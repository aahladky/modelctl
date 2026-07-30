# Decode-time cache activation — investigated, not fixed, two concrete blockers found

Follow-up to `2026-07-30-cache-inactivity-rootcause.md`, which found the MoE
cache never engages during single-token decode because the SYCL backend's
`op_offload_min_batch_size` threshold (default 32, `GGML_OP_OFFLOAD_MIN_BATCH`
env var, set once at backend-registry init in
`ggml_backend_sycl_reg()`, `ggml/src/ggml-sycl/ggml-sycl.cpp:6810`) keeps
`MUL_MAT_ID` on CPU below that batch size — decode is always batch=1, so the
cache's `copy_experts` hook (`ggml/src/ggml-backend.cpp:1643`) is never
reached. The user asked for a scoped fix attempt. **No fix was made** — two
independent blockers were found, one architectural and one a live
correctness bug unrelated to the cache itself.

## Blocker 1 (architectural): a precise cache-hit bypass isn't possible without Phase G's execution model

The offload decision (`ggml_backend_sched_backend_id_from_cur`,
`ggml/src/ggml-backend.cpp:887-949`, calling
`ggml_backend_offload_op`/`ggml_backend_sycl_device_offload_op` at
`ggml-sycl.cpp:6410`) happens at **schedule-build time**, using only static
tensor shape/buffer information. The actual selected-expert `ids` values
that `copy_experts` needs to make its real per-expert hit/miss decision are
only read later, at **schedule-execution time**
(`ggml_backend_tensor_get_async` on the ids tensor,
`ggml-backend.cpp:1608-1621`) — i.e. after the backend assignment for that
node is already locked in.

Critically, this assignment is not re-evaluated per token: llama.cpp
reuses the built schedule across decode steps (`graphs reused = N` in every
server log this project has captured). A schedule-time decision is
structurally incapable of being "this specific token's specific experts are
cached" aware, because by the time that information exists, the schedule —
and the backend it assigned this node to — has already been fixed and is
being replayed unchanged for every subsequent token. Making the assignment
data-dependent per token would mean abandoning graph reuse for these nodes,
or restructuring execution so the hit/miss decision happens *inside* an
already-scheduled node's compute call rather than at backend-assignment
time — i.e. Phase G's actual design (Task G1's "classify work into GPU-hit
and CPU-miss rows" as a runtime partition, not a schedule-time backend
choice). This is not a small, scoped change; it's the thing Phase G already
exists to build.

## Blocker 2 (bug, but not this feature's bug): forcing small-batch offload breaks correctness generally

There's a cheap, zero-code experiment that doesn't require the above: the
existing `GGML_OP_OFFLOAD_MIN_BATCH` env var already lets you lower the
threshold process-wide. Tried it:

- **Baseline** (`min_batch=32`, cache enabled, 9-token prompt, greedy
  decode, `n_predict=80`): 1.24 tok/s, zero `moe_cache_*` activity (as
  expected), coherent output —
  `[271, 248068, 271, 248069, 271, 623, 279, 11012, 11, 15333, ...]`
  ("...Unit 734—known simply as 'Seven'—spent its days polishing the brass
  gears...").
- **`GGML_OP_OFFLOAD_MIN_BATCH=1`**, same prompt/seed/model, cache enabled:
  4.80 tok/s (**3.9x faster**), and the cache genuinely activated —
  52,193 hits / 15,733 misses / 1,885 evictions / 3,305 promotions / 5.85GB
  H2D bytes, 76.8% hit ratio. Looked like a real win.
- **But the output is wrong**: tokens diverge from the baseline after
  position 2 and degenerate into an infinite repeat of token `18` —
  `[271, 248068, 18, 18, 18, 18, 18, 18, ...]`. Not a benign reordering;
  broken generation.
- **Isolated the cause**: reran `GGML_OP_OFFLOAD_MIN_BATCH=1` with **no**
  `--moe-cache-*` flags at all, same prompt/seed. **Identical broken
  output** — same immediate collapse into repeated `18`. This reproduces
  with the MoE cache entirely absent from the command line, so it is a
  general small-batch SYCL offload correctness issue on this
  architecture/backend, not a bug in the cache feature. (Could not confirm
  whether this is pre-existing in unmodified upstream or something this
  port's other changes expose, since the stock-upstream oracle build from
  earlier tasks was no longer present in this session and rebuilding one
  was out of scope for this pass — worth checking before anyone relies on
  `GGML_OP_OFFLOAD_MIN_BATCH` for anything on this architecture.)

The practical implication: even the "crude, no-code-change" version of a
decode-time fix is not safe to recommend, independent of the cache
question — something about small-batch (`batch < 32`, likely `batch == 1`
specifically, untested at intermediate sizes) offload for this
architecture's `MUL_MAT_ID` produces wrong output today. This deserves its
own separate bug report/investigation; it happens to be the same knob that
would have made the cache's crude bypass "free," which is why it surfaced
here, but it is not part of the cache feature and shouldn't be conflated
with it.

## Bottom line

No fix implemented, and none should be attempted as a "scoped" change
today:

1. A *correct*, per-request cache-hit-aware bypass requires Phase G's
   runtime row-partitioned dispatch — there is no cheaper mechanism
   available given how schedule/graph reuse works.
2. The one available cheap shortcut (globally lowering
   `GGML_OP_OFFLOAD_MIN_BATCH`) produces wrong output, for reasons
   unrelated to and outside the cache — a separate, real, currently-unfixed
   correctness bug in small-batch SYCL offload for this architecture,
   worth its own investigation before Phase E, independent of the MoE
   cache work entirely.

The characterization from the root-cause doc stands: this cache currently
only helps prompt processing and sufficiently-large continuous-batching
decode, not lone single-token interactive generation, and that will remain
true until Phase G's actual hybrid execution work lands.
