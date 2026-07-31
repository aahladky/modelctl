# Hybrid GPU-hit / CPU-miss MoE execution — runtime design

Roadmap Task G1. Written 2026-07-30 against fork commit `ad4903d40`.

Phase G's requirement is **useful interactive decoding**, not large prompt
batches. Everything below is designed for batch 1 and judged on it.

## 0. Read this before building on the existing scaffolding

`ggml/src/ggml-sycl/moe-hybrid.{hpp,cpp}` exists and is compiled, but no
backend code calls any of it, and `moe_hybrid_cpu_miss` is hardcoded
`false` in `common/arg.cpp` with a `NOT IMPLEMENTED` comment. It is dead
code, and it has four defects that a reader could easily mistake for a
working starting point:

1. **The partition builder classifies every row as a miss.**
   `moe_build_partition` probes with
   `cache->lookup(layer, expert, 0, /*proj_bytes=*/0)`. The geometry guard
   in `lookup` rejects any `proj_bytes` that does not equal the init-time
   projection size, so the probe returns `nullptr` unconditionally — even
   against a fully populated cache. The partition has never been observed
   to contain a hit.
2. **`lookup` is the wrong primitive for partitioning.** Even fixed, it
   mutates `hits`/`misses`, updates the SLRU recency tick and clears
   admission progress. Partitioning must be able to ask "is this resident"
   without changing what is resident.
3. **No routing coefficient anywhere.** A routed MoE output is the
   *weighted sum* of `n_expert_used` expert outputs per token. The
   partition carries only `(contiguous_row, original_row, expert_id)`, and
   `moe_merge_outputs` `memcpy`s rows into `dst + original_row * ne0`. With
   `n_expert_used = 2` the second expert overwrites the first, unweighted.
   This is not an optimisation gap; it is wrong output.
4. **The CPU path assumes f32 experts.** It casts `src0_host` to
   `const float *` and hand-rolls a dense matmul. Every model this stack
   serves is quantised — IQ1_M, IQ4_XS, Q4_K_M — so it would reinterpret
   quantised blocks as floats. G3 also explicitly calls for reusing the
   existing CPU kernels rather than writing new ones.

Defect 3 is the one that shapes the design: the partition is not a
2-way split of rows, it is a *scatter-accumulate* with weights.

## 1. The binding constraint

`docs/upstream-sync/2026-07-30-decode-cache-fix.md` established this and it
still holds:

- The offload decision (`ggml_backend_sched_backend_id_from_cur` →
  `ggml_backend_sycl_device_offload_op`) runs at **schedule-build time**,
  with only static tensor shape and buffer information.
- The selected-expert `ids` that a real hit/miss split depends on are only
  readable at **schedule-execution time**.
- Graphs are reused across decode steps, so a decision baked in at build
  time cannot vary per token.

**Consequence for this design:** the partition cannot influence op
placement. Hybrid execution has to live *inside* the execution of one
`MUL_MAT_ID`, not in the scheduler's choice of where to run it. Any design
that tries to route whole ops per token is re-deriving a blocker that has
already been measured.

## 2. What changed since Phase G was written

Phase G was justified by "the transfer cache cannot serve interactive
decode". That claim came from one model and did not survive re-measurement
(`modelctl/docs/moe-cache-batch1-decode-2026-07-30.md`):

| model | threshold cost | cache gain | net |
|---|---|---|---|
| 35B-A3B IQ4_XS | -52% | +21% | -42% |
| **122B-A10B IQ1_M** | -41% | **+48%** | **-12%** |

Two further facts, measured 2026-07-30 against a stock-upstream oracle at
`9b2a08881`:

- `GGML_OP_OFFLOAD_MIN_BATCH=1` produces **byte-identical** greedy output
  to the default on 122B-A10B, on both the fork and stock upstream. The
  earlier "produces wrong output" warning does not reproduce.
- Its ~38% decode cost also reproduces on stock upstream (22.76 → 14.13
  t/s), so it is an upstream characteristic, not something the fork
  introduced.

**Therefore:** an op-aware offload threshold — exempting routed MoE expert
ops from `op_offload_min_batch_size` in
`ggml_backend_sycl_device_offload_op` — is a far smaller change that would
engage the existing cache at batch 1 without dragging unrelated ops
through PCIe. It is not part of Phase G, it is a plausible substitute for
it, and **G's value must be judged against that baseline rather than
against the default configuration**. This design proceeds anyway, at the
user's direction, but no G result should be reported as a win without the
op-aware threshold measured alongside it.

## 3. Design

### 3.1 Where it runs

Inside the SYCL `MUL_MAT_ID` implementation, after `ids` is available on
host. One partition per (layer, projection) per token batch. The scheduler
is not involved and no graph is rebuilt.

### 3.2 Partition representation (Task G2)

The partition is a list of **contributions**, not a split of rows. Each
contribution records everything needed to compute one expert's share of
one token's output and to place it correctly:

| field | why |
|---|---|
| `original_row` | destination row in `dst`, pre-sort order |
| `contiguous_row` | source row in the reordered `src1` |
| `expert_id` | which expert |
| `routing_weight` | the router's coefficient; the merge is a weighted sum |
| `projection` | gate / up / down; residency is per projection, not per expert |
| `tier` | GPU_HIT or CPU_MISS |
| `slot_ptr` | for hits, the resident device region (avoids a second lookup) |

`n_expert_used` contributions share one `original_row`. That is the whole
reason merge must accumulate.

### 3.3 Residency query

A new `moe_expert_cache::contains(layer, expert, projection) const`:
takes the lock, reports whether that projection is filled, and **touches
nothing** — no stats, no recency, no admission. `lookup` keeps its current
behaviour for the transfer-cache path, which does want the side effects.

### 3.4 CPU miss execution (Task G3)

Reuse `ggml`'s existing quantised kernels via the CPU backend's
`ggml_compute_forward_mul_mat_id` machinery over the mmap-backed weights.
Do not hand-roll a matmul and do not introduce custom SSD I/O — the
weights are already mapped and the page cache is the right layer for
residency.

Initial scope: one architecture (`qwen35moe`) and the quants actually
served here (IQ1_M, IQ4_XS, Q4_K_M). Anything else fails closed to the
existing non-hybrid path rather than guessing.

### 3.5 GPU hit execution (Task G4)

Hit contributions consume the persistent cache slot directly. Staging them
into a transient tensor would reintroduce the per-token copy the cache
exists to avoid.

### 3.6 Merge (Task G5)

`dst[original_row] = Σ routing_weight_i × expert_output_i` over all
contributions for that row, regardless of tier. Accumulation happens in
f32. Ordering is fixed by `(original_row, expert_slot_index)` rather than
by completion order, so results do not depend on which tier finished
first — otherwise the same input yields different output run to run.

### 3.7 Synchronisation

The CPU miss path and the GPU hit path run concurrently. The merge waits
on both. The GPU-side accumulation is ordered on the compute queue; the
CPU results are copied in with one H2D transfer per batch, not per row.
No `.wait()` in the steady-state loop except the single join before merge.

### 3.8 Temporary buffers

Sized once per context from `(max_batch × n_expert_used)`, not per token:
one host activation buffer, one host output buffer, one device staging
buffer for the merged CPU results. Reused across decode steps.

### 3.9 Numerical tolerance

The reference is the same model with hybrid disabled, greedy, temperature
0, fixed seed. Expert outputs computed on CPU and on GPU differ in the
last bits, so the acceptance criterion is **token-identical output over a
fixed prompt set**, not bit-identical logits — the same oracle method Task
0.6 uses. Divergence in token IDs is a failure, not a tolerance.

### 3.10 Asynchronous promotion (Task G6)

A miss enqueues a promotion; it never blocks the miss path. Record
promoted bytes, queue delay, transfer time, overfetch, eviction-before-
reuse, and H2D bytes avoided.

## 4. Scope

**Delivered (G2):** the contribution representation, the side-effect-free
residency query, a corrected partition builder, and host-only unit tests
in the shape F8 established.

**Delivered (G3, 2026-07-31):** `moe_cpu_execute_misses()` in
`moe-hybrid.cpp` — batch-1 CPU execution of a partition's miss
contributions over host/mmap weights, one dequantized row at a time so no
weight byte is read twice. One deviation from §3.4, made deliberately:
the executor dequantizes through **ggml-base's type traits**
(`ggml_get_type_traits()->to_float`) plus an f32 dot, instead of calling
the CPU backend's `mul_mat_id` machinery. The SYCL backend cannot reach
ggml-cpu's kernels in shared/dynamic builds without inventing a new
cross-backend interface; the ggml-base dequantizer is the same reference
math those kernels are validated against; and the miss path at batch 1 is
bound by reading the weights (page faults / storage), not FLOPs. Types
without a dequantizer fail closed (return -1 → non-hybrid path). Covered
by five host-only cases in `tests/test-moe-hybrid.cpp`, including a
quantized (Q8_0 / Q4_K / Q6_K) comparison against an independent
dequantize-and-dot reference and an end-to-end
partition→execute→weighted-merge check.

**Delivered (G4/G5/G7, 2026-07-31, same-day continuation):** true hybrid
execution, wired and hardware-validated. Under `--moe-hybrid-mode on`, a
miss the cache declines to admit is **never staged to the device** — the
scheduler hook records it in a per-staged-tensor plan (keyed by the
`input_cpy` device base, which two models cannot collide on) and copies
only the 512-byte MMQ padding head. `ggml_sycl_mul_mat_id` takes the
plan exactly once, keeps fused kernels off while one is pending, queues
the GPU rows asynchronously, computes the skipped experts' rows on CPU
(threaded `moe_cpu_execute_gemvs`, contiguous chunks, bit-identical to
sequential) **while the GPU works**, and lands the results through the
same in-order queue. Both the batch-1 and the batched-prompt branch are
covered; metrics (`moe_hybrid_*` in `/metrics`) record cpu/gpu rows,
avoided H2D bytes, staging skips, and CPU/merge time. Note one deviation
from §3.2/§3.6: because `MUL_MAT_ID` output rows are per-(token,expert)
and the graph applies routing weights downstream, the in-op merge is a
disjoint-row scatter — the weighted-sum merge machinery remains for any
future merge at the fused-MoE level.

**§3.9 oracle results:** token-identical to the non-hybrid reference on
the tiny-MoE fixture (6 distinct prompts × 16 tokens, including an
all-CPU-tier stress with 5,577 CPU rows and a main+draft two-context
run) AND on the real target (Qwen3.5-122B-A10B IQ1_M, 24-token greedy,
identical to the cache-only condition). Zero device loss.

**Performance verdict (2026-07-31, this machine):** hybrid **loses** to
the plain transfer cache on the current target and placement — 2.46 vs
4.33 t/s decode (96 tokens, threshold=1, 4 GiB cache, admission 2),
despite avoiding 35.9 GB of expert H2D transfer. The threaded CPU tier
(ggml-base dequant + dot) moved it from 1.91 to 2.46 t/s but IQ1_M
dequantization cost still exceeds the PCIe transfer it replaces at this
miss rate. §2's warning held. Consequence: `moe_hybrid_cpu_miss` now
truthfully reports **implemented**, the feature stays **opt-in and
never auto-selected** (the H2 experimental-margin guardrail sees the
measured loss), and the acceptance matrix gained a `hybrid-cpu-miss`
cell so the comparison reruns in one command when anything changes.

**Not delivered / future:** G6's full asynchrony (promotions are already
queue-async and never block the miss path, but promotion-delay and
eviction-before-reuse counters are not recorded); a vec_dot-grade CPU
tier (reaching ggml-cpu's optimized kernels from the SYCL backend needs
a cross-backend interface — the single change most likely to flip the
verdict); per-projection (rather than per-expert) skip decisions.
