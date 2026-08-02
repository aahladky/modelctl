# How to actually test the MoE expert cache

Written 2026-07-30 after three rounds of measuring the cache, two of
which produced misleading numbers for avoidable reasons. Read this before
running another cache benchmark.

The recurring failure is the same each time: **the benchmark ran, produced
plausible numbers, and measured something other than what was intended.**
None of these were detected by the numbers looking wrong. They were caught
by checking what the machine was actually doing.

Measured results for this machine live in
[../evidence/](../evidence/) — in particular the batch-1 decode and
Q4_K_M offload-threshold reports.

## 1. Confirm the cache is running at all

The cache attaches to a scheduler hook that only fires for cross-backend
weight copies, gated on the op-offload batch threshold. **Do not assume
the threshold is 32**: it is build- and environment-specific. Read
`constraints.moe_cache_min_batch` from
`llama-server --modelctl-capabilities` *run under the effective launch
environment* — the probe honors `GGML_OP_OFFLOAD_MOE_MIN_BATCH` the same
way the runtime will. On a default environment the threshold is 32,
decode is batch 1, and **the cache never activates during generation** —
it is configured, logged as enabled, and completely inert.

An earlier measuring round concluded `cache-enabled ≈ cache-disabled`
and attributed it to the model fitting in VRAM. The real reason was that
no cache existed.

Before trusting any comparison:

```bash
grep "moe_cache: initialized" server.log      # must be present
curl localhost:$PORT/metrics | grep moe_cache # must be non-empty
```

`MoE expert cache enabled: ...` in the log means the *config* was
accepted. It does **not** mean a cache was created. Only
`moe_cache: initialized on device N` means that.

To reach the cache at batch 1, set `GGML_OP_OFFLOAD_MOE_MIN_BATCH=1`
(routed MoE ops only, requires `moe_offload_threshold_control`) or
`GGML_OP_OFFLOAD_MIN_BATCH=1` (global) — and see §3, because the global
variant especially is not free.

## 2. Derive placement for the model under test; never inherit it

`--tensor-split 4,1` was copied from the `qwen3-5-122b-a10b-ud` profile
into a Q4_K_M run. That value was tuned for IQ1_M (31.9 GiB). Q4_K_M is
71.3 GiB — 2.2× larger — and on 31.9 + 11.9 GiB of VRAM the 80/20 split
pinned SYCL0 at 98% while **~10 GiB of SYCL1 sat unused**, forcing far
more onto storage than the hardware required.

The GPUs here are a **2.7:1** capacity ratio, so a capacity-proportional
split is roughly `8,3`, not `4,1`.

This matters beyond wasted VRAM: an unnecessarily storage-bound baseline
**flatters whatever is being tested**, because it leaves more for the
optimisation to recover. A cache measured against a handicap you
introduced will look better than it is.

Check mid-run, not just at the end:

```bash
modelctl doctor            # or read free VRAM per device
```

If one GPU is saturated and the other has room, the placement is wrong
and the numbers are not about the cache.

## 3. Prove correctness before measuring speed

A throughput number from a run that produced wrong output measures
nothing. Before any A/B comparison, run the deterministic oracle:
greedy decoding, temperature 0, fixed seed, a fixed prompt set, and
compare **token IDs** between the condition under test and its
reference (cache off, or the stock-upstream oracle). Token divergence
is a failure, not a tolerance. The correctness fixtures under the
fork's `tests/` (tiny-MoE model, `/cache/reset` between conditions)
exist for exactly this; the invalidated 07-28 benchmark is what
skipping this step produces.

**First check that the oracle reproduces itself.** For a month this
machine's 122B runs did not, and every A/B built on top of them was
comparing two samples from a distribution. The cause was oneDNN's GPU
matmul, whose default permits a reduction order that varies between
executions; it is reached whenever a batch exceeds
`MMQ_MAX_BATCH_SIZE` (32), i.e. during prompt processing, in **every**
condition including static placement. `GGML_SYCL_DETERMINISTIC` now
defaults to 1 and pins it. Details and the localization in
[../evidence/2026-08-01-onednn-determinism.md](../evidence/2026-08-01-onednn-determinism.md).

Two consequences for anyone running this protocol:

- Run the reference against **itself** before comparing anything to it.
  Two runs of one condition must be token-identical. If they are not,
  no downstream comparison means anything, and that is a defect to
  chase rather than a tolerance to widen.
- Token identity is a weak instrument. Greedy argmax survives large
  logit perturbations, so sequences can agree for dozens of steps while
  the numbers underneath differ by nats. Ask for `n_probs` and compare
  **logprobs**, not only token IDs — within one step, differences of
  log-softmax values are exactly differences of logits.

`GGML_MOE_FINGERPRINT=<path>` in the fork records, per routed
`MUL_MAT_ID`, which branch ran and which experts it was routed to, plus
every staging decision (hit / promote / hybrid-skip / fallback). Diffing
two runs' fingerprints answers "did these two runs execute the same
ops?" before anyone argues about arithmetic. It is inert when unset.

## 4. Measure the cost of reaching the cache, not just its benefit

`GGML_OP_OFFLOAD_MIN_BATCH` is **global**. Lowering it to 1 forces every
small-batch op through a cross-backend copy, not only the MoE ones. On
the models tested this cost 41–52% before the cache did anything.
(`GGML_OP_OFFLOAD_MOE_MIN_BATCH` exists precisely to avoid that; measure
it as its own condition.)

So a two-condition test (cache on vs off, both at min-batch 1) answers
the wrong question. Always run **three**:

| | min-batch | cache | what it tells you |
|---|---|---|---|
| A | default | on | the real-world baseline; cache will be inert |
| B | 1 | off | what lowering the threshold costs on its own |
| C | 1 | on | what the cache recovers |

`C vs B` is the cache's benefit. **`C vs A` is whether any of it was
worth doing.** Reporting only `C vs B` overstates the case badly.

## 5. Test the quant you actually serve

The answer changes with the model, and not slightly (2026-07-30
measurements, this machine, details in
[../evidence/](../evidence/)):

| model | threshold cost (A→B) | cache gain (B→C) | net (A→C) | hit ratio |
|---|---|---|---|---|
| 35B-A3B, IQ4_XS (16.3 GiB) | -52% | +21% | **-42%** | 71.6% |
| 122B-A10B, IQ1_M (31.9 GiB) | -41% | +48% | **-12%** | 85.3% |
| 122B-A10B, Q4_K_M (71.3 GiB) | **+28%** | *not measured* | — | — |

A conclusion drawn from the first row ("the cache cannot serve
interactive decode") did not survive the second, and the third inverted
the sign of the threshold cost. Active expert bytes per token is the
variable that matters — A3B activates ~3B parameters, A10B ~10B — and
whether the model is VRAM-, RAM-, or storage-bound changes it again.

Do not generalise from one quant. State which one was measured.

## 6. Keep the harness from lying to you

- **`llama-sycl-env.sh` terminates a script that sources it.** Source it
  in the caller, then invoke the script.
- **`pkill -f` does not reliably kill `llama-server`.** Kill by PID, then
  verify with `pgrep` *and* by watching VRAM come back.
- **Verify VRAM is released between conditions.** A 30 GiB model that has
  not finished unloading changes the next run's placement silently.
- **Watch for orphaned wait loops.** An `until grep -q DONE ...; sleep`
  loop polling a file that never gets its sentinel spins until the
  session ends.

## 7. Prefer the harness over ad-hoc scripts

`modelctl_acceptance.run_matrix()` already handles placement,
preconditions, per-cell cache state, and honest skip reasons, and records
every measurement through `test_launch_plan` with storage counters
attached — RSS, PSS, page faults, read bytes, storage-activity
classification and bottleneck attribution. Its cells vary
`GGML_OP_OFFLOAD_MIN_BATCH` and `GGML_OP_OFFLOAD_MOE_MIN_BATCH`
per condition, and the `hybrid-cpu-miss` cell reruns the hybrid-vs-cache
comparison in one command. Every lesson above is something the harness
already gets right; extend the harness rather than writing another
one-off script.
