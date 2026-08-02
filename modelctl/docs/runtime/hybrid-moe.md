# Hybrid MoE execution (`--moe-hybrid-mode`)

Current state of GPU-hit / CPU-miss hybrid execution in the llama.cpp
fork. **Implemented and hardware-validated (2026-07-31); opt-in; never
auto-selected** — on the current target it measures slower than the
plain transfer cache (see the performance verdict below).

## What it does

Under `--moe-hybrid-mode on`, a routed-expert weight that the cache
declines to admit is **not staged to the device**. The rows that needed
that expert execute on CPU over the original host/mmap weights,
concurrently with the GPU rows, and land in the same output tensor.
Avoiding the H2D transfer is the entire saving; with hybrid off, every
miss is staged host→device as usual (the transfer-cache path, see
[moe-cache.md](moe-cache.md)).

## Hit and miss paths

- The scheduler hook records each declined miss in a per-staged-tensor
  plan keyed by the staged copy's device base address (unique per
  staged tensor; two models cannot collide on it the way tensor names
  can) and copies only the 512-byte MMQ padding head instead of the
  expert's weights.
- `ggml_sycl_mul_mat_id` takes the plan exactly once per execution.
  While a plan is pending, fused kernels stay off — they would read the
  unstaged regions.
- GPU rows (cache hits and staged experts) are queued asynchronously on
  the device's in-order queue.
- The CPU tier (`moe_cpu_execute_gemvs` in `moe-hybrid.cpp`) computes
  the skipped experts' rows **while the GPU works**, through
  ggml-cpu's quantized `vec_dot` kernels: the activation is quantized
  once into the weight type's `vec_dot_type` (Q8_K for the IQ2 family)
  and dotted straight against the quantized weight blocks, so nothing
  is ever materialized as f32. Work is split by output-row slice over
  a persistent worker pool, claimed dynamically; every output row has
  exactly one writer, so the result is bit-identical to sequential
  execution however the slices are scheduled.
- Reaching those kernels is a deliberate link edge: backends normally
  see only ggml-base, and `ggml-sycl` additionally links `ggml-cpu`
  (guarded on the target existing). Where it does not exist — CPU
  backend off, or the multi-variant DL build — the tier falls back to
  the dequantize-and-dot path, which is the same math and about 25x
  slower per row. `moe_hybrid_cpu_kernel_rows_total` says which one
  ran, so the fallback is visible rather than silent.
- The tier calls `ggml_cpu_init()` before first use. Without it every
  quantized `vec_dot` returns exactly 0.0 — silently, which is how a
  host-only test that linked the tier directly produced all-zero
  expert outputs that no throughput number would have revealed.
- CPU results are copied in with one H2D transfer per batch and land
  through the same in-order queue. The single synchronization point is
  the join before the output is consumed. Both the batch-1 decode
  branch and the batched-prompt branch are covered.

## Invariants

- **A partition is a list of weighted contributions, not unique
  destination rows.** With `n_expert_used > 1`, several contributions
  share one destination row, so any merge at the fused-MoE level must
  accumulate `Σ routing_weight × expert_output` per token — a plain
  copy overwrites all but the last expert.
- In the current in-op integration, `MUL_MAT_ID` output rows are
  per-(token, expert) and the graph applies routing weights downstream,
  so the in-op merge is a disjoint-row scatter. The weighted-sum merge
  machinery in `moe-hybrid.cpp` remains for any future fused-level
  merge.
- Result ordering is fixed by row, never by completion order — output
  must not depend on which tier finished first.
- Per-layer barrier: all expert outputs for a layer are complete before
  the next layer's attention consumes them.
- Residency classification uses `moe_expert_cache::contains()`, which
  touches nothing; `lookup()` counts hits/misses and updates recency,
  which is correct for the transfer-cache path and wrong for
  classifying rows.

## Why it lives inside `MUL_MAT_ID`

Offload decisions run at schedule-build time with only static
shape/buffer information; the selected-expert `ids` are readable only
at execution time; and graphs are reused across decode steps. So the
hit/miss split cannot influence op placement — hybrid execution has to
live inside the execution of one `MUL_MAT_ID`, not in the scheduler's
choice of where to run it. Any design that routes whole ops per token
re-derives this blocker (established 2026-07-30, decode-cache-fix
investigation).

## Supported architectures and quants

The mechanism is tensor-level, with no per-architecture assumptions
(`moe_hybrid_supported_archs: ["any"]`). The CPU tier handles every
weight type ggml has a dequantizer for; a type without one is simply
never skipped from staging — it stages and runs on GPU as usual
(fail-safe, not fail-wrong).

## Metrics (`/metrics`)

Execution: `moe_hybrid_cpu_rows_total`, `moe_hybrid_gpu_rows_total`,
`moe_hybrid_gpu_fallback_rows_total`,
`moe_hybrid_h2d_bytes_avoided_total`, `moe_hybrid_staging_skips_total`,
`moe_hybrid_cpu_time_ms`, `moe_hybrid_merge_time_ms`.

Miss-path profile: `moe_hybrid_cpu_tier_calls_total`,
`moe_hybrid_cpu_tier_jobs_total`, `moe_hybrid_cpu_weight_rows_total`,
`moe_hybrid_cpu_kernel_rows_total`,
`moe_hybrid_cpu_weight_bytes_total`, `moe_hybrid_cpu_threads_used`,
`moe_hybrid_cpu_wall_ns_total`, `moe_hybrid_cpu_dispatch_ns_total`,
`moe_hybrid_cpu_quant_act_ns_total`.

`weight_bytes / wall_ns` is the number to watch: it says whether the
tier is reading the weight stream at memory speed or is stuck behind
something else. `kernel_rows == weight_rows` confirms the quantized
kernels ran.

`moe_hybrid_cpu_dequant_ns_total` and `moe_hybrid_cpu_matmul_ns_total`
split one row's time and cost two clock reads per row to collect, so
they are recorded only under `GGML_MOE_HYBRID_PROFILE=1`;
`moe_hybrid_cpu_profiled_rows_total` says how many rows they cover, so
zero cannot be misread as "no time spent".

## Environment switches

- `GGML_MOE_HYBRID_PROFILE=1` -- record the per-row ns breakdown.
- `GGML_MOE_HYBRID_NO_VEC_DOT=1` -- force the dequantize-and-dot path,
  for A/B measurement and for bisecting a numerical difference back to
  the kernel change.
- `GGML_MOE_HYBRID_THREADS=N` -- pool size, counting the caller.
  Default is `nproc - 2`.

## Correctness evidence

**2026-07-31.** Token-identical greedy output against the non-hybrid
reference on the tiny-MoE fixture (6 distinct prompts × 16 tokens,
including an all-CPU-tier stress with 5,577 CPU rows and a main+draft
two-context run) and on the real target (Qwen3.5-122B-A10B, 24-token
greedy, identical to the cache-only condition). Zero device loss.

**2026-08-01, after the CPU-kernel pass.** Re-verified on the tiny-MoE
fixture: 4 prompts × 2 repeats, hybrid on vs off, identical token
arrays in all 8 sequences, with the kernel path exercised (72 staging
skips, all 62,592 weight rows through `vec_dot`). The kernel path dots
against a Q8_K-quantized activation, so it is not bit-identical to
dequantize-and-dot: relative RMS 0.0073 against that reference on real
IQ2_XXS weights. See also the known defect below.

## Performance verdict

**2026-07-31, before the CPU-kernel pass.** Hybrid **lost** to the
plain transfer cache: 2.46 vs 4.33 t/s decode (Qwen3.5-122B-A10B, 96
tokens, MoE offload threshold 1, 4 GiB cache, admission 2), despite
avoiding 35.9 GB of expert H2D transfer. Threading the CPU tier moved
it from 1.91 to 2.46 t/s; dequantization cost still exceeded the PCIe
transfer it replaced at that miss rate.

**2026-08-01, after it**
([evidence](../evidence/2026-08-01-hybrid-cpu-kernel-pass.md)). The
miss path is 74x faster on the microbenchmark and 1.84x faster
in-server (103 -> 56 ns/row over the same ~158M rows and 75 GB), and
the tier is now bound by the weight stream rather than by FLOPs. On the
122B, three replicates: cache+hybrid **6.033** vs cache-only **5.394**
t/s — hybrid is no longer behind. The per-condition spread is 4-18%,
so three runs do not separate those means on their own; the tier
counters do.

Static placement still wins that comparison outright at **23.975**
t/s, because this model fits in 42.8 GiB of VRAM and nothing has to
stream. Hybrid remains opt-in and is never auto-selected. The
acceptance matrix has a `hybrid-cpu-miss` cell, so the comparison
reruns in one command when anything changes.

## Not implemented

- Promotion-delay and eviction-before-reuse counters. (Promotions are
  already queue-asynchronous and never block the miss path; the two
  counters just aren't recorded.)
- Per-projection (rather than per-expert) skip decisions.
- Weight reuse across a batched prompt: the batched branch issues one
  job per (expert, row), so an expert's weights are re-streamed once
  per routed row. Decode is batch 1 and unaffected, and with
  `--moe-cache-prefill-admission off` no skips happen during prefill at
  all, so this has not been worth fixing yet.

## The nondeterminism that blocked this rail (fixed)

An identical 122B condition used not to reproduce its own greedy token
sequence run to run, which left "hybrid on matches hybrid off" with no
fixed reference on that model. It was recorded here as a defect in the
surrounding cache path. **It was not in the cache path.** It reproduced
with no cache configured at all, and — once logprobs rather than token
IDs were compared — under static placement too. The cause was oneDNN's
GPU matmul running without oneDNN's `deterministic` attribute, reached
by any matmul whose batch exceeds `MMQ_MAX_BATCH_SIZE`, i.e. during
prompt processing in every condition.

`GGML_SYCL_DETERMINISTIC` (default 1) pins it, and the rail is now
evaluable on the 122B rather than only on the tiny fixture. The
localization, the controls that excluded the cache, and the acceptance
numbers are in
[../evidence/2026-08-01-onednn-determinism.md](../evidence/2026-08-01-onednn-determinism.md).
