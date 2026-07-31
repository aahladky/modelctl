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
- The CPU tier (`moe_cpu_execute_gemvs` in `moe-hybrid.cpp`, threaded
  over contiguous chunks, bit-identical to sequential execution)
  computes the skipped experts' rows **while the GPU works**, using
  ggml-base's dequantizer (`ggml_get_type_traits()->to_float`) plus an
  f32 dot. It cannot reach ggml-cpu's optimized vec_dot kernels from
  the SYCL backend (see "Not implemented").
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

`moe_hybrid_cpu_rows_total`, `moe_hybrid_gpu_rows_total`,
`moe_hybrid_gpu_fallback_rows_total`,
`moe_hybrid_h2d_bytes_avoided_total`, `moe_hybrid_staging_skips_total`,
`moe_hybrid_cpu_time_ms`, `moe_hybrid_merge_time_ms`.

## Correctness evidence (2026-07-31)

Token-identical greedy output against the non-hybrid reference on the
tiny-MoE fixture (6 distinct prompts × 16 tokens, including an
all-CPU-tier stress with 5,577 CPU rows and a main+draft two-context
run) and on the real target (Qwen3.5-122B-A10B IQ1_M, 24-token greedy,
identical to the cache-only condition). Zero device loss.

## Performance verdict (2026-07-31, this machine)

Hybrid **loses** to the plain transfer cache on the current target and
placement: 2.46 vs 4.33 t/s decode (Qwen3.5-122B-A10B IQ1_M, 96
tokens, MoE offload threshold 1, 4 GiB cache, admission 2), despite
avoiding 35.9 GB of expert H2D transfer. Threading the CPU tier moved
it from 1.91 to 2.46 t/s; IQ1_M dequantization cost still exceeds the
PCIe transfer it replaces at this miss rate. The feature therefore
stays opt-in and is never auto-selected — the experimental-margin
guardrail sees the measured loss. The acceptance matrix has a
`hybrid-cpu-miss` cell, so the comparison reruns in one command when
anything changes.

## Not implemented

- Promotion-delay and eviction-before-reuse counters. (Promotions are
  already queue-asynchronous and never block the miss path; the two
  counters just aren't recorded.)
- A vec_dot-grade CPU tier: reaching ggml-cpu's optimized kernels from
  the SYCL backend needs a cross-backend interface. This is the single
  change most likely to flip the performance verdict.
- Per-projection (rather than per-expert) skip decisions.
