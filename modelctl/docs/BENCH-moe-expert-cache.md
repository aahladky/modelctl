# MoE expert cache — benchmark results

Model: Laguna-S-2.1, 19/47 expert layers on CPU (`ffn_.*_exps=CPU`).
Config: 1 GiB cache budget, SLRU policy, admission=1.
Baseline (cache disabled): **prompt 4.7 t/s, gen 11.2 t/s**.

## Latest: with F0 scheduler hook, 2026-07-28 (fork @ 67ab58096)

| | prompt t/s | gen t/s |
|---|---|---|
| Baseline | 4.7 | 11.2 |
| Cache + hook | 27.6 | 30.1 |
| **Δ** | **+487%** | **+169%** |

The scheduler hook intercepts every host→device expert copy, so the
cache sees CPU-resident experts (not just GPU-resident ones going
through `ggml_sycl_mul_mat_id`). That's why generation nearly triples
here versus the pre-hook run below — decode re-selects experts every
token, so caching the CPU-resident 19/47 layers pays off per token.

## Earlier: pre-hook, 2026-07-28 (fork @ a64c80eee)

| | prompt t/s | gen t/s |
|---|---|---|
| Cache enabled (inline path only) | 28.1 | 15.1 |
| **Δ vs baseline** | **+497%** | **+35%** |

## Caveats

- **Both runs predate the C1 slot-geometry fix** (`f0750e8a4`). Before
  that fix, up/down projection cache hits served the wrong (gate)
  weights — throughput was unaffected (identical copies), so the speed
  numbers should still hold, but output quality during those runs was
  degraded. Re-validate both speed and output correctness against
  `f42f2fe4e` or later before quoting these.

## Re-validation checklist

1. Same model + flags, cache off vs on (`--moe-cache-bytes 0` vs 1 GiB).
2. Output sanity: fixed prompts, compare cache-on output against
   cache-off for coherence (first run with *correct* cached weights).
3. Throughput: `llama-bench` or server timings, prompt + gen.
4. Check `moe_cache_*` metrics (`/metrics`) for hit ratio on the
   CPU-resident layers.

## Post-C1-fix validation, 2026-07-29 (fork @ f42f2fe4e, qwen3-5-122b-a10b-ud)

First run with *correct* cached weights. 19/48 expert layers forced to
CPU (`-ot blk.(2[9]|3[0-9]|4[0-7]).ffn_.*_exps=CPU`), 1 GiB cache, SLRU,
admission=1, `--metrics` on.

- **Correctness: PASS.** All final `content` outputs bit-identical
  cache-off vs cache-on (and vs a cache-off/off control). The one
  reasoning-stream divergence observed also occurs off-vs-off — baseline
  SYCL nondeterminism, not the cache.
- **Cache engages only for batches >= the op-offload threshold (~32
  tokens).** Short prompts and single-user decode (batch 1) run
  CPU-expert MUL_MAT_ID on the CPU backend; the scheduler hook only
  fires on the GPU split path. Cache benefit is prefill/batched-decode
  only under current upstream gating.
- **Throughput: neutral here.** prompt ~130-210 t/s and gen ~22 t/s
  with and without cache; hit ratio only 9.6% — 1 GiB (441 slots)
  thrashes against this model's expert working set. Laguna's gains do
  not generalize to every geometry; budget sizing matters.
- Metrics, /cache/reset, and the modelctl web telemetry + reset proxy
  all verified end-to-end.
