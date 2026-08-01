# Promotion report, 2026-08-01: audit fix pass + cache observability

Promotes the pair produced by the 2026-07-31/08-01 full-project audit and
its fix pass:

- modelctl `3df7a60` (web console acquisition chain, cancel semantics,
  error visibility; probe env cache warm-before-promote; service-layer
  write serialization; docs)
- llama.cpp `54da29dc8` (`feature/sycl-moe-expert-cache`: report
  still-learning MoE caches in stats, `llamacpp:moe_cache_learning` gauge)

Previous validated pair: `b5a0c44` + `9accaf4fb`. It stays the rollback
target until this one is superseded.

## Why this promotion exists

The audit found the Add Model wizard broken end-to-end for HF repos,
advisory job cancellation, unlocked service-layer writes, a probe env
cache that dropped candidates on cold promotion, and a cache
observability hole: a still-learning MoE cache was invisible in
/metrics, indistinguishable from "no cache configured". The fixes span
both repos, so the pair moves together.

## Verified on this machine

Hardware: 2x Intel Arc (SYCL0 = B70 32GB, SYCL1 = B580 12GB), 31GB RAM,
oneAPI 2026.1, `build-sycl` rebuilt from `54da29dc8` with icx/icpx.

| Check | Result |
|---|---|
| `test-moe-cache` (host) | pass |
| modelctl suite (`unittest discover`) | pass (exit 0, full suite) |
| Console deployed + live walk | all pages 200; wizard inspect renders the repo that 500'd pre-fix |
| laguna-s2.1 (54.7 GiB MoE, static split + CPU experts) | 14.2 tok/s generation via console -> job -> speed.py -> llama-swap |
| ornith-397b (182.6 GiB, tier-4 SSD mmap) | 0.37 tok/s static (planner's config; NVMe-bound, matches its own prediction) |
| Cache/hybrid full pipeline on ornith (`GGML_OP_OFFLOAD_MOE_MIN_BATCH=1`, `--moe-hybrid-mode on`) | engages as designed: 50.5% hit ratio, 57% expert rows on GPU, 182 GB H2D avoided -- and still loses to static pinning (0.27 vs 0.37); rolled back |
| Learning gauge, live | `moe_cache_learning{SYCL0} 1` sampled during load; flips to 0 at finalization with counters registered |

## Findings recorded, not fixed here

- The scheduler's offload pass always selects the first capable backend,
  so only SYCL0's cache budget ever engages; a SYCL1 budget is dead
  weight in this topology.
- Static expert pinning beats the transfer cache whenever the expert
  working set vastly exceeds the cache budget; the cache's expected
  sweet spot on this box is the 50-70 GiB MoE class.
- `speed.py` (outside this repo, `~/services/llama-swap/`) now scales
  its request timeout with the token budget; the fixed 300 s could
  never measure the SSD tier.

## Not covered

- Full acceptance throughput matrix (`modelctl_acceptance.py` cells)
  not re-run for this pair.
- ASan/UBSan runs not repeated (the fork delta is 16 lines of stats
  emission; no allocation or kernel paths touched).
- K-quant determinism (`run_kquant.sh`) not repeated for the same
  reason.
