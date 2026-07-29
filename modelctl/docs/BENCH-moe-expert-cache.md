# MoE expert cache — benchmark results

## Laguna-S-2.1, 2026-07-28 (llama.cpp fork @ a64c80eee)

Config: 1 GiB cache budget, SLRU policy, admission=1.
Laguna-S-2.1 runs 19/47 expert layers on CPU (`ffn_.*_exps=CPU`).

| | prompt t/s | gen t/s |
|---|---|---|
| Cache disabled (baseline) | 4.7 | 11.2 |
| Cache enabled | 28.1 | 15.1 |
| **Δ** | **+497%** | **+35%** |

The cache speeds prompt processing ~6x by keeping hot experts resident
across repeated expert selections, avoiding redundant host→device copies.

## Caveats

- **These numbers predate the C1 slot-geometry fix** (`f0750e8a4`).
  Before that fix, up/down projection cache hits served the wrong (gate)
  weights — throughput was unaffected (identical copies), so the speed
  numbers should still hold, but output quality during that run was
  degraded. Re-validate both speed and output correctness against
  `f42f2fe4e` or later before quoting these.
- Historical note from the original run: at the time, F0 (scheduler
  hook for CPU-resident experts) was still open — the cache could only
  see GPU-resident experts through `ggml_sycl_mul_mat_id`. F0 landed in
  `67ab58096`, so a re-run should show *larger* gains: the 19
  CPU-resident layers are now cacheable too.

## Re-validation checklist

1. Same model + flags, cache off vs on (`--moe-cache-bytes 0` vs 1 GiB).
2. Output sanity: fixed prompts, compare cache-on output against
   cache-off for coherence (first run with *correct* cached weights).
3. Throughput: `llama-bench` or server timings, prompt + gen.
4. Check `moe_cache_*` metrics (`/metrics`) for hit ratio on the
   CPU-resident layers — should now be nonzero, unlike the original run.
