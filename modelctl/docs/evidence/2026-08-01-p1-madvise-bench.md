# P1 madvise bench — ornith-397b, 2026-08-01

Raw numbers only. Full per-condition detail (commands, environments,
token arrays, 5 s NVMe samples, metric scrapes, system snapshots) in
[2026-08-01-p1-madvise-bench.json](2026-08-01-p1-madvise-bench.json).

## Setup

- Fork commit `f4d390349ff7be5a36f6945b1cb136bd82e857ca` (P1: mmap-tier
  madvise management). `build-sycl` rebuilt incrementally from it at
  ~12:45 EDT before any condition — the prior build (04:47) was from
  `54da29dc8` and predated the P1 commit. Post-rebuild capability probe:
  `moe_cache_mmap_advise: true`, schema 3.
- Model: ornith-397b (Ornith-1.0-397B UD-IQ4_NL, 182.6 GiB, 5 shards)
  on `/home` = `nvme1n1` (WD_BLACK SN850X 1TB).
- Protocol per `modelctl/docs/runtime/moe-cache-testing.md`: fixed
  prompt (in the JSON), greedy (temperature 0), seed 42,
  `cache_prompt=false`, 32-token warmup, 128-token measured decode,
  fresh server per condition on port 18131, health polled at 1 Hz from
  launch. Cache conditions: engagement verified after warmup
  (`llamacpp:moe_cache_learning == 0` AND `misses_total > 0`), then
  `POST /cache/reset` so the measured run's counters are clean.
- Commands assembled through modelctl's canonical launch path
  (`modelctl_launch.launch_command_for_profile`) from the live profile;
  cache conditions used an in-memory profile override only — the saved
  profile, artifacts, and llama-swap config were never modified.
- No iostat binary on the host: NVMe MB/s from
  `/sys/block/nvme1n1/stat` sector deltas, 5 s samples during decode.
- llama-swap: zero resident models throughout; no evictions performed.
- Concurrent load: three orphaned tiny-moe llama-server test fixtures
  (`build-sycl-int`, ports 18209/18211/18213) ran throughout, holding
  ~2.3 GB VRAM on SYCL0. Left untouched, recorded here. Load averages
  per condition are in the JSON snapshots.
- Starting state 12:49: SYCL0 31.7/34.2 GB free, SYCL1 12.8/12.8 GB
  free, loadavg 0.88, MemAvailable 22.6 GiB, ornith page-cache
  residency 12.0%.

## Conditions

| | config | env |
|---|---|---|
| C1 | saved profile: static planner config, moe_cache off, `-ot blk\.[0-5]\...=SYCL0, blk\.[6-7]\...=SYCL1, ffn_.*_exps=CPU` (+ `--metrics` appended for scrape only) | — |
| C2 | promotion-report cache+hybrid config: `--moe-cache-bytes SYCL0=18253611008,SYCL1=5368709120 --moe-cache-policy slru --moe-cache-admission-misses 2 --moe-cache-prefill-admission off --moe-hybrid-mode on`, `-ot ffn_.*_exps=CPU` | `GGML_OP_OFFLOAD_MOE_MIN_BATCH=1` |
| C3 | identical to C2 | C2 env + `GGML_MOE_CACHE_MMAP_ADVISE=1` |
| C4A | C2 flags minus `--moe-hybrid-mode` (pure transfer cache) | `GGML_OP_OFFLOAD_MOE_MIN_BATCH=1` |
| C4B | identical to C4A | C4A env + `GGML_MOE_CACHE_MMAP_ADVISE=1` |

C2 first attempt (13:02) used the profile's static 0–5/6–7 GPU expert
pins together with the 17 GiB SYCL0 cache budget: `moe_cache: pool
allocation of 18249154560 bytes returned null`, then
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` → SIGABRT in
`ggml_backend_sycl_set_tensor_async` during warmup. The promotion
report's run had all routed experts on CPU (web job `cc745c366ac1`,
2026-08-01 00:14); C2/C3/C4 use that placement. Crash log preserved in
the failed-attempt record inside the JSON
(`conditions["C2-attempt1-oom-staticpins"]`).

## Results (measured 128-token decode)

| | gen tok/s | prompt tok/s | wall s | load s | warmup gen tok/s | NVMe avg MB/s | NVMe min–max MB/s | NVMe bytes read |
|---|---|---|---|---|---|---|---|---|
| C1 | 0.3531 | 0.7625 | 420.2 | 117.6 | 0.3912 | 1816.7 | 1245.7–1962.3 | 769,119,068,160 |
| C2 | 0.2553 | 0.5955 | 575.3 | 82.3 | 0.2916 | 1295.0 | 1112.7–1436.0 | 749,173,592,064 |
| C3 | 0.2600 | 0.5831 | 567.9 | 83.4 | 0.3021 | 1320.7 | 1160.7–1583.6 | 754,188,447,744 |
| C4A | 0.2723 | 0.6007 | 543.3 | 84.2 | 0.3175 | 1370.0 | 1287.0–1429.2 | 748,780,912,640 |
| C4B | 0.2672 | 0.4643 | 573.8 | 83.2 | 0.3092 | 1191.0 | 873.3–1428.3 | 687,246,991,360 |

Cache counters, measured run only (post-reset; SYCL0 — the only device
that creates a cache in this topology; SYCL1 budget emitted but no
cache instance):

| | hit ratio | hits | misses | H2D bytes | willneed | dontneed | dropped | evictions | promotions | fallbacks | slots used |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | no cache (moe_cache off; /metrics has no moe_cache series) | | | | | | | | | | |
| C2 | 0.45246 | 114,044 | 138,009 | 91,258,355,712 | 0 | 0 | 0 | 12,468 | 46,013 | 25,281 | 3060/3060 |
| C3 | 0.45140 | 113,802 | 138,306 | 91,624,374,272 | 120,085 | 36,584 | 22,456 | 12,527 | 46,196 | 25,317 | 3060/3060 |
| C4A | 0.45274 | 114,099 | 137,919 | 91,341,717,504 | 0 | 0 | 0 | 12,480 | 46,054 | 99,005 | 3060/3060 |
| C4B | 0.45319 | 114,225 | 137,820 | 91,338,375,168 | 119,623 | 36,437 | 22,390 | 12,483 | 46,054 | 98,903 | 3060/3060 |

Engagement: C2, C3, C4A, C4B all verified engaged before timing
(learning gauge 0, nonzero misses after warmup; geometry
`gate=1802240 up=1802240 down=2359296`, 3060 slots × 5,963,776 B).

Page cache (measured-run window): residency of the five shards moved
≤0.3 pp in every condition (C1 12.04→11.94%, C2 11.85→11.88%, C3
12.03→12.06%, C4A 11.86→11.78%, C4B 12.22→12.21%); `Cached` deltas in
the JSON snapshots.

## Correctness rail

- **C2 vs C3: token-identical** — warmup (32) and measured (128)
  sequences match exactly. The rail required by the work order passes.
- C4A vs C2: token-identical (128/128).
- C4A vs C4B (optional pair, not part of the rail): warmups identical,
  measured sequences identical through token 119, diverge at token 120
  of 128 (token id 974 vs 31733); both runs complete with
  `stop_type=limit`. Raw sequences in the JSON.
- C1 vs C2: sequences differ (different placement/execution path).

## Acceptance battery (2026-08-01 14:20–14:50 EDT)

- laguna-s2.1 anchor (`speed.py laguna-s2.1 256 3` through llama-swap,
  laguna's own pinned binary `~/src/llama.cpp-laguna`, untouched by
  this work), reference 14.20 tok/s (2026-07-31 23:49):
  - run at 14:20 (immediately after the ornith bench, loadavg ~2.9):
    12.92 / 13.81 / 13.71, avg **13.48** tok/s — 5.07% below reference,
    outside the 5% line by 0.01 tok/s. **Flagged.**
  - repeat at 14:25 (loadavg 3.42 at start): 13.39 / 13.58 / 14.18,
    avg **13.71** tok/s — 3.45% below reference, within 5%.
  - laguna unloaded through the router afterwards.
- `ci/checks.sh` (full): every check PASS — 1104 tests, CPU-only build,
  MoE cache + hybrid host tests, capability truthfulness, ASan/UBSan
  pass, layering — except the expected pre-update pin item ("manifest
  names 54da29dc8, pin is f4d390349").
- `integration-manifest.json` updated (validated pair
  `dfcebc8` + `f4d390349ff7be5a36f6945b1cb136bd82e857ca`,
  `moe_cache_mmap_advise` added to supported features), then
  `ci/checks.sh --quick`: all green including the pin/manifest checks.

## Timeline

C1 12:50–13:01, C2 failed attempt 13:02–13:20, C2 13:22–13:36,
C3 13:36–13:51, C4A 13:51–14:05, C4B 14:05–14:19 EDT. Every server
exited on SIGTERM, VRAM release verified, no orphans on port 18131
after each condition.
