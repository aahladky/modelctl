# P5 pin landing — ornith-397b (applied, pins-only)

2026-08-01, console phase 2 order step 5. Planning inputs were
backfilled from the placement's era, the dry replan classified as
pins-only (gate kind "pins", requires_accept False), and the plan was
applied with `place --tiers --apply --no-hermes --no-router-restart`.
The running instance was NOT reloaded; it picks the pins up at its next
natural reload. laguna-s2.1's replan classified structural and was NOT
applied — its diff is in moe-review/replan-diff-laguna-s2.1/.

## Recorded planning inputs (profile.planning.inputs)

- version: 1
- ram_available_bytes: 23684980736 (22.06 GiB — MemAvailable_kB 23129864
  from docs/evidence/2026-08-01-p1-madvise-bench.json starting_state,
  measured the day this placement was last written)
- vram_limit_pct: 90 (defaults.json)
- primary: SYCL0
- inventory: SYCL0 "Intel(R) Arc(TM) Pro B70 Graphics" 34242297856 B,
  SYCL1 "Intel(R) Arc(TM) B580 Graphics" 12809404416 B
- hw_settings.devices: SYCL0 {memory_bandwidth_gbs_override 608.0,
  enabled true}, SYCL1 {456.0, enabled true} (hardware.json)
- capabilities.features.moe_cache_per_device_budgets: true
  (moe-serving fork f4d390349, capability schema 3)

## Replan verdict (from stored inputs, source "stored")

- tier: 4
- gate: kind "pins", requires_accept False
- change (verbatim): extra pins: [] ->
  [('blk\\\\.[0-5]\\\\.ffn_.*_shexp', 'SYCL0'),
   ('blk\\\\.[6-7]\\\\.ffn_.*_shexp', 'SYCL1'),
   ('ffn_.*_shexp', 'SYCL0')]
- tensor_split, routed-expert rules, -ngl/mmap/device/fit and all other
  flags: unchanged.

## Admission record

- fits: true
- degradations: []
- warning carried by the plan: "model exceeds GPU+RAM: the CPU-resident
  portion streams from SSD via mmap -- expect low single-digit tok/s on
  cold cache." (pre-existing tier-4 reality, not introduced by the pins)

## Applied config.extra (after)

    --fit off --device SYCL0,SYCL1 -ot blk\\.[0-5]\\.ffn_.*_shexp=SYCL0,blk\\.[6-7]\\.ffn_.*_shexp=SYCL1,ffn_.*_shexp=SYCL0,blk\\.[0-5]\\.ffn_.*_exps=SYCL0,blk\\.[6-7]\\.ffn_.*_exps=SYCL1,ffn_.*_exps=CPU --no-warmup --ubatch-size 128

(doubled-backslash on-disk convention; single-backslash after JSON
decode, plain regex after shlex)

## Deployment state

- profile saved, artifacts regenerated (run.sh carries the pins),
  llama-swap config.yaml rewritten with backup
  config.yaml.bak.20260801-175906
- llama-swap service NOT restarted (pid 2587028, started 04:51:12,
  unchanged) — pins take effect at the next natural reload
