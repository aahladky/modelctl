# Bench infrastructure + deploy verification — 2026-08-02

Raw numbers only. No benchmark was run in this session; the maiden runs
are queued for the quiet window.

## Machine state during this session

- 00:31 EDT: loadavg(1m/5m/15m) **0.13 / 0.44 / 0.98**, MemAvailable
  **19 GiB** of 31 GiB, swap 59 GiB free of 71 GiB.
- llama-swap (`:9292`): `{"running":[]}` — zero resident models
  throughout. Not restarted, not reconfigured.
- Console `:9293` listening throughout. Not restarted.
- No orphaned test fixtures on the 18xxx scratch ports (contrast the P1
  bench, which ran with three).
- SYCL0 Intel Arc Pro B70, 32656 MiB total / **30776 MiB free**;
  SYCL1 Intel Arc B580, 12216 MiB total / **12178 MiB free**.
- 00:57 EDT, during `ci/checks.sh`: loadavg(1m) **3.07**, MemAvailable
  16.2 GiB. That is this session's own test/build load, recorded because
  it is the concurrent load any measurement taken now would have seen.

## Step 0 — the validated pair was already deployed

`llama.cpp/build-sycl` did not need rebuilding. The order allows for
this ("If already deployed, verify and say so").

| check | result |
|---|---|
| fork pin (superproject `HEAD:llama.cpp`) | `85b7e6556b6b83026d1a17df2635bc1173db1f97` |
| submodule working tree | same commit, clean |
| pin commit date | 2026-08-01 21:55:10 −0400 |
| `build-sycl/bin/llama-server` mtime | 2026-08-01 22:09:05 −0400 (after the pin) |
| `make -q llama-server` in `build-sycl` | **rc=0** — target up to date, an incremental rebuild is a no-op |

The 2026-08-01 determinism record's closing note ("`llama.cpp/build-sycl`
was deliberately not rebuilt") is superseded: it was rebuilt at 22:08–22:09
that evening, after that record was written.

### Capability probe (raw, through `modelctl_capabilities._probe_raw`)

```
schema: 3
build: {commit: 85b7e6556, number: 10217, compiler: IntelLLVM 2026.1.0,
        target: Linux x86_64, build_type: Release,
        backends: [SYCL, CPU], dynamic_backends: false}
features:
  moe_weight_transfer_cache      true
  moe_hybrid_cpu_miss            true
  moe_cache_metrics              true
  moe_cache_prefill_policy       true
  moe_cache_reset                true
  moe_cache_prefetch             false
  moe_cache_mmap_advise          true
  moe_offload_threshold_control  true
  moe_cache_per_device_budgets   true
constraints:
  moe_cache_backend              SYCL
  moe_cache_min_batch            32
  moe_cache_supported_projections [gate, up, down]
  moe_hybrid_supported_archs     [any]
  moe_hybrid_supported_quant     [any_with_dequantizer]
  moe_hybrid_can_overlap         true
```

### Determinism default (no `GGML_SYCL_DETERMINISTIC` set in the env)

From the deployed binary's own SYCL init block, via
`llama-bench -v` on the tiny-moe fixture:

```
GGML_SYCL_DNNL: yes
GGML_SYCL_ENABLE_DNN: 1
GGML_SYCL_DETERMINISTIC: 1
GGML_SYCL_FA_ONEDNN: 1
GGML_SYCL_FA_ONEDNN_MAX_KV: 0
```

Both required properties hold: the binary reports the pinned commit, it
probes `moe_cache_mmap_advise: true`, and the determinism attribute
defaults to on.

Laguna's own pinned binary (`~/src/llama.cpp-laguna`) was not touched,
not probed and not run.

## Defect found while verifying step 0

The binary has emitted `moe_cache_mmap_advise` since `f4d390349`, and
`integration-manifest.json` lists it under `supported_runtime_features`.
`modelctl_capabilities.normalize_capabilities()` built its canonical
feature set from a fixed whitelist that **did not contain the key**, so
the flag was dropped on every probe: `_raw_features` carried it, the
canonical `features` dict did not, and every modelctl caller asking
whether the runtime had the feature — including the deploy check whose
job is to confirm it landed — was answered "no".

Fixed in all three code paths (schema 3, schema 1, and the fail-closed
probe-failure set), with `supports_mmap_advise()` added alongside the
other `supports_*` helpers, and a test asserting that every feature the
manifest names is one modelctl can represent.

**Consequence to expect:** the capability fingerprint of this binary
moves `db3f81a8c2fec07f` → `89706a136eed7f5b`, because the feature set
genuinely gained a key. Stored observations keyed on the old fingerprint
will read as stale and their plans will be re-tested. That is the
intended behaviour of the staleness rule, not a side effect to suppress.

## What was built

| module | what it is |
|---|---|
| `modelctl_load.py` | `/proc`-only load sampling, a background `LoadRecorder`, and a comparability check. An unreadable field stays unreadable into the summary; never 0.0. |
| `modelctl_paired.py` | Paired benchmarking: alternating schedule, per-pair deltas, two-sided exact sign test, per-pair load comparability. No threshold, no verdict, no winner field. |
| `modelctl_anchors.py` + `anchors.json` | Anchor registry. Fingerprint = build commit + profile hash + env hash + driver, compared field by field. `void` and `always_run` are separate from staleness. |
| `modelctl_nightlane.py` (extended) | Window gate (llama-swap idle AND load below ceiling, both fail closed), dispatch onto the job store's benchmark lane, per-run load traces, evidence filing with one-line summaries, and the `GGML_OP_OFFLOAD_MIN_BATCH ≥ 32` floor enforced before anything runs. |

Documented in
[../runtime/benchmarking.md](../runtime/benchmarking.md).

## Anchor registry state

Seeded from `2026-08-01-onednn-determinism.json` by script, not retyped.

| anchor | value | runs | state |
|---|---|---|---|
| `c1-static-122b` | 6.2365 tok/s | 5 | **void** — load-contaminated |
| `c2-cache-122b` | 5.0413 tok/s | 10 | **void** — load-contaminated |
| `c3-cache-hybrid-122b` | 5.1646 tok/s | 10 | **void** — load-contaminated |
| `laguna-s2.1-canary` | 14.20 tok/s | — | not void, `always_run` |

Void reason recorded on all three: taken across a battery whose
loadavg(1m) ran 2.63–17.15 (mean 8.99) over 442 samples, so the
conditions were not measured under one machine. The values and every raw
run are **kept**, not deleted.

Fingerprints on the three carry `build_commit: 85b7e6556` and leave
`profile_hash`, `env_hash` and `driver` empty — those were never
recorded for those runs, and an unrecorded field reads as unrecorded
rather than as a match. The recorded `env_overrides` and the binary path
(`moe-fork-det/build-sycl-det/bin/llama-server`) are kept in `extra`.

`effective_cache_budget_bytes_per_run` is `null` on all three with the
note that the 2026-08-01 battery recorded only the *requested*
`--moe-cache-bytes`. Closing that gap is part of the re-anchor job.

A battery planned against this machine today runs all four anchors:
three void, one exempt, none reusable.

## Night-queue state

| job | enabled | mode | size | arms |
|---|---|---|---|---|
| `ornith-rpc-criterion-2026-08-02` | **no** | block | — | 2 |
| `qwen122b-remote-experts-hypothesis-2026-08-02` | **no** | block | — | 2 |
| `determinism-cost-c1-static-2026-08-02` | yes | paired | 5 pairs | 2 |
| `determinism-cost-c2-cache-2026-08-02` | yes | paired | 5 pairs | 2 |
| `re-anchor-c1-c2-c3-2026-08-02` | yes | battery | 5 runs/arm | 3 |
| `sdpa-reproducibility-2026-08-02` | yes | reproducibility | 3 runs | 1 |

The two RPC pairs are **byte-identical** to how the RPC session left
them. The registration appended 195 lines and deleted none; the new
schema fields (`mode`, `pairs`, `runs`, `metric`) serialize only when
they carry a non-default value, so re-saving the registry cannot make an
untouched job look edited.

Nothing was dispatched. The lane starts nothing on import, and no
console restart was performed or authorized.

## The queued runs

**4a — determinism cost, paired.** Two jobs, C1-static and C2, five
pairs each, `GGML_SYCL_DETERMINISTIC` 1 vs 0, delta = det-off − det-on.
Replaces the void −15.20% / −8.01% / −14.87% block figures. Pair order
alternates; a pair whose arms did not see the same machine is reported
with that fact attached, not dropped; no arm is re-run to break a tie.

**4b — re-anchor C1/C2/C3.** Five runs per condition on the deployed
binary, hit ratio and effective cache budget recorded *per run*, each
anchor written with its fingerprint.

**4c — SDPA probe.** `run_reproducibility.py --shape sdpa-heavy`, a
~2.3k-token prompt (9195 chars, ~1526 words) built by cycling three
distinct sentence templates rather than repeating one, so the attention
pattern does not degenerate. Three fresh servers, pre-registered pass
condition max abs Δlogprob **exactly 0.0** at the logit level.

The shape is committed on fork branch **`agent/sdpa-probe`**
(`e73d47680`), not on the pinned commit, and the submodule working tree
was returned to `85b7e6556`. Reason: it is a script-only change that
never touched the binary, and advancing the pin for it would re-stamp
`validated_llama_commit` with a commit that passed no hardware
acceptance. The job's arm records `_requires_fork_branch:
agent/sdpa-probe` and its note says the runner must refuse rather than
fall back to the default shape, whose ~20-token prompt answers a
different question.

## Checks and tests

The modelctl suite ran exactly twice: once as a baseline before any edit,
once at the gate inside `ci/checks.sh`.

| run | result | wall |
|---|---|---|
| baseline, before any edit | 1286 passed, 11 skipped | 44.23 s |
| gate, inside `ci/checks.sh` | **1424 passed, 11 skipped** | 44 s |

138 tests added. Cumulative test wall-time across the session, including
the targeted single-file runs during iteration: **~100 s**, against the
10-minute tripwire.

`ci/checks.sh` (full), once, at the gate — **every check PASS**, 47 s
wall:

```
submodule pin        pin and working tree agree (85b7e6556b6b)
                     submodule URL declared: ../llama.cpp.git
                     manifest agrees with the pin
static checks        compileall / every modelctl module imports
test suite           1424 passed, 11 skipped in 44s wall
console offline build  builds offline from the vendored tree
CPU-only build       build / MoE cache host tests / MoE hybrid host tests
                     CPU-only build reports every SYCL/cache feature false
sanitizer pass       test-moe-cache, test-moe-hybrid under ASan/UBSan
layering             no leaf module imports modelctl at module level
```

The pin items are green, unlike the two prior sessions: the pin is not
being advanced here, and the SDPA script change deliberately stays off it.

Nothing died. No process was killed, no service restarted, no server
launched beyond the two read-only probes of the deployed binary
(`--modelctl-capabilities` and `llama-bench -v` on the 76 MiB tiny-moe
fixture), both of which exited 0.
