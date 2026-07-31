# Runtime promotion report — 2026-07-31

Promotes `feature/sycl-moe-expert-cache@724f705c1` into the umbrella
repository, replacing the pin at `2dbf94801`.

**Why this was needed.** The previous pin predated every Phase F fix. A
clone of the umbrella repo built a runtime with the shared gate/up/down
admission counter, `cpu_expert_calls`, duplicate Prometheus metadata and
unpooled slot allocation — while the docs, the manifest and every
measurement described a runtime five commits further on. The drift was
also invisible: `manifest_status()` read the pin from `git submodule
status`, which reports the *checked-out* commit, so the check compared a
value against itself and could never fire. That is fixed (`c0676b7`).

## What is being promoted

| commit | contents |
|---|---|
| `0b4b3a2b7` | F3/F4/F7 — per-projection admission, prefill wiring, metric renames |
| `9d175c28a` | F6 — pooled allocation, per-projection geometry |
| `ad4903d40` | F8 — policy/device split; the cache tests build and run host-only |
| `b43bb7aa6` | G2 — corrected hybrid partition representation |
| `724f705c1` | Per-op-type offload threshold + `moe_offload_threshold_control` |

## Gate 0.5 — build matrix

| form | result |
|---|---|
| CPU-only, Release | clean; every SYCL/cache feature reports **false** |
| SYCL, cache compiled, Release, `GGML_BACKEND_DL=OFF` | clean |
| SYCL, cache disabled at runtime | clean (sweep condition E) |
| SYCL, `GGML_BACKEND_DL=ON` | clean; all cache procedures resolve across the dlopen boundary |
| Debug / assertions | clean; `test-moe-cache` and `test-moe-hybrid` both pass under assertions |
| Release used for benchmarks | `build-sycl-f`, the binary every measurement used |

`GGML_BACKEND_DL=ON` additionally requires `GGML_NATIVE=OFF` and
`GGML_CPU_ALL_VARIANTS=ON`; upstream refuses the combination otherwise.

## Gate 0.6 — correctness matrix

Run against a genuinely unmodified `origin/master@9b2a08881` oracle built
with matching flags. Cache disabled, admission thresholds 1/2/3, SLRU and
LRU eviction under a forced 2-slot budget, prefill admission on and off,
reset while loaded, and two contexts on one GPU — every case
token-identical to the oracle, with real cache activity (172 evictions,
174 promotions) rather than an inert cache.

On the real model, all four conditions of the 122B-A10B IQ1_M
default-vs-min-batch-1 comparison produced byte-identical greedy output on
both the fork and the oracle.

Not covered: unequal-projection geometry rejection (the fixture has equal
projections), MTP + draft coexistence (deferred to Phase E originally).

## Gate 0.8 — integration chain

All thirteen links pass: capability probe → normalization → plan
generation → preflight → launch command built and valid → distinct binary,
environment and command fingerprints → binary fingerprint differs pre/post
sync → cache flags present → command identity stable across rebuilds →
llama-swap entry rendered carrying the same flags including all four
`--moe-cache-*` → launch → metrics → unload and reload.

**Task B2's acceptance holds:** a stock upstream binary with every cache
setting enabled emits **zero** cache flags and returns an invalid command
with a structured `cache_feature_unsupported` refusal.

## Gate 0.7 — performance

Not a pass/fail gate, and deliberately not blocking. The relevant
measurement is `modelctl/docs/moe-offload-threshold-q4km-2026-07-30.md`:
on Q4_K_M, decode is 2.5× faster than the default when routed MoE ops may
offload at batch 1.

## Known limitations carried forward

- A `GGML_BACKEND_DL=ON` build reports `build.dynamic_backends: false` in
  its capability response although the backend is dlopened and every cache
  procedure resolves. The field is wrong; the functionality is not. Same
  class as the always-empty `build.commit`.
- The MoE cache initialised on **device 0 only** in every condition of the
  Q4_K_M sweep, despite experts living on SYCL1. Unresolved.
- `--moe-cache-bytes` is a single uniform per-GPU budget, so per-device
  budgets collapse to their maximum. The tier planner reserves the
  per-device figure it was given, which can be *less* than the runtime
  would allocate — latent, and worth closing before a differently shaped
  model hits it.
- `moe_hybrid_cpu_miss` remains false. G3–G7 are unimplemented and the
  capability must not lead the implementation.

## Verdict

**Promoted.** Gates 0.5, 0.6 and 0.8 pass against `724f705c1`; the
manifest and submodule pointer now describe the runtime that every
measurement in the docs was taken on.
