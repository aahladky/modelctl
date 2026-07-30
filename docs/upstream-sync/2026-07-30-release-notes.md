# Release notes — llama.cpp runtime synced to upstream 2026-07-30

**Pinned commit:** `2dbf9480 15d3476ce94e3fe0c3b38b941a32a685` (branch `feature/sycl-moe-expert-cache`, based on upstream `origin/master@9b2a08881`)
**Previous pinned state:** `e7af6cf1996cb3d850963213b043d876ffe959ea`, preserved as tag `moe-cache-pre-upstream-2026-07-30`
**Release channel:** experimental

## What changed

The MoE expert-weight transfer cache has been re-ported onto current
`llama.cpp` upstream, closing a 348-commit gap. This was a from-scratch
reimplementation against current upstream code (not a merge), so upstream
changes since the feature's original base are now included, notably:

- SYCL fused top-k MoE expert selection (`efb3036c1`)
- A revised offload-to-weight's-backend scheduling rule (`dee2a846b`)
- The server's `common_memory` abstraction for KV-cache lifecycle (`ee3d1b54c`)
- Battlemage-specific SYCL tuning (relevant to this hardware's Arc Pro B70 / Arc B580)
- The CLI's move to an HTTP-based implementation (`c264f65ff`)

## Architecture corrections made during this sync

- The cache's server-to-backend wiring no longer uses weak global symbols;
  it goes through a versioned backend-registry procedure API
  (`ggml_backend_moe_cache_configure_v1` and siblings), which now works
  correctly under `GGML_BACKEND_DL=ON` (dynamic backend loading) — confirmed
  via a build with no static SYCL link at all still correctly reporting and
  using the cache.
- Capability reporting is now backend-derived (queries whether the proc
  actually exists) rather than inferred from compile-time flags.
- Cache ownership moved from a raw one-pointer-per-device array (a
  use-after-free hazard with two contexts on one device) to a ref-counted,
  mutex-guarded registry.
- Fixed a real bug where `--moe-cache-prefill-admission` was silently a
  no-op.
- Fixed a missing weak-symbol guard on `--moe-hybrid-mode` that would very
  likely have broken CPU-only builds' link step.

## Known experimental limitations

- **The cache does not accelerate plain single-token interactive decode.**
  It only engages during prompt processing and continuous batching large
  enough to cross the SYCL backend's general 32-token offload threshold.
  This is a real architectural limitation, not a bug — a proper fix needs
  runtime hit/miss-aware dispatch (planned as Phase G), not a scoped patch.
- **True hybrid CPU-miss execution does not exist.** `moe_hybrid_cpu_miss`
  correctly reports `false` everywhere; the hybrid dispatch code is
  present but unreferenced scaffolding for that future work.
- **MTP/draft-model coexistence with the cache is untested** — deferred to
  real-hardware acceptance work, not validated by this sync.
- A **separate, pre-existing correctness bug** was found in general
  small-batch SYCL offload (unrelated to this feature): forcing
  `GGML_OP_OFFLOAD_MIN_BATCH` below its default of 32 produces incorrect
  output on this architecture/hardware. Do not lower that setting until
  it's independently root-caused.
- The runtime's self-reported `build.commit` field is always empty;
  binary identity for this promotion was verified via an external content
  hash, not runtime self-report.

## Full evidence trail

See `docs/upstream-sync/2026-07-30-acceptance-report.md` for the complete
task-by-task record and links to every supporting document.
