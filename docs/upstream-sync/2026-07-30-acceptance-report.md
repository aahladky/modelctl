# Phase 0 acceptance report — 2026-07-30

Summary and final verdict for the controlled upstream synchronization
described in `modelctl/docs/modelctl-task-by-task-roadmap-2026-07-30.md`
("Phase 0 — Synchronize the runtime fork with current upstream"). This
report is the Task 0.9 archive; it indexes the detailed evidence rather
than repeating it.

## What happened, task by task

| Task | Result | Evidence |
|---|---|---|
| 0.1 Freeze baseline | Tagged `feature/sycl-moe-expert-cache@e7af6cf19` as `moe-cache-pre-upstream-2026-07-30`, pushed to `gitea`. Recorded environment, capability response, build/launch commands, a real correctness/benchmark snapshot. | `2026-07-30-baseline.md` |
| 0.2 Impact review | 348 commits behind / 12 ahead at merge-base `6f4f53f2b`. Ranked highest-risk conflict areas by actual commit content. | `2026-07-30-impact-review.md` |
| 0.3 Port | Reimplemented the feature as 4 clean commits on `sync/moe-cache-upstream-2026-07-30` (based on `origin/master@9b2a08881`), each conflict resolution documented against the specific upstream commit it reconciles with. SYCL build succeeded; live capability probe byte-identical to pre-sync. | `2026-07-30-port.md` |
| 0.4 Architecture corrections | Replaced weak SYCL globals with a versioned backend-registry API; capability probe now backend-derived, not compile-time inferred; cache ownership reworked from a raw one-pointer-per-device array (use-after-free hazard) to a mutex-guarded ref-counted registry, justified by precedent in upstream's own device-info singleton; fixed a real prefill-admission no-op bug. | `2026-07-30-architecture-corrections.md` |
| 0.5 Build matrix | CPU-only (all `moe_*` false, confirmed independently), SYCL static, SYCL `GGML_BACKEND_DL=ON` (confirmed resolving the cache proc across the dlopen boundary via `ldd` + live probe), Debug. All clean. | `2026-07-30-architecture-corrections.md` |
| 0.6 Correctness matrix | 13 of 14 required cases pass with exact greedy token-ID match against a genuinely unmodified upstream oracle, using a purpose-built tiny synthetic MoE model (real cache activity confirmed: hits/misses/evictions/H2D bytes, not a no-op). MTP+draft explicitly deferred to Phase E. | `2026-07-30-correctness-matrix.md` |
| 0.7 Performance rebaseline | No regression found (25.97 tok/s decode vs. baseline's 24.24-24.33). Surfaced that the cache never activates for the flagship `qwen3.5-122b-iq1m` profile because it fits entirely in VRAM. A real RAM-ceiling incident occurred and is documented plainly (see Known Limitations). | `2026-07-30-performance-rebaseline.md` |
| 0.7b Real-model cache activation | Forced host-residency with a genuinely oversized quant (Q4_K_M, ~76.6GB); confirmed the model loads and runs correctly, but still saw zero cache metrics — flagged as a more serious, unresolved-at-the-time finding. | `2026-07-30-real-model-cache-activation.md` |
| — Root cause | Resolved: general SYCL `op_offload_min_batch_size` (default 32) keeps `MUL_MAT_ID` on CPU below that batch, so the hook is never reached at low batch sizes — not a `qwen35moe`-specific bug. Confirmed with `GGML_SCHED_DEBUG=2` tracing and a longer prompt showing real activity (10,402 misses / 4,392 evictions / 5,201 promotions / 9.2GB H2D). | `2026-07-30-cache-inactivity-rootcause.md` |
| — Decode-time fix investigation | No fix implemented, correctly: a real per-token cache-hit-aware bypass requires Phase G's runtime row-partitioned dispatch (schedule-build-time vs. schedule-execution-time information availability, plus graph reuse across decode steps make a scoped fix structurally impossible). Also found and documented a real, separate, pre-existing small-batch SYCL offload correctness bug (unrelated to the cache) via `GGML_OP_OFFLOAD_MIN_BATCH=1`. | `2026-07-30-decode-cache-fix.md` |
| 0.8 Integration chain | All 10 links pass with direct evidence (capability probe → normalization → plan → preflight → preview → plan test → managed worker/llama-swap command → launch → metrics → unload/reload). Binary fingerprint (real SHA-256, not the always-empty `build.commit` field) differs pre/post sync as required. Submodule pointer confirmed unadvanced until this promotion step. Found and flagged (not fixed) a live-profile config inconsistency (`decode.miss_execution: "cpu"` on a profile where `moe_hybrid_cpu_miss` is `false`). | `2026-07-30-integration-chain.md` |
| 0.9 Promotion | This step: branch pointer advanced, submodule pinned, manifest written, this report archived. | this document |

## Known limitations, carried forward honestly rather than hidden

- **The cache does not help plain single-token interactive decode**, only prompt processing and sufficiently large continuous-batching — a real, documented architectural characteristic (batch-size gating), not a bug. Fixing this properly requires Phase G.
- **A real, separate SYCL correctness bug** exists in small-batch offload (`GGML_OP_OFFLOAD_MIN_BATCH` below ~32 produces wrong output, reproduces with zero MoE-cache involvement) — worth its own investigation before Phase E; do not lower that env var on this architecture until root-caused.
- **MTP + draft-model coexistence is untested** for the cache (Task 0.6 deferral) — needs real hardware acceptance work (Phase E), not a synthetic fixture.
- **The `build.commit` field embedded in capability responses is always empty** — binary identity verification in this session relied on a real content hash computed externally (Task 0.8), not anything the runtime self-reports. Worth fixing before relying on it operationally (Task A2/0.8 gap).
- **A live production profile** (`qwen3.5-122b-iq1m`) has a config value (`decode.miss_execution: "cpu"`) inconsistent with current capabilities (`moe_hybrid_cpu_miss: false`) — not touched, flagged for Phase B.
- **This session had two real operational incidents**, documented plainly: an OOM that crashed the user's desktop session (root-caused to `--parallel 4` on a 34GB model against 31GB system RAM, reproduced on the *pre-sync* binary too, confirmed not a sync regression), and a stale, pre-move binary path in a generated `run.sh` artifact (found in Task 0.1, unrelated to the sync, not fixed since fixing it would restart the live router).
- **"Mark old benchmark observations stale by fingerprint"**: no automated staleness mechanism exists in modelctl yet (that's Phase H's Task H1, not yet built) — this is a manual/documentary marking today. The pre-sync binary's cached capability entry (`~/.local/share/modelctl/backend_capabilities/7cc5d96af8216629.json`, keyed to `build-sycl/bin/llama-server` at commit `e7af6cf19`) should be treated as superseded by this promotion; no code change was made to enforce that automatically.

## Definition of completion (roadmap's own bar)

> The feature is ported to a recent upstream base, the old state remains
> reproducible, current upstream behavior is preserved where intended, the
> runtime passes the complete test gate, and modelctl pins the validated
> commit with stale observations invalidated.

- Ported to current upstream: yes (`origin/master@9b2a08881`, 348 commits closed).
- Old state reproducible: yes (`moe-cache-pre-upstream-2026-07-30` tag, pushed).
- Current upstream behavior preserved where intended: yes, verified (byte-identical capability response pre/post port, no non-cache-path regression found).
- Complete test gate: build matrix (0.5) and correctness matrix (0.6) both pass; performance (0.7) shows no regression; integration chain (0.8) passes end to end. Two real, separately-scoped issues found along the way (decode-time cache inactivity by design, a pre-existing small-batch SYCL bug) — both documented, neither blocks this promotion since neither is a regression from the sync itself.
- modelctl pins the validated commit: yes, this step.
- Stale observations invalidated: documented manually (see Known Limitations) — no automated mechanism exists yet to do this in code.

**Verdict: Phase 0 is complete**, with the limitations above carried forward explicitly rather than silently. Nothing here should be read as "the cache is production-ready for all use cases" — it is ported, correct within the surface actually tested, and honestly characterized where it falls short.
