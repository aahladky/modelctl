# P1 — madvise WILLNEED/DONTNEED per-expert SSD-tier management

Session: 2026-08-01, autonomous (daytime kickoff, bench window closed).
Fork branch: `agent/P1` off `feature/sycl-moe-expert-cache` (54da29dc8).
Superproject branch: `agent/P1` off `staging` (f9479a4).
Primary source (WebFetched per protocol): llama.cpp issue #20757 — the
design sketch names `POSIX_MADV_WILLNEED` per expert after each decode
step (pattern at the model-load site in `llama-mmap.cpp`) and
`POSIX_MADV_DONTNEED` on tier eviction, estimating ~60 LOC, and notes
DONTNEED does not exist in the tree.

## What was built

Advice flows through three layers, each with one job:

1. **`src/llama-mmap.cpp` — the bridge and the safety boundary.** A
   process-global registry tracks every live mapped fragment (registered
   at `mmap`, updated by `unmap_fragment`, removed — under the same
   mutex, before `munmap` — at teardown). The advise bridge refuses any
   range not wholly inside a registered fragment, rounds WILLNEED starts
   down (fragment starts are page-aligned) and DONTNEED inward (a page
   shared with a neighbouring tensor is never dropped), and issues the
   syscalls. DONTNEED is raw `madvise(MADV_DONTNEED)` on Linux because
   glibc's `posix_madvise` deliberately ignores `POSIX_MADV_DONTNEED`;
   that is safe here precisely because the registry guarantees the range
   is a read-only file-backed mapping, where MADV_DONTNEED only drops
   PTEs and a refault re-reads identical bytes. The bridge is exposed to
   ggml via `ggml_backend_moe_set_mmap_advise_fn` (fn-pointer idiom,
   same shape as the existing scheduler hooks).
2. **`ggml/src/ggml-sycl/moe-cache.{hpp,cpp}` — policy.** The cache
   batches, per step: the host ranges miss-path staged copies read
   (every miss outcome — promote, hybrid skip, fallback — reads those
   host bytes), and the origins of evicted projections (captured at the
   eviction site before `proj_origin[]` is destroyed). At step end the
   batches are swapped out under the mutex and fired outside it:
   DONTNEED first, skipping any range that overlaps a range also used
   this step (it is about to be re-read), then WILLNEED over the
   coalesced used ranges (consecutive experts of one tensor merge into
   one call). Batches are capped at 4096 entries per step with a
   `advise_dropped` counter — dropped loudly, never silently. `reset()`
   clears batches without advising (administrative, not pressure — a
   `/cache/reset` between bench conditions must not drop the page
   cache).
3. **`ggml/src/ggml-backend.cpp` + `ggml-sycl.cpp` — the step
   boundary.** A new scheduler step-end hook fires once per completed
   `compute_splits` pass (one ubatch); the SYCL backend walks the
   per-device cache registry and flushes. Abandoned graphs do not
   flush — the existing abandon hook now also drops the advice batches.

Opt-in: `GGML_MOE_CACHE_MMAP_ADVISE=1` (same idiom as
`GGML_OP_OFFLOAD_MOE_MIN_BATCH`). Capability probe reports
`features.moe_cache_mmap_advise` (true where the cache is compiled and
the platform is POSIX; no schema bump — modelctl treats features
additively and gates schema behavior on `>= 3`). Prometheus:
`moe_cache_advise_willneed_total`, `moe_cache_advise_dontneed_total`,
`moe_cache_advise_dropped_total`.

## Scope decisions (deviations from the one-line spec, with reasons)

- **"Experts just used" = the step's cache *misses*, not hits.** A hit's
  bytes are device-resident; WILLNEED on its host range would pull NVMe
  reads for pages nothing will read while the slot holds them — on a
  31 GB box that is page-cache pollution, the exact failure mode the
  landscape doc's pollution rule warns about. Miss ranges are the bytes
  the next steps will actually fault.
- **DONTNEED fires on eviction-for-repurpose only,** not on `reset()`
  (administrative) and not on unload (munmap already releases).
- **No behavior change when resident, enforced twice:** the feature is
  opt-in (default off), and even when on, the bridge no-ops for
  non-mmap pointers, `--no-mmap` loads never register a bridge, and
  Windows never registers one.
- **The engagement condition is inherited from the cache.** The hook
  only sees staged copies above the offload threshold, so this feature
  acts on the batch/prefill path (and threshold-lowered decode), not on
  routed experts statically pinned to CPU at batch 1. That matches the
  gate's workload — the ornith batch bench — and the owner's
  batch-first framing of the SSD tier (RQ6). Documented in
  modelctl/docs/runtime/moe-cache.md.
- **In-flight async H2D copies vs DONTNEED:** staging copies are queued
  on the in-order device queue and may still be running at step end. A
  dropped page refaults with identical contents (read-only file
  backing), so this is a bounded perf hiccup, never a correctness
  hazard. Noted in the header contract.

## Correctness evidence

- `test-moe-cache` (31 existing cases + 7 new advise cases): batching,
  coalescing, DONTNEED-before-WILLNEED ordering, same-step-use
  suppression, reset/abandon drop semantics, off-by-default, cap
  accounting, stats_json fields. RESULT: **pass** on the SYCL build,
  the CPU-only build, and the clang ASan/UBSan build.
- `test-mmap-advise` (new, Linux): real `llama_mmap` over a temp file
  with a page-odd tail; asserts via `/proc/self/smaps` Rss that
  DONTNEED drops exactly the fully-covered pages, inward rounding never
  drops a shared page, ranges outside the mapping (or spanning an
  unmapped fragment, or after teardown) are refused, and — the safety
  contract — DONTNEED through the bridge never touches anonymous
  memory (0xAB-filled heap pages remain intact; MADV_DONTNEED would
  have zeroed them). RESULT: **pass** on all three builds. (One test
  bug caught and fixed during bring-up: a probe range that contained no
  complete page was expected to drop one — the no-op was the correct
  behavior.)
- `test-moe-hybrid`: unchanged paths; rerun as regression. RESULT:
  **pass** on all three builds.
- Capability probe: SYCL build reports `moe_cache_mmap_advise: true`;
  CPU-only build reports it **false** (the ci/checks.sh truthfulness
  invariant holds with the new key). RESULT: **pass**.
- modelctl suite: full run under the project venv (worktree source,
  main checkout's .venv interpreter). RESULT: **Ran 1104 tests, OK
  (skipped=11)**.

## Gate status

Unit/suite portion: **green** (fork commit f4d390349). The performance
leg of the gate — ornith-397b batch bench, three-condition protocol,
>= +10% over the 0.37 tok/s static baseline — **requires the bench
window (02:00–06:00)** and was not run in this daytime session
(kickoff: bench window closed, nothing may contend for SYCL0); the
item is `gated (bench window)`. No merge to staging until the full
gate is met. Promotion-time follow-up: add `test-mmap-advise` to
ci/checks.sh sections 4 and 5 when the submodule pin bumps (adding it
now would fail checks.sh against the current pin, which lacks the
target). Bench notes
for the window session: the deployed ornith profile is static-pinned
(cache rolled back), so the advise conditions must run a cache-enabled
cell (the manifest's 50.5%-hit configuration) with
`GGML_MOE_CACHE_MMAP_ADVISE=0/1` as the A/B axis, plus the static
baseline for the +10% comparison; use
`modelctl_acceptance.run_matrix()` per moe-cache-testing.md §7, and
watch `moe_cache_advise_*_total` to confirm engagement before trusting
any number.
