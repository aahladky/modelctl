# LocalAI Stage 0 — M1: cost of tracking LocalAI's llama.cpp pin — 2026-08-06

Measurement 1 of the LocalAI Stage 0 spike. The question it answers: if the
moe-serving stack adopts a LocalAI fork as its front end, every LocalAI release
taken drags `grpc-server.cpp`'s pinned llama.cpp base forward, and the MoE fork
must be merged onto that base whether or not there was an independent reason to
move. **What does one such hop cost, in wall clock?**

Measured in a throwaway lane worktree. Nothing was landed on any branch; the
fork is untouched.

## Config

- MoE fork HEAD: `f3e7141dd` on `vibe/async-fills` (the ratified
  `llama_cpp_commit` in integration-manifest.json), upstream base `9b2a08881`
- LocalAI v4.8.1 (`8052c950`) pins llama.cpp `221f0f635` via
  `backend/cpp/llama-cpp/Makefile:2 LLAMA_VERSION`
- Gap merged: `9b2a08881..221f0f635` = 45 upstream commits, 2026-07-30 → 2026-08-02
- Fork delta carried across: 33 files, +7816/−7
- Lane: `~/workspace/.lanes/sync-221f0f635`, branch `sync/upstream-221f0f635`
- Build: cmake + `GGML_SYCL=ON GGML_NATIVE=ON GGML_SYCL_F16=OFF
  GGML_CCACHE=ON BUILD_SHARED_LIBS=ON CMAKE_BUILD_TYPE=Release`, icx/icpx from
  oneAPI 2026.1 — mirrors `llama.cpp/build-sycl/CMakeCache.txt`
- Machine: 28 threads, 31 GiB RAM. Build tree under
  `~/.cache/modelctl/ci/`, never `/tmp` (tmpfs)
- Concurrent load: no model resident, no benchmark running. Load average
  pre/post per run recorded below.

## Merge

| Measure | Value |
|---|---|
| Conflicts | 0 |
| Wall clock | 0.30 s |
| Resulting tree | `9209e2629509d51b806283df154992e6683902d3` |
| Predicted tree (`git merge-tree --write-tree`, run before the merge) | `9209e2629509d51b806283df154992e6683902d3` — identical |
| Integrity | `git diff --numstat 221f0f635 HEAD` is byte-identical to `git diff --numstat 9b2a08881..f3e7141dd`; merged tree differs from upstream by exactly the fork delta and nothing else |

`merge-tree` was clean in both parent orders. A forced-conflict control at a
different base correctly reported exit 1 with 47 paths, so the clean prediction
is not a false negative.

## Build — three runs, three different quantities

| Run | Tree | ccache going in | configure | build | rc | ccache delta |
|---|---|---|---|---|---|---|
| A | merged, from empty build dir | warm from project history | 2.49 s | 93.49 s (aborted at link, see below) | 2 | — |
| B | same build dir, resumed | as left by A | 1.03 s | 58.82 s | 0 | **+2 hits, +500 misses** |
| C | merged, fresh build dir | as left by B | 2.23 s | 34.75 s | 0 | **+497 hits, +4 misses** |

Load averages: A pre 0.79 / post 16.44 · B post 7.46 · C pre 2.23 / post 4.54.

Reading these:

- **A + B ≈ 152 s of build wall clock** is the cost of compiling this upstream
  delta the first time. ccache served almost nothing (+2 hits against +500
  misses in B), so this is real compilation.
- **C's 34.75 s is not a from-scratch build cost.** The tree was fresh but the
  cache was hot from A and B: 497 of 501 compilations were replayed from
  ccache. It measures rebuilding an *already-compiled* merge, e.g. after
  wiping a build directory.
- For the recurring per-release fee, **A + B is the applicable figure**: each
  new LocalAI release carries upstream commits ccache has never seen.

## Finding: `bench-moe-hybrid` does not link at the ratified pin

Run A failed at `bin/bench-moe-hybrid` with undefined references to
`moe_cache_device_{create_transfer_queue,destroy_transfer_queue,
copy_async_after,event_complete,event_free}`. **The merge did not cause this.**

- `ggml/src/ggml-sycl/moe-cache-device.cpp` is deliberately the only TU that
  includes the SYCL headers; the policy side compiles as plain C++ so tests can
  build on a GPU-less machine, and test targets supply aborting stubs instead
  of linking the device TU.
- `moe-cache.cpp` calls 8 device entry points. `tests/test-moe-hybrid.cpp`
  stubs all 8. `tests/bench-moe-hybrid.cpp` stubs 3.
- The 5 missing ones were added by `e1957ebed` ("sycl: step-end stale-plan
  purge + async admission fills on a transfer queue"). That commit updated
  `test-moe-hybrid.cpp` and never touched `bench-moe-hybrid.cpp`.
- `f3e7141dd` — the ratified pin — contains `e1957ebed`. The bench target has
  therefore been unbuildable since 2026-08-02.
- The merge left the target byte-identical (`add_executable(bench-moe-hybrid …)`
  diffed pre- and post-merge: no change).
- Not caught by CI because `ci/checks.sh` builds named targets only —
  `llama-server test-moe-cache` (L209) and `test-moe-cache test-moe-hybrid`
  (L277). `bench-moe-hybrid` is configured but never built, and nothing builds
  `all`. The last full build of the tree, `build-sycl/bin/bench-moe-hybrid`, is
  dated 2026-08-01 22:08 — before `e1957ebed`.
- `bench-moe-hybrid.cpp:42` states the invariant it violates: "Same aborting
  stubs as test-moe-hybrid.cpp, for the same reason."

Runs B and C carried a lane-local patch adding the 5 stubs, mirroring
`test-moe-hybrid.cpp:62-66`, solely to let the build complete. **The fork was
not modified.**

## Not covered

- Whether a *future* LocalAI release's llama.cpp gap merges as cleanly. n=1,
  and this gap was 3 days / 45 commits. The number is a floor, not a rate.
- Wall clock is not the whole fee. Establishing that this clean merge was
  semantically sound — i.e. that the link failure predated it — took
  substantially longer than the 152 s of compilation, and that forensic step
  recurs on every hop because a clean three-way merge carries no evidence about
  semantics.
- No functional validation of the merged tree: `test-moe-cache` not run, no
  acceptance pass, no throughput measurement. This measured build cost only.
