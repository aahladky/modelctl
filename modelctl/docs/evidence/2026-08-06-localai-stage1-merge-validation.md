# LocalAI Stage 1 -- work item 1: merge onto 221f0f635, validated -- 2026-08-06

Stage 1 work item 1 is "rebase our llama.cpp fork onto LocalAI's pin and build
their grpc-server against it". M1 (`2026-08-06-localai-stage0-m1-rebase-cost.md`)
measured that merge's build cost but ran no functional validation. This pass
reproduces the merge independently and validates the merged tree.

**Scope was bounded by the operator (2026-08-06): merge, build and host-only
tests only -- do not re-ratify.** `integration-manifest.json` is untouched;
`llama_cpp_commit` still reads `f3e7141dd` and `validated_llama_commit` still
reads `85b7e6556`. The fork's own branches are unmodified; all work is in a
throwaway worktree.

## Config

- Fork HEAD: `f3e7141dd` on `vibe/async-fills` (the ratified `llama_cpp_commit`)
- LocalAI v4.8.1 (`8052c950`, confirmed this session to be exactly tag `v4.8.1`)
  pins llama.cpp `221f0f635` via `backend/cpp/llama-cpp/Makefile:2`
- Worktree: `~/workspace/.lanes/sync-221f0f635`, branch `sync/upstream-221f0f635`,
  created from `f3e7141dd`
- Build: CPU-only, `GGML_SYCL=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON
  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release`, GNU 16.1.1, build dir
  `~/.cache/modelctl/ci/ci-build-cpu-lane-sync221` (never `/tmp`)
- Machine: 28 threads, 31 GiB RAM. Load average 0.62 at merge, 3.55 after the
  build; no model resident, no inference process running, llama-swap and OVMS
  untouched.

## Merge -- reproduces M1 exactly

| Measure | This pass | M1 |
|---|---|---|
| `git merge-tree --write-tree` prediction | `9209e2629509d51b806283df154992e6683902d3` | same |
| Actual merged tree | `9209e2629509d51b806283df154992e6683902d3` | same |
| Conflicts | 0 | 0 |
| Wall clock | 0.307 s | 0.30 s |

Integrity, re-run: `git diff --numstat 221f0f635 HEAD` and
`git diff --numstat 9b2a08881..f3e7141dd` hash identically
(`sha256 e0623d73adb764dd...`), so the merged tree differs from upstream by
exactly the fork delta and nothing else.

## Prerequisite: bench-moe-hybrid had to be fixed first

The merged tree cannot build the full CI target set until the open
`bench-moe-hybrid` link break is fixed -- `tests/bench-moe-hybrid.cpp` stubbed
3 of the 8 `moe_cache_device_*` entry points, and `e1957ebed` added 5 more
without updating it. Confirmed still present at `f3e7141dd` this session
(bench: 3 stubs, `test-moe-hybrid.cpp`: 8).

The five missing stubs were added, mirroring `test-moe-hybrid.cpp:62-66`, with
signatures checked against `ggml/src/ggml-sycl/moe-cache.hpp:235-240` rather
than copied blind. In this lane the fix is a working-tree patch; the fix itself
is staged separately against the main checkout, since it is not caused by the
merge and stands on its own.

## Build and tests -- merged tree

| Step | Result |
|---|---|
| cmake configure | rc 0 |
| build `llama-server test-moe-cache test-moe-hybrid bench-moe-hybrid` | rc 0 |
| `test-moe-cache` | rc 0 -- "all MoE cache tests passed" |
| `test-moe-hybrid` | rc 0 -- "all MoE hybrid partition tests passed" |
| `bench-moe-hybrid --experts 4 --threads 4 --reps 2 --check` | rc 0 -- check "max abs 0, max rel 0 vs the warm-up pass (bit-identical)" |

The bench figure it printed (0.356 ms/op, 4 experts) was taken under load
average 3.55 immediately after the build. It is a smoke test that the target
runs, not a benchmark, and is not comparable to any recorded bench number.

## Capability surface survived the upstream delta

The 45-commit gap touches `common/arg.cpp` (+36) and `common/common.h` (+3) --
the two files the MoE-cache flags are defined in, which is the specific reason
this merge could have broken something silently.

`llama-server --modelctl-capabilities` on the merged tree:

- `schema: 3`, matching `integration-manifest.json`'s `capability_schema`
- CI's own assertion (`ci/checks.sh:220-238`, every feature false in a CPU-only
  build) returns `none` -- pass
- all five CLI flag families still registered: `--moe-cache-bytes`,
  `--moe-cache-policy`, `--moe-cache-admission-misses`,
  `--moe-cache-prefill-admission`, `--moe-hybrid-mode`

## Not covered

- No SYCL build in this pass. M1 built LocalAI's `grpc-server` against this same
  merged tree with `BUILD_TYPE=sycl_f32` and it linked and served; that is not
  re-verified here.
- No hardware acceptance pass, no GPU test, no throughput measurement. The pin
  is deliberately not advanced, so the ratified/validated pair is unchanged.
- Host-only tests exercise the policy side of the cache; every
  `moe_cache_device_*` path aborts by design in these binaries.
- n=1 on the merge. This is the same gap M1 measured, re-run, not a second hop.
