# Upstream drift impact review — 2026-07-30

Task 0.2 deliverable. Compares `feature/sycl-moe-expert-cache` (frozen at
`e7af6cf19`, tag `moe-cache-pre-upstream-2026-07-30`, see
`2026-07-30-baseline.md`) against `upstream/master` (`origin` =
github.com/ggerganov/llama.cpp), fetched 2026-07-30 at `9b2a08881`.

## Top-line drift

- Total: **348 commits** behind (equivalent under `e7af6cf19..origin/master`
  or `6f4f53f2b..origin/master` — see note below), **12 commits** ahead (the
  cache feature series itself).
- True merge-base of `e7af6cf19` and `origin/master` is **`6f4f53f2b`**
  ("common: dedup preset and cached model entries in /v1/models (#25131)"),
  not `e7af6cf19` — the 12 feature commits sit linearly on top of
  `6f4f53f2b`, which is itself an ordinary ancestor commit of
  `origin/master`. Since none of upstream's 348 commits are reachable from
  the feature tip, `git log e7af6cf19..origin/master` and
  `git log 6f4f53f2b..origin/master` enumerate the identical commit set —
  the counts above are unaffected, only the merge-base label was corrected
  post-review.

Per-path commit counts touching the areas the roadmap calls out as
conflict-sensitive:

| Path | Commits since merge-base |
|---|---|
| `ggml/src/ggml-backend.cpp` | 2 |
| `ggml/src/ggml-sycl/` | 20 |
| `tools/server/` | 47 |
| `common/` | 44 |
| `src/` | 45 |
| `tests/` | 46 |

## Scheduler copy / backend-assignment logic

Only 2 commits directly touch `ggml-backend.cpp`, but both are structurally
significant for a cache that intercepts expert-weight copies at the
scheduler level:

- **`dee2a846b`** — "ggml: adjust logic for offloading ops to weight's
  backend (#25832)". Touches `ggml/src/ggml-backend.cpp` (43 lines),
  `src/llama-context.cpp`, and `src/models/deepseek4.cpp`. This changes the
  rule for *which backend an op runs on when its weight lives on a
  different backend than the compute device* — i.e. the exact decision
  point a GPU-resident expert-weight cache needs to intercept. Expect a
  real conflict here, not just textual.
- **`86b94708f`** — "Revert 'sched: reintroduce less synchronizations
  during split compute (#20793)' (#25138)". Touches `ggml-backend.cpp` (10
  lines) and `ggml-cuda.cu`. A revert of a synchronization-reduction
  change — worth checking whether the feature branch's hook assumes either
  the reverted or the reverting behavior for split-compute sync ordering.

## `mul_mat_id` / selected-expert handling

No commit in the reviewed window touches `mul_mat_id` directly by name.
The closest relevant activity is CUDA-side top-k MoE fusion work —
**`846e991ec`** ("cuda: add sqrt_softplus in topk-moe for dsv4", #25896) and
**`75a48a905`** ("cuda: enable topk-moe fusion for 288 experts", #25267).
Both are CUDA-only today, so they don't directly conflict with the SYCL
cache, but they establish a fused topk-MoE pattern upstream is investing in;
if that fusion later lands for SYCL, it would change the granularity at
which the cache can observe "selected experts," so this is a trend to watch
rather than an immediate conflict.

## SYCL queues / events / allocation / device-context lifetime

20 commits touched `ggml/src/ggml-sycl/`. Highest relevance:

- **`efb3036c1`** — "sycl: add fused top-k MoE (#25217)" — direct overlap
  with expert-selection logic on the SYCL backend specifically. This is the
  single highest-risk SYCL commit for the cache's admission/hook logic and
  needs a careful read against the cache's expert-copy interception point.
- **`c1063ac9d`** — "sycl: set fattn_vec_nthreads to 256 for Battlemage
  (#25205)" and **`32b741c33`** — "[SYCL] Flash Attention with XMX engine
  via oneDNN (#25222)" — both target Battlemage specifically, which is this
  machine's actual hardware (Arc Pro B70 / Arc B580, both Battlemage). Low
  conflict risk against the cache itself (attention path, not expert
  weights) but directly affects any re-benchmark after sync (Task 0.7) since
  Battlemage FA performance characteristics changed upstream since the
  feature branch's base.
- **`0e148a573`** — "sycl: Increase minimum buffer size for USM system
  allocations (#25525)" — relevant if the cache's allocation path shares the
  USM allocator wrapper.
- **`26145b3db`** — "sycl: rename the env vars from 'disable' to 'enable'
  (#25042)" — an env-var rename for backend toggles. Worth double-checking
  the cache/launch code doesn't reference the old env var names anywhere.
- No commit in this window touches SYCL device enumeration/selection
  (`dev_mgr`, `select_device`) specifically. **I found nothing upstream that
  explains the SYCL device-selection crash observed against the current
  pre-sync `build-sycl` binary** (`sycl::_V1::detail::select_device`
  throwing "No device of requested type available" in `dpct::dev_mgr::dev_mgr()`
  — see baseline doc). Since that binary predates all 348 of these commits
  and the crash reproduces with unset/forced `ONEAPI_DEVICE_SELECTOR` alike,
  it looks like a local build/runtime-drift issue (stale binary vs. the
  currently installed Level-Zero/compute-runtime stack), not something
  upstream changed or will fix. Treat this as an open pre-existing question
  for Task 0.5, unrelated to the sync itself.

## Server memory abstraction / model unload/reload paths

- **`ee3d1b54c`** — "server: abstract llama_memory calls to common_memory
  (#26221)". Touches `common/common.cpp` (+32), `common/common.h` (+15),
  and **`tools/server/server-context.cpp`** (59 lines, net reduction) —
  replaces direct `llama_memory` calls in the server with a
  `common_memory` wrapper. This is exactly the surface the roadmap flags as
  conflict-sensitive, and it's a wide enough refactor of
  `server-context.cpp` that any cache hook into model unload/reload will
  need re-verification line by line, not just a conflict-marker resolve.
- **`40b740ad0`** — "server: properly handle null llama_context (#25868)"
  — relevant to unload/reload edge cases (context set to null mid-lifecycle).
- **`bf2c86ddc`** — "server: refactor prompt cache state ownership (#25649)"
  — ownership refactor adjacent to any cache-owns-state assumptions.

## Capability and argument parsing

No commits touched `common/arg.cpp` directly in this window. Related
argument-parsing changes to be aware of:

- **`e6dd0e29a`** — "args: refactor mlock/mmap/directio into load-mode
  (#20834)" and **`ad256ded3`** — "args: add `-lm mlock` where it mlocks
  but doesn't mmap (#26135)". These change how mmap/mlock/directio are
  selected, which the roadmap's storage-mode claims (Task D1) and the
  cache's RAM/storage assumptions (`ram.mode: page_cache`,
  `storage.mode: mmap` in the current profile config) depend on.
- **`0e4a03622`** — "common: add `common_print_available_devices()`
  (#26170)" — new device-listing helper; worth checking whether it
  duplicates or should replace any ad-hoc device enumeration the
  capability-probe patch added.
- **`c264f65ff`** — "cli: move to HTTP-based implementation (#24948)" — a
  large CLI-entry refactor; the `--modelctl-capabilities` flag's argument
  registration should be re-verified against wherever CLI argument handling
  now lives post-refactor.

## MTP / NextN layer counting and placement

- **`0324696b8`** — "fit: count nextn (MTP) blocks in n_gpu_layers so
  front layers stay on GPU (#26177)", `common/fit.cpp` (1 line changed but
  semantically significant — changes layer-count arithmetic that
  auto-fit/placement depends on). Directly relevant: the current baseline
  already showed `common_fit_params` aborting when `tensor_split` is
  user-set (see baseline doc); this commit changes the layer-counting input
  to that same fit logic and should be re-tested against MTP-enabled
  profiles specifically (e.g. `gemma4-26b-mtp`).
- **`7be2c65dc`** — "model: add NextN/MTP speculative decoding support for
  GLM_DSA (GLM-5.2) (#25980)", **`64d528be7`** — "mimo2: address MTP review
  feedback (#26228)", **`2969d6d15`** — "model: add Hy3 (hy_v3) support with
  MTP speculative decoding (#25395)" — three separate MTP-support additions
  for different architectures. None conflict with the cache directly, but
  confirm MTP is an active, moving area upstream — re-validate
  main+draft/MTP coexistence (Task 0.4/0.6) against whichever of these
  landed most recently, not just the original MTP implementation the
  feature branch was built against.

## Dynamic backend loading

- **`5735e10c4`** — "ggml-openvino: Add GGML_BACKEND_DL_IMPL invocation for
  OpenVINO backend (#25795)" and **`082b326fc`** — "ggml-et: Initial ET
  backend (#24179)" — both are new backends adopting the
  `GGML_BACKEND_DL_IMPL` dynamic-loading pattern. No direct conflict (SYCL
  registration untouched in this window), but useful as a fresh reference
  implementation for the "expose a backend API through the backend
  registry" work in Task F2.

## Model metadata and expert tensor layouts

No commits in this window touch expert tensor layout/metadata generically
(the tensor-layout-adjacent changes found — `dee2a846b`'s
`src/models/deepseek4.cpp` fixes, `4937ca83f` "llama-quant: exclude i32
ffn_gate_tid2eid routing table from quantization (#25787)" — are
architecture-specific to DeepSeek V4, not the general expert-tensor
metadata path the cache generalizes over). No evidence of a metadata-layout
break for the cache's currently-supported projections (`gate`, `up`,
`down`), but this should be spot-checked against whichever architecture the
post-sync correctness matrix (Task 0.6) actually targets.

## Highest-risk conflict points, ranked

1. **`ee3d1b54c`** (server memory abstraction) — widest surface (59 lines in
   `server-context.cpp`) directly on the unload/reload path the cache's
   lifetime management depends on.
2. **`efb3036c1`** (SYCL fused top-k MoE) — same subsystem (SYCL expert
   selection) as the cache's core hook; highest chance of a *semantic*
   conflict, not just a textual one.
3. **`dee2a846b`** (offload-to-weight's-backend logic) — changes the exact
   backend-assignment decision the cache's scheduler hook intercepts.
4. **`0324696b8`** (MTP layer counting in fit) — small diff, but changes
   input to fit/placement logic that already shows a related warning
   (`tensor_split already set by user, abort`) in the current baseline;
   compounds with `c264f65ff`'s CLI refactor risk for anything depending on
   argument-driven placement.
5. **`c264f65ff`** (CLI moved to HTTP-based implementation) — broad enough
   a refactor that the `--modelctl-capabilities` argument hook should be
   re-verified structurally, not assumed to still parse the same way.
