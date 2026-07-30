# modelctl Live-Repository Implementation Roadmap

**Date:** July 30, 2026  
**Revision:** Upstream synchronization and maintenance strategy integrated  
**Source of truth:**

- Control plane / umbrella repository: <https://github.com/aahladky/modelctl>
- Runtime fork: <https://github.com/aahladky/llama.cpp>
- Runtime development branch: `feature/sycl-moe-expert-cache`

**Primary goal:** Deliver a web-first, hardware-aware local model management and serving system that can acquire, inspect, place, benchmark, serve, monitor, and remove models across heterogeneous GPUs, system RAM, and storage, with practical optimization for oversized sparse-MoE models.

---

## 1. Live-repository correction

The July 29 roadmap was written from an older review bundle. The live `modelctl` repository has already implemented much of its early and middle phases.

The repository now includes:

- a canonical `ResolvedBackend` and `LaunchCommand` layer;
- schema-2 backend capability normalization and binary-fingerprint caching;
- structured validation errors;
- profile, plan, runtime, hardware, and cache service modules;
- atomic multi-file transactions;
- storage topology discovery;
- explicit benchmark cache-state modes;
- a persistent browser add-model wizard;
- local GGUF import;
- plan comparison, plan testing, registration, and warm-load steps;
- settings and hardware pages;
- Release A and Release B control-plane acceptance tests.

This means the project should **not** spend another cycle rebuilding those abstractions. The remaining work is to prove that they are authoritative in every production path, finish the observable RAM/SSD serving loop, correct the runtime cache, and make the browser the actual product entry point.

The live repository is best viewed as an umbrella project:

```text
modelctl repository
├── modelctl/      Python control plane and web application
└── llama.cpp/     pinned runtime submodule
```

The top-level repository should remain the reproducible integration unit. A checkout of one commit should identify the exact modelctl code, llama.cpp commit, schema contract, tests, and target build instructions that belong together.

---

## 2. Updated project status

| Area | Live state | Remaining work |
|---|---|---|
| Canonical backend resolution | Implemented in `modelctl_launch.py` | Prove every preview, artifact, worker, test, and llama-swap path uses it |
| Capability schema | Schema 2 implemented in modelctl | Runtime probe is still not truthful or backend-derived |
| Cache preflight | Integrated into `build_launch_command()` | Audit all legacy callers and prohibit bypasses |
| Operation/service layer | Profile, plan, runtime, hardware, and cache services exist | Move remaining route and CLI mutation logic behind services |
| Atomic mutations | `Transaction` exists | Ensure all multi-file operations use it and add crash/fault tests |
| Storage topology | `StorageInfo` and mount/device probing exist | Integrate process I/O/page faults and storage claims into observations and UI |
| Benchmark modes | Natural, process-cold, page-cache-warm, expert-cache-warm, storage-cold defined | Implement and validate execution semantics end to end |
| Browser model lifecycle | Persistent source → inspect → download/import → analyze → plans → test → register → done wizard exists | Improve failure recovery, job-state visibility, plan evidence, and product polish |
| Web settings/hardware | Implemented | Expand backend/build/storage setup and first-run diagnostics |
| Release A tests | Control-plane acceptance tests exist | Add real-process, llama-swap, storage, concurrency, and target-hardware acceptance |
| Release B tests | Control-plane scaffolding exists | Runtime hybrid execution does not exist |
| SYCL expert transfer cache | Experimental implementation exists | Admission, prefill, capability, ownership, allocation, and deterministic correctness remain blockers |
| True CPU-miss hybrid execution | Modelctl contract and test scaffolding exist | Entire runtime data path remains to be implemented |
| Runtime upstream alignment | Feature branch is materially behind current llama.cpp master | Sync now on a protected integration branch, then maintain a regular drift and promotion cadence |

---

## 3. Revised release sequence

```text
Phase 0  Controlled sync to current llama.cpp upstream
    │
    ▼
Phase A  Reproducible repository and integration contract
    │
    ▼
Phase B  Prove the canonical launch path is truly canonical
    │
    ├──────────────┐
    ▼              ▼
Phase C        Phase D
Web-first      RAM/SSD observability
product pass   and benchmark execution
    │              │
    └──────┬───────┘
           ▼
Phase E  Release A real-hardware acceptance
           │
           ▼
Phase F  Correct and stabilize the SYCL transfer cache
           │
           ▼
Phase G  True GPU-hit / CPU-miss execution
           │
           ▼
Phase H  Measured automatic plan selection
           │
           ▼
Ongoing  Scheduled upstream drift review and tested promotion
```

The immediate product milestone remains:

> A user can add or import an oversized MoE from the browser, compare truthful GPU/RAM/mmap/cache plans, run labeled cold and warm tests, select a measured plan, register it with llama-swap, load it, and see where its bytes and time are going.

---

# Phase 0 — Synchronize the runtime fork with current upstream

The runtime branch is far enough behind `ggml-org/llama.cpp` master that continued feature development would increase rebase risk. This matters more than raw commit count because the cache touches high-churn areas: the scheduler, SYCL backend, server lifecycle, memory handling, MoE placement, and MTP integration.

The goal is **not** to run production from upstream master. The goal is to port the feature onto a recent upstream base, validate it, and then pin the tested commit in the umbrella repository.

## Task 0.1 — Freeze the current known-good baseline

**Goal:** Preserve a reproducible rollback point before touching upstream history.

**Work:**

- [ ] Tag the current runtime feature branch with an annotated pre-sync tag.
- [ ] Push the tag to the fork.
- [ ] Record the exact modelctl commit and llama.cpp commit.
- [ ] Record oneAPI compiler/runtime, kernel, driver, firmware, and GPU identities.
- [ ] Export the current capability response.
- [ ] Save representative build commands, launch commands, correctness results, and benchmark observations.
- [ ] Generate a support bundle containing the integration manifest and relevant logs, with secrets redacted.

Suggested tag:

```bash
git tag -a moe-cache-pre-upstream-2026-07-30 \
  -m "Known-good MoE cache state before upstream synchronization"
git push origin moe-cache-pre-upstream-2026-07-30
```

**Acceptance:**

The previous runtime can be rebuilt and relaunched from the tag without relying on an uncommitted working tree or remembered environment details.

---

## Task 0.2 — Quantify drift and identify conflict-sensitive changes

**Goal:** Review upstream changes by subsystem rather than treating the update as a blind commit-count exercise.

**Work:**

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream master

git rev-list --left-right --count \
  feature/sycl-moe-expert-cache...upstream/master

BASE=$(git merge-base feature/sycl-moe-expert-cache upstream/master)

git log --oneline --no-merges \
  feature/sycl-moe-expert-cache..upstream/master -- \
  ggml/src/ggml-backend.cpp \
  ggml/src/ggml-sycl \
  tools/server \
  common \
  src \
  tests
```

Review the upstream diff in these areas:

- scheduler copy and backend-assignment logic;
- `mul_mat_id` and selected-expert handling;
- SYCL queues, events, memory allocation, and device context lifetime;
- server memory abstraction and model unload/reload paths;
- capability and argument parsing;
- MTP/NextN layer counting and placement;
- dynamic backend loading;
- model metadata and expert tensor layouts.

**Deliverable:** `docs/upstream-sync/2026-07-30-impact-review.md` containing relevant upstream commits, expected conflicts, required retests, and any feature assumptions invalidated by upstream.

---

## Task 0.3 — Port the feature onto a fresh upstream integration branch

**Goal:** Avoid permanently tangling the feature history with a large upstream merge commit.

Preferred approach:

```bash
git switch -c sync/moe-cache-upstream-2026-07-30 upstream/master
```

Then port a cleaned logical commit series:

1. backend cache implementation;
2. scheduler integration;
3. server configuration and telemetry;
4. capability/backend interface;
5. deterministic tests.

Use cherry-pick when the current history contains multiple corrective commits. Use a rebase only when the feature commits are already clean and independently reviewable.

Do **not** resolve conflicts by taking all of “ours” or all of “theirs” in the following files:

- `ggml/src/ggml-backend.cpp`;
- `ggml/src/ggml-sycl/ggml-sycl.cpp`;
- `ggml/src/ggml-sycl/moe-cache.*`;
- `tools/server/server.cpp`;
- `tools/server/server-context.cpp`;
- `common/arg.cpp`;
- MTP/model-loading and layer-placement code.

Each conflict resolution should document which upstream behavior is being preserved and how the cache integrates with it.

---

## Task 0.4 — Re-evaluate architecture while resolving conflicts

The upstream port should be used to remove known integration debt rather than reproducing it unchanged.

Required corrections during or immediately after the port:

- [ ] Replace weak SYCL globals with a versioned backend procedure API.
- [ ] Make capabilities backend-derived and truthful for CPU-only, static-backend, and dynamic-backend builds.
- [ ] Attach cache ownership to a safe backend/device/context lifetime model.
- [ ] Use current upstream SYCL allocation and synchronization helpers.
- [ ] Revalidate the scheduler interception point and its batch-size behavior.
- [ ] Revalidate MTP/main-model coexistence and GPU-layer interpretation.
- [ ] Keep true hybrid CPU-miss support false until the data path exists.

This avoids spending effort perfecting an interface that must be rewritten immediately after the sync.

---

## Task 0.5 — Pass the post-sync build matrix

Required builds:

```text
CPU-only
SYCL with cache disabled
SYCL with transfer-cache support compiled
GGML_BACKEND_DL=OFF
GGML_BACKEND_DL=ON
Debug/assertion-enabled build
Release build used for benchmarks
```

The CPU-only capability response must report every SYCL/cache capability as false. A dynamically loaded SYCL backend must expose configuration, metrics, phase, reset, and capability procedures after loading.

**Acceptance:**

All supported build forms configure, compile, link, and return internally consistent capability responses.

---

## Task 0.6 — Pass the post-sync correctness matrix

Required cases:

- cache disabled versus unmodified current upstream;
- forced all-hit;
- forced all-miss;
- alternating mixed hit/miss;
- partial projection population;
- admission thresholds 1 and 2+;
- prefill admission on and off;
- LRU and SLRU eviction;
- cache reset while loaded;
- unload/reload;
- two contexts on one GPU;
- main plus draft/MTP context;
- two asymmetric GPUs;
- dynamic backend loading.

Compare logits, expert outputs, or deterministic token IDs. Coherent prose is not a correctness oracle.

**Acceptance:**

The cache-disabled integration matches current upstream within the declared numerical tolerance, and every enabled path passes deterministic correctness checks.

---

## Task 0.7 — Rebaseline performance after the sync

Upstream scheduler and kernel changes can alter cache eligibility and apparent speed without any cache-code change. Re-run:

```text
prompt batch 1
prompt batch below the SYCL/offload threshold
prompt batch above the threshold
interactive decode batch 1
continuous-batching decode
cold page cache
warm page cache
warm expert cache
```

Record:

- actual operation backend by phase;
- cache lookups/hits/misses;
- H2D and D2D bytes;
- prompt throughput;
- generation throughput;
- TTFT;
- CPU time;
- storage reads and page faults;
- model and backend fingerprints.

Do not compare new results against old observations unless the UI marks them as historical and stale.

---

## Task 0.8 — Validate the complete modelctl integration chain

Run the exact integration sequence:

```text
runtime capability probe
→ capability normalization
→ plan generation
→ preflight
→ browser preview
→ plan test
→ managed worker command
→ llama-swap command
→ launch
→ metrics
→ unload/reload
```

Verify that:

- the runtime binary fingerprint changes after the sync;
- stale observations are excluded from automatic selection;
- unsupported flags never appear;
- the displayed command and launched command share one fingerprint;
- the submodule pointer is not advanced before all gates pass.

---

## Task 0.9 — Promote the tested runtime into the umbrella repository

Only after Tasks 0.5–0.8 pass:

- [ ] Merge the integration branch into the runtime feature branch.
- [ ] Pin the tested runtime commit in the top-level modelctl submodule.
- [ ] Update `integration-manifest.json`.
- [ ] Archive the acceptance report and benchmark comparison.
- [ ] Mark old benchmark observations stale by fingerprint rather than deleting them.
- [ ] Publish release notes describing relevant upstream changes and known experimental limitations.

Production and normal development should use the pinned submodule commit, never an untested moving branch head.

---

## Task 0.10 — Establish recurring upstream maintenance

**Cadence:**

- weekly automated drift report;
- planned integration branch every two to four weeks;
- immediate review for relevant SYCL correctness fixes, security changes, scheduler changes, model-format changes, or MTP/MoE placement fixes.

CI should report:

```bash
git rev-list --count HEAD..upstream/master
```

and summarize upstream changes touching:

```text
ggml/src/ggml-sycl/
ggml/src/ggml-backend.cpp
tools/server/
common/
src/llama-model*
src/llama-context*
tests/
```

The drift report is informational. It must not automatically advance the production submodule. Promotion always requires the build, correctness, performance, and integration gates above.

**Definition of completion for Phase 0:**

The feature is ported to a recent upstream base, the old state remains reproducible, current upstream behavior is preserved where intended, the runtime passes the complete test gate, and modelctl pins the validated commit with stale observations invalidated.

---

# Phase A — Make the umbrella repository reproducible

## Task A1 — Correct repository metadata and links

**Goal:** Make the GitHub repositories themselves the documented source of truth.

**Files:**

- top-level `README.md`
- `.gitmodules`
- `modelctl/README.md`
- installation and contributor documentation

**Work:**

- [ ] Replace stale Gitea and `moe-serving/*` references with the public GitHub repository locations.
- [ ] Explain that the top-level repo is the integration repo and `modelctl/` contains the application source.
- [ ] Document the runtime branch and the fact that the submodule commit, not branch head, is authoritative for a release.
- [ ] Add clone instructions for HTTPS and SSH.
- [ ] Add a command that verifies the submodule commit exists in the configured remote.
- [ ] Change the product description from “CLI primary, UI for edge cases” to the intended web-first model.

**Acceptance:**

A new user can clone recursively, install, build the pinned runtime, start the web service, and identify exactly which runtime commit is in use without private infrastructure.

---

## Task A2 — Add integration metadata

**Goal:** Tie modelctl, runtime, schema, and build artifacts together.

**Add:** `integration-manifest.json` generated or validated in CI.

Suggested shape:

```json
{
  "modelctl_commit": "...",
  "llama_cpp_commit": "...",
  "capability_schema": 2,
  "profile_schema": 2,
  "supported_runtime_features": ["moe_weight_transfer_cache"],
  "release_channel": "experimental"
}
```

**Work:**

- [ ] Generate the manifest during release or CI.
- [ ] Display it on the web diagnostics/settings page.
- [ ] Include it in support bundles.
- [ ] Warn when the working tree or submodule does not match the manifest.

---

## Task A3 — Add CI for the integration repository

**Required jobs:**

- Python unit and web tests with all declared dependencies.
- `compileall` and static import checks.
- CPU-only llama.cpp build.
- CPU-only `--modelctl-capabilities` assertion: every SYCL/cache feature must be false.
- Submodule URL and commit validation.
- Golden command/provenance tests.
- Artifact generation and transaction rollback tests.

A SYCL hardware runner can remain separate, but CPU-only capability truthfulness must be enforced on every push.

---

# Phase B — Prove one authoritative launch path

The canonical types now exist. This phase is an integration audit, not another abstraction rewrite.

## Task B1 — Inventory every command-construction call site

**Goal:** Eliminate all independent reconstruction after validation.

**Search targets:**

- `build_server_args`
- backend `build_command`
- string concatenation involving `llama-server`
- generated `run.sh`
- llama-swap config rendering
- plan tests
- smoke tests
- worker launches
- web command previews
- CLI previews

**Rule:** Every production path must receive or derive from a `LaunchCommand` created by `modelctl_launch.build_launch_command()`.

Legacy helper functions may remain as internal adapter implementation, but no caller may bypass backend resolution and capability validation.

---

## Task B2 — Make validation blocking and explicit

**Goal:** A command containing unsupported experimental flags can never launch.

**Work:**

- [ ] Add `LaunchCommand.is_valid` or a mandatory `raise_for_errors()` operation.
- [ ] Require workers, tests, artifact writers, and llama-swap sync to call it.
- [ ] Make manual cache plans fail with a structured message when unsupported.
- [ ] Make automatic planning omit unsupported cache candidates.
- [ ] Treat probe failure, stale capability data, device mismatch, and contradictory constraints as fail-closed.
- [ ] Expose all validation messages in web plan and runtime pages.

**Acceptance test:**

Use a stock upstream `llama-server`, enable every cache setting in a profile, and assert that no generated or launched command contains a `--moe-cache-*` argument.

---

## Task B3 — Prove command equality across surfaces

For one profile and plan, compare normalized command identity from:

1. browser preview;
2. CLI preview;
3. plan-test process;
4. managed worker;
5. generated `run.sh`;
6. llama-swap configuration.

Only assigned port and wrapper-specific quoting may differ. Binary, environment, model path, placement, cache flags, context, and backend options must be identical.

Store `command_fingerprint`, binary fingerprint, environment fingerprint, plan ID, and capability fingerprint with every observation and runtime event.

---

## Task B4 — Remove recursive or duplicated preflight behavior

`resolve_backend()` currently uses existing preflight logic and `build_launch_command()` invokes preflight again. Consolidate this so binary selection, environment preparation, file checks, capability probing, and plan validation occur once in a clearly ordered pipeline.

Suggested shape:

```text
resolve candidate binary
→ construct effective environment
→ fingerprint binary and environment
→ probe capabilities in that environment
→ validate profile and plan
→ build argv once
→ return immutable LaunchCommand
```

This avoids discrepancies where preflight discovers an alternate binary but a later path launches the original one.

---

# Phase C — Make the browser the actual first-class product

The wizard and management pages now exist. The next work is cohesion, recovery, and default product behavior.

## Task C1 — Change the product entry point

- [ ] Make the web console the primary documented workflow.
- [ ] Add a first-run page when hardware, backend, llama-swap, or model directories are unconfigured.
- [ ] Provide one command to install/start the user service and print the URL/token.
- [ ] Keep the CLI as bootstrap, automation, diagnostics, and recovery tooling.

---

## Task C2 — Harden the add-model wizard

The current wizard already covers:

```text
source
→ inspect
→ download/import
→ analyze
→ plans
→ test
→ register
→ done
```

Finish it as a durable workflow:

- [ ] Persist structured job outcomes and errors rather than parsing logs.
- [ ] Show resumable state after browser/server restart.
- [ ] Make every failed step retryable without restarting the wizard.
- [ ] Verify local files before profile creation: GGUF header, shard completeness, file readability, duplicate identity, and available destination space.
- [ ] Show download checksums or Hugging Face metadata where available.
- [ ] Prevent advancing while prerequisite jobs are running or failed.
- [ ] On registration failure, leave the profile valid and display an explicit rollback/retry action.
- [ ] Show the final endpoint, selected plan, command fingerprint, and measured result on the done page.

---

## Task C3 — Make plan comparison evidence-first

Each plan card should show:

- binary/build and capability state;
- plan source and selection reason;
- exact per-device VRAM claim;
- RAM claim;
- storage mode;
- expected context;
- cache budget and eligibility constraints;
- estimated versus observed values;
- cold/warm label;
- stale/untested/failed status;
- exact launch command and validation messages;
- fallback position.

The page should visually separate:

- safe baseline;
- tested preferred plan;
- untested estimate;
- experimental cache plan;
- unavailable plan and reason.

---

## Task C4 — Complete settings and diagnostics

Expand the current settings/hardware pages to include:

- model and state directories;
- llama-swap binary, service, URL, and configuration path;
- runtime build directories and pinned binary selection;
- oneAPI environment discovery/test;
- storage devices and calibration state;
- default benchmark mode;
- experimental-feature policy;
- current integration manifest;
- submodule/runtime commit mismatch;
- capability probe result and raw response;
- a downloadable support bundle with secrets redacted.

---

## Task C5 — Finish the service-layer migration

The service package exists, but route and CLI code should no longer directly coordinate multi-step mutations.

Add or complete:

- acquisition service;
- settings service;
- routing/llama-swap service;
- benchmark service.

All service operations should return one consistent result type containing:

```text
ok
messages
warnings
data
changed_resources
job_id
rollback_status
```

---

# Phase D — Make RAM and SSD behavior observable

Storage topology and benchmark mode definitions are present. The missing work is to connect them to real process execution, observations, planning, and UI.

## Task D1 — Extend resource claims

Every `ResourceClaim` should separately report:

- static VRAM by device;
- KV-cache VRAM by device;
- compute/overhead reserve;
- dynamic expert-cache VRAM;
- process-resident RAM;
- mmap-addressed model bytes;
- expected page-cache working set;
- storage path and device identity;
- temporary download/staging space.

Do not present mmap model size as guaranteed resident RAM.

---

## Task D2 — Sample process memory and I/O

During load and benchmark runs, collect:

- RSS, PSS where available, and peak RSS;
- process read bytes and system calls;
- major and minor page faults;
- per-device storage read throughput where practical;
- model-load wall time;
- TTFT;
- prompt and generation throughput;
- GPU VRAM before, peak, and after;
- cache metrics when supported.

Linux sources may include `/proc/<pid>/status`, `/proc/<pid>/io`, `/proc/<pid>/stat`, `smaps_rollup`, and block-device counters.

Store raw counters and derived rates. Never infer “SSD streaming” from `mmap=true` alone.

---

## Task D3 — Execute benchmark modes honestly

The enum exists; implement each mode as a controlled procedure.

### Natural

Record the current cache state without claims.

### Process cold

Start a new backend process while explicitly labeling page-cache state unknown.

### Page-cache warm

Perform a complete controlled load/warmup, unload, then measure the next run.

### Expert-cache warm

Warm until cache counters stabilize or a declared token/work budget is reached. Record whether stabilization was achieved.

### Storage cold

Require explicit opt-in and a scoped cache-eviction mechanism. Validate that eviction actually changed residency/read behavior; otherwise label the result `cold_unverified`.

---

## Task D4 — Add storage calibration and health

Calibration should record:

- sequential read throughput using a file large enough to exceed easy cache effects;
- representative random-read latency/IOPS;
- filesystem and mount options;
- RAID topology;
- timestamp and staleness;
- whether results appear page-cache contaminated.

Calibration data should guide estimates, not automatically determine the preferred plan.

---

## Task D5 — Add storage and memory views

The runtime and history pages should answer:

- How much of the model is in VRAM?
- How much RAM is resident?
- Is the process taking major faults?
- How many bytes were read during load and per generated token?
- Is this result storage-cold, process-cold, page-cache-warm, or expert-cache-warm?
- Is performance limited by CPU compute, H2D transfer, GPU compute, or storage reads?

---

# Phase E — Release A real-hardware acceptance

The current `test_release_a.py` is useful control-plane coverage. Release A is not complete until the actual processes and target hardware pass.

## Task E1 — Define baseline candidate set

For each oversized sparse MoE, compare at least:

1. stock mmap with CPU spill;
2. fully resident RAM spill, when feasible;
3. static heterogeneous GPU/CPU expert placement;
4. static placement plus experimental transfer cache.

Use the same model file, context, prompts, binary fingerprint, and correctness checks.

---

## Task E2 — Add real-process integration tests

Tests should launch actual small GGUF fixtures or purpose-built deterministic models through:

- direct worker launch;
- plan test;
- llama-swap registration and load;
- unload and reload;
- service restart;
- transaction failure and rollback;
- cancellation during download and benchmark;
- two competing resource reservations.

---

## Task E3 — Run the target-hardware matrix

At minimum:

- CPU-only build and machine;
- one Intel SYCL GPU;
- two asymmetric Intel SYCL GPUs;
- model fitting one GPU;
- model fitting combined VRAM;
- model requiring RAM spill;
- model requiring mmap/storage backing;
- transfer-cache disabled and enabled;
- concurrent prompt and decode requests;
- main plus draft/MTP model where applicable.

Release A passes only when the browser can complete the full lifecycle and the resulting measurements are correctly labeled and reproducible.

---

# Phase F — Correct the current SYCL expert-weight transfer cache

The live runtime branch still contains the core issues identified in review. These are blockers before the cache can be considered a stable plan variant.

## Task F1 — Make capability reporting truthful and backend-derived

The capability response must reflect the loaded runtime, devices, and compiled feature implementation.

- CPU-only build: every SYCL/cache feature false.
- SYCL build without cache implementation: cache false.
- Transfer-cache build: `moe_weight_transfer_cache=true`.
- `moe_hybrid_cpu_miss=false` until CPU execution and merge exist.
- Prefill policy false until the option works end to end.

Retrieve capabilities from a versioned backend procedure API, not hardcoded common-layer booleans.

---

## Task F2 — Replace weak global symbols

Remove the server’s direct weak references to SYCL globals and functions.

Expose a backend API through the backend registry, covering:

- capability query;
- configuration;
- current phase;
- metrics collection;
- reset;
- supported geometry/architectures;
- lifecycle/version information.

This must work with dynamic backend loading.

---

## Task F3 — Fix admission semantics

Current miss counts are still shared across gate/up/down projections. Replace them with projection-specific or activation-level admission state.

Acceptance tests must prove:

- threshold 2 means second eligible use, not second projection within one use;
- a cached projection cannot reset another projection’s admission;
- all cacheable projections eventually populate;
- unsupported geometry does not distort admission;
- prefill-disabled admission does not warm from prompt processing.

---

## Task F4 — Wire prefill configuration and phase

- Pass `--moe-cache-prefill-admission` into backend configuration.
- Reject invalid values rather than treating typos as off.
- Preserve the current phase before lazy cache initialization.
- Represent concurrent mixed prefill/decode accurately or conservatively without corrupting metrics.

---

## Task F5 — Correct ownership and lifetime

Replace the raw one-pointer-per-device registry with either:

- a ref-counted device-level cache shared by contexts; or
- a synchronized registry of context-owned caches.

Prove safe behavior for:

- two contexts on one device;
- main and draft models;
- metrics during unload;
- reset during active requests;
- repeated load/unload;
- multi-GPU contexts.

---

## Task F6 — Correct allocation and geometry

- Use the backend’s native SYCL allocation wrapper.
- Check every allocation result.
- Prefer a contiguous cache pool.
- Obtain exact projection geometry from model/backend metadata.
- Handle fused gate/up tensors and unequal projection sizes explicitly.
- Fail closed on unknown layouts.

---

## Task F7 — Rename misleading metrics

Until hybrid CPU execution exists:

- replace `cpu_expert_calls` with a host-projection or host-weight fallback metric;
- distinguish expert selections, projection copies, cache lookups, hit rows, and bytes;
- emit valid Prometheus metadata once;
- expose structured JSON with a schema version.

---

## Task F8 — Add deterministic runtime correctness tests

Required cases:

- cache disabled;
- forced all-hit;
- forced all-miss;
- alternating hit/miss;
- partial projection fill;
- admission thresholds 1 and 2+;
- LRU and SLRU eviction;
- reset;
- unequal geometry rejection;
- two contexts;
- multi-GPU;
- prefill on/off;
- dynamic backend loading.

Compare logits, expert outputs, or deterministic token IDs—not merely coherent final text.

---

# Phase G — Implement true GPU-hit / CPU-miss execution

The existing modelctl Release B tests are a useful contract scaffold. They do not constitute runtime implementation.

## Task G1 — Write the runtime design around batch-one decode

The key requirement is useful interactive decoding, not only large prompt batches.

For each routed MoE operation:

1. obtain selected experts and row assignments;
2. classify work into GPU-resident hits and CPU/RAM/mmap misses;
3. execute hit rows on persistent GPU expert weights;
4. execute miss rows directly on CPU-accessible weights;
5. merge outputs into original row order;
6. update admission asynchronously.

The design must specify synchronization, quant support, architecture constraints, temporary buffers, and numerical tolerances.

---

## Task G2 — Implement partition representation

Represent work in terms of selected rows and experts rather than merely projection-copy interception.

The partition must preserve:

- original token/row indices;
- expert weight and routing coefficient;
- projection/tensor identity;
- target execution tier;
- output destination.

---

## Task G3 — Implement CPU miss execution

Start with one architecture and limited quant types. Reuse existing CPU kernels and mmap weights rather than introducing custom SSD I/O.

Measure CPU miss compute independently from storage faults and output merge.

---

## Task G4 — Execute GPU hit partitions

Persistent cache slots should be consumed directly by the GPU operation, rather than copied into transient staging tensors where avoidable.

---

## Task G5 — Merge outputs correctly

Prove equivalence for:

- all hits;
- all misses;
- mixed experts in one token;
- mixed rows in one batch;
- shared experts;
- duplicate expert selections;
- multi-GPU placement;
- concurrent sequences.

---

## Task G6 — Add asynchronous promotion

Promotion must not stall the current miss path. Record:

- promoted bytes;
- queue delay;
- transfer time;
- overfetch;
- eviction before reuse;
- H2D bytes avoided.

---

## Task G7 — Complete Release B control-plane integration

Only advertise hybrid plans when all runtime constraints match:

- architecture;
- quantization;
- device/backend;
- context and batch constraints;
- CPU kernel support;
- merge support.

Show measured CPU miss, GPU hit, merge, promotion, and storage costs in the browser.

---

# Phase H — Select plans from measurements

## Task H1 — Define observation validity

An observation becomes stale when any material identity changes:

- model or shard fingerprint;
- profile schema/config;
- command fingerprint;
- binary fingerprint;
- environment fingerprint;
- GPU/driver/kernel fingerprint;
- storage device or filesystem;
- benchmark mode;
- cache schema/configuration.

---

## Task H2 — Rank by objective with guardrails

Support objectives such as:

- interactive latency;
- generation throughput;
- prompt throughput;
- minimum RAM pressure;
- minimum storage traffic;
- balanced.

Experimental plans should outrank safe baselines only when:

- correctness passed;
- a current observation exists;
- improvement exceeds a configurable margin;
- failure rate is acceptable;
- resource claims remain feasible.

---

## Task H3 — Explain every automatic decision

The UI should provide a concise trace:

```text
Selected: hybrid-balanced
Reason: 22% lower warm generation latency than static mmap baseline
Constraints: 64 GiB RAM maximum, context >= 32k
Rejected: transfer-cache-only — no decode improvement
Fallback: static-mmap-safe
Evidence: 3 successful current runs, binary abc..., hardware def...
```

---

# Immediate next sprint

The best next sprint is not to add more planner variants. It should close the gap between the sophisticated control-plane scaffolding and the still-experimental runtime.

## Sprint 0 — Controlled upstream synchronization

1. Freeze and tag the current runtime baseline.
2. Produce the upstream impact review.
3. Port the cache feature onto a fresh current-upstream integration branch.
4. Correct backend capability/configuration integration while resolving conflicts.
5. Run the complete build, deterministic correctness, performance, and modelctl integration gates.
6. Promote only the validated runtime commit into the modelctl submodule.
7. Establish the weekly drift report and two-to-four-week integration cadence.

This sprint should happen before additional SYCL runtime feature work. Web and storage work that does not depend on runtime internals may continue in parallel, but the production submodule must remain pinned until the sync passes.

## Sprint 1 — Repository and launch truth

1. Correct public repository links and web-first documentation.
2. Add integration manifest and CPU-only CI capability assertion.
3. Audit every command-construction path.
4. Add blocking launch validation and command-equality tests.
5. Remove duplicated/recursive backend resolution and preflight.

## Sprint 2 — Observable Release A

1. Connect storage topology to resource claims.
2. Add process RSS, page-fault, and I/O sampling.
3. Execute and verify benchmark modes.
4. Display storage/RAM evidence in wizard plan tests and runtime history.
5. Run the full browser lifecycle on the target workstation.

## Sprint 3 — Runtime cache correctness

1. Truthful backend capability API.
2. Replace weak globals.
3. Projection-correct admission.
4. Prefill wiring.
5. Safe ownership/allocation.
6. Deterministic cache tests.

Only after those sprints should implementation move to true CPU-miss hybrid execution.

---

# Definition of done

## Web-first lifecycle

A user can configure the machine, add or import a model, compare plans, test, register, load, observe, unload, and remove it from the browser. Failure and restart do not lose the workflow.

## Hardware-aware serving

Every plan and runtime instance has an exact, explainable claim across each GPU, RAM, mmap/page cache, and storage. Reservations prevent conflicting launches.

## Truthful runtime integration

The exact binary is probed in its effective environment. Unsupported flags never launch. Previewed, tested, generated, and launched commands share one fingerprint.

## Measured RAM/SSD behavior

Cold and warm states are labeled honestly. The UI records RSS, page faults, read bytes, load time, TTFT, prompt speed, generation speed, VRAM, and cache behavior.

## Sparse-MoE optimization

The transfer cache is numerically correct and selected only when measured. True hybrid execution performs GPU hits and CPU/RAM/mmap misses during interactive decode and merges outputs correctly.

## Operational safety

Mutations are transactional, jobs are cancellable, runtime operations are recoverable, observations are fingerprinted, and every automatic choice has a visible fallback and decision trace.

## Upstream sustainability

The runtime fork is periodically ported onto current upstream through a protected integration branch. Production remains pinned to a tested SHA, the previous state remains reproducible, stale observations are invalidated by fingerprint, and no submodule promotion occurs without build, deterministic correctness, performance, and full modelctl integration gates.
