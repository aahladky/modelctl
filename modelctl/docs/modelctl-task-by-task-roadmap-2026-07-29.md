# modelctl Task-by-Task Implementation Roadmap

**Date:** July 29, 2026  
**Starting point:** Current `modelctl` repository, current web console and managed worker, and the current `llama.cpp` SYCL expert-transfer-cache branch supplied in the July 29 review bundle.  
**Primary goal:** Deliver a web-first, hardware-aware local model management and serving system that can discover, acquire, analyze, place, test, serve, monitor, and remove models across heterogeneous GPUs, system RAM, and storage, with practical optimization for oversized sparse-MoE models.

---

## 1. Roadmap outcome

The next major release should not be defined as “the MoE cache works.” It should be defined as:

> A user can add or import an oversized sparse-MoE model in the browser, inspect truthful placement candidates, run reproducible cold and warm tests, select a measured plan, register it with llama-swap, load it, observe its use of VRAM/RAM/storage, and safely fall back when the selected backend or placement fails.

The release must provide at least these comparable serving modes:

1. Static GPU placement with CPU/mmap spill.
2. RAM-resident spill where the machine can support it.
3. The current experimental GPU expert-weight transfer cache, only when the exact runtime binary reports support.

True GPU-hit/CPU-miss execution is the next runtime milestone after that release. A managed RAM expert cache and expert prediction remain later evidence-driven work.

---

## 2. Implementation principles

The tasks below follow these constraints.

### 2.1 Preserve the existing architecture

Keep the boundary between:

- `modelctl` as the control plane;
- llama-swap as the stable serving front door;
- backend adapters as the launch and capability boundary;
- `llama.cpp` as the tensor-execution data plane.

Do not move tensor execution or expert routing into Python. Do not move product policy, profile mutation, or machine-wide scheduling into the `llama.cpp` fork.

### 2.2 Build one authoritative launch path

The same resolved launch object must drive:

- browser command preview;
- CLI command preview;
- plan tests;
- managed worker launches;
- fixed-profile llama-swap entries;
- generated artifacts;
- runtime history and decision traces.

No path may reconstruct the command independently after validation.

### 2.3 Ship usefulness before sophistication

Complete the browser workflow, storage observability, and truthful baseline comparison before attempting direct SSD-to-GPU I/O, custom GGUF packing, learned prediction, or a generic plugin system.

### 2.4 Measurements outrank estimates

Planner estimates may generate and prune candidates. Only current, non-stale observations should cause automatic preference for an experimental plan.

### 2.5 Experimental features fail closed

An unsupported, unprobed, stale, or contradictory backend must never receive experimental cache flags. Automatic mode should omit the candidate. Manual mode should block it with an explicit reason.

---

## 3. Release sequence and dependencies

```text
Phase 0  Reproducible baseline and safety net
    │
    ▼
Phase 1  Truthful backend capability and canonical launch contract
    │
    ├──────────────┐
    ▼              ▼
Phase 2        Phase 3
Operation API  Storage as a first-class resource
and rollback   and benchmark dimension
    │              │
    └──────┬───────┘
           ▼
Phase 4  Complete browser add/import/test/register workflow
           │
           ▼
Phase 5  Stabilize the current expert-transfer-cache variant
           │
           ▼
Release A: Web-validated oversized-MoE serving baseline
           │
           ▼
Phase 6  True GPU-hit/CPU-miss hybrid execution
           │
           ▼
Release B: Interactive hybrid sparse-MoE serving
           │
           ▼
Phase 7  Managed RAM tier, only if measurements justify it
           │
           ▼
Phase 8  Prefetch and prediction, only if measurements justify it
```

Phase 1 is the hard dependency for every later phase. Phases 2 and 3 can proceed in parallel after the canonical launch types and capability contract are stable. Phase 5 can be developed alongside the web work, but it must not be presented as production-capable until its runtime tests pass.

---

# Phase 0 — Establish a reproducible baseline

## Task 0.1 — Record the current test and behavior baseline

**Goal:** Capture a known starting point before changing shared launch and profile paths.

**Files:**

- Add: `docs/BASELINE-2026-07-hardware-serving.md`
- Add or update: project test scripts under `scripts/` if that directory is introduced
- Existing tests: `test_modelctl*.py`

**Work:**

- [ ] Record the exact modelctl commit and pinned llama.cpp commit.
- [ ] Record Python, Textual, FastAPI, llama-swap, compiler, oneAPI, kernel, and driver versions.
- [ ] Run every modelctl test module with all declared requirements installed.
- [ ] Build a CPU-only `llama-server` and run `--modelctl-capabilities`.
- [ ] Build the SYCL server on the actual target machine and run the same probe.
- [ ] Save one known-good fixed-profile command and one managed-worker command.
- [ ] Save one baseline plan-test observation for a small model and one oversized MoE.
- [ ] Document currently known failures rather than silently accepting them.

**Tests/verification:**

```bash
python -m unittest discover -v
python -m compileall .
llama-server --modelctl-capabilities
```

**Exit criteria:**

- A baseline document identifies expected failures and distinguishes environment failures from product failures.
- Every later task can demonstrate whether it changed launch behavior, plan identity, or stored observations.

---

## Task 0.2 — Add golden fixtures for profiles, plans, commands, and capabilities

**Goal:** Protect compatibility while the launch path is consolidated.

**Files:**

- Add: `tests/fixtures/profiles/*.json`
- Add: `tests/fixtures/capabilities/*.json`
- Add: `tests/fixtures/plans/*.json`
- Add: `tests/fixtures/commands/*.json`
- Modify: test modules or split into a `tests/` package

**Fixtures should include:**

- [ ] Legacy llama.cpp profile with `kv_quant` only.
- [ ] New profile with separate K/V cache types.
- [ ] OVMS profile.
- [ ] Fixed CPU/mmap spill profile.
- [ ] Managed automatic profile.
- [ ] Experimental transfer-cache profile.
- [ ] Stock llama.cpp capability response: unsupported probe.
- [ ] CPU-only modelctl-capability response.
- [ ] SYCL build without cache support.
- [ ] SYCL build with transfer-cache support.
- [ ] Future hybrid-cache response.

**Exit criteria:**

- Tests can compare normalized profiles, plan IDs, and final tokenized commands without depending on the developer’s machine.

---

# Phase 1 — Make the launch contract truthful and canonical

This phase is the immediate blocker. It addresses the false capability response, disconnected cache preflight, duplicated command construction, and weak-symbol backend integration identified in the review.

## Task 1.1 — Define backend capability schema version 2

**Goal:** Establish one explicit, versioned meaning for backend features.

**Files:**

- Modify: `modelctl_capabilities.py`
- Add: `docs/backend-capability-schema-v2.md`
- Modify in llama.cpp: capability declarations and probe output
- Tests: `test_modelctl_capabilities.py`

**Schema:**

```json
{
  "schema": 2,
  "backend": "llama.cpp",
  "build": {
    "commit": "...",
    "compiler": "...",
    "dynamic_backends": true
  },
  "devices": [
    {
      "type": "SYCL",
      "name": "Intel Arc ...",
      "index": 0,
      "features": {
        "moe_weight_transfer_cache": true
      }
    }
  ],
  "features": {
    "moe_weight_transfer_cache": true,
    "moe_hybrid_cpu_miss": false,
    "moe_cache_metrics": true,
    "moe_cache_prefill_policy": false,
    "moe_cache_reset": true,
    "moe_cache_prefetch": false
  },
  "constraints": {
    "moe_cache_backend": "SYCL",
    "moe_cache_min_batch": 32,
    "moe_cache_supported_projections": ["gate", "up", "down"]
  },
  "cli": {
    "moe_cache_bytes": "--moe-cache-bytes",
    "moe_cache_admission": "--moe-cache-admission",
    "moe_cache_policy": "--moe-cache-policy",
    "moe_cache_prefill": "--moe-cache-prefill-admission"
  }
}
```

**Work:**

- [ ] Rename the current implemented runtime feature to `moe_weight_transfer_cache`.
- [ ] Keep `moe_hybrid_cpu_miss` false until CPU miss execution and output merge exist.
- [ ] Report the actual device/backend requirement.
- [ ] Report known eligibility constraints such as minimum batch or offload path.
- [ ] Treat unknown schema versions conservatively.
- [ ] Preserve schema-0 behavior for stock binaries that reject the probe.
- [ ] Add a normalization function that converts schema 1 responses into a safe internal representation without inventing support.

**Tests:**

- [ ] CPU-only response is never cache-capable.
- [ ] SYCL transfer cache does not imply hybrid CPU misses.
- [ ] Missing or malformed feature fields evaluate false.
- [ ] Unknown newer schema fails closed but preserves diagnostic details.
- [ ] CLI flag names are accepted only when the corresponding feature is true.

**Exit criteria:**

- Feature names describe behavior rather than aspiration.
- A CPU-only binary cannot be classified as SYCL-cache-capable.

---

## Task 1.2 — Replace weak globals with a versioned llama.cpp backend API

**Goal:** Make capability, configuration, metrics, reset, and phase control work with static and dynamically loaded backends.

**Runtime files:**

- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`
- Modify: `ggml/src/ggml-sycl/ggml-sycl.h` or add a private cache API header
- Modify: backend registry/procedure exposure
- Modify: `common/arg.cpp`
- Modify: `tools/server/server.cpp`
- Modify: `tools/server/server-context.cpp`
- Add tests in llama.cpp test targets

**Proposed API:**

```cpp
struct ggml_backend_moe_cache_api_v1 {
    uint32_t abi_version;
    bool (*get_capabilities)(ggml_backend_moe_cache_caps * out);
    bool (*configure)(const ggml_backend_moe_cache_config * config);
    bool (*set_phase)(ggml_backend_moe_phase phase);
    bool (*collect_metrics)(ggml_backend_moe_cache_metrics * out);
    bool (*reset)(uint32_t device_index);
};
```

Retrieve it through `ggml_backend_reg_get_proc_address()` after the backend is loaded.

**Work:**

- [ ] Remove server references to SYCL cache globals and weak functions.
- [ ] Expose one ABI-versioned procedure from the SYCL backend.
- [ ] Make the common capability probe enumerate loaded backend APIs.
- [ ] Configure only backend instances that report the API.
- [ ] Return structured errors for unsupported configuration.
- [ ] Ensure `GGML_BACKEND_DL=ON` works.
- [ ] Ensure non-GNU builds do not depend on weak-symbol behavior.

**Tests:**

- [ ] Static SYCL build finds the API.
- [ ] Dynamic SYCL module finds the API after loading.
- [ ] CPU-only build reports no cache API.
- [ ] Server metrics and reset return unsupported rather than crashing when no API exists.
- [ ] API version mismatch is rejected cleanly.

**Exit criteria:**

- The capability probe is generated from the loaded implementation.
- Dynamic-backend builds can configure and query the cache.

---

## Task 1.3 — Introduce canonical resolved-backend and launch-command types

**Goal:** Stop passing loosely related binary, environment, plan, and argument values through separate code paths.

**Files:**

- Modify: `modelctl_backends.py`
- Modify: `modelctl_plans.py`
- Add: `modelctl_launch.py`
- Modify: `modelctl.py`
- Modify: `modelctl_worker.py`
- Modify: `modelctl_tune.py`
- Tests: add `test_modelctl_launch.py`

**Types:**

```python
@dataclass(frozen=True)
class ResolvedBackend:
    name: str
    binary: str
    binary_fingerprint: str
    environment: dict[str, str]
    environment_fingerprint: str
    capabilities: dict

@dataclass(frozen=True)
class LaunchCommand:
    argv: tuple[str, ...]
    environment: dict[str, str]
    backend: ResolvedBackend
    profile_name: str
    plan_id: str
    port: int | None
    warnings: tuple[str, ...]
    validation: tuple[ValidationMessage, ...]
    command_fingerprint: str
```

**Work:**

- [ ] Move binary selection, oneAPI environment resolution, and capability probing into the backend adapter.
- [ ] Make command generation return tokenized arguments, never a shell string.
- [ ] Separate port-independent command identity from the assigned runtime port.
- [ ] Include the final normalized profile and plan fingerprints in command identity.
- [ ] Make preview rendering quote the canonical token list rather than rebuilding it.
- [ ] Store a redacted environment preview while retaining the actual environment for launch.

**Tests:**

- [ ] Preview tokens equal worker launch tokens apart from port.
- [ ] Plan-test tokens equal managed-worker tokens for the same plan.
- [ ] Fixed llama-swap entry derives from the same validated command.
- [ ] Binary or environment changes alter the command fingerprint.
- [ ] Values containing spaces remain single tokens.

**Exit criteria:**

- There is exactly one authoritative command builder per backend adapter.

---

## Task 1.4 — Integrate capability validation into every launch path

**Goal:** Ensure no unsupported flags reach a binary.

**Files:**

- Modify: `modelctl.py`
- Modify: `modelctl_backends.py`
- Modify: `modelctl_plans.py`
- Modify: `modelctl_worker.py`
- Modify: `modelctl_tune.py`
- Modify: `modelctl_matrix.py`
- Modify: `modelctl_web/app.py`
- Tests across capability, plan, worker, web, and main test modules

**Paths to cover:**

- [ ] Fixed-profile artifact generation.
- [ ] Managed worker launch.
- [ ] Plan testing.
- [ ] Autotuning.
- [ ] CLI smoke test.
- [ ] Web smoke test.
- [ ] llama-swap config generation.
- [ ] Matrix preview and apply.
- [ ] Profile edit/save that triggers regeneration.
- [ ] Cache settings save and restart.

**Behavior:**

- Manual/fixed plan with unsupported requested feature: fail validation with a precise error.
- Automatic planning: omit the unsupported candidate and retain safe plans.
- Probe failure: fail closed for experimental features; allow baseline features according to the adapter’s stock capability policy.
- Stale capability fingerprint: reprobe before launch.

**Tests:**

- [ ] Stock llama.cpp plus cache-enabled profile never receives a cache argument.
- [ ] CPU-only schema-2 binary cannot launch a SYCL cache plan.
- [ ] A changed binary invalidates cached capabilities and observations.
- [ ] UI displays the same rejection reason returned by the operation API.
- [ ] Fixed and managed paths behave consistently.

**Exit criteria:**

- The dead-code gap around `preflight_moe_cache()` is eliminated.

---

## Task 1.5 — Add structured validation and failure classes

**Goal:** Replace free-form preflight text with machine-readable results that the UI and fallback policy can act on.

**Files:**

- Add: `modelctl_errors.py`
- Modify: `modelctl.py`
- Modify: `modelctl_backends.py`
- Modify: `modelctl_plans.py`
- Modify: `modelctl_worker.py`
- Modify: web error templates and job result rendering

**Types:**

```python
@dataclass(frozen=True)
class ValidationMessage:
    code: str
    severity: Literal["info", "warning", "error"]
    summary: str
    detail: str = ""
    recovery: str = ""
    field: str | None = None
```

**Initial codes:**

- `binary_missing`
- `backend_probe_unsupported`
- `backend_feature_missing`
- `backend_device_missing`
- `invalid_cache_budget`
- `insufficient_vram`
- `insufficient_ram`
- `storage_path_unavailable`
- `model_file_missing`
- `invalid_backend_argument`
- `reservation_conflict`
- `health_timeout`
- `backend_crash`
- `numerical_mismatch`

**Exit criteria:**

- The web UI can render a recovery action without parsing English text.
- Fallback suppression keys on stable failure codes.

---

## Task 1.6 — Add command provenance and decision trace persistence

**Goal:** Make every running or historical plan explainable.

**Files:**

- Modify: `modelctl_runtime.py`
- Modify: `modelctl_worker.py`
- Modify: `modelctl_tune.py`
- Modify: runtime and history templates

**Database additions:**

- command fingerprint;
- redacted argv JSON;
- binary path and fingerprint;
- environment fingerprint;
- capability schema and digest;
- normalized claim JSON;
- decision JSON;
- parent job ID;
- selected/fallback ordinal.

**Work:**

- [ ] Add additive migrations rather than replacing the database.
- [ ] Record the exact command before process start.
- [ ] Record why the plan was selected and which candidates were rejected.
- [ ] Link fallback attempts into one launch episode.
- [ ] Show command/capability/claim provenance in runtime and history views.

**Exit criteria:**

- A user can answer “why is this model running this way?” from the browser alone.

---

# Phase 2 — Create one application-operation layer with transactions

## Task 2.1 — Define profile schema version 2 and migration

**Goal:** Make profiles safe to evolve without ad hoc `setdefault()` behavior spread across modules.

**Files:**

- Add: `modelctl_profiles.py`
- Modify: `modelctl.py`
- Modify: `modelctl_web/mutate.py`
- Tests: add `test_modelctl_profiles.py`

**Schema sections:**

```json
{
  "schema": 2,
  "name": "...",
  "source": {},
  "model": {},
  "backend": {},
  "runtime": {},
  "placement": {},
  "cache": {},
  "acquisition": {},
  "advanced": {}
}
```

Existing flat fields should continue to load through a migration function. Do not force an immediate rewrite of all stored profiles.

**Work:**

- [ ] Implement `load_profile_document()`, `normalize_profile_document()`, and `save_profile_document()`.
- [ ] Preserve unknown fields for forward compatibility.
- [ ] Validate types and enumerations.
- [ ] Give experimental cache configuration a distinct structured section.
- [ ] Store source/repository/file information separately from local model paths.
- [ ] Add a schema migration dry-run command and UI preview.

**Exit criteria:**

- Old profiles load identically.
- New code no longer mutates loaded dictionaries in place merely to add defaults.

---

## Task 2.2 — Split callable application services from CLI and HTTP routes

**Goal:** Let the browser and CLI call the same authoritative operations.

**Files to add:**

- `modelctl_services/profile_service.py`
- `modelctl_services/acquisition_service.py`
- `modelctl_services/plan_service.py`
- `modelctl_services/runtime_service.py`
- `modelctl_services/routing_service.py`
- `modelctl_services/hardware_service.py`

**Modify:**

- `modelctl.py`
- `modelctl_web/app.py`
- `modelctl_web/mutate.py`
- `modelctl_worker.py`

**Rules:**

- Service functions receive typed inputs and return typed results.
- Services may not print or call `sys.exit()`.
- CLI handlers convert results to terminal output and exit codes.
- HTTP routes convert results to HTML, redirects, or JSON.
- Long-running services accept `JobContext` for logging, cancellation, and subprocess registration.

**Migration order:**

1. Runtime load/unload/restart.
2. Profile save/remove/regenerate.
3. Plan compile/select/test/tune.
4. Acquisition and import.
5. Routing apply/rollback.
6. Hardware settings.

**Exit criteria:**

- No web route shells out to the modelctl CLI.
- No CLI handler contains the only implementation of a product operation.

---

## Task 2.3 — Standardize job callback signatures

**Goal:** Remove the mixed job-ID/store and `JobContext` patterns.

**Files:**

- Modify: `modelctl_web/jobs.py`
- Modify: `modelctl_web/mutate.py`
- Modify: service modules
- Tests: `test_modelctl_web.py` and new job tests

**Work:**

- [ ] Require all long operations to use `fn(context: JobContext) -> JobResult`.
- [ ] Add structured progress events: phase, current, total, unit, message.
- [ ] Add structured result payload and failure payload.
- [ ] Preserve process-group cancellation.
- [ ] Add child-job support for wizard stages if needed, but retain one parent operation in the UI.

**Exit criteria:**

- Every long browser operation is cancellable and exposes consistent progress.

---

## Task 2.4 — Add atomic multi-file mutation transactions

**Goal:** Prevent partial profiles, artifacts, or llama-swap configuration after a failed operation.

**Files:**

- Add: `modelctl_transactions.py`
- Modify: profile, routing, and acquisition services
- Modify: artifact and sync functions

**Transaction behavior:**

1. Stage files in the same filesystem.
2. Validate generated JSON/YAML and commands.
3. Capture hashes/backups of managed targets.
4. Atomically replace profile and generated artifacts.
5. Apply the managed llama-swap section.
6. Restart or reload.
7. Run health validation.
8. Roll back all managed files if the operation fails.

**Tests:**

- [ ] Failure after profile staging leaves no profile.
- [ ] Failure during llama-swap validation restores previous configuration.
- [ ] Restart failure rolls back configuration and reports the recovery result.
- [ ] Concurrent profile edits serialize or fail with a version conflict.

**Exit criteria:**

- A failed mutation cannot leave a half-created model or broken managed routing section.

---

## Task 2.5 — Add optimistic profile revision checks

**Goal:** Prevent one browser tab or background job from overwriting newer changes.

**Files:**

- Modify profile schema/service
- Modify forms and API responses

**Work:**

- [ ] Store a profile revision or content fingerprint.
- [ ] Include it in edit, cache, plan-selection, and policy forms.
- [ ] Reject stale writes with a conflict page showing current versus submitted values.
- [ ] Let explicitly idempotent operations retry against the latest revision.

**Exit criteria:**

- Parallel edits do not silently lose data.

---

# Phase 3 — Make storage a first-class hardware and measurement tier

The existing `StorageSnapshot` is only `path`, `kind`, and `allow_mmap`. This phase expands it enough to support truthful RAM/SSD planning and benchmarking.

## Task 3.1 — Expand `StorageSnapshot` and probe storage topology

**Files:**

- Modify: `modelctl_hardware.py`
- Add: `modelctl_storage.py`
- Modify: hardware settings schema and page
- Tests: add `test_modelctl_storage.py`

**Fields:**

```python
@dataclass(frozen=True)
class StorageSnapshot:
    path: str
    mount_point: str
    filesystem: str
    block_devices: tuple[str, ...]
    transport: str
    rotational: bool | None
    raid_level: str | None
    total_bytes: int
    free_bytes: int
    allow_mmap: bool
    measured_sequential_read_bps: int | None
    measured_random_read_bps: int | None
    measurement_age_seconds: float | None
```

**Work:**

- [ ] Resolve a model path through mount info to backing block devices.
- [ ] Detect mdraid/device-mapper membership without assuming a single disk.
- [ ] Read rotational and transport information from sysfs where possible.
- [ ] Preserve user overrides for unusual arrays and filesystems.
- [ ] Include storage policy in the hardware fingerprint.
- [ ] Do not run destructive or sustained benchmarks automatically.

**Exit criteria:**

- The hardware page identifies the actual storage backing each model directory.

---

## Task 3.2 — Add storage and RAM detail to resource claims

**Files:**

- Modify: `modelctl_plans.py`
- Modify: `modelctl_matrix.py`
- Modify: `modelctl_runtime.py`
- Modify: plans and routing templates

**Extend `ResourceClaim`:**

- `ram_bytes`: total resident/staging estimate;
- `ram_breakdown`: model pages, CPU tensors, staging, overhead;
- `storage_mode`: none, mmap, expected-page-cache, active-streaming;
- `storage_path`;
- `model_bytes`;
- `expected_resident_bytes`;
- `expected_read_bytes_per_token`, initially optional/estimated;
- `cache_bytes` separately from static VRAM.

**Work:**

- [ ] Derive storage path from the actual model file rather than a global default.
- [ ] Show claim confidence and which values are estimates.
- [ ] Include storage mode in stable plan identity.
- [ ] Make matrix compatibility consider RAM as well as VRAM.

**Exit criteria:**

- A plan explanation says which bytes are expected in VRAM, RAM, and storage.

---

## Task 3.3 — Add process I/O and page-fault sampling

**Files:**

- Modify: `modelctl_tune.py`
- Modify: `modelctl_runtime.py`
- Modify: history templates
- Tests with fixture `/proc` data

**Sampler metrics:**

- `/proc/<pid>/io`: read bytes and syscall bytes;
- `/proc/<pid>/stat` or `status`: minor/major faults where available;
- RSS, peak RSS, and child-process aggregate;
- GPU memory by device;
- elapsed load I/O;
- read bytes during warmup and measured generation separately.

**Database additions:**

- `read_bytes`;
- `read_bytes_warmup`;
- `read_bytes_generation`;
- `major_faults`;
- `minor_faults`;
- `storage_path`;
- `cache_state` and benchmark phase.

**Exit criteria:**

- A plan run can distinguish compute speed from active storage reads.

---

## Task 3.4 — Define explicit cold and warm benchmark modes

**Files:**

- Modify: `modelctl_tune.py`
- Modify: plan-test/autotune service and web forms
- Add documentation: `docs/benchmark-protocol.md`

**Modes:**

- `natural`: no cache manipulation; record current state.
- `process-cold`: new backend process, OS page cache unchanged.
- `page-cache-warm`: run a controlled warmup before measurement.
- `expert-cache-warm`: warmup until cache counters stabilize or a token budget is reached.
- `storage-cold`: optional privileged operation with explicit consent and a clear scope.

**Safety:**

- [ ] Never issue global `drop_caches` silently.
- [ ] Prefer file-specific eviction/advice if a reliable platform method is available.
- [ ] Label results `cold_unverified` when cold state cannot be guaranteed.
- [ ] Store the method used in `details_json`.

**Exit criteria:**

- The UI cannot label a page-cache-warm run as an SSD-cold benchmark.

---

## Task 3.5 — Add storage calibration jobs

**Files:**

- Add service operation and web route/page section
- Modify hardware settings

**Work:**

- [ ] Add an opt-in, bounded sequential read calibration using a user-selected test file or model file.
- [ ] Add an optional representative random-read calibration with a capped read volume.
- [ ] Record method, file size, direct/buffered mode, time, and result age.
- [ ] Allow manual overrides for arrays where synthetic tests are misleading.
- [ ] Do not rank a plan solely from calibration numbers; use them as planner hints.

**Exit criteria:**

- Storage estimates use machine measurements when available and clearly say when they do not.

---

# Phase 4 — Complete the browser model lifecycle

## Task 4.1 — Add persistent add-model wizard state

**Goal:** Turn the separate pull, plan, tune, and runtime pages into one resumable workflow.

**Files:**

- Add: `modelctl_web/wizard.py`
- Add: wizard templates and fragments
- Modify: `modelctl_runtime.py` or add a small wizard-state table
- Modify: web navigation

**Wizard state fields:**

- wizard ID and owner/session;
- source type;
- repository or local path;
- selected files/quant/extras;
- download job and verification state;
- temporary profile draft;
- analysis result;
- candidate plan IDs;
- test observations;
- selected runtime policy;
- transaction result;
- expiry time.

**Work:**

- [ ] Persist enough state to survive browser reloads and service restarts.
- [ ] Make each step idempotent.
- [ ] Allow back navigation without repeating downloads.
- [ ] Expire abandoned drafts and clean temporary files safely.

**Exit criteria:**

- A failed benchmark or browser refresh does not force the user to restart acquisition.

---

## Task 4.2 — Implement source selection and local import

**Files:**

- Modify acquisition service
- Add wizard source template
- Add API endpoints

**Sources:**

- Hugging Face repository;
- existing local GGUF file;
- existing local model directory;
- future connector/import types remain out of scope.

**Local import behavior:**

- [ ] Validate readability and supported file types.
- [ ] Detect split GGUF parts.
- [ ] Detect mmproj and MTP companions.
- [ ] Let the user reference files in place or copy/move into the managed model directory.
- [ ] Never delete source files during a failed transaction.

**Exit criteria:**

- A local GGUF can be added, analyzed, and served entirely from the browser.

---

## Task 4.3 — Build repository and quant inspection as a typed wizard step

**Files:**

- Refactor existing `/pull` functionality into acquisition service
- Modify `pull.html`/`pull_repo.html` or replace with wizard templates

**Work:**

- [ ] Preserve repository search, tags, sorting, and size filters.
- [ ] Display quant groups, shard count, total size, mmproj, and MTP companions.
- [ ] Show required free storage before submission.
- [ ] Show estimated RAM/VRAM fit ranges as advisory only.
- [ ] Validate the selected files against the repository listing at submit time.

**Exit criteria:**

- The user knows exactly which files will be downloaded and how large they are.

---

## Task 4.4 — Add resumable download and verification records

**Files:**

- Modify acquisition service
- Modify jobs/progress handling
- Add acquisition tables or sidecar manifest

**Work:**

- [ ] Record every selected file, expected size, ETag/revision where available, and local path.
- [ ] Resume partial downloads using the supported Hugging Face client behavior.
- [ ] Verify size and available checksum/ETag after completion.
- [ ] Mark files reusable by a restarted wizard.
- [ ] Distinguish cancelled, interrupted, corrupt, and complete states.
- [ ] Keep download jobs in the dedicated lane.

**Exit criteria:**

- A service restart or cancellation does not require discarding valid completed shards.

---

## Task 4.5 — Run automatic post-download GGUF analysis

**Files:**

- Refactor existing `modelctl_vram.py` analysis into an acquisition/analysis service
- Add analysis result model and wizard template

**Analysis output:**

- architecture;
- total weight bytes;
- expert and shared-expert tensor geometry;
- layer count and expert count;
- context metadata;
- vision/MTP companions;
- exact tensor families relevant to placement;
- unsupported or uncertain architecture warnings.

**Work:**

- [ ] Cache analysis by model-file fingerprint.
- [ ] Invalidate when any shard changes.
- [ ] Display confidence and unsupported assumptions.
- [ ] Block experimental cache plans for unrecognized expert geometry.

**Exit criteria:**

- Candidate plans are generated from the files actually downloaded, not repository-name heuristics.

---

## Task 4.6 — Embed plan comparison into the wizard

**Files:**

- Reuse/refactor `plans.html`
- Modify plan service and wizard

**Display for each candidate:**

- source and label;
- validation state;
- exact binary/capability fingerprint;
- exact command preview;
- per-GPU VRAM breakdown;
- RAM and storage claims;
- expected context;
- estimated status versus current measured result;
- warnings and rejected alternatives;
- experimental badge and eligibility constraint.

**Work:**

- [ ] Default to safe baseline candidates.
- [ ] Hide experimental variants behind a clear opt-in setting unless already enabled globally.
- [ ] Let the user choose “test selected,” “autotune bounded candidates,” or “register without testing.”
- [ ] Warn that untested estimates may be wrong.

**Exit criteria:**

- The user need not leave the wizard to understand or test placement options.

---

## Task 4.7 — Add wizard-integrated test and autotune stages

**Files:**

- Modify plan-test/autotune service
- Modify jobs and wizard templates

**Work:**

- [ ] Let the user select objective: fastest generation, fastest prompt, lowest RAM, lowest power proxy, or balanced.
- [ ] Show cold/warm benchmark mode explicitly.
- [ ] Show live phase, load time, prompt speed, generation speed, RAM, VRAM, I/O, and cache metrics.
- [ ] Preserve failed candidates with classified reasons.
- [ ] Never auto-select an experimental plan without a successful current observation.

**Exit criteria:**

- The wizard can rank at least static mmap, RAM-resident where feasible, and transfer-cache variants from real measurements.

---

## Task 4.8 — Add runtime policy, registration, and warm-load verification

**Files:**

- Modify runtime/profile/routing services
- Add final wizard steps

**Work:**

- [ ] Choose fixed, manually selected managed, or automatically ranked policy.
- [ ] Configure idle TTL, load timeout, fallback count, and pinning.
- [ ] Stage profile, artifacts, and managed llama-swap matrix in one transaction.
- [ ] Validate llama-swap configuration before replacement.
- [ ] Load through the public front door, not a private test-only endpoint.
- [ ] Send a small warm-load request and verify the OpenAI-compatible response.
- [ ] Display final endpoint, selected plan, command fingerprint, and links to runtime/history.

**Exit criteria:**

- A model can go from source selection to a working endpoint without a terminal.

---

## Task 4.9 — Add a first-class settings page

**Files:**

- Add settings template/routes/service
- Migrate relevant defaults from CLI-only prompts

**Settings:**

- managed model directories;
- profile/state directories;
- llama.cpp binary candidates;
- oneAPI environment scripts;
- llama-swap config/service/base URL;
- RAM reserve;
- device roles and reserves;
- storage locations and calibration;
- experimental-feature opt-in;
- default context and cache settings;
- authentication token rotation.

**Exit criteria:**

- Normal machine configuration is possible from the browser.

---

# Phase 5 — Stabilize the existing expert-weight transfer cache

This phase deliberately narrows the runtime claim. It does not implement hybrid CPU misses. It makes the current D2D-on-hit/H2D-on-miss mechanism correct, measurable, and safely selectable.

## Task 5.1 — Rename the feature and metrics to match behavior

**Files:**

- Modify capability schema and llama.cpp output
- Modify `modelctl_capabilities.py`
- Modify profile/cache labels and templates
- Modify benchmark docs

**Renames:**

- Feature: `moe_weight_transfer_cache`.
- Fallback metric: `host_projection_fallbacks` or `host_weight_copy_fallbacks`.
- Do not call fallback counts CPU expert executions.

**Exit criteria:**

- UI text accurately states that misses still transfer weights to GPU.

---

## Task 5.2 — Make admission projection-specific

**Runtime files:**

- `ggml/src/ggml-sycl/moe-cache.hpp`
- `ggml/src/ggml-sycl/moe-cache.cpp`
- Runtime unit tests

**Work:**

- [ ] Index miss counts by layer, expert, and projection.
- [ ] Reset only the projection that hits.
- [ ] Do not count geometry-incompatible projections toward admission.
- [ ] Define whether threshold means N misses or N routed expert activations and document it.
- [ ] Add an admission-1 fast path without changing threshold-N semantics.

**Tests:**

- [ ] Gate/up/down all eventually promote at threshold 2.
- [ ] One projection hit cannot starve another projection.
- [ ] Fused or unsupported projections do not distort counters.

**Exit criteria:**

- Default admission behavior matches its documented meaning.

---

## Task 5.3 — Wire prefill policy and phase through lazy initialization

**Runtime files:**

- Backend cache API/configuration
- server batch-phase signaling
- cache constructor/init

**Work:**

- [ ] Validate `on`/`off` CLI values strictly.
- [ ] Store configured prefill admission in backend configuration.
- [ ] Store current phase even before a cache instance exists.
- [ ] Make newly created caches inherit that phase.
- [ ] Represent mixed continuous-batching phase explicitly or document conservative behavior.
- [ ] Split prefill/decode metrics.

**Tests:**

- [ ] Initial prefill cannot populate cache when disabled.
- [ ] Initial prefill can populate cache when enabled.
- [ ] Decode admission works after prefill.
- [ ] Lazy creation does not misclassify the first batch.

---

## Task 5.4 — Correct cache ownership and multi-context lifetime

**Runtime files:**

- SYCL cache registry and backend context teardown

**Choose one design:**

1. Device-level shared cache with reference-counted contexts and namespaced model identity; or
2. Synchronized registry of context-owned caches aggregated by metrics/reset APIs.

For the current milestone, the second option is likely safer unless slots are intentionally reusable across models.

**Work:**

- [ ] Replace one raw pointer per device.
- [ ] Protect registry lookup and destruction.
- [ ] Ensure metrics hold a safe reference while collecting.
- [ ] Ensure resetting one model does not reset another unless explicitly requested.
- [ ] Account for main and draft/MTP contexts.
- [ ] Enforce total per-device cache budget across contexts or expose separate claims.

**Tests:**

- [ ] Two contexts on one GPU remain visible.
- [ ] Destroying context A does not hide or free context B.
- [ ] Concurrent metrics and teardown do not race.
- [ ] Reset targets the requested context/device.

---

## Task 5.5 — Use backend-native allocation and exact geometry

**Runtime files:**

- SYCL cache allocator and geometry code

**Work:**

- [ ] Use `ggml_sycl_malloc_device()`/matching free path.
- [ ] Check allocation failure and reduce/disable cache gracefully.
- [ ] Prefer one contiguous pool per cache over hundreds of USM allocations.
- [ ] Represent projection sizes independently.
- [ ] Key cache entries by exact tensor identity where architecture geometry is uncertain.
- [ ] Handle fused gate/up tensors explicitly.
- [ ] Report unsupported tensor layout through capabilities or model-load diagnostics.

**Tests:**

- [ ] Unequal gate/up/down sizes.
- [ ] Fused gate/up layout.
- [ ] Partial allocation failure.
- [ ] Cache budget smaller than one slot.
- [ ] Multi-GPU allocations use the correct device context.

---

## Task 5.6 — Make metrics valid Prometheus and structured JSON

**Runtime files:**

- server metrics endpoint
- cache metrics API

**Metrics:**

- lookups, hits, misses, promotions, evictions;
- fallback projection copies;
- bytes H2D avoided and bytes D2D copied;
- cache bytes allocated/used;
- entries by protected/probation segment;
- prefill/decode/mixed phase counters;
- per-device and per-context labels;
- reset generation.

**Work:**

- [ ] Emit HELP/TYPE metadata once.
- [ ] Keep counter names and units stable.
- [ ] Add a schema/version field to JSON metrics.
- [ ] Make reset atomic relative to metric snapshots.

**Exit criteria:**

- Standard Prometheus parsers accept multi-GPU output.

---

## Task 5.7 — Add deterministic cache-runtime correctness tests

**Files:**

- Add llama.cpp cache unit tests and integration harness
- Add modelctl acceptance wrapper for real hardware

**Required cases:**

- cache disabled;
- all forced misses;
- all forced hits;
- alternating hit/miss;
- multiple selected experts;
- shared expert plus routed expert;
- eviction and re-admission;
- reset during idle state;
- two contexts;
- two GPUs;
- prefill disabled/enabled;
- admission thresholds 1 and 2;
- unequal projections.

**Comparison:**

- Prefer logits or expert output tensors within a documented tolerance.
- If token comparison is used, force deterministic settings and compare token IDs, not merely final prose.

**Exit criteria:**

- The cache cannot be called numerically correct based only on coherent generated text.

---

## Task 5.8 — Encode cache eligibility in planning

**Files:**

- Modify `modelctl_plans.py`
- Modify capability normalization
- Modify plan UI

**Work:**

- [ ] Require the exact backend/device feature.
- [ ] Require recognized expert geometry.
- [ ] Reserve cache budget before static VRAM placement.
- [ ] Warn when the backend hook is inactive at the expected batch size.
- [ ] Do not rank the variant for interactive generation when it cannot affect batch-one decode.
- [ ] Keep the variant disabled by default unless the user enables experimental plans.

**Exit criteria:**

- The planner does not imply a decode benefit that the runtime cannot provide.

---

# Phase 6 — Release A: Web-validated oversized-MoE serving baseline

This is an integration and product milestone, not a new algorithm.

## Task 6.1 — Define the baseline candidate set

For each suitable oversized MoE, generate a bounded set containing:

- [ ] Stock/static CPU+mmap spill.
- [ ] RAM-resident CPU spill when estimated to fit with reserve.
- [ ] Best static heterogeneous-GPU placement.
- [ ] Experimental transfer-cache variant when eligible.
- [ ] One conservative fallback with reduced context.

Do not generate near-duplicate variants that differ only by insignificant cache increments.

---

## Task 6.2 — Implement the standardized benchmark suite

**Prompt sets:**

- short interactive generation;
- long prefill followed by generation;
- repeated locality prompt;
- varied-routing prompts;
- optional code task, clearly separated from generic text.

**Measurements:**

- load time;
- TTFT;
- prompt TPS;
- generation TPS;
- actual context;
- per-GPU peak VRAM;
- peak RAM;
- read bytes and page faults;
- cache metrics;
- correctness result;
- energy/power only if a reliable existing sensor is available.

**Rules:**

- [ ] Store cold and warm observations separately.
- [ ] Tie observations to hardware, backend, capability, model, and command fingerprints.
- [ ] Mark old observations stale automatically.
- [ ] Require current successful observation before an experimental plan can win automatic ranking.

---

## Task 6.3 — Add objective-aware measured ranking

**Files:**

- Modify `modelctl_plans.py`
- Modify `modelctl_tune.py`
- Modify runtime-policy UI

**Objectives:**

- interactive generation;
- prompt ingestion;
- balanced;
- minimum RAM;
- minimum active storage reads.

**Ranking behavior:**

- [ ] Correctness failure always disqualifies.
- [ ] Unsupported or stale capability disqualifies.
- [ ] Repeated backend crash strongly suppresses.
- [ ] Estimated plans rank below current successful measured plans.
- [ ] Experimental plans require a configurable minimum improvement over the safe baseline.
- [ ] Display the score breakdown.

---

## Task 6.4 — Complete the runtime decision trace UI

**Files:**

- Modify runtime, plans, and history templates

**Display:**

- selected plan and objective;
- baseline comparison;
- current command and binary fingerprint;
- resource claims versus actual peaks;
- storage and cache state;
- fallback chain;
- rejected candidates and reasons;
- stale observation warnings.

---

## Task 6.5 — Run the Release A acceptance matrix

**Hardware cases:**

- single GPU, model fits;
- asymmetric two-GPU system;
- model exceeds total VRAM but fits RAM;
- model exceeds practical RAM and relies on mmap/storage;
- supported SYCL transfer-cache binary;
- stock/unsupported llama.cpp binary;
- service restart and reboot survival where approved.

**Required product flow:**

1. Open browser.
2. Add/import model.
3. Download or reference local files.
4. Analyze.
5. Compare plans.
6. Test selected candidates.
7. Register.
8. Load through llama-swap.
9. Observe resource use.
10. Unload and reload.
11. Remove or disable safely.

**Release A definition of done:**

- No terminal is required for the normal flow.
- Unsupported flags never reach a backend.
- The shown command matches the launched command.
- RAM/VRAM/storage use is visible.
- The selected plan has an explainable measured basis.
- Failed mutations and launches recover cleanly.

---

# Phase 7 — Implement true GPU-hit/CPU-miss execution

This phase fulfills the central MoE optimization goal. It should begin only after the control-plane contract and deterministic runtime harness are stable.

## Task 7.1 — Write a focused hybrid-execution design and spike

**Goal:** Resolve graph, tensor-layout, synchronization, and ownership questions before broad implementation.

**Files:**

- Add: `docs/runtime/hybrid-moe-execution-design.md`
- Add a small experimental branch/harness in llama.cpp

**Questions to answer:**

- Where are router-selected token/expert pairs available in a form suitable for partitioning?
- Can existing CPU and SYCL `mul_mat_id` implementations operate on disjoint row subsets without graph-wide duplication?
- What is the minimum synchronization boundary for output merge?
- How are shared experts handled?
- How are quantized expert tensors represented on CPU and GPU?
- Can one activation select a transfer path for some misses and CPU execution for others?
- What batch-one path currently bypasses the GPU split hook?

**Exit criteria:**

- The design identifies exact graph/backend insertion points and includes a numerical test strategy.

---

## Task 7.2 — Define hybrid capability and configuration contracts

**Capability fields:**

- `moe_hybrid_cpu_miss`;
- supported architectures/projection layouts;
- supported quant types;
- minimum/maximum batch constraints;
- whether CPU and GPU work can overlap;
- metrics version.

**Configuration:**

- mode: off, transfer-cache, hybrid;
- GPU cache budget per device;
- CPU miss policy;
- transfer-versus-CPU threshold, initially fixed/conservative;
- prefill policy;
- synchronization/debug mode.

**Exit criteria:**

- Modelctl can distinguish transfer-cache and hybrid plans without guessing.

---

## Task 7.3 — Extract and represent routed work partitions

**Runtime work:**

- [ ] Build a compact representation of selected `(token row, expert)` work.
- [ ] Mark GPU-cache hits and misses before expert computation.
- [ ] Preserve deterministic row ordering for merge.
- [ ] Handle multiple experts per token.
- [ ] Handle shared experts separately.
- [ ] Avoid host round trips for hit classification where possible, but accept a correct initial implementation before optimizing.

**Tests:**

- synthetic router selections;
- repeated experts;
- no hits, all hits, and mixed;
- multiple token rows and experts.

---

## Task 7.4 — Implement CPU execution for misses

**Runtime work:**

- [ ] Execute only miss rows against CPU-resident or mmap-backed expert weights.
- [ ] Reuse existing quantized CPU kernels where possible.
- [ ] Avoid materializing complete expert copies unnecessarily.
- [ ] Track CPU execution time and page faults/read bytes if available.
- [ ] Ensure CPU miss execution does not force a synchronous GPU weight copy.

**Exit criteria:**

- A forced-miss test completes with zero expert H2D weight copies for the miss path.

---

## Task 7.5 — Execute hit partitions on persistent GPU slots

**Runtime work:**

- [ ] Use persistent cached expert tensors directly or copy from slots into the expected staging layout, whichever can be proven correct first.
- [ ] Submit work on an explicit queue/event chain.
- [ ] Do not globally synchronize after every expert.
- [ ] Account for selected-expert reuse within the same batch.

**Exit criteria:**

- Forced-hit output matches the cache-disabled baseline.

---

## Task 7.6 — Merge CPU and GPU outputs correctly

**Runtime work:**

- [ ] Allocate an output representation that supports disjoint CPU/GPU contributions.
- [ ] Merge by original token row and expert weight.
- [ ] Preserve accumulation order where required for numerical stability.
- [ ] Support all-hit and all-miss cases without special corruption-prone paths.
- [ ] Add debug assertions for row coverage and duplicate contribution.

**Tests:**

- exact synthetic small tensors;
- random mixed selections;
- multiple expert contributions per token;
- shared expert plus routed experts;
- comparison across quant types.

**Exit criteria:**

- Mixed execution passes deterministic numerical comparison.

---

## Task 7.7 — Add asynchronous admission and promotion

**Work:**

- [ ] Record misses without blocking current-token completion.
- [ ] Promote eligible experts after the miss result is no longer dependent on staging buffers.
- [ ] Use explicit events rather than queue-wide waits.
- [ ] Rate-limit promotion to avoid saturating H2D bandwidth.
- [ ] Prevent duplicate concurrent promotion of the same projection.
- [ ] Preserve correctness when eviction occurs during in-flight work.

**Exit criteria:**

- Promotion overhead is visible and does not introduce hidden global synchronization.

---

## Task 7.8 — Add hybrid metrics and cost evidence

**Metrics:**

- routed expert calls;
- hit rows and miss rows;
- CPU miss execution count/time;
- GPU hit execution count/time;
- output merge time;
- promotion bytes/time;
- H2D bytes avoided;
- active storage read bytes during misses;
- per-layer hit ratio and cost;
- queue overlap where measurable.

**Exit criteria:**

- A slow hybrid result can be diagnosed as CPU compute, storage, merge, or promotion overhead.

---

## Task 7.9 — Add hybrid plans to modelctl

**Files:**

- Modify capabilities, plans, claims, adapter, tune, runtime DB, and web templates

**Work:**

- [ ] Generate hybrid variants only for supported binary/model/device combinations.
- [ ] Claim GPU cache bytes and estimated CPU/RAM working set.
- [ ] Label CPU/mmap backing explicitly.
- [ ] Add conservative fallback to static CPU/mmap.
- [ ] Require measured success before automatic selection.
- [ ] Rank batch-one generation separately from prompt throughput.

---

## Task 7.10 — Run Release B acceptance

**Required success:**

- [ ] Correct all-hit, all-miss, and mixed output.
- [ ] Batch-one interactive decode enters the hybrid path.
- [ ] A miss does not automatically transfer full expert weights to GPU.
- [ ] Representative oversized MoE warm generation improves meaningfully over the best static CPU/mmap baseline.
- [ ] No unacceptable regression when cache is disabled.
- [ ] Browser shows CPU/GPU/storage work and selection reason.

Release B should not be declared from hit-rate alone.

---

# Phase 8 — Add a managed RAM tier only if measurements justify it

## Entry condition

Begin this phase only when Release A/B observations demonstrate at least one of:

- repeated major faults during steady-state generation;
- unstable page-cache residency under normal machine pressure;
- excessive storage read amplification;
- a reproducible advantage from pinning a bounded expert working set in RAM.

## Task 8.1 — Design a byte-budgeted complete-expert RAM cache

**Requirements:**

- complete projection set per cached expert where architecture permits;
- byte-based budget;
- asynchronous reads;
- per-model/context ownership;
- admission tied to measured reuse;
- no duplicate copy when the page cache already provides equivalent residency.

## Task 8.2 — Integrate RAM cache claims and metrics

Track:

- pinned/resident RAM bytes;
- cache hits/misses;
- SSD read bytes avoided;
- read queue depth and latency;
- eviction reason;
- interaction with OS page cache.

## Task 8.3 — Prove benefit against normal mmap

The managed RAM tier should ship only if it improves total token latency or stability compared with normal mmap/page cache under realistic memory pressure.

---

# Phase 9 — Add prefetch and prediction only after the miss path is proven

## Task 9.1 — Implement simple heuristic prefetch

Start with cheap, explainable heuristics:

- recent expert transitions;
- layer-local reuse;
- small lookahead where router output is already available;
- strict byte and bandwidth budget.

## Task 9.2 — Measure overfetch cost

Record:

- useful prefetched bytes;
- unused prefetched bytes;
- promotion delay hidden;
- bandwidth stolen from active execution;
- total latency impact.

## Task 9.3 — Consider learned prediction only if heuristics show headroom

Do not adopt a predictor because it improves hit rate. Require a reproducible reduction in total token latency after accounting for overfetch and compute overhead.

---

# 10. Cross-cutting test plan

## 10.1 Modelctl unit tests

- Profile migration and validation.
- Capability schema normalization.
- Exact binary/environment resolution.
- Canonical command equality across paths.
- Resource claims and plan IDs.
- Storage topology fixtures.
- Reservation accounting for VRAM, RAM, and dynamic cache.
- Ranking and staleness.
- Structured failure behavior.
- Transaction rollback.
- Wizard state transitions and resumability.

## 10.2 Modelctl concurrency tests

- Two simultaneous reservation requests.
- Runtime unload while download/benchmark lanes are busy.
- Cancellation during download, test, and launch.
- Profile revision conflict.
- Routing apply versus profile mutation.
- Service restart with pending wizard/job state.

## 10.3 Web tests

- Authentication and CSRF-equivalent protection appropriate to the current app.
- Complete add-model wizard happy path.
- Local import path.
- Interrupted download resume.
- Unsupported plan rejection.
- Plan test progress and cancellation.
- Transaction rollback surfaced to user.
- Runtime decision trace.
- Storage and cache metrics rendering with missing/partial metrics.

## 10.4 Runtime tests

- Capability API static and dynamic backend builds.
- Cache geometry and admission.
- Prefill/decode state.
- Multi-context lifetime.
- Metrics/reset concurrency.
- Forced hit/miss/mixed numerical correctness.
- Fault injection for allocation and transfer failures.
- Hybrid partition/CPU/GPU/merge correctness.

## 10.5 Real-hardware acceptance matrix

At minimum, keep a documented matrix for:

- CPU-only build;
- supported Intel SYCL GPU;
- asymmetric multi-GPU machine;
- model fitting one GPU;
- model fitting aggregate VRAM only;
- model fitting RAM but not VRAM;
- model relying on active storage;
- stock llama.cpp;
- experimental transfer-cache build;
- future hybrid build.

---

# 11. Branch and commit strategy

Keep the control plane and runtime fork separate.

## modelctl branches

- `feature/launch-contract-v2`
- `feature/operation-services`
- `feature/storage-observability`
- `feature/web-add-model-workflow`
- `feature/transfer-cache-productization`
- `feature/hybrid-plan-integration`

## llama.cpp branches

- `feature/modelctl-backend-api`
- `fix/moe-transfer-cache-correctness`
- `feature/moe-hybrid-cpu-miss`

Each task above should normally be one reviewable commit or a short commit series with tests introduced before or alongside implementation. Avoid a single cross-repository “big bang” branch.

The modelctl submodule pin should advance only after:

1. the runtime commit passes its own tests;
2. modelctl capability fixtures are updated;
3. the modelctl integration tests pass;
4. the real-hardware acceptance note is updated.

---

# 12. Immediate first sprint

The first sprint should stay narrowly focused on trust and should not add new UI features or cache policies.

## Sprint Task A — Capability schema and truthful runtime response

- Implement schema 2.
- Report transfer cache only from a loaded SYCL backend.
- Report hybrid CPU miss and prefill policy false until implemented.
- Add CPU-only and dynamic-backend tests.

## Sprint Task B — Canonical launch object

- Add `ResolvedBackend` and `LaunchCommand`.
- Move binary/environment/capability resolution into `LlamaCppAdapter`.
- Make worker and plan test consume the same object.

## Sprint Task C — Connect validation to all launch paths

- Fixed profile.
- Managed worker.
- Plan test/autotune.
- llama-swap generation.
- Web cache save/restart.

## Sprint Task D — Command equality and regression tests

- Add golden fixtures.
- Prove preview equals launch.
- Prove unsupported binaries receive no cache flags.
- Prove binary changes stale capabilities and observations.

## Sprint exit criteria

- CPU-only `llama-server --modelctl-capabilities` reports no SYCL cache.
- Stock llama.cpp cannot be launched with modelctl cache flags through any supported path.
- The browser command preview and worker command are derived from the same immutable launch object.
- Existing non-cache profiles still render and launch identically.

This sprint creates the trustworthy foundation required for every web and runtime feature that follows.

---

# 13. Overall definition of done

The primary project goal is achieved when all of the following are true:

## Web-first lifecycle

- A user can discover or import, acquire, analyze, configure, test, register, serve, monitor, unload, and remove a model through the browser.
- Every long operation is observable and cancellable.
- Failed mutations roll back.

## Hardware-aware serving

- Plans account for per-device VRAM, RAM, dynamic cache, and storage mode.
- Reservations prevent concurrent overcommit.
- The exact backend and environment are validated before launch.
- Automatic choices are explainable and recoverable.

## Measured optimization

- Observations are tied to hardware, backend, capability, model, and command fingerprints.
- Cold and warm behavior are not conflated.
- The UI shows estimates versus actual resource use and performance.

## Oversized sparse-MoE support

- Safe static CPU/mmap and RAM-resident baselines are first-class.
- The transfer cache is correctly named, tested, capability-gated, and selected only when measured to help.
- True hybrid mode can execute GPU hits and CPU/RAM/mmap misses during batch-one decode and merge outputs correctly.
- A representative oversized MoE demonstrates a meaningful improvement over the best static baseline.

## Operational trust

- Unsupported flags never reach a backend.
- Previewed and launched commands are the same.
- Runtime state, logs, metrics, claims, and decision history are visible from the browser.
- The service survives normal restart and recovery scenarios without manual reconstruction.

At that point, modelctl is not merely a launcher with experimental cache controls. It is a coherent local inference platform built for the exact problem that motivated the project: making unusual combinations of GPU memory, system RAM, CPU compute, and fast local storage usable through one dependable web control plane.
