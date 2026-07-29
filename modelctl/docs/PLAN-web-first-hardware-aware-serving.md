# modelctl Web-First Hardware-Aware Serving Plan

## 1. Purpose

Turn `modelctl` into a browser-operated control plane for discovering, installing, configuring, placing, testing, serving, monitoring, and removing local AI models across unusual combinations of GPUs, system RAM, storage, and inference backends.

The web console becomes the primary product. The CLI remains available for bootstrap, recovery, diagnostics, and automation, but every normal model-management workflow must be possible through the browser.

This plan is intentionally **functionality-first**. Broad repository restructuring, naming cleanup, removal of retired code, profile-schema normalization, and general refactoring are deferred until the required behavior works end to end.

---

## 2. Target outcome

A user should be able to perform this complete workflow without opening a terminal:

1. Open the modelctl control page.
2. Inspect detected GPUs, RAM, storage, drivers, and backend binaries.
3. Search Hugging Face or import an existing local model.
4. Compare available quants and estimated hardware compatibility.
5. Download and verify the selected files.
6. Generate several viable launch plans.
7. Test individual plans or autotune the model.
8. Choose a runtime policy:

   * fastest generation;
   * largest context;
   * lowest load time;
   * fixed placement;
   * automatic placement with fallback.
9. Register the model with llama-swap.
10. Load, unload, restart, cancel, or evict models from the dashboard.
11. View startup progress, backend logs, resource use, and measured performance.
12. Review past tests and understand why a launch plan was selected.
13. Manage model coexistence and routing without manually editing llama-swap YAML.

The system should make conservative automatic decisions while always showing:

* The exact command that will run.
* The expected resource claim.
* The reason a plan was selected.
* The alternatives that were rejected.
* The observed result after launch.

---

## 3. Scope

### Included

* Web-based runtime lifecycle controls.
* Multiple launch plans per profile.
* Live hardware-aware plan selection.
* Runtime resource reservations.
* Backend process supervision and startup fallback.
* Persistent benchmark and launch observations.
* Web-based plan testing and autotuning.
* Editable hardware topology and reserve settings.
* Generated llama-swap coexistence rules.
* Live job, runtime, and log updates.
* Browser-driven model acquisition and configuration.
* Existing llama.cpp/SYCL and OVMS support.
* A design that can later add CUDA, vLLM, remote workers, or other backends.

### Deferred

* Large-scale package restructuring.
* Rewriting the control plane in another language.
* Replacing llama-swap.
* A chat interface or agent framework.
* Multi-user roles and permissions.
* Kubernetes or distributed scheduling.
* A generic third-party plugin framework.
* Full migration of every legacy router path.
* Cosmetic UI redesign beyond what the workflows require.

---

## 4. Design principles

### 4.1 Web-first, not web-only

All core behavior must be exposed as ordinary Python operations. FastAPI routes call those operations directly. CLI commands may call the same operations.

Avoid this pattern:

```python
args = type("Args", (), {...})()
cmd_remove(args)
```

Prefer:

```python
result = remove_profile(
    name=name,
    sync_runtime=True,
    sync_hermes=False,
)
```

The CLI wrapper can translate argparse values into this function call.

### 4.2 Preserve the current working path

Existing profiles must remain valid. Introduce managed runtime behavior as an optional mode:

```json
{
  "runtime": {
    "mode": "fixed"
  }
}
```

Profiles without a `runtime` section behave exactly as they do now.

A user can enable the new path per profile:

```json
{
  "runtime": {
    "mode": "managed",
    "policy": "balanced",
    "fallback": true
  }
}
```

### 4.3 Measured behavior outranks theoretical behavior

The VRAM and tier planners generate candidates. Actual launch and benchmark results determine which candidates should be preferred.

The program must distinguish:

* estimated to fit;
* tested successfully;
* tested on the current hardware and backend build;
* failed on the current build;
* stale because the binary, driver, or hardware changed.

### 4.4 Every automatic decision must be explainable

The runtime should produce a decision trace:

```text
Selected: b70-b580-10-2

Reasons:
- Fits current free VRAM after pending reservations.
- Successfully tested on this backend build.
- Median generation rate: 6.4 tok/s.
- 12% faster than the 8:3 split.
- B70-only plan cannot satisfy the requested 64K context.

Rejected:
- b70-only: estimated VRAM shortfall of 1.8 GiB.
- b580-only: model weights exceed device capacity.
- b70-cpu: valid, but 31% slower in previous tests.
```

### 4.5 Keep the candidate search bounded

Do not generate hundreds of tiny variations. Candidate generation should be deterministic and deliberately limited.

For a typical profile, generate no more than roughly 5–12 meaningful candidates.

---

## 5. Runtime architecture

```text
Browser
   │
   ▼
FastAPI + HTMX
   │
   ├── Read-only status operations
   ├── Persistent job manager
   └── Application operations
           │
           ├── Model/profile operations
           ├── Hardware inventory
           ├── Launch-plan compiler
           ├── Observation store
           ├── Reservation manager
           ├── llama-swap client
           └── Backend adapters
                    │
                    ▼
             modelctl _worker
                    │
                    ▼
       llama-server / OVMS / later backends
```

llama-swap remains responsible for:

* The public OpenAI-compatible endpoint.
* Selecting models based on request model IDs.
* Starting a configured command on demand.
* Assigning `${PORT}`.
* Request queuing.
* TTL-based unloading.
* Matrix-based eviction and coexistence.
* Forwarding requests to the selected upstream.
* Capturing and exposing upstream logs.

modelctl becomes responsible for:

* Determining which concrete backend command should run.
* Understanding the current hardware state.
* Reserving resources while a model starts.
* Trying fallback plans.
* Recording outcomes.
* Generating safe llama-swap configuration.

---

## 6. New core data structures

Add these as dataclasses or typed dictionaries. They can initially live in new focused modules without moving existing code.

### 6.1 HardwareSnapshot

```python
@dataclass(frozen=True)
class HardwareSnapshot:
    captured_at: float
    fingerprint: str
    gpus: tuple["GpuSnapshot", ...]
    ram_total_bytes: int
    ram_available_bytes: int
    storage: tuple["StorageSnapshot", ...]
    backend_fingerprints: dict[str, str]
```

```python
@dataclass(frozen=True)
class GpuSnapshot:
    device: str
    name: str
    pci_address: str | None
    total_bytes: int
    free_bytes: int
    reserve_bytes: int
    enabled: bool
    role: str
    pcie_width: int | None
    memory_bandwidth_gbs: float | None
```

The fingerprint should include, where available:

* GPU device IDs and total memory.
* PCI addresses and link widths.
* Kernel version.
* GPU driver/runtime version.
* llama-server path and binary hash.
* OVMS image or version.
* Relevant environment/runtime identifiers.

### 6.2 ResourceClaim

```python
@dataclass(frozen=True)
class ResourceClaim:
    vram_bytes: dict[str, int]
    ram_bytes: int
    storage_mode: str
    expected_context: int | None
```

The claim represents expected use, not observed use.

### 6.3 LaunchPlan

```python
@dataclass(frozen=True)
class LaunchPlan:
    id: str
    profile_name: str
    backend: str
    label: str
    argv: tuple[str, ...]
    env: dict[str, str]
    claim: ResourceClaim
    estimated: dict[str, float | int | None]
    source: str
    warnings: tuple[str, ...]
    decision_data: dict
```

`id` should be a stable hash of the normalized plan:

```text
profile revision
backend
device placement
tensor split
context behavior
cache configuration
offload flags
load mode
binary fingerprint
```

### 6.4 RuntimePolicy

```python
@dataclass(frozen=True)
class RuntimePolicy:
    objective: str
    pinned_plan_id: str | None
    allow_fallback: bool
    allow_untested: bool
    minimum_context: int | None
    maximum_cpu_bytes: int | None
    maximum_storage_tier: int
```

Supported objectives:

```text
balanced
fastest_generation
fastest_prompt
largest_context
fastest_load
lowest_ram
fixed
```

### 6.5 PlanRun

```python
@dataclass
class PlanRun:
    profile_name: str
    plan_id: str
    hardware_fingerprint: str
    backend_fingerprint: str
    success: bool
    failure_class: str | None
    load_seconds: float | None
    time_to_first_token: float | None
    prompt_tps: float | None
    generation_tps: float | None
    peak_vram_bytes: dict[str, int]
    peak_ram_bytes: int | None
    actual_context: int | None
    exit_code: int | None
    log_path: str
```

### 6.6 Reservation

```python
@dataclass
class Reservation:
    id: str
    profile_name: str
    plan_id: str
    owner_pid: int
    state: str
    claim: ResourceClaim
    created_at: float
    updated_at: float
```

States:

```text
pending
starting
active
releasing
stale
```

---

## 7. Persistent runtime database

Keep profile JSON files and the existing web jobs database unchanged for now.

Add:

```text
~/.local/share/modelctl/runtime.db
```

Suggested schema:

```sql
CREATE TABLE IF NOT EXISTS hardware_snapshots (
    id INTEGER PRIMARY KEY,
    captured_at REAL NOT NULL,
    fingerprint TEXT NOT NULL,
    snapshot_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_runs (
    id INTEGER PRIMARY KEY,
    profile_name TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    hardware_fingerprint TEXT NOT NULL,
    backend_fingerprint TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    success INTEGER NOT NULL,
    failure_class TEXT,
    load_seconds REAL,
    ttft_seconds REAL,
    prompt_tps REAL,
    generation_tps REAL,
    peak_vram_json TEXT NOT NULL DEFAULT '{}',
    peak_ram_bytes INTEGER,
    actual_context INTEGER,
    exit_code INTEGER,
    log_path TEXT NOT NULL DEFAULT '',
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_plan_runs_lookup
ON plan_runs (
    profile_name,
    plan_id,
    hardware_fingerprint,
    backend_fingerprint,
    started_at
);

CREATE TABLE IF NOT EXISTS reservations (
    id TEXT PRIMARY KEY,
    profile_name TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    owner_pid INTEGER NOT NULL,
    state TEXT NOT NULL,
    claim_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_events (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    profile_name TEXT,
    event_type TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}'
);
```

Use SQLite transactions with `BEGIN IMMEDIATE` when acquiring or changing reservations.

---

# Implementation milestones

## Milestone 1: Complete runtime control from the web

### Objective

Allow models and llama-swap to be operated from the browser before introducing dynamic placement.

### Backend work

Create a dedicated llama-swap client:

```python
class LlamaSwapClient:
    def health(self) -> RuntimeHealth: ...
    def registered_models(self) -> list[dict]: ...
    def running_models(self) -> list[dict]: ...
    def unload(self, model_id: str) -> None: ...
    def unload_all(self) -> None: ...
    def warm_load(self, model_id: str) -> WarmLoadResult: ...
    def logs(self, model_id: str | None = None) -> str: ...
```

Use:

* `/health` for service health.
* `/v1/models` for registered models.
* `/running` for live workers.
* `POST /api/models/unload/{encoded_model_id}` for one model.
* `POST /api/models/unload` for all models.
* A minimal OpenAI request to warm-load a model.
* `/logs/stream/{encoded_model_id}` for live model logs.

Warm load should issue a minimal request through the normal public path:

```json
{
  "model": "profile-name",
  "messages": [
    {
      "role": "user",
      "content": "Reply with OK."
    }
  ],
  "max_tokens": 1,
  "temperature": 0
}
```

A warm-load job is successful once:

* The model appears in `/running`.
* The request receives a valid response.
* The backend remains running after the request.

### Web routes

```text
GET  /runtime
GET  /api/runtime
GET  /api/runtime/models/{name}

POST /models/{name}/load
POST /models/{name}/unload
POST /models/{name}/restart
POST /runtime/unload-all
POST /runtime/restart-swap

GET  /runtime/logs/{name}
GET  /api/runtime/logs/{name}
```

### Dashboard changes

Each model row receives:

* Load.
* Unload.
* Restart.
* View logs.
* Current state.
* Current PID.
* Start time.
* Assigned upstream port, when available.
* Last runtime error.

States shown by the UI:

```text
unregistered
stopped
queued
loading
ready
unloading
failed
unknown
```

### Tests

* Mock llama-swap health, running, and model endpoints.
* Confirm model IDs containing `/` are safely URL-encoded.
* Confirm load submits a job rather than blocking the HTTP request.
* Confirm restart performs unload followed by warm load.
* Confirm unavailable llama-swap produces a visible failed job.
* Confirm unload-all requires an explicit confirmation form.

### Acceptance criteria

* A currently registered fixed-command profile can be loaded, unloaded, and restarted entirely through the web console.
* Per-model logs can be opened from the dashboard.
* Failures appear as structured job errors, not blank pages or generic HTTP 500 responses.

---

## Milestone 2: Replace the single global job worker with job lanes

### Objective

Prevent a download or benchmark from blocking urgent runtime actions.

### Current problem

All profile mutations, downloads, tests, and benchmarks use one FIFO worker. Marking a job cancelled changes its database state but does not necessarily terminate a running subprocess.

### New JobManager

```python
class JobManager:
    def __init__(self, store: JobStore):
        self.mutation = JobLane("mutation", workers=1)
        self.runtime = JobLane("runtime", workers=2)
        self.download = JobLane("download", workers=2)
        self.benchmark = JobLane("benchmark", workers=1)
```

Recommended lane assignment:

| Lane      | Operations                                                           |
| --------- | -------------------------------------------------------------------- |
| mutation  | Save profile, apply configuration, sync llama-swap, settings changes |
| runtime   | Load, unload, restart, eviction, reservation cleanup                 |
| download  | Hugging Face downloads, verification, repair                         |
| benchmark | Smoke test, plan test, tuning, evaluation                            |

### JobContext

Every job function should receive:

```python
class JobContext:
    job_id: str

    def log(self, line: str) -> None: ...
    def set_progress(self, value: float, detail: str = "") -> None: ...
    def is_cancelled(self) -> bool: ...
    def raise_if_cancelled(self) -> None: ...
    def register_process(self, process: subprocess.Popen) -> None: ...
    def register_cancel_callback(self, fn: Callable) -> None: ...
```

### Real cancellation

For subprocess-backed jobs:

* Launch in a new process group.
* Record the PID.
* On cancellation, send `SIGTERM` to the process group.
* Wait for a bounded graceful period.
* Send `SIGKILL` if still alive.
* Run registered cleanup callbacks.
* Release any pending reservation.

Add states:

```text
queued
running
cancelling
cancelled
done
failed
interrupted
```

### Job schema additions

```sql
ALTER TABLE jobs ADD COLUMN lane TEXT NOT NULL DEFAULT 'mutation';
ALTER TABLE jobs ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN pid INTEGER;
ALTER TABLE jobs ADD COLUMN cancellable INTEGER NOT NULL DEFAULT 1;
ALTER TABLE jobs ADD COLUMN parent_id TEXT;
```

Increment `revision` whenever progress, status, detail, result, or outcome changes.

### SSE endpoints

```text
GET /events/jobs/{job_id}
GET /events/jobs
```

The event stream sends a new event only when `revision` changes.

### Compatibility

Retain a thin `JobRunner` compatibility wrapper so current tests and call sites can migrate incrementally.

### Tests

* A download job and unload job can run without blocking one another.
* Two mutation jobs remain serialized.
* Cancelling a subprocess actually terminates it.
* Restarting the web service marks abandoned jobs interrupted.
* An interrupted runtime job causes reservation cleanup.
* SSE exits after a terminal job state.

### Acceptance criteria

* Runtime unload remains responsive while a download or benchmark is running.
* Cancel stops actual work rather than only changing the displayed state.
* Existing pull, edit, tier, smoke, and benchmark routes continue working.

---

## Milestone 3: Introduce launch-plan generation

### Objective

Change placement from one configuration recommendation into a bounded set of comparable launch plans.

### New module

Add:

```text
modelctl_plans.py
```

Do not move the existing planner. Call it as one candidate source.

Primary API:

```python
def compile_launch_plans(
    profile: dict,
    hardware: HardwareSnapshot,
    *,
    include_experimental: bool = False,
) -> list[LaunchPlan]:
    ...
```

### Candidate sources

#### A. Current profile

Always produce a baseline plan representing the exact existing configuration.

```text
source: current-profile
```

This ensures managed mode can begin without changing behavior.

#### B. Existing tier planner

Convert the output of `plan_tiers()` into a `LaunchPlan`.

```text
source: tier-planner
```

#### C. Single-GPU plans

Generate one plan for each eligible GPU when the model can fit:

```text
B70 only
B580 only
future GPU only
```

Account for:

* configured reserve;
* estimated KV cache;
* expected overhead;
* fixed or fit-managed context.

#### D. Multi-GPU plans

Generate:

1. Capacity-ratio split.
2. One split biased toward the primary GPU.
3. One split biased away from the primary GPU.

For a 32 GB and 12 GB combination, candidate ratios might be:

```text
8,3
10,2
11,1
```

Do not generate every integer ratio.

#### E. CPU-spill plans

For MoE models:

* Keep attention, KV cache, and shared tensors on the primary GPU where possible.
* Generate one aggressive GPU-expert placement.
* Generate one more conservative plan leaving additional headroom.
* Spill remaining routed experts to CPU.

For dense models:

* Generate at most two `-ngl` boundary plans:

  * highest estimated safe GPU layer count;
  * a lower fallback with additional headroom.

#### F. Storage plans

Only generate storage/load-mode alternatives during explicit tuning unless a profile already uses them.

Possible variants:

```text
mmap
mmap + mlock
no-mmap
DirectIO, when supported and explicitly enabled
```

### Plan normalization

Create one canonical representation before hashing:

```python
{
    "backend": "llama-cpp",
    "binary": "...",
    "device": ["SYCL0", "SYCL1"],
    "split_mode": "layer",
    "tensor_split": [10, 2],
    "ctx_mode": "fixed",
    "ctx": 65536,
    "cache_type_k": "q8_0",
    "cache_type_v": "q4_0",
    "fit": "off",
    "offload": {...},
    "load_mode": "mmap"
}
```

Argument order or cosmetic formatting must not change the plan ID.

### Plan status

Derived from observation history:

```text
untested
validated
failed
stale
disabled
```

A validated plan becomes stale when its hardware or backend fingerprint no longer matches.

### API and web routes

```text
GET /profiles/{name}/plans
GET /api/profiles/{name}/plans
GET /api/profiles/{name}/plans/{plan_id}
```

The page should display:

* Plan label.
* Estimated resource use.
* Expected context.
* Placement.
* Warnings.
* Validation status.
* Last measured speed.
* Exact command.
* Why the candidate exists.

### Tests

Cover:

* One B70.
* One B580.
* B70 plus B580.
* Integrated GPU excluded by policy.
* Dense model that fits one GPU.
* Dense model requiring partial offload.
* MoE model requiring CPU experts.
* Model requiring mmap.
* Unknown GGUF tensor types.
* Existing manual profile preserved as the baseline plan.
* Stable plan IDs.

### Acceptance criteria

* Every runnable profile produces at least one baseline plan.
* Candidate generation is deterministic.
* Candidate generation has no side effects.
* The web page can compare candidates without modifying the profile or llama-swap config.

---

## Milestone 4: Add resource reservations and the managed worker

### Objective

Choose a launch plan when a model starts, using current hardware availability, and safely fall back when startup fails.

### Hidden worker command

Add:

```bash
modelctl _worker PROFILE_NAME --port PORT
```

llama-swap entries for managed profiles become:

```yaml
models:
  profile-name:
    cmd: modelctl _worker profile-name --port ${PORT}
    checkEndpoint: /health
    ttl: 600
```

Fixed profiles continue rendering their direct backend command.

### Worker lifecycle

```python
def worker_main(profile_name: str, port: int) -> int:
    profile = load_profile(profile_name)
    policy = load_runtime_policy(profile)

    snapshot = capture_hardware_snapshot()
    plans = compile_launch_plans(profile, snapshot)
    ranked = rank_feasible_plans(profile, policy, plans, snapshot)

    for plan in ranked:
        reservation = acquire_pending_reservation(profile, plan, snapshot)

        try:
            child = launch_backend(plan, port)
            forward_worker_signals_to(child)

            if wait_until_ready(child, port, plan):
                mark_reservation_active(reservation)
                record_successful_start(...)
                return supervise_until_exit(child, reservation)

            record_start_failure(...)
            stop_process_group(child)
        finally:
            release_or_update_reservation(reservation)

    return NO_PLAN_SUCCEEDED
```

### Reservation accounting

Use live free GPU memory as the source of truth for running workloads.

Subtract only **pending or starting** reservations when evaluating another launch. Do not subtract active reservations from live free VRAM, because active allocations are already reflected by the GPU driver.

```text
effective free =
    currently reported free
    - pending reservation claims
    - configured safety reserve
```

Active reservations remain in the database for:

* status display;
* ownership tracking;
* matrix generation;
* diagnostics.

### Atomic reservation acquisition

Inside one SQLite `BEGIN IMMEDIATE` transaction:

1. Delete reservations whose owner PID is dead.
2. Read all pending claims.
3. Capture or accept a fresh hardware snapshot.
4. Recheck candidate feasibility.
5. Insert the pending reservation.
6. Commit.

Only then launch the child process.

### Signal handling

The worker must forward:

* `SIGTERM`
* `SIGINT`
* `SIGHUP`, when useful

to the child process group.

On worker shutdown:

1. Ask the child to terminate.
2. Wait for the configured unload timeout.
3. Kill if necessary.
4. Release the reservation.
5. Record a runtime event.

### Backend adapters

Use a small explicit interface:

```python
class BackendAdapter(Protocol):
    name: str

    def build_command(
        self,
        profile: dict,
        plan: LaunchPlan,
        port: int,
    ) -> list[str]: ...

    def effective_environment(
        self,
        profile: dict,
        plan: LaunchPlan,
    ) -> dict[str, str]: ...

    def readiness_url(
        self,
        profile: dict,
        port: int,
    ) -> str: ...

    def classify_failure(
        self,
        exit_code: int | None,
        log_tail: str,
    ) -> str: ...
```

Initial adapters:

```text
LlamaCppAdapter
OvmsAdapter
```

Do not build a generic plugin loader yet. Use a dictionary:

```python
BACKENDS = {
    "llama-cpp": LlamaCppAdapter(),
    "ovms": OvmsAdapter(),
}
```

### Failure classifications

At minimum:

```text
out_of_vram
out_of_ram
unsupported_architecture
invalid_argument
missing_binary
missing_library
device_unavailable
health_timeout
backend_crash
cancelled
unknown
```

A plan that fails with `invalid_argument` or `unsupported_architecture` should be strongly suppressed for the matching backend fingerprint.

An OOM failure should suppress the exact plan but permit lower-resource fallbacks.

### Feature rollout

Add a web toggle:

```text
Runtime mode:
  Fixed command
  Managed placement
```

Managed placement must be opt-in until the worker is proven reliable.

### Tests

* Worker starts a fake backend and remains alive while it runs.
* Worker forwards SIGTERM.
* Worker releases the reservation after normal exit.
* Worker releases the reservation after failed startup.
* Two concurrent workers cannot reserve the same free VRAM.
* A failed first plan falls back to the second plan.
* A cancelled start does not continue trying fallbacks.
* Fixed profiles still render the old command.
* Managed profiles render `_worker`.

### Acceptance criteria

* A managed model requested through llama-swap selects a feasible plan at start time.
* Concurrent cold starts cannot both claim the same unallocated memory.
* Failed startup automatically tries a valid fallback when permitted.
* The dashboard shows the selected plan and decision trace.

---

## Milestone 5: Runtime policy and plan controls in the web UI

### Objective

Make dynamic placement understandable and controllable through the browser.

### Model detail page

Add tabs:

```text
Overview
Runtime
Placement
Tuning
Configuration
Artifacts
History
Logs
```

### Placement tab

Display a comparison table:

| Plan            | Status    | Placement   | Context |  VRAM |   RAM | Last speed | Actions |
| --------------- | --------- | ----------- | ------: | ----: | ----: | ---------: | ------- |
| B70 only        | Validated | SYCL0       |     32K | 29 GB |  2 GB |    6.8 t/s | Select  |
| B70+B580 10:2   | Validated | Split       |     64K | 38 GB |  3 GB |    6.3 t/s | Select  |
| B70+CPU experts | Untested  | Hybrid      |    128K | 30 GB | 28 GB |          — | Test    |
| mmap fallback   | Untested  | GPU+RAM+SSD |    256K | 30 GB | 50 GB |          — | Test    |

Each plan expands to show:

* Exact command.
* Environment.
* Resource claim.
* Estimated calculations.
* Warnings.
* Last test result.
* Failure history.
* Backend and hardware fingerprints.
* Decision explanation.

### Runtime-policy form

```text
Mode:
  Fixed command
  Managed placement

Objective:
  Balanced
  Fastest generation
  Fastest prompt processing
  Largest context
  Fastest load
  Lowest RAM use
  Fixed plan

Fallback:
  Enabled / Disabled

Untested plans:
  Never
  Only when no validated plan fits
  Allowed

Minimum context:
  [value]

Maximum CPU/RAM offload:
  [value]

Maximum tier:
  GPU only
  GPU + RAM
  GPU + RAM + storage
```

### Routes

```text
POST /profiles/{name}/runtime-policy
POST /api/profiles/{name}/runtime-policy

POST /profiles/{name}/plans/{plan_id}/select
POST /profiles/{name}/plans/{plan_id}/disable
POST /profiles/{name}/plans/{plan_id}/enable
```

### Profile storage

Add only one optional section:

```json
{
  "runtime": {
    "mode": "managed",
    "objective": "balanced",
    "pinned_plan_id": null,
    "allow_fallback": true,
    "allow_untested": false,
    "minimum_context": 32768,
    "maximum_cpu_bytes": null,
    "maximum_storage_tier": 3,
    "disabled_plan_ids": []
  }
}
```

### Configuration preview

Before saving runtime policy, show:

* The new generated llama-swap entry.
* Whether the service configuration will change.
* Whether the model must be unloaded.
* Which current observations will become inapplicable.
* The selected fallback order.

### Acceptance criteria

* A user can enable managed runtime for a profile through the web page.
* A user can pin or disable plans.
* A user can preview the exact generated worker command.
* The currently effective policy is visible from both the model page and dashboard.

---

## Milestone 6: Plan testing and autotuning

### Objective

Measure real performance and use those results for runtime decisions.

### Plan-test operation

```python
def test_launch_plan(
    profile_name: str,
    plan_id: str,
    job: JobContext,
    *,
    prompt: str | None = None,
    max_tokens: int = 256,
    runs: int = 3,
) -> PlanRun:
    ...
```

The operation should:

1. Ensure the model is not already running under an incompatible test.
2. Capture the initial hardware snapshot.
3. Reserve the candidate resources.
4. Start the backend directly on a temporary port.
5. Poll readiness.
6. Record load time.
7. Sample GPU and RAM usage during startup and inference.
8. Run a deterministic prompt workload.
9. Record prompt and generation throughput.
10. Stop the backend.
11. Release the reservation.
12. Persist the result.
13. Restore normal runtime state if the test temporarily displaced a model.

### Resource sampling

Sample during launch and inference:

```text
GPU free memory per device
process RSS
system available RAM
optional temperature
optional power
```

Start with one-second sampling. Avoid making high-frequency monitoring a prerequisite.

### Autotuning operation

```python
def autotune_profile(
    profile_name: str,
    objective: str,
    candidate_ids: list[str] | None,
    job: JobContext,
) -> TuneResult:
    ...
```

Default behavior:

* Test only safe candidates.
* Skip candidates already validated under the current fingerprints unless retest is requested.
* Stop early when remaining plans cannot beat the current winner for the selected objective.
* Never automatically select a result unless the user enabled “select winner after tuning.”

### Ranking

First filter by hard constraints:

* Fits current resources.
* Satisfies minimum context.
* Respects maximum CPU offload.
* Respects maximum storage tier.
* Not disabled.
* Not known incompatible.

Then score.

Example balanced score:

```python
score = (
    normalized_generation_tps * 0.50
    + normalized_prompt_tps * 0.15
    + normalized_context * 0.15
    - normalized_load_time * 0.10
    - cpu_spill_penalty * 0.05
    - storage_penalty * 0.05
)
```

Do not hard-code this as an opaque truth. Store each score component and display it.

### Web interface

Actions:

```text
Test
Retest
Tune safe plans
Tune selected plans
Cancel
Select winner
Mark result invalid
```

Live tuning page:

```text
Plan 1 of 6
Starting B70+B580 10:2
Loading weights...
B70: 27.8 / 32 GB
B580: 8.4 / 12 GB
RAM: 21 / 64 GB
Prompt: 41.2 tok/s
Generation: 6.3 tok/s
```

### Routes

```text
POST /profiles/{name}/plans/{plan_id}/test
POST /profiles/{name}/tune
POST /profiles/{name}/tune/cancel

GET  /profiles/{name}/history
GET  /api/profiles/{name}/history
GET  /api/plan-runs/{run_id}
```

### Acceptance criteria

* A plan can be tested without permanently modifying the profile.
* Test cancellation terminates the backend and releases resources.
* Results remain available after restarting the web service.
* Runtime plan ranking uses matching measured results.
* Results become stale when the hardware or backend fingerprint changes.

---

## Milestone 7: Hardware control page

### Objective

Represent machine-specific facts that automatic probing cannot reliably infer.

### Settings file

Add:

```text
~/.local/share/modelctl/hardware.json
```

Example:

```json
{
  "version": 1,
  "devices": {
    "SYCL0": {
      "enabled": true,
      "role": "primary",
      "reserve_bytes": 2147483648,
      "pcie_width_override": 16,
      "memory_bandwidth_gbs_override": 608.0
    },
    "SYCL1": {
      "enabled": true,
      "role": "secondary",
      "reserve_bytes": 1073741824,
      "pcie_width_override": 4,
      "memory_bandwidth_gbs_override": 456.0
    }
  },
  "ram": {
    "reserve_bytes": 8589934592
  },
  "storage": {
    "models": {
      "path": "/models",
      "kind": "nvme-raid0",
      "allow_mmap": true,
      "allow_direct_io": false
    }
  }
}
```

### Probe sources

Use:

* Existing `get_gpu_inventory()`.
* `xpu-smi`.
* `llama-server --list-devices`.
* `/sys/bus/pci/devices`.
* `/proc/meminfo`.
* `lsblk` or sysfs storage metadata.
* Backend binary versions and hashes.

Explicit overrides always win over probed values.

### Web page

```text
/hardware
```

Show:

* All detected accelerators.
* Excluded devices.
* Total and current free memory.
* Primary/secondary role.
* Reserve.
* PCI topology.
* Measured or configured bandwidth.
* Driver/backend fingerprint.
* System RAM and reserve.
* Model storage.
* Current pending reservations.
* Active model claims.

Actions:

```text
Refresh inventory
Save overrides
Set primary
Enable/disable device
Clear stale reservations
Test device mapping
Capture new fingerprint
```

### Device-mapping test

Launch a tiny known model or a backend device-list command to confirm:

* `SYCL0` maps to the expected physical device.
* `SYCL1` maps to the expected physical device.
* Excluded integrated devices are not accidentally used.

### Acceptance criteria

* A user can model the B70/B580 asymmetry from the browser.
* The planner uses configured reserves and overrides.
* Hardware changes visibly mark relevant plan runs stale.
* Clearing stale reservations is possible without manually editing SQLite.

---

## Milestone 8: Generate llama-swap coexistence and eviction policy

### Objective

Make llama-swap’s runtime solver reflect modelctl’s resource understanding.

### Resource claims

For each enabled model, determine the claim used for routing:

* Pinned plan claim, when fixed.
* Selected validated plan claim.
* Conservative maximum among eligible automatic plans.
* Explicit manual override, when configured.

### Compatibility calculation

A combination is valid when:

```python
for resource in resources:
    sum(model.claim[resource] for model in combination) <= budget[resource]
```

Resources include:

```text
SYCL0 VRAM
SYCL1 VRAM
system RAM
exclusive backend/container resources
```

Initially generate only:

* Single-model sets.
* Valid pairs.
* Valid triples involving small helper models.

Avoid exhaustive powerset generation.

A useful reduction algorithm:

1. Classify large models as exclusive candidates.
2. Identify small resident/helper models.
3. Generate large-plus-helper combinations.
4. Generate combinations among helper models.
5. Remove combinations that are strict subsets of another valid combination because llama-swap matrix sets already permit subsets.

### Eviction cost

Derive from measured cold-start cost:

```python
evict_cost = clamp(
    round(median_load_seconds / 5),
    minimum=1,
    maximum=100,
)
```

Allow user override.

### Configuration preview

Add:

```text
/runtime/routing
```

Display:

* Existing unmanaged matrix.
* Proposed managed entries.
* Models not included and why.
* Generated eviction costs.
* YAML diff.
* Validation warnings.

Actions:

```text
Enable managed routing
Preview
Apply
Rollback
Disable management
```

### Ownership boundaries

Do not overwrite the entire llama-swap file.

Continue preserving hand-authored sections. Add a clearly marked managed region or merge only known modelctl-owned keys:

```yaml
# BEGIN MODELCTL MANAGED MATRIX
...
# END MODELCTL MANAGED MATRIX
```

Before writing:

1. Render into memory.
2. Parse as YAML.
3. Write a timestamped backup.
4. Write to a temporary file.
5. `fsync`.
6. Atomically replace.
7. reload/restart llama-swap;
8. check `/health`;
9. restore the backup if health fails.

### Acceptance criteria

* Compatible models can remain loaded together.
* Models that do not fit together are never declared compatible.
* Existing hand-written config outside the managed region survives.
* The user can preview and roll back every routing change.
* Eviction decisions account for measured cold-start cost.

---

## Milestone 9: Complete the browser model-acquisition workflow

### Objective

Connect model search, quant choice, placement, tuning, and serving into one continuous workflow.

### Add-model wizard

Steps:

```text
1. Source
2. Repository/files
3. Quant or implementation
4. Download
5. Analyze
6. Placement candidates
7. Test/tune
8. Runtime policy
9. Register
10. Finish
```

### Supported sources

Initial:

```text
Hugging Face GGUF repository
Existing local GGUF
Existing model directory
OVMS conversion/import
```

Later:

```text
Remote worker
vLLM/Hugging Face implementation
```

### Quant comparison

For every quant group show:

* Download size.
* Multipart status.
* Estimated weight memory.
* Estimated KV cache at several context points.
* Best predicted placement.
* Highest likely context on the primary GPU.
* Whether CPU or storage spill is expected.
* MTP availability.
* Multimodal projector availability.
* Recommendation reason.

### Download job

Keep downloads in the download lane.

Improve progress tracking:

* File count.
* Current file.
* Downloaded bytes.
* Total bytes.
* Transfer rate.
* Verification state.
* Cancellation.
* Resume status.

### Post-download pipeline

After verification:

1. Read actual GGUF metadata and tensor table.
2. Create the profile.
3. Generate candidate plans.
4. Show plans to the user.
5. Allow immediate tuning.
6. Save the selected runtime policy.
7. Sync the model into llama-swap.
8. Offer a load test.

Do not silently select a risky CPU or storage-spill plan merely because it technically fits.

### Acceptance criteria

A new GGUF model can move from Hugging Face search to a working llama-swap endpoint entirely in the browser.

---

## 10. Web navigation and page map

Top navigation:

```text
Dashboard
Models
Add Model
Runtime
Hardware
Jobs
Settings
```

### Dashboard

Focus: immediate system state.

Show:

* llama-swap health.
* GPUs and RAM.
* Running/loading/failed models.
* Pending runtime work.
* Recent failures.
* Load/unload controls.

### Models

Focus: model inventory.

Filters:

```text
Running
Enabled
Managed
Needs tuning
Failed
Stale measurements
llama.cpp
OVMS
```

### Runtime

Focus: serving stack.

Show:

* Running workers.
* PIDs and ports.
* Selected plans.
* Pending and active reservations.
* Queue.
* llama-swap logs.
* Generated routing matrix.

### Hardware

Focus: topology and budgets.

### Jobs

Focus: all background work with lane, progress, logs, and cancellation.

### Settings

Focus:

* State paths.
* Model paths.
* Backend binaries.
* oneAPI environment.
* llama-swap URL/service.
* Hermes integration.
* Default runtime policy.
* Web token.
* Managed-routing toggle.

---

## 11. Application operation API

Add ordinary callable functions rather than placing more logic directly in FastAPI routes.

```python
# Runtime
get_runtime_status()
load_model(name, job)
unload_model(name, job)
restart_model(name, job)
unload_all_models(job)

# Plans
get_launch_plans(name)
test_plan(name, plan_id, job)
autotune_model(name, objective, plan_ids, job)
update_runtime_policy(name, policy)

# Hardware
capture_hardware_snapshot()
get_hardware_settings()
update_hardware_settings(changes)
clear_stale_reservations()

# Routing
preview_runtime_config()
apply_runtime_config(job)
preview_managed_matrix()
apply_managed_matrix(job)

# Models
search_model_sources(query, filters)
inspect_model_source(source)
pull_model_from_web(selection, job)
import_local_model(path, options, job)
remove_model(name, options, job)
```

FastAPI routes should validate input, submit operations, and render results. They should not implement placement or process-management logic themselves.

---

## 12. Error handling

All application operations should raise structured errors:

```python
class ModelctlError(Exception):
    code: str
    message: str
    detail: dict
    recoverable: bool
    suggested_action: str | None
```

Examples:

```text
LLAMA_SWAP_UNAVAILABLE
PROFILE_NOT_FOUND
NO_FEASIBLE_PLAN
PLAN_STALE
PLAN_DISABLED
RESERVATION_CONFLICT
BACKEND_START_FAILED
BACKEND_HEALTH_TIMEOUT
MODEL_DOWNLOAD_FAILED
MODEL_VERIFICATION_FAILED
CONFIG_VALIDATION_FAILED
CONFIG_RELOAD_FAILED
```

The web UI should render:

* Human-readable summary.
* Technical detail.
* Suggested recovery action.
* Relevant log link.
* Retry button when appropriate.

---

## 13. Atomic writes and rollback

Even before general cleanup, all newly touched critical files should use one helper:

```python
def atomic_write_text(path: Path, text: str, mode: int | None = None) -> None:
    ...
```

Required behavior:

1. Create a temporary file in the destination directory.
2. Write content.
3. Flush.
4. `fsync`.
5. Apply mode if supplied.
6. `os.replace`.
7. `fsync` the directory when practical.

Use it for:

* Runtime policy changes.
* Hardware settings.
* Generated llama-swap config.
* New or updated profile JSON.
* Managed matrix changes.

Before multi-file changes, render and validate every output in memory. Do not restart services until all writes succeed.

---

## 14. Observability requirements

Every managed model start should produce runtime events:

```text
plan_generation_started
plan_selected
reservation_acquired
backend_started
backend_ready
plan_fallback
backend_failed
reservation_released
model_unloaded
```

Dashboard status must expose:

```json
{
  "model": "qwen-coder",
  "state": "ready",
  "plan_id": "b70-b580-10-2-a91c",
  "placement": "B70 + B580",
  "pid": 23172,
  "port": 5821,
  "started_at": 1785251172,
  "claim": {
    "vram": {
      "SYCL0": 28991029248,
      "SYCL1": 8589934592
    },
    "ram": 4294967296
  },
  "decision_summary": "...",
  "last_error": null
}
```

---

## 15. Testing strategy

### Unit tests

* Plan normalization and stable IDs.
* Candidate generation.
* Policy filtering and scoring.
* Observation matching and staleness.
* Hardware settings precedence.
* Resource-claim arithmetic.
* Matrix compatibility.
* Failure classification.
* URL encoding and llama-swap response parsing.

### Concurrency tests

* Simultaneous reservation attempts.
* Pending versus active reservation accounting.
* Dead PID cleanup.
* Concurrent download and unload.
* Mutation serialization.
* Cancellation during backend startup.
* Cancellation during benchmark inference.

### Worker tests

Use a small fake Python backend that can:

* Delay readiness.
* Return `/health`.
* Return a minimal OpenAI response.
* Allocate configurable memory.
* Exit with a selected code.
* Print simulated OOM or unsupported-argument errors.
* Ignore SIGTERM to test forced termination.

### Web tests

* Every route requires authentication.
* Lifecycle actions return job IDs.
* SSE delivers revisions.
* Plan comparison renders.
* Policy updates persist.
* Hardware overrides persist.
* Config previews do not mutate files.
* Apply and rollback paths work.
* Destructive operations require confirmation.

### Real-hardware acceptance matrix

Run manually against:

1. Small model on B580.
2. Medium model on B70.
3. Model split across B70+B580.
4. MoE model with CPU experts.
5. Oversized model using RAM/mmap.
6. Concurrent small helper plus primary model.
7. Two simultaneous cold-start requests.
8. Failed high-memory plan falling back to a safer plan.
9. Backend binary update making observations stale.
10. Web-service restart during an active job.

---

## 16. Rollout strategy

### Stage A: Web lifecycle controls

Use existing fixed llama-swap commands. No serving behavior changes.

### Stage B: Managed worker with baseline-only plan

Managed mode invokes `_worker`, but the worker has only the existing profile command as a candidate.

This proves process supervision and reservations without changing placement.

### Stage C: Multiple plans, manually selected

Expose candidate plans and testing. The user pins the selected plan.

### Stage D: Automatic fallback

Allow the worker to use validated alternatives when the pinned/default plan fails or does not fit.

### Stage E: Objective-based automatic selection

Use measurements and live resources to choose among validated plans.

### Stage F: Managed routing matrix

Generate coexistence and eviction policy after resource claims have proven trustworthy.

At every stage, fixed-command mode remains available as an immediate fallback.

---

## 17. Recommended implementation order

1. Add the llama-swap client and web lifecycle controls.
2. Split jobs into lanes and implement real cancellation.
3. Add SSE job and log updates.
4. Define `HardwareSnapshot`, `ResourceClaim`, `LaunchPlan`, and `RuntimePolicy`.
5. Generate baseline and existing-tier plans.
6. Add plan comparison pages.
7. Implement runtime database and observations.
8. Add `_worker` with baseline-only managed mode.
9. Add reservation acquisition and stale cleanup.
10. Add worker startup fallback.
11. Add single-GPU and multi-GPU candidate generation.
12. Add CPU and storage-tier candidates.
13. Add plan testing.
14. Add autotuning and measured ranking.
15. Add hardware settings and topology page.
16. Add managed matrix generation.
17. Complete the add-model wizard.
18. Only then begin broad repo cleanup and consolidation.

---

## 18. Definition of done

The original goal is achieved when all of the following are true:

* The normal user never needs the CLI after initial installation.
* Models can be searched, downloaded, imported, configured, tested, and removed through the web console.
* The UI shows every meaningful placement available for a model.
* The program can select placement using current GPU and RAM availability.
* Simultaneous cold starts cannot overcommit the same free resources.
* Failed launches can fall back to safer validated plans.
* Actual performance is recorded and influences later decisions.
* Hardware-specific results are invalidated when the environment materially changes.
* Exact backend commands and selection reasons remain visible.
* Models can be loaded, unloaded, restarted, cancelled, and monitored through the web console.
* llama-swap coexistence and eviction rules can be generated, previewed, applied, and rolled back through the browser.
* Existing fixed profiles remain usable throughout the transition.
* A failure cannot leave a permanent stale reservation or a partially written runtime configuration.
* The entire workflow works on the B70/B580/RAM/NVMe system without relying on assumptions that both GPUs are symmetric.

At that point, repository cleanup becomes worthwhile because the correct subsystem boundaries will have emerged from working behavior rather than being guessed in advance.
