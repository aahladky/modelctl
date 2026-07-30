# modelctl Release A — July 29, 2026

## Starting Point

- **modelctl commit**: `484021d` (Add review packet: MoE cache + modelctl integration)
- **llama.cpp commit**: `f42f2fe4e` (SYCL MoE expert-cache fork, submodule pin)
- **Python**: 3.14 (system)
- **FastAPI + HTMX**: web console
- **Platform**: Linux, Intel multi-GPU SYCL workstation

## Test Baseline

```
Ran 555 tests in ~88s — ALL PASS
```

Test modules:
- `test_modelctl` — CLI, profiles, placement, artifacts, download, GPU inventory, import
- `test_modelctl_capabilities` — schema v2, normalization, preflight (41 tests)
- `test_modelctl_vram` — VRAM estimation, GGUF layout parsing
- `test_modelctl_tui` — Textual wizard
- `test_modelctl_web` — FastAPI web console, cache variants, tiers, runtime
- `test_modelctl_tiers` — tier planner
- `test_modelctl_profiles` — profile schema v2, validation, migration (19 tests)
- `test_modelctl_errors` — structured validation messages (8 tests)
- `test_modelctl_launch` — canonical launch types (9 tests)
- `test_modelctl_services` — service layer (8 tests)
- `test_modelctl_transactions` — atomic mutations (5 tests)
- `test_modelctl_storage` — storage topology probing (7 tests)
- `test_modelctl_benchmark` — benchmark modes (9 tests)
- `test_modelctl_wizard` — wizard state persistence (12 tests)
- `test_modelctl_cache_service` — cache metrics, calibration (6 tests)
- `test_release_a` — Release A acceptance (20 tests)

## Release A Definition of Done — Status

| Requirement | Status | Evidence |
|---|---|---|
| No terminal required for normal flow | Done | `/add` wizard, `/import`, `/settings` pages |
| Unsupported flags never reach backend | Done | `normalize_capabilities()` fail-closed, `preflight_moe_cache()` |
| Shown command = launched command | Done | `LaunchCommand` single authoritative builder |
| RAM/VRAM/storage visible | Done | `ResourceClaim` expanded, runtime page shows cache metrics |
| Plan has explainable measured basis | Done | `decision_data`, `command_fingerprint`, provenance in `plan_runs` |
| Failed mutations recover cleanly | Done | `Transaction` with rollback |

## Architecture

### Control Plane (Python)
| Module | Role |
|---|---|
| `modelctl.py` | CLI, profiles, artifacts, sync |
| `modelctl_capabilities.py` | Schema v2 capability probing and normalization |
| `modelctl_launch.py` | `ResolvedBackend`, `LaunchCommand` canonical types |
| `modelctl_errors.py` | `ValidationMessage` structured errors |
| `modelctl_profiles.py` | Profile schema v2, validation, migration |
| `modelctl_plans.py` | Plan compilation, ranking, resource claims |
| `modelctl_transactions.py` | Atomic multi-file mutations |
| `modelctl_storage.py` | Storage topology probing |
| `modelctl_benchmark.py` | Benchmark mode definitions |
| `modelctl_runtime.py` | Runtime DB with provenance columns |
| `modelctl_worker.py` | Managed worker with provenance recording |
| `modelctl_services/` | Service layer (profile, plan, runtime, hardware, cache) |
| `modelctl_web/` | FastAPI + HTMX console with wizard |

### Runtime (llama.cpp fork)
| Module | Role |
|---|---|
| `common/arg.cpp` | Schema 2 `--modelctl-capabilities` probe |

## Web Routes

| Route | Purpose |
|---|---|
| `/` | Dashboard with VRAM gauges and profile table |
| `/add` | Add-model wizard (source → inspect → download → analyze → plans → test → register → done) |
| `/import` | Local GGUF import |
| `/pull` | Hugging Face model search and pull |
| `/runtime` | Runtime monitoring with cache metrics |
| `/hardware` | Hardware snapshot and device settings |
| `/settings` | Profile defaults and paths |
| `/tiers` | Tier planning |
| `/plans/{name}` | Plan comparison and selection |
| `/jobs` | Background job tracking |

## Key Design Decisions

1. **`moe_weight_transfer_cache`** — canonical feature name replacing `moe_expert_cache` + `moe_cache_sycl`
2. **`moe_hybrid_cpu_miss` always false** — forced by normalization until Phase 7 implements it
3. **Schema 0/1 auto-normalized** — old binaries and early forks work transparently
4. **Provenance additive** — new columns via `ALTER TABLE`, old rows have empty defaults
5. **Services don't print** — `modelctl_services/` returns typed results, CLI/web handle presentation
6. **Transactions rollback** — staged changes are atomic, failures restore previous state
7. **Benchmark modes explicit** — enum prevents conflating cold/warm results

## Remaining Work (Post Release A)

| Phase | Description | Type |
|---|---|---|
| 7 | True GPU-hit/CPU-miss hybrid execution | Runtime C++ |
| 8 | Managed RAM tier (if measurements justify) | Runtime + control plane |
| 9 | Prefetch and prediction (if measurements justify) | Runtime + control plane |
| 3.3 | Process I/O sampling (`/proc/<pid>/io`) | Runtime integration |
| 3.5 | Storage calibration jobs | Service + web |
