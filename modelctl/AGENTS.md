# Agent Guide — modelctl

Web console (FastAPI + HTMX) backed by a Python CLI, for managing local
GGUF models on multi-GPU Intel SYCL workstations. **The browser is the
product entry point**: `modelctl web install` starts it and prints
URL + token; `/setup` reports first-run readiness. The CLI is for
bootstrap, automation, diagnostics, and recovery — fully supported, but
new user-facing capability belongs in the console first.

This is a **local-first** project: one operator, one machine, real
hardware. Correctness on the actual GPUs/RAM/storage, accurate resource
accounting, reproducible runtime builds, and safe recovery outrank
public packaging, portability, stable external APIs, or multi-user
concerns.

## How to work

Inspect the current code, implement the requested change through the
real production path, add or update tests, run them, and report the
result in plain English (file, function, what was wrong, what changed,
which test proves it). Do not create a roadmap or an unrelated
refactor; the user's prompt determines the task. Finish vertical
slices — a helper alone is not done until the planner/preview/worker/
launch/web path that uses it works and is tested. When hardware or a
live service is unavailable, finish everything deterministic, then
state the exact remaining validation command and its expected result.
Ask only before destructive changes or when a missing product decision
makes implementation genuinely ambiguous.

Invariants: experimental features fail closed; one canonical launch
path (`modelctl_launch.py` types) for every preview/artifact/launch;
cold/warm measurements never conflated; control plane stays in Python,
tensor execution stays in the `../llama.cpp` fork.

## Setup

```bash
cd ~/workspace/moe-serving/modelctl
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements.txt
```

System Python is externally managed (PEP 668). Always use the `.venv`.

## Running Tests

```bash
cd ~/workspace/moe-serving/modelctl
.venv/bin/python -m unittest discover -p "test_*.py"
```

`discover` picks up every `test_*.py` file automatically -- don't hand-list
modules here, that list has gone stale in the past (new test files added
without updating this command, so `python -m unittest test_a test_b ...`
silently skipped most of the suite).

Or run a single test file:
```bash
.venv/bin/python -m unittest test_modelctl_vram -v
```

The suite is stdlib `unittest`; data fixtures live in `tests/fixtures/`.
Web tests use FastAPI's `TestClient` with mocked modelctl state.
Hardware/reproduction scripts outside the unit suite are documented
where they live (`docs/runtime/`).

## Architecture

| File | Role |
|---|---|
| `modelctl` (no ext) | Shell launcher — resolves `.venv` and runs `modelctl.py` |
| `modelctl.py` | Main CLI: profile lifecycle, placement, router management |
| `modelctl_backends.py` | Backend adapters (managed runtime beyond llama.cpp) |
| `modelctl_benchmark.py` | Benchmark mode definitions and safety |
| `modelctl_capabilities.py` | Backend capability probe/cache (`--modelctl-capabilities`) |
| `modelctl_errors.py` | Structured validation messages and failure classes |
| `modelctl_hardware.py` | Hardware settings, snapshots, and fingerprinting |
| `modelctl_launch.py` | Canonical resolved-backend and launch-command types |
| `modelctl_matrix.py` | Managed llama-swap routing matrix |
| `modelctl_plans.py` | Launch-plan generation and resource claims |
| `modelctl_profiles.py` | Profile schema, migration, and validation |
| `modelctl_runtime.py` | Runtime database: reservations and runtime events |
| `modelctl_services/` | Application services shared by CLI and web |
| `modelctl_setup.py` | First-run readiness checks behind `/setup` |
| `modelctl_storage.py` | Storage topology probing |
| `modelctl_tiers.py` | Tier planner for `place --tiers` |
| `modelctl_transactions.py` | Atomic multi-file mutation transactions |
| `modelctl_tui.py` | Textual wizard (`pull --tui`); calls into `modelctl.py` |
| `modelctl_tune.py` | Plan testing and autotuning |
| `modelctl_vram.py` | Pure-stdlib VRAM math. No `modelctl` import — standalone calculator |
| `modelctl_web/` | FastAPI + HTMX console. Reads concurrent; profile/config writes serialize on the `mutation` job lane |
| `modelctl_worker.py` | Managed worker process |

## Key Conventions

- **Profiles** are JSON at `~/.local/share/modelctl/profiles/<name>.json`. Plain files, hand-editable.
- **No hidden directories** for project state. Generated artifacts belong in visible paths.
- `modelctl_vram.py` has zero imports from `modelctl.py` — it's designed to be copied out standalone.
- Web console auth: Bearer token at `~/.local/share/modelctl/web_token`, or `MODELCTL_WEB_TOKEN` env.
- **Profile schema v2** adds `moe_cache` section. `normalize_profile()` in `modelctl.py` fills defaults for v1 profiles lazily. Never assume a profile has `moe_cache` — always go through `normalize_profile()` or use `.get("moe_cache", {})`.

## Environment Variables

All have defaults. Override via env or `modelctl defaults` (persisted JSON):
- `MODELCTL_HOME` — state dir (default `~/.local/share/modelctl`)
- `MODELCTL_MODELS_DIR` — pulled GGUFs (default `~/models`)
- `MODELCTL_LLAMA_SERVER` — path to llama-server binary
- `MODELCTL_ROUTER_*` — router preset/service/URL/port
- `MODELCTL_DEFAULT_*` — profile defaults (device, ctx, KV quant, flash-attn, TTL, etc.)
- `MODELCTL_GPU_EXCLUDE` — regex to exclude devices (e.g. iGPUs)
- `MODELCTL_WEB_TOKEN`, `MODELCTL_WEB_BIND` — web console config
