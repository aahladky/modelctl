# Agent Guide — modelctl

Python CLI + optional TUI/web console for managing local GGUF models on multi-GPU Intel SYCL workstations.

## Active Roadmap

The current implementation plan is `docs/modelctl-task-by-task-roadmap-2026-07-30.md` (supersedes the 2026-07-29 revision — the live repository had already implemented much of that plan's early/middle phases, and this revision adds a Phase 0 controlled upstream-sync/maintenance strategy ahead of further runtime work). Read the relevant phase/task before implementing. Key invariants: experimental features fail closed (§2.5), one canonical launch path (§2.2), cold/warm measurements never conflated, control plane stays in Python, tensor execution stays in the `../llama.cpp` fork.

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

Tests are pure unittest (no fixtures, no services). Web tests use FastAPI's `TestClient` with mocked modelctl state.

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
| `modelctl_plans.py` | Launch-plan generation |
| `modelctl_profiles.py` | Profile schema, migration, and validation |
| `modelctl_runtime.py` | Runtime database: reservations and runtime events |
| `modelctl_services/` | Application services (cache, hardware, plan, profile, runtime) shared by CLI and web |
| `modelctl_storage.py` | Storage topology probing |
| `modelctl_tiers.py` | Tier planner for `place --tiers` |
| `modelctl_transactions.py` | Atomic multi-file mutation transactions |
| `modelctl_tui.py` | Textual wizard (`pull --tui`). Calls into `modelctl.py`, no logic duplication |
| `modelctl_tune.py` | Plan testing and autotuning |
| `modelctl_vram.py` | Pure-stdlib VRAM math. No `modelctl` import — standalone calculator |
| `modelctl_web/` | FastAPI + HTMX console (`modelctl web`). Reads concurrent, writes via single JobRunner |
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
