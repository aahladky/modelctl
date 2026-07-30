# modelctl

A **web console** (with a CLI behind it) for managing local GGUF models
served by `llama-server`, on a workstation with multiple Intel SYCL GPUs.
`modelctl` handles the whole lifecycle: find or import a model, inspect
it, compare placement plans against your actual hardware, test them, and
register the winner with `llama-swap` so it loads on demand.

## Start here

```
modelctl web install
```

Installs and starts the console as a systemd user service, then prints
its URL and token. Open that URL — everything below is reachable from
it, and the console's `/setup` page tells you what (if anything) this
machine is still missing.

The CLI remains fully supported, and is the right tool for bootstrap,
automation, diagnostics, and recovery — but the browser is the primary
workflow, because placement decisions are comparisons and comparisons
want a screen.

## Why this exists

Running several local models on limited VRAM means every load is a
placement decision: which GPU(s), how much context, how the KV cache
is quantized. `modelctl` makes that decision mechanical instead of
manual — it estimates a model's VRAM footprint from the GGUF header
itself (weights + KV cache + overhead) and recommends where it fits,
before you ever start the server.

## Components

| File | Role |
|---|---|
| [modelctl](modelctl) | Stable shell launcher. Resolves the checkout's `.venv` and runs `modelctl.py` with it, so the `~/.local/bin/modelctl` symlink doesn't depend on system-Python packages. |
| [modelctl.py](modelctl.py) | Main CLI. Profile lifecycle (search/pull/list/show/edit/regen/verify), placement (`place`), and router management (`router status/stats/load/unload`). |
| [modelctl_vram.py](modelctl_vram.py) | Pure-stdlib VRAM math: GGUF header + tensor-table parsing, KV-cache/weights/overhead estimation, GPU probing (`xpu-smi`, with `llama-server --list-devices` fallback), and the placement rule. No `modelctl` import — also works as a **standalone calculator** (see below). |
| [modelctl_tiers.py](modelctl_tiers.py) | Pure tier planner for `place --tiers`: tier 1–4 decisions, MoE expert-layer assignment (bandwidth-ordered), dense `-ngl` math, llama-server flag emission. |
| [modelctl_tui.py](modelctl_tui.py) | Textual wizard for `modelctl pull --tui`. Pure interaction layer; every screen calls an existing function from `modelctl.py` rather than duplicating logic. |
| [modelctl_web/](modelctl_web/) | FastAPI + HTMX console (`modelctl web`) — the primary interface. Reads run concurrently; writes go through a single JobRunner. |
| [modelctl_setup.py](modelctl_setup.py) | First-run readiness checks behind the console's `/setup` page. |

Tests: run the whole suite with
`.venv/bin/python -m unittest discover -p "test_*.py"` — `discover` picks
up every `test_*.py`, which a hand-listed set has failed to do before.

## Installation

This checkout keeps its Python dependencies in a project-local virtualenv;
system Python on this host is externally managed and does not provide them.

```
cd ~/workspace/moe-serving/modelctl
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements.txt
ln -sfn "$PWD/modelctl" ~/.local/bin/modelctl
```

Run `modelctl ...` normally afterward. The pinned requirements include
`huggingface-hub`, `PyYAML`, and `textual` (for `pull --tui`).

## Profiles

A **profile** is a JSON file (`~/.local/share/modelctl/profiles/<name>.json`)
describing one model: its HF repo/file, local model/mmproj/mtp paths, and a
`config` dict (context length, GPU device/tensor-split, KV cache quant
type(s), flash-attn, TTL, etc.). From a profile, `modelctl` generates:

- `run.sh` — a standalone `llama-server` invocation
- a section in the router's `preset.ini` (router mode, load-on-demand)
- an Ollama-style `Modelfile`

Profiles are plain JSON so they're easy to inspect, hand-edit, or
regenerate (`modelctl regen <name>`) after a manual tweak.

## Zero-config pulls (`pull --yes`)

For the common case — "new ~27B-class fine-tune, get it serving" —
`modelctl pull <repo> --yes` asks nothing:

- **Auto-quant**: picks the *largest* quant whose estimated footprint
  (weights + KV + overhead) fits the primary GPU — best quality that stays
  tier 1. If nothing fits, it takes the smallest quant and tells you to
  re-plan with `place --tiers` afterward. (`imatrix` files are never
  mistaken for models.)
- **Auto-config**: your saved defaults (`modelctl defaults`), pinned to the
  primary GPU when the quant fits. KV cache types, flash-attn, TTL — all
  from defaults, no prompts.
- **Auto-context**: after download, the context is computed from the actual
  GGUF — the largest step (8k…1M, capped at the model's own max) whose
  *exact* KV cache (sliding-window aware) fits the tier-1 budget. No more
  guessing 32k vs 131k per model; if even 8k doesn't fit, it says so and
  points at `place --tiers`.
- **Load-time adaptation**: tier-1 profiles get `--fit on` and no fixed
  `-ngl`/`-c` flags, so llama.cpp sizes the context to whatever *actually*
  fits when the model loads — more than the ceiling on an empty card, a
  graceful shrink (not an OOM) when another model is resident. The computed
  auto-context stays in the profile as the advertised ceiling clients see
  (fit refuses to adjust user-set values, which is why the flags must be
  omitted). Tier-3/4 offload plans keep `--fit off` -- the fit simulation
  crashes against multi-device `-ot` overrides.
- **Auto-env**: populates the oneAPI `LD_LIBRARY_PATH` from your env script
  so SYCL launches don't depend on how modelctl was invoked.
- **Auto-MTP**: if the repo ships a separate MTP draft-head file, it's
  pulled and enabled (free speculative-decoding speed).
- **Auto-name**: the repo label, quant-stripped and slugified; mmproj is
  skipped (text-only) — re-run interactively if you want vision.

## GPU placement

`modelctl place` estimates every enabled profile's VRAM footprint and
recommends a device/tensor-split:

- Pins to the primary GPU if the model fits within `VRAM_LIMIT_PCT` of it.
- Otherwise splits across GPUs in proportion to their capacity.
- Warns (doesn't silently truncate) if a model exceeds combined VRAM.

`--apply` rewrites the recommendation into the profile and re-syncs.
The intended usage pattern is at most one large (15GB+) model loaded
at a time, with `router load --evict` used for guarded swaps.

Gemma-family (sliding-window-attention) models get per-layer-aware KV
math — a naive full-context formula over-counts their footprint by
close to 17x, since SWA layers only cache `swa_window` tokens, not the
full context.

## Tiered placement (`place --tiers`)

`modelctl place --tiers` is the multi-tier variant for models that don't
fit VRAM. It parses the GGUF tensor table (exact per-layer and per-expert
bytes) and plans across four tiers:

| tier | model fits in | what the planner emits |
|---|---|---|
| 1 | primary GPU alone | `--device SYCL0` |
| 2 | all GPUs | `--split-mode layer --tensor-split <capacity ratio>` |
| 3 | GPUs + system RAM | GPU spill to CPU, `--no-mmap` (fully resident) |
| 4 | beyond RAM (SSD streaming) | same layout, mmap left on |

MoE models get **expert-granular** placement — attention/KV/shared experts
stay on the primary GPU, routed-expert layers are assigned to each GPU
fastest-bandwidth-first (per the card table in `modelctl_tiers.py`), and
the remainder goes to CPU as an `-ot` override bundle. Dense models get a
computed `-ngl` instead. Generated flags encode the llama.cpp quirks this
stack needs: `-ot` specific-ranges-before-catch-all (first match wins),
`--split-mode layer` + explicit `--device` for multi-SYCL, and `--fit off`
when overrides span devices.

`--apply` rewrites the profile's placement fields (preserving non-placement
`extra` flags like `--ubatch-size`), fills in the oneAPI `env` from your env
script if the profile has none, and re-syncs. Per-profile `"binary"` pins
in the profile JSON override the global llama-server resolution — use them
for models that need a specific build (e.g. a fork with a not-yet-upstream
architecture), so env-less regen/sync runs can't clobber the choice.

## Independent K/V cache quantization

K and V caches can be quantized independently
(`--cache-type-k`/`--cache-type-v` in llama.cpp), which `modelctl`
exposes as `cache_type_k`/`cache_type_v` in a profile's config —
e.g. K at `q8_0` for quality, V at `q4_0` to shrink the cache further.
Profiles saved before this existed still work: a single legacy
`kv_quant` field is used for both, via `_resolve_cache_types()`.

### Standalone VRAM calculator

`modelctl_vram.py` has no dependency on `modelctl.py` and can be
copied out and run on its own:

```
python3 modelctl_vram.py <model.gguf> --ctx 131072 --cache-type-k q8_0 --cache-type-v q4_0
```

Prints a weights/KV-cache/overhead/total breakdown, with K and V
shown separately when they diverge.

## Router mode

Models load on demand behind a router-mode `llama-server`
(systemd `--user` unit, default `llama-router.service`). `modelctl
router status` shows what's loaded and where, with a VRAM footer;
`modelctl router stats` reads per-model Prometheus metrics
(throughput, etc.) exposed by the router when `metrics = true`.
`modelctl router load --evict` loads a model, unloading another
first if needed to fit — API/autoload requests always take
precedence over any local heuristics.

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `MODELCTL_HOME` | State dir (profiles, defaults) — default `~/.local/share/modelctl` |
| `MODELCTL_MODELS_DIR` | Where pulled GGUFs land — default `~/models` |
| `MODELCTL_LLAMA_SERVER` | Path to the `llama-server` binary |
| `MODELCTL_ROUTER_PRESET`, `MODELCTL_ROUTER_SERVICE`, `MODELCTL_ROUTER_BASE_URL`, `MODELCTL_ROUTER_PORT` | Router-mode preset path / systemd unit / API base |
| `MODELCTL_DEFAULT_DEVICE`, `_CTX`, `_SPLIT_MODE`, `_TENSOR_SPLIT`, `_FLASH_ATTN`, `_TTL`, `_MTP` | Defaults for newly-created profiles |
| `MODELCTL_DEFAULT_KV_QUANT` | Legacy single K/V quant default (fallback for both) |
| `MODELCTL_DEFAULT_CACHE_TYPE_K`, `MODELCTL_DEFAULT_CACHE_TYPE_V` | Independent K/V quant defaults (override the legacy one) |
| `MODELCTL_DEFAULT_PRIMARY_GPU`, `MODELCTL_DEFAULT_VRAM_LIMIT_PCT` | Placement policy |
| `MODELCTL_GPU_EXCLUDE` | Regex to exclude devices from placement inventory (e.g. iGPUs that misreport shared RAM as VRAM) |
| `MODELCTL_HERMES_CONFIG` | Path to sync an external agent config's custom-provider list |
| `MODELCTL_PASSTHROUGH_ENV` | Extra env vars to forward into generated `run.sh`/preset entries |

`modelctl defaults` reads/writes these as a persisted JSON file so you
don't need to export them every session.

## Further reading

Design rationale and implementation history for individual features
live in `docs/superpowers/specs/` (design docs) and
`docs/superpowers/plans/` (task-by-task implementation plans):

- VRAM-aware placement + router observability
- Split K/V cache quantization + standalone calculator
- TUI pull wizard

## Web console (`modelctl web` / modelctl-web.service)

A FastAPI + HTMX console on `:9293` for the cases the CLI's auto-config
doesn't cover. The CLI remains the primary path; the UI is for visibility
and edge-case overrides.

- **Dashboard**: all profiles with live state (llama-swap loaded/registered),
  per-GPU VRAM and RAM gauges.
- **Profile edit**: every config field (device, split, ctx, KV types, fit,
  extra flags, binary pin, env), with the rendered command preview. Saves
  regenerate artifacts and re-sync.
- **Tier planner**: the `place --tiers` dry-run rendered per profile (layout
  table, warnings, config diff) with one-click apply.
- **Pull wizard**: HF search → quant table with the auto recommendation →
  background download job with progress → auto-config profile.
- **Benchmarks**: speed.py runs as jobs with persistent history.
- **Jobs**: everything long-running is a serialized background job (SQLite at
  `~/.local/share/modelctl/web_jobs.db`) — all mutations flow through a
  single writer, since profiles and the llama-swap config are plain files.

Auth: one shared token (Bearer header, `?token=`, or login cookie). Stored at
`~/.local/share/modelctl/web_token` (created on first start, override with
`MODELCTL_WEB_TOKEN`). Binds `MODELCTL_WEB_BIND` (default `0.0.0.0:9293`).
Run in the foreground with `modelctl web`, or as a service:
`systemctl --user enable --now modelctl-web`.
