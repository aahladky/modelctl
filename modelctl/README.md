# modelctl

A CLI (and optional TUI) for managing local GGUF models served by
`llama-server`, on a workstation with multiple Intel SYCL GPUs.
`modelctl` handles the whole lifecycle: search Hugging Face, pull a
quant, configure runtime settings, size it against available VRAM,
and push it into a router-mode `llama-server` so it loads on demand.

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
| [modelctl.py](modelctl.py) | Main CLI. Profile lifecycle (search/pull/list/show/edit/regen/verify), placement (`place`), and router management (`router status/stats/load/unload`). |
| [modelctl_vram.py](modelctl_vram.py) | Pure-stdlib VRAM math: GGUF header parsing, KV-cache/weights/overhead estimation, GPU probing (`xpu-smi`), and the placement rule. No `modelctl` import — also works as a **standalone calculator** (see below). |
| [modelctl_tui.py](modelctl_tui.py) | Textual wizard for `modelctl pull --tui`. Pure interaction layer; every screen calls an existing function from `modelctl.py` rather than duplicating logic. |

Tests: `test_modelctl.py`, `test_modelctl_vram.py`, `test_modelctl_tui.py` (`python3 -m unittest test_modelctl test_modelctl_vram test_modelctl_tui`).

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
