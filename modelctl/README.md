# modelctl

A **web console** (with a CLI behind it) for managing local GGUF models
served by `llama-server`, on a workstation with multiple Intel SYCL GPUs.
`modelctl` handles the whole lifecycle: find or import a model, inspect
it, compare placement plans against your actual hardware, test them, and
register the winner with `llama-swap` so it loads on demand.

This is a local-first tool, built for and operated on one machine. This
README is the operator guide; the design boundaries live in
[docs/architecture.md](docs/architecture.md), day-2 operations in
[docs/operations.md](docs/operations.md), and runtime (MoE cache /
hybrid) documentation under [docs/runtime/](docs/runtime/).

## Start here

```
modelctl web install
```

Installs and starts the console as a systemd user service, then prints
its URL and token. Open that URL — everything below is reachable from
it, and the console's `/setup` page tells you what (if anything) this
machine is still missing.

The browser is the primary workflow, because placement decisions are
comparisons and comparisons want a screen. The CLI remains fully
supported and is the right tool for bootstrap, automation, diagnostics,
and recovery.

## Why this exists

Running several local models on limited VRAM means every load is a
placement decision: which GPU(s), how much context, how the KV cache
is quantized. `modelctl` makes that decision mechanical instead of
manual — it estimates a model's VRAM footprint from the GGUF header
itself (weights + KV cache + overhead) and recommends where it fits,
before you ever start the server.

## Installation

This checkout keeps its Python dependencies in a project-local virtualenv;
system Python on this host is externally managed and does not provide them.

```
cd ~/workspace/moe-serving/modelctl
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements.txt
ln -sfn "$PWD/modelctl" ~/.local/bin/modelctl
```

Run `modelctl ...` normally afterward. Tests:
`.venv/bin/python -m unittest discover -p "test_*.py"`.

## The web console

FastAPI + HTMX on `:9293` (`modelctl web` foreground, or the installed
`modelctl-web.service`).

- **Add model (`/add`)**: the acquisition workflow — HF search or local
  file, verification, quant inspection, download, analysis, plan
  selection, measured testing, registration.
- **Dashboard**: all profiles with live state (llama-swap
  loaded/registered), per-GPU VRAM and RAM gauges.
- **Profile edit**: every config field with a rendered command preview;
  saves regenerate artifacts and re-sync.
- **Tier planner**: the `place --tiers` dry-run per profile (layout
  table, warnings, config diff) with one-click apply.
- **Runtime (`/runtime`)**: MoE-cache metrics scraped from the server,
  cache reset.
- **Benchmarks**: speed runs as jobs with persistent history.
- **Jobs**: long-running work runs as background jobs in lanes (SQLite
  at `~/.local/share/modelctl/web_jobs.db`); profile and config
  mutations serialize on a single-worker lane because profiles and the
  llama-swap config are plain files.
- **Launch plans (`/profiles/<name>/plans`)**: candidate plans with
  evidence, select/disable/test/tune actions.
- **History (`/profiles/<name>/history`)**: past plan runs with
  bottleneck classification.
- **Hardware (`/hardware`)**: per-device reserves/roles/bandwidth,
  storage calibration.
- **Settings (`/settings`)**: persisted defaults, diagnostics, support
  bundle download.
- **Routing (`/runtime/routing`)**: the managed llama-swap routing
  matrix with preview, apply, rollback.
- A JSON API mirrors most of this under `/api/*`, and job progress
  streams over SSE at `/events/jobs/<id>`.

Beyond the console: `modelctl ovms-add`/`ovms-convert` manage OpenVINO
Model Server profiles (a second backend), `modelctl test --evals` runs
lm-eval suites, and `modelctl doctor [--bundle]` produces diagnostics.

Auth: one shared token (Bearer header or login cookie; tokens in URLs
are rejected), stored at `~/.local/share/modelctl/web_token`.

### Network exposure

Binds `MODELCTL_WEB_BIND` (default `0.0.0.0:9293`, i.e. LAN-accessible).
This is a trusted-user control plane that can launch processes and change
runtime configuration, so its reach is always shown explicitly: `modelctl
web`, `modelctl web url`, and the `/setup` page all report whether the
console is loopback-only or LAN-accessible and on which address. LAN
access uses plain HTTP — the token travels unencrypted — which is an
acceptable personal default on a trusted home LAN. **Exposure to
untrusted networks is unsupported**; there is deliberately no
multi-user/permissions system. For loopback-only operation set
`MODELCTL_WEB_BIND=127.0.0.1:9293`.

## Profiles

A **profile** is a JSON file (`~/.local/share/modelctl/profiles/<name>.json`)
describing one model: its HF repo/file, local model/mmproj/mtp paths, and a
`config` dict (context length, GPU device/tensor-split, KV cache quant
type(s), flash-attn, TTL, etc.). From a profile, `modelctl` generates:

- `run.sh` — a standalone `llama-server` invocation
- a section in the router's config (llama-swap, load-on-demand)
- an Ollama-style `Modelfile`

Profiles are plain JSON so they're easy to inspect, hand-edit, or
regenerate (`modelctl regen <name>`) after a manual tweak. K and V
caches can be quantized independently (`cache_type_k`/`cache_type_v`);
a per-profile `"binary"` pin overrides the global llama-server
resolution for models that need a specific build.

`modelctl pull <repo> --yes` is the zero-question path: it picks the
largest quant that fits the primary GPU, applies your saved defaults
(`modelctl defaults`), computes the context ceiling from the actual
GGUF, and fills the oneAPI env from your env script. If the repo ships
a separate MTP draft head it is pulled and enabled — MTP is
model-dependent and opt-in, not a universal speedup.

## Placement

`modelctl place` estimates every enabled profile's VRAM footprint and
recommends a device/tensor-split; `--apply` writes it back to the
profile and re-syncs. `modelctl place --tiers` is the multi-tier
variant for models that don't fit VRAM: it parses the GGUF tensor
table (exact per-layer and per-expert bytes) and plans across four
tiers — primary GPU, all GPUs, GPUs + resident RAM, and SSD-backed
mmap. MoE models get expert-granular placement (attention/KV/shared
experts on the primary GPU, routed-expert layers assigned
fastest-bandwidth-first, remainder to CPU); dense models get a
computed `-ngl`. The generated flags encode the llama.cpp quirks this
stack needs — the details live with the code in `modelctl_tiers.py`.

Sliding-window-attention models (Gemma family) get per-layer-aware KV
math; a naive full-context formula over-counts their footprint by
close to 17×.

`modelctl_vram.py` also works standalone, with no `modelctl` import:

```
python3 modelctl_vram.py <model.gguf> --ctx 131072 --cache-type-k q8_0 --cache-type-v q4_0
```

## Serving

Models load on demand behind llama-swap (`llama-swap.service`, port
9292 — see [docs/operations.md](docs/operations.md)). `modelctl router
status` shows what's loaded and where with a VRAM footer; `modelctl
router stats` reads per-model Prometheus metrics; `modelctl router
load --evict` loads a model, unloading another first if needed. The
intended pattern is at most one large (15GB+) model loaded at a time.

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `MODELCTL_HOME` | State dir (profiles, defaults) — default `~/.local/share/modelctl` |
| `MODELCTL_MODELS_DIR` | Where pulled GGUFs land — default `~/models` |
| `MODELCTL_LLAMA_SERVER` | Path to the `llama-server` binary |
| `MODELCTL_LLAMA_SWAP_CONFIG`, `MODELCTL_LLAMA_SWAP_SERVICE`, `MODELCTL_LLAMA_SWAP_BASE_URL`, `MODELCTL_LLAMA_SWAP_DIR` | Router config path / systemd unit / API base / install dir (the port comes from the base URL) |
| `MODELCTL_OVMS_*` | OpenVINO Model Server backend knobs (`ovms-add`/`ovms-convert`) |
| `MODELCTL_PROBE_TIMEOUT`, `MODELCTL_WEB_SECURE_COOKIE`, `MODELCTL_BENCH_SH`, `MODELCTL_SPEED_PY` | Probe timeout / cookie policy / benchmark script overrides |
| `MODELCTL_DEFAULT_*` | Defaults for new profiles (device, ctx, split, KV quant, flash-attn, TTL, MTP, primary GPU, VRAM limit) |
| `MODELCTL_GPU_EXCLUDE` | Regex to exclude devices from placement inventory (e.g. iGPUs that misreport shared RAM as VRAM) |
| `MODELCTL_HERMES_CONFIG` | Path to sync an external agent config's custom-provider list |
| `MODELCTL_PASSTHROUGH_ENV` | Extra env vars to forward into generated `run.sh`/preset entries |
| `MODELCTL_WEB_TOKEN`, `MODELCTL_WEB_BIND` | Web console auth/bind |

`modelctl defaults` reads/writes these as a persisted JSON file so you
don't need to export them every session.
