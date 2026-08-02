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
it, and the settings page's readiness section tells you what (if anything)
this machine is still missing.

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

FastAPI on `:9293` (`modelctl web` foreground, or the installed
`modelctl-web.service`) serving a compiled Preact SPA from
`console/dist`. One service, one port, one token. Everything operational
streams over a single SSE tick at `/api/v2/events` — there is no manual
refresh anywhere. Five pages:

- **Operate (`/v2/`)**: service chips, per-GPU VRAM and RAM meters with
  sparklines, MoE-cache hit ratios, resident models with live tok/s, and
  load/unload. Each region degrades on its own and says which probe
  failed rather than rendering a zero as fact.
- **Model hub (`/v2/models`, `/v2/models/<name>`)**: per-model overview
  and placement, compiled launch plans joined with the measurement store
  (measured vs estimated, always tagged), full plan-run history with the
  bottleneck judgement, log tail, and a typed configure form whose save
  shows the planner's admission preview and gates structural changes
  behind an explicit confirm.
- **Add (`/v2/add`, `/v2/add/<id>`)**: the acquisition workflow — HF
  search or local file, verification, quant inspection, download,
  analysis, plan selection, measured testing, registration — as a
  stepper with inline job progress, visible blocked-advance reasons and
  single-shot retry.
- **Jobs (`/v2/jobs`)**: running / queued / history across the lanes
  (SQLite at `~/.local/share/modelctl/web_jobs.db`); profile and config
  mutations serialize on a single-worker lane because profiles and the
  llama-swap config are plain files. Cancel is optimistic and un-happens
  loudly on refusal.
- **Settings (`/v2/settings`)**: readiness checklist, typed profile
  defaults, hardware policy (per-device reserves/roles/bandwidth, RAM
  reserve, storage calibration), access and state paths, integration
  manifest and capability report, support bundle. No JSON anywhere in
  the UI; the files on disk stay hand-editable.

The typed surface lives under `/api/v2/*`; a legacy JSON API remains
under `/api/*`. Every URL the old server-rendered console published
301s to its `/v2` equivalent.

Beyond the console: `modelctl ovms-add`/`ovms-convert` manage OpenVINO
Model Server profiles (a second backend), `modelctl test --evals` runs
lm-eval suites, and `modelctl doctor [--bundle]` produces diagnostics.

Auth: one shared token (Bearer header or login cookie; tokens in URLs
are rejected), stored at `~/.local/share/modelctl/web_token`.

### Network exposure

Binds `MODELCTL_WEB_BIND` (default `0.0.0.0:9293`, i.e. LAN-accessible).
This is a trusted-user control plane that can launch processes and change
runtime configuration, so its reach is always shown explicitly: `modelctl
web`, `modelctl web url`, and the settings page's readiness section all
report whether the
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

## Benchmarking

Three pieces sit above the per-run protocol in
[docs/runtime/moe-cache-testing.md](docs/runtime/moe-cache-testing.md),
all documented in
[docs/runtime/benchmarking.md](docs/runtime/benchmarking.md):

- **Paired comparisons** (`modelctl_paired.py`) alternate two conditions
  back-to-back and compare the delta *within* each pair, so machine
  drift between blocks cannot land in the answer. Reports every raw run,
  per-pair deltas with signs, and an exact sign test — never a verdict.
- **Anchors** (`anchors.json`) store a reference measurement beside the
  fingerprint of what produced it (build commit, profile hash,
  environment, driver). A battery re-runs an anchor only when that
  fingerprint is stale, when the anchor is marked `void`, or when it is
  the laguna canary, which always runs.
- **The night lane** (`night-lane.json`) holds pre-registered
  comparisons — question, criterion and sample size committed *before*
  any numbers exist. Enabled jobs run unattended on the benchmark lane,
  but only through a window that requires llama-swap to be holding
  nothing and the load to be below a ceiling; both halves fail closed.
  Evidence lands in `docs/evidence/` with one-line summaries.

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
