# Operations

How to set up, run, recover, and update this machine's serving stack.
The what-and-why is in [architecture.md](architecture.md).

## Services

| unit (systemd --user) | role |
|---|---|
| `llama-swap.service` | the serving front door — OpenAI-compatible API on `127.0.0.1:9292`, loads models on demand |
| `modelctl-web.service` | the management console on `:9293` (`modelctl web install` creates it) |
| `moe-upstream-drift.timer` | weekly llama.cpp upstream drift report (repo `systemd/`) |

`systemctl --user status llama-swap` answers "is my model serving up?".
Lingering is enabled so units survive logout.

## Setup on a fresh checkout

```bash
git clone --recursive git@gitea:moe-serving/modelctl.git ~/workspace/moe-serving
cd ~/workspace/moe-serving/modelctl
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r requirements.txt
ln -sfn "$PWD/modelctl" ~/.local/bin/modelctl
modelctl web install         # installs + starts the console, prints URL/token
```

The console's readiness section (settings) reports what the machine is
still missing
(model directory, runnable llama.cpp build, GPUs, llama-swap).

Build the fork (SYCL needs the oneAPI env; `llama-sycl-env.sh` exits a
script that sources it, so source it in your shell first):

```bash
source llama.cpp/llama-sycl-env.sh
cmake --build llama.cpp/build-sycl -j --target llama-server
```

## State — what is precious, what is regenerable

Precious (backup targets; everything else can be regenerated):

- **model weights** — `~/models` (never delete a directory containing
  GGUF/weights to "clean up"; they are expensive to re-download)
- **profiles** — `~/.local/share/modelctl/profiles/*.json` (plain JSON,
  hand-editable)
- this repo (with the submodule pin)

Regenerable state under `~/.local/share/modelctl/`:

| path | contents |
|---|---|
| `defaults.json` | persisted `modelctl defaults` |
| `hardware.json`, `gpu_map.json` | hardware settings and GPU inventory |
| `backend_capabilities/` | cached capability probes |
| `runtime.db` | plan runs, reservations, runtime events (benchmark/observation history) |
| `web_jobs.db` | web console job queue/history |
| `web_token` | console auth token (regenerated on first start) |

The llama-swap config lives at `~/services/llama-swap/config.yaml`
(override `MODELCTL_LLAMA_SWAP_CONFIG`); modelctl owns the per-profile
entries in it via `sync`, and regenerates `run.sh`/preset artifacts
with `modelctl regen <name>`.

## Recovery

- Console unreachable: `systemctl --user status modelctl-web`, then
  `modelctl web` in the foreground for the traceback. Token is in
  `~/.local/share/modelctl/web_token`.
- Serving broken after a config change: profiles are plain JSON — fix
  or `git`-restore the profile, then `modelctl regen <name>` and
  `systemctl --user restart llama-swap.service`.
- A wedged model process: `modelctl router status` /
  `modelctl router load --evict`; kill by PID (not `pkill -f`) and
  verify VRAM returns before relaunching.
- Disaster recovery = this repo + `~/models` + `profiles/`. Reinstall
  per Setup, restore profiles, `modelctl sync`.

## Update and rollback

The repo pins the exact fork commit modelctl was validated with;
`integration-manifest.json` records the last pair that passed hardware
acceptance together (`validated_modelctl_commit` +
`validated_llama_commit`) — that pair is the rollback target.

Update:

```bash
git -C ~/workspace/moe-serving pull && git submodule update --init
source llama.cpp/llama-sycl-env.sh && cmake --build llama.cpp/build-sycl -j --target llama-server
ci/checks.sh          # pin/manifest consistency + tests (--quick to skip the build check)
```

Rollback: check out the validated pair from the manifest, rebuild, and
re-run `ci/checks.sh`. Capability probes are cached per binary, so a
rolled-back binary is re-probed automatically; if in doubt, run
`llama-server --modelctl-capabilities` and compare against
[runtime/backend-capability-schema.md](runtime/backend-capability-schema.md).

## Network exposure

The console binds `MODELCTL_WEB_BIND` (default `0.0.0.0:9293`,
LAN-accessible, plain HTTP, single shared token). This is a
trusted-home-LAN default; exposure to untrusted networks is
unsupported. Loopback-only: `MODELCTL_WEB_BIND=127.0.0.1:9293`.
llama-swap binds loopback only.

Fleet RPC nodes are the sharpest edge here: `ggml-rpc-server` has no
authentication at all, and ph16-71's two ports are currently reachable
from the whole `192.168.0.0/24` subnet, not just the rig. The rules that
would fix that are written but unapplied — see `docs/fleet/README.md`
and `docs/fleet/ph16-71-ufw.sh`.

## Fleet nodes

Remote RPC nodes are **optional** planning targets. Registering one
cannot change any launch that worked without it: the planner adds
fleet variants on top of the local plan set, and with every node absent
the compiled plans are byte-identical to a fleet-free checkout.

* Registry: `$MODELCTL_HOME/fleet.json` (`modelctl_fleet`).
* Presence: `$MODELCTL_HOME/fleet-presence.json`, written only by an
  explicit `refresh_presence()`. Planning reads it and never opens a
  socket, so plan compilation stays deterministic and the test suite
  stays hermetic. A node nobody has probed is treated as absent.
* A node is used only when it is enabled, reachable, and its recorded
  build pin equals this checkout's llama.cpp pin. A node on a different
  commit is reported, not used.
* Per-node budgets are operator-declared ceilings and are admitted
  through the same `_usable_vram_map` / `_admission_overflow` path local
  GPUs use, under a namespaced device key (`RPC:<node>:<device>`).

Node-specific detail, including the security caveat and the rebuild
procedure when the pin moves: `docs/fleet/README.md`.
