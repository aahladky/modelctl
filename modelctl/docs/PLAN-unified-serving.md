# Plan: Consolidate local LLM serving into a single llama-swap front door

## Context (read first)

This machine is an Ubuntu 26.04 workstation with two Intel Arc GPUs:

- **Arc Pro B70, 32GB VRAM** — intended for one large model at a time (27–31B INT4)
- **Arc B580, 12GB VRAM** — intended for small always-on utility models (8B chat, embeddings)

Current state (the problem): model serving has accreted across multiple
mechanisms — a Docker Compose file running OVMS containers with
`restart: unless-stopped`, one or more llama-swap configs managed by a
custom CLI tool (`modelctl.py`), possibly stray systemd units and stopped
containers from past experiments (vLLM, llama.cpp-SYCL, OVMS variants).
Clients (primarily a "Hermes" agent stack) have accumulated multiple
provider endpoints, some pointing at dead ports.

Target state:

1. **Exactly one serving entry point**: llama-swap on `127.0.0.1:9292`,
   run as a single systemd user unit.
2. **llama-swap owns all model process lifecycles.** Models are started
   on demand via `docker run` and stopped via `docker stop`. No model
   server has its own compose file, restart policy, or systemd unit.
3. **VRAM contention handled declaratively** via llama-swap's `matrix`
   solver: big models on GPU.0 evict each other; GPU.1 utility models are
   never evicted.
4. **Everything except model weights is regenerable.** The YAML config +
   the models directory fully describe the stack.

## Hard guardrails

- **NEVER delete or move `/home/aaron/ovms/models`** or any directory
  containing model weights (GGUF files, OpenVINO IR dirs). Weights are
  expensive to re-download. When in doubt about whether a directory
  contains weights, ask.
- **Do not run `docker image prune` or `docker system prune` without
  showing the user what will be removed and getting confirmation.**
- Do not modify `modelctl.py` beyond what Phase 5 describes without
  asking — it has functionality unrelated to this plan (GGUF management,
  Hermes provider sync, OVMS compose integration).
- Stop and ask if you find a running service you cannot identify.
- Make a dated backup copy of every config file you modify or retire
  (compose files, llama-swap configs, Hermes provider config) into
  `~/config-backups/<date>/` before touching it.

## Phase 0 — Audit (no changes)

Produce a written inventory before changing anything:

```bash
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}'
docker images
systemctl list-units --type=service --all | grep -iE 'ovms|llama|vllm|hermes|model'
systemctl --user list-units --type=service --all
ss -tlnp
ls -la /home/aaron/ovms/
```

Also locate:
- All docker-compose files related to model serving (`find ~ -name 'docker-compose*.yml' 2>/dev/null`, check `~/ovms/`)
- All llama-swap config files and how llama-swap is currently launched
- The Hermes provider configuration (ask the user where it lives if not obvious)
- What `modelctl.py` currently manages (read it; it lives in the user's projects — ask for the path if needed)

Verify GPU enumeration — this determines GPU.0 vs GPU.1 in all configs:

```bash
docker run --rm --device /dev/dri --group-add $(stat -c "%g" /dev/dri/renderD128) \
  --entrypoint python3 openvino/model_server:latest-gpu -c \
  "from openvino import Core; c=Core(); [print(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices]"
```

Record which index is the B70 (32GB) and which is the B580 (12GB).
**All later steps assume GPU.0 = B70; swap indices everywhere if reversed.**

Also record the render group gid: `stat -c "%g" /dev/dri/renderD128`
(configs below assume 990).

Deliverable: `~/config-backups/<date>/AUDIT.md` summarizing findings,
including a table of every container/unit/port and its verdict
(keep / retire / unknown-ask-user).

## Phase 1 — Unified llama-swap config

Create `/home/aaron/llama-swap/config.yaml`. A starting version is
provided alongside this plan as `ovms-llama-swap-config.yaml` — use it
as the base, then:

1. Fix GPU indices per the Phase 0 enumeration check.
2. Fix the render group gid if not 990.
3. Fold in the user's existing llama.cpp GGUF models from the current
   llama-swap/modelctl-managed config(s) as additional model entries.
   Preserve their existing command lines and context settings. Add each
   to the `matrix` section: GGUF models that run on the B70 join the
   big-model eviction pool (cost 1, mutually exclusive sets with the
   OVMS big models); small ones that fit the B580 alongside the utility
   models get their own sets.
4. Every containerized entry MUST have `cmdStop: docker stop ${MODEL_ID}`
   and `checkEndpoint` set appropriately (`/v2/health/ready` for OVMS,
   `/health` for llama-server).
5. Set `healthCheckTimeout: 600` (first OVMS GPU compile is slow).
6. Mount a persistent OpenVINO compile cache into every OVMS container:
   `-v /home/aaron/ovms/cache:/cache` + `--cache_dir /cache`. Create the
   directory.
7. Pin the OVMS image to a specific version tag (check what
   `openvino/model_server:latest-gpu` currently resolves to and pin
   that), not `latest-gpu`.
8. Naming convention for model IDs: short, stable, lowercase-hyphenated
   (`big-qwen`, `big-gemma`, `fast-8b`, `embed`). Set OVMS
   `--model_name` equal to the llama-swap model ID so request routing
   matches without `useModelName` mapping.

Validate: `llama-swap --config /home/aaron/llama-swap/config.yaml --check`
(or whatever the installed version's validation flag is; check
`llama-swap --help`).

## Phase 2 — systemd unit (the ONLY unit)

Create a systemd user unit `~/.config/systemd/user/llama-swap.service`:

```ini
[Unit]
Description=llama-swap unified model serving front door
After=docker.service network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/llama-swap --config /home/aaron/llama-swap/config.yaml --listen 127.0.0.1:9292
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Adjust the binary path to wherever llama-swap is installed (find it in
Phase 0; if it's currently launched some other way, retire that
mechanism). Enable lingering so it survives logout:
`loginctl enable-linger aaron`. Then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now llama-swap
```

## Phase 3 — Teardown of legacy mechanisms

Only after Phase 2 is verified working (see Phase 4 checks — run at
least the health check first):

1. `docker compose down` on the OVMS compose stack; move the compose
   file into `~/config-backups/<date>/` (do not leave it in place —
   its `restart: unless-stopped` policies will fight llama-swap if it
   is ever `up`'d again by muscle memory).
2. Disable/remove any other model-serving systemd units found in
   Phase 0 (after confirming with the user per the audit verdicts).
3. `docker container prune` (show list first).
4. Image cleanup: list images not referenced by the new config, show
   the user, prune only with confirmation.
5. Retire old llama-swap config files (move to backups).

## Phase 4 — Verification

All must pass:

```bash
# 1. Front door is up
curl -s http://127.0.0.1:9292/v1/models | jq

# 2. Utility model works (B580)
curl -s http://127.0.0.1:9292/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"fast-8b","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}'

# 3. Big model cold start (B70) — time it
time curl -s http://127.0.0.1:9292/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"big-qwen","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}'

# 4. Eviction swap: request the OTHER big model; verify big-qwen container
#    stops and big-gemma starts, while fast-8b/embed containers stay up
curl -s http://127.0.0.1:9292/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"big-gemma","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}'
docker ps --format '{{.Names}}'

# 5. Embeddings endpoint
curl -s http://127.0.0.1:9292/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embed","input":"test"}' | jq '.data[0].embedding | length'

# 6. Compile cache effectiveness: stop big-qwen (idle TTL or restart the
#    unit), re-request it, confirm second cold start is materially faster
#    than the first. Report both timings.

# 7. Reboot survival: systemctl --user is enabled, linger is on. If the
#    user approves a reboot test, verify the stack comes back with zero
#    manual steps and `docker ps` shows nothing until first request.
```

If a swap produces an OOM on the incoming model (VRAM not released fast
enough after `docker stop`), change `cmdStop` to
`docker stop ${MODEL_ID} && sleep 3` and retest.

## Phase 5 — Client and tooling cleanup

1. Hermes: reduce providers to exactly one — `http://127.0.0.1:9292/v1`.
   Remove every other local provider entry. Back up the provider config
   first. Verify Hermes can list and call models through the front door.
2. `modelctl.py`: repoint it so its "OVMS" and "llama-swap" concepts both
   resolve to editing `/home/aaron/llama-swap/config.yaml` + 
   `systemctl --user reload-or-restart llama-swap`. Do NOT rewrite the
   tool wholesale — the minimal change is: (a) stop it from launching or
   managing compose stacks / separate llama-swap profiles, (b) add or
   adapt a command that regenerates/validates the unified config and
   restarts the unit. Propose the diff to the user before applying.
3. Add a short `README.md` in `/home/aaron/llama-swap/` documenting: the
   one port, the one unit, the one config file, how to add a model
   (entry + matrix set), and the experiment convention below.

## Operating conventions (encode in the README)

- **New experiments** enter as `unlisted: true` model entries (callable
  by explicit name, hidden from /v1/models so clients never see them),
  OR run as foreground `docker run --rm` in a terminal. Nothing new gets
  a compose file, a restart policy, or a systemd unit.
- Promotion rule: an experiment that survives two weeks of real use gets
  a listed entry and matrix membership; otherwise its entry gets deleted.
- Disaster recovery = this config file + `/home/aaron/ovms/models` +
  re-pull of pinned images. Nothing else is state.

## Success criteria

- `systemctl --user status llama-swap` is the single answer to "is my
  model serving up?"
- `docker ps` on an idle machine shows at most the two B580 utility
  containers (after first request) and nothing else model-related.
- Requesting any configured model by name Just Works, including
  automatic eviction of whatever is hogging the B70.
- All Phase 4 checks pass, including reboot survival.
- The audit's "retire" list is fully gone; the "unknown" list is empty
  (resolved with the user).
