# MoE serving stack

Combined project: the **modelctl** control plane plus the **llama.cpp
fork** it drives, pinned as a submodule.

This Gitea repository is the source of truth. The GitHub copies
(`aahladky/modelctl`, `aahladky/llama.cpp`) are review mirrors only.

## Mission

Automate away what makes local AI painful — file management, downloads,
launch configs, config files, terminals, linux-after-one-failed-command —
while keeping granular control available from an interface that doesn't
raise your blood pressure, and use the existing hardware to the fullest
extent reasonably possible.

- **Top-level goal:** from "model found on Hugging Face" to "model
  available under the local endpoint" as quickly as possible. The console
  wizard (search → download → place → register → test) is the golden
  path; everything else supports it.
- **Key constraint:** hardware optimization. Automatic placement across
  heterogeneous tiers (GPU / CPU RAM / SSD / fleet RPC) is the hard part.
  The fork and the benchmark lane exist to feed placement rules — research
  earns its keep when it changes a planner decision.
- **The two numbers that matter:** wall-clock time-to-serve, and the
  fraction of best-known tok/s the planner's automatic choice achieves.
- **A workflow that requires a terminal is a gap.** Failures must be
  recoverable from the console.

| Piece | Path | Repo |
|---|---|---|
| modelctl (CLI + web console) | [modelctl/](modelctl/) | this repo |
| llama.cpp fork (SYCL MoE expert cache) | [llama.cpp/](llama.cpp/) | [moe-serving/llama.cpp](../../llama.cpp), branch `feature/sycl-moe-expert-cache` |

## Clone

```
git clone --recursive git@gitea:moe-serving/modelctl.git
```

`--recursive` materializes the llama.cpp fork at the exact commit
modelctl expects. Already cloned without it? `git submodule update --init`.

## Start here

```
cd modelctl && modelctl web install
```

Installs and starts the web console as a systemd user service and prints
its URL and token. The console is the primary interface; the readiness
section of its settings page reports whether this machine has a model
directory, a runnable llama.cpp build, GPUs, and llama-swap. The CLI stays available for
bootstrap, automation, diagnostics, and recovery.

## How they fit together

- modelctl probes the fork's capabilities via
  `llama-server --modelctl-capabilities` and only enables MoE-cache
  features when the fork reports them.
- The fork adds a per-GPU MoE expert weight cache
  (`--moe-cache-bytes`, a uniform per-device budget or a `DEV=N,DEV=N`
  per-device map), Prometheus
  `moe_cache_*` metrics, and a `/cache/reset` endpoint; modelctl's web
  console's operate page scrapes the metrics and proxies the reset.
- Cache-aware tier planning (`modelctl place --tiers`) reserves the
  uniform runtime cache budget on every GPU named in the profile's
  cache budget map before placing static experts, so the two never
  collide on those devices.

Docs: [operations](modelctl/docs/operations.md) ·
[runtime (MoE cache, hybrid, capability schema, testing)](modelctl/docs/runtime/).

See [modelctl/README.md](modelctl/README.md) for modelctl itself
(install, profiles, placement, web console).
