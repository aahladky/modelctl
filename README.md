# MoE serving stack

Combined project: the **modelctl** control plane plus the **llama.cpp
fork** it drives, pinned as a submodule.

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

## How they fit together

- modelctl probes the fork's capabilities via
  `llama-server --modelctl-capabilities` and only enables MoE-cache
  features when the fork reports them.
- The fork adds a per-GPU MoE expert weight cache
  (`--moe-cache-bytes`, uniform budget applied per device), Prometheus
  `moe_cache_*` metrics, and a `/cache/reset` endpoint; modelctl's web
  console (`/runtime`) scrapes the metrics and proxies the reset.
- Cache-aware tier planning (`modelctl place --tiers`) reserves the
  cache budget on every participating GPU before placing static
  experts, so the two never collide.

Design docs live in [modelctl/docs/](modelctl/docs/) and
[modelctl/modelctl_sparse_moe_expert_cache_coding_plan.md](modelctl/modelctl_sparse_moe_expert_cache_coding_plan.md).

See [modelctl/README.md](modelctl/README.md) for modelctl itself
(install, profiles, placement, web console).
