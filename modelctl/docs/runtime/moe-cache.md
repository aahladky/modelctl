# MoE expert weight transfer cache

Current behavior and limits of the fork's per-GPU expert cache
(`moe_weight_transfer_cache` in the capability probe). For the
GPU-hit/CPU-miss execution mode built on top of it, see
[hybrid-moe.md](hybrid-moe.md). For how to measure it without fooling
yourself, see [moe-cache-testing.md](moe-cache-testing.md).

## What it is

A persistent slot pool on each SYCL device that keeps routed-expert
projection weights resident, so a repeated expert use is served from
device memory instead of re-staged host→device. Slots hold the
projection's bytes **as stored in the model (quantized, copied
verbatim)** — not dequantized. It is a *transfer* cache: it changes
where weight bytes come from, not where the math runs.

- One cache per device; every model on that device shares it. Slot
  contents are identified by (layer, expert, projection) **plus the
  host source pointer**, because two models reuse identical tensor
  names.
- Admission is per projection (gate/up/down tracked separately), with
  partial slot fill; fused `gate_up` layouts are handled by geometry
  learning (below).
- Policy: SLRU (20% probationary / 80% protected) with
  admission-after-N-misses and prefill protection, or plain LRU.
- The slot pool is one contiguous device allocation carved into slots.

## Engagement conditions — read before benchmarking

The cache attaches to a scheduler hook that only fires for
**cross-backend weight copies**. If the whole model is device-resident,
or the op's batch is below the offload threshold, there is nothing to
hook and the cache is inert *even though it is configured and logged
as enabled*.

- Default op-offload threshold: batch ≥ 32 — decode is batch 1, so on
  defaults the cache never engages during generation.
- `GGML_OP_OFFLOAD_MOE_MIN_BATCH=N` lowers the threshold for routed
  MoE ops only (`moe_offload_threshold_control` capability);
  `GGML_OP_OFFLOAD_MIN_BATCH=N` lowers it globally, which drags every
  small-batch op through PCIe and is much more expensive.
- `moe_cache: initialized on device N` in the server log means a cache
  exists; `MoE expert cache enabled: ...` only means the config was
  accepted.

## Flags

| flag | meaning |
|---|---|
| `--moe-cache-bytes N` | per-GPU budget. **One uniform value**: the server stores a single global and every device that creates a cache gets the same budget. A profile declaring different per-device budgets collapses to the max. |
| `--moe-cache-policy slru\|lru` | eviction policy |
| `--moe-cache-admission-misses N` | a projection is admitted after N misses (N=1: first miss) |
| `--moe-cache-prefill-admission on\|off` | whether prefill-phase misses count toward admission; off protects the cache from long prompts flooding it with one-use experts |

## Geometry learning

Cache creation is lazy and starts in learning mode: it observes real
staged copies to discover which projections exist and how large each
one is (fused gate+up tensors, unequal gate/up/down sizes), then
allocates the pool from the observed sizes. A projection the learning
pass never observes is simply never cached (fail-safe). The learned
geometry is logged at finalization.

## Observability

- Prometheus `/metrics`: `moe_cache_hits_total`,
  `moe_cache_misses_total`, `moe_cache_hit_ratio`,
  `moe_cache_evictions_total`, `moe_cache_promotions_total`,
  `moe_cache_h2d_bytes_total`, `moe_cache_served_projections_total`,
  `moe_cache_host_weight_copy_fallbacks_total`, `moe_cache_slots`,
  `moe_cache_slots_used`.
- `POST /cache/reset` clears slots and counters (modelctl's `/runtime`
  page proxies this).
- Hits/misses are counted per **projection**, not per expert: one
  expert use touches gate, up, and down separately.

## Limits

- SYCL backend only; requires the fork (probe with
  `llama-server --modelctl-capabilities`, see
  [backend-capability-schema.md](backend-capability-schema.md)).
- Uniform per-device budget (no per-device sizing).
- Whether the cache helps is model-, quant-, and placement-specific;
  measured results ranged from a 48% decode gain to a net loss on the
  same hardware (see `../evidence/`). Never quote a benefit without
  the three-condition protocol in
  [moe-cache-testing.md](moe-cache-testing.md).
