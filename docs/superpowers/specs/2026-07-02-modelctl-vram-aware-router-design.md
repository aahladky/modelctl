# modelctl: VRAM-aware llama-router integration

**Date:** 2026-07-02
**Status:** Draft for review

## Problem

modelctl currently treats GPU placement as a static per-profile choice made
blind (defaults: split `3,1` across both cards) and gives the router no VRAM
awareness at all. The real constraint on this machine is bytes, not model
count: four 1–2GB models coexist fine, while a single 15+GB model needs the
primary card mostly to itself. Concretely:

- Primary GPU: Intel 0xe223, 32GB (xpu-smi device 0)
- Secondary GPU: Intel Arc B580, 12GB (xpu-smi device 1)

Desired policy: models run on the primary card alone when they fit; the B580
is overflow only. Explicit loads should refuse (or evict) rather than
oversubscribe. Throughput and load-time stats should be visible without
duplicating what nvtop already shows.

## Non-goals (this spec)

- **Autoload eviction** ("request for qwen3.6 27B evicts gemma 4 26B"): the
  router has no size-aware eviction and no per-model autoload control, so
  this is delegated to **Project 2: a llama.cpp router patch** — on load
  failure (or models-max block), evict least-recently-used instance(s) and
  retry the requested load. Local build first, upstream PR candidate. No
  modelctl watcher process will be built, since the patch obsoletes it.
- Managing the router's systemd unit or `--models-max` (count is the wrong
  knob; left as the operator's coarse backstop).
- NVIDIA/CUDA support for the VRAM probe (SYCL/xpu-smi only; the probe is
  isolated behind one function so other backends can be added later).

## Component A: VRAM estimation + static placement

### `read_gguf_kv_metadata(path) -> dict`

Self-contained GGUF header reader (no new dependency; the format's KV
section is straightforward binary parsing). Extracts, for arch `X`:
`X.block_count`, `X.attention.head_count`, `X.attention.head_count_kv`,
`X.embedding_length`, and `X.attention.key_length`/`value_length` when
present. Returns `{}` on any parse failure (never raises).

### `estimate_vram_footprint(profile) -> dict`

```
kv_bytes   = 2 * block_count * ctx * n_kv_heads * head_dim * bytes_per_el(kv_quant)
             (key_length/value_length override head_dim when present)
weights    = model file size (sum of shards) + mmproj file size
overhead   = max(1 GiB, 10% of weights)          # compute buffers, fragmentation
total      = weights + kv_bytes + overhead
```

`bytes_per_el`: f16/bf16 = 2.0, q8_0 = 1.0625, q5_1 = 0.75, q4_0 = 0.5625
(llama.cpp block sizes). Unknown quant → 2.0 (conservative).

Fallback when GGUF parse fails: `kv_bytes = ctx * 96KiB` (heuristic sized on
~30B dense models) and the result is marked `"estimate_quality": "heuristic"`
vs `"exact"` so callers can hedge their output.

The estimate is computed on demand, not stored in the profile (files change,
ctx changes on edit; recomputing is cheap and can't go stale).

### GPU inventory: `gpu_inventory() -> list`

One function isolates the platform probe:
`xpu-smi discovery -j` per device → `{sycl_device, total_bytes, free_bytes,
name}`. The xpu-smi index → SYCL index mapping is established by matching
total memory sizes against `llama-server --list-devices` output once, then
cached in `defaults.json` (`"gpu_map"`); a `--remap` flag on `modelctl place`
refreshes it. Returns `[]` if xpu-smi is missing (all VRAM features then
degrade to warnings, never errors).

### Placement rule

New defaults.json keys: `vram_limit_pct` (default 90), `primary_gpu`
(default: the SYCL device with the most VRAM).

```
budget = primary.total_bytes * vram_limit_pct / 100
if estimate <= budget:            device = primary, no split
elif estimate <= combined budget: split_mode = layer,
                                  tensor_split = ratio of card totals (32:12 -> "8,3")
else:                             keep combined-split placement, warn loudly
```

### Surfacing

- **`modelctl place [name] [--apply] [--remap]`** — new subcommand. Without
  `--apply`: prints each profile's estimate, current placement, and
  recommended placement. With `--apply`: rewrites placement in the profile(s),
  regenerates artifacts, syncs. Never touches profiles without `--apply`.
- **`cmd_pull` / `prompt_config`**: the device/split prompts default to the
  computed recommendation (with the estimate shown inline) instead of the
  static defaults. The user can still override anything.

## Component B: VRAM-guarded explicit loads

`modelctl router load <name>` gains a pre-check:

1. Estimate the profile's footprint (Component A).
2. Read live `free_bytes` for its target GPU(s) from `gpu_inventory()`.
3. Fits → proceed as today.
4. Doesn't fit → print the shortfall, the currently loaded models (from
   `router_status()`) with their estimates, and abort with exit 1.
   - `--evict`: unload loaded modelctl-profile models (largest-estimate
     first) until the target fits, then load.
   - `--force`: skip the check entirely.
5. xpu-smi unavailable or estimate heuristic-quality → warn, proceed
   (guard degrades open, never blocks on missing tooling).

## Component C: Observability

### Preset `[*]` global section

`sync_router_preset()` emits a `version = 1` header and a `[*]` section
containing `metrics = true` (enables per-instance Prometheus metrics, which
the router forwards at `GET /metrics?model=<name>`). Shared per-instance
defaults belong here later; only `metrics` for now.

### `modelctl router stats`

For each **loaded** model (from `/v1/models`), fetch
`/metrics?model=<urlencoded name>` and render one row:

| column | source metric |
|---|---|
| gen tok/s (avg) | `n_tokens_predicted_total / t_tokens_generation_total` |
| prompt tok/s (avg) | `n_prompt_tokens_processed_total / t_prompt_processing_total` |
| requests | `n_requests_processed_total` |
| KV cache used | `kv_cache_usage_ratio` (as %) |

Plus a per-GPU footer: `VRAM used/total` from `gpu_inventory()`. Metric
names are verified against the actual `/metrics` output during
implementation; missing metrics render as `?` rather than failing the row.

### `router status` additions

- Per-GPU `VRAM used/total free` footer (same helper as stats).
- Per-model estimated footprint column (`~18.2GB` / `~?` for non-profile
  models).
- `failed` rows get a hint: `check: journalctl --user -u llama-router` .

### Load timing

`cmd_router_load` polls `/v1/models` after the load POST until the model
reaches `loaded` (or `failed` / 300s timeout) and prints the elapsed
wall-clock time. No history persistence (YAGNI — add if trends matter later).

## Error handling summary

| failure | behavior |
|---|---|
| xpu-smi missing/erroring | placement uses static defaults; guard warns + proceeds; status/stats omit VRAM lines |
| GGUF parse failure | heuristic estimate, marked as such in output |
| /metrics fetch failure per model | `?` row values, other models unaffected |
| router unreachable | existing RuntimeError path, unchanged |

## Testing

Unit tests (unittest, matching existing style — mock subprocess/urllib/HF):

- `read_gguf_kv_metadata`: crafted minimal GGUF fixtures (valid, truncated,
  wrong magic).
- `estimate_vram_footprint`: exact-path math for a known config; heuristic
  fallback; sharded file size summing; mmproj inclusion.
- Placement rule: fits-primary / needs-split / doesn't-fit boundaries around
  `vram_limit_pct`.
- `cmd_router_load` guard: fits, blocked, `--evict` ordering
  (largest-first), `--force`, xpu-smi-missing degrade-open.
- Preset writer: `[*]` section + `version = 1` present, existing sections
  unchanged (update existing sync tests).
- `router stats`: metrics parsing from a canned Prometheus text body;
  missing-metric rendering.

Manual verification: `modelctl place` against the real profile set;
`router stats` against the live router with one loaded model.

## Follow-on

**Project 2 (separate spec, llama.cpp repo):** router-side LRU
evict-and-retry on failed/blocked load, so API load requests always take
precedence. Once live, `--evict` on explicit loads becomes mostly moot but
harmless.
