# Backend Capability Schema

## Overview

The `--modelctl-capabilities` probe returns a JSON object describing
what the llama-server binary supports. **Schema 3 is the canonical
format** (`modelctl_capabilities.CAPABILITY_SCHEMA_VERSION`). Schema 0
(unsupported probe), 1 (early fork), and 2 are automatically normalized
to it by `normalize_capabilities()`; features an older schema cannot
express normalize to `false`, never to an assumption.

Schema 3 adds `moe_cache_per_device_budgets`: `--moe-cache-bytes`
accepts a device map (`SYCL0=8589934592,SYCL1=2147483648`) as well as a
single uniform value. This matters to placement, not just the CLI --
see the reservation rule below.

The probe runs under the actual launch environment (modelctl passes
it), so environment-sensitive constraints reflect the values the
runtime will really use.

## Schema 2 Format

The example below is the verbatim output of
`./build-sycl/bin/llama-server --modelctl-capabilities` on this
machine (2026-07-31; source of truth: the handler in `common/arg.cpp`;
`test_modelctl_capabilities.py` pins the parsing). Probe a real binary
rather than trusting this snapshot.

```json
{
  "schema": 2,
  "backend": "llama.cpp",
  "build": {
    "commit": "b9ce49c29",
    "number": 10206,
    "compiler": "IntelLLVM 2026.1.0",
    "target": "Linux x86_64",
    "build_type": "Release",
    "backends": ["SYCL","CPU"],
    "dynamic_backends": false
  },
  "devices": [
    {"type": "CPU", "name": "CPU", "index": 0, "features": {"moe_weight_transfer_cache": false}},
    {"type": "SYCL", "name": "SYCL0", "index": 1, "features": {"moe_weight_transfer_cache": true}},
    {"type": "SYCL", "name": "SYCL1", "index": 2, "features": {"moe_weight_transfer_cache": true}}
  ],
  "features": {
    "moe_weight_transfer_cache": true,
    "moe_hybrid_cpu_miss": true,
    "moe_cache_metrics": true,
    "moe_cache_prefill_policy": true,
    "moe_cache_reset": true,
    "moe_cache_prefetch": false,
    "moe_offload_threshold_control": true
  },
  "constraints": {
    "moe_cache_backend": "SYCL",
    "moe_cache_min_batch": 32,
    "moe_cache_supported_projections": ["gate", "up", "down"],
    "moe_hybrid_supported_archs": ["any"],
    "moe_hybrid_supported_quant": ["any_with_dequantizer"],
    "moe_hybrid_can_overlap": true
  },
  "cli": {
    "moe_cache_bytes": "--moe-cache-bytes",
    "moe_cache_policy": "--moe-cache-policy",
    "moe_cache_admission": "--moe-cache-admission-misses",
    "moe_cache_prefill": "--moe-cache-prefill-admission",
    "moe_hybrid_mode": "--moe-hybrid-mode"
  }
}
```

## Feature Names

| Feature | Meaning | Status in the pinned fork |
|---|---|---|
| `moe_weight_transfer_cache` | D2D-on-hit / H2D-on-miss expert weight cache | Implemented |
| `moe_hybrid_cpu_miss` | True GPU-hit / CPU-miss execution | Implemented; opt-in, never auto-selected (see [hybrid-moe.md](hybrid-moe.md)) |
| `moe_cache_metrics` | Prometheus /metrics and stats JSON | Implemented |
| `moe_cache_prefill_policy` | Prefill/decode phase admission control | Implemented |
| `moe_cache_reset` | Cache reset via API | Implemented |
| `moe_offload_threshold_control` | `GGML_OP_OFFLOAD_MOE_MIN_BATCH` (per-op-type offload threshold) | Implemented |
| `moe_cache_per_device_budgets` | `--moe-cache-bytes` accepts a per-device map (schema 3) | Implemented |
| `moe_cache_prefetch` | Expert prefetching | Not implemented; forced `false` by normalization |

**Fail-closed rule**: a feature a binary does not affirmatively report
is treated as absent, and features known to be unimplemented
(`moe_cache_prefetch`) are forced false regardless of what the binary
claims. Schema semantics (what a field means) are fixed here; which
features are true is the binary's report, not this document's.

## Cache budget reservation

The budget contract decides how placement reserves VRAM, and emitting
one contract's flags while reserving the other's is how the runtime
cache collides with statically placed experts:

- **`moe_cache_per_device_budgets: true`** (schema 3+): the profile's
  `moe_cache.gpu.budgets_bytes` map is passed through literally. Each
  named device is reserved its own budget; **a device the map does not
  name gets no cache**, and none is reserved for it.
- **`false`** (schema 1/2): the runtime has ONE global budget applied
  to every device that creates a cache. The map collapses to its
  **maximum**, and that figure is reserved on every participating
  device. Summing, or sending a smaller device's figure, would hand
  some card more cache than was reserved there.

If a budget does not fit its device's usable VRAM, the planner
disables the cache for that plan rather than reserving half of it --
a partial reservation would diverge from what the server allocates.

## Constraint semantics

- `moe_cache_min_batch` is a **reported, environment-dependent value**,
  not a constant: the probe honors `GGML_OP_OFFLOAD_MOE_MIN_BATCH` the
  same way the runtime will, so it is 32 only on a default
  environment. Always read it from the probe response for the actual
  launch environment.
- `moe_hybrid_supported_quant: ["any_with_dequantizer"]` means the CPU
  tier handles every weight type ggml can dequantize; other types are
  never skipped from staging (fail-safe).

## Normalization to the canonical schema

| Schema 1 | Canonical |
|---|---|
| `moe_expert_cache` + `moe_cache_sycl` | `moe_weight_transfer_cache` |
| `moe_cache_prefetch` | forced `false` |
| (absent) | `moe_offload_threshold_control: false` |
| (absent, schema 1/2) | `moe_cache_per_device_budgets: false` |
| `cache_bytes` (cli) | `moe_cache_bytes` |
| `cache_policy` (cli) | `moe_cache_policy` |
| `admission_misses` (cli) | `moe_cache_admission` |
| `prefill_admission` (cli) | `moe_cache_prefill` |

## Schema 0 (Unsupported)

Stock llama.cpp binaries that reject the probe get all features set to
false. The `_probe_status` is `"unsupported"`.

## Consumers

- `modelctl_capabilities.is_cache_capable()` — checks `moe_weight_transfer_cache`
- `modelctl_capabilities.is_weight_transfer_cache_capable()` — same
- `modelctl_capabilities.is_sycl_cache_capable()` — same (SYCL is the only implementation)
- `modelctl_capabilities.supports_hybrid_miss()` — checks `moe_hybrid_cpu_miss`
- `modelctl_capabilities.supports_metrics()` — checks `moe_cache_metrics`
- `modelctl.preflight_moe_cache()` — validates cache config against features
- `modelctl.build_moe_cache_args()` — uses `cli` keys for flag names
- `modelctl_plans.compile_launch_plans()` — filters cache variants
