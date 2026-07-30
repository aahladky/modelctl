# Backend Capability Schema v2

## Overview

The `--modelctl-capabilities` probe returns a JSON object describing
what the llama-server binary supports. Schema 2 is the canonical
format. Schema 0 (unsupported probe) and schema 1 (early fork) are
automatically normalized to schema 2 by `normalize_capabilities()`.

## Schema 2 Format

```json
{
  "schema": 2,
  "backend": "llama.cpp",
  "build": {
    "commit": "f42f2fe4e",
    "compiler": "icpx-2025",
    "dynamic_backends": true
  },
  "devices": [
    {
      "type": "SYCL",
      "name": "SYCL0",
      "index": 0,
      "features": {
        "moe_weight_transfer_cache": true
      }
    }
  ],
  "features": {
    "moe_weight_transfer_cache": true,
    "moe_hybrid_cpu_miss": false,
    "moe_cache_metrics": true,
    "moe_cache_prefill_policy": true,
    "moe_cache_reset": true,
    "moe_cache_prefetch": false
  },
  "constraints": {
    "moe_cache_backend": "SYCL",
    "moe_cache_min_batch": 32,
    "moe_cache_supported_projections": ["gate", "up", "down"]
  },
  "cli": {
    "moe_cache_bytes": "--moe-cache-bytes",
    "moe_cache_policy": "--moe-cache-policy",
    "moe_cache_admission": "--moe-cache-admission-misses",
    "moe_cache_prefill": "--moe-cache-prefill-admission"
  }
}
```

## Feature Names

| Feature | Meaning | Status |
|---|---|---|
| `moe_weight_transfer_cache` | D2D-on-hit / H2D-on-miss expert weight cache | Implemented |
| `moe_hybrid_cpu_miss` | True GPU-hit / CPU-miss execution with output merge | Phase 7 |
| `moe_cache_metrics` | Prometheus /metrics and stats JSON | Implemented |
| `moe_cache_prefill_policy` | Prefill/decode phase admission control | Implemented |
| `moe_cache_reset` | Cache reset via API | Implemented |
| `moe_cache_prefetch` | Expert prefetching | Phase 9 |

**Fail-closed rule**: Even if a backend claims `moe_hybrid_cpu_miss` or
`moe_cache_prefetch`, `normalize_capabilities()` forces them false
until the features are actually implemented.

## Schema 1 → 2 Normalization

| Schema 1 | Schema 2 |
|---|---|
| `moe_expert_cache` + `moe_cache_sycl` | `moe_weight_transfer_cache` |
| `moe_hybrid_cpu_miss` | forced `false` |
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
