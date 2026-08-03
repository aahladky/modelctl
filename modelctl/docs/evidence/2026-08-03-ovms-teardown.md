# OVMS teardown archive — 2026-08-03

OVMS (OpenVINO Model Server) support was removed from modelctl on
2026-08-03 by owner decision. This file preserves the four live OVMS
profiles exactly as they were at teardown, so the work is recoverable.
The full implementation is at git tag `ovms-final` (last commit that
contains the backend code, CLI commands, and tests).

OVMS service + weights under ~/services were NOT touched (owner's lane).
Router config regenerated without the profiles; takes effect at the
next llama-swap restart.

## Profiles at teardown

### big-gemma

```json
{
  "name": "big-gemma",
  "backend": "ovms",
  "repo_id": "OpenVINO/gemma-4-31B-it-int4-ov",
  "file": "big-gemma",
  "model_path": null,
  "mmproj_path": null,
  "mtp_path": null,
  "config": {
    "target_device": "GPU.0",
    "task": "text_generation",
    "cache_size": 8,
    "tool_parser": "hermes3",
    "ttl": 1800
  },
  "env": [],
  "enabled": true,
  "artifacts_dir": "/home/aaron/.local/share/modelctl/profiles/big-gemma"
}
```

### big-qwen

```json
{
  "name": "big-qwen",
  "backend": "ovms",
  "repo_id": "OpenVINO/Qwen3.6-27B-int4-ov",
  "file": "big-qwen",
  "model_path": null,
  "mmproj_path": null,
  "mtp_path": null,
  "config": {
    "target_device": "GPU.0",
    "task": "text_generation",
    "cache_size": 6,
    "tool_parser": "hermes3",
    "reasoning_parser": "qwen3",
    "ttl": 1800
  },
  "env": [],
  "enabled": true,
  "artifacts_dir": "/home/aaron/.local/share/modelctl/profiles/big-qwen"
}
```

### big-qwen-moe

```json
{
  "name": "big-qwen-moe",
  "backend": "ovms",
  "repo_id": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
  "file": "big-qwen-moe",
  "model_path": null,
  "mmproj_path": null,
  "mtp_path": null,
  "config": {
    "target_device": "GPU.0",
    "task": "text_generation",
    "cache_size": 8,
    "tool_parser": "hermes3",
    "reasoning_parser": "qwen3",
    "ttl": 1800
  },
  "env": [],
  "enabled": true,
  "artifacts_dir": "/home/aaron/.local/share/modelctl/profiles/big-qwen-moe"
}
```

### ornith-35b-ov

```json
{
  "name": "ornith-35b-ov",
  "backend": "ovms",
  "repo_id": "OpenVINO/Ornith-1.0-35B-int4",
  "file": "ornith-35b-ov",
  "model_path": null,
  "mmproj_path": null,
  "mtp_path": null,
  "config": {
    "target_device": "GPU.0",
    "task": "text_generation",
    "cache_size": null,
    "tool_parser": "hermes3",
    "reasoning_parser": "qwen3",
    "ttl": 1800,
    "device": "",
    "split_mode": "",
    "tensor_split": "",
    "ctx": 8192,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "flash_attn": "auto",
    "mtp": "off",
    "fit": "off",
    "extra": ""
  },
  "env": [],
  "enabled": false,
  "artifacts_dir": "/home/aaron/.local/share/modelctl/profiles/ornith-35b-ov",
  "profile_version": 2,
  "binary": "",
  "moe_cache": {
    "mode": "off",
    "gpu": {
      "budgets_bytes": {},
      "policy": "slru",
      "probationary_fraction": 0.2,
      "admission_misses": 2,
      "pin_shared_experts": true,
      "pin_static_experts": []
    },
    "ram": {
      "mode": "page_cache",
      "budget_bytes": 0,
      "mlock_hot_set": false
    },
    "storage": {
      "mode": "mmap",
      "readahead": "adaptive",
      "release_cold_pages": false
    },
    "prefill": {
      "admit_to_gpu_cache": false,
      "protect_decode_entries": true
    },
    "decode": {
      "admit_to_gpu_cache": true,
      "miss_execution": "gpu"
    },
    "prefetch": {
      "enabled": false,
      "method": "none",
      "max_overfetch_ratio": 1.5
    }
  }
}
```
