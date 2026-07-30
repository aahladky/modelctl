"""Profile schema, migration, and validation for modelctl.

Profiles are plain JSON at ~/.local/share/modelctl/profiles/<name>.json.
This module owns the schema definition, normalization (v1→v2 migration),
validation, and persistence contract.

Public API:
    normalize_profile(raw: dict) -> dict
    validate_profile(profile: dict) -> list[ValidationMessage]
    profile_schema_version(profile: dict) -> int
    migrate_profile_dry_run(profile: dict) -> dict
"""
import json
from pathlib import Path

from modelctl_errors import ValidationMessage


# Canonical profile schema (v2):
#   schema: 2
#   name: str
#   repo_id: str (Hugging Face repo, or empty for local)
#   file: str (selected GGUF filename in repo)
#   model_path: str (local absolute path)
#   mmproj_path: str (optional multimodal projector)
#   mtp_path: str (optional multi-token prediction draft)
#   backend: "llama-cpp" | "ovms"
#   binary: str (optional pinned server binary)
#   config: {device, split_mode, tensor_split, ctx, cache_type_k,
#            cache_type_v, flash_attn, ttl, mtp, fit, extra}
#   env: ["K=V", ...]
#   enabled: bool
#   runtime (optional): {mode, objective, pinned_plan_id, allow_fallback,
#            allow_untested, minimum_context, maximum_cpu_bytes,
#            maximum_storage_tier, disabled_plan_ids, health_timeout}
#   moe_cache: {mode, gpu, ram, storage, prefill, decode, prefetch}
#   profile_version: 2

# The schema normalize_profile() writes. The integration manifest records
# this number; modelctl_diagnostics reports a mismatch when the two drift,
# so bumping the schema without updating the manifest is visible rather
# than silent.
PROFILE_SCHEMA_VERSION = 2

_DEFAULT_MOE_CACHE = {
    "mode": "off",
    "gpu": {
        "budgets_bytes": {},
        "policy": "slru",
        "probationary_fraction": 0.2,
        "admission_misses": 2,
        "pin_shared_experts": True,
        "pin_static_experts": [],
    },
    "ram": {
        "mode": "page_cache",
        "budget_bytes": 0,
        "mlock_hot_set": False,
    },
    "storage": {
        "mode": "mmap",
        "readahead": "adaptive",
        "release_cold_pages": False,
    },
    "prefill": {
        "admit_to_gpu_cache": False,
        "protect_decode_entries": True,
    },
    "decode": {
        "admit_to_gpu_cache": True,
        # "gpu" = weight transfer on cache miss (works on any cache-capable
        # fork).  "cpu" = hybrid CPU miss execution, which no shipping
        # backend implements yet; opt in explicitly, never by default.
        "miss_execution": "gpu",
    },
    "prefetch": {
        "enabled": False,
        "method": "none",
        "max_overfetch_ratio": 1.5,
    },
}

_DEFAULT_CONFIG = {
    "device": "",
    "split_mode": "",
    "tensor_split": "",
    "ctx": 8192,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "flash_attn": "auto",
    "ttl": 3600,
    "mtp": "off",
    "fit": "on",
    "extra": "",
}

_VALID_BACKENDS = {"llama-cpp", "ovms"}
_VALID_CACHE_MODES = {"off", "auto", "manual"}
_VALID_OBJECTIVES = {
    "balanced", "fastest_generation", "fastest_prompt",
    "largest_context", "fastest_load", "lowest_ram", "fixed",
}
_VALID_CACHE_POLICIES = {"slru", "lru", "lfu", "fifo"}
_VALID_STORAGE_MODES = {"none", "mmap", "mlock", "page_cache"}
_VALID_MISS_EXECUTION = {"cpu", "gpu"}
_VALID_FLASH_ATTN = {"on", "off", "auto"}
_VALID_FIT = {"on", "off"}


def _deep_copy_dict(d):
    out = {}
    for k, v in d.items():
        out[k] = _deep_copy_dict(v) if isinstance(v, dict) else v
    return out


def profile_schema_version(profile: dict) -> int:
    """Return the profile's schema version. 0 = no version field (v1)."""
    return int(profile.get("profile_version", 0))


def normalize_profile(raw: dict) -> dict:
    """Normalize a loaded profile into the canonical v2 schema.

    Idempotent, non-destructive: unknown fields are preserved, legacy
    ones are mapped (kv_quant → cache_type_k/v, string ctx/ttl → int),
    and moe_cache defaults are filled for profiles that lack them.
    """
    p = dict(raw)

    # Ensure schema version.
    if "profile_version" not in p:
        p["profile_version"] = PROFILE_SCHEMA_VERSION

    p.setdefault("name", "")
    p.setdefault("repo_id", "")
    p.setdefault("file", "")
    p.setdefault("model_path", "")
    p.setdefault("mmproj_path", "")
    p.setdefault("mtp_path", "")
    p.setdefault("enabled", True)
    p.setdefault("backend", "llama-cpp")
    p.setdefault("binary", "")
    p.setdefault("env", [])

    # Config: fill defaults for missing keys.
    cfg = p.setdefault("config", {})

    # Legacy migration: kv_quant → cache_type_k/v (BEFORE defaults fill
    # cache_type_k/v, so the legacy value wins).
    legacy = cfg.get("kv_quant")
    if legacy and not cfg.get("cache_type_k"):
        cfg["cache_type_k"] = legacy
    if legacy and not cfg.get("cache_type_v"):
        cfg["cache_type_v"] = legacy

    for k, v in _DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)

    # Type coercion: string ctx/ttl → int
    for key in ("ctx", "ttl"):
        if key in cfg:
            try:
                cfg[key] = int(cfg[key])
            except (TypeError, ValueError):
                pass

    # moe_cache: fill defaults for profiles that predate the feature.
    if "moe_cache" not in p:
        p["moe_cache"] = _deep_copy_dict(_DEFAULT_MOE_CACHE)
    else:
        mc = p["moe_cache"]
        for section, defaults in _DEFAULT_MOE_CACHE.items():
            if section not in mc:
                mc[section] = _deep_copy_dict(defaults)
            elif isinstance(defaults, dict):
                for k, v in defaults.items():
                    mc[section].setdefault(k, v)

    # Runtime: ensure int types for numeric fields.
    rt = p.get("runtime")
    if rt:
        for key in ("minimum_context", "maximum_cpu_bytes",
                     "maximum_storage_tier", "health_timeout"):
            if key in rt and rt[key] is not None:
                try:
                    rt[key] = int(rt[key])
                except (TypeError, ValueError):
                    pass

    return p


def validate_profile(profile: dict) -> list[ValidationMessage]:
    """Validate a normalized profile. Returns a list of messages.

    Errors prevent launch. Warnings are advisory.
    """
    messages = []
    name = profile.get("name", "")
    if not name:
        messages.append(ValidationMessage(
            code="invalid_profile",
            severity="error",
            summary="Profile has no name",
            field="name",
        ))

    backend = profile.get("backend", "llama-cpp")
    if backend not in _VALID_BACKENDS:
        messages.append(ValidationMessage(
            code="invalid_backend",
            severity="error",
            summary=f"Unknown backend '{backend}'",
            detail=f"Valid backends: {', '.join(sorted(_VALID_BACKENDS))}",
            field="backend",
        ))

    model_path = profile.get("model_path", "")
    if model_path and not Path(model_path).exists():
        messages.append(ValidationMessage(
            code="model_file_missing",
            severity="warning",
            summary=f"Model file not found: {model_path}",
            field="model_path",
        ))

    cfg = profile.get("config", {})
    ctx = cfg.get("ctx", 0)
    try:
        ctx = int(ctx)
    except (TypeError, ValueError):
        ctx = 0
    if ctx <= 0:
        messages.append(ValidationMessage(
            code="invalid_config",
            severity="error",
            summary=f"Invalid context length: {ctx}",
            field="config.ctx",
        ))

    flash_attn = cfg.get("flash_attn", "auto")
    if flash_attn not in _VALID_FLASH_ATTN:
        messages.append(ValidationMessage(
            code="invalid_config",
            severity="warning",
            summary=f"Unknown flash_attn value: {flash_attn}",
            field="config.flash_attn",
        ))

    fit = cfg.get("fit", "on")
    if fit not in _VALID_FIT:
        messages.append(ValidationMessage(
            code="invalid_config",
            severity="warning",
            summary=f"Unknown fit value: {fit}",
            field="config.fit",
        ))

    # moe_cache validation
    mc = profile.get("moe_cache", {})
    mc_mode = mc.get("mode", "off")
    if mc_mode not in _VALID_CACHE_MODES:
        messages.append(ValidationMessage(
            code="invalid_cache_mode",
            severity="error",
            summary=f"Unknown moe_cache mode: {mc_mode}",
            detail=f"Valid modes: {', '.join(sorted(_VALID_CACHE_MODES))}",
            field="moe_cache.mode",
        ))

    gpu_section = mc.get("gpu", {})
    policy = gpu_section.get("policy", "slru")
    if policy not in _VALID_CACHE_POLICIES:
        messages.append(ValidationMessage(
            code="invalid_cache_policy",
            severity="warning",
            summary=f"Unknown cache policy: {policy}",
            field="moe_cache.gpu.policy",
        ))

    admission = gpu_section.get("admission_misses", 2)
    try:
        admission = int(admission)
    except (TypeError, ValueError):
        admission = -1
    if admission < 0:
        messages.append(ValidationMessage(
            code="invalid_cache_config",
            severity="warning",
            summary=f"Invalid admission_misses: {admission}",
            field="moe_cache.gpu.admission_misses",
        ))

    # Runtime validation
    rt = profile.get("runtime")
    if rt:
        obj = rt.get("objective", "balanced")
        if obj not in _VALID_OBJECTIVES:
            messages.append(ValidationMessage(
                code="invalid_objective",
                severity="error",
                summary=f"Unknown runtime objective: {obj}",
                detail=f"Valid: {', '.join(sorted(_VALID_OBJECTIVES))}",
                field="runtime.objective",
            ))

    if not messages:
        messages.append(ValidationMessage(
            code="profile_valid",
            severity="info",
            summary="Profile is valid",
        ))

    return messages


def migrate_profile_dry_run(profile: dict) -> dict:
    """Show what migration would do without saving.

    Returns {"original": {...}, "migrated": {...}, "changes": [...]}
    """
    original = _deep_copy_dict(profile)
    migrated = normalize_profile(profile)

    changes = []
    orig_ver = profile_schema_version(original)
    new_ver = profile_schema_version(migrated)
    if orig_ver != new_ver:
        changes.append(f"profile_version: {orig_ver} → {new_ver}")

    if "moe_cache" not in original and "moe_cache" in migrated:
        changes.append("moe_cache section added (default: off)")

    orig_cfg = original.get("config", {})
    new_cfg = migrated.get("config", {})
    if "kv_quant" in orig_cfg and "cache_type_k" not in orig_cfg:
        changes.append(f"kv_quant → cache_type_k/v = {new_cfg.get('cache_type_k')}")

    return {
        "original": original,
        "migrated": migrated,
        "changes": changes,
    }
