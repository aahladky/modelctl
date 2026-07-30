"""Profile operations service.

All profile mutations flow through here. Both CLI and web call these
functions. They return structured results, never print or sys.exit.
"""
from dataclasses import dataclass, field
from pathlib import Path

import modelctl
import modelctl_profiles
from modelctl_errors import ValidationMessage


@dataclass
class ProfileResult:
    """Result of a profile operation."""
    ok: bool
    profile: dict | None = None
    messages: list = field(default_factory=list)
    validation: list = field(default_factory=list)


def save_profile(profile: dict, regenerate: bool = True,
                 resync: bool = True) -> ProfileResult:
    """Save a profile, optionally regenerate artifacts and sync backends.

    Validates before saving. Returns ProfileResult with any issues.
    """
    validation = modelctl_profiles.validate_profile(
        modelctl_profiles.normalize_profile(profile))
    errors = [v for v in validation if v.severity == "error"]

    if errors:
        return ProfileResult(
            ok=False, profile=profile,
            messages=[f"validation error: {e.summary}" for e in errors],
            validation=validation,
        )

    normalized = modelctl_profiles.normalize_profile(profile)
    modelctl.save_profile(normalized)

    if regenerate:
        modelctl.generate_artifacts(normalized)

    if resync:
        modelctl.sync_all_backends(restart_router=True, restart_openarc=True)

    return ProfileResult(ok=True, profile=normalized, validation=validation)


def remove_profile(name: str, resync: bool = True) -> ProfileResult:
    """Remove a profile and its artifacts."""
    profile_dir = modelctl.PROFILES_DIR / name
    if not profile_dir.exists() and not (modelctl.PROFILES_DIR / f"{name}.json").exists():
        return ProfileResult(ok=False, messages=[f"profile '{name}' not found"])

    modelctl.remove_profile(name, resync=resync)
    return ProfileResult(ok=True, messages=[f"profile '{name}' removed"])


def update_config(name: str, updates: dict,
                  resync: bool = True, ctx=None) -> ProfileResult:
    """Apply field-level updates to a profile.

    The canonical non-interactive edit path for both CLI and web
    (modelctl_web/mutate.py's submit_edit). `ctx`, if given, is a
    modelctl_web.jobs.JobContext -- update_profile_config's own preflight
    messages get logged through it as they're produced, same as
    submit_edit did before delegating here.
    """
    if ctx:
        ctx.log(f"applying updates to '{name}': {', '.join(sorted(updates))}")
    try:
        profile, messages = modelctl.update_profile_config(
            name, updates, resync=resync)
    except (SystemExit, Exception) as e:
        return ProfileResult(ok=False, messages=[str(e)])
    if ctx:
        for m in messages:
            ctx.log(m)

    validation = modelctl_profiles.validate_profile(
        modelctl_profiles.normalize_profile(profile))
    if ctx:
        for v in validation:
            if v.severity != "info":
                ctx.log(f"{v.severity}: {v.summary}")
    return ProfileResult(
        ok=True, profile=profile, messages=messages, validation=validation)


def update_runtime_policy(name: str, runtime: dict,
                          resync: bool = True) -> ProfileResult:
    """Set or clear a profile's runtime policy section."""
    try:
        profile = modelctl.update_runtime_policy(name, runtime, resync=resync)
    except (SystemExit, Exception) as e:
        return ProfileResult(ok=False, messages=[str(e)])
    return ProfileResult(ok=True, profile=profile)


def list_profiles() -> list[dict]:
    """List all profiles with summary info."""
    profiles = []
    for p in sorted(modelctl.PROFILES_DIR.glob("*.json")):
        try:
            profile = modelctl.load_profile(p.stem)
            profiles.append({
                "name": profile["name"],
                "backend": profile.get("backend", "llama-cpp"),
                "model_path": profile.get("model_path", ""),
                "enabled": profile.get("enabled", True),
                "runtime_mode": profile.get("runtime", {}).get("mode", "fixed"),
                "moe_cache_mode": profile.get("moe_cache", {}).get("mode", "off"),
            })
        except Exception:
            pass
    return profiles
