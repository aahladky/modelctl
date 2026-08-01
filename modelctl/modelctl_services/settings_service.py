"""Settings service: defaults, and the diagnostics the console reports.

`modelctl defaults` (CLI) and the settings page (web)
both mutated the defaults file directly; this is the one implementation,
and it validates rather than silently discarding a bad value the way the
route's try/except ValueError did.
"""
import modelctl

from .result import ServiceResult

# Keys the console is allowed to write, and how to coerce them. Anything
# outside this map is rejected rather than stored, so a typo'd field name
# cannot quietly become a permanent entry in defaults.json.
_STRING_KEYS = ("device", "split_mode", "tensor_split", "cache_type_k",
                "cache_type_v", "flash_attn", "mtp", "primary_gpu")
_INT_KEYS = ("ctx", "ttl", "vram_limit_pct")

_BOUNDS = {
    "ctx": (512, 1 << 22),
    "ttl": (0, 86400 * 7),
    "vram_limit_pct": (10, 100),
}


def load_defaults() -> ServiceResult:
    return ServiceResult(data={"defaults": modelctl.load_defaults()})


def update_defaults(updates: dict) -> ServiceResult:
    """Validate and persist profile defaults.

    Rejected values are reported as warnings and left unchanged, rather
    than being dropped silently -- a context length that quietly refused
    to save is worse than one that says why.
    """
    defaults = modelctl.load_defaults()
    result = ServiceResult()
    applied = {}

    for key, raw in (updates or {}).items():
        if key in _STRING_KEYS:
            applied[key] = str(raw)
        elif key in _INT_KEYS:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                result.add_warning(f"{key}: '{raw}' is not a number — unchanged")
                continue
            low, high = _BOUNDS.get(key, (None, None))
            if low is not None and not (low <= value <= high):
                result.add_warning(
                    f"{key}: {value} is outside {low}–{high} — unchanged")
                continue
            applied[key] = value
        else:
            result.add_warning(f"{key}: not a settable default — ignored")

    if not applied:
        result.add_message("nothing changed")
        return result

    try:
        # Re-read and write under the state lock so a concurrent writer
        # (CLI `modelctl defaults`, another request) can't be dropped by
        # this read-modify-write.
        import modelctl_fsutil
        with modelctl_fsutil.state_lock():
            defaults = modelctl.load_defaults()
            defaults.update(applied)
            modelctl.save_defaults(defaults)
    except Exception as e:
        return ServiceResult.failure(f"could not save defaults: {e}",
                                     warnings=result.warnings)

    result.data["defaults"] = defaults
    result.data["applied"] = applied
    result.add_message(f"saved {len(applied)} default(s)")
    return result.changed("settings:defaults")


def diagnostics() -> ServiceResult:
    """The full diagnostic picture the settings page and `doctor` show."""
    import modelctl_diagnostics
    import modelctl_setup

    manifest = modelctl_diagnostics.manifest_status()
    setup = modelctl_setup.probe_setup(probe_backend=True)
    result = ServiceResult(
        ok=setup.ready and manifest.ok,
        data={
            "manifest": {
                "present": manifest.present, "ok": manifest.ok,
                "path": manifest.path, "error": manifest.error,
                "content": manifest.manifest,
                "modelctl_commit": manifest.modelctl_commit,
                "validated_modelctl_commit": manifest.validated_modelctl_commit,
                "validated_llama_commit": manifest.validated_llama_commit,
                "upstream_base": manifest.upstream_base,
                "validation_report": manifest.validation_report,
                "newer_than_validated": manifest.newer_than_validated,
                "submodule_pinned": manifest.submodule_pinned,
                "submodule_checked_out": manifest.submodule_checked_out,
                "working_tree_dirty": manifest.working_tree_dirty,
            },
            "capabilities": modelctl_diagnostics.capability_report(),
            "environment": modelctl_diagnostics.environment_report(),
            "ready": setup.ready,
        })
    result.warnings.extend(manifest.mismatches)
    result.warnings.extend(c.detail for c in setup.warnings)
    result.messages.extend(c.detail for c in setup.blocking)
    return result


def support_bundle(path) -> ServiceResult:
    import modelctl_diagnostics
    try:
        written = modelctl_diagnostics.write_support_bundle(path)
    except Exception as e:
        return ServiceResult.failure(f"could not write support bundle: {e}")
    return ServiceResult(messages=[f"wrote {written}"],
                         data={"path": str(written)})
