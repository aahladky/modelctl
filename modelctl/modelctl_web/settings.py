"""Data assembly + validation for the /v2 settings page (console phase 3).

Same contract as hub.py: plain JSON-serializable data, routes stay thin,
a dead probe yields its section's empty shape with an error string rather
than an exception out of the endpoint.

Scope rule from the phase-3 order: this module exposes only settings that
already exist and already have a backing implementation. Everything
writable here routes through `settings_service.update_defaults` (profile
defaults) or `modelctl_hardware.load/save_settings` (device policy) --
the same code paths the CLI uses -- plus token rotation, which is a write
to the token file the auth layer already reads. Nothing here invents a
settings store. Settings the system has but exposes no API for (bind
address, state paths) are reported read-only, labelled with where they
actually come from.
"""
import os
import secrets

import modelctl

from modelctl_services import settings_service


# ---- profile defaults ---------------------------------------------------

# Form metadata for the typed defaults editor. The int bounds are read
# from settings_service, not restated, so an input's min/max is the same
# number the write validates against. `kind` picks the control; `choices`
# turns a field the code only reads a fixed vocabulary from into a select
# instead of a free-text box that can be typo'd into silence.
_CHOICES = {
    "split_mode": ["layer", "row", "none"],
    "cache_type_k": ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"],
    "cache_type_v": ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"],
    "flash_attn": ["auto", "on", "off"],
    "mtp": ["off", "on"],
}

_UNITS = {"ctx": "tokens", "ttl": "seconds", "vram_limit_pct": "%"}

_ORDER = ("device", "primary_gpu", "split_mode", "tensor_split", "ctx",
          "ttl", "cache_type_k", "cache_type_v", "flash_attn", "mtp",
          "vram_limit_pct")


def default_fields():
    """Descriptors for every settable default, in form order.

    Anything settings_service accepts appears here; the assertion that
    the two agree is a test, so a key added to the service can't quietly
    become uneditable in the console.
    """
    fields = []
    for key in _ORDER:
        if key in settings_service.INT_KEYS:
            low, high = settings_service.BOUNDS.get(key, (None, None))
            fields.append({"name": key, "kind": "int", "min": low,
                           "max": high, "unit": _UNITS.get(key, ""),
                           "choices": []})
        elif key in settings_service.STRING_KEYS:
            fields.append({"name": key,
                           "kind": "choice" if key in _CHOICES else "text",
                           "min": None, "max": None,
                           "unit": _UNITS.get(key, ""),
                           "choices": _CHOICES.get(key, [])})
    return fields


def shadowed_defaults():
    """Fields whose value comes from the environment, not the saved file.

    modelctl.load_defaults() gives MODELCTL_DEFAULT_<KEY> precedence over
    defaults.json, so saving a shadowed field writes the file and changes
    nothing observable. That is a state the page has to show, not a
    surprise the operator gets to discover.
    """
    out = {}
    for key in _ORDER:
        env_key = f"MODELCTL_DEFAULT_{key.upper()}"
        if os.environ.get(env_key) is not None:
            out[key] = env_key
    return out


def defaults_section():
    try:
        values = modelctl.load_defaults()
    except Exception as e:
        return {"values": {}, "fields": default_fields(), "shadowed": {},
                "error": str(e)}
    return {"values": {k: values.get(k, "") for k in _ORDER},
            "fields": default_fields(), "shadowed": shadowed_defaults(),
            "error": ""}


def update_defaults(updates):
    """Typed write. Returns the service's own warnings verbatim.

    Unknown keys are rejected by the service, so a hand-crafted body
    cannot become a JSON side door around the typed form.
    """
    result = settings_service.update_defaults(updates or {})
    warnings = list(result.warnings)
    # A saved-but-shadowed field is a write that changed the file and
    # nothing else; say so rather than reporting a clean save.
    shadowed = shadowed_defaults()
    for key in result.data.get("applied", {}):
        if key in shadowed:
            warnings.append(
                f"{key}: saved to disk but {shadowed[key]} is set in the "
                "service environment and wins — the running value is "
                "unchanged")
    return {
        "ok": result.ok,
        "applied": result.data.get("applied", {}),
        "values": {k: (result.data.get("defaults") or {}).get(k, "")
                   for k in _ORDER},
        "shadowed": shadowed,
        "warnings": warnings,
        "messages": list(result.messages),
    }


# ---- hardware policy ----------------------------------------------------

# The only role the planner reads is "primary" (it sorts that GPU first);
# every other string is indistinguishable from empty. A select of what
# the code actually understands beats the old free-text box.
DEVICE_ROLES = ["", "primary"]


def hardware_section():
    """Device policy joined with the live snapshot.

    Reserve/enabled/role/bandwidth are user policy; totals, names and
    measured bandwidth are machine facts shown beside them so the number
    being edited has its context inline.
    """
    import modelctl_hardware
    out = {"devices": [], "ram": {}, "storage": [], "roles": DEVICE_ROLES,
           "fingerprint": "", "error": ""}
    try:
        stored = modelctl_hardware.load_settings()
        snap = modelctl_hardware.capture_hardware_snapshot(stored)
    except Exception as e:
        out["error"] = f"hardware probe failed: {e}"
        return out
    dev_settings = stored.get("devices", {}) or {}
    out["fingerprint"] = snap.fingerprint
    for g in snap.gpus:
        ds = dev_settings.get(g.device, {}) or {}
        out["devices"].append({
            "device": g.device,
            "name": g.name,
            "total_bytes": g.total_bytes,
            "free_bytes": g.free_bytes,
            "reserve_bytes": g.reserve_bytes,
            "enabled": bool(g.enabled),
            "role": g.role or "",
            "bandwidth_gbs": g.bandwidth_gbs,
            # Whether this device's bandwidth is an override or the
            # probed value -- the measured/estimated distinction the
            # spec makes visible everywhere else.
            "bandwidth_overridden":
                "memory_bandwidth_gbs_override" in ds,
            "pci_address": g.pci_address,
            "pcie_width": g.pcie_width,
        })
    out["ram"] = {"total_bytes": snap.ram_total_bytes,
                  "available_bytes": snap.ram_available_bytes,
                  "reserve_bytes": snap.ram_reserve_bytes}
    for s in snap.storage:
        out["storage"].append({
            "path": s.path, "kind": s.kind, "mount_point": s.mount_point,
            "filesystem": s.filesystem, "transport": s.transport,
            "allow_mmap": s.allow_mmap, "total_bytes": s.total_bytes,
            "free_bytes": s.free_bytes,
            "measured_sequential_read_bps": s.measured_sequential_read_bps,
            "measured_random_read_bps": s.measured_random_read_bps,
            "measurement_age_seconds": s.measurement_age_seconds,
        })
    return out


def update_hardware(payload):
    """Validate and persist device policy.

    The old form route coerced with `except ValueError: pass`, so a
    mistyped reserve silently kept the old value. Here every rejection
    is a warning the page shows, matching the defaults form.
    """
    import modelctl_fsutil
    import modelctl_hardware

    payload = payload or {}
    warnings = []
    applied = 0

    devices = payload.get("devices")
    if devices is not None and not isinstance(devices, dict):
        return {"ok": False, "applied": 0,
                "warnings": ["devices: expected an object keyed by device"],
                "messages": []}

    def _bytes(raw, label, low=0, high=1 << 44):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            warnings.append(f"{label}: '{raw}' is not a number — unchanged")
            return None
        if not (low <= value <= high):
            fmt = lambda n: f"{n / 2 ** 30:.0f} GiB"
            warnings.append(
                f"{label}: {fmt(value)} is outside {fmt(low)}–{fmt(high)}"
                " — unchanged")
            return None
        return value

    try:
        with modelctl_fsutil.state_lock():
            stored = modelctl_hardware.load_settings()
            dev_settings = stored.setdefault("devices", {})
            snap = modelctl_hardware.capture_hardware_snapshot(stored)
            known = {g.device for g in snap.gpus}

            for dev, spec in (devices or {}).items():
                if dev not in known:
                    warnings.append(f"{dev}: not a device on this machine"
                                    " — ignored")
                    continue
                if not isinstance(spec, dict):
                    warnings.append(f"{dev}: expected an object — ignored")
                    continue
                ds = dev_settings.setdefault(dev, {})
                gpu = snap.gpu_by_device(dev)
                if "reserve_bytes" in spec:
                    ceiling = gpu.total_bytes if gpu else 1 << 44
                    value = _bytes(spec["reserve_bytes"],
                                   f"{dev} reserve", 0, ceiling)
                    if value is not None:
                        ds["reserve_bytes"] = value
                        applied += 1
                if "enabled" in spec:
                    ds["enabled"] = bool(spec["enabled"])
                    applied += 1
                if "role" in spec:
                    role = str(spec["role"] or "")
                    if role not in DEVICE_ROLES:
                        warnings.append(
                            f"{dev} role: '{role}' is not one of"
                            f" {', '.join(r or '(none)' for r in DEVICE_ROLES)}"
                            " — unchanged")
                    else:
                        ds["role"] = role
                        applied += 1
                if "memory_bandwidth_gbs_override" in spec:
                    raw = spec["memory_bandwidth_gbs_override"]
                    if raw in (None, ""):
                        # Clearing an override is a real edit: drop back
                        # to the probed value rather than storing 0.
                        ds.pop("memory_bandwidth_gbs_override", None)
                        applied += 1
                    else:
                        try:
                            value = float(raw)
                        except (TypeError, ValueError):
                            warnings.append(f"{dev} bandwidth: '{raw}' is not"
                                            " a number — unchanged")
                            value = None
                        if value is not None:
                            if not (0 < value <= 10000):
                                warnings.append(
                                    f"{dev} bandwidth: {value} GB/s is outside"
                                    " 0–10000 — unchanged")
                            else:
                                ds["memory_bandwidth_gbs_override"] = value
                                applied += 1

            ram = payload.get("ram")
            if isinstance(ram, dict) and "reserve_bytes" in ram:
                value = _bytes(ram["reserve_bytes"], "RAM reserve", 0,
                               snap.ram_total_bytes or 1 << 44)
                if value is not None:
                    stored.setdefault("ram", {})["reserve_bytes"] = value
                    applied += 1

            if applied:
                modelctl_hardware.save_settings(stored)
    except Exception as e:
        return {"ok": False, "applied": 0,
                "warnings": warnings + [f"could not save hardware policy: {e}"],
                "messages": []}

    messages = [f"saved {applied} hardware setting(s)"] if applied \
        else ["nothing changed"]
    return {"ok": True, "applied": applied, "warnings": warnings,
            "messages": messages}


# ---- access + paths (read-only unless noted) ----------------------------

def access_section(token_path, bind_default="0.0.0.0:9293"):
    """How the console is reached, and where its token lives.

    Bind is process configuration (systemd unit / MODELCTL_WEB_BIND) --
    reported with its source so the page can say why it is not editable
    here, rather than offering a control that could not take effect
    without a restart nobody asked for.
    """
    bind = os.environ.get("MODELCTL_WEB_BIND", "")
    return {
        "bind": bind or bind_default,
        "bind_source": "MODELCTL_WEB_BIND" if bind else "built-in default",
        "bind_editable": False,
        "token_path": str(token_path),
        "token_source": ("MODELCTL_WEB_TOKEN"
                         if os.environ.get("MODELCTL_WEB_TOKEN", "").strip()
                         else "token file"),
        # An env-supplied token is owned by whoever set the variable;
        # rotating the file underneath it would change nothing and read
        # as a no-op failure.
        "token_rotatable": not os.environ.get("MODELCTL_WEB_TOKEN", "").strip(),
        "secure_cookie": os.environ.get("MODELCTL_WEB_SECURE_COOKIE", "") == "1",
    }


def rotate_token(token_path):
    """Write a fresh token and return it.

    The caller re-issues the session cookie with the new value, so the
    operator who rotated stays signed in and every other session is the
    one that gets logged out -- which is the point of rotating.
    """
    if os.environ.get("MODELCTL_WEB_TOKEN", "").strip():
        return {"ok": False, "token": "",
                "error": "token comes from MODELCTL_WEB_TOKEN; rotate it "
                         "where that variable is set (the systemd unit), "
                         "not here"}
    token = secrets.token_hex(16)
    try:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = token_path.with_suffix(".tmp")
        tmp.write_text(token + "\n")
        os.chmod(tmp, 0o600)
        tmp.replace(token_path)
    except OSError as e:
        return {"ok": False, "token": "", "error": f"could not write token: {e}"}
    return {"ok": True, "token": token, "error": ""}


def paths_section():
    """The state paths the install actually uses, and their source."""
    from .jobs import STATE_DIR
    rows = [
        ("models directory", str(modelctl.DEFAULT_MODELS_DIR),
         "MODELCTL_MODELS_DIR"),
        ("profiles directory", str(modelctl.PROFILES_DIR), "MODELCTL_HOME"),
        ("console state", str(STATE_DIR), "MODELCTL_HOME"),
        ("llama-server binary", str(modelctl.LLAMA_SERVER_BIN),
         "resolved from candidates"),
        ("llama-swap config", str(modelctl.LLAMA_SWAP_CONFIG_PATH),
         "MODELCTL_LLAMA_SWAP_CONFIG"),
        ("llama-swap service", str(modelctl.LLAMA_SWAP_SERVICE_NAME),
         "MODELCTL_LLAMA_SWAP_SERVICE"),
        ("llama-swap base URL", str(modelctl.LLAMA_SWAP_BASE_URL),
         "MODELCTL_LLAMA_SWAP_BASE_URL"),
    ]
    return [{"label": label, "value": value, "source": source,
             "overridden": bool(os.environ.get(source, ""))}
            for label, value, source in rows]


# ---- integration + diagnostics ------------------------------------------

def diagnostics_section():
    """Manifest, backend capabilities and environment, degrading per part.

    A failing capability probe must not blank the manifest beside it, so
    each part is tried on its own.
    """
    import modelctl_diagnostics
    out = {"manifest": {}, "capabilities": {}, "environment": {},
           "errors": []}
    try:
        status = modelctl_diagnostics.manifest_status()
        out["manifest"] = {
            "path": status.path,
            "present": status.present,
            "ok": status.ok,
            "error": status.error,
            "modelctl_commit": status.modelctl_commit,
            "validated_modelctl_commit": status.validated_modelctl_commit,
            "validated_llama_commit": status.validated_llama_commit,
            "upstream_base": status.upstream_base,
            "validation_report": status.validation_report,
            "newer_than_validated": status.newer_than_validated,
            "submodule_pinned": status.submodule_pinned,
            "submodule_checked_out": status.submodule_checked_out,
            "working_tree_dirty": status.working_tree_dirty,
            "mismatches": list(status.mismatches),
            "notes": list(status.notes),
        }
    except Exception as e:
        out["errors"].append(f"manifest: {e}")
    try:
        out["capabilities"] = modelctl_diagnostics.capability_report()
    except Exception as e:
        out["errors"].append(f"capabilities: {e}")
    try:
        out["environment"] = modelctl_diagnostics.environment_report()
    except Exception as e:
        out["errors"].append(f"environment: {e}")
    return out


def readiness_section():
    """The first-run checklist the old /setup page owned.

    Folded in here at the phase-3 cutover: it is diagnostics, and the
    four-domain IA has no separate setup domain. `fix_command` is carried
    verbatim -- the remediations are deliberately shell/systemd
    instructions, because that is where these settings actually live.
    """
    import modelctl_setup
    try:
        status = modelctl_setup.probe_setup(probe_backend=True)
    except Exception as e:
        return {"ready": False, "first_run": False, "checks": [],
                "error": f"readiness probe failed: {e}"}
    checks = []
    for c in status.checks:
        checks.append({"id": c.id, "title": c.title, "severity": c.severity,
                       "detail": c.detail, "fix": c.fix,
                       "fix_url": c.fix_url, "fix_command": c.fix_command})
    return {"ready": bool(status.ready), "first_run": bool(status.first_run),
            "checks": checks, "error": ""}


def overview(token_path):
    """Everything the settings page renders on first paint."""
    return {
        "readiness": readiness_section(),
        "defaults": defaults_section(),
        "hardware": hardware_section(),
        "access": access_section(token_path),
        "paths": paths_section(),
        "diagnostics": diagnostics_section(),
    }
