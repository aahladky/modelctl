"""Data assembly for the /v2 model hub and add wizard (console phase 2).

Every function here returns plain JSON-serializable data for the typed
/api/v2 surface; routes in app.py stay thin. Same degrade contract as
telemetry: a dead probe yields its section's empty shape with an error
string, never an exception out of the endpoint.
"""
import copy
import json
import time

import modelctl


# ---- model hub ----------------------------------------------------------

# The typed configure form edits exactly the fields the old profile edit
# form owned, plus `enabled`; anything else in a save body is rejected so
# the API can't become a JSON side door around the typed form.
CONFIG_FIELDS = ["device", "split_mode", "tensor_split", "ctx",
                 "cache_type_k", "cache_type_v", "flash_attn", "ttl",
                 "mtp", "fit", "extra", "binary"]

# The runtime-policy objectives the ranker actually scores differently
# (modelctl_plans._objective_score / _plan_score). The typed form offers
# this list and the endpoint validates against it, so the console cannot
# offer an objective the ranker ignores -- the old form's select listed
# six of these and silently omitted interactive_latency and
# lowest_storage, which were therefore reachable only by hand-editing the
# profile JSON.
RUNTIME_OBJECTIVES = ("balanced", "fastest_generation", "fastest_prompt",
                      "interactive_latency", "largest_context",
                      "fastest_load", "lowest_ram", "lowest_storage")


def _profile_config(profile):
    cfg = profile.get("config", {}) or {}
    return {f: cfg.get(f, "") for f in CONFIG_FIELDS}


def _budgets(profile):
    mc = profile.get("moe_cache") or {}
    raw = ((mc.get("gpu") or {}).get("budgets_bytes")) or {}
    out = {}
    for dev, val in raw.items():
        try:
            out[dev] = int(val)
        except (TypeError, ValueError):
            continue
    return {"mode": mc.get("mode", "off"), "budgets_bytes": out}


def model_detail(profile, runtime_row=None, inventory=None):
    """One profile -> the per-model page's overview payload."""
    rt = runtime_row or {}
    size = None
    if profile.get("model_path"):
        try:
            import modelctl_vram
            size = modelctl_vram.weights_bytes_on_disk(profile["model_path"])
        except Exception:
            size = None
    planning = profile.get("planning") or {}
    return {
        "name": profile.get("name", ""),
        "repo_id": profile.get("repo_id") or "",
        "file": profile.get("file") or "",
        "model_path": profile.get("model_path") or "",
        "backend": profile.get("backend") or "llama-cpp",
        "binary": profile.get("binary") or "",
        "enabled": bool(profile.get("enabled", True)),
        "size_bytes": size,
        "config": _profile_config(profile),
        "moe_cache": _budgets(profile),
        "runtime": {
            "state": rt.get("state") or (
                "stopped" if rt.get("registered") else "unregistered"),
            "state_class": rt.get("state_class") or "",
            "running": bool(rt.get("running")),
            "registered": bool(rt.get("registered")),
            "port": rt.get("port"),
            "pid": rt.get("pid"),
            "started": rt.get("started"),
        },
        "planning": {
            "recorded_at": planning.get("recorded_at") or "",
            "has_stored_inputs": bool(planning.get("inputs")),
        },
        "gpus": [{"device": g["device"], "name": g["name"],
                  "total_bytes": g["total_bytes"]}
                 for g in (inventory or [])],
    }


def _measured_from_observation(ev):
    obs = ev.observation or {}
    if not ev.tested or not obs:
        return None
    out = {k: obs.get(k) for k in
           ("generation_tps", "prompt_tps", "load_seconds", "ttft_seconds",
            "actual_context")
           if obs.get(k) is not None}
    out["cache_state"] = ev.cache_state or ""
    out["measured_at"] = ev.measured_at
    out["stale"] = bool(ev.stale)
    return out


def plan_rows(profile, snapshot=None):
    """Compiled launch plans joined with measurements: the hub's
    "measurements outrank estimates" view. Every row says whether its
    numbers are measured or estimated -- the tag the spec makes
    first-class."""
    import modelctl_evidence
    import modelctl_fleet
    import modelctl_hardware
    import modelctl_launch
    import modelctl_plans
    import modelctl_runtime

    name = profile.get("name", "")
    # Presence gates the fleet plan family; refresh it here (bounded, one
    # probe per stale node) so a plans view is never silently computed
    # against a fleet nobody has looked at inside the TTL.
    try:
        modelctl_fleet.ensure_fresh_presence()
    except Exception:
        pass
    snap = snapshot or modelctl_hardware.capture_hardware_snapshot()
    plans = modelctl_plans.compile_launch_plans(profile, snap)
    rdb = modelctl_runtime.RuntimeDB()
    failures = rdb.failures_for_profile(name)
    try:
        backend = modelctl_launch.resolve_backend(profile)
    except Exception:
        backend = None
    observations = rdb.observations_for_profile(
        name, identity=modelctl_runtime.ObservationIdentity.current(
            snapshot=snap, backend=backend,
            profile_name=profile.get("backend", "llama-cpp")))
    evidence = modelctl_evidence.build_plan_evidence(
        profile, plans, observations, failures, backend=backend)
    rows = []
    for ev in evidence:
        pl = ev.plan
        rows.append({
            "id": pl.id,
            "label": pl.label,
            "source": pl.source,
            "category": ev.category,
            "category_label": ev.category_label,
            "tested": bool(ev.tested),
            "stale": bool(ev.stale),
            "pinned": bool(ev.pinned),
            "disabled": bool(ev.disabled),
            "estimated": dict(pl.estimated or {}),
            "measured": _measured_from_observation(ev),
            "warnings": list(pl.warnings or ()),
            "reason": ev.reason,
            "admission": (pl.decision_data or {}).get("admission"),
        })
    return rows


def history_rows(name, limit=50):
    """Full measurement history from the store (every plan_runs row, not
    the one-bucket observation view), with the bottleneck judgement the
    old history page already makes."""
    import modelctl_runtime
    import modelctl_tune
    rows = []
    for r in modelctl_runtime.RuntimeDB().plan_runs_for(name, limit=limit):
        d = dict(r)
        label, why = modelctl_tune.classify_bottleneck(d)
        rows.append({
            "started_at": d.get("started_at"),
            "plan_id": d.get("plan_id") or "",
            "run_kind": d.get("run_kind") or "",
            "success": bool(d.get("success")),
            "failure_class": d.get("failure_class") or "",
            "generation_tps": d.get("generation_tps"),
            "prompt_tps": d.get("prompt_tps"),
            "load_seconds": d.get("load_seconds"),
            "ttft_seconds": d.get("ttft_seconds"),
            "actual_context": d.get("actual_context"),
            "cache_state": d.get("cache_state") or "",
            "bottleneck": label,
            "bottleneck_why": why,
        })
    return rows


def log_tail(profile, swap_client=None, lines=120):
    """Best log tail available for a model: llama-swap's per-model log via
    its API first (covers the running case), then the artifact-dir files
    llama-swap and the smoke test leave behind. Returns the source with
    the text so the page can say where the log came from."""
    name = profile.get("name", "")
    if swap_client is not None:
        try:
            data = swap_client.logs(model_id=name)
            text = data if isinstance(data, str) else json.dumps(data, indent=2)
            if text.strip():
                return {"source": "llama-swap", "tail": _last_lines(text, lines),
                        "error": ""}
        except Exception:
            pass
    from pathlib import Path
    art = Path(profile.get("artifacts_dir") or "")
    candidates = []
    if art.is_dir():
        # Fixed names first (runtime log, smoke-test log), then whatever
        # is newest -- plan tests log as plan-test-<id8>.log.
        for fname in ("llama-swap.log", "test.log"):
            if (art / fname).is_file():
                candidates.append(art / fname)
        others = [p for p in art.glob("*.log")
                  if p.name not in ("llama-swap.log", "test.log")]
        candidates.extend(sorted(others, key=lambda p: p.stat().st_mtime,
                                 reverse=True))
    for p in candidates:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if text.strip():
            return {"source": p.name, "tail": _last_lines(text, lines),
                    "error": ""}
    return {"source": "", "tail": "",
            "error": "no log yet -- the model has not been started or tested"}


def _last_lines(text, n):
    return "\n".join(text.splitlines()[-n:])


# ---- fit preview + typed configure save ---------------------------

def admission_preview(name, ctx=None, budgets_bytes=None, moe_mode=None):
    """The tier planner's answer for a draft config, without saving.

    Draft values overlay a deep copy; stored machine snapshot still win, so
    the preview is exactly what apply would compute (same path as
    /api/tiers/{name}). The gate compares against the SAVED profile, so a
    structural draft shows its confirm before anything is written.

    moe_mode must be draftable alongside the budgets: the planner only
    charges cache budgets when the mode is not "off", so previewing
    enable-the-cache against a stored mode of "off" would silently drop
    the drafted budgets and predict a fit the real plan won't have."""
    import modelctl_tiers
    saved = modelctl.load_profile(name)
    draft = copy.deepcopy(saved)
    if ctx is not None:
        draft.setdefault("config", {})["ctx"] = int(ctx)
    if budgets_bytes is not None or moe_mode is not None:
        mc = draft.setdefault("moe_cache", {})
        if moe_mode is not None:
            mc["mode"] = str(moe_mode)
        if budgets_bytes is not None:
            mc.setdefault("gpu", {})["budgets_bytes"] = {
                dev: int(v) for dev, v in budgets_bytes.items() if int(v) > 0}
    plan, inputs, source = modelctl.plan_tiers_for_profile(draft)
    gate = (modelctl_tiers.tier_change_gate(saved, plan)
            if plan is not None else None)
    if plan is None:
        return {"plan": None, "planning_inputs": inputs,
                "planning_inputs_source": source, "gate": None}
    return {"plan": plan, "planning_inputs": inputs,
            "planning_inputs_source": source, "gate": gate}


# ---- placement ----------------------------------------------------------

# The admission key for the rig's own memory. The planner calls this share
# "CPU" in its layout rows; the fleet surface lists it as a device named
# RAM, and the selection the operator sends back is keyed the fleet's way.
HOST_KEY = "RAM"
_HOST_ROW_LABEL = "CPU"


def _host_share(plan):
    """(bytes, resident) for the part of the model the host holds.

    Read off the emitted plan, not off planner intermediates, for the same
    reason _admission_report is: only the emitted config describes what
    llama.cpp will actually do. `--no-mmap` is the whole difference between
    a host share that lives in memory and one that streams off the SSD --
    the planner decides it by fit, so the flag is the answer, not the tier.

    Bytes come back through the layout row's GiB, so they carry that row's
    display precision (~0.1%). This number is shown, never charged.
    """
    extra = ((plan.get("config") or {}).get("extra") or "").split()
    for label, gib, _detail in plan.get("layout") or []:
        if label == _HOST_ROW_LABEL:
            return int(gib * (1 << 30)), "--no-mmap" in extra
    return 0, True


def _placement_devices(plan):
    """Bytes per device for one plan, keyed by admission key.

    Every entry is the planner's own number: GPUs from the admission
    report's demand, remote devices from the RPC admission the config
    carries, the host from its layout row. Nothing here re-derives a split
    -- a second implementation of placement is exactly the failure this
    endpoint exists to prevent.
    """
    devices = {}
    report = (plan.get("admission") or {}).get("devices") or {}
    for dev, row in report.items():
        devices[dev] = {
            "bytes": int(row.get("demand_bytes", 0)),
            "backing": "VRAM",
            "fits": bool(row.get("fits", True)),
            "capacity_bytes": int(row.get("capacity_bytes", 0)),
            "usable_bytes": int(row.get("usable_bytes", 0)),
        }
    rpc = ((plan.get("config") or {}).get("rpc") or {}).get("admission") or {}
    for key, byte_count in rpc.items():
        devices[key] = {"bytes": int(byte_count), "backing": "over RPC",
                        "fits": True}
    host_bytes, resident = _host_share(plan)
    if host_bytes:
        devices[HOST_KEY] = {"bytes": host_bytes, "fits": True,
                             "backing": "RAM" if resident else "SSD via mmap"}
    return devices


def placement_preview(profile, selection):
    """Where this model's weights land for a chosen set of devices.

    The one read the placement screen needs. `selection` maps an admission
    key to {"on", "ceiling_bytes"}; an empty selection is the automatic
    placement, which is what the operator sees before touching anything.

    Returns None when the model's layout cannot be analyzed, so the caller
    can say so rather than render an empty machine.
    """
    import modelctl_plans
    plan = modelctl_plans.plan_for_selection(profile, selection)
    if plan is None:
        return None
    devices = _placement_devices(plan)
    # Asked of the plan again rather than read back off the rendered row:
    # deriving the headline number from a display string means a reworded
    # backing label silently reports "nothing spills".
    host_bytes, resident = _host_share(plan)
    return {
        "name": profile.get("name", ""),
        "selection": dict(selection or {}),
        "tier": plan.get("tier"),
        "config": plan.get("config"),
        "warnings": list(plan.get("warnings") or ()),
        "analysis": plan.get("analysis"),
        "admission": plan.get("admission"),
        "cache_budgets": plan.get("cache_budgets"),
        "layout": [{"label": label, "gib": gib, "detail": detail}
                   for label, gib, detail in plan.get("layout") or []],
        "devices": devices,
        # The headline number: bytes with nowhere to go but the SSD. The
        # invariant this screen enforces is that this stays 0 while any
        # enabled device still has room.
        "spill_bytes": 0 if resident else host_bytes,
    }


def classify_config_save(profile, updates, budgets_bytes=None,
                         moe_mode=None):
    """The configure form's structural-change gate.

    Reuses the planner's own diff classifier so "structural" means the
    same thing here as on the auto-place path; cache-budget and
    cache-mode changes are structural for the same reason
    tier_change_gate treats an effective budget change that way."""
    import modelctl_tiers
    current = profile.get("config", {}) or {}
    draft = dict(current)
    for k, v in updates.items():
        if k in ("enabled",):
            continue
        draft[k] = v
    gate = modelctl_tiers.classify_config_diff(current, draft)
    changes = list(gate.get("changes") or [])
    kind = gate.get("kind", "none")
    cur_mc = _budgets(profile)
    if budgets_bytes is not None:
        cur_b = cur_mc["budgets_bytes"]
        new_b = {dev: int(v) for dev, v in budgets_bytes.items() if int(v) > 0}
        if new_b != cur_b:
            kind = "structural"
            changes.append(
                f"cache budgets: {json.dumps(cur_b, sort_keys=True)} -> "
                f"{json.dumps(new_b, sort_keys=True)}")
    if moe_mode is not None and str(moe_mode) != cur_mc["mode"]:
        # Flipping the cache on or off moves expert bytes as surely as a
        # budget edit does.
        kind = "structural"
        changes.append(f"moe_cache mode: {cur_mc['mode']} -> {moe_mode}")
    # First placement never needs the confirm, same rule as the tier gate.
    requires = (kind == "structural"
                and modelctl_tiers._has_placement(current))
    return {"kind": kind, "changes": changes, "requires_accept": requires}


# ---- wizard: GGUF analysis ---------------------------------------------

def analyze_model(profile):
    """The analyze step's real content: what the GGUF header says.

    This is where the register form's hard numbers come from -- ctx max is
    read here, not typed in. Returns None when the header can't be parsed
    (the page degrades to profile facts and says the header was
    unreadable)."""
    import modelctl_vram
    path = profile.get("model_path") or ""
    meta = modelctl_vram.read_gguf_kv_metadata(path) if path else {}
    if not meta:
        return None
    arch = meta.get("general.architecture") or ""
    def g(suffix):
        return meta.get(f"{arch}.{suffix}") if arch else None
    params = modelctl_vram.gguf_kv_params(meta)
    kv_per_token = None
    if params:
        cfg = profile.get("config", {}) or {}
        try:
            kv_per_token = modelctl_vram.kv_cache_bytes(
                params, 1, cfg.get("cache_type_k") or "f16",
                cfg.get("cache_type_v") or None)
        except Exception:
            kv_per_token = None
    layout = None
    try:
        layout = modelctl_vram.gguf_model_layout(path)
    except Exception:
        layout = None
    return {
        "arch": arch,
        "name": meta.get("general.name") or "",
        "model_max_ctx": g("context_length"),
        "block_count": g("block_count"),
        "embedding_length": g("embedding_length"),
        "expert_count": g("expert_count"),
        "is_moe": bool(layout and layout.get("is_moe")),
        "weight_bytes": (layout or {}).get("weight_bytes"),
        "kv_bytes_per_token": kv_per_token,
    }


# ---- wizard: typed state ------------------------------------------------

def wizard_summary(state):
    return {
        "wizard_id": state.wizard_id,
        "step": state.step,
        "source_type": state.source_type,
        "repo_id": state.repo_id,
        "local_path": state.local_path,
        "profile_name": state.profile_name,
        "updated_at": state.updated_at,
        "created_at": state.created_at,
    }


def wizard_detail(state, store):
    """Full wizard state for the SPA: the persisted dataclass plus the
    derived per-step gate answers, so blocked-advance renders the server's
    reason rather than re-deriving one client-side."""
    from .wizard import STEPS
    d = state.to_dict()
    d["steps"] = list(STEPS)
    d["step_gates"] = {
        step: {"blocking_reason": state.blocking_reason(step),
               "outcome": state.outcome(step)}
        for step in ("download", "test", "register")}
    d["jobs"] = {}
    for label, job_id in (("download", state.download_job_id),
                          ("test", state.test_job_id)):
        if job_id:
            job = store.get(job_id)
            if job:
                from .telemetry import job_row
                d["jobs"][label] = job_row(job)
    return d
