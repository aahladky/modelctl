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


class UnknownDevices(ValueError):
    """A selection naming devices this machine does not have.

    modelctl_tiers.select_inputs only ever looks up keys it already knows
    -- inventory devices, RAM, remote rungs -- so a key nothing knows was
    accepted and then never read, and the answer described a placement the
    operator did not ask for. That is the same silent no-op the query
    parser already refuses for an unreadable ceiling, so it gets the same
    422 rather than a quietly different layout.
    """

    def __init__(self, unknown, known):
        self.unknown = sorted(unknown)
        self.known = sorted(known)
        verb = "is not a device" if len(self.unknown) == 1 else "are not devices"
        super().__init__(
            f"{', '.join(self.unknown)} {verb} on this machine -- "
            f"known devices: {', '.join(self.known) or 'none'}")


def known_devices(inputs):
    """Every admission key a selection may name.

    The remote half comes from the recorded fleet budgets, which are
    presence-independent on purpose (modelctl_fleet.budget_input): a
    device must not stop being a valid choice because a laptop is closed.
    Whether it can be reached is its state, reported per device, and not
    a reason to refuse the key.
    """
    known = {d.get("device") for d in (inputs.get("inventory") or [])}
    known.discard(None)
    known.add(HOST_KEY)
    known.update(inputs.get("fleet_budgets") or {})
    # Plus every device the fleet registry currently declares. A node
    # enrolled since this profile's inputs were recorded is a device the
    # planner will happily place on, so refusing to let the operator name
    # it would make the valid set narrower than the planner's.
    known.update(_device_states(inputs))
    return known


def refuse_unknown_devices(selection, inputs):
    """The UnknownDevices this selection earns, or None.

    One derivation and one message for both the read and the apply: the
    two must not be able to disagree about which keys are legal, or the
    screen would preview a placement the apply then refuses.
    """
    known = known_devices(inputs)
    unknown = set(selection or {}) - known
    return UnknownDevices(unknown, known) if unknown else None


def _device_states(inputs):
    """{admission key: {"state", "detail"}} for every device a selection
    may name.

    Presence belongs on the device, not in the decision about whether the
    device exists. A remote key used to vanish from the answer when its
    node could not be reached, so ticking one produced a response
    byte-identical to the baseline and the screen could not tell a request
    that was honoured from one that was quietly dropped.

    The states are the fleet page's own -- PRESENT / STALE /
    PIN_MISMATCH, tri-state because a pin-mismatched node is *up* and must
    never look available. Derived by calling that page's presence_state
    rather than re-reading the rule here: two derivations of "is this
    machine usable" is how the two surfaces come to disagree.

    Nothing here probes. This reads what was last recorded, so drawing a
    placement is not a network event.
    """
    import time

    import modelctl_fleet

    from . import fleet as fleet_view

    states = {d.get("device"): {"state": fleet_view.PRESENT, "detail": ""}
              for d in (inputs.get("inventory") or []) if d.get("device")}
    # The rig's own memory is present by construction, like its cards.
    states[HOST_KEY] = {"state": fleet_view.PRESENT, "detail": ""}
    try:
        nodes = modelctl_fleet.load_fleet()
        presence = modelctl_fleet.load_presence()
    except Exception:
        nodes, presence = [], {}
    now = time.time()
    for node in nodes:
        state, detail = fleet_view.presence_state(presence.get(node.name),
                                                  now=now)
        for device in node.devices:
            # Every device the registry declares, not only those the
            # recorded snapshot named. A plan can place weights on a node
            # enrolled after its inputs were recorded -- the live 122B
            # does exactly that -- and such a device arrived with no
            # bound and no state, which is the hole this whole task is
            # about.
            states[modelctl_fleet.admission_key(node.name, device.name)] = {
                "state": state,
                "detail": detail,
                # total_bytes is the only place the remote hardware's own
                # size is known; without it a remote bar would claim
                # capacity == budget, saying a card is exactly as big as
                # the ceiling set on it.
                "total_bytes": int(getattr(device, "total_bytes", 0) or 0),
                "budget_bytes": int(getattr(device, "budget_bytes", 0) or 0),
            }
    # A budget recorded against a node the registry no longer lists. The
    # key stays valid -- it is in the snapshot this plan was built from --
    # but nothing can vouch for the machine behind it.
    for key in set(inputs.get("fleet_budgets") or {}) - set(states):
        states[key] = {"state": fleet_view.STALE,
                       "detail": "no longer in the fleet registry"}
    return states


def legacy_pin(profile):
    """The pinned plan still deciding this profile's launch, or None.

    A pin names a compiled artifact; this surface owns intent. The two
    cannot be reconciled here, so it reports the pin rather than drawing
    an automatic placement that is not what runs -- which is exactly how
    qwen3-5-122b-a10b-ud came to read as "on automatic" while a
    four-device ladder plan was launching.

    A profile with a selection is unaffected: the launcher stops reading
    the pin once a selection exists, so going on reporting it would be
    the same lie in the other direction.
    """
    runtime = profile.get("runtime") or {}
    pin = runtime.get("pinned_plan_id")
    if not pin or (profile.get("planning") or {}).get("selection"):
        return None
    return {"plan_id": pin, "mode": runtime.get("mode") or ""}


class NoPinToAdopt(ValueError):
    """Nothing for this profile to adopt, with the reason."""


def pin_adoption_preview(profile):
    """The selection that would use the devices the pinned plan uses,
    together with the layout it produces.

    Offered, never installed. Adopting is the operator sending this
    selection to the ordinary apply -- the same write every other
    placement takes -- so nothing here writes: a derivation that
    installed itself would be a fifth writer winning quietly, which is
    the thing this whole alignment removes.

    Device on/off only. A compiled plan does not record the ceiling that
    produced it, so inventing one would be a number nobody chose. That
    makes the derived selection an approximation, and the caveat and the
    previewed layout beside it are how the operator sees what adopting
    actually costs before accepting it.

    Compiled on request rather than on every placement read: finding the
    pin means compiling the candidate set, which probes the machine.
    """
    import modelctl_plans

    pin = legacy_pin(profile)
    if not pin:
        raise NoPinToAdopt("this model is not running a pinned plan")
    plan = next((p for p in modelctl_plans.compile_launch_plans(profile)
                 if p.id == pin["plan_id"]), None)
    if plan is None:
        # Plan ids hash the compiled config, so a planner improvement can
        # strand a pin -- the 122B's moved cf4274bf -> 2b426a8f after the
        # --no-mmap fix. Say that, rather than offering a selection
        # derived from some other plan.
        raise NoPinToAdopt(
            f"pinned plan {pin['plan_id']} is not among the plans this "
            f"machine compiles today -- the planner has changed since it "
            f"was pinned, so there is no layout left to copy")
    used = set(plan.claim.vram_admission_bytes() or {})
    if plan.claim.ram_admission_bytes():
        used.add(HOST_KEY)
    inputs, _source = modelctl.resolve_planning_inputs(profile)
    # Every known device gets an explicit answer: an absent key means "as
    # the machine offers", which is not what copying a fixed layout means.
    selection = {key: {"on": key in used} for key in known_devices(inputs)}
    return {
        "pinned": {**pin, "label": getattr(plan, "label", "")},
        "selection": selection,
        "placement": placement_preview(profile, selection),
        "caveat": ("this copies which devices the pinned plan uses, not the "
                   "ceilings that produced it -- compare the layout below "
                   "against what is running before accepting"),
    }


_STORAGE_TIER_LABEL = {1: "GPU only", 2: "GPU + RAM"}


def _storage_floor(profile, host_bytes, resident):
    """Whether this layout crosses the model's own storage limit.

    `maximum_storage_tier` (1=gpu, 2=gpu+ram, 3=gpu+ram+storage) is a hard
    constraint on the ranked path -- modelctl_plans:1697 drops any plan
    whose claim is storage_mode "mmap" when the tier is below 3. The
    placement path never read it, so the screen offered layouts the
    profile forbids: live on 2026-08-04, qwen3-5-122b-a10b-ud is set to
    tier 2 and previewed 39.52 GiB of SSD streaming.

    A host share that is not resident is exactly that claim's "mmap" --
    bytes addressed off the disk rather than held in memory.

    This reports; it does not choose. There is one plan for a selection,
    not a ranked list, so there is nothing here to fall back to -- picking
    a different layout is the degradation ladder's job, and a preview that
    quietly substituted one would be answering for a selection nobody
    made.
    """
    tier = (profile.get("runtime") or {}).get("maximum_storage_tier", 3)
    try:
        tier = int(tier)
    except (TypeError, ValueError):
        tier = 3
    crossed = bool(host_bytes) and not resident and tier < 3
    detail = ""
    if crossed:
        detail = (f"this layout streams {host_bytes / float(1 << 30):.1f} GiB "
                  f"from the SSD, and this model's storage limit is "
                  f"{_STORAGE_TIER_LABEL.get(tier, tier)}")
    return {"maximum_storage_tier": tier, "crossed": crossed,
            "detail": detail}


def _placement_devices(plan, inputs=None):
    """Bytes per device for one plan, keyed by admission key.

    Every byte count is the planner's own number: GPUs from the admission
    report's demand, remote devices from the RPC admission the config
    carries, the host from its layout row. Nothing here re-derives a split
    -- a second implementation of placement is exactly the failure this
    endpoint exists to prevent.

    Every row also carries the bound it is measured against, because the
    control this feeds is a bar with a ceiling you can drag, and a bar
    needs a maximum. GPUs get theirs from the admission report. The other
    two rows came back with neither, so the host row held 39.5 GiB against
    a capacity of zero and the screen had nothing to draw for the one
    device the spill actually lands on:

      usable   -- what the planner was allowed to spend, taken from the
                  SAME inputs the plan was built from, so the bound agrees
                  with planned_against rather than with a fresher reading.
                  Remote ceilings come from the recorded fleet budgets,
                  presence-independent on purpose: a closed laptop must
                  not collapse the bar its device draws.
      capacity -- the hardware's own limit. For memory that is installed
                  RAM, read live because how much is fitted does not drift
                  with load; see modelctl_vram.system_ram_total.

    `fits` answers whether the bytes are inside the bound, on every row.
    It was hardcoded True on the remote and host rows, which put
    "fits: True" beside a host share streaming off the SSD -- the backing
    field telling the truth while the boolean next to it said otherwise.
    """
    import modelctl_vram

    inputs = inputs or {}
    states = _device_states(inputs)

    def remote_usable(key):
        """The room a remote device offers.

        The recorded snapshot first, because that is what the plan was
        built against. A device the snapshot never named -- a node
        enrolled since -- falls back to the ceiling the registry declares
        for it now, which is better than the zero it used to report while
        holding real bytes.
        """
        recorded = int((inputs.get("fleet_budgets") or {}).get(key, 0) or 0)
        return recorded or int((states.get(key) or {}).get("budget_bytes") or 0)

    def remote_capacity(key, ceiling):
        """A remote device's own size, falling back to its ceiling.

        A registry that no longer lists the node leaves nothing to read,
        and a bar drawn to its ceiling is the honest degradation: it
        understates headroom rather than inventing it.
        """
        return int((states.get(key) or {}).get("total_bytes") or 0) or ceiling

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
        held = int(byte_count)
        ceiling = remote_usable(key)
        devices[key] = {
            "bytes": held,
            "backing": "over RPC",
            # A node the registry no longer declares leaves no ceiling to
            # judge against; say so with False rather than assert a fit
            # that nothing backs.
            "fits": bool(ceiling) and held <= ceiling,
            "capacity_bytes": remote_capacity(key, ceiling),
            "usable_bytes": ceiling,
        }
    host_bytes, resident = _host_share(plan)
    if host_bytes:
        devices[HOST_KEY] = {
            "bytes": host_bytes,
            "backing": "RAM" if resident else "SSD via mmap",
            # Held in memory is the only way a host share fits: a share
            # that streams is by definition bytes memory had no room for.
            "fits": resident,
            "capacity_bytes": modelctl_vram.system_ram_total(),
            "usable_bytes": int(inputs.get("ram_available_bytes") or 0),
        }
    # Every device the machine has, whether or not this layout used it: a
    # console where placement means ticking devices cannot offer a tick
    # for a device it never draws. An unused device is an empty bar, not
    # an absent one.
    frac = (inputs.get("vram_limit_pct") or 100) / 100.0
    local_totals = {d.get("device"): int(d.get("total_bytes", 0) or 0)
                    for d in (inputs.get("inventory") or []) if d.get("device")}
    for key in known_devices(inputs):
        if key in devices:
            continue
        if key == HOST_KEY:
            capacity = modelctl_vram.system_ram_total()
            usable = int(inputs.get("ram_available_bytes") or 0)
        elif key in local_totals:
            # The same arithmetic select_inputs spends: limit_pct of the
            # card's total is what plan_tiers is allowed to place on it.
            capacity = local_totals[key]
            usable = int(capacity * frac)
        else:
            usable = remote_usable(key)
            capacity = remote_capacity(key, usable)
        devices[key] = {"bytes": 0, "backing": "", "fits": True,
                        "capacity_bytes": capacity, "usable_bytes": usable}
    for key, row in devices.items():
        found = states.get(key) or {}
        # state and detail only: total_bytes is how capacity was worked
        # out, not a field the answer carries.
        row["state"] = found.get("state", "")
        row["detail"] = found.get("detail", "")
    return devices


def placement_preview(profile, selection, refresh=False):
    """Where this model's weights land for a chosen set of devices.

    The one read the placement screen needs. `selection` maps an admission
    key to {"on", "ceiling_bytes"}; an empty selection is the automatic
    placement, which is what the operator sees before touching anything.

    Planner inputs are resolved HERE rather than left to
    plan_for_selection, for two reasons. The answer has to say which
    machine snapshot it planned against -- the screen puts the planner's
    numbers beside live free memory, and without the snapshot's date a
    three-day-old picture reads as a broken layout instead of a stale one.
    And resolving once means the snapshot named is provably the snapshot
    spent.

    refresh=True re-reads the machine instead of the profile's record: the
    "that was then" escape hatch, for when the recorded snapshot no longer
    describes the computer. It writes nothing; only an apply records.

    Returns None when the model's layout cannot be analyzed, so the caller
    can say so rather than render an empty machine.
    """
    import modelctl_plans
    inputs, source = modelctl.resolve_planning_inputs(profile,
                                                      refresh=refresh)
    bad = refuse_unknown_devices(selection, inputs)
    if bad:
        raise bad
    plan = modelctl_plans.plan_for_selection(profile, selection,
                                             inputs=inputs)
    if plan is None:
        return None
    devices = _placement_devices(plan, inputs)
    # Asked of the plan again rather than read back off the rendered row:
    # deriving the headline number from a display string means a reworded
    # backing label silently reports "nothing spills".
    host_bytes, resident = _host_share(plan)
    floor = _storage_floor(profile, host_bytes, resident)
    # The crossing rides the plan's own warnings channel as well as its
    # own field: _apply_tier_plan logs plan["warnings"] on every apply, so
    # a layout that breaks the profile's storage limit cannot be written
    # without the job having said so.
    warnings = list(plan.get("warnings") or ())
    if floor["crossed"]:
        warnings.append(floor["detail"])
    # Turning on a device nothing can reach must not read as success. The
    # planner will not place weights there, so without this the answer is
    # the one the operator would have got by not asking at all.
    for key, choice in (selection or {}).items():
        if not (choice or {}).get("on"):
            continue
        row = devices.get(key) or {}
        if row.get("state") and row["state"] != "PRESENT":
            detail = row.get("detail") or "not reachable"
            warnings.append(f"{key} was turned on but its machine is "
                            f"{row['state']} ({detail}), so the planner "
                            f"placed nothing there")
    return {
        "name": profile.get("name", ""),
        "selection": dict(selection or {}),
        # What is set to run right now, as recorded by the last apply. The
        # screen opens on this and compares against it to know whether it
        # has unsaved changes; without it the only way to tell would be to
        # reconstruct the choice from the emitted -ot rules, which is the
        # second reader of placement this endpoint exists to remove.
        "applied_selection": (profile.get("planning") or {})
                             .get("selection") or {},
        # Which picture of the machine these numbers were computed from.
        # "stored" means the profile's record -- deliberate, so a layout
        # does not drift every time free memory moves -- and recorded_at
        # is how old that picture is. "live" means the machine as it is.
        "planned_against": {
            "source": source,
            "recorded_at": (profile.get("planning") or {}).get("recorded_at"),
            "ram_available_bytes": int(inputs.get("ram_available_bytes") or 0),
        },
        "tier": plan.get("tier"),
        "config": plan.get("config"),
        "warnings": warnings,
        # The profile's own ceiling on how far down the memory tiers a
        # layout may reach. Reported, not enforced here -- see
        # _storage_floor.
        "storage_floor": floor,
        # Set when a pinned plan id, not this selection, is still what
        # decides the launch. The screen must say so rather than render
        # an automatic placement that is not running.
        "legacy_pin": legacy_pin(profile),
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
