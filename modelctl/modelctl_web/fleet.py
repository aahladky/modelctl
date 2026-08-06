"""The fleet read model: every node the planner can place on, as one list.

A node is a node. The rig is the local node -- it appears in the same
list, with the same identity/devices/presence shape, because the
question an operator opens this view to answer ("where can this model
go, and how much of each place may it have") does not change when the
answer happens to be the machine under the desk. The differences that
are real are carried as fields, not as a second layout: the local node
is always present, its pin is the checkout's own by construction, and
its per-device budget is derived from the VRAM limit and reserve on the
settings page rather than stored here, so it is not editable from this
surface. One editor per number.

Presence is deliberately tri-state, not a boolean:

    PRESENT       last probe reachable, on the pinned commit, inside the
                  presence TTL -- the planner will actually use it
    STALE         never probed, probe expired, or the last probe could
                  not reach it. Not usable, and the reason is carried
    PIN_MISMATCH  reachable, but built from a different llama.cpp commit

The third one is why a boolean will not do. A pin-mismatched node is
*up* -- it answers a handshake immediately -- and it is the one state
where the honest rendering is furthest from the cheerful one: placing a
graph across two ggml builds that disagree on kernels is how you get
wrong numbers rather than an error. It must never look available.

Nothing here probes. `probe_node` is an explicit call the console makes
on page open and on the per-node button; this assembles what was last
recorded, so a render is not a network event.
"""
import time

import modelctl
import modelctl_fleet


PRESENT = "PRESENT"
STALE = "STALE"
PIN_MISMATCH = "PIN_MISMATCH"


def presence_state(probe, now=None, ttl=modelctl_fleet.PRESENCE_TTL_SECONDS):
    """(state, detail) for one node's last recorded probe.

    `probe` is None for a node nobody has ever probed, which is STALE
    rather than absent-with-a-reason: "we have never looked" and "we
    looked and it was gone" are different facts and the detail says
    which.
    """
    if probe is None:
        return STALE, "never probed"
    age = (time.time() if now is None else now) - probe.probed_at
    if probe.reachable and not probe.pin_agrees:
        return PIN_MISMATCH, (probe.detail or "built from a different commit "
                                              "than this checkout pins")
    if not probe.reachable:
        return STALE, probe.detail or "unreachable at the last probe"
    if ttl is not None and age > ttl:
        return STALE, (f"last probe was {int(age)}s ago, past the "
                       f"{int(ttl)}s presence TTL")
    return PRESENT, probe.detail or ""


def _device_row(node, device):
    ceiling, basis = modelctl_fleet.device_ceiling(device)
    return {
        "name": device.name,
        "kind": device.kind,
        "budget_bytes": int(device.budget_bytes),
        "total_bytes": int(device.total_bytes),
        "cap_bytes": int(device.cap_bytes),
        "ceiling_bytes": int(ceiling),
        "ceiling_basis": basis,
        "admission_key": modelctl_fleet.admission_key(node.name, device.name),
        "editable": True,
        "edit_note": "",
    }


def _remote_node(node, presence, now, expected_pin):
    probe = presence.get(node.name)
    state, detail = presence_state(probe, now=now)
    return {
        "name": node.name,
        "location": "remote",
        "host": node.host,
        "port": node.port,
        "endpoint": node.endpoint,
        "variant": node.variant,
        "enabled": bool(node.enabled),
        "note": node.note or "",
        # Pin agreement is a first-class field: it decides a presence
        # state of its own, so it cannot be a footnote on the card.
        "pin": {
            "node": node.pin or "",
            "expected": expected_pin,
            "agrees": bool(node.pin and expected_pin
                           and node.pin == expected_pin),
        },
        "presence": {
            "state": state,
            "detail": detail,
            "reachable": bool(probe.reachable) if probe else False,
            "protocol": probe.protocol if probe else "",
            "probed_at": probe.probed_at if probe else None,
            "ttl_seconds": modelctl_fleet.PRESENCE_TTL_SECONDS,
        },
        "devices": [_device_row(node, d) for d in node.devices],
    }


def _local_node(errors):
    """The rig, in the same shape.

    Its budget per device is what admission actually charges a local
    card -- limit_pct of the card minus its reserve -- rather than a
    number stored here, which is why it is read-only on this surface:
    the two knobs that produce it live on the settings page and having
    two editors for one number is how they drift.
    """
    devices = []
    try:
        import modelctl_hardware
        snapshot = modelctl_hardware.capture_hardware_snapshot()
        try:
            frac = modelctl.load_defaults().get("vram_limit_pct", 90) / 100.0
        except Exception:
            frac = 0.90
        for g in modelctl_hardware.enabled_gpus(snapshot):
            budget = max(0, int(g.total_bytes * frac) - g.reserve_bytes)
            devices.append({
                "name": g.device,
                "kind": "gpu",
                "label": getattr(g, "name", "") or "",
                "budget_bytes": budget,
                "total_bytes": int(g.total_bytes),
                "cap_bytes": 0,
                "ceiling_bytes": int(g.total_bytes),
                "ceiling_basis": "reported device total",
                "admission_key": g.device,
                "editable": False,
                # An em dash, not two hyphens: this string is rendered to
                # a browser, not read in a source file.
                "edit_note": (f"{int(frac * 100)}% VRAM limit minus a "
                              f"{g.reserve_bytes / (1 << 30):.1f} GiB reserve "
                              f"— both on the settings page"),
            })
        # Memory is a budget device too. reservation_budgets() has always
        # charged a "RAM" key, and the tier planner spills routed experts
        # onto it -- but this surface only ever walked enabled_gpus(), so
        # the rig read as GPUs and nothing else while the laptop's CPU
        # node showed up in full. A console where placement means ticking
        # devices has to be able to tick memory.
        reserve = int(getattr(snapshot, "ram_reserve_bytes", 0) or 0)
        total = int(getattr(snapshot, "ram_total_bytes", 0) or 0)
        available = int(getattr(snapshot, "ram_available_bytes", 0) or 0)
        devices.append({
            "name": "RAM",
            "kind": "ram",
            "label": "system memory",
            # Free-minus-reserve, not total-minus-reserve: this is the
            # number the reservation gate charges, and a row showing
            # installed capacity would promise room the desktop is
            # already using.
            "budget_bytes": max(0, available - reserve),
            "total_bytes": total,
            "cap_bytes": 0,
            "ceiling_bytes": total,
            "ceiling_basis": "installed memory",
            "admission_key": "RAM",
            "editable": False,
            "edit_note": (f"free memory minus a "
                          f"{reserve / (1 << 30):.1f} GiB reserve "
                          f"— the reserve is on the settings page"),
        })
    except Exception as e:
        errors["local_devices"] = str(e) or "local device inventory failed"
    return {
        "name": "rig (this machine)",
        "location": "local",
        "host": "127.0.0.1",
        "port": None,
        "endpoint": "local",
        "variant": "local",
        "enabled": True,
        "note": "the checkout that plans and launches; always present",
        "pin": {"node": modelctl_fleet.local_pin(),
                "expected": modelctl_fleet.local_pin(), "agrees": True},
        "presence": {"state": PRESENT, "detail": "", "reachable": True,
                     "protocol": "", "probed_at": None,
                     "ttl_seconds": modelctl_fleet.PRESENCE_TTL_SECONDS},
        "devices": devices,
    }


def _attach_holdings(nodes):
    """What live models hold, written onto the device rows holding it.

    Display only -- budgets are untouched and admission never reads
    these fields. The RAM row's budget is free-minus-reserve, which is
    admission truth but cannot say whether the missing gigabytes are the
    desktop's or a served model's own weights; `held` names the holder,
    so "no room" and "no room because a model already holds it" stop
    rendering identically. Remote rows get the same treatment: an
    active model's rungs show up on the laptop device they occupy.

    Returns what each model streams off the SSD instead of holding,
    keyed by profile. That one is per-MODEL, not per-device -- there is
    no device to put it on, which is precisely what makes those bytes
    invisible to every bar. A profile absent from the map streams
    nothing; the map absent altogether means we could not look.

    Raises on a runtime DB it cannot read; fleet_view catches and files
    it under errors["holdings"], leaving the fields ABSENT -- a zero
    here would claim "nothing is held" on the strength of not having
    been able to look.
    """
    import modelctl_runtime
    per_key = {}
    streamed = {}
    for holding in modelctl_runtime.RuntimeDB().live_holdings():
        for key, held in holding["devices"].items():
            per_key.setdefault(key, []).append(
                {"profile": holding["profile"], "bytes": int(held)})
        mmap_bytes = int(holding.get("mmap_bytes") or 0)
        if mmap_bytes > 0:
            streamed[holding["profile"]] = (
                streamed.get(holding["profile"], 0) + mmap_bytes)
    for node in nodes:
        for row in node.get("devices") or []:
            held = per_key.get(row.get("admission_key")) or []
            row["held"] = held
            row["held_bytes"] = sum(h["bytes"] for h in held)
    return streamed


def night_lane_rows():
    """The night-lane jobs that need a fleet node, READ-ONLY.

    On this page because the question "may I change this node" and the
    question "is a pre-registered comparison waiting on it" are the same
    operator's, one keystroke apart -- moving a budget under an enabled
    job changes what that job measures. Enabling one is not offered
    here: a pre-registration is a commitment, and the surface that can
    flip it in passing is the surface that quietly invalidates it.
    """
    import modelctl_nightlane
    rows = []
    for job in modelctl_nightlane.load_jobs():
        required = sorted(job.required_nodes)
        if not required:
            continue
        rows.append({
            "id": job.id,
            "title": job.title,
            "enabled": bool(job.enabled),
            "mode": job.mode,
            "registered": job.registered,
            "requires_nodes": required,
            "arms": [{"name": a.name, "profile": a.profile,
                      "requires_nodes": sorted(a.requires_nodes)}
                     for a in job.arms],
        })
    return rows


def fleet_view(now=None):
    """Every node, local first, plus what a budget edit would disturb.

    Same degrade contract as the rest of the /v2 surface: a section that
    throws yields its empty shape and an error string naming it, never
    an exception out of the endpoint.
    """
    errors = {}
    view = {"nodes": [_local_node(errors)], "night_lane": [],
            "stale_profiles": [], "local_pin": "", "errors": errors}
    try:
        view["local_pin"] = modelctl_fleet.local_pin()
    except Exception as e:
        errors["pin"] = str(e) or "could not read this checkout's pin"
    try:
        nodes = modelctl_fleet.load_fleet()
        presence = modelctl_fleet.load_presence()
        for node in nodes:
            view["nodes"].append(
                _remote_node(node, presence, now, view["local_pin"]))
    except Exception as e:
        errors["nodes"] = str(e) or "the fleet registry could not be read"
    try:
        # After every node is assembled, so a holding lands on remote
        # rows too. On failure the fields stay absent, not zero -- and
        # `mmap_bytes` is only set on success for the same reason: the
        # console renders "nothing on disk" on a known zero, which is a
        # lie to tell about a model we could not look up.
        view["mmap_bytes"] = _attach_holdings(view["nodes"])
    except Exception as e:
        errors["holdings"] = str(e) or "live holdings could not be read"
    try:
        view["night_lane"] = night_lane_rows()
    except Exception as e:
        errors["night_lane"] = str(e) or "the night-lane registry is unreadable"
    try:
        # Which stored-input plans are already out of step with the
        # budgets in force. Shown before an edit, not after: it is the
        # cost of the edit the operator is about to make.
        view["stale_profiles"] = modelctl_fleet.stale_input_profiles()
    except Exception as e:
        errors["stale_profiles"] = str(e) or "profiles could not be read"
    return view
