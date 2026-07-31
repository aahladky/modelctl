"""Managed llama-swap routing matrix (Milestone 8).

Generates coexistence rules from modelctl's resource understanding instead of
hand-maintained `matrix:` sets. Managed entries are merged into the existing
matrix with an `mc_` prefix so they are identifiable, removable, and never
trample hand-authored entries.

Compatibility rule: for every resource (per-device VRAM, system RAM), the sum
of member claims must fit the budget. Eviction cost derives from measured
median load time (plan_runs) with the doc's clamp(load_s/5, 1, 100).
"""
import statistics

import modelctl
import modelctl_plans
import modelctl_runtime
import modelctl_vram

MANAGED_PREFIX = "mc_"

# A model is a "helper" when its total claim fits this fraction of the
# primary GPU budget -- helpers can coexist with one large model.
HELPER_FRAC = 0.25


def profile_claim(profile, inventory, defaults=None):
    """Resource claim used for routing: {resource: bytes} with per-device
    VRAM keys and 'RAM'. Uses the pinned plan's claim when the profile is
    managed with a pinned plan; otherwise a config-derived estimate."""
    d = defaults or modelctl.load_defaults()
    ctx = int(profile.get("config", {}).get("ctx", 32768))
    cfg = profile.get("config", {})

    runtime = profile.get("runtime") or {}
    if profile.get("backend") == "ovms":
        import modelctl_hardware
        snap = modelctl_hardware.capture_hardware_snapshot()
        plans = modelctl_plans.compile_launch_plans(profile, snap)
        if runtime.get("mode") == "managed" and runtime.get("pinned_plan_id"):
            plans = [p for p in plans if p.id == runtime["pinned_plan_id"]]
        # Conservative envelope: max admission claim per resource across
        # eligible plans (peak VRAM including cache, resident RAM).
        env = {}
        for plan in plans:
            for dev, b in plan.claim.vram_admission_bytes().items():
                env[dev] = max(env.get(dev, 0), b)
            ram = plan.claim.ram_admission_bytes()
            if ram:
                env["RAM"] = max(env.get("RAM", 0), ram)
        if not any(env.values()):
            return None  # unknown claim -- do not guess
        return env

    if runtime.get("mode") == "managed":
        import modelctl_hardware
        snap = modelctl_hardware.capture_hardware_snapshot()
        plans = modelctl_plans.compile_launch_plans(profile, snap)
        if runtime.get("pinned_plan_id"):
            plans = [p for p in plans if p.id == runtime["pinned_plan_id"]
                     or p.source == "current-profile"]
        env = {}
        for plan in plans:
            for dev, b in plan.claim.vram_admission_bytes().items():
                env[dev] = max(env.get(dev, 0), b)
            ram = plan.claim.ram_admission_bytes()
            if ram:
                env["RAM"] = max(env.get("RAM", 0), ram)
        return env if any(env.values()) else None

    weights = 0
    try:
        weights = modelctl_vram.weights_bytes_on_disk(profile["model_path"])
    except (OSError, KeyError, TypeError):
        pass
    kv = 0
    meta = modelctl_vram.read_gguf_kv_metadata(profile.get("model_path", "")) \
        if profile.get("model_path") else {}
    params = modelctl_vram.gguf_kv_params(meta)
    if params:
        kv = modelctl_vram.kv_cache_bytes(
            params, ctx, cfg.get("cache_type_k", "q8_0"),
            cfg.get("cache_type_v") or cfg.get("cache_type_k", "q8_0"))
    overhead = max(1 << 30, int(weights * 0.10)) if weights else 1 << 30
    total = weights + kv + overhead

    extra = cfg.get("extra", "") or ""
    cpu_share = 0
    if "exps=CPU" in extra or "-ngl" in extra:
        cpu_share = int(weights * 0.5)  # rough: offload profiles stream about half

    device = cfg.get("device", "")
    split = cfg.get("tensor_split", "")
    claim = {}
    if split and "," in split:
        parts = [float(x) for x in split.split(",")]
        devs = [g["device"] for g in inventory[: len(parts)]] or ["SYCL0"]
        denom = sum(parts) or 1
        for dev, frac in zip(devs, parts):
            claim[dev] = int((total - cpu_share) * frac / denom)
    else:
        claim[device or (inventory[0]["device"] if inventory else "SYCL0")] = total - cpu_share
    if cpu_share:
        claim["RAM"] = cpu_share

    # Peak admission includes the dynamic expert-cache budget, same rule
    # as ResourceClaim.vram_admission_bytes(): the runtime allocates one
    # UNIFORM per-GPU budget (the max of the declared ones) on every
    # device the profile names.
    mc = profile.get("moe_cache", {})
    if mc.get("mode", "off") != "off":
        budgets_cfg = mc.get("gpu", {}).get("budgets_bytes", {})
        declared = {d: b for d, b in budgets_cfg.items() if b > 0}
        uniform = max(declared.values(), default=0)
        for dev in declared:
            claim[dev] = claim.get(dev, 0) + uniform
    return claim


def _budgets(inventory, defaults):
    """Stable budgets: card capacity * limit minus any configured hardware
    reserves (hardware.json), and RAM from TOTAL memory minus reserve -- not
    transient free RAM, which would make the matrix flap with desktop load."""
    import modelctl_hardware
    settings = modelctl_hardware.load_settings()
    dev_cfg = settings.get("devices", {})
    frac = defaults["vram_limit_pct"] / 100.0
    budgets = {}
    for d in inventory:
        reserve = dev_cfg.get(d["device"], {}).get("reserve_bytes", 0)
        budgets[d["device"]] = d["total_bytes"] * frac - reserve
    ram_reserve = (settings.get("ram") or {}).get("reserve_bytes",
                                                 4 * (1 << 30))
    budgets["RAM"] = max(0, modelctl_hardware._system_ram() - ram_reserve)
    return budgets


def _fits(claims, names, budgets):
    need = {}
    for n in names:
        for res, b in claims[n].items():
            need[res] = need.get(res, 0) + b
    return all(need.get(res, 0) <= budgets.get(res, 0) for res in need)


def _evict_cost(profile_name, rdb):
    runs = [r for r in rdb.plan_runs_for(profile_name, limit=10)
            if r.get("success") and r.get("load_seconds")]
    if not runs:
        return 1
    median = statistics.median(r["load_seconds"] for r in runs)
    return max(1, min(100, round(median / 5)))


def generate_matrix(inventory=None, defaults=None):
    """Propose managed matrix entries from current profiles and claims.
    Returns {vars, evict_costs, sets, claims, excluded}."""
    defaults = defaults or modelctl.load_defaults()
    inventory = inventory or modelctl.get_gpu_inventory()
    budgets = _budgets(inventory, defaults)
    primary_budget = max((budgets[d["device"]] for d in inventory), default=0)

    rdb = modelctl_runtime.RuntimeDB()
    profiles = [p for p in
                (modelctl.load_profile(f.stem)
                 for f in sorted(modelctl.PROFILES_DIR.glob("*.json")))
                if p.get("enabled", True)]

    claims, excluded = {}, []
    for p in profiles:
        try:
            claim = profile_claim(p, inventory, defaults)
            if claim is None or not any(claim.values()):
                excluded.append({"name": p["name"],
                                 "reason": "claim unknown -- not guessed"})
                continue
            claims[p["name"]] = claim
        except Exception as e:
            excluded.append({"name": p["name"], "reason": str(e)})

    large, helpers = [], []
    for name, claim in claims.items():
        total = sum(claim.values())
        if total > primary_budget * HELPER_FRAC:
            large.append(name)
        else:
            helpers.append(name)

    sets = {}
    # every model alone (fallback; llama-swap permits subsets of sets)
    for name in claims:
        if _fits(claims, [name], budgets):
            sets[f"mc_{name}"] = name
        else:
            excluded.append({"name": name, "reason": "exceeds budgets alone"})
    # large + one helper pairs
    for big in large:
        for small in helpers:
            if big == small:
                continue
            if _fits(claims, [big, small], budgets):
                sets[f"mc_{big}_plus_{small}"] = f"{big} & {small}"
    # helper-only combos (all helpers together if they fit, else pairs)
    if len(helpers) > 1 and _fits(claims, helpers, budgets):
        sets["mc_helpers_all"] = " & ".join(sorted(helpers))
    else:
        for i, a in enumerate(helpers):
            for b in helpers[i + 1:]:
                if _fits(claims, [a, b], budgets):
                    sets[f"mc_{a}_plus_{b}"] = f"{a} & {b}"

    return {
        "vars": {f"mc_{name}": name for name in claims},
        "evict_costs": {f"mc_{name}": _evict_cost(name, rdb) for name in claims},
        "sets": sets,
        "claims": claims,
        "excluded": excluded,
    }


def merge_matrix(existing, generated, managed_only=True):
    """Merge generated entries into the existing matrix dict. Managed entries
    (mc_ prefix) are fully replaced inside vars/evict_costs/sets; hand-authored
    keys AND any unknown matrix-level fields survive untouched."""
    matrix = dict(existing or {})
    out = {k: v for k, v in matrix.items()
           if k not in ("vars", "evict_costs", "sets")}
    for section in ("vars", "evict_costs", "sets"):
        old = matrix.get(section, {}) or {}
        kept = {k: v for k, v in old.items() if not k.startswith(MANAGED_PREFIX)}
        kept.update(generated[section])
        out[section] = kept
    return out
