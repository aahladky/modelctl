"""Tier-aware placement planner for modelctl.

Decides how a profile's model should be spread across the machine's memory
tiers and emits the llama-server flags that realize it:

  tier 1 -- fits the primary GPU alone          -> pin: --device SYCL0
  tier 2 -- fits all GPUs combined              -> layer split by capacity
  tier 3 -- fits GPUs + system RAM              -> expert/layer spill to CPU,
                                                   --no-mmap (fully resident)
  tier 4 -- needs SSD streaming beyond RAM      -> same layout, mmap on

MoE models get expert-granular placement (the highest-value split: routed
experts are ~90%+ of weight bytes and run cold), dense models get a computed
-ngl. GPU-resident expert layers are assigned greedily, fastest-bandwidth
tier first -- optimal for the linear decode-time cost sum(bytes/bandwidth).
Shared experts (_shexp, active every token) and leading dense layers
(GLM/DeepSeek pattern) are pinned to a GPU by explicit -ot rules -- only
routed experts ever flow to the CPU/offload tiers (P5 hard rule, landscape
doc RQ7).

Hard-won llama.cpp quirks this module encodes (each cost a debugging session):
  * -ot overrides are FIRST-match-wins -> specific layer ranges before the
    CPU catch-all.
  * Placing tensors on a second SYCL device requires --split-mode layer AND
    an explicit --device list; with split-mode none the scheduler has no
    backend for SYCL1 and aborts ("buffer cannot run the operation").
  * llama.cpp's --fit simulation crashes when -ot spans multiple devices,
    so multi-device plans always carry --fit off.
  * shlex.split (build_server_args) eats single backslashes, so regexes here
    are written with doubled backslashes to survive to llama-server intact.

Pure planner: no modelctl import, no printing, no side effects. Callers pass
profiles/inventory in and get plan dicts (or None) back.
"""
import math
import re
import shlex

import modelctl_vram

# Approximate usable memory bandwidth (GB/s) per tier, used to order GPUs
# when assigning expert layers (fastest first). Matched by substring against
# the card name reported by xpu-smi; unknown cards get the default.
_GPU_BANDWIDTH_TABLE = [
    ("B70", 608.0),
    ("BMG G31", 608.0),  # Vulkan-style name for the B70 die
    ("A770", 560.0),
    ("B580", 456.0),
    ("B60", 456.0),
    ("B65", 456.0),
    ("BMG G21", 456.0),  # Vulkan-style name for the B580/B60 die
    ("B50", 384.0),
    ("A750", 512.0),
]
DEFAULT_GPU_BANDWIDTH_GBS = 400.0
# Tiers below the GPUs; only their relative order vs GPU bandwidth matters.
CPU_BANDWIDTH_GBS = 75.0

# Remote (RPC) rungs, calibrated on the 2026-08-03 122B fleet A/B
# (docs/evidence/2026-08-03-qwen122b-fleet-calibration.md): experts on the
# remote GPU rank above local RAM, experts in remote RAM above SSD but
# below local RAM. The absolute numbers only encode that order -- the
# greedy assigner consumes rank, not throughput.
RPC_GPU_BANDWIDTH_GBS = 200.0
RPC_CPU_BANDWIDTH_GBS = 60.0
# Never ship less than this to a node: the fixed per-token wire cost
# (~2 ms/MB measured in the calibration) swamps the bandwidth win on
# tiny placements.
MIN_RPC_PLACEMENT_BYTES = 2 * (1 << 30)

# Per-GPU runtime reserve: compute buffers, graph workspace, allocator
# slack. llama.cpp's own needs scale with the model, and MoE expert matmuls
# are hungry -- too small a reserve is a load-time Level Zero OOM (learned
# the hard way with Ornith-397B on the B580).
GPU_COMPUTE_RESERVE_BYTES = int(1.5 * (1 << 30))

# Don't plan the CPU tier right up to MemAvailable -- leave headroom for the
# OS, desktop, and llama-swap itself.
RAM_RESERVE_BYTES = 4 * (1 << 30)

# Flags the planner owns (and therefore strips from a profile's existing
# extra before re-emitting). Anything else in extra is preserved verbatim.
_PLACEMENT_FLAG_ARITY = {"-ot": 1, "-ngl": 1, "--fit": 1, "--device": 1,
                         "--no-mmap": 0}


# KV heuristic for QUANT SELECTION only. The vram module's 256 KiB/token is a
# deliberate over-estimate for OOM guarding; using it here would penalize big
# contexts into needlessly tiny quants. 48 KiB/token approximates a modern
# GQA model (~32 layers, 4-8 KV heads, 128 dim, q8_0) with a little margin.
SELECT_KV_BYTES_PER_TOKEN = 48 * 1024


# Context steps for auto-sizing: chosen from largest to smallest, so the
# first one whose KV cache fits the budget wins. Powers-of-two-ish keeps the
# advertised context predictable for clients.
CTX_STEPS = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

AUTO_CTX_FLOOR = 8192


def auto_ctx(model_path, budget_bytes, cache_type_k="q8_0", cache_type_v=None,
             weights_bytes=None):
    """Largest context (from CTX_STEPS, capped at the model's own max) whose
    weights + exact KV + overhead fit budget_bytes. The KV math is EXACT once
    the GGUF is on disk -- including sliding-window patterns -- so this
    replaces per-model ctx guessing with the actual memory equation.

    Returns {ctx, kv_bytes, model_max, fits, note}; fits=False when even the
    floor context doesn't fit (caller should re-plan tiers).
    """
    meta = modelctl_vram.read_gguf_kv_metadata(model_path)
    if not meta:
        return None
    arch = meta.get("general.architecture", "")
    model_max = meta.get(f"{arch}.context_length") or 0
    params = modelctl_vram.gguf_kv_params(meta)
    if weights_bytes is None:
        try:
            weights_bytes = modelctl_vram.weights_bytes_on_disk(model_path)
        except OSError:
            return None
    overhead = modelctl_vram.compute_overhead_bytes(weights_bytes)
    ctv = cache_type_v or cache_type_k

    def kv_for(ctx):
        if params:
            return modelctl_vram.kv_cache_bytes(params, ctx, cache_type_k, ctv)
        return ctx * SELECT_KV_BYTES_PER_TOKEN

    for ctx in sorted((s for s in CTX_STEPS if s <= model_max), reverse=True):
        kv = kv_for(ctx)
        if weights_bytes + kv + overhead <= budget_bytes:
            return {"ctx": ctx, "kv_bytes": kv, "model_max": model_max,
                    "fits": True,
                    "note": f"largest context fitting the budget"
                            + (f" (model max {model_max})" if ctx == model_max else "")}
    kv = kv_for(AUTO_CTX_FLOOR)
    return {"ctx": AUTO_CTX_FLOOR, "kv_bytes": kv, "model_max": model_max,
            "fits": False,
            "note": f"even {AUTO_CTX_FLOOR} ctx doesn't fit the budget -- "
                    "re-plan with `place --tiers` or shrink KV quant"}


def recommend_quant_group(groups, budget_bytes, ctx, kv_per_token=None):
    """Pick a quant group for zero-config pulls: the LARGEST whose estimated
    footprint (weights + heuristic KV + overhead) fits the tier-1 budget
    (best quality that stays fully on the primary GPU). When nothing fits,
    falls back to the SMALLEST quant -- the least SSD streaming, which is the
    right failure mode. imatrix files are data, not servable models, and are
    excluded.

    groups        -- get_repo_contents()['quant_groups'] entries
    budget_bytes  -- usable bytes on the primary GPU (total * limit_pct)
    Returns {group, fits, est_total, reason} or None when no candidates.
    """
    kv_per_token = kv_per_token or SELECT_KV_BYTES_PER_TOKEN
    candidates = [g for g in groups
                  if g.get("total_size") and "imatrix" not in g["label"].lower()]
    if not candidates:
        return None

    def footprint(g):
        w = g["total_size"]
        return w + ctx * kv_per_token + modelctl_vram.compute_overhead_bytes(w)

    fitting = [g for g in candidates if footprint(g) <= budget_bytes]
    if fitting:
        best = max(fitting, key=lambda g: g["total_size"])
        return {"group": best, "fits": True, "est_total": footprint(best),
                "reason": "largest quant that fits the primary GPU"}
    smallest = min(candidates, key=lambda g: g["total_size"])
    return {"group": smallest, "fits": False, "est_total": footprint(smallest),
            "reason": "nothing fits the primary GPU -- smallest quant, will "
                      "need offload (re-plan with `place --tiers` after pull)"}


def gpu_bandwidth_gbs(card_name):
    """Effective bandwidth for a card by name substring; default when unknown."""
    for needle, bw in _GPU_BANDWIDTH_TABLE:
        if needle.lower() in (card_name or "").lower():
            return bw
    return DEFAULT_GPU_BANDWIDTH_GBS


def split_extra_flags(extra):
    """Split a profile's extra-flags string into (placement, other) token
    lists. Placement flags are ones the planner generates itself -- stripping
    them makes re-planning idempotent instead of accreting stale -ot/-ngl."""
    toks = shlex.split(extra or "")
    placement, other = [], []
    i = 0
    while i < len(toks):
        tok = toks[i]
        arity = _PLACEMENT_FLAG_ARITY.get(tok)
        if arity is None:
            other.append(tok)
            i += 1
        else:
            placement.extend(toks[i:i + 1 + arity])
            i += 1 + arity
    return placement, other


def _same_digit_range_regex(lo, hi):
    """Regex fragment for integers lo..hi where both have the same digit count."""
    s_lo, s_hi = str(lo), str(hi)
    n = len(s_lo)

    def rec(pos, floor, ceil):
        if pos == n:
            return ""
        d_lo = int(s_lo[pos]) if floor else 0
        d_hi = int(s_hi[pos]) if ceil else 9
        if floor and ceil and d_lo == d_hi:
            return str(d_lo) + rec(pos + 1, True, True)
        if pos == n - 1:
            # Final digit: any floor/ceil mix collapses to one range.
            if d_lo == d_hi:
                return str(d_lo)
            if d_lo == 0 and d_hi == 9:
                return "[0-9]"
            return f"[{d_lo}-{d_hi}]"
        if not floor and not ceil:
            if d_lo == 0 and d_hi == 9:
                return "[0-9]" * (n - pos)
            return f"[{d_lo}-{d_hi}]" + "[0-9]" * (n - pos - 1)
        parts = []
        for d in range(d_lo, d_hi + 1):
            if d == d_lo and floor:
                parts.append(str(d) + rec(pos + 1, True, False))
            elif d == d_hi and ceil:
                parts.append(str(d) + rec(pos + 1, False, True))
            else:
                parts.append(str(d) + "[0-9]" * (n - pos - 1))
        if len(parts) == 1:
            return parts[0]
        return "(?:" + "|".join(parts) + ")"

    return rec(0, True, True)


def range_regex(a, b):
    """Regex fragment matching exactly the integers a..b (inclusive)."""
    if a > b:
        a, b = b, a
    if a == b:
        return str(a)
    # Split into sub-ranges of uniform digit count (0-9, 10-99, ...).
    parts = []
    start = a
    while start <= b:
        end = min(b, 10 ** len(str(start)) - 1)
        parts.append(_same_digit_range_regex(start, end))
        start = end + 1
    if len(parts) == 1:
        return parts[0]
    return "(?:" + "|".join(parts) + ")"


def layers_regex(layers):
    """Regex fragment matching any of the given layer indices, with
    contiguous runs compressed (e.g. [1..19] -> '(?:[1-9]|1[0-9])')."""
    layers = sorted(set(layers))
    if not layers:
        return "(?!)"  # matches nothing
    runs = []
    run_start = prev = layers[0]
    for n in layers[1:]:
        if n == prev + 1:
            prev = n
        else:
            runs.append((run_start, prev))
            run_start = prev = n
    runs.append((run_start, prev))
    parts = [range_regex(a, b) for a, b in runs]
    if len(parts) == 1:
        return parts[0]
    return "(?:" + "|".join(parts) + ")"


def _gib(n):
    return n / (1 << 30)


# --- stable planning inputs ------------------------------------------------
# Everything plan_tiers reads from the live machine, captured as one
# explicit record: defaulted from the machine on a profile's first plan,
# stored in the profile, and reused on replan. Without this, replanning
# reads instantaneous free RAM and can demote a profile a tier and rewrite
# its split though nothing about the profile changed.

PLANNING_INPUTS_VERSION = 1


def stored_planning_inputs(profile):
    """The planner inputs recorded in a profile, or None (legacy profile
    that has never planned with recorded inputs)."""
    inputs = (profile.get("planning") or {}).get("inputs")
    return inputs if isinstance(inputs, dict) and inputs else None


def make_planning_inputs(inventory, limit_pct, primary, ram_available_bytes,
                         hw_settings=None, capabilities=None, display=None,
                         fleet_budgets=None):
    """Canonicalize live machine values into the recorded-input schema.

    Only what plan_tiers actually reads is recorded -- with one stated
    exception -- because anything else is dead weight that still forces a
    diff when it drifts.

    `display` is that exception: the desktop-up/headless mode the plan
    was built under (modelctl_display.planning_input()). plan_tiers does
    not read it and no budget spends it; it is recorded so a launch can
    tell the operator that the machine is not the one the plan was built
    for. A profile planned before this field existed simply has no
    display block, which reads as "no claim".

    `fleet_budgets` is not an exception: a remote device's operator
    budget is spent by the same admission path a local device's VRAM is
    (modelctl_fleet.admission_budgets -> _usable_vram_map), so it is a
    planner input in the strict sense, and a plan built when a node
    offered 16 GiB is not a plan for a node offering 24. Presence is
    deliberately NOT part of it -- see modelctl_fleet.budget_input."""
    return {
        "version": PLANNING_INPUTS_VERSION,
        "ram_available_bytes": int(ram_available_bytes or 0),
        "vram_limit_pct": limit_pct,
        "primary": primary,
        "inventory": [{"device": d["device"], "name": d.get("name", ""),
                       "total_bytes": int(d.get("total_bytes", 0) or 0)}
                      for d in inventory],
        "hw_settings": {"devices": dict((hw_settings or {}).get("devices", {}))},
        "capabilities": {"features": {
            "moe_cache_per_device_budgets": bool(
                ((capabilities or {}).get("features") or {})
                .get("moe_cache_per_device_budgets"))}},
        "display": {
            "mode": str((display or {}).get("mode") or "unknown"),
            "freed_bytes": int((display or {}).get("freed_bytes") or 0),
        },
        "fleet": {"budgets_bytes": {str(k): int(v) for k, v
                                    in sorted((fleet_budgets or {}).items())}},
    }


def planning_input_fleet_budgets(inputs):
    """The fleet budgets recorded with one profile's planner inputs, or
    None for a record written before they were an input.

    None and {} are different answers and the caller must keep them
    apart: {} is "there were no fleet budgets when this was planned",
    which a node enrolled later contradicts; None is "this record
    predates the field", which nothing contradicts."""
    fleet = (inputs or {}).get("fleet")
    if not isinstance(fleet, dict):
        return None
    budgets = fleet.get("budgets_bytes")
    if not isinstance(budgets, dict):
        return None
    return {str(k): int(v) for k, v in budgets.items()}


def fleet_budget_mismatch(inputs, live_budgets):
    """One operator-readable line when a profile's recorded fleet budgets
    are not the budgets in force now, else None.

    Same contract as display_input_mismatch: a statement about the
    recorded inputs, never a block. The plan still launches on the
    budget it was built for -- what the operator needs to know is that a
    replan would now place differently."""
    recorded = planning_input_fleet_budgets(inputs)
    if recorded is None:
        return None
    live = {str(k): int(v) for k, v in (live_budgets or {}).items()}
    changed = sorted(set(recorded) | set(live))
    parts = []
    for key in changed:
        was, now = recorded.get(key), live.get(key)
        if was == now:
            continue
        parts.append(f"{key}: "
                     f"{'none' if was is None else f'{_gib(was):.2f} GiB'} -> "
                     f"{'none' if now is None else f'{_gib(now):.2f} GiB'}")
    if not parts:
        return None
    return ("fleet budgets changed since this plan's inputs were recorded ("
            + "; ".join(parts) + "); the stored plan still places against "
            "the old numbers -- replan with --refresh-inputs to use the new "
            "ones")


def planning_input_display_mode(inputs):
    """The display mode recorded with one profile's planner inputs, or
    "unknown" (legacy record, or a machine that never toggled)."""
    mode = ((inputs or {}).get("display") or {}).get("mode")
    return mode if mode in ("graphical", "headless") else "unknown"


def display_input_mismatch(inputs, live_mode):
    """One operator-readable line when a plan's recorded display mode is
    not the mode the machine is in now, else None.

    Both sides must make a claim: "unknown" on either side is silence,
    not a mismatch, so profiles planned before display was recorded --
    and machines that have never toggled -- say nothing. This is a
    statement about the recorded inputs, never a block: the plan still
    launches, and the operator decides whether to replan."""
    recorded = planning_input_display_mode(inputs)
    live = live_mode if live_mode in ("graphical", "headless") else "unknown"
    if recorded == "unknown" or live == "unknown" or recorded == live:
        return None
    freed = int(((inputs or {}).get("display") or {}).get("freed_bytes") or 0)
    detail = f" (~{_gib(freed):.2f} GiB measured)" if freed else ""
    return (f"planning inputs were recorded with the display {recorded}, "
            f"but the machine is {live} now{detail}; free VRAM differs from "
            f"what this plan was built against -- replan with "
            f"--refresh-inputs to rebuild it for this mode")


def record_planning_inputs(profile, inputs, recorded_at=None):
    """Write the resolved inputs into the profile (in place). Returns True
    when the profile changed."""
    planning = profile.setdefault("planning", {})
    if planning.get("inputs") == inputs:
        return False
    planning["inputs"] = inputs
    if recorded_at:
        planning["recorded_at"] = recorded_at
    return True


# --- replan diff classification --------------------------------------------

def _ot_parts(extra):
    """-ot parts of an extra string as [(raw_pattern, target)], in flag
    order, WITHOUT shlex de-escaping (raw profile text, comparable across
    plans)."""
    toks = (extra or "").split()
    parts = []
    for i, tok in enumerate(toks):
        if tok == "-ot" and i + 1 < len(toks):
            for part in toks[i + 1].split(","):
                pattern, _, target = part.rpartition("=")
                if pattern and target:
                    parts.append((pattern, target))
    return parts


def _is_pin_rule(pattern):
    """Pin rules place shared experts (_shexp) or leading dense layers
    (whole-layer ffn) on a GPU; routed-expert (_exps) rules are placement
    structure. P5 pins may be added by a replan without changing where any
    routed expert or attention byte lives."""
    return "_shexp" in pattern or "_exps" not in pattern


def _placement_view(extra):
    """Structural placement content of an extra string, pins separated."""
    parts = _ot_parts(extra)
    toks = shlex.split(extra or "")
    view = {"pins": [p for p in parts if _is_pin_rule(p[0])],
            "routed": [p for p in parts if not _is_pin_rule(p[0])],
            "ngl": None, "no_mmap": "--no-mmap" in toks,
            "device_list": "", "fit": "", "other": []}
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok == "-ngl" and i + 1 < len(toks):
            view["ngl"] = toks[i + 1]
            i += 2
        elif tok == "--device" and i + 1 < len(toks):
            view["device_list"] = toks[i + 1]
            i += 2
        elif tok == "--fit" and i + 1 < len(toks):
            view["fit"] = toks[i + 1]
            i += 2
        elif tok in ("--no-mmap", "-ot"):
            i += 2 if tok == "-ot" else 1
        else:
            view["other"].append(tok)
            i += 1
    return view


def _describe_rules(rules):
    """A rule list as people read it: where things go, not the regexes.

    The exact patterns are machine-generated and visible in the profile;
    the diff only needs to say what moved at a glance.
    """
    if not rules:
        return "(none)"
    targets = []
    counts = {}
    for _pattern, device in rules:
        if device not in counts:
            targets.append(device)
        counts[device] = counts.get(device, 0) + 1
    body = ", ".join(f"{d} x{counts[d]}" if counts[d] > 1 else d
                     for d in targets)
    n = len(rules)
    return f"{n} rule{'s' if n != 1 else ''} ({body})"


_DIFF_LABELS = {
    "routed": "expert placement",
    "ngl": "GPU layer count (-ngl)",
    "no_mmap": "memory-map mode",
    "device_list": "devices",
    "fit": "auto-fit",
    "other": "other flags",
}


def _describe_view_value(key, value):
    if key == "routed":
        return _describe_rules(value)
    if key == "no_mmap":
        return "off" if value else "on"
    if isinstance(value, list):
        return " ".join(str(v) for v in value) or "(none)"
    return str(value) if value not in (None, "") else "(none)"


def classify_config_diff(old_cfg, new_cfg):
    """Classify a replan's config diff: "none", "pins", or "structural".

    "pins" means only P5 pin rules (shexp / leading-dense -ot overrides)
    differ -- tier, split, device, routed-expert placement, -ngl, mmap
    mode, and every non-placement flag are unchanged. Anything beyond that
    is "structural" and (in the apply paths) requires an explicit
    --accept-tier-change. The change strings are user-facing (CLI and
    console) -- keep them in words, never raw rule-list reprs."""
    changes = []
    structural = False
    for key in ("device", "split_mode", "tensor_split"):
        old, new = old_cfg.get(key, "") or "", new_cfg.get(key, "") or ""
        if old != new:
            structural = True
            changes.append(f"{key}: {old or '(none)'} -> {new or '(none)'}")
    old_v = _placement_view(old_cfg.get("extra", ""))
    new_v = _placement_view(new_cfg.get("extra", ""))
    for key in ("routed", "ngl", "no_mmap", "device_list", "fit", "other"):
        if old_v[key] != new_v[key]:
            structural = True
            changes.append(f"{_DIFF_LABELS[key]}: "
                           f"{_describe_view_value(key, old_v[key])} -> "
                           f"{_describe_view_value(key, new_v[key])}")
    pins_changed = old_v["pins"] != new_v["pins"]
    if pins_changed:
        changes.append(f"pinned tensors: {_describe_rules(old_v['pins'])} -> "
                       f"{_describe_rules(new_v['pins'])}")
    if structural:
        kind = "structural"
    elif pins_changed:
        kind = "pins"
    else:
        kind = "none"
    return {"kind": kind, "changes": changes}


def _has_placement(cfg):
    """Does this config carry any placement content at all? A profile with
    none is receiving its initial placement, not a placement change."""
    if cfg.get("device") or cfg.get("tensor_split") or cfg.get("split_mode"):
        return True
    placement, _ = split_extra_flags(cfg.get("extra", ""))
    return bool(placement)


def tier_change_gate(profile, plan):
    """Decide whether applying `plan` to `profile` needs an explicit
    --accept-tier-change.

    Returns {"kind", "changes", "requires_accept"}. A no-op or pins-only
    diff never needs the flag; neither does a profile with no existing
    placement (first plan). Everything else -- tier, split, or placement
    beyond pins, including an effective cache-budget change -- does."""
    diff = classify_config_diff(profile.get("config", {}), plan["config"])
    mc = profile.get("moe_cache", {})
    if mc.get("mode", "off") != "off":
        requested = {d: b for d, b in mc.get("gpu", {})
                     .get("budgets_bytes", {}).items() if b > 0}
        effective = plan.get("cache_budgets") or {}
        if effective != requested:
            diff["changes"].append(
                f"moe_cache budgets: {requested} -> {effective}")
            diff["kind"] = "structural"
    requires = (diff["kind"] == "structural"
                and _has_placement(profile.get("config", {})))
    return {"kind": diff["kind"], "changes": diff["changes"],
            "requires_accept": requires}


def plan_tiers(profile, inventory, limit_pct, primary, ram_available=None,
               layout=None, cache_request=None, capabilities=None,
               hw_settings=None, remote_rungs=None):
    """Compute a tiered placement for one profile.

    profile     -- modelctl profile dict (config carries ctx/cache types/extra)
    inventory   -- get_gpu_inventory() output [{device, name, total_bytes, ...}]
    limit_pct   -- per-card capacity budget (e.g. 90)
    primary     -- device name of the primary GPU
    ram_available -- bytes; defaults to live /proc/meminfo reading
    layout      -- optional pre-parsed gguf_model_layout() (tests, batch use)
    cache_request -- optional dict: {mode, gpu: {budgets_bytes: {dev: N}}}
                    When non-None and mode != "off", the planner reserves
                    per-device dynamic cache space before assigning static
                    expert layers -- preventing the cache from overwriting
                    or being overwritten by statically placed experts.
    hw_settings -- parsed hardware settings ({"devices": {dev: {enabled,
                    reserve_bytes}}}); None reads the live settings file.
                    Passed explicitly so a stored planning-inputs record
                    fully determines the plan.
    remote_rungs -- optional RPC devices for the MoE expert ladder, each
                    {"name": admission key, "endpoint": host:port,
                     "local_device_index": int, "kind": "gpu"|"cpu",
                     "budget_bytes": int}. The caller (compile_launch_plans)
                    builds them from the fleet registry; this module stays
                    pure and never opens a socket. Dense spill ignores
                    them -- whole-layer remote ranges are the fleet plan
                    family's job.

    Returns None when the model can't be analyzed; otherwise:
      {tier, config: {device, split_mode, tensor_split, extra},
       layout: [(label, gib, description)], warnings: [...],
       analysis: {weights_gib, kv_gib, is_moe, n_layers, ram_budget_gib,
                  cache_budgets_gib: {...} or None},
       cache_budgets: {dev: bytes} or None,
       admission: {fits, devices: {...}, requested, assumed, chosen,
                   degradations: [...]}}
    cache_budgets is the effective UNIFORM per-GPU reserve the planner
    actually accounted for (None when the cache is off or didn't fit);
    apply paths should write it back to the profile so the server
    allocates exactly what the planner reserved.

    Admission contract: the emitted config is verified per device --
    resident weights + KV at the configured ctx + cache budget + pinned
    expert bytes + the compute reserve against usable capacity -- and an
    oversubscribed plan is never returned as-is: the cache budget shrinks
    toward 0 first, then further expert layers spill to CPU (the replan
    under the smaller cache does the spilling), with every step recorded
    machine-readably in plan["admission"]["degradations"] and echoed in
    plan["warnings"].
    """
    cfg = profile.get("config", {})
    model_path = profile.get("model_path")
    if layout is None:
        if not model_path:
            return None
        layout = modelctl_vram.gguf_model_layout(model_path)
    if not layout or not layout.get("weight_bytes"):
        return None

    meta = layout.get("meta") or {}
    kv_params = modelctl_vram.gguf_kv_params(meta)
    # 8192, matching the schema default (modelctl_profiles._DEFAULT_CONFIG)
    # and every other planner fallback: this one used to assume 32768, so
    # the tier plan was sized for a 4x bigger KV cache than the claim
    # admitted or the launch used.
    ctx = int(cfg.get("ctx", 8192))
    ctk = cfg.get("cache_type_k") or "f16"
    ctv = cfg.get("cache_type_v") or ctk
    if kv_params:
        kv = modelctl_vram.kv_cache_bytes(kv_params, ctx, ctk, ctv)
    else:
        kv = ctx * modelctl_vram.HEURISTIC_KV_BYTES_PER_TOKEN

    if ram_available is None:
        ram_available = modelctl_vram.system_ram_available()
    ram_budget = max(0, ram_available - RAM_RESERVE_BYTES)

    frac = limit_pct / 100.0
    devs = ([d for d in inventory if d["device"] == primary]
            + [d for d in inventory if d["device"] != primary])
    if not devs:
        return None
    # Hardware policy: disabled devices must not receive any placement, and
    # configured reserves shrink the usable budget -- the tier planner must
    # live in the same hardware reality as the worker and matrix.
    if hw_settings is None:
        try:
            import modelctl_hardware
            hw_settings = modelctl_hardware.load_settings()
        except Exception:
            hw_settings = {}
    dev_cfg = (hw_settings or {}).get("devices", {})
    devs = [d for d in devs if dev_cfg.get(d["device"], {}).get("enabled", True)]
    if not devs:
        return None
    if primary not in [d["device"] for d in devs]:
        primary = devs[0]["device"]
    usable_raw = {d["device"]: d["total_bytes"] * frac
                        - dev_cfg.get(d["device"], {}).get("reserve_bytes", 0)
                  for d in devs}

    weights = layout["weight_bytes"]
    overhead = modelctl_vram.compute_overhead_bytes(weights)
    total = weights + kv + overhead
    base_warnings = []
    _, other_flags = split_extra_flags(cfg.get("extra", ""))
    other = " ".join(other_flags)

    if layout.get("unknown_type_tensors"):
        base_warnings.append(
            f"{layout['unknown_type_tensors']} tensors use quant types this "
            "planner doesn't know -- layout math undercounts; check the plan.")

    # Reserve the dynamic cache BEFORE assigning static expert layers, or
    # the runtime cache collides with statically placed experts (OOM).
    # A budget that doesn't fit is SHRUNK toward 0 (never emitted as-is,
    # never silently honored): the degradation order is cache first, then
    # expert spill -- see _reserve_cache_budgets.
    requested_budgets = {}
    if cache_request and cache_request.get("mode", "off") != "off":
        requested_budgets = {d: b for d, b in cache_request.get("gpu", {})
                             .get("budgets_bytes", {}).items() if b > 0}
    per_device_caps = bool((capabilities or {}).get("features", {})
                           .get("moe_cache_per_device_budgets"))
    degradations = []
    cache_warnings = []
    cache_budgets = _reserve_cache_budgets(
        requested_budgets, per_device_caps, usable_raw, layout, kv, devs,
        cache_warnings, degradations)

    def plan_once(cache_now):
        usable = {d: max(0, usable_raw[d] - cache_now.get(d, 0))
                  for d in usable_raw}
        warnings = list(base_warnings) + list(cache_warnings)
        analysis = {"weights_gib": _gib(weights), "kv_gib": _gib(kv),
                    "is_moe": layout["is_moe"],
                    "n_layers": layout["block_count"],
                    "ram_budget_gib": _gib(ram_budget),
                    "cache_budgets_gib": ({d: _gib(b)
                                           for d, b in cache_now.items()}
                                          if cache_now else None)}

        def result(tier, config, layout_rows):
            return {"tier": tier, "config": config, "layout": layout_rows,
                    "warnings": warnings, "analysis": analysis,
                    "cache_budgets": dict(cache_now) if cache_now else None}

        # --- tier 1: primary GPU alone ----------------------------------
        if total <= usable[primary]:
            return result(1, {"device": primary, "split_mode": "",
                              "tensor_split": "", "extra": other},
                          [(f"{primary} (whole model)", _gib(total),
                            "weights + KV + overhead")])

        # --- tier 2: all GPUs --------------------------------------------
        combined = sum(usable.values())
        if len(devs) > 1 and total <= combined:
            # Ratio from USABLE bytes, not raw capacity: llama.cpp
            # distributes the model by these weights, and a device carrying
            # a reserve or the uniform cache reservation must receive
            # proportionally less. A raw-capacity ratio put static bytes
            # into the reserved space -- aggregate admission passed, the
            # small card OOMed at load.
            ts = modelctl_vram.tensor_split_ratio(
                [usable[d["device"]] for d in devs])
            # The emitted ratio is GiB-rounded and GCD-reduced; recheck
            # what each device actually receives under the quantized
            # weights before admitting, and spill instead when quantization
            # overflows someone's budget.
            ts_weights = [int(w) for w in ts.split(",")]
            wsum = sum(ts_weights) or 1
            per_device_fits = all(
                total * w / wsum <= usable[d["device"]]
                for w, d in zip(ts_weights, devs))
            if per_device_fits:
                # Explicit --device list: without one, llama.cpp maps
                # tensor_split positions to its own enumeration order,
                # while this ratio is in devs order (primary first) -- a
                # non-SYCL0 primary would receive the wrong share at
                # runtime.
                dev_list = ",".join(d["device"] for d in devs)
                extra2 = f"--device {dev_list}" + (f" {other}" if other else "")
                return result(2, {"device": "", "split_mode": "layer",
                                  "tensor_split": ts, "extra": extra2},
                              [("all GPUs", _gib(total),
                                f"layer split {ts} by usable VRAM")])

        # --- tiers 3/4: spill to CPU (RAM), maybe SSD streaming ----------
        tier = 3 if total <= combined + ram_budget else 4
        if tier == 4:
            warnings.append(
                "model exceeds GPU+RAM: the CPU-resident portion streams "
                "from SSD via mmap -- expect low single-digit tok/s on "
                "cold cache.")

        if layout["is_moe"]:
            return _plan_moe_spill(
                tier, layout, devs, usable, primary, kv, ram_budget, other,
                warnings, analysis, result, cache_now,
                remote_rungs=remote_rungs)
        return _plan_dense_spill(
            tier, layout, devs, usable, primary, kv, ram_budget, other,
            warnings, analysis, result)

    # Admission-and-degradation loop: verify the EMITTED config per device
    # and shrink the cache until it fits (the replan under a smaller cache
    # spills further expert layers on its own). Bounded: each pass either
    # fits or strictly shrinks the cache, and a zero cache ends the loop.
    plan = report = None
    for _ in range(4):
        plan = plan_once(cache_budgets)
        report = _admission_report(plan, layout, devs, usable_raw, kv,
                                   cache_budgets)
        if report["fits"] or not any(cache_budgets.values()):
            break
        overflow = {d: r["demand_bytes"] - r["usable_bytes"]
                    for d, r in report["devices"].items() if not r["fits"]}
        if per_device_caps:
            new_budgets = {}
            for d, b in cache_budgets.items():
                got = max(0, b - overflow.get(d, 0))
                if got != b:
                    degradations.append({
                        "action": "shrink_cache", "device": d,
                        "requested_bytes": int(b), "chosen_bytes": int(got),
                        "reason": "emitted plan oversubscribed the device"})
                    cache_warnings.append(_cache_fallback_warning(
                        d, b, usable_raw.get(d, 0), got))
                if got:
                    new_budgets[d] = got
        else:
            uniform = max(cache_budgets.values())
            got = max(0, uniform - max(overflow.values()))
            degradations.append({
                "action": "shrink_cache", "device": "*",
                "requested_bytes": int(uniform), "chosen_bytes": int(got),
                "reason": "emitted plan oversubscribed a device; the "
                          "backend applies one per-GPU budget to every "
                          "device"})
            cache_warnings.append(_cache_fallback_warning(
                "every device", uniform, min(usable_raw.values()), got))
            new_budgets = {d: got for d in cache_budgets} if got else {}
        cache_budgets = new_budgets

    if not report["fits"]:
        bad = sorted(d for d, r in report["devices"].items() if not r["fits"])
        plan["warnings"].append(
            f"plan oversubscribes {', '.join(bad)} even with the cache "
            "disabled and every spillable expert on CPU -- the fixed set "
            "(attention/KV/overhead) does not fit; shrink ctx or the KV "
            "quant. This plan is expected to fail VRAM admission.")

    plan["admission"] = {
        "fits": report["fits"],
        "devices": report["devices"],
        "requested": {
            "cache_budgets_bytes": {d: int(b)
                                    for d, b in requested_budgets.items()},
            "ctx": ctx,
        },
        "assumed": {
            "vram_limit_pct": limit_pct,
            "ram_available_bytes": int(ram_available),
            "capacity_bytes": {d["device"]: int(d["total_bytes"])
                               for d in devs},
            "reserve_bytes": {d["device"]: int(dev_cfg.get(d["device"], {})
                                               .get("reserve_bytes", 0))
                              for d in devs},
        },
        "chosen": {
            "tier": plan["tier"],
            "cache_budgets_bytes": {d: int(b)
                                    for d, b in cache_budgets.items()},
        },
        "degradations": degradations,
    }
    return plan


def _cache_fallback_warning(where, requested, usable, chosen):
    """The approved contract copy: requested, measured/assumed, fallback."""
    req = f"{requested / (1 << 30):.1f} GiB"
    have = f"{usable / (1 << 30):.1f} GiB"
    if chosen > 0:
        return (f"requested expert cache of {req} on {where} exceeds the "
                f"free VRAM measured ({have} usable) -- launch will fall "
                f"back to a smaller cache ({chosen / (1 << 30):.1f} GiB)")
    return (f"requested expert cache of {req} on {where} exceeds the free "
            f"VRAM measured ({have} usable) -- the cache is disabled for "
            "this plan")


def _reserve_cache_budgets(requested, per_device_caps, usable, layout, kv,
                           devs, warnings, degradations):
    """Per-device cache reservation with shrink-to-fit semantics.

    Two runtime contracts, and the reservation must match whichever one
    the backend implements:
      schema 3+ (per_device_budgets): --moe-cache-bytes takes a device
        map. Each named device is reserved its own budget; unnamed
        devices get no cache, and none is reserved for them.
      older: ONE global budget is applied to every device that creates a
        cache, so per-device requests collapse to their max, reserved on
        every participating device.

    A budget that exceeds what the device can give is never honored as-is
    and never silently dropped: it shrinks to the device's usable bytes
    here (the coarse cap), and the post-plan admission loop in plan_tiers
    shrinks it further if the emitted config still oversubscribes.
    Shrinking beats disabling: the plan keeps as much cache as admission
    allows, which is the exact fallback contract the console copy
    promises."""
    if not requested:
        return {}
    known = {d: b for d, b in requested.items() if d in usable}
    if per_device_caps:
        unknown = sorted(set(requested) - set(known))
        if unknown:
            warnings.append(
                f"cache budget names {', '.join(unknown)}, which this "
                "plan does not use -- those budgets are ignored")
        chosen = {}
        for d, b in sorted(known.items()):
            cap = max(0, int(usable[d]))
            got = min(b, cap)
            if got < b:
                degradations.append({
                    "action": "shrink_cache", "device": d,
                    "requested_bytes": int(b), "chosen_bytes": int(got),
                    "reason": "requested budget exceeds the usable VRAM "
                              "assumed for this device"})
                warnings.append(_cache_fallback_warning(d, b, usable[d], got))
            if got > 0:
                chosen[d] = got
        return chosen

    uniform = max(requested.values())
    cap = min(max(0, int(usable[d])) for d in usable)
    got = min(uniform, cap)
    if got < uniform:
        degradations.append({
            "action": "shrink_cache", "device": "*",
            "requested_bytes": int(uniform), "chosen_bytes": int(got),
            "reason": "this backend applies one per-GPU budget to every "
                      "device, and the requested figure exceeds what the "
                      "smallest device can give"})
        warnings.append(_cache_fallback_warning(
            "every device", uniform, min(usable.values()), got))
    return {d: got for d in usable} if got > 0 else {}


def _admission_report(plan, layout, devs, usable, kv, cache_budgets):
    """Per-device admission math for an EMITTED tier-plan config.

    Recomputed from the config itself (tensor split ratio, -ot rules,
    -ngl) rather than from planner intermediates, so the report can only
    describe what llama.cpp will actually allocate: the fixed set
    distributed by the tensor-split ratio, statically pinned experts by
    first-match -ot rules, the cache budget, and (spill tiers) the
    per-card compute reserve. Values are bytes; `fits` is per device and
    overall."""
    config = plan["config"]
    tier = plan["tier"]
    by_dev = {d["device"]: d for d in devs}
    toks = shlex.split(config.get("extra", "") or "")

    device_list = None
    ngl = None
    for i, tok in enumerate(toks):
        if tok == "--device" and i + 1 < len(toks):
            device_list = toks[i + 1].split(",")
        elif tok == "-ngl" and i + 1 < len(toks):
            try:
                ngl = int(toks[i + 1])
            except ValueError:
                pass
    if config.get("device"):
        order = [config["device"]]
    elif device_list:
        order = device_list
    else:
        order = [d["device"] for d in devs]

    split = config.get("tensor_split", "")
    if split and "," in split:
        ratio = [float(x) for x in split.split(",")]
    else:
        ratio = [1.0]
    rsum = sum(ratio) or 1.0
    shares = {dev: (ratio[i] / rsum if i < len(ratio) else 0.0)
              for i, dev in enumerate(order)}

    rules = []
    for i, tok in enumerate(toks):
        if tok == "-ot" and i + 1 < len(toks):
            for part in toks[i + 1].split(","):
                pattern, _, target = part.rpartition("=")
                if not pattern or not target:
                    continue
                try:
                    rules.append((re.compile(pattern), target))
                except re.error:
                    continue

    expert_pins = {}
    cpu_expert_bytes = 0
    spill = tier >= 3
    if layout.get("is_moe") and spill:
        for layer, nbytes in layout["expert_bytes_per_layer"].items():
            probe = f"blk.{layer}.ffn_gate_exps.weight"
            target = order[0]
            for rx, tgt in rules:
                if rx.search(probe):
                    target = tgt
                    break
            if target == "CPU":
                cpu_expert_bytes += nbytes
            else:
                expert_pins[target] = expert_pins.get(target, 0) + nbytes

    if not spill:
        # Whole model GPU-resident: everything (experts included) follows
        # the split ratio; overhead over all weights already covers the
        # compute buffers, so no separate reserve line.
        gpu_weights = layout["weight_bytes"]
        overhead = modelctl_vram.compute_overhead_bytes(gpu_weights)
        reserve = 0
    elif layout.get("is_moe"):
        gpu_weights = layout["non_expert_bytes"]
        overhead = modelctl_vram.compute_overhead_bytes(gpu_weights + kv)
        reserve = GPU_COMPUTE_RESERVE_BYTES
    else:
        block_count = layout.get("block_count") or 1
        per_layer = (layout["weight_bytes"] - layout["other_bytes"]) / block_count
        gpu_weights = layout["other_bytes"] + (ngl or 0) * per_layer
        overhead = modelctl_vram.compute_overhead_bytes(layout["weight_bytes"])
        reserve = GPU_COMPUTE_RESERVE_BYTES

    devices = {}
    fits = True
    # Local inventory devices only: remote client names ("RPC0") in the
    # --device list have zero usable bytes here and would each read as a
    # spurious compute-reserve oversubscription; their budgets are
    # charged via config["rpc"]["admission"] at reservation time.
    for dev in [d for d in order if d in by_dev]:
        share = shares.get(dev, 0.0)
        w = int(gpu_weights * share)
        k = int(kv * share)
        ov = int(overhead * share)
        pinned = int(expert_pins.get(dev, 0))
        cache = int(cache_budgets.get(dev, 0))
        demand = w + k + ov + pinned + cache + reserve
        u = int(usable.get(dev, 0))
        ok = demand <= u
        fits = fits and ok
        devices[dev] = {
            "capacity_bytes": int(by_dev.get(dev, {}).get("total_bytes", 0)),
            "usable_bytes": u,
            "weights_bytes": w,
            "kv_bytes": k,
            "overhead_bytes": ov,
            "pinned_expert_bytes": pinned,
            "cache_bytes": cache,
            "compute_reserve_bytes": int(reserve),
            "demand_bytes": int(demand),
            "fits": ok,
        }
    return {"fits": fits, "devices": devices,
            "cpu_expert_bytes": int(cpu_expert_bytes)}


def apply_plan_cache_budgets(profile, plan, log=print):
    """Write the plan's effective cache budgets back into the profile.

    Keeps the runtime consistent with the plan: the planner reserves a
    uniform per-GPU cache budget (the fork applies one budget to every
    device) and may disable the cache entirely when it doesn't fit.
    Every apply path -- CLI and web alike -- must call this before saving,
    so build_moe_cache_args emits exactly what the planner reserved,
    never the stale request. Returns True if the profile was modified.
    """
    mc = profile.get("moe_cache", {})
    if mc.get("mode", "off") == "off":
        return False
    requested = mc.get("gpu", {}).get("budgets_bytes", {})
    effective = plan.get("cache_budgets") or {}
    if effective == requested:
        return False
    if not effective:
        log("planner disabled the expert cache (budget exceeds usable "
            "VRAM); clearing moe_cache budgets so the server matches the plan")
    else:
        gib = {d: round(b / (1 << 30), 1) for d, b in effective.items()}
        log(f"planner reserved a uniform per-GPU cache budget: {gib} GiB")
    mc.setdefault("gpu", {})["budgets_bytes"] = effective
    profile["moe_cache"] = mc
    return True


def _expert_assignment(layers_bytes, ordered_devs, budgets):
    """Greedy fastest-tier-first whole-layer assignment.
    Returns (assignment: {device: [layers]}, cpu_layers: [layers])."""
    remaining = sorted(layers_bytes)
    assignment = {}
    for d in ordered_devs:
        dev = d["device"]
        budget = budgets.get(dev, 0)
        chosen = []
        for layer in list(remaining):
            if layers_bytes[layer] <= budget:
                chosen.append(layer)
                budget -= layers_bytes[layer]
                remaining.remove(layer)
        if chosen:
            assignment[dev] = chosen
    return assignment, remaining


def _plan_moe_spill(tier, layout, devs, usable, primary, kv, ram_budget,
                    other, warnings, analysis, result, cache_budgets=None,
                    remote_rungs=None):
    layers_bytes = layout["expert_bytes_per_layer"]
    expert_total = sum(layers_bytes.values())
    non_expert = layout["non_expert_bytes"]

    # Fixed set: attention, norms, embeddings, shared experts, KV cache,
    # plus compute-buffer overhead. With --split-mode layer (the only working
    # multi-SYCL mode) llama.cpp distributes these across ALL cards by the
    # tensor-split ratio, so each card's expert budget must absorb its share
    # -- pinning the fixed set to the primary in the math while it splits at
    # runtime is how you get a load-time VRAM OOM on the small card.
    fixed = non_expert + kv
    gpu_overhead = modelctl_vram.compute_overhead_bytes(fixed)
    fixed_total = fixed + gpu_overhead
    if fixed_total > sum(usable.values()):
        warnings.append(
            "attention/KV alone exceeds the combined GPU budget -- consider "
            "a smaller ctx or KV quant; this plan will likely OOM.")

    # Distribute the fixed set only across the devices that actually end
    # up in the emitted --device list. A device whose expert budget comes
    # out empty is dropped from the plan entirely, and the fixed bytes it
    # would have carried land on the remaining cards -- so their expert
    # budgets must be recomputed against the larger share. Skipping this
    # re-distribution is how a single-device plan got experts budgeted as
    # if another card were still absorbing 27% of the KV/attention set:
    # aggregate math passed, the emitting device oversubscribed at load.
    active = list(devs)
    while True:
        cap_sum = sum(d["total_bytes"] for d in active)
        budgets = {}
        for d in active:
            share = fixed_total * d["total_bytes"] / cap_sum
            budgets[d["device"]] = max(
                0, usable[d["device"]] - share - GPU_COMPUTE_RESERVE_BYTES)
        ordered = sorted(active, key=lambda d: -gpu_bandwidth_gbs(d["name"]))
        assignment, cpu_layers = _expert_assignment(layers_bytes, ordered,
                                                    budgets)
        used = [d for d in active if d["device"] in assignment]
        if not used or len(used) == len(active):
            break
        active = used

    used_devs = [d["device"] for d in active if d["device"] in assignment]
    if not used_devs:
        used_devs = [primary]
        warnings.append("no expert layers fit any GPU budget -- all routed "
                        "experts land on CPU; this will be slow.")

    # Remote rungs take what the local GPUs could not, before the CPU
    # tail. Eligibility follows the calibrated ladder order: in tier 3
    # the tail is resident RAM (75 GB/s), which only the remote-GPU rung
    # outranks; in tier 4 the tail streams from SSD, which every rung
    # outranks. A placement under the wire floor is skipped -- the fixed
    # per-token network cost would eat the win.
    rpc_used = []
    if remote_rungs and cpu_layers:
        def rung_gbs(r):
            return (RPC_GPU_BANDWIDTH_GBS if r.get("kind") == "gpu"
                    else RPC_CPU_BANDWIDTH_GBS)
        tail_gbs = CPU_BANDWIDTH_GBS if tier == 3 else 0.0
        eligible = sorted([r for r in remote_rungs if rung_gbs(r) > tail_gbs],
                          key=lambda r: -rung_gbs(r))
        for rung in eligible:
            budget = rung.get("budget_bytes", 0)
            chosen = []
            for layer in list(cpu_layers):
                if layers_bytes[layer] <= budget:
                    chosen.append(layer)
                    budget -= layers_bytes[layer]
                    cpu_layers.remove(layer)
            if sum(layers_bytes[l] for l in chosen) < MIN_RPC_PLACEMENT_BYTES:
                cpu_layers.extend(chosen)
                cpu_layers.sort()
                continue
            rpc_used.append((rung, sorted(chosen)))

    # -ot parts: specific layer ranges FIRST (first-match-wins), CPU last.
    # Backslashes are doubled so shlex.split in build_server_args passes a
    # literal '\.' through to llama-server.
    #
    # Hard rule (P5, landscape RQ7): shared experts run every token and
    # leading dense layers (GLM/DeepSeek pattern, e.g. blk.0-2) never
    # route, so both get explicit GPU pins ahead of every other rule --
    # only routed experts (_exps) may flow to the CPU/offload tiers.
    # Their bytes already sit in the fixed set the budget math spreads
    # across cards; the pins concentrate a small slice of it (a few
    # hundred MB on real models) on named devices, absorbed by the
    # per-card compute reserve.
    ot_parts = []
    dense_leading = list(range(min(layers_bytes)))
    if dense_leading:
        ot_parts.append(
            f"blk\\\\.{layers_regex(dense_leading)}\\\\.ffn_.*={primary}")
    if layout.get("has_shexp"):
        # The shared-expert output sums with the routed-expert output, so
        # each layer's shexp lives with that layer's experts (the FFN
        # input activation is already shipped there); layers whose
        # experts spill to CPU keep their shexp on the primary GPU.
        for dev in used_devs:
            ls = sorted(assignment.get(dev, []))
            if ls:
                ot_parts.append(
                    f"blk\\\\.{layers_regex(ls)}\\\\.ffn_.*_shexp={dev}")
        ot_parts.append(f"ffn_.*_shexp={primary}")
    for dev in used_devs:
        ls = sorted(assignment.get(dev, []))
        if ls:
            ot_parts.append(f"blk\\\\.{layers_regex(ls)}\\\\.ffn_.*_exps={dev}")
    for rung, ls in rpc_used:
        # -ot names the buffer type: the endpoint qualifier with the
        # node-local device index, NOT the --device list position (see
        # bench_qwen122b_fleet.py arm D: two single-device nodes are both
        # RPC0[endpoint] in -ot while --device says RPC0,RPC1).
        target = (f"RPC{rung.get('local_device_index', 0)}"
                  f"[{rung['endpoint']}]")
        ot_parts.append(f"blk\\\\.{layers_regex(ls)}\\\\.ffn_.*_exps={target}")
    if cpu_layers:
        ot_parts.append("ffn_.*_exps=CPU")

    extra = ["--fit off"]
    if rpc_used:
        # --rpc must precede --device in the argv: the RPC backend
        # registers its devices when --rpc is parsed (laguna R2 gotcha).
        extra.insert(0, "--rpc "
                     + ",".join(rung["endpoint"] for rung, _ in rpc_used))
    if len(used_devs) > 1 or rpc_used:
        split_mode = "layer"
        split_devs = [d for d in devs if d["device"] in used_devs]
        if len(split_devs) > 1:
            tensor_split = modelctl_vram.tensor_split_ratio(
                [d["total_bytes"] for d in split_devs])
        else:
            tensor_split = "1"
        rpc_clients = [f"RPC{i}" for i in range(len(rpc_used))]
        if rpc_used:
            tensor_split += ",0" * len(rpc_used)
        device = ""
        extra.append("--device " + ",".join(used_devs + rpc_clients))
    else:
        split_mode = tensor_split = ""
        device = used_devs[0]
    if ot_parts:
        extra.append("-ot " + ",".join(ot_parts))
    if tier == 3:
        # Whole model fits GPU+RAM: load CPU-resident tensors into anonymous
        # RAM instead of page-faulting them off the SSD on every cold token.
        extra.append("--no-mmap")
    if other:
        extra.append(other)

    cpu_gib = sum(layers_bytes[l] for l in cpu_layers) / (1 << 30)
    if tier == 3 and cpu_gib > analysis["ram_budget_gib"]:
        # Fixed+GPU consumed the model but the CPU share still exceeds RAM:
        # effective streaming anyway -- downgrade the label, keep the plan.
        warnings.append("the CPU share is bigger than the RAM budget, so "
                        "part of the model will stream from disk -- expect "
                        "tier-4 (slowest) speed in practice.")
    layout_rows = [("all GPUs (fixed)" if len(used_devs) > 1 else f"{primary} (fixed)",
                    _gib(fixed_total),
                    "attention/embeddings/KV, split by capacity"
                    if len(used_devs) > 1 else "attention/embeddings/KV")]
    for dev in used_devs:
        ls = sorted(assignment.get(dev, []))
        if ls:
            gib = sum(layers_bytes[l] for l in ls) / (1 << 30)
            layout_rows.append((dev, gib,
                                f"experts layers {ls[0]}-{ls[-1]} ({len(ls)})"))
    if cpu_layers:
        backing = "RAM" if tier == 3 else "SSD via mmap"
        layout_rows.append(("CPU", cpu_gib,
                            f"experts layers {cpu_layers[0]}-{cpu_layers[-1]} "
                            f"({len(cpu_layers)}), {backing}"))
    if cache_budgets:
        for dev, b in cache_budgets.items():
            layout_rows.append((f"{dev} (cache)", _gib(b),
                                "dynamic expert cache (LRU slots)"))

    config = {"device": device, "split_mode": split_mode,
              "tensor_split": tensor_split, "extra": " ".join(extra)}
    if rpc_used:
        remote_gib = sum(layers_bytes[l] for _, ls in rpc_used
                         for l in ls) / (1 << 30)
        names = ", ".join(rung["name"] for rung, _ in rpc_used)
        warnings.append(
            f"{remote_gib:.1f} GiB of experts load onto {names} over the "
            "network -- loads measured 10-90 s slower (2026-08-03 "
            "calibration), and serving this plan requires the "
            "RPC-enabled build")
        for rung, ls in rpc_used:
            gib = sum(layers_bytes[l] for l in ls) / (1 << 30)
            layout_rows.append((rung["name"], gib,
                                f"experts layers {ls[0]}-{ls[-1]} "
                                f"({len(ls)}), over RPC"))
        config["rpc"] = {
            "endpoints": [rung["endpoint"] for rung, _ in rpc_used],
            "placements": [],
            "admission": {rung["name"]: int(sum(layers_bytes[l] for l in ls))
                          for rung, ls in rpc_used},
        }
    return result(tier, config, layout_rows)


def _plan_dense_spill(tier, layout, devs, usable, primary, kv, ram_budget,
                      other, warnings, analysis, result):
    block_count = layout["block_count"]
    weights = layout["weight_bytes"]
    other_bytes = layout["other_bytes"]
    layer_total = weights - other_bytes
    per_layer = layer_total / block_count if block_count else layer_total

    # Per-card compute reserve comes off the top, matching the admission
    # math: overhead covers buffers scaled to the weights, the reserve
    # covers the allocator slack that admission charges per device.
    gpu_budget = (sum(usable.values())
                  - len(devs) * GPU_COMPUTE_RESERVE_BYTES)
    fixed = other_bytes + kv + modelctl_vram.compute_overhead_bytes(weights)
    n_gpu = int(max(0, min(block_count,
                           math.floor((gpu_budget - fixed) / per_layer))))
    if n_gpu < block_count * 0.2:
        warnings.append(
            f"only {n_gpu}/{block_count} layers fit on GPU -- expect "
            "CPU-bound single-digit tok/s; a smaller quant would help more "
            "than any placement.")

    extra = ["--fit off", f"-ngl {n_gpu}"]
    if len(devs) > 1:
        split_mode = "layer"
        tensor_split = modelctl_vram.tensor_split_ratio(
            [d["total_bytes"] for d in devs])
        device = ""
        extra.append("--device " + ",".join(d["device"] for d in devs))
    else:
        split_mode = tensor_split = ""
        device = primary
    if tier == 3:
        extra.append("--no-mmap")
    if other:
        extra.append(other)

    cpu_gib = (block_count - n_gpu) * per_layer / (1 << 30)
    layout_rows = [("all GPUs", _gib(fixed + n_gpu * per_layer),
                    f"layers 0..{n_gpu - 1} + embeddings/KV" if n_gpu
                    else "embeddings/KV only"),
                   ("CPU", cpu_gib,
                    f"layers {n_gpu}..{block_count - 1} "
                    f"({'RAM' if tier == 3 else 'SSD via mmap'})")]
    config = {"device": device, "split_mode": split_mode,
              "tensor_split": tensor_split, "extra": " ".join(extra)}
    return result(tier, config, layout_rows)
