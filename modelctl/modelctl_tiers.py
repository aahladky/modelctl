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
    overhead = max(1 << 30, int(weights_bytes * 0.10))
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
        return w + ctx * kv_per_token + max(1 << 30, int(w * 0.10))

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


def plan_tiers(profile, inventory, limit_pct, primary, ram_available=None,
               layout=None, cache_request=None):
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

    Returns None when the model can't be analyzed; otherwise:
      {tier, config: {device, split_mode, tensor_split, extra},
       layout: [(label, gib, description)], warnings: [...],
       analysis: {weights_gib, kv_gib, is_moe, n_layers, ram_budget_gib,
                  cache_budgets_gib: {...} or None},
       cache_budgets: {dev: bytes} or None}
    cache_budgets is the effective UNIFORM per-GPU reserve the planner
    actually accounted for (None when the cache is off or didn't fit);
    apply paths should write it back to the profile so the server
    allocates exactly what the planner reserved.
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
    ctx = int(cfg.get("ctx", 32768))
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
    try:
        import modelctl_hardware
        hw_settings = modelctl_hardware.load_settings()
    except Exception:
        hw_settings = {}
    dev_cfg = hw_settings.get("devices", {})
    devs = [d for d in devs if dev_cfg.get(d["device"], {}).get("enabled", True)]
    if not devs:
        return None
    if primary not in [d["device"] for d in devs]:
        primary = devs[0]["device"]
    usable = {d["device"]: d["total_bytes"] * frac
                    - dev_cfg.get(d["device"], {}).get("reserve_bytes", 0)
              for d in devs}

    weights = layout["weight_bytes"]
    overhead = max(1 << 30, int(weights * 0.10))
    total = weights + kv + overhead
    warnings = []
    _, other_flags = split_extra_flags(cfg.get("extra", ""))
    other = " ".join(other_flags)

    # Extract the dynamic cache budget from cache_request.  The llama.cpp
    # fork's --moe-cache-bytes is a single UNIFORM per-GPU budget: one global
    # (g_moe_cache_budget_bytes) is applied to every device's lazily-created
    # cache instance -- there is no per-device flag.  So per-device requests
    # collapse to their max, and that uniform budget must be reserved on
    # EVERY participating device BEFORE static expert layers are assigned,
    # or the runtime cache can collide with statically placed experts (OOM).
    # If the uniform budget doesn't fit on even one device, the cache is
    # disabled for the whole plan -- a half-reserved cache would diverge
    # from what the server actually allocates.
    cache_budgets = {}
    if cache_request and cache_request.get("mode", "off") != "off":
        requested = [b for b in cache_request.get("gpu", {})
                     .get("budgets_bytes", {}).values() if b > 0]
        if requested:
            uniform = max(requested)
            too_small = sorted(d for d, u in usable.items() if uniform > u)
            if too_small:
                warnings.append(
                    f"cache budget {uniform / (1<<30):.1f} GiB exceeds usable "
                    f"VRAM on {', '.join(too_small)} -- the fork applies one "
                    "per-GPU budget to every device, so the cache is disabled "
                    "for this plan")
            else:
                cache_budgets = {d: uniform for d in usable}
                for d in usable:
                    usable[d] -= uniform
    if cache_budgets:
        for dev, b in cache_budgets.items():
            if b > usable[dev]:
                warnings.append(
                    f"cache reserve {b / (1<<30):.1f} GiB on {dev} leaves "
                    f"only {usable[dev] / (1<<30):.1f} GiB for static experts "
                    "-- this may force more experts to CPU than the non-cache plan")

    analysis = {"weights_gib": _gib(weights), "kv_gib": _gib(kv),
                "is_moe": layout["is_moe"], "n_layers": layout["block_count"],
                "ram_budget_gib": _gib(ram_budget),
                "cache_budgets_gib": ({d: _gib(b) for d, b in cache_budgets.items()}
                                      if cache_budgets else None)}
    if layout.get("unknown_type_tensors"):
        warnings.append(
            f"{layout['unknown_type_tensors']} tensors use quant types this "
            "planner doesn't know -- layout math undercounts; check the plan.")

    def result(tier, config, layout_rows):
        return {"tier": tier, "config": config, "layout": layout_rows,
                "warnings": warnings, "analysis": analysis,
                "cache_budgets": dict(cache_budgets) if cache_budgets else None}

    # --- tier 1: primary GPU alone --------------------------------------
    if total <= usable[primary]:
        return result(1, {"device": primary, "split_mode": "",
                          "tensor_split": "", "extra": other},
                      [(f"{primary} (whole model)", _gib(total),
                        "weights + KV + overhead")])

    # --- tier 2: all GPUs ------------------------------------------------
    combined = sum(usable.values())
    if len(devs) > 1 and total <= combined:
        # Ratio from USABLE bytes, not raw capacity: llama.cpp distributes
        # the model by these weights, and a device carrying a reserve or
        # the uniform cache reservation must receive proportionally less.
        # A raw-capacity ratio put static bytes into the reserved space --
        # aggregate admission passed, the small card OOMed at load.
        ts = modelctl_vram.tensor_split_ratio(
            [usable[d["device"]] for d in devs])
        # The emitted ratio is GiB-rounded and GCD-reduced; recheck what
        # each device actually receives under the quantized weights
        # before admitting, and spill instead when quantization overflows
        # someone's budget.
        weights = [int(w) for w in ts.split(",")]
        wsum = sum(weights) or 1
        per_device_fits = all(
            total * w / wsum <= usable[d["device"]]
            for w, d in zip(weights, devs))
        if per_device_fits:
            # Explicit --device list: without one, llama.cpp maps
            # tensor_split positions to its own enumeration order, while
            # this ratio is in devs order (primary first) -- a non-SYCL0
            # primary would receive the wrong share at runtime.
            dev_list = ",".join(d["device"] for d in devs)
            extra2 = f"--device {dev_list}" + (f" {other}" if other else "")
            return result(2, {"device": "", "split_mode": "layer",
                              "tensor_split": ts, "extra": extra2},
                          [("all GPUs", _gib(total),
                            f"layer split {ts} by usable VRAM")])

    # --- tiers 3/4: spill to CPU (RAM), maybe SSD streaming --------------
    tier = 3 if total <= combined + ram_budget else 4
    if tier == 4:
        warnings.append(
            "model exceeds GPU+RAM: the CPU-resident portion streams from "
            "SSD via mmap -- expect low single-digit tok/s on cold cache.")

    if layout["is_moe"]:
        return _plan_moe_spill(
            tier, layout, devs, usable, primary, kv, ram_budget, other,
            warnings, analysis, result, cache_budgets)
    return _plan_dense_spill(
        tier, layout, devs, usable, primary, kv, ram_budget, other,
        warnings, analysis, result)


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
                    other, warnings, analysis, result, cache_budgets=None):
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
    gpu_overhead = max(1 << 30, int(fixed * 0.10))
    fixed_total = fixed + gpu_overhead
    cap_sum = sum(d["total_bytes"] for d in devs)
    budgets = {}
    for d in devs:
        share = fixed_total * d["total_bytes"] / cap_sum
        budgets[d["device"]] = max(
            0, usable[d["device"]] - share - GPU_COMPUTE_RESERVE_BYTES)
    if fixed_total > sum(usable.values()):
        warnings.append(
            "attention/KV alone exceeds the combined GPU budget -- consider "
            "a smaller ctx or KV quant; this plan will likely OOM.")

    ordered = sorted(devs, key=lambda d: -gpu_bandwidth_gbs(d["name"]))
    assignment, cpu_layers = _expert_assignment(layers_bytes, ordered, budgets)
    used_devs = [d["device"] for d in devs if d["device"] in assignment]
    if not used_devs:
        used_devs = [primary]
        warnings.append("no expert layers fit any GPU budget -- all routed "
                        "experts land on CPU; this will be slow.")

    # -ot parts: specific layer ranges FIRST (first-match-wins), CPU last.
    # Backslashes are doubled so shlex.split in build_server_args passes a
    # literal '\.' through to llama-server.
    ot_parts = []
    for dev in used_devs:
        ls = sorted(assignment.get(dev, []))
        if ls:
            ot_parts.append(f"blk\\\\.{layers_regex(ls)}\\\\.ffn_.*_exps={dev}")
    if cpu_layers:
        ot_parts.append("ffn_.*_exps=CPU")

    extra = ["--fit off"]
    if len(used_devs) > 1:
        split_mode = "layer"
        split_devs = [d for d in devs if d["device"] in used_devs]
        tensor_split = modelctl_vram.tensor_split_ratio(
            [d["total_bytes"] for d in split_devs])
        device = ""
        extra.append(f"--device {','.join(used_devs)}")
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
        warnings.append("CPU-resident share exceeds the RAM budget even "
                        "though the whole model fits on paper -- treat this "
                        "as tier 4 in practice.")
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
    return result(tier, config, layout_rows)


def _plan_dense_spill(tier, layout, devs, usable, primary, kv, ram_budget,
                      other, warnings, analysis, result):
    block_count = layout["block_count"]
    weights = layout["weight_bytes"]
    other_bytes = layout["other_bytes"]
    layer_total = weights - other_bytes
    per_layer = layer_total / block_count if block_count else layer_total

    gpu_budget = sum(usable.values())
    fixed = other_bytes + kv + max(1 << 30, int(weights * 0.05))
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
