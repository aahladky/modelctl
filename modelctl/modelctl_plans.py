"""Launch-plan generation for modelctl.

Compiles a bounded set of comparable launch plans from a profile and
current hardware snapshot.  Each plan represents a concrete way to run
the model — different GPU placements, tensor splits, CPU offload
strategies, or storage modes.

Primary API:
    compile_launch_plans(profile, hardware, ...) -> list[LaunchPlan]

Candidate sources:
    A. Current profile (baseline, always generated)
    B. Existing tier planner output
    C. Single-GPU plans (one per eligible GPU)
    D. Multi-GPU split plans (capacity-ratio variants)
    E. CPU-spill plans (partial offload)

Plan IDs are stable hashes of the normalized plan representation.
"""
import hashlib
import re
import json
import math
import os
from dataclasses import dataclass, field, asdict

import modelctl
import modelctl_tiers
import modelctl_vram


@dataclass(frozen=True)
class ResourceClaim:
    vram_bytes: dict          # {device: bytes}
    ram_bytes: int
    storage_mode: str         # "none", "mmap", "direct"
    expected_context: int | None
    breakdown: dict = field(default_factory=dict)
    # breakdown is an optional per-device decomposition for UI/debug:
    # {"vram": {"SYCL0": {"fixed": N, "kv": N, "static_experts": N,
    #           "dynamic_expert_cache": N, "reserve": N}},
    #  "ram": {"expert_backing": N, "staging": N}}
    # It does NOT affect vram_bytes, which remains the total reservation
    # used by the worker and matrix logic.


@dataclass(frozen=True)
class LaunchPlan:
    id: str
    profile_name: str
    backend: str
    label: str
    argv: tuple
    env: dict
    claim: ResourceClaim
    estimated: dict           # {total_vram, per_device_vram, ram, context}
    source: str               # "current-profile", "tier-planner", "single-gpu", etc.
    warnings: tuple
    decision_data: dict       # extra info for explainability


@dataclass(frozen=True)
class RuntimePolicy:
    objective: str            # "balanced", "fastest_generation", etc.
    pinned_plan_id: str | None
    allow_fallback: bool
    allow_untested: bool
    minimum_context: int | None
    maximum_cpu_bytes: int | None
    maximum_storage_tier: int  # 1=gpu, 2=gpu+ram, 3=gpu+ram+storage


# Default policy
DEFAULT_POLICY = RuntimePolicy(
    objective="balanced",
    pinned_plan_id=None,
    allow_fallback=True,
    allow_untested=False,
    minimum_context=8192,
    maximum_cpu_bytes=None,
    maximum_storage_tier=3,
)


def _plan_id(normalized):
    """Stable hash of the canonical plan representation."""
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _plan_label(config, source, gpu_names=None):
    """Human-readable label for a plan."""
    device = config.get("device", "")
    split = config.get("tensor_split", "")
    extra = config.get("extra", "")

    if source == "current-profile":
        return "current profile"
    if source == "tier-planner":
        tier = config.get("_tier", "?")
        return f"tier {tier} plan"

    if split and "," in split:
        names = ",".join(gpu_names) if gpu_names else device
        return f"split {split} ({names})"
    if device:
        return f"{device} only"
    if "exps=CPU" in extra or "-ngl" in extra:
        return "CPU offload"
    return source


def _parse_ot_rules(extra):
    """Extract -ot tensor placement rules from a config's extra flags.
    Returns [(compiled_regex, target)] in first-match-wins order (llama.cpp
    stops at the first matching override)."""
    import shlex
    try:
        toks = shlex.split(extra or "")
    except ValueError:
        toks = (extra or "").split()
    rules = []
    i = 0
    while i < len(toks):
        if toks[i] == "-ot" and i + 1 < len(toks):
            for part in toks[i + 1].split(","):
                pattern, _, target = part.rpartition("=")
                if not target or not pattern:
                    continue
                try:
                    rules.append((re.compile(pattern), target))
                except re.error:
                    continue
            i += 2
        else:
            i += 1
    return rules


def _make_claim(profile, config, hardware):
    """Placement-aware resource claim.

    Uses the GGUF tensor layout (exact per-layer/expert bytes) plus the
    config's actual offload directives -- -ot override rules (first match
    wins, like llama.cpp), -ngl partial offload, --no-mmap -- to split
    weights across devices and RAM, instead of charging the whole model to
    VRAM. KV cache and compute overhead are GPU-resident and distributed the
    same way as the non-expert weights."""
    model_path = profile.get("model_path", "")
    ctx = int(config.get("ctx", 8192))
    cache_k = config.get("cache_type_k", "q8_0")
    cache_v = config.get("cache_type_v", cache_k)
    extra = config.get("extra", "") or ""

    layout = None
    if model_path and os.path.exists(model_path):
        try:
            layout = modelctl_vram.gguf_model_layout(model_path)
        except Exception:
            layout = None

    if layout:
        weights = layout["weight_bytes"]
        non_expert = layout["non_expert_bytes"]
        meta = layout.get("meta") or {}
        params = modelctl_vram.gguf_kv_params(meta)
        if params:
            kv = modelctl_vram.kv_cache_bytes(params, ctx, cache_k, cache_v)
        else:
            kv = ctx * modelctl_vram.HEURISTIC_KV_BYTES_PER_TOKEN
    else:
        weights = non_expert = 0
        if model_path and os.path.exists(model_path):
            try:
                weights = modelctl_vram.weights_bytes_on_disk(model_path)
            except Exception:
                pass
        non_expert = weights
        est = modelctl_vram.estimate_from_parts(weights, ctx, cache_k, cache_type_v=cache_v)
        kv = est["kv_bytes"]

    # ---- weight placement ---------------------------------------------
    assigned = {}       # device -> bytes pinned by -ot <pattern>=DEVICE
    cpu_bytes = 0
    default_gpu_experts = 0
    if layout and layout.get("is_moe") and extra:
        rules = _parse_ot_rules(extra)
        ngl_m = re.search(r"-ngl\s+(\d+)", extra)
        if ngl_m and not rules:
            # Dense-style -ngl on a MoE model offloads WHOLE layers: layers
            # >= ngl put all their tensors (attention AND experts) on CPU.
            ngl = int(ngl_m.group(1))
            block_count = layout.get("block_count") or 0
            per_layer_nonexp = (layout.get("layer_bytes", 0) / block_count
                                if block_count else 0)
            for layer, nbytes in layout["expert_bytes_per_layer"].items():
                if layer >= ngl:
                    cpu_bytes += nbytes + per_layer_nonexp
                else:
                    default_gpu_experts += nbytes
            non_expert -= int(per_layer_nonexp *
                              sum(1 for l in layout["expert_bytes_per_layer"]
                                  if l >= ngl))
            non_expert = max(0, non_expert)
        elif rules:
            for layer, nbytes in layout["expert_bytes_per_layer"].items():
                probe = f"blk.{layer}.ffn_gate_exps.weight"
                target = None
                for rx, tgt in rules:
                    if rx.search(probe):
                        target = tgt
                        break
                if target is None:
                    default_gpu_experts += nbytes
                elif target == "CPU":
                    cpu_bytes += nbytes
                else:
                    assigned[target] = assigned.get(target, 0) + nbytes
        else:
            default_gpu_experts = layout["weight_bytes"] - non_expert
    elif layout and layout.get("is_moe"):
        default_gpu_experts = layout["weight_bytes"] - non_expert

    if layout and not layout.get("is_moe") and extra:
        m = re.search(r"-ngl\s+(\d+)", extra)
        if m:
            ngl = int(m.group(1))
            per_layer = ((weights - layout["other_bytes"])
                         / max(1, layout["block_count"]))
            gpu_part = layout["other_bytes"] + ngl * per_layer
            cpu_bytes += max(0, weights - gpu_part)
            non_expert = min(non_expert, gpu_part)

    # Auxiliary GPU-resident artifacts: mmproj (multimodal projector) and
    # the MTP draft model are loaded onto the GPU alongside the weights.
    aux_gpu = 0
    for aux_path in (profile.get("mmproj_path"), profile.get("mtp_path")):
        if aux_path and os.path.exists(aux_path):
            try:
                aux_gpu += os.path.getsize(aux_path)
            except OSError:
                pass

    if not layout:
        # No layout available: keep the old whole-model estimate behavior.
        est = modelctl_vram.estimate_from_parts(weights, ctx, cache_k, cache_type_v=cache_v)
        gpu_total = est["total"]
        shared_gpu = gpu_total + aux_gpu
        base_gpu = gpu_total
    else:
        # GPU-resident total: non-expert + unpinned experts + KV + overhead
        base_gpu = non_expert + default_gpu_experts
        shared_gpu = base_gpu + kv + max(1 << 30, int(base_gpu * 0.10)) + aux_gpu

    # ---- distribute shared GPU bytes across devices ---------------------
    device = config.get("device", "")
    split = config.get("tensor_split", "")
    vram_map = dict(assigned)
    if split and "," in split and hardware:
        import modelctl_hardware as _hw
        ratios = [float(x) for x in split.split(",")]
        gpus = list(_hw.enabled_gpus(hardware))
        denom = sum(ratios) or 1
        for i, ratio in enumerate(ratios):
            if i < len(gpus):
                dev = gpus[i].device
                vram_map[dev] = vram_map.get(dev, 0) + int(shared_gpu * ratio / denom)
    elif device:
        vram_map[device] = vram_map.get(device, 0) + shared_gpu
    elif hardware:
        import modelctl_hardware as _hw
        _eg = _hw.enabled_gpus(hardware)
        dev = _eg[0].device if _eg else ""
        vram_map[dev] = vram_map.get(dev, 0) + shared_gpu

    storage = "mmap" if cpu_bytes and "--no-mmap" not in extra else "none"

    return ResourceClaim(
        vram_bytes=vram_map,
        ram_bytes=cpu_bytes,
        storage_mode=storage,
        expected_context=ctx,
    )


def _make_plan(profile, config, source, hardware, extra_warnings=(), decision=None):
    """Build a LaunchPlan from profile + config overrides."""
    merged = {**profile.get("config", {}), **config}
    args = modelctl.build_server_args({**profile, "config": merged})
    env = {k: v for e in profile.get("env", []) for k, v in [e.split("=", 1)]} if profile.get("env") else {}

    gpu_names = []
    if hardware:
        device = merged.get("device", "")
        split = merged.get("tensor_split", "")
        if split and "," in split:
            import modelctl_hardware as _hw
            for g in list(_hw.enabled_gpus(hardware))[:len(split.split(","))]:
                gpu_names.append(g.name)
        elif device:
            g = hardware.gpu_by_device(device)
            if g:
                gpu_names.append(g.name)

    backend_bin = profile.get("binary") or modelctl.LLAMA_SERVER_BIN
    normalized = {
        "backend": profile.get("backend", "llama-cpp"),
        "binary": merged.get("binary", "") or profile.get("binary", ""),
        "binary_fp": modelctl_vram.file_fingerprint(backend_bin),
        "model_fp": modelctl_vram.file_fingerprint(profile.get("model_path")),
        "mmproj_fp": modelctl_vram.file_fingerprint(profile.get("mmproj_path")),
        "mtp_fp": modelctl_vram.file_fingerprint(profile.get("mtp_path")),
        "device": merged.get("device", ""),
        "split_mode": merged.get("split_mode", ""),
        "tensor_split": merged.get("tensor_split", ""),
        "ctx": merged.get("ctx", 8192),
        "cache_type_k": merged.get("cache_type_k", "q8_0"),
        "cache_type_v": merged.get("cache_type_v", "q8_0"),
        "fit": merged.get("fit", "on"),
        "extra": merged.get("extra", ""),
        "env": sorted(profile.get("env") or []),
    }
    pid = _plan_id(normalized)
    label = _plan_label(merged, source, gpu_names)
    claim = _make_claim(profile, merged, hardware)

    return LaunchPlan(
        id=pid,
        profile_name=profile.get("name", ""),
        backend=profile.get("backend", "llama-cpp"),
        label=label,
        argv=tuple(args),
        env=env,
        claim=claim,
        estimated={"total_vram": sum(claim.vram_bytes.values()),
                   "ram": claim.ram_bytes, "context": claim.expected_context},
        source=source,
        warnings=tuple(extra_warnings),
        decision_data=decision or {},
    )


def _compile_ovms_plans(profile, hardware):
    """OVMS plan candidates: one per enabled GPU (target_device), plus a
    reduced-cache fallback per GPU. Claims come from the local IR directory
    size when present (OVMS pulls its own copy on first start)."""
    plans = []
    cfg = profile["config"]
    name = profile["name"]

    ir_bytes = 0
    try:
        repo_dir = modelctl.OVMS_MODEL_REPOSITORY / profile.get("repo_id", "")
        if repo_dir.is_dir():
            ir_bytes = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
    except Exception:
        ir_bytes = 0

    for idx, gpu in enumerate(hardware.gpus):
        if not gpu.enabled:
            continue
        target = f"GPU.{idx}"
        for cache_mult, suffix in ((1.0, ""), (0.5, " low-cache")):
            base_cache = cfg.get("cache_size")
            cache = max(1, int(base_cache * cache_mult)) if base_cache else None
            label = f"{target}{suffix}"
            claim = ResourceClaim(
                vram_bytes={gpu.device: int(ir_bytes * 1.2)},
                ram_bytes=2 << 30,
                storage_mode="none",
                expected_context=int(cfg.get("ctx", 8192) or 8192))
            pid = _plan_id({"backend": "ovms", "model": profile.get("repo_id"),
                            "target": target, "cache": cache,
                            "task": cfg.get("task")})
            plans.append(LaunchPlan(
                id=pid, profile_name=name, backend="ovms", label=label,
                argv=(), env={},
                claim=claim,
                estimated={"total_vram": int(ir_bytes * 1.2), "ram": 2 << 30,
                           "context": claim.expected_context},
                source="ovms-adapter",
                warnings=(() if ir_bytes else
                          ("IR size unknown until OVMS downloads the model",)),
                decision_data={"target_device": target, "cache_size": cache}))
    return plans


def compile_launch_plans(profile, hardware=None, include_experimental=False):
    """Generate a bounded set of launch plans for a profile.

    Returns 3-12 plans depending on hardware configuration.
    Plans are deterministic and side-effect free.
    """
    if profile.get("backend", "llama-cpp") == "ovms":
        if not hardware:
            import modelctl_hardware
            hardware = modelctl_hardware.capture_hardware_snapshot()
        baseline_claim = {}
        try:
            repo_dir = modelctl.OVMS_MODEL_REPOSITORY / profile.get("repo_id", "")
            if repo_dir.is_dir():
                ir = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
                tgt = profile["config"].get("target_device", "GPU.0")
                idx = int(tgt.split(".")[1]) if "." in tgt else 0
                dev = (hardware.gpus[idx].device if idx < len(hardware.gpus)
                       else hardware.gpus[0].device)
                baseline_claim = {dev: int(ir * 1.2)}
        except Exception:
            pass
        baseline = LaunchPlan(
            id=_plan_id({"backend": "ovms", "model": profile.get("repo_id"),
                         "target": profile["config"].get("target_device"),
                         "cache": profile["config"].get("cache_size")}),
            profile_name=profile["name"], backend="ovms",
            label="current profile", argv=(), env={},
            claim=ResourceClaim(baseline_claim, 2 << 30, "none",
                                profile["config"].get("ctx")),
            estimated={}, source="current-profile", warnings=(),
            decision_data={
                "target_device": profile["config"].get("target_device"),
                "cache_size": profile["config"].get("cache_size")})
        return [baseline] + _compile_ovms_plans(profile, hardware)
    if not hardware:
        from modelctl_hardware import capture_hardware_snapshot
        hardware = capture_hardware_snapshot()

    plans = []
    seen_ids = set()

    def add(plan):
        if plan.id not in seen_ids:
            plans.append(plan)
            seen_ids.add(plan.id)

    # A. Current profile baseline
    add(_make_plan(profile, {}, "current-profile", hardware))

    # B. Tier planner output
    try:
        d = modelctl.load_defaults()
        inventory = modelctl.get_gpu_inventory()
        primary = modelctl.resolve_primary_gpu(inventory, d)
        tier = modelctl_tiers.plan_tiers(
            profile, inventory, d["vram_limit_pct"], primary)
        if tier and tier.get("config"):
            tc = dict(tier["config"])
            tc["_tier"] = tier.get("tier", "?")
            add(_make_plan(profile, tc, "tier-planner", hardware,
                           extra_warnings=tuple(tier.get("warnings", [])),
                           decision={"tier": tier.get("tier"),
                                     "analysis": tier.get("analysis")}))
    except Exception:
        pass

    # F/G/H. MoE cache variants (small/balanced/large) — only when the
    # profile has moe_cache enabled and the binary supports it.
    mc = profile.get("moe_cache", {})
    if mc.get("mode", "off") != "off":
        try:
            import modelctl_capabilities
            binary = profile.get("binary") or modelctl.LLAMA_SERVER_BIN
            caps = modelctl_capabilities.probe_backend(binary)
            if modelctl_capabilities.is_cache_capable(caps):
                gpus_enabled = [g for g in hardware.gpus if g.enabled]
                for frac, label_suffix in ((0.20, "cache-small"),
                                             (0.50, "cache-balanced"),
                                             (0.80, "cache-large")):
                    for g in gpus_enabled:
                        usable = g.total_bytes * 0.9  # VRAM limit pct
                        budget = int(usable * frac)
                        if budget < 64 * (1 << 20):  # minimum 64 MiB
                            continue
                        # Build a plan with this cache budget.
                        cache_cfg = {
                            "device": g.device,
                            "split_mode": "",
                            "tensor_split": "",
                            "extra": f"--moe-cache-bytes {budget}",
                        }
                        cache_cfg["_moe_cache_budget"] = budget
                        cache_cfg["_moe_cache_device"] = g.device
                        plan = _make_plan(profile, cache_cfg,
                                          label_suffix, hardware)
                        add(plan)
        except Exception:
            pass

    # C. Single-GPU plans
    gpus = [g for g in hardware.gpus if g.enabled]
    weights = 0
    model_path = profile.get("model_path", "")
    if model_path and os.path.exists(model_path):
        try:
            weights = modelctl_vram.weights_bytes_on_disk(model_path)
        except Exception:
            pass
    ctx = int(profile.get("config", {}).get("ctx", 8192))
    cache_k = profile.get("config", {}).get("cache_type_k", "q8_0")
    cache_v = profile.get("config", {}).get("cache_type_v", "q8_0")
    est = modelctl_vram.estimate_from_parts(weights, ctx, cache_k, cache_type_v=cache_v)
    total_need = est.get("total", 0)

    for gpu in gpus:
        budget = gpu.total_bytes - gpu.reserve_bytes
        if total_need <= budget:
            add(_make_plan(profile, {"device": gpu.device, "split_mode": "",
                                     "tensor_split": ""},
                           "single-gpu", hardware,
                           decision={"gpu": gpu.device, "budget": budget}))

    # D. Multi-GPU split plans (capacity-ratio variants)
    if len(gpus) >= 2:
        combined = sum(g.total_bytes - g.reserve_bytes for g in gpus)
        if total_need <= combined:
            # Capacity-ratio split
            caps = [g.total_bytes - g.reserve_bytes for g in gpus]
            gib = [max(1, round(c / (1 << 30))) for c in caps]
            divisor = math.gcd(*gib)
            ratio = ",".join(str(g // divisor) for g in gib)

            dev_list = ",".join(g.device for g in gpus)
            add(_make_plan(profile, {"device": "", "split_mode": "layer",
                                     "tensor_split": ratio,
                                     "extra": f"--device {dev_list}"},
                           "multi-gpu", hardware,
                           decision={"ratio": ratio, "combined_budget": combined,
                                     "device_list": dev_list}))

            # Biased variants: favor primary, then secondary
            if len(gpus) == 2 and sum(gib) > 3:
                for bias_idx in range(2):
                    biased = list(gib)
                    biased[bias_idx] += 1
                    biased[1 - bias_idx] = max(1, biased[1 - bias_idx] - 1)
                    b_ratio = ",".join(str(b) for b in biased)
                    if b_ratio != ratio:
                        dev_list = ",".join(g.device for g in gpus)
                        add(_make_plan(profile, {"device": "", "split_mode": "layer",
                                                 "tensor_split": b_ratio,
                                                 "extra": f"--device {dev_list}"},
                                       "multi-gpu", hardware,
                                       decision={"ratio": b_ratio, "bias": gpus[bias_idx].device,
                                                 "device_list": dev_list}))

    # E. CPU-spill plans
    if total_need > 0 and gpus:
        primary_gpu = gpus[0]
        budget = primary_gpu.total_bytes - primary_gpu.reserve_bytes
        if total_need > budget:
            ngl = _compute_ngl(weights, budget, profile, hardware)
            if ngl > 0:
                # Preserve the user's non-placement extra flags (ubatch-size
                # etc.) -- only -ngl is planner-owned here.
                import modelctl_tiers
                _, other_flags = modelctl_tiers.split_extra_flags(
                    profile.get("config", {}).get("extra", ""))
                spill_extra = " ".join([f"-ngl {ngl}"] + other_flags)
                add(_make_plan(profile, {"device": primary_gpu.device, "split_mode": "",
                                         "tensor_split": "",
                                         "extra": spill_extra},
                               "cpu-spill", hardware,
                               extra_warnings=("partial GPU offload",),
                               decision={"ngl": ngl, "gpu": primary_gpu.device}))

    return plans


def _compute_ngl(weights_bytes, gpu_budget, profile, hardware):
    """Layers that fit on the GPU, from the model's ACTUAL block count and
    per-layer bytes (GGUF layout) -- never a hardcoded 32."""
    if not weights_bytes or not gpu_budget:
        return 0
    model_path = profile.get("model_path", "")
    layout = None
    if model_path and os.path.exists(model_path):
        try:
            layout = modelctl_vram.gguf_model_layout(model_path)
        except Exception:
            layout = None
    if not layout or not layout.get("block_count"):
        return 0

    n_layers = layout["block_count"]
    per_layer = (layout["weight_bytes"] - layout["other_bytes"]) / n_layers
    ctx = int(profile.get("config", {}).get("ctx", 8192))
    cache_k = profile.get("config", {}).get("cache_type_k", "q8_0")
    cache_v = profile.get("config", {}).get("cache_type_v", cache_k)
    params = modelctl_vram.gguf_kv_params(layout.get("meta") or {})
    kv = (modelctl_vram.kv_cache_bytes(params, ctx, cache_k, cache_v) if params
          else ctx * modelctl_vram.HEURISTIC_KV_BYTES_PER_TOKEN)
    fixed = layout["other_bytes"] + kv + max(1 << 30, int(layout["weight_bytes"] * 0.05))
    if per_layer <= 0 or fixed >= gpu_budget:
        return 0
    return max(0, min(n_layers - 1, int((gpu_budget - fixed) / per_layer)))


def plan_status(plan, observations=None):
    """Derive plan status from observation history.

    Returns: untested, validated, failed, stale, disabled
    """
    if observations is None:
        return "untested"
    for obs in reversed(observations):
        if obs.get("success"):
            return "validated"
        return "failed"
    return "untested"


def rank_plans(plans, policy=None, observations=None, failures=None):
    """Filter and rank plans by policy constraints.

    Enforces every RuntimePolicy field:
    - minimum_context: plans below it are dropped.
    - maximum_cpu_bytes: plans claiming more RAM are dropped.
    - maximum_storage_tier: mmap plans dropped when tier < 3.
    - allow_untested=False: only plans validated (successful, non-stale
      measurement) under the current fingerprints survive; if that would
      leave nothing, untested plans are allowed back in as a bootstrap.
    - pinned_plan_id: promoted to the front (not exclusive) -- fallback to
      the remaining ranked plans stays available unless the caller filters
      (worker only filters when allow_fallback is False).

    Returns list of (plan, score) tuples, sorted best-first.
    """
    if policy is None:
        policy = DEFAULT_POLICY
    observations = observations or {}

    def is_validated(plan):
        obs = observations.get(plan.id)
        return bool(obs and not obs.get("stale"))

    ranked = []
    untested_pool = []
    for plan in plans:
        # Hard constraints
        if policy.minimum_context:
            if plan.claim.expected_context and plan.claim.expected_context < policy.minimum_context:
                continue
        if policy.maximum_storage_tier < 3 and plan.claim.storage_mode == "mmap":
            continue
        if (policy.maximum_cpu_bytes is not None
                and plan.claim.ram_bytes > policy.maximum_cpu_bytes):
            continue

        score = _score_plan(plan, policy, observations, failures)
        if not policy.allow_untested and not is_validated(plan):
            untested_pool.append((plan, score))
            continue
        ranked.append((plan, score))

    if not policy.allow_untested and not ranked:
        # No validated plan satisfies the constraints. Bootstrap ONLY with
        # the exact current-profile baseline -- an arbitrary generated split
        # or CPU plan is not a safe first launch without tuning/permission.
        ranked = [(p, sc) for p, sc in untested_pool if p.source == "current-profile"]

    # Suppressed plans (incompatible on this backend build) are removed,
    # not merely scored down -- they must never be attempted.
    ranked = [(p, sc) for p, sc in ranked if sc != float("-inf")]

    # Pinned plan promotes to the front but never deletes alternatives.
    if policy.pinned_plan_id:
        pinned = [(p, s) for p, s in ranked if p.id == policy.pinned_plan_id]
        rest = [(p, s) for p, s in ranked if p.id != policy.pinned_plan_id]
        rest.sort(key=lambda x: -x[1])
        return [(p, float("inf")) for p, s in pinned] + rest

    ranked.sort(key=lambda x: -x[1])
    return ranked


# Failure classes that permanently disqualify a plan for a backend build
# (re-testing them is pointless until the backend changes).
_SUPPRESS_FAILURES = {"unsupported_architecture", "invalid_argument"}


def _score_plan(plan, policy, observations, failures=None):
    """Score a plan for a given objective. Higher is better.

    failures: {plan_id: [failure_class, ...]} from RuntimeDB. Recent
    failures penalize; unsupported-architecture/invalid-argument failures
    suppress the plan entirely (-inf) until the backend fingerprint changes
    (observations/failures are keyed to it by the caller)."""
    obs = observations.get(plan.id, {})
    plan_failures = (failures or {}).get(plan.id, [])
    if any(f in _SUPPRESS_FAILURES for f in plan_failures):
        return float("-inf")
    fail_penalty = 10 * len([f for f in plan_failures
                             if f in ("out_of_vram", "out_of_ram", "health_timeout",
                                      "backend_crash")])
    gen_tps = obs.get("generation_tps", 0) or 0
    prompt_tps = obs.get("prompt_tps", 0) or 0
    load_time = obs.get("load_seconds", 0) or 0
    ctx = plan.claim.expected_context or 0

    if policy.objective == "fastest_generation":
        return (gen_tps * 10 if gen_tps else -1) - fail_penalty
    elif policy.objective == "fastest_prompt":
        return (prompt_tps * 10 if prompt_tps else -1) - fail_penalty
    elif policy.objective == "largest_context":
        return ctx - fail_penalty
    elif policy.objective == "fastest_load":
        return (-load_time * 10 if load_time else 0) - fail_penalty
    elif policy.objective == "lowest_ram":
        return -(plan.claim.ram_bytes / (1 << 30)) - fail_penalty
    elif policy.objective == "fixed":
        return 0
    else:  # balanced
        score = 0
        if gen_tps:
            score += gen_tps * 5
        if prompt_tps:
            score += prompt_tps * 1.5
        score += ctx / 10000
        if load_time:
            score -= load_time * 0.5
        if plan.claim.storage_mode == "mmap":
            score -= 2
        return score - fail_penalty
