"""Baseline candidate set for acceptance comparisons.

Acceptance needs four named, directly comparable ways of serving the same
oversized sparse MoE:

    A. stock-mmap        stock placement, CPU spill reached through mmap
    B. resident-ram      the same spill held in RAM (--no-mmap)
    C. static-placement  experts explicitly pinned across GPUs and CPU
    D. static-plus-cache C plus the experimental expert transfer cache

They are only comparable if everything else is held fixed, so this module
builds all four from one profile and one hardware snapshot, and refuses to
emit a candidate whose preconditions do not hold (no second GPU, no
cache-capable runtime, not enough RAM) rather than emitting one that
silently measures something else.

Public API:
    CANDIDATES
    BaselineCandidate dataclass
    build_candidate_set(profile, hardware, backend) -> BaselineSet
    comparability_report(runs) -> dict
"""
import copy
from dataclasses import dataclass, field

import modelctl
import modelctl_plans


# Candidate identity is a stable string, because acceptance reports and
# stored observations refer to these by name across runs and machines.
CANDIDATES = ("stock-mmap", "resident-ram", "static-placement",
              "static-plus-cache")

DESCRIPTIONS = {
    "stock-mmap":
        "Stock placement with CPU spill reached through mmap. The "
        "do-nothing baseline every other candidate has to beat.",
    "resident-ram":
        "The same spill held in RAM instead of mmap (--no-mmap). Isolates "
        "how much of the cost is paging rather than CPU compute.",
    "static-placement":
        "Experts explicitly pinned across the GPUs and CPU. Isolates the "
        "value of placement alone, with no experimental runtime feature.",
    "static-plus-cache":
        "Static placement plus the experimental expert transfer cache. "
        "Only meaningful measured against static-placement.",
}


@dataclass
class BaselineCandidate:
    """One member of the comparison set."""
    name: str
    description: str
    available: bool
    plan: object = None
    reason: str = ""          # why unavailable
    config: dict = field(default_factory=dict)


@dataclass
class BaselineSet:
    """The four candidates plus what is being held constant across them."""
    profile_name: str
    candidates: list = field(default_factory=list)
    invariants: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def available(self):
        return [c for c in self.candidates if c.available]

    def by_name(self, name):
        return next((c for c in self.candidates if c.name == name), None)


def _expert_offload_rule(hardware, profile):
    """An -ot rule pinning the upper expert layers to CPU.

    Deliberately simple and deterministic: acceptance wants a placement
    that is the same every run, not the best one the tier planner can
    find. The tier planner's own output is already a separate plan.
    """
    try:
        import modelctl_vram
        layout = modelctl_vram.gguf_model_layout(profile.get("model_path", ""))
    except Exception:
        layout = None
    if not layout or not layout.get("is_moe"):
        return ""
    blocks = layout.get("block_count") or 0
    if blocks < 4:
        return ""
    # Upper half of the expert layers to CPU: enough to exercise the
    # heterogeneous path on any model big enough to need it.
    first = blocks // 2
    return rf"-ot blk\.({first}|[{first + 1}-9]|[1-9][0-9])\.ffn_.*_exps\.=CPU"


def build_candidate_set(profile, hardware=None, backend=None) -> BaselineSet:
    """Build the four Release A candidates for one profile.

    backend is a modelctl_launch.ResolvedBackend; without it the
    cache candidate cannot be shown to be supported and is marked
    unavailable rather than assumed.
    """
    profile = modelctl.normalize_profile(dict(profile))
    name = profile.get("name", "")
    result = BaselineSet(profile_name=name)

    if hardware is None:
        import modelctl_hardware
        hardware = modelctl_hardware.capture_hardware_snapshot()

    base_config = dict(profile.get("config", {}))
    extra = base_config.get("extra", "") or ""

    # --- what is held fixed across the set --------------------------------
    caps = getattr(backend, "capabilities", None) or {}
    result.invariants = {
        "model_path": profile.get("model_path", ""),
        "context": base_config.get("ctx"),
        "cache_type_k": base_config.get("cache_type_k"),
        "cache_type_v": base_config.get("cache_type_v"),
        "binary": getattr(backend, "binary", ""),
        "binary_fingerprint": getattr(backend, "binary_fingerprint", ""),
        "capability_fingerprint": getattr(backend, "capability_fingerprint", ""),
        "hardware_fingerprint": getattr(hardware, "fingerprint", ""),
    }
    if not result.invariants["binary_fingerprint"]:
        result.warnings.append(
            "no resolved backend: results cannot be tied to a specific "
            "binary, so they are not comparable across rebuilds")

    def plan_for(config_overrides, cache_profile=None):
        p = copy.deepcopy(profile)
        if cache_profile is not None:
            p["moe_cache"] = cache_profile
        merged = {**base_config, **config_overrides}
        p["config"] = merged
        return modelctl_plans._make_plan(
            p, {}, "release-a-baseline", hardware, capabilities=caps)

    # --- A. stock mmap with CPU spill -------------------------------------
    stock_extra = modelctl._strip_moe_cache_flags(extra.split()) if extra else []
    stock_extra = " ".join(t for t in stock_extra if t != "--no-mmap")
    result.candidates.append(BaselineCandidate(
        name="stock-mmap",
        description=DESCRIPTIONS["stock-mmap"],
        available=True,
        config={"extra": stock_extra},
        plan=plan_for({"extra": stock_extra}, cache_profile={"mode": "off"})))

    # --- B. fully resident RAM spill --------------------------------------
    # Only offered when the machine can actually hold it; a --no-mmap run
    # that OOMs measures nothing.
    spill = result.candidates[0].plan.claim.ram_bytes
    available_ram = getattr(hardware, "ram_available_bytes", 0)
    if spill <= 0:
        avail, why = False, "this plan spills nothing to the host"
    elif available_ram and spill > available_ram * 0.9:
        avail, why = False, (
            f"needs {modelctl._format_size(spill)} resident but only "
            f"{modelctl._format_size(available_ram)} is available")
    else:
        avail, why = True, ""
    resident_extra = (stock_extra + " --no-mmap").strip()
    result.candidates.append(BaselineCandidate(
        name="resident-ram",
        description=DESCRIPTIONS["resident-ram"],
        available=avail, reason=why,
        config={"extra": resident_extra},
        plan=(plan_for({"extra": resident_extra}, cache_profile={"mode": "off"})
              if avail else None)))

    # --- C. static heterogeneous placement --------------------------------
    rule = _expert_offload_rule(hardware, profile)
    gpus = [g for g in getattr(hardware, "gpus", ()) if g.enabled]
    if not rule:
        avail, why = False, "not a MoE model, or its layer count is unknown"
    else:
        avail, why = True, ""
    static_extra = (stock_extra + " " + rule).strip()
    if len(gpus) > 1:
        static_config = {"extra": static_extra,
                         "device": ",".join(g.device for g in gpus[:2]),
                         "fit": "off"}
    else:
        static_config = {"extra": static_extra, "fit": "off"}
    result.candidates.append(BaselineCandidate(
        name="static-placement",
        description=DESCRIPTIONS["static-placement"],
        available=avail, reason=why,
        config=static_config,
        plan=(plan_for(static_config, cache_profile={"mode": "off"})
              if avail else None)))

    # --- D. static placement plus the transfer cache ----------------------
    cache_ok = False
    if backend is not None:
        try:
            import modelctl_capabilities
            cache_ok = modelctl_capabilities.is_cache_capable(caps)
        except Exception:
            cache_ok = False
    if not avail:
        cache_why = "static-placement is unavailable, so there is nothing to add the cache to"
    elif not cache_ok:
        cache_why = ("this runtime does not report moe_weight_transfer_cache; "
                     "the cache candidate would measure a command that never "
                     "carries the flag")
    else:
        cache_why = ""
    cache_profile = None
    if cache_ok and avail:
        budget = min((g.total_bytes for g in gpus), default=0) // 8
        cache_profile = {
            "mode": "manual",
            "gpu": {"budgets_bytes": {g.device: budget for g in gpus},
                    "policy": "slru", "admission_misses": 2,
                    "pin_shared_experts": True},
            "decode": {"admit_to_gpu_cache": True, "miss_execution": "gpu"},
            "prefill": {"admit_to_gpu_cache": False},
        }
    result.candidates.append(BaselineCandidate(
        name="static-plus-cache",
        description=DESCRIPTIONS["static-plus-cache"],
        available=bool(cache_ok and avail), reason=cache_why,
        config=static_config,
        plan=(plan_for(static_config, cache_profile=cache_profile)
              if (cache_ok and avail) else None)))

    for c in result.candidates:
        if not c.available and c.reason:
            result.warnings.append(f"{c.name}: {c.reason}")
    return result


def comparability_report(runs) -> dict:
    """Are these measurements actually comparable with each other?

    Release A's whole claim is "these four were measured the same way". A
    set where the binary or the context changed halfway through is a set
    of unrelated numbers, and saying so is more useful than ranking them.
    """
    fields = ("binary_fingerprint", "hardware_fingerprint",
              "capability_digest", "actual_context", "cache_state")
    seen = {f: set() for f in fields}
    for run in runs or []:
        for f in fields:
            value = run.get(f) if hasattr(run, "get") else None
            if value not in (None, ""):
                seen[f].add(value)

    differing = {f: sorted(str(v) for v in vals)
                 for f, vals in seen.items() if len(vals) > 1}
    return {
        "comparable": not differing,
        "differing": differing,
        "runs": len(runs or []),
        "explanation": (
            "all runs share one binary, hardware, capability set, context "
            "and cache state" if not differing else
            "these runs do not describe the same conditions: "
            + ", ".join(f"{f} varies" for f in differing)),
    }
