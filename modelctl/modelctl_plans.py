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
    F. Remote fleet plans (a contiguous layer range on an RPC node)

Plan IDs are stable hashes of the normalized plan representation.
"""
import hashlib
import re
import json
import math
import os
from dataclasses import dataclass, field, asdict, replace

import modelctl
import modelctl_tiers
import modelctl_vram


@dataclass(frozen=True)
class ResourceClaim:
    # {device: bytes} -- the STATIC reservation per device (weights + KV
    # + overhead). NOT the peak: the dynamic expert-cache budget is an
    # additional allocation on top. Admission goes through
    # vram_admission_bytes(), never this field alone.
    vram_bytes: dict
    # CPU-addressed model bytes. NOT a promise of residency: when
    # storage_mode is "mmap" most of it is addressed rather than copied,
    # and it is not what admission charges -- ram_admission_bytes()
    # defines that. Use ram_resident_bytes for "how much RAM this will
    # actually hold".
    ram_bytes: int
    storage_mode: str         # "none", "mmap", "direct"
    expected_context: int | None
    breakdown: dict = field(default_factory=dict)
    storage_path: str = ""    # actual model file path
    model_bytes: int = 0      # total model weight bytes
    cache_bytes: int = 0      # dynamic cache budget separate from static VRAM

    # --- per-device VRAM decomposition -------------------------------
    # Reported separately because "12 GiB on SYCL0" answers nothing about
    # whether raising the context or the cache budget will still fit.
    # static + kv + overhead sum to vram_bytes[device].
    vram_static_bytes: dict = field(default_factory=dict)   # weights
    vram_kv_bytes: dict = field(default_factory=dict)       # KV cache
    vram_overhead_bytes: dict = field(default_factory=dict)  # compute reserve
    # The dynamic expert-cache budget is ADDITIONAL to vram_bytes, matching
    # cache_bytes: it is a separate allocation the tier planner reserves
    # before placing static experts, not a slice of the static claim.
    vram_cache_bytes: dict = field(default_factory=dict)

    # --- RAM and storage ---------------------------------------------
    # Bytes that really will be resident: the non-mmap portion. An mmap
    # plan's model bytes are NOT counted here -- presenting mmap size as
    # guaranteed resident RAM is the specific error this split exists to
    # prevent.
    ram_resident_bytes: int = 0
    # Model bytes reachable through mmap. Resident only as far as the page
    # cache decides, which is why this is separate from the above.
    mmap_bytes: int = 0
    # The subset of mmap_bytes expected to stay hot. For a sparse MoE this
    # is far smaller than the model: only routed experts are touched.
    page_cache_working_set_bytes: int = 0
    # Temporary space an acquisition needs on top of the final file.
    staging_bytes: int = 0
    # Block device backing storage_path, so two plans on different disks
    # are not compared as if their read costs were the same.
    storage_device: str = ""

    # Kept for compatibility; equals ram_resident_bytes. It never meant
    # VRAM, though it used to be computed as if it did.
    expected_resident_bytes: int = 0

    # "gguf" when the claim was computed from the parsed tensor layout
    # (exact per-device math), "estimate" when it fell back to whole-model
    # heuristics. Plan-time admission only trusts exact claims: degrading
    # a plan over an estimate would act on numbers known to be wrong.
    layout_source: str = ""

    # breakdown is the same decomposition in nested dict form, for UI and
    # support bundles:
    # {"vram": {"SYCL0": {"static": N, "kv": N, "overhead": N, "cache": N}},
    #  "ram": {"resident": N, "mmap": N, "page_cache_working_set": N,
    #          "staging": N}}
    # It is display data only; admission reads vram_admission_bytes()
    # and ram_admission_bytes(), never this dict.

    def vram_total(self) -> int:
        return sum(self.vram_bytes.values())

    def device_breakdown(self, device: str) -> dict:
        """Per-component claim on one device.

        `total` is the static reservation (static + kv + overhead).
        `cache` is additional; `peak` is what the device actually needs
        once the expert cache is allocated.
        """
        static = self.vram_static_bytes.get(device, 0)
        kv = self.vram_kv_bytes.get(device, 0)
        overhead = self.vram_overhead_bytes.get(device, 0)
        cache = self.vram_cache_bytes.get(device, 0)
        total = self.vram_bytes.get(device, 0)
        return {
            "static": static, "kv": kv, "overhead": overhead,
            "cache": cache, "total": total, "peak": total + cache,
        }

    # --- canonical admission semantics -------------------------------
    # Feasibility, reservations, the coexistence matrix, and capacity
    # display must all admit against the same numbers. These two methods
    # are the only place those numbers are defined; callers never rebuild
    # them from the decomposed fields.

    def vram_admission_bytes(self) -> dict:
        """Peak per-device VRAM the plan needs admitted: the static
        reservation (weights + KV + overhead) plus the dynamic expert-cache
        budget, which is an additional allocation, not a slice of the
        static claim."""
        peak = dict(self.vram_bytes)
        for dev, cache in self.vram_cache_bytes.items():
            peak[dev] = peak.get(dev, 0) + cache
        return peak

    def ram_admission_bytes(self) -> int:
        """RAM the plan needs admitted: bytes that will actually be
        resident, plus staging buffers. mmap-addressed model bytes are
        NOT included -- the page cache holds them only as far as it
        decides, and charging them as resident refuses every oversized
        MoE served through mmap. They stay visible in mmap_bytes /
        page_cache_working_set_bytes as storage-pressure information,
        and the global page-cache headroom is the budget side's
        configured RAM reserve.

        ram_resident_bytes is 0 for claims built before the D1 split, so
        a non-mmap claim with no resident figure falls back to ram_bytes
        (for such a plan the two mean the same thing)."""
        resident = self.ram_resident_bytes
        if self.storage_mode != "mmap" and not resident:
            resident = self.ram_bytes
        return resident + self.staging_bytes


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
    config: dict = field(default_factory=dict)  # profile["config"] fields this
                                                  # plan implies -- select_plan()
                                                  # applies this back onto the
                                                  # profile when a user picks it.


@dataclass(frozen=True)
class RuntimePolicy:
    objective: str            # "balanced", "fastest_generation", etc.
    pinned_plan_id: str | None
    allow_fallback: bool
    allow_untested: bool
    minimum_context: int | None
    maximum_cpu_bytes: int | None
    maximum_storage_tier: int  # 1=gpu, 2=gpu+ram, 3=gpu+ram+storage
    # How much better an experimental plan must measure than the best safe
    # baseline before it is allowed to outrank it, as a fraction (0.10 =
    # 10%). An experimental path that is merely equal is not worth
    # its risk, and one that wins by noise is not winning. Set to 0 to rank
    # experimental plans purely on score.
    experimental_margin: float = 0.10


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


def is_experimental_plan(plan) -> bool:
    """Does this plan depend on the fork's experimental MoE features?

    Lives here rather than in modelctl_evidence because ranking (the
    experimental-margin guardrail) needs it too and evidence already
    imports this module, so
    the other direction would be circular.
    """
    if (getattr(plan.claim, "cache_bytes", 0) or 0) > 0:
        return True
    return bool((getattr(plan, "decision_data", None) or {}).get("hybrid"))


def is_baseline_plan(plan) -> bool:
    """The profile's own configuration -- the plan a user already trusts."""
    return getattr(plan, "source", "") == "current-profile"


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
    if source == "fleet-rpc":
        pl = ((config.get("rpc") or {}).get("placements") or [{}])[0]
        first, last = pl.get("first_layer"), pl.get("last_layer")
        # The buffer type carries the endpoint in brackets; the label
        # wants the endpoint, since that is what names the machine.
        buf = pl.get("buffer_type", "RPC")
        where = buf.partition("[")[2].rstrip("]") or buf
        # "experts of", not "layers": only routed experts cross the wire.
        return f"experts of layers {first}-{last} on {where}"
    if source == "tier-planner":
        tier = config.get("_tier", "?")
        return f"tier {tier} plan"
    if source.startswith("cache-"):
        base = f"{device} only" if device else source
        return f"{base} {source}"

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


# Fraction of CPU-side MoE expert weights expected to stay hot in the page
# cache. Only the experts a request actually routes to are touched, so the
# working set is far smaller than the offloaded bytes. A rough constant is
# honest here in a way that "all of it" is not; storage calibration and
# measured page faults are what refine it.
_MOE_PAGE_CACHE_FRACTION = 0.25


def _block_device_for(path):
    """Block device backing `path`, or "" when it cannot be determined.

    Two plans reading from different disks do not have the same read cost,
    so the claim records which one it is rather than leaving the reader to
    assume there is only ever one.
    """
    if not path:
        return ""
    try:
        import modelctl_storage
        info = modelctl_storage.probe_storage(path)
        devices = getattr(info, "block_devices", ()) or ()
        return devices[0] if devices else (info.mount_point or "")
    except Exception:
        return ""


def _make_claim(profile, config, hardware, capabilities=None):
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
        base_gpu = gpu_total
        overhead = 0
        shared_gpu = gpu_total + aux_gpu
        # estimate_from_parts folds KV and overhead into one number; the
        # split is not recoverable, so report it as static rather than
        # inventing a decomposition.
        shared_static, shared_kv, shared_overhead = shared_gpu, 0, 0
    else:
        # GPU-resident total: non-expert + unpinned experts + KV + overhead
        base_gpu = non_expert + default_gpu_experts
        overhead = modelctl_vram.compute_overhead_bytes(base_gpu)
        shared_gpu = base_gpu + kv + overhead + aux_gpu
        shared_static, shared_kv, shared_overhead = base_gpu + aux_gpu, kv, overhead

    # ---- distribute shared GPU bytes across devices ---------------------
    # Each component is distributed by the same rule as the total, so the
    # parts always sum back to vram_bytes.
    device = config.get("device", "")
    split = config.get("tensor_split", "")
    vram_map = dict(assigned)
    # Bytes pinned by -ot rules are static weights by definition.
    static_map = dict(assigned)
    kv_map = {}
    overhead_map = {}

    def _add(dev, total, static, kvb, over):
        vram_map[dev] = vram_map.get(dev, 0) + total
        static_map[dev] = static_map.get(dev, 0) + static
        kv_map[dev] = kv_map.get(dev, 0) + kvb
        overhead_map[dev] = overhead_map.get(dev, 0) + over

    if split and "," in split and hardware:
        import modelctl_hardware as _hw
        ratios = [float(x) for x in split.split(",")]
        # Position i of --tensor-split maps to position i of the command's
        # --device list when it carries one, else to llama.cpp's own
        # enumeration order (SYCL index order). It never maps to the
        # role/size ordering of enabled_gpus() -- attributing shares by
        # that order charged the wrong cards whenever the two disagreed,
        # admitting coexisting models onto a card that was already full.
        m = re.search(r"--device[= ]([A-Za-z0-9_,]+)", extra or "")
        if m:
            order = m.group(1).split(",")
        else:
            def _dev_index(g):
                num = re.search(r"(\d+)$", g.device)
                return int(num.group(1)) if num else 0
            order = [g.device for g in
                     sorted(_hw.enabled_gpus(hardware), key=_dev_index)]
        denom = sum(ratios) or 1
        for i, ratio in enumerate(ratios):
            if i < len(order):
                share = ratio / denom
                _add(order[i],
                     int(shared_gpu * share), int(shared_static * share),
                     int(shared_kv * share), int(shared_overhead * share))
    elif device:
        _add(device, shared_gpu, shared_static, shared_kv, shared_overhead)
    elif hardware:
        import modelctl_hardware as _hw
        _eg = _hw.enabled_gpus(hardware)
        dev = _eg[0].device if _eg else ""
        _add(dev, shared_gpu, shared_static, shared_kv, shared_overhead)

    uses_mmap = "--no-mmap" not in extra
    storage = "mmap" if cpu_bytes and uses_mmap else "none"

    # Dynamic cache budget (from moe_cache config), per device.
    cache_budget = 0
    cache_map = {}
    mc = profile.get("moe_cache", {})
    if mc.get("mode", "off") != "off":
        budgets = mc.get("gpu", {}).get("budgets_bytes", {})
        declared = {d: b for d, b in budgets.items() if b > 0}
        per_device_caps = bool((capabilities or {}).get("features", {})
                               .get("moe_cache_per_device_budgets"))
        # The runtime's --moe-cache-bytes is a single UNIFORM per-GPU
        # budget: server.cpp stores one global and ggml-sycl's lazy init
        # gives that same value to every device that creates a cache. So a
        # profile asking for {SYCL0: 4 GiB, SYCL1: 2 GiB} actually gets
        # 4 GiB on *both*. Reserving the declared 2 GiB on SYCL1 would
        # under-reserve by 2 GiB against what the runtime allocates, and
        # the cache would collide with statically placed experts.
        #
        # Reserve the uniform figure on every device the profile's budget
        # map names -- the planner's half of the collapse that
        # modelctl.build_server_args() applies to the flag. Known limit:
        # the runtime allocates on every device that CREATES a cache,
        # which placement decides; a participating device the budget map
        # does not name is still unreserved here.
        if per_device_caps:
            # Schema 3+: the runtime honours the map literally, so the
            # claim charges each named device its own budget and nothing
            # to the devices the map omits (they get no cache).
            cache_map = dict(declared)
            cache_budget = max(declared.values(), default=0)
        else:
            cache_budget = max(declared.values(), default=0)
            cache_map = {d: cache_budget for d in declared}

    # ---- RAM vs mmap ---------------------------------------------------
    # CPU-side weights are only *resident* when mmap is off. With mmap on
    # they are addressed, and residency is the page cache's decision --
    # reporting them as RAM is what makes an oversized MoE look like it
    # needs 200 GiB of RAM it does not need.
    if cpu_bytes and uses_mmap:
        mmap_bytes = cpu_bytes
        ram_resident = 0
        # Only routed experts get touched per token, so the hot set is a
        # fraction of the CPU-side weights rather than all of them. For a
        # dense model there is no such saving.
        if layout and layout.get("is_moe"):
            working_set = int(cpu_bytes * _MOE_PAGE_CACHE_FRACTION)
        else:
            working_set = cpu_bytes
    else:
        mmap_bytes = 0
        ram_resident = cpu_bytes
        working_set = 0

    storage_device = _block_device_for(model_path)

    return ResourceClaim(
        vram_bytes=vram_map,
        ram_bytes=cpu_bytes,
        storage_mode=storage,
        expected_context=ctx,
        storage_path=model_path,
        model_bytes=weights,
        cache_bytes=cache_budget,
        vram_static_bytes=static_map,
        vram_kv_bytes=kv_map,
        vram_overhead_bytes=overhead_map,
        vram_cache_bytes=cache_map,
        ram_resident_bytes=ram_resident,
        mmap_bytes=mmap_bytes,
        page_cache_working_set_bytes=working_set,
        staging_bytes=0,
        storage_device=storage_device,
        expected_resident_bytes=ram_resident,
        layout_source="gguf" if layout else "estimate",
        breakdown={
            "vram": {d: {"static": static_map.get(d, 0),
                         "kv": kv_map.get(d, 0),
                         "overhead": overhead_map.get(d, 0),
                         "cache": cache_map.get(d, 0),
                         "total": vram_map.get(d, 0)}
                     for d in vram_map},
            "ram": {"resident": ram_resident,
                    "mmap": mmap_bytes,
                    "page_cache_working_set": working_set,
                    "staging": 0},
        },
    )


def _usable_vram_map(hardware):
    """Per-device usable VRAM capacity: limit_pct of the card minus its
    configured reserve. Capacity-based on purpose -- free bytes are
    volatile and belong to launch-time feasibility, not plan admission."""
    try:
        frac = modelctl.load_defaults().get("vram_limit_pct", 90) / 100.0
    except Exception:
        frac = 0.90
    import modelctl_hardware as _hw
    usable = {g.device: max(0, int(g.total_bytes * frac) - g.reserve_bytes)
              for g in _hw.enabled_gpus(hardware)}
    # Remote fleet devices join the same map, so a remote placement is
    # admitted -- or refused, or degraded -- by the identical code path a
    # local card goes through. The budget is the operator's declared
    # ceiling, NOT limit_pct of what the node reports: we do not get to
    # decide how much of someone else's machine to take.
    try:
        import modelctl_fleet
        usable.update(modelctl_fleet.admission_budgets())
    except Exception:
        pass
    return usable


def _apply_rpc_placement(claim, merged):
    """Re-attribute remotely-placed weight bytes from local devices to the
    node's admission key.

    Without this a fleet plan would be charged twice: once on the local
    card the layers no longer live on, and once against the node budget.
    The local side keeps its KV cache and overhead -- only static weight
    bytes move, because only weights are what the -ot rule relocated.
    """
    import modelctl_fleet
    adm = ((merged or {}).get("rpc") or {}).get("admission") or {}
    if not adm:
        return claim
    static = dict(claim.vram_static_bytes)
    vram = dict(claim.vram_bytes)

    # Fold the client-side buffer names in FIRST. The two plan families
    # arrive here in different states: _compile_rpc_plans leaves its
    # placement out of config["extra"], so the moved bytes are still
    # charged to a local card and the debit below is what relocates them.
    # A tier-ladder plan writes its placement into extra as -ot rules, so
    # _make_claim has ALREADY charged those bytes to the llama.cpp-side
    # buffer name ("RPC0[host:port]", plus the bare "RPC0"/"RPC1" from
    # --device). Those are the same bytes adm describes, under a second
    # name no budget map contains -- left in place they read as an extra
    # unbudgeted device and deny the plan at reservation (2026-08-04).
    # Matches what the two producers actually emit -- buffer_type_for's
    # "RPC{i}[{endpoint}]" and device_name_for's "RPC{i}" -- and never an
    # admission key, which is_remote_key claims by its "RPC:" prefix.
    # For every planner-emitted config vram[alias] == static[alias] (an
    # alias reaches the claim only through the -ot `assigned` map, which
    # seeds both, or through a --device slot whose tensor-split share is
    # 0), so summing static loses nothing the pop discards.
    already_remote = 0
    for dev in [d for d in set(vram) | set(static)
                if re.match(r"RPC\d", str(d))
                and not modelctl_fleet.is_remote_key(d)]:
        already_remote += static.get(dev, 0)
        vram.pop(dev, None)
        static.pop(dev, None)

    # Debit only what the -ot rules had NOT already moved off the local
    # cards. Without the subtraction the fold above and this loop each
    # relocate the same bytes, and the second pass eats a local card's
    # own resident experts: the 122B ladder plan was charging SYCL0 1.02
    # GiB instead of 25.9 GiB, understating the local claim by 24.9 GiB.
    remaining = max(0, sum(adm.values()) - already_remote)
    # Debit the device carrying the most static weight: that is where the
    # relocated layers were otherwise going to sit.
    for dev in sorted(static, key=lambda d: -static[d]):
        if remaining <= 0:
            break
        take = min(static[dev], remaining)
        static[dev] = static[dev] - take
        vram[dev] = max(0, vram.get(dev, 0) - take)
        remaining -= take
    for key, byts in adm.items():
        vram[key] = int(byts)
    return replace(claim, vram_bytes=vram, vram_static_bytes=static)


def _admission_overflow(claim, usable):
    """Per-device oversubscription of a claim against usable capacity.

    Demand is the canonical peak (vram_admission_bytes) with the
    per-device compute allowance floored at the tier planner's 1.5 GiB
    reserve: the claim's own overhead is 10% of resident weights, which
    covers the reserve on big residents but not on small ones."""
    overflow = {}
    peak = claim.vram_admission_bytes()
    for dev, need in peak.items():
        if dev not in usable:
            continue
        floor_delta = max(0, modelctl_tiers.GPU_COMPUTE_RESERVE_BYTES
                          - claim.vram_overhead_bytes.get(dev, 0))
        over = need + floor_delta - usable[dev]
        if over > 0:
            overflow[dev] = over
    return overflow


def _shrink_cache_budgets(profile, claim, usable, capabilities):
    """Degradation step 1: shrink the profile's cache budgets so the peak
    fits. Returns (new_moe_cache or None, {dev: chosen_bytes})."""
    mc = profile.get("moe_cache", {})
    declared = {d: b for d, b in mc.get("gpu", {}).get("budgets_bytes", {})
                .items() if b > 0}
    if not declared or mc.get("mode", "off") == "off":
        return None, {}
    per_device = bool((capabilities or {}).get("features", {})
                      .get("moe_cache_per_device_budgets"))
    # Room left for cache on each device the runtime would allocate on
    # (claim.vram_cache_bytes holds the collapse-aware reservation map).
    room = {}
    for dev in claim.vram_cache_bytes:
        static = claim.vram_bytes.get(dev, 0)
        floor_delta = max(0, modelctl_tiers.GPU_COMPUTE_RESERVE_BYTES
                          - claim.vram_overhead_bytes.get(dev, 0))
        room[dev] = max(0, usable.get(dev, 0) - static - floor_delta)
    if per_device:
        chosen = {d: min(b, room.get(d, b)) for d, b in declared.items()}
    else:
        uniform = max(declared.values())
        allowed = min(room.values()) if room else 0
        got = min(uniform, allowed)
        chosen = {d: got for d in declared}
    chosen = {d: int(b) for d, b in chosen.items() if b > 0}
    new_mc = {**mc, "gpu": {**mc.get("gpu", {}), "budgets_bytes": chosen}}
    return new_mc, chosen


def _spill_pinned_expert_layers(extra, layout, overflow):
    """Degradation step 2: move statically pinned routed-expert layers to
    CPU until each device's overflow is covered.

    Rewrites only the routed (_exps) -ot rules; pin rules (shexp, leading
    dense) and every other flag stay verbatim. Returns (new_extra,
    {dev: [moved layers]}); a no-op returns (extra, {})."""
    if not layout or not layout.get("is_moe"):
        return extra, {}
    rules = _parse_ot_rules(extra)
    if not rules:
        return extra, {}
    raw_toks = (extra or "").split()
    try:
        ot_idx = raw_toks.index("-ot")
    except ValueError:
        return extra, {}
    if ot_idx + 1 >= len(raw_toks):
        return extra, {}

    layers_bytes = layout["expert_bytes_per_layer"]
    placement = {}
    for layer in layers_bytes:
        probe = f"blk.{layer}.ffn_gate_exps.weight"
        for rx, tgt in rules:
            if rx.search(probe):
                placement[layer] = tgt
                break
    moved = {}
    for dev, over in sorted(overflow.items()):
        mine = sorted((l for l, t in placement.items() if t == dev),
                      reverse=True)
        freed = 0
        take = []
        for layer in mine:
            if freed >= over:
                break
            take.append(layer)
            freed += layers_bytes[layer]
        for layer in take:
            placement[layer] = "CPU"
        if take:
            moved[dev] = sorted(take)
    if not moved:
        return extra, {}

    # Rebuild the -ot value: keep non-routed parts (pins) verbatim in
    # their original raw (double-backslash) form, regenerate the routed
    # rules from the new placement, CPU catch-all last (first-match-wins).
    raw_parts = []
    for part in raw_toks[ot_idx + 1].split(","):
        pattern, _, target = part.rpartition("=")
        if pattern and target and "_exps" in pattern and "_shexp" not in pattern:
            continue  # routed rule: regenerated below
        raw_parts.append(part)
    gpu_layers = {}
    for layer, tgt in placement.items():
        if tgt != "CPU":
            gpu_layers.setdefault(tgt, []).append(layer)
    for dev in sorted(gpu_layers):
        rx = modelctl_tiers.layers_regex(sorted(gpu_layers[dev]))
        raw_parts.append(f"blk\\\\.{rx}\\\\.ffn_.*_exps={dev}")
    raw_parts.append("ffn_.*_exps=CPU")
    raw_toks[ot_idx + 1] = ",".join(raw_parts)
    return " ".join(raw_toks), moved


def _make_plan(profile, config, source, hardware, extra_warnings=(), decision=None,
               capabilities=None):
    """Build a LaunchPlan from profile + config overrides.

    Plan-time VRAM admission: when a hardware snapshot is available, the
    claim is checked per device against usable capacity (limit_pct minus
    reserves) BEFORE the plan is emitted. An oversubscribed plan is never
    emitted as-is -- the cache budget shrinks toward 0, then statically
    pinned expert layers spill to CPU -- and the plan carries both a
    human-readable warning and a machine-readable record
    (decision_data["admission"]) stating requested, assumed, and chosen.
    A planner-emitted config must never be able to crash at launch for
    VRAM admission reasons."""
    merged = {**profile.get("config", {}), **config}

    def _claim_for(prof, mrg):
        # Every claim in this function goes through here: the degradation
        # steps below rebuild the claim, and a rebuild that forgot the
        # remote re-attribution would silently charge the moved layers to
        # the local card again.
        return _apply_rpc_placement(
            _make_claim(prof, mrg, hardware, capabilities), mrg)

    claim = _claim_for(profile, merged)

    admission = None
    adm_warnings = []
    # Admission needs numbers it can trust and a plan whose VRAM claim
    # means what it says: estimate-quality claims (no parsed GGUF layout)
    # and hybrid plans (whose deliberately conservative claim charges
    # CPU-miss weights to VRAM) are left to launch-time feasibility,
    # which refuses rather than degrades.
    if (hardware is not None and claim.layout_source == "gguf"
            and not (decision or {}).get("hybrid")):
        usable = _usable_vram_map(hardware)
        overflow = _admission_overflow(claim, usable)
        if overflow:
            requested_cache = {d: int(b) for d, b in
                               claim.vram_cache_bytes.items()}
            requested_peak = {d: int(b) for d, b in
                              claim.vram_admission_bytes().items()}
            degradations = []
            # Step 1: fall back to a smaller cache.
            if any(requested_cache.values()):
                new_mc, chosen = _shrink_cache_budgets(
                    profile, claim, usable, capabilities)
                if new_mc is not None:
                    profile = {**profile, "moe_cache": new_mc}
                    claim = _claim_for(profile, merged)
                    for dev in sorted(overflow):
                        req = requested_cache.get(dev, 0)
                        got = claim.vram_cache_bytes.get(dev, 0)
                        if got < req:
                            degradations.append({
                                "action": "shrink_cache", "device": dev,
                                "requested_bytes": req,
                                "chosen_bytes": int(got)})
                            adm_warnings.append(
                                modelctl_tiers._cache_fallback_warning(
                                    dev, req, usable.get(dev, 0), got))
                    overflow = _admission_overflow(claim, usable)
            # Step 2: spill statically pinned expert layers to CPU.
            if overflow:
                layout = None
                model_path = profile.get("model_path", "")
                if model_path and os.path.exists(model_path):
                    try:
                        layout = modelctl_vram.gguf_model_layout(model_path)
                    except Exception:
                        layout = None
                new_extra, moved = _spill_pinned_expert_layers(
                    merged.get("extra", ""), layout, overflow)
                if moved:
                    merged = {**merged, "extra": new_extra}
                    claim = _claim_for(profile, merged)
                    for dev, layers in sorted(moved.items()):
                        degradations.append({
                            "action": "spill_experts", "device": dev,
                            "layers": layers})
                        adm_warnings.append(
                            f"statically pinned expert layers "
                            f"{layers} exceed the free VRAM measured on "
                            f"{dev} -- they spill to CPU for this plan")
                    overflow = _admission_overflow(claim, usable)
            if overflow:
                adm_warnings.append(
                    "plan oversubscribes "
                    + ", ".join(sorted(overflow))
                    + " even after degradation -- it is expected to fail "
                    "VRAM admission at launch")
            admission = {
                "fits": not overflow,
                "requested": {"per_device_peak_bytes": requested_peak,
                              "cache_budgets_bytes": requested_cache},
                "assumed": {"usable_bytes": {d: int(b)
                                             for d, b in usable.items()}},
                "chosen": {"per_device_peak_bytes":
                           {d: int(b) for d, b in
                            claim.vram_admission_bytes().items()},
                           "cache_budgets_bytes":
                           {d: int(b) for d, b in
                            claim.vram_cache_bytes.items()}},
                "degradations": degradations,
            }

    args = modelctl.build_server_args({**profile, "config": merged},
                                      capabilities=capabilities)
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
        # Every shard, not just the named file: swapping shard 2 of 3 has to
        # produce a different plan. Identical to file_fingerprint
        # for unsharded models, so single-file plan IDs are unchanged.
        "model_fp": modelctl_vram.model_fingerprint(profile.get("model_path")),
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
        # Cache configuration is part of plan identity: a budget or policy
        # change must produce a different plan ID.
        "moe_cache": profile.get("moe_cache", {}),
        # Remote placement likewise. Without this, a fleet plan and the
        # local plan it was derived from normalize identically and
        # collide on ID -- the second one would be silently dropped by
        # compile_launch_plans' dedupe, which is exactly the plan a user
        # asked for.
        "rpc": merged.get("rpc", {}),
        "env": sorted(profile.get("env") or []),
    }
    pid = _plan_id(normalized)
    label = _plan_label(merged, source, gpu_names)

    decision_data = dict(decision or {})
    if admission is not None:
        decision_data["admission"] = admission

    return LaunchPlan(
        id=pid,
        profile_name=profile.get("name", ""),
        backend=profile.get("backend", "llama-cpp"),
        label=label,
        argv=tuple(args),
        env=env,
        claim=claim,
        # Capacity figures shown in the UI are the admission values, so
        # what the wizard displays is what the worker will actually admit
        # against. The mmap-addressed size stays as separate information.
        estimated={"total_vram": sum(claim.vram_admission_bytes().values()),
                   "ram": claim.ram_admission_bytes(),
                   "mmap": claim.mmap_bytes,
                   "context": claim.expected_context},
        source=source,
        warnings=tuple(extra_warnings) + tuple(adm_warnings),
        decision_data=decision_data,
        config=dict(merged),
    )


def _remote_rungs() -> list:
    """Fleet devices the expert ladder may place experts on.

    Recorded-present nodes with a declared budget. usable_nodes() opens
    no socket -- presence was refreshed by whoever last probed -- so an
    unreachable or expired node contributes nothing and the plan set
    degrades to local-only rather than promising memory that is not
    there.

    Extracted so the selection preview and compile_launch_plans build
    rungs identically: a preview that offered a different set of remote
    devices than the compiler would answer for a machine the operator
    does not have.
    """
    try:
        import modelctl_fleet
        rungs = []
        for node in modelctl_fleet.usable_nodes():
            for dev_index, device in enumerate(node.devices):
                if device.budget_bytes <= 0:
                    continue
                rungs.append({
                    "name": modelctl_fleet.admission_key(node.name,
                                                         device.name),
                    "endpoint": node.endpoint,
                    "local_device_index": dev_index,
                    "kind": device.kind,
                    "budget_bytes": int(device.budget_bytes)})
        return rungs
    except Exception:
        return []


def plan_for_selection(profile, selection, inputs=None, remote_rungs=None,
                       layout=None):
    """Where the planner puts this model's weights, for a chosen set of
    devices.

    This is the one entry point the placement UI needs. The operator's
    choices -- which devices may be used, and how much of each -- are
    expressed as filtered planner inputs (see
    modelctl_tiers.select_inputs), so the answer comes from the planner
    that actually launches rather than from a second implementation
    beside it. A screen that works out the split itself is a screen that
    can disagree with what runs.

    An empty selection returns the automatic placement: what the machine
    would do on its own, which is what the operator sees before touching
    anything.

    inputs/remote_rungs/layout are injectable for tests; production
    resolves them from the profile and the fleet.
    """
    import modelctl_tiers
    if inputs is None:
        inputs, _source = modelctl.resolve_planning_inputs(profile)
    rungs = _remote_rungs() if remote_rungs is None else remote_rungs
    inventory, ram, rungs = modelctl_tiers.select_inputs(
        inputs["inventory"], inputs.get("ram_available_bytes", 0), rungs,
        selection, inputs["vram_limit_pct"])
    return modelctl_tiers.plan_tiers(
        profile, inventory, inputs["vram_limit_pct"], inputs["primary"],
        ram_available=ram, layout=layout,
        cache_request=profile.get("moe_cache"),
        capabilities=inputs.get("capabilities"),
        hw_settings=inputs.get("hw_settings"), remote_rungs=rungs)


def launch_plan_for_selection(profile, selection, hardware=None, inputs=None,
                              remote_rungs=None, layout=None,
                              capabilities=None):
    """The LaunchPlan for a stored selection -- what actually runs.

    plan_for_selection answers where the weights land; this turns that
    answer into the thing the worker can exec: argv, a resource claim to
    charge against the admission gate, and an id to record events under.
    Built by handing the tier plan's config to the SAME _make_plan() every
    other candidate goes through, so a selection is not a second kind of
    plan the rest of the system has to learn.

    The point of the indirection is that the launcher and the placement
    screen now derive their layout from one function and one selection.
    Before this they were two implementations of "where does this model
    run" -- the screen asking the planner, the launcher compiling its own
    candidates and promoting a pinned id -- and they could, and did,
    disagree.

    The id still hashes the compiled config rather than the selection.
    That was an open question and the answer fell out of removing the pin:
    an id was fragile only because something stored it, and a stored id
    went stale every time the planner improved. Nothing pins by id here --
    the selection is what is durable -- which frees the id to go on being
    what it is, a content hash of one layout, and that is precisely what
    the evidence trail needs to key a measurement to the layout that
    produced it.
    """
    tier_plan = plan_for_selection(profile, selection, inputs=inputs,
                                   remote_rungs=remote_rungs, layout=layout)
    if tier_plan is None:
        return None
    if capabilities is None:
        capabilities = (inputs or {}).get("capabilities")
    return _make_plan(profile, tier_plan.get("config") or {}, "selection",
                      hardware,
                      extra_warnings=tuple(tier_plan.get("warnings") or ()),
                      capabilities=capabilities)


def plans_for_selection_launch(profile, selection, policy, hardware=None,
                               resolve_inputs=None, remote_rungs=None,
                               layout=None, log=None):
    """Compilations of one stored intent, best first.

    The launcher's replacement for compile-rank-pin when a profile carries
    a selection. It returns a list because a launch may need to try more
    than once, but every entry is the SAME intent compiled against a
    different picture of the machine -- not a ranked set of alternative
    placements the operator never chose.

    The stored selection is never rewritten here, and that is the rule the
    whole ladder hangs on. A closed laptop makes its node unavailable for
    this launch; it does not un-tick it. When the laptop comes back the
    next launch gets it again with nothing to re-select. Degradation is a
    property of one compilation; intent is a property of the profile.

    Most of the ladder already existed inside the planner and is left
    there rather than reimplemented: _remote_rungs omits a node nobody can
    reach (rung 1, unavailable devices drop out), select_inputs clamps a
    ceiling to what a device actually has (part of rung 2), and plan_tiers
    IS the descent from VRAM to RAM to SSD (rung 3). What this adds is the
    rest of rung 2 -- re-planning against the machine as it is now instead
    of the recorded snapshot, which is what "ceilings relax to real
    capacity" means once reality has got tighter -- and the floor.

    The floor is `maximum_storage_tier`. Degradation may not solve the
    problem by streaming from the disk when the profile forbids it;
    reaching it is a named refusal, never a silent slow launch.

    `allow_fallback` finally means something honest. It used to decide
    whether the worker could leave a pinned plan for another compiled
    candidate; with no candidate list to leave for, it now decides whether
    this launch may degrade the selection at all. False is the strict
    reading -- launch exactly what was chosen or fail -- which is the
    right setting for a profile whose measured numbers depend on an exact
    layout.
    """
    log = log or (lambda _message: None)
    resolve = resolve_inputs or modelctl.resolve_planning_inputs
    name = profile.get("name", "this model")
    plans, seen = [], set()
    for refresh in (False, True):
        if refresh and not policy.allow_fallback:
            break
        try:
            inputs, source = resolve(profile, refresh=refresh)
        except Exception as e:
            log(f"could not read the machine ({e})")
            continue
        plan = launch_plan_for_selection(profile, selection, hardware=hardware,
                                         inputs=inputs,
                                         remote_rungs=remote_rungs,
                                         layout=layout)
        if plan is None:
            continue
        # The same layout twice is a wasted launch attempt, not a rung: it
        # happens whenever the live machine still matches the record.
        if plan.id in seen:
            continue
        seen.add(plan.id)
        if (policy.maximum_storage_tier < 3
                and plan.claim.storage_mode == "mmap"):
            log(f"NOT LAUNCHED: this selection needs to stream from storage "
                f"and {name} is limited to "
                f"{'GPU only' if policy.maximum_storage_tier == 1 else 'GPU + RAM'}"
                f" -- widen the selection, raise the storage limit, or free "
                f"memory")
            continue
        if refresh:
            log(f"degraded: the selection did not fit the recorded machine, "
                f"replanned against the machine as it is now ({source})")
        plans.append((plan, 0.0))
    return plans


def current_profile_plan(profile, hardware=None, capabilities=None):
    """The single LaunchPlan meaning "this profile exactly as saved".

    Preview, artifact, llama-swap, and smoke-test paths need that one plan
    without compiling the whole candidate set (which probes hardware and
    the GGUF layout for every variant).  Built through the same
    _make_plan() as every other candidate, so its argv and plan ID are
    identical to the "current profile" candidate compile_launch_plans()
    produces for the same profile.

    hardware is optional: it only refines the resource claim and the
    plan's display label, neither of which a command preview needs.
    """
    return _make_plan(profile, {}, "current-profile", hardware,
                      capabilities=capabilities)


def compile_launch_plans(profile, hardware=None, include_experimental=False):
    """Generate a bounded set of launch plans for a profile.

    Returns 3-12 plans depending on hardware configuration.
    Plans are deterministic and side-effect free.

    Experimental variants (MoE transfer-cache, hybrid CPU-miss) are only
    generated when include_experimental is True or the profile's runtime
    section sets allow_experimental_plans (the variant
    stays disabled by default unless the user opts in).
    """
    if not hardware:
        from modelctl_hardware import capture_hardware_snapshot
        hardware = capture_hardware_snapshot()

    plans = []
    seen_ids = set()

    def add(plan):
        if plan.id not in seen_ids:
            plans.append(plan)
            seen_ids.add(plan.id)

    # Probe once when the profile uses moe_cache so every plan's argv is
    # compiled against the same capabilities the launch-time validation
    # will see (probe result is session/disk-cached).  Without this, a
    # cold probe cache would compile cache-less argv that launch-time
    # validation then approves -- silently under-delivering manual mode.
    caps_for_args = None
    if profile.get("moe_cache", {}).get("mode", "off") != "off":
        try:
            import modelctl_capabilities
            binary = profile.get("binary") or modelctl.LLAMA_SERVER_BIN
            caps_for_args = modelctl_capabilities.probe_backend(binary)
        except Exception:
            caps_for_args = None

    # A. Current profile baseline
    add(_make_plan(profile, {}, "current-profile", hardware,
                   capabilities=caps_for_args))

    # B. Tier planner output. Inputs resolve stored-first: a profile that
    # has planned before replans from its recorded inputs (planning RAM
    # among them), so this candidate cannot drift with instantaneous free
    # memory while the profile is unchanged.
    try:
        inputs, _source = modelctl.resolve_planning_inputs(profile)
        # RPC rungs for the expert ladder: recorded-present fleet devices
        # with declared budgets. usable_nodes() opens no socket (presence
        # was refreshed by the plans view); an empty fleet yields no
        # rungs and a byte-identical local plan set.
        remote_rungs = _remote_rungs()
        tier = modelctl_tiers.plan_tiers(
            profile, inputs["inventory"], inputs["vram_limit_pct"],
            inputs["primary"],
            ram_available=inputs["ram_available_bytes"],
            cache_request=profile.get("moe_cache"),
            capabilities=inputs.get("capabilities") or caps_for_args,
            hw_settings=inputs.get("hw_settings"),
            remote_rungs=remote_rungs)
        if tier and tier.get("config"):
            tc = dict(tier["config"])
            tc["_tier"] = tier.get("tier", "?")
            add(_make_plan(profile, tc, "tier-planner", hardware,
                           extra_warnings=tuple(tier.get("warnings", [])),
                           decision={"tier": tier.get("tier"),
                                     "analysis": tier.get("analysis")},
                           capabilities=caps_for_args))
    except Exception:
        pass

    # F/G/H. MoE cache variants (small/balanced/large) — experimental:
    # only when the user opted in, the profile has moe_cache enabled, and
    # the binary supports it.
    experimental = include_experimental or bool(
        (profile.get("runtime") or {}).get("allow_experimental_plans"))
    mc = profile.get("moe_cache", {})
    if experimental and mc.get("mode", "off") != "off":
        try:
            import modelctl_capabilities
            import modelctl_hardware
            caps = caps_for_args
            if caps and modelctl_capabilities.is_cache_capable(caps):
                d = modelctl.load_defaults()
                limit_pct = d.get("vram_limit_pct", 90)
                try:
                    hw_settings = modelctl_hardware.load_settings()
                except Exception:
                    hw_settings = {}
                dev_cfg = hw_settings.get("devices", {})
                # GPU-resident floor for the feasibility check: KV cache +
                # compute overhead stay on the card even in a spill plan;
                # the cache budget must fit in what they leave behind.
                # (Weights may spill to CPU, so the full static footprint
                # is NOT the right bar -- it would kill every variant for
                # exactly the oversized MoE models the cache is for.)
                cfg0 = profile.get("config", {})
                weights = 0
                model_path = profile.get("model_path", "")
                if model_path and os.path.exists(model_path):
                    try:
                        weights = modelctl_vram.weights_bytes_on_disk(model_path)
                    except Exception:
                        pass
                gpu_resident = 0
                if weights:
                    est = modelctl_vram.estimate_from_parts(
                        weights, int(cfg0.get("ctx", 8192)),
                        cfg0.get("cache_type_k", "q8_0"),
                        cache_type_v=cfg0.get("cache_type_v", "q8_0"))
                    gpu_resident = est["kv_bytes"] + est["overhead"]
                gpus_enabled = [g for g in hardware.gpus if g.enabled]
                for frac, label_suffix in ((0.20, "cache-small"),
                                             (0.50, "cache-balanced"),
                                             (0.80, "cache-large")):
                    for g in gpus_enabled:
                        # Honor the configured per-card limit and hardware
                        # reserve -- the same reality the tier planner uses.
                        usable = (g.total_bytes * limit_pct / 100.0
                                  - dev_cfg.get(g.device, {}).get("reserve_bytes", 0))
                        budget = int(usable * frac)
                        if budget < 64 * (1 << 20):  # minimum 64 MiB
                            continue
                        if (gpu_resident
                                and gpu_resident + budget > usable):
                            continue  # infeasible: cache crowds out KV/overhead
                        # Pass the budget through the STRUCTURED moe_cache
                        # path, not a raw extra flag -- build_server_args
                        # would otherwise emit two --moe-cache-bytes values.
                        variant_mc = {**mc,
                                      "gpu": {**mc.get("gpu", {}),
                                              "budgets_bytes": {g.device: budget}}}
                        cache_cfg = {
                            "device": g.device,
                            "split_mode": "",
                            "tensor_split": "",
                        }
                        plan = _make_plan({**profile, "moe_cache": variant_mc},
                                          cache_cfg, label_suffix, hardware,
                                          decision={"moe_cache": variant_mc},
                                          capabilities=caps)
                        add(plan)
        except Exception:
            pass

    # I. Hybrid plans (GPU cache hit + CPU miss execution) — only when
    # the user opted into experimental plans, the backend supports
    # moe_hybrid_cpu_miss, and the model is oversized.
    if (experimental and mc.get("mode", "off") != "off"
            and mc.get("decode", {}).get("miss_execution") == "cpu"):
        try:
            import modelctl_capabilities
            import modelctl_hardware
            caps = caps_for_args
            if (caps
                    and modelctl_capabilities.supports_hybrid_miss(caps)
                    and modelctl_capabilities.is_cache_capable(caps)):
                d = modelctl.load_defaults()
                limit_pct = d.get("vram_limit_pct", 90)
                try:
                    hw_settings = modelctl_hardware.load_settings()
                except Exception:
                    hw_settings = {}
                dev_cfg = hw_settings.get("devices", {})
                gpus_enabled = [g for g in hardware.gpus if g.enabled]
                cfg0 = profile.get("config", {})
                weights_h = 0
                model_path = profile.get("model_path", "")
                if model_path and os.path.exists(model_path):
                    try:
                        weights_h = modelctl_vram.weights_bytes_on_disk(model_path)
                    except Exception:
                        pass
                # Hybrid needs oversized models that exceed GPU VRAM.
                total_est = modelctl_vram.estimate_from_parts(
                    weights_h, int(cfg0.get("ctx", 8192)),
                    cfg0.get("cache_type_k", "q8_0"),
                    cache_type_v=cfg0.get("cache_type_v", "q8_0"))
                total_gpu_budget = sum(
                    int(g.total_bytes * limit_pct / 100.0) - g.reserve_bytes
                    for g in gpus_enabled)
                if total_est.get("total", 0) > total_gpu_budget:
                    # Use a moderate cache budget — the miss path handles
                    # the overflow, so we don't need to fill the GPU.
                    for frac, label_suffix in ((0.30, "hybrid-balanced"),
                                                 (0.60, "hybrid-large")):
                        for g in gpus_enabled:
                            usable = (g.total_bytes * limit_pct / 100.0
                                      - dev_cfg.get(g.device, {}).get("reserve_bytes", 0))
                            budget = int(usable * frac)
                            if budget < 64 * (1 << 20):
                                continue
                            variant_mc = {
                                **mc,
                                "gpu": {**mc.get("gpu", {}),
                                        "budgets_bytes": {g.device: budget}},
                                "decode": {**mc.get("decode", {}),
                                           "miss_execution": "cpu"},
                            }
                            hybrid_cfg = {
                                "device": g.device,
                                "split_mode": "",
                                "tensor_split": "",
                            }
                            plan = _make_plan(
                                {**profile, "moe_cache": variant_mc},
                                hybrid_cfg, label_suffix, hardware,
                                extra_warnings=("hybrid: CPU miss execution",),
                                decision={"moe_cache": variant_mc,
                                          "hybrid": True},
                                capabilities=caps)
                            add(plan)
        except Exception:
            pass

    # C. Single-GPU plans
    # Sections C and D admit against the same usable budget as the tier
    # planner and the worker: limit_pct applied, reserves subtracted.
    # Raw total-minus-reserve quietly bypassed the configured capacity
    # margin for exactly these plan families.
    try:
        _limit_frac = modelctl.load_defaults().get("vram_limit_pct", 90) / 100.0
    except Exception:
        _limit_frac = 0.90
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
        budget = int(gpu.total_bytes * _limit_frac) - gpu.reserve_bytes
        if total_need <= budget:
            add(_make_plan(profile, {"device": gpu.device, "split_mode": "",
                                     "tensor_split": ""},
                           "single-gpu", hardware,
                           decision={"gpu": gpu.device, "budget": budget},
                           capabilities=caps_for_args))

    # D. Multi-GPU split plans (capacity-ratio variants)
    if len(gpus) >= 2:
        combined = sum(int(g.total_bytes * _limit_frac) - g.reserve_bytes
                       for g in gpus)
        if total_need <= combined:
            # Usable-budget-ratio split (see the tier-2 rationale in
            # modelctl_tiers: llama.cpp distributes by these weights)
            caps = [int(g.total_bytes * _limit_frac) - g.reserve_bytes
                    for g in gpus]
            gib = [max(1, round(c / (1 << 30))) for c in caps]
            divisor = math.gcd(*gib)
            ratio = ",".join(str(g // divisor) for g in gib)

            dev_list = ",".join(g.device for g in gpus)
            add(_make_plan(profile, {"device": "", "split_mode": "layer",
                                     "tensor_split": ratio,
                                     "extra": f"--device {dev_list}"},
                           "multi-gpu", hardware,
                           decision={"ratio": ratio, "combined_budget": combined,
                                     "device_list": dev_list},
                           capabilities=caps_for_args))

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
                                                 "device_list": dev_list},
                                       capabilities=caps_for_args))

    # E. CPU-spill plans
    if total_need > 0 and gpus:
        primary_gpu = gpus[0]
        budget = primary_gpu.total_bytes - primary_gpu.reserve_bytes
        if total_need > budget:
            ngl = _compute_ngl(weights, budget, profile, hardware)
            if ngl > 0:
                # Preserve the user's non-placement extra flags (ubatch-size
                # etc.) -- only -ngl is planner-owned here.
                # NB: no local "import modelctl_tiers" here -- a function-
                # local import shadows the module-level one and makes the
                # section-B plan_tiers() call above die with
                # UnboundLocalError (silently swallowed by its except).
                _, other_flags = modelctl_tiers.split_extra_flags(
                    profile.get("config", {}).get("extra", ""))
                spill_extra = " ".join([f"-ngl {ngl}"] + other_flags)
                add(_make_plan(profile, {"device": primary_gpu.device, "split_mode": "",
                                         "tensor_split": "",
                                         "extra": spill_extra},
                               "cpu-spill", hardware,
                               extra_warnings=("partial GPU offload",),
                               decision={"ngl": ngl, "gpu": primary_gpu.device},
                               capabilities=caps_for_args))

    # F. Remote fleet variants -- strictly additive.
    #
    # Everything above is the local plan set. These are extra candidates
    # that exist only while a node is registered, enabled, and recorded
    # present at the pinned commit. With no such node this loop adds
    # nothing and `plans` is byte-identical to a fleet-free checkout,
    # which is the property test_fleet_rpc pins down.
    for rpc_plan in _compile_rpc_plans(profile, hardware, caps_for_args):
        add(rpc_plan)

    return plans


def _rpc_layer_budget(profile, budget_bytes):
    """The trailing routed-expert layers that fit a remote device's budget.

    Returns (first_layer, last_layer, expert_bytes), or None when nothing
    can be placed -- including when the model's layout is unknown, since
    an estimate is not good enough to place tensors on another machine
    with.

    Sized in ROUTED-EXPERT bytes, because those are the only bytes that
    move: the -ot rule is expert-only (modelctl_fleet.contiguous_range),
    so each layer's attention and norm tensors stay on the local card.
    Spending whole-layer bytes here was wrong in both directions -- it
    bought too few layers, and because _apply_rpc_placement debits the
    local devices by exactly what this reports, it also understated the
    local claim by the bytes that never left.

    A dense model places nothing: with no routed experts an expert-only
    rule moves zero bytes, and a plan whose remote range is empty pays
    the wire cost for no work.
    """
    model_path = profile.get("model_path", "")
    if not model_path or not os.path.exists(model_path) or budget_bytes <= 0:
        return None
    try:
        layout = modelctl_vram.gguf_model_layout(model_path)
    except Exception:
        return None
    if not layout or not layout.get("block_count"):
        return None
    try:
        # OverflowError too: int(float("inf")) raises it, and this runs on
        # every plan compilation -- a raise here wedges the console, so
        # the house rule is degrade, never raise.
        per_layer = {}
        for key, value in (layout.get("expert_bytes_per_layer") or {}).items():
            if (nbytes := int(value)) > 0:
                per_layer[int(key)] = nbytes
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None
    if not per_layer:
        return None

    # Contiguous and trailing (see _compile_rpc_plans for why), so walk
    # back from the last expert-bearing layer and stop at the first layer
    # that does not fit or that hosts no experts at all.
    last = max(per_layer)
    # Never every expert layer: a remote range that swallows them all
    # leaves the local side holding nothing but attention and the KV
    # cache, which is not a serving topology anyone asked for -- and it
    # makes the local-fallback comparison meaningless.
    limit = len(per_layer) - 1
    chosen, spent = [], 0
    for layer in range(last, -1, -1):
        nbytes = per_layer.get(layer)
        if not nbytes or spent + nbytes > budget_bytes or len(chosen) >= limit:
            break
        chosen.append(layer)
        spent += nbytes
    if not chosen:
        return None
    return min(chosen), last, spent


# Both spellings, matching _make_claim's parser for the same field a few
# hundred lines up: llama.cpp accepts --device=X, and a second parser in
# this file that saw only the space form would miss an existing list and
# append a CONFLICTING --device token instead of extending it.
_DEVICE_ARG_RE = re.compile(r"--device[= ]([A-Za-z0-9_,]+)")


def _attach_rpc_device(merged, client_device):
    """Add the RPC client to the plan's device list with a zero split share.

    Three-valued on purpose, so the caller must test `is None` and not
    truthiness:
      dict -- the config overrides to merge
      {}   -- already correctly configured; nothing to change
      None -- cannot be made safe, so the caller DROPS the plan rather
              than emit one that cannot load

    llama.cpp only offloads to devices named in --device, and every fleet
    config that has ever served -- the tier-4 ladder and the laguna R2
    promotion -- names the RPC client there with a ZERO tensor-split
    share. The share must be zero because the -ot rule already decides
    what lands remotely; a nonzero share would ALSO hand it a slice of
    the automatic layer split, including the attention and norm tensors
    the RPC buffer cannot run.

    --device goes into `extra`, not config["device"], because
    build_server_args emits the split instead of config["device"]
    whenever split_mode and tensor_split are both set -- so a device list
    written to config would silently never reach the argv. `extra` is
    also emitted after placement_args, which keeps --rpc ahead of
    --device (the laguna R2 gotcha).
    """
    extra = str(merged.get("extra", "") or "")
    match = _DEVICE_ARG_RE.search(extra)
    if match:
        devices = [d for d in match.group(1).split(",") if d]
    else:
        devices = [d for d in str(merged.get("device", "") or "").split(",")
                   if d]
    if not devices:
        # No explicit device list to extend. Inventing one would restrict
        # the model to fewer devices than the profile intends.
        return None

    split = str(merged.get("tensor_split", "") or "")
    shares = [s for s in split.split(",") if s]

    if client_device in devices:
        # Being NAMED is not enough. Without a matching zero share the
        # automatic layer split still hands this device a slice of the
        # model -- attention and norms included -- on top of whatever the
        # -ot rule places, which is the exact failure this function
        # exists to prevent. Accept only a provably zero share.
        if len(shares) != len(devices):
            return None
        try:
            if float(shares[devices.index(client_device)]) != 0.0:
                return None
        except ValueError:
            return None
        return {}

    if shares:
        if len(shares) != len(devices):
            # Position i of --tensor-split maps to position i of --device;
            # a mapping that is already broken cannot be extended without
            # guessing which share belongs to the remote device.
            return None
        new_split = ",".join(shares + ["0"])
    elif len(devices) == 1:
        new_split = "1,0"
    else:
        # Several local cards and no split: llama.cpp is spreading layers
        # by its own rule, and there is no share list to append a 0 to.
        return None

    new_devices = ",".join(devices + [client_device])
    if match:
        new_extra = extra[:match.start()] + f"--device {new_devices}" \
            + extra[match.end():]
    else:
        new_extra = (extra + " " if extra else "") + f"--device {new_devices}"
    return {"extra": new_extra, "device": "", "split_mode": "layer",
            "tensor_split": new_split}


def _compile_rpc_plans(profile, hardware, capabilities):
    """One plan per usable remote device: the routed experts of a
    contiguous trailing layer range.

    Contiguous and trailing on purpose. llama.cpp streams activations
    across the RPC boundary once per contiguous region, so a scattered
    placement pays the network cost repeatedly per token; a single
    trailing block crosses the wire twice. The work order allows
    contiguous ranges only, and this is why.

    Routed experts only, never whole layers -- see contiguous_range. A
    model with no routed experts therefore yields no plans here.
    """
    out = []
    try:
        import modelctl_fleet
    except Exception:
        return out
    try:
        nodes = modelctl_fleet.usable_nodes()
    except Exception:
        return out
    if not nodes:
        return out

    endpoints = [n.endpoint for n in nodes]
    for node in nodes:
        for dev_index, device in enumerate(node.devices):
            if device.budget_bytes <= 0:
                continue
            placement = _rpc_layer_budget(profile, device.budget_bytes)
            if placement is None:
                continue
            first, last, moved_bytes = placement
            n_layers = last - first + 1
            rpc_device = modelctl_fleet.device_name_for(
                endpoints, node.endpoint, dev_index)
            buf_type = modelctl_fleet.buffer_type_for(
                node.endpoint, dev_index)
            key = modelctl_fleet.admission_key(node.name, device.name)
            rpc_cfg = {
                "endpoints": endpoints,
                "placements": [{"buffer_type": buf_type,
                                "first_layer": first, "last_layer": last}],
                # Carried so admission can charge the remote budget to the
                # right key without re-deriving it from the argv. The
                # routed-expert bytes only: that is all the -ot rule moves.
                "admission": {key: int(moved_bytes)},
            }
            # The remote device must be in --device or nothing reaches it.
            attach = _attach_rpc_device(
                {**profile.get("config", {})}, rpc_device)
            if attach is None:
                continue
            out.append(_make_plan(
                profile, {"rpc": rpc_cfg, **attach}, "fleet-rpc", hardware,
                extra_warnings=(
                    f"the routed experts of layers {first}-{last} execute "
                    f"on {node.name} ({device.name}) over the network; "
                    f"the local-only plans remain available if the node "
                    f"goes away",),
                decision={"rpc": {
                    "node": node.name, "endpoint": node.endpoint,
                    "device": device.name, "client_device": rpc_device,
                    "buffer_type": buf_type,
                    "first_layer": first, "last_layer": last,
                    "layers": n_layers,
                    "budget_bytes": int(device.budget_bytes),
                    "claimed_bytes": int(moved_bytes),
                    "admission_key": key}},
                capabilities=capabilities))
    return out


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
    fixed = (layout["other_bytes"] + kv
             + modelctl_vram.compute_overhead_bytes(layout["weight_bytes"]))
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


def policy_for_profile(profile):
    """The RuntimePolicy a profile's saved runtime section implies.

    One definition, so the plans page, the plan service, and the managed
    worker cannot disagree about which policy is in force while showing
    the user a ranking.
    """
    rt = profile.get("runtime") or {}
    if not rt:
        return DEFAULT_POLICY
    return RuntimePolicy(
        objective=rt.get("objective", "balanced"),
        pinned_plan_id=rt.get("pinned_plan_id"),
        allow_fallback=rt.get("allow_fallback", True),
        allow_untested=rt.get("allow_untested", False),
        minimum_context=rt.get("minimum_context"),
        maximum_cpu_bytes=rt.get("maximum_cpu_bytes"),
        maximum_storage_tier=rt.get("maximum_storage_tier", 3),
    )


def rank_plans(plans, policy=None, observations=None, failures=None,
               explain=None):
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
    - experimental_margin: an experimental plan sorts below every safe one
      unless it has a current measurement beating the best measured safe
      plan by that fraction.

    `explain` is an optional dict, populated with plan_id -> reason for
    every plan the guardrail demoted. Opt-in so the return shape is
    unchanged for existing callers; the decision-trace renderer uses it.

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

    # An experimental plan sorts below the safe ones unless it has earned
    # its place. Demotion is a sort tier rather than a score change: the
    # score still reports what was measured, which is what the plan card
    # shows, and only the ordering reflects the guardrail.
    demoted = _experimental_demotions(ranked, policy, observations, explain)

    def order(entry):
        plan, score = entry
        return (plan.id in demoted, -score)

    # Pinned plan promotes to the front but never deletes alternatives.
    # An explicit pin is the user overriding the guardrail on purpose.
    if policy.pinned_plan_id:
        pinned = [(p, s) for p, s in ranked if p.id == policy.pinned_plan_id]
        rest = [(p, s) for p, s in ranked if p.id != policy.pinned_plan_id]
        rest.sort(key=order)
        return [(p, float("inf")) for p, s in pinned] + rest

    ranked.sort(key=order)
    return ranked


# Failure classes that permanently disqualify a plan for a backend build
# (re-testing them is pointless until the backend changes).
_SUPPRESS_FAILURES = {"unsupported_architecture", "invalid_argument"}


def _objective_value(plan, policy, observations):
    """Higher-is-better magnitude for the active objective.

    Used only for margin comparisons, which is why it is not the score:
    scores are on arbitrary per-objective scales and several are negative,
    so "10% better than" is not meaningful on them. These are all positive
    quantities where larger is genuinely better, so a ratio means something.

    Zero when the plan has no measurement for the objective, which fails
    any positive margin -- that is how "a current observation exists"
    becomes a precondition rather than a separate check.
    """
    obs = observations.get(plan.id, {}) or {}

    def inverse(x):
        return 1.0 / x if x else 0.0

    if policy.objective == "fastest_prompt":
        return obs.get("prompt_tps") or 0
    if policy.objective == "interactive_latency":
        return inverse(obs.get("ttft_seconds") or 0)
    if policy.objective == "fastest_load":
        return inverse(obs.get("load_seconds") or 0)
    if policy.objective == "lowest_storage":
        return inverse((obs.get("read_bytes") or 0) / float(1 << 30))
    if policy.objective == "lowest_ram":
        # A claim, not a measurement, so it is always available -- the
        # non-stale observation check in the guardrail still applies.
        return inverse(plan.claim.ram_bytes / float(1 << 30))
    if policy.objective == "largest_context":
        return plan.claim.expected_context or 0
    # fastest_generation and balanced both turn on generation throughput.
    return obs.get("generation_tps") or 0


def _experimental_demotions(ranked, policy, observations, explain=None):
    """Which experimental plans have not earned the right to outrank a
    safe baseline.

    An experimental plan may lead only when it has a current, non-stale
    measurement and beats the best measured safe plan by
    policy.experimental_margin. Failing that it is demoted, not dropped:
    it stays selectable and stays visible with its numbers, but an
    automatic choice will not land on it.

    With no measured safe plan to compare against there is no baseline to
    protect, so the margin does not apply and the objective score decides.
    """
    if policy.experimental_margin is None:
        return set()

    experimental = [(p, s) for p, s in ranked if is_experimental_plan(p)]
    if not experimental:
        return set()

    safe_values = [_objective_value(p, policy, observations)
                   for p, _s in ranked if not is_experimental_plan(p)
                   and (observations.get(p.id) or {}).get("generation_tps") is not None
                   and not (observations.get(p.id) or {}).get("stale")]
    best_safe = max(safe_values, default=0.0)
    if best_safe <= 0:
        return set()

    required = best_safe * (1.0 + policy.experimental_margin)
    demoted = set()
    for plan, _score in experimental:
        obs = observations.get(plan.id) or {}
        value = _objective_value(plan, policy, observations)
        if not obs:
            reason = "no measurement of its own"
        elif obs.get("stale"):
            reasons = obs.get("stale_reasons") or []
            reason = "measurement is stale" + (
                f" ({'; '.join(reasons)})" if reasons else "")
        elif value < required:
            pct = policy.experimental_margin * 100
            reason = (f"{value:.4g} does not clear the {pct:.0f}% margin over "
                      f"the best safe plan's {best_safe:.4g}")
        else:
            continue
        demoted.add(plan.id)
        if explain is not None:
            explain[plan.id] = f"experimental plan ranked below safe plans: {reason}"
    return demoted


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
    ttft = obs.get("ttft_seconds", 0) or 0
    read_bytes = obs.get("read_bytes", 0) or 0
    ctx = plan.claim.expected_context or 0

    if policy.objective == "interactive_latency":
        # Time to first token, which is what a user waiting at a prompt
        # feels -- distinct from fastest_load, which is a one-off cost paid
        # before any request. Unmeasured plans score below every measured
        # one rather than winning by having no latency to report.
        return (-ttft * 100 if ttft else -1e6) - fail_penalty
    elif policy.objective == "lowest_storage":
        # Bytes actually read during the run. Never inferred from mmap being
        # configured: a fully page-cached mmap plan reads nothing, and a
        # resident plan still pays to load once.
        if not read_bytes:
            return -1e6 - fail_penalty
        return -(read_bytes / float(1 << 30)) - fail_penalty
    elif policy.objective == "fastest_generation":
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
