"""Evidence-first plan comparison (roadmap Task C3).

A plan table that shows only estimates invites the user to pick the
biggest number. This module turns each candidate into a record that
answers "why would I pick this one, and what do we actually know?" --
what it claims, what was measured, whether that measurement is still
valid, whether it can run at all, and where it sits relative to the safe
baseline.

Nothing here recomputes placement or scoring; it joins plans with the
capability probe, the observation history, the failure history, and the
canonical launch validation.

Public API:
    PlanEvidence dataclass
    CATEGORIES
    build_plan_evidence(profile, plans, observations, failures, backend) -> list
    group_evidence(evidence) -> list[(category, label, description, items)]
    DecisionTrace dataclass
    build_decision_trace(evidence, ranked, policy, observations, explain) -> DecisionTrace
"""
import time
from dataclasses import dataclass, field

import modelctl_plans


# Ordered worst-to-best is deliberately *not* the display order; the
# display order below goes from "what you should probably use" to "what
# you cannot use", so the safe answer is the first thing on screen.
CATEGORIES = (
    ("baseline", "safe baseline",
     "The profile exactly as saved. Chosen automatically when nothing "
     "better has been measured."),
    ("tested", "tested",
     "A current measurement exists for this hardware and this backend "
     "build."),
    ("untested", "untested estimate",
     "Claims are computed from the GGUF layout and your hardware. Nothing "
     "has been run yet."),
    ("experimental", "experimental (MoE cache)",
     "Uses the fork's expert cache or hybrid CPU-miss execution. Correct "
     "only as far as it has been tested here."),
    ("unavailable", "unavailable",
     "Cannot be launched as configured. The reason is shown per plan."),
)

CATEGORY_ORDER = {name: i for i, (name, _l, _d) in enumerate(CATEGORIES)}


@dataclass
class PlanEvidence:
    """One plan plus everything known about whether it works."""
    plan: object
    category: str
    # --- state flags (a plan can be several of these at once) ---
    experimental: bool = False
    blocked: bool = False
    tested: bool = False
    stale: bool = False
    failed: bool = False
    suppressed: bool = False
    is_baseline: bool = False
    # --- explanation ---
    reason: str = ""                     # why this category
    blocked_reasons: tuple = ()          # validation errors, suppressions
    failure_classes: tuple = ()          # recent failure classes
    # --- evidence ---
    observation: dict | None = None
    measured_at: float | None = None
    cache_state: str = ""                # "cold" / "warm" / ""
    estimated: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)
    # --- provenance ---
    binary: str = ""
    binary_fingerprint: str = ""
    capability_fingerprint: str = ""
    capability_status: str = ""
    build: str = ""
    command: tuple = ()
    validation: tuple = ()
    warnings: tuple = ()
    # --- policy position ---
    pinned: bool = False
    disabled: bool = False
    fallback_ordinal: int | None = None  # 0 = would be chosen first

    @property
    def cache_bytes(self):
        return getattr(self.plan.claim, "cache_bytes", 0) or 0

    @property
    def category_label(self):
        for name, label, _d in CATEGORIES:
            if name == self.category:
                return label
        return self.category


# Defined in modelctl_plans because ranking needs the same predicate for
# Task H2's guardrail, and this module already imports that one.
_is_experimental = modelctl_plans.is_experimental_plan


def _observed_values(obs) -> dict:
    if not obs:
        return {}
    return {k: obs.get(k) for k in
            ("generation_tps", "prompt_tps", "load_seconds", "actual_context")
            if obs.get(k) is not None}


def build_plan_evidence(profile, plans, observations=None, failures=None,
                        backend=None, ranked_ids=None):
    """Join plans with capability, measurement, failure, and launch state.

    backend: a modelctl_launch.ResolvedBackend. Supplying it is what makes
    the card able to say *which build* the claims and cache eligibility
    belong to -- without it the card would repeat the same estimate for a
    stock binary and the fork, which is the confusion Task C3 exists to
    remove.

    ranked_ids: plan ids in the order the current policy would try them,
    from modelctl_plans.rank_plans(). Used only to show fallback position.
    """
    observations = observations or {}
    failures = failures or {}
    rt = profile.get("runtime") or {}
    pinned_id = rt.get("pinned_plan_id")
    disabled_ids = set(rt.get("disabled_plan_ids") or [])
    ranked_ids = list(ranked_ids or [])

    caps = getattr(backend, "capabilities", None) or {}
    build = caps.get("build") or ""
    if isinstance(build, dict):
        build = build.get("commit", "")

    out = []
    for plan in plans:
        obs = observations.get(plan.id)
        plan_failures = tuple(failures.get(plan.id) or ())
        suppressed = any(f in modelctl_plans._SUPPRESS_FAILURES
                         for f in plan_failures)

        experimental = _is_experimental(plan)
        stale = bool(obs and obs.get("stale"))
        tested = bool(obs and not stale)

        blocked_reasons = []
        if suppressed:
            blocked_reasons.append(
                f"suppressed after {plan_failures[0]} on this backend build — "
                "re-testing is pointless until the runtime changes")
        if experimental and backend is not None:
            import modelctl_capabilities
            if not modelctl_capabilities.is_cache_capable(caps):
                blocked_reasons.append(
                    "this build does not report moe_weight_transfer_cache")
            if ((plan.decision_data or {}).get("hybrid")
                    and not modelctl_capabilities.supports_hybrid_miss(caps)):
                blocked_reasons.append(
                    "this build does not report moe_hybrid_cpu_miss")

        blocked = bool(blocked_reasons)
        is_baseline = modelctl_plans.is_baseline_plan(plan)

        # Precedence: a plan that cannot run is never advertised as tested
        # or safe, and an experimental plan stays labelled experimental
        # even once it has a measurement -- "it worked once" is not the
        # same claim as "this is the safe choice".
        if blocked:
            category = "unavailable"
            reason = blocked_reasons[0]
        elif experimental:
            category = "experimental"
            reason = ("measured on this build" if tested
                      else "no current measurement")
        elif is_baseline:
            category = "baseline"
            reason = ("the profile as saved, with a current measurement" if tested
                      else "the profile as saved; used as the safe first launch")
        elif tested:
            category = "tested"
            reason = "current measurement on this hardware and build"
        else:
            category = "untested"
            reason = ("previous measurement is stale" if stale
                      else "estimate only — never run here")

        try:
            ordinal = ranked_ids.index(plan.id)
        except ValueError:
            ordinal = None

        out.append(PlanEvidence(
            plan=plan,
            category=category,
            experimental=experimental,
            blocked=blocked,
            tested=tested,
            stale=stale,
            failed=bool(plan_failures),
            suppressed=suppressed,
            is_baseline=is_baseline,
            reason=reason,
            blocked_reasons=tuple(blocked_reasons),
            failure_classes=plan_failures,
            observation=obs,
            measured_at=(obs or {}).get("measured_at"),
            cache_state=(obs or {}).get("cache_state", ""),
            estimated=dict(plan.estimated or {}),
            observed=_observed_values(obs),
            binary=getattr(backend, "binary", ""),
            binary_fingerprint=getattr(backend, "binary_fingerprint", ""),
            capability_fingerprint=getattr(backend, "capability_fingerprint", ""),
            capability_status=caps.get("_probe_status", ""),
            build=build,
            command=tuple(plan.argv or ()),
            warnings=tuple(plan.warnings or ()),
            pinned=(pinned_id == plan.id),
            disabled=(plan.id in disabled_ids),
            fallback_ordinal=ordinal,
        ))

    out.sort(key=lambda e: (CATEGORY_ORDER.get(e.category, 99),
                            e.fallback_ordinal if e.fallback_ordinal is not None else 999))
    return out


@dataclass
class DecisionTrace:
    """Why the automatic choice is what it is (roadmap Task H3).

    Every field is allowed to be empty, and an empty one is simply not
    rendered. A trace that invents a reason it does not have is worse than
    a short one -- the point is that the user can check the machine's
    reasoning, which requires the reasoning to be the real one.
    """
    selected: str = ""
    selected_plan_id: str = ""
    reason: str = ""
    constraints: tuple = ()
    rejected: tuple = ()          # (label, why) pairs
    fallback: str = ""
    evidence: tuple = ()

    @property
    def present(self) -> bool:
        return bool(self.selected)


def _describe_constraints(policy) -> tuple:
    """The policy limits actually in force, in the user's units."""
    out = []
    objective = getattr(policy, "objective", "")
    if objective:
        out.append(f"objective: {objective.replace('_', ' ')}")
    if getattr(policy, "minimum_context", None):
        out.append(f"context >= {policy.minimum_context:,}")
    if getattr(policy, "maximum_cpu_bytes", None):
        out.append(f"RAM <= {policy.maximum_cpu_bytes / float(1 << 30):.0f} GiB")
    tier = getattr(policy, "maximum_storage_tier", 3)
    if tier < 3:
        out.append("storage tier: " + ("GPU only" if tier == 1 else "GPU + RAM"))
    if not getattr(policy, "allow_untested", True):
        out.append("untested plans not selectable")
    margin = getattr(policy, "experimental_margin", None)
    if margin:
        out.append(f"experimental plans must beat the safe baseline by "
                   f"{margin * 100:.0f}%")
    return tuple(out)


def _selection_reason(winner, observations, policy) -> str:
    """Why the winner won, stated against the safe baseline it displaced.

    Falls back to naming the absence of evidence rather than inventing a
    justification: "nothing has been measured" is the honest reason a
    baseline is chosen, and it tells the user what to do about it.
    """
    obs = (observations or {}).get(winner.plan.id) or {}
    metric = {
        "fastest_prompt": ("prompt_tps", "prompt throughput", True),
        "interactive_latency": ("ttft_seconds", "time to first token", False),
        "fastest_load": ("load_seconds", "load time", False),
        "lowest_storage": ("read_bytes", "storage read", False),
    }.get(getattr(policy, "objective", ""),
          ("generation_tps", "generation throughput", True))
    key, label, higher_better = metric

    if not obs.get(key):
        if winner.is_baseline:
            return ("nothing has been measured for this profile yet, so the "
                    "profile's own configuration is used unchanged")
        return f"no measurement of {label} yet; ranked on estimated claims"

    state = f"{winner.cache_state} " if winner.cache_state else ""
    return f"best measured {state}{label} of the selectable plans"


def build_decision_trace(evidence, ranked=None, policy=None, observations=None,
                         explain=None, backend=None, hardware_fingerprint=""):
    """Assemble the trace for the plan the policy would launch.

    `explain` is rank_plans()' demotion dict, so a rejection reads with the
    guardrail's own words rather than a rephrasing that could drift from
    what the ranker actually did.
    """
    ranked = list(ranked or [])
    by_id = {e.plan.id: e for e in evidence}
    order = [pid for pid, in ((p.id,) for p, _s in ranked)] if ranked else [
        e.plan.id for e in evidence if e.fallback_ordinal == 0]
    chosen = next((by_id[pid] for pid in order if pid in by_id), None)
    if chosen is None:
        return DecisionTrace()

    trace = DecisionTrace(
        selected=chosen.plan.label,
        selected_plan_id=chosen.plan.id,
        reason=_selection_reason(chosen, observations, policy),
        constraints=_describe_constraints(policy) if policy else (),
    )
    # Read the pin off the policy that actually did the ranking, not off the
    # evidence: PlanEvidence.pinned reflects the saved profile, and a caller
    # can rank under a policy that differs from it.
    if chosen.pinned or getattr(policy, "pinned_plan_id", None) == chosen.plan.id:
        trace.reason = "pinned by the profile, overriding automatic ranking"

    # Rejections: the guardrail's demotions first (those are choices the
    # ranker made), then plans that could not run at all.
    rejected = []
    for pid, why in (explain or {}).items():
        e = by_id.get(pid)
        if e is not None and pid != chosen.plan.id:
            rejected.append((e.plan.label, why))
    for e in evidence:
        if e.plan.id == chosen.plan.id or e.plan.id in (explain or {}):
            continue
        if e.blocked or e.suppressed:
            why = "; ".join(e.blocked_reasons) or e.reason or "cannot be launched"
            rejected.append((e.plan.label, why))
    trace.rejected = tuple(rejected)

    nxt = next((by_id[pid] for pid in order[1:] if pid in by_id), None)
    trace.fallback = nxt.plan.label if nxt is not None else ""

    ev = []
    obs = (observations or {}).get(chosen.plan.id) or {}
    if obs:
        when = obs.get("measured_at")
        ev.append("measured " + (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(when)) if when
            else "at an unrecorded time")
            + (f", {obs['cache_state']} cache" if obs.get("cache_state") else ""))
        if obs.get("stale"):
            ev.append("measurement is stale: " +
                      "; ".join(obs.get("stale_reasons") or ["identity changed"]))
    else:
        ev.append("no measurement; claims are computed from the GGUF layout")
    if chosen.binary_fingerprint:
        ev.append(f"binary {chosen.binary_fingerprint[:12]}")
    if hardware_fingerprint:
        ev.append(f"hardware {hardware_fingerprint[:12]}")
    trace.evidence = tuple(ev)
    return trace


def group_evidence(evidence):
    """Group evidence into the display sections Task C3 asks for.

    Empty sections are dropped, so a profile with no experimental plans
    does not show an empty "experimental" heading.
    """
    groups = []
    for name, label, description in CATEGORIES:
        items = [e for e in evidence if e.category == name]
        if items:
            groups.append((name, label, description, items))
    return groups
