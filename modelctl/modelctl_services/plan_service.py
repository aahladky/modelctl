"""Plan compilation, selection, and testing service.

Plan operations are pure computations plus optional runtime interactions.
They return structured results, never print or sys.exit.
"""
from dataclasses import dataclass, field

import modelctl
import modelctl_hardware
import modelctl_plans
import modelctl_runtime


@dataclass
class PlanResult:
    """Result of a plan operation."""
    ok: bool
    plans: list = field(default_factory=list)
    selected_plan: dict | None = None
    messages: list = field(default_factory=list)
    observations: dict = field(default_factory=dict)


def compile_plans(profile_name: str) -> PlanResult:
    """Compile launch plans for a profile.

    Returns all candidate plans with their resource claims.
    """
    try:
        profile = modelctl.load_profile(profile_name)
    except (SystemExit, Exception) as e:
        return PlanResult(ok=False, messages=[f"failed to load profile: {e}"])

    snap = modelctl_hardware.capture_hardware_snapshot()
    plans = modelctl_plans.compile_launch_plans(profile, snap)
    rdb = modelctl_runtime.RuntimeDB()
    observations = rdb.observations_for_profile(
        profile_name, snap.fingerprint,
        snap.backend_fingerprints.get(profile.get("backend", "llama-cpp"), ""))

    return PlanResult(ok=True, plans=plans, observations=observations)


def rank_plans(profile_name: str, policy=None) -> PlanResult:
    """Compile and rank plans according to policy.

    Returns plans sorted by score, filtered by policy constraints.
    """
    try:
        profile = modelctl.load_profile(profile_name)
    except (SystemExit, Exception) as e:
        return PlanResult(ok=False, messages=[f"failed to load profile: {e}"])

    snap = modelctl_hardware.capture_hardware_snapshot()
    plans = modelctl_plans.compile_launch_plans(profile, snap)

    if policy is None:
        rt = profile.get("runtime", {})
        if rt:
            policy = modelctl_plans.RuntimePolicy(
                objective=rt.get("objective", "balanced"),
                pinned_plan_id=rt.get("pinned_plan_id"),
                allow_fallback=rt.get("allow_fallback", True),
                allow_untested=rt.get("allow_untested", False),
                minimum_context=rt.get("minimum_context"),
                maximum_cpu_bytes=rt.get("maximum_cpu_bytes"),
                maximum_storage_tier=rt.get("maximum_storage_tier", 3),
            )
        else:
            policy = modelctl_plans.DEFAULT_POLICY

    rdb = modelctl_runtime.RuntimeDB()
    observations = rdb.observations_for_profile(
        profile_name, snap.fingerprint,
        snap.backend_fingerprints.get(profile.get("backend", "llama-cpp"), ""))
    failures = rdb.failures_for_profile(profile_name)
    ranked = modelctl_plans.rank_plans(plans, policy, observations, failures)

    return PlanResult(
        ok=True,
        plans=[p for p, _ in ranked],
        observations=observations,
    )


def select_plan(profile_name: str, plan_id: str) -> PlanResult:
    """Pin a plan as the preferred choice for a managed profile.

    Clears the plan from disabled_plan_ids if it's there -- pinning a
    previously-disabled plan should un-disable it, not leave it pinned
    and disabled at once (rank_plans would then never actually select it).
    """
    try:
        profile = modelctl.load_profile(profile_name)
    except (SystemExit, Exception) as e:
        return PlanResult(ok=False, messages=[f"failed to load profile: {e}"])

    rt = dict(profile.get("runtime") or {
        "mode": "managed", "objective": "balanced",
        "allow_fallback": True, "allow_untested": False,
    })
    rt["pinned_plan_id"] = plan_id
    rt["mode"] = "managed"
    disabled = [p for p in rt.get("disabled_plan_ids", []) if p != plan_id]
    rt["disabled_plan_ids"] = disabled
    modelctl.update_runtime_policy(profile_name, rt)
    return PlanResult(ok=True, selected_plan={"id": plan_id},
                      messages=[f"pinned plan {plan_id}"])


def disable_plan(profile_name: str, plan_id: str) -> PlanResult:
    """Disable a plan so it's never selected.

    Clears pinned_plan_id if this was the pinned plan -- "disabled" must
    mean never selected, not "disabled but still pinned."
    """
    try:
        profile = modelctl.load_profile(profile_name)
    except (SystemExit, Exception) as e:
        return PlanResult(ok=False, messages=[f"failed to load profile: {e}"])

    rt = dict(profile.get("runtime") or {"mode": "managed"})
    rt["mode"] = "managed"
    disabled = set(rt.get("disabled_plan_ids", []))
    disabled.add(plan_id)
    rt["disabled_plan_ids"] = sorted(disabled)
    if rt.get("pinned_plan_id") == plan_id:
        rt["pinned_plan_id"] = None
    modelctl.update_runtime_policy(profile_name, rt)
    return PlanResult(ok=True, messages=[f"disabled plan {plan_id}"])


def enable_plan(profile_name: str, plan_id: str) -> PlanResult:
    """Re-enable a previously disabled plan.

    Unlike select/disable, this never forces a profile into managed mode --
    a profile with no runtime policy at all has nothing to enable, so it's
    a no-op rather than fabricating a fresh managed section.
    """
    try:
        profile = modelctl.load_profile(profile_name)
    except (SystemExit, Exception) as e:
        return PlanResult(ok=False, messages=[f"failed to load profile: {e}"])

    rt = dict(profile.get("runtime") or {})
    disabled = set(rt.get("disabled_plan_ids", []))
    disabled.discard(plan_id)
    rt["disabled_plan_ids"] = sorted(disabled)
    modelctl.update_runtime_policy(
        profile_name, rt if rt.get("mode") == "managed" else None)
    return PlanResult(ok=True, messages=[f"enabled plan {plan_id}"])


def smoke_test(profile_name: str, timeout: int = 600,
               prompt: str | None = None,
               proc_register=None) -> dict:
    """Run a smoke test on a profile. Returns structured result.

    This is already callable from both CLI and web - just wrapping
    the existing function for consistency.
    """
    return modelctl.smoke_test_profile(
        profile_name, timeout=timeout, prompt=prompt,
        proc_register=proc_register)
