"""Task C3: plan comparison is evidence-first.

The behaviour these lock down is *classification*: which bucket a plan
lands in, and why. Getting that wrong is how a user ends up picking an
experimental plan because it showed the biggest number.
"""
import unittest

import modelctl_evidence
import modelctl_plans
from modelctl_plans import LaunchPlan, ResourceClaim


def plan(pid, source="tier-planner", cache_bytes=0, decision=None,
         label=None, ctx=8192):
    return LaunchPlan(
        id=pid, profile_name="m1", backend="llama-cpp",
        label=label or pid, argv=("/bin/llama-server", "--model", "/m.gguf"),
        env={},
        claim=ResourceClaim(vram_bytes={"SYCL0": 8 << 30}, ram_bytes=4 << 30,
                            storage_mode="mmap", expected_context=ctx,
                            cache_bytes=cache_bytes),
        estimated={"total_vram": 8 << 30, "ram": 4 << 30, "context": ctx},
        source=source, warnings=(), decision_data=decision or {})


class FakeBackend:
    binary = "/bin/llama-server"
    binary_fingerprint = "binfp"
    capability_fingerprint = "capfp"

    def __init__(self, **features):
        self.capabilities = {
            "schema": 2, "build": {"commit": "abc123"},
            "features": features, "cli": {}, "_probe_status": "ok"}


PROFILE = {"name": "m1", "config": {}}


def by_id(evidence):
    return {e.plan.id: e for e in evidence}


class TestCategorisation(unittest.TestCase):
    def test_current_profile_is_the_safe_baseline(self):
        ev = modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("a", source="current-profile")],
            backend=FakeBackend())
        self.assertEqual(ev[0].category, "baseline")
        self.assertTrue(ev[0].is_baseline)

    def test_untested_plan_says_so(self):
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("a")], backend=FakeBackend()))
        self.assertEqual(ev["a"].category, "untested")
        self.assertFalse(ev["a"].tested)
        self.assertIn("estimate only", ev["a"].reason)

    def test_current_measurement_makes_a_plan_tested(self):
        obs = {"a": {"generation_tps": 20.0, "stale": False,
                     "cache_state": "warm", "measured_at": 1.0}}
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("a")], observations=obs, backend=FakeBackend()))
        self.assertEqual(ev["a"].category, "tested")
        self.assertTrue(ev["a"].tested)
        self.assertEqual(ev["a"].observed["generation_tps"], 20.0)
        self.assertEqual(ev["a"].cache_state, "warm")

    def test_stale_measurement_does_not_count_as_tested(self):
        # A measurement from different hardware or a different build is not
        # evidence about this one.
        obs = {"a": {"generation_tps": 99.0, "stale": True,
                     "cache_state": "warm", "measured_at": 1.0}}
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("a")], observations=obs, backend=FakeBackend()))
        self.assertEqual(ev["a"].category, "untested")
        self.assertTrue(ev["a"].stale)
        self.assertFalse(ev["a"].tested)
        self.assertIn("stale", ev["a"].reason)

    def test_cache_plan_stays_experimental_even_once_measured(self):
        # "It worked once" is not the same claim as "this is the safe
        # choice" -- an experimental plan must not graduate into the tested
        # bucket and read as ordinary.
        obs = {"c": {"generation_tps": 40.0, "stale": False,
                     "cache_state": "warm", "measured_at": 1.0}}
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("c", cache_bytes=2 << 30)], observations=obs,
            backend=FakeBackend(moe_weight_transfer_cache=True)))
        self.assertEqual(ev["c"].category, "experimental")
        self.assertTrue(ev["c"].tested)
        self.assertTrue(ev["c"].experimental)

    def test_hybrid_plan_is_experimental(self):
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("h", decision={"hybrid": True})],
            backend=FakeBackend(moe_weight_transfer_cache=True,
                                moe_hybrid_cpu_miss=True)))
        self.assertTrue(ev["h"].experimental)
        self.assertEqual(ev["h"].category, "experimental")


class TestUnavailablePlans(unittest.TestCase):
    def test_cache_plan_on_incapable_build_is_unavailable_with_reason(self):
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("c", cache_bytes=2 << 30)],
            backend=FakeBackend(moe_weight_transfer_cache=False)))
        e = ev["c"]
        self.assertEqual(e.category, "unavailable")
        self.assertTrue(e.blocked)
        self.assertTrue(any("moe_weight_transfer_cache" in r
                            for r in e.blocked_reasons))

    def test_hybrid_plan_without_hybrid_support_is_unavailable(self):
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("h", cache_bytes=1 << 30, decision={"hybrid": True})],
            backend=FakeBackend(moe_weight_transfer_cache=True,
                                moe_hybrid_cpu_miss=False)))
        self.assertEqual(ev["h"].category, "unavailable")
        self.assertTrue(any("moe_hybrid_cpu_miss" in r
                            for r in ev["h"].blocked_reasons))

    def test_suppressed_failure_makes_a_plan_unavailable(self):
        failures = {"a": ["unsupported_architecture"]}
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("a")], failures=failures, backend=FakeBackend()))
        self.assertEqual(ev["a"].category, "unavailable")
        self.assertTrue(ev["a"].suppressed)
        self.assertIn("unsupported_architecture", ev["a"].blocked_reasons[0])

    def test_blocked_beats_tested(self):
        # A plan that cannot run must never be advertised as tested.
        obs = {"c": {"generation_tps": 40.0, "stale": False,
                     "cache_state": "warm", "measured_at": 1.0}}
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("c", cache_bytes=2 << 30)], observations=obs,
            backend=FakeBackend(moe_weight_transfer_cache=False)))
        self.assertEqual(ev["c"].category, "unavailable")

    def test_soft_failures_do_not_block(self):
        # An OOM is a reason to warn, not a reason to hide the plan: a
        # different placement or a freed GPU can make it work.
        failures = {"a": ["out_of_vram"]}
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("a")], failures=failures, backend=FakeBackend()))
        self.assertFalse(ev["a"].blocked)
        self.assertTrue(ev["a"].failed)
        self.assertEqual(ev["a"].failure_classes, ("out_of_vram",))


class TestPolicyPosition(unittest.TestCase):
    def test_fallback_ordinal_follows_the_ranking(self):
        plans = [plan("a"), plan("b"), plan("c")]
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, plans, backend=FakeBackend(),
            ranked_ids=["c", "a", "b"]))
        self.assertEqual(ev["c"].fallback_ordinal, 0)
        self.assertEqual(ev["a"].fallback_ordinal, 1)
        self.assertEqual(ev["b"].fallback_ordinal, 2)

    def test_plan_outside_the_ranking_has_no_position(self):
        ev = by_id(modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("a"), plan("b")], backend=FakeBackend(),
            ranked_ids=["a"]))
        self.assertEqual(ev["a"].fallback_ordinal, 0)
        self.assertIsNone(ev["b"].fallback_ordinal)

    def test_pinned_and_disabled_come_from_the_profile(self):
        profile = {"name": "m1", "config": {},
                   "runtime": {"mode": "managed", "pinned_plan_id": "a",
                               "disabled_plan_ids": ["b"]}}
        ev = by_id(modelctl_evidence.build_plan_evidence(
            profile, [plan("a"), plan("b")], backend=FakeBackend()))
        self.assertTrue(ev["a"].pinned)
        self.assertFalse(ev["a"].disabled)
        self.assertTrue(ev["b"].disabled)


class TestProvenanceAndGrouping(unittest.TestCase):
    def test_evidence_records_which_build_it_describes(self):
        ev = modelctl_evidence.build_plan_evidence(
            PROFILE, [plan("a")], backend=FakeBackend())[0]
        self.assertEqual(ev.binary, "/bin/llama-server")
        self.assertEqual(ev.binary_fingerprint, "binfp")
        self.assertEqual(ev.capability_fingerprint, "capfp")
        self.assertEqual(ev.build, "abc123")
        self.assertEqual(ev.capability_status, "ok")

    def test_works_without_a_resolved_backend(self):
        # The page must still render when the backend cannot be resolved.
        ev = modelctl_evidence.build_plan_evidence(PROFILE, [plan("a")])
        self.assertEqual(ev[0].binary, "")
        self.assertEqual(ev[0].category, "untested")

    def test_groups_are_ordered_safe_first_and_drop_empties(self):
        plans = [plan("base", source="current-profile"),
                 plan("exp", cache_bytes=2 << 30)]
        ev = modelctl_evidence.build_plan_evidence(
            PROFILE, plans,
            backend=FakeBackend(moe_weight_transfer_cache=True))
        groups = modelctl_evidence.group_evidence(ev)
        names = [g[0] for g in groups]
        self.assertEqual(names, ["baseline", "experimental"])
        for _n, _label, _desc, items in groups:
            self.assertTrue(items, "empty group should have been dropped")

    def test_evidence_sorted_safe_before_unavailable(self):
        plans = [plan("bad", cache_bytes=2 << 30),
                 plan("base", source="current-profile")]
        ev = modelctl_evidence.build_plan_evidence(
            PROFILE, plans, backend=FakeBackend(moe_weight_transfer_cache=False))
        self.assertEqual(ev[0].plan.id, "base")
        self.assertEqual(ev[-1].plan.id, "bad")


class TestPolicyForProfile(unittest.TestCase):
    def test_no_runtime_section_gives_the_default_policy(self):
        self.assertIs(modelctl_plans.policy_for_profile({"name": "m"}),
                      modelctl_plans.DEFAULT_POLICY)

    def test_runtime_section_is_honoured(self):
        pol = modelctl_plans.policy_for_profile({
            "runtime": {"mode": "managed", "objective": "fastest_generation",
                        "pinned_plan_id": "x", "allow_untested": True,
                        "minimum_context": 16384, "maximum_storage_tier": 2}})
        self.assertEqual(pol.objective, "fastest_generation")
        self.assertEqual(pol.pinned_plan_id, "x")
        self.assertTrue(pol.allow_untested)
        self.assertEqual(pol.minimum_context, 16384)
        self.assertEqual(pol.maximum_storage_tier, 2)


if __name__ == "__main__":
    unittest.main()
