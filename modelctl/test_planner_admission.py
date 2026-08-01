"""Plan-time VRAM admission (the C2 crash class).

The disease: a config whose static GPU expert pins plus cache pool
oversubscribed SYCL0 was emitted as a plan, launched, and died at warmup
with a null pool allocation (SIGABRT) instead of being rejected -- or
degraded -- at plan time. Preserved in
docs/evidence/2026-08-01-p1-madvise-bench.json, condition
C2-attempt1-oom-staticpins: pins blk.[0-5] on SYCL0 plus a 17 GiB cache
pool against a 31.9 GiB card.

These tests drive that exact shape (and its boundaries) through the real
launch-plan compiler and the tier planner. Contract under test: an
oversubscribed plan is never emitted as-is -- the cache shrinks toward 0
first, then pinned expert layers spill to CPU -- and the emitted plan
carries both a human warning and a machine-readable admission record
stating requested, assumed, and chosen.

Hermetic: hardware snapshots, GGUF layouts, and defaults are all synthetic;
no probe, no state dir, no real model files.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_plans
import modelctl_tiers

GIB = 1 << 30

# The real cards, straight from the C2 evidence record.
B70_TOTAL = 34242297856
B580_TOTAL = 12809404416

# The C2 request: 17 GiB pool on SYCL0, 5 GiB on SYCL1.
C2_CACHE = {"SYCL0": 18253611008, "SYCL1": 5368709120}
C2_PINS = (r"--fit off --device SYCL0,SYCL1 "
           r"-ot blk\\.[0-5]\\.ffn_.*_exps=SYCL0,"
           r"blk\\.[6-7]\\.ffn_.*_exps=SYCL1,"
           r"ffn_.*_exps=CPU --no-warmup --ubatch-size 128")


class FakeGpu:
    def __init__(self, device, total_bytes):
        self.device = device
        self.name = device
        self.enabled = True
        self.role = ""
        self.total_bytes = total_bytes
        self.free_bytes = total_bytes
        self.reserve_bytes = 0


class FakeSnapshot:
    def __init__(self, gpus):
        self.gpus = tuple(gpus)

    def gpu_by_device(self, device):
        return next((g for g in self.gpus if g.device == device), None)


def ornith_class_layout():
    """DeepSeek-pattern synthetic: dense blk.0-2, 61 expert layers of
    2.9 GiB from layer 3, shexp -- the documented ornith-397b shape."""
    layers = {i: int(2.9 * GIB) for i in range(3, 64)}
    return {"arch": "testmoe", "meta": {}, "block_count": 64,
            "is_moe": True,
            "weight_bytes": sum(layers.values()) + 4 * GIB,
            "non_expert_bytes": 4 * GIB, "other_bytes": GIB,
            "layer_bytes": 3 * GIB, "expert_bytes_per_layer": layers,
            "has_shexp": True, "unknown_type_tensors": 0}


class AdmissionBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "m.gguf"
        self.model.write_bytes(b"\0" * 4096)

    def c2_profile(self):
        return modelctl.normalize_profile({
            "name": "ornith-397b",
            "model_path": str(self.model),
            "backend": "llama-cpp",
            "binary": "/fake/llama-server",
            "config": {"ctx": 4096, "cache_type_k": "q8_0",
                       "cache_type_v": "q4_0", "device": "",
                       "split_mode": "layer", "tensor_split": "8,3",
                       "fit": "off", "extra": C2_PINS},
            "moe_cache": {"mode": "manual",
                          "gpu": {"budgets_bytes": dict(C2_CACHE)}},
        })

    def make_plan(self, profile, layout, gpus=None, capabilities=None):
        snap = FakeSnapshot(gpus or [FakeGpu("SYCL0", B70_TOTAL),
                                     FakeGpu("SYCL1", B580_TOTAL)])
        with mock.patch.object(modelctl_plans.modelctl_vram,
                               "gguf_model_layout", return_value=layout), \
             mock.patch.object(modelctl_plans.modelctl_vram,
                               "gguf_kv_params", return_value=None), \
             mock.patch.object(modelctl_plans, "_block_device_for",
                               return_value="/dev/test0"), \
             mock.patch.object(modelctl, "load_defaults",
                               return_value={"vram_limit_pct": 90,
                                             "primary_gpu": "SYCL0"}):
            return modelctl_plans.current_profile_plan(
                profile, hardware=snap, capabilities=capabilities)

    def usable(self):
        return {"SYCL0": int(B70_TOTAL * 0.9), "SYCL1": int(B580_TOTAL * 0.9)}


class TestC2ShapeThroughLaunchPlanner(AdmissionBase):
    """The exact C2 shape must plan to a degraded config with a warning --
    never to a config that can crash at launch for VRAM admission."""

    def test_c2_is_degraded_never_emitted_as_is(self):
        plan = self.make_plan(self.c2_profile(), ornith_class_layout())
        adm = plan.decision_data.get("admission")
        self.assertIsNotNone(adm, "no admission record on a C2-shape plan")

        # Fixture validity: the request really was oversubscribed.
        usable = self.usable()
        requested = adm["requested"]["per_device_peak_bytes"]
        self.assertGreater(requested["SYCL0"], usable["SYCL0"],
                           "fixture no longer oversubscribes; test is void")

        # The emitted plan admits on every device.
        self.assertTrue(adm["fits"])
        chosen = adm["chosen"]["per_device_peak_bytes"]
        for dev, peak in chosen.items():
            self.assertLessEqual(peak, usable[dev], dev)
        final_peak = plan.claim.vram_admission_bytes()
        for dev, peak in final_peak.items():
            self.assertLessEqual(peak, usable[dev], dev)

        # Degradation order: the cache shrank first.
        actions = [d["action"] for d in adm["degradations"]]
        self.assertIn("shrink_cache", actions)
        self.assertLess(max(adm["chosen"]["cache_budgets_bytes"].values(),
                            default=0),
                        max(C2_CACHE.values()))
        if "spill_experts" in actions:
            self.assertLess(actions.index("shrink_cache"),
                            actions.index("spill_experts"),
                            "cache must shrink before experts spill")

        # The warning is surfaced on the plan itself, in the approved
        # contract wording.
        self.assertTrue(any("fall back to a smaller cache" in w
                            for w in plan.warnings), plan.warnings)

        # Machine-readable: requested / assumed / chosen all present.
        self.assertEqual(adm["requested"]["cache_budgets_bytes"]
                         ["SYCL0"], C2_CACHE["SYCL0"])
        self.assertEqual(adm["assumed"]["usable_bytes"], usable)

    def test_c2_degraded_argv_carries_the_shrunk_cache(self):
        caps = {"schema": 2,
                "features": {"moe_weight_transfer_cache": True,
                             "moe_cache_per_device_budgets": False},
                "cli": {}}
        plan = self.make_plan(self.c2_profile(), ornith_class_layout(),
                              capabilities=caps)
        argv = list(plan.argv)
        self.assertIn("--moe-cache-bytes", argv)
        emitted = int(argv[argv.index("--moe-cache-bytes") + 1])
        self.assertLess(emitted, max(C2_CACHE.values()),
                        "argv still asks for the crashing pool size")
        self.assertEqual(
            emitted,
            max(plan.decision_data["admission"]["chosen"]
                ["cache_budgets_bytes"].values()),
            "argv and the admission record disagree about the cache")


class TestOversubscriptionBoundaries(AdmissionBase):
    """Exactly-fits is honored untouched; one byte over degrades."""

    TOTAL = 16 * GIB

    def boundary_profile(self, budget):
        return modelctl.normalize_profile({
            "name": "boundary",
            "model_path": str(self.model),
            "backend": "llama-cpp",
            "config": {"ctx": 4096, "cache_type_k": "q8_0",
                       "cache_type_v": "q8_0", "device": "SYCL0",
                       "fit": "off", "extra": ""},
            "moe_cache": {"mode": "manual",
                          "gpu": {"budgets_bytes": {"SYCL0": budget}}},
        })

    def small_layout(self):
        return {"arch": "testmoe", "meta": {}, "block_count": 1,
                "is_moe": True, "weight_bytes": 6 * GIB,
                "non_expert_bytes": 4 * GIB, "other_bytes": GIB,
                "layer_bytes": 3 * GIB,
                "expert_bytes_per_layer": {0: 2 * GIB},
                "has_shexp": False, "unknown_type_tensors": 0}

    CAPS = {"schema": 3,
            "features": {"moe_weight_transfer_cache": True,
                         "moe_cache_per_device_budgets": True},
            "cli": {}}

    def exact_budget(self):
        # static = weights 6 GiB + KV (4096 * 256 KiB = 1 GiB)
        #        + overhead max(1 GiB, 10% of 6 GiB) = 1 GiB  -> 8 GiB,
        # plus the 1.5 GiB compute-reserve floor delta (1.5 - 1 = 0.5).
        usable = int(self.TOTAL * 0.9)
        return usable - 8 * GIB - GIB // 2

    def test_exact_fit_is_untouched(self):
        budget = self.exact_budget()
        plan = self.make_plan(self.boundary_profile(budget),
                              self.small_layout(),
                              gpus=[FakeGpu("SYCL0", self.TOTAL)],
                              capabilities=self.CAPS)
        self.assertEqual(plan.claim.vram_cache_bytes, {"SYCL0": budget},
                         "an exactly-fitting budget must be honored as-is")
        self.assertNotIn("admission", plan.decision_data,
                         "no degradation record on a fitting plan")
        self.assertEqual(plan.warnings, ())

    def test_one_byte_over_shrinks_by_one_byte(self):
        budget = self.exact_budget()
        plan = self.make_plan(self.boundary_profile(budget + 1),
                              self.small_layout(),
                              gpus=[FakeGpu("SYCL0", self.TOTAL)],
                              capabilities=self.CAPS)
        adm = plan.decision_data["admission"]
        self.assertTrue(adm["fits"])
        self.assertEqual(plan.claim.vram_cache_bytes, {"SYCL0": budget})
        self.assertEqual(adm["degradations"][0]["action"], "shrink_cache")
        self.assertTrue(any("fall back to a smaller cache" in w
                            for w in plan.warnings))

    def test_pins_alone_oversubscribed_spill_to_cpu(self):
        # No cache to shrink: 10 pinned expert layers of 2 GiB against a
        # 16 GiB card. Degradation step 2 must move layers to CPU until
        # the device admits, rewriting only the routed -ot rules.
        layout = {"arch": "testmoe", "meta": {}, "block_count": 10,
                  "is_moe": True, "weight_bytes": 24 * GIB,
                  "non_expert_bytes": 4 * GIB, "other_bytes": GIB,
                  "layer_bytes": 3 * GIB,
                  "expert_bytes_per_layer": {i: 2 * GIB for i in range(10)},
                  "has_shexp": False, "unknown_type_tensors": 0}
        profile = modelctl.normalize_profile({
            "name": "pinned",
            "model_path": str(self.model),
            "backend": "llama-cpp",
            "config": {"ctx": 4096, "cache_type_k": "q8_0",
                       "cache_type_v": "q8_0", "device": "SYCL0",
                       "fit": "off",
                       "extra": r"--fit off -ot blk\\.[0-9]\\.ffn_.*_exps=SYCL0"
                                r" --ubatch-size 128"},
        })
        plan = self.make_plan(profile, layout,
                              gpus=[FakeGpu("SYCL0", self.TOTAL)])
        adm = plan.decision_data["admission"]
        self.assertTrue(adm["fits"], adm)
        spills = [d for d in adm["degradations"]
                  if d["action"] == "spill_experts"]
        self.assertEqual(len(spills), 1)
        self.assertEqual(spills[0]["device"], "SYCL0")
        self.assertTrue(spills[0]["layers"])
        usable = int(self.TOTAL * 0.9)
        self.assertLessEqual(plan.claim.vram_admission_bytes()["SYCL0"],
                             usable)
        extra = plan.config["extra"]
        self.assertIn("ffn_.*_exps=CPU", extra,
                      "spilled layers need the CPU catch-all rule")
        self.assertIn("--ubatch-size 128", extra,
                      "non-placement flags must survive the rewrite")
        self.assertTrue(any("spill to CPU" in w for w in plan.warnings))

    def test_estimate_quality_claims_are_left_alone(self):
        # No parsed layout -> whole-model estimate claim. Admission must
        # not degrade plans over numbers it knows are wrong; launch-time
        # feasibility owns that refusal.
        profile = self.boundary_profile(4 * GIB)
        snap = FakeSnapshot([FakeGpu("SYCL0", self.TOTAL)])
        with mock.patch.object(modelctl_plans.modelctl_vram,
                               "gguf_model_layout", return_value=None), \
             mock.patch.object(modelctl_plans.modelctl_vram,
                               "weights_bytes_on_disk",
                               return_value=64 * GIB), \
             mock.patch.object(modelctl_plans, "_block_device_for",
                               return_value="/dev/test0"), \
             mock.patch.object(modelctl, "load_defaults",
                               return_value={"vram_limit_pct": 90,
                                             "primary_gpu": "SYCL0"}):
            plan = modelctl_plans.current_profile_plan(
                profile, hardware=snap, capabilities=self.CAPS)
        self.assertEqual(plan.claim.layout_source, "estimate")
        self.assertNotIn("admission", plan.decision_data)


class TestC2ShapeThroughTierPlanner(unittest.TestCase):
    """plan_tiers on the C2 request: degraded, warned, admissible."""

    INVENTORY = [
        {"device": "SYCL0", "name": "Intel(R) Arc(TM) Pro B70 Graphics",
         "total_bytes": B70_TOTAL, "free_bytes": 0},
        {"device": "SYCL1", "name": "Intel(R) Arc(TM) B580 Graphics",
         "total_bytes": B580_TOTAL, "free_bytes": 0},
    ]

    def test_c2_cache_request_plans_degraded_with_warning(self):
        profile = {"name": "ornith-397b", "model_path": "/fake/m.gguf",
                   "config": {"ctx": 4096, "cache_type_k": "q8_0",
                              "cache_type_v": "q4_0", "extra": C2_PINS}}
        plan = modelctl_tiers.plan_tiers(
            profile, self.INVENTORY, 90, "SYCL0",
            ram_available=25 * GIB, layout=ornith_class_layout(),
            cache_request={"mode": "manual",
                           "gpu": {"budgets_bytes": dict(C2_CACHE)}},
            capabilities=None, hw_settings={})
        adm = plan["admission"]
        self.assertTrue(adm["fits"], adm)
        for dev, row in adm["devices"].items():
            self.assertLessEqual(row["demand_bytes"], row["usable_bytes"],
                                 dev)
        # Uniform backend: 17 GiB collapses onto the B580 too, so the
        # budget must shrink -- and the shrink is recorded and warned.
        self.assertTrue(plan["cache_budgets"])
        self.assertLess(max(plan["cache_budgets"].values()),
                        max(C2_CACHE.values()))
        self.assertTrue(any(d["action"] == "shrink_cache"
                            for d in adm["degradations"]))
        self.assertTrue(any("fall back to a smaller cache" in w
                            for w in plan["warnings"]))
        # Raw admission math is reported per device with every component.
        for dev, row in adm["devices"].items():
            self.assertEqual(row["demand_bytes"],
                             row["weights_bytes"] + row["kv_bytes"]
                             + row["overhead_bytes"]
                             + row["pinned_expert_bytes"]
                             + row["cache_bytes"]
                             + row["compute_reserve_bytes"], dev)

    def test_infeasible_fixed_set_is_flagged_not_silent(self):
        # KV so large even the cache-free, all-spilled plan cannot admit:
        # the plan says so machine-readably instead of pretending.
        profile = {"name": "kvbomb", "model_path": "/fake/m.gguf",
                   "config": {"ctx": 1048576, "cache_type_k": "q8_0",
                              "cache_type_v": "q8_0", "extra": ""}}
        plan = modelctl_tiers.plan_tiers(
            profile, self.INVENTORY, 90, "SYCL0",
            ram_available=25 * GIB, layout=ornith_class_layout(),
            capabilities=None, hw_settings={})
        self.assertFalse(plan["admission"]["fits"])
        self.assertTrue(any("oversubscribes" in w for w in plan["warnings"]))


class TestPlanIsJsonSerializable(unittest.TestCase):
    def test_admission_record_serializes(self):
        plan = modelctl_tiers.plan_tiers(
            {"name": "m", "model_path": "/fake/m.gguf",
             "config": {"ctx": 4096, "extra": ""}},
            TestC2ShapeThroughTierPlanner.INVENTORY, 90, "SYCL0",
            ram_available=25 * GIB, layout=ornith_class_layout(),
            hw_settings={})
        json.dumps(plan)  # must not raise


if __name__ == "__main__":
    unittest.main()
