"""Stable planner inputs and replan gating.

The disease's second symptom: the planner read instantaneous free RAM,
so replanning laguna on a busy afternoon demoted it tier 3 -> 4 and
rewrote its split though nothing about the profile changed.

Contract under test:
  * everything plan_tiers reads from the live machine (planning RAM
    first) is an explicit recorded input -- defaulted from the machine on
    first plan, stored in the profile, reused on replan unless refreshed;
  * planning twice with identical stored inputs is byte-identical;
  * for migrated legacy profiles (no stored inputs), a replan must be a
    no-op or a pins-only diff -- any change to tier, split, or placement
    beyond pins requires an explicit --accept-tier-change, without which
    the plan is shown and the profile untouched.

Hermetic: synthetic inventories and layouts, tmp profile dirs, no live
probes (hw_settings and capabilities always passed explicitly).
"""
import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_tiers
import modelctl_vram

GIB = 1 << 30

B70 = {"device": "SYCL0", "name": "Intel(R) Arc(TM) Pro B70 Graphics",
       "total_bytes": 32656 * (1 << 20), "free_bytes": 0}
B580 = {"device": "SYCL1", "name": "Intel(R) Arc(TM) B580 Graphics",
        "total_bytes": 12216 * (1 << 20), "free_bytes": 0}
INVENTORY = [B70, B580]


def laguna_class_layout():
    """qwen35moe pattern: shexp, MoE from layer 0, RAM-resident tier 3."""
    layers = {i: int(1.1 * GIB) for i in range(48)}
    return {"arch": "testmoe", "meta": {}, "block_count": 48,
            "is_moe": True, "weight_bytes": sum(layers.values()) + 4 * GIB,
            "non_expert_bytes": 4 * GIB, "other_bytes": GIB,
            "layer_bytes": 3 * GIB, "expert_bytes_per_layer": layers,
            "has_shexp": True, "unknown_type_tensors": 0}


def ornith_class_layout():
    """DeepSeek pattern: shexp + dense blk.0-2, SSD tier 4."""
    layers = {i: int(2.9 * GIB) for i in range(3, 64)}
    return {"arch": "testmoe", "meta": {}, "block_count": 64,
            "is_moe": True, "weight_bytes": sum(layers.values()) + 4 * GIB,
            "non_expert_bytes": 4 * GIB, "other_bytes": GIB,
            "layer_bytes": 3 * GIB, "expert_bytes_per_layer": layers,
            "has_shexp": True, "unknown_type_tensors": 0}


def make_profile(name="m1", ctx=8192, extra=""):
    return {"name": name, "model_path": "/fake/m.gguf",
            "config": {"ctx": ctx, "cache_type_k": "q8_0",
                       "cache_type_v": "q8_0", "device": "",
                       "split_mode": "", "tensor_split": "", "extra": extra},
            "moe_cache": {"mode": "off"}}


def inputs_for(ram_gib):
    return modelctl_tiers.make_planning_inputs(
        INVENTORY, 90, "SYCL0", ram_gib * GIB,
        hw_settings={}, capabilities=None)


def plan_with(profile, inputs, layout):
    return modelctl_tiers.plan_tiers(
        profile, inputs["inventory"], inputs["vram_limit_pct"],
        inputs["primary"], ram_available=inputs["ram_available_bytes"],
        layout=layout, cache_request=profile.get("moe_cache"),
        capabilities=inputs.get("capabilities"),
        hw_settings=inputs.get("hw_settings"))


class TestStoredInputRoundTrip(unittest.TestCase):
    def test_record_store_load_is_lossless(self):
        profile = make_profile()
        inputs = inputs_for(31)
        self.assertTrue(
            modelctl_tiers.record_planning_inputs(profile, inputs,
                                                  recorded_at="2026-08-01"))
        # Through JSON, exactly as save_profile/load_profile would.
        loaded = json.loads(json.dumps(profile))
        self.assertEqual(modelctl_tiers.stored_planning_inputs(loaded),
                         inputs)
        self.assertEqual(loaded["planning"]["recorded_at"], "2026-08-01")
        # Re-recording identical inputs is a no-op.
        self.assertFalse(
            modelctl_tiers.record_planning_inputs(loaded, inputs))

    def test_resolver_prefers_stored_and_never_probes(self):
        profile = make_profile()
        modelctl_tiers.record_planning_inputs(profile, inputs_for(31))
        with mock.patch.object(modelctl, "get_gpu_inventory",
                               side_effect=AssertionError("live probe")), \
             mock.patch.object(modelctl_vram, "system_ram_available",
                               side_effect=AssertionError("live probe")):
            inputs, source = modelctl.resolve_planning_inputs(profile)
        self.assertEqual(source, "stored")
        self.assertEqual(inputs["ram_available_bytes"], 31 * GIB)

    def test_refresh_rereads_the_machine(self):
        profile = make_profile()
        modelctl_tiers.record_planning_inputs(profile, inputs_for(31))
        with mock.patch.object(modelctl_vram, "system_ram_available",
                               return_value=21 * GIB), \
             mock.patch.object(modelctl, "get_gpu_inventory",
                               return_value=INVENTORY), \
             mock.patch.object(modelctl, "load_defaults",
                               return_value={"vram_limit_pct": 90,
                                             "primary_gpu": "SYCL0"}), \
             mock.patch.object(modelctl, "resolve_primary_gpu",
                               return_value="SYCL0"):
            inputs, source = modelctl.resolve_planning_inputs(profile,
                                                              refresh=True)
        self.assertEqual(source, "live")
        self.assertEqual(inputs["ram_available_bytes"], 21 * GIB)


class TestByteIdenticalReplan(unittest.TestCase):
    def assert_byte_identical(self, layout, ctx=8192):
        profile = make_profile(ctx=ctx)
        inputs = inputs_for(31)
        one = plan_with(profile, inputs, layout)
        two = plan_with(profile, json.loads(json.dumps(inputs)), layout)
        self.assertEqual(json.dumps(one, sort_keys=True),
                         json.dumps(two, sort_keys=True))

    def test_laguna_class(self):
        self.assert_byte_identical(laguna_class_layout())

    def test_ornith_class(self):
        self.assert_byte_identical(ornith_class_layout(), ctx=4096)

    def test_stored_ram_beats_machine_drift(self):
        # The drift that motivated all of this: RAM 31 GiB when laguna was
        # planned, ~21 GiB on the replanning afternoon. With stored inputs
        # the replan cannot see the drift at all.
        profile = make_profile()
        rich = plan_with(profile, inputs_for(31), laguna_class_layout())
        replan = plan_with(profile, inputs_for(31), laguna_class_layout())
        self.assertEqual(rich["tier"], replan["tier"])
        self.assertEqual(rich["config"], replan["config"])
        # And the counterfactual really would have drifted: live 21 GiB
        # demotes the tier, which is exactly what stored inputs prevent.
        drifted = plan_with(profile, inputs_for(21), laguna_class_layout())
        self.assertNotEqual(rich["tier"], drifted["tier"],
                            "fixture no longer demonstrates drift; "
                            "adjust the RAM figures")


class TestLegacyReplanProperties(unittest.TestCase):
    """A replanned legacy profile: no-op or pins-only, never silently
    structural."""

    def replan_applied_profile(self, layout, ctx=8192):
        profile = make_profile(ctx=ctx)
        inputs = inputs_for(31)
        first = plan_with(profile, inputs, layout)
        # Apply exactly as the apply paths do.
        modelctl_tiers.apply_plan_cache_budgets(profile, first,
                                                log=lambda m: None)
        profile["config"].update(first["config"])
        second = plan_with(profile, inputs, layout)
        return profile, first, second

    def test_laguna_class_replan_is_noop(self):
        profile, first, second = self.replan_applied_profile(
            laguna_class_layout())
        self.assertEqual(first["config"], second["config"])
        diff = modelctl_tiers.classify_config_diff(profile["config"],
                                                   second["config"])
        self.assertEqual(diff["kind"], "none", diff)
        gate = modelctl_tiers.tier_change_gate(profile, second)
        self.assertFalse(gate["requires_accept"], gate)

    def test_ornith_class_replan_is_noop(self):
        profile, first, second = self.replan_applied_profile(
            ornith_class_layout(), ctx=4096)
        self.assertEqual(first["config"], second["config"])
        gate = modelctl_tiers.tier_change_gate(profile, second)
        self.assertEqual(gate["kind"], "none", gate)
        self.assertFalse(gate["requires_accept"])

    def test_pins_only_diff_passes_without_flag(self):
        # A pre-P5 profile: routed rules and split in place, no shexp
        # pins. The replan adds pins and nothing else.
        old = {"device": "", "split_mode": "layer", "tensor_split": "8,3",
               "extra": r"--fit off --device SYCL0,SYCL1 "
                        r"-ot blk\\.(?:[0-9]|1[0-9])\\.ffn_.*_exps=SYCL0,"
                        r"blk\\.2[0-5]\\.ffn_.*_exps=SYCL1,"
                        r"ffn_.*_exps=CPU --no-mmap"}
        new = {"device": "", "split_mode": "layer", "tensor_split": "8,3",
               "extra": r"--fit off --device SYCL0,SYCL1 "
                        r"-ot blk\\.(?:[0-9]|1[0-9])\\.ffn_.*_shexp=SYCL0,"
                        r"blk\\.2[0-5]\\.ffn_.*_shexp=SYCL1,"
                        r"ffn_.*_shexp=SYCL0,"
                        r"blk\\.(?:[0-9]|1[0-9])\\.ffn_.*_exps=SYCL0,"
                        r"blk\\.2[0-5]\\.ffn_.*_exps=SYCL1,"
                        r"ffn_.*_exps=CPU --no-mmap"}
        diff = modelctl_tiers.classify_config_diff(old, new)
        self.assertEqual(diff["kind"], "pins", diff)
        gate = modelctl_tiers.tier_change_gate(
            {"config": old, "moe_cache": {"mode": "off"}},
            {"config": new, "cache_budgets": None})
        self.assertFalse(gate["requires_accept"])

    def test_structural_diff_requires_flag(self):
        old = {"device": "SYCL0", "split_mode": "", "tensor_split": "",
               "extra": ""}
        new = {"device": "", "split_mode": "layer", "tensor_split": "8,3",
               "extra": "--fit off --device SYCL0,SYCL1"}
        diff = modelctl_tiers.classify_config_diff(old, new)
        self.assertEqual(diff["kind"], "structural")
        gate = modelctl_tiers.tier_change_gate(
            {"config": old, "moe_cache": {"mode": "off"}},
            {"config": new, "cache_budgets": None})
        self.assertTrue(gate["requires_accept"])
        self.assertTrue(gate["changes"])

    def test_cache_budget_change_is_structural(self):
        cfg = {"device": "SYCL0", "split_mode": "", "tensor_split": "",
               "extra": ""}
        gate = modelctl_tiers.tier_change_gate(
            {"config": cfg,
             "moe_cache": {"mode": "manual",
                           "gpu": {"budgets_bytes": {"SYCL0": 8 * GIB}}}},
            {"config": dict(cfg), "cache_budgets": {"SYCL0": 2 * GIB}})
        self.assertTrue(gate["requires_accept"])

    def test_first_placement_needs_no_flag(self):
        empty = {"device": "", "split_mode": "", "tensor_split": "",
                 "extra": ""}
        new = {"device": "", "split_mode": "layer", "tensor_split": "8,3",
               "extra": "--fit off --device SYCL0,SYCL1"}
        gate = modelctl_tiers.tier_change_gate(
            {"config": empty, "moe_cache": {"mode": "off"}},
            {"config": new, "cache_budgets": None})
        self.assertFalse(gate["requires_accept"],
                         "initial placement is not a placement change")


class TestApplyPathGating(unittest.TestCase):
    """cmd_place_tiers with --apply: a structural replan without
    --accept-tier-change leaves the profile untouched on disk."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profiles = Path(self.tmp.name)
        p = mock.patch.object(modelctl, "PROFILES_DIR", self.profiles)
        p.start()
        self.addCleanup(p.stop)

    def write_profile(self, profile):
        (self.profiles / f"{profile['name']}.json").write_text(
            json.dumps(profile))

    def args(self, apply=True, accept=False, refresh=False):
        return types.SimpleNamespace(
            apply=apply, accept_tier_change=accept, refresh_inputs=refresh,
            no_router_restart=True, no_hermes=True)

    def run_place(self, args):
        with mock.patch.object(modelctl_vram, "gguf_model_layout",
                               return_value=laguna_class_layout()), \
             mock.patch.object(modelctl_vram, "system_ram_available",
                               return_value=31 * GIB), \
             mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends"), \
             mock.patch.object(modelctl, "sync_hermes_custom_providers"), \
             mock.patch("modelctl_capabilities.probe_backend",
                        side_effect=RuntimeError("no probe in tests")):
            modelctl.cmd_place_tiers(
                args, INVENTORY, {"vram_limit_pct": 90,
                                  "primary_gpu": "SYCL0"},
                "SYCL0", ["m1"])

    def read_profile(self):
        return json.loads((self.profiles / "m1.json").read_text())

    def test_structural_apply_without_flag_leaves_profile_untouched(self):
        legacy = make_profile()
        legacy["config"]["device"] = "SYCL0"  # legacy pin, plan will split
        self.write_profile(legacy)
        before = (self.profiles / "m1.json").read_text()
        self.run_place(self.args(accept=False))
        self.assertEqual((self.profiles / "m1.json").read_text(), before,
                         "profile changed despite the missing flag")

    def test_structural_apply_with_flag_applies_and_records_inputs(self):
        legacy = make_profile()
        legacy["config"]["device"] = "SYCL0"
        # Nonempty env keeps the apply path off the env-script discovery
        # (this single-file run has no suite-wide glob redirect).
        legacy["env"] = ["LD_LIBRARY_PATH=/x"]
        self.write_profile(legacy)
        self.run_place(self.args(accept=True))
        saved = self.read_profile()
        self.assertEqual(saved["config"]["split_mode"], "layer")
        inputs = modelctl_tiers.stored_planning_inputs(saved)
        self.assertIsNotNone(inputs, "first apply must record the inputs")
        self.assertEqual(inputs["ram_available_bytes"], 31 * GIB)
        # Replan after a RAM drop: stored inputs win, config is stable.
        with mock.patch.object(modelctl_vram, "gguf_model_layout",
                               return_value=laguna_class_layout()), \
             mock.patch.object(modelctl_vram, "system_ram_available",
                               return_value=21 * GIB):
            plan, _inputs, source = modelctl.plan_tiers_for_profile(saved)
        self.assertEqual(source, "stored")
        self.assertEqual(plan["config"], {k: saved["config"][k]
                                          for k in plan["config"]})


if __name__ == "__main__":
    unittest.main()
