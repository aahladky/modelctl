"""Per-device MoE cache budgets (Phase 5 remediation).

Two runtime contracts have to stay distinguishable end to end:
  schema 3+ -- --moe-cache-bytes takes a device map, honoured literally;
               devices the map omits get no cache and none is reserved.
  older     -- one uniform budget applied to every device that creates a
               cache, so the map collapses to its max and that figure is
               reserved everywhere.
Emitting one contract's flags while reserving the other's is how the
runtime cache collides with statically placed experts.
"""
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import mock

import modelctl
import modelctl_plans
import modelctl_tiers

GIB = 1 << 30

CAPS_PER_DEVICE = {
    "schema": 3,
    "features": {"moe_weight_transfer_cache": True,
                 "moe_cache_per_device_budgets": True},
    "cli": {},
}
CAPS_UNIFORM = {
    "schema": 2,
    "features": {"moe_weight_transfer_cache": True,
                 "moe_cache_per_device_budgets": False},
    "cli": {},
}

BUDGETS = {"SYCL0": 8 * GIB, "SYCL1": 2 * GIB}


def profile_with_budgets(budgets=None):
    return {
        "name": "m1", "model_path": "/x/m.gguf",
        "config": {"device": "", "ctx": 8192, "extra": ""},
        "moe_cache": {"mode": "manual",
                      "gpu": {"budgets_bytes": dict(
                          BUDGETS if budgets is None else budgets)}},
    }


class TestFlagEmission(unittest.TestCase):
    def test_schema3_emits_the_device_map(self):
        args = modelctl.build_moe_cache_args(
            profile_with_budgets(), capabilities=CAPS_PER_DEVICE)
        self.assertIn("--moe-cache-bytes", args)
        spec = args[args.index("--moe-cache-bytes") + 1]
        self.assertEqual(spec, f"SYCL0={8 * GIB},SYCL1={2 * GIB}")

    def test_older_backend_collapses_to_the_max(self):
        args = modelctl.build_moe_cache_args(
            profile_with_budgets(), capabilities=CAPS_UNIFORM)
        spec = args[args.index("--moe-cache-bytes") + 1]
        self.assertEqual(spec, str(8 * GIB),
                         "a uniform-budget backend must get one number, and "
                         "the max -- a smaller one under-reserves somewhere")

    def test_incapable_backend_gets_no_cache_flags(self):
        args = modelctl.build_moe_cache_args(
            profile_with_budgets(),
            capabilities={"features": {}, "cli": {}})
        self.assertNotIn("--moe-cache-bytes", args)


class TestTierReservation(unittest.TestCase):
    INVENTORY = [
        {"device": "SYCL0", "name": "B70", "total_bytes": 32 * GIB,
         "free_bytes": 32 * GIB},
        {"device": "SYCL1", "name": "B580", "total_bytes": 12 * GIB,
         "free_bytes": 12 * GIB},
    ]

    def _plan(self, caps, budgets=None):
        layout = {"is_moe": True, "block_count": 8, "weight_bytes": 6 * GIB,
                  "expert_bytes": 4 * GIB, "other_bytes": 2 * GIB,
                  "expert_bytes_per_layer": {i: (4 * GIB) // 8 for i in range(8)},
                  "meta": {}}
        return modelctl_tiers.plan_tiers(
            profile_with_budgets(budgets), self.INVENTORY, 90, "SYCL0",
            ram_available=32 * GIB, layout=layout,
            cache_request=profile_with_budgets(budgets)["moe_cache"],
            capabilities=caps)

    def test_schema3_reserves_each_device_its_own_budget(self):
        plan = self._plan(CAPS_PER_DEVICE)
        self.assertEqual(plan["cache_budgets"], BUDGETS,
                         "per-device backend must reserve the map as written")

    def test_older_backend_reserves_the_uniform_max_everywhere(self):
        plan = self._plan(CAPS_UNIFORM)
        self.assertEqual(set(plan["cache_budgets"].values()), {8 * GIB},
                         "uniform backend allocates the same budget on every "
                         "device that creates a cache")

    def test_budget_too_big_for_its_own_device_shrinks_to_fit(self):
        # 20 GiB named on the 12 GiB card: never honored as-is, never
        # disabled outright -- it falls back to what the device can give,
        # loudly, with a machine-readable degradation record.
        plan = self._plan(CAPS_PER_DEVICE,
                          budgets={"SYCL0": 4 * GIB, "SYCL1": 20 * GIB})
        self.assertEqual(plan["cache_budgets"]["SYCL0"], 4 * GIB,
                         "the budget that fit must be honored as written")
        self.assertLess(plan["cache_budgets"]["SYCL1"],
                        int(12 * GIB * 0.9) + 1)
        self.assertLess(plan["cache_budgets"]["SYCL1"], 20 * GIB)
        self.assertTrue(any("fall back to a smaller cache" in w
                            for w in plan["warnings"]))
        self.assertTrue(any(d["action"] == "shrink_cache"
                            and d["device"] == "SYCL1"
                            and d["requested_bytes"] == 20 * GIB
                            for d in plan["admission"]["degradations"]))
        self.assertTrue(plan["admission"]["fits"])

    def test_unused_device_in_the_map_is_reported(self):
        plan = self._plan(CAPS_PER_DEVICE,
                          budgets={"SYCL0": 4 * GIB, "SYCL9": 1 * GIB})
        self.assertNotIn("SYCL9", plan["cache_budgets"])
        self.assertTrue(any("SYCL9" in w for w in plan["warnings"]))


class TestClaimCharging(unittest.TestCase):
    def _claim(self, caps):
        import modelctl_hardware
        hw = modelctl_hardware.HardwareSnapshot(
            captured_at=0.0, fingerprint="t",
            gpus=(modelctl_hardware.GpuSnapshot(
                      device="SYCL0", name="B70",
                      total_bytes=32 * GIB, free_bytes=32 * GIB,
                      reserve_bytes=0, enabled=True, role="primary",
                      bandwidth_gbs=608.0),
                  modelctl_hardware.GpuSnapshot(
                      device="SYCL1", name="B580",
                      total_bytes=12 * GIB, free_bytes=12 * GIB,
                      reserve_bytes=0, enabled=True, role="",
                      bandwidth_gbs=456.0)),
            ram_total_bytes=64 * GIB, ram_available_bytes=32 * GIB,
            ram_reserve_bytes=0, storage=(), backend_fingerprints={})
        profile = profile_with_budgets()
        return modelctl_plans._make_claim(
            profile, profile["config"], hw, caps)

    def test_schema3_charges_each_device_its_own_budget(self):
        claim = self._claim(CAPS_PER_DEVICE)
        self.assertEqual(claim.vram_cache_bytes.get("SYCL1"), 2 * GIB,
                         "per-device backend charges the declared budget")

    def test_older_backend_charges_the_uniform_budget_everywhere(self):
        claim = self._claim(CAPS_UNIFORM)
        self.assertEqual(claim.vram_cache_bytes.get("SYCL1"), 8 * GIB,
                         "uniform backend allocates the max on every named "
                         "device, so admission must charge that")


class TestCacheEditorRoundTrip(unittest.TestCase):
    """The console's per-GPU budget editor was a no-op: the add-device
    input had no value field and was explicitly skipped, and every save
    rebuilt moe_cache from scratch, dropping CLI-set sections.

    The form moved to POST /api/v2/models/{name}/config (console phase 3);
    the same two defects are still reachable there -- it takes a
    budgets_bytes map and a mode and must merge them into the stored
    moe_cache, not replace it."""

    def test_config_save_adds_new_device_and_preserves_other_sections(self):
        from modelctl_web.app import create_app
        from modelctl_web.jobs import JobStore, JobRunner
        from fastapi.testclient import TestClient

        with TemporaryDirectory() as tmp:
            profiles = Path(tmp) / "profiles"
            profiles.mkdir()
            (profiles / "m1.json").write_text(json.dumps({
                "name": "m1", "model_path": "/x/m.gguf",
                "config": {"device": "SYCL0", "ctx": 8192, "ttl": 3600,
                           "flash_attn": "auto", "extra": ""},
                # Every section here but gpu.budgets_bytes and mode is one
                # the typed save never writes -- i.e. CLI-set state it can
                # only ever destroy.
                "moe_cache": {"mode": "manual",
                              "gpu": {"budgets_bytes": {},
                                      "policy": "lru",
                                      "admission_misses": 5},
                              "prefetch": {"enabled": True},
                              "ram": {"budget_bytes": 123},
                              "decode": {"miss_execution": "cpu"},
                              "prefill": {"admit_to_gpu_cache": True},
                              "storage": {"mode": "direct"}},
                "enabled": True,
            }))
            store = JobStore(Path(tmp) / "jobs.db")
            runner = JobRunner(store)
            client = TestClient(create_app(store=store, runner=runner))

            with mock.patch.object(modelctl, "PROFILES_DIR", profiles), \
                 mock.patch("modelctl_web.mutate.submit_moe_cache",
                            return_value="job1") as submit:
                r = client.post(
                    "/api/v2/models/m1/config",
                    json={"moe_mode": "manual",
                          "budgets_bytes": {"SYCL2": 4 * GIB},
                          # A budget edit on a placed profile is structural.
                          "accept_structural": True},
                    headers={"Authorization": "Bearer t"})

        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["jobs"]["moe_cache"], "job1")
        mc = submit.call_args.args[2]
        self.assertEqual(mc.get("gpu", {}).get("budgets_bytes"),
                         {"SYCL2": 4 * GIB},
                         "a device named only in the request still cannot "
                         "get a budget")
        self.assertTrue(mc.get("prefetch", {}).get("enabled"),
                        "save dropped a section the form does not render")
        self.assertEqual(mc.get("ram", {}).get("budget_bytes"), 123)
        self.assertEqual(mc.get("decode", {}).get("miss_execution"), "cpu")
        self.assertTrue(mc.get("prefill", {}).get("admit_to_gpu_cache"))
        self.assertEqual(mc.get("storage", {}).get("mode"), "direct")
        # Siblings of the one gpu key the save owns have to survive too.
        self.assertEqual(mc["gpu"]["policy"], "lru")
        self.assertEqual(mc["gpu"]["admission_misses"], 5)


if __name__ == "__main__":
    unittest.main()
