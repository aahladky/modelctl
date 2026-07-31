"""Task D1: resource claims report where the bytes actually go.

The rule these exist to enforce: mmap-addressed model bytes are never
presented as guaranteed resident RAM. An oversized MoE served by mmap
does not need 200 GiB of RAM, and a claim that says it does makes every
plan look infeasible.
"""
import unittest
from unittest import mock

import modelctl_plans
from modelctl_plans import ResourceClaim


class FakeGpu:
    def __init__(self, device, total=16 << 30):
        self.device = device
        self.name = device
        self.total_bytes = total
        self.free_bytes = total
        self.enabled = True
        self.role = ""


class FakeHardware:
    def __init__(self, devices=("SYCL0", "SYCL1")):
        self.gpus = tuple(FakeGpu(d) for d in devices)

    def gpu_by_device(self, device):
        return next((g for g in self.gpus if g.device == device), None)


def claim_for(config=None, moe_cache=None, model_bytes=32 << 30, is_moe=True):
    """Build a claim with the GGUF layout stubbed to a known shape."""
    profile = {
        "name": "m1", "model_path": "/models/m.gguf",
        "config": {"ctx": 8192, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                   "device": "SYCL0", "extra": "", **(config or {})},
    }
    if moe_cache:
        profile["moe_cache"] = moe_cache
    layout = {
        "weight_bytes": model_bytes,
        "non_expert_bytes": model_bytes // 8,
        "layer_bytes": model_bytes // 2,
        "block_count": 48,
        "is_moe": is_moe,
        "expert_bytes_per_layer": {i: (model_bytes // 2) // 48 for i in range(48)},
        "other_bytes": model_bytes // 8,
        "meta": {},
    }
    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("modelctl_vram.gguf_model_layout", return_value=layout), \
         mock.patch("modelctl_vram.gguf_kv_params", return_value=None), \
         mock.patch.object(modelctl_plans, "_block_device_for",
                           return_value="/dev/nvme0n1p1"):
        return modelctl_plans._make_claim(
            profile, profile["config"], FakeHardware())


class TestMmapIsNotResidentRam(unittest.TestCase):
    def test_mmap_bytes_are_not_counted_as_resident(self):
        c = claim_for({"extra": "-ngl 10"})
        self.assertEqual(c.storage_mode, "mmap")
        self.assertGreater(c.mmap_bytes, 0)
        self.assertEqual(c.ram_resident_bytes, 0)

    def test_no_mmap_makes_cpu_weights_resident(self):
        c = claim_for({"extra": "-ngl 10 --no-mmap"})
        self.assertEqual(c.storage_mode, "none")
        self.assertEqual(c.mmap_bytes, 0)
        self.assertGreater(c.ram_resident_bytes, 0)
        self.assertEqual(c.ram_resident_bytes, c.ram_bytes)

    def test_expected_resident_bytes_no_longer_includes_vram(self):
        # It was computed as cpu_bytes + sum(vram), which is neither RSS
        # nor anything else meaningful.
        c = claim_for({"extra": "-ngl 10"})
        self.assertEqual(c.expected_resident_bytes, c.ram_resident_bytes)
        self.assertLess(c.expected_resident_bytes, sum(c.vram_bytes.values()))

    def test_moe_working_set_is_smaller_than_the_mmap_size(self):
        # Only routed experts are touched, so the hot set is a fraction.
        c = claim_for({"extra": "-ngl 10"}, is_moe=True)
        self.assertGreater(c.page_cache_working_set_bytes, 0)
        self.assertLess(c.page_cache_working_set_bytes, c.mmap_bytes)

    def test_dense_model_working_set_is_the_whole_offload(self):
        # A dense model has no sparse-routing saving to claim.
        c = claim_for({"extra": "-ngl 10"}, is_moe=False)
        if c.mmap_bytes:
            self.assertEqual(c.page_cache_working_set_bytes, c.mmap_bytes)

    def test_fully_resident_plan_has_no_working_set(self):
        c = claim_for({"extra": "--no-mmap"})
        self.assertEqual(c.page_cache_working_set_bytes, 0)


class TestVramDecomposition(unittest.TestCase):
    def test_components_sum_to_the_device_total(self):
        c = claim_for()
        for device in c.vram_bytes:
            b = c.device_breakdown(device)
            self.assertEqual(b["static"] + b["kv"] + b["overhead"], b["total"],
                             f"{device} components do not sum to its total")

    def test_each_component_is_reported_separately(self):
        c = claim_for()
        b = c.device_breakdown("SYCL0")
        self.assertGreater(b["static"], 0)
        self.assertGreater(b["overhead"], 0)
        self.assertIn("kv", b)

    def test_tensor_split_distributes_every_component(self):
        c = claim_for({"device": "", "tensor_split": "3,1"})
        self.assertEqual(set(c.vram_bytes), {"SYCL0", "SYCL1"})
        for device in ("SYCL0", "SYCL1"):
            b = c.device_breakdown(device)
            self.assertGreater(b["static"], 0)
            self.assertEqual(b["static"] + b["kv"] + b["overhead"], b["total"])
        # The larger share really is larger.
        self.assertGreater(c.vram_static_bytes["SYCL0"],
                           c.vram_static_bytes["SYCL1"])

    def test_cache_budget_is_additional_not_a_slice(self):
        # The tier planner reserves the cache budget before placing static
        # experts, so peak > static reservation.
        c = claim_for(moe_cache={"mode": "manual",
                                 "gpu": {"budgets_bytes": {"SYCL0": 2 << 30}}})
        b = c.device_breakdown("SYCL0")
        self.assertEqual(b["cache"], 2 << 30)
        self.assertEqual(b["peak"], b["total"] + (2 << 30))

    def test_no_cache_configured_means_no_cache_claim(self):
        c = claim_for()
        self.assertEqual(c.vram_cache_bytes, {})
        self.assertEqual(c.cache_bytes, 0)
        self.assertEqual(c.device_breakdown("SYCL0")["peak"],
                         c.device_breakdown("SYCL0")["total"])


class TestCanonicalAdmission(unittest.TestCase):
    """vram_admission_bytes()/ram_admission_bytes() are the only place
    admission semantics are defined; every feasibility, reservation, and
    matrix path charges these values."""

    def test_vram_admission_includes_the_cache_budget(self):
        c = claim_for(moe_cache={"mode": "manual",
                                 "gpu": {"budgets_bytes": {"SYCL0": 2 << 30}}})
        peak = c.vram_admission_bytes()
        self.assertEqual(peak["SYCL0"],
                         c.vram_bytes["SYCL0"] + (2 << 30))

    def test_vram_admission_matches_the_breakdown_peak(self):
        c = claim_for(moe_cache={"mode": "manual",
                                 "gpu": {"budgets_bytes": {"SYCL0": 2 << 30}}})
        peak = c.vram_admission_bytes()
        for dev in peak:
            self.assertEqual(peak[dev], c.device_breakdown(dev)["peak"])

    def test_vram_admission_without_cache_is_the_static_claim(self):
        c = claim_for()
        self.assertEqual(c.vram_admission_bytes(), c.vram_bytes)

    def test_cache_on_a_device_with_no_static_claim_is_still_admitted(self):
        # A cache budget on a device the static placement skipped is a
        # real allocation on that device.
        c = ResourceClaim(vram_bytes={"SYCL0": 8 << 30}, ram_bytes=0,
                          storage_mode="none", expected_context=8192,
                          vram_cache_bytes={"SYCL1": 2 << 30})
        peak = c.vram_admission_bytes()
        self.assertEqual(peak["SYCL0"], 8 << 30)
        self.assertEqual(peak["SYCL1"], 2 << 30)

    def test_mmap_bytes_are_not_charged_as_ram_admission(self):
        c = claim_for({"extra": "-ngl 10"})
        self.assertEqual(c.storage_mode, "mmap")
        self.assertGreater(c.mmap_bytes, 0)
        self.assertEqual(c.ram_admission_bytes(), 0)

    def test_no_mmap_charges_resident_bytes(self):
        c = claim_for({"extra": "-ngl 10 --no-mmap"})
        self.assertEqual(c.ram_admission_bytes(), c.ram_resident_bytes)
        self.assertGreater(c.ram_admission_bytes(), 0)

    def test_legacy_claim_without_resident_split_falls_back_to_ram_bytes(self):
        # Claims built before the D1 split have ram_resident_bytes == 0;
        # for a non-mmap claim ram_bytes is the resident figure.
        c = ResourceClaim(vram_bytes={}, ram_bytes=4 << 30,
                          storage_mode="none", expected_context=8192)
        self.assertEqual(c.ram_admission_bytes(), 4 << 30)

    def test_staging_bytes_are_part_of_ram_admission(self):
        c = ResourceClaim(vram_bytes={}, ram_bytes=0, storage_mode="mmap",
                          expected_context=8192, ram_resident_bytes=1 << 30,
                          mmap_bytes=32 << 30, staging_bytes=1 << 29)
        self.assertEqual(c.ram_admission_bytes(), (1 << 30) + (1 << 29))


class TestStorageIdentity(unittest.TestCase):
    def test_claim_records_the_backing_device(self):
        # Two plans reading from different disks do not have the same read
        # cost; the claim has to say which disk.
        c = claim_for()
        self.assertEqual(c.storage_device, "/dev/nvme0n1p1")
        self.assertEqual(c.storage_path, "/models/m.gguf")

    def test_undeterminable_device_is_empty_not_guessed(self):
        self.assertEqual(modelctl_plans._block_device_for(""), "")


class TestBreakdownDict(unittest.TestCase):
    def test_breakdown_is_populated(self):
        # It was a documented field that nothing ever wrote.
        c = claim_for()
        self.assertTrue(c.breakdown)
        self.assertIn("vram", c.breakdown)
        self.assertIn("ram", c.breakdown)
        self.assertIn("SYCL0", c.breakdown["vram"])

    def test_breakdown_ram_separates_mmap_from_resident(self):
        c = claim_for({"extra": "-ngl 10"})
        ram = c.breakdown["ram"]
        self.assertEqual(ram["resident"], 0)
        self.assertGreater(ram["mmap"], 0)
        self.assertIn("page_cache_working_set", ram)
        self.assertIn("staging", ram)


class TestDefaultsRemainSafe(unittest.TestCase):
    def test_a_bare_claim_still_constructs(self):
        # Every new field has a default, so existing constructions in
        # tests and the OVMS path keep working.
        c = ResourceClaim(vram_bytes={"SYCL0": 1 << 30}, ram_bytes=0,
                          storage_mode="none", expected_context=8192)
        self.assertEqual(c.mmap_bytes, 0)
        self.assertEqual(c.vram_total(), 1 << 30)
        self.assertEqual(c.device_breakdown("SYCL0")["total"], 1 << 30)


if __name__ == "__main__":
    unittest.main()
