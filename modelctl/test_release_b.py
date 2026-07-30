"""Release B acceptance tests.

Validates the hybrid MoE execution control plane:
- Hybrid capability declaration and detection
- Hybrid plan generation
- Hybrid metrics integration
- Hybrid preflight validation
- Hybrid configuration contract
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_capabilities
import modelctl_plans


class TestReleaseBHybridCapability(unittest.TestCase):
    """Req: modelctl can distinguish transfer-cache and hybrid plans."""

    def test_hybrid_feature_detected(self):
        caps = {
            "schema": 2,
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": True,
            },
            "constraints": {
                "moe_hybrid_supported_archs": ["deepseek_v2"],
            },
        }
        self.assertTrue(modelctl_capabilities.supports_hybrid_miss(caps))
        self.assertIn("deepseek_v2",
                      caps["constraints"]["moe_hybrid_supported_archs"])

    def test_hybrid_feature_absent(self):
        caps = {"features": {"moe_weight_transfer_cache": True}}
        self.assertFalse(modelctl_capabilities.supports_hybrid_miss(caps))

    def test_hybrid_constraints_normalize(self):
        raw = {
            "schema": 2,
            "features": {"moe_hybrid_cpu_miss": True},
            "constraints": {
                "moe_hybrid_supported_archs": ["deepseek_v2", "qwen3_moe"],
                "moe_hybrid_supported_quant": ["q4_0", "f16"],
                "moe_hybrid_can_overlap": True,
            },
        }
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertIn("deepseek_v2",
                      norm["constraints"]["moe_hybrid_supported_archs"])
        self.assertTrue(norm["constraints"]["moe_hybrid_can_overlap"])


class TestReleaseBHybridPlans(unittest.TestCase):
    """Req: hybrid plans generated for oversized MoE with hybrid-capable binary."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_profiles = modelctl.PROFILES_DIR
        modelctl.PROFILES_DIR = Path(self.tmp.name) / "profiles"
        modelctl.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        self.addCleanup(setattr, modelctl, "PROFILES_DIR", self._orig_profiles)

    def _mock_hardware(self):
        return mock.MagicMock(
            gpus=[mock.MagicMock(
                device="SYCL0", name="Arc A770",
                total_bytes=16 * (1 << 30), free_bytes=14 * (1 << 30),
                reserve_bytes=0, enabled=True, role="primary",
                bandwidth_gbs=560, pci_address="", pcie_width=None,
            )],
            fingerprint="test-hw",
            ram_total_bytes=64 * (1 << 30),
            ram_available_bytes=48 * (1 << 30),
            ram_reserve_bytes=0,
            backend_fingerprints={"llama-cpp": "test-bf"},
            captured_at=0,
            storage=(),
        )

    def test_hybrid_plan_generated_for_oversized_moe(self):
        """An oversized MoE with hybrid-capable binary gets hybrid plans."""
        profile = modelctl.normalize_profile({
            "name": "test-moe",
            "model_path": "/tmp/huge-moe.gguf",
            "config": {"ctx": 8192, "flash_attn": "auto", "fit": "off",
                       "mtp": "off", "extra": ""},
            "moe_cache": {
                "mode": "manual",
                "gpu": {"budgets_bytes": {"SYCL0": 4 * (1 << 30)},
                        "policy": "slru", "admission_misses": 2},
                "decode": {"miss_execution": "cpu"},
            },
        })
        modelctl.save_profile(profile)

        # Mock: model is 80 GiB, GPU is 16 GiB -> oversized
        with mock.patch("modelctl_capabilities.probe_backend") as mock_caps, \
             mock.patch("modelctl_vram.weights_bytes_on_disk", return_value=80 << 30), \
             mock.patch("modelctl_vram.estimate_from_parts",
                       return_value={"total": 80 << 30, "kv_bytes": 2 << 30,
                                     "overhead": 2 << 30}), \
             mock.patch("modelctl_vram.file_fingerprint", return_value="abc"), \
             mock.patch("modelctl.load_defaults",
                       return_value={"vram_limit_pct": 90, "primary_gpu": "SYCL0"}), \
             mock.patch("modelctl.get_gpu_inventory", return_value=[]), \
             mock.patch("modelctl_hardware.load_settings", return_value={}):
            mock_caps.return_value = {
                "schema": 2,
                "features": {
                    "moe_weight_transfer_cache": True,
                    "moe_hybrid_cpu_miss": True,
                },
                "constraints": {},
                "cli": {},
            }
            plans = modelctl_plans.compile_launch_plans(
                profile, self._mock_hardware(), include_experimental=True)

        hybrid_plans = [p for p in plans if p.source.startswith("hybrid-")]
        self.assertTrue(hybrid_plans,
                        "no hybrid plans generated for oversized MoE")
        for p in hybrid_plans:
            self.assertTrue(p.decision_data.get("hybrid"))
            self.assertIn("hybrid", p.source)

    def test_no_hybrid_when_binary_lacks_support(self):
        """No hybrid plans when the binary doesn't report moe_hybrid_cpu_miss."""
        profile = modelctl.normalize_profile({
            "name": "test-moe",
            "model_path": "/tmp/huge-moe.gguf",
            "config": {"ctx": 8192, "flash_attn": "auto", "fit": "off",
                       "mtp": "off", "extra": ""},
            "moe_cache": {
                "mode": "manual",
                "gpu": {"budgets_bytes": {"SYCL0": 4 * (1 << 30)}},
                "decode": {"miss_execution": "cpu"},
            },
        })
        modelctl.save_profile(profile)

        with mock.patch("modelctl_capabilities.probe_backend") as mock_caps, \
             mock.patch("modelctl_vram.weights_bytes_on_disk", return_value=80 << 30), \
             mock.patch("modelctl_vram.estimate_from_parts",
                       return_value={"total": 80 << 30, "kv_bytes": 2 << 30,
                                     "overhead": 2 << 30}), \
             mock.patch("modelctl_vram.file_fingerprint", return_value="abc"), \
             mock.patch("modelctl.load_defaults",
                       return_value={"vram_limit_pct": 90}), \
             mock.patch("modelctl.get_gpu_inventory", return_value=[]), \
             mock.patch("modelctl_hardware.load_settings", return_value={}):
            mock_caps.return_value = {
                "schema": 2,
                "features": {
                    "moe_weight_transfer_cache": True,
                    "moe_hybrid_cpu_miss": False,  # no hybrid
                },
                "cli": {},
            }
            plans = modelctl_plans.compile_launch_plans(
                profile, self._mock_hardware(), include_experimental=True)

        hybrid_plans = [p for p in plans if p.source.startswith("hybrid-")]
        self.assertFalse(hybrid_plans,
                         "hybrid plans generated despite missing capability")

    def test_no_experimental_plans_without_opt_in(self):
        """Task 5.8: cache/hybrid variants stay disabled by default, even
        when the profile enables moe_cache and the binary is capable."""
        profile = modelctl.normalize_profile({
            "name": "test-moe",
            "model_path": "/tmp/huge-moe.gguf",
            "config": {"ctx": 8192, "flash_attn": "auto", "fit": "off",
                       "mtp": "off", "extra": ""},
            "moe_cache": {
                "mode": "manual",
                "gpu": {"budgets_bytes": {"SYCL0": 4 * (1 << 30)}},
                "decode": {"miss_execution": "cpu"},
            },
        })
        modelctl.save_profile(profile)

        with mock.patch("modelctl_capabilities.probe_backend") as mock_caps, \
             mock.patch("modelctl_vram.weights_bytes_on_disk", return_value=80 << 30), \
             mock.patch("modelctl_vram.estimate_from_parts",
                       return_value={"total": 80 << 30, "kv_bytes": 2 << 30,
                                     "overhead": 2 << 30}), \
             mock.patch("modelctl_vram.file_fingerprint", return_value="abc"), \
             mock.patch("modelctl.load_defaults",
                       return_value={"vram_limit_pct": 90}), \
             mock.patch("modelctl.get_gpu_inventory", return_value=[]), \
             mock.patch("modelctl_hardware.load_settings", return_value={}):
            mock_caps.return_value = {
                "schema": 2,
                "features": {
                    "moe_weight_transfer_cache": True,
                    "moe_hybrid_cpu_miss": True,
                },
                "cli": {},
            }
            plans = modelctl_plans.compile_launch_plans(
                profile, self._mock_hardware())

        experimental = [p for p in plans
                        if p.source.startswith(("hybrid-", "cache-"))]
        self.assertFalse(experimental,
                         "experimental plans generated without opt-in")


class TestReleaseBHybridPreflight(unittest.TestCase):
    """Req: preflight validates hybrid mode against capabilities."""

    def test_hybrid_ok_when_supported(self):
        profile = {
            "moe_cache": {
                "mode": "auto",
                "decode": {"miss_execution": "cpu"},
            }
        }
        caps = {
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": True,
            }
        }
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        ok_msgs = [m for m in msgs if m[0] == "ok"]
        self.assertTrue(any("hybrid" in m[1].lower() for m in ok_msgs))

    def test_hybrid_warns_when_unsupported(self):
        profile = {
            "moe_cache": {
                "mode": "auto",
                "decode": {"miss_execution": "cpu"},
            }
        }
        caps = {
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": False,
            }
        }
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        warnings = [m for m in msgs if m[0] == "warning"]
        self.assertTrue(any("hybrid" in m[1].lower() for m in warnings))

    def test_hybrid_errors_when_unsupported_manual(self):
        """§2.5: manual mode blocks with an explicit reason when hybrid is
        requested but the backend lacks moe_hybrid_cpu_miss."""
        profile = {
            "moe_cache": {
                "mode": "manual",
                "decode": {"miss_execution": "cpu"},
            }
        }
        caps = {
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": False,
            }
        }
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        errors = [m for m in msgs if m[0] == "error"]
        self.assertTrue(any("hybrid" in m[1].lower() or
                            "miss_execution" in m[1] for m in errors))


class TestReleaseBHybridMetrics(unittest.TestCase):
    """Req: hybrid metrics distinguish CPU compute from GPU hit."""

    def test_cache_metrics_has_hybrid_fields(self):
        from modelctl_services.cache_service import CacheMetrics
        m = CacheMetrics(
            lookups=100, hits=70, misses=30,
            hit_rows=700, miss_rows=300,
            cpu_miss_time_ms=150.0,
            gpu_hit_time_ms=50.0,
            merge_time_ms=10.0,
            promotion_bytes=1024 * 1024,
            h2d_bytes_avoided=10 * 1024 * 1024,
        )
        self.assertEqual(m.hit_rows, 700)
        self.assertEqual(m.miss_rows, 300)
        self.assertTrue(m.cpu_miss_time_ms > 0)
        self.assertTrue(m.gpu_hit_time_ms > 0)
        self.assertTrue(m.merge_time_ms > 0)
        self.assertTrue(m.h2d_bytes_avoided > 0)


class TestReleaseBAcceptance(unittest.TestCase):
    """Release B acceptance: hybrid plans are explainable and recoverable."""

    def test_hybrid_plan_has_decision_data(self):
        plan = modelctl_plans.LaunchPlan(
            id="hybrid-test", profile_name="m", backend="llama-cpp",
            label="hybrid-balanced", argv=(), env={},
            claim=modelctl_plans.ResourceClaim(
                vram_bytes={"SYCL0": 4 << 30}, ram_bytes=60 << 30,
                storage_mode="mmap", expected_context=8192,
                cache_bytes=4 << 30),
            estimated={}, source="hybrid-balanced", warnings=(),
            decision_data={"hybrid": True, "moe_cache": {"mode": "manual"}})
        self.assertTrue(plan.decision_data.get("hybrid"))
        self.assertIn("hybrid", plan.label)

    def test_hybrid_plan_excluded_from_storage_tier_1(self):
        """Hybrid plans use mmap, so they should be excluded when
        maximum_storage_tier < 3."""
        plan = modelctl_plans.LaunchPlan(
            id="hybrid", profile_name="m", backend="llama-cpp",
            label="hybrid", argv=(), env={},
            claim=modelctl_plans.ResourceClaim(
                vram_bytes={}, ram_bytes=60 << 30,
                storage_mode="mmap", expected_context=8192),
            estimated={}, source="hybrid", warnings=(),
            decision_data={})
        policy = modelctl_plans.RuntimePolicy(
            objective="balanced", pinned_plan_id=None,
            allow_fallback=True, allow_untested=True,
            minimum_context=None, maximum_cpu_bytes=None,
            maximum_storage_tier=1)  # GPU only
        ranked = modelctl_plans.rank_plans([plan], policy)
        self.assertEqual(len(ranked), 0)
