import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_hardware
import modelctl_plans
import modelctl_services.profile_service as profile_service
import modelctl_services.plan_service as plan_service
import modelctl_services.hardware_service as hardware_service
import modelctl_services.runtime_service as runtime_service


class TestProfileService(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_profiles = modelctl.PROFILES_DIR
        self._orig_models = modelctl.DEFAULT_MODELS_DIR
        self.profiles_dir = Path(self.tmp.name) / "profiles"
        modelctl.PROFILES_DIR = self.profiles_dir
        modelctl.DEFAULT_MODELS_DIR = Path(self.tmp.name) / "models"
        self.addCleanup(setattr, modelctl, "PROFILES_DIR", self._orig_profiles)
        self.addCleanup(setattr, modelctl, "DEFAULT_MODELS_DIR", self._orig_models)

    def test_save_valid_profile(self):
        profile = {
            "name": "test-model",
            "config": {"ctx": 4096, "device": "SYCL0"},
        }
        with mock.patch("modelctl.generate_artifacts"), \
             mock.patch("modelctl.sync_all_backends"):
            result = profile_service.save_profile(profile)
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.profile)
        self.assertEqual(result.profile["name"], "test-model")

    def test_save_invalid_profile_fails(self):
        profile = {"name": "", "config": {"ctx": 0}}
        result = profile_service.save_profile(profile)
        self.assertFalse(result.ok)
        self.assertTrue(result.messages)

    def test_update_config(self):
        profile = modelctl.normalize_profile({
            "name": "test-model",
            "model_path": "/tmp/test.gguf",
            "config": {"ctx": 4096},
        })
        modelctl.save_profile(profile)

        with mock.patch("modelctl.generate_artifacts"), \
             mock.patch("modelctl.sync_all_backends"):
            result = profile_service.update_config("test-model", {"ctx": "8192"})
        self.assertTrue(result.ok)

    def test_update_config_logs_through_ctx(self):
        profile = modelctl.normalize_profile({
            "name": "test-model",
            "model_path": "/tmp/test.gguf",
            "config": {"ctx": 4096},
        })
        modelctl.save_profile(profile)
        ctx = _FakeCtx()

        with mock.patch("modelctl.generate_artifacts"), \
             mock.patch("modelctl.sync_all_backends"):
            result = profile_service.update_config("test-model", {"ctx": "8192"}, ctx=ctx)
        self.assertTrue(result.ok)
        self.assertTrue(any("applying updates" in line for line in ctx.lines))

    def test_list_profiles_empty(self):
        profiles = profile_service.list_profiles()
        self.assertEqual(profiles, [])

    def test_list_profiles_with_profiles(self):
        modelctl.save_profile({"name": "alpha", "config": {"ctx": 4096}})
        modelctl.save_profile({"name": "beta", "config": {"ctx": 8192}})
        profiles = profile_service.list_profiles()
        names = [p["name"] for p in profiles]
        self.assertIn("alpha", names)
        self.assertIn("beta", names)


class TestPlanService(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_profiles = modelctl.PROFILES_DIR
        modelctl.PROFILES_DIR = Path(self.tmp.name) / "profiles"
        self.addCleanup(setattr, modelctl, "PROFILES_DIR", self._orig_profiles)

    def test_compile_plans_nonexistent_profile(self):
        result = plan_service.compile_plans("nonexistent")
        self.assertFalse(result.ok)

    def test_compile_plans_valid(self):
        profile = modelctl.normalize_profile({
            "name": "test",
            "model_path": "/tmp/test.gguf",
            "config": {"ctx": 4096, "flash_attn": "auto", "fit": "on",
                       "mtp": "off", "extra": ""},
        })
        modelctl.save_profile(profile)
        with mock.patch("modelctl_hardware.capture_hardware_snapshot") as mock_snap, \
             mock.patch("modelctl_vram.file_fingerprint", return_value="abc"):
            mock_snap.return_value = mock.MagicMock(
                gpus=[], fingerprint="test",
                ram_total_bytes=16 * (1 << 30),
                ram_available_bytes=8 * (1 << 30),
                ram_reserve_bytes=0,
                backend_fingerprints={},
                captured_at=0,
            )
            result = plan_service.compile_plans("test")
        self.assertTrue(result.ok)

    def test_apply_plan_writes_config_and_syncs(self):
        profile = modelctl.normalize_profile({
            "name": "test",
            "model_path": "/tmp/test.gguf",
            "config": {"ctx": 4096, "device": "SYCL0", "flash_attn": "auto",
                       "fit": "on", "mtp": "off", "extra": ""},
        })
        modelctl.save_profile(profile)
        snap = mock.MagicMock(
            gpus=[], fingerprint="test",
            ram_total_bytes=16 * (1 << 30), ram_available_bytes=8 * (1 << 30),
            ram_reserve_bytes=0, backend_fingerprints={}, captured_at=0,
        )
        with mock.patch("modelctl_hardware.capture_hardware_snapshot", return_value=snap), \
             mock.patch("modelctl_vram.file_fingerprint", return_value="abc"):
            plans = modelctl_plans.compile_launch_plans(profile, snap)
            target = plans[0]
            with mock.patch("modelctl.generate_artifacts"), \
                 mock.patch("modelctl.sync_all_backends") as mock_sync:
                result = plan_service.apply_plan("test", target.id)
        self.assertTrue(result.ok, result.messages)
        mock_sync.assert_called_once()
        saved = modelctl.load_profile("test")
        self.assertEqual(saved["config"], target.config)

    def test_apply_plan_unknown_id_fails(self):
        profile = modelctl.normalize_profile({
            "name": "test", "model_path": "/tmp/test.gguf",
            "config": {"ctx": 4096, "flash_attn": "auto", "fit": "on",
                       "mtp": "off", "extra": ""},
        })
        modelctl.save_profile(profile)
        with mock.patch("modelctl_hardware.capture_hardware_snapshot") as mock_snap:
            mock_snap.return_value = mock.MagicMock(
                gpus=[], fingerprint="test", ram_total_bytes=16 << 30,
                ram_available_bytes=8 << 30, ram_reserve_bytes=0,
                backend_fingerprints={}, captured_at=0,
            )
            result = plan_service.apply_plan("test", "nonexistent-plan-id")
        self.assertFalse(result.ok)


class TestPlanServiceSelection(unittest.TestCase):
    """select_plan/disable_plan must never leave a plan both pinned and
    disabled -- rank_plans would then never actually select it."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_profiles = modelctl.PROFILES_DIR
        modelctl.PROFILES_DIR = Path(self.tmp.name) / "profiles"
        self.addCleanup(setattr, modelctl, "PROFILES_DIR", self._orig_profiles)
        modelctl.save_profile({"name": "m1", "config": {"ctx": 4096}})

    def test_selecting_a_disabled_plan_undisables_it(self):
        with mock.patch("modelctl.generate_artifacts"), \
             mock.patch("modelctl.sync_all_backends"):
            plan_service.disable_plan("m1", "plan-a")
            result = plan_service.select_plan("m1", "plan-a")
        self.assertTrue(result.ok)
        profile = modelctl.load_profile("m1")
        self.assertEqual(profile["runtime"]["pinned_plan_id"], "plan-a")
        self.assertNotIn("plan-a", profile["runtime"]["disabled_plan_ids"])

    def test_disabling_the_pinned_plan_unpins_it(self):
        with mock.patch("modelctl.generate_artifacts"), \
             mock.patch("modelctl.sync_all_backends"):
            plan_service.select_plan("m1", "plan-a")
            result = plan_service.disable_plan("m1", "plan-a")
        self.assertTrue(result.ok)
        profile = modelctl.load_profile("m1")
        self.assertIsNone(profile["runtime"]["pinned_plan_id"])
        self.assertIn("plan-a", profile["runtime"]["disabled_plan_ids"])

    def test_disabling_a_different_plan_leaves_pin_intact(self):
        with mock.patch("modelctl.generate_artifacts"), \
             mock.patch("modelctl.sync_all_backends"):
            plan_service.select_plan("m1", "plan-a")
            result = plan_service.disable_plan("m1", "plan-b")
        self.assertTrue(result.ok)
        profile = modelctl.load_profile("m1")
        self.assertEqual(profile["runtime"]["pinned_plan_id"], "plan-a")
        self.assertIn("plan-b", profile["runtime"]["disabled_plan_ids"])

    def test_enable_on_profile_with_no_runtime_policy_is_a_noop(self):
        # No prior managed runtime policy -- enabling a plan that was
        # never disabled must not fabricate a fresh managed section.
        with mock.patch("modelctl.generate_artifacts"), \
             mock.patch("modelctl.sync_all_backends"):
            result = plan_service.enable_plan("m1", "plan-a")
        self.assertTrue(result.ok)
        profile = modelctl.load_profile("m1")
        self.assertNotIn("runtime", profile)

    def test_enable_clears_a_disabled_plan(self):
        with mock.patch("modelctl.generate_artifacts"), \
             mock.patch("modelctl.sync_all_backends"):
            plan_service.disable_plan("m1", "plan-a")
            result = plan_service.enable_plan("m1", "plan-a")
        self.assertTrue(result.ok)
        profile = modelctl.load_profile("m1")
        self.assertNotIn("plan-a", profile["runtime"]["disabled_plan_ids"])


class TestHardwareService(unittest.TestCase):
    def test_load_settings_returns_dict(self):
        settings = hardware_service.load_settings()
        self.assertIsInstance(settings, dict)


class TestHardwareServiceCalibration(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_models = modelctl.DEFAULT_MODELS_DIR
        modelctl.DEFAULT_MODELS_DIR = Path(self.tmp.name) / "models"
        self.addCleanup(setattr, modelctl, "DEFAULT_MODELS_DIR", self._orig_models)

    def test_pick_calibration_file_finds_largest_gguf(self):
        modelctl.DEFAULT_MODELS_DIR.mkdir(parents=True)
        small = modelctl.DEFAULT_MODELS_DIR / "small.gguf"
        small.write_bytes(b"\x00" * 100)
        sub = modelctl.DEFAULT_MODELS_DIR / "sub"
        sub.mkdir()
        big = sub / "big.gguf"
        big.write_bytes(b"\x00" * 1000)
        self.assertEqual(hardware_service.pick_calibration_file(), str(big))

    def test_pick_calibration_file_none_when_no_models_dir(self):
        self.assertIsNone(hardware_service.pick_calibration_file())

    def test_pick_calibration_file_none_when_no_gguf(self):
        modelctl.DEFAULT_MODELS_DIR.mkdir(parents=True)
        (modelctl.DEFAULT_MODELS_DIR / "readme.txt").write_text("x")
        self.assertIsNone(hardware_service.pick_calibration_file())

    def test_calibrate_storage_persists_measurement(self):
        modelctl.DEFAULT_MODELS_DIR.mkdir(parents=True)
        model_file = modelctl.DEFAULT_MODELS_DIR / "m.gguf"
        model_file.write_bytes(b"\x00" * (2 * 1024 * 1024))

        settings_path = Path(self.tmp.name) / "hardware.json"
        with mock.patch("modelctl_hardware.SETTINGS_PATH", settings_path):
            result = hardware_service.calibrate_storage(str(model_file))
            self.assertTrue(result.ok, result.messages)
            saved = modelctl_hardware.load_settings()

        import modelctl_storage
        mount = modelctl_storage.probe_storage(str(model_file)).mount_point
        entry = saved["storage"][mount]
        self.assertGreater(entry["measured_sequential_read_bps"], 0)
        self.assertIn("measured_at", entry)

        # And the hardware snapshot picks it up.
        with mock.patch("modelctl_hardware.SETTINGS_PATH", settings_path), \
             mock.patch.object(modelctl_hardware, "probe_gpus", return_value=[]):
            snap = modelctl_hardware.capture_hardware_snapshot()
        matching = [s for s in snap.storage if s.mount_point == mount]
        self.assertTrue(matching)
        self.assertGreater(matching[0].measured_sequential_read_bps, 0)
        self.assertIsNotNone(matching[0].measurement_age_seconds)


class _FakeCtx:
    """Minimal stand-in for modelctl_web.jobs.JobContext."""
    def __init__(self, cancelled=False):
        self.lines = []
        self._cancelled = cancelled

    def log(self, line):
        self.lines.append(line)

    def raise_if_cancelled(self):
        if self._cancelled:
            from modelctl_web.jobs import CancelledError
            raise CancelledError("job cancelled")


class TestRuntimeService(unittest.TestCase):
    """The one implementation behind both `modelctl router load/unload`
    (no ctx) and the web console's runtime page (modelctl_web/mutate.py,
    ctx=a JobContext)."""

    def test_load_model_success(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.warm_load",
                        return_value={"loaded": True, "elapsed_s": 4.2, "response_ok": True}):
            result = runtime_service.load_model("qwen")
        self.assertTrue(result.ok)
        self.assertTrue(result.loaded)
        self.assertEqual(result.elapsed_s, 4.2)

    def test_load_model_not_ready(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.warm_load",
                        return_value={"loaded": False, "elapsed_s": 300.0, "response_ok": False}):
            result = runtime_service.load_model("qwen")
        self.assertFalse(result.ok)

    def test_load_model_swap_error(self):
        from modelctl_web.swap import ModelctlSwapError
        with mock.patch("modelctl_web.swap.LlamaSwapClient.warm_load",
                        side_effect=ModelctlSwapError("LLAMA_SWAP_UNAVAILABLE", "down")):
            result = runtime_service.load_model("qwen")
        self.assertFalse(result.ok)
        self.assertIn("down", result.messages[0])

    def test_load_model_logs_and_checks_cancellation_with_ctx(self):
        ctx = _FakeCtx()
        with mock.patch("modelctl_web.swap.LlamaSwapClient.warm_load",
                        return_value={"loaded": True, "elapsed_s": 1.0, "response_ok": True}):
            runtime_service.load_model("qwen", ctx=ctx)
        self.assertTrue(any("warm-loading" in line for line in ctx.lines))
        self.assertTrue(any("loaded=True" in line for line in ctx.lines))

    def test_load_model_propagates_cancellation(self):
        from modelctl_web.jobs import CancelledError
        ctx = _FakeCtx(cancelled=True)
        with self.assertRaises(CancelledError):
            runtime_service.load_model("qwen", ctx=ctx)

    def test_unload_model_success(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.unload",
                        return_value={}) as mock_unload:
            result = runtime_service.unload_model("qwen")
        self.assertTrue(result.ok)
        mock_unload.assert_called_once_with("qwen")

    def test_unload_model_swap_error(self):
        from modelctl_web.swap import ModelctlSwapError
        with mock.patch("modelctl_web.swap.LlamaSwapClient.unload",
                        side_effect=ModelctlSwapError("LLAMA_SWAP_UNAVAILABLE", "down")):
            result = runtime_service.unload_model("qwen")
        self.assertFalse(result.ok)

    def test_restart_model_unloads_running_then_loads(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.running_model_ids",
                        return_value={"qwen"}), \
             mock.patch("modelctl_web.swap.LlamaSwapClient.unload") as mock_unload, \
             mock.patch("modelctl_web.swap.LlamaSwapClient.warm_load",
                        return_value={"loaded": True, "elapsed_s": 2.0, "response_ok": True}):
            result = runtime_service.restart_model("qwen")
        self.assertTrue(result.ok)
        mock_unload.assert_called_once_with("qwen")

    def test_restart_model_skips_unload_when_not_running(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.running_model_ids",
                        return_value=set()), \
             mock.patch("modelctl_web.swap.LlamaSwapClient.unload") as mock_unload, \
             mock.patch("modelctl_web.swap.LlamaSwapClient.warm_load",
                        return_value={"loaded": True, "elapsed_s": 2.0, "response_ok": True}):
            result = runtime_service.restart_model("qwen")
        self.assertTrue(result.ok)
        mock_unload.assert_not_called()

    def test_unload_all(self):
        with mock.patch("modelctl_web.swap.LlamaSwapClient.running_model_ids",
                        return_value={"a", "b"}), \
             mock.patch("modelctl_web.swap.LlamaSwapClient.unload_all") as mock_unload_all:
            result = runtime_service.unload_all()
        self.assertTrue(result.ok)
        self.assertEqual(result.models, ["a", "b"])
        mock_unload_all.assert_called_once()
