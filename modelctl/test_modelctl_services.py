import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_services.profile_service as profile_service
import modelctl_services.plan_service as plan_service
import modelctl_services.hardware_service as hardware_service


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
        self.assertTrue(result.plans)


class TestHardwareService(unittest.TestCase):
    def test_load_settings_returns_dict(self):
        settings = hardware_service.load_settings()
        self.assertIsInstance(settings, dict)
