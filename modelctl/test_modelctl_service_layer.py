"""Task C5: one service result contract, and the new services.

The contract test is the important one: it asserts that every service
result type really is a ServiceResult, so a caller can handle any of them
without knowing which service produced it.
"""
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
from modelctl_services import ServiceResult, ROLLBACK_DONE, ROLLBACK_FAILED
from modelctl_services import (acquisition_service, benchmark_service,
                               hardware_service, plan_service,
                               profile_service, routing_service,
                               runtime_service, settings_service)


class TestResultContract(unittest.TestCase):
    def test_every_domain_result_is_a_service_result(self):
        for cls in (profile_service.ProfileResult, plan_service.PlanResult,
                    hardware_service.HardwareResult,
                    runtime_service.RuntimeResult):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, ServiceResult))

    def test_every_result_carries_the_full_envelope(self):
        for cls in (ServiceResult, profile_service.ProfileResult,
                    plan_service.PlanResult, hardware_service.HardwareResult,
                    runtime_service.RuntimeResult):
            r = cls(ok=True)
            with self.subTest(cls=cls.__name__):
                for field in ("ok", "messages", "warnings", "data",
                              "changed_resources", "job_id", "rollback_status"):
                    self.assertTrue(hasattr(r, field), field)

    def test_changed_resources_are_deduplicated(self):
        # An operation that saves a profile twice changed one profile.
        r = ServiceResult().changed("profile:m1", "profile:m1", "llama-swap:config")
        self.assertEqual(r.changed_resources, ["profile:m1", "llama-swap:config"])

    def test_failure_helper_marks_not_ok(self):
        r = ServiceResult.failure("nope")
        self.assertFalse(r.ok)
        self.assertEqual(r.messages, ["nope"])
        self.assertEqual(r.changed_resources, [])

    def test_as_dict_is_json_serialisable(self):
        r = ServiceResult(data={"n": 1}).changed("x:y").add_warning("w")
        json.dumps(r.as_dict())
        self.assertEqual(r.as_dict()["changed_resources"], ["x:y"])


class TestGgufVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        profiles = self.root / "profiles"
        profiles.mkdir()
        p = mock.patch.object(modelctl, "PROFILES_DIR", profiles)
        p.start()
        self.addCleanup(p.stop)
        self.profiles = profiles

    def gguf(self, name="model.gguf", size=4096):
        f = self.root / name
        f.write_bytes(b"GGUF" + b"\0" * (size - 4))
        return f

    def test_valid_gguf_passes(self):
        r = acquisition_service.verify_local_gguf(self.gguf())
        self.assertTrue(r.ok)
        self.assertEqual(r.data["size_bytes"], 4096)

    def test_missing_file_is_rejected(self):
        r = acquisition_service.verify_local_gguf(self.root / "nope.gguf")
        self.assertFalse(r.ok)
        self.assertIn("no such file", r.messages[0])

    def test_non_gguf_is_rejected(self):
        f = self.root / "notes.txt"
        f.write_bytes(b"hello" * 500)
        r = acquisition_service.verify_local_gguf(f)
        self.assertFalse(r.ok)
        self.assertIn("not a GGUF", r.messages[0])

    def test_truncated_file_is_rejected(self):
        f = self.root / "tiny.gguf"
        f.write_bytes(b"GGUF")
        r = acquisition_service.verify_local_gguf(f)
        self.assertFalse(r.ok)
        self.assertIn("truncated", r.messages[0])

    def test_incomplete_shard_set_is_rejected(self):
        # Importing shard 1 of 3 produces a profile that fails at load time
        # with a far more confusing error than this one.
        self.gguf("big-00001-of-00003.gguf")
        r = acquisition_service.verify_local_gguf(
            self.root / "big-00001-of-00003.gguf")
        self.assertFalse(r.ok)
        self.assertIn("shard", r.messages[0])
        self.assertEqual(len(r.data["missing_shards"]), 2)

    def test_complete_shard_set_passes(self):
        for i in (1, 2, 3):
            self.gguf(f"big-{i:05d}-of-00003.gguf")
        r = acquisition_service.verify_local_gguf(
            self.root / "big-00001-of-00003.gguf")
        self.assertTrue(r.ok, r.messages)

    def test_duplicate_import_warns_but_does_not_block(self):
        f = self.gguf()
        (self.profiles / "existing.json").write_text(json.dumps(
            {"name": "existing", "model_path": str(f)}))
        r = acquisition_service.verify_local_gguf(f)
        self.assertTrue(r.ok)
        self.assertEqual(r.data["duplicate_profile"], "existing")
        self.assertTrue(any("already points at" in w for w in r.warnings))


class TestDestinationSpace(unittest.TestCase):
    def test_insufficient_space_refuses_before_download(self):
        with TemporaryDirectory() as d:
            usage = mock.Mock(free=1 << 30, total=2 << 30, used=1 << 30)
            with mock.patch("shutil.disk_usage", return_value=usage):
                r = acquisition_service.check_destination_space(
                    10 << 30, destination=d)
        self.assertFalse(r.ok)
        self.assertIn("free", r.messages[0])

    def test_missing_destination_refuses(self):
        r = acquisition_service.check_destination_space(
            1 << 30, destination="/nonexistent/place")
        self.assertFalse(r.ok)

    def test_ample_space_passes(self):
        with TemporaryDirectory() as d:
            usage = mock.Mock(free=500 << 30, total=1000 << 30, used=500 << 30)
            with mock.patch("shutil.disk_usage", return_value=usage):
                r = acquisition_service.check_destination_space(
                    10 << 30, destination=d)
        self.assertTrue(r.ok)


class TestSettingsService(unittest.TestCase):
    def setUp(self):
        self.saved = {}
        for p in (mock.patch.object(modelctl, "load_defaults",
                                    return_value={"ctx": 8192, "ttl": 3600}),
                  mock.patch.object(modelctl, "save_defaults",
                                    side_effect=self.saved.update)):
            p.start()
            self.addCleanup(p.stop)

    def test_valid_values_are_applied(self):
        r = settings_service.update_defaults({"ctx": "16384", "device": "SYCL1"})
        self.assertTrue(r.ok)
        self.assertEqual(r.data["applied"]["ctx"], 16384)
        self.assertEqual(r.data["applied"]["device"], "SYCL1")
        self.assertIn("settings:defaults", r.changed_resources)

    def test_non_numeric_value_is_reported_not_swallowed(self):
        # The route used to `except ValueError: pass`, so a mistyped
        # context length silently kept the old value.
        r = settings_service.update_defaults({"ctx": "banana"})
        self.assertTrue(any("not a number" in w for w in r.warnings))
        self.assertNotIn("ctx", r.data.get("applied", {}))

    def test_out_of_range_value_is_rejected(self):
        r = settings_service.update_defaults({"vram_limit_pct": "500"})
        self.assertTrue(any("outside" in w for w in r.warnings))

    def test_unknown_key_is_ignored_with_a_warning(self):
        r = settings_service.update_defaults({"nonsense": "1"})
        self.assertTrue(any("not a settable default" in w for w in r.warnings))


class TestRoutingRollback(unittest.TestCase):
    """apply_matrix must always say what state the config was left in."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Path(self.tmp.name) / "config.yaml"
        self.cfg.write_text("models: {}\n")
        p = mock.patch.object(modelctl, "LLAMA_SWAP_CONFIG_PATH", self.cfg)
        p.start()
        self.addCleanup(p.stop)
        for target, value in (("generate_matrix", {"sets": []}),
                              ("merge_matrix", {"sets": [{"name": "a"}]})):
            import modelctl_matrix
            q = mock.patch.object(modelctl_matrix, target, return_value=value)
            q.start()
            self.addCleanup(q.stop)

    def test_successful_apply_reports_no_rollback(self):
        with mock.patch.object(routing_service, "_restart_swap",
                               return_value=mock.Mock(returncode=0, stdout="", stderr="")), \
             mock.patch.object(routing_service, "_wait_healthy", return_value=True):
            r = routing_service.apply_matrix()
        self.assertTrue(r.ok)
        self.assertEqual(r.rollback_status, "")
        self.assertIn("llama-swap:config", r.changed_resources)

    def test_unhealthy_after_apply_rolls_back(self):
        with mock.patch.object(routing_service, "_restart_swap",
                               return_value=mock.Mock(returncode=0, stdout="", stderr="")), \
             mock.patch.object(routing_service, "_wait_healthy", return_value=False):
            r = routing_service.apply_matrix()
        self.assertFalse(r.ok)
        self.assertEqual(r.rollback_status, ROLLBACK_DONE)
        self.assertEqual(self.cfg.read_text(), "models: {}\n")

    def test_failed_rollback_is_reported_distinctly(self):
        # "The apply failed" and "the apply failed and we could not put the
        # old config back" need different responses from whoever is looking.
        with mock.patch.object(routing_service, "_restart_swap",
                               return_value=mock.Mock(returncode=1, stdout="", stderr="boom")), \
             mock.patch("shutil.copy2", side_effect=[None, OSError("disk gone")]):
            r = routing_service.apply_matrix()
        self.assertFalse(r.ok)
        self.assertEqual(r.rollback_status, ROLLBACK_FAILED)
        self.assertTrue(any("ROLLBACK FAILED" in m for m in r.messages))


class TestBenchmarkService(unittest.TestCase):
    def test_modes_are_described(self):
        r = benchmark_service.describe_modes()
        self.assertTrue(r.data["modes"])
        for m in r.data["modes"]:
            self.assertIn("value", m)
            self.assertIn("description", m)

    def test_unknown_mode_is_rejected(self):
        r = benchmark_service.validate_mode("teleportation")
        self.assertFalse(r.ok)

    def test_unmet_preconditions_change_the_recorded_label(self):
        # A mode whose preconditions were not met must not be recorded
        # under that mode's name, or ranking later compares a
        # "page-cache-warm" number against one that was nothing of the kind.
        import modelctl_benchmark
        mode = modelctl_benchmark.BenchmarkMode.STORAGE_COLD
        r = benchmark_service.validate_mode(mode)
        if r.warnings:
            self.assertNotEqual(r.data["label"], mode.value)


if __name__ == "__main__":
    unittest.main()
