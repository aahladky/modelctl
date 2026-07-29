import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl_capabilities


class TestBinaryFingerprint(unittest.TestCase):
    def test_same_content_same_fingerprint(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "bin"
            p.write_bytes(b"hello world")
            fp1 = modelctl_capabilities._binary_fingerprint(str(p))
            fp2 = modelctl_capabilities._binary_fingerprint(str(p))
            self.assertEqual(fp1, fp2)
            self.assertEqual(len(fp1), 16)

    def test_different_content_different_fingerprint(self):
        with TemporaryDirectory() as d:
            p1 = Path(d) / "bin1"
            p1.write_bytes(b"hello")
            p2 = Path(d) / "bin2"
            p2.write_bytes(b"world")
            self.assertNotEqual(
                modelctl_capabilities._binary_fingerprint(str(p1)),
                modelctl_capabilities._binary_fingerprint(str(p2)),
            )

    def test_missing_binary_uses_path(self):
        fp = modelctl_capabilities._binary_fingerprint("/nonexistent/binary")
        self.assertEqual(len(fp), 16)


class TestClassifyProbeFailure(unittest.TestCase):
    def test_returns_unsupported_status(self):
        caps = modelctl_capabilities._classify_probe_failure("/usr/bin/llama-server")
        self.assertEqual(caps["_probe_status"], "unsupported")
        self.assertEqual(caps["schema"], 0)
        self.assertEqual(caps["features"], {})


class TestProbeRaw(unittest.TestCase):
    def test_returns_none_on_nonzero_exit(self):
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text("#!/bin/sh\nexit 1\n")
            script.chmod(0o755)
            self.assertIsNone(modelctl_capabilities._probe_raw(str(script)))

    def test_returns_none_on_invalid_json(self):
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text("#!/bin/sh\necho 'not json'\n")
            script.chmod(0o755)
            self.assertIsNone(modelctl_capabilities._probe_raw(str(script)))

    def test_parses_valid_json(self):
        caps_json = json.dumps({
            "schema": 1,
            "backend": "llama.cpp",
            "build": "test",
            "features": {"moe_expert_cache": True},
            "cli": {"cache_bytes": "--moe-cache-bytes"},
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            result = modelctl_capabilities._probe_raw(str(script))
            self.assertIsNotNone(result)
            self.assertTrue(result["features"]["moe_expert_cache"])


class TestProbeBackend(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._orig_cap_dir = modelctl_capabilities.CAPABILITIES_DIR
        self.cap_dir = Path(self.tmp.name) / "caps"
        modelctl_capabilities.CAPABILITIES_DIR = self.cap_dir
        modelctl_capabilities._session_cache.clear()
        self.addCleanup(setattr, modelctl_capabilities,
                         "CAPABILITIES_DIR", self._orig_cap_dir)

    def test_unsupported_binary(self):
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text("#!/bin/sh\nexit 1\n")
            script.chmod(0o755)
            caps = modelctl_capabilities.probe_backend(str(script))
            self.assertEqual(caps["_probe_status"], "unsupported")
            self.assertFalse(modelctl_capabilities.is_cache_capable(caps))

    def test_supported_binary(self):
        caps_json = json.dumps({
            "schema": 1,
            "backend": "llama.cpp",
            "build": "b123",
            "features": {
                "moe_expert_cache": True,
                "moe_cache_sycl": True,
                "moe_hybrid_cpu_miss": True,
                "moe_cache_metrics": True,
            },
            "cli": {"cache_bytes": "--moe-cache-bytes"},
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            caps = modelctl_capabilities.probe_backend(str(script))
            self.assertEqual(caps["_probe_status"], "ok")
            self.assertTrue(modelctl_capabilities.is_cache_capable(caps))
            self.assertTrue(modelctl_capabilities.is_sycl_cache_capable(caps))
            self.assertTrue(modelctl_capabilities.supports_hybrid_miss(caps))
            self.assertTrue(modelctl_capabilities.supports_metrics(caps))
            self.assertFalse(modelctl_capabilities.supports_prefetch(caps))

    def test_session_cache_hit(self):
        caps_json = json.dumps({
            "schema": 1, "backend": "llama.cpp", "build": "x",
            "features": {"moe_expert_cache": True}, "cli": {},
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            caps1 = modelctl_capabilities.probe_backend(str(script))
            caps2 = modelctl_capabilities.probe_backend(str(script))
            self.assertIs(caps1, caps2)

    def test_disk_cache_persists(self):
        caps_json = json.dumps({
            "schema": 1, "backend": "llama.cpp", "build": "y",
            "features": {}, "cli": {},
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            modelctl_capabilities.probe_backend(str(script))
            modelctl_capabilities._session_cache.clear()
            cached = modelctl_capabilities.get_cached_capabilities(str(script))
            self.assertIsNotNone(cached)
            self.assertEqual(cached["_probe_status"], "ok")

    def test_clear_cache_removes_files(self):
        caps_json = json.dumps({
            "schema": 1, "backend": "llama.cpp", "build": "z",
            "features": {}, "cli": {},
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            modelctl_capabilities.probe_backend(str(script))
            self.assertTrue(any(self.cap_dir.glob("*.json")))
            modelctl_capabilities.clear_cache()
            self.assertFalse(any(self.cap_dir.glob("*.json")))


class TestMoeCacheArgsIntegration(unittest.TestCase):
    """Test that modelctl.build_moe_cache_args emits correct flags."""

    def test_off_mode_returns_empty(self):
        import modelctl
        profile = {"moe_cache": {"mode": "off"}}
        self.assertEqual(modelctl.build_moe_cache_args(profile), [])

    def test_no_moe_cache_returns_empty(self):
        import modelctl
        profile = {}
        self.assertEqual(modelctl.build_moe_cache_args(profile), [])

    def test_auto_mode_with_budgets(self):
        import modelctl
        profile = {
            "moe_cache": {
                "mode": "auto",
                "gpu": {
                    "budgets_bytes": {"SYCL0": 10 * (1 << 30)},
                    "policy": "slru",
                    "admission_misses": 2,
                },
                "decode": {"miss_execution": "cpu"},
                "prefill": {"admit_to_gpu_cache": False},
                "storage": {"mode": "mmap"},
            }
        }
        args = modelctl.build_moe_cache_args(profile)
        self.assertIn("--moe-cache-device", args)
        self.assertIn("--moe-cache-bytes", args)
        self.assertIn("--moe-cache-policy", args)
        self.assertIn("slru", args)
        self.assertIn("--moe-cache-miss-execution", args)
        self.assertIn("cpu", args)
        self.assertIn("--load-mode", args)
        self.assertIn("mmap", args)

    def test_capabilities_override_flag_names(self):
        import modelctl
        profile = {
            "moe_cache": {
                "mode": "manual",
                "gpu": {"budgets_bytes": {"SYCL0": 5 * (1 << 30)},
                         "policy": "lru", "admission_misses": 1},
                "decode": {"miss_execution": "cpu"},
                "prefill": {"admit_to_gpu_cache": False},
                "storage": {"mode": "mmap"},
            }
        }
        caps = {
            "cli": {
                "cache_bytes": "--custom-cache-bytes",
                "cache_policy": "--custom-policy",
                "admission_misses": "--custom-admission",
                "prefill_admission": "--custom-prefill-admission",
                "miss_execution": "--custom-miss-exec",
            }
        }
        args = modelctl.build_moe_cache_args(profile, capabilities=caps)
        self.assertIn("--custom-cache-bytes", args)
        self.assertIn("--custom-policy", args)
        self.assertIn("--custom-admission", args)
        self.assertIn("--custom-prefill-admission", args)
        self.assertIn("--custom-miss-exec", args)


class TestPreflightMoeCache(unittest.TestCase):
    def test_off_mode_no_messages(self):
        import modelctl
        profile = {"moe_cache": {"mode": "off"}}
        msgs = modelctl.preflight_moe_cache(profile)
        self.assertEqual(msgs, [])

    def test_no_capabilities_warns(self):
        import modelctl
        profile = {"moe_cache": {"mode": "auto"}}
        msgs = modelctl.preflight_moe_cache(profile, capabilities=None)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0][0], "warning")

    def test_unsupported_binary_error_for_manual(self):
        import modelctl
        profile = {"moe_cache": {"mode": "manual"}}
        caps = {"features": {}}
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        self.assertEqual(msgs[0][0], "error")

    def test_unsupported_binary_warning_for_auto(self):
        import modelctl
        profile = {"moe_cache": {"mode": "auto"}}
        caps = {"features": {}}
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        self.assertEqual(msgs[0][0], "warning")

    def test_supported_binary_ok(self):
        import modelctl
        profile = {
            "moe_cache": {
                "mode": "auto",
                "gpu": {"budgets_bytes": {"SYCL0": 10 * (1 << 30)}},
                "decode": {"miss_execution": "cpu"},
                "prefill": {"admit_to_gpu_cache": False},
                "prefetch": {"enabled": False},
            }
        }
        caps = {
            "features": {
                "moe_expert_cache": True,
                "moe_cache_sycl": True,
                "moe_hybrid_cpu_miss": True,
            }
        }
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        levels = [m[0] for m in msgs]
        self.assertIn("ok", levels)
        self.assertNotIn("error", levels)

    def test_prefetch_unsupported_warns(self):
        import modelctl
        profile = {
            "moe_cache": {
                "mode": "auto",
                "decode": {"miss_execution": "cpu"},
                "prefetch": {"enabled": True},
            }
        }
        caps = {
            "features": {
                "moe_expert_cache": True,
                "moe_hybrid_cpu_miss": True,
                "moe_cache_prefetch": False,
            }
        }
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        warnings = [m for m in msgs if m[0] == "warning"]
        self.assertTrue(any("prefetch" in m[1] for m in warnings))


class TestNormalizeMoeCache(unittest.TestCase):
    def test_old_profile_gets_moe_cache(self):
        import modelctl
        profile = {"name": "test", "config": {}}
        normalized = modelctl.normalize_profile(profile)
        self.assertIn("moe_cache", normalized)
        self.assertEqual(normalized["moe_cache"]["mode"], "off")
        self.assertIn("gpu", normalized["moe_cache"])
        self.assertIn("ram", normalized["moe_cache"])
        self.assertIn("storage", normalized["moe_cache"])
        self.assertIn("prefill", normalized["moe_cache"])
        self.assertIn("decode", normalized["moe_cache"])
        self.assertIn("prefetch", normalized["moe_cache"])

    def test_existing_moe_cache_preserved(self):
        import modelctl
        profile = {
            "name": "test",
            "config": {},
            "moe_cache": {
                "mode": "manual",
                "gpu": {"budgets_bytes": {"SYCL0": 999}},
            }
        }
        normalized = modelctl.normalize_profile(profile)
        self.assertEqual(normalized["moe_cache"]["mode"], "manual")
        self.assertEqual(normalized["moe_cache"]["gpu"]["budgets_bytes"]["SYCL0"], 999)
        # Missing sub-sections are filled.
        self.assertIn("ram", normalized["moe_cache"])
        self.assertIn("storage", normalized["moe_cache"])

    def test_partial_moe_cache_filled(self):
        import modelctl
        profile = {
            "name": "test",
            "config": {},
            "moe_cache": {"mode": "auto", "gpu": {"policy": "lru"}},
        }
        normalized = modelctl.normalize_profile(profile)
        self.assertEqual(normalized["moe_cache"]["gpu"]["policy"], "lru")
        self.assertEqual(normalized["moe_cache"]["gpu"]["admission_misses"], 2)
        self.assertIn("decode", normalized["moe_cache"])

    def test_idempotent(self):
        import modelctl
        profile = {"name": "test", "config": {}}
        n1 = modelctl.normalize_profile(profile)
        n2 = modelctl.normalize_profile(n1)
        self.assertEqual(n1["moe_cache"], n2["moe_cache"])


class TestBuildServerArgsMoeCache(unittest.TestCase):
    """build_server_args should include moe_cache flags when configured."""

    def test_moe_cache_flags_included(self):
        import modelctl
        profile = {
            "model_path": "/m.gguf",
            "config": {
                "flash_attn": "on", "ctx": 32768,
                "cache_type_k": "q8_0", "cache_type_v": "q4_0",
                "fit": "off", "extra": "",
            },
            "moe_cache": {
                "mode": "manual",
                "gpu": {
                    "budgets_bytes": {"SYCL0": 10 * (1 << 30)},
                    "policy": "slru",
                    "admission_misses": 2,
                },
                "decode": {"miss_execution": "cpu"},
                "prefill": {"admit_to_gpu_cache": False},
                "storage": {"mode": "mmap"},
            },
        }
        args = modelctl.build_server_args(profile)
        self.assertIn("--moe-cache-bytes", args)
        self.assertIn("--moe-cache-policy", args)

    def test_moe_cache_off_no_flags(self):
        import modelctl
        profile = {
            "model_path": "/m.gguf",
            "config": {
                "flash_attn": "on", "ctx": 32768,
                "cache_type_k": "q8_0", "cache_type_v": "q4_0",
                "fit": "off", "extra": "",
            },
            "moe_cache": {"mode": "off"},
        }
        args = modelctl.build_server_args(profile)
        self.assertNotIn("--moe-cache-bytes", args)
