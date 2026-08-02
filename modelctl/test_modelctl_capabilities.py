import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl_capabilities


class TestVersionStringProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        import modelctl
        self.modelctl = modelctl
        self._old_cache = getattr(modelctl, "_PROBE_ENV_CACHE", None)
        modelctl._PROBE_ENV_CACHE = None
        self.addCleanup(setattr, modelctl, "_PROBE_ENV_CACHE", self._old_cache)

    def _fake_binary(self, script_body):
        path = Path(self.tmp.name) / "llama-server"
        path.write_text("#!/bin/sh\n" + script_body)
        path.chmod(0o755)
        return str(path)

    def test_crash_text_never_becomes_version(self):
        # A SYCL binary run without the oneAPI env aborts printing an
        # uncaught-exception banner; that text must not be stamped into the
        # capability cache as a version identity.
        bin_path = self._fake_binary(
            "echo \"terminate called after throwing an instance of "
            "'sycl::_V1::exception'\" >&2\n"
            "exit 134\n")
        with mock.patch.object(self.modelctl, "find_env_script_candidates",
                               return_value=[]):
            self.assertEqual(
                modelctl_capabilities._version_string(bin_path), "")

    def test_env_fallback_recovers_version(self):
        bin_path = self._fake_binary(
            'if [ -n "$MODELCTL_TEST_PROBE_OK" ]; then\n'
            '  echo "version: b6100 (test)"\n'
            '  exit 0\n'
            'fi\n'
            'exit 134\n')
        with mock.patch.object(self.modelctl, "find_env_script_candidates",
                               return_value=["/fake/env.sh"]), \
             mock.patch.object(self.modelctl, "source_env_script",
                               return_value={"MODELCTL_TEST_PROBE_OK": "1"}):
            self.assertEqual(
                modelctl_capabilities._version_string(bin_path),
                "version: b6100 (test)")


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
        self.assertEqual(caps["schema"],
                         modelctl_capabilities.CAPABILITY_SCHEMA_VERSION)
        self.assertFalse(caps["features"]["moe_weight_transfer_cache"])


class TestProbeRaw(unittest.TestCase):
    def test_silent_nonzero_exit_is_error_not_unsupported(self):
        # No rejection message on stderr: indistinguishable from a crash,
        # so it must classify "error" (transient, never persisted) -- a
        # cached "unsupported" from one bad run silently stripped every
        # fork feature until the binary was rebuilt.
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text("#!/bin/sh\nexit 1\n")
            script.chmod(0o755)
            verdict, raw = modelctl_capabilities._probe_raw(str(script))
        self.assertEqual(verdict, "error")
        self.assertIsNone(raw)

    def test_flag_rejection_is_unsupported(self):
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(
                "#!/bin/sh\necho \"error: invalid argument: $1\" >&2\nexit 1\n")
            script.chmod(0o755)
            verdict, raw = modelctl_capabilities._probe_raw(str(script))
        self.assertEqual(verdict, "rejected")

    def test_garbage_output_is_error(self):
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text("#!/bin/sh\necho 'not json'\n")
            script.chmod(0o755)
            verdict, raw = modelctl_capabilities._probe_raw(str(script))
        self.assertEqual(verdict, "error")

    def test_non_object_json_is_error(self):
        # Valid JSON that is not an object used to AttributeError out of
        # every launch/preview path instead of failing closed.
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text("#!/bin/sh\necho '[]'\n")
            script.chmod(0o755)
            verdict, raw = modelctl_capabilities._probe_raw(str(script))
        self.assertEqual(verdict, "error")

    def test_parses_valid_json(self):
        caps_json = json.dumps({
            "schema": 2,
            "backend": "llama.cpp",
            "build": "test",
            "features": {"moe_weight_transfer_cache": True},
            "cli": {"moe_cache_bytes": "--moe-cache-bytes"},
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            verdict, raw = modelctl_capabilities._probe_raw(str(script))
            self.assertEqual(verdict, "ok")
            self.assertTrue(raw["features"]["moe_weight_transfer_cache"])

    def test_retries_with_env_script_after_bare_failure(self):
        # SYCL binaries crash without their oneAPI env even for the probe;
        # _probe_raw must retry with the launch path's env scripts.
        caps_json = json.dumps({"schema": 2,
                                "features": {"moe_weight_transfer_cache": True}})
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(
                "#!/bin/sh\n"
                '[ -n "$PROBE_TEST_MARKER" ] || exit 1\n'
                f"echo '{caps_json}'\n")
            script.chmod(0o755)
            import modelctl
            with mock.patch.object(modelctl, "find_env_script_candidates",
                                   return_value=["/fake/env.sh"]), \
                 mock.patch.object(modelctl, "source_env_script",
                                   return_value={"PROBE_TEST_MARKER": "1"}):
                verdict, raw = modelctl_capabilities._probe_raw(str(script))
            self.assertEqual(verdict, "ok")
            self.assertTrue(raw["features"]["moe_weight_transfer_cache"])

    def test_env_fallback_not_used_when_bare_probe_works(self):
        caps_json = json.dumps({"schema": 2, "features": {}})
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            import modelctl
            with mock.patch.object(
                    modelctl, "find_env_script_candidates",
                    side_effect=AssertionError("should not be called")):
                verdict, _ = modelctl_capabilities._probe_raw(str(script))
            self.assertEqual(verdict, "ok")


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
            script.write_text(
                "#!/bin/sh\necho \"error: invalid argument: $1\" >&2\nexit 1\n")
            script.chmod(0o755)
            caps = modelctl_capabilities.probe_backend(str(script))
            self.assertEqual(caps["_probe_status"], "unsupported")
            self.assertFalse(modelctl_capabilities.is_cache_capable(caps))

    def test_supported_binary(self):
        caps_json = json.dumps({
            "schema": 2,
            "backend": "llama.cpp",
            "build": "b123",
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": False,
                "moe_cache_metrics": True,
                "moe_cache_prefill_policy": True,
                "moe_cache_reset": True,
            },
            "cli": {
                "moe_cache_bytes": "--moe-cache-bytes",
                "moe_cache_policy": "--moe-cache-policy",
                "moe_cache_admission": "--moe-cache-admission-misses",
                "moe_cache_prefill": "--moe-cache-prefill-admission",
            },
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            caps = modelctl_capabilities.probe_backend(str(script))
            self.assertEqual(caps["_probe_status"], "ok")
            self.assertTrue(modelctl_capabilities.is_cache_capable(caps))
            self.assertTrue(modelctl_capabilities.is_weight_transfer_cache_capable(caps))
            self.assertTrue(modelctl_capabilities.is_sycl_cache_capable(caps))
            self.assertFalse(modelctl_capabilities.supports_hybrid_miss(caps))
            self.assertTrue(modelctl_capabilities.supports_metrics(caps))
            self.assertFalse(modelctl_capabilities.supports_prefetch(caps))

    def test_supported_binary_with_hybrid(self):
        caps_json = json.dumps({
            "schema": 2,
            "backend": "llama.cpp",
            "build": "b123",
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": True,
                "moe_cache_metrics": True,
            },
            "constraints": {
                "moe_hybrid_supported_archs": ["deepseek_v2", "qwen3_moe"],
            },
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            caps = modelctl_capabilities.probe_backend(str(script))
            self.assertTrue(modelctl_capabilities.supports_hybrid_miss(caps))
            self.assertIn("deepseek_v2",
                          caps["constraints"]["moe_hybrid_supported_archs"])

    def test_session_cache_hit(self):
        caps_json = json.dumps({
            "schema": 2, "backend": "llama.cpp", "build": "x",
            "features": {"moe_weight_transfer_cache": True}, "cli": {},
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
            "schema": 2, "backend": "llama.cpp", "build": "y",
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
            "schema": 2, "backend": "llama.cpp", "build": "z",
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
        caps = {"features": {"moe_weight_transfer_cache": True}}
        args = modelctl.build_moe_cache_args(profile, capabilities=caps)
        self.assertIn("--moe-cache-bytes", args)
        self.assertIn("--moe-cache-policy", args)
        self.assertIn("slru", args)

    def test_multiple_budgets_emit_uniform_max_not_sum(self):
        # The fork applies --moe-cache-bytes per GPU (one global copied into
        # every device's cache), so per-device budgets collapse to their max.
        # Summing would over-allocate on every card vs the planner's reserve.
        import modelctl
        profile = {
            "moe_cache": {
                "mode": "auto",
                "gpu": {"budgets_bytes": {"SYCL0": 10 * (1 << 30),
                                          "SYCL1": 4 * (1 << 30)}},
            }
        }
        caps = {"features": {"moe_weight_transfer_cache": True}}
        args = modelctl.build_moe_cache_args(profile, capabilities=caps)
        self.assertEqual(args.count("--moe-cache-bytes"), 1)
        i = args.index("--moe-cache-bytes")
        self.assertEqual(args[i + 1], str(10 * (1 << 30)))

    def test_structured_budget_overrides_extra_flag(self):
        # build_server_args: structured moe_cache settings take precedence --
        # raw --moe-cache-* tokens in extra are dropped, not duplicated.
        import modelctl
        profile = {
            "model_path": "/x.gguf",
            "config": {"device": "", "split_mode": "", "tensor_split": "",
                       "ctx": 8192, "flash_attn": "auto", "fit": "on",
                       "mtp": "off",
                       "extra": "--moe-cache-bytes 123 --verbose-prompt"},
            "moe_cache": {"mode": "auto",
                          "gpu": {"budgets_bytes": {"SYCL0": 10 * (1 << 30)}}},
        }
        caps = {"features": {"moe_weight_transfer_cache": True}}
        args = modelctl.build_server_args(profile, capabilities=caps)
        self.assertEqual(args.count("--moe-cache-bytes"), 1)
        i = args.index("--moe-cache-bytes")
        self.assertEqual(args[i + 1], str(10 * (1 << 30)))
        self.assertIn("--verbose-prompt", args)

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
            "features": {"moe_weight_transfer_cache": True},
            "cli": {
                "moe_cache_bytes": "--custom-cache-bytes",
                "moe_cache_policy": "--custom-policy",
                "moe_cache_admission": "--custom-admission",
                "moe_cache_prefill": "--custom-prefill-admission",
            }
        }
        args = modelctl.build_moe_cache_args(profile, capabilities=caps)
        self.assertIn("--custom-cache-bytes", args)
        self.assertIn("--custom-policy", args)
        self.assertIn("--custom-admission", args)
        self.assertIn("--custom-prefill-admission", args)

    def test_unprobed_backend_fails_closed(self):
        # §2.5: an unprobed backend must never receive experimental cache
        # flags.  No capabilities -> only the stock --metrics flag.
        import modelctl
        profile = {
            "moe_cache": {
                "mode": "auto",
                "gpu": {"budgets_bytes": {"SYCL0": 10 * (1 << 30)}},
                "decode": {"miss_execution": "cpu"},
            }
        }
        args = modelctl.build_moe_cache_args(profile, capabilities=None)
        self.assertEqual(args, ["--metrics"])

    def test_incapable_backend_fails_closed(self):
        import modelctl
        profile = {
            "moe_cache": {
                "mode": "manual",
                "gpu": {"budgets_bytes": {"SYCL0": 10 * (1 << 30)}},
                "decode": {"miss_execution": "cpu"},
            }
        }
        caps = {"features": {"moe_weight_transfer_cache": False,
                             "moe_hybrid_cpu_miss": False}}
        args = modelctl.build_moe_cache_args(profile, capabilities=caps)
        self.assertEqual(args, ["--metrics"])

    def test_hybrid_flag_only_with_real_capability(self):
        import modelctl
        profile = {
            "moe_cache": {
                "mode": "auto",
                "gpu": {"budgets_bytes": {"SYCL0": 10 * (1 << 30)}},
                "decode": {"miss_execution": "cpu"},
            }
        }
        # Cache-capable but NOT hybrid-capable: cache flags yes, hybrid no.
        caps = {"features": {"moe_weight_transfer_cache": True,
                             "moe_hybrid_cpu_miss": False}}
        args = modelctl.build_moe_cache_args(profile, capabilities=caps)
        self.assertIn("--moe-cache-bytes", args)
        self.assertNotIn("--moe-hybrid-mode", args)
        # Hybrid-capable: flag emitted.
        caps["features"]["moe_hybrid_cpu_miss"] = True
        args = modelctl.build_moe_cache_args(profile, capabilities=caps)
        self.assertIn("--moe-hybrid-mode", args)

    def test_stock_binary_never_gets_cache_flags(self):
        # Task 1.4: stock llama.cpp plus cache-enabled profile never
        # receives a cache argument through build_server_args.
        import modelctl
        profile = {
            "model_path": "/x.gguf",
            "binary": "/nonexistent/llama-server",  # unprobed
            "config": {"device": "", "split_mode": "", "tensor_split": "",
                       "ctx": 8192, "flash_attn": "auto", "fit": "on",
                       "mtp": "off", "extra": ""},
            "moe_cache": {"mode": "manual",
                          "gpu": {"budgets_bytes": {"SYCL0": 10 * (1 << 30)}},
                          "decode": {"miss_execution": "cpu"}},
        }
        args = modelctl.build_server_args(profile)
        self.assertFalse(any(a.startswith("--moe-") for a in args), args)
        # An explicit incapable capability set gives the same result.
        args = modelctl.build_server_args(
            profile, capabilities={"features": {}, "cli": {}})
        self.assertFalse(any(a.startswith("--moe-") for a in args), args)


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

    def test_no_capabilities_errors_for_manual(self):
        # §2.5: manual mode against an unprobed backend blocks with an
        # explicit reason instead of emitting unvalidated flags.
        import modelctl
        profile = {"moe_cache": {"mode": "manual"}}
        msgs = modelctl.preflight_moe_cache(profile, capabilities=None)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0][0], "error")

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
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": False,
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
                "gpu": {"budgets_bytes": {"SYCL0": 10 * (1 << 30)}},
                "decode": {"miss_execution": "cpu"},
                "prefetch": {"enabled": True},
            }
        }
        caps = {
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": False,
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
        caps = {"features": {"moe_weight_transfer_cache": True}}
        args = modelctl.build_server_args(profile, capabilities=caps)
        self.assertIn("--moe-cache-bytes", args)
        self.assertIn("--moe-cache-policy", args)
        # cache telemetry needs the Prometheus endpoint up
        self.assertIn("--metrics", args)
        # hybrid CPU miss is not claimed by these capabilities, so the
        # experimental flag must not be emitted even though the profile
        # asks for cpu miss execution
        self.assertNotIn("--moe-hybrid-mode", args)

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
        self.assertNotIn("--metrics", args)


class TestMmapAdviseSurvivesNormalization(unittest.TestCase):
    """The binary has emitted moe_cache_mmap_advise since f4d390349 and
    integration-manifest.json lists it supported, but it was missing from
    normalize_capabilities()' canonical whitelist -- so every caller that
    asked modelctl whether the runtime had it was told no, including the
    deploy check whose job is to confirm the feature landed."""

    def test_a_schema3_probe_carries_the_flag_through(self):
        raw = {"schema": 3, "backend": "llama.cpp",
               "build": {"commit": "85b7e6556"},
               "features": {"moe_weight_transfer_cache": True,
                            "moe_cache_mmap_advise": True}}
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertTrue(norm["features"]["moe_cache_mmap_advise"])
        self.assertTrue(modelctl_capabilities.supports_mmap_advise(norm))

    def test_a_build_without_it_reports_false_not_missing(self):
        # Fail-closed, and present as a key: a consumer doing
        # features["moe_cache_mmap_advise"] must not KeyError on an older
        # build, which is how a gate turns into a crash.
        raw = {"schema": 3, "backend": "llama.cpp", "build": {"commit": "x"},
               "features": {"moe_weight_transfer_cache": True}}
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertIs(norm["features"]["moe_cache_mmap_advise"], False)
        self.assertFalse(modelctl_capabilities.supports_mmap_advise(norm))

    def test_schema1_predates_it_entirely(self):
        raw = {"schema": 1, "features": {"moe_expert_cache": True,
                                         "moe_cache_sycl": True}}
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertIs(norm["features"]["moe_cache_mmap_advise"], False)

    def test_a_failed_probe_reports_it_false(self):
        caps = modelctl_capabilities._classify_probe_failure("", status="error")
        self.assertIs(caps["features"]["moe_cache_mmap_advise"], False)

    def test_the_manifest_and_the_whitelist_agree(self):
        # The drift this bug was: the manifest declared the feature
        # supported while modelctl could not see it at all.
        import json
        from pathlib import Path
        manifest = json.loads(
            (Path(__file__).resolve().parent.parent
             / "integration-manifest.json").read_text())
        declared = set(manifest["supported_runtime_features"]) | set(
            manifest["unsupported_runtime_features"])
        raw = {"schema": 3, "features": {}}
        known = set(modelctl_capabilities.normalize_capabilities(
            raw)["features"])
        self.assertEqual(declared - known, set(),
                         "manifest names features modelctl cannot represent")


class TestNormalizeCapabilities(unittest.TestCase):
    """Test normalize_capabilities() converts schema 0/1 to canonical form."""

    def test_schema0_unsupported(self):
        raw = {"schema": 0, "_probe_status": "unsupported", "features": {}}
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertEqual(norm["schema"],
                         modelctl_capabilities.CAPABILITY_SCHEMA_VERSION)
        self.assertFalse(norm["features"]["moe_weight_transfer_cache"])
        self.assertFalse(norm["features"]["moe_hybrid_cpu_miss"])
        self.assertEqual(norm["_probe_status"], "unsupported")

    def test_schema1_maps_to_canonical_features(self):
        raw = {
            "schema": 1,
            "backend": "llama.cpp",
            "build": "test",
            "devices": ["CPU", "SYCL0"],
            "features": {
                "moe_expert_cache": True,
                "moe_cache_sycl": True,
                "moe_hybrid_cpu_miss": True,  # was wrongly true in schema 1
                "moe_cache_metrics": True,
                "moe_cache_prefill_policy": True,
            },
            "cli": {
                "cache_bytes": "--moe-cache-bytes",
                "cache_policy": "--moe-cache-policy",
                "admission_misses": "--moe-cache-admission-misses",
                "prefill_admission": "--moe-cache-prefill-admission",
            },
        }
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertEqual(norm["schema"],
                         modelctl_capabilities.CAPABILITY_SCHEMA_VERSION)
        self.assertTrue(norm["features"]["moe_weight_transfer_cache"])
        # hybrid is allowed through from schema 1 when both flags are true
        self.assertTrue(norm["features"]["moe_hybrid_cpu_miss"])
        self.assertTrue(norm["features"]["moe_cache_metrics"])
        self.assertTrue(norm["features"]["moe_cache_prefill_policy"])
        self.assertTrue(norm["features"]["moe_cache_reset"])
        self.assertFalse(norm["features"]["moe_cache_prefetch"])
        self.assertEqual(norm["_raw_schema"], 1)

    def test_schema1_cli_mapped_to_canonical_keys(self):
        raw = {
            "schema": 1,
            "features": {"moe_expert_cache": True, "moe_cache_sycl": True},
            "cli": {
                "cache_bytes": "--my-cache-bytes",
                "cache_policy": "--my-policy",
                "admission_misses": "--my-admission",
                "prefill_admission": "--my-prefill",
            },
        }
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertEqual(norm["cli"]["moe_cache_bytes"], "--my-cache-bytes")
        self.assertEqual(norm["cli"]["moe_cache_policy"], "--my-policy")
        self.assertEqual(norm["cli"]["moe_cache_admission"], "--my-admission")
        self.assertEqual(norm["cli"]["moe_cache_prefill"], "--my-prefill")

    def test_schema1_devices_normalized_to_objects(self):
        raw = {
            "schema": 1,
            "features": {"moe_expert_cache": True, "moe_cache_sycl": True},
            "devices": ["CPU", "SYCL0", "SYCL1"],
            "cli": {},
        }
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertEqual(len(norm["devices"]), 3)
        self.assertEqual(norm["devices"][0]["type"], "CPU")
        self.assertEqual(norm["devices"][1]["type"], "SYCL")
        self.assertEqual(norm["devices"][1]["name"], "SYCL0")
        self.assertTrue(norm["devices"][1]["features"]["moe_weight_transfer_cache"])

    def test_schema1_without_sycl_not_cache_capable(self):
        raw = {
            "schema": 1,
            "features": {"moe_expert_cache": True, "moe_cache_sycl": False},
            "cli": {},
        }
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertFalse(norm["features"]["moe_weight_transfer_cache"])

    def test_schema2_passthrough(self):
        raw = {
            "schema": 2,
            "backend": "llama.cpp",
            "build": {"commit": "abc", "compiler": "gcc", "dynamic_backends": True},
            "devices": [
                {"type": "SYCL", "name": "SYCL0", "index": 0,
                 "features": {"moe_weight_transfer_cache": True}},
            ],
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": False,
                "moe_cache_metrics": True,
                "moe_cache_prefill_policy": True,
                "moe_cache_reset": True,
                "moe_cache_prefetch": False,
            },
            "constraints": {
                "moe_cache_backend": "SYCL",
                "moe_cache_min_batch": 32,
                "moe_cache_supported_projections": ["gate", "up", "down"],
            },
            "cli": {
                "moe_cache_bytes": "--moe-cache-bytes",
                "moe_cache_policy": "--moe-cache-policy",
                "moe_cache_admission": "--moe-cache-admission-misses",
                "moe_cache_prefill": "--moe-cache-prefill-admission",
            },
        }
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertEqual(norm["schema"],
                         modelctl_capabilities.CAPABILITY_SCHEMA_VERSION)
        self.assertTrue(norm["features"]["moe_weight_transfer_cache"])
        self.assertEqual(norm["constraints"]["moe_cache_backend"], "SYCL")
        self.assertEqual(norm["constraints"]["moe_cache_min_batch"], 32)

    def test_schema2_forced_fail_closed(self):
        """Prefetch is forced false; hybrid passes through."""
        raw = {
            "schema": 2,
            "features": {
                "moe_weight_transfer_cache": True,
                "moe_hybrid_cpu_miss": True,
                "moe_cache_prefetch": True,   # not implemented
            },
        }
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertTrue(norm["features"]["moe_hybrid_cpu_miss"])  # now allowed
        self.assertFalse(norm["features"]["moe_cache_prefetch"])  # still forced false
        self.assertTrue(norm["_raw_features"]["moe_cache_prefetch"])

    def test_schema1_missing_features_default_false(self):
        raw = {"schema": 1, "features": {}, "cli": {}}
        norm = modelctl_capabilities.normalize_capabilities(raw)
        self.assertFalse(norm["features"]["moe_weight_transfer_cache"])
        self.assertFalse(norm["features"]["moe_hybrid_cpu_miss"])
        self.assertFalse(norm["features"]["moe_cache_metrics"])

    def test_normalize_idempotent(self):
        raw = {
            "schema": 2,
            "features": {"moe_weight_transfer_cache": True},
        }
        n1 = modelctl_capabilities.normalize_capabilities(raw)
        n2 = modelctl_capabilities.normalize_capabilities(n1)
        self.assertEqual(n1["features"], n2["features"])
