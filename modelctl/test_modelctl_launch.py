import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl_capabilities
import modelctl_launch


class TestResolvedBackend(unittest.TestCase):
    def test_fields(self):
        backend = modelctl_launch.ResolvedBackend(
            name="llama-cpp",
            binary="/usr/bin/llama-server",
            binary_fingerprint="abc123",
            environment={"PATH": "/usr/bin"},
            environment_fingerprint="def456",
            capabilities={"schema": 2, "features": {}},
        )
        self.assertEqual(backend.name, "llama-cpp")
        self.assertEqual(backend.binary, "/usr/bin/llama-server")
        self.assertEqual(backend.binary_fingerprint, "abc123")

    def test_frozen(self):
        backend = modelctl_launch.ResolvedBackend(
            name="test", binary="/bin/test", binary_fingerprint="x",
            environment={}, environment_fingerprint="y", capabilities={},
        )
        with self.assertRaises(AttributeError):
            backend.name = "changed"


class TestLaunchCommand(unittest.TestCase):
    def test_fields(self):
        backend = modelctl_launch.ResolvedBackend(
            name="llama-cpp", binary="/bin/llama-server",
            binary_fingerprint="abc", environment={},
            environment_fingerprint="def", capabilities={},
        )
        cmd = modelctl_launch.LaunchCommand(
            argv=("/bin/llama-server", "--model", "/m.gguf"),
            environment={},
            backend=backend,
            profile_name="test-model",
            plan_id="plan123",
            port=8080,
            warnings=("partial GPU offload",),
            validation=(),
            command_fingerprint="fp123",
        )
        self.assertEqual(cmd.profile_name, "test-model")
        self.assertEqual(cmd.plan_id, "plan123")
        self.assertEqual(cmd.port, 8080)
        self.assertIn("partial GPU offload", cmd.warnings)

    def test_frozen(self):
        backend = modelctl_launch.ResolvedBackend(
            name="t", binary="/b", binary_fingerprint="f",
            environment={}, environment_fingerprint="e", capabilities={},
        )
        cmd = modelctl_launch.LaunchCommand(
            argv=(), environment={}, backend=backend,
            profile_name="", plan_id="", port=None,
            warnings=(), validation=(), command_fingerprint="x",
        )
        with self.assertRaises(AttributeError):
            cmd.port = 9999


class TestCommandFingerprint(unittest.TestCase):
    def test_port_excluded_from_identity(self):
        """Same command on different ports shares a fingerprint."""
        fp1 = modelctl_launch._command_fingerprint(
            ("/bin/srv", "--model", "m.gguf", "--port", "8080"),
            "env1", "bin1")
        fp2 = modelctl_launch._command_fingerprint(
            ("/bin/srv", "--model", "m.gguf", "--port", "9090"),
            "env1", "bin1")
        self.assertEqual(fp1, fp2)

    def test_different_args_different_fingerprint(self):
        fp1 = modelctl_launch._command_fingerprint(
            ("/bin/srv", "--model", "a.gguf"), "env1", "bin1")
        fp2 = modelctl_launch._command_fingerprint(
            ("/bin/srv", "--model", "b.gguf"), "env1", "bin1")
        self.assertNotEqual(fp1, fp2)

    def test_different_env_different_fingerprint(self):
        fp1 = modelctl_launch._command_fingerprint(
            ("/bin/srv",), "env1", "bin1")
        fp2 = modelctl_launch._command_fingerprint(
            ("/bin/srv",), "env2", "bin1")
        self.assertNotEqual(fp1, fp2)


class TestResolveBackend(unittest.TestCase):
    def test_stock_binary_returns_unsupported(self):
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            # Coherent rejection (message + exit 1): a silent exit 1 is
            # now "error" (transient, uncached), not "unsupported".
            script.write_text(
                '#!/bin/sh\necho "error: invalid argument: $1" >&2\nexit 1\n')
            script.chmod(0o755)
            profile = {"backend": "llama-cpp", "binary": str(script),
                      "model_path": str(Path(d) / "model.gguf")}
            with mock.patch("modelctl_launch.modelctl.find_env_script_candidates",
                           return_value=[]):
                backend = modelctl_launch.resolve_backend(profile)
            self.assertEqual(backend.name, "llama-cpp")
            self.assertEqual(backend.binary, str(script))
            self.assertFalse(backend.capabilities["features"]["moe_weight_transfer_cache"])
            self.assertEqual(backend.capabilities["_probe_status"], "unsupported")

    def test_cache_capable_binary(self):
        caps_json = json.dumps({
            "schema": 2,
            "backend": "llama.cpp",
            "features": {"moe_weight_transfer_cache": True},
            "cli": {"moe_cache_bytes": "--moe-cache-bytes"},
        })
        with TemporaryDirectory() as d:
            script = Path(d) / "fake-server"
            script.write_text(f"#!/bin/sh\necho '{caps_json}'\n")
            script.chmod(0o755)
            profile = {"backend": "llama-cpp", "binary": str(script),
                      "model_path": str(Path(d) / "model.gguf")}
            with mock.patch("modelctl_launch.modelctl.find_env_script_candidates",
                           return_value=[]):
                backend = modelctl_launch.resolve_backend(profile)
            self.assertTrue(backend.capabilities["features"]["moe_weight_transfer_cache"])

    def test_unresolved_default_uses_preflight_autofix_search(self):
        # The real-world case this was broken for: MODELCTL_LLAMA_SERVER
        # unset and "llama-server" not on PATH, but a real build exists
        # in one of the usual locations. resolve_backend() must find it
        # via the same auto-fix search preflight() does (modelctl.py's
        # find_llama_server_candidates), not hand back the unresolved
        # "llama-server" fallback string that a real launch can't exec.
        import modelctl
        with TemporaryDirectory() as d:
            script = Path(d) / "found-server"
            script.write_text("#!/bin/sh\nexit 1\n")
            script.chmod(0o755)
            profile = {"backend": "llama-cpp",
                      "model_path": str(Path(d) / "m.gguf"),
                      "config": {"device": ""}}
            with mock.patch.object(modelctl, "LLAMA_SERVER_RESOLVED", False), \
                 mock.patch.object(modelctl, "find_llama_server_candidates",
                                   return_value=[str(script)]), \
                 mock.patch("modelctl_launch.modelctl.find_env_script_candidates",
                           return_value=[]):
                backend = modelctl_launch.resolve_backend(profile)
        self.assertEqual(backend.binary, str(script))
        self.assertNotEqual(backend.binary, "llama-server")


class TestBuildLaunchCommandValidation(unittest.TestCase):
    """The canonical launch object carries fail-closed validation that
    workers and plan tests must enforce."""

    def _profile(self, mode="manual"):
        return {
            "name": "m", "backend": "llama-cpp",
            "model_path": "/m.gguf",
            "config": {"device": "", "split_mode": "", "tensor_split": "",
                       "ctx": 8192, "flash_attn": "auto", "fit": "on",
                       "mtp": "off", "extra": "", "ttl": 3600},
            "moe_cache": {"mode": mode,
                          "gpu": {"budgets_bytes": {"SYCL0": 1 << 30}},
                          "decode": {"miss_execution": "cpu"}},
        }

    def _backend(self, features, preflight_messages=()):
        return modelctl_launch.ResolvedBackend(
            name="llama-cpp", binary="/bin/llama-server",
            binary_fingerprint="abc", environment={},
            environment_fingerprint="def",
            capabilities={"schema": 2, "features": features, "cli": {}},
            preflight_messages=tuple(preflight_messages))

    class _Plan:
        id = "plan1"
        env = {}

    def test_manual_cache_on_incapable_backend_is_error(self):
        backend = self._backend({"moe_weight_transfer_cache": False,
                                 "moe_hybrid_cpu_miss": False})
        with mock.patch("modelctl_backends.get_backend") as gb:
            gb.return_value.build_command.return_value = ["/bin/llama-server"]
            cmd = modelctl_launch.build_launch_command(
                self._profile("manual"), self._Plan(), backend=backend, port=8080)
        errors = [v for v in cmd.validation if v.severity == "error"]
        self.assertTrue(errors, "manual cache on incapable backend not blocked")
        self.assertFalse(cmd.is_valid)
        with self.assertRaises(modelctl_launch.LaunchValidationError):
            cmd.raise_for_errors()

    def test_auto_cache_on_incapable_backend_is_warning(self):
        backend = self._backend({"moe_weight_transfer_cache": False,
                                 "moe_hybrid_cpu_miss": False})
        with mock.patch("modelctl_backends.get_backend") as gb:
            gb.return_value.build_command.return_value = ["/bin/llama-server"]
            cmd = modelctl_launch.build_launch_command(
                self._profile("auto"), self._Plan(), backend=backend, port=8080)
        self.assertFalse([v for v in cmd.validation if v.severity == "error"])
        self.assertTrue([v for v in cmd.validation if v.severity == "warning"])
        self.assertTrue(cmd.is_valid)
        cmd.raise_for_errors()

    def test_real_preflight_message_strings_do_not_crash(self):
        # modelctl.preflight() returns plain message strings (severity as
        # a text prefix), not (severity, summary) tuples -- every other
        # test here uses an empty message list, which never exercises
        # this. Real profiles almost always produce at least one message
        # (an auto-fix note, a pinned-binary note, a warning), so this
        # reproduces the actual crash a real launch hit. resolve_backend()
        # now records those strings on the ResolvedBackend rather than
        # build_launch_command() re-running preflight for them.
        backend = self._backend(
            {"moe_weight_transfer_cache": True, "moe_hybrid_cpu_miss": False},
            preflight_messages=[
                "using pinned binary: /bin/llama-server",
                "WARNING: configured llama-server doesn't appear to support 'SYCL0'",
                "ERROR: mmproj file not found on disk: /x.mmproj"])
        with mock.patch("modelctl_backends.get_backend") as gb:
            gb.return_value.build_command.return_value = ["/bin/llama-server"]
            cmd = modelctl_launch.build_launch_command(
                self._profile("off"), self._Plan(), backend=backend, port=8080)
        self.assertIn("configured llama-server doesn't appear to support 'SYCL0'",
                      cmd.warnings)
        errors = [v.summary for v in cmd.validation if v.severity == "error"]
        self.assertIn("mmproj file not found on disk: /x.mmproj", errors)
        self.assertFalse(cmd.is_valid)

    def test_environment_fingerprint_ignores_process_env(self):
        # Only profile-declared env feeds the fingerprint: unrelated shell
        # changes must not mass-stale observations.
        with mock.patch("modelctl_capabilities.probe_backend",
                        return_value={"schema": 2, "features": {}, "cli": {}}), \
             mock.patch("modelctl_vram.file_fingerprint", return_value="fp"), \
             mock.patch("modelctl.find_env_script_candidates", return_value=[]), \
             mock.patch.dict("os.environ", {"SSH_AUTH_SOCK": "/tmp/a"}):
            b1 = modelctl_launch.resolve_backend(self._profile())
        with mock.patch("modelctl_capabilities.probe_backend",
                        return_value={"schema": 2, "features": {}, "cli": {}}), \
             mock.patch("modelctl_vram.file_fingerprint", return_value="fp"), \
             mock.patch("modelctl.find_env_script_candidates", return_value=[]), \
             mock.patch.dict("os.environ", {"SSH_AUTH_SOCK": "/tmp/b",
                                            "RANDOM_VAR": "x"}):
            b2 = modelctl_launch.resolve_backend(self._profile())
        self.assertEqual(b1.environment_fingerprint, b2.environment_fingerprint)
