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
            script.write_text("#!/bin/sh\nexit 1\n")
            script.chmod(0o755)
            profile = {"backend": "llama-cpp", "binary": str(script)}
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
            profile = {"backend": "llama-cpp", "binary": str(script)}
            with mock.patch("modelctl_launch.modelctl.find_env_script_candidates",
                           return_value=[]):
                backend = modelctl_launch.resolve_backend(profile)
            self.assertTrue(backend.capabilities["features"]["moe_weight_transfer_cache"])
