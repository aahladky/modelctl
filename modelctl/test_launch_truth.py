"""Phase B acceptance tests: one authoritative launch path.

Task B2 -- a command containing unsupported experimental flags can never
be generated or launched.
Task B3 -- the browser preview, CLI preview, plan-test process, managed
worker, generated run.sh, and llama-swap configuration all describe the
same command.

These are integration tests over a real (fake) binary on disk rather than
unit tests with mocked capabilities: the failure mode they exist to catch
is precisely a path that quietly resolves a *different* binary or skips
the capability probe, which mocking the probe would hide.
"""
import json
import re
import shlex
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

import modelctl
import modelctl_capabilities
import modelctl_launch
import modelctl_plans
from modelctl_web.app import create_app
from modelctl_web.jobs import JobStore, JobRunner

TOKEN = "test-token"

# A stock upstream llama-server: --version works, --modelctl-capabilities
# is an unrecognized argument and exits non-zero.  This is what the probe
# classifies as schema 0 / "unsupported".
STOCK_SERVER = """#!/bin/sh
case "$1" in
  --version) echo "version: 6000 (abcdef)"; exit 0 ;;
  *) echo "error: invalid argument: $1" >&2; exit 1 ;;
esac
"""

# Every cache setting a profile can turn on, in manual mode -- the most
# aggressive request the control plane can make of a backend.
FULL_CACHE = {
    "mode": "manual",
    "gpu": {"budgets_bytes": {"SYCL0": 8 << 30}, "policy": "slru",
            "probationary_fraction": 0.2, "admission_misses": 2,
            "pin_shared_experts": True, "pin_static_experts": []},
    "ram": {"mode": "page_cache", "budget_bytes": 16 << 30, "mlock_hot_set": True},
    "storage": {"mode": "mmap", "readahead": "adaptive",
                "release_cold_pages": True},
    "prefill": {"admit_to_gpu_cache": True, "protect_decode_entries": True},
    "decode": {"admit_to_gpu_cache": True, "miss_execution": "cpu"},
    "prefetch": {"enabled": True, "method": "sequential",
                 "max_overfetch_ratio": 1.5},
}


class LaunchTruthBase(unittest.TestCase):
    """A profile pinned to a stock binary, with a real model file on disk."""

    moe_cache = None

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)

        self.binary = root / "llama-server"
        self.binary.write_text(STOCK_SERVER)
        self.binary.chmod(0o755)

        # preflight() requires the model file to exist, or it fails before
        # any command is built and the test proves nothing.
        self.model = root / "model.gguf"
        self.model.write_bytes(b"GGUF" + b"\0" * 64)

        self.profiles_dir = root / "profiles"
        self.profiles_dir.mkdir()

        self.profile = {
            "name": "m1", "repo_id": "r/m", "file": "m-Q4_K_M",
            "model_path": str(self.model), "backend": "llama-cpp",
            "binary": str(self.binary),
            "config": {"device": "SYCL0", "split_mode": "", "tensor_split": "",
                       "ctx": 32768, "cache_type_k": "q8_0",
                       "cache_type_v": "q4_0", "flash_attn": "auto",
                       "ttl": 3600, "mtp": "off", "fit": "off", "extra": ""},
            "env": [], "enabled": True,
        }
        if self.moe_cache is not None:
            self.profile["moe_cache"] = self.moe_cache
        (self.profiles_dir / "m1.json").write_text(json.dumps(self.profile))

        # The probe caches by binary content hash; a stale entry from another
        # test's fake binary would defeat the point of using a real one.
        modelctl_capabilities.clear_cache()
        self.addCleanup(modelctl_capabilities.clear_cache)

        caps_dir = root / "capabilities"
        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir),
                  mock.patch.object(modelctl_capabilities, "CAPABILITIES_DIR",
                                    caps_dir)):
            p.start()
            self.addCleanup(p.stop)

    def web_client(self):
        store = JobStore(Path(self.tmp.name) / "jobs.db")
        runner = JobRunner(store)
        self.addCleanup(lambda: runner._thread.join(timeout=1) or None)
        client = TestClient(create_app(token=TOKEN, store=store, runner=runner))
        return client, {"Authorization": f"Bearer {TOKEN}"}


class TestStockBinaryNeverGetsCacheFlags(LaunchTruthBase):
    """Task B2 acceptance: stock upstream llama-server + every cache setting
    enabled => no --moe-cache-* argument anywhere, on any surface."""

    moe_cache = FULL_CACHE

    CACHE_FLAG = re.compile(r"--moe[-_]")

    def assertNoCacheFlags(self, text, surface):
        found = self.CACHE_FLAG.findall(text)
        self.assertFalse(
            found,
            f"{surface} emitted an experimental cache flag against a stock "
            f"binary that cannot support it:\n{text}")

    def test_probe_reports_stock_binary_as_incapable(self):
        # Guards the premise: if the fake binary ever started answering the
        # probe, every assertion below would pass vacuously.
        caps = modelctl_capabilities.probe_backend(str(self.binary))
        self.assertEqual(caps["_probe_status"], "unsupported")
        self.assertFalse(caps["features"]["moe_weight_transfer_cache"])

    def test_canonical_command_has_no_cache_flags(self):
        cmd, _ok, _msgs = modelctl.canonical_launch_command(self.profile)
        self.assertNoCacheFlags(" ".join(cmd.argv), "canonical command")

    def test_manual_mode_blocks_the_launch(self):
        # Omitting the flags is not enough: manual mode asked for a feature
        # the backend does not have, so the command must be unlaunchable
        # rather than silently degraded.
        cmd, _ok, _msgs = modelctl.canonical_launch_command(self.profile)
        self.assertFalse(cmd.is_valid)
        self.assertTrue(cmd.errors)
        with self.assertRaises(modelctl_launch.LaunchValidationError):
            cmd.raise_for_errors()

    def test_generated_run_sh_has_no_cache_flags(self):
        modelctl.generate_artifacts(dict(self.profile))
        run_sh = (self.profiles_dir / "m1" / "run.sh").read_text()
        self.assertNoCacheFlags(run_sh, "generated run.sh")

    def test_llama_swap_entry_has_no_cache_flags(self):
        entry, ok, _msgs = modelctl.render_llama_swap_entry(dict(self.profile))
        self.assertNoCacheFlags(entry, "llama-swap entry")
        self.assertFalse(ok, "llama-swap entry reported a blocked profile as resolved")

    def test_browser_preview_has_no_cache_flags(self):
        client, auth = self.web_client()
        resp = client.get("/profiles/m1/run.sh", headers=auth)
        self.assertEqual(resp.status_code, 200)
        self.assertNoCacheFlags(resp.text, "browser run.sh preview")

    def test_browser_shows_why_the_command_is_blocked(self):
        # Task B2: validation messages must be visible where the user makes
        # the decision, not only in a job log.
        client, auth = self.web_client()
        resp = client.get("/profiles/m1", headers=auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("moe_weight_transfer_cache", resp.text)
        self.assertIn("blocked", resp.text)

    def test_plan_argv_has_no_cache_flags(self):
        # Plans feed the worker and the plan test; a cache flag surviving
        # here would reach both.
        backend = modelctl_launch.resolve_backend(self.profile)
        plan = modelctl_plans.current_profile_plan(
            self.profile, capabilities=backend.capabilities)
        self.assertNoCacheFlags(" ".join(plan.argv), "launch plan argv")

    def test_smoke_test_refuses_to_launch(self):
        res = modelctl.smoke_test_profile("m1", timeout=1)
        self.assertFalse(res["ok"])
        self.assertEqual(res["stage"], "preflight",
                         "smoke test got past validation with a blocked profile")


class TestCommandEqualityAcrossSurfaces(LaunchTruthBase):
    """Task B3: one profile, one plan, six surfaces, one command.

    Only the assigned port and wrapper-specific quoting may differ.  Binary,
    model path, placement, cache flags, context, and backend options must be
    identical -- so all six must share one command_fingerprint, which
    excludes the port by construction.
    """

    def canonical(self, port=None):
        return modelctl.canonical_launch_command(dict(self.profile), port=port)

    def test_all_surfaces_share_one_fingerprint(self):
        backend = modelctl_launch.resolve_backend(self.profile)
        plan = modelctl_plans.current_profile_plan(
            self.profile, capabilities=backend.capabilities)

        fingerprints = {}

        # 1. browser preview (web route)
        client, auth = self.web_client()
        resp = client.get("/profiles/m1/run.sh", headers=auth)
        self.assertEqual(resp.status_code, 200)
        browser_argv = tuple(
            tok for tok in resp.text.replace("\\\n", " ").split() if tok)

        # 2. CLI preview / rendering entry point
        cli_cmd, _ok, _msgs = self.canonical()
        fingerprints["cli-preview"] = cli_cmd.command_fingerprint

        # 3. plan-test process (modelctl_tune's construction, port assigned)
        plan_test = modelctl_launch.build_launch_command(
            self.profile, plan, backend=backend, port=45001)
        fingerprints["plan-test"] = plan_test.command_fingerprint

        # 4. managed worker (modelctl_worker's construction, different port)
        worker = modelctl_launch.build_launch_command(
            self.profile, plan, backend=backend, port=45002)
        fingerprints["managed-worker"] = worker.command_fingerprint

        # 5. generated run.sh
        modelctl.generate_artifacts(dict(self.profile))
        run_sh = (self.profiles_dir / "m1" / "run.sh").read_text()

        # 6. llama-swap configuration
        entry, _ok, _msgs = modelctl.render_llama_swap_entry(dict(self.profile))

        self.assertEqual(len(set(fingerprints.values())), 1,
                         f"surfaces disagree on command identity: {fingerprints}")

        # The two rendered surfaces are shell text, so compare them by
        # re-tokenizing rather than by fingerprint.
        expected = list(cli_cmd.argv)
        self.assertEqual(list(browser_argv), expected,
                         "browser preview differs from the canonical command")
        self.assertEqual(self._argv_from_run_sh(run_sh), expected,
                         "generated run.sh differs from the canonical command")
        self.assertEqual(self._argv_from_swap_entry(entry), expected,
                         "llama-swap entry differs from the canonical command")

    @staticmethod
    def _strip_port(argv):
        out, skip = [], False
        for a in argv:
            if skip:
                skip = False
                continue
            if a == "--port":
                skip = True
                continue
            out.append(a)
        return out

    def _argv_from_run_sh(self, text):
        line = text.split("\n$", 1)[0]
        cmd_line = [ln for ln in text.replace("\\\n", " ").splitlines()
                    if "llama-server" in ln][-1]
        return self._strip_port(shlex.split(cmd_line))

    def _argv_from_swap_entry(self, text):
        cmd_line = [ln for ln in text.replace("\\\n", " ").splitlines()
                    if "llama-server" in ln][-1]
        # ${PORT} is llama-swap's placeholder, not part of command identity.
        return self._strip_port(shlex.split(cmd_line.replace("${PORT}", "0")))

    def test_port_is_the_only_difference_between_launches(self):
        backend = modelctl_launch.resolve_backend(self.profile)
        plan = modelctl_plans.current_profile_plan(
            self.profile, capabilities=backend.capabilities)
        a = modelctl_launch.build_launch_command(
            self.profile, plan, backend=backend, port=45001)
        b = modelctl_launch.build_launch_command(
            self.profile, plan, backend=backend, port=45002)
        self.assertNotEqual(a.argv, b.argv)
        self.assertEqual(self._strip_port(a.argv), self._strip_port(b.argv))
        self.assertEqual(a.command_fingerprint, b.command_fingerprint)

    def test_rendering_paths_carry_no_literal_port(self):
        # port=None must leave --port out entirely so renderers can insert
        # ${PORT} -- a literal "--port None" would be launched verbatim.
        cmd, _ok, _msgs = self.canonical()
        self.assertNotIn("--port", cmd.argv)
        self.assertNotIn("None", cmd.argv)

    def test_resolved_binary_is_argv0_on_every_surface(self):
        cmd, _ok, _msgs = self.canonical()
        self.assertEqual(cmd.argv[0], str(self.binary))
        modelctl.generate_artifacts(dict(self.profile))
        run_sh = (self.profiles_dir / "m1" / "run.sh").read_text()
        self.assertIn(str(self.binary), run_sh)
        entry, _ok, _msgs = modelctl.render_llama_swap_entry(dict(self.profile))
        self.assertIn(str(self.binary), entry)


class TestObservationProvenance(LaunchTruthBase):
    """Task B3: every observation carries the full identity of what ran."""

    def test_backend_exposes_every_required_fingerprint(self):
        backend = modelctl_launch.resolve_backend(self.profile)
        for field in ("binary_fingerprint", "environment_fingerprint",
                      "capability_fingerprint"):
            self.assertTrue(getattr(backend, field),
                            f"{field} is empty -- observations cannot be staled by it")

    def test_capability_fingerprint_tracks_reported_features(self):
        caps_a = {"schema": 2, "features": {"moe_weight_transfer_cache": False}}
        caps_b = {"schema": 2, "features": {"moe_weight_transfer_cache": True}}
        fp = modelctl_capabilities.capability_fingerprint
        self.assertNotEqual(fp(caps_a), fp(caps_b))
        self.assertEqual(fp(caps_a), fp(dict(caps_a)))

    def test_capability_fingerprint_ignores_probe_bookkeeping(self):
        # Re-probing the same binary rewrites _version/_binary; that must
        # not look like a capability change and stale every observation.
        base = {"schema": 2, "features": {"moe_cache_metrics": True}}
        fp = modelctl_capabilities.capability_fingerprint
        self.assertEqual(
            fp(base),
            fp({**base, "_version": "b6000", "_binary": "/some/other/path"}))

    def test_probe_status_change_is_a_capability_change(self):
        fp = modelctl_capabilities.capability_fingerprint
        self.assertNotEqual(
            fp({"schema": 2, "features": {}, "_probe_status": "ok"}),
            fp({"schema": 2, "features": {}, "_probe_status": "unsupported"}))


class TestArtifactsExportOnlyProfileEnvironment(LaunchTruthBase):
    """Rendered artifacts must export the profile's environment overrides,
    never the whole inherited process environment."""

    def test_run_sh_does_not_leak_process_env(self):
        with mock.patch.dict("os.environ",
                             {"MODELCTL_TEST_SECRET": "do-not-export"}):
            modelctl.generate_artifacts(dict(self.profile))
        run_sh = (self.profiles_dir / "m1" / "run.sh").read_text()
        self.assertNotIn("MODELCTL_TEST_SECRET", run_sh)

    def test_swap_entry_does_not_leak_process_env(self):
        with mock.patch.dict("os.environ",
                             {"MODELCTL_TEST_SECRET": "do-not-export"}):
            entry, _ok, _msgs = modelctl.render_llama_swap_entry(dict(self.profile))
        self.assertNotIn("MODELCTL_TEST_SECRET", entry)

    def test_launch_environment_still_inherits_process_env(self):
        # The rendered artifacts are narrow; the actual launch env is not.
        with mock.patch.dict("os.environ", {"MODELCTL_TEST_INHERIT": "yes"}):
            backend = modelctl_launch.resolve_backend(self.profile)
        self.assertEqual(backend.environment.get("MODELCTL_TEST_INHERIT"), "yes")
        self.assertNotIn("MODELCTL_TEST_INHERIT", backend.environment_overrides)


if __name__ == "__main__":
    unittest.main()
