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


class TestProbeEnvHandling(unittest.TestCase):
    """The probe env cache and the relocated-RUNPATH workaround.

    A cold-cache promote used to seed the cache as [env, None], dropping
    every other discovered env script for the life of the process -- a
    second binary needing a different env was then silently declared
    unsupported. And run_llama_probe skipped the LD_LIBRARY_PATH
    workaround every other probe path applies.
    """

    def setUp(self):
        self._saved = modelctl._PROBE_ENV_CACHE
        modelctl._PROBE_ENV_CACHE = None
        self.addCleanup(
            lambda: setattr(modelctl, "_PROBE_ENV_CACHE", self._saved))

    def test_promote_on_cold_cache_keeps_all_candidates(self):
        env_a = {"ONEAPI_ROOT": "/opt/a"}
        env_b = {"ONEAPI_ROOT": "/opt/b"}
        with mock.patch.object(modelctl, "find_env_script_candidates",
                               return_value=["/a.sh", "/b.sh"]), \
             mock.patch.object(
                 modelctl, "source_env_script",
                 side_effect=lambda s: env_a if s == "/a.sh" else env_b):
            modelctl._promote_probe_env(env_b)
        cache = modelctl._PROBE_ENV_CACHE
        self.assertEqual(cache[0], env_b)
        self.assertIn(env_a, cache)
        self.assertIn(None, cache)

    def test_promote_on_warm_cache_reorders(self):
        env_a = {"A": "1"}
        env_b = {"B": "2"}
        modelctl._PROBE_ENV_CACHE = [None, env_a, env_b]
        modelctl._promote_probe_env(env_b)
        self.assertEqual(modelctl._PROBE_ENV_CACHE, [env_b, None, env_a])

    def test_run_llama_probe_applies_library_path_workaround(self):
        import subprocess as _sp
        seen = {}

        def fake_run(cmd, capture_output, text, timeout, env):
            seen["env"] = env
            return _sp.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with TemporaryDirectory() as td:
            binary = Path(td) / "bin" / "llama-server"
            binary.parent.mkdir()
            binary.write_text("#!/bin/sh\nexit 0\n")
            modelctl._PROBE_ENV_CACHE = [None]
            with mock.patch.object(modelctl.subprocess, "run",
                                   side_effect=fake_run):
                modelctl.run_llama_probe(str(binary), ["--version"])
        self.assertIn(str(binary.parent.resolve()),
                      seen["env"].get("LD_LIBRARY_PATH", ""))


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

        caps_dir = root / "capabilities"
        no_binaries = str(root / "no-binaries" / "*")
        for p in (mock.patch.object(modelctl, "PROFILES_DIR", self.profiles_dir),
                  mock.patch.object(modelctl_capabilities, "CAPABILITIES_DIR",
                                    caps_dir),
                  # Hardware fingerprints and preflight auto-fix sweep these
                  # globs and probe whatever they find. On this box that
                  # means the real SYCL builds, which SIGABRT bare-env; the
                  # only binary these tests may execute is self.binary.
                  mock.patch.object(modelctl, "COMMON_LLAMA_SERVER_GLOBS",
                                    [no_binaries]),
                  mock.patch.object(modelctl, "COMMON_ENV_SCRIPT_GLOBS",
                                    [no_binaries])):
            p.start()
            self.addCleanup(p.stop)

        # The probe caches by binary content hash; a stale entry from another
        # test's fake binary would defeat the point of using a real one.
        # Both clears must run inside the CAPABILITIES_DIR patch (cleanups
        # are LIFO), or they delete the REAL probe cache -- which is exactly
        # what this setUp used to do on every test.
        modelctl_capabilities.clear_cache()
        self.addCleanup(modelctl_capabilities.clear_cache)

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

    def test_web_plan_api_has_no_cache_flags(self):
        # Was the /profiles/{name}/run.sh preview page, demolished with the
        # server-rendered console; /api/profiles/{name}/plans is the web
        # surface that still publishes argv. Checking every published plan,
        # not just one, because a cache flag on any of them is launchable
        # from the console's plan table.
        client, auth = self.web_client()
        resp = client.get("/api/profiles/m1/plans", headers=auth)
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        self.assertTrue(rows, "web plan API published no plans at all")
        for row in rows:
            self.assertNoCacheFlags(" ".join(row["argv"]),
                                    f"web plan API ({row['id']})")

    def test_plans_api_says_why_the_command_is_blocked(self):
        # Task B2: validation messages must be visible where the user makes
        # the decision, not only in a job log. The decision now happens on
        # the /v2 per-model page, which reads this endpoint: a plan the
        # resolved backend cannot run comes back in the "unavailable"
        # bucket, naming the capability that is missing.
        client, auth = self.web_client()
        resp = client.get("/api/v2/models/m1/plans", headers=auth)
        self.assertEqual(resp.status_code, 200)
        rows = [r for r in resp.json() if r["source"] == "current-profile"]
        self.assertEqual(len(rows), 1,
                         "per-model plan API dropped the profile as saved")
        self.assertEqual(rows[0]["category"], "unavailable")
        self.assertIn("moe_weight_transfer_cache", rows[0]["reason"])

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

        # 1. web surface (the console's plan API)
        # Was the /profiles/{name}/run.sh preview page. That page rendered
        # the whole canonical argv; the API publishes the plan's option
        # vector without argv[0], because the binary is whatever
        # resolve_backend() picks at launch -- frequently not the one the
        # profile names. argv[0] is put back below from the binary this
        # fixture actually pinned, never from the row itself, so a surface
        # that quietly resolved a different binary still fails here.
        client, auth = self.web_client()
        resp = client.get("/api/profiles/m1/plans", headers=auth)
        self.assertEqual(resp.status_code, 200)
        rows = [r for r in resp.json() if r["source"] == "current-profile"]
        self.assertEqual(len(rows), 1,
                         "web plan API dropped the profile exactly as saved")
        browser_argv = tuple(rows[0]["argv"])

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

        # The web surface publishes argv as JSON; the two rendered
        # surfaces are shell text, so compare those by re-tokenizing
        # rather than by fingerprint.
        expected = list(cli_cmd.argv)
        self.assertEqual([str(self.binary)] + list(browser_argv), expected,
                         "web plan API differs from the canonical command")
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


# Answers the capability probe only when the marker variable is present:
# a stand-in for a SYCL build that cannot even start without its oneAPI
# environment.
ENV_DEPENDENT_SERVER = """#!/bin/sh
case "$1" in
  --version) echo "version: 6000 (abcdef)"; exit 0 ;;
  --modelctl-capabilities)
    if [ "$MODELCTL_TEST_REQUIRES_ME" = "1" ]; then
      echo '{"schema": 2, "features": {}}'
      exit 0
    fi
    exit 1 ;;
  *) echo "error: invalid argument: $1" >&2; exit 1 ;;
esac
"""


class TestLaunchEnvironmentIsPartOfIdentity(LaunchTruthBase):
    """P0 launch-contract tests: the complete final environment lives
    inside the immutable LaunchCommand, and the behavior-affecting subset
    of it is part of command identity.

    The motivating hardware measurement: changing
    GGML_OP_OFFLOAD_MOE_MIN_BATCH alone reverses the performance ranking
    between storage-bound and VRAM-resident placements. Two commands that
    differ only in it must not share a fingerprint, reuse each other's
    observations, or preview identically.
    """

    def build(self, plan_env=None, backend=None):
        import dataclasses
        backend = backend or modelctl_launch.resolve_backend(self.profile)
        plan = modelctl_plans.current_profile_plan(
            self.profile, capabilities=backend.capabilities)
        if plan_env is not None:
            plan = dataclasses.replace(plan, env=plan_env)
        return modelctl_launch.build_launch_command(
            self.profile, plan, backend=backend)

    def test_plan_env_is_inside_the_launch_environment(self):
        # No caller assembly required: what the plan sets is already in
        # the environment the command carries.
        launch = self.build(plan_env={"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "64"})
        self.assertEqual(
            launch.environment.get("GGML_OP_OFFLOAD_MOE_MIN_BATCH"), "64")

    def test_offload_threshold_changes_command_identity(self):
        backend = modelctl_launch.resolve_backend(self.profile)
        base = self.build(backend=backend)
        tuned = self.build(
            plan_env={"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "1"}, backend=backend)
        self.assertEqual(base.argv, tuned.argv,
                         "fixture drift: argv must be identical so only "
                         "the environment differs")
        self.assertNotEqual(base.environment_fingerprint,
                            tuned.environment_fingerprint)
        self.assertNotEqual(base.command_fingerprint, tuned.command_fingerprint)

    def test_plan_env_cannot_drop_the_library_path(self):
        backend = modelctl_launch.resolve_backend(self.profile)
        expected = backend.environment.get("LD_LIBRARY_PATH", "")
        launch = self.build(plan_env={"LD_LIBRARY_PATH": "/nowhere"},
                            backend=backend)
        got = launch.environment.get("LD_LIBRARY_PATH", "")
        # The plan's value survives, but the binary's own directory is
        # re-appended so the process can still load its runtime.
        self.assertIn(str(Path(self.binary).parent), got)
        if expected:
            self.assertNotEqual(got, expected)

    def test_ambient_noise_does_not_change_identity(self):
        base = self.build()
        with mock.patch.dict("os.environ", {"SSH_CLIENT": "10.0.0.5 1 22",
                                            "LC_ALL": "C.UTF-8"}):
            noisy = self.build()
        self.assertEqual(base.environment_fingerprint,
                         noisy.environment_fingerprint)
        self.assertEqual(base.command_fingerprint, noisy.command_fingerprint)

    def test_ambient_behavior_variable_changes_identity(self):
        # The profile never declared it, but it changes what the runtime
        # does -- so it is part of identity (this is what the old
        # profile-declared-only fingerprint missed).
        base = self.build()
        with mock.patch.dict("os.environ",
                             {"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "1"}):
            ambient = self.build()
        self.assertNotEqual(base.environment_fingerprint,
                            ambient.environment_fingerprint)

    def test_identity_environment_is_a_normalized_whitelist(self):
        ident = modelctl_launch.identity_environment({
            "GGML_OP_OFFLOAD_MOE_MIN_BATCH": "32",
            "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
            "LD_LIBRARY_PATH": "/opt/oneapi/lib",
            "SSH_CLIENT": "10.0.0.5",       # noise: excluded
            "LANG": "en_US.UTF-8",          # noise: excluded
            "ZE_AFFINITY_MASK": "",          # empty: excluded
        })
        self.assertEqual(ident, {
            "GGML_OP_OFFLOAD_MOE_MIN_BATCH": "32",
            "LD_LIBRARY_PATH": "/opt/oneapi/lib",
            "ONEAPI_DEVICE_SELECTOR": "level_zero:0",
        })

    def test_fingerprint_describes_the_carried_environment(self):
        # Self-consistency: the fingerprint is derivable from the object,
        # so no hidden inputs exist.
        launch = self.build(plan_env={"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "8"})
        expected = modelctl_launch._env_fingerprint(
            modelctl_launch.identity_environment(launch.environment))
        self.assertEqual(launch.environment_fingerprint, expected)

    def test_probe_runs_under_the_resolved_environment(self):
        # A binary that cannot answer the probe without its environment
        # (like a SYCL build without oneAPI) must be probed with the env
        # it will launch under, not the bare process env.
        self.binary.write_text(ENV_DEPENDENT_SERVER)
        modelctl_capabilities.clear_cache()
        profile = dict(self.profile)
        profile["env"] = ["MODELCTL_TEST_REQUIRES_ME=1"]
        backend = modelctl_launch.resolve_backend(profile)
        self.assertEqual(backend.capabilities.get("_probe_status"), "ok",
                         "probe did not run under the resolved launch "
                         "environment")


if __name__ == "__main__":
    unittest.main()
