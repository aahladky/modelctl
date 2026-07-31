"""Regression tripwires for the 2026-07 code review's critical findings.

Each test fails on the code as it stood before the corresponding fix.
They pin exact failure modes -- missing-profile handling, job-lane
survival, cache-reservation parity on the CLI apply path, the login
flow, auth failing closed, and the Release A offload regex -- not
general feature coverage.
"""
import io
import json
import re
import time
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_baseline
import modelctl_errors
import modelctl_tiers
import modelctl_vram
from modelctl_web import app as web_app
from modelctl_web.jobs import JobRunner, JobStore

from test_launch_truth import LaunchTruthBase
from test_modelctl_web import TOKEN, WebTestBase


class TestLoadProfileContract(unittest.TestCase):
    """load_profile() used to print and sys.exit(1) on a missing profile,
    which killed web job-lane worker threads (SystemExit is not an
    Exception) and turned stale bookmarks into 500s."""

    def test_missing_profile_raises_not_exits(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.object(modelctl, "PROFILES_DIR", Path(tmp)):
                with self.assertRaises(modelctl_errors.ProfileNotFoundError) as cm:
                    modelctl.load_profile("ghost")
        self.assertIn("ghost", str(cm.exception))

    def test_cli_boundary_translates_to_exit_1(self):
        # main() is the one place CLI exit semantics are applied.
        buf = io.StringIO()
        with TemporaryDirectory() as tmp, \
             mock.patch.object(modelctl, "PROFILES_DIR", Path(tmp)), \
             mock.patch("sys.argv", ["modelctl", "test", "ghost"]), \
             redirect_stdout(buf), \
             self.assertRaises(SystemExit) as cm:
            modelctl.main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("No profile named 'ghost'", buf.getvalue())


class TestJobLaneSurvivesSystemExit(unittest.TestCase):
    """A job that raises SystemExit must fail the job, not silently kill
    the lane's only worker thread and strand every queued job forever."""

    def _wait(self, store, job_id, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = store.get(job_id)
            if j["status"] not in ("queued", "running"):
                return j
            time.sleep(0.05)
        self.fail(f"job {job_id} never finished: {store.get(job_id)['status']}")

    def test_lane_survives_and_next_job_runs(self):
        with TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs.db")
            runner = JobRunner(store)

            def boom(ctx):
                raise SystemExit(1)

            def fine(ctx):
                return {"ok": True}

            j1 = runner.submit("t", "boom", boom, lane="mutation")
            j2 = runner.submit("t", "fine", fine, lane="mutation")
            self.assertEqual(self._wait(store, j1)["status"], "failed")
            # The second job only completes if the lane worker survived.
            self.assertEqual(self._wait(store, j2)["status"], "done")


class TestMissingProfileRoutes(WebTestBase):
    """Routes for a deleted/never-existing profile must 404, not
    propagate SystemExit past Starlette as a 500 with a traceback."""

    def test_profile_page_404(self):
        r = self.client.get("/profiles/ghost", headers=self.auth)
        self.assertEqual(r.status_code, 404)

    def test_tiers_preview_404(self):
        r = self.client.get("/profiles/ghost/tiers", headers=self.auth)
        self.assertEqual(r.status_code, 404)

    def test_api_route_404_json(self):
        r = self.client.get("/api/profiles/ghost/plans", headers=self.auth)
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.json())


class TestSmokeTestVerdict(LaunchTruthBase):
    """smoke_test_profile left stage=="generate" on success, so cmd_test's
    failure check fired on every healthy run and the PASS branch was
    unreachable."""

    # Stays alive when launched with server args (the stock script exits
    # immediately on any unknown flag, which reads as a load failure).
    RUNNING_SERVER = """#!/bin/sh
case "$1" in
  --version) echo "version: 6000 (abcdef)"; exit 0 ;;
  *) exec /bin/sleep 30 ;;
esac
"""

    class _FakeResp:
        status = 200

        def __init__(self, body=b""):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(self, req, timeout=None):
        if isinstance(req, str):  # health poll
            return self._FakeResp()
        body = json.dumps({
            "choices": [{"message": {"content": "117"},
                         "finish_reason": "stop"}],
            "timings": {"predicted_per_second": 42.0},
        }).encode()
        return self._FakeResp(body)

    def _successful_run(self):
        self.binary.write_text(self.RUNNING_SERVER)
        (self.profiles_dir / "m1").mkdir(exist_ok=True)
        with mock.patch.object(modelctl.urllib.request, "urlopen",
                               side_effect=self._fake_urlopen):
            return modelctl.smoke_test_profile("m1", timeout=10)

    def test_success_advances_stage_past_generate(self):
        res = self._successful_run()
        self.assertTrue(res["ok"], res["messages"])
        self.assertEqual(res["stage"], "done")

    def test_cmd_test_prints_pass_on_success(self):
        res = self._successful_run()
        buf = io.StringIO()
        args = Namespace(name="m1", evals=None, timeout=10)
        with mock.patch.object(modelctl, "smoke_test_profile",
                               return_value=res), \
             redirect_stdout(buf):
            modelctl.cmd_test(args)
        out = buf.getvalue()
        self.assertIn("PASS", out)
        self.assertNotIn("FAIL", out)

    def test_cmd_test_still_fails_when_generation_request_errors(self):
        res = {"ok": False, "stage": "generate",
               "messages": ["completion request errored: boom"],
               "log_path": "/tmp/t.log", "finish_reason": None,
               "content": None, "tok_per_s": None}
        buf = io.StringIO()
        args = Namespace(name="m1", evals=None, timeout=10)
        with mock.patch.object(modelctl, "smoke_test_profile",
                               return_value=res), \
             redirect_stdout(buf):
            modelctl.cmd_test(args)
        self.assertIn("FAIL", buf.getvalue())


class TestApplyPlanCacheBudgets(unittest.TestCase):
    """The effective-budget write-back is one shared helper; both apply
    paths (CLI and web) must produce the same profile."""

    def test_mode_off_is_untouched(self):
        profile = {"moe_cache": {"mode": "off"}}
        changed = modelctl_tiers.apply_plan_cache_budgets(
            profile, {"cache_budgets": {"SYCL0": 1}}, log=lambda m: None)
        self.assertFalse(changed)
        self.assertNotIn("gpu", profile["moe_cache"])

    def test_matching_budgets_are_untouched(self):
        profile = {"moe_cache": {"mode": "manual",
                                 "gpu": {"budgets_bytes": {"SYCL0": 1}}}}
        changed = modelctl_tiers.apply_plan_cache_budgets(
            profile, {"cache_budgets": {"SYCL0": 1}}, log=lambda m: None)
        self.assertFalse(changed)

    def test_planner_disabled_cache_clears_budgets(self):
        profile = {"moe_cache": {"mode": "manual",
                                 "gpu": {"budgets_bytes": {"SYCL0": 8 << 30}}}}
        changed = modelctl_tiers.apply_plan_cache_budgets(
            profile, {"cache_budgets": {}}, log=lambda m: None)
        self.assertTrue(changed)
        self.assertEqual(profile["moe_cache"]["gpu"]["budgets_bytes"], {})

    def test_effective_budgets_written_back(self):
        logs = []
        profile = {"moe_cache": {"mode": "manual",
                                 "gpu": {"budgets_bytes": {"SYCL0": 8 << 30}}}}
        changed = modelctl_tiers.apply_plan_cache_budgets(
            profile, {"cache_budgets": {"SYCL0": 4 << 30}}, log=logs.append)
        self.assertTrue(changed)
        self.assertEqual(profile["moe_cache"]["gpu"]["budgets_bytes"],
                         {"SYCL0": 4 << 30})
        self.assertTrue(logs)


class TestPlaceTiersCacheParity(unittest.TestCase):
    """cmd_place_tiers planned without cache_request and never wrote the
    effective budgets back, so `place --tiers --apply` produced a static
    placement that collided with the runtime cache at launch."""

    def test_cli_apply_reserves_and_writes_back(self):
        profile = {"name": "m1", "model_path": "/x/m.gguf", "env": ["A=1"],
                   "config": {"ctx": 8192},
                   "moe_cache": {"mode": "manual",
                                 "gpu": {"budgets_bytes": {"SYCL0": 8 << 30}}}}
        plan = {"tier": 2, "config": {"device": "SYCL0"}, "layout": [],
                "warnings": [], "cache_budgets": {"SYCL0": 4 << 30}}
        saved = {}
        args = Namespace(apply=True, no_router_restart=True, no_hermes=True)
        with mock.patch.object(modelctl, "load_profile", return_value=profile), \
             mock.patch.object(modelctl_vram, "system_ram_available",
                               return_value=64 << 30), \
             mock.patch.object(modelctl_tiers, "plan_tiers",
                               return_value=plan) as plan_spy, \
             mock.patch.object(modelctl, "_print_tier_plan",
                               return_value=True), \
             mock.patch.object(modelctl, "save_profile",
                               side_effect=lambda p: saved.update(p)), \
             mock.patch.object(modelctl, "generate_artifacts"), \
             mock.patch.object(modelctl, "sync_all_backends"), \
             mock.patch.object(modelctl, "sync_hermes_custom_providers"), \
             redirect_stdout(io.StringIO()):
            modelctl.cmd_place_tiers(
                args, [{"device": "SYCL0"}], {"vram_limit_pct": 90},
                "SYCL0", ["m1"])
        self.assertEqual(plan_spy.call_args.kwargs.get("cache_request"),
                         profile["moe_cache"],
                         "CLI tier planning must reserve the cache budget")
        self.assertEqual(saved["moe_cache"]["gpu"]["budgets_bytes"],
                         {"SYCL0": 4 << 30},
                         "effective budgets must be written back before save")


class TestAuthFailsClosed(unittest.TestCase):
    def test_empty_token_file_regenerates(self):
        # An empty/whitespace token file used to be returned as "", which
        # matched an anonymous request's empty credentials: auth wide open.
        with TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "web_token"
            token_path.write_text("\n")
            with mock.patch.object(web_app, "TOKEN_PATH", token_path), \
                 mock.patch.dict("os.environ",
                                 {"MODELCTL_WEB_TOKEN": ""}, clear=False):
                tok = web_app.load_or_create_token()
        self.assertTrue(tok.strip(),
                        "empty token file must fail closed, not auth-open")

    def test_whitespace_env_token_rejected(self):
        with TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "web_token"
            with mock.patch.object(web_app, "TOKEN_PATH", token_path), \
                 mock.patch.dict("os.environ",
                                 {"MODELCTL_WEB_TOKEN": "   "}, clear=False):
                tok = web_app.load_or_create_token()
        self.assertTrue(tok.strip())


class TestLoginFlow(WebTestBase):
    """POST /login used Starlette's default 307 redirect, which re-POSTs:
    a correct login ended on 405, a wrong token looped, and a crafted
    ?next= could re-POST a fresh login into a mutating endpoint."""

    def test_successful_login_lands_on_a_get(self):
        # /healthz is GET-only: under the old 307 the follow-up re-POSTed
        # and got 405; under 303 the browser switches to GET and succeeds.
        r = self.client.post(
            "/login", data={"token_field": TOKEN, "next": "/healthz"},
            follow_redirects=True)
        self.assertEqual(r.status_code, 200)

    def test_successful_login_is_303(self):
        r = self.client.post(
            "/login", data={"token_field": TOKEN, "next": "/healthz"},
            follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("set-cookie", r.headers)

    def test_wrong_token_renders_error_not_redirect_loop(self):
        r = self.client.post("/login", data={"token_field": "wrong"},
                             follow_redirects=False)
        self.assertEqual(r.status_code, 401)
        self.assertIn("Wrong token", r.text)

    def test_repeated_failures_backoff(self):
        for _ in range(5):
            r = self.client.post("/login", data={"token_field": "wrong"},
                                 follow_redirects=False)
            self.assertEqual(r.status_code, 401)
        r = self.client.post("/login", data={"token_field": "wrong"},
                             follow_redirects=False)
        self.assertEqual(r.status_code, 429)


class TestBaselineOffloadRegex(unittest.TestCase):
    """The Release A -ot rule used [N-9] as a numeric range; it is a
    character class, so models with >=20 blocks had the LOWER half of
    their expert layers pinned to CPU and the A/B comparison measured a
    placement nobody asked for."""

    def _matched_blocks(self, blocks):
        layout = {"is_moe": True, "block_count": blocks}
        with mock.patch.object(modelctl_vram, "gguf_model_layout",
                               return_value=layout):
            rule = modelctl_baseline._expert_offload_rule(
                {}, {"model_path": "/x/m.gguf"})
        m = re.fullmatch(r"-ot (.+)=CPU", rule)
        self.assertIsNotNone(m, f"unexpected rule shape: {rule!r}")
        pat = re.compile(m.group(1))
        return {i for i in range(blocks)
                if pat.match(f"blk.{i}.ffn_gate_exps.weight")}

    def test_upper_half_pinned_48_blocks(self):
        self.assertEqual(self._matched_blocks(48), set(range(24, 48)))

    def test_upper_half_pinned_94_blocks(self):
        self.assertEqual(self._matched_blocks(94), set(range(47, 94)))

    def test_upper_half_pinned_120_blocks(self):
        # >=100 blocks never matched at all under the old pattern.
        self.assertEqual(self._matched_blocks(120), set(range(60, 120)))

    def test_tiny_models_get_no_rule(self):
        layout = {"is_moe": True, "block_count": 3}
        with mock.patch.object(modelctl_vram, "gguf_model_layout",
                               return_value=layout):
            rule = modelctl_baseline._expert_offload_rule(
                {}, {"model_path": "/x/m.gguf"})
        self.assertEqual(rule, "")


if __name__ == "__main__":
    unittest.main()
