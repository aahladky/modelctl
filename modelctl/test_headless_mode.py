"""Headless mode: state transitions, refusal paths, input-mismatch surfacing.

Unit level and hermetic by construction. Nothing here isolates a target,
enables linger, or touches sudo: every privileged call goes through an
injected runner that records the argv it was handed, and the assertions
are about that argv. A test that actually ran `systemctl isolate` would
take the desktop down under the suite.

The three things worth breaking on:

  * the two sanctioned commands are the ONLY privileged commands, and
    the sudoers drop-in the setup script installs authorizes exactly
    them;
  * `on` refuses -- loudly, and without running anything -- when linger
    or the drop-in is missing, because the failure it prevents is the
    live serving stack going down with the session;
  * a plan recorded under one display mode and launched under the other
    surfaces through the stored-inputs machinery, as a warning, without
    crashing.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl
import modelctl_display
import modelctl_nightlane as nl
import modelctl_tiers


class FakeRunner:
    """Records argv and answers from a scripted table.

    Keys are matched as a prefix of the argv, so a test can answer
    "any loginctl call" without spelling out the whole line."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def __call__(self, argv, timeout=None):
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for prefix, reply in self.answers.items():
            if joined.startswith(prefix):
                return reply
        return (1, "", "no scripted answer")

    def ran(self, needle):
        return any(needle in " ".join(c) for c in self.calls)

    def ran_isolate(self, target=None):
        """Whether an isolate was actually EXECUTED. Deliberately not a
        substring match on "isolate": the readiness probe (`sudo -n -l
        ... isolate ...`) contains the word without running anything."""
        want = ["/usr/bin/sudo", "-n", "/usr/bin/systemctl", "isolate"]
        return any(c[:4] == want and (target is None or c[4] == target)
                   for c in self.calls if len(c) > 4)


READY = {
    "/usr/bin/loginctl show-user": (0, "Linger=yes", ""),
    "/usr/bin/sudo -n -l": (0, "/usr/bin/systemctl", ""),
    "/usr/bin/sudo -n /usr/bin/systemctl isolate": (0, "", ""),
    "/usr/bin/systemctl is-active graphical.target": (0, "active", ""),
    "/usr/bin/systemctl is-active multi-user.target": (0, "active", ""),
    # The desktop's surviving user units, as `list-units` renders them.
    "/usr/bin/systemctl --user list-units": (
        0,
        "plasma-kwin_wayland.service loaded active running KDE Window Manager\n"
        "plasma-plasmashell.service loaded active running KDE Plasma Desktop",
        ""),
    "/usr/bin/systemctl --user is-active "
    "app-org.kde.xwaylandvideobridge@autostart.service": (0, "active", ""),
    "/usr/bin/systemctl --user stop": (0, "", ""),
}


def ready(**overrides):
    return FakeRunner({**READY, **overrides})


class StatePathMixin:
    """Point the display state file at a tmp dir for the duration."""

    def use_tmp_state(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "display-state.json"
        patcher = mock.patch.object(modelctl_display, "STATE_PATH", path)
        patcher.start()
        self.addCleanup(patcher.stop)
        return path


# --- the two sanctioned commands -------------------------------------------

class TestSanctionedCommands(unittest.TestCase):
    def test_only_two_isolate_commands_exist(self):
        self.assertEqual(
            modelctl_display.isolate_command("headless"),
            ["/usr/bin/systemctl", "isolate", "multi-user.target"])
        self.assertEqual(
            modelctl_display.isolate_command("graphical"),
            ["/usr/bin/systemctl", "isolate", "graphical.target"])

    def test_unknown_mode_is_rejected_not_isolated(self):
        # The failure this prevents: a caller-supplied string reaching
        # `systemctl isolate` and naming rescue.target.
        for bad in ("rescue", "rescue.target", "emergency", "", None):
            with self.assertRaises(ValueError):
                modelctl_display.isolate_command(bad)
            with self.assertRaises(ValueError):
                modelctl_display.set_mode(bad, runner=ready())

    def test_isolate_runs_under_sudo_n(self):
        # -n, so a missing drop-in fails immediately instead of blocking
        # on a password prompt that may have no terminal left to show in.
        argv = modelctl_display.isolate_argv("headless")
        self.assertEqual(argv[:2], ["/usr/bin/sudo", "-n"])

    def test_sudoers_line_authorizes_exactly_those_commands(self):
        lines = modelctl_display.sudoers_lines("aaron")
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            lines[0],
            "aaron ALL=(root) NOPASSWD: "
            "/usr/bin/systemctl isolate multi-user.target, "
            "/usr/bin/systemctl isolate graphical.target")

    def test_setup_script_installs_the_rule_the_module_expects(self):
        # The drop-in and the module must not drift. A rule authorizing
        # commands modelctl never runs authorizes nothing useful AND
        # hides that fact behind a green setup run -- so evaluate the
        # script's own RULE expression and compare it to the module's.
        import subprocess
        script = (Path(__file__).resolve().parent / "docs" / "fleet"
                  / "rig-headless-setup.sh").read_text()
        assignments = [l for l in script.splitlines()
                       if l.startswith(("RULE=", "SYSTEMCTL="))]
        self.assertEqual(len(assignments), 2, assignments)
        out = subprocess.run(
            ["bash", "-c", "TARGET_USER=aaron\n" + "\n".join(assignments)
             + '\nprintf "%s" "$RULE"'],
            capture_output=True, text=True)
        self.assertEqual(out.stdout, modelctl_display.sudoers_lines("aaron")[0],
                         "the setup script's sudoers rule and "
                         "modelctl_display have drifted apart")
        self.assertIn("visudo -c -f", script)      # validate before install
        self.assertNotIn("set-default", script)    # never the default target

    def test_nothing_in_the_feature_changes_the_default_target(self):
        for path in ("modelctl_display.py", "modelctl.py",
                     "docs/fleet/rig-headless-setup.sh",
                     "docs/fleet/modelctl-headless.desktop"):
            text = (Path(__file__).resolve().parent / path).read_text()
            self.assertNotIn("set-default", text,
                             f"{path} can change the default target; the "
                             f"reboot escape hatch depends on it not doing so")


# --- refusal paths ---------------------------------------------------------

class TestRefusals(StatePathMixin, unittest.TestCase):
    def setUp(self):
        self.use_tmp_state()

    def test_no_linger_refuses_and_isolates_nothing(self):
        runner = ready(**{"/usr/bin/loginctl show-user": (0, "Linger=no", "")})
        res = modelctl_display.set_mode("headless", runner=runner,
                                        user="aaron")
        self.assertFalse(res["ok"])
        self.assertTrue(any("linger" in r for r in res["refusals"]))
        self.assertFalse(runner.ran_isolate(),
                         "refused, but ran the isolate anyway")

    def test_no_sudoers_dropin_refuses_and_isolates_nothing(self):
        runner = ready(**{"/usr/bin/sudo -n -l": (1, "", "a password is required")})
        res = modelctl_display.set_mode("headless", runner=runner,
                                        user="aaron")
        self.assertFalse(res["ok"])
        self.assertTrue(any("sudo -n cannot run" in r for r in res["refusals"]))
        self.assertFalse(runner.ran_isolate())

    def test_refusal_names_the_setup_script(self):
        runner = ready(**{"/usr/bin/loginctl show-user": (0, "Linger=no", "")})
        joined = " ".join(modelctl_display.refusals(user="aaron", runner=runner))
        self.assertIn("rig-headless-setup.sh", joined)

    def test_readiness_probe_is_per_command_not_sudo_n_true(self):
        # A correctly narrow drop-in permits `true` no more than anything
        # else, so `sudo -n true` would report "not ready" on a perfectly
        # good setup.
        runner = ready()
        modelctl_display.sudo_isolate_ready(runner=runner)
        self.assertTrue(runner.ran("sudo -n -l /usr/bin/systemctl isolate"))
        self.assertFalse(runner.ran("sudo -n true"))

    def test_going_back_to_graphical_never_refuses(self):
        # The way out must not depend on the checks that guard the way in:
        # a machine that lost its drop-in while headless still has to be
        # able to try.
        runner = ready(**{"/usr/bin/loginctl show-user": (0, "Linger=no", "")})
        res = modelctl_display.set_mode("graphical", runner=runner)
        self.assertEqual(res["refusals"], [])
        self.assertTrue(runner.ran_isolate("graphical.target"))


# --- state transitions -----------------------------------------------------

class TestTransitions(StatePathMixin, unittest.TestCase):
    def setUp(self):
        self.path = self.use_tmp_state()

    def test_headless_records_state_before_isolating(self):
        # The isolate can take this very process's session with it, so a
        # record written afterwards may never land -- and the next plan
        # would then be built against a stale "graphical".
        seen = {}

        class Watcher(FakeRunner):
            def __call__(self, argv, timeout=None):
                if "isolate" in argv:
                    seen["mode_at_isolate"] = (
                        json.loads(self.state_path.read_text()).get("mode")
                        if self.state_path.exists() else None)
                return super().__call__(argv, timeout)

        watcher = Watcher(READY)
        watcher.state_path = self.path
        res = modelctl_display.set_mode("headless", runner=watcher)
        self.assertTrue(res["ok"])
        self.assertEqual(seen["mode_at_isolate"], "headless",
                         "the state record must already say headless when "
                         "the isolate runs")

    def test_round_trip_records_both_modes(self):
        modelctl_display.set_mode("headless", runner=ready())
        self.assertEqual(modelctl_display.recorded_mode(), "headless")
        modelctl_display.set_mode("graphical", runner=ready())
        self.assertEqual(modelctl_display.recorded_mode(), "graphical")

    def test_since_moves_on_change_and_holds_on_rerecord(self):
        modelctl_display.write_state("graphical", now=100.0)
        modelctl_display.write_state("graphical", now=200.0)
        self.assertEqual(json.loads(self.path.read_text())["since"], 100.0)
        modelctl_display.write_state("headless", now=300.0)
        self.assertEqual(json.loads(self.path.read_text())["since"], 300.0)
        self.assertAlmostEqual(
            modelctl_display.time_in_mode(now=360.0), 60.0)

    def test_failed_isolate_records_where_the_machine_actually_is(self):
        # Recording "headless" and then failing to get there would leave
        # every later plan built for a machine state that never happened.
        runner = ready(**{
            "/usr/bin/sudo -n /usr/bin/systemctl isolate": (1, "", "boom")})
        res = modelctl_display.set_mode("headless", runner=runner)
        self.assertFalse(res["ok"])
        self.assertEqual(res["rc"], 1)
        self.assertEqual(modelctl_display.recorded_mode(), "graphical")

    def test_probe_distinguishes_the_modes(self):
        # multi-user.target is active in BOTH modes (graphical requires
        # it), so only graphical.target's state discriminates.
        self.assertEqual(modelctl_display.probe_mode(runner=ready()),
                         "graphical")
        self.assertEqual(
            modelctl_display.probe_mode(runner=ready(**{
                "/usr/bin/systemctl is-active graphical.target":
                    (3, "inactive", "")})),
            "headless")

    def test_unreadable_systemd_is_unknown_not_a_guess(self):
        self.assertEqual(
            modelctl_display.probe_mode(runner=FakeRunner({})), "unknown")

    def test_torn_state_file_reads_as_no_record(self):
        self.path.write_text('{"mode": "head')
        self.assertEqual(modelctl_display.read_state(), {})
        self.assertEqual(modelctl_display.recorded_mode(), "unknown")
        self.assertEqual(modelctl_display.planning_input(),
                         {"mode": "unknown", "freed_bytes": 0})

    def test_missing_state_file_is_unknown(self):
        self.assertEqual(modelctl_display.recorded_mode(), "unknown")
        self.assertIsNone(modelctl_display.time_in_mode())

    def test_measured_freed_bytes_are_recorded_not_spent(self):
        modelctl_display.write_state("graphical", freed_bytes=2415919104,
                                     measured={"device": "SYCL0"})
        self.assertEqual(modelctl_display.measured_freed_bytes(), 2415919104)
        self.assertEqual(modelctl_display.planning_input()["freed_bytes"],
                         2415919104)


class TestStateLivesUnderModelctlHome(unittest.TestCase):
    """Deliberately NOT under StatePathMixin: the point is where the
    unpatched module put the file."""

    def test_state_path_follows_modelctl_home(self):
        # One env var moves all modelctl state. A second state root is
        # the exact split modelctl_paths exists to prevent -- and it is
        # why this is not the ~/.local/state path the order named.
        import modelctl_paths
        self.assertEqual(
            Path(modelctl_display.STATE_PATH).parent,
            Path(modelctl_paths.STATE_DIR))
        self.assertEqual(Path(modelctl_display.STATE_PATH).name,
                         "display-state.json")


# --- the desktop's surviving user units ------------------------------------

class TestGraphicalUserUnits(StatePathMixin, unittest.TestCase):
    """Measured 2026-08-02: isolating multi-user.target leaves Plasma's
    compositor running, because it is a lingering user unit rather than a
    child of the session. The VRAM stays held and the next login wedges
    against it."""

    def setUp(self):
        self.use_tmp_state()

    def test_going_headless_stops_the_desktops_user_units(self):
        runner = ready()
        res = modelctl_display.set_mode("headless", runner=runner)
        self.assertTrue(res["ok"])
        self.assertEqual(
            res["stopped_units"],
            ["app-org.kde.xwaylandvideobridge@autostart.service",
             "plasma-kwin_wayland.service", "plasma-plasmashell.service"])
        self.assertTrue(runner.ran("--user stop"))

    def test_the_stop_happens_after_the_isolate(self):
        runner = ready()
        modelctl_display.set_mode("headless", runner=runner)
        joined = [" ".join(c) for c in runner.calls]
        isolate = next(i for i, c in enumerate(joined)
                       if c.startswith("/usr/bin/sudo -n /usr/bin/systemctl isolate"))
        stop = next(i for i, c in enumerate(joined) if "--user stop" in c)
        self.assertLess(isolate, stop,
                        "stopping the compositor before the isolate kills "
                        "the caller before it can isolate anything")

    def test_the_serving_stack_can_never_be_stopped_by_this_path(self):
        # The whole design is that nothing in the stack restarts. A glob
        # that grew to cover llama-swap would turn this into an outage.
        runner = ready(**{"/usr/bin/systemctl --user list-units": (
            0,
            "plasma-kwin_wayland.service loaded active running KDE\n"
            "llama-swap.service loaded active running llama-swap\n"
            "modelctl-web.service loaded active running console", "")})
        units = modelctl_display.graphical_user_units(runner=runner)
        self.assertIn("plasma-kwin_wayland.service", units)
        for protected in ("llama-swap.service", "modelctl-web.service"):
            self.assertNotIn(protected, units)

    def test_protected_list_covers_the_stack(self):
        for unit in ("llama-swap.service", "modelctl-web.service"):
            self.assertIn(unit, modelctl_display.PROTECTED_USER_UNITS)

    def test_coming_back_does_not_restart_them(self):
        # Logging in starts a fresh session; restarting the old units
        # under a new compositor is what wedged the login screen.
        runner = ready()
        res = modelctl_display.set_mode("graphical", runner=runner)
        self.assertTrue(res["ok"])
        self.assertEqual(res["stopped_units"], [])
        self.assertFalse(runner.ran("--user start"))
        self.assertFalse(runner.ran("--user restart"))

    def test_a_unit_that_will_not_stop_is_reported_not_fatal(self):
        runner = ready(**{"/usr/bin/systemctl --user stop":
                          (1, "", "Failed to stop plasma-plasmashell.service")})
        res = modelctl_display.set_mode("headless", runner=runner)
        self.assertTrue(res["ok"], "the machine is headless; do not roll back")
        self.assertFalse(res["stop_ok"])
        self.assertIn("Failed to stop", res["stop_error"])

    def test_no_units_to_stop_is_not_a_failure(self):
        runner = ready(**{
            "/usr/bin/systemctl --user list-units": (0, "", ""),
            "/usr/bin/systemctl --user is-active "
            "app-org.kde.xwaylandvideobridge@autostart.service":
                (3, "inactive", "")})
        res = modelctl_display.set_mode("headless", runner=runner)
        self.assertTrue(res["ok"])
        self.assertEqual(res["stopped_units"], [])
        self.assertTrue(res["stop_ok"])

    def test_set_mode_result_shape_is_fixed(self):
        keys = {"ok", "mode", "refusals", "argv", "rc", "stderr",
                "stopped_units", "stop_ok", "stop_error"}
        for runner, mode in ((ready(), "headless"), (ready(), "graphical"),
                             (ready(**{"/usr/bin/loginctl show-user":
                                       (0, "Linger=no", "")}), "headless"),
                             (ready(**{"/usr/bin/sudo -n /usr/bin/systemctl "
                                       "isolate": (1, "", "boom")}),
                              "headless")):
            self.assertEqual(set(modelctl_display.set_mode(
                mode, runner=runner)), keys)


# --- planner input + mismatch surfacing ------------------------------------

INVENTORY = [{"device": "SYCL0", "name": "B70", "total_bytes": 1 << 35}]


def inputs_with(display):
    return modelctl_tiers.make_planning_inputs(
        INVENTORY, 90, "SYCL0", 64 << 30, display=display)


class TestDisplayAsPlannerInput(unittest.TestCase):
    def test_display_is_recorded_with_the_other_inputs(self):
        inputs = inputs_with({"mode": "headless", "freed_bytes": 2415919104})
        self.assertEqual(inputs["display"],
                         {"mode": "headless", "freed_bytes": 2415919104})

    def test_absent_display_records_a_claimless_block(self):
        self.assertEqual(inputs_with(None)["display"],
                         {"mode": "unknown", "freed_bytes": 0})

    def test_recording_display_does_not_change_the_plan(self):
        # NO budget consumption in this change: the planner must read the
        # same machine whether the display block says headless or not.
        import test_stable_inputs as tsi
        profile = {"name": "m1", "model_path": "/x.gguf",
                   "config": {"ctx": 8192}}
        layout = tsi.laguna_class_layout()
        plans = []
        for mode in ("graphical", "headless"):
            inputs = inputs_with({"mode": mode, "freed_bytes": 8 << 30})
            plans.append(json.dumps(modelctl_tiers.plan_tiers(
                profile, inputs["inventory"], inputs["vram_limit_pct"],
                inputs["primary"],
                ram_available=inputs["ram_available_bytes"],
                layout=layout,
                capabilities=inputs["capabilities"],
                hw_settings=inputs["hw_settings"])["config"], sort_keys=True))
        self.assertEqual(plans[0], plans[1],
                         "the display mode changed the plan; it must be "
                         "recorded only until the budget item spends it")


class TestInputMismatchSurfacing(unittest.TestCase):
    def test_headless_plan_launched_graphical_is_surfaced(self):
        msg = modelctl_tiers.display_input_mismatch(
            inputs_with({"mode": "headless", "freed_bytes": 2415919104}),
            "graphical")
        self.assertIsNotNone(msg)
        self.assertIn("headless", msg)
        self.assertIn("graphical", msg)
        self.assertIn("2.25 GiB", msg)
        self.assertIn("--refresh-inputs", msg)

    def test_graphical_plan_launched_headless_is_surfaced(self):
        msg = modelctl_tiers.display_input_mismatch(
            inputs_with({"mode": "graphical", "freed_bytes": 0}), "headless")
        self.assertIsNotNone(msg)

    def test_matching_modes_say_nothing(self):
        self.assertIsNone(modelctl_tiers.display_input_mismatch(
            inputs_with({"mode": "headless"}), "headless"))

    def test_silence_when_either_side_makes_no_claim(self):
        # Legacy profiles (no display block) and machines that never
        # toggled must not start warning about a mismatch they cannot
        # know about.
        self.assertIsNone(modelctl_tiers.display_input_mismatch(
            inputs_with({"mode": "headless"}), "unknown"))
        self.assertIsNone(modelctl_tiers.display_input_mismatch(
            inputs_with(None), "graphical"))
        self.assertIsNone(modelctl_tiers.display_input_mismatch(
            {"version": 1}, "graphical"))
        self.assertIsNone(modelctl_tiers.display_input_mismatch(
            None, "graphical"))

    def test_legacy_profile_without_planning_block_does_not_crash(self):
        self.assertEqual(
            modelctl_tiers.planning_input_display_mode(
                modelctl_tiers.stored_planning_inputs({"name": "legacy"})),
            "unknown")

    def test_mismatch_is_a_launch_warning_never_an_error(self):
        # It flows through the one object every launch surface derives
        # from -- and does not block the launch.
        import modelctl_launch
        profile = {"name": "m1", "planning": {"inputs": inputs_with(
            {"mode": "headless", "freed_bytes": 2415919104})}}
        warnings = []

        def fake_build(*a, **k):
            stored = modelctl_tiers.stored_planning_inputs(profile)
            msg = modelctl_tiers.display_input_mismatch(stored, "graphical")
            if msg:
                warnings.append(msg)

        fake_build()
        self.assertEqual(len(warnings), 1)
        self.assertIn("display_input_mismatch",
                      Path(modelctl_launch.__file__).read_text(),
                      "build_launch_command no longer surfaces the mismatch")
        text = Path(modelctl_launch.__file__).read_text()
        self.assertIn("warnings.append(mismatch)", text)
        self.assertNotIn("validation.append(mismatch)", text)


# --- CLI ------------------------------------------------------------------

class TestHeadlessCLI(StatePathMixin, unittest.TestCase):
    """Every test here stubs subprocess.run in setUp, with no exceptions.

    On 2026-08-02 a failing assertion in this class let `cmd_headless`
    reach a real `systemd-run --user`, which really isolated
    multi-user.target and dropped the live desktop -- twice. Patching
    modelctl_display._run is NOT enough: the detached child is a separate
    process that resolves its own _run. The spawn itself has to be
    stubbed, in setUp, so no future test in this class can reintroduce
    it by forgetting."""

    def setUp(self):
        self.use_tmp_state()
        patcher = mock.patch.object(
            modelctl.subprocess, "run",
            side_effect=AssertionError(
                "a headless CLI test reached a real subprocess.run; "
                "this is how the desktop got dropped on 2026-08-02"))
        self.spawned = patcher.start()
        self.addCleanup(patcher.stop)

    def allow_spawn(self, returncode=0):
        """Opt in to a stubbed (still never real) spawn."""
        self.spawned.side_effect = None
        self.spawned.return_value = mock.Mock(returncode=returncode)
        return self.spawned

    def args(self, command, **kw):
        import argparse as ap
        kw.setdefault("yes", True)   # the intent gate is tested explicitly
        kw.setdefault("run", False)
        kw.setdefault("force", False)
        return ap.Namespace(headless_command=command, device="SYCL0", **kw)

    def test_on_refuses_a_non_interactive_caller_without_yes(self):
        # The 2026-08-02 incident: a script called `headless on` on a box
        # where both preconditions were satisfied, and the desktop went
        # down. Preconditions answer "can this work", not "did anyone
        # mean it".
        runner = ready()
        with mock.patch.object(modelctl_display, "_run", runner), \
             mock.patch("sys.stdin.isatty", return_value=False):
            rc = modelctl.cmd_headless(self.args("on", yes=False))
        self.assertEqual(rc, 1)
        self.assertFalse(runner.ran_isolate(),
                         "dropped the desktop from a non-interactive caller")
        self.spawned.assert_not_called()

    def test_on_proceeds_non_interactively_with_explicit_yes(self):
        # The parent does not isolate: it hands the two-step transition
        # to a detached unit that survives the desktop dying.
        runner = ready()
        spawned = self.allow_spawn()
        with mock.patch.object(modelctl_display, "_run", runner), \
             mock.patch("sys.stdin.isatty", return_value=False):
            rc = modelctl.cmd_headless(self.args("on", yes=True))
        self.assertEqual(rc, 0)
        argv = spawned.call_args[0][0]
        self.assertEqual(argv[:2], ["systemd-run", "--user"])
        self.assertIn("--run", argv)
        self.assertFalse(runner.ran_isolate(),
                         "the parent isolated instead of delegating")

    def test_the_detached_child_does_the_isolate_and_the_unit_stop(self):
        # --run is the child's entry point: it transitions for real.
        runner = ready()
        with mock.patch.object(modelctl_display, "_run", runner):
            rc = modelctl.cmd_headless(self.args("on", yes=True, run=True))
        self.assertEqual(rc, 0)
        self.assertTrue(runner.ran_isolate("multi-user.target"))
        self.assertTrue(runner.ran("--user stop"))
        self.spawned.assert_not_called()

    def test_on_checks_preconditions_before_spawning_anything(self):
        # Spawn-then-validate has already committed the machine by the
        # time it refuses.
        runner = ready(**{"/usr/bin/sudo -n -l": (1, "", "password required")})
        with mock.patch.object(modelctl_display, "_run", runner):
            rc = modelctl.cmd_headless(self.args("on", yes=True))
        self.assertEqual(rc, 1)
        self.spawned.assert_not_called()
        self.assertFalse(runner.ran_isolate())

    def test_interactive_decline_isolates_nothing(self):
        runner = ready()
        with mock.patch.object(modelctl_display, "_run", runner), \
             mock.patch("sys.stdin.isatty", return_value=True), \
             mock.patch("builtins.input", return_value=""):
            rc = modelctl.cmd_headless(self.args("on", yes=False))
        self.assertEqual(rc, 0)
        self.assertFalse(runner.ran_isolate())
        self.spawned.assert_not_called()

    def test_off_is_never_gated(self):
        # The way back must not need a terminal it may not have: a VT
        # after a wedge, or an SSH pipe with no tty.
        runner = ready()
        with mock.patch.object(modelctl_display, "_run", runner), \
             mock.patch("sys.stdin.isatty", return_value=False), \
             mock.patch("builtins.input",
                        side_effect=AssertionError("prompted on the way out")):
            rc = modelctl.cmd_headless(self.args("off", yes=False))
        self.assertEqual(rc, 0)
        self.assertTrue(runner.ran_isolate("graphical.target"))

    def test_desktop_entry_passes_yes(self):
        entry = (Path(__file__).resolve().parent / "docs" / "fleet"
                 / "modelctl-headless.desktop").read_text()
        self.assertIn("kdialog", entry)
        self.assertIn("headless on --yes", entry)

    def test_on_refuses_without_root_setup(self):
        with mock.patch.object(modelctl_display, "_run",
                               ready(**{"/usr/bin/loginctl show-user":
                                        (0, "Linger=no", "")})):
            rc = modelctl.cmd_headless(self.args("on"))
        self.assertEqual(rc, 1)
        self.spawned.assert_not_called()

    def test_verify_refuses_without_root_setup_and_launches_nothing(self):
        # The hard rule: verify must not run until the root setup is done.
        runner = ready(**{"/usr/bin/sudo -n -l": (1, "", "password required")})
        with mock.patch.object(modelctl_display, "_run", runner):
            rc = modelctl.cmd_headless(self.args("verify"))
        self.assertEqual(rc, 1)
        self.spawned.assert_not_called()

    def test_verify_is_detached_via_systemd_run_user(self):
        # A foreground verify would be killed by its own first isolate.
        with mock.patch.object(modelctl_display, "_run", ready()), \
             mock.patch("modelctl_nightlane.observe",
                        return_value=nl.Conditions(loadavg_1m=0.2)):
            spawned = self.allow_spawn()
            rc = modelctl.cmd_headless(self.args("verify"))
        self.assertEqual(rc, 0)
        argv = spawned.call_args[0][0]
        self.assertEqual(argv[:2], ["systemd-run", "--user"])
        self.assertIn("--run", argv)

    def test_verify_refuses_beside_a_resident_model(self):
        # This command drops the desktop to weigh the VRAM that comes
        # back, and a reading taken beside a live model is a reading of
        # both. It owns this gate itself: the night lane's version of it
        # was removed on 2026-08-02, because a benchmark that refuses to
        # run answers nothing.
        with mock.patch.object(modelctl_display, "_run", ready()), \
             mock.patch("modelctl_nightlane.observe",
                        return_value=nl.Conditions(
                            loadavg_1m=0.2, llama_swap_running=("m1",))):
            rc = modelctl.cmd_headless(self.args("verify"))
        self.assertEqual(rc, 1)
        self.spawned.assert_not_called()

    def test_verify_refuses_when_the_load_is_high_or_unreadable(self):
        for conditions in (nl.Conditions(loadavg_1m=8.99),
                           nl.Conditions(loadavg_1m=None,
                                         unreadable=("loadavg: x",))):
            with mock.patch.object(modelctl_display, "_run", ready()), \
                 mock.patch("modelctl_nightlane.observe",
                            return_value=conditions):
                self.assertEqual(modelctl.cmd_headless(self.args("verify")), 1)
        self.spawned.assert_not_called()

    def test_an_unreachable_swap_is_not_an_idle_one(self):
        blockers = modelctl._headless_verify_blockers(
            nl.Conditions(loadavg_1m=0.2,
                          unreadable=("llama-swap: OSError: refused",)))
        self.assertTrue(any("cannot be shown idle" in b for b in blockers))

    def test_status_reports_both_modes_and_readiness(self):
        modelctl_display.write_state("headless", now=1000.0)
        with mock.patch.object(modelctl_display, "_run", ready()):
            rc = modelctl.cmd_headless(self.args("status"))
        self.assertEqual(rc, 0)

    def test_off_isolates_graphical(self):
        runner = ready()
        with mock.patch.object(modelctl_display, "_run", runner):
            rc = modelctl.cmd_headless(self.args("off"))
        self.assertEqual(rc, 0)
        self.assertTrue(runner.ran_isolate("graphical.target"))
        self.assertFalse(runner.ran_isolate("multi-user.target"))

    def test_parser_exposes_the_four_subcommands(self):
        parser = modelctl.build_arg_parser()
        for sub in ("on", "off", "status", "verify"):
            args = parser.parse_args(["headless", sub])
            self.assertEqual(args.headless_command, sub)
            self.assertIs(args.func, modelctl._cmd_headless_cli)


if __name__ == "__main__":
    unittest.main()
