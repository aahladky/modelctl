"""Display mode (desktop up vs text console) as a recorded planner input.

The rig boots graphical, and the desktop holds VRAM on the B70 that a
model could otherwise use. `modelctl headless on` drops the machine to a
text console; `off` brings the desktop back. Nothing in the serving
stack restarts in between: llama-swap and the web console are user
services under user@1000, and with linger enabled they outlive the
graphical session that happened to start them.

Three things this module is deliberately careful about.

* It never touches the default target. A reboot always comes up
  graphical, so the escape hatch out of a wedged headless rig is the
  power button -- and that stays true no matter what state this module
  leaves behind.
* It refuses rather than guesses. `on` requires linger enabled (without
  it, ending the last session takes user@1000 -- and the live stack --
  down with it) and a sudoers drop-in narrow enough to permit exactly
  the two isolate commands and nothing else. Both are root setup that
  only Aaron performs; this module detects their absence and stops.
* Nothing here restarts a service. The whole design survives the toggle
  precisely because no service is restarted across it.

The recorded mode is a planner input like any other (see
modelctl_tiers.make_planning_inputs): a plan computed with the desktop
up saw less free VRAM than one computed headless, so a launch whose
recorded mode disagrees with the machine says so rather than silently
running a plan built for a different one. This module records the mode
and the measured freed bytes; it charges nothing to any budget.

Live truth comes from systemd (`probe_mode`), and the recorded input
comes from the state file (`recorded_mode`). They are separate on
purpose: the planner and every launch path read the file, which is a
cheap, hermetic read that honours MODELCTL_HOME, while only the
`headless` command itself shells out to systemd.

This module must stay a leaf: no modelctl imports.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import modelctl_fsutil
from modelctl_paths import STATE_DIR

# Deliberately under STATE_DIR rather than the ~/.local/state path the
# order named: every other piece of modelctl state resolves through
# modelctl_paths so MODELCTL_HOME moves all of it at once (the test
# suite and every scratch walk depend on that, and a second state root
# is the exact split modelctl_paths exists to prevent).
STATE_PATH = STATE_DIR / "display-state.json"
STATE_VERSION = 1

GRAPHICAL = "graphical"
HEADLESS = "headless"
UNKNOWN = "unknown"
MODES = (GRAPHICAL, HEADLESS)

TARGETS = {GRAPHICAL: "graphical.target", HEADLESS: "multi-user.target"}

# Absolute paths, because sudoers matches the command line literally: a
# drop-in naming /usr/bin/systemctl does not authorize `systemctl`
# resolved through PATH to something else.
SYSTEMCTL = "/usr/bin/systemctl"
LOGINCTL = "/usr/bin/loginctl"
SUDO = "/usr/bin/sudo"


# --- process helpers -------------------------------------------------------

def _run(argv, timeout=15):
    """(rc, stdout, stderr), stripped. Never raises: a missing binary or a
    timeout is a failed probe, not a traceback in the middle of a toggle."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 127, "", f"{type(e).__name__}: {e}"


def current_user():
    return os.environ.get("USER") or os.environ.get("LOGNAME") or ""


# --- the two sanctioned commands -------------------------------------------

def isolate_command(mode):
    """The privileged command for one mode, WITHOUT sudo. This exact argv
    is what the sudoers drop-in authorizes, so it has one definition."""
    if mode not in TARGETS:
        raise ValueError(f"unknown display mode: {mode!r}")
    return [SYSTEMCTL, "isolate", TARGETS[mode]]


def isolate_argv(mode):
    """The full command the toggle runs. -n so a missing drop-in fails
    immediately instead of blocking on a password prompt nobody will
    ever see -- the desktop may already be gone when it is asked."""
    return [SUDO, "-n"] + isolate_command(mode)


def sudoers_lines(user=None):
    """The exact drop-in content the root setup script installs. Two
    commands, both fully qualified, nothing else -- `systemctl isolate`
    with an unconstrained argument would be a general-purpose way to
    reach any target, including rescue and emergency."""
    user = user or current_user() or "aaron"
    cmds = ", ".join(" ".join(isolate_command(m)) for m in (HEADLESS, GRAPHICAL))
    return [f"{user} ALL=(root) NOPASSWD: {cmds}"]


# --- live probes -----------------------------------------------------------

def probe_mode(runner=None):
    """The machine's actual mode, from systemd.

    graphical.target is the discriminator: multi-user.target is active in
    both modes (graphical.target requires it), so only graphical.target
    going inactive distinguishes them. Anything unreadable is UNKNOWN,
    never a guess."""
    run = runner or _run
    _, out, _ = run([SYSTEMCTL, "is-active", TARGETS[GRAPHICAL]])
    if out == "active":
        return GRAPHICAL
    _, multi, _ = run([SYSTEMCTL, "is-active", TARGETS[HEADLESS]])
    if out == "inactive" and multi == "active":
        return HEADLESS
    return UNKNOWN


def linger_enabled(user=None, runner=None):
    """Whether user@N survives the last session ending.

    This is the precondition the whole design rests on: with Linger=no,
    dropping the graphical session ends the user manager and takes
    llama-swap and the console with it."""
    run = runner or _run
    user = user or current_user()
    if not user:
        return False
    rc, out, _ = run([LOGINCTL, "show-user", user, "--property=Linger"])
    return rc == 0 and out.endswith("=yes")


def sudo_isolate_ready(runner=None):
    """(ok, detail) for the sudoers drop-in.

    Probes each isolate command with `sudo -n -l`, which asks "may I run
    exactly this, without a password?" and answers without running it.
    A plain `sudo -n true` would be the wrong probe: a correctly narrow
    drop-in permits neither `true` nor anything else."""
    run = runner or _run
    for mode in (HEADLESS, GRAPHICAL):
        cmd = isolate_command(mode)
        rc, _, err = run([SUDO, "-n", "-l"] + cmd)
        if rc != 0:
            return False, (f"sudo -n cannot run `{' '.join(cmd)}`"
                           + (f": {err}" if err else ""))
    return True, ""


# --- the desktop's own user units ------------------------------------------
# Measured on 2026-08-02, and the reason this section exists: isolating
# multi-user.target tears down the login SESSION, but Plasma's compositor
# and shell are user-manager services under
# user@1000.service/session.slice/. Linger -- the thing that keeps
# llama-swap alive across the toggle -- keeps those alive too. After a
# full isolate round-trip, kwin_wayland was still running, 1h55m old,
# still holding wayland-0, DRM master, and its GPU buffers.
#
# Two consequences, both observed:
#   * the VRAM this feature exists to free was never freed;
#   * coming back, the new session's startplasma-wayland wedged against
#     the surviving compositor and the login screen hung.
#
# So going headless stops them explicitly. Coming back does not restart
# them: logging in starts a fresh session, which is how Plasma expects to
# be started anyway.

GRAPHICAL_UNIT_PATTERNS = ("plasma-*.service",)
GRAPHICAL_EXTRA_UNITS = ("app-org.kde.xwaylandvideobridge@autostart.service",)

# Never stoppable by this path, whatever a pattern matches. The serving
# stack surviving the toggle is the entire point; a glob that ever grew
# to cover it would silently turn this feature into an outage.
PROTECTED_USER_UNITS = ("llama-swap.service", "modelctl-web.service",
                        "ovms.service", "dbus.service", "dbus-broker.service")


def graphical_user_units(runner=None):
    """Active user units keeping the desktop -- and its VRAM -- alive
    after the graphical session is gone."""
    run = runner or _run
    found = []
    for pattern in GRAPHICAL_UNIT_PATTERNS:
        rc, out, _ = run([SYSTEMCTL, "--user", "list-units", "--state=active",
                          "--plain", "--no-legend", "--no-pager", pattern])
        if rc == 0:
            found += [line.split()[0] for line in out.splitlines()
                      if line.split()]
    for unit in GRAPHICAL_EXTRA_UNITS:
        _, out, _ = run([SYSTEMCTL, "--user", "is-active", unit])
        if out.strip() == "active":
            found.append(unit)
    return sorted({u for u in found if u not in PROTECTED_USER_UNITS})


def stop_graphical_user_units(runner=None):
    """Stop the desktop's surviving user units. Returns (units, ok, err).

    Not fatal when it fails: the machine is already headless by the time
    this runs, and a desktop unit that would not stop is worth reporting
    rather than worth rolling the whole transition back for."""
    run = runner or _run
    units = graphical_user_units(runner=run)
    if not units:
        return [], True, ""
    rc, _, err = run([SYSTEMCTL, "--user", "stop"] + units, timeout=90)
    return units, rc == 0, err


def refusals(user=None, runner=None):
    """Why `headless on` must not proceed, as operator-readable lines.
    Empty means the root setup is in place."""
    out = []
    if not linger_enabled(user, runner):
        out.append(
            f"linger is not enabled for {user or current_user() or 'this user'}"
            " -- dropping the desktop would end user@1000 and kill "
            "llama-swap and the console with it")
    ok, detail = sudo_isolate_ready(runner)
    if not ok:
        out.append(detail)
    if out:
        out.append("run the one-time root setup first: "
                   "sudo modelctl/docs/fleet/rig-headless-setup.sh")
    return out


# --- recorded state --------------------------------------------------------

def read_state():
    """The recorded display state, or {} when absent or unreadable. A
    torn or hand-edited file reads as "no record", never as an
    exception on a planning path."""
    try:
        data = json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def recorded_mode(state=None):
    """The mode as recorded -- what plans were built against. UNKNOWN when
    nothing has been recorded, which every consumer treats as "no claim"
    rather than as graphical."""
    state = read_state() if state is None else state
    mode = state.get("mode")
    return mode if mode in MODES else UNKNOWN


def write_state(mode, freed_bytes=None, measured=None, now=None):
    """Record the mode (and optionally a freed-VRAM measurement).

    `since` only moves when the mode actually changes, so "time in mode"
    survives a re-record. Written atomically: the planner reads this file
    on every plan and must never see half of it."""
    if mode not in MODES:
        raise ValueError(f"unknown display mode: {mode!r}")
    now = time.time() if now is None else now
    state = read_state()
    if state.get("mode") != mode or not state.get("since"):
        state["since"] = now
    state["version"] = STATE_VERSION
    state["mode"] = mode
    if freed_bytes is not None or measured is not None:
        block = dict(state.get("measured") or {})
        if freed_bytes is not None:
            block["freed_bytes"] = int(freed_bytes)
        if measured:
            block.update(measured)
        block["at"] = now
        state["measured"] = block
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    modelctl_fsutil.atomic_write_text(STATE_PATH, json.dumps(state, indent=2))
    return state


def measured_freed_bytes(state=None):
    """Last measured B70 bytes the desktop was holding, or 0 when never
    measured. Recorded only -- no budget consumes this."""
    state = read_state() if state is None else state
    try:
        return int((state.get("measured") or {}).get("freed_bytes") or 0)
    except (TypeError, ValueError):
        return 0


def time_in_mode(state=None, now=None):
    """Seconds since the recorded mode was entered, or None."""
    state = read_state() if state is None else state
    since = state.get("since")
    if not isinstance(since, (int, float)):
        return None
    return max(0.0, (time.time() if now is None else now) - since)


def format_duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def planning_input(state=None):
    """The display block recorded with a profile's planner inputs.

    Deliberately small: the mode a plan was built under, and the bytes
    the desktop was measured to be holding. The per-device budget item
    is what will later spend those bytes; recording them here charges
    nothing."""
    state = read_state() if state is None else state
    return {"mode": recorded_mode(state),
            "freed_bytes": measured_freed_bytes(state)}


# --- transitions -----------------------------------------------------------

def set_mode(mode, runner=None, user=None, now=None):
    """Switch the machine into `mode`.

    Returns a dict with a fixed shape whatever happened: {"ok", "mode",
    "refusals", "argv", "rc", "stderr", "stopped_units", "stop_ok",
    "stop_error"}. Callers read it without guarding for missing keys.

    Going headless writes the state record BEFORE isolating: the isolate
    may take the session running this process with it, and a plan built
    afterwards must not read a stale "graphical". Coming back records
    after, when the desktop is up again and the write is certain to
    land."""
    if mode not in MODES:
        raise ValueError(f"unknown display mode: {mode!r}")
    run = runner or _run
    result = {"ok": False, "mode": mode, "refusals": [], "argv": None,
              "rc": None, "stderr": "", "stopped_units": [],
              "stop_ok": True, "stop_error": ""}

    blocked = refusals(user=user, runner=run) if mode == HEADLESS else []
    if blocked:
        return {**result, "refusals": blocked}

    argv = isolate_argv(mode)
    if mode == HEADLESS:
        write_state(HEADLESS, now=now)
    rc, _, err = run(argv, timeout=120)
    result.update(argv=argv, rc=rc, stderr=err)

    if rc != 0:
        # The isolate failed, so the machine did not move. Put the record
        # back where the machine actually is rather than leaving a claim
        # nothing supports.
        probed = probe_mode(runner=run)
        if probed in MODES:
            write_state(probed, now=now)
        return result

    if mode == HEADLESS:
        # The isolate freed the SESSION, not the desktop: its compositor
        # and shell are lingering user units still holding the VRAM this
        # whole feature exists to release.
        units, stop_ok, stop_err = stop_graphical_user_units(runner=run)
        result.update(stopped_units=units, stop_ok=stop_ok,
                      stop_error=stop_err)
    else:
        write_state(GRAPHICAL, now=now)
    return {**result, "ok": True}


def status(runner=None, user=None):
    """Everything `modelctl headless status` prints, as data."""
    state = read_state()
    return {
        "recorded_mode": recorded_mode(state),
        "probed_mode": probe_mode(runner=runner),
        "since": state.get("since"),
        "time_in_mode": time_in_mode(state),
        "measured": dict(state.get("measured") or {}),
        "linger": linger_enabled(user=user, runner=runner),
        "sudoers_ready": sudo_isolate_ready(runner=runner)[0],
        "state_path": str(STATE_PATH),
    }
