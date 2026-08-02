# Unintended desktop bounce during headless-mode development, 2026-08-02

Raw record of an unintended live state change made by the work session
that built `modelctl headless`. Facts and timestamps only.

## What happened

While exercising the new CLI, the session ran a scratch script that
invoked, in sequence, against a redirected `MODELCTL_HOME`:

    modelctl headless --help
    modelctl headless status
    modelctl headless on
    modelctl headless off
    modelctl headless verify

`on` and `off` were expected to refuse. They did not. Both executed,
each returning rc=0:

    $ modelctl headless on
    [rc=0]
    display mode: headless

    $ modelctl headless off
    [rc=0]
    display mode: graphical

`MODELCTL_HOME` redirection isolates state files. It does not isolate
`sudo systemctl isolate`, and the script assumed a refusal instead of
verifying one.

## Confirmation that the transition was real

    graphical.target         ActiveState=active   ActiveEnterTimestamp=2026-08-02 07:18:57 EDT
    multi-user.target        ActiveState=active   ActiveEnterTimestamp=2026-08-02 01:16:33 EDT
    display-manager.service  ActiveState=active   ActiveEnterTimestamp=2026-08-02 07:18:57 EDT  NRestarts=0

`multi-user.target` still carries its boot timestamp; `graphical.target`
and `display-manager.service` both re-entered active at 07:18:57, which
is the moment of the run. The graphical session was torn down and a new
one started. `sddm.service` is inactive (`display-manager.service` is the
active unit).

Session table afterwards:

    17 1000 aaron seat0 348295 user    tty3  no  -
     2 1000 aaron -     3008   user    -     no  -
     3 1000 aaron -     3014   manager -     no  -
     4 1000 aaron -     3327   user    pts/0 yes 6h ago

Session 3 (`manager`, pid 3014) predates the bounce and survived it.

## State of the serving stack after the round trip

    llama-swap  :9292   ALIVE
    console     :9293   ALIVE
    systemctl --user is-active llama-swap.service modelctl-web.service -> active active

Nothing was restarted by the session. `user@1000` survived the graphical
teardown.

## Why the refusal did not fire

    $ sudo -n -l
    User aaron may run the following commands on aaron:
        (ALL) ALL
        (ALL) NOPASSWD: ALL
        (root) NOPASSWD: /usr/bin/systemctl isolate multi-user.target, /usr/bin/systemctl isolate graphical.target

    $ loginctl show-user aaron --property=Linger --property=State
    State=active
    Linger=yes

Both preconditions the order recorded as missing on 2026-08-02 (`Linger=no`,
no sudoers drop-in) were satisfied by the time this session ran. The
blanket `(ALL) NOPASSWD: ALL` entry grants the isolate regardless of the
narrow drop-in. `modelctl headless status` reported `linger: yes` and
`sudoers: ready` immediately before the toggle ran, and that reading was
correct.

## Change made in response

`modelctl headless on` gained a second gate, independent of the
preconditions: it prompts interactively, and refuses outright when stdin
is not a terminal unless `--yes` is passed. Preconditions answer whether
the transition *can* work; they do not answer whether anyone *meant* it.
`off` is deliberately not gated — the way back must not require a
terminal that may not exist.

Covered by `test_headless_mode.py`:
`test_on_refuses_a_non_interactive_caller_without_yes`,
`test_on_proceeds_non_interactively_with_explicit_yes`,
`test_interactive_decline_isolates_nothing`, `test_off_is_never_gated`.

The scratch script was neutralized rather than deleted, with a header
recording what it did.

## Second bounce, 07:33 — a unit test spawned a real transition

While fixing the above, `test_on_refuses_without_root_setup` failed an
assertion and, in doing so, let `cmd_headless` reach a real
`systemd-run --user`:

    Running as unit: modelctl-headless-on-1785670654.service

Confirmed by `graphical.target ActiveState=inactive` immediately
afterwards. The desktop went down a second time and was restored with a
deliberate `sudo -n /usr/bin/systemctl isolate graphical.target`
(`graphical.target ActiveEnterTimestamp=2026-08-02 07:38:54 EDT`).
llama-swap :9292 and the console :9293 stayed active throughout both
bounces.

Two defects, both fixed:

* `cmd_headless` spawned the detached transition *before* checking
  preconditions. Spawn-then-validate has already committed the machine
  by the time it refuses. Preconditions are now checked in the parent
  first; `set_mode()` re-checks in the child as a second line.
* The CLI tests patched `modelctl_display._run`, which the detached
  child does not inherit — it is a separate process that resolves its
  own. `TestHeadlessCLI.setUp` now stubs `subprocess.run` for the whole
  class with a raising `side_effect`, so a test must opt in via
  `allow_spawn()` and cannot reach a real spawn by forgetting.

## The finding that matters: the isolate did not free any VRAM

Process listing taken while the machine was headless, before any of the
teardown work below existed:

    206788  6929s  Ssl  kwin_wayland_wrapper --xwayland
    206800  6929s  Sl   kwin_wayland --wayland-fd 7 --socket wayland-0 ...
    207148  6928s  Sl   Xwayland :0 ...
    207664  6928s  Ssl  plasmashell --no-respawn

Their cgroups:

    /user.slice/user-1000.slice/user@1000.service/session.slice/plasma-kwin_wayland.service
    /user.slice/user-1000.slice/user@1000.service/session.slice/plasma-plasmashell.service

`isolate multi-user.target` tears down the login **session**. Plasma's
compositor and shell are **user-manager services** under
`user@1000.service`, and linger — the property that keeps llama-swap
alive across the toggle — keeps those alive too. After a full isolate
round-trip `kwin_wayland` was still running, 1h55m old, still holding
`wayland-0`, DRM master, and its GPU buffers.

So the feature as specified in the order does not do what the order says
it is for: **the desktop's VRAM was never released.**

Second consequence, which is what wedged the login screen: on return,
the new session's `startplasma-wayland` (pid 348303, `session-17.scope`)
hung for 8 minutes against the surviving compositor.
`plasmalogin-helper exited with 255` at 07:19:11. Clearing it needed

    systemctl --user stop plasma-plasmashell.service \
        plasma-kwin_wayland.service \
        app-org.kde.xwaylandvideobridge@autostart.service

after which a fresh session came up normally (session 19, tty2) with the
serving stack untouched.

`modelctl headless on` now performs that stop itself, after the isolate,
via `modelctl_display.stop_graphical_user_units()`. Unit discovery is by
`plasma-*.service` glob plus the xwayland bridge, filtered through
`PROTECTED_USER_UNITS` so llama-swap, the console, OVMS and dbus can
never be matched. Coming back does not restart them: logging in starts a
fresh session, which is how Plasma expects to start.

Because the transition is now two steps and the caller does not survive
the first, `headless on` runs both from a detached
`systemd-run --user` unit — the same pattern `verify` uses.

## Not measured

No VRAM samples were taken across either transition. The freed-bytes
figure the order asks for is still unmeasured, and the number will now be
different from any measurement taken before the unit teardown existed,
because before it nothing was freed at all. `modelctl headless verify`
remains untriggered.
