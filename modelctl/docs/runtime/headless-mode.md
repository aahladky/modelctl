# Headless mode

Drop the rig to a text console on demand, freeing the VRAM the desktop
holds on the B70, and bring the desktop back just as easily. No hardware
changes. Nothing in the serving stack restarts.

    modelctl headless on        # desktop away, models keep serving
    modelctl headless off       # desktop back
    modelctl headless status    # mode, time in mode, measured freed VRAM
    modelctl headless verify    # detached round-trip self-test

## What actually frees the VRAM

Isolating `multi-user.target` is **not** enough, and this is the single
most important thing on this page.

`isolate` tears down the login *session*. Plasma's compositor and shell
are not in it — they are user-manager services under
`user@1000.service/session.slice/plasma-kwin_wayland.service` and
`.../plasma-plasmashell.service`. Linger, the very property that keeps
llama-swap alive across the toggle, keeps those alive too. Measured
2026-08-02: after a full isolate round-trip `kwin_wayland` was still
running, 1h55m old, still holding `wayland-0`, DRM master, and its GPU
buffers. Nothing was freed.

So `modelctl headless on` stops them explicitly, after the isolate:

    systemctl --user stop plasma-*.service \
        app-org.kde.xwaylandvideobridge@autostart.service

Discovery is a `plasma-*.service` glob plus the xwayland bridge, filtered
through `modelctl_display.PROTECTED_USER_UNITS` — `llama-swap.service`,
`modelctl-web.service`, `ovms.service`, `dbus*` can never be matched, no
matter what a future glob picks up.

Coming back does **not** restart them. Logging in starts a fresh session,
which is how Plasma expects to start; restarting the old units under a
new compositor is what wedged the login screen on 2026-08-02.

Because the transition is two steps now, and the caller does not survive
the first when it is the desktop entry or a GUI terminal, `headless on`
runs both from a detached `systemd-run --user` unit.

## Why this works at all

`llama-swap.service` and `modelctl-web.service` are **user** services
under `user@1000/app.slice`, not children of the graphical session. When
the graphical target goes away they are untouched — provided
**linger is enabled**, so that the user manager itself survives the last
session ending. Without linger, dropping the desktop ends `user@1000`
and takes the live stack down with it.

Nothing is restarted across the toggle. That is the whole design, not an
optimisation: `modelctl headless` never restarts llama-swap (`:9292`),
OVMS, or the console (`:9293`), and it never changes the default systemd
target.

## Getting out

Know these before you use it the first time. The desktop is gone; both
displays are dark.

| route | how |
|---|---|
| virtual terminal | `ctrl-alt-F2`, log in, `modelctl headless off` |
| SSH from the laptop | `ssh aaron@192.168.0.184` then `modelctl headless off` |
| reboot | always comes up **graphical** — the default target is never changed |

The reboot route is the escape hatch, and it is why the default target
must stay `graphical.target`. Do not "tidy" that up.

## One-time root setup

Two root actions, in one script that prints what it will do before doing
it, and is safe to re-run:

```bash
sudo modelctl/docs/fleet/rig-headless-setup.sh
```

1. `loginctl enable-linger aaron` — see above. Worth having on its own
   merits: it also stops a plain logout from killing the serving stack.
2. `/etc/sudoers.d/modelctl-headless`, containing exactly one line:

```
aaron ALL=(root) NOPASSWD: /usr/bin/systemctl isolate multi-user.target, /usr/bin/systemctl isolate graphical.target
```

Two commands, both fully qualified, nothing else. `systemctl isolate`
with an *unconstrained* argument would be a password-free route to
`rescue.target` and `emergency.target` — a root shell. Naming both
targets in full removes that. The paths are absolute because sudoers
matches the command line literally.

Until both are in place, `modelctl headless on` and `modelctl headless
verify` refuse and say which half is missing. They probe with
`sudo -n -l <the exact command>`, not `sudo -n true` — a correctly narrow
drop-in permits `true` no more than it permits anything else.

**State on this rig as of 2026-08-02: both are already done.** Linger is
`yes`, and `sudo -l` shows the two-command rule in place (alongside a
pre-existing blanket `(ALL) NOPASSWD: ALL` for `aaron`, which is what
actually grants the isolate). Re-running the setup script is a no-op.

## The intent gate on `on`

`modelctl headless on` asks for confirmation, and **refuses outright when
stdin is not a terminal** unless `--yes` is passed:

```
headless on: refusing to drop the desktop from a non-interactive caller.
  This closes the graphical session and every window in it. Pass --yes if
  that is what you mean.
```

This exists because of a real incident. On 2026-08-02 a scripted
`modelctl headless on`, written on the assumption that the preconditions
would refuse, ran on a box where both were already satisfied — and took
the live desktop down. Preconditions answer *can this work*; they do not
answer *did anyone mean it*. Hence the second gate.

`off` is never gated. The way back must not require a terminal it might
not have — a VT after a wedge, or an SSH pipe with no tty.

## Desktop entry

`modelctl/docs/fleet/modelctl-headless.desktop` is a launcher that shows
a kdialog confirmation naming the exit routes, then runs `modelctl
headless on`. Install it for your user (no root needed):

```bash
cp modelctl/docs/fleet/modelctl-headless.desktop ~/.local/share/applications/
```

It runs with `Terminal=false` on purpose: a terminal emulator is a child
of the graphical session, so the isolate would kill the window running it
partway through. For the same reason `modelctl headless on` writes its
state record *before* it isolates.

## verify: the round-trip self-test

```bash
modelctl headless verify
```

Aaron triggers this; it is not something a work session runs. It refuses
unless the root setup is complete, and refuses unless the night lane's
machine is quiet (llama-swap holding nothing, load below this command's
own 1.5 ceiling; the night lane's version of this gate was removed on
2026-08-02, and `headless verify` now owns the only one left) —
`--force` overrides the second, never the first.

It runs **detached**, in a transient `systemd-run --user` unit, because
it has to survive the desktop dying under it. What it does, in order:

1. measures free VRAM on the B70 with the desktop up (3 samples, via the
   same `get_gpu_inventory()` probe the planner reads);
2. isolates `multi-user.target`;
3. within seconds, liveness-checks `:9292` and `:9293`;
4. measures free VRAM headless (3 samples);
5. loads the tiny fixture on scratch port 9411, takes 8 greedy tokens,
   tears it down by PID;
6. isolates `graphical.target` — in a `finally`, so this happens whatever
   went wrong above;
7. re-checks both services and files the raw log under
   `modelctl/docs/evidence/<date>-headless-verify.md`.

A ten-minute deadline applies: a round-trip that has not finished by then
skips the remaining work and returns to graphical rather than sitting
there with the desktop gone.

Follow it live with `journalctl --user -u modelctl-headless-verify-<ts> -f`.

## The recorded display state

`$MODELCTL_HOME/display-state.json` (default
`~/.local/share/modelctl/display-state.json`):

```json
{
  "version": 1,
  "mode": "graphical",
  "since": 1754130000.0,
  "measured": {
    "freed_bytes": 0,
    "device": "SYCL0",
    "graphical_free_bytes": [],
    "headless_free_bytes": [],
    "at": 1754130000.0
  }
}
```

It lives under `MODELCTL_HOME` rather than the `~/.local/state` path the
original order named, so that one environment variable still moves *all*
modelctl state at once — the property the test suite and every scratch
walk depend on, and the exact split `modelctl_paths` exists to prevent.

The mode is recorded with a profile's planner inputs
(`planning.inputs.display`) like any other input, alongside RAM,
inventory, and the VRAM limit. **It is recorded only.** No budget spends
the freed bytes in this change; the per-device budget work is what will
use them.

What it buys today: a plan built headless and launched with the desktop
up (or the reverse) says so, through the same stored-inputs machinery as
every other input:

```
WARNING: planning inputs were recorded with the display headless, but the
machine is graphical now (~2.25 GiB measured); free VRAM differs from what
this plan was built against -- replan with --refresh-inputs to rebuild it
for this mode
```

It is a warning on the launch command, never an error — the plan still
launches, and whether to replan is the operator's call. Both sides have
to make a claim: a profile planned before this field existed, or a
machine that has never toggled, records `unknown` and says nothing.

## Rules this respects

* Never restarts llama-swap, OVMS, or the console.
* Never changes the default target; never runs `systemctl isolate`
  outside the two sanctioned commands.
* Root steps are Aaron's; the CLI detects their absence and stops.
* `GGML_OP_OFFLOAD_MIN_BATCH` below 32 refuses (the verify fixture load
  checks it explicitly).
* The fixture server is torn down by PID, never by a `pkill` pattern
  that would match the harness's own command line.
