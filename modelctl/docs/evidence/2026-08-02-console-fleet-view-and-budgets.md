# Console fleet view + node budget editing — 2026-08-02

Order: `moe-review/console-fleet-order.md`. Lane `console-fleet`
(ports 9500-9509), on top of phase 4 (`5e40ebf`).

Raw record. No judgements.

## 1. The budget-mutation primitive

New in `modelctl_fleet.py`:

| function | what it does |
|---|---|
| `FleetDevice.cap_bytes` | new field: the OS-enforced limit on the unit serving the device (systemd `MemoryMax`), `0` = not recorded |
| `runtime_headroom(base)` | `max(1 GiB, 5% of base)` |
| `device_ceiling(device)` | `(ceiling_bytes, basis)`; cpu → `MemoryMax`, gpu → reported total, cpu without a cap → reported total with the basis saying so |
| `set_device_budget(node, device, bytes)` | the one writer; `state_lock`, ceiling check, returns the change + staled profiles |
| `set_device_cap(node, device, bytes)` | records a cap read off the live unit; refuses a cap below the budget in force |
| `budget_input(nodes=None)` | `{admission_key: budget_bytes}` for **enabled** nodes, presence-independent — the planning-input twin of `admission_budgets()` |
| `stale_input_profiles(live=None)` | profiles whose recorded planning inputs disagree with the budgets in force |

CLI: `modelctl fleet set-budget <node> <device> <bytes>` and
`modelctl fleet set-cap <node> <device> <bytes>`.

Ceilings in force after this pass:

```
ph16-71-cpu0  / CPU    budget 16.00 GiB  total 30.57 GiB  cap 20.00 GiB  ceiling 19.00 GiB (systemd MemoryMax)
ph16-71-cuda0 / CUDA0  budget 10.00 GiB  total 11.60 GiB  cap  0.00 GiB  ceiling 10.60 GiB (reported device total)
```

### Budgets as a recorded planning input

`modelctl_tiers.make_planning_inputs()` gained a `fleet` block
(`{"budgets_bytes": {...}}`), fed from `modelctl_fleet.budget_input()`
by `modelctl.resolve_planning_inputs()`. `fleet_budget_mismatch()` is the
drift reporter, surfaced as a launch warning in `modelctl_launch.py`
next to the existing display-mode mismatch.

Presence is deliberately **not** part of the recorded input: admission
reads presence (a node nobody can reach may not be spent), a recorded
input must not move because a laptop was closed.

`None` vs `{}` are kept apart: a record written before the field existed
makes no claim and never goes stale; a record that says "no fleet
budgets" *is* contradicted by a node enrolled afterwards.

## 2. The laptop cap — APPLIED, unit not restarted

The `set-property` was refused by this session's permission layer
(read-only SSH to the same host worked in the same session, so it was a
mutation block, not connectivity; nothing was worked around). Aaron ran
it by hand at ~11:26 EDT. Everything below is the read-back.

Cap re-read off the live unit — **26 GiB, and the unit never restarted**:

```
ActiveState=active
SubState=running
NRestarts=0
ExecMainStartTimestamp=Sat 2026-08-01 23:45:48 EDT
ExecMainPID=1472071
MemoryMax=27917287424          # 26 GiB
```

PID and start timestamp are identical to the pre-change reading
(1472071 / Sat 2026-08-01 23:45:48 EDT) and `NRestarts=0`: the process
serving RPC has been up since the night before and did not blink.
`set-property` writes a drop-in and applies it to the live cgroup, which
is why the hard rule holds and presence survived.

Post-bump probe of both nodes (live registry, real wire), immediately
after:

```
ph16-71-cuda0    reachable=True protocol=5.0.0 pin_agrees=True
ph16-71-cpu0     reachable=True protocol=5.0.0 pin_agrees=True
```

Registry. Two writes, in sequence:

```
$ modelctl fleet set-cap ph16-71-cpu0 CPU 21474836480      # 11:06, the pre-bump truth
ph16-71-cpu0/CPU cap: 0.00 GiB -> 20.00 GiB (ceiling now 19.00 GiB)

$ modelctl fleet set-cap ph16-71-cpu0 CPU 27917287424      # 11:28, after the bump
ph16-71-cpu0/CPU cap: 20.00 GiB -> 26.00 GiB (ceiling now 24.70 GiB)
```

The `set-cap` line in the hand-run block did not take effect —
`fleet.json` mtime was still 11:06:33 at 11:28, after the other three
commands had visibly landed — so it was re-applied here against the
value read back off the unit. Recording the cap is what moves the
ceiling: the setter derives it from this number, not from the node.

State after, and the point of the whole step: **only the cap moved.**

```
ph16-71-cpu0  / CPU    budget 16.00 GiB  total 30.57 GiB  cap 26.00 GiB  ceiling 24.70 GiB (systemd MemoryMax)
ph16-71-cuda0 / CUDA0  budget 10.00 GiB  total 11.60 GiB  cap  0.00 GiB  ceiling 10.60 GiB (reported device total)
```

The declared budget is still 16 GiB. What changed is that a budget up to
24.70 GiB (26 GiB − 1.30 GiB headroom) is now something `set-budget`
will accept instead of refuse; 19.00 GiB was the old wall.

## 3-5. Read model, console surface, one mutation path

* `modelctl_web/fleet.py` — `fleet_view()`: local rig first, then every
  registered node, in one shape (same keys, asserted by test). Per
  device: budget / total / cap / ceiling / basis. Per node: identity,
  pin (`node`, `expected`, `agrees` — a field, not a footnote), presence
  tri-state, night-lane rows, stale profiles. Opens no socket.
* Presence: `PRESENT` (reachable, pin agrees, inside the 900 s TTL) /
  `STALE` (never probed, expired, or unreachable — detail says which) /
  `PIN_MISMATCH` (reachable on the wrong commit). A test pins that the
  view's `PRESENT` and `usable_nodes()` never disagree.
* Routes: `GET /api/v2/fleet`, `POST /api/v2/fleet/probe`,
  `POST /api/v2/fleet/nodes/{node}/probe`,
  `POST /api/v2/fleet/nodes/{node}/devices/{device}/budget`.
* **The located phase-4 mutation entry: `modelctl_web/mutate.py`'s
  `submit_*` helpers onto `JobRunner.submit(..., lane="mutation")`**
  (found by reading `test_console_phase4.py`, which mocks
  `modelctl_web.mutate.submit_*` for every phase-4 control, and the
  routes in `modelctl_web/app.py`). Budget editing adds exactly one
  helper, `mutate.submit_fleet_budget`, which calls
  `modelctl_fleet.set_device_budget` and nothing else. No route writes
  the registry directly; the ceiling check on the route is a
  fail-early copy whose authoritative twin runs inside the lock.
* Console: `console/src/pages/fleet.tsx`, route `/v2/fleet`, nav entry
  next to operate. Node cards; presence chips `ok` / `warn` / `err` with
  the non-present cards also carrying the `widget stale` treatment;
  budget field capped at the ceiling with the ceiling shown; submission
  through `submitAction` + the configure-style `ConfirmButton`;
  auto-probe on open plus a per-node "probe now"; night-lane pairs
  read-only.

## 6. Scratch walk

Full transcript: `fleet-walk-transcript.txt` (driver:
`fleet_walk.py`). `MODELCTL_WEB_SCRATCH=1`, all five redirections, a
throwaway state universe, lane port 9500, torn down at the end. The real
registry was never the one edited — the three fixture nodes live only in
the scratch tree, and the live registry is printed unchanged at the end.

Three presence states, from one page load:

```
  rig (this machine)     local  PRESENT       pin_agrees=True
  ph16-71-cpu0           remote PRESENT       pin_agrees=True
  ph16-71-cuda0          remote STALE         pin_agrees=True  never probed
  ph16-71-oldbuild       remote PIN_MISMATCH  pin_agrees=False node built at 111111111111, this checkout pins 85b7e6556b6b
```

Both budget writes refused by the scratch middleware, with the request
named:

```
POST .../devices/CPU/budget  budget_bytes=19327352832 (18.00 GiB, under the ceiling)
  -> 405 scratch-safe mode (MODELCTL_WEB_SCRATCH=1): this instance refuses mutations; POST /api/v2/fleet/nodes/ph16-71-cpu0/devices/CPU/budget would write state or drive the serving stack
POST .../devices/CPU/budget  budget_bytes=26843545600 (25.00 GiB, over the ceiling)
  -> 405 (same reason)
```

A scratch console refuses a write *before* the handler, so the ceiling
is never consulted there and the console transcript alone cannot prove
it exists. The same scratch state, through the CLI — the same primitive
the console's mutation entry calls:

```
$ modelctl fleet set-budget ph16-71-cpu0 CPU 26843545600   # 25 GiB
  rc=2
  refused: budget 25.00 GiB exceeds the ceiling for ph16-71-cpu0/CPU: 19.00 GiB
           (systemd MemoryMax 20.00 GiB minus 1.00 GiB runtime headroom).
           Raise the cap on the node first, or ask for less.

$ modelctl fleet set-budget ph16-71-cpu0 CPU 19327352832   # 18 GiB
  rc=0
  ph16-71-cpu0/CPU budget: 16.00 GiB -> 18.00 GiB (ceiling 19.00 GiB, systemd MemoryMax)
    stale planning inputs: walk-dependent
```

The over-ceiling refusal on the console's own endpoint (422, naming the
ceiling, before anything is queued) is pinned by
`test_console_fleet.TestFleetRoutes.test_an_over_ceiling_budget_is_refused_before_it_is_queued`.
Client-side, the editor caps the input at the ceiling and refuses a
typed over-ceiling value without sending the request.

Live registry after the walk, unchanged:

```
ph16-71-cuda0          budget  10.00 GiB  cap   0.00 GiB
ph16-71-cpu0           budget  16.00 GiB  cap  20.00 GiB
```

## The authorized console restart — used, once

`modelctl-web.service` runs `python -m modelctl_web` from the main
checkout with no `--reload`, so it cannot hot-load and the restart *was*
the cutover. The command was refused by the same permission layer that
refused the laptop bump (the process was **not** killed to let
`Restart=` respawn it — that would circumvent the denial, not adjust to
it); Aaron ran it by hand.

The window, from the service's own record:

```
before   MainPID 869250   started Sun 2026-08-02 10:37:00 EDT
after    MainPID 975412   started Sun 2026-08-02 11:26:17 EDT
```

No job was in flight across it (`web_jobs.db` at the time: 58 done, 13
failed, 0 running, 0 queued), so nothing was marked interrupted. Nothing
else was restarted: `llama-swap :9292` still answers 200 on the same
process it has been running on throughout.

The cutover, verified against the live instance on :9293:

```
GET  /api/v2/fleet   -> 200
  rig (this machine)     local  PRESENT   SYCL0 budget 28.70 / ceiling 31.89 GiB (editable=False)
                                          SYCL1 budget 10.74 / ceiling 11.93 GiB (editable=False)
  ph16-71-cuda0          remote PRESENT   CUDA0 budget 10.00 / ceiling 10.60 GiB (reported device total)
  ph16-71-cpu0           remote PRESENT   CPU   budget 16.00 / ceiling 24.70 GiB (systemd MemoryMax)
  night_lane: ornith-rpc-criterion-2026-08-02 (enabled=False),
              qwen122b-remote-experts-hypothesis-2026-08-02 (enabled=False)
  stale_profiles: []
POST /api/v2/fleet/probe -> 200   both nodes reachable, protocol 5.0.0, pin_agrees=True
GET  /v2/fleet       -> 200   (SPA shell)
```

The live surface carries the 24.70 GiB ceiling, so step 2's cap and
step 1's ceiling are joined up end to end. Both remote nodes read STALE
on the first load — the stored presence record was from 01:15, past the
900 s TTL — and flipped to PRESENT after the probe route ran, which is
the tri-state doing its job on live data rather than on fixtures.

## 7. Tests

New: `test_fleet_budget.py` (28 cases) and `test_console_fleet.py`
(32 cases). The concurrency case was verified to have teeth — with
`state_lock` stubbed out it fails on the lost update
(`17179869184 != 19327352832`).

Full suite, run 1 (baseline/gate 1):

```
1872 passed, 11 skipped in 46.18s
```

`ci/checks.sh`, run 2 (contains the second full-suite run), with the
lane build-dir overrides:

```
submodule pin        PASS  pin and working tree agree (85b7e6556b6b)
static checks        PASS  compileall / every modelctl module imports
test suite           PASS  1872 passed, 11 skipped in 46s wall
console offline      PASS  console builds offline from the vendored tree
CPU-only build       PASS  build, MoE cache tests, hybrid tests,
                           CPU-only reports every SYCL/cache feature false
sanitizer pass       PASS  test-moe-cache, test-moe-hybrid under ASan/UBSan
layering             PASS  no leaf module imports modelctl at module level
result               all checks passed          2:02.41 elapsed
```

Cumulative **test** wall-time for this pass: **~2 min** — 46 s (run 1)
+ 46 s (the suite inside checks.sh) + ~26 s of targeted per-file runs
while iterating. Inside the 10-minute tripwire. (`ci/checks.sh` total
elapsed was 2:02, most of it the CPU-only and sanitizer builds.)
