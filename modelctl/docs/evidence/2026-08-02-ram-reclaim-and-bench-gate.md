# 2026-08-02 — RAM reclaimed from tmpfs, and the night-lane gate removed

Raw record. No reading of any of it.

## What changed

* Lane teardown (`land`, `sweep --delete`) now deletes the lane's build
  scratch. It previously freed the worktree, the branch and the port
  block and left the build trees.
* `modelctl lane sweep --orphans` collects lane-suffixed scratch whose
  lane is not in the ledger. `--keep <slug>` spares one.
* `ci/checks.sh` build dirs and logs default to
  `~/.cache/modelctl/ci` instead of `/tmp` (tmpfs).
  `MODELCTL_CI_SCRATCH_ROOT` overrides; `modelctl_lanes` reads the same
  variable.
* The night lane's `window_state()` gate is gone. Conditions are
  observed and recorded per run; a cleanup pass runs before each job;
  the GPU lock is the only remaining wait.

## Bytes reclaimed from /tmp (tmpfs = RAM)

`free -m`, same session, 2026-08-02:

| point | used | free | shared | available | /tmp used |
| --- | ---: | ---: | ---: | ---: | ---: |
| before the sweep | 14432 | 3467 | 3369 | 17396 | 3.5 G |
| after `sweep --orphans --keep console-fleet` | 12680 | 5200 | 1678 | 19148 | 1.8 G |
| after removing the main checkout's dead build dirs | 12778 | 9397 | 2389 | 19051 | 1.8 G |

Directories deleted by the sweep (its own accounting):

    /tmp/ci-build-cpu-lane-phase4          0.8 GB   lane phase4 gone
    /tmp/ci-build-san-lane-phase4          0.1 GB   lane phase4 gone
    /tmp/ci-console-dist-lane-phase4       0.0 GB   lane phase4 gone
    /tmp/modelctl-lane-console-phase4      0.8 GB   lane console-phase4 gone
                                    freed  1.7 GB

Deleted separately, after the scratch root moved off tmpfs (these are
the main checkout's own CI build trees, not lane-suffixed, so the sweep
does not consider them):

    /tmp/ci-build-cpu      838.6 MB
    /tmp/ci-build-san       60.6 MB
    /tmp/ci-console-dist     0.2 MB
                    total  899.4 MB

Still in /tmp at the end of the session: 784 MB + 58 MB + 0.2 MB under
`*-lane-console-fleet` (the order said not to remove anything suffixed
`console-fleet`), 842 MB under `/tmp/modelctl-lane-ram-bench-gate` (this
session's own tmpfs measurement tree, removed when the lane landed), and
121 MB of `/tmp/node-compile-cache`, which nothing in this change owns.

## ci/checks.sh wall-time, tmpfs vs disk

Same machine, same commit (llama.cpp pin `85b7e6556b6b`), same lane
worktree, back to back. Only `MODELCTL_CI_SCRATCH_ROOT` differs. ccache
counters are the delta across each run.

| build dir | ccache | tmpfs | disk |
| --- | --- | ---: | ---: |
| fresh, ccache cold for that path | 4 hits / 330 misses | 128.8 s | 125.8 s |
| fresh, ccache warm for that path | 316 hits / 20 misses | 79.3 s | 80.6 s |
| already built (incremental no-op) | 0 calls | not measured | 49.4 s |

ccache hashes the build directory's path (it lies outside
`CCACHE_BASEDIR`, so the `-I` flags into it are hashed verbatim), so a
build directory that has never been used misses on every object
regardless of which filesystem it is on. The middle row is what a lane
pays: a fresh build directory with the cache already holding that
path's objects.

Test-suite portion of each run: 46 s wall (1919 passed, 11 skipped).

Concurrent machine load: sampled only for the last two runs.
loadavg(1m) 9.14 → 6.93 across the disk incremental run, 6.23 → 7.13
across the disk fresh/warm run — the load is this session's own back-to-
back checks runs. The earlier three runs were taken under the same
pattern of work with no other job dispatched from this session, but
their load was not sampled.

## Test wall-time for the change itself

Five full `ci/checks.sh` runs, each containing one full suite run:

| run | result | wall |
| --- | --- | ---: |
| tmpfs, fresh/cold | 1 FAIL | 128.8 s |
| tmpfs, fresh/warm | all passed | 79.3 s |
| disk, fresh/cold | all passed | 125.8 s |
| disk, incremental | all passed | 49.4 s |
| disk, fresh/warm | all passed | 80.6 s |

Cumulative: 464 s (7 min 44 s) of `checks.sh`, of which 5 × 46 s =
230 s (3 min 50 s) was the test suite.

The one FAIL was `test__hermeticity.py::test_ci_scratch_root_is_redirected`,
the tripwire added in this change, firing because the run exported
`MODELCTL_CI_SCRATCH_ROOT=/tmp` to measure tmpfs and the bootstrap used
`setdefault`. The bootstrap now assigns unconditionally; every run after
that was green.

Unrelated stderr noise present in every suite run, before and after this
change: `sqlite3.OperationalError: attempt to write a readonly database`
from `modelctl_web/jobs.py` worker threads outliving their test's temp
directory, and one `OSError: Directory not empty` from a tempdir
teardown racing the same threads. Warnings, not failures; nothing in
this change touches that code.

## Proof the night lane runs under conditions the old gate refused

By test, in `test_nightlane.py`:

* `test_a_busy_machine_dispatches_everything_anyway` — loadavg 8.99
  (the mean the void 2026-08-01 battery ran at, six times the old 1.5
  ceiling) with `ornith-397b` resident in llama-swap: all four enabled
  jobs dispatch.
* `test_an_unreadable_loadavg_dispatches_too` — loadavg `None` and
  llama-swap unreachable: all four dispatch. The old gate skipped on
  both readings, on the grounds that an unread load "cannot be shown
  low".
* `test_a_lock_that_never_frees_is_a_failure_naming_the_holder` — the
  GPU lock still serialises; a wait that runs out is
  `status: failed` with the holder's note and pid in the message, not a
  skip.
* `test_the_leaf_cgroup_decides_not_a_substring` — the reaper's
  systemd-service check reads the leaf cgroup, because every user
  process on this machine sits under `user@1000.service`.

## Skip paths removed

* `modelctl_nightlane.WindowState` (the `open`/`reasons` type)
* `modelctl_nightlane.window_state()` — both halves: the llama-swap
  reading and the loadavg ceiling
* `modelctl_nightlane.DEFAULT_LOADAVG_CEILING` (1.5)
* the "measurement window is shut" branch of `dispatch_due`, which
  moved every runnable job into `skipped`
* the `refused` status `run_job` gave a held GPU lock, replaced by
  `failed` with the holder named

`headless verify` was the other caller of `window_state()`. It keeps a
quietness gate — it drops the desktop to weigh the VRAM that comes back
— and now owns it at its own call site
(`modelctl._headless_verify_blockers`, ceiling 1.5, `--force` past it)
rather than sharing the night lane's.

GPU lock wait: 60 s → 21600 s (6 h).
