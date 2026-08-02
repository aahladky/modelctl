# Parallel session lanes

Several Claude Code sessions run orders at once without any of them
seeing the others, and without Aaron ever meeting a branch, a worktree
or a merge. A lane is a hidden checkout that exists for exactly as long
as the session that owns it; landing it is a fast-forward of `master`.

    modelctl lane start <slug>      # make one
    modelctl lane land <slug>       # rebase, re-check if needed, fast-forward, delete
    modelctl lane list              # age, sessions, unlanded commits, flags
    modelctl lane sweep             # the same, and delete only what is named
    modelctl lane env <slug>        # the lane's shell environment
    modelctl lane gpu-lock -- <cmd> # run <cmd> holding the machine's GPU lock

Aaron's surface is unchanged: he pastes an order into a session, and the
work turns up on `master`. Nothing on this page is something he has to
do.

## What a lane is

`modelctl lane start review-console` prints a path and stops:

    /home/aaron/workspace/.lanes/review-console
    branch lane/review-console, ports 9500-9509
    environment: eval "$(modelctl lane env review-console)"
    land it with: modelctl lane land review-console

That is a real git worktree of the main checkout on branch
`lane/review-console`, with the `llama.cpp` submodule checked out
`--reference`d to the main checkout's copy, so it shares the fork's
object store instead of refetching several gigabytes.

It lives in `/home/aaron/workspace/.lanes/`, **beside** the repository
and not inside it. A worktree inside the checkout would show up in `git
status`, in every `find`, and in every glob a future order writes —
lanes are meant to be invisible, and a dot-directory one level up is
what makes that true.

### The clone source, when gitea is down

`--reference` supplies objects; it does not remove the need to reach the
configured remote. So a failed submodule clone is retried with the URL
pointed at the main checkout's own `llama.cpp`, which has every object
the pin needs and is on the same disk. The override is passed with `git
-c`, never written: worktrees share one `.git/config`, and persisting it
would repoint the main checkout's submodule too. `lane start` says so
when it takes that path.

If the submodule cannot be checked out either way, the whole lane is
rolled back — worktree, branch and port block. A half-made lane is
worse than none, because the session only discovers it at build time.

## Landing

`modelctl lane land <slug>`, under a global flock so two lands can never
fast-forward the same `master`:

1. **The lane must be committed.** Uncommitted files stop the land.
   Land is not a stasher, and a file that is uncommitted here is work
   that would be deleted with the worktree three steps later.
2. **The main checkout must be on `master`.**
3. **Operator edits in the main checkout become a commit** — message
   `journal: local edits` — before anything lands on top. Never a stash:
   a stash is where work goes to be forgotten, a commit is in the log
   where the next session sees it.
4. **Rebase onto `master`**, unless `master` has not moved under the
   lane at all. On a conflict the rebase is aborted and the land stops:
   the lane is exactly as it was, `master` is untouched.
5. **If the rebase actually replayed anything, the full `ci/checks.sh`
   runs inside the lane** — not `--quick`. Rebased code is code no run
   has ever seen; the lane's green checks were about a different parent,
   and the checks that catch a semantic conflict are the ones `--quick`
   skips. Red checks stop the land with `master` untouched.
6. **`git merge --ff-only`** in the main checkout.
7. **The lane is deleted**: worktree, branch, port block, ledger entry.

A stop is a non-zero exit with the reason on stderr, always from a state
where `master` is byte-identical to what it was and the lane still holds
its commits. There is no path that lands half a lane.

### The fork pin is never advanced as a side effect

The journal commit in step 3 excludes the submodule, and if the main
checkout's `llama.cpp` pointer has moved the land stops there and says
so. Advancing the pin is a decision an order makes deliberately; it is
never a thing that happens because a worktree was dirty.

## Nothing strands silently

`lane list` and `lane sweep` print every lane with its age, the PIDs of
any live session inside it (processes whose cwd is in the lane — no
registration, because a registered PID goes stale exactly when it
matters), how many commits it holds that `master` does not, and flags:

    review-console       9500-9509  age  26h04m  sessions none  unlanded 3
      /home/aaron/workspace/.lanes/review-console  (lane/review-console)
      flags: older than 24 h; no live session

`sweep` flags and stops. It deletes only lanes named with `--delete`,
and refuses even then if the lane has unlanded commits or a live
session; `--force` overrides. A sweep that deleted stale lanes on its
own would be the thing that forgets somebody's work.

## Ports

Each lane gets a block of ten from a ledger, so two scratch consoles
never fight over one port. Blocks are allocated lowest-free-first from
**9500–9699** (twenty blocks), a range chosen to step around everything
the rig already answers on: 9090 metrics, 9292 llama-swap, 9293 console,
9294 remote hands, 9411 headless verify, 9728/9832 fleet RPC. Landing or
deleting a lane frees its block for reuse.

## The GPU lock

    modelctl lane gpu-lock -- python3 sweep.py

One machine-wide advisory `flock`, held for the duration of the command,
at `~/.local/share/modelctl/gpu.lock` (moves with `MODELCTL_HOME`;
overridable with `MODELCTL_GPU_LOCK`). Anything can take it with
`flock(1)` on that path — no modelctl code required, which is the point
of a well-known path.

The night lane takes the same lock around every measurement
(`modelctl_nightlane.run_job`), waiting up to
`GPU_LOCK_WAIT_SECONDS` (60 s) and otherwise **refusing** the job with
that reason recorded in its evidence. The benchmark job lane's single
worker already stops two night jobs overlapping; the lock is what stops
a night job and a session's bench in a parallel lane from measuring each
other, and those two schedulers know nothing about one another.

## ccache

ccache hashes the absolute path of every source file, so the same commit
built from a lane would miss on every object the main checkout already
compiled — a fork-touching lane would pay a full rebuild instead of
minutes. Two settings fix that, and `ci/checks.sh` and `lane env` both
set them:

    CCACHE_BASEDIR=<checkout root>   # rewrite paths under the root to relative
    CCACHE_NOHASHDIR=1               # keep the cwd out of the hash

Every `cmake` invocation in `ci/checks.sh` runs from the checkout root,
because `CCACHE_BASEDIR` rewrites relative to the build's cwd and the
two checkouts have to agree. ggml already turns ccache on by itself when
it is installed (`GGML_CCACHE=ON`); this only makes its entries
shareable between the main checkout and a lane. No sloppiness is set:
headers, macros and compiler identity are hashed as strictly as ccache's
defaults, and the only thing made path-independent is the path.

## The lane environment

`eval "$(modelctl lane env <slug>)"` exports:

| Variable | Why |
| --- | --- |
| `MODELCTL_LANE`, `MODELCTL_LANE_PORT_BASE` | which lane this shell is in |
| `MODELCTL_WEB_BIND` | `127.0.0.1:<port base>` — the lane's console port |
| `MODELCTL_CI_BUILD_DIR`, `MODELCTL_CI_SAN_BUILD_DIR`, `MODELCTL_CI_CONSOLE_DIR` | per-lane build scratch under `/tmp/modelctl-lane-<slug>/`, so two lanes running `ci/checks.sh` do not share one cmake cache configured against a different source path |
| `CCACHE_DIR`, `CCACHE_BASEDIR`, `CCACHE_NOHASHDIR` | shared, path-independent ccache |

Note what it does **not** set: `MODELCTL_HOME` still points at the real
state directory, so a lane session that touches saved profiles touches
the real ones. Console walks stay scratch-safe the same way they always
have — `MODELCTL_WEB_SCRATCH=1` plus the five redirections.

## Paths

| What | Where |
| --- | --- |
| Lane worktrees | `/home/aaron/workspace/.lanes/<slug>` (`MODELCTL_LANES_ROOT`) |
| Ledger | `~/.local/share/modelctl/lanes.json` (`MODELCTL_LANES_LEDGER`) |
| Land lock | `~/.local/share/modelctl/lanes-land.lock` (`MODELCTL_LANES_LAND_LOCK`) |
| GPU lock | `~/.local/share/modelctl/gpu.lock` (`MODELCTL_GPU_LOCK`) |
| Build scratch | `/tmp/modelctl-lane-<slug>/` (`MODELCTL_LANES_SCRATCH`) |

The ledger and the locks sit under `MODELCTL_HOME` rather than
`~/.local/state/modelctl`, the same call `modelctl_display` and
`modelctl_remote_hands` made: every piece of modelctl state resolves
through `modelctl_paths` so one variable moves all of it, and a second
state root is the exact split `modelctl_paths` exists to prevent. The
filenames are the ones the order asked for.

Ledger schema (`version` 1):

```json
{
  "version": 1,
  "lanes": {
    "review-console": {
      "slug": "review-console",
      "branch": "lane/review-console",
      "path": "/home/aaron/workspace/.lanes/review-console",
      "main": "/home/aaron/workspace/moe-serving",
      "port_base": 9500,
      "port_end": 9509,
      "created": 1754130000.0,
      "base_commit": "bd867e5…",
      "submodules": {"referenced": ["llama.cpp"], "cloned_from_main": []}
    }
  }
}
```

## Writing orders for parallel lanes

This part lives with whoever writes the orders, not with the tool — the
tool cannot check it:

* **Parallel orders get disjoint file scopes.** Two lanes editing
  `modelctl.py` will both land, and the second one's rebase either
  conflicts (a stop, and a person's afternoon) or replays into a
  semantic conflict that only the re-run checks catch.
* **At most one in-flight order may advance the fork pin**, and fork
  orders are serialized. A pin advance moves the runtime under every
  other lane at once; two of them in flight cannot be rebased into a
  sensible order after the fact.
* **No two lanes benchmark at the same time.** They physically cannot,
  because of the GPU lock — but an order that assumes it can measure
  while another lane measures will just block, so write it knowing that.

## Deliberate limits

* `land` never stashes and never force-pushes; there is no `--force`
  anywhere in the land path.
* `sweep` never deletes on its own.
* A lane is not created inside the repository, and the ledger is not
  version-controlled: nothing about lane machinery ever appears in `git
  status` in the main checkout.
* `main_checkout()` resolves through `--git-common-dir`, so a session
  running `lane land` from inside a lane still lands into the main
  checkout rather than into itself.
