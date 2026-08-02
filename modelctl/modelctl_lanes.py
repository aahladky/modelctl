"""Parallel work lanes: a hidden worktree per session, landed on master.

Several Claude Code sessions run orders at once. Without machinery they
share one checkout and overwrite each other; with the obvious machinery
(branches, worktrees, merges) Aaron acquires a job he never asked for --
remembering which branch a session left work on. This module is the
third option: the lane exists for exactly as long as the session, it
lives outside the checkout where nothing lists it, and landing it is a
fast-forward of master. Aaron's surface is unchanged -- he pastes an
order into a session, and the work appears on master.

What that costs, and why each cost is paid here rather than by a person:

* **Nothing strands silently.** `land` is mandatory and it is the only
  exit; `list`/`sweep` flag every lane older than 24 h or with no live
  session, in every report. Sweep never deletes on its own -- it names
  what is stale and stops, because a lane with unmerged commits is
  somebody's work and only a person knows whether it is finished.
* **Master is never half-landed.** Rebase conflict, red checks, dirty
  lane: every one of them stops with the lane intact and master exactly
  as it was. There is no path in here that leaves master holding half a
  lane, which is the failure a person cannot undo by rerunning anything.
* **The main checkout is never stashed.** If it has operator edits at
  land time they are journalled as a commit first ("journal: local
  edits") and landed on top. A stash is a place work goes to be
  forgotten; a commit is in the log where the next session sees it.
* **The fork pin is not advanced as a side effect.** The journal commit
  excludes the submodule, and a lane whose fork checkout disagrees with
  the pin stops the land. Advancing the pin is a decision an order
  makes, never a thing that happens because a worktree was dirty.

Two shared resources the lanes contend for, both handled by one flock
each: the land lock (only one land at a time, so two lanes cannot both
fast-forward the same master) and the GPU lock (only one benchmark at a
time -- the night lane takes the same lock, so a lane bench and a 03:00
night job can never measure each other).

Ports: a lane gets a block of ten, allocated from the ledger, so two
scratch consoles never fight over 9293. The ledger lives in the state
directory, not the repo -- it is machine state, and a lane that appeared
in `git status` would be exactly the thing this module exists to hide.

Scratch: a lane's build trees live outside the lane, on the pool, and
both exits delete them. Neither half was true before 2026-08-02 -- they
defaulted into /tmp, which is tmpfs, and nothing ever came back for
them -- so every lane that ran CI took ~850 MB of RAM away from the
thing this machine exists to run, permanently. `sweep --orphans`
collects what the old teardown left behind.

This module must stay a leaf: no modelctl imports (paths/fsutil only).
"""
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import modelctl_fsutil
from modelctl_paths import STATE_DIR

LEDGER_VERSION = 1
BRANCH_PREFIX = "lane/"
JOURNAL_MESSAGE = "journal: local edits"

# Ten ports per lane, out of a range that steps around everything the rig
# already answers on: 9090 (metrics), 9292 (llama-swap), 9293 (console),
# 9294 (remote hands), 9411 (headless verify), 9728/9832 (fleet RPC).
# Twenty blocks is far more lanes than sessions.
PORT_BLOCK_SIZE = 10
PORT_RANGE_START = 9500
PORT_RANGE_END = 9699

# A lane older than this is flagged in every listing. Not deleted: age is
# evidence that something needs a person's attention, not permission to
# throw work away.
STALE_AFTER_SECONDS = 24 * 3600

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


class LaneError(Exception):
    """A refusal or a stop.

    Every raise site in this module is a point where the working trees
    and master are exactly as they were before the call -- the message
    is the whole outcome, and rerunning after fixing what it names is
    always safe.
    """


class LockBusy(LaneError):
    """A lock (the land lock or the GPU lock) is held and the wait ran out."""


# --- paths -----------------------------------------------------------------
# Read through functions, not frozen at import: the suite redirects these
# per-test, and STATE_DIR itself moves with MODELCTL_HOME.
#
# The ledger and the locks sit under STATE_DIR rather than the
# ~/.local/state/modelctl path the order named -- the same call
# modelctl_display and modelctl_remote_hands made, for the same reason:
# every other piece of modelctl state resolves through modelctl_paths so
# MODELCTL_HOME moves all of it at once, and a second state root is the
# exact split modelctl_paths exists to prevent. The filenames are the
# ones the order asked for.

def ledger_path():
    override = os.environ.get("MODELCTL_LANES_LEDGER", "").strip()
    return Path(override) if override else STATE_DIR / "lanes.json"


def land_lock_path():
    override = os.environ.get("MODELCTL_LANES_LAND_LOCK", "").strip()
    return Path(override) if override else STATE_DIR / "lanes-land.lock"


def gpu_lock_path():
    """The well-known GPU lock.

    Well-known is the entire point: a lane bench, the night lane and a
    hand-run sweep all have to name the same file, or the lock protects
    nothing. Anything outside modelctl can take it with `flock` on this
    path -- no modelctl code required.
    """
    override = os.environ.get("MODELCTL_GPU_LOCK", "").strip()
    return Path(override) if override else STATE_DIR / "gpu.lock"


def lanes_root(main=None):
    """Where lane worktrees live: a dot-directory beside the checkout.

    Beside rather than inside, because a worktree inside the repo shows
    up in `git status`, in `find`, and in every glob a future order
    writes -- and lanes are supposed to be invisible.
    """
    override = os.environ.get("MODELCTL_LANES_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(main or main_checkout()).parent / ".lanes"


# --- scratch ---------------------------------------------------------------
# /tmp on this machine is tmpfs, which is to say RAM the inference stack
# could be using. Two things were wrong with putting build trees there,
# and both are fixed here:
#
#   * the default lived in RAM at all -- ~850 MB per lane that ran CI,
#     against a pool with hundreds of gigabytes free and a warm ccache;
#   * nothing ever deleted it. `land` freed the worktree, the branch and
#     the port block and walked straight past the build trees CI had put
#     beside them, so on 2026-08-02 four lanes' worth (3.4 GB) sat in
#     tmpfs, one of them belonging to a lane that had been swept hours
#     earlier.
#
# Scratch is therefore named in one place, defaults to the pool, and is
# torn down by whatever tears the lane down.

LEGACY_SCRATCH_BASE = Path("/tmp")

# Used when ci/checks.sh cannot be read (a fixture checkout, a lane whose
# worktree is already gone). Names, not paths: the parent is whichever
# scratch base is configured.
CI_SCRATCH_FALLBACK = ("ci-build-cpu", "ci-build-san", "ci-console-dist")

_CI_DIR_DEFAULT_RE = re.compile(r"\$\{(MODELCTL_CI_[A-Z0-9_]+_DIR):-([^}]+)\}")


def scratch_base():
    """Where build scratch goes: the pool, not tmpfs.

    `MODELCTL_CI_SCRATCH_ROOT` puts it back on tmpfs (or anywhere else)
    for anyone who wants that -- ci/checks.sh reads the same variable, so
    one export moves every build tree the project makes.
    """
    override = os.environ.get("MODELCTL_CI_SCRATCH_ROOT", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "modelctl" / "ci"


def lane_scratch_base():
    """The directory lane scratch roots sit in."""
    override = os.environ.get("MODELCTL_LANES_SCRATCH", "").strip()
    return Path(override) if override else scratch_base()


def scratch_root(slug):
    """Per-lane build/scratch space, outside both the lane and the repo.

    Two lanes running ci/checks.sh at once would otherwise share one
    cmake cache configured against a different source path, and the
    second one to arrive silently reconfigures the first one's build.
    """
    return lane_scratch_base() / f"modelctl-lane-{slug}"


def scratch_search_dirs():
    """Every directory a lane's scratch can be found in.

    The configured bases, plus /tmp -- but /tmp only when nothing has
    been redirected. Every lane before 2026-08-02 built in /tmp, so a
    sweep blind to it would leave in RAM exactly the megabytes it exists
    to reclaim; and a caller that has redirected scratch (a scratch-safe
    walk, the test suite) has said that the machine's /tmp is not its to
    delete from.
    """
    redirected = bool(os.environ.get("MODELCTL_CI_SCRATCH_ROOT", "").strip()
                      or os.environ.get("MODELCTL_LANES_SCRATCH", "").strip())
    candidates = [lane_scratch_base(), scratch_base()]
    if not redirected:
        candidates.append(LEGACY_SCRATCH_BASE)
    out = []
    for path in candidates:
        path = Path(path)
        if path not in out:
            out.append(path)
    return out


def ci_scratch_names(main=None):
    """Leaf directory names ci/checks.sh builds into.

    Read out of the script rather than listed in this module. The list
    grows whenever someone adds a build to CI, and a second copy of it
    here would go stale silently -- which is the precise shape of the bug
    being fixed: teardown knew about the lane's worktree and not about
    the build trees CI had created beside it.
    """
    try:
        text = (Path(main or main_checkout()) / "ci" / "checks.sh").read_text()
    except (OSError, LaneError, subprocess.SubprocessError):
        return CI_SCRATCH_FALLBACK
    names = []
    for _var, default in _CI_DIR_DEFAULT_RE.findall(text):
        leaf = default.strip().rstrip("/").rsplit("/", 1)[-1]
        # A default that is itself a variable names nothing sweepable.
        if leaf and "$" not in leaf and leaf not in names:
            names.append(leaf)
    return tuple(names) or CI_SCRATCH_FALLBACK


def scratch_dir_names(slug, main=None):
    """The directory names that belong to `slug`, in any scratch base."""
    names = [f"modelctl-lane-{slug}"]
    for leaf in ci_scratch_names(main):
        # The convention sessions use when they export MODELCTL_CI_*_DIR
        # by hand rather than through `lane env`.
        names.append(f"{leaf}-lane-{slug}")
    return names


def scratch_dirs_for(slug, main=None, search_dirs=None):
    """Existing scratch directories belonging to `slug`."""
    out = []
    for base in (search_dirs if search_dirs is not None
                 else scratch_search_dirs()):
        for name in scratch_dir_names(slug, main):
            path = Path(base) / name
            if path not in out and path.is_dir() and not path.is_symlink():
                out.append(path)
    return out


def dir_bytes(path):
    """Blocks a directory occupies, the way du counts them.

    Blocks rather than apparent size because the number that matters is
    the RAM a tmpfs tree is holding, and that is what it allocated.
    """
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_blocks * 512
            except OSError:
                continue
    return total


def _remove_dir(path):
    """Delete one scratch directory; return its size, or None if it stayed."""
    size = dir_bytes(path)
    shutil.rmtree(path, ignore_errors=True)
    return None if path.exists() else size


def remove_scratch(slug, main=None, search_dirs=None):
    """Delete every scratch directory belonging to `slug`.

    Called by both exits from a lane -- `land` and `delete` -- because
    the lane system is what created these directories and nothing else
    is ever going to come back for them.
    """
    removed = []
    for path in scratch_dirs_for(slug, main=main, search_dirs=search_dirs):
        size = _remove_dir(path)
        if size is not None:
            removed.append({"path": str(path), "slug": slug, "bytes": size})
    return removed


def find_orphan_scratch(ledger=None, main=None, search_dirs=None, keep=()):
    """Lane-suffixed scratch directories whose lane no longer exists.

    The ledger is the authority on what a lane is: `start` writes it and
    both exits remove it, so a scratch directory whose slug is not in it
    is nobody's. `keep` spares a slug anyway, for the case the ledger
    cannot cover -- a session that exported the build dirs by hand and
    never took a ledger entry looks exactly like an orphan while it is
    still compiling.
    """
    live = set(read_ledger(ledger)["lanes"]) | set(keep)
    names = ci_scratch_names(main)
    found = []
    seen = set()
    for base in (search_dirs if search_dirs is not None
                 else scratch_search_dirs()):
        base = Path(base)
        try:
            entries = sorted(base.iterdir())
        except OSError:
            continue
        for path in entries:
            if path in seen or not path.is_dir() or path.is_symlink():
                continue
            slug = _slug_of(path.name, names)
            if slug is None or slug in live:
                continue
            seen.add(path)
            found.append({"path": str(path), "slug": slug,
                          "bytes": dir_bytes(path)})
    return found


def _slug_of(name, ci_names):
    """The lane slug a scratch directory name belongs to, or None."""
    prefixes = ["modelctl-lane-"] + [f"{leaf}-lane-" for leaf in ci_names]
    for prefix in prefixes:
        if name.startswith(prefix):
            slug = name[len(prefix):]
            # Validated, not just non-empty: this is the check standing
            # between a sweep and someone else's directory.
            return slug if _SLUG_RE.match(slug) else None
    return None


def sweep_scratch_orphans(ledger=None, main=None, search_dirs=None, keep=()):
    """Delete the orphans and return what went, each one named."""
    removed = []
    for orphan in find_orphan_scratch(ledger=ledger, main=main,
                                      search_dirs=search_dirs, keep=keep):
        size = _remove_dir(Path(orphan["path"]))
        if size is not None:
            removed.append({**orphan, "bytes": size})
    return removed


# --- git -------------------------------------------------------------------

def git(repo, *args, check=True, timeout=600, env=None):
    """Run one git command in `repo` and return the CompletedProcess."""
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=timeout,
                          env=env)
    if check and proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip()
                  or f"exit {proc.returncode}")
        raise LaneError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return proc


def main_checkout(start=None):
    """The main checkout, even when the caller is inside a lane.

    Resolved through --git-common-dir rather than from __file__: a lane
    is a full copy of this tree, so a session inside one that computed
    the repo root from its own module path would land a lane into
    itself. The common dir of a linked worktree is the main checkout's
    .git, which is the one thing that cannot be confused.
    """
    start = Path(start or Path(__file__).resolve().parent)
    common = git(start, "rev-parse", "--path-format=absolute",
                 "--git-common-dir").stdout.strip()
    if not common:
        raise LaneError(f"{start} is not inside a git checkout")
    return Path(common).parent


def _branch_exists(repo, branch):
    return git(repo, "rev-parse", "--verify", "--quiet",
               f"refs/heads/{branch}", check=False).returncode == 0


def _rev(repo, ref):
    return git(repo, "rev-parse", ref).stdout.strip()


def _current_branch(repo):
    proc = git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False)
    return proc.stdout.strip()


def _porcelain(repo, *pathspec):
    return [line for line in git(repo, "status", "--porcelain",
                                 *pathspec).stdout.splitlines() if line.strip()]


def submodule_paths(repo):
    """Submodule paths declared by .gitmodules, in declaration order."""
    proc = git(repo, "config", "-f", ".gitmodules", "--get-regexp",
               r"^submodule\..*\.path$", check=False)
    out = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip():
            out.append(parts[1].strip())
    return out


# --- ledger ----------------------------------------------------------------

def read_ledger(path=None):
    p = Path(path or ledger_path())
    if not p.exists():
        return {"version": LEDGER_VERSION, "lanes": {}}
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise LaneError(f"lane ledger {p} is unreadable: {e}")
    if not isinstance(data, dict):
        raise LaneError(f"lane ledger {p} is not an object")
    data.setdefault("version", LEDGER_VERSION)
    if not isinstance(data.get("lanes"), dict):
        data["lanes"] = {}
    return data


def write_ledger(data, path=None):
    p = Path(path or ledger_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    modelctl_fsutil.atomic_write_text(
        p, json.dumps(data, indent=2, sort_keys=True) + "\n")


def allocate_port_block(data):
    """Lowest free block base. Deterministic, so a freed block is reused
    rather than the range drifting upward until it runs out."""
    taken = set()
    for entry in data.get("lanes", {}).values():
        base = entry.get("port_base")
        if isinstance(base, int):
            taken.add(base)
    for base in range(PORT_RANGE_START,
                      PORT_RANGE_END - PORT_BLOCK_SIZE + 2, PORT_BLOCK_SIZE):
        if base not in taken:
            return base
    raise LaneError(
        f"no free port block in {PORT_RANGE_START}-{PORT_RANGE_END}: "
        f"{len(taken)} lanes hold one. `modelctl lane list` shows them.")


# --- environment a lane session builds in ----------------------------------

def build_env(root, base=None):
    """ccache settings that make a lane rebuild of the fork share the
    main checkout's cache.

    ccache hashes the absolute path of every source file, so the same
    commit built from a second worktree misses on every object and a
    fork-touching lane pays a full rebuild. CCACHE_BASEDIR rewrites
    absolute paths under the checkout root to paths relative to the
    build's cwd, and hash_dir=false keeps the cwd itself out of the
    hash; with both set, and both builds run from their own root, the
    lane hits the entries the main checkout created.

    Nothing here relaxes correctness: no sloppiness is set, so headers,
    macros and compiler identity are hashed as strictly as ccache's
    defaults. The only thing made path-independent is the path.
    """
    env = dict(os.environ if base is None else base)
    env.setdefault("CCACHE_DIR", str(Path.home() / ".cache" / "ccache"))
    env["CCACHE_BASEDIR"] = str(Path(root))
    env["CCACHE_NOHASHDIR"] = "1"
    return env


def lane_env(entry, base=None):
    """The full environment for a lane: ccache, scratch dirs, ports."""
    slug = entry["slug"]
    scratch = scratch_root(slug)
    env = build_env(entry["path"], base=base)
    env["MODELCTL_LANE"] = slug
    env["MODELCTL_LANE_PORT_BASE"] = str(entry["port_base"])
    env["MODELCTL_WEB_BIND"] = f"127.0.0.1:{entry['port_base']}"
    env["MODELCTL_CI_BUILD_DIR"] = str(scratch / "build-cpu")
    env["MODELCTL_CI_SAN_BUILD_DIR"] = str(scratch / "build-san")
    env["MODELCTL_CI_CONSOLE_DIR"] = str(scratch / "console-dist")
    return env


LANE_ENV_KEYS = ("MODELCTL_LANE", "MODELCTL_LANE_PORT_BASE",
                 "MODELCTL_WEB_BIND", "MODELCTL_CI_BUILD_DIR",
                 "MODELCTL_CI_SAN_BUILD_DIR", "MODELCTL_CI_CONSOLE_DIR",
                 "CCACHE_DIR", "CCACHE_BASEDIR", "CCACHE_NOHASHDIR")


def env_exports(entry, base=None):
    """`lane env` output: shell export lines, in a stable order."""
    env = lane_env(entry, base=base)
    return [f"export {k}={env[k]}" for k in LANE_ENV_KEYS if k in env]


# --- sessions --------------------------------------------------------------

def sessions_for(path, proc_root="/proc"):
    """PIDs whose working directory is inside the lane.

    Liveness by observation rather than by registration: the session
    that owns a lane is a Claude Code process with its cwd in it, and it
    never told anyone it started. A registered pid would go stale the
    first time a session was killed, which is precisely the case this
    has to detect.
    """
    root = str(Path(path).resolve())
    out = []
    try:
        names = os.listdir(proc_root)
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        try:
            cwd = os.readlink(os.path.join(proc_root, name, "cwd"))
        except OSError:
            continue
        if cwd == root or cwd.startswith(root + os.sep):
            out.append(int(name))
    return sorted(out)


# --- start -----------------------------------------------------------------

def _validate_slug(slug):
    if not _SLUG_RE.match(slug or ""):
        raise LaneError(
            f"invalid lane slug {slug!r}: lowercase letters, digits and "
            f"dashes only, starting with a letter or digit, 40 chars max "
            f"(it becomes a branch name and a directory name)")


# Cloning a submodule from a local directory is what the fallback below
# does, and git refuses local-path clones by default (CVE-2022-39253).
# Allowed here because the path is not attacker-supplied: it is the main
# checkout on this machine, the same tree the caller already runs code
# from. Harmless for the declared ssh URL.
_LOCAL_CLONE = ("-c", "protocol.file.allow=always")


def _init_submodules(main, lane):
    """Check out the lane's submodules, sharing the main checkout's
    objects.

    --reference makes the clone an alternate of the object store that is
    already on disk: no refetch of a multi-gigabyte fork history, and no
    second copy of it. The reference is the main checkout's submodule,
    which is never garbage-collected out from under the lane because the
    lane is deleted at land time and the main checkout outlives it.

    The fallback matters more than it looks. The clone source is the
    configured URL -- gitea -- and --reference only supplies objects, it
    does not remove the need to reach it. A lane could therefore not be
    created while gitea was down, and "start a lane" is the one command
    that has to work on a bad day. So a failed clone is retried with the
    URL pointed at the main checkout's own submodule, which has every
    object the pin needs and is on the same disk. The override is passed
    with -c, never written: worktrees share one .git/config, and
    persisting it would repoint the main checkout's submodule too.
    """
    referenced, local = [], []
    for path in submodule_paths(lane):
        reference = Path(main) / path
        args = list(_LOCAL_CLONE) + ["submodule", "update", "--init"]
        has_reference = (reference / ".git").exists()
        if has_reference:
            args += ["--reference", str(reference)]
        proc = git(lane, *args, "--", path, check=False, timeout=1800)
        if proc.returncode == 0:
            referenced.append(path)
            continue
        if not has_reference:
            raise LaneError(
                f"git submodule update --init -- {path} failed in {lane}: "
                f"{(proc.stderr or proc.stdout).strip()}")
        name = _submodule_name(lane, path) or path
        git(lane, *_LOCAL_CLONE, "-c", f"submodule.{name}.url={reference}",
            "submodule", "update", "--init", "--reference", str(reference),
            "--", path, timeout=1800)
        local.append(path)
    return {"referenced": referenced, "cloned_from_main": local}


def _submodule_name(repo, path):
    proc = git(repo, "config", "-f", ".gitmodules", "--get-regexp",
               r"^submodule\..*\.path$", check=False)
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip() == path:
            return parts[0][len("submodule."):-len(".path")]
    return None


def start(slug, main=None, root=None, ledger=None, clock=time.time):
    """Create a lane and return its ledger entry."""
    _validate_slug(slug)
    main = Path(main or main_checkout())
    root = Path(root if root is not None else lanes_root(main))
    path = root / slug
    branch = BRANCH_PREFIX + slug

    with modelctl_fsutil.state_lock():
        data = read_ledger(ledger)
        if slug in data["lanes"]:
            raise LaneError(
                f"lane {slug} already exists at "
                f"{data['lanes'][slug].get('path')} -- pick another slug, or "
                f"land that one first")
        if path.exists():
            raise LaneError(f"{path} already exists but no lane claims it; "
                            f"`modelctl lane sweep` lists what is known")
        if _branch_exists(main, branch):
            raise LaneError(
                f"branch {branch} already exists in {main} but no lane claims "
                f"it -- a previous land stopped part way. Inspect it before "
                f"reusing the slug.")

        port_base = allocate_port_block(data)
        root.mkdir(parents=True, exist_ok=True)
        git(main, "worktree", "add", "-b", branch, str(path), "master")
        try:
            submodules = _init_submodules(main, path)
        except LaneError:
            # A lane whose fork checkout never initialised is not a lane;
            # rolling all the way back is better than leaving a half one
            # for a session to discover at build time.
            git(main, "worktree", "remove", "--force", str(path), check=False)
            git(main, "branch", "-D", branch, check=False)
            git(main, "worktree", "prune", check=False)
            raise

        entry = {
            "slug": slug,
            "branch": branch,
            "path": str(path),
            "main": str(main),
            "port_base": port_base,
            "port_end": port_base + PORT_BLOCK_SIZE - 1,
            "created": clock(),
            "base_commit": _rev(main, "master"),
            "submodules": submodules,
        }
        data["lanes"][slug] = entry
        write_ledger(data, ledger)
    return entry


# --- listing and sweeping --------------------------------------------------

def describe(entry, clock=time.time, proc_root="/proc"):
    """One lane's live state: age, sessions, unlanded commits, flags."""
    path = Path(entry["path"])
    main = Path(entry.get("main") or main_checkout())
    age = max(0.0, clock() - float(entry.get("created") or 0))
    sessions = sessions_for(path, proc_root) if path.exists() else []
    flags = []
    if not path.exists():
        flags.append("worktree is missing")
    if age > STALE_AFTER_SECONDS:
        flags.append(f"older than {STALE_AFTER_SECONDS // 3600} h")
    if not sessions:
        flags.append("no live session")
    ahead = None
    if _branch_exists(main, entry["branch"]):
        proc = git(main, "rev-list", "--count",
                   f"master..{entry['branch']}", check=False)
        if proc.returncode == 0 and proc.stdout.strip().isdigit():
            ahead = int(proc.stdout.strip())
    else:
        flags.append("branch is missing")
    return {**entry, "age_seconds": age, "sessions": sessions,
            "unlanded_commits": ahead, "flags": flags}


def lane_list(ledger=None, clock=time.time, proc_root="/proc"):
    data = read_ledger(ledger)
    return [describe(data["lanes"][slug], clock=clock, proc_root=proc_root)
            for slug in sorted(data["lanes"])]


def delete(slug, ledger=None, force=False, clock=time.time,
           proc_root="/proc"):
    """Delete a named lane, its branch and its port block.

    Refuses a lane with unlanded commits unless forced: sweep exists to
    find work that is about to be forgotten, and deleting it by default
    would make sweep the thing that forgets it.
    """
    with modelctl_fsutil.state_lock():
        data = read_ledger(ledger)
        entry = data["lanes"].get(slug)
        if entry is None:
            raise LaneError(f"no lane named {slug}")
        state = describe(entry, clock=clock, proc_root=proc_root)
        if state["unlanded_commits"] and not force:
            raise LaneError(
                f"lane {slug} has {state['unlanded_commits']} commit(s) not on "
                f"master. Land it, or delete it with --force if the work is "
                f"meant to be thrown away.")
        if state["sessions"] and not force:
            raise LaneError(
                f"lane {slug} has a live session (pid "
                f"{', '.join(str(p) for p in state['sessions'])}). "
                f"--force deletes it anyway.")
        main = Path(entry.get("main") or main_checkout())
        git(main, "worktree", "remove", "--force", entry["path"], check=False)
        git(main, "worktree", "prune", check=False)
        if Path(entry["path"]).exists():
            shutil.rmtree(entry["path"], ignore_errors=True)
        if _branch_exists(main, entry["branch"]):
            git(main, "branch", "-D", entry["branch"], check=False)
        state["scratch_removed"] = remove_scratch(slug, main=main)
        del data["lanes"][slug]
        write_ledger(data, ledger)
    return state


# --- the land lock ---------------------------------------------------------

@contextmanager
def _exclusive(path, timeout=None, poll=0.2, note=None, clock=time.monotonic):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    deadline = None if timeout is None else clock() + timeout
    try:
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if deadline is not None and clock() >= deadline:
                    raise LockBusy(
                        f"the lock at {path} is held by another run "
                        f"({_lock_holder(path)}) and the {timeout}s wait ran "
                        f"out")
                time.sleep(poll)
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps({"pid": os.getpid(), "since": time.time(),
                                 "what": note or ""}) + "\n")
            fh.flush()
            yield path
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def _lock_holder(path):
    try:
        return json.loads(Path(path).read_text() or "{}")
    except (OSError, json.JSONDecodeError):
        return {}


@contextmanager
def gpu_lock(timeout=None, path=None, note=None, poll=0.2):
    """Hold the well-known GPU lock for the duration of the block.

    Every benchmark takes this, from anywhere: a lane's, a hand-run
    sweep's, the night lane's. Two benchmarks sharing the GPUs measure
    each other, and the 2026-08-01 battery is void for exactly that
    reason -- a lock is the only version of "don't run two at once" that
    survives nobody being awake to check.
    """
    with _exclusive(path or gpu_lock_path(), timeout=timeout, poll=poll,
                    note=note or "gpu") as p:
        yield p


def run_with_gpu_lock(argv, timeout=None, path=None, cwd=None, env=None):
    """Run argv holding the GPU lock; return its exit status."""
    if not argv:
        raise LaneError("gpu-lock needs a command to run after `--`")
    with gpu_lock(timeout=timeout, path=path, note=" ".join(argv)):
        return subprocess.run(list(argv), cwd=cwd, env=env).returncode


# --- land ------------------------------------------------------------------

def default_checks(lane_path, env, log_path=None, timeout=7200):
    """Run the repository's full checks inside the lane.

    Full, never --quick: the reason to re-run at all is that master moved
    under the lane, and the checks that would catch a semantic conflict
    (the CPU-only build, the sanitizer pass, the suite) are the ones
    --quick skips.
    """
    script = Path(lane_path) / "ci" / "checks.sh"
    if not script.exists():
        raise LaneError(f"{script} is missing; cannot verify the rebase")
    proc = subprocess.run([str(script)], cwd=str(lane_path), env=env,
                          capture_output=True, text=True, timeout=timeout)
    output = proc.stdout + proc.stderr
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(output)
    return proc.returncode, output


def _journal_main(main, subs):
    """Commit the main checkout's operator edits, excluding submodules.

    Never stashed, per the invariant, and never including a submodule
    pointer: a dirty submodule means the fork checkout moved, and turning
    that into a commit would advance the pin because a worktree happened
    to be dirty. That is a decision, so it stops the land instead.
    """
    if not _porcelain(main):
        return None
    exclude = [f":(exclude){p}" for p in subs]
    git(main, "add", "-A", "--", ".", *exclude)
    staged = git(main, "diff", "--cached", "--name-only").stdout.split()
    if not staged:
        return None
    git(main, "commit", "-m", JOURNAL_MESSAGE)
    return {"commit": _rev(main, "HEAD"), "files": staged}


def land(slug, ledger=None, checks=None, clock=time.time, timeout=None,
         proc_root="/proc"):
    """Land a lane onto master. Returns a report; raises LaneError on a
    stop, always with master untouched and the lane intact."""
    report = {"slug": slug, "landed": False, "journal": None,
              "rebase": None, "checks": "not run", "master_before": None,
              "master_after": None, "checks_log": None,
              "scratch_removed": [], "scratch_bytes": 0}
    with _exclusive(land_lock_path(), timeout=timeout, note=f"land {slug}"):
        data = read_ledger(ledger)
        entry = data["lanes"].get(slug)
        if entry is None:
            raise LaneError(
                f"no lane named {slug}; `modelctl lane list` shows what "
                f"exists")
        lane = Path(entry["path"])
        main = Path(entry.get("main") or main_checkout())
        branch = entry["branch"]
        if not lane.exists():
            raise LaneError(
                f"lane {slug}'s worktree {lane} is gone. Nothing was landed. "
                f"`modelctl lane sweep --delete {slug}` clears the ledger "
                f"entry once you have checked branch {branch} for commits.")
        report["master_before"] = _rev(main, "master")

        # 1. The lane must be committed. Land is not a stasher, and a
        #    file that is uncommitted at land time is work that would be
        #    deleted with the worktree three steps later.
        dirty = _porcelain(lane)
        if dirty:
            raise LaneError(
                f"lane {slug} has uncommitted changes:\n  "
                + "\n  ".join(dirty[:20])
                + f"\ncommit them in {lane} and land again. Land never "
                  f"stashes.")

        # 2. The main checkout must be on master; landing anywhere else
        #    would fast-forward a branch nobody asked about.
        current = _current_branch(main) or "a detached HEAD"
        if current != "master":
            raise LaneError(
                f"the main checkout {main} is on {current}, not master. "
                f"Nothing was landed.")

        # 3. Operator edits become a commit, and the submodule is the one
        #    thing that never does.
        subs = submodule_paths(main)
        report["journal"] = _journal_main(main, subs)
        left = _porcelain(main)
        if left:
            raise LaneError(
                "the main checkout still has changes that cannot be "
                "journalled (the submodule pointer moved -- advancing the "
                "fork pin is an order's decision, not a side effect of "
                "landing):\n  " + "\n  ".join(left[:20]))

        # 4. Rebase, unless master has not moved under the lane at all.
        #    "Clean no-op replay" is exactly that case: master is already
        #    an ancestor of the lane, so the replay would change nothing
        #    and the checks the lane already ran still describe this code.
        noop = git(main, "merge-base", "--is-ancestor", "master", branch,
                   check=False).returncode == 0
        if noop:
            report["rebase"] = "no-op (master has not moved under the lane)"
        else:
            proc = git(lane, "rebase", "master", check=False, timeout=1800)
            if proc.returncode != 0:
                git(lane, "rebase", "--abort", check=False)
                raise LaneError(
                    f"lane {slug} does not rebase cleanly onto master:\n"
                    f"{(proc.stderr or proc.stdout).strip()}\n"
                    f"The rebase was aborted: master is untouched and the "
                    f"lane is exactly as it was. Resolve it in {lane} "
                    f"(`git rebase master`) and land again.")
            report["rebase"] = "replayed onto master"

            # 5. Rebased code is code no run has ever seen. The lane's
            #    green checks were about a different parent.
            runner = checks or default_checks
            log_path = scratch_root(slug) / "land-checks.log"
            rc, output = runner(lane, lane_env(entry), log_path)
            report["checks_log"] = str(log_path)
            if rc != 0:
                report["checks"] = "failed"
                tail = "\n".join(output.strip().splitlines()[-25:])
                raise LaneError(
                    f"ci/checks.sh failed in lane {slug} after the rebase "
                    f"(exit {rc}). master is untouched and the lane is "
                    f"intact -- fix it in {lane} and land again.\n"
                    f"full log: {log_path}\n{tail}")
            report["checks"] = "passed"

        # 6. Fast-forward only. A merge commit here would mean master
        #    moved between the rebase and now, which the land lock makes
        #    impossible -- so if --ff-only ever fails, something outside
        #    this module moved master and stopping is correct.
        git(main, "merge", "--ff-only", branch)
        report["master_after"] = _rev(main, "master")

        # 7. The lane is gone the moment its work is on master. --force
        #    is safe here and nowhere else: step 1 proved there was
        #    nothing uncommitted, so all it discards is build scratch.
        git(main, "worktree", "remove", "--force", str(lane), check=False)
        git(main, "worktree", "prune", check=False)
        if lane.exists():
            shutil.rmtree(lane, ignore_errors=True)
        git(main, "branch", "-d", branch, check=False)
        if _branch_exists(main, branch):
            git(main, "branch", "-D", branch, check=False)

        # 8. And so is its scratch. The checks log lived in there, so the
        #    report stops naming a file that no longer exists -- a failed
        #    run keeps its log, because that is the run whose log anyone
        #    wants, and a failed run never reaches this line.
        report["scratch_removed"] = remove_scratch(slug, main=main)
        report["scratch_bytes"] = sum(d["bytes"]
                                      for d in report["scratch_removed"])
        if report["checks_log"] and not Path(report["checks_log"]).exists():
            report["checks_log"] = None

        with modelctl_fsutil.state_lock():
            data = read_ledger(ledger)
            data["lanes"].pop(slug, None)
            write_ledger(data, ledger)
        report["landed"] = True
        report["ports_freed"] = [entry["port_base"], entry["port_end"]]
    return report
