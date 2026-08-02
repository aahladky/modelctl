"""Lanes: lifecycle, the ledger, and every stop that keeps master whole.

Fixture repositories, never the real one. Each test builds a throwaway
superproject with a submodule in a temp dir and drives the real git
commands against it -- the interesting behaviour here IS git's (does a
rebase abort leave the branch where it was, does --ff-only refuse, does
--reference produce an alternates file), and a mocked git would only
prove that the mock agrees with the assertions.

What is worth breaking on, in order:

  * a stop is a stop: conflict, red checks, dirty lane and a moved fork
    pin each leave master byte-identical and the lane exactly as it was,
    because "landed half of it" is the one failure nobody can undo by
    rerunning;
  * the main checkout's operator edits become a commit and never a
    stash, and that commit never carries the submodule;
  * ports are allocated from the ledger and freed on land, or two
    scratch consoles fight over one port and both walks are void;
  * the GPU lock actually excludes, in-process and across processes,
    since it is the only thing standing between a lane bench and a
    03:00 night job.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl_fsutil
import modelctl_lanes as lanes


def run_git(repo, *args, check=True):
    """Test-side git. Local-path clones are allowed for the fixtures."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "-c", "protocol.file.allow=always", *args],
        capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"fixture git {' '.join(args)} failed in {repo}: "
                             f"{proc.stderr or proc.stdout}")
    return proc.stdout.strip()


CHECKS_STUB = """#!/bin/sh
# Fixture stand-in for ci/checks.sh.
echo "checks ran in $PWD"
echo "CCACHE_BASEDIR=$CCACHE_BASEDIR"
exit ${FIXTURE_CHECKS_EXIT:-0}
"""


class LaneCase(unittest.TestCase):
    """A fixture superproject, a fixture fork, and a lanes root."""

    def setUp(self):
        tmp = TemporaryDirectory(prefix="modelctl-lanes-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

        self.state = self.tmp / "state"
        self.state.mkdir()
        # state_lock() reads the STATE_DIR frozen into fsutil at import.
        patcher = mock.patch.object(modelctl_fsutil, "STATE_DIR", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Both scratch bases are redirected, and setting either one also
        # takes the machine's /tmp out of scratch_search_dirs(): these
        # tests delete whole directories under those roots.
        env = mock.patch.dict(os.environ, {
            "MODELCTL_LANES_LEDGER": str(self.state / "lanes.json"),
            "MODELCTL_LANES_LAND_LOCK": str(self.state / "lanes-land.lock"),
            "MODELCTL_GPU_LOCK": str(self.state / "gpu.lock"),
            "MODELCTL_LANES_SCRATCH": str(self.tmp / "scratch"),
            "MODELCTL_CI_SCRATCH_ROOT": str(self.tmp / "ci-scratch"),
            "MODELCTL_LANES_ROOT": str(self.tmp / ".lanes"),
        })
        env.start()
        self.addCleanup(env.stop)
        self.ledger = self.state / "lanes.json"
        self.root = self.tmp / ".lanes"
        self.scratch = self.tmp / "scratch"
        self.ci_scratch = self.tmp / "ci-scratch"
        self.scratch.mkdir()
        self.ci_scratch.mkdir()

        self.fork = self.init_repo(self.tmp / "forks" / "fork")
        self.commit(self.fork, "kernel.c", "int one(void);\n", "fork: first")
        self.main = self.init_repo(self.tmp / "main")
        self.commit(self.main, "a.txt", "one\n", "main: first")
        checks = self.main / "ci" / "checks.sh"
        checks.parent.mkdir()
        checks.write_text(CHECKS_STUB)
        checks.chmod(0o755)
        run_git(self.main, "add", "-A")
        run_git(self.main, "commit", "-qm", "main: checks")
        run_git(self.main, "submodule", "add", "-q", str(self.fork), "fork")
        run_git(self.main, "commit", "-qm", "main: add fork")

        self.proc_root = self.tmp / "proc"
        self.proc_root.mkdir()
        self.checks_calls = []

    # --- fixture helpers ---------------------------------------------------

    def init_repo(self, path):
        path.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "-b", "master", str(path)],
                       check=True, capture_output=True)
        for key, value in (("user.email", "lane@test.invalid"),
                           ("user.name", "lane test"),
                           ("commit.gpgsign", "false"),
                           ("protocol.file.allow", "always")):
            run_git(path, "config", key, value)
        return path

    def commit(self, repo, name, text, message):
        path = Path(repo) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        run_git(repo, "add", "-A")
        run_git(repo, "commit", "-qm", message)
        return run_git(repo, "rev-parse", "HEAD")

    def fake_checks(self, lane_path, env, log_path=None):
        self.checks_calls.append((Path(lane_path), dict(env), log_path))
        return 0, "fixture checks: pass\n"

    def failing_checks(self, lane_path, env, log_path=None):
        self.checks_calls.append((Path(lane_path), dict(env), log_path))
        return 1, "fixture checks: FAIL cache tests\n"

    def start(self, slug="alpha", clock=time.time):
        return lanes.start(slug, main=self.main, root=self.root,
                           ledger=self.ledger, clock=clock)

    def land(self, slug="alpha", checks=None, timeout=10):
        return lanes.land(slug, ledger=self.ledger,
                          checks=checks or self.fake_checks, timeout=timeout,
                          proc_root=str(self.proc_root))

    def entries(self):
        return lanes.read_ledger(self.ledger)["lanes"]

    def fake_session(self, pid, cwd):
        d = self.proc_root / str(pid)
        d.mkdir(parents=True)
        os.symlink(str(cwd), str(d / "cwd"))

    def make_scratch(self, base, name, kbytes=4):
        """A stand-in build tree, with enough bytes in it to be counted."""
        path = Path(base) / name
        (path / "CMakeFiles").mkdir(parents=True)
        (path / "CMakeCache.txt").write_text("x" * (kbytes * 1024))
        return path


# --- start -----------------------------------------------------------------

class TestStart(LaneCase):
    def test_start_creates_worktree_branch_and_ledger_entry(self):
        entry = self.start("alpha")
        lane = Path(entry["path"])
        self.assertEqual(lane, self.root / "alpha")
        self.assertTrue((lane / "a.txt").exists())
        self.assertEqual(entry["branch"], "lane/alpha")
        self.assertEqual(run_git(lane, "rev-parse", "--abbrev-ref", "HEAD"),
                         "lane/alpha")
        self.assertEqual(entry["base_commit"],
                         run_git(self.main, "rev-parse", "master"))
        self.assertEqual(self.entries()["alpha"]["path"], str(lane))

    def test_lane_lives_outside_the_checkout(self):
        # The whole invisibility claim: nothing about a lane shows up in
        # the main checkout's status or its directory listing.
        entry = self.start("alpha")
        self.assertNotIn(str(self.main), str(Path(entry["path"]).parent))
        self.assertEqual(run_git(self.main, "status", "--porcelain"), "")

    def test_duplicate_slug_is_refused(self):
        self.start("alpha")
        with self.assertRaises(lanes.LaneError) as cm:
            self.start("alpha")
        self.assertIn("already exists", str(cm.exception))
        self.assertEqual(len(self.entries()), 1)

    def test_existing_directory_without_a_ledger_entry_is_refused(self):
        (self.root / "alpha").mkdir(parents=True)
        with self.assertRaises(lanes.LaneError) as cm:
            self.start("alpha")
        self.assertIn("already exists", str(cm.exception))
        self.assertEqual(self.entries(), {})

    def test_existing_branch_without_a_ledger_entry_is_refused(self):
        run_git(self.main, "branch", "lane/alpha")
        with self.assertRaises(lanes.LaneError) as cm:
            self.start("alpha")
        self.assertIn("stopped part way", str(cm.exception))
        self.assertEqual(self.entries(), {})

    def test_bad_slugs_are_refused_before_anything_is_created(self):
        for slug in ("Alpha", "a/b", "", "-lead", "x" * 41, "with space"):
            with self.subTest(slug=slug):
                with self.assertRaises(lanes.LaneError):
                    self.start(slug)
        self.assertEqual(self.entries(), {})
        self.assertFalse(self.root.exists())


class TestSubmoduleReference(LaneCase):
    def _alternates(self, lane):
        gitdir = run_git(lane / "fork", "rev-parse", "--absolute-git-dir")
        return Path(gitdir) / "objects" / "info" / "alternates"

    def test_submodule_is_checked_out_sharing_the_main_objects(self):
        entry = self.start("alpha")
        lane = Path(entry["path"])
        self.assertTrue((lane / "fork" / "kernel.c").exists())
        self.assertEqual(entry["submodules"]["referenced"], ["fork"])
        alternates = self._alternates(lane)
        self.assertTrue(alternates.exists(),
                        "no alternates file: the lane refetched the fork "
                        "instead of sharing the main checkout's objects")
        self.assertIn(str(self.main), alternates.read_text())

    def break_the_declared_url(self):
        """Both places git looks: .gitmodules and the shared .git/config
        (worktrees share it, so the config value is what a lane reads)."""
        run_git(self.main, "config", "-f", ".gitmodules",
                "submodule.fork.url", str(self.tmp / "gone"))
        run_git(self.main, "commit", "-aqm", "main: break the fork url")
        run_git(self.main, "config", "submodule.fork.url",
                str(self.tmp / "gone"))

    def test_unreachable_declared_url_falls_back_to_the_main_checkout(self):
        # gitea being down must not stop a lane from starting.
        self.break_the_declared_url()
        entry = self.start("alpha")
        lane = Path(entry["path"])
        self.assertTrue((lane / "fork" / "kernel.c").exists())
        self.assertEqual(entry["submodules"]["cloned_from_main"], ["fork"])
        self.assertTrue(self._alternates(lane).exists())

    def test_submodule_failure_rolls_the_whole_lane_back(self):
        # A half-made lane is worse than none: the session would only
        # find out at build time. Both the declared URL and the local
        # fallback are unusable here, which is the only way the checkout
        # can genuinely fail.
        self.break_the_declared_url()
        real_git = lanes.git

        def refuse_submodule_clones(repo, *args, **kwargs):
            if "submodule" in args:
                if kwargs.get("check", True):
                    raise lanes.LaneError("fixture: the fork is unreachable")
                return subprocess.CompletedProcess(
                    args, 1, "", "fixture: the fork is unreachable")
            return real_git(repo, *args, **kwargs)

        with mock.patch.object(lanes, "git", refuse_submodule_clones):
            with self.assertRaises(lanes.LaneError):
                self.start("alpha")

        self.assertEqual(self.entries(), {})
        self.assertFalse((self.root / "alpha").exists())
        self.assertEqual(run_git(self.main, "branch", "--list", "lane/alpha"),
                         "")


# --- ports -----------------------------------------------------------------

class TestPortBlocks(LaneCase):
    def test_blocks_are_disjoint_and_ten_wide(self):
        a = self.start("alpha")
        b = self.start("bravo")
        self.assertEqual((a["port_base"], a["port_end"]), (9500, 9509))
        self.assertEqual((b["port_base"], b["port_end"]), (9510, 9519))
        self.assertEqual(b["port_base"] - a["port_base"],
                         lanes.PORT_BLOCK_SIZE)

    def test_landing_frees_the_block_for_the_next_lane(self):
        a = self.start("alpha")
        self.start("bravo")
        self.commit(Path(a["path"]), "lane.txt", "work\n", "lane: work")
        report = self.land("alpha")
        self.assertEqual(report["ports_freed"], [9500, 9509])
        again = self.start("charlie")
        self.assertEqual(again["port_base"], 9500)

    def test_exhausted_range_refuses_rather_than_overlapping(self):
        data = {"version": 1, "lanes": {
            f"l{base}": {"slug": f"l{base}", "port_base": base}
            for base in range(lanes.PORT_RANGE_START,
                              lanes.PORT_RANGE_END, lanes.PORT_BLOCK_SIZE)}}
        with self.assertRaises(lanes.LaneError) as cm:
            lanes.allocate_port_block(data)
        self.assertIn("no free port block", str(cm.exception))


# --- land ------------------------------------------------------------------

class TestLandCleanReplay(LaneCase):
    def test_land_fast_forwards_master_and_removes_the_lane(self):
        entry = self.start("alpha")
        lane = Path(entry["path"])
        sha = self.commit(lane, "lane.txt", "work\n", "lane: work")

        report = self.land("alpha")

        self.assertTrue(report["landed"])
        self.assertEqual(run_git(self.main, "rev-parse", "master"), sha)
        self.assertEqual((self.main / "lane.txt").read_text(), "work\n")
        self.assertFalse(lane.exists())
        self.assertEqual(run_git(self.main, "branch", "--list", "lane/alpha"),
                         "")
        self.assertEqual(self.entries(), {})
        self.assertNotIn("lane/alpha",
                         run_git(self.main, "worktree", "list"))

    def test_untouched_master_skips_the_re_run_of_checks(self):
        # The lane already ran the checks on exactly this code. Re-running
        # them for a replay that changes nothing is minutes per land.
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        report = self.land("alpha")
        self.assertEqual(self.checks_calls, [])
        self.assertIn("no-op", report["rebase"])
        self.assertEqual(report["checks"], "not run")

    def test_land_is_a_fast_forward_not_a_merge_commit(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        self.land("alpha")
        parents = run_git(self.main, "rev-list", "--parents", "-n", "1",
                          "master").split()
        self.assertEqual(len(parents), 2, "master gained a merge commit")


class TestLandAfterMasterMoved(LaneCase):
    def setUp(self):
        super().setUp()
        self.entry = self.start("alpha")
        self.lane = Path(self.entry["path"])

    def test_rebase_replays_and_re_runs_the_full_checks(self):
        self.commit(self.lane, "lane.txt", "work\n", "lane: work")
        self.commit(self.main, "b.txt", "meanwhile\n", "main: other work")

        report = self.land("alpha")

        self.assertTrue(report["landed"])
        self.assertEqual(report["checks"], "passed")
        self.assertEqual(len(self.checks_calls), 1)
        self.assertEqual(self.checks_calls[0][0], self.lane)
        self.assertEqual((self.main / "lane.txt").read_text(), "work\n")
        self.assertEqual((self.main / "b.txt").read_text(), "meanwhile\n")

    def test_checks_run_with_the_lane_environment(self):
        self.commit(self.lane, "lane.txt", "work\n", "lane: work")
        self.commit(self.main, "b.txt", "meanwhile\n", "main: other work")
        self.land("alpha")
        env = self.checks_calls[0][1]
        self.assertEqual(env["CCACHE_BASEDIR"], str(self.lane))
        self.assertEqual(env["MODELCTL_LANE"], "alpha")
        self.assertIn("modelctl-lane-alpha", env["MODELCTL_CI_BUILD_DIR"])

    def test_red_checks_stop_with_master_untouched_and_the_lane_intact(self):
        sha = self.commit(self.lane, "lane.txt", "work\n", "lane: work")
        self.commit(self.main, "b.txt", "meanwhile\n", "main: other work")
        before = run_git(self.main, "rev-parse", "master")

        with self.assertRaises(lanes.LaneError) as cm:
            self.land("alpha", checks=self.failing_checks)

        self.assertIn("checks.sh failed", str(cm.exception))
        self.assertIn("master is untouched", str(cm.exception))
        self.assertEqual(run_git(self.main, "rev-parse", "master"), before)
        self.assertFalse((self.main / "lane.txt").exists())
        self.assertTrue(self.lane.exists())
        self.assertIn("lane: work", run_git(self.lane, "log", "--oneline"))
        self.assertIn("alpha", self.entries())
        # The rebase itself is kept -- the lane's commit is on top of the
        # new master, which is where the session has to fix it.
        self.assertNotEqual(run_git(self.lane, "rev-parse", "HEAD"), sha)

    def test_conflict_aborts_the_rebase_and_reports(self):
        self.commit(self.lane, "a.txt", "lane version\n", "lane: edit a")
        self.commit(self.main, "a.txt", "master version\n", "main: edit a")
        before = run_git(self.main, "rev-parse", "master")
        lane_head = run_git(self.lane, "rev-parse", "HEAD")

        with self.assertRaises(lanes.LaneError) as cm:
            self.land("alpha")

        self.assertIn("does not rebase cleanly", str(cm.exception))
        self.assertIn("master is untouched", str(cm.exception))
        self.assertEqual(run_git(self.main, "rev-parse", "master"), before)
        self.assertEqual((self.main / "a.txt").read_text(), "master version\n")
        # Lane exactly as it was: same head, no rebase left in progress.
        self.assertEqual(run_git(self.lane, "rev-parse", "HEAD"), lane_head)
        self.assertEqual(run_git(self.lane, "status", "--porcelain"), "")
        gitdir = Path(run_git(self.lane, "rev-parse", "--absolute-git-dir"))
        self.assertFalse((gitdir / "rebase-merge").exists())
        self.assertFalse((gitdir / "rebase-apply").exists())
        self.assertIn("alpha", self.entries())
        self.assertEqual(self.checks_calls, [])


class TestLandRefusals(LaneCase):
    def test_uncommitted_lane_work_is_never_stashed_or_dropped(self):
        entry = self.start("alpha")
        lane = Path(entry["path"])
        (lane / "wip.txt").write_text("half a thought\n")

        with self.assertRaises(lanes.LaneError) as cm:
            self.land("alpha")

        self.assertIn("uncommitted changes", str(cm.exception))
        self.assertIn("never stashes", str(cm.exception))
        self.assertEqual((lane / "wip.txt").read_text(), "half a thought\n")
        self.assertEqual(run_git(lane, "stash", "list"), "")
        self.assertIn("alpha", self.entries())

    def test_main_checkout_on_another_branch_is_refused(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        run_git(self.main, "checkout", "-q", "-b", "sidequest")
        with self.assertRaises(lanes.LaneError) as cm:
            self.land("alpha")
        self.assertIn("not master", str(cm.exception))
        self.assertIn("alpha", self.entries())

    def test_unknown_slug_is_refused(self):
        with self.assertRaises(lanes.LaneError) as cm:
            self.land("nosuch")
        self.assertIn("no lane named", str(cm.exception))

    def test_missing_worktree_stops_without_touching_the_branch(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        before = run_git(self.main, "rev-parse", "master")
        subprocess.run(["rm", "-rf", entry["path"]], check=True)
        with self.assertRaises(lanes.LaneError) as cm:
            self.land("alpha")
        self.assertIn("worktree", str(cm.exception))
        self.assertEqual(run_git(self.main, "rev-parse", "master"), before)
        self.assertNotEqual(
            run_git(self.main, "branch", "--list", "lane/alpha"), "")


class TestDirtyMainIsJournalled(LaneCase):
    def test_operator_edits_become_a_commit_under_the_lane(self):
        entry = self.start("alpha")
        lane = Path(entry["path"])
        self.commit(lane, "lane.txt", "work\n", "lane: work")
        (self.main / "a.txt").write_text("operator edited this\n")
        (self.main / "notes.txt").write_text("and left this\n")

        report = self.land("alpha")

        self.assertTrue(report["landed"])
        self.assertIsNotNone(report["journal"])
        self.assertEqual(sorted(report["journal"]["files"]),
                         ["a.txt", "notes.txt"])
        log = run_git(self.main, "log", "--format=%s", "-3").splitlines()
        self.assertEqual(log[0], "lane: work")
        self.assertEqual(log[1], lanes.JOURNAL_MESSAGE)
        self.assertEqual((self.main / "a.txt").read_text(),
                         "operator edited this\n")
        self.assertEqual((self.main / "notes.txt").read_text(),
                         "and left this\n")
        self.assertEqual(run_git(self.main, "stash", "list"), "",
                         "operator edits were stashed instead of journalled")
        self.assertEqual(run_git(self.main, "status", "--porcelain"), "")

    def test_clean_main_gets_no_journal_commit(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        report = self.land("alpha")
        self.assertIsNone(report["journal"])
        self.assertNotIn(lanes.JOURNAL_MESSAGE,
                         run_git(self.main, "log", "--format=%s"))

    def test_a_moved_fork_pin_stops_the_land_and_is_never_journalled(self):
        # Advancing the submodule pin is an order's decision. A land that
        # committed it because a worktree was dirty would move the whole
        # runtime under everyone else.
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        second = self.commit(self.fork, "kernel.c", "int two(void);\n",
                             "fork: second")
        run_git(self.main / "fork", "fetch", "-q", "origin")
        run_git(self.main / "fork", "checkout", "-q", second)
        before = run_git(self.main, "rev-parse", "master")

        with self.assertRaises(lanes.LaneError) as cm:
            self.land("alpha")

        self.assertIn("fork pin", str(cm.exception))
        self.assertEqual(run_git(self.main, "rev-parse", "master"), before)
        self.assertNotIn(lanes.JOURNAL_MESSAGE,
                         run_git(self.main, "log", "--format=%s"))
        self.assertIn("alpha", self.entries())

    def test_a_moved_pin_beside_operator_edits_still_stops(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        second = self.commit(self.fork, "kernel.c", "int two(void);\n",
                             "fork: second")
        run_git(self.main / "fork", "fetch", "-q", "origin")
        run_git(self.main / "fork", "checkout", "-q", second)
        (self.main / "a.txt").write_text("operator edited this\n")

        with self.assertRaises(lanes.LaneError):
            self.land("alpha")

        # The journal commit is allowed to have happened (it is step one,
        # by design) but it must not carry the submodule.
        head_files = run_git(self.main, "show", "--name-only",
                             "--format=", "HEAD").split()
        self.assertNotIn("fork", head_files)
        self.assertNotIn("lane.txt", head_files)
        self.assertIn("alpha", self.entries())


# --- list and sweep --------------------------------------------------------

class TestListAndSweep(LaneCase):
    def test_a_lane_with_no_live_session_is_flagged(self):
        self.start("alpha")
        [row] = lanes.lane_list(self.ledger, proc_root=str(self.proc_root))
        self.assertEqual(row["sessions"], [])
        self.assertIn("no live session", row["flags"])

    def test_a_live_session_is_seen_and_not_flagged(self):
        entry = self.start("alpha")
        self.fake_session(4242, Path(entry["path"]) / "modelctl")
        [row] = lanes.lane_list(self.ledger, proc_root=str(self.proc_root))
        self.assertEqual(row["sessions"], [4242])
        self.assertNotIn("no live session", row["flags"])

    def test_a_lane_older_than_a_day_is_flagged_in_every_listing(self):
        self.start("alpha", clock=lambda: time.time() - 25 * 3600)
        self.fake_session(4242, self.root / "alpha")
        [row] = lanes.lane_list(self.ledger, proc_root=str(self.proc_root))
        self.assertIn("older than 24 h", row["flags"])
        self.assertGreater(row["age_seconds"], 24 * 3600)

    def test_unlanded_commits_are_counted(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        [row] = lanes.lane_list(self.ledger, proc_root=str(self.proc_root))
        self.assertEqual(row["unlanded_commits"], 1)

    def test_sweep_refuses_to_delete_work_that_was_never_landed(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        with self.assertRaises(lanes.LaneError) as cm:
            lanes.delete("alpha", ledger=self.ledger,
                         proc_root=str(self.proc_root))
        self.assertIn("not on master", str(cm.exception))
        self.assertIn("alpha", self.entries())
        self.assertTrue(Path(entry["path"]).exists())

    def test_forced_delete_removes_worktree_branch_and_ports(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "throwaway\n", "lane: x")
        lanes.delete("alpha", ledger=self.ledger, force=True,
                     proc_root=str(self.proc_root))
        self.assertFalse(Path(entry["path"]).exists())
        self.assertEqual(run_git(self.main, "branch", "--list", "lane/alpha"),
                         "")
        self.assertEqual(self.entries(), {})
        self.assertEqual(self.start("bravo")["port_base"], 9500)

    def test_delete_refuses_a_lane_with_a_live_session(self):
        entry = self.start("alpha")
        self.fake_session(4242, Path(entry["path"]))
        with self.assertRaises(lanes.LaneError) as cm:
            lanes.delete("alpha", ledger=self.ledger,
                         proc_root=str(self.proc_root))
        self.assertIn("live session", str(cm.exception))
        self.assertTrue(Path(entry["path"]).exists())

    def test_delete_of_an_unknown_slug_is_refused(self):
        with self.assertRaises(lanes.LaneError):
            lanes.delete("nosuch", ledger=self.ledger,
                         proc_root=str(self.proc_root))


# --- build scratch ---------------------------------------------------------

class TestScratchLocation(LaneCase):
    """Scratch is on the pool, not in tmpfs.

    /tmp on this machine is RAM. A default that put every lane's cmake
    output there took ~850 MB per lane away from the models, and nobody
    noticed because nothing ever reported it.
    """

    def test_the_default_is_not_tmpfs(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            base = lanes.scratch_base()
        self.assertNotEqual(base, lanes.LEGACY_SCRATCH_BASE)
        self.assertFalse(str(base).startswith("/tmp"), base)

    def test_the_env_override_moves_every_lane_at_once(self):
        # The same variable ci/checks.sh reads, so one export moves every
        # build tree the project makes.
        with mock.patch.dict(os.environ,
                             {"MODELCTL_CI_SCRATCH_ROOT": "/mnt/fast"},
                             clear=True):
            self.assertEqual(lanes.scratch_root("alpha"),
                             Path("/mnt/fast/modelctl-lane-alpha"))

    def test_the_lane_specific_override_wins_over_the_shared_one(self):
        with mock.patch.dict(os.environ,
                             {"MODELCTL_CI_SCRATCH_ROOT": "/mnt/fast",
                              "MODELCTL_LANES_SCRATCH": "/mnt/lanes"},
                             clear=True):
            self.assertEqual(lanes.scratch_root("alpha"),
                             Path("/mnt/lanes/modelctl-lane-alpha"))

    def test_tmp_is_searched_only_when_nothing_is_redirected(self):
        # The legacy scan is how the megabytes every earlier lane left in
        # tmpfs get collected; a caller that has redirected scratch has
        # said the machine's /tmp is not its to delete from.
        self.assertNotIn(lanes.LEGACY_SCRATCH_BASE, lanes.scratch_search_dirs())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIn(lanes.LEGACY_SCRATCH_BASE,
                          lanes.scratch_search_dirs())

    def test_both_configured_bases_are_searched(self):
        dirs = lanes.scratch_search_dirs()
        self.assertIn(self.scratch, dirs)
        self.assertIn(self.ci_scratch, dirs)


class TestScratchNamesComeFromChecks(LaneCase):
    """The list of build directories is read out of ci/checks.sh.

    Hardcoding it here is how teardown fell behind CI in the first
    place: the script grew a console build and a sanitizer build, and
    nothing that cleans up was told.
    """

    def test_a_new_ci_build_dir_is_picked_up_without_editing_this_module(self):
        checks = self.main / "ci" / "checks.sh"
        checks.write_text(CHECKS_STUB + '\nB="${MODELCTL_CI_WIDGET_DIR:'
                                        '-$CI_SCRATCH_ROOT/ci-widget}"\n')
        self.assertIn("ci-widget", lanes.ci_scratch_names(self.main))
        self.assertIn("ci-widget-lane-alpha",
                      lanes.scratch_dir_names("alpha", self.main))

    def test_an_unreadable_script_falls_back_rather_than_sweeping_nothing(self):
        self.assertEqual(lanes.ci_scratch_names(self.tmp / "nowhere"),
                         lanes.CI_SCRATCH_FALLBACK)

    def test_a_default_that_is_itself_a_variable_names_nothing(self):
        checks = self.main / "ci" / "checks.sh"
        checks.write_text('X="${MODELCTL_CI_ODD_DIR:-$SOMEWHERE}"\n')
        self.assertEqual(lanes.ci_scratch_names(self.main),
                         lanes.CI_SCRATCH_FALLBACK)


class TestScratchTeardown(LaneCase):
    """Both exits from a lane take its build trees with them.

    Before this, `land` freed the worktree, the branch and the port
    block and left the build trees: on 2026-08-02 four lanes' worth
    (3.4 GB) sat in tmpfs, one of them belonging to a lane that had been
    swept hours earlier.
    """

    def scratch_for(self, slug):
        """One of each naming convention: `lane env`'s, and the one a
        session uses when it exports MODELCTL_CI_*_DIR by hand."""
        return [self.make_scratch(self.scratch, f"modelctl-lane-{slug}"),
                self.make_scratch(self.ci_scratch,
                                  f"ci-build-cpu-lane-{slug}"),
                self.make_scratch(self.ci_scratch,
                                  f"ci-console-dist-lane-{slug}")]

    def test_land_removes_every_scratch_directory_of_the_lane(self):
        self.start("alpha")
        made = self.scratch_for("alpha")
        report = self.land("alpha")
        self.assertTrue(report["landed"])
        for path in made:
            self.assertFalse(path.exists(), path)
        self.assertEqual({Path(d["path"]) for d in report["scratch_removed"]},
                         set(made))
        self.assertGreater(report["scratch_bytes"], 0)

    def test_another_lanes_scratch_is_untouched(self):
        self.start("alpha")
        self.start("bravo")
        mine, theirs = self.scratch_for("alpha"), self.scratch_for("bravo")
        self.land("alpha")
        self.assertFalse(any(p.exists() for p in mine))
        self.assertTrue(all(p.exists() for p in theirs))

    def test_delete_removes_the_scratch_too(self):
        self.start("alpha")
        made = self.scratch_for("alpha")
        state = lanes.delete("alpha", ledger=self.ledger, force=True,
                             proc_root=str(self.proc_root))
        for path in made:
            self.assertFalse(path.exists(), path)
        self.assertEqual(len(state["scratch_removed"]), len(made))

    def test_a_failed_land_keeps_the_scratch_and_its_checks_log(self):
        # The log of the run that failed is the one anybody wants, and
        # the lane is still there to be fixed and landed again.
        entry = self.start("alpha")
        self.commit(self.main, "b.txt", "moved\n", "main: moved")
        self.commit(Path(entry["path"]), "c.txt", "lane\n", "lane: work")
        made = self.scratch_for("alpha")

        def failing_checks_with_a_log(lane_path, env, log_path=None):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("fixture checks: FAIL cache tests\n")
            return 1, "fixture checks: FAIL cache tests\n"

        with self.assertRaises(lanes.LaneError) as cm:
            self.land("alpha", checks=failing_checks_with_a_log)
        self.assertTrue(all(p.exists() for p in made))
        log = lanes.scratch_root("alpha") / "land-checks.log"
        self.assertTrue(log.exists())
        self.assertIn(str(log), str(cm.exception))

    def test_a_landed_report_stops_naming_a_log_it_deleted(self):
        entry = self.start("alpha")
        self.commit(self.main, "b.txt", "moved\n", "main: moved")
        self.commit(Path(entry["path"]), "c.txt", "lane\n", "lane: work")
        report = self.land("alpha")
        self.assertEqual(report["checks"], "passed")
        self.assertIsNone(report["checks_log"])

    def test_nothing_to_remove_is_not_an_error(self):
        self.start("alpha")
        report = self.land("alpha")
        self.assertEqual(report["scratch_removed"], [])
        self.assertEqual(report["scratch_bytes"], 0)


class TestOrphanSweep(LaneCase):
    """`sweep --orphans`: scratch whose lane no longer exists.

    The ledger is the authority. `start` writes it and both exits remove
    it, so a lane-suffixed directory whose slug is not in it belongs to
    nobody -- which described 3.4 GB of tmpfs on the morning this was
    written.
    """

    def orphans(self, **kw):
        return lanes.find_orphan_scratch(ledger=self.ledger, main=self.main,
                                         **kw)

    def sweep(self, **kw):
        return lanes.sweep_scratch_orphans(ledger=self.ledger, main=self.main,
                                           **kw)

    def test_scratch_of_a_lane_that_is_gone_is_found_named_and_deleted(self):
        dead = self.make_scratch(self.scratch, "modelctl-lane-gone")
        found = self.orphans()
        self.assertEqual([Path(f["path"]) for f in found], [dead])
        self.assertEqual(found[0]["slug"], "gone")
        self.assertGreater(found[0]["bytes"], 0)

        removed = self.sweep()
        self.assertFalse(dead.exists())
        self.assertEqual([Path(r["path"]) for r in removed], [dead])
        self.assertGreater(removed[0]["bytes"], 0)

    def test_a_live_lanes_scratch_is_spared(self):
        self.start("alive")
        live = self.make_scratch(self.scratch, "modelctl-lane-alive")
        live_ci = self.make_scratch(self.ci_scratch, "ci-build-cpu-lane-alive")
        self.sweep()
        self.assertTrue(live.exists())
        self.assertTrue(live_ci.exists())

    def test_keep_spares_a_slug_the_ledger_cannot_see(self):
        # A session that exported the build dirs by hand and never took
        # a ledger entry looks exactly like an orphan while it compiles.
        busy = self.make_scratch(self.ci_scratch, "ci-build-cpu-lane-busy")
        other = self.make_scratch(self.ci_scratch, "ci-build-cpu-lane-old")
        removed = self.sweep(keep=["busy"])
        self.assertTrue(busy.exists())
        self.assertFalse(other.exists())
        self.assertEqual([r["slug"] for r in removed], ["old"])

    def test_directories_that_are_not_lane_scratch_are_never_touched(self):
        keepers = [self.make_scratch(self.scratch, "ci-build-cpu"),
                   self.make_scratch(self.scratch, "modelctl-lane-"),
                   self.make_scratch(self.scratch, "modelctl-lane-Bad_Slug"),
                   self.make_scratch(self.scratch, "somebody-elses-work"),
                   self.make_scratch(self.ci_scratch, "ci-build-cpu-lane")]
        self.assertEqual(self.orphans(), [])
        self.sweep()
        self.assertTrue(all(p.exists() for p in keepers))

    def test_a_symlinked_directory_is_not_followed(self):
        # rmtree on a symlinked scratch dir would delete its target's
        # contents, which is not a directory this sweep ever chose.
        target = self.tmp / "elsewhere"
        (target / "keep").mkdir(parents=True)
        os.symlink(str(target), str(self.scratch / "modelctl-lane-link"))
        self.sweep()
        self.assertTrue((target / "keep").exists())

    def test_every_ci_naming_convention_is_collected(self):
        made = [self.make_scratch(self.scratch, "modelctl-lane-gone"),
                self.make_scratch(self.ci_scratch, "ci-build-cpu-lane-gone"),
                self.make_scratch(self.ci_scratch, "ci-build-san-lane-gone"),
                self.make_scratch(self.ci_scratch,
                                  "ci-console-dist-lane-gone")]
        removed = self.sweep()
        self.assertEqual(len(removed), len(made))
        self.assertFalse(any(p.exists() for p in made))

    def test_a_missing_search_directory_is_not_an_error(self):
        shutil.rmtree(self.ci_scratch)
        self.assertEqual(self.sweep(), [])


class TestSessionDetection(LaneCase):
    def test_only_processes_inside_the_lane_count(self):
        self.fake_session(11, self.tmp / "elsewhere")
        self.fake_session(12, self.root / "alpha" / "modelctl")
        self.fake_session(13, self.root / "alpha-other")
        (self.root / "alpha").mkdir(parents=True)
        found = lanes.sessions_for(self.root / "alpha",
                                   proc_root=str(self.proc_root))
        self.assertEqual(found, [12],
                         "a sibling path prefix was mistaken for the lane")

    def test_unreadable_proc_entries_do_not_crash_the_listing(self):
        (self.proc_root / "not-a-pid").mkdir()
        (self.proc_root / "99").mkdir()      # no cwd link at all
        self.assertEqual(
            lanes.sessions_for(self.tmp, proc_root=str(self.proc_root)), [])


# --- locks -----------------------------------------------------------------

class TestGpuLock(LaneCase):
    def test_the_lock_excludes_a_second_holder(self):
        with lanes.gpu_lock():
            with self.assertRaises(lanes.LockBusy) as cm:
                with lanes.gpu_lock(timeout=0.2, poll=0.05):
                    self.fail("two runs held the GPU lock at once")
        self.assertIn("held by another run", str(cm.exception))

    def test_the_lock_is_released_at_the_end_of_the_block(self):
        with lanes.gpu_lock():
            pass
        with lanes.gpu_lock(timeout=0.2, poll=0.05):
            pass

    def test_the_lock_is_released_when_the_body_raises(self):
        with self.assertRaises(ZeroDivisionError):
            with lanes.gpu_lock():
                1 / 0
        with lanes.gpu_lock(timeout=0.2, poll=0.05):
            pass

    def test_it_excludes_other_processes_too(self):
        # In-process exclusion could be a thread artefact; the case that
        # matters is a night job and a lane bench in different processes.
        rc = lanes.run_with_gpu_lock(
            [sys.executable, "-c", _TRY_LOCK, str(lanes.gpu_lock_path())])
        self.assertEqual(rc, 3, "the command could take the lock while "
                                "run_with_gpu_lock was holding it")

    def test_the_holder_is_recorded_for_the_report(self):
        with lanes.gpu_lock(note="paired sweep"):
            holder = json.loads(lanes.gpu_lock_path().read_text())
        self.assertEqual(holder["pid"], os.getpid())
        self.assertEqual(holder["what"], "paired sweep")

    def test_run_with_gpu_lock_returns_the_command_status(self):
        rc = lanes.run_with_gpu_lock([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(rc, 7)

    def test_gpu_lock_needs_a_command(self):
        with self.assertRaises(lanes.LaneError):
            lanes.run_with_gpu_lock([])


_TRY_LOCK = """
import fcntl, sys
fh = open(sys.argv[1], 'a+')
try:
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    raise SystemExit(3)
raise SystemExit(0)
"""


class TestLandLock(LaneCase):
    def test_a_second_land_waits_rather_than_racing_master(self):
        entry = self.start("alpha")
        self.commit(Path(entry["path"]), "lane.txt", "work\n", "lane: work")
        with lanes._exclusive(lanes.land_lock_path(), note="another land"):
            with self.assertRaises(lanes.LockBusy):
                self.land("alpha", timeout=0.2)
        # Nothing was landed while it waited.
        self.assertIn("alpha", self.entries())
        self.assertTrue(Path(entry["path"]).exists())


# --- the build environment -------------------------------------------------

class TestBuildEnv(LaneCase):
    def test_ccache_is_made_path_independent(self):
        env = lanes.build_env("/somewhere/lane", base={})
        self.assertEqual(env["CCACHE_BASEDIR"], "/somewhere/lane")
        self.assertEqual(env["CCACHE_NOHASHDIR"], "1")
        self.assertTrue(env["CCACHE_DIR"].endswith("/.cache/ccache"))

    def test_an_existing_ccache_dir_is_left_alone(self):
        env = lanes.build_env("/somewhere/lane",
                              base={"CCACHE_DIR": "/mnt/big/ccache"})
        self.assertEqual(env["CCACHE_DIR"], "/mnt/big/ccache")

    def test_no_sloppiness_is_relaxed(self):
        # Path independence is the only thing being bought here; a
        # sloppiness setting would buy hits by hashing less.
        env = lanes.build_env("/somewhere/lane", base={})
        self.assertNotIn("CCACHE_SLOPPINESS", env)

    def test_two_lanes_get_separate_build_and_console_scratch(self):
        a = lanes.lane_env(self.start("alpha"), base={})
        b = lanes.lane_env(self.start("bravo"), base={})
        for key in ("MODELCTL_CI_BUILD_DIR", "MODELCTL_CI_SAN_BUILD_DIR",
                    "MODELCTL_CI_CONSOLE_DIR"):
            self.assertNotEqual(a[key], b[key], key)
        self.assertEqual(a["MODELCTL_WEB_BIND"], "127.0.0.1:9500")
        self.assertEqual(b["MODELCTL_WEB_BIND"], "127.0.0.1:9510")

    def test_env_exports_are_shell_lines(self):
        lines = lanes.env_exports(self.start("alpha"), base={})
        self.assertTrue(all(line.startswith("export ") for line in lines))
        self.assertIn("export MODELCTL_LANE=alpha", lines)


class TestDefaultChecks(LaneCase):
    def test_it_runs_the_lane_checks_and_captures_the_log(self):
        entry = self.start("alpha")
        log = self.tmp / "checks.log"
        rc, output = lanes.default_checks(
            entry["path"], lanes.lane_env(entry, base=dict(os.environ)), log)
        self.assertEqual(rc, 0)
        self.assertIn("checks ran in", output)
        self.assertIn(f"CCACHE_BASEDIR={entry['path']}", log.read_text())

    def test_a_failing_run_reports_its_status(self):
        entry = self.start("alpha")
        env = lanes.lane_env(entry, base=dict(os.environ))
        env["FIXTURE_CHECKS_EXIT"] = "1"
        rc, _ = lanes.default_checks(entry["path"], env)
        self.assertEqual(rc, 1)

    def test_a_missing_checks_script_is_a_stop_not_a_pass(self):
        entry = self.start("alpha")
        (Path(entry["path"]) / "ci" / "checks.sh").unlink()
        with self.assertRaises(lanes.LaneError):
            lanes.default_checks(entry["path"], dict(os.environ))


# --- resolution ------------------------------------------------------------

class TestMainCheckoutResolution(LaneCase):
    def test_a_lane_resolves_the_main_checkout_not_itself(self):
        # The failure this prevents: a session inside a lane running
        # `lane land` and landing the lane into the lane.
        entry = self.start("alpha")
        self.assertEqual(lanes.main_checkout(entry["path"]),
                         self.main.resolve())

    def test_a_directory_outside_a_checkout_is_refused(self):
        outside = self.tmp / "nowhere"
        outside.mkdir()
        with self.assertRaises(lanes.LaneError):
            lanes.main_checkout(outside)


class TestLedger(LaneCase):
    def test_a_missing_ledger_reads_as_empty(self):
        data = lanes.read_ledger(self.tmp / "nothing.json")
        self.assertEqual(data, {"version": lanes.LEDGER_VERSION, "lanes": {}})

    def test_a_corrupt_ledger_is_a_loud_refusal(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json")
        with self.assertRaises(lanes.LaneError) as cm:
            lanes.read_ledger(path)
        self.assertIn("unreadable", str(cm.exception))

    def test_the_ledger_lives_outside_the_repository(self):
        self.start("alpha")
        self.assertEqual(run_git(self.main, "status", "--porcelain"), "")
        self.assertFalse((self.main / "lanes.json").exists())


class TestLaneCli(unittest.TestCase):
    """The command surface, and the one thing about it that is not
    cosmetic: a stop must not exit 0."""

    def parse(self, *argv):
        import modelctl
        return modelctl.build_arg_parser().parse_args(["lane", *argv])

    def test_a_stop_exits_non_zero(self):
        # A land that stopped but exited 0 tells the session its work is
        # on master when it is still in the lane -- and the next thing
        # that session does is delete the lane.
        import modelctl
        args = self.parse("land", "alpha")
        with mock.patch.object(lanes, "land",
                               side_effect=lanes.LaneError("conflict")):
            with self.assertRaises(SystemExit) as cm:
                modelctl._cmd_lane_cli(args)
        self.assertEqual(cm.exception.code, 1)

    def test_a_successful_land_exits_zero(self):
        import modelctl
        report = {"slug": "alpha", "landed": True, "journal": None,
                  "rebase": "no-op", "checks": "not run", "checks_log": None,
                  "master_before": "a" * 40, "master_after": "b" * 40,
                  "ports_freed": [9500, 9509]}
        with mock.patch.object(lanes, "land", return_value=report):
            with self.assertRaises(SystemExit) as cm:
                modelctl._cmd_lane_cli(self.parse("land", "alpha"))
        self.assertEqual(cm.exception.code, 0)

    def test_a_land_reports_the_scratch_it_reclaimed(self):
        import modelctl
        report = {"slug": "alpha", "landed": True, "journal": None,
                  "rebase": "no-op", "checks": "not run", "checks_log": None,
                  "master_before": "a" * 40, "master_after": "b" * 40,
                  "ports_freed": [9500, 9509],
                  "scratch_removed": [{"path": "/s/modelctl-lane-alpha",
                                       "slug": "alpha", "bytes": 850_000_000}],
                  "scratch_bytes": 850_000_000}
        buf = io.StringIO()
        with mock.patch.object(lanes, "land", return_value=report), \
             contextlib.redirect_stdout(buf):
            modelctl.cmd_lane(self.parse("land", "alpha"))
        self.assertIn("/s/modelctl-lane-alpha", buf.getvalue())

    def test_sweep_orphans_names_each_directory_it_removed(self):
        import modelctl
        removed = [{"path": "/tmp/modelctl-lane-gone", "slug": "gone",
                    "bytes": 842_000_000}]
        buf = io.StringIO()
        with mock.patch.object(lanes, "lane_list", return_value=[]), \
             mock.patch.object(lanes, "sweep_scratch_orphans",
                               return_value=removed) as sweep, \
             contextlib.redirect_stdout(buf):
            rc = modelctl.cmd_lane(self.parse("sweep", "--orphans"))
        self.assertEqual(rc, 0)
        self.assertEqual(sweep.call_args.kwargs["keep"], [])
        out = buf.getvalue()
        self.assertIn("/tmp/modelctl-lane-gone", out)
        self.assertIn("gone", out)

    def test_sweep_without_orphans_deletes_no_scratch(self):
        # `sweep` on its own still only reports: the flag is the consent.
        import modelctl
        with mock.patch.object(lanes, "lane_list", return_value=[]), \
             mock.patch.object(lanes, "sweep_scratch_orphans") as sweep, \
             contextlib.redirect_stdout(io.StringIO()):
            modelctl.cmd_lane(self.parse("sweep"))
        sweep.assert_not_called()

    def test_keep_reaches_the_sweep(self):
        import modelctl
        with mock.patch.object(lanes, "lane_list", return_value=[]), \
             mock.patch.object(lanes, "sweep_scratch_orphans",
                               return_value=[]) as sweep, \
             contextlib.redirect_stdout(io.StringIO()):
            modelctl.cmd_lane(self.parse("sweep", "--orphans",
                                         "--keep", "console-fleet"))
        self.assertEqual(sweep.call_args.kwargs["keep"], ["console-fleet"])

    def test_gpu_lock_passes_the_command_after_the_separator(self):
        import modelctl
        args = self.parse("gpu-lock", "--", "echo", "hi")
        with mock.patch.object(lanes, "run_with_gpu_lock",
                               return_value=0) as run:
            modelctl.cmd_lane(args)
        self.assertEqual(run.call_args.args[0], ["echo", "hi"])

    def test_gpu_lock_returns_the_command_status(self):
        import modelctl
        with mock.patch.object(lanes, "run_with_gpu_lock", return_value=7):
            self.assertEqual(modelctl.cmd_lane(self.parse("gpu-lock", "--",
                                                          "false")), 7)

    def test_sweep_deletes_nothing_unless_a_lane_is_named(self):
        import modelctl
        with mock.patch.object(lanes, "lane_list", return_value=[]), \
             mock.patch.object(lanes, "delete") as delete:
            self.assertEqual(modelctl.cmd_lane(self.parse("sweep")), 0)
        delete.assert_not_called()

    def test_sweep_delete_names_the_lane_and_honours_force(self):
        import modelctl
        with mock.patch.object(lanes, "lane_list", return_value=[]), \
             mock.patch.object(lanes, "delete") as delete:
            modelctl.cmd_lane(self.parse("sweep", "--delete", "alpha",
                                         "--force"))
        delete.assert_called_once_with("alpha", force=True)

    def test_a_subcommand_is_required(self):
        with self.assertRaises(SystemExit):
            self.parse()


if __name__ == "__main__":
    unittest.main()
