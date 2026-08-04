"""Byte-accurate progress for a running `modelctl pull`.

The console's pull bar sat at 0 for entire multi-hour downloads. The old
poller read progress from two things that never arrive during a
download:

* the child's ``downloading X ->`` line, which block-buffers in a plain
  CPython child writing to a pipe, so it lands after the transfer it
  announced; and
* ``stat()`` on the destination path, which ``hf_hub_download`` does not
  create until the file is finished -- with ``local_dir=`` it unlinks any
  outdated destination first and streams into a spool file, then moves it
  into place at the end.

So the bytes are measured where they actually are: the spool. On
huggingface_hub 1.24.0 that is

    <models>/<repo_id>/.cache/huggingface/download/<subdir>/
        <short-hash-of-filename>.<etag>[.<uuid8>].incomplete

-- the file's own name never appears in it, which is why a
filename-shaped glob finds nothing. This module globs by extension
instead, so an upgrade that changes the hash scheme cannot silently stop
the bar.
"""
import threading
from pathlib import Path

import modelctl

POLL_INTERVAL_SECONDS = 3.0

# The child's exit code is the only thing that may declare a pull
# finished. A bar that reaches 100% while the process is still running is
# the same lie the byte counting exists to stop.
MAX_REPORTED_FRACTION = 0.99

# Backstop on the wait for an in-flight tick at shutdown. A tick is a
# handful of stat()s and one row update; anything near this means the
# filesystem is wedged, and the pull must not wait on its own reporter.
STOP_JOIN_TIMEOUT_SECONDS = 5.0

SPOOL_PARTS = (".cache", "huggingface", "download")


def _spool_dirs(repo_dir: Path, needed_files) -> set:
    """Spool directories for this pull's files.

    hf mirrors the repo-relative path under the download folder, so a
    quant shipped inside ``BF16/`` spools into ``download/BF16/``. Only
    the directories this pull's own files use are searched: a repo dir
    can hold an earlier quant's leftovers, and counting those would pin
    the bar near 100% before a byte of the new download landed.
    """
    root = repo_dir.joinpath(*SPOOL_PARTS)
    return {(root / f).parent for f in needed_files}


def bytes_on_disk(repo_id: str, needed_files) -> int:
    """Bytes of THIS pull that are already on disk.

    Counts each requested file that has finished (a file already present
    from an earlier run counts too -- it genuinely does not need
    fetching) plus whatever is currently in the matching spool
    directories.

    A file vanishing between the glob and the stat is normal: that is the
    spool being moved into place, and the next tick counts it at its
    destination. Such a file is skipped rather than raising, so an
    unreadable path can never take down the download it is reporting on.
    """
    repo_dir = Path(modelctl.DEFAULT_MODELS_DIR) / repo_id
    total = 0
    for f in needed_files:
        try:
            total += (repo_dir / f).stat().st_size
        except OSError:
            continue
    for spool_dir in _spool_dirs(repo_dir, needed_files):
        try:
            spools = list(spool_dir.glob("*.incomplete"))
        except OSError:
            continue
        for spool in spools:
            try:
                total += spool.stat().st_size
            except OSError:
                continue
    return total


class PullProgressPoller:
    """Reports a pull's byte progress on a timer until told to stop.

    Structured like ``modelctl_nodestats.NodeStatsPoller``: an
    ``Event.wait`` pause a stop can interrupt, and a ``poll_once`` seam so
    the measurement is testable without a thread or a real download.

    Caveats, all bounded by the 0.99 cap and none of them silent -- the
    job log's per-file lines remain the exact record:

    * mmproj/MTP companion files are outside ``needed_bytes``, so their
      bytes are not counted;
    * a quant sitting at the repo root shares its spool directory with
      those companions, whose in-flight bytes ARE counted; and
    * a stale spool left by an earlier failed pull of a different quant
      in that same directory is counted too. ``_cleanup_partial_downloads``
      is meant to remove those and currently cannot -- it globs the
      pre-1.x spool naming, which no longer matches anything.
    """

    def __init__(self, repo_id, needed_files, needed_bytes, on_progress,
                 should_continue=None, label_fn=None, on_error=None,
                 interval=None):
        self._repo_id = repo_id
        self._needed_files = tuple(needed_files or ())
        self._needed_bytes = int(needed_bytes or 0)
        self._on_progress = on_progress
        self._should_continue = should_continue or (lambda: True)
        self._label_fn = label_fn or (lambda: None)
        self._on_error = on_error or (lambda msg: None)
        # Resolved here rather than as a default argument so the module
        # constant can be lowered for tests.
        self._interval = (POLL_INTERVAL_SECONDS if interval is None
                          else interval)
        self._stop = threading.Event()
        self._thread = None

    @property
    def can_measure(self) -> bool:
        """A fraction needs both a denominator and something to measure.

        Repo metadata can be unreachable when the pull is submitted; the
        honest response is to report nothing rather than invent a number.
        """
        return bool(self._needed_bytes and self._needed_files)

    def start(self):
        """Start the timer thread once. A no-op when nothing is measurable."""
        if not self.can_measure or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop,
                                        name="pull-progress-poller",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def shutdown(self, timeout=STOP_JOIN_TIMEOUT_SECONDS):
        """Stop, and wait for a tick that is already in flight.

        The waiting is the point. Setting the flag does not recall a tick
        that is mid-write, and that write would land after the job runner
        stamps progress=1.0 -- rewinding a finished pull to 99%. The
        timeout is a backstop so a wedged filesystem can never hold the
        download's own thread hostage.
        """
        self.stop()
        self.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        while not self._stop.is_set():
            if not self._should_continue():
                return
            try:
                self.poll_once()
            except Exception as e:
                # Progress is cosmetic; the download must outlive a
                # failure to describe it. Say so once, then stand down --
                # a tick that failed will fail again every interval.
                self._on_error(str(e) or e.__class__.__name__)
                return
            if self._stop.wait(self._interval):
                return

    def poll_once(self):
        """Measure once and report. Used by the thread and by tests."""
        if not self.can_measure:
            return 0
        have = bytes_on_disk(self._repo_id, self._needed_files)
        fraction = min(MAX_REPORTED_FRACTION, have / self._needed_bytes)
        sizes = (f"{modelctl._format_size(have)} / "
                 f"{modelctl._format_size(self._needed_bytes)}")
        label = self._label_fn()
        self._on_progress(fraction, f"{label}: {sizes}" if label else sizes)
        return have
