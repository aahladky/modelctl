"""Lane-based job queue for modelctl-web.

Jobs are routed into named lanes so that downloads and benchmarks cannot block
urgent runtime actions like load/unload.  Each lane has its own worker thread
pool and FIFO queue.

Lane assignment:
  mutation  - profile save, config apply, sync (serialized)
  runtime   - load, unload, restart, eviction (2 workers)
  download  - HF downloads, verification, repair (2 workers)
  benchmark - smoke test, plan test, tuning (1 worker)

JobContext wraps every job function call, providing progress tracking,
cancellation checks, and subprocess registration for real SIGTERM/SIGKILL
cancellation via process groups.

Job records live in SQLite so they survive service restarts; a job left in
'running' state at startup is marked 'interrupted' (the worker is gone).
"""
import json
import os
import queue
import signal
import sqlite3
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path

STATE_DIR = Path(os.environ.get(
    "MODELCTL_STATE_DIR",
    Path.home() / ".local" / "share" / "modelctl"))
DB_PATH = STATE_DIR / "web_jobs.db"

_LANE_DEFAULTS = {
    "mutation": 1,
    "runtime": 2,
    "download": 2,
    "benchmark": 1,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    lane TEXT NOT NULL DEFAULT 'mutation',
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    result TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    revision INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    cancellable INTEGER NOT NULL DEFAULT 1,
    parent_id TEXT,
    created REAL NOT NULL,
    started REAL,
    finished REAL
);
"""

_MIGRATE_COLUMNS = {
    "lane": "TEXT NOT NULL DEFAULT 'mutation'",
    "revision": "INTEGER NOT NULL DEFAULT 0",
    "pid": "INTEGER",
    "cancellable": "INTEGER NOT NULL DEFAULT 1",
    "parent_id": "TEXT",
}


class JobStore:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(_SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
            for col, decl in _MIGRATE_COLUMNS.items():
                if col not in cols:
                    c.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")
            c.execute(
                "UPDATE jobs SET status='interrupted' WHERE status IN ('queued','running')")

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def create(self, job_type, title, payload=None, lane="mutation"):
        job_id = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs (id, type, title, lane, status, payload, created) "
                "VALUES (?,?,?,?,?,?,?)",
                (job_id, job_type, title, lane, "queued",
                 json.dumps(payload or {}), time.time()))
        return job_id

    def update(self, job_id, **fields):
        if not fields:
            return
        # Single statement: revision increments atomically in SQL, so
        # concurrent updaters can't read-then-write the same value (which
        # would silently drop an SSE notification).
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = []
        for k, v in fields.items():
            if k in ("payload", "result") and not isinstance(v, str):
                vals.append(json.dumps(v))
            else:
                vals.append(v)
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {cols}, revision=revision+1 WHERE id=?",
                      (*vals, job_id))

    def get(self, job_id):
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list(self, limit=50):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def append_result_line(self, job_id, line):
        # Atomic append in SQL -- the read-modify-write version dropped
        # lines whenever two writers raced.
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET result = substr(result || ?, -20000), "
                "revision=revision+1 WHERE id=?", (line + "\n", job_id))


class JobContext:
    """Per-job context passed to every job function.

    Provides logging, progress, cancellation checks, and subprocess
    registration for real process-group termination.
    """

    def __init__(self, store, job_id):
        self._store = store
        self._job_id = job_id
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._processes = []
        self._callbacks = []

    @property
    def store(self):
        return self._store

    @property
    def job_id(self):
        return self._job_id

    def log(self, line):
        self._store.append_result_line(self._job_id, line)

    def set_progress(self, value, detail=""):
        fields = {}
        if value is not None:
            fields["progress"] = max(0.0, min(1.0, float(value)))
        if detail:
            fields["detail"] = detail
        if fields:
            self._store.update(self._job_id, **fields)

    def is_cancelled(self):
        return self._cancel.is_set()

    def raise_if_cancelled(self):
        if self._cancel.is_set():
            raise CancelledError("job cancelled")

    def set_cancelled(self):
        self._cancel.set()

    def register_process(self, proc):
        with self._lock:
            self._processes.append(proc)

    def register_cancel_callback(self, fn):
        with self._lock:
            self._callbacks.append(fn)

    def cancel_all(self):
        """Terminate registered processes and run cleanup callbacks."""
        with self._lock:
            procs = list(self._processes)
            cbs = list(self._callbacks)
        for proc in procs:
            _terminate_process(proc)
        for fn in cbs:
            try:
                fn()
            except Exception:
                pass


class CancelledError(Exception):
    pass


def _terminate_process(proc, grace=5.0):
    """Send SIGTERM to process group, wait, then SIGKILL."""
    try:
        if hasattr(os, "killpg") and hasattr(proc, "pgid"):
            os.killpg(proc.pgid, signal.SIGTERM)
        elif proc.poll() is None:
            proc.terminate()
    except (ProcessLookupError, OSError):
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        if hasattr(os, "killpg") and hasattr(proc, "pgid"):
            os.killpg(proc.pgid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _make_job_event(job):
    """Convert a job dict to a JSON-serializable event dict."""
    return {
        "id": job["id"],
        "type": job["type"],
        "title": job["title"],
        "lane": job.get("lane", "mutation"),
        "status": job["status"],
        "progress": job["progress"],
        "detail": job.get("detail", ""),
        "error": job.get("error", ""),
        "revision": job.get("revision", 0),
        "pid": job.get("pid"),
        "created": job.get("created"),
        "started": job.get("started"),
        "finished": job.get("finished"),
    }


def sse_job_stream(store, job_id, timeout=1800):
    """Generator yielding SSE events when the job's revision changes.
    Heartbeat comments every ~15s so disconnected clients are detected on
    the next failed write instead of lingering for the full timeout."""
    last_rev = -1
    idle = 0
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            job = store.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                return
            rev = job.get("revision", 0)
            if rev != last_rev:
                last_rev = rev
                idle = 0
                yield f"data: {json.dumps(_make_job_event(job))}\n\n"
                if job["status"] not in ("queued", "running"):
                    return
            else:
                idle += 1
                if idle >= 15:
                    idle = 0
                    yield ": heartbeat\n\n"
            time.sleep(1)
    except (GeneratorExit, BrokenPipeError, ConnectionResetError):
        return


class JobLane:
    """A named lane with its own FIFO queue and worker pool."""

    def __init__(self, name, workers=1):
        self.name = name
        self._q = queue.Queue()
        self._threads = []
        for i in range(workers):
            t = threading.Thread(target=self._run, daemon=True,
                                 name=f"job-{name}-{i}")
            t.start()
            self._threads.append(t)

    def submit(self, item):
        self._q.put(item)

    def _run(self):
        while True:
            job_id, fn, store, cancel_set, cancel_lock = self._q.get()
            job = store.get(job_id)
            if not job or job["status"] == "cancelled":
                continue
            ctx = JobContext(store, job_id)
            store.update(job_id, status="running", started=time.time())
            try:
                with cancel_lock:
                    cancel_set[job_id] = ctx
                result = fn(ctx)
                store.update(job_id, status="done", progress=1.0,
                             outcome=json.dumps(result or {}),
                             finished=time.time())
            except CancelledError:
                store.update(job_id, status="cancelled", finished=time.time())
            except Exception as e:
                store.update(
                    job_id, status="failed",
                    error=f"{e}\n{traceback.format_exc(limit=5)}",
                    finished=time.time())
            finally:
                ctx.cancel_all()
                with cancel_lock:
                    cancel_set.pop(job_id, None)


class JobManager:
    """Lane-based job manager replacing the single-worker JobRunner."""

    def __init__(self, store: JobStore, lanes=None):
        self.store = store
        self._cancel = {}
        self._cancel_lock = threading.Lock()
        lane_cfg = lanes or _LANE_DEFAULTS
        self.lanes = {name: JobLane(name, workers=w)
                      for name, w in lane_cfg.items()}

    def submit(self, job_type, title, fn, payload=None, lane="mutation", **kwargs):
        """fn(ctx, **kwargs) -> result dict (or raises)."""
        job_id = self.store.create(job_type, title, payload, lane=lane)
        self.lanes[lane].submit((job_id, fn, self.store,
                                 self._cancel, self._cancel_lock))
        return job_id

    def cancel(self, job_id):
        with self._cancel_lock:
            ctx = self._cancel.get(job_id)
        if ctx:
            # Immediate cleanup: terminate registered process groups and run
            # cleanup callbacks NOW (the lane's finally runs them again
            # harmlessly when the job function unwinds).
            ctx.set_cancelled()
            ctx.cancel_all()
        self.store.update(job_id, status="cancelled", finished=time.time())

    def is_cancelled(self, job_id):
        with self._cancel_lock:
            ctx = self._cancel.get(job_id)
        return ctx.is_cancelled() if ctx else True


# --- backward-compatible wrapper ---

class JobRunner:
    """Compatibility wrapper with the same API as before.

    Creates all lanes so that mutation functions can route to the right one.
    """

    def __init__(self, store: JobStore):
        self._mgr = JobManager(store)

    @property
    def store(self):
        return self._mgr.store

    def submit(self, job_type, title, fn, payload=None, **kwargs):
        """fn(ctx) -> result dict (or raises).  For backward compat, if fn
        takes 2 positional args it is wrapped as fn(store, job_id)."""
        lane = kwargs.pop("lane", "mutation")
        import inspect
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        positional = [p for p in params
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(positional) >= 2:
            def adapted(ctx, **kw):
                return fn(ctx.store, ctx.job_id, **kw)
            return self._mgr.submit(job_type, title, adapted, payload,
                                    lane=lane, **kwargs)
        return self._mgr.submit(job_type, title, fn, payload,
                                lane=lane, **kwargs)

    def cancel(self, job_id):
        self._mgr.cancel(job_id)

    def is_cancelled(self, job_id):
        return self._mgr.is_cancelled(job_id)

    @property
    def _threads(self):
        threads = []
        for lane in self._mgr.lanes.values():
            threads.extend(lane._threads)
        return threads

    @property
    def _thread(self):
        """Backward-compat: return first worker thread."""
        threads = self._threads
        return threads[0] if threads else None
