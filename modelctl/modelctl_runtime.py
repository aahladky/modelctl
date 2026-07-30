"""Runtime database for modelctl: reservations and runtime events.

Provides persistent storage for resource reservations (preventing
concurrent workers from overcommitting the same GPU memory) and
runtime events (for diagnostics and observability).

Database: ~/.local/share/modelctl/runtime.db
"""
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

STATE_DIR = Path(os.environ.get(
    "MODELCTL_STATE_DIR",
    Path.home() / ".local" / "share" / "modelctl"))
DB_PATH = STATE_DIR / "runtime.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
    id TEXT PRIMARY KEY,
    profile_name TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    owner_pid INTEGER NOT NULL,
    state TEXT NOT NULL,
    claim_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_events (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    profile_name TEXT,
    event_type TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_reservations_owner
ON reservations (owner_pid, state);

CREATE INDEX IF NOT EXISTS idx_reservations_profile
ON reservations (profile_name, state);

CREATE TABLE IF NOT EXISTS plan_runs (
    id INTEGER PRIMARY KEY,
    profile_name TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    hardware_fingerprint TEXT NOT NULL,
    backend_fingerprint TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    success INTEGER NOT NULL,
    failure_class TEXT,
    load_seconds REAL,
    ttft_seconds REAL,
    prompt_tps REAL,
    generation_tps REAL,
    peak_vram_json TEXT NOT NULL DEFAULT '{}',
    peak_ram_bytes INTEGER,
    actual_context INTEGER,
    exit_code INTEGER,
    log_path TEXT NOT NULL DEFAULT '',
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_plan_runs_lookup
ON plan_runs (profile_name, plan_id, hardware_fingerprint, backend_fingerprint, started_at);
"""

_STATES = {"pending", "starting", "active", "releasing", "stale"}

# Additive migrations for plan_runs provenance columns (Task 1.6).
# Each tuple is (column_name, column_type, default).
_MIGRATIONS = [
    ("command_fingerprint", "TEXT NOT NULL DEFAULT ''", ""),
    ("command_argv_json", "TEXT NOT NULL DEFAULT '[]'", "[]"),
    ("binary_path", "TEXT NOT NULL DEFAULT ''", ""),
    ("binary_fingerprint", "TEXT NOT NULL DEFAULT ''", ""),
    ("environment_fingerprint", "TEXT NOT NULL DEFAULT ''", ""),
    ("capability_schema", "INTEGER NOT NULL DEFAULT 0", 0),
    ("capability_digest", "TEXT NOT NULL DEFAULT ''", ""),
    ("claim_json", "TEXT NOT NULL DEFAULT '{}'", "{}"),
    ("decision_json", "TEXT NOT NULL DEFAULT '{}'", "{}"),
    ("parent_job_id", "TEXT", None),
    ("fallback_ordinal", "INTEGER", None),
    # Cold/warm separation (Task 6.2): measurements carry their cache
    # state ("cold"/"warm"/"" for legacy rows) and ranking never compares
    # across states.
    ("cache_state", "TEXT NOT NULL DEFAULT ''", ""),
    # Process I/O and page-fault sampling (Task 3.3): distinguishes compute
    # speed from active storage reads. read_bytes_warmup/_generation split
    # /proc/<pid>/io's read_bytes at the warmup/measured-run boundary;
    # read_bytes is the run total. NULL for rows recorded before this
    # migration (the sampler didn't track it yet).
    ("read_bytes", "INTEGER", None),
    ("read_bytes_warmup", "INTEGER", None),
    ("read_bytes_generation", "INTEGER", None),
    ("major_faults", "INTEGER", None),
    ("minor_faults", "INTEGER", None),
    # Task D2. PSS divides shared pages by their sharers, so a main model
    # and a draft sharing mmap'd pages are not double-counted; it is not
    # always readable, and NULL means unknown rather than zero.
    ("peak_pss_bytes", "INTEGER", None),
    # Read syscall count separates "many small reads" from "few large
    # ones" at the same byte total.
    ("read_syscalls", "INTEGER", None),
    # Bytes read from the model's own block device over the run, and which
    # device that was -- process counters alone cannot attribute reads to
    # a disk.
    ("disk_read_bytes", "INTEGER", None),
    ("storage_device", "TEXT NOT NULL DEFAULT ''", ""),
    # VRAM at process start and after exit, alongside the existing peak.
    ("baseline_vram_json", "TEXT NOT NULL DEFAULT '{}'", "{}"),
    ("final_vram_json", "TEXT NOT NULL DEFAULT '{}'", "{}"),
    # What the storage actually did, derived from counters. Never inferred
    # from mmap being enabled -- a fully page-cached mmap model reads
    # nothing at all.
    ("storage_activity", "TEXT NOT NULL DEFAULT ''", ""),
    ("storage_activity_detail", "TEXT NOT NULL DEFAULT ''", ""),
    # Derived rates, stored next to the raw counters they come from.
    ("rates_json", "TEXT NOT NULL DEFAULT '{}'", "{}"),
]


class RuntimeDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(_SCHEMA)
            self._run_migrations(c)

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _run_migrations(self, cursor):
        """Additive migrations: add columns that don't exist yet."""
        existing = {row[1] for row in cursor.execute("PRAGMA table_info(plan_runs)").fetchall()}
        for col_name, col_type, _default in _MIGRATIONS:
            if col_name not in existing:
                cursor.execute(f"ALTER TABLE plan_runs ADD COLUMN {col_name} {col_type}")

    # --- reservations ---

    def acquire_reservation(self, profile_name, plan_id, claim_dict, owner_pid,
                            budgets=None):
        """Atomically acquire a reservation.

        Uses BEGIN IMMEDIATE to serialize concurrent acquisitions.
        Cleans up stale reservations from dead PIDs first.

        When `budgets` ({resource: bytes}, with per-device VRAM keys and
        "RAM") is given, admission is BYTE-BASED: the claim plus every other
        live pending/starting claim must fit the budget for each resource.
        Without budgets it falls back to the legacy conservative rule (any
        device overlap conflicts).

        Returns the reservation dict on success, None if resources
        are already claimed by another live process.
        """
        claim_json = json.dumps(claim_dict)
        now = time.time()
        res_id = uuid.uuid4().hex[:12]

        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                self._cleanup_stale(c)
                existing = c.execute(
                    "SELECT * FROM reservations WHERE state IN ('pending', 'starting')"
                ).fetchall()
                if budgets is not None:
                    pending = {}
                    for row in existing:
                        if row["owner_pid"] == owner_pid or not _pid_alive(row["owner_pid"]):
                            continue
                        ex_claim = json.loads(row["claim_json"])
                        for dev, b in ex_claim.get("vram_bytes", {}).items():
                            pending[dev] = pending.get(dev, 0) + b
                        pending["RAM"] = pending.get("RAM", 0) + ex_claim.get("ram_bytes", 0)
                    for dev, need in claim_dict.get("vram_bytes", {}).items():
                        if need + pending.get(dev, 0) > budgets.get(dev, 0):
                            c.execute("ROLLBACK")
                            return None
                    ram_need = claim_dict.get("ram_bytes", 0)
                    if ram_need + pending.get("RAM", 0) > budgets.get("RAM", 0):
                        c.execute("ROLLBACK")
                        return None
                else:
                    for row in existing:
                        if row["owner_pid"] == owner_pid:
                            continue
                        if not _pid_alive(row["owner_pid"]):
                            continue
                        existing_claim = json.loads(row["claim_json"])
                        for dev in claim_dict.get("vram_bytes", {}):
                            if dev in existing_claim.get("vram_bytes", {}):
                                c.execute("ROLLBACK")
                                return None

                c.execute(
                    "INSERT INTO reservations "
                    "(id, profile_name, plan_id, owner_pid, state, claim_json, "
                    "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (res_id, profile_name, plan_id, owner_pid, "pending",
                     claim_json, now, now))
                c.execute("COMMIT")
                return {"id": res_id, "profile_name": profile_name,
                        "plan_id": plan_id, "owner_pid": owner_pid,
                        "state": "pending", "claim": claim_dict,
                        "created_at": now, "updated_at": now}
            except Exception:
                c.execute("ROLLBACK")
                raise

    def update_reservation(self, res_id, state=None, owner_pid=None):
        """Update reservation state."""
        fields = {"updated_at": time.time()}
        if state:
            fields["state"] = state
        if owner_pid is not None:
            fields["owner_pid"] = owner_pid
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values())
        with self._conn() as c:
            c.execute(f"UPDATE reservations SET {cols} WHERE id=?",
                      (*vals, res_id))

    def release_reservation(self, res_id):
        """Remove a reservation."""
        with self._conn() as c:
            c.execute("DELETE FROM reservations WHERE id=?", (res_id,))

    def get_reservations(self, profile_name=None, state=None):
        """Get reservations, optionally filtered."""
        query = "SELECT * FROM reservations WHERE 1=1"
        params = []
        if profile_name:
            query += " AND profile_name=?"
            params.append(profile_name)
        if state:
            query += " AND state=?"
            params.append(state)
        with self._conn() as c:
            rows = c.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def pending_claims(self, exclude_pid=None):
        """Get all pending/starting claims from live processes, excluding one PID."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM reservations WHERE state IN ('pending', 'starting')"
            ).fetchall()
        claims = []
        for r in rows:
            if exclude_pid and r["owner_pid"] == exclude_pid:
                continue
            if not _pid_alive(r["owner_pid"]):
                continue
            claims.append(json.loads(r["claim_json"]))
        return claims

    def cleanup_stale(self):
        """Public wrapper to clean up stale reservations."""
        with self._conn() as c:
            self._cleanup_stale(c)

    def _cleanup_stale(self, cursor):
        """Delete reservations whose owner PID is dead."""
        rows = cursor.execute(
            "SELECT id, owner_pid, state FROM reservations "
            "WHERE state IN ('pending', 'starting', 'active')"
        ).fetchall()
        for row in rows:
            if not _pid_alive(row["owner_pid"]):
                cursor.execute(
                    "UPDATE reservations SET state='stale' WHERE id=?",
                    (row["id"],))

    # --- events ---

    def record_event(self, event_type, profile_name=None, detail=None):
        """Record a runtime event."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO runtime_events "
                "(created_at, profile_name, event_type, detail_json) VALUES (?,?,?,?)",
                (time.time(), profile_name, event_type,
                 json.dumps(detail or {})))

    def get_events(self, profile_name=None, limit=50):
        """Get recent runtime events."""
        query = "SELECT * FROM runtime_events"
        params = []
        if profile_name:
            query += " WHERE profile_name=?"
            params.append(profile_name)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(query, params).fetchall()
        return [dict(r) for r in rows]


    # --- plan runs (measured observations) ---

    def record_plan_run(self, run):
        """Insert a PlanRun-shaped dict into plan_runs."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO plan_runs (profile_name, plan_id, hardware_fingerprint, "
                "backend_fingerprint, started_at, finished_at, success, failure_class, "
                "load_seconds, ttft_seconds, prompt_tps, generation_tps, peak_vram_json, "
                "peak_ram_bytes, actual_context, exit_code, log_path, details_json, "
                "command_fingerprint, command_argv_json, binary_path, binary_fingerprint, "
                "environment_fingerprint, capability_schema, capability_digest, "
                "claim_json, decision_json, parent_job_id, fallback_ordinal, "
                "cache_state, read_bytes, read_bytes_warmup, read_bytes_generation, "
                "major_faults, minor_faults, "
                "peak_pss_bytes, read_syscalls, disk_read_bytes, storage_device, "
                "baseline_vram_json, final_vram_json, storage_activity, "
                "storage_activity_detail, rates_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?,?,?,?)",
                (run["profile_name"], run["plan_id"],
                 run.get("hardware_fingerprint", ""), run.get("backend_fingerprint", ""),
                 run.get("started_at", time.time()), run.get("finished_at"),
                 int(bool(run.get("success"))), run.get("failure_class"),
                 run.get("load_seconds"), run.get("ttft_seconds"),
                 run.get("prompt_tps"), run.get("generation_tps"),
                 json.dumps(run.get("peak_vram_bytes", {})),
                 run.get("peak_ram_bytes"), run.get("actual_context"),
                 run.get("exit_code"), run.get("log_path", ""),
                 json.dumps(run.get("details", {})),
                 run.get("command_fingerprint", ""),
                 json.dumps(run.get("command_argv", [])),
                 run.get("binary_path", ""),
                 run.get("binary_fingerprint", ""),
                 run.get("environment_fingerprint", ""),
                 run.get("capability_schema", 0),
                 run.get("capability_digest", ""),
                 json.dumps(run.get("claim", {})),
                 json.dumps(run.get("decision", {})),
                 run.get("parent_job_id"),
                 run.get("fallback_ordinal"),
                 run.get("cache_state", ""),
                 run.get("read_bytes"),
                 run.get("read_bytes_warmup"),
                 run.get("read_bytes_generation"),
                 run.get("major_faults"),
                 run.get("minor_faults"),
                 run.get("peak_pss_bytes"),
                 run.get("read_syscalls"),
                 run.get("disk_read_bytes"),
                 run.get("storage_device", "") or "",
                 json.dumps(run.get("baseline_vram_bytes", {})),
                 json.dumps(run.get("final_vram_bytes", {})),
                 run.get("storage_activity", "") or "",
                 run.get("storage_activity_detail", "") or "",
                 json.dumps(run.get("rates", {}))))
            return c.execute("SELECT last_insert_rowid()").fetchone()[0]

    def plan_runs_for(self, profile_name, plan_id=None, limit=50):
        q = ("SELECT * FROM plan_runs WHERE profile_name=?"
             + (" AND plan_id=?" if plan_id else "")
             + " ORDER BY started_at DESC LIMIT ?")
        args = (profile_name, plan_id, limit) if plan_id else (profile_name, limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def observations_for_profile(self, profile_name, hardware_fingerprint="",
                                 backend_fingerprint=""):
        """Latest successful measurement per plan, keyed by plan_id, shaped
        for rank_plans' `observations` argument. Includes a `stale` flag when
        the run's fingerprints don't match the current environment.

        Cold and warm measurements are never conflated: a single cache_state
        is chosen for the whole profile (the state with the widest plan
        coverage, ties prefer the colder/conservative state), and only plans
        with a successful run in that state get an observation. Legacy rows
        and serving runs have cache_state "" — semantically "unknown" — and
        are bucketed with "cold" (the conservative interpretation) rather
        than forming a third state that could shadow real measurements."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM plan_runs WHERE profile_name=? AND success=1 "
                "ORDER BY started_at DESC", (profile_name,)).fetchall()
        # Newest successful run per (cache_state, plan_id); rows are
        # newest-first so the first sighting of a plan wins.
        by_state = {}
        for r in rows:
            d = dict(r)
            state = d.get("cache_state") or "cold"
            plans_for_state = by_state.setdefault(state, {})
            if d["plan_id"] not in plans_for_state:
                plans_for_state[d["plan_id"]] = d
        if not by_state:
            return {}
        chosen_state = sorted(
            by_state,
            key=lambda s: (-len(by_state[s]), 0 if s == "cold" else 1))[0]
        out = {}
        for plan_id, d in by_state[chosen_state].items():
            stale = ((hardware_fingerprint and d["hardware_fingerprint"] != hardware_fingerprint)
                     or (backend_fingerprint and d["backend_fingerprint"] != backend_fingerprint))
            out[plan_id] = {
                "generation_tps": d["generation_tps"],
                "prompt_tps": d["prompt_tps"],
                "load_seconds": d["load_seconds"],
                "actual_context": d["actual_context"],
                "cache_state": d.get("cache_state") or "cold",
                "stale": bool(stale),
                "run_id": d["id"],
                "measured_at": d["started_at"],
            }
        return out


    def failures_for_profile(self, profile_name, limit=5):
        """Recent failure classes per plan: {plan_id: [failure_class, ...]},
        newest first. Suppression decisions (unsupported arch, invalid args)
        key on these."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT plan_id, failure_class FROM plan_runs "
                "WHERE profile_name=? AND success=0 AND failure_class IS NOT NULL "
                "ORDER BY started_at DESC", (profile_name,)).fetchall()
        out = {}
        for r in rows:
            out.setdefault(r["plan_id"], [])
            if len(out[r["plan_id"]]) < limit:
                out[r["plan_id"]].append(r["failure_class"])
        return out


def _pid_alive(pid):
    """Check if a process is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False

