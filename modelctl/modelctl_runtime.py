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
from dataclasses import dataclass
from pathlib import Path

from modelctl_paths import STATE_DIR
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


@dataclass(frozen=True)
class ObservationIdentity:
    """What must still hold for a past measurement to describe this setup.

    Most of the identities that matter are
    already load-bearing in the plan ID itself: _plan_id() hashes the model
    fingerprint, binary fingerprint, device, split, context, cache types,
    fit, extra args, moe_cache config and env. Change any of those and the
    old run is filed under a different plan, so it is never offered as
    evidence for the new one -- that is a stronger guarantee than a
    staleness flag, and duplicating it here would be dead code.

    What plan identity does NOT cover, and what this therefore checks:

      hardware   GPU/driver/kernel; the same plan on re-seated hardware
      backend    the runtime behind the plan
      env        the *effective* launch environment, which includes ambient
                 variables the profile never declared
      storage    which device the model was actually read from
      capability what the binary reported it could do

    Benchmark mode is deliberately absent: observations_for_profile()
    buckets by cache_state and never compares across states, so a cold run
    cannot masquerade as a warm one in the first place.

    Every field is compared only when both sides are non-empty. A blank
    means "not recorded", and unknown must not be reported as changed --
    rows predating a column would otherwise all read stale, which trains
    the user to ignore the flag.
    """
    hardware_fingerprint: str = ""
    backend_fingerprint: str = ""
    environment_fingerprint: str = ""
    storage_device: str = ""
    capability_digest: str = ""

    @classmethod
    def current(cls, snapshot=None, backend=None, profile_name="",
                storage_device=""):
        """Identity of the setup a measurement would run in right now.

        `backend` is a ResolvedBackend when the caller already has one.
        Resolving one solely to build this is not worth it -- the caller
        then simply gets the hardware and backend checks, which is what it
        had before.
        """
        backend_fp = ""
        if snapshot is not None and profile_name:
            backend_fp = getattr(snapshot, "backend_fingerprints", {}).get(
                profile_name, "")
        return cls(
            hardware_fingerprint=getattr(snapshot, "fingerprint", "") or "",
            backend_fingerprint=backend_fp,
            environment_fingerprint=getattr(
                backend, "environment_fingerprint", "") or "",
            storage_device=storage_device,
            capability_digest=getattr(
                backend, "capability_fingerprint", "") or "")

    def stale_reasons(self, row) -> tuple:
        """Why `row` no longer describes this setup; empty when it still does.

        Reasons rather than a bare bool because the decision trace has to
        explain automatic choices, and "rejected: stale" is not an
        explanation.
        """
        checks = (
            ("hardware", self.hardware_fingerprint, row.get("hardware_fingerprint")),
            ("backend", self.backend_fingerprint, row.get("backend_fingerprint")),
            ("environment", self.environment_fingerprint,
             row.get("environment_fingerprint")),
            ("storage device", self.storage_device, row.get("storage_device")),
            ("runtime capabilities", self.capability_digest,
             row.get("capability_digest")),
        )
        out = []
        for label, current, recorded in checks:
            if current and recorded and current != recorded:
                out.append(f"{label} changed since this was measured")
        return tuple(out)

# Additive migrations for plan_runs provenance columns.
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
    # What produced this row. "benchmark" rows are measurements: a
    # controlled run that reports throughput. "serving" rows are
    # bookkeeping from the managed worker -- a real llama-swap load,
    # which has no throughput numbers and whose duration says nothing
    # about the plan. Ranking, validation, and autotune consider only
    # benchmarks; a serving row used to shadow the real measurement for
    # the same plan (newest-per-plan wins), so a measured plan read back
    # as "validated (None tok/s)" and lost its measured advantage.
    ("run_kind", "TEXT NOT NULL DEFAULT 'benchmark'", "benchmark"),
    # Cold/warm separation: measurements carry their cache
    # state ("cold"/"warm"/"" for legacy rows) and ranking never compares
    # across states.
    ("cache_state", "TEXT NOT NULL DEFAULT ''", ""),
    # Process I/O and page-fault sampling: distinguishes compute
    # speed from active storage reads. read_bytes_warmup/_generation split
    # /proc/<pid>/io's read_bytes at the warmup/measured-run boundary;
    # read_bytes is the run total. NULL for rows recorded before this
    # migration (the sampler didn't track it yet).
    ("read_bytes", "INTEGER", None),
    ("read_bytes_warmup", "INTEGER", None),
    ("read_bytes_generation", "INTEGER", None),
    ("major_faults", "INTEGER", None),
    ("minor_faults", "INTEGER", None),
    # PSS divides shared pages by their sharers, so a main model
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
        """Back-compat wrapper: the reservation dict on success, None on
        denial. Callers who need to know WHY use
        acquire_reservation_verdict()."""
        reservation, _denial = self.acquire_reservation_verdict(
            profile_name, plan_id, claim_dict, owner_pid, budgets=budgets)
        return reservation

    def acquire_reservation_verdict(self, profile_name, plan_id, claim_dict,
                                    owner_pid, budgets=None):
        """Atomically acquire a reservation; on denial, say why.

        Uses BEGIN IMMEDIATE to serialize concurrent acquisitions.
        Cleans up stale reservations from dead PIDs first.

        claim_dict's "vram_bytes" and "ram_bytes" are ADMISSION values:
        peak per-device VRAM including the dynamic expert-cache budget,
        and resident (not mmap-addressed) RAM. Callers produce them with
        ResourceClaim.vram_admission_bytes()/ram_admission_bytes() rather
        than rebuilding them from decomposed fields; extra keys (e.g.
        vram_cache_bytes, mmap_bytes) are explanation only and are not
        charged here.

        When `budgets` ({resource: bytes}, with per-device VRAM keys and
        "RAM") is given, admission is BYTE-BASED: the claim plus every other
        live claim must fit the budget for each resource.
        Without budgets it falls back to the legacy conservative rule (any
        device overlap conflicts) over pending/starting rows only.

        "Every other live claim" is asymmetric by device, because the
        budgets are: a LOCAL card contributes driver-reported free bytes,
        so a model that finished loading has already subtracted itself
        and only pending/starting claims may be charged on top -- but a
        REMOTE device contributes the operator's declared ceiling, which
        no reading ever reduces. An `RPC:`-namespaced key is therefore
        charged from ACTIVE reservations too, or two managed models both
        pass the gate against one laptop's ceiling and overshoot there is
        an OOM kill rather than a swap-out (2026-08-04).

        Returns (reservation, None) on success, (None, denial) on refusal.
        The denial dict distinguishes a claim that does not fit the budget
        (code insufficient_vram / insufficient_ram -- possible with ZERO
        other workers) from a genuine collision (code reservation_conflict):
        {"code", "resource", "need_bytes", "budget_bytes", "pending_bytes",
         "holders": [profile names of live pending/starting claimants,
         plus active ones charging a remote (`RPC:`) key]}.
        """
        claim_json = json.dumps(claim_dict)
        now = time.time()
        res_id = uuid.uuid4().hex[:12]

        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                self._cleanup_stale(c)
                existing = c.execute(
                    "SELECT * FROM reservations "
                    "WHERE state IN ('pending', 'starting', 'active')"
                ).fetchall()
                if budgets is not None:
                    pending, holders = _fold_live_claims(existing, owner_pid)
                    for dev, need in claim_dict.get("vram_bytes", {}).items():
                        if need + pending.get(dev, 0) > budgets.get(dev, 0):
                            c.execute("ROLLBACK")
                            return None, {
                                "code": "insufficient_vram", "resource": dev,
                                "need_bytes": int(need),
                                "budget_bytes": int(budgets.get(dev, 0)),
                                "pending_bytes": int(pending.get(dev, 0)),
                                "holders": holders}
                    ram_need = claim_dict.get("ram_bytes", 0)
                    if ram_need + pending.get("RAM", 0) > budgets.get("RAM", 0):
                        c.execute("ROLLBACK")
                        return None, {
                            "code": "insufficient_ram", "resource": "RAM",
                            "need_bytes": int(ram_need),
                            "budget_bytes": int(budgets.get("RAM", 0)),
                            "pending_bytes": int(pending.get("RAM", 0)),
                            "holders": holders}
                else:
                    for row in existing:
                        if row["state"] == "active":
                            # The legacy rule is "any device overlap
                            # conflicts", which is what lets llama-swap
                            # hand a local card from one model to the
                            # next. Counting active rows here would
                            # deadlock that handover, and without a
                            # budget map there is no remote ceiling to
                            # protect anyway.
                            continue
                        if row["owner_pid"] == owner_pid:
                            continue
                        if not _pid_alive(row["owner_pid"]):
                            continue
                        existing_claim = json.loads(row["claim_json"])
                        for dev in claim_dict.get("vram_bytes", {}):
                            if dev in existing_claim.get("vram_bytes", {}):
                                c.execute("ROLLBACK")
                                return None, {
                                    "code": "reservation_conflict",
                                    "resource": dev,
                                    "need_bytes": int(
                                        claim_dict["vram_bytes"][dev]),
                                    "budget_bytes": 0, "pending_bytes": 0,
                                    "holders": [row["profile_name"]]}

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
                        "created_at": now, "updated_at": now}, None
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

    def charged_bytes(self, exclude_pid=None):
        """{resource: bytes} already charged against the budget map.

        What `acquire_reservation_verdict` would count if it ran right
        now -- the same fold, so a caller that pre-filters plans by
        feasibility reaches the same verdict the gate will. Includes
        remote (`RPC:`) bytes held by ACTIVE reservations, which no
        driver reading reports and which `pending_claims` therefore
        cannot see.

        Advisory only: this reads outside a transaction, so the gate
        remains the authority.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM reservations "
                "WHERE state IN ('pending', 'starting', 'active')"
            ).fetchall()
        return _fold_live_claims(rows, exclude_pid)[0]

    def live_holdings(self, exclude_pid=None):
        """Who holds memory right now, for display:
        [{profile, state, devices: {resource: bytes}}].

        NOT the admission fold. `charged_bytes` answers "what would the
        gate count", and deliberately charges an ACTIVE row's `RPC:`
        keys only -- local free bytes already have a running model
        subtracted, so counting it again would refuse a launch that
        fits. That subtraction is also why no memory view could ever say
        WHO holds the RAM: free shrinks with no holder named, and "no
        room" renders exactly like "no room because this model already
        holds it". This fold answers that second question: every live
        row contributes everything it holds, local devices and RAM
        included, attributed to the profile holding it.

        Same hygiene as `_fold_live_claims`: rows owned by `exclude_pid`
        or a dead process, claims that do not decode, and unpriceable
        byte counts contribute nothing. Advisory and read-only --
        admission never reads this.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM reservations "
                "WHERE state IN ('pending', 'starting', 'active')"
            ).fetchall()
        holdings = []
        for row in rows:
            if (row["owner_pid"] == exclude_pid
                    or not _pid_alive(row["owner_pid"])):
                continue
            try:
                claim = json.loads(row["claim_json"])
            except (TypeError, ValueError):
                continue
            if not isinstance(claim, dict):
                continue
            vram = claim.get("vram_bytes")
            vram = vram if isinstance(vram, dict) else {}
            # Strictly positive, unlike the gate's >= 0: a zero-byte
            # entry charges nothing there, but here it would name a
            # profile as a holder of nothing.
            devices = {dev: int(b) for dev, b in vram.items()
                       if isinstance(b, (int, float)) and b > 0}
            ram = claim.get("ram_bytes", 0)
            if isinstance(ram, (int, float)) and ram > 0:
                devices["RAM"] = devices.get("RAM", 0) + int(ram)
            # What this model addresses off the SSD instead of holding.
            # Deliberately NOT a device entry: mmap'd weights occupy no
            # budget anywhere (plans.py books them outside
            # `ram_admission_bytes`, because charging them as resident
            # refuses every oversized MoE served through mmap), and the
            # RAM row's free-minus-reserve budget already counts those
            # pages as reclaimable. Folding them into `devices` would put
            # a held figure next to a budget that disagrees with it.
            # They ride separately so a screen can say where the rest of
            # a model is without any bar claiming to hold it.
            streamed = claim.get("mmap_bytes", 0)
            streamed = (int(streamed)
                        if isinstance(streamed, (int, float)) and streamed > 0
                        else 0)
            # A model served entirely through mmap reserves no device at
            # all. Dropping it here made the one model whose bytes are
            # all on disk the one model no screen could name.
            if not devices and not streamed:
                continue
            holdings.append({"profile": row["profile_name"],
                             "state": row["state"], "devices": devices,
                             "mmap_bytes": streamed})
        return holdings

    def pending_claims(self, exclude_pid=None):
        """Pending/starting claims from live processes, excluding one PID.

        Deliberately NOT the admission view: a model that finished
        loading has left this list while still holding its remote
        ceiling. Callers weighing whether a plan can be admitted want
        `charged_bytes`.
        """
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
                "cache_state, run_kind, read_bytes, read_bytes_warmup, read_bytes_generation, "
                "major_faults, minor_faults, "
                "peak_pss_bytes, read_syscalls, disk_read_bytes, storage_device, "
                "baseline_vram_json, final_vram_json, storage_activity, "
                "storage_activity_detail, rates_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?,?,?,?,?)",
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
                 run.get("run_kind", "benchmark"),
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
                                 backend_fingerprint="", identity=None):
        """Latest successful measurement per plan, keyed by plan_id, shaped
        for rank_plans' `observations` argument. Includes a `stale` flag, and
        `stale_reasons` naming what changed, when the run's recorded identity
        no longer matches the current one.

        `identity` is an ObservationIdentity covering everything plan identity
        does not (see that class). The two positional fingerprints are the
        older, narrower form and are still honoured; passing both takes the
        union, so an existing caller cannot lose a check by adopting the
        richer argument.

        Cold and warm measurements are never conflated: a single cache_state
        is chosen for the whole profile (the state with the widest plan
        coverage, ties prefer the colder/conservative state), and only plans
        with a successful run in that state get an observation. Legacy rows
        and serving runs have cache_state "" — semantically "unknown" — and
        are bucketed with "cold" (the conservative interpretation) rather
        than forming a third state that could shadow real measurements."""
        with self._conn() as c:
            # Benchmarks only. A serving row carries no throughput and
            # its duration reflects a user session, not the plan; letting
            # one into this set made the newest-per-plan rule discard the
            # real measurement.
            rows = c.execute(
                "SELECT * FROM plan_runs WHERE profile_name=? AND success=1 "
                "AND run_kind='benchmark' "
                "ORDER BY started_at DESC", (profile_name,)).fetchall()
        # Newest successful run per (cache_state, plan_id); rows are
        # newest-first so the first sighting of a plan wins.
        by_state = {}
        for r in rows:
            d = dict(r)
            state = d.get("cache_state") or "unknown"
            plans_for_state = by_state.setdefault(state, {})
            if d["plan_id"] not in plans_for_state:
                plans_for_state[d["plan_id"]] = d
        if not by_state:
            return {}
        # Coverage first; conservatism breaks ties. "warm" (page cache
        # only) sits between "cold" and "warm-expert" (page cache plus a
        # filled GPU expert cache) -- three distinct states, never
        # compared against each other.
        # "unknown" (legacy rows with no recorded state) is the most
        # conservative: it cannot be claimed as either cold or warm.
        conservatism = {"unknown": -1, "cold": 0, "warm": 1, "warm-expert": 2}
        chosen_state = sorted(
            by_state,
            key=lambda s: (-len(by_state[s]), conservatism.get(s, 1)))[0]
        # The positional fingerprints and the identity object describe the
        # same current setup, so fold them together rather than choosing.
        ident = identity or ObservationIdentity()
        ident = ObservationIdentity(
            hardware_fingerprint=hardware_fingerprint or ident.hardware_fingerprint,
            backend_fingerprint=backend_fingerprint or ident.backend_fingerprint,
            environment_fingerprint=ident.environment_fingerprint,
            storage_device=ident.storage_device,
            capability_digest=ident.capability_digest)

        out = {}
        for plan_id, d in by_state[chosen_state].items():
            reasons = ident.stale_reasons(d)
            out[plan_id] = {
                "generation_tps": d["generation_tps"],
                "prompt_tps": d["prompt_tps"],
                "load_seconds": d["load_seconds"],
                "actual_context": d["actual_context"],
                # Ranking reads these directly: interactive latency is
                # TTFT, not load time, and storage traffic is the bytes the
                # run actually read rather than whether mmap was configured.
                "ttft_seconds": d.get("ttft_seconds"),
                "read_bytes": d.get("read_bytes"),
                "cache_state": d.get("cache_state") or "cold",
                "stale": bool(reasons),
                "stale_reasons": list(reasons),
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


def _fold_live_claims(rows, exclude_pid=None):
    """Fold live reservation rows into ({resource: bytes}, [holders]).

    The ONE charging rule, shared by the atomic gate
    (`acquire_reservation_verdict`) and by the worker's soft feasibility
    pre-filter, because two copies of it drift: the gate learned to
    charge ACTIVE remote claims on 2026-08-04, and a hand-rolled second
    fold went on calling plans feasible that the gate then denied -- a
    wasted launch, or a hard failure when fallback is off.

    A pending or starting row is charged in full. An ACTIVE row is
    charged for its `RPC:`-namespaced devices only: a remote budget is a
    declared ceiling that a running model never reduces, while local
    budgets are driver-reported free bytes and available RAM, which
    already have that model subtracted. Charging an active row's local
    bytes would refuse a launch that fits.

    A row owned by `exclude_pid` or by a dead process contributes
    nothing. So does one whose claim does not decode to the expected
    shape: it cannot be priced, and raising instead would raise on EVERY
    admission for as long as that row lives -- an active row lives for
    the life of a served model, so one bad row would wedge every launch
    on the box. Same degrade direction as `modelctl_fleet.load_fleet`.
    """
    # Imported inside the call, as every other caller of the fleet
    # module does: this file is the runtime DB and must not grow a
    # module-level dependency on cluster state. Only the pure namespace
    # predicate is used -- nothing here reads a fleet file.
    import modelctl_fleet
    charged = {}
    holders = []
    for row in rows:
        if row["owner_pid"] == exclude_pid or not _pid_alive(row["owner_pid"]):
            continue
        try:
            claim = json.loads(row["claim_json"])
        except (TypeError, ValueError):
            # Deliberately not reported through record_event: that method
            # opens its own `with self._conn()`, and the connection is
            # thread-local, so leaving that block would COMMIT the gate's
            # still-open BEGIN IMMEDIATE early. A skipped row stays
            # visible in the table for anyone who looks.
            continue
        if not isinstance(claim, dict):
            continue
        vram = claim.get("vram_bytes")
        vram = vram if isinstance(vram, dict) else {}
        # Drop unpriceable entries before anything reads the dict, so a
        # row holding nothing chargeable is never named as a holder. A
        # negative count would be worse than a non-numeric one -- it
        # would subtract from another row's occupancy, the one direction
        # an admission gate must never fail in.
        vram = {dev: b for dev, b in vram.items()
                if isinstance(b, (int, float)) and b >= 0}
        if row["state"] == "active":
            vram = {dev: b for dev, b in vram.items()
                    if modelctl_fleet.is_remote_key(dev)}
            if not vram:
                continue
        else:
            ram = claim.get("ram_bytes", 0)
            if isinstance(ram, (int, float)) and ram >= 0:
                charged["RAM"] = charged.get("RAM", 0) + ram
        holders.append(row["profile_name"])
        for dev, b in vram.items():
            charged[dev] = charged.get(dev, 0) + b
    return charged, holders

