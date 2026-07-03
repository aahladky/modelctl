# Model Performance Stats Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist per-profile model performance data (load times, token throughput, error rates) by parsing `llama-router.service`'s journal, so `modelctl stats show` can answer "how has each profile actually performed" instead of only the live-only `modelctl router stats`.

**Architecture:** A new stdlib-only module, `modelctl_stats.py`, parses journald JSON entries into structured SQLite rows (cursor-synced, so each `modelctl stats sync` only processes genuinely new lines), correlating per-instance `[pid]`-tagged lines to profile names via a persisted pid→profile map. `modelctl.py` gains a `stats sync`/`stats show` subcommand pair that calls into this module. A systemd `--user` timer (files only, not auto-installed) invokes `stats sync` periodically so events aren't lost to journald's rotation.

**Tech Stack:** Python 3 stdlib only (`sqlite3`, `subprocess`, `re`, `fcntl`, `json`), `unittest` + `unittest.mock` for tests (matching this repo's existing convention — no pytest, no new dependencies).

---

## Before you start

Read [docs/superpowers/specs/2026-07-02-model-perf-stats-design.md](../specs/2026-07-02-model-perf-stats-design.md) — the approved design this plan implements. It explains *why* each piece exists (the pid/port correlation approach, what counts as an error, why journald-parsing instead of a watcher, etc.). This plan only covers *how*.

All journald line formats referenced below were captured directly from this system's actual `llama-router.service` output — they are not hypothetical.

---

### Task 1: `modelctl_stats.py` skeleton + schema

**Files:**
- Create: `modelctl_stats.py`
- Create: `test_modelctl_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# test_modelctl_stats.py
import sqlite3
import unittest

import modelctl_stats


class TestEnsureSchema(unittest.TestCase):
    def test_creates_all_tables(self):
        conn = sqlite3.connect(":memory:")
        modelctl_stats.ensure_schema(conn)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(
            tables,
            {"sync_state", "pending_loads", "pid_map", "events"})

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        modelctl_stats.ensure_schema(conn)
        modelctl_stats.ensure_schema(conn)  # must not raise

    def test_events_rejects_unknown_event_type(self):
        conn = sqlite3.connect(":memory:")
        modelctl_stats.ensure_schema(conn)
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO events (ts, profile, event_type) "
                "VALUES (0, 'x', 'not_a_real_type')")


class TestConnect(unittest.TestCase):
    def test_creates_parent_dir_and_schema(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "nested" / "stats.db"
            conn = modelctl_stats.connect(db_path)
            try:
                self.assertTrue(db_path.exists())
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertIn("events", tables)
            finally:
                conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl_stats -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelctl_stats'`

- [ ] **Step 3: Write minimal implementation**

```python
# modelctl_stats.py
"""Per-profile performance stats for modelctl, sourced from
llama-router.service's journal.

Pure module: no modelctl import. sync_journal() and query_stats() are the
two entry points modelctl.py's `stats sync`/`stats show` commands call.
Everything else is an implementation detail of those two.
"""
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cursor TEXT
);

CREATE TABLE IF NOT EXISTS pending_loads (
    profile TEXT PRIMARY KEY,
    started_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pid_map (
    boot_id TEXT NOT NULL,
    port INTEGER NOT NULL,
    profile TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (boot_id, port)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    profile TEXT,
    event_type TEXT NOT NULL CHECK (event_type IN
        ('load_success', 'load_failure',
         'request_complete', 'request_error', 'service_crash')),
    duration_ms REAL,
    prompt_tokens INTEGER,
    gen_tokens INTEGER,
    prompt_tps REAL,
    gen_tps REAL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_profile ON events(profile);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the stats database at db_path, with its
    schema ensured. Caller is responsible for closing the connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    return conn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_modelctl_stats -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add modelctl_stats.py test_modelctl_stats.py
git commit -m "stats: add modelctl_stats module skeleton and SQLite schema"
```

---

### Task 2: Load-lifecycle line parsers

**Files:**
- Modify: `modelctl_stats.py`
- Modify: `test_modelctl_stats.py`

These three functions each take one journald entry's `MESSAGE` string and
return `None` if it doesn't match, or the matched data if it does. Sample
lines below were captured verbatim via `journalctl --user -u
llama-router.service -o json`.

- [ ] **Step 1: Write the failing test**

```python
# add to test_modelctl_stats.py
class TestParseLoadStart(unittest.TestCase):
    def test_matches_loading_line(self):
        msg = ("1.42.150.889 I srv  ensure_model: model name="
               "qwen3-5-35b-a3b-ud is not loaded, loading...")
        self.assertEqual(modelctl_stats.parse_load_start(msg),
                         "qwen3-5-35b-a3b-ud")

    def test_no_match_on_unrelated_line(self):
        self.assertIsNone(modelctl_stats.parse_load_start(
            "0.00.128.628 I srv   load_models: Loaded 1 cached model presets"))


class TestParseLoadSuccess(unittest.TestCase):
    def test_matches_model_loaded_line(self):
        msg = "[52035] 0.09.675.693 I srv  llama_server: model loaded"
        self.assertEqual(modelctl_stats.parse_load_success(msg), 52035)

    def test_no_match_without_port_prefix(self):
        self.assertIsNone(modelctl_stats.parse_load_success(
            "0.00.133.295 I srv  llama_server: listening on http://0.0.0.0:7071"))


class TestParseLoadFailure(unittest.TestCase):
    def test_matches_failed_to_load_exception(self):
        msg = ('0.05.964.076 W srv    operator(): got exception: '
               '{"error":{"code":500,"message":"model name='
               'qwen3.6-moe-mtp-longctx failed to load","type":"server_error"}}')
        profile, detail = modelctl_stats.parse_load_failure(msg)
        self.assertEqual(profile, "qwen3.6-moe-mtp-longctx")
        self.assertIn("failed to load", detail)

    def test_no_match_on_unrelated_exception(self):
        self.assertIsNone(modelctl_stats.parse_load_failure(
            '1.00.000.000 W srv operator(): got exception: '
            '{"error":{"code":404,"message":"model not found","type":"not_found"}}'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl_stats.TestParseLoadStart test_modelctl_stats.TestParseLoadSuccess test_modelctl_stats.TestParseLoadFailure -v`
Expected: FAIL with `AttributeError: module 'modelctl_stats' has no attribute 'parse_load_start'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to modelctl_stats.py, near the top after the schema block
import re

_RE_LOAD_START = re.compile(
    r"I srv\s+ensure_model: model name=(?P<profile>\S+) is not loaded, loading\.\.\.")
_RE_LOAD_SUCCESS = re.compile(
    r"^\[(?P<port>\d+)\] [\d.]+ I srv\s+llama_server: model loaded$")
_RE_LOAD_FAILURE = re.compile(
    r'got exception: (?P<detail>\{.*"message":"model name=(?P<profile>\S+) '
    r'failed to load".*\})')


def parse_load_start(message: str) -> str | None:
    """Return the profile name from an 'ensure_model: ... loading...' line,
    or None if the message isn't one."""
    m = _RE_LOAD_START.search(message)
    return m.group("profile") if m else None


def parse_load_success(message: str) -> int | None:
    """Return the port from a '[PORT] ... model loaded' line, or None."""
    m = _RE_LOAD_SUCCESS.match(message)
    return int(m.group("port")) if m else None


def parse_load_failure(message: str) -> tuple[str, str] | None:
    """Return (profile, raw_detail) from a 'failed to load' exception line,
    or None."""
    m = _RE_LOAD_FAILURE.search(message)
    return (m.group("profile"), m.group("detail")) if m else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_modelctl_stats -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelctl_stats.py test_modelctl_stats.py
git commit -m "stats: add load-lifecycle journal line parsers"
```

---

### Task 3: Request-timing line parsers

**Files:**
- Modify: `modelctl_stats.py`
- Modify: `test_modelctl_stats.py`

Each completed request produces three `slot print_timing` lines sharing the
same `[port]` and `task N`: prompt eval, generation eval, and total. These
parsers extract each independently; a later task groups them by
`(port, task)` into one `request_complete` event.

- [ ] **Step 1: Write the failing test**

```python
# add to test_modelctl_stats.py
class TestParsePromptEvalLine(unittest.TestCase):
    def test_matches(self):
        msg = ("[52035] 0.10.612.795 I slot print_timing: id  0 | task 0 | "
               "prompt eval time =     824.79 ms /    11 tokens "
               "(   74.98 ms per token,    13.34 tokens per second)")
        r = modelctl_stats.parse_prompt_eval_line(msg)
        self.assertEqual(r, {"port": 52035, "task": 0, "tokens": 11, "tps": 13.34})

    def test_no_match_on_gen_eval_line(self):
        msg = ("[52035] 0.10.612.797 I slot print_timing: id  0 | task 0 | "
               "       eval time =     100.31 ms /     5 tokens "
               "(   20.06 ms per token,    49.84 tokens per second)")
        self.assertIsNone(modelctl_stats.parse_prompt_eval_line(msg))


class TestParseGenEvalLine(unittest.TestCase):
    def test_matches(self):
        msg = ("[52035] 0.10.612.797 I slot print_timing: id  0 | task 0 | "
               "       eval time =     100.31 ms /     5 tokens "
               "(   20.06 ms per token,    49.84 tokens per second)")
        r = modelctl_stats.parse_gen_eval_line(msg)
        self.assertEqual(r, {"port": 52035, "task": 0, "tokens": 5, "tps": 49.84})

    def test_no_match_on_prompt_eval_line(self):
        msg = ("[52035] 0.10.612.795 I slot print_timing: id  0 | task 0 | "
               "prompt eval time =     824.79 ms /    11 tokens "
               "(   74.98 ms per token,    13.34 tokens per second)")
        self.assertIsNone(modelctl_stats.parse_gen_eval_line(msg))


class TestParseTotalTimeLine(unittest.TestCase):
    def test_matches(self):
        msg = ("[52035] 0.10.612.798 I slot print_timing: id  0 | task 0 | "
               "      total time =     925.10 ms /    16 tokens")
        r = modelctl_stats.parse_total_time_line(msg)
        self.assertEqual(r, {"port": 52035, "task": 0, "duration_ms": 925.10,
                              "tokens": 16})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl_stats.TestParsePromptEvalLine test_modelctl_stats.TestParseGenEvalLine test_modelctl_stats.TestParseTotalTimeLine -v`
Expected: FAIL with `AttributeError: module 'modelctl_stats' has no attribute 'parse_prompt_eval_line'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to modelctl_stats.py
_RE_PROMPT_EVAL = re.compile(
    r"^\[(?P<port>\d+)\] [\d.]+ I slot print_timing: id\s+\d+ \| "
    r"task (?P<task>\d+) \| prompt eval time =\s+[\d.]+ ms /\s+"
    r"(?P<tokens>\d+) tokens \(\s*[\d.]+ ms per token,\s*"
    r"(?P<tps>[\d.]+) tokens per second\)")
_RE_GEN_EVAL = re.compile(
    r"^\[(?P<port>\d+)\] [\d.]+ I slot print_timing: id\s+\d+ \| "
    r"task (?P<task>\d+) \|\s+(?<!prompt )eval time =\s+[\d.]+ ms /\s+"
    r"(?P<tokens>\d+) tokens \(\s*[\d.]+ ms per token,\s*"
    r"(?P<tps>[\d.]+) tokens per second\)")
_RE_TOTAL_TIME = re.compile(
    r"^\[(?P<port>\d+)\] [\d.]+ I slot print_timing: id\s+\d+ \| "
    r"task (?P<task>\d+) \|\s+total time =\s+(?P<ms>[\d.]+) ms /\s+"
    r"(?P<tokens>\d+) tokens$")


def parse_prompt_eval_line(message: str) -> dict | None:
    m = _RE_PROMPT_EVAL.match(message)
    if not m:
        return None
    return {"port": int(m.group("port")), "task": int(m.group("task")),
            "tokens": int(m.group("tokens")), "tps": float(m.group("tps"))}


def parse_gen_eval_line(message: str) -> dict | None:
    m = _RE_GEN_EVAL.match(message)
    if not m:
        return None
    return {"port": int(m.group("port")), "task": int(m.group("task")),
            "tokens": int(m.group("tokens")), "tps": float(m.group("tps"))}


def parse_total_time_line(message: str) -> dict | None:
    m = _RE_TOTAL_TIME.match(message)
    if not m:
        return None
    return {"port": int(m.group("port")), "task": int(m.group("task")),
            "duration_ms": float(m.group("ms")), "tokens": int(m.group("tokens"))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_modelctl_stats -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelctl_stats.py test_modelctl_stats.py
git commit -m "stats: add request-timing journal line parsers"
```

---

### Task 4: Request-error and service-crash line parsers

**Files:**
- Modify: `modelctl_stats.py`
- Modify: `test_modelctl_stats.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_modelctl_stats.py
class TestParseCancelTask(unittest.TestCase):
    def test_matches(self):
        msg = ("[60305] 47.06.408.543 W srv          stop: cancel task, "
               "id_task = 14396")
        self.assertEqual(modelctl_stats.parse_cancel_task_line(msg),
                         {"port": 60305, "task": 14396})

    def test_no_match(self):
        self.assertIsNone(modelctl_stats.parse_cancel_task_line(
            "[60305] 47.06.408.543 W srv    load_model: something else"))


class TestParseHttpClientError(unittest.TestCase):
    def test_matches_unbracketed_line(self):
        msg = ("48.15.121.590 E srv    operator(): http client error: "
               "Failed to read connection")
        r = modelctl_stats.parse_http_client_error_line(msg)
        self.assertEqual(r, {"port": None, "detail": "Failed to read connection"})

    def test_matches_bracketed_line(self):
        msg = ("[60305] 48.15.121.590 E srv    operator(): http client error: "
               "Connection handling canceled")
        r = modelctl_stats.parse_http_client_error_line(msg)
        self.assertEqual(r, {"port": 60305, "detail": "Connection handling canceled"})

    def test_no_match(self):
        self.assertIsNone(modelctl_stats.parse_http_client_error_line(
            "0.00.133.295 I srv  llama_server: listening on http://0.0.0.0:7071"))


class TestParseOomKill(unittest.TestCase):
    def test_matches(self):
        msg = "llama-router.service: Failed with result 'oom-kill'."
        self.assertEqual(modelctl_stats.parse_oom_kill_line(msg), "oom-kill")

    def test_no_match_on_other_unit(self):
        self.assertIsNone(modelctl_stats.parse_oom_kill_line(
            "some-other.service: Failed with result 'exit-code'."))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl_stats.TestParseCancelTask test_modelctl_stats.TestParseHttpClientError test_modelctl_stats.TestParseOomKill -v`
Expected: FAIL with `AttributeError: module 'modelctl_stats' has no attribute 'parse_cancel_task_line'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to modelctl_stats.py
_RE_CANCEL_TASK = re.compile(
    r"^\[(?P<port>\d+)\] [\d.]+ W srv\s+stop: cancel task, "
    r"id_task = (?P<task>\d+)$")
_RE_HTTP_CLIENT_ERROR = re.compile(
    r"^(?:\[(?P<port>\d+)\] )?[\d.]+ \w srv\s+operator\(\): "
    r"http client error: (?P<detail>.+)$")
_RE_OOM_KILL = re.compile(
    r"^llama-router\.service: Failed with result '(?P<reason>[\w-]+)'\.$")


def parse_cancel_task_line(message: str) -> dict | None:
    m = _RE_CANCEL_TASK.match(message)
    if not m:
        return None
    return {"port": int(m.group("port")), "task": int(m.group("task"))}


def parse_http_client_error_line(message: str) -> dict | None:
    m = _RE_HTTP_CLIENT_ERROR.match(message)
    if not m:
        return None
    port = m.group("port")
    return {"port": int(port) if port else None, "detail": m.group("detail")}


def parse_oom_kill_line(message: str) -> str | None:
    m = _RE_OOM_KILL.match(message)
    return m.group("reason") if m else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_modelctl_stats -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelctl_stats.py test_modelctl_stats.py
git commit -m "stats: add request-error and service-crash journal line parsers"
```

---

### Task 5: pending-loads and pid-map persistence helpers

**Files:**
- Modify: `modelctl_stats.py`
- Modify: `test_modelctl_stats.py`

These operate directly on an open `sqlite3.Connection`, wrapping the
`pending_loads`/`pid_map` tables. `sync_journal` (Task 6) is the only
caller.

- [ ] **Step 1: Write the failing test**

```python
# add to test_modelctl_stats.py
class TestPendingLoads(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        modelctl_stats.ensure_schema(self.conn)

    def test_record_and_pop(self):
        modelctl_stats.record_pending_load(self.conn, "profile-a", 100.0)
        started = modelctl_stats.pop_pending_load(self.conn, "profile-a")
        self.assertEqual(started, 100.0)
        # popped once, gone now
        self.assertIsNone(modelctl_stats.pop_pending_load(self.conn, "profile-a"))

    def test_pop_missing_profile_returns_none(self):
        self.assertIsNone(modelctl_stats.pop_pending_load(self.conn, "nope"))

    def test_record_overwrites_existing(self):
        modelctl_stats.record_pending_load(self.conn, "profile-a", 100.0)
        modelctl_stats.record_pending_load(self.conn, "profile-a", 200.0)
        self.assertEqual(
            modelctl_stats.pop_pending_load(self.conn, "profile-a"), 200.0)


class TestPidMap(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        modelctl_stats.ensure_schema(self.conn)

    def test_record_and_lookup(self):
        modelctl_stats.record_pid_mapping(self.conn, "boot-1", 52035,
                                           "profile-a", 100.0)
        self.assertEqual(
            modelctl_stats.lookup_profile_for_pid(self.conn, "boot-1", 52035),
            "profile-a")

    def test_lookup_unknown_pid_returns_none(self):
        self.assertIsNone(
            modelctl_stats.lookup_profile_for_pid(self.conn, "boot-1", 9999))

    def test_different_boot_ids_isolated(self):
        modelctl_stats.record_pid_mapping(self.conn, "boot-1", 52035,
                                           "profile-a", 100.0)
        self.assertIsNone(
            modelctl_stats.lookup_profile_for_pid(self.conn, "boot-2", 52035))

    def test_port_reuse_overwrites_mapping(self):
        modelctl_stats.record_pid_mapping(self.conn, "boot-1", 52035,
                                           "profile-a", 100.0)
        modelctl_stats.record_pid_mapping(self.conn, "boot-1", 52035,
                                           "profile-b", 200.0)
        self.assertEqual(
            modelctl_stats.lookup_profile_for_pid(self.conn, "boot-1", 52035),
            "profile-b")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl_stats.TestPendingLoads test_modelctl_stats.TestPidMap -v`
Expected: FAIL with `AttributeError: module 'modelctl_stats' has no attribute 'record_pending_load'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to modelctl_stats.py
def record_pending_load(conn: sqlite3.Connection, profile: str, ts: float) -> None:
    conn.execute(
        "INSERT INTO pending_loads (profile, started_at) VALUES (?, ?) "
        "ON CONFLICT(profile) DO UPDATE SET started_at = excluded.started_at",
        (profile, ts))


def pop_pending_load(conn: sqlite3.Connection, profile: str) -> float | None:
    row = conn.execute(
        "SELECT started_at FROM pending_loads WHERE profile = ?",
        (profile,)).fetchone()
    if row is None:
        return None
    conn.execute("DELETE FROM pending_loads WHERE profile = ?", (profile,))
    return row[0]


def record_pid_mapping(conn: sqlite3.Connection, boot_id: str, port: int,
                        profile: str, ts: float) -> None:
    conn.execute(
        "INSERT INTO pid_map (boot_id, port, profile, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(boot_id, port) DO UPDATE SET "
        "profile = excluded.profile, updated_at = excluded.updated_at",
        (boot_id, port, profile, ts))


def lookup_profile_for_pid(conn: sqlite3.Connection, boot_id: str,
                            port: int) -> str | None:
    row = conn.execute(
        "SELECT profile FROM pid_map WHERE boot_id = ? AND port = ?",
        (boot_id, port)).fetchone()
    return row[0] if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_modelctl_stats -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelctl_stats.py test_modelctl_stats.py
git commit -m "stats: add pending-loads and pid-map persistence helpers"
```

---

### Task 6: `sync_journal` orchestration

**Files:**
- Modify: `modelctl_stats.py`
- Modify: `test_modelctl_stats.py`

This is the one function that shells out (`journalctl`) and touches the
filesystem lock. It reads `sync_state.cursor`, fetches new journal entries
via `journalctl -o json --after-cursor=<cursor>` (or a full read if no
cursor is stored yet), dispatches each entry's `MESSAGE` through the Task
2-4 parsers, groups the three per-task timing lines into one
`request_complete` event each, and commits everything (new events + the
new cursor) in a single transaction. An `fcntl.flock` on
`<db_path>.lock` prevents two overlapping syncs from racing on
`pending_loads`/`pid_map`.

- [ ] **Step 1: Write the failing test**

```python
# add to test_modelctl_stats.py
from unittest import mock


def _journal_entry(message, ts_us, boot_id="boot-1", cursor="c1"):
    return {"MESSAGE": message, "__REALTIME_TIMESTAMP": str(ts_us),
            "_BOOT_ID": boot_id, "__CURSOR": cursor}


class TestSyncJournal(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "stats.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _run_sync(self, entries):
        lines = "\n".join(json.dumps(e) for e in entries)
        fake_result = mock.Mock(returncode=0, stdout=lines, stderr="")
        with mock.patch.object(modelctl_stats.subprocess, "run",
                                return_value=fake_result) as mock_run:
            summary = modelctl_stats.sync_journal(self.db_path)
        return summary, mock_run

    def test_first_sync_has_no_after_cursor_flag(self):
        _, mock_run = self._run_sync([])
        args = mock_run.call_args[0][0]
        self.assertNotIn("--after-cursor", " ".join(args))

    def test_load_success_produces_event_with_duration(self):
        entries = [
            _journal_entry(
                "1.00.000.000 I srv  ensure_model: model name=profile-a "
                "is not loaded, loading...", ts_us=1_000_000_000, cursor="c1"),
            _journal_entry(
                "[52035] 0.09.675.693 I srv  llama_server: model loaded",
                ts_us=1_010_000_000, cursor="c2"),
        ]
        summary, _ = self._run_sync(entries)
        self.assertEqual(summary["load_success"], 1)
        conn = modelctl_stats.connect(self.db_path)
        row = conn.execute(
            "SELECT profile, event_type, duration_ms FROM events").fetchone()
        conn.close()
        self.assertEqual(row, ("profile-a", "load_success", 10_000.0))

    def test_load_failure_produces_event(self):
        entries = [
            _journal_entry(
                "1.00.000.000 I srv  ensure_model: model name=profile-a "
                "is not loaded, loading...", ts_us=1_000_000_000, cursor="c1"),
            _journal_entry(
                '0.05.964.076 W srv    operator(): got exception: '
                '{"error":{"code":500,"message":"model name=profile-a '
                'failed to load","type":"server_error"}}',
                ts_us=1_005_000_000, cursor="c2"),
        ]
        summary, _ = self._run_sync(entries)
        self.assertEqual(summary["load_failure"], 1)

    def test_request_lines_grouped_into_one_event(self):
        entries = [
            _journal_entry(
                "1.00.000.000 I srv  ensure_model: model name=profile-a "
                "is not loaded, loading...", ts_us=1_000_000_000, cursor="c1"),
            _journal_entry(
                "[52035] 0.09.675.693 I srv  llama_server: model loaded",
                ts_us=1_010_000_000, cursor="c2"),
            _journal_entry(
                "[52035] 0.10.612.795 I slot print_timing: id  0 | task 0 | "
                "prompt eval time =     824.79 ms /    11 tokens "
                "(   74.98 ms per token,    13.34 tokens per second)",
                ts_us=1_020_000_000, cursor="c3"),
            _journal_entry(
                "[52035] 0.10.612.797 I slot print_timing: id  0 | task 0 | "
                "       eval time =     100.31 ms /     5 tokens "
                "(   20.06 ms per token,    49.84 tokens per second)",
                ts_us=1_020_100_000, cursor="c4"),
            _journal_entry(
                "[52035] 0.10.612.798 I slot print_timing: id  0 | task 0 | "
                "      total time =     925.10 ms /    16 tokens",
                ts_us=1_020_200_000, cursor="c5"),
        ]
        summary, _ = self._run_sync(entries)
        self.assertEqual(summary["request_complete"], 1)
        conn = modelctl_stats.connect(self.db_path)
        row = conn.execute(
            "SELECT profile, prompt_tokens, gen_tokens, prompt_tps, gen_tps, "
            "duration_ms FROM events WHERE event_type = 'request_complete'"
        ).fetchone()
        conn.close()
        self.assertEqual(row, ("profile-a", 11, 5, 13.34, 49.84, 925.10))

    def test_second_sync_uses_stored_cursor(self):
        self._run_sync([_journal_entry(
            "0.00.000.000 I srv unrelated line", ts_us=1, cursor="c1")])
        _, mock_run = self._run_sync([])
        args = mock_run.call_args[0][0]
        self.assertIn("--after-cursor=c1", args)

    def test_unattributed_request_error_has_null_profile(self):
        entries = [_journal_entry(
            "48.15.121.590 E srv    operator(): http client error: "
            "Failed to read connection", ts_us=1, cursor="c1")]
        summary, _ = self._run_sync(entries)
        self.assertEqual(summary["request_error"], 1)
        conn = modelctl_stats.connect(self.db_path)
        row = conn.execute(
            "SELECT profile FROM events WHERE event_type = 'request_error'"
        ).fetchone()
        conn.close()
        self.assertIsNone(row[0])

    def test_oom_kill_produces_unattributed_service_crash(self):
        entries = [_journal_entry(
            "llama-router.service: Failed with result 'oom-kill'.",
            ts_us=1, cursor="c1")]
        summary, _ = self._run_sync(entries)
        self.assertEqual(summary["service_crash"], 1)

    def test_concurrent_sync_raises_when_locked(self):
        import fcntl
        lock_path = self.db_path.parent / (self.db_path.name + ".lock")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            with self.assertRaises(modelctl_stats.SyncInProgressError):
                modelctl_stats.sync_journal(self.db_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl_stats.TestSyncJournal -v`
Expected: FAIL with `AttributeError: module 'modelctl_stats' has no attribute 'sync_journal'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to modelctl_stats.py, near the top
import fcntl
import json
import subprocess
import time

ROUTER_SERVICE_NAME = "llama-router.service"


class SyncInProgressError(Exception):
    """Raised when another sync_journal() call already holds the lock."""


def _get_cursor(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT cursor FROM sync_state WHERE id = 1").fetchone()
    return row[0] if row else None


def _set_cursor(conn: sqlite3.Connection, cursor: str) -> None:
    conn.execute(
        "INSERT INTO sync_state (id, cursor) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET cursor = excluded.cursor", (cursor,))


def _fetch_journal_entries(unit: str, after_cursor: str | None) -> list[dict]:
    cmd = ["journalctl", "--user", "-u", unit, "-o", "json", "--no-pager"]
    if after_cursor:
        cmd.append(f"--after-cursor={after_cursor}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def sync_journal(db_path: Path, unit: str = ROUTER_SERVICE_NAME) -> dict:
    """Parse new llama-router.service journal entries since the last sync
    into db_path, returning a summary count by event type. Raises
    SyncInProgressError if another sync already holds the lock."""
    lock_path = db_path.parent / (db_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        raise SyncInProgressError(f"another sync already holds {lock_path}")

    try:
        conn = connect(db_path)
        try:
            cursor = _get_cursor(conn)
            entries = _fetch_journal_entries(unit, cursor)

            summary = {"load_success": 0, "load_failure": 0,
                       "request_complete": 0, "request_error": 0,
                       "service_crash": 0}
            pending_tasks: dict[tuple[int, int], dict] = {}
            last_cursor = cursor

            for entry in entries:
                message = entry.get("MESSAGE", "")
                boot_id = entry.get("_BOOT_ID", "")
                ts = int(entry["__REALTIME_TIMESTAMP"]) / 1_000_000
                last_cursor = entry.get("__CURSOR", last_cursor)

                profile = parse_load_start(message)
                if profile:
                    record_pending_load(conn, profile, ts)
                    continue

                port = parse_load_success(message)
                if port is not None:
                    # Attribute to the most recently started pending load --
                    # see "Known limitations" in the design spec for the
                    # concurrent-different-model-load edge case this doesn't
                    # handle.
                    row = conn.execute(
                        "SELECT profile, started_at FROM pending_loads "
                        "ORDER BY started_at DESC LIMIT 1").fetchone()
                    if row:
                        load_profile, started_at = row
                        pop_pending_load(conn, load_profile)
                        record_pid_mapping(conn, boot_id, port, load_profile, ts)
                        conn.execute(
                            "INSERT INTO events (ts, profile, event_type, "
                            "duration_ms) VALUES (?, ?, 'load_success', ?)",
                            (ts, load_profile, (ts - started_at) * 1000))
                        summary["load_success"] += 1
                    continue

                failure = parse_load_failure(message)
                if failure:
                    load_profile, detail = failure
                    pop_pending_load(conn, load_profile)
                    conn.execute(
                        "INSERT INTO events (ts, profile, event_type, detail) "
                        "VALUES (?, ?, 'load_failure', ?)",
                        (ts, load_profile, detail))
                    summary["load_failure"] += 1
                    continue

                prompt = parse_prompt_eval_line(message)
                if prompt:
                    key = (prompt["port"], prompt["task"])
                    pending_tasks.setdefault(key, {})
                    pending_tasks[key]["prompt_tokens"] = prompt["tokens"]
                    pending_tasks[key]["prompt_tps"] = prompt["tps"]
                    continue

                gen = parse_gen_eval_line(message)
                if gen:
                    key = (gen["port"], gen["task"])
                    pending_tasks.setdefault(key, {})
                    pending_tasks[key]["gen_tokens"] = gen["tokens"]
                    pending_tasks[key]["gen_tps"] = gen["tps"]
                    continue

                total = parse_total_time_line(message)
                if total:
                    key = (total["port"], total["task"])
                    acc = pending_tasks.pop(key, {})
                    request_profile = lookup_profile_for_pid(
                        conn, boot_id, total["port"])
                    conn.execute(
                        "INSERT INTO events (ts, profile, event_type, "
                        "duration_ms, prompt_tokens, gen_tokens, prompt_tps, "
                        "gen_tps) VALUES (?, ?, 'request_complete', ?, ?, ?, ?, ?)",
                        (ts, request_profile, total["duration_ms"],
                         acc.get("prompt_tokens"), acc.get("gen_tokens"),
                         acc.get("prompt_tps"), acc.get("gen_tps")))
                    summary["request_complete"] += 1
                    continue

                cancel = parse_cancel_task_line(message)
                if cancel:
                    request_profile = lookup_profile_for_pid(
                        conn, boot_id, cancel["port"])
                    conn.execute(
                        "INSERT INTO events (ts, profile, event_type, detail) "
                        "VALUES (?, ?, 'request_error', ?)",
                        (ts, request_profile, "cancel task"))
                    summary["request_error"] += 1
                    continue

                http_err = parse_http_client_error_line(message)
                if http_err:
                    request_profile = (
                        lookup_profile_for_pid(conn, boot_id, http_err["port"])
                        if http_err["port"] is not None else None)
                    conn.execute(
                        "INSERT INTO events (ts, profile, event_type, detail) "
                        "VALUES (?, ?, 'request_error', ?)",
                        (ts, request_profile, http_err["detail"]))
                    summary["request_error"] += 1
                    continue

                oom_reason = parse_oom_kill_line(message)
                if oom_reason:
                    conn.execute(
                        "INSERT INTO events (ts, profile, event_type, detail) "
                        "VALUES (?, NULL, 'service_crash', ?)",
                        (ts, oom_reason))
                    summary["service_crash"] += 1
                    continue

            if last_cursor:
                _set_cursor(conn, last_cursor)
            conn.commit()
            return summary
        finally:
            conn.close()
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_modelctl_stats -v`
Expected: PASS (all TestSyncJournal cases plus everything from Tasks 1-5)

- [ ] **Step 5: Commit**

```bash
git add modelctl_stats.py test_modelctl_stats.py
git commit -m "stats: add sync_journal orchestration (journald -> stats.db)"
```

---

### Task 7: `query_stats` aggregation

**Files:**
- Modify: `modelctl_stats.py`
- Modify: `test_modelctl_stats.py`

`query_stats` reads `events`, groups by profile, and joins each profile's
current `config.ctx` from its JSON file (path: `profiles_dir /
f"{profile}.json"`) for filtering/display. Unattributed events
(`profile IS NULL`) are summarized separately, not folded into any
profile's row.

- [ ] **Step 1: Write the failing test**

```python
# add to test_modelctl_stats.py
class TestQueryStats(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "stats.db"
        self.profiles_dir = Path(self._tmp.name) / "profiles"
        self.profiles_dir.mkdir()
        self.conn = modelctl_stats.connect(self.db_path)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def _write_profile(self, name, ctx):
        (self.profiles_dir / f"{name}.json").write_text(
            json.dumps({"name": name, "config": {"ctx": ctx}}))

    def _insert(self, ts, profile, event_type, **kw):
        cols = ["ts", "profile", "event_type"] + list(kw.keys())
        placeholders = ", ".join("?" for _ in cols)
        self.conn.execute(
            f"INSERT INTO events ({', '.join(cols)}) VALUES ({placeholders})",
            [ts, profile, event_type] + list(kw.values()))
        self.conn.commit()

    def test_aggregates_per_profile(self):
        self._write_profile("profile-a", 32768)
        self._insert(1, "profile-a", "load_success", duration_ms=1000)
        self._insert(2, "profile-a", "request_complete", gen_tps=10.0,
                     prompt_tps=100.0)
        self._insert(3, "profile-a", "request_complete", gen_tps=20.0,
                     prompt_tps=200.0)
        self._insert(4, "profile-a", "request_error")

        rows = modelctl_stats.query_stats(self.db_path, self.profiles_dir)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["profile"], "profile-a")
        self.assertEqual(row["ctx"], 32768)
        self.assertEqual(row["loads"], 1)
        self.assertEqual(row["load_failures"], 0)
        self.assertEqual(row["avg_load_ms"], 1000)
        self.assertEqual(row["requests"], 3)
        self.assertEqual(row["request_errors"], 1)
        self.assertEqual(row["avg_gen_tps"], 15.0)
        self.assertEqual(row["avg_prompt_tps"], 150.0)

    def test_ctx_filter(self):
        self._write_profile("small-ctx", 8192)
        self._write_profile("big-ctx", 131072)
        self._insert(1, "small-ctx", "request_complete", gen_tps=1.0)
        self._insert(1, "big-ctx", "request_complete", gen_tps=1.0)

        rows = modelctl_stats.query_stats(self.db_path, self.profiles_dir,
                                          ctx_min=100000)
        self.assertEqual([r["profile"] for r in rows], ["big-ctx"])

    def test_profile_filter(self):
        self._write_profile("profile-a", 8192)
        self._write_profile("profile-b", 8192)
        self._insert(1, "profile-a", "request_complete", gen_tps=1.0)
        self._insert(1, "profile-b", "request_complete", gen_tps=1.0)

        rows = modelctl_stats.query_stats(self.db_path, self.profiles_dir,
                                          profile="profile-a")
        self.assertEqual([r["profile"] for r in rows], ["profile-a"])

    def test_missing_profile_json_shows_unknown_ctx(self):
        self._insert(1, "deleted-profile", "request_complete", gen_tps=1.0)
        rows = modelctl_stats.query_stats(self.db_path, self.profiles_dir)
        self.assertEqual(rows[0]["ctx"], None)

    def test_unattributed_events_summarized_separately(self):
        self._write_profile("profile-a", 8192)
        self._insert(1, "profile-a", "request_complete", gen_tps=1.0)
        self._insert(2, None, "service_crash", detail="oom-kill")
        self._insert(3, None, "request_error", detail="Failed to read connection")

        rows = modelctl_stats.query_stats(self.db_path, self.profiles_dir)
        self.assertEqual(len(rows), 1)  # unattributed isn't a profile row
        unattributed = modelctl_stats.unattributed_summary(self.db_path)
        self.assertEqual(unattributed["service_crash"], 1)
        self.assertEqual(unattributed["request_error"], 1)

    def test_since_filter(self):
        self._write_profile("profile-a", 8192)
        self._insert(100, "profile-a", "request_complete", gen_tps=1.0)
        self._insert(1_000_000, "profile-a", "request_complete", gen_tps=2.0)

        rows = modelctl_stats.query_stats(self.db_path, self.profiles_dir,
                                          since_ts=500_000)
        self.assertEqual(rows[0]["requests"], 1)
        self.assertEqual(rows[0]["avg_gen_tps"], 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl_stats.TestQueryStats -v`
Expected: FAIL with `AttributeError: module 'modelctl_stats' has no attribute 'query_stats'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to modelctl_stats.py
def _profile_ctx(profiles_dir: Path, profile: str) -> int | None:
    path = profiles_dir / f"{profile}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ctx = data.get("config", {}).get("ctx")
    return int(ctx) if ctx is not None else None


def query_stats(db_path: Path, profiles_dir: Path, profile: str | None = None,
                 ctx_min: int | None = None, ctx_max: int | None = None,
                 since_ts: float | None = None) -> list[dict]:
    conn = connect(db_path)
    try:
        where = ["profile IS NOT NULL"]
        params: list = []
        if profile:
            where.append("profile = ?")
            params.append(profile)
        if since_ts is not None:
            where.append("ts >= ?")
            params.append(since_ts)
        where_sql = " AND ".join(where)

        rows = conn.execute(f"""
            SELECT
                profile,
                SUM(event_type = 'load_success') AS loads,
                SUM(event_type = 'load_failure') AS load_failures,
                AVG(CASE WHEN event_type = 'load_success' THEN duration_ms END)
                    AS avg_load_ms,
                SUM(event_type IN ('request_complete', 'request_error'))
                    AS requests,
                SUM(event_type = 'request_error') AS request_errors,
                AVG(CASE WHEN event_type = 'request_complete'
                    THEN gen_tps END) AS avg_gen_tps,
                AVG(CASE WHEN event_type = 'request_complete'
                    THEN prompt_tps END) AS avg_prompt_tps
            FROM events
            WHERE {where_sql}
            GROUP BY profile
            ORDER BY profile
        """, params).fetchall()

        results = []
        for row in rows:
            (name, loads, load_failures, avg_load_ms, requests,
             request_errors, avg_gen_tps, avg_prompt_tps) = row
            # loads also counts load_failure events themselves toward the
            # attempted-load total (loads here is load_success only, so add
            # failures back in for "attempts")
            ctx = _profile_ctx(profiles_dir, name)
            if ctx_min is not None and (ctx is None or ctx < ctx_min):
                continue
            if ctx_max is not None and (ctx is None or ctx > ctx_max):
                continue
            results.append({
                "profile": name, "ctx": ctx,
                "loads": loads + load_failures, "load_failures": load_failures,
                "avg_load_ms": avg_load_ms,
                "requests": requests, "request_errors": request_errors,
                "avg_gen_tps": avg_gen_tps, "avg_prompt_tps": avg_prompt_tps,
            })
        return results
    finally:
        conn.close()


def unattributed_summary(db_path: Path, since_ts: float | None = None) -> dict:
    conn = connect(db_path)
    try:
        where = ["profile IS NULL"]
        params: list = []
        if since_ts is not None:
            where.append("ts >= ?")
            params.append(since_ts)
        where_sql = " AND ".join(where)
        rows = conn.execute(
            f"SELECT event_type, COUNT(*) FROM events WHERE {where_sql} "
            f"GROUP BY event_type", params).fetchall()
        return dict(rows)
    finally:
        conn.close()
```

Note: `test_aggregates_per_profile` expects `row["loads"] == 1` with only a
`load_success` inserted (no failure) — the implementation's `loads =
loads + load_failures` correctly yields `1 + 0 = 1` for that case, and
`test_missing_profile_json_shows_unknown_ctx`/`test_unattributed_events...`
don't insert any load events so `loads` is `0 + 0 = 0`, not asserted
directly but consistent.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_modelctl_stats -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelctl_stats.py test_modelctl_stats.py
git commit -m "stats: add query_stats aggregation with ctx/profile/since filters"
```

---

### Task 8: `modelctl.py` CLI wiring — `stats sync` / `stats show`

**Files:**
- Modify: `modelctl.py` (add near `cmd_router_stats` around line 2254; add argparse wiring near the `router` subparser block around line 2442)
- Modify: `test_modelctl.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_modelctl.py
class TestCmdStatsSync(unittest.TestCase):
    def test_prints_summary_counts(self):
        fake_summary = {"load_success": 2, "load_failure": 0,
                         "request_complete": 43, "request_error": 2,
                         "service_crash": 0}
        args = argparse.Namespace()
        with mock.patch.object(modelctl.modelctl_stats, "sync_journal",
                                return_value=fake_summary) as mock_sync:
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                modelctl.cmd_stats_sync(args)
        mock_sync.assert_called_once_with(modelctl.STATE_DIR / "stats.db")
        self.assertIn("2 loads", out.getvalue())
        self.assertIn("2 request errors", out.getvalue())

    def test_reports_error_on_sync_in_progress(self):
        args = argparse.Namespace()
        with mock.patch.object(
                modelctl.modelctl_stats, "sync_journal",
                side_effect=modelctl.modelctl_stats.SyncInProgressError("busy")):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                with self.assertRaises(SystemExit) as cm:
                    modelctl.cmd_stats_sync(args)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("busy", out.getvalue())


class TestCmdStatsShow(unittest.TestCase):
    def test_prints_table_and_footer(self):
        fake_rows = [{
            "profile": "profile-a", "ctx": 32768, "loads": 1,
            "load_failures": 0, "avg_load_ms": 1000.0, "requests": 2,
            "request_errors": 0, "avg_gen_tps": 15.0, "avg_prompt_tps": 150.0,
        }]
        args = argparse.Namespace(profile=None, ctx=None, ctx_min=None,
                                   ctx_max=None, since=None)
        with mock.patch.object(modelctl.modelctl_stats, "query_stats",
                                return_value=fake_rows), \
             mock.patch.object(modelctl.modelctl_stats, "unattributed_summary",
                                return_value={}):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                modelctl.cmd_stats_show(args)
        self.assertIn("profile-a", out.getvalue())
        self.assertIn("32768", out.getvalue())

    def test_prints_unattributed_footer_when_present(self):
        args = argparse.Namespace(profile=None, ctx=None, ctx_min=None,
                                   ctx_max=None, since=None)
        with mock.patch.object(modelctl.modelctl_stats, "query_stats",
                                return_value=[]), \
             mock.patch.object(modelctl.modelctl_stats, "unattributed_summary",
                                return_value={"service_crash": 2,
                                               "request_error": 1}):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                modelctl.cmd_stats_show(args)
        self.assertIn("2 service crashes", out.getvalue())
        self.assertIn("1 unattributed request error", out.getvalue())

    def test_ctx_exact_sets_min_and_max(self):
        args = argparse.Namespace(profile=None, ctx=131072, ctx_min=None,
                                   ctx_max=None, since=None)
        with mock.patch.object(modelctl.modelctl_stats, "query_stats",
                                return_value=[]) as mock_query, \
             mock.patch.object(modelctl.modelctl_stats, "unattributed_summary",
                                return_value={}):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                modelctl.cmd_stats_show(args)
        _, kwargs = mock_query.call_args
        self.assertEqual(kwargs["ctx_min"], 131072)
        self.assertEqual(kwargs["ctx_max"], 131072)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_modelctl.TestCmdStatsSync test_modelctl.TestCmdStatsShow -v`
Expected: FAIL with `AttributeError: module 'modelctl' has no attribute 'cmd_stats_sync'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to modelctl.py, imports section near "import modelctl_vram"
import modelctl_stats
```

```python
# add to modelctl.py, near cmd_router_stats (around line 2254)
def _parse_since(value: str | None) -> float | None:
    """Parse a relative duration like '24h' or '7d' into a unix timestamp
    cutoff (now - duration), or None if value is None."""
    if value is None:
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    m = re.match(r"^(\d+)([smhd])$", value)
    if not m:
        print(f"Error: --since must look like '24h' or '7d', got {value!r}")
        sys.exit(1)
    amount, unit = m.groups()
    return time.time() - int(amount) * units[unit]


def cmd_stats_sync(args):
    db_path = STATE_DIR / "stats.db"
    try:
        summary = modelctl_stats.sync_journal(db_path)
    except modelctl_stats.SyncInProgressError as e:
        print(f"Error: {e}")
        sys.exit(1)
    total_new = sum(summary.values())
    print(f"Synced {total_new} new events "
          f"({summary['load_success']} loads, "
          f"{summary['load_failure']} load failures, "
          f"{summary['request_complete']} requests, "
          f"{summary['request_error']} request errors).")


def cmd_stats_show(args):
    db_path = STATE_DIR / "stats.db"
    ctx_min = args.ctx_min
    ctx_max = args.ctx_max
    if args.ctx is not None:
        ctx_min = ctx_max = args.ctx
    since_ts = _parse_since(args.since)

    rows = modelctl_stats.query_stats(
        db_path, PROFILES_DIR, profile=args.profile,
        ctx_min=ctx_min, ctx_max=ctx_max, since_ts=since_ts)

    if not rows:
        print("No stats recorded yet -- run `modelctl stats sync` first.")
    else:
        print(f"{'PROFILE':<28} {'CTX':>8} {'LOADS':>6} {'LOAD-FAIL':>9} "
              f"{'AVG LOAD':>9} {'REQS':>6} {'REQ-ERR':>8} "
              f"{'AVG GEN T/S':>12} {'AVG PROMPT T/S':>15}")
        for r in rows:
            ctx = r["ctx"] if r["ctx"] is not None else "?"
            avg_load = f"{r['avg_load_ms'] / 1000:.1f}s" if r["avg_load_ms"] else "-"
            avg_gen = f"{r['avg_gen_tps']:.1f}" if r["avg_gen_tps"] else "-"
            avg_prompt = f"{r['avg_prompt_tps']:.1f}" if r["avg_prompt_tps"] else "-"
            print(f"{r['profile']:<28} {ctx!s:>8} {r['loads']:>6} "
                  f"{r['load_failures']:>9} {avg_load:>9} {r['requests']:>6} "
                  f"{r['request_errors']:>8} {avg_gen:>12} {avg_prompt:>15}")

    unattributed = modelctl_stats.unattributed_summary(db_path, since_ts=since_ts)
    if unattributed:
        parts = []
        if unattributed.get("service_crash"):
            parts.append(f"{unattributed['service_crash']} service crashes "
                         f"(oom-kill)")
        if unattributed.get("request_error"):
            parts.append(f"{unattributed['request_error']} unattributed "
                         f"request error"
                         f"{'s' if unattributed['request_error'] != 1 else ''}")
        if parts:
            print()
            print(", ".join(parts) + " -- not tied to a specific profile.")
```

```python
# add to modelctl.py's argparse wiring, right after the router subcommand
# block (after p_router_unload.set_defaults(func=cmd_router_unload), before
# "return parser")
    p_stats = sub.add_parser("stats", help="per-profile performance history "
                              "(load times, throughput, error rates)")
    stats_sub = p_stats.add_subparsers(dest="stats_command", required=True)

    p_stats_sync = stats_sub.add_parser(
        "sync", help="parse new llama-router.service journal entries into stats.db")
    p_stats_sync.set_defaults(func=cmd_stats_sync)

    p_stats_show = stats_sub.add_parser(
        "show", help="show aggregated performance stats per profile")
    p_stats_show.add_argument("--profile", help="show only this profile")
    p_stats_show.add_argument("--ctx", type=int,
                               help="show only profiles configured with exactly this ctx")
    p_stats_show.add_argument("--ctx-min", type=int, dest="ctx_min",
                               help="show only profiles configured with ctx >= this")
    p_stats_show.add_argument("--ctx-max", type=int, dest="ctx_max",
                               help="show only profiles configured with ctx <= this")
    p_stats_show.add_argument("--since",
                               help="only include events from the last N "
                                    "units, e.g. 24h, 7d")
    p_stats_show.set_defaults(func=cmd_stats_show)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_modelctl -v`
Expected: PASS (full suite, including the new stats CLI tests)

- [ ] **Step 5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "stats: wire up 'modelctl stats sync'/'stats show' CLI"
```

---

### Task 9: systemd timer files + README

**Files:**
- Create: `systemd/modelctl-stats-sync.service`
- Create: `systemd/modelctl-stats-sync.timer`
- Modify: `README.md`

- [ ] **Step 1: Create the service unit**

```ini
# systemd/modelctl-stats-sync.service
[Unit]
Description=modelctl stats sync (parse llama-router journal into stats.db)

[Service]
Type=oneshot
ExecStart=%h/.local/bin/modelctl stats sync
```

- [ ] **Step 2: Create the timer unit**

```ini
# systemd/modelctl-stats-sync.timer
[Unit]
Description=Periodic modelctl stats sync

[Timer]
OnUnitActiveSec=10min
Persistent=true

[Install]
WantedBy=timers.target
```

- [ ] **Step 3: Update README.md's components table and test-run line**

In [README.md](../../../README.md), add a row after the `modelctl_vram.py`
row (currently line 23):

```markdown
| [modelctl_stats.py](modelctl_stats.py) | Per-profile performance history: parses `llama-router.service`'s journal into `~/.local/share/modelctl/stats.db` (load times, throughput, error rates), cursor-synced so repeated syncs only process new lines. No `modelctl` import. Backs `modelctl stats sync`/`show`. |
```

And update the test-run line (currently line 26) to:

```markdown
Tests: `test_modelctl.py`, `test_modelctl_vram.py`, `test_modelctl_tui.py`, `test_modelctl_stats.py` (`python3 -m unittest test_modelctl test_modelctl_vram test_modelctl_tui test_modelctl_stats`).
```

- [ ] **Step 4: Add a "Performance stats" section to README.md**

Append near the end of README.md (after the existing "Profiles" section
or wherever the file's last section currently ends):

```markdown
## Performance stats

`modelctl stats sync` parses new `llama-router.service` journal entries
into `~/.local/share/modelctl/stats.db` -- load times, token throughput,
and error rates per profile. `modelctl stats show` reads them back,
filterable by `--profile`, `--ctx`/`--ctx-min`/`--ctx-max` (the profile's
configured context size, not per-request length), and `--since` (e.g.
`24h`, `7d`).

journald's default rotation means events can be lost if sync only runs
manually. To keep it current automatically, copy the unit files from
`systemd/` into `~/.config/systemd/user/` and enable the timer:

```bash
cp systemd/modelctl-stats-sync.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now modelctl-stats-sync.timer
```

See [docs/superpowers/specs/2026-07-02-model-perf-stats-design.md](docs/superpowers/specs/2026-07-02-model-perf-stats-design.md)
for the full design, including known limitations (concurrent different-model
auto-loads, oom-kill attribution).
```

- [ ] **Step 5: Commit**

```bash
git add systemd/modelctl-stats-sync.service systemd/modelctl-stats-sync.timer README.md
git commit -m "stats: add systemd timer units and README documentation"
```

---

### Task 10: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `python3 -m unittest test_modelctl test_modelctl_vram test_modelctl_tui test_modelctl_stats -v`
Expected: PASS, 0 failures, 0 errors

- [ ] **Step 2: Manually smoke-test against the real journal**

```bash
python3 -c "
from pathlib import Path
import modelctl_stats
print(modelctl_stats.sync_journal(Path('/tmp/smoke-stats.db')))
print(modelctl_stats.query_stats(Path('/tmp/smoke-stats.db'),
                                  Path.home() / '.local/share/modelctl/profiles'))
"
```

Expected: a summary dict with nonzero counts (this system's journal has
real history to parse), and a list of per-profile rows with plausible
`avg_gen_tps`/`avg_load_ms` values matching what `modelctl router stats`
shows for currently-loaded models. Delete `/tmp/smoke-stats.db` afterward
-- it's a scratch file, not the real `~/.local/share/modelctl/stats.db`.

- [ ] **Step 3: Run the real CLI end-to-end**

```bash
modelctl stats sync
modelctl stats show
```

Expected: sync prints a summary line; show prints a populated table (or
"No stats recorded yet" only if sync found zero events, which would be
surprising given this system's journal history).
