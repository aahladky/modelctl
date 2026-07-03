# Model performance stats logging — design

Date: 2026-07-02
Status: approved, not yet implemented

## Context

Today the only performance visibility is `modelctl router stats`
([modelctl.py:2254](../../../modelctl.py)), which reads each loaded model's
live Prometheus counters via the router's forwarded `/metrics` endpoint
([modelctl.py:2211](../../../modelctl.py)). Those counters are lifetime
averages *since that instance loaded* — they reset to zero on unload/evict
and on every router restart, and there's no history, no load-time tracking,
no error tracking, and nothing broken out per profile over time.

This spec adds a persisted, per-profile performance log: token throughput,
load times, and error rates, filterable by the profile's configured context
size. High-level by design — not a request-by-request trace tool.

The router auto-loads models directly from the first inbound API request
(`llama_server: starting server in router mode. models will be automatically
loaded on-demand`), with no modelctl invocation involved. Real usage happens
this way more often than through explicit `modelctl router load` calls, so
the data source has to be something that observes the router independently
of modelctl's own actions.

## Decisions made during brainstorming

- **Event scope: everything, any trigger** — auto-loads and direct API
  activity count, not just modelctl-initiated load/unload calls.
- **Collection: journald parsing, not a watcher.** A `modelctl stats sync`
  command parses new `llama-router.service` journal lines since the last
  sync and writes structured events to a local store. This matches the
  earlier decision not to build a modelctl watcher process for the
  evict-and-retry follow-on — sync is a short-lived invocation that reads,
  writes, and exits, not a long-running observer.
- **Sync is invoked by a systemd --user timer**, not left purely manual.
  journald has no custom retention configured (347M and growing, default
  rotation), so a sync that only runs when remembered risks losing events
  before they're ever read. The timer unit is a oneshot service + timer,
  same shape as a cron job — still no persistent process.
- **Storage: SQLite** at `~/.local/share/modelctl/stats.db` (same
  `STATE_DIR` as profiles). Filtering/aggregation (by profile, by ctx range,
  averages, error rates) is simple in SQL and `sqlite3` is stdlib — no new
  dependency.
- **Error definition: load failures *and* failed/aborted requests.** Load
  failures are clean (`model name=X failed to load`, names the profile
  directly). Request-level errors/cancellations are real but noisier —
  they're journald lines like `http client error: Failed to read connection`
  and `stop: cancel task`, tagged only by a `[pid]` prefix, not a profile
  name. These get attributed via the pid→profile correlation described
  below. Some can't be attributed at all (unbracketed lines, whole-service
  crashes) — those are recorded unattributed rather than dropped or guessed
  at.
- **Retention: keep forever in v1.** Event volume is load/unload/error/
  request-summary rows, not per-token — not worth pruning logic until it's
  actually a problem.
- **CLI surface: new `modelctl stats` subcommand family** (`sync`, `show`),
  kept separate from the existing live `modelctl router stats`, which
  answers a different question ("what's loaded right now") from this
  feature's ("how has each profile performed over time").
- **Context size filter = the profile's configured `config.ctx`**, read
  live from the profile's JSON at query time — not the actual per-request
  prompt length. It's a static, per-profile attribute you filter profiles
  by, not a dynamic bucket. (If a profile's `ctx` is edited later, historical
  events join against the *current* value, not what it was at the time —
  accepted simplification; profiles change rarely.)
- **Incremental read: journald cursor**, not a timestamp watermark or full
  replay. `journalctl -o json --after-cursor=<last_cursor>` only returns
  genuinely new lines and survives router restarts (the cursor is
  journal-global, not tied to a PID). The alternative of a timestamp
  watermark risks duplicate/skipped lines at second-resolution boundaries;
  full replay gets slower every sync for no benefit on a job that fires
  every few minutes.

## Architecture

New module **`modelctl_stats.py`**, stdlib-only, following the same
standalone-module convention as `modelctl_vram.py`
([README.md](../../../README.md)) — parsing and storage logic that doesn't
need to import `modelctl.py`, so it stays independently testable and usable.

Three journald signal families, keyed off the exact lines observed on this
system:

1. **Load lifecycle** (profile name directly in the line, no correlation
   needed):
   - Start: `srv ensure_model: model name=X is not loaded, loading...`
   - Success: the next `[PID] ... llama_server: model loaded` line —
     assumed to correspond to the most recently started pending load
     (router loads are observed to proceed one at a time; see Known
     limitations). This line's `[PID]` also seeds the pid→profile map.
   - Failure: `got exception: {"error":{...,"message":"model name=X failed
     to load",...}}`.
2. **Request completion** (pid-tagged, profile via pid map): the three
   `[PID] slot print_timing: id 0 | task N | ...` lines for a given task
   (`prompt eval time`, `eval time`, `total time`) are grouped by task id
   into one `request_complete` event carrying prompt/gen token counts and
   both throughput figures.
3. **Request errors** (pid-tagged where possible): `http client error: ...`
   and `stop: cancel task` lines. Bracketed lines resolve to a profile via
   the pid map; unbracketed ones (seen in practice — some `http client
   error` lines have no `[pid]` prefix) are recorded with `profile = NULL`.
4. **Service-level crash** (unattributable by nature): `systemd[...]:
   llama-router.service: Failed with result 'oom-kill'`. Recorded as its
   own `service_crash` event with `profile = NULL` — guessing which model
   was responsible would be fabricating data `stats show` doesn't actually
   have.

### pid → profile correlation

A model's `[PID]` (== the port `llama-server` listens on for that instance)
is only meaningful within one boot of `llama-router.service`. Persisted
table `pid_map(boot_id, port) -> profile`, updated the moment a `model
loaded` line resolves a pending load. Later `[PID]`-tagged lines look up
`(current_boot_id, PID)` to find their profile. Keying by `boot_id`
(journald's `_BOOT_ID` field) means a restart can't cause a stale mapping
from before the restart to leak into the new boot.

## Data model

```sql
CREATE TABLE sync_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cursor TEXT
);

CREATE TABLE pending_loads (
    profile TEXT PRIMARY KEY,
    started_at REAL NOT NULL
);

CREATE TABLE pid_map (
    boot_id TEXT NOT NULL,
    port INTEGER NOT NULL,
    profile TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (boot_id, port)
);

CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    profile TEXT,              -- NULL for unattributed events
    event_type TEXT NOT NULL CHECK (event_type IN
        ('load_success', 'load_failure',
         'request_complete', 'request_error', 'service_crash')),
    duration_ms REAL,          -- load_success: load time; request_complete: total time
    prompt_tokens INTEGER,     -- request_complete only
    gen_tokens INTEGER,        -- request_complete only
    prompt_tps REAL,           -- request_complete only
    gen_tps REAL,              -- request_complete only
    detail TEXT                -- raw error message/snippet
);
CREATE INDEX idx_events_profile ON events(profile);
CREATE INDEX idx_events_ts ON events(ts);
```

`pending_loads` and `pid_map` are working state for correlation across sync
invocations (a load or a long-lived instance can span multiple 10-minute
sync windows) — not queried directly by `stats show`.

## Components

- **`modelctl_stats.py`**
  - Line parsers (regexes) for each of the four signal families above.
  - `sync_journal(db_path, unit="llama-router.service") -> dict` — takes an
    exclusive `fcntl.flock` on `STATE_DIR/.stats-sync.lock`, runs
    `journalctl --user -u <unit> -o json --after-cursor=<stored cursor>`
    (full read on first-ever run), parses, resolves `pending_loads`/
    `pid_map`, and commits new `events` rows + the new cursor in one SQLite
    transaction. Returns a summary dict (counts by event type) for the CLI
    to print.
  - `ensure_schema(conn)` — idempotent `CREATE TABLE IF NOT EXISTS`.
  - `query_stats(db_path, profile=None, ctx_min=None, ctx_max=None,
    since=None) -> list[dict]` — aggregates `events` grouped by profile
    (loads, load failures, avg load time, requests, request errors, avg
    gen/prompt tok/s), joins each profile's current `config.ctx` from
    `PROFILES_DIR/{profile}.json` for filtering/display, and separately
    surfaces unattributed counts (`service_crash` rows, `profile IS NULL`
    request errors).
- **`modelctl.py`**: new `stats` argparse subcommand group (mirrors the
  `router` subcommand group at [modelctl.py:2448](../../../modelctl.py)):
  - `cmd_stats_sync(args)` — calls `sync_journal`, prints the summary.
  - `cmd_stats_show(args)` — calls `query_stats` with `--profile`,
    `--ctx`/`--ctx-min`/`--ctx-max`, `--since` (relative duration string,
    e.g. `24h`, `7d`) parsed from CLI flags, prints a table plus the
    unattributed-events footer line.

## CLI

```
$ modelctl stats sync
Synced 47 new events (2 loads, 0 load failures, 43 requests, 2 request errors).

$ modelctl stats show --ctx-min 100000 --since 7d
PROFILE                     CTX      LOADS  LOAD-FAIL  AVG LOAD   REQS   REQ-ERR  AVG GEN T/S  AVG PROMPT T/S
qwen3.6-moe-mtp-longctx     262144   12     0          38.2s      340    2        4.47         172.3
qwen3.6-27b-131k            131072   3      1          51.0s      12     0        9.8          195.0

2 service crashes (oom-kill), 1 unattributed request error -- not tied to a specific profile.
```

## Systemd timer

Not auto-installed by any modelctl command in v1 (YAGNI — can add a
`modelctl stats install-timer` convenience command later if manual setup
proves annoying). Documented unit files, copied to
`~/.config/systemd/user/` and enabled by hand:

```ini
# modelctl-stats-sync.service
[Unit]
Description=modelctl stats sync (parse llama-router journal into stats.db)

[Service]
Type=oneshot
ExecStart=%h/.local/bin/modelctl stats sync
```

```ini
# modelctl-stats-sync.timer
[Unit]
Description=Periodic modelctl stats sync

[Timer]
OnUnitActiveSec=10min
Persistent=true

[Install]
WantedBy=timers.target
```

## Error handling

- `journalctl` missing/permission denied/unit not found: `sync_journal`
  raises, `cmd_stats_sync` prints the error and exits 1; cursor is
  untouched since the failure happens before the commit.
- Unrecognized journal lines: silently skipped — most lines aren't events
  this feature tracks, and that's expected, not an error.
- Cursor + event inserts commit in a single transaction, so a crash
  mid-sync can't leave the cursor advanced past events that weren't
  actually stored (or vice versa) — the next run just re-fetches the same
  window.
- Concurrent sync invocations (manual run overlapping the timer): the
  `flock` on `.stats-sync.lock` makes the second call fail fast with an
  explicit "sync already in progress" message instead of racing on
  `pending_loads`/`pid_map` state.
- Profile JSON deleted after events were logged: `stats show` still lists
  the profile by name (from `events`), with `ctx` shown as `?`; ctx filters
  simply won't match it.

### Known limitations (documented, not solved)

- **Concurrent auto-loads of two different models** could race the
  "next `model loaded` line belongs to the most recent pending load"
  assumption. Router loads have been observed proceeding one at a time in
  practice, and `--models-max 2` makes true concurrent *different-model*
  auto-loads an edge case rather than the common path — not worth the
  complexity of solving in v1.
- **Port/pid reuse within the same boot** for a different model, before a
  fresh `model loaded` line re-establishes the mapping, could briefly
  misattribute a stray line. Any actual request line for the new instance
  is always preceded by its own `model loaded` line, so the exposure window
  is effectively empty in practice.
- **`oom-kill` service crashes** aren't attributed to whichever profile was
  active — recorded unattributed rather than guessed.

## Testing

New `test_modelctl_stats.py`, mirroring `test_modelctl_vram.py`'s
standalone-module test pattern:

- Parser unit tests against literal journald line fixtures (the exact lines
  captured on this system) for each event type: load success, load
  failure, request_complete (3-line group), request_error (bracketed and
  unbracketed), service_crash.
- Correlation test: a synthetic ordered line sequence through the parsing
  pass covering load → pid-map seeding → subsequent requests → a
  reload/reuse of the same port by a different profile.
- `ensure_schema`/insert/`query_stats` round-trip tests against
  `sqlite3.connect(":memory:")`.
- Cursor persistence test: two sequential `sync_journal` calls against
  mocked `journalctl` output, asserting the second only requests
  `--after-cursor=<cursor from first call>`.
- `query_stats` aggregation tests: fixed event fixtures → expected
  avg/error-rate math, `--ctx-min`/`--ctx-max` filtering, and a profile
  whose JSON is missing (ctx shows as unknown, still included by name).
- Lock-contention test: a held lock causes a second `sync_journal` call to
  fail fast without touching `events`/`sync_state`.

`test_modelctl.py` gains argparse-wiring + output-formatting tests for
`cmd_stats_sync`/`cmd_stats_show`, following the existing pattern used for
`cmd_router_stats`.

## Out of scope (deliberately deferred)

- Retention/pruning of `stats.db`.
- Auto-installing the systemd timer via a modelctl command.
- Per-request context-length tracking/filtering (ctx filter is the
  profile's static configured value, not actual prompt length per request).
- Solving the concurrent-load pid race or oom-kill attribution beyond
  "log it unattributed."
- Any TUI surface for these stats — CLI only.
