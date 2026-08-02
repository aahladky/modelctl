"""Remote hands: the rig's files and shell, reachable from a plain chat.

Today the only way a Claude chat reaches this box is the electron
desktop app, because that is where the local MCP servers live. Headless
mode takes the desktop away on purpose, and a phone never had it. This
module is the other route: a small MCP server over HTTP that claude.ai
contacts as a *custom connector*, so a browser or phone session gets the
same files and the same shell.

Why it is shaped like this (verified against Anthropic's connector docs
on 2026-08-02, which outrank any summary of them):

* Custom connectors are contacted FROM Anthropic's servers, not from the
  browser. "your MCP server must be reachable over the public internet
  from Anthropic's IP ranges" -- egress is 160.79.104.0/21. So the
  endpoint needs a public name and a trusted certificate, which is
  exactly what `tailscale funnel` hands out. Nothing here is reachable
  until Aaron turns that on.
* Auth is a fixed credential in a request header (`static_headers`, in
  beta at the time of writing). Claude accepts an allowlisted set of
  header names -- `authorization`, `x-api-key`, `x-auth-token` -- and
  sends the value verbatim, so an Authorization value must include the
  `Bearer ` prefix itself. This server accepts any of the three.
* Credentials never go in the URL. The MCP authorization spec prohibits
  access tokens in the query string, and a URL leaks into logs, proxies
  and history. There is no query-parameter fallback in here to reach
  for on a bad day.

Why bespoke rather than a maintained wrapper. The order preferred
mcp-proxy or supergateway with a gateway in front. Both were checked on
2026-08-02: mcp-proxy has no inbound authentication at all, and
supergateway's `--oauth2Bearer`/`--header` flags set headers on
*outbound* connections, not requirements on inbound ones. Neither can
satisfy "auth check on EVERY request" without a second reverse proxy,
and neither can write the per-tool-call audit record, which needs to see
the decoded JSON-RPC. So the transport, the auth and the audit are
local, and they are the only things that are.

Exposure is default-off in three independent places, and none of them
is flipped by importing this module or by running the tests: the systemd
unit ships disabled, the listener binds 127.0.0.1, and the funnel is not
configured. `modelctl remote-hands on` is the only thing that changes
that, and it refuses without a token file.

This module must stay a leaf: no modelctl imports beyond paths/fsutil.
"""
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import modelctl_fsutil
import modelctl_remote_hands_oauth as oauth
from modelctl_paths import STATE_DIR

SERVER_NAME = "modelctl-remote-hands"
SERVER_VERSION = "1.0.0"

# Loopback, always. The funnel is what makes this reachable, and it is a
# separate, explicit, revocable step -- binding 0.0.0.0 would put the
# endpoint on the LAN the moment the service starts, which is not what
# "default-off exposure" means.
DEFAULT_HOST = "127.0.0.1"
# 9292 is llama-swap, 9293 the console, 9411 the headless verify probe.
DEFAULT_PORT = 9294
MCP_PATH = "/mcp"

# Newest first. `initialize` echoes the client's version when it is one
# of these, which is what the spec asks for; otherwise it answers with
# the newest we implement and lets the client decide.
SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
PROTOCOL_VERSION = SUPPORTED_PROTOCOLS[0]

# The three header names Claude's connector UI will send. Checked in this
# order; the first one present decides, so a stale second header cannot
# quietly rescue a wrong first one.
AUTH_HEADERS = ("authorization", "x-api-key", "x-auth-token")

# Anthropic's published egress range, recorded for the audit log and for
# `status` to compare against. Not enforced: Funnel terminates TLS and
# proxies, so the peer address this process sees is the local tailscaled,
# and rejecting on it would reject everything.
ANTHROPIC_EGRESS = "160.79.104.0/21"

# Filesystem allowlist. moe-review lives under workspace/ already; it is
# named anyway so that moving it later does not silently drop it.
DEFAULT_ROOTS = (
    "/home/aaron/workspace",
    "/home/aaron/models",
    "/home/aaron/services",
    "/home/aaron/workspace/moe-review",
)

# Response caps. A tool result travels back through a model's context, so
# an unbounded read is an expensive way to fail; truncation is reported
# in the text rather than hidden.
MAX_READ_BYTES = 1_000_000
MAX_OUTPUT_BYTES = 200_000
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_LIST_ENTRIES = 2000

DEFAULT_COMMAND_TIMEOUT = 120
MAX_COMMAND_TIMEOUT = 600

SERVICE_NAME = "modelctl-remote-hands.service"

# The serving stack. CLAUDE.md's floor is "never restart llama-swap,
# OVMS or the console", and this is the one place where that floor can be
# made mechanical rather than remembered: a chat session driving this
# server is exactly the caller most likely to reach for a restart out of
# habit. It is a speed bump and is documented as one -- a determined
# caller can spell the same command another way -- so it is not part of
# the security boundary, which is the token.
PROTECTED_PATTERN = re.compile(
    r"\b(?:systemctl(?:\s+--user)?\s+(?:restart|stop|kill|reload-or-restart)"
    r"|pkill|killall)\b[^\n]*\b(?:llama-swap|llama_swap|ovms|modelctl-web)\b",
    re.IGNORECASE)


# --- paths -----------------------------------------------------------------
# Read through functions, not frozen at import: the suite redirects these
# per-test, and STATE_DIR itself moves with MODELCTL_HOME.
#
# Deliberately under STATE_DIR rather than the ~/.local/state/modelctl
# path the order named -- the same call modelctl_display made, for the
# same reason: every other piece of modelctl state resolves through
# modelctl_paths so MODELCTL_HOME moves all of it at once, and a second
# state root is the exact split modelctl_paths exists to prevent. The
# filename is the one the order asked for.

def token_path():
    override = os.environ.get("MODELCTL_REMOTE_HANDS_TOKEN_PATH", "").strip()
    return Path(override) if override else STATE_DIR / "remote-hands-token"


def audit_path():
    override = os.environ.get("MODELCTL_REMOTE_HANDS_AUDIT_PATH", "").strip()
    return Path(override) if override else STATE_DIR / "remote-hands.log"


def allow_roots():
    """The filesystem allowlist, as resolved absolute paths.

    Resolved once per call and compared against resolved candidates, so a
    symlink anywhere in either path cannot smuggle a target out of the
    allowlist. A root that does not exist is kept rather than dropped:
    dropping it would silently widen nothing but would make `status`
    claim an allowlist the server is not using."""
    raw = os.environ.get("MODELCTL_REMOTE_HANDS_ROOTS", "").strip()
    parts = raw.split(os.pathsep) if raw else DEFAULT_ROOTS
    roots, seen = [], set()
    for part in parts:
        if not part:
            continue
        resolved = Path(part).expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    return roots


# --- token -----------------------------------------------------------------

class RemoteHandsError(Exception):
    """A refusal with an operator-readable reason."""


def read_token():
    """The configured token, or None.

    An empty token file is None, never "": an anonymous request presents
    "" too, and compare_digest("", "") is True, which would open the
    shell of this box to the internet."""
    env = os.environ.get("MODELCTL_REMOTE_HANDS_TOKEN", "").strip()
    if env:
        return env
    try:
        stored = token_path().read_text().strip()
    except OSError:
        return None
    return stored or None


def create_token(force=False):
    """Generate the bearer token. Returns (token, created).

    32 random bytes, hex-encoded, 0600. Written with the umask cleared so
    the mode is the mode asked for rather than whatever the caller's
    shell subtracted. Never overwrites unless asked: rotating a live
    token silently would break a connector that is working."""
    path = token_path()
    existing = read_token()
    if existing and not force:
        return existing, False
    token = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    modelctl_fsutil.atomic_write_text(path, token + "\n")
    os.chmod(path, 0o600)
    return token, True


def presented_credentials(headers):
    """Every credential the request offered, in header precedence order.

    `getattr(headers, "get", ...)` rather than dict indexing so this
    works with both http.client message objects (case-insensitive) and
    plain dicts in tests."""
    out = []
    for name in AUTH_HEADERS:
        value = headers.get(name) or headers.get(name.title())
        if not value:
            continue
        value = value.strip()
        if name == "authorization" and value.lower().startswith("bearer "):
            value = value[7:].strip()
        if value:
            out.append(value)
    return out


def authorized(headers, token=None):
    """Whether a request may proceed. Constant-time, and False-by-default.

    A missing server-side token is a hard no rather than an open door:
    the service can start before `create_token` has ever run, and that
    window must not be an unauthenticated shell."""
    token = token if token is not None else read_token()
    if not token:
        return False
    ok = False
    for candidate in presented_credentials(headers):
        # No early return: every presented credential is compared, so the
        # time taken does not depend on which one matched.
        if secrets.compare_digest(candidate, token):
            ok = True
    return ok


# --- redaction -------------------------------------------------------------
# The patterns are modelctl_diagnostics.redact's, copied rather than
# imported: modelctl_diagnostics imports the 5k-line CLI hub at module
# level, and this file is a leaf on purpose (ci/checks.sh enforces it) --
# the audit path of an internet-facing service is the last place to
# acquire a dependency on argparse and sys.exit. The copy is small, and
# it is exercised by tests here rather than trusted to stay in sync.

REDACTED = "<redacted>"

# `Bearer <value>` wherever it appears. First, because the value is the
# credential and the word in front of it is not a name=value pair.
_BEARER = re.compile(r"(?i)\b(bearer\s+)\S+")

# name=value / name: value where the NAME looks like a credential. Value
# rather than name matching is not enough on its own: a 64-hex operator
# token is indistinguishable from a digest by inspection. The name part
# has no mandatory leading character -- "Authorization" is the header
# this exists for, and a required prefix character would eat its "A" and
# then fail to find "auth".
_SECRET_PAIR = re.compile(
    r"(?i)\b([\w-]*"
    r"(?:token|secret|key|password|passwd|credential|auth|session|cookie)"
    r"[\w-]*)(\s*[=:]\s*)(\S+)")

# ...and values that are a credential regardless of what they are called.
_SECRET_VALUE = re.compile(r"\b(?:hf_|sk-|ghp_|gho_|github_pat_)\S+")

_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|KEY|PASSWORD|PASSWD|CREDENTIAL|AUTH|SESSION|COOKIE)",
    re.IGNORECASE)


def redact_mapping(mapping):
    """modelctl_diagnostics.redact, for a name -> value mapping."""
    out = {}
    for key, value in (mapping or {}).items():
        text = str(value)
        if _SECRET_NAME.search(str(key)) or _SECRET_VALUE.match(text):
            out[key] = REDACTED
        else:
            out[key] = text
    return out


def redact_text(text):
    """Free text, safe to write to the audit log.

    Three passes, because a log line is prose rather than a mapping: the
    live operator token by exact value (the one secret this process knows
    for certain, and the one whose leak matters most), then credential
    -shaped name=value pairs, then values that announce themselves."""
    text = str(text)
    try:
        secret = read_token()
    except Exception:                          # noqa: BLE001 -- never fail
        secret = None
    if secret and secret in text:
        text = text.replace(secret, REDACTED)
    text = _BEARER.sub(lambda m: m.group(1) + REDACTED, text)
    text = _SECRET_PAIR.sub(lambda m: m.group(1) + m.group(2) + REDACTED, text)
    return _SECRET_VALUE.sub(REDACTED, text)


# --- audit -----------------------------------------------------------------

def args_digest(args):
    """A stable fingerprint of a tool's arguments.

    The arguments themselves are not logged. A `write_file` call carries
    the file's contents and a `run_command` call can carry a secret; the
    log has to be safe to read out loud, and a digest still answers "was
    this the same call as that one"."""
    try:
        blob = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(args)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


def audit(tool, outcome, args=None, digest=None, peer=None, ms=None,
          detail=None, now=None, event="tool", **fields):
    """Append one audit record. Never raises.

    Appended with O_APPEND in a single write so concurrent handler
    threads interleave whole lines rather than halves, and created 0600
    because it records what was touched and when.

    `event` separates the two kinds of line the log carries: one
    `request` per HTTP request (what arrived, from where, how it was
    authenticated, what it got back) and one `tool` per tool invocation
    (what ran, on which arguments, how long it took). A tools/call
    produces both, and neither is derivable from the other -- the request
    line is the only record of a call that never reached a tool."""
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S",
                            time.localtime(now if now else time.time())),
        "event": event,
    }
    # A fixed order, so the log reads as columns rather than as whatever
    # the caller happened to pass.
    for key in ("method", "path", "status", "auth"):
        value = fields.pop(key, None)
        if value not in (None, ""):
            entry[key] = value
    if peer:
        entry["peer"] = peer
    for key in ("xff",):
        value = fields.pop(key, None)
        if value not in (None, ""):
            entry[key] = redact_text(value)[:200]
    entry["tool"] = tool
    if digest is not None:
        entry["args_sha256"] = digest
    elif args is not None:
        entry["args_sha256"] = args_digest(args)
    # ...and otherwise omitted: a request that never named a tool has no
    # arguments, and printing the digest of None on every 401 is a column
    # of one constant pretending to be evidence.
    entry["outcome"] = outcome
    if ms is not None:
        entry["ms"] = int(ms)
    if detail:
        entry["detail"] = redact_text(detail)[:300]
    for key, value in fields.items():
        if value not in (None, ""):
            entry[key] = value
    line = json.dumps(entry) + "\n"
    try:
        path = audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass
    return entry


def audit_tail(count=5):
    """The last `count` audit records, oldest first. [] when there is no
    log yet, which is what a machine that has never been used looks
    like."""
    try:
        lines = audit_path().read_text(errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-count:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            out.append({"raw": line})
    return out


# Outcomes that mean "a credential was offered and refused", as opposed
# to "this endpoint is public" or "the request was malformed". These are
# what `status` counts, because they are the only ones that answer "is
# something hammering it".
AUTH_FAILURE_OUTCOMES = ("unauthorized", "rate-limited")


def audit_stats(window=3600, now=None):
    """Line count, and the auth failures in the last `window` seconds.

    Reads the whole log rather than tailing it: the file is one line per
    request and rotates by nobody, so this is the honest count. If it
    ever gets large enough for that to hurt, the fix is rotation, not a
    number that quietly means something else."""
    now = time.time() if now is None else now
    stats = {"lines": 0, "auth_failures_last_hour": 0,
             "last_auth_failure": None, "window_seconds": int(window)}
    try:
        text = audit_path().read_text(errors="replace")
    except OSError:
        return stats
    cutoff = now - window
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        stats["lines"] += 1
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("outcome") not in AUTH_FAILURE_OUTCOMES:
            continue
        stamp = entry.get("at") or ""
        stats["last_auth_failure"] = stamp
        try:
            when = time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%S"))
        except (TypeError, ValueError):
            continue
        if when >= cutoff:
            stats["auth_failures_last_hour"] += 1
    return stats


# --- rate limiting and lockout ---------------------------------------------
# Nothing here survives a restart, and that is the right trade: the state
# is a few hundred bytes of "who has been failing lately", it is only
# useful while an attempt is in progress, and persisting it would put a
# writable file in the path of every unauthenticated request.
#
# Two limiters, because they answer different questions. The per-source
# bucket is the one that makes a single client back off. The global
# bucket is the ceiling that holds when the source cannot be told apart
# -- which is the normal case here: Funnel terminates TLS and proxies, so
# every request arrives from 127.0.0.1 and the only hint of a real client
# is an X-Forwarded-For the funnel wrote. That header is trusted for
# bucketing and for nothing else; a forger can spread their own load
# across keys, and the global bucket is what stops them.

AUTH_RATE = 1.0            # sustained requests per second, per source
AUTH_BURST = 20            # ...and what a source may spend at once
GLOBAL_RATE = 5.0
GLOBAL_BURST = 60
# Consecutive credential failures from one source before it is banned,
# then doubling: 30s, 60s, 120s... to an hour.
LOCKOUT_THRESHOLD = 5
LOCKOUT_BASE = 30.0
LOCKOUT_MAX = 3600.0
# Keys are attacker-supplied (an XFF value), so the tables are capped and
# oldest-out rather than left to grow.
MAX_TRACKED_SOURCES = 4096

# Every credential failure takes at least this long, whatever the reason.
# The point is not to defeat a patient attacker with a stopwatch; it is
# that "unknown client" (no work) and "wrong secret" (a hash compare)
# must not be separable by timing the response.
AUTH_FAILURE_DELAY = 0.15


class TokenBucket:
    """Classic token bucket, keyed. `take` is the whole interface."""

    def __init__(self, rate, burst, maximum=MAX_TRACKED_SOURCES):
        self.rate, self.burst, self.maximum = float(rate), float(burst), maximum
        self._state = {}

    def take(self, key="-", now=None, cost=1.0):
        now = time.time() if now is None else now
        tokens, stamp = self._state.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - stamp) * self.rate)
        if tokens < cost:
            self._state[key] = (tokens, now)
            return False
        self._state[key] = (tokens - cost, now)
        self._evict(now)
        return True

    def _evict(self, now):
        if len(self._state) <= self.maximum:
            return
        # Full buckets are indistinguishable from absent ones, so drop
        # those first; if that is not enough, drop the least recently
        # touched.
        for key, (tokens, stamp) in list(self._state.items()):
            if tokens + (now - stamp) * self.rate >= self.burst:
                self._state.pop(key, None)
        while len(self._state) > self.maximum:
            oldest = min(self._state, key=lambda k: self._state[k][1])
            self._state.pop(oldest, None)

    def reset(self):
        self._state.clear()


class Lockout:
    """Consecutive-failure tempban with exponential backoff."""

    def __init__(self, threshold=LOCKOUT_THRESHOLD, base=LOCKOUT_BASE,
                 maximum=LOCKOUT_MAX, tracked=MAX_TRACKED_SOURCES):
        self.threshold, self.base, self.maximum = threshold, base, maximum
        self.tracked = tracked
        self._state = {}

    def failure(self, key, now=None):
        """Record one failure. Returns the seconds this source is now
        banned for (0 while it is still under the threshold)."""
        now = time.time() if now is None else now
        count, _ = self._state.get(key, (0, 0.0))
        count += 1
        if count < self.threshold:
            self._state[key] = (count, 0.0)
            return 0.0
        span = min(self.maximum,
                   self.base * (2 ** (count - self.threshold)))
        self._state[key] = (count, now + span)
        self._prune(now)
        return span

    def success(self, key):
        """A credential that worked clears the streak. Without this a
        connector that fluffs its first refresh would stay penalised for
        the rest of the hour."""
        self._state.pop(key, None)

    def retry_after(self, key, now=None):
        now = time.time() if now is None else now
        _, until = self._state.get(key, (0, 0.0))
        return max(0.0, until - now)

    def _prune(self, now):
        for key, (_, until) in list(self._state.items()):
            if until and until <= now:
                self._state.pop(key, None)
        while len(self._state) > self.tracked:
            soonest = min(self._state, key=lambda k: self._state[k][1])
            self._state.pop(soonest, None)

    def reset(self):
        self._state.clear()


SOURCE_BUCKET = TokenBucket(AUTH_RATE, AUTH_BURST)
GLOBAL_BUCKET = TokenBucket(GLOBAL_RATE, GLOBAL_BURST)
LOCKOUT = Lockout()


def reset_rate_limits():
    """Drop every limiter's memory. The suite calls this between tests;
    nothing in the service does."""
    SOURCE_BUCKET.reset()
    GLOBAL_BUCKET.reset()
    LOCKOUT.reset()


def rate_check(source, distinguishable=True, now=None):
    """(allowed, retry_after_seconds, reason).

    The lockout is skipped for an indistinguishable source. Behind the
    funnel with no X-Forwarded-For every caller shares one key, and
    banning that key would let a stranger's failed guesses lock out the
    connector -- a denial of service handed over for free. The global
    bucket still applies, which is the part that cannot be turned against
    us."""
    now = time.time() if now is None else now
    if distinguishable:
        banned = LOCKOUT.retry_after(source, now=now)
        if banned > 0:
            return False, banned, "locked-out"
        if not SOURCE_BUCKET.take(source, now=now):
            return False, 1.0 / AUTH_RATE, "source-rate"
    if not GLOBAL_BUCKET.take("-", now=now):
        return False, 1.0 / GLOBAL_RATE, "global-rate"
    return True, 0.0, ""


# --- filesystem allowlist --------------------------------------------------

def resolve_allowed(raw):
    """Resolve `raw` and confirm it is inside the allowlist.

    Resolution happens before the containment test, so `..` segments and
    symlinks are already collapsed when the test runs -- checking the
    string first and resolving later is the classic way to let
    `/home/aaron/workspace/../../etc/shadow` through.

    Works for paths that do not exist yet (writes): resolve(strict=False)
    resolves the existing prefix and normalises the rest."""
    if not raw or not str(raw).strip():
        raise RemoteHandsError("path is required")
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        raise RemoteHandsError(
            f"path must be absolute: {raw!r} (the server has no working "
            f"directory a caller can reason about)")
    resolved = candidate.resolve()
    for root in allow_roots():
        if resolved == root or root in resolved.parents:
            return resolved
    raise RemoteHandsError(
        f"path is outside the allowlist: {resolved}\nallowed roots: "
        + ", ".join(str(r) for r in allow_roots()))


# --- tools -----------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description":
            "Read a UTF-8 text file on the rig. Only paths under the "
            "allowlist (workspace, models, services, moe-review) are "
            "readable. Large files are truncated and say so.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "absolute path on the rig"},
                "offset": {"type": "integer",
                           "description": "first line to return (1-based)"},
                "limit": {"type": "integer",
                          "description": "how many lines to return"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description":
            "Write or append a UTF-8 text file on the rig, under the "
            "allowlist. Parent directories are created as needed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["rewrite", "append"],
                         "description": "default rewrite"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description":
            "List one directory on the rig, under the allowlist. Entries "
            "are marked [DIR]/[FILE] with sizes.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_command",
        "description":
            "Run a shell command on the rig as aaron and return its exit "
            "code, stdout and stderr. Synchronous, with a timeout. "
            "Commands that would restart llama-swap, OVMS or the web "
            "console are refused.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string",
                        "description": "working directory (default $HOME)"},
                "timeout": {"type": "integer",
                            "description":
                                f"seconds, default {DEFAULT_COMMAND_TIMEOUT}, "
                                f"max {MAX_COMMAND_TIMEOUT}"},
            },
            "required": ["command"],
        },
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOLS)


def _truncate(text, limit, what):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[truncated: {what} exceeded {limit} bytes]"


def tool_read_file(args):
    path = resolve_allowed(args.get("path"))
    if path.is_dir():
        raise RemoteHandsError(f"{path} is a directory (use list_directory)")
    try:
        data = path.read_text(errors="replace")
    except OSError as e:
        raise RemoteHandsError(f"cannot read {path}: {e}")
    offset, limit = args.get("offset"), args.get("limit")
    if offset or limit:
        lines = data.splitlines(keepends=True)
        start = max(0, int(offset or 1) - 1)
        end = start + int(limit) if limit else len(lines)
        data = "".join(lines[start:end])
    return _truncate(data, MAX_READ_BYTES, "file")


def tool_write_file(args):
    path = resolve_allowed(args.get("path"))
    content = args.get("content")
    if not isinstance(content, str):
        raise RemoteHandsError("content must be a string")
    mode = (args.get("mode") or "rewrite").lower()
    if mode not in ("rewrite", "append"):
        raise RemoteHandsError(f"unknown mode {mode!r} (rewrite or append)")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with path.open("a", encoding="utf-8") as fh:
                fh.write(content)
        else:
            # Atomic: a half-written file under workspace/ is a file some
            # other tool in this repo will happily read as complete.
            modelctl_fsutil.atomic_write_text(path, content)
    except OSError as e:
        raise RemoteHandsError(f"cannot write {path}: {e}")
    return f"{mode} {path} ({len(content)} bytes)"


def tool_list_directory(args):
    path = resolve_allowed(args.get("path"))
    if not path.is_dir():
        raise RemoteHandsError(f"{path} is not a directory")
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except OSError as e:
        raise RemoteHandsError(f"cannot list {path}: {e}")
    out = []
    for entry in entries[:MAX_LIST_ENTRIES]:
        try:
            size = entry.stat().st_size if entry.is_file() else 0
        except OSError:
            size = 0
        out.append(f"[DIR]  {entry.name}" if entry.is_dir()
                   else f"[FILE] {entry.name} ({size} bytes)")
    if len(entries) > MAX_LIST_ENTRIES:
        out.append(f"[truncated: {len(entries)} entries, showing "
                   f"{MAX_LIST_ENTRIES}]")
    return "\n".join(out) or "(empty)"


def protected_refusal(command):
    """The reason a command is refused, or None."""
    if PROTECTED_PATTERN.search(command or ""):
        return ("refused: this would restart or kill part of the live "
                "serving stack (llama-swap, OVMS or the web console). "
                "Those stay up; do it at the console if it is really "
                "needed.")
    return None


def tool_run_command(args):
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        raise RemoteHandsError("command must be a non-empty string")
    refusal = protected_refusal(command)
    if refusal:
        raise RemoteHandsError(refusal)
    timeout = args.get("timeout") or DEFAULT_COMMAND_TIMEOUT
    try:
        timeout = max(1, min(int(timeout), MAX_COMMAND_TIMEOUT))
    except (TypeError, ValueError):
        raise RemoteHandsError("timeout must be an integer number of seconds")
    cwd = args.get("cwd")
    if cwd:
        # Not allowlist-checked on purpose: this is a shell, and the
        # posture is the one the desktop session already has. The
        # allowlist governs the file tools, which are the ones that
        # promise it.
        cwd = str(Path(str(cwd)).expanduser())
        if not os.path.isdir(cwd):
            raise RemoteHandsError(f"cwd does not exist: {cwd}")
    else:
        cwd = str(Path.home())
    started = time.time()
    try:
        proc = subprocess.run(command, shell=True, cwd=cwd, text=True,
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RemoteHandsError(
            f"timed out after {timeout}s: {command[:200]}\n"
            f"(nothing was killed beyond the command itself; if it left "
            f"work running, check for it)")
    except OSError as e:
        raise RemoteHandsError(f"cannot run command: {e}")
    elapsed = time.time() - started
    parts = [f"exit {proc.returncode} in {elapsed:.1f}s"]
    if proc.stdout:
        parts.append(_truncate(proc.stdout, MAX_OUTPUT_BYTES, "stdout"))
    if proc.stderr:
        parts.append("--- stderr ---\n"
                     + _truncate(proc.stderr, MAX_OUTPUT_BYTES, "stderr"))
    if not proc.stdout and not proc.stderr:
        parts.append("(no output)")
    return "\n".join(parts)


TOOL_IMPLS = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_directory": tool_list_directory,
    "run_command": tool_run_command,
}


def call_tool(name, args, peer=None, xff=None):
    """Run one tool and audit the outcome. Returns (text, is_error).

    Every path through here writes exactly one audit line -- including
    the unknown-tool and refusal paths, which are the ones worth seeing
    in a log after the fact."""
    args = args if isinstance(args, dict) else {}
    digest = args_digest(args)
    started = time.time()
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        audit(name, "unknown-tool", digest=digest, peer=peer, xff=xff,
              ms=(time.time() - started) * 1000)
        return f"unknown tool: {name}", True
    try:
        text = impl(args)
    except RemoteHandsError as e:
        audit(name, "denied", digest=digest, peer=peer, xff=xff,
              ms=(time.time() - started) * 1000, detail=str(e))
        return str(e), True
    except Exception as e:                     # noqa: BLE001 -- see below
        # A tool raising anything else is a bug in this file, but a
        # traceback out of a handler thread would kill the request and
        # leave no record. The record is the point.
        audit(name, "error", digest=digest, peer=peer, xff=xff,
              ms=(time.time() - started) * 1000,
              detail=f"{type(e).__name__}: {e}")
        return f"{type(e).__name__}: {e}", True
    audit(name, "ok", digest=digest, peer=peer, xff=xff,
          ms=(time.time() - started) * 1000)
    return text, False


# --- JSON-RPC / MCP --------------------------------------------------------

def _result(msg_id, payload):
    return {"jsonrpc": "2.0", "id": msg_id, "result": payload}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}


def payload_tools(payload):
    """[(tool name, args digest)] for every tools/call in a body.

    Read before dispatch so the request-level audit line can name the
    tool even when the call never runs -- a rate-limited or rejected
    tools/call is exactly the line worth having."""
    messages = payload if isinstance(payload, list) else [payload]
    out = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("method") != "tools/call":
            continue
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            continue
        args = params.get("arguments")
        out.append((str(params.get("name")),
                    args_digest(args if isinstance(args, dict) else {})))
    return out


def handle_message(msg, peer=None, xff=None):
    """One JSON-RPC message in, one response out (or None).

    None means "nothing to send back", which is the correct answer for a
    notification -- the transport turns that into 202 Accepted with an
    empty body rather than inventing a response id."""
    if not isinstance(msg, dict):
        return _error(None, -32600, "invalid request: not an object")
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}
    is_notification = "id" not in msg

    if method == "initialize":
        asked = (params or {}).get("protocolVersion")
        version = asked if asked in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION
        return _result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions":
                "Files and shell on Aaron's rig. Filesystem tools are "
                "restricted to " + ", ".join(str(r) for r in allow_roots())
                + ". The serving stack (llama-swap, OVMS, the web console) "
                  "must never be restarted.",
        })
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return None if is_notification else _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        text, is_error = call_tool(name, params.get("arguments"), peer=peer,
                                   xff=xff)
        return _result(msg_id, {"content": [{"type": "text", "text": text}],
                                "isError": is_error})
    if is_notification:
        # An unknown notification is not an error the client can act on,
        # and per JSON-RPC a notification never gets a response.
        return None
    return _error(msg_id, -32601, f"method not found: {method}")


def handle_payload(payload, peer=None, xff=None):
    """A whole request body in, a whole response body out (or None).

    Arrays are accepted because protocol revisions up to 2025-03-26
    allowed batching; 2025-11-25 dropped it. Accepting one is cheaper
    than a client-version matrix."""
    if isinstance(payload, list):
        responses = [r for r in (handle_message(m, peer, xff)
                                 for m in payload) if r is not None]
        return responses or None
    return handle_message(payload, peer, xff)


# --- HTTP transport (MCP streamable HTTP) ----------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Streamable HTTP, stateless.

    No `Mcp-Session-Id` is issued. A session id would oblige this server
    to remember sessions and answer 404 for expired ones; with one
    connector and no server-initiated messages there is nothing for a
    session to hold, and statelessness means a restart of the unit does
    not strand a live connector.
    """

    server_version = f"{SERVER_NAME}/{SERVER_VERSION}"
    protocol_version = "HTTP/1.1"
    token = None

    # One audit record per request, opened when the request line parses
    # and written when the request is done -- see `parse_request` and
    # `handle_one_request`. None outside a request.
    _record = None
    _started = 0.0

    # --- the request-level audit record -----------------------------------
    # Hooked at parse/dispatch rather than inside each handler method so
    # there is no way to add an endpoint that answers without leaving a
    # line. A verb with no do_* method (501) and a request line that does
    # not parse (400) are logged too: those are what a scanner produces,
    # and "nothing in the log" must never be one of the possible answers
    # to "what reached this port".

    def parse_request(self):
        self._started = time.time()
        # `malformed` up front and cleared on success, rather than set
        # afterwards on failure: BaseHTTPRequestHandler answers a bad
        # request line from inside parse_request, which reaches
        # end_headers, which flushes and clears the record -- so by the
        # time it returns False there is nothing left to label. Writing
        # the label after the fact both lost it and raised on the None.
        self._record = {"method": "?", "path": "?", "auth": "none",
                        "tool": "-", "digest": None,
                        "outcome": "malformed", "detail": None}
        ok = BaseHTTPRequestHandler.parse_request(self)
        if ok:
            self._note(method=self.command,
                       path=self.path.split("?")[0][:200],
                       peer=self._peer(), xff=self._forwarded_for(),
                       outcome=None)
        return ok

    def handle_one_request(self):
        try:
            BaseHTTPRequestHandler.handle_one_request(self)
        finally:
            # The backstop, for a request that never sent headers at all
            # (a dropped connection, a handler that raised). The normal
            # path already flushed in end_headers.
            self._flush_record()

    def end_headers(self):
        # Written before the response reaches the client, not after: a
        # caller that has its answer can rely on the line already being
        # in the log, which is what makes "make a call, show the line"
        # a test rather than a race. It also means a crash between
        # answering and logging is not a possible state.
        self._flush_record()
        BaseHTTPRequestHandler.end_headers(self)

    def _flush_record(self):
        record, self._record = self._record, None
        if record is not None:
            self._write_request_audit(record)

    def send_response(self, code, message=None):
        # Every response goes through here, including BaseHTTPRequest-
        # Handler's own send_error for unsupported verbs, so the status
        # in the log is the status that was actually sent.
        if self._record is not None and self._record.get("status") is None:
            self._record["status"] = int(code)
        BaseHTTPRequestHandler.send_response(self, code, message)

    @staticmethod
    def _outcome_for(status, explicit=None):
        if explicit:
            return explicit
        if status is None:
            return "no-response"
        if status == 401:
            return "unauthorized"
        if status == 429:
            return "rate-limited"
        if status >= 500:
            return "error"
        if status >= 400:
            return "rejected"
        return "ok"

    def _write_request_audit(self, record):
        audit(record.get("tool") or "-",
              self._outcome_for(record.get("status"), record.get("outcome")),
              event="request", digest=record.get("digest"),
              peer=record.get("peer"), xff=record.get("xff"),
              method=record.get("method"), path=record.get("path"),
              status=record.get("status"), auth=record.get("auth"),
              detail=record.get("detail"),
              ms=(time.time() - self._started) * 1000)

    def _note(self, **fields):
        if self._record is not None:
            self._record.update(fields)

    # --- plumbing ---------------------------------------------------------

    def _base(self):
        return oauth.base_url(self.headers.get("Host"))

    def _send_json(self, code, payload, extra=None):
        self._send(code, json.dumps(payload).encode(), extra=extra)

    def _form(self):
        """A urlencoded request body as a flat dict. The token endpoint
        must accept application/x-www-form-urlencoded per RFC 6749 --
        Claude sends both the code exchange and refreshes that way, and a
        JSON-only parser answers 415 and strands the connection."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(min(length, 64 * 1024)) if length else b""
        return {k: v[0] for k, v in urllib.parse.parse_qs(
            raw.decode("utf-8", "replace"), keep_blank_values=True).items()}

    def log_message(self, fmt, *args):
        # journalctl gets method/path/status; the audit log gets the tool
        # calls. Neither ever gets a header value.
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()

    def _peer(self):
        # Behind the funnel this is tailscaled on loopback, not Anthropic;
        # recorded for what it is rather than dressed up as a client IP.
        try:
            return self.client_address[0]
        except (AttributeError, IndexError):
            return "?"

    def _forwarded_for(self):
        """The X-Forwarded-For header verbatim, or ""."""
        try:
            return (self.headers.get("X-Forwarded-For") or "").strip()[:200]
        except AttributeError:
            return ""

    def _source_key(self):
        """What the limiters count against.

        The LAST X-Forwarded-For entry, not the first: a proxy appends
        the peer it saw, so the last hop is the one our funnel wrote and
        everything before it is whatever the caller chose to send. It is
        still only a hint -- see `rate_check` -- but it is the hint that
        costs an attacker something to move."""
        forwarded = self._forwarded_for()
        if forwarded and self._peer_is_loopback():
            return "xff:" + forwarded.split(",")[-1].strip()[:64]
        return "peer:" + self._peer()

    def _peer_is_loopback(self):
        peer = self._peer()
        return peer.startswith("127.") or peer in ("::1", "localhost")

    def _source_is_distinguishable(self):
        """Whether one caller can be told from another.

        False is the normal case behind a funnel that forwards no
        X-Forwarded-For: every request then shares one key, and a tempban
        on that key would be a stranger's ability to lock the connector
        out. The global bucket covers that case instead."""
        return bool(self._forwarded_for()) or not self._peer_is_loopback()

    def _too_many(self, retry_after, reason):
        """429, with the reason in the log and not in the body."""
        self._note(auth="rate-limited", outcome="rate-limited", detail=reason)
        # The request body may not have been read (the gate runs before
        # the read on purpose), so this connection can no longer be
        # trusted to frame a next request on.
        self.close_connection = True
        self._send_json(429, {"error": "slow_down",
                              "error_description": "too many requests"},
                        extra={"Retry-After":
                               str(max(1, int(retry_after + 0.999)))})

    def _rate_gate(self):
        """Lockout + both buckets, ahead of any credential work.

        Applied to the unauthenticated OAuth surface, where an attacker
        picks the cost of each request and there is no credential to stop
        them earlier."""
        allowed, retry, reason = rate_check(
            self._source_key(),
            distinguishable=self._source_is_distinguishable())
        if allowed:
            return True
        self._too_many(retry, reason)
        return False

    def _lockout_gate(self):
        """The lockout only, for the authenticated surface.

        Deliberately not the buckets: a working connector can make a long
        burst of legitimate tool calls, and throttling those would be
        this feature breaking the thing it protects. A source that has
        failed five credentials in a row is a different animal."""
        if not self._source_is_distinguishable():
            return True
        banned = LOCKOUT.retry_after(self._source_key())
        if banned <= 0:
            return True
        self._too_many(banned, "locked-out")
        return False

    def _auth_failure(self, reason):
        """Book one credential failure against the source, then pause.

        The pause is the whole "constant time regardless of reason" rule:
        every failure -- unknown client, wrong secret, expired token, no
        credential at all -- leaves by this door and takes the same
        minimum time to do it."""
        if self._source_is_distinguishable():
            LOCKOUT.failure(self._source_key())
        self._note(auth="rejected", detail=reason)
        remaining = AUTH_FAILURE_DELAY - (time.time() - self._started)
        if remaining > 0:
            time.sleep(remaining)

    def _send(self, code, body=b"", content_type="application/json",
              extra=None):
        self.send_response(code)
        if body:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _unauthorized(self):
        """401 with the header that starts the OAuth flow.

        `resource_metadata` is what Claude follows to discover this
        server's authorization server; without it, an unauthenticated
        connector attempt is a dead end rather than a Connect card. The
        static-token path produces the same 401, so a client that has a
        header credential and got it wrong sees the same thing.

        The request-level record carries the 401; no separate audit call
        here, or a rejected request would write two lines saying the same
        thing and the count in `status` would double."""
        self._auth_failure("no valid credential")
        # The one ceiling that holds when sources cannot be told apart:
        # a flood of failures eventually gets 429s instead of 401s,
        # whoever it is coming from. Successful traffic never draws on
        # this bucket.
        if not GLOBAL_BUCKET.take("auth-failure"):
            self._too_many(1.0 / GLOBAL_RATE, "global-auth-failure-rate")
            return
        self._send_json(401, {"error": "invalid_token",
                              "error_description": "Authentication required"},
                        extra={"WWW-Authenticate":
                               oauth.www_authenticate(self._base(), MCP_PATH)})

    def _check_auth(self):
        """Either credential opens the door.

        The static header token stays because it is what works the day
        the `static_headers` beta reaches this account, and an OAuth
        access token because that is what works today. Neither is
        weakened by the other's presence: both are unguessable secrets
        and both are checked before routing."""
        if not self._lockout_gate():
            return False
        if authorized(self.headers, self.token):
            self._note(auth="static-token")
            LOCKOUT.success(self._source_key())
            return True
        presented = presented_credentials(self.headers)
        if any(oauth.valid_access_token(c) for c in presented):
            self._note(auth="oauth")
            LOCKOUT.success(self._source_key())
            return True
        self._unauthorized()
        return False

    # --- methods ----------------------------------------------------------

    # --- OAuth surface ----------------------------------------------------
    # These are the only unauthenticated paths, and they have to be: a
    # client with no credential yet is exactly who fetches discovery and
    # walks the consent flow. Everything they expose is either public by
    # design (the metadata documents) or gated on a secret the caller
    # must already hold (the consent form takes the operator token; the
    # token endpoint takes a PKCE verifier for a code it was issued).

    def _oauth_audit(self, tool, outcome, args=None, detail=None):
        """The domain event, beside the request line.

        Both are written: the request line says a POST /token got a 400,
        this says which grant and why. Neither is recoverable from the
        other, and only this one names the client."""
        audit(tool, outcome, args=args, peer=self._peer(),
              xff=self._forwarded_for(), event="oauth", detail=detail)

    def _route_oauth_get(self, path, query):
        base = self._base()
        if path in ("/.well-known/oauth-protected-resource",
                    f"/.well-known/oauth-protected-resource{MCP_PATH}"):
            self._note(auth="public")
            self._send_json(200, oauth.protected_resource_metadata(
                base, MCP_PATH))
            return True
        if path in ("/.well-known/oauth-authorization-server",
                    f"/.well-known/oauth-authorization-server{MCP_PATH}"):
            self._note(auth="public")
            self._send_json(200, oauth.authorization_server_metadata(base))
            return True
        if path == "/authorize":
            self._note(auth="public")
            if self._rate_gate():
                self._authorize_page(query)
            return True
        return False

    def _authorize_page(self, query):
        params = {k: v[0] for k, v in query.items()}
        self._note(tool="oauth/authorize", digest=args_digest(params))
        try:
            oauth.validate_authorize(params)
            oauth.resolve_client(params.get("client_id"),
                                 params.get("redirect_uri"))
        except oauth.OAuthError as e:
            # Rendered as a page, not redirected: an invalid client or
            # redirect_uri is exactly the case where bouncing the user
            # onward would be the vulnerability.
            #
            # Still descriptive, unlike the POST below: no credential has
            # been offered yet, so there is nothing here an error message
            # could leak, and this text is how a misconfigured connector
            # tells its operator what is wrong.
            self._oauth_audit("oauth/authorize", "denied", args=params,
                              detail=f"{e.code}: {e.description}")
            self._send(400, f"<h1>{e.code}</h1><p>{oauth._esc(e.description)}"
                            f"</p>".encode(), content_type="text/html")
            return
        self._oauth_audit("oauth/authorize", "ok", args=params,
                          detail=params.get("client_id"))
        html = oauth.consent_html(params["client_id"], params["redirect_uri"],
                                  oauth.authorize_params(params))
        self._send(200, html.encode(), content_type="text/html")

    def _consent_denied(self, form, reason):
        """One failure page for every way the consent POST can fail.

        A credential is in play here, so the reason goes to the log and
        not to the caller: "unknown client" and "wrong secret" must not
        be separable by reading the response, and `_auth_failure` sees to
        it that they are not separable by timing it either. The form is
        re-rendered rather than redirected -- a denied consent hands the
        client nothing, and the operator may simply have pasted the wrong
        thing."""
        self._oauth_audit("oauth/consent", "denied", args=form, detail=reason)
        self._auth_failure(reason)
        client_id = form.get("client_id") or ""
        redirect_uri = form.get("redirect_uri") or ""
        html = oauth.consent_html(client_id, redirect_uri,
                                  oauth.authorize_params(form),
                                  error="That did not match.")
        self._send(401, html.encode(), content_type="text/html")

    def _authorize_submit(self):
        form = self._form()
        supplied = form.pop("operator_token", "")
        self._note(tool="oauth/consent", digest=args_digest(form),
                   auth="public")
        if not self._rate_gate():
            return
        try:
            oauth.validate_authorize(form)
            oauth.resolve_client(form.get("client_id"),
                                 form.get("redirect_uri"))
        except oauth.OAuthError as e:
            self._consent_denied(form, f"{e.code}: {e.description}")
            return
        token = self.token if self.token is not None else read_token()
        if not token or not supplied or not secrets.compare_digest(
                supplied, token):
            self._consent_denied(form, "operator token did not match")
            return
        LOCKOUT.success(self._source_key())
        code = oauth.create_code(form["client_id"], form["redirect_uri"],
                                 form["code_challenge"])
        self._oauth_audit("oauth/consent", "ok", args=form,
                          detail=form.get("client_id"))
        self._note(auth="operator-token")
        self._send(302, b"", extra={
            "Location": oauth.redirect_with_code(form["redirect_uri"], code,
                                                 form.get("state"))})

    def _token_endpoint(self):
        form = self._form()
        grant = form.get("grant_type")
        self._note(tool=f"oauth/{grant or 'token'}", digest=args_digest(form),
                   auth="public")
        if not self._rate_gate():
            return
        try:
            if grant == "authorization_code":
                entry = oauth.redeem_code(
                    form.get("code"), form.get("client_id"),
                    form.get("redirect_uri"), form.get("code_verifier"))
                tokens = oauth.issue_tokens(entry["client_id"],
                                            entry.get("scope"))
                outcome = "oauth/token"
            elif grant == "refresh_token":
                tokens = oauth.refresh_tokens(form.get("refresh_token"),
                                              form.get("client_id"))
                outcome = "oauth/refresh"
            else:
                raise oauth.OAuthError("unsupported_grant_type",
                                       f"unsupported grant_type: {grant}")
        except oauth.OAuthError as e:
            self._oauth_audit(f"oauth/{grant or 'token'}", "denied", args=form,
                              detail=f"{e.code}: {e.description}")
            # The RFC error CODE stays exact, unlike the consent page:
            # Claude re-authorizes only on `invalid_grant` and gives up on
            # anything else, so flattening these to one message would
            # strand the connector rather than protect it. The
            # description is dropped and the timing floor applies, which
            # is the part that costs an attacker something.
            self._auth_failure(f"{e.code}: {e.description}")
            self._send_json(e.status, {"error": e.code},
                            extra={"Cache-Control": "no-store"})
            return
        LOCKOUT.success(self._source_key())
        self._note(auth="oauth-grant")
        self._oauth_audit(outcome, "ok", args=form)
        self._send_json(200, tokens, extra={"Cache-Control": "no-store"})

    def _register_endpoint(self):
        self._note(tool="oauth/register", auth="public")
        if not self._rate_gate():
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(min(length, 64 * 1024)) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            payload = None
        self._note(digest=args_digest(payload))
        try:
            if not isinstance(payload, dict):
                raise oauth.OAuthError("invalid_client_metadata",
                                       "body must be a JSON object")
            result = oauth.register_client(payload)
        except oauth.OAuthError as e:
            self._oauth_audit("oauth/register", "denied", args=payload,
                              detail=f"{e.code}: {e.description}")
            self._send_json(e.status, e.body())
            return
        self._oauth_audit("oauth/register", "ok", args=payload,
                          detail=result["client_id"])
        self._send_json(201, result)

    # --- methods ----------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0]
        query = urllib.parse.parse_qs(
            self.path.partition("?")[2], keep_blank_values=True)
        if self._route_oauth_get(path, query):
            return
        # Auth first, then routing: an unauthenticated caller learns
        # nothing about which paths exist here.
        if not self._check_auth():
            return
        if path == MCP_PATH:
            # No server-initiated SSE stream. The spec allows 405 for
            # exactly this case, and clients fall back to POST-only.
            self._send(405, b"", extra={"Allow": "POST, DELETE"})
            return
        self._send(404, json.dumps({"error": "not found"}).encode())

    def do_DELETE(self):
        if not self._check_auth():
            return
        # Stateless: there is no session to end, so this always succeeds.
        self._send(204)

    def do_POST(self):
        path = self.path.split("?")[0]
        # The three unauthenticated POSTs, routed before the auth gate
        # for the same reason as their GET counterparts: they are how a
        # caller with no credential gets one.
        if path == "/authorize":
            self._authorize_submit()
            return
        if path == "/token":
            self._token_endpoint()
            return
        if path == "/register":
            self._register_endpoint()
            return
        if not self._check_auth():
            return
        if path != MCP_PATH:
            self._send(404, json.dumps({"error": "not found"}).encode())
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            self._send(413, json.dumps({"error": "body too large"}).encode())
            return
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._note(outcome="malformed", detail="body is not JSON")
            self._send(400, json.dumps(
                _error(None, -32700, "parse error")).encode())
            return

        # Named before dispatch so a call that dies mid-flight is still
        # attributable to a tool and a digest.
        called = payload_tools(payload)
        if called:
            self._note(tool=",".join(name for name, _ in called[:5])[:120],
                       digest=called[0][1] if len(called) == 1 else
                       args_digest([digest for _, digest in called]))

        response = handle_payload(payload, peer=self._peer(),
                                  xff=self._forwarded_for())
        if response is None:
            # Notification-only body. 202 with no content is what the
            # streamable HTTP spec asks for.
            self._send(202)
            return

        body = json.dumps(response)
        accept = (self.headers.get("Accept") or "").lower()
        if "text/event-stream" in accept:
            # The path a real MCP client takes. One `message` event, then
            # the stream ends -- there is nothing further to push.
            data = ("event: message\ndata: " + body + "\n\n").encode()
            self._send(200, data, content_type="text/event-stream",
                       extra={"Cache-Control": "no-cache"})
        else:
            self._send(200, body.encode(), content_type="application/json")


def make_server(host=DEFAULT_HOST, port=DEFAULT_PORT, token=None):
    """A bound, not-yet-serving HTTP server.

    Refuses to exist without a token. Starting an unauthenticated shell
    endpoint and relying on "it is only on loopback" is precisely the
    thing that becomes untrue the moment the funnel goes up."""
    token = token if token is not None else read_token()
    if not token:
        raise RemoteHandsError(
            f"no token: {token_path()} is missing or empty. Run "
            f"`modelctl remote-hands install` first.")
    handler = type("_BoundHandler", (_Handler,), {"token": token})
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    httpd.daemon_threads = True
    return httpd


def serve(host=DEFAULT_HOST, port=DEFAULT_PORT, token=None):
    httpd = make_server(host, port, token)
    print(f"{SERVER_NAME} listening on http://{host}:{port}{MCP_PATH} "
          f"(loopback only; exposure is a separate step)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


# --- systemd ---------------------------------------------------------------

UNIT_TEMPLATE = """\
[Unit]
Description=modelctl remote hands (MCP over HTTP, loopback only)
After=network.target

[Service]
ExecStart="{python}" -m modelctl_remote_hands --host {host} --port {port}
WorkingDirectory={workdir}
{env_lines}Restart=on-failure
RestartSec=3

# No [Install] section on purpose: this unit is never enabled, so it
# cannot come back after a reboot. `modelctl remote-hands on` starts it
# for the session and `off` stops it. Exposure that survives a reboot
# without anyone asking is the failure mode this avoids.
"""


def unit_path():
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "systemd" / "user" / SERVICE_NAME


def render_unit(host=DEFAULT_HOST, port=DEFAULT_PORT, python=None,
                workdir=None, env=None):
    env = os.environ if env is None else env
    # The installing shell's MODELCTL_* environment is part of the
    # install, the same reasoning `web install` uses: a service reading a
    # different state dir than the CLI that installed it is two control
    # planes that disagree about where the audit log lives.
    #
    # Except the secrets. A unit file under ~/.config is 0644 and shows
    # up in full in `systemctl --user cat`; the token belongs in its own
    # 0600 file, which the service reads at startup.
    lines = [f'Environment="{k}={env[k]}"'
             for k in sorted(env)
             if k.startswith("MODELCTL_") and not k.endswith("_TOKEN")]
    return UNIT_TEMPLATE.format(
        python=python or sys.executable,
        host=host, port=int(port),
        workdir=workdir or Path(__file__).resolve().parent,
        env_lines="".join(line + "\n" for line in lines))


def _systemctl(*args, timeout=30):
    argv = ["systemctl", "--user"] + list(args)
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 127, "", f"{type(e).__name__}: {e}"


def service_active():
    rc, out, _ = _systemctl("is-active", SERVICE_NAME)
    return out == "active"


def service_uptime():
    """Seconds the unit has been running, or None. Read from systemd
    rather than tracked here so a manual `systemctl --user start` is not
    invisible to `status`."""
    rc, out, _ = _systemctl("show", SERVICE_NAME,
                            "--property=ActiveEnterTimestampMonotonic")
    if rc != 0 or "=" not in out:
        return None
    try:
        started_us = int(out.split("=", 1)[1])
    except ValueError:
        return None
    if started_us <= 0:
        return None
    return max(0.0, time.clock_gettime(time.CLOCK_MONOTONIC)
               - started_us / 1_000_000)


def port_listening(host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=2):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


# --- tailscale funnel ------------------------------------------------------

def _tailscale(*args, timeout=60):
    binary = shutil.which("tailscale") or "/usr/bin/tailscale"
    try:
        p = subprocess.run([binary] + list(args), capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 127, "", f"{type(e).__name__}: {e}"


def funnel_config(runner=None):
    """The current serve/funnel config as a dict, or {}.

    Read before every teardown: `tailscale funnel reset` clears the whole
    serve config, so it must not run while something that is not ours is
    configured."""
    run = runner or _tailscale
    rc, out, _ = run("funnel", "status", "--json")
    if rc != 0 or not out:
        return {}
    try:
        data = json.loads(out)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def funnel_targets(config=None, runner=None):
    """Every local target the serve config proxies to, as strings.

    Used to tell "the funnel we put up" from "someone else's serve
    config", which is the difference between a safe reset and clobbering
    an unrelated service."""
    config = funnel_config(runner) if config is None else config
    out = []
    for handlers in (config.get("Web") or {}).values():
        for handler in ((handlers or {}).get("Handlers") or {}).values():
            proxy = (handler or {}).get("Proxy")
            if proxy:
                out.append(str(proxy))
    for port, target in (config.get("TCP") or {}).items():
        if not isinstance(target, dict) or not target:
            continue
        # `{"HTTPS": true}` on 443 is not a publication -- it is how the
        # Web handlers above are reached, and every funnel has one. It
        # used to be counted as a target, which made it look foreign to
        # `funnel_off`: the kill switch refused its own config and asked
        # for --force, and --force is precisely the flag that would
        # clobber a genuinely foreign one. A raw TCP proxy is a real
        # second publication and still counts.
        forward = target.get("TCPForward")
        if not forward:
            continue
        out.append(f"tcp:{port} -> {forward}")
    return out


def funnel_exposed(port=DEFAULT_PORT, config=None, runner=None):
    """Whether the funnel currently points at our port."""
    targets = funnel_targets(config=config, runner=runner)
    return any(f":{int(port)}" in t for t in targets)


def public_url(port=DEFAULT_PORT, runner=None):
    """The URL to paste into the connector dialog, or None.

    Built from tailscaled's own idea of this node's DNS name, so it is
    the name the certificate will actually be issued for."""
    run = runner or _tailscale
    rc, out, _ = run("status", "--json")
    if rc != 0 or not out:
        return None
    try:
        name = ((json.loads(out).get("Self") or {}).get("DNSName") or "")
    except ValueError:
        return None
    name = name.strip().rstrip(".")
    return f"https://{name}{MCP_PATH}" if name else None


def funnel_on(port=DEFAULT_PORT, runner=None):
    """Expose the loopback port. Returns (ok, detail)."""
    run = runner or _tailscale
    rc, out, err = run("funnel", "--bg", "--yes", str(int(port)))
    if rc != 0:
        detail = (err or out or "").strip()
        if "operator" in detail.lower() or "access denied" in detail.lower():
            detail += ("\nThis needs operator rights. Run the one-time root "
                       "setup: sudo modelctl/docs/fleet/rig-headless-setup.sh")
        return False, detail
    return True, (out or "").strip()


def funnel_off(port=DEFAULT_PORT, runner=None, force=False):
    """Tear the funnel down. Returns (ok, detail).

    Refuses when the serve config contains something that is not ours:
    `reset` is the only teardown this tailscale version offers and it
    clears everything, so the alternative to refusing is silently taking
    down whatever else was published."""
    run = runner or _tailscale
    config = funnel_config(runner=run)
    targets = funnel_targets(config=config)
    if not targets:
        return True, "no funnel configured"
    foreign = [t for t in targets if f":{int(port)}" not in t]
    if foreign and not force:
        return False, ("the serve config also publishes something else "
                       "(" + ", ".join(foreign) + ") and `tailscale funnel "
                       "reset` clears all of it. Re-run with --force if that "
                       "is what you want.")
    rc, out, err = run("funnel", "reset")
    if rc != 0:
        return False, (err or out or "").strip()
    return True, "funnel reset"


# --- status ----------------------------------------------------------------

def status(port=DEFAULT_PORT, host=DEFAULT_HOST, runner=None):
    """Everything `modelctl remote-hands status` prints, as data."""
    config = funnel_config(runner=runner)
    exposed = funnel_exposed(port=port, config=config)
    stats = audit_stats()
    return {
        "exposure": "EXPOSED" if exposed else "hidden",
        # The one-glance answer to "is anything hammering it". Read out
        # of the audit log rather than out of the limiters, because the
        # limiters live in the service process and this runs in the CLI's
        # -- an in-memory counter read from here would always say zero.
        "security": {
            "funnel": "up" if exposed else "down",
            "audit_lines": stats["lines"],
            "auth_failures_last_hour": stats["auth_failures_last_hour"],
            "last_auth_failure": stats["last_auth_failure"],
        },
        "public_url": public_url(port=port, runner=runner) if exposed else None,
        "funnel_targets": funnel_targets(config=config),
        "service": SERVICE_NAME,
        "service_active": service_active(),
        "uptime_seconds": service_uptime(),
        "listening": port_listening(host=host, port=port),
        "bind": f"{host}:{int(port)}",
        "token_present": bool(read_token()),
        "token_path": str(token_path()),
        "audit_path": str(audit_path()),
        "allow_roots": [str(r) for r in allow_roots()],
        "oauth": oauth.grant_summary(),
        "recent": audit_tail(5),
    }


def format_duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def main(argv=None):
    """`python -m modelctl_remote_hands` -- the unit's ExecStart."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    try:
        return serve(args.host, args.port)
    except RemoteHandsError as e:
        print(f"remote-hands: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
