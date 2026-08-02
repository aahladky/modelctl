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
          detail=None, now=None):
    """Append one audit record. Never raises.

    Appended with O_APPEND in a single write so concurrent handler
    threads interleave whole lines rather than halves, and created 0600
    because it records what was touched and when."""
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S",
                            time.localtime(now if now else time.time())),
        "tool": tool,
        "args_sha256": digest if digest is not None else args_digest(args),
        "outcome": outcome,
    }
    if peer:
        entry["peer"] = peer
    if ms is not None:
        entry["ms"] = int(ms)
    if detail:
        entry["detail"] = str(detail)[:300]
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


def call_tool(name, args, peer=None):
    """Run one tool and audit the outcome. Returns (text, is_error).

    Every path through here writes exactly one audit line -- including
    the unknown-tool and refusal paths, which are the ones worth seeing
    in a log after the fact."""
    args = args if isinstance(args, dict) else {}
    digest = args_digest(args)
    started = time.time()
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        audit(name, "unknown-tool", digest=digest, peer=peer,
              ms=(time.time() - started) * 1000)
        return f"unknown tool: {name}", True
    try:
        text = impl(args)
    except RemoteHandsError as e:
        audit(name, "denied", digest=digest, peer=peer,
              ms=(time.time() - started) * 1000, detail=str(e))
        return str(e), True
    except Exception as e:                     # noqa: BLE001 -- see below
        # A tool raising anything else is a bug in this file, but a
        # traceback out of a handler thread would kill the request and
        # leave no record. The record is the point.
        audit(name, "error", digest=digest, peer=peer,
              ms=(time.time() - started) * 1000,
              detail=f"{type(e).__name__}: {e}")
        return f"{type(e).__name__}: {e}", True
    audit(name, "ok", digest=digest, peer=peer,
          ms=(time.time() - started) * 1000)
    return text, False


# --- JSON-RPC / MCP --------------------------------------------------------

def _result(msg_id, payload):
    return {"jsonrpc": "2.0", "id": msg_id, "result": payload}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}


def handle_message(msg, peer=None):
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
        text, is_error = call_tool(name, params.get("arguments"), peer=peer)
        return _result(msg_id, {"content": [{"type": "text", "text": text}],
                                "isError": is_error})
    if is_notification:
        # An unknown notification is not an error the client can act on,
        # and per JSON-RPC a notification never gets a response.
        return None
    return _error(msg_id, -32601, f"method not found: {method}")


def handle_payload(payload, peer=None):
    """A whole request body in, a whole response body out (or None).

    Arrays are accepted because protocol revisions up to 2025-03-26
    allowed batching; 2025-11-25 dropped it. Accepting one is cheaper
    than a client-version matrix."""
    if isinstance(payload, list):
        responses = [r for r in (handle_message(m, peer) for m in payload)
                     if r is not None]
        return responses or None
    return handle_message(payload, peer)


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
        header credential and got it wrong sees the same thing."""
        audit("-", "unauthorized", args=None, peer=self._peer(),
              detail=f"{self.command} {self.path}")
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
        if authorized(self.headers, self.token):
            return True
        presented = presented_credentials(self.headers)
        if any(oauth.valid_access_token(c) for c in presented):
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

    def _route_oauth_get(self, path, query):
        base = self._base()
        if path in ("/.well-known/oauth-protected-resource",
                    f"/.well-known/oauth-protected-resource{MCP_PATH}"):
            self._send_json(200, oauth.protected_resource_metadata(
                base, MCP_PATH))
            return True
        if path in ("/.well-known/oauth-authorization-server",
                    f"/.well-known/oauth-authorization-server{MCP_PATH}"):
            self._send_json(200, oauth.authorization_server_metadata(base))
            return True
        if path == "/authorize":
            self._authorize_page(query)
            return True
        return False

    def _authorize_page(self, query):
        params = {k: v[0] for k, v in query.items()}
        try:
            oauth.validate_authorize(params)
            oauth.resolve_client(params.get("client_id"),
                                 params.get("redirect_uri"))
        except oauth.OAuthError as e:
            # Rendered as a page, not redirected: an invalid client or
            # redirect_uri is exactly the case where bouncing the user
            # onward would be the vulnerability.
            audit("oauth/authorize", "denied", args=params, peer=self._peer(),
                  detail=f"{e.code}: {e.description}")
            self._send(400, f"<h1>{e.code}</h1><p>{oauth._esc(e.description)}"
                            f"</p>".encode(), content_type="text/html")
            return
        audit("oauth/authorize", "ok", args=params, peer=self._peer(),
              detail=params.get("client_id"))
        html = oauth.consent_html(params["client_id"], params["redirect_uri"],
                                  oauth.authorize_params(params))
        self._send(200, html.encode(), content_type="text/html")

    def _authorize_submit(self):
        form = self._form()
        supplied = form.pop("operator_token", "")
        try:
            oauth.validate_authorize(form)
            oauth.resolve_client(form.get("client_id"),
                                 form.get("redirect_uri"))
        except oauth.OAuthError as e:
            audit("oauth/consent", "denied", args=form, peer=self._peer(),
                  detail=f"{e.code}: {e.description}")
            self._send(400, f"<h1>{e.code}</h1>".encode(),
                       content_type="text/html")
            return
        token = self.token if self.token is not None else read_token()
        if not token or not supplied or not secrets.compare_digest(
                supplied, token):
            # Re-render rather than redirect: a denied consent must not
            # hand the client anything, and the operator may simply have
            # pasted the wrong thing.
            audit("oauth/consent", "denied", args=form, peer=self._peer(),
                  detail="operator token did not match")
            html = oauth.consent_html(form["client_id"], form["redirect_uri"],
                                      oauth.authorize_params(form),
                                      error="That token did not match.")
            self._send(401, html.encode(), content_type="text/html")
            return
        code = oauth.create_code(form["client_id"], form["redirect_uri"],
                                 form["code_challenge"])
        audit("oauth/consent", "ok", args=form, peer=self._peer(),
              detail=form.get("client_id"))
        self._send(302, b"", extra={
            "Location": oauth.redirect_with_code(form["redirect_uri"], code,
                                                 form.get("state"))})

    def _token_endpoint(self):
        form = self._form()
        grant = form.get("grant_type")
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
            audit(f"oauth/{grant or 'token'}", "denied", args=form,
                  peer=self._peer(), detail=f"{e.code}: {e.description}")
            self._send_json(e.status, e.body(),
                            extra={"Cache-Control": "no-store"})
            return
        audit(outcome, "ok", args=form, peer=self._peer())
        self._send_json(200, tokens, extra={"Cache-Control": "no-store"})

    def _register_endpoint(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(min(length, 64 * 1024)) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            payload = None
        try:
            if not isinstance(payload, dict):
                raise oauth.OAuthError("invalid_client_metadata",
                                       "body must be a JSON object")
            result = oauth.register_client(payload)
        except oauth.OAuthError as e:
            audit("oauth/register", "denied", args=payload, peer=self._peer(),
                  detail=f"{e.code}: {e.description}")
            self._send_json(e.status, e.body())
            return
        audit("oauth/register", "ok", args=payload, peer=self._peer(),
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
            self._send(400, json.dumps(
                _error(None, -32700, "parse error")).encode())
            return

        response = handle_payload(payload, peer=self._peer())
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
    for target in (config.get("TCP") or {}).values():
        if target:
            out.append(json.dumps(target, sort_keys=True))
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
    return {
        "exposure": "EXPOSED" if exposed else "hidden",
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
