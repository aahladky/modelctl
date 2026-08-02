"""OAuth for the remote-hands connector, because header auth is a beta.

claude.ai's Add-custom-connector dialog offers request-header auth
(`static_headers`) only to accounts in the beta. This one is not in it:
the dialog shows OAuth Client ID / Client Secret and nothing else, and
those fields configure a client for *your* authorization server -- they
do not authenticate anyone to the rig. So the rig has to be an
authorization server.

That is smaller than it sounds, because of CIMD. Anthropic's own sample
does not implement Dynamic Client Registration; it advertises **Client
ID Metadata Documents** instead, where the `client_id` is itself an
HTTPS URL that dereferences to the client's registration metadata.
There is no client database and no `POST /register` round trip in the
common path -- at `/authorize` the server fetches the `client_id` URL,
checks the document is self-referential, and checks the requested
`redirect_uri` against it. Claude selects CIMD only when the metadata
advertises BOTH `client_id_metadata_document_supported: true` AND
`"none"` in `token_endpoint_auth_methods_supported` (its CIMD client
authenticates as a public client), and falls back to looking for a
`registration_endpoint` otherwise -- so a minimal DCR endpoint is here
as well, purely so a CIMD selection failure is not a dead connector.

Who authenticates at the consent screen: Aaron, with the same 32-byte
token the static-header path uses. It is posted in a form body, never
placed in a URL. That keeps one credential for the whole feature and
means turning the beta on later changes nothing about who can get in.

Verified against Anthropic's connector docs on 2026-08-02. Two of their
constraints shape the code more than the RFCs do:

* Discovery is cached globally, keyed by URL, for about five minutes.
  Metadata is computed per-request from the Host header rather than
  written to a file, so there is no second copy to go stale.
* The token endpoint gets ten seconds (thirty for refresh). Everything
  here is local except the one CIMD fetch, which is bounded.

This module must stay a leaf: no modelctl imports.
"""
import base64
import hashlib
import http.client
import ipaddress
import json
import os
import secrets
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import modelctl_fsutil
from modelctl_paths import STATE_DIR

# One scope. The tools are all-or-nothing -- a shell is not meaningfully
# divisible into read and write -- and advertising a scope tree that
# grants nothing different would be decoration on a consent screen.
SCOPE = "rig"

ACCESS_TTL = 3600
REFRESH_TTL = 30 * 24 * 3600
# Long enough for a redirect and a token call, short enough that a code
# captured from a proxy log is dead before it can be replayed.
CODE_TTL = 60

# Anthropic's callback for the hosted surfaces. Not an allowlist -- the
# CIMD document decides -- but it is what a working connection uses, and
# `status` prints it so a mismatch is visible.
CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback"

CIMD_TIMEOUT = 10
CIMD_MAX_BYTES = 128 * 1024
# The CIMD fetch is the one outbound request an unauthenticated caller
# can make this server perform, and the caller names the destination. 443
# only: a client_id pointing at :9292 or :22 is not a metadata document,
# it is a port scan with a JSON parser attached.
CIMD_ALLOWED_PORTS = (443,)

# An unauthenticated endpoint that writes to disk needs a ceiling. DCR is
# the fallback path, so this only fills up if CIMD is failing and
# something is retrying; oldest-out keeps the live client working.
MAX_REGISTERED_CLIENTS = 50

STATE_VERSION = 1

# Authorization codes live in memory only. They are valid for a minute,
# they must not survive a restart (a code issued before a restart is a
# code whose PKCE challenge is gone), and writing them to disk would put
# a bearer-equivalent secret in a file for no benefit.
_CODES = {}


class OAuthError(Exception):
    """An RFC 6749 error. `code` is the error field the client parses;
    getting these right matters -- Claude only retries a refresh on
    `invalid_grant`, and a custom code strands the connection."""

    def __init__(self, code, description="", status=400):
        super().__init__(description or code)
        self.code = code
        self.description = description
        self.status = status

    def body(self):
        out = {"error": self.code}
        if self.description:
            out["error_description"] = self.description
        return out


# --- state -----------------------------------------------------------------

def state_path():
    override = os.environ.get("MODELCTL_REMOTE_HANDS_OAUTH_PATH", "").strip()
    return Path(override) if override else STATE_DIR / "remote-hands-oauth.json"


def _read_state():
    try:
        data = json.loads(state_path().read_text())
    except (OSError, ValueError):
        return {"version": STATE_VERSION, "clients": {}, "grants": {},
                "access": {}}
    if not isinstance(data, dict):
        return {"version": STATE_VERSION, "clients": {}, "grants": {},
                "access": {}}
    for key in ("clients", "grants", "access"):
        if not isinstance(data.get(key), dict):
            data[key] = {}
    return data


def _write_state(state):
    state["version"] = STATE_VERSION
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    modelctl_fsutil.atomic_write_text(path, json.dumps(state, indent=2))
    os.chmod(path, 0o600)


def _digest(token):
    """Tokens are stored as hashes.

    The grant file would otherwise be a list of live bearer credentials
    at rest, and it has to be readable by the process that validates
    them -- a hash is enough for that job and useless to anyone who
    reads the file."""
    return hashlib.sha256(token.encode()).hexdigest()


def prune(state=None, now=None):
    """Drop expired access and refresh grants. Returns the state.

    Called on every issue and every validation, because the alternative
    is a file that only ever grows and a revoked-by-expiry token that
    still appears in `status`."""
    now = time.time() if now is None else now
    state = _read_state() if state is None else state
    for bucket in ("access", "grants"):
        state[bucket] = {k: v for k, v in state[bucket].items()
                         if float(v.get("expires") or 0) > now}
    return state


# --- discovery documents ---------------------------------------------------

def base_url(host=None, scheme=None):
    """The origin this server is reachable at, as Claude will see it.

    Computed from the request's Host header rather than stored, because
    the protected-resource document's `resource` field must match the
    URL the user typed into the dialog exactly, and the funnel hostname
    is the only thing that knows what that is. The env override exists
    for the case where a proxy rewrites Host."""
    override = os.environ.get("MODELCTL_REMOTE_HANDS_BASE_URL", "").strip()
    if override:
        return override.rstrip("/")
    host = (host or "127.0.0.1:9294").strip()
    if scheme is None:
        # Funnel terminates TLS and forwards plain HTTP to us, so the
        # scheme cannot be read off the connection; the hostname is what
        # distinguishes "reached through the funnel" from "reached on
        # loopback by a smoke test".
        loopback = host.startswith("127.0.0.1") or host.startswith("localhost")
        scheme = "http" if loopback else "https"
    return f"{scheme}://{host}"


def protected_resource_metadata(base, mcp_path="/mcp"):
    """RFC 9728. `resource` must equal the MCP URL exactly as entered."""
    return {
        "resource": f"{base}{mcp_path}",
        "authorization_servers": [base],
        "scopes_supported": [SCOPE],
        "bearer_methods_supported": ["header"],
    }


def authorization_server_metadata(base):
    """RFC 8414, shaped so Claude selects CIMD.

    Both `client_id_metadata_document_supported` and `"none"` in
    `token_endpoint_auth_methods_supported` are required for that -- the
    CIMD client is a public client and must be able to redeem a code
    with PKCE and no secret. `registration_endpoint` stays advertised as
    the documented fallback."""
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "registration_endpoint": f"{base}/register",
        "scopes_supported": [SCOPE],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none"],
        "code_challenge_methods_supported": ["S256"],
        "client_id_metadata_document_supported": True,
    }


def www_authenticate(base, mcp_path="/mcp", error="invalid_token",
                     description="Authentication required"):
    """The 401 header that starts the whole flow.

    Claude only runs OAuth on a transport-level 401 carrying this; a 200
    wrapping an isError tool result is passed to the model as text and
    the user never sees a Connect card."""
    return (f'Bearer error="{error}", error_description="{description}", '
            f'resource_metadata="{base}/.well-known/'
            f'oauth-protected-resource{mcp_path}", scope="{SCOPE}"')


# --- clients ---------------------------------------------------------------

def _origin(url):
    parts = urllib.parse.urlsplit(url)
    return (parts.scheme, parts.hostname, parts.port)


def _is_loopback(url):
    host = urllib.parse.urlsplit(url).hostname or ""
    return host in ("127.0.0.1", "::1", "localhost")


def redirect_uri_allowed(client_id, redirect_uri, registered):
    """Whether this client may be sent to this redirect_uri.

    Three rules, and the last two are the ones that are easy to get
    wrong.

    A loopback redirect is compared with the PORT IGNORED, because a
    native client binds an ephemeral port at runtime (RFC 8252 7.3);
    Claude Code declares `http://localhost/callback` and arrives on
    whatever port it got.

    The same-origin requirement applies ONLY to CIMD clients, whose
    client_id is a URL. A CIMD document is self-asserted, so without it
    any URL serving a self-referential document could name someone
    else's callback and turn this endpoint into an open redirect. A
    DCR-registered client_id is an opaque id with no origin to compare
    against, and its redirect_uris were registered explicitly rather
    than asserted at authorize time -- holding it to a same-origin test
    would reject every registration this server itself issued."""
    if not redirect_uri:
        return False
    for candidate in registered or []:
        if candidate == redirect_uri:
            break
        if (_is_loopback(candidate) and _is_loopback(redirect_uri)
                and _origin(candidate)[:2] == _origin(redirect_uri)[:2]
                and urllib.parse.urlsplit(candidate).path
                == urllib.parse.urlsplit(redirect_uri).path):
            break
    else:
        return False
    if _is_loopback(redirect_uri):
        return True
    if not client_id.startswith("https://"):
        return True
    return _origin(client_id)[:2] == _origin(redirect_uri)[:2]


# --- SSRF guard on the CIMD fetch ------------------------------------------
# `client_id` is an https URL supplied by an unauthenticated caller, and
# this server dereferences it. Size and time were already bounded and the
# document has to be self-referential, so nothing comes *back* to the
# caller -- but the destination was not bounded, which makes the endpoint
# a blind-SSRF primitive against everything the rig can reach that the
# internet cannot: llama-swap on 9292, the console on 9293, the tailnet,
# a cloud metadata service. The fix is destination vetting, and it has to
# survive DNS rebinding: a name that answers public on the check and
# private on the fetch defeats a check-then-fetch. So the vetted address
# is the address connected to, with the certificate still validated
# against the original hostname.


def _ip_block_reason(ip):
    """The class an address is blocked as, or None if it is routable.

    Ordered most-specific first because IPv4 `is_private` is true for
    loopback and link-local too, and a report saying "private" when the
    answer is "127.0.0.1" is a worse log line. v6 addresses that embed a
    v4 one (4-in-6, 6to4, Teredo) are unwrapped first: `::ffff:127.0.0.1`
    is a loopback address wearing a hat."""
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip.teredo:
            embedded = ip.teredo[1]
        if embedded is not None:
            reason = _ip_block_reason(embedded)
            return f"{reason} via {ip}" if reason else None
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if getattr(ip, "is_site_local", False):
        return "site-local"
    if ip.is_private:
        return "private"
    if not ip.is_global:
        return "non-global"
    return None


def vet_cimd_target(client_id, resolver=None):
    """(host, port, [vetted addresses]) for a client_id URL, or raise.

    EVERY address the name resolves to has to be routable, not just the
    first: a name that answers with one public and one private address
    would otherwise be a coin flip, and the caller picks the coin."""
    parts = urllib.parse.urlsplit(client_id)
    if parts.scheme != "https":
        raise OAuthError("invalid_client", "client_id must be an https URL")
    if parts.username or parts.password:
        raise OAuthError("invalid_client",
                         "client_id must not carry credentials")
    try:
        port = parts.port
    except ValueError:
        raise OAuthError("invalid_client", "client_id has an invalid port")
    port = 443 if port is None else port
    if port not in CIMD_ALLOWED_PORTS:
        raise OAuthError("invalid_client",
                         f"client_id port {port} is not allowed "
                         f"(https on {CIMD_ALLOWED_PORTS[0]} only)")
    host = parts.hostname
    if not host:
        raise OAuthError("invalid_client", "client_id has no host")
    resolve = resolver or socket.getaddrinfo
    try:
        infos = resolve(host, port, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as e:
        raise OAuthError("invalid_client", f"cannot resolve {host}: {e}")
    addresses = []
    for info in infos or []:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except (IndexError, TypeError, ValueError):
            continue
        reason = _ip_block_reason(ip)
        if reason:
            raise OAuthError(
                "invalid_client",
                f"client_id host {host} resolves to a {reason} address; "
                f"only publicly routable destinations are fetched")
        addresses.append(ip)
    if not addresses:
        raise OAuthError("invalid_client", f"cannot resolve {host}")
    return host, port, addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to one vetted IP, with the certificate checked against the
    name. Connecting by address is what closes the rebind window; passing
    the hostname as `server_hostname` is what keeps that from also
    disabling verification."""

    def __init__(self, host, *args, pinned_ip=None, **kwargs):
        super().__init__(host, *args, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection(
            (str(self._pinned_ip), self.port), self.timeout,
            self.source_address)
        self.sock = self._context.wrap_socket(self.sock,
                                              server_hostname=self.host)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):

    def __init__(self, pinned_ip, context=None):
        super().__init__(context=context or ssl.create_default_context())
        self._pinned_ip = pinned_ip

    def https_open(self, req):
        return self.do_open(self._new_connection, req)

    def _new_connection(self, host, **kwargs):
        return _PinnedHTTPSConnection(host, pinned_ip=self._pinned_ip,
                                      context=self._context, **kwargs)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects are not followed.

    Following one would re-open the destination question after it was
    settled, and nothing is lost by refusing: the document must name the
    exact URL it was fetched from, so a redirect that changes the URL
    fails the self-referentiality check anyway. Returning None here makes
    urllib raise the 3xx as an HTTPError instead of chasing it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _pinned_opener(pinned_ip):
    return urllib.request.build_opener(
        _PinnedHTTPSHandler(pinned_ip), _NoRedirect).open


def cimd_request(client_id):
    """The GET this server makes for a client_id document.

    The User-Agent is not decoration. urllib's default is
    `Python-urllib/<version>`, and the CDN in front of claude.ai answers
    that with 403 Forbidden -- observed 2026-08-02 against
    https://claude.ai/oauth/mcp-oauth-client-metadata, where curl got
    200 and urllib got 403 with nothing else changed. Any real
    User-Agent is accepted; this one names the service so the request is
    identifiable at the other end. Accept is sent because the document
    is JSON and asking for it is correct regardless."""
    req = urllib.request.Request(client_id)
    req.add_header("User-Agent", f"{__name__.replace('_', '-')}/1.0")
    req.add_header("Accept", "application/json")
    return req


def fetch_cimd(client_id, opener=None, resolver=None):
    """Resolve a Client ID Metadata Document.

    Self-referentiality is the only thing that makes this safe to trust:
    the document must claim the exact URL it was served from, so a
    client_id cannot point at somebody else's document and inherit its
    redirect_uris.

    The destination is vetted before anything is sent, and the vetted
    address is the one connected to -- see `vet_cimd_target`. An explicit
    `opener` skips the pinning (it is the test seam and the caller has
    already chosen the transport), but never skips the vetting."""
    if not client_id.startswith("https://"):
        raise OAuthError("invalid_client",
                         "client_id must be an https URL or a registered id")
    _, _, addresses = vet_cimd_target(client_id, resolver=resolver)
    try:
        open_url = opener or _pinned_opener(addresses[0])
        with open_url(cimd_request(client_id), timeout=CIMD_TIMEOUT) as resp:
            raw = resp.read(CIMD_MAX_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise OAuthError("invalid_client",
                         f"cannot fetch client_id document: {e}")
    if len(raw) > CIMD_MAX_BYTES:
        raise OAuthError("invalid_client", "client_id document is too large")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise OAuthError("invalid_client", "client_id document is not JSON")
    if not isinstance(doc, dict) or doc.get("client_id") != client_id:
        raise OAuthError("invalid_client",
                         "client_id document is not self-referential")
    return doc


def resolve_client(client_id, redirect_uri, opener=None, state=None,
                   resolver=None):
    """(client_name, redirect_uris) for a client, or raise.

    CIMD first, then the DCR store. A registered client_id is an opaque
    id and will never look like a URL, so the two cannot collide."""
    if not client_id:
        raise OAuthError("invalid_request", "client_id is required")
    if client_id.startswith("https://"):
        doc = fetch_cimd(client_id, opener=opener, resolver=resolver)
        registered = doc.get("redirect_uris") or []
        name = doc.get("client_name") or ""
    else:
        state = _read_state() if state is None else state
        entry = (state.get("clients") or {}).get(client_id)
        if not entry:
            raise OAuthError("invalid_client", "unknown client_id")
        registered = entry.get("redirect_uris") or []
        name = entry.get("client_name") or ""
    if not redirect_uri_allowed(client_id, redirect_uri, registered):
        raise OAuthError("invalid_request",
                         "redirect_uri is not registered for this client")
    return name, registered


def register_client(payload, now=None):
    """Minimal RFC 7591. Public clients only.

    Only reached when Claude does not select CIMD. Unauthenticated by
    necessity, so it is capped and oldest-out: the failure mode to avoid
    is a store that grows without bound and eventually evicts the client
    that is actually working."""
    now = time.time() if now is None else now
    redirect_uris = payload.get("redirect_uris")
    if not isinstance(redirect_uris, list) or not redirect_uris:
        raise OAuthError("invalid_redirect_uri", "redirect_uris is required")
    for uri in redirect_uris:
        if not isinstance(uri, str) or not uri:
            raise OAuthError("invalid_redirect_uri", "bad redirect_uri")
        if not (uri.startswith("https://") or _is_loopback(uri)):
            raise OAuthError("invalid_redirect_uri",
                             "redirect_uri must be https or loopback")
    state = _read_state()
    clients = state.setdefault("clients", {})
    if len(clients) >= MAX_REGISTERED_CLIENTS:
        oldest = sorted(clients.items(),
                        key=lambda kv: kv[1].get("registered_at") or 0)
        for key, _ in oldest[:len(clients) - MAX_REGISTERED_CLIENTS + 1]:
            clients.pop(key, None)
    client_id = "rh-" + secrets.token_hex(16)
    clients[client_id] = {
        "redirect_uris": list(redirect_uris),
        "client_name": str(payload.get("client_name") or "")[:120],
        "registered_at": now,
    }
    _write_state(state)
    return {
        "client_id": client_id,
        "client_id_issued_at": int(now),
        "redirect_uris": list(redirect_uris),
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }


# --- authorization code + PKCE --------------------------------------------

def _b64url(raw):
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def verify_pkce(verifier, challenge):
    """S256 only. `plain` is accepted by neither the MCP spec nor this."""
    if not verifier or not challenge:
        return False
    expected = _b64url(hashlib.sha256(verifier.encode()).digest())
    return secrets.compare_digest(expected, challenge)


def create_code(client_id, redirect_uri, challenge, scope=SCOPE, now=None):
    now = time.time() if now is None else now
    code = secrets.token_urlsafe(32)
    _CODES[code] = {"client_id": client_id, "redirect_uri": redirect_uri,
                    "challenge": challenge, "scope": scope,
                    "expires": now + CODE_TTL}
    # Opportunistic sweep: codes are short-lived and this dict is the
    # only thing holding them, so an abandoned flow should not leak.
    for key, entry in list(_CODES.items()):
        if entry["expires"] <= now:
            _CODES.pop(key, None)
    return code


def redeem_code(code, client_id, redirect_uri, verifier, now=None):
    """One-time. Popped before anything can fail, so a code cannot be
    retried after a rejected PKCE check."""
    now = time.time() if now is None else now
    entry = _CODES.pop(code or "", None)
    if entry is None:
        raise OAuthError("invalid_grant", "unknown or already-used code")
    if entry["expires"] <= now:
        raise OAuthError("invalid_grant", "code has expired")
    if entry["client_id"] != client_id:
        raise OAuthError("invalid_grant", "code was issued to another client")
    if redirect_uri and entry["redirect_uri"] != redirect_uri:
        raise OAuthError("invalid_grant", "redirect_uri does not match")
    if not verify_pkce(verifier, entry["challenge"]):
        raise OAuthError("invalid_grant", "PKCE verification failed")
    return entry


# --- tokens ----------------------------------------------------------------

def issue_tokens(client_id, scope=SCOPE, now=None):
    now = time.time() if now is None else now
    access = secrets.token_urlsafe(32)
    refresh = secrets.token_urlsafe(32)
    state = prune(now=now)
    state["access"][_digest(access)] = {
        "client_id": client_id, "scope": scope, "expires": now + ACCESS_TTL}
    state["grants"][_digest(refresh)] = {
        "client_id": client_id, "scope": scope, "issued": now,
        "expires": now + REFRESH_TTL}
    _write_state(state)
    return {"access_token": access, "token_type": "Bearer",
            "expires_in": ACCESS_TTL, "refresh_token": refresh,
            "scope": scope}


def refresh_tokens(refresh_token, client_id=None, now=None):
    """Rotate. The old refresh token is invalidated in the same write
    that issues the new one, which is what the MCP spec requires of
    public clients."""
    now = time.time() if now is None else now
    state = prune(now=now)
    entry = state["grants"].pop(_digest(refresh_token or ""), None)
    if entry is None:
        # Must be invalid_grant: any other code and Claude gives up on
        # the connection instead of re-authorizing.
        _write_state(state)
        raise OAuthError("invalid_grant", "unknown or expired refresh token")
    if client_id and entry.get("client_id") != client_id:
        _write_state(state)
        raise OAuthError("invalid_grant", "refresh token belongs to another "
                                          "client")
    _write_state(state)
    return issue_tokens(entry["client_id"], entry.get("scope") or SCOPE,
                        now=now)


def valid_access_token(token, now=None):
    """Whether an OAuth bearer is live. Constant-time is not needed here:
    the lookup is by digest, so no comparison of the secret happens."""
    if not token:
        return False
    now = time.time() if now is None else now
    state = _read_state()
    entry = state["access"].get(_digest(token))
    return bool(entry) and float(entry.get("expires") or 0) > now


def revoke_all():
    """Drop every issued grant. What `remote-hands off` could call and
    what a rotated operator token should imply."""
    state = _read_state()
    count = len(state["access"]) + len(state["grants"])
    state["access"], state["grants"] = {}, {}
    _write_state(state)
    return count


def grant_summary(now=None):
    now = time.time() if now is None else now
    state = prune(now=now)
    return {"clients": len(state.get("clients") or {}),
            "access_tokens": len(state["access"]),
            "refresh_tokens": len(state["grants"])}


# --- the consent screen ----------------------------------------------------

CONSENT_CSS = """
body{font:16px/1.5 system-ui,sans-serif;max-width:34rem;margin:4rem auto;
padding:0 1.5rem;background:#111;color:#eee}
h1{font-size:1.3rem;margin:0 0 1.5rem}
.rp{background:#1c1c1c;border:1px solid #333;border-radius:6px;
padding:1rem;margin:1.5rem 0}
.rp b{font-family:ui-monospace,monospace;color:#fff}
label{display:block;margin:1.5rem 0 .4rem;font-size:.9rem;color:#aaa}
input{width:100%;padding:.6rem;font-family:ui-monospace,monospace;
font-size:.95rem;background:#000;color:#eee;border:1px solid #444;
border-radius:4px;box-sizing:border-box}
button{margin-top:1.2rem;padding:.6rem 1.4rem;font-size:1rem;
background:#eee;color:#111;border:0;border-radius:4px;cursor:pointer}
.warn{color:#f0a;font-size:.9rem}
ul{color:#bbb;font-size:.92rem;padding-left:1.2rem}
"""


def consent_html(client_id, redirect_uri, params, error=None):
    """The page Aaron sees in the popup.

    It displays the HOST OF THE client_id URL as the relying party, not
    the document's `client_name`: the document is self-asserted, so the
    name is whatever it says it is, while the host is the one fact the
    fetch actually established."""
    host = urllib.parse.urlsplit(client_id).hostname or client_id
    hidden = "".join(
        f'<input type="hidden" name="{_esc(k)}" value="{_esc(v)}">'
        for k, v in params.items() if v is not None)
    banner = (f'<p class="warn">{_esc(error)}</p>' if error else "")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Authorize rig access</title><style>{CONSENT_CSS}</style></head><body>
<h1>Authorize rig access</h1>
<div class="rp">Requested by <b>{_esc(host)}</b><br>
<span style="color:#888;font-size:.9rem">redirecting to
{_esc(redirect_uri)}</span></div>
<p>Granting this gives the connected Claude session:</p>
<ul>
<li>read, write and list under <code>/home/aaron/workspace</code>,
<code>/home/aaron/models</code>, <code>/home/aaron/services</code></li>
<li>a shell on this machine as <code>aaron</code></li>
</ul>
{banner}
<form method="post" action="/authorize">{hidden}
<label for="tok">Paste the remote-hands token to approve</label>
<input id="tok" name="operator_token" type="password" autocomplete="off"
autofocus>
<button type="submit">Approve</button>
</form>
<p style="color:#666;font-size:.85rem">Close this window to deny.</p>
</body></html>"""


def _esc(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def redirect_with_code(redirect_uri, code, state_param):
    """Append the code to the client's callback.

    Built with urlencode rather than string concatenation because a
    redirect_uri may already carry a query, and a hand-built `?` there
    produces a URL that silently drops the code."""
    parts = urllib.parse.urlsplit(redirect_uri)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.append(("code", code))
    if state_param:
        query.append(("state", state_param))
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path,
         urllib.parse.urlencode(query), parts.fragment))


def authorize_params(query):
    """Pull the parameters forward through the consent POST.

    `state` and `resource` are carried verbatim: dropping `state` breaks
    the client's CSRF check, and dropping `resource` breaks audience
    binding on servers that enforce it."""
    return {k: query.get(k) for k in
            ("client_id", "redirect_uri", "state", "code_challenge",
             "code_challenge_method", "scope", "resource")}


def validate_authorize(params):
    """Everything that must be true before a consent screen is shown.

    Checked before rendering, not after submitting: a page that asks for
    the operator token and then rejects the request has already asked
    for the token."""
    if params.get("response_type") not in (None, "code"):
        raise OAuthError("unsupported_response_type",
                         "only response_type=code is supported")
    method = params.get("code_challenge_method")
    if method != "S256":
        raise OAuthError("invalid_request",
                         "code_challenge_method must be S256")
    if not params.get("code_challenge"):
        raise OAuthError("invalid_request", "code_challenge is required")
    if not params.get("redirect_uri"):
        raise OAuthError("invalid_request", "redirect_uri is required")
    return True
