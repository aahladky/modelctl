# Remote hands: rig access from a plain Claude chat

The desktop app is where the local MCP servers live, so today it is the
only Claude that can reach this box. Headless mode takes the desktop away
on purpose, and a phone never had it. This is the other route: a small
MCP server that claude.ai contacts as a **custom connector**, over the
public internet, authenticated by a bearer token in a request header.

Nothing here is reachable until you turn it on. Three separate things
have to be true, and `modelctl remote-hands on` is the only thing that
changes any of them:

| | default | turned on by |
|---|---|---|
| the systemd unit | installed, **never enabled** (no `[Install]` section) | `remote-hands on`, for the session |
| the listener | `127.0.0.1:9294` | never; it is loopback-only, always |
| the funnel | not configured | `remote-hands on` |

A reboot therefore always comes back hidden.

---

## One-time setup

```bash
sudo modelctl/docs/fleet/rig-headless-setup.sh
```

Step 3 of that script runs `tailscale set --operator=aaron`, without
which `tailscale funnel` needs root. Granting the operator publishes
nothing by itself — it grants the right to *configure* serve/funnel.
Revoke it with `tailscale set --operator=` (empty).

```bash
modelctl remote-hands install
```

Generates a 32-byte token in `~/.local/share/modelctl/remote-hands-token`
(mode 0600), writes the systemd user unit, and **prints the token once**.
It does not start anything and does not expose anything. Re-running it
leaves an existing token alone; `--rotate-token` replaces it, which
breaks any connector already configured with the old one.

---

## Adding the connector on claude.ai

Requirements verified against Anthropic's connector docs on 2026-08-02.
They drift — re-read them if something below does not match what the
dialog shows you.

There are two ways in, and which one you get depends on your account.
**OAuth is the one that works on this account today.**

1. `modelctl remote-hands on`. It prints the public URL, which looks
   like `https://aaron-2.tailb51646.ts.net/mcp`.
2. On claude.ai: **Settings → Connectors → Add custom connector**.
3. Paste the URL. Nothing goes in the query string — see below.
4. Leave OAuth Client ID / Client Secret **empty** — those configure a
   client for *your* authorization server, and this server registers
   Claude automatically.
5. **Add**, then enable it per-conversation with the **+** button.
6. The first tool call shows a **Connect** card. Clicking it opens the
   rig's own consent page, which asks for the remote-hands token.
   Paste it. That is the only time you need it.

### If the dialog has a Request headers section

Then `static_headers` is enabled on your account and you can skip the
consent step entirely:

1. Open **Request headers**, pick `authorization`.
2. Enter the value **including the scheme**: `Bearer <token>`, with the
   space. Claude sends the value verbatim and adds no prefix of its own,
   so a bare token arrives as `Authorization: <token>` and is rejected.
   `x-api-key` and `x-auth-token` take the bare token with no prefix.
3. Mark it **Required**, so a connection with no stored value fails
   rather than reaching the server unauthenticated.

Both credentials work at once; neither disables the other. Header auth
is in beta ("being slowly rolled out to customers; contact Anthropic for
early access"), which is why OAuth exists here at all.

### Why the token is never in the URL

The MCP authorization spec prohibits access tokens in the URI query
string, and Anthropic's docs call a credential in a connector URL a
security vulnerability: URLs land in server logs, proxies and browsing
history. There is no query-parameter mode in this server to fall back
on, deliberately — the fallback is the thing you would reach for on a
bad day.

---

## How the OAuth side works

The rig is its own authorization server. It is smaller than that sounds
because of **CIMD** (Client ID Metadata Document): Claude's `client_id`
is an HTTPS URL that dereferences to its own registration metadata, so
there is no client database and no registration round-trip in the normal
path. Claude selects CIMD only when the metadata advertises **both**
`client_id_metadata_document_supported: true` **and** `"none"` in
`token_endpoint_auth_methods_supported`; if either is missing it falls
back to looking for a `registration_endpoint`, so a minimal RFC 7591
`/register` is implemented too — purely so a CIMD selection failure is
not a dead connector.

| endpoint | auth |
|---|---|
| `/.well-known/oauth-protected-resource[/mcp]` | none (public by design) |
| `/.well-known/oauth-authorization-server` | none (public by design) |
| `GET /authorize` | none — renders the consent page |
| `POST /authorize` | the remote-hands token, in the form body |
| `POST /token` | the PKCE verifier for a code it issued |
| `POST /register` | none, capped at 50 clients, oldest-out |
| `/mcp` | the remote-hands token **or** an OAuth access token |

These are the only unauthenticated paths, and they have to be — a client
with no credential yet is exactly who fetches discovery and walks the
consent flow.

Details worth knowing when something breaks:

- **You** are the user being authenticated, at the consent screen, with
  the same 32-byte token the header path uses. It is posted in a form
  body, never a URL.
- The consent screen shows the **host of the `client_id` URL**, not the
  document's `client_name`. The document is self-asserted; the host is
  the one fact the fetch established.
- Non-loopback redirect URIs must be same-origin with the `client_id`
  URL. Without that rule a self-referential document could name someone
  else's callback and make `/authorize` an open redirect. Loopback URIs
  are matched with the **port ignored** (RFC 8252 §7.3), because native
  clients bind an ephemeral port.
- PKCE **S256 only**; `plain` is refused. Codes are single-use, live 60
  seconds, and are popped before the PKCE check so a rejected attempt
  cannot be retried.
- Refresh tokens rotate: the old one is invalidated in the same write
  that issues the new one. An unknown or expired one returns
  `invalid_grant` specifically — Claude only re-authorizes on that code,
  and anything else strands the connection.
- Access and refresh tokens are stored **as SHA-256 hashes** in
  `remote-hands-oauth.json` (0600), so the grant file is not a list of
  live credentials at rest. Codes are in memory only and do not survive
  a restart.
- Claude caches discovery documents globally, keyed by URL, for about
  five minutes. Metadata is computed per-request from the `Host` header
  rather than stored, so there is no second copy to go stale — but
  expect a delay after changing the hostname. Override with
  `MODELCTL_REMOTE_HANDS_BASE_URL` if a proxy rewrites `Host`.
- Claude allows ten seconds for discovery and token calls, thirty for
  refresh. Everything is local except the one CIMD fetch, which is
  bounded at ten seconds and 128 KiB.
- The CIMD fetch is the one outbound request an **unauthenticated**
  caller can make this server perform, at a destination they choose, so
  the destination is vetted before anything is sent. `https` on port 443
  only; no credentials in the URL; the host is resolved and **every**
  address it answers with must be publicly routable — loopback, RFC-1918,
  link-local (including `169.254.169.254`), ULA, CGNAT/tailnet,
  multicast and reserved are all refused, as are the v4 addresses
  embedded in 4-in-6, 6to4 and Teredo forms. The socket is then opened
  to the address that passed, with the certificate still checked against
  the hostname, so a DNS answer that changes between the check and the
  fetch has nothing to swap. Redirects are not followed. Without this the
  endpoint is a blind-SSRF primitive against everything the rig can reach
  and the internet cannot: llama-swap on 9292, the console on 9293, the
  tailnet, a cloud metadata service.

Revoking access: rotating the operator token stops new consents, but
already-issued grants survive it. Clear those by deleting
`remote-hands-oauth.json`, or take the whole thing down with
`remote-hands off`.

---

## What it exposes

Four tools.

| tool | scope |
|---|---|
| `read_file` | allowlist only |
| `write_file` | allowlist only (rewrite or append) |
| `list_directory` | allowlist only |
| `run_command` | a shell as `aaron`, with a timeout |

The filesystem allowlist:

- `/home/aaron/workspace` (which contains `moe-review`)
- `/home/aaron/models`
- `/home/aaron/services`

Paths are resolved before they are checked, so `..` segments and
symlinks out of an allowed root are refused rather than followed.

`run_command` is deliberately **not** confined to the allowlist: it is a
shell, and the posture is the one the desktop session already has. The
allowlist governs the file tools, which are the ones that promise it.

One guard sits in front of it: commands that would `restart`, `stop`,
`kill` or `pkill` llama-swap, OVMS or the web console are refused. That
is a speed bump against habit, not a security boundary — a determined
caller can spell the same command another way. The security boundary is
the token.

---

## The kill switch

```bash
modelctl remote-hands off
```

Tears the funnel down **first**, then stops the service. That order is
the point: stopping the service first would leave a public URL pointing
at a dead port.

`off` refuses if the tailscale serve config publishes something other
than port 9294 — `tailscale funnel reset` is the only teardown this
tailscale version offers and it clears everything. `--force` overrides
that, after you have read what it lists.

Harder switches, in increasing order of blast radius:

```bash
tailscale funnel reset          # drops all serve/funnel config
tailscale set --operator=       # revokes the right to configure it
```

and rotating the token (`modelctl remote-hands install --rotate-token`)
invalidates any connector already holding the old one.

---

## Status and the audit log

```bash
modelctl remote-hands status
```

Prints exposure (`EXPOSED` / `hidden`), the public URL when exposed,
service state and uptime, whether anything is listening, the token and
allowlist in use, the last five audit entries, and a security line:

```
security: funnel up, 412 audit line(s), 7 auth failure(s) in the last hour, last 2026-08-02T09:14:03
```

That line is the one-glance answer to "is anything hammering it". It is
computed from the audit log, not from the running service's counters —
`status` runs in the CLI's process, and an in-memory counter read from
there would always say zero.

**Every** request appends one JSON line to
`~/.local/share/modelctl/remote-hands.log` (0600), and a tool call
appends a second one for the tool:

```json
{"at": "2026-08-02T09:14:01", "event": "request", "method": "POST", "path": "/mcp", "status": 200, "auth": "oauth", "peer": "127.0.0.1", "xff": "160.79.104.10", "tool": "read_file", "args_sha256": "4713e1ebfe05c38a", "outcome": "ok", "ms": 3}
{"at": "2026-08-02T09:14:01", "event": "tool", "peer": "127.0.0.1", "xff": "160.79.104.10", "tool": "read_file", "args_sha256": "4713e1ebfe05c38a", "outcome": "ok", "ms": 2}
```

The two are not derivable from each other: the request line is the only
record of a call that never reached a tool — a 401, a body that is not
JSON, a verb with no handler — and the tool line is the only record of
what actually ran and for how long.

`event: request` outcomes are `ok`, `unauthorized`, `rate-limited`,
`rejected`, `malformed` and `error`, with the HTTP status beside them.
`auth` says how the request got in: `static-token`, `oauth`,
`operator-token`, `oauth-grant`, `public` (an endpoint that needs no
credential), `rejected` or `rate-limited`.

`event: tool` outcomes are `ok`, `denied`, `error` and `unknown-tool`.
OAuth steps appear as `event: oauth` with tool `oauth/authorize`,
`oauth/consent`, `oauth/token`, `oauth/refresh` or `oauth/register` — so
a failed consent, a replayed code and a rotated refresh token are all
visible after the fact, with the reason in `detail`.

A 401 line carries no tool name. Filling one in would mean parsing an
unauthenticated request body just to populate a log field, and "no work
before auth" is worth more than a tidier line.

The line is written **before** the response goes out (at `end_headers`),
so a caller holding an answer can rely on the record already existing.

The **arguments are not logged**, only a digest of them: `write_file`
carries file contents and `run_command` can carry a secret, and the log
has to be safe to read out loud. The digest still answers "was this the
same call as that one".

`peer` is the local tailscaled when the funnel is up, not Anthropic —
Funnel terminates TLS and proxies, so the connecting address this
process sees is always loopback. `xff` is the `X-Forwarded-For` the
funnel wrote, logged verbatim beside it. **Both are recorded and neither
is trusted for policy**: the socket peer is real but says nothing, and
the header says something but the caller can write into it. Anthropic's
egress is `160.79.104.0/21` if you want to correlate against tailscaled's
own logs.

---

## Rate limiting and lockout

In memory, per process, gone on restart — the state is "who has been
failing lately", it is only useful while an attempt is in progress, and
persisting it would put a writable file in the path of every
unauthenticated request.

| what | where | limit |
|---|---|---|
| per-source token bucket | `/authorize`, `/token`, `/register` | 20 burst, 1/s sustained |
| global token bucket | the same, plus every auth *failure* | 60 burst, 5/s sustained |
| lockout | any credential failure | 5 in a row → 30s, doubling to 1h |

Successful authenticated traffic is never throttled: a working connector
makes long bursts of legitimate tool calls, and rate-limiting those
would be this feature breaking the thing it protects. Only the
unauthenticated surface and the failure path draw on a bucket.

The source key is the **last** `X-Forwarded-For` entry (a proxy appends
the peer it saw, so the last hop is the one the funnel wrote; everything
before it is caller-supplied). Two consequences worth knowing:

- With no `X-Forwarded-For` at all, every caller behind the funnel
  shares one key. In that case the lockout is **skipped** — banning the
  shared key would let a stranger's failed guesses lock the connector
  out. The global bucket is what covers that case, and it cannot be
  turned against you.
- With one, a forger can spread their own load across keys. That is why
  the global bucket exists as well; the per-source key is the part that
  costs an attacker something to move, not the part that has to hold.

A limited request gets `429` and a `Retry-After`, and never reaches a
tool.

Every credential failure takes at least 150 ms, whatever the reason, and
the consent screen answers "That did not match." to all of them — an
unknown client, an unregistered `redirect_uri` and a wrong token are one
response. The real reason goes to the audit log's `detail`, which is
where the operator looks and the attacker does not.

The token endpoint is the exception: its RFC 6749 error **codes** stay
exact, because Claude re-authorizes only on `invalid_grant` and gives up
on anything else — flattening them would strand the connector rather
than protect it. The `error_description` is dropped and the timing floor
still applies.

State paths live under `MODELCTL_HOME` (default
`~/.local/share/modelctl`) rather than the `~/.local/state` the original
order named, so that one variable moves all of modelctl's state at once
— the same call `modelctl_display` made, for the same reason.

---

## Why this is a bespoke server

The order preferred wrapping a maintained stdio MCP server behind an
HTTP gateway. Both candidates were checked on 2026-08-02:

- **mcp-proxy** has no inbound authentication at all.
- **supergateway**'s `--oauth2Bearer` and `--header` set headers on
  *outbound* connections; there is no inbound check.

Neither can satisfy "auth on every request" without a second reverse
proxy in front, and neither can write the per-tool-call audit record,
which needs the decoded JSON-RPC. So the transport, the auth and the
audit are local — and they are the only things that are.

The transport is MCP streamable HTTP, stateless: no `Mcp-Session-Id` is
issued, so restarting the unit does not strand a live connector. `POST
/mcp` answers with SSE when the client's `Accept` asks for it and plain
JSON otherwise; `GET /mcp` is 405 (there is no server-initiated stream);
everything unauthenticated is 401 before routing, so an anonymous caller
learns nothing about which paths exist.
