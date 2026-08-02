"""Tests for the remote-hands MCP server.

The four things worth failing over, in the order the order named them:
auth rejection, allowlist enforcement, audit logging, and on/off
ordering. Everything here runs against a loopback listener on an
ephemeral port with a throwaway state dir -- no funnel is configured, no
systemd unit is touched, and no real allowlist root is written to.
"""
import base64
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

import modelctl_remote_hands as rh
import modelctl_remote_hands_oauth as oauth

TOKEN = "a" * 64


class RemoteHandsBase(unittest.TestCase):
    """Redirect the token, the audit log and the allowlist per test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.allowed = self.root / "allowed"
        self.allowed.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self._env = {
            "MODELCTL_REMOTE_HANDS_TOKEN_PATH": str(self.root / "token"),
            "MODELCTL_REMOTE_HANDS_AUDIT_PATH": str(self.root / "audit.log"),
            "MODELCTL_REMOTE_HANDS_OAUTH_PATH": str(self.root / "oauth.json"),
            "MODELCTL_REMOTE_HANDS_ROOTS": str(self.allowed),
            "MODELCTL_REMOTE_HANDS_TOKEN": "",
            "MODELCTL_REMOTE_HANDS_BASE_URL": "",
        }
        self._saved = {k: os.environ.get(k) for k in self._env}
        os.environ.update(self._env)
        (self.root / "token").write_text(TOKEN + "\n")
        # Authorization codes are process-global and deliberately not
        # persisted; a code left over from another test would make a
        # one-time-use assertion pass for the wrong reason.
        oauth._CODES.clear()
        self.addCleanup(oauth._CODES.clear)
        self.addCleanup(self._restore)

    def _restore(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def audit_lines(self):
        path = Path(os.environ["MODELCTL_REMOTE_HANDS_AUDIT_PATH"])
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]


# --- auth ------------------------------------------------------------------

class TokenTests(RemoteHandsBase):

    def test_empty_token_file_is_not_a_token(self):
        """An interrupted write must not mean "no auth required": an
        anonymous request presents "" too."""
        Path(os.environ["MODELCTL_REMOTE_HANDS_TOKEN_PATH"]).write_text("\n")
        self.assertIsNone(rh.read_token())
        self.assertFalse(rh.authorized({"authorization": "Bearer "}))
        self.assertFalse(rh.authorized({}))

    def test_missing_token_rejects_everything(self):
        Path(os.environ["MODELCTL_REMOTE_HANDS_TOKEN_PATH"]).unlink()
        self.assertFalse(rh.authorized({"x-api-key": "anything"}))
        self.assertFalse(rh.authorized({"authorization": f"Bearer {TOKEN}"}))

    def test_create_token_is_32_bytes_and_0600(self):
        Path(os.environ["MODELCTL_REMOTE_HANDS_TOKEN_PATH"]).unlink()
        token, created = rh.create_token()
        self.assertTrue(created)
        self.assertEqual(len(token), 64)          # 32 bytes, hex
        mode = rh.token_path().stat().st_mode & 0o777
        self.assertEqual(mode, 0o600, f"token file is {oct(mode)}")

    def test_create_token_does_not_clobber_without_force(self):
        token, created = rh.create_token()
        self.assertFalse(created)
        self.assertEqual(token, TOKEN)
        rotated, created = rh.create_token(force=True)
        self.assertTrue(created)
        self.assertNotEqual(rotated, TOKEN)

    def test_all_three_header_names_are_accepted(self):
        for header, value in (("authorization", f"Bearer {TOKEN}"),
                              ("authorization", TOKEN),
                              ("x-api-key", TOKEN),
                              ("x-auth-token", TOKEN)):
            self.assertTrue(rh.authorized({header: value}),
                            f"{header}: {value[:12]}... was rejected")

    def test_wrong_and_prefix_tokens_are_rejected(self):
        for value in (TOKEN[:-1], TOKEN + "b", "b" * 64, "", "Bearer"):
            self.assertFalse(rh.authorized({"x-api-key": value}),
                             f"{value!r} was accepted")


# --- allowlist -------------------------------------------------------------

class AllowlistTests(RemoteHandsBase):

    def test_inside_the_allowlist_resolves(self):
        target = self.allowed / "sub" / "file.txt"
        self.assertEqual(rh.resolve_allowed(str(target)), target.resolve())

    def test_the_root_itself_is_allowed(self):
        self.assertEqual(rh.resolve_allowed(str(self.allowed)),
                         self.allowed.resolve())

    def test_outside_is_refused(self):
        with self.assertRaises(rh.RemoteHandsError):
            rh.resolve_allowed(str(self.outside / "secret"))

    def test_dotdot_escape_is_refused(self):
        """The classic: a path that is textually under the root and
        resolves outside it."""
        escape = str(self.allowed / ".." / "outside" / "secret")
        with self.assertRaises(rh.RemoteHandsError):
            rh.resolve_allowed(escape)

    def test_symlink_escape_is_refused(self):
        (self.outside / "secret").write_text("nope")
        link = self.allowed / "link"
        link.symlink_to(self.outside)
        with self.assertRaises(rh.RemoteHandsError):
            rh.resolve_allowed(str(link / "secret"))

    def test_relative_paths_are_refused(self):
        with self.assertRaises(rh.RemoteHandsError):
            rh.resolve_allowed("relative/path")

    def test_tools_enforce_the_allowlist(self):
        outside = str(self.outside / "secret")
        (self.outside / "secret").write_text("nope")
        for name, args in (
                ("read_file", {"path": outside}),
                ("write_file", {"path": outside, "content": "x"}),
                ("list_directory", {"path": str(self.outside)})):
            text, is_error = rh.call_tool(name, args)
            self.assertTrue(is_error, f"{name} did not refuse")
            self.assertIn("outside the allowlist", text)
        self.assertEqual((self.outside / "secret").read_text(), "nope")


# --- tools -----------------------------------------------------------------

class ToolTests(RemoteHandsBase):

    def test_write_then_read_roundtrip(self):
        path = str(self.allowed / "note.txt")
        text, is_error = rh.call_tool("write_file",
                                      {"path": path, "content": "hello\n"})
        self.assertFalse(is_error, text)
        text, is_error = rh.call_tool("read_file", {"path": path})
        self.assertFalse(is_error, text)
        self.assertEqual(text, "hello\n")

    def test_append_mode(self):
        path = str(self.allowed / "note.txt")
        rh.call_tool("write_file", {"path": path, "content": "a\n"})
        rh.call_tool("write_file",
                     {"path": path, "content": "b\n", "mode": "append"})
        text, _ = rh.call_tool("read_file", {"path": path})
        self.assertEqual(text, "a\nb\n")

    def test_read_offset_and_limit(self):
        path = self.allowed / "lines.txt"
        path.write_text("one\ntwo\nthree\nfour\n")
        text, _ = rh.call_tool("read_file",
                               {"path": str(path), "offset": 2, "limit": 2})
        self.assertEqual(text, "two\nthree\n")

    def test_list_directory(self):
        (self.allowed / "sub").mkdir()
        (self.allowed / "f.txt").write_text("xx")
        text, is_error = rh.call_tool("list_directory",
                                      {"path": str(self.allowed)})
        self.assertFalse(is_error, text)
        self.assertIn("[DIR]  sub", text)
        self.assertIn("[FILE] f.txt (2 bytes)", text)

    def test_run_command(self):
        text, is_error = rh.call_tool("run_command", {"command": "echo hi"})
        self.assertFalse(is_error, text)
        self.assertIn("exit 0", text)
        self.assertIn("hi", text)

    def test_run_command_reports_failure_without_raising(self):
        text, is_error = rh.call_tool("run_command", {"command": "exit 3"})
        self.assertFalse(is_error, "a non-zero exit is a result, not a refusal")
        self.assertIn("exit 3", text)

    def test_run_command_timeout(self):
        text, is_error = rh.call_tool("run_command",
                                      {"command": "sleep 5", "timeout": 1})
        self.assertTrue(is_error)
        self.assertIn("timed out", text)

    def test_run_command_is_not_restricted_to_the_allowlist(self):
        """The shell posture is deliberately the desktop session's. If
        this ever starts failing, the change was intentional -- update
        the docs with it."""
        text, is_error = rh.call_tool(
            "run_command", {"command": "pwd", "cwd": str(self.outside)})
        self.assertFalse(is_error, text)

    def test_serving_stack_restarts_are_refused(self):
        for command in ("systemctl --user restart llama-swap.service",
                        "systemctl --user stop llama-swap",
                        "systemctl restart ovms",
                        "pkill -f llama-swap",
                        "systemctl --user restart modelctl-web.service"):
            text, is_error = rh.call_tool("run_command", {"command": command})
            self.assertTrue(is_error, f"{command!r} was not refused")
            self.assertIn("serving stack", text)

    def test_unrelated_restarts_are_not_refused(self):
        self.assertIsNone(rh.protected_refusal(
            "systemctl --user restart act-runner.service"))
        self.assertIsNone(rh.protected_refusal("echo llama-swap"))

    def test_unknown_tool(self):
        text, is_error = rh.call_tool("rm_rf", {})
        self.assertTrue(is_error)
        self.assertIn("unknown tool", text)


# --- audit -----------------------------------------------------------------

class AuditTests(RemoteHandsBase):

    def test_every_call_is_logged_with_digest_and_outcome(self):
        path = str(self.allowed / "note.txt")
        rh.call_tool("write_file", {"path": path, "content": "hello"})
        rh.call_tool("read_file", {"path": "/etc/shadow"})
        rh.call_tool("nope", {})
        lines = self.audit_lines()
        self.assertEqual([e["tool"] for e in lines],
                         ["write_file", "read_file", "nope"])
        self.assertEqual([e["outcome"] for e in lines],
                         ["ok", "denied", "unknown-tool"])
        for entry in lines:
            self.assertIn("at", entry)
            self.assertEqual(len(entry["args_sha256"]), 16)
            self.assertIn("ms", entry)

    def test_arguments_are_not_written_to_the_log(self):
        """The log has to be safe to read out loud: `write_file` carries
        file contents and `run_command` can carry a secret."""
        rh.call_tool("write_file", {"path": str(self.allowed / "s"),
                                    "content": "hunter2-secret-content"})
        rh.call_tool("run_command", {"command": "echo swordfish-secret"})
        blob = Path(os.environ["MODELCTL_REMOTE_HANDS_AUDIT_PATH"]).read_text()
        self.assertNotIn("hunter2", blob)
        self.assertNotIn("swordfish", blob)

    def test_digest_is_stable_and_argument_sensitive(self):
        first = rh.args_digest({"a": 1, "b": 2})
        self.assertEqual(first, rh.args_digest({"b": 2, "a": 1}))
        self.assertNotEqual(first, rh.args_digest({"a": 1, "b": 3}))

    def test_audit_tail_returns_the_last_five(self):
        for i in range(8):
            rh.audit("read_file", "ok", args={"i": i})
        tail = rh.audit_tail(5)
        self.assertEqual(len(tail), 5)
        self.assertEqual(tail[-1]["args_sha256"], rh.args_digest({"i": 7}))

    def test_audit_never_raises_on_an_unwritable_log(self):
        os.environ["MODELCTL_REMOTE_HANDS_AUDIT_PATH"] = "/proc/nope/audit.log"
        rh.audit("read_file", "ok", args={})     # must not raise
        self.assertEqual(rh.audit_tail(5), [])


# --- the HTTP transport, end to end ---------------------------------------

class ServerFixture(RemoteHandsBase):
    """A real listener on an ephemeral loopback port."""

    def setUp(self):
        super().setUp()
        self.httpd = rh.make_server(host="127.0.0.1", port=0, token=TOKEN)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)

    def _stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def post(self, payload, token=TOKEN, header="authorization",
             accept="application/json", path="/mcp", method="POST"):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", accept)
        if token is not None:
            value = f"Bearer {token}" if header == "authorization" else token
            req.add_header(header, value)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def rpc(self, method, params=None, msg_id=1, **kwargs):
        status, body = self.post(
            {"jsonrpc": "2.0", "id": msg_id, "method": method,
             "params": params or {}}, **kwargs)
        return status, (json.loads(body) if body else None)


class ServerTests(ServerFixture):
    """Auth and protocol behaviour of the static-token path."""

    # --- auth rejection ---------------------------------------------------

    def test_no_token_is_401(self):
        status, body = self.post({"jsonrpc": "2.0", "id": 1,
                                  "method": "tools/list"}, token=None)
        self.assertEqual(status, 401)

    def test_wrong_token_is_401(self):
        status, _ = self.post({"jsonrpc": "2.0", "id": 1,
                               "method": "tools/list"}, token="b" * 64)
        self.assertEqual(status, 401)

    def test_unauthenticated_requests_are_audited(self):
        self.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                  token=None)
        entries = [e for e in self.audit_lines()
                   if e["outcome"] == "unauthorized"]
        self.assertTrue(entries, "a 401 left no audit trail")

    def test_unauthenticated_requests_run_no_tool(self):
        target = self.allowed / "must-not-exist"
        self.post({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "write_file",
                              "arguments": {"path": str(target),
                                            "content": "x"}}}, token=None)
        self.assertFalse(target.exists())

    def test_auth_is_required_on_every_verb_and_path(self):
        for method, path in (("GET", "/mcp"), ("GET", "/"),
                             ("DELETE", "/mcp"), ("POST", "/anything")):
            status, _ = self.post(None if method != "POST" else {},
                                  token=None, path=path, method=method)
            self.assertEqual(status, 401, f"{method} {path} was not 401")

    def test_x_api_key_and_x_auth_token_work_over_http(self):
        for header in ("x-api-key", "x-auth-token"):
            status, body = self.rpc("tools/list", header=header)
            self.assertEqual(status, 200, header)
            self.assertEqual(len(body["result"]["tools"]), 4)

    # --- protocol ---------------------------------------------------------

    def test_initialize_echoes_a_known_protocol_version(self):
        status, body = self.rpc("initialize",
                                {"protocolVersion": "2025-06-18"})
        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(body["result"]["serverInfo"]["name"], rh.SERVER_NAME)

    def test_initialize_falls_back_for_an_unknown_version(self):
        _, body = self.rpc("initialize", {"protocolVersion": "1999-01-01"})
        self.assertEqual(body["result"]["protocolVersion"],
                         rh.PROTOCOL_VERSION)

    def test_tools_list_advertises_the_four_tools(self):
        _, body = self.rpc("tools/list")
        names = sorted(t["name"] for t in body["result"]["tools"])
        self.assertEqual(names, ["list_directory", "read_file",
                                 "run_command", "write_file"])
        for tool in body["result"]["tools"]:
            self.assertIn("inputSchema", tool)

    def test_tools_call_over_http(self):
        target = self.allowed / "http.txt"
        _, body = self.rpc("tools/call",
                           {"name": "write_file",
                            "arguments": {"path": str(target),
                                          "content": "via http"}})
        self.assertFalse(body["result"]["isError"], body)
        self.assertEqual(target.read_text(), "via http")

    def test_tools_call_error_is_a_result_not_a_transport_error(self):
        status, body = self.rpc("tools/call",
                                {"name": "read_file",
                                 "arguments": {"path": "/etc/shadow"}})
        self.assertEqual(status, 200)
        self.assertTrue(body["result"]["isError"])
        self.assertIn("outside the allowlist",
                      body["result"]["content"][0]["text"])

    def test_sse_response_when_the_client_asks_for_it(self):
        status, body = self.post(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            accept="application/json, text/event-stream")
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith("event: message\ndata: "), body[:60])
        payload = json.loads(body.split("data: ", 1)[1])
        self.assertEqual(len(payload["result"]["tools"]), 4)

    def test_notification_gets_202_and_no_body(self):
        status, body = self.post({"jsonrpc": "2.0",
                                  "method": "notifications/initialized"})
        self.assertEqual(status, 202)
        self.assertEqual(body, "")

    def test_unknown_method_is_a_jsonrpc_error(self):
        _, body = self.rpc("tools/delete_everything")
        self.assertEqual(body["error"]["code"], -32601)

    def test_bad_json_is_a_parse_error(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/mcp",
                                     data=b"{not json", method="POST")
        req.add_header("authorization", f"Bearer {TOKEN}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        self.assertEqual(status, 400)

    def test_get_on_mcp_is_405(self):
        status, _ = self.post(None, path="/mcp", method="GET")
        self.assertEqual(status, 405)

    def test_unknown_path_is_404_when_authenticated(self):
        status, _ = self.post({}, path="/admin", method="POST")
        self.assertEqual(status, 404)

    def test_server_refuses_to_start_without_a_token(self):
        Path(os.environ["MODELCTL_REMOTE_HANDS_TOKEN_PATH"]).unlink()
        with self.assertRaises(rh.RemoteHandsError):
            rh.make_server(host="127.0.0.1", port=0)

    def test_listener_is_loopback_only(self):
        self.assertEqual(self.httpd.server_address[0], "127.0.0.1")


# --- on/off ordering and funnel handling ----------------------------------

class FunnelTests(RemoteHandsBase):
    """`tailscale` is never actually run: a fake runner records the argv."""

    def fake(self, responses):
        calls = []

        def runner(*args, **kwargs):
            calls.append(list(args))
            for prefix, result in responses:
                if list(args)[:len(prefix)] == list(prefix):
                    return result
            return 0, "", ""
        return runner, calls

    def config_json(self, *ports):
        handlers = {f"/{i}": {"Proxy": f"http://127.0.0.1:{p}"}
                    for i, p in enumerate(ports)}
        return json.dumps({"Web": {"aaron-2.tailb51646.ts.net:443":
                                   {"Handlers": handlers}}})

    def test_off_refuses_to_clobber_someone_elses_serve_config(self):
        runner, calls = self.fake([
            (["funnel", "status"], (0, self.config_json(8080), "")),
        ])
        ok, detail = rh.funnel_off(port=9294, runner=runner)
        self.assertFalse(ok)
        self.assertIn("8080", detail)
        self.assertNotIn(["funnel", "reset"], calls,
                         "reset ran despite a foreign serve config")

    def test_off_resets_when_only_our_port_is_published(self):
        runner, calls = self.fake([
            (["funnel", "status"], (0, self.config_json(9294), "")),
        ])
        ok, _ = rh.funnel_off(port=9294, runner=runner)
        self.assertTrue(ok)
        self.assertIn(["funnel", "reset"], calls)

    def test_off_with_force_resets_anyway(self):
        runner, calls = self.fake([
            (["funnel", "status"], (0, self.config_json(8080), "")),
        ])
        ok, _ = rh.funnel_off(port=9294, runner=runner, force=True)
        self.assertTrue(ok)
        self.assertIn(["funnel", "reset"], calls)

    def test_off_is_a_no_op_when_nothing_is_published(self):
        runner, calls = self.fake([(["funnel", "status"], (0, "{}", ""))])
        ok, detail = rh.funnel_off(port=9294, runner=runner)
        self.assertTrue(ok)
        self.assertIn("no funnel", detail)
        self.assertNotIn(["funnel", "reset"], calls)

    def test_on_passes_the_port_and_runs_in_the_background(self):
        runner, calls = self.fake([])
        ok, _ = rh.funnel_on(port=9294, runner=runner)
        self.assertTrue(ok)
        self.assertEqual(calls, [["funnel", "--bg", "--yes", "9294"]])

    def test_on_explains_an_operator_rights_failure(self):
        runner, _ = self.fake([
            (["funnel"], (1, "", "access denied: funnel"))])
        ok, detail = rh.funnel_on(port=9294, runner=runner)
        self.assertFalse(ok)
        self.assertIn("rig-headless-setup.sh", detail)

    def test_exposed_detection(self):
        runner, _ = self.fake([
            (["funnel", "status"], (0, self.config_json(9294), ""))])
        self.assertTrue(rh.funnel_exposed(port=9294, runner=runner))
        self.assertFalse(rh.funnel_exposed(port=9999, runner=runner))

    def test_public_url_comes_from_tailscaled(self):
        runner, _ = self.fake([
            (["status"], (0, json.dumps(
                {"Self": {"DNSName": "aaron-2.tailb51646.ts.net."}}), ""))])
        self.assertEqual(rh.public_url(runner=runner),
                         "https://aaron-2.tailb51646.ts.net/mcp")

    def test_no_token_never_reaches_the_url(self):
        """A credential in a URL leaks into logs, proxies and history, and
        the MCP spec prohibits it. There must be no code path that puts
        one there."""
        runner, _ = self.fake([
            (["status"], (0, json.dumps(
                {"Self": {"DNSName": "aaron-2.tailb51646.ts.net."}}), ""))])
        url = rh.public_url(runner=runner)
        self.assertNotIn("?", url)
        self.assertNotIn(TOKEN, url)


# --- the systemd unit ------------------------------------------------------

class UnitTests(RemoteHandsBase):

    def test_unit_has_no_install_section(self):
        """No [Install] means `enable` cannot silently make exposure
        survive a reboot."""
        unit = rh.render_unit(port=9294)
        sections = [line.strip() for line in unit.splitlines()
                    if line.strip().startswith("[")]
        self.assertEqual(sections, ["[Unit]", "[Service]"])
        self.assertNotIn("WantedBy", unit)

    def test_unit_binds_loopback(self):
        unit = rh.render_unit(host="127.0.0.1", port=9294)
        self.assertIn("--host 127.0.0.1 --port 9294", unit)
        self.assertNotIn("0.0.0.0", unit)

    def test_unit_carries_the_installing_modelctl_environment(self):
        unit = rh.render_unit(env={"MODELCTL_HOME": "/tmp/x", "PATH": "/bin"})
        self.assertIn('Environment="MODELCTL_HOME=/tmp/x"', unit)
        self.assertNotIn("PATH=/bin", unit)

    def test_unit_never_contains_the_token(self):
        """A unit file under ~/.config is 0644 and is echoed by
        `systemctl --user cat`. The token lives in a 0600 file, so an
        installing shell that happens to export it must not bake it in."""
        unit = rh.render_unit(env={"MODELCTL_REMOTE_HANDS_TOKEN": TOKEN,
                                   "MODELCTL_WEB_TOKEN": "webtok",
                                   "MODELCTL_HOME": "/tmp/x"})
        self.assertNotIn(TOKEN, unit)
        self.assertNotIn("webtok", unit)
        self.assertIn('Environment="MODELCTL_HOME=/tmp/x"', unit)


# --- OAuth: discovery documents -------------------------------------------

class DiscoveryTests(RemoteHandsBase):

    BASE = "https://aaron-2.tailb51646.ts.net"

    def test_protected_resource_metadata_shape(self):
        """`resource` must equal the MCP URL exactly as the user typed it
        into the dialog, or Claude rejects the document."""
        doc = oauth.protected_resource_metadata(self.BASE, "/mcp")
        self.assertEqual(doc["resource"], f"{self.BASE}/mcp")
        self.assertEqual(doc["authorization_servers"], [self.BASE])
        self.assertEqual(doc["bearer_methods_supported"], ["header"])

    def test_authorization_server_metadata_selects_cimd(self):
        """Claude picks CIMD only when BOTH of these are advertised; if
        either goes missing it silently falls back to hunting for a
        registration_endpoint."""
        doc = oauth.authorization_server_metadata(self.BASE)
        self.assertIs(doc["client_id_metadata_document_supported"], True)
        self.assertIn("none", doc["token_endpoint_auth_methods_supported"])
        self.assertEqual(doc["code_challenge_methods_supported"], ["S256"])
        self.assertEqual(doc["issuer"], self.BASE)
        self.assertEqual(doc["authorization_endpoint"], f"{self.BASE}/authorize")
        self.assertEqual(doc["token_endpoint"], f"{self.BASE}/token")

    def test_base_url_from_host_header(self):
        self.assertEqual(oauth.base_url("aaron-2.tailb51646.ts.net"),
                         "https://aaron-2.tailb51646.ts.net")
        self.assertEqual(oauth.base_url("127.0.0.1:9294"),
                         "http://127.0.0.1:9294")

    def test_base_url_env_override_wins(self):
        os.environ["MODELCTL_REMOTE_HANDS_BASE_URL"] = "https://example.test/"
        self.assertEqual(oauth.base_url("anything"), "https://example.test")

    def test_www_authenticate_points_at_the_metadata(self):
        header = oauth.www_authenticate(self.BASE, "/mcp")
        self.assertIn('resource_metadata="https://aaron-2.tailb51646.ts.net'
                      '/.well-known/oauth-protected-resource/mcp"', header)
        self.assertTrue(header.startswith("Bearer "))


# --- OAuth: CIMD ----------------------------------------------------------

def fake_cimd(doc, status=200):
    """An opener that answers one CIMD fetch with `doc`."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner, n=-1):
            raw = doc if isinstance(doc, bytes) else json.dumps(doc).encode()
            return raw[:n] if n and n > 0 else raw

    def opener(url, timeout=None):
        return _Resp()
    return opener


class CimdTests(RemoteHandsBase):

    CLIENT = "https://claude.ai/oauth/client-metadata"

    def test_self_referential_document_is_accepted(self):
        doc = {"client_id": self.CLIENT,
               "redirect_uris": [oauth.CLAUDE_CALLBACK]}
        got = oauth.fetch_cimd(self.CLIENT, opener=fake_cimd(doc))
        self.assertEqual(got["client_id"], self.CLIENT)

    def test_non_self_referential_document_is_refused(self):
        """Without this, any URL serving someone else's document could
        inherit its redirect_uris."""
        doc = {"client_id": "https://evil.test/other",
               "redirect_uris": ["https://evil.test/cb"]}
        with self.assertRaises(oauth.OAuthError) as ctx:
            oauth.fetch_cimd(self.CLIENT, opener=fake_cimd(doc))
        self.assertEqual(ctx.exception.code, "invalid_client")

    def test_fetch_sends_a_real_user_agent(self):
        """urllib's default User-Agent is `Python-urllib/<version>`, and
        the CDN in front of claude.ai answers that with 403 -- observed
        2026-08-02 against the live client_id document, where curl got
        200 and urllib got 403 with nothing else changed. This is the
        regression that cost a working connector."""
        req = oauth.cimd_request("https://claude.ai/oauth/x")
        agent = req.get_header("User-agent") or ""
        self.assertTrue(agent)
        self.assertNotIn("Python-urllib", agent)
        self.assertEqual(req.get_header("Accept"), "application/json")

    def test_fetch_passes_a_request_object_not_a_bare_url(self):
        """A bare URL string would carry none of those headers."""
        seen = {}

        def opener(req, timeout=None):
            seen["req"] = req
            return fake_cimd({"client_id": self.CLIENT,
                              "redirect_uris": []})(req, timeout)
        oauth.fetch_cimd(self.CLIENT, opener=opener)
        self.assertIsInstance(seen["req"], urllib.request.Request)
        self.assertEqual(seen["req"].full_url, self.CLIENT)

    def test_non_https_client_id_is_refused(self):
        with self.assertRaises(oauth.OAuthError):
            oauth.fetch_cimd("http://claude.ai/x", opener=fake_cimd({}))

    def test_non_json_document_is_refused(self):
        with self.assertRaises(oauth.OAuthError):
            oauth.fetch_cimd(self.CLIENT, opener=fake_cimd(b"<html>"))

    def test_oversized_document_is_refused(self):
        blob = b'{"client_id":"' + b"x" * (oauth.CIMD_MAX_BYTES + 10) + b'"}'
        with self.assertRaises(oauth.OAuthError):
            oauth.fetch_cimd(self.CLIENT, opener=fake_cimd(blob))

    def test_unreachable_document_is_refused(self):
        def opener(url, timeout=None):
            raise urllib.error.URLError("boom")
        with self.assertRaises(oauth.OAuthError):
            oauth.fetch_cimd(self.CLIENT, opener=opener)


class RedirectUriTests(RemoteHandsBase):

    CLIENT = "https://claude.ai/oauth/client-metadata"

    def test_registered_same_origin_is_allowed(self):
        self.assertTrue(oauth.redirect_uri_allowed(
            self.CLIENT, oauth.CLAUDE_CALLBACK, [oauth.CLAUDE_CALLBACK]))

    def test_unregistered_is_refused(self):
        self.assertFalse(oauth.redirect_uri_allowed(
            self.CLIENT, "https://claude.ai/other", [oauth.CLAUDE_CALLBACK]))

    def test_cross_origin_is_refused_even_when_registered(self):
        """A CIMD document is self-asserted; without the same-origin rule
        this endpoint is an open redirect."""
        self.assertFalse(oauth.redirect_uri_allowed(
            self.CLIENT, "https://evil.test/cb", ["https://evil.test/cb"]))

    def test_same_origin_does_not_apply_to_registered_clients(self):
        """A DCR client_id is an opaque id with no origin to compare, and
        its redirect_uris were registered explicitly rather than asserted
        at authorize time. Holding it to the CIMD rule would reject every
        registration this server issues."""
        self.assertTrue(oauth.redirect_uri_allowed(
            "rh-abc123", oauth.CLAUDE_CALLBACK, [oauth.CLAUDE_CALLBACK]))
        # ...but an unregistered URI is still refused.
        self.assertFalse(oauth.redirect_uri_allowed(
            "rh-abc123", "https://evil.test/cb", [oauth.CLAUDE_CALLBACK]))

    def test_loopback_matches_with_the_port_ignored(self):
        """Native clients bind an ephemeral port at runtime (RFC 8252
        7.3), so Claude Code declares localhost/callback and arrives on
        whatever port it got."""
        self.assertTrue(oauth.redirect_uri_allowed(
            self.CLIENT, "http://localhost:3118/callback",
            ["http://localhost/callback"]))
        self.assertTrue(oauth.redirect_uri_allowed(
            self.CLIENT, "http://127.0.0.1:51234/callback",
            ["http://127.0.0.1/callback"]))

    def test_loopback_path_still_has_to_match(self):
        self.assertFalse(oauth.redirect_uri_allowed(
            self.CLIENT, "http://localhost:3118/steal",
            ["http://localhost/callback"]))

    def test_empty_redirect_uri_is_refused(self):
        self.assertFalse(oauth.redirect_uri_allowed(
            self.CLIENT, "", [oauth.CLAUDE_CALLBACK]))


# --- OAuth: PKCE, codes and tokens ----------------------------------------

def challenge_for(verifier):
    return base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")


class PkceTests(RemoteHandsBase):

    def test_matching_verifier(self):
        self.assertTrue(oauth.verify_pkce("v" * 43, challenge_for("v" * 43)))

    def test_mismatched_verifier(self):
        self.assertFalse(oauth.verify_pkce("wrong", challenge_for("v" * 43)))

    def test_empty_inputs_never_match(self):
        self.assertFalse(oauth.verify_pkce("", ""))
        self.assertFalse(oauth.verify_pkce(None, challenge_for("v")))

    def test_plain_method_is_rejected_at_validation(self):
        with self.assertRaises(oauth.OAuthError):
            oauth.validate_authorize({"code_challenge": "abc",
                                      "code_challenge_method": "plain",
                                      "redirect_uri": "https://x.test/cb"})

    def test_missing_challenge_is_rejected(self):
        with self.assertRaises(oauth.OAuthError):
            oauth.validate_authorize({"code_challenge_method": "S256",
                                      "redirect_uri": "https://x.test/cb"})


class CodeTests(RemoteHandsBase):

    VERIFIER = "verifier-" + "z" * 40

    def make_code(self):
        return oauth.create_code("client", "https://x.test/cb",
                                 challenge_for(self.VERIFIER))

    def test_redeem_happy_path(self):
        entry = oauth.redeem_code(self.make_code(), "client",
                                  "https://x.test/cb", self.VERIFIER)
        self.assertEqual(entry["client_id"], "client")

    def test_code_is_single_use(self):
        code = self.make_code()
        oauth.redeem_code(code, "client", "https://x.test/cb", self.VERIFIER)
        with self.assertRaises(oauth.OAuthError) as ctx:
            oauth.redeem_code(code, "client", "https://x.test/cb",
                              self.VERIFIER)
        self.assertEqual(ctx.exception.code, "invalid_grant")

    def test_bad_verifier_burns_the_code(self):
        """Popped before the PKCE check, so a rejected attempt cannot be
        retried with a guessed verifier."""
        code = self.make_code()
        with self.assertRaises(oauth.OAuthError):
            oauth.redeem_code(code, "client", "https://x.test/cb", "nope")
        with self.assertRaises(oauth.OAuthError):
            oauth.redeem_code(code, "client", "https://x.test/cb",
                              self.VERIFIER)

    def test_expired_code(self):
        code = self.make_code()
        with self.assertRaises(oauth.OAuthError):
            oauth.redeem_code(code, "client", "https://x.test/cb",
                              self.VERIFIER,
                              now=time.time() + oauth.CODE_TTL + 1)

    def test_code_bound_to_its_client_and_redirect(self):
        for client, redirect in (("other", "https://x.test/cb"),
                                 ("client", "https://x.test/elsewhere")):
            with self.assertRaises(oauth.OAuthError):
                oauth.redeem_code(self.make_code(), client, redirect,
                                  self.VERIFIER)


class TokenTests_(RemoteHandsBase):

    def test_issue_and_validate(self):
        tokens = oauth.issue_tokens("client")
        self.assertEqual(tokens["token_type"], "Bearer")
        self.assertEqual(tokens["expires_in"], oauth.ACCESS_TTL)
        self.assertTrue(oauth.valid_access_token(tokens["access_token"]))

    def test_expired_access_token_is_invalid(self):
        tokens = oauth.issue_tokens("client")
        self.assertFalse(oauth.valid_access_token(
            tokens["access_token"], now=time.time() + oauth.ACCESS_TTL + 1))

    def test_unknown_access_token_is_invalid(self):
        oauth.issue_tokens("client")
        self.assertFalse(oauth.valid_access_token("not-a-token"))
        self.assertFalse(oauth.valid_access_token(""))

    def test_tokens_are_stored_as_hashes(self):
        """The grant file must not be a list of live bearer credentials
        at rest."""
        tokens = oauth.issue_tokens("client")
        blob = Path(os.environ["MODELCTL_REMOTE_HANDS_OAUTH_PATH"]).read_text()
        self.assertNotIn(tokens["access_token"], blob)
        self.assertNotIn(tokens["refresh_token"], blob)

    def test_state_file_is_0600(self):
        oauth.issue_tokens("client")
        mode = oauth.state_path().stat().st_mode & 0o777
        self.assertEqual(mode, 0o600, f"grant file is {oct(mode)}")

    def test_refresh_rotates_and_invalidates_the_old_token(self):
        first = oauth.issue_tokens("client")
        second = oauth.refresh_tokens(first["refresh_token"], "client")
        self.assertNotEqual(first["refresh_token"], second["refresh_token"])
        self.assertTrue(oauth.valid_access_token(second["access_token"]))
        with self.assertRaises(oauth.OAuthError) as ctx:
            oauth.refresh_tokens(first["refresh_token"], "client")
        self.assertEqual(ctx.exception.code, "invalid_grant")

    def test_unknown_refresh_token_is_invalid_grant(self):
        """It must be invalid_grant specifically -- Claude only
        re-authorizes on that code, and anything else strands the
        connection."""
        with self.assertRaises(oauth.OAuthError) as ctx:
            oauth.refresh_tokens("nope")
        self.assertEqual(ctx.exception.code, "invalid_grant")

    def test_refresh_bound_to_its_client(self):
        tokens = oauth.issue_tokens("client")
        with self.assertRaises(oauth.OAuthError):
            oauth.refresh_tokens(tokens["refresh_token"], "someone-else")

    def test_revoke_all(self):
        tokens = oauth.issue_tokens("client")
        self.assertEqual(oauth.revoke_all(), 2)
        self.assertFalse(oauth.valid_access_token(tokens["access_token"]))

    def test_prune_drops_expired_grants(self):
        oauth.issue_tokens("client")
        state = oauth.prune(now=time.time() + oauth.REFRESH_TTL + 1)
        self.assertEqual(state["access"], {})
        self.assertEqual(state["grants"], {})


class RegistrationTests(RemoteHandsBase):

    def test_register_returns_a_public_client(self):
        result = oauth.register_client(
            {"redirect_uris": [oauth.CLAUDE_CALLBACK], "client_name": "x"})
        self.assertTrue(result["client_id"].startswith("rh-"))
        self.assertEqual(result["token_endpoint_auth_method"], "none")

    def test_registered_client_resolves(self):
        result = oauth.register_client(
            {"redirect_uris": [oauth.CLAUDE_CALLBACK]})
        name, uris = oauth.resolve_client(result["client_id"],
                                          oauth.CLAUDE_CALLBACK)
        self.assertEqual(uris, [oauth.CLAUDE_CALLBACK])

    def test_registration_requires_redirect_uris(self):
        for payload in ({}, {"redirect_uris": []}, {"redirect_uris": "x"}):
            with self.assertRaises(oauth.OAuthError):
                oauth.register_client(payload)

    def test_registration_rejects_plain_http_non_loopback(self):
        with self.assertRaises(oauth.OAuthError):
            oauth.register_client({"redirect_uris": ["http://evil.test/cb"]})

    def test_registration_is_capped(self):
        """Unauthenticated by necessity, so it must not grow without
        bound."""
        for _ in range(oauth.MAX_REGISTERED_CLIENTS + 5):
            oauth.register_client({"redirect_uris": [oauth.CLAUDE_CALLBACK]})
        state = json.loads(oauth.state_path().read_text())
        self.assertLessEqual(len(state["clients"]),
                             oauth.MAX_REGISTERED_CLIENTS)

    def test_unknown_client_id_is_refused(self):
        with self.assertRaises(oauth.OAuthError):
            oauth.resolve_client("rh-nope", oauth.CLAUDE_CALLBACK)


class RedirectBuildingTests(RemoteHandsBase):

    def test_code_appended_to_a_callback_that_already_has_a_query(self):
        url = oauth.redirect_with_code("https://claude.ai/cb?x=1", "CODE", "S")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(query["x"], ["1"])
        self.assertEqual(query["code"], ["CODE"])
        self.assertEqual(query["state"], ["S"])

    def test_state_omitted_when_absent(self):
        url = oauth.redirect_with_code("https://claude.ai/cb", "CODE", None)
        self.assertNotIn("state=", url)


# --- OAuth over HTTP, end to end ------------------------------------------

class OAuthHttpTests(ServerFixture):
    """The whole flow against the real listener, with a DCR-registered
    client so no network fetch is involved."""

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path, headers=None):
        req = urllib.request.Request(self.url(path), method="GET")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(), dict(e.headers)

    def form_post(self, path, fields, follow=True):
        data = urllib.parse.urlencode(fields).encode()
        req = urllib.request.Request(self.url(path), data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **kw):
                return None

        opener = (urllib.request.build_opener() if follow
                  else urllib.request.build_opener(NoRedirect))
        try:
            with opener.open(req, timeout=10) as resp:
                return resp.status, resp.read().decode(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(), dict(e.headers)

    def register(self):
        req = urllib.request.Request(
            self.url("/register"),
            data=json.dumps({"redirect_uris": ["http://localhost/callback"],
                             "client_name": "test"}).encode(),
            method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    # --- discovery is reachable without a credential ---------------------

    def test_discovery_documents_need_no_credential(self):
        for path in ("/.well-known/oauth-protected-resource",
                     "/.well-known/oauth-protected-resource/mcp",
                     "/.well-known/oauth-authorization-server"):
            status, body, _ = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertIn("issuer" if "authorization-server" in path
                          else "resource", json.loads(body))

    def test_protected_resource_resource_matches_the_request_host(self):
        _, body, _ = self.get("/.well-known/oauth-protected-resource/mcp")
        self.assertEqual(json.loads(body)["resource"],
                         f"http://127.0.0.1:{self.port}/mcp")

    def test_mcp_401_carries_the_discovery_pointer(self):
        status, _, headers = self.get("/mcp")
        self.assertEqual(status, 401)
        self.assertIn("resource_metadata=", headers["WWW-Authenticate"])
        self.assertIn("oauth-protected-resource/mcp",
                      headers["WWW-Authenticate"])

    # --- the flow ---------------------------------------------------------

    def authorize_query(self, client_id, verifier):
        return urllib.parse.urlencode({
            "response_type": "code", "client_id": client_id,
            "redirect_uri": "http://localhost/callback",
            "code_challenge": challenge_for(verifier),
            "code_challenge_method": "S256", "state": "st-123",
            "scope": oauth.SCOPE})

    def test_consent_page_shows_the_client_host_not_its_claimed_name(self):
        client = self.register()
        status, body, _ = self.get(
            "/authorize?" + self.authorize_query(client["client_id"], "v" * 43))
        self.assertEqual(status, 200)
        self.assertIn("Authorize rig access", body)
        self.assertIn("operator_token", body)

    def test_authorize_rejects_a_plain_pkce_method(self):
        client = self.register()
        query = urllib.parse.urlencode({
            "response_type": "code", "client_id": client["client_id"],
            "redirect_uri": "http://localhost/callback",
            "code_challenge": "abc", "code_challenge_method": "plain"})
        status, _, _ = self.get("/authorize?" + query)
        self.assertEqual(status, 400)

    def test_authorize_rejects_an_unregistered_redirect_uri(self):
        client = self.register()
        query = urllib.parse.urlencode({
            "response_type": "code", "client_id": client["client_id"],
            "redirect_uri": "https://evil.test/cb",
            "code_challenge": challenge_for("v" * 43),
            "code_challenge_method": "S256"})
        status, _, _ = self.get("/authorize?" + query)
        self.assertEqual(status, 400)

    def test_wrong_operator_token_issues_no_code(self):
        client = self.register()
        status, body, headers = self.form_post("/authorize", {
            "client_id": client["client_id"],
            "redirect_uri": "http://localhost/callback",
            "code_challenge": challenge_for("v" * 43),
            "code_challenge_method": "S256", "state": "st-123",
            "operator_token": "b" * 64}, follow=False)
        self.assertEqual(status, 401)
        self.assertNotIn("Location", headers)
        self.assertIn("did not match", body)
        self.assertEqual(oauth._CODES, {})

    def full_flow(self, verifier="verifier-" + "q" * 40):
        client = self.register()
        _, _, headers = self.form_post("/authorize", {
            "client_id": client["client_id"],
            "redirect_uri": "http://localhost/callback",
            "code_challenge": challenge_for(verifier),
            "code_challenge_method": "S256", "state": "st-123",
            "operator_token": TOKEN}, follow=False)
        location = headers["Location"]
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(location).query)
        return client, query, verifier

    def test_consent_redirects_with_a_code_and_the_state(self):
        client, query, _ = self.full_flow()
        self.assertEqual(query["state"], ["st-123"])
        self.assertTrue(query["code"][0])

    def test_the_operator_token_never_reaches_the_redirect(self):
        _, query, _ = self.full_flow()
        self.assertNotIn(TOKEN, json.dumps(query))

    def test_code_exchange_returns_a_working_access_token(self):
        client, query, verifier = self.full_flow()
        status, body, _ = self.form_post("/token", {
            "grant_type": "authorization_code", "code": query["code"][0],
            "client_id": client["client_id"],
            "redirect_uri": "http://localhost/callback",
            "code_verifier": verifier})
        self.assertEqual(status, 200)
        tokens = json.loads(body)
        self.assertEqual(tokens["token_type"], "Bearer")
        # and it opens the MCP endpoint
        code, resp = self.rpc("tools/list", token=tokens["access_token"])
        self.assertEqual(code, 200)
        self.assertEqual(len(resp["result"]["tools"]), 4)

    def test_code_exchange_with_a_bad_verifier_is_invalid_grant(self):
        client, query, _ = self.full_flow()
        status, body, _ = self.form_post("/token", {
            "grant_type": "authorization_code", "code": query["code"][0],
            "client_id": client["client_id"],
            "redirect_uri": "http://localhost/callback",
            "code_verifier": "wrong-verifier"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "invalid_grant")

    def test_code_cannot_be_replayed(self):
        client, query, verifier = self.full_flow()
        fields = {"grant_type": "authorization_code", "code": query["code"][0],
                  "client_id": client["client_id"],
                  "redirect_uri": "http://localhost/callback",
                  "code_verifier": verifier}
        self.assertEqual(self.form_post("/token", fields)[0], 200)
        status, body, _ = self.form_post("/token", fields)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "invalid_grant")

    def test_refresh_over_http_rotates(self):
        client, query, verifier = self.full_flow()
        _, body, _ = self.form_post("/token", {
            "grant_type": "authorization_code", "code": query["code"][0],
            "client_id": client["client_id"],
            "redirect_uri": "http://localhost/callback",
            "code_verifier": verifier})
        first = json.loads(body)
        status, body, _ = self.form_post("/token", {
            "grant_type": "refresh_token",
            "refresh_token": first["refresh_token"],
            "client_id": client["client_id"]})
        self.assertEqual(status, 200)
        second = json.loads(body)
        self.assertNotEqual(first["refresh_token"], second["refresh_token"])
        self.assertEqual(self.rpc("tools/list",
                                  token=second["access_token"])[0], 200)

    def test_unsupported_grant_type(self):
        status, body, _ = self.form_post("/token", {
            "grant_type": "client_credentials"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "unsupported_grant_type")

    def test_token_endpoint_sets_no_store(self):
        status, _, headers = self.form_post("/token", {"grant_type": "nope"})
        self.assertEqual(headers.get("Cache-Control"), "no-store")

    def test_cimd_client_completes_the_flow_over_http(self):
        """The path Claude actually takes: no /register call, the
        client_id is a URL the server dereferences at authorize time."""
        client_id = "https://claude.ai/oauth/client-metadata"
        doc = {"client_id": client_id,
               "client_name": "Claude",
               "redirect_uris": [oauth.CLAUDE_CALLBACK]}
        verifier = "verifier-" + "c" * 40
        # Patch the fetch, not urllib: `oauth.urllib.request` is the same
        # module object this test's own HTTP client uses, so patching
        # there breaks the requests the test is trying to make. The real
        # fetch and its self-referentiality check are covered by
        # CimdTests.
        with mock.patch.object(oauth, "fetch_cimd",
                               lambda cid, opener=None: doc):
            status, body, _ = self.get("/authorize?" + urllib.parse.urlencode({
                "response_type": "code", "client_id": client_id,
                "redirect_uri": oauth.CLAUDE_CALLBACK,
                "code_challenge": challenge_for(verifier),
                "code_challenge_method": "S256", "state": "st-9"}))
            self.assertEqual(status, 200)
            # The relying party shown is the host of the client_id URL,
            # not the document's self-asserted client_name.
            self.assertIn("claude.ai", body)
            _, _, headers = self.form_post("/authorize", {
                "client_id": client_id,
                "redirect_uri": oauth.CLAUDE_CALLBACK,
                "code_challenge": challenge_for(verifier),
                "code_challenge_method": "S256", "state": "st-9",
                "operator_token": TOKEN}, follow=False)
        query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(headers["Location"]).query)
        self.assertEqual(query["state"], ["st-9"])
        status, body, _ = self.form_post("/token", {
            "grant_type": "authorization_code", "code": query["code"][0],
            "client_id": client_id, "redirect_uri": oauth.CLAUDE_CALLBACK,
            "code_verifier": verifier})
        self.assertEqual(status, 200)
        self.assertEqual(
            self.rpc("tools/list",
                     token=json.loads(body)["access_token"])[0], 200)

    def test_static_token_still_works_alongside_oauth(self):
        """The header-auth path must keep working: it is what runs the
        day the static_headers beta reaches this account."""
        self.assertEqual(self.rpc("tools/list")[0], 200)

    def test_oauth_events_are_audited(self):
        self.full_flow()
        tools = [e["tool"] for e in self.audit_lines()]
        self.assertIn("oauth/register", tools)
        self.assertIn("oauth/consent", tools)

    def test_operator_token_is_not_written_to_the_audit_log(self):
        self.full_flow()
        blob = Path(os.environ["MODELCTL_REMOTE_HANDS_AUDIT_PATH"]).read_text()
        self.assertNotIn(TOKEN, blob)


if __name__ == "__main__":
    unittest.main()
