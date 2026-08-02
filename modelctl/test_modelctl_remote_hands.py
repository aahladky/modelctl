"""Tests for the remote-hands MCP server.

The four things worth failing over, in the order the order named them:
auth rejection, allowlist enforcement, audit logging, and on/off
ordering. Everything here runs against a loopback listener on an
ephemeral port with a throwaway state dir -- no funnel is configured, no
systemd unit is touched, and no real allowlist root is written to.
"""
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import modelctl_remote_hands as rh

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
            "MODELCTL_REMOTE_HANDS_ROOTS": str(self.allowed),
            "MODELCTL_REMOTE_HANDS_TOKEN": "",
        }
        self._saved = {k: os.environ.get(k) for k in self._env}
        os.environ.update(self._env)
        (self.root / "token").write_text(TOKEN + "\n")
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

class ServerTests(RemoteHandsBase):
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


if __name__ == "__main__":
    unittest.main()
