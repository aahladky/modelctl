"""LlamaSwapClient: typed access to the llama-swap control surface.

Covers service health, registered/running models, unload (one/all), warm
load (via a minimal OpenAI request through the public path), and logs.
All methods degrade to structured ModelctlSwapError on failure -- callers
never see bare URLError/timeout exceptions.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:9292"


class ModelctlSwapError(Exception):
    def __init__(self, code, message, detail=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}


class LlamaSwapClient:
    def __init__(self, base_url=DEFAULT_BASE, timeout=10):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # -- transport --------------------------------------------------------
    def _request(self, method, path, body=None, timeout=None):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"} if body else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                raw = r.read()
                if not raw:
                    return {}
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    # Several llama-swap endpoints answer plain text ("OK")
                    # rather than JSON -- that's a success, not a parse error.
                    return {"text": raw.decode(errors="replace").strip()}
        except urllib.error.HTTPError as e:
            raise ModelctlSwapError(
                "SWAP_HTTP_ERROR", f"llama-swap {method} {path} -> {e.code}",
                {"status": e.code, "body": e.read()[:500].decode(errors="replace")})
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise ModelctlSwapError(
                "LLAMA_SWAP_UNAVAILABLE", f"llama-swap unreachable: {e}")

    @staticmethod
    def _enc(model_id):
        return urllib.parse.quote(model_id, safe="")

    # -- read -------------------------------------------------------------
    def health(self):
        return self._request("GET", "/health")

    def registered_models(self):
        data = self._request("GET", "/v1/models")
        return data.get("data", [])

    def running_models(self):
        data = self._request("GET", "/running")
        return data.get("running", [])

    def running_model_ids(self):
        return {m.get("model") for m in self.running_models()}

    def logs(self, model_id=None, lines=200):
        if model_id:
            return self._request("GET", f"/logs/stream/{self._enc(model_id)}")
        return self._request("GET", "/logs")

    # -- lifecycle --------------------------------------------------------
    def unload(self, model_id):
        return self._request("POST", f"/api/models/unload/{self._enc(model_id)}")

    def unload_all(self):
        return self._request("POST", "/api/models/unload")

    def warm_load(self, model_id, timeout=1800):
        """Trigger a load via a minimal request through the public path.
        Success = valid response AND the model shows in /running afterwards."""
        body = {"model": model_id,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 1, "temperature": 0}
        started = time.time()
        resp = self._request("POST", "/v1/chat/completions", body,
                             timeout=timeout)
        running = model_id in self.running_model_ids()
        return {"loaded": running, "elapsed_s": round(time.time() - started, 1),
                "response_ok": bool(resp.get("choices"))}

    def model_state(self, model_id, registered=None, running=None):
        """One of: unregistered, stopped, ready, loading, queued, failed, unknown."""
        try:
            registered = registered if registered is not None else self.registered_models()
            running = running if running is not None else self.running_models()
        except ModelctlSwapError:
            return {"state": "unavailable", "worker": None}
        for m in running:
            if m.get("model") == model_id:
                return {"state": m.get("state", "ready"), "worker": m}
        if any(m.get("id") == model_id for m in registered):
            return {"state": "stopped", "worker": None}
        return {"state": "unregistered", "worker": None}

    @staticmethod
    def _worker_port(worker):
        """Upstream port for a running model. Newer llama-swap /running
        payloads omit "port"/"pid"/"started" and expose the upstream as
        "proxy": "http://localhost:PORT" -- derive the port from that."""
        port = worker.get("port")
        if port:
            return port
        proxy = worker.get("proxy") or ""
        try:
            return int(urllib.parse.urlparse(proxy).port)
        except (TypeError, ValueError):
            return None

    def runtime_state(self):
        """Aggregate state for all registered + running models.

        Returns dict keyed by model_id with state, worker info, and flags.
        Degrades gracefully to empty dict when llama-swap is unreachable.
        """
        try:
            registered = self.registered_models()
            running = self.running_models()
        except ModelctlSwapError:
            return {}
        reg_ids = {m.get("id") for m in registered}
        run_map = {}
        for m in running:
            mid = m.get("model", "")
            run_map[mid] = m
        out = {}
        for mid in reg_ids | set(run_map):
            worker = run_map.get(mid)
            if worker:
                raw = worker.get("state", "ready")
            elif mid in reg_ids:
                raw = "stopped"
            else:
                raw = "unknown"
            out[mid] = {
                "model_id": mid,
                "state": raw,
                "registered": mid in reg_ids,
                "running": mid in run_map,
                "pid": worker.get("pid") if worker else None,
                "port": self._worker_port(worker) if worker else None,
                "started": worker.get("started") if worker else None,
                "state_class": self._state_class(raw),
            }
        return out

    @staticmethod
    def _state_class(state):
        if state in ("ready",):
            return "ok"
        if state in ("loading", "queued", "unloading"):
            return "active"
        if state in ("failed",):
            return "warn"
        if state in ("stopped",):
            return "stopped"
        return ""
