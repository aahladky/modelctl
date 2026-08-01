"""FastAPI application for the modelctl web console.

Reads are direct calls into modelctl (concurrent); every write goes through
the single JobRunner worker (see jobs.py). Auth: one shared token, checked
as Bearer header or the login cookie (never a query param -- tokens in URLs
leak into logs and history).
"""
import json
import os
import secrets
import time
import urllib.request
from dataclasses import asdict
from urllib.parse import quote
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, RedirectResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import modelctl
import modelctl_errors
import modelctl_vram

from . import mutate, telemetry
from .jobs import (JobRunner, JobStore, STATE_DIR, scratch_safe_mode,
                   scratch_missing_redirections)

TOKEN_PATH = STATE_DIR / "web_token"
COOKIE_NAME = "modelctl_web_token"
TEMPLATE_DIR = Path(__file__).parent / "templates"
CONSOLE_DIST = Path(__file__).parent.parent / "console" / "dist"

from .swap import LlamaSwapClient, ModelctlSwapError

LLAMA_SWAP_BASE = os.environ.get("MODELCTL_LLAMA_SWAP_BASE_URL",
                                 "http://127.0.0.1:9292/v1/")
LLAMA_SWAP_ROOT = LLAMA_SWAP_BASE.rsplit("/v1/", 1)[0]


def load_or_create_token():
    env = os.environ.get("MODELCTL_WEB_TOKEN", "").strip()
    if env:
        return env
    try:
        stored = TOKEN_PATH.read_text().strip()
        # An empty token file (interrupted write, `touch`, full disk) must
        # never mean "no auth": an anonymous request supplies "" too, and
        # "" == "" would open the whole console. Regenerate instead.
        if stored:
            return stored
    except OSError:
        pass
    token = secrets.token_hex(16)
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token + "\n")
    os.chmod(TOKEN_PATH, 0o600)
    return token


def _safe_back(target):
    """An internal path to return to, or "/".

    `back` is reflected into an <a href> and into the polling URL; a
    "javascript:..." value there executes in the console's origin, with
    the session cookie attached, able to drive any mutating API.
    """
    target = str(target or "/")
    if not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


def _fetch_json(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


# Shared with the measurement path, which needs the same counters to say
# whether a benchmark's cache was actually running.
_scrape_moe_cache_metrics = modelctl.scrape_moe_cache_metrics


# ---- scratch-safe mode (MODELCTL_WEB_SCRATCH=1) -------------------------
# A scratch instance exists to be walked, never to drive the stack: every
# mutating endpoint answers 405 with the reason; reads and SSE behave
# exactly as live. The one carve-out is the add-wizard chain running its
# own scratch jobs -- and only when every path/service that chain can
# touch has been redirected off the live install, so the carve-out is
# provably unable to reach it.

_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# Legacy convenience GETs that write wizard state as a side effect.
# (Other wizard GETs refresh state opportunistically; those writes are
# no-op'd centrally in WizardStore when scratch shares the live state.)
_MUTATING_GET_PREFIXES = ("/pull/", "/add/start/")
# The add-wizard chain: old server-rendered steps and the /v2 SPA API.
_WIZARD_PREFIXES = ("/add", "/api/v2/wizard", "/api/v2/wizards")


def _is_wizard_path(path):
    return any(path == p or path.startswith(p + "/")
               for p in _WIZARD_PREFIXES)


def _scratch_refusal(request):
    """A 405 response if scratch-safe mode refuses this request, else None."""
    path = request.url.path
    mutating = (request.method in _MUTATING_METHODS
                or any(path.startswith(p) for p in _MUTATING_GET_PREFIXES))
    if not mutating:
        return None
    if _is_wizard_path(path):
        missing = scratch_missing_redirections()
        if not missing:
            return None  # hermetic universe: the wizard's own scratch jobs may run
        reason = ("scratch-safe mode (MODELCTL_WEB_SCRATCH=1): the add-wizard "
                  "chain is only allowed once the whole state universe is "
                  "redirected off the live install; not redirected: "
                  + ", ".join(missing))
    else:
        reason = (f"scratch-safe mode (MODELCTL_WEB_SCRATCH=1): this instance "
                  f"refuses mutations; {request.method} {path} would write "
                  f"state or drive the serving stack")
    if path.startswith("/api/"):
        return JSONResponse({"error": "scratch-safe mode", "reason": reason},
                            status_code=405)
    return PlainTextResponse(reason + "\n", status_code=405)


def create_app(token=None, store=None, runner=None, collector=None,
               tick_interval=2.0, tick_max_seconds=3600):
    token = token or load_or_create_token()
    store = store or JobStore()
    runner = runner or JobRunner(store)
    collector = collector or telemetry.TelemetryCollector(store=store)
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    templates.env.filters["fmt_size"] = modelctl._format_size
    templates.env.filters["fromjson"] = json.loads

    app = FastAPI(title="modelctl-web", docs_url=None, redoc_url=None)
    app.state.token = token
    app.state.store = store
    app.state.runner = runner
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")),
              name="static")

    @app.exception_handler(modelctl_errors.ProfileNotFoundError)
    async def profile_not_found(request: Request, exc):
        # A stale bookmark or a profile deleted in another tab is a 404,
        # not a 500 with a traceback in the journal.
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": str(exc)}, status_code=404)
        return PlainTextResponse(f"{exc}\n", status_code=404)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        # A bare "Internal Server Error" is a dead end; keep the operator
        # in the app with the error visible and a way back. The traceback
        # still goes to the journal.
        import sys
        import traceback as _tb
        print(f"unhandled error at {request.url.path}", file=sys.stderr)
        _tb.print_exception(exc, file=sys.stderr)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                                status_code=500)
        # Minimal context on purpose: ctx() itself touches setup probing,
        # which must not be able to take the error page down with it.
        return templates.TemplateResponse(
            request=request, name="error.html",
            context={"heading": "something went wrong",
                     "message": f"{type(exc).__name__}: {exc}",
                     "setup_blocking": []},
            status_code=500)

    scratch = scratch_safe_mode()

    @app.middleware("http")
    async def auth(request: Request, call_next):
        # /login stays reachable in scratch mode too: authenticating is
        # how a scratch walk gets in, and it mutates nothing server-side.
        if request.url.path in ("/login", "/healthz") or request.url.path.startswith("/static"):
            return await call_next(request)
        supplied = (
            request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            or request.cookies.get(COOKIE_NAME, ""))
        if not supplied or not secrets.compare_digest(supplied, token):
            if request.url.path.startswith("/api/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return RedirectResponse(f"/login?next={request.url.path}",
                                    status_code=303)
        if request.method == "POST":
            origin = (request.headers.get("origin")
                      or request.headers.get("referer") or "")
            if origin:
                from urllib.parse import urlparse
                if urlparse(origin).netloc != request.headers.get("host", ""):
                    return JSONResponse({"error": "cross-origin POST rejected"},
                                        status_code=403)
        if scratch:
            refusal = _scratch_refusal(request)
            if refusal is not None:
                return refusal
        return await call_next(request)

    # ---- helpers --------------------------------------------------------
    def profiles():
        out = []
        for p in sorted(modelctl.PROFILES_DIR.glob("*.json")):
            try:
                out.append(modelctl.load_profile(p.stem))
            except Exception:
                continue
        return out

    def _swap_client():
        return LlamaSwapClient(base_url=LLAMA_SWAP_ROOT)

    def live_state():
        running = _fetch_json(f"{LLAMA_SWAP_ROOT}/running") or {}
        models = _fetch_json(f"{LLAMA_SWAP_BASE}models") or {}
        loaded = {m["model"] for m in running.get("running", [])}
        registered = {m["id"] for m in models.get("data", [])}
        return loaded, registered

    def _runtime_state():
        return _swap_client().runtime_state()

    def placement_summary(profile):
        cfg = profile.get("config", {})
        if cfg.get("split_mode") and cfg.get("tensor_split"):
            base = f"split {cfg['tensor_split']} ({cfg['split_mode']})"
        else:
            base = cfg.get("device") or "(backend default)"
        extras = []
        if cfg.get("fit") == "on":
            extras.append("fit")
        if "exps=CPU" in (cfg.get("extra") or ""):
            extras.append("CPU experts")
        if "-ngl" in (cfg.get("extra") or ""):
            extras.append("partial ngl")
        return base + (f" +{','.join(extras)}" if extras else "")

    def ctx(request, **kw):
        d = {"request": request, "setup_blocking": _setup_banner()}
        d.update(kw)
        return d

    # Readiness for the nav banner, on every page. Cached briefly because
    # it runs on every render and the underlying state (a directory
    # appearing, llama-swap coming up) does not change per request.
    _setup_cache = {"at": 0.0, "blocking": (), "first_run": False}

    def _refresh_setup_cache():
        now = time.time()
        if now - _setup_cache["at"] > 30:
            try:
                import modelctl_setup
                # probe_backend=False: no subprocess on a page render.
                status = modelctl_setup.probe_setup()
                _setup_cache["blocking"] = status.blocking
                _setup_cache["first_run"] = status.first_run
            except Exception:
                _setup_cache["blocking"] = ()
                _setup_cache["first_run"] = False
            _setup_cache["at"] = now
        return _setup_cache

    def _setup_banner():
        return _refresh_setup_cache()["blocking"]

    # ---- template filters ------------------------------------------------
    def _fmt_elapsed(start_ts):
        if not start_ts:
            return ""
        # `started` comes verbatim from the external llama-swap /running
        # payload, whose shape has drifted before -- a string timestamp
        # must degrade, not 500 the whole runtime page.
        try:
            s = max(0, int(time.time() - float(start_ts)))
        except (TypeError, ValueError):
            return ""
        if s < 60:
            return f"{s}s ago"
        if s < 3600:
            return f"{s // 60}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"

    templates.env.filters["elapsed"] = _fmt_elapsed

    def _state_badge_filter(state, state_class=""):
        labels = {"ready": "ready", "loading": "loading", "queued": "queued",
                  "stopped": "stopped", "failed": "failed", "unloading": "unloading",
                  "unregistered": "unregistered", "unavailable": "swap down",
                  "unknown": "unknown"}
        cls = state_class or LlamaSwapClient._state_class(state)
        label = labels.get(state, state)
        return f'<span class="{cls}">{label}</span>'

    templates.env.filters["state_badge"] = _state_badge_filter

    # ---- auth pages -----------------------------------------------------
    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz():
        return "ok"

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/"):
        if not next.startswith("/") or next.startswith("//"):
            next = "/"
        return templates.TemplateResponse(request=request, name="login.html", context=ctx(
            request, next=next, error=""))

    # Global (not per-IP): this is a single-user console; the goal is to
    # blunt unattended guessing, not to referee concurrent users.
    login_failures = {"count": 0, "last": 0.0}

    @app.post("/login")
    def login(request: Request, next: str = Form("/"), token_field: str = Form("")):
        if not next.startswith("/") or next.startswith("//"):
            next = "/"
        now = time.monotonic()
        if login_failures["count"] >= 5 and now - login_failures["last"] < 30.0:
            return templates.TemplateResponse(
                request=request, name="login.html", status_code=429,
                context=ctx(request, next=next,
                            error="Too many failed attempts -- wait 30 seconds."))
        if not token_field or not secrets.compare_digest(token_field, token):
            login_failures["count"] += 1
            login_failures["last"] = now
            # Render the form with the error instead of redirecting: a 307
            # redirect used to re-POST the same body in a loop, and the old
            # template had no error slot at all.
            return templates.TemplateResponse(
                request=request, name="login.html", status_code=401,
                context=ctx(request, next=next, error="Wrong token."))
        login_failures["count"] = 0
        # 303 turns the follow-up into a GET; the default 307 re-POSTed to
        # `next`, which both broke the login flow (405 on /) and let a
        # crafted ?next= re-POST a fresh login into a mutating endpoint.
        resp = RedirectResponse(next, status_code=303)
        resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict",
                        secure=os.environ.get("MODELCTL_WEB_SECURE_COOKIE", "") == "1")
        return resp

    @app.get("/logout")
    def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME)
        return resp

    # ---- dashboard ------------------------------------------------------
    # ---- first run / setup ----------------------------------------------
    @app.post("/setup/reprobe")
    def setup_reprobe():
        # The reprobe escape hatch: drop every cached capability verdict
        # and let the setup page's live probe rebuild them. Without this,
        # recovering from a poisoned entry meant hand-deleting files in
        # the state dir (documented nowhere).
        import modelctl_capabilities
        modelctl_capabilities.clear_cache()
        return RedirectResponse("/setup", status_code=303)

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page(request: Request):
        import modelctl_setup
        # The setup page is the one place a live capability probe is worth
        # a subprocess: it is what makes "runtime found" mean "runtime that
        # answers for its own features" rather than "a file exists".
        status = modelctl_setup.probe_setup(probe_backend=True)
        bind = os.environ.get("MODELCTL_WEB_BIND", "0.0.0.0:9293")
        _host, _, port = bind.rpartition(":")
        return templates.TemplateResponse(request=request, name="setup.html", context=ctx(
            request, status=status,
            console_url=f"http://{request.url.hostname}:{port or 9293}/",
            token_path=str(TOKEN_PATH),
            service_name="modelctl-web.service"))

    @app.get("/api/setup")
    def api_setup():
        import modelctl_setup
        status = modelctl_setup.probe_setup(probe_backend=True)
        return {"ready": status.ready, "first_run": status.first_run,
                "checks": [asdict(c) for c in status.checks]}

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        # A machine nobody has configured has nothing to show on a
        # dashboard -- send them where the work actually starts. Only on a
        # genuine first run: an established install with llama-swap down
        # keeps its dashboard.
        # Reuses the 30s-cached probe the nav banner already runs on
        # every render; a second uncached probe here meant five HTTP
        # calls with 2s timeouts each whenever llama-swap was down.
        if _refresh_setup_cache()["first_run"]:
            return RedirectResponse("/setup", status_code=303)
        loaded, registered = live_state()
        runtime = _runtime_state()
        inventory = modelctl.get_gpu_inventory()
        ram_avail = modelctl_vram.system_ram_available()
        rows = []
        for p in profiles():
            name = p["name"]
            rt = runtime.get(name, {})
            rows.append({
                "name": name,
                "repo": p.get("repo_id", ""),
                "file": p.get("file", ""),
                "backend": p.get("backend", "llama-cpp"),
                "placement": placement_summary(p),
                "ctx": p.get("config", {}).get("ctx", ""),
                "enabled": p.get("enabled", True),
                "loaded": name in loaded,
                "registered": name in registered,
                "rt_state": rt.get("state", "stopped" if name in registered else "unregistered"),
                "rt_state_class": rt.get("state_class", ""),
                "pid": rt.get("pid"),
                "port": rt.get("port"),
                "started": rt.get("started"),
            })
        return templates.TemplateResponse(request=request, name="dashboard.html", context=ctx(
            request, rows=rows, inventory=inventory, ram_avail=ram_avail))

    # ---- profile edit ---------------------------------------------------
    EDIT_FIELDS = ["device", "split_mode", "tensor_split", "ctx",
                   "cache_type_k", "cache_type_v", "flash_attn", "ttl",
                   "mtp", "fit", "extra", "binary"]

    @app.get("/profiles/{name}", response_class=HTMLResponse)
    def profile_edit(request: Request, name: str, saved: str = ""):
        p = modelctl.load_profile(name)
        # The preview is the canonical command, not a re-derivation of it:
        # this page used to probe whatever binary the profile *named*, which
        # is frequently not the one preflight's auto-fix search resolves and
        # actually launches.
        cmd, _ok, messages = modelctl.canonical_launch_command(p)
        caps = cmd.backend.capabilities
        return templates.TemplateResponse(request=request, name="profile_edit.html", context=ctx(
            request, p=p, fields=EDIT_FIELDS, saved=saved,
            run_sh=" \\\n  ".join(cmd.argv), messages=messages,
            command_fingerprint=cmd.command_fingerprint,
            validation=cmd.validation, warnings=cmd.warnings,
            capabilities=caps))

    @app.post("/profiles/{name}")
    async def profile_save(request: Request, name: str):
        form = await request.form()
        updates = {}
        for f in EDIT_FIELDS:
            if f in form:
                updates[f] = str(form[f])
        # Checkbox-absent means "unchecked" only when the full edit form
        # was submitted (it carries _full_form). A partial programmatic
        # POST that just set one field used to silently disable the
        # profile, which unregistered the model from llama-swap on sync.
        if "_full_form" in form:
            updates["enabled"] = "enabled" in form
        elif "enabled" in form:
            updates["enabled"] = str(form["enabled"]).lower() in ("1", "true", "on")
        job_id = mutate.submit_edit(runner, name, updates)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}", status_code=303)

    @app.post("/profiles/{name}/moe-cache")
    async def profile_moe_cache_save(request: Request, name: str):
        form = await request.form()
        # Start from the profile's existing moe_cache, not a fresh skeleton:
        # rebuilding from scratch silently dropped every section this form
        # does not render (ram, prefetch, storage details, and `mode` when
        # the select was absent), so a CLI-set option vanished on any save.
        existing = modelctl.load_profile(name).get("moe_cache", {})
        mc = json.loads(json.dumps(existing)) if existing else {}
        mc.setdefault("gpu", {}).setdefault("budgets_bytes", {})
        for section in ("ram", "storage", "prefill", "decode", "prefetch"):
            mc.setdefault(section, {})
        budgets = {}
        new_device = str(form.get("gpu.budgets_bytes._new_device", "")).strip()
        new_budget = str(form.get("gpu.budgets_bytes._new_budget", "")).strip()
        for key, val in form.items():
            val = str(val)
            if key == "mode":
                mc["mode"] = val
            elif key.startswith("gpu.budgets_bytes."):
                dev = key.split(".", 2)[2]
                if dev.startswith("_"):
                    continue  # the add-device inputs, handled below
                try:
                    budgets[dev] = int(val)
                except ValueError:
                    pass
            elif key == "gpu.policy":
                mc["gpu"]["policy"] = val
            elif key == "gpu.admission_misses":
                try:
                    mc["gpu"]["admission_misses"] = int(val)
                except ValueError:
                    pass
            elif key == "decode.miss_execution":
                mc["decode"]["miss_execution"] = val
            elif key == "prefill.admit_to_gpu_cache":
                mc["prefill"]["admit_to_gpu_cache"] = val == "true"
            elif key == "storage.mode":
                mc["storage"]["mode"] = val
        # The add-device row: previously the device name was read and
        # discarded and there was no field for its value at all, so a new
        # per-GPU budget could never be set from the console.
        if new_device and new_budget:
            try:
                budgets[new_device] = int(new_budget)
            except ValueError:
                pass
        # Strip empty budgets (also how a device is removed: clear its box).
        mc["gpu"]["budgets_bytes"] = {k: v for k, v in budgets.items() if v > 0}
        job_id = mutate.submit_moe_cache(runner, name, mc)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}", status_code=303)

    @app.post("/profiles/{name}/delete")
    def profile_delete(name: str):
        def fn(ctx):
            modelctl.cmd_remove(type("A", (), {"name": name, "no_hermes": True,
                                               "no_router_restart": False})())
            return {"removed": name}
        job_id = runner.submit("remove", f"remove {name}", fn)
        return RedirectResponse(f"/jobs/{job_id}?back=/", status_code=303)

    @app.get("/profiles/{name}/run.sh", response_class=PlainTextResponse)
    def profile_runsh(name: str):
        p = modelctl.load_profile(name)
        # Was the one preview that passed no capabilities at all and omitted
        # the resolved binary, so it could show a command the profile's own
        # generated run.sh contradicted.
        cmd, _ok, _messages = modelctl.canonical_launch_command(p)
        return " \\\n  ".join(cmd.argv)

    # ---- tier planner ---------------------------------------------------
    def _plan_for(name):
        import modelctl_tiers
        p = modelctl.load_profile(name)
        # Stored-first inputs and the same plan path submit_tier_apply
        # uses, so the preview is exactly what apply would compute.
        plan, inputs, source = modelctl.plan_tiers_for_profile(p)
        gate = (modelctl_tiers.tier_change_gate(p, plan)
                if plan is not None else None)
        return p, plan, inputs, source, gate

    @app.get("/profiles/{name}/tiers", response_class=HTMLResponse)
    def tiers_one(request: Request, name: str):
        p, plan, inputs, source, gate = _plan_for(name)
        return templates.TemplateResponse(request=request, name="tiers.html", context=ctx(
            request, p=p, plan=plan, current=p.get("config", {}),
            inputs=inputs, inputs_source=source, gate=gate))

    @app.post("/profiles/{name}/tiers/apply")
    def tiers_apply(name: str, accept_tier_change: bool = Form(False)):
        job_id = mutate.submit_tier_apply(runner, name,
                                          accept_tier_change=accept_tier_change)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}/tiers",
                                status_code=303)

    @app.get("/tiers", response_class=HTMLResponse)
    def tiers_all(request: Request):
        inventory = modelctl.get_gpu_inventory()
        d = modelctl.load_defaults()
        primary = modelctl.resolve_primary_gpu(inventory, d)
        plans = []
        for p in profiles():
            if not p.get("model_path"):
                continue
            plan, _inputs, source = modelctl.plan_tiers_for_profile(
                p, inventory=inventory, defaults=d, primary=primary)
            plans.append({"name": p["name"], "plan": plan,
                          "inputs_source": source})
        return templates.TemplateResponse(request=request, name="tiers_all.html", context=ctx(
            request, plans=plans))

    # ---- legacy acquisition routes --------------------------------------
    # /add is the ONE acquisition workflow: it owns validation, job state,
    # local-file verification, analysis, plan generation, and registration.
    # These routes exist only for old bookmarks and muscle memory -- they
    # create a pre-populated wizard and redirect into it, so there is no
    # second path that half-owns acquisition.

    def _wizard_for_repo(repo_id: str):
        from .wizard import WizardState, WizardStore
        state = WizardState()
        state.source_type = "hf_repo"
        state.repo_id = repo_id
        state.advance("inspect")
        WizardStore().save(state)
        return state

    @app.get("/pull", response_class=HTMLResponse)
    def pull_form(q: str = ""):
        # Search now lives on the wizard's source step.
        target = f"/add?q={q}" if q else "/add"
        return RedirectResponse(target, status_code=303)

    @app.get("/pull/{repo_id:path}", response_class=HTMLResponse)
    def pull_repo(repo_id: str):
        state = _wizard_for_repo(repo_id)
        return RedirectResponse(f"/add/{state.wizard_id}/inspect",
                                status_code=303)

    @app.post("/pull/{repo_id:path}")
    def pull_start(repo_id: str, quant: str = Form("")):
        # Quant selection happens on the wizard's inspect step, where the
        # recommendation logic lives.
        state = _wizard_for_repo(repo_id)
        return RedirectResponse(f"/add/{state.wizard_id}/inspect",
                                status_code=303)

    @app.get("/import", response_class=HTMLResponse)
    def import_form():
        return RedirectResponse("/add", status_code=303)

    @app.post("/import")
    def import_start(file_path: str = Form(...), name: str = Form(""),
                     copy: bool = Form(False)):
        from .wizard import WizardState, WizardStore
        state = WizardState()
        state.source_type = "local_file"
        state.local_path = file_path
        WizardStore().save(state)
        # The source step re-verifies on submit, so a bad path becomes a
        # form error instead of a background-job failure.
        return RedirectResponse(f"/add/{state.wizard_id}/source",
                                status_code=303)

    # ---- add-model wizard -----------------------------------------------
    @app.get("/add", response_class=HTMLResponse)
    def wizard_list(request: Request, q: str = ""):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        active = store_wiz.list_active()
        # Hugging Face search lives here now (it used to be the separate
        # /pull page): searching and starting the acquisition are one
        # workflow, not two. HF being down is a message, not a 500.
        results, search_error = [], ""
        if q:
            try:
                results = modelctl.search_models(q)
            except Exception as e:
                search_error = f"search failed: {e}"
        return templates.TemplateResponse(request=request, name="wizard_list.html",
                                          context=ctx(request, wizards=active,
                                                      q=q, results=results,
                                                      search_error=search_error))

    @app.get("/add/start/{repo_id:path}")
    def wizard_start_from_repo(repo_id: str):
        """One click from a search result into the wizard's inspect step."""
        state = _wizard_for_repo(repo_id)
        return RedirectResponse(f"/add/{state.wizard_id}/inspect",
                                status_code=303)

    @app.post("/add")
    def wizard_create():
        from .wizard import WizardState, WizardStore
        store_wiz = WizardStore()
        state = WizardState()
        store_wiz.save(state)
        return RedirectResponse(f"/add/{state.wizard_id}/source", status_code=303)

    @app.get("/add/{wizard_id}/source", response_class=HTMLResponse)
    def wizard_source(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        return templates.TemplateResponse(request=request, name="wizard_source.html",
                                          context=ctx(request, w=state))

    @app.post("/add/{wizard_id}/source")
    async def wizard_source_submit(request: Request, wizard_id: str):
        from .wizard import WizardStore, WizardState
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        form = await request.form()
        source_type = form.get("source_type", "")
        state.source_type = source_type
        state.clear_error()
        if source_type == "hf_repo":
            # Validate before advancing: an empty or malformed repo id used
            # to sail through to the inspect step, which then silently
            # bounced back to /add -- a wizard stuck in a redirect loop with
            # no error ever shown.
            repo_id = (form.get("repo_id") or "").strip()
            if not repo_id or "/" not in repo_id.strip("/"):
                state.set_error(
                    "enter a Hugging Face repo id like org/model-name"
                    if not repo_id else
                    f"'{repo_id}' is not a repo id -- expected org/model-name")
                store_wiz.save(state)
                return RedirectResponse(f"/add/{wizard_id}/source",
                                        status_code=303)
            state.repo_id = repo_id
            state.advance("inspect")
        elif source_type == "local_file":
            state.local_path = form.get("local_path", "")
            # Verify the file BEFORE a profile exists: GGUF
            # magic, truncation, shard completeness, readability, duplicate
            # identity. A bad path caught here is a form error; caught
            # later it is a broken profile that fails at load time.
            from modelctl_services import acquisition_service
            check = acquisition_service.verify_local_gguf(state.local_path)
            state.source_verification = {
                "ok": check.ok,
                "messages": list(check.messages),
                "warnings": list(check.warnings),
                "data": dict(check.data),
            }
            if not check.ok:
                state.set_error(check.messages[0] if check.messages
                                else "the selected file is not usable")
                store_wiz.save(state)
                return RedirectResponse(f"/add/{wizard_id}/source", status_code=303)
            state.advance("download")
            _submit_download(state)
        else:
            # An unknown source_type used to fall through silently, landing
            # the user back on the same form with no explanation.
            state.set_error("pick a source: Hugging Face repository or local file")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/{state.step}", status_code=303)

    @app.get("/add/{wizard_id}/inspect", response_class=HTMLResponse)
    def wizard_inspect(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        if not state.repo_id:
            # A wizard on the inspect step with no repo (pre-validation
            # wizards could get here) used to bounce silently to /add,
            # making its Resume link a dead loop. Send it back to source
            # with the reason instead.
            state.set_error("no repository selected yet")
            state.advance("source")
            store_wiz.save(state)
            return RedirectResponse(f"/add/{wizard_id}/source", status_code=303)
        # A typo'd repo id, a private repo, or HF being unreachable raises
        # out of the HF client -- render the template's error branch instead
        # of a bare 500 so the user can go back and correct the source.
        try:
            contents = modelctl.get_repo_contents(state.repo_id)
        except Exception as e:
            state.set_error(f"could not fetch {state.repo_id}: {e}")
            store_wiz.save(state)
            contents = None
        return templates.TemplateResponse(request=request, name="wizard_inspect.html",
                                          context=ctx(request, w=state, contents=contents))

    def _submit_download(state):
        """Submit the wizard's acquisition job exactly once.

        Submission lives in POST handlers; the download GET used to submit
        as a side effect, so two tabs (or a browser prefetch) raced the
        job-id guard into two concurrent pulls of the same repo and a
        duplicate `name-2` profile.
        """
        if state.download_job_id:
            return
        if state.source_type == "local_file" and state.local_path:
            state.download_job_id = mutate.submit_import_local(
                runner, state.local_path)
        elif state.repo_id:
            state.download_job_id = mutate.submit_pull(
                runner, state.repo_id,
                quant_label=state.selected_quant or None)

    @app.post("/add/{wizard_id}/inspect")
    async def wizard_inspect_submit(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        form = await request.form()
        state.selected_quant = form.get("quant", "")
        state.advance("download")
        _submit_download(state)
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/download", status_code=303)

    def _refresh_download_outcome(state):
        """Fold the acquisition job's structured result into wizard state.

        `download_complete` used to be set the moment the job was
        *submitted*, so a failed or still-running download read as done and
        the wizard walked on to a profile that did not exist.
        """
        from .wizard import outcome_from_job
        if not state.download_job_id:
            return
        o = outcome_from_job(store, state.download_job_id)
        if not o:
            return
        state.record_outcome(
            "download", ok=o["ok"], job_id=state.download_job_id,
            status=o["status"], warnings=o.get("warnings"),
            error=o.get("error", ""))
        state.download_complete = o["ok"]
        data = o.get("data") or {}
        if o["ok"] and data.get("profile") and not state.profile_name:
            state.profile_name = data["profile"]

    @app.get("/add/{wizard_id}/download", response_class=HTMLResponse)
    def wizard_download(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        is_local = state.source_type == "local_file" and bool(state.local_path)
        # Render-only: submission happens in the POST handlers (inspect
        # submit, source submit, retry, download/start) so reloads and
        # extra tabs can't start duplicate downloads.
        _refresh_download_outcome(state)
        store_wiz.save(state)
        return templates.TemplateResponse(
            request=request, name="wizard_download.html",
            context=ctx(request, w=state, is_local=is_local,
                        outcome=state.outcome("download"),
                        blocked=state.blocking_reason("download")))

    @app.post("/add/{wizard_id}/download")
    def wizard_download_next(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        _refresh_download_outcome(state)
        # Refuse to advance over a running or failed acquisition.
        if not state.can_advance("download"):
            state.set_error(f"cannot continue: {state.blocking_reason('download')}")
            store_wiz.save(state)
            return RedirectResponse(f"/add/{wizard_id}/download", status_code=303)
        state.clear_error()
        state.advance("analyze")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/analyze", status_code=303)

    @app.post("/add/{wizard_id}/delete")
    def wizard_delete(wizard_id: str):
        """Abandon a wizard. Without this, a stuck wizard sat in the
        active list for a day with a Resume button as its only affordance."""
        from .wizard import WizardStore
        WizardStore().delete(wizard_id)
        return RedirectResponse("/add", status_code=303)

    @app.post("/add/{wizard_id}/download/start")
    def wizard_download_start(wizard_id: str):
        """Explicit (re)start for a wizard that has no download job yet --
        the landing point for wizards created before submission moved into
        the POST handlers."""
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        _submit_download(state)
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/download", status_code=303)

    @app.post("/add/{wizard_id}/retry/{step}")
    def wizard_retry(wizard_id: str, step: str):
        """Retry one failed step without restarting the wizard."""
        from .wizard import WizardStore, STEPS
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state or step not in STEPS:
            return RedirectResponse("/add", status_code=303)
        state.clear_outcome(step)
        state.clear_error()
        if step == "download":
            state.download_job_id = ""
            state.download_complete = False
            # Resubmit right away -- the download page no longer submits
            # on GET, so a cleared job id would otherwise strand the
            # retry on a page waiting for a job that never starts.
            _submit_download(state)
        elif step == "test":
            state.test_job_id = ""
        elif step == "register":
            state.registration_complete = False
            state.registration_error = ""
        state.advance(step)
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/{step}", status_code=303)

    @app.get("/add/{wizard_id}/analyze", response_class=HTMLResponse)
    def wizard_analyze(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        # If the pull job completed, load the created profile.
        profile = None
        if state.profile_name:
            try:
                profile = modelctl.load_profile(state.profile_name)
            except Exception:
                pass
        elif state.download_job_id:
            job = store.get(state.download_job_id)
            if job and job.get("status") == "done" and job.get("outcome"):
                # job["result"] is the plain-text log (see append_result_line);
                # the structured return value submit_pull()/submit_import_local()
                # return is JSON-encoded in job["outcome"] (job_fragment.html
                # reads it the same way via the `fromjson` filter). Legacy
                # rows can hold a JSON scalar -- treating one as a dict
                # 500'd this page on Jul 30.
                try:
                    outcome = json.loads(job["outcome"])
                except (ValueError, TypeError):
                    outcome = {}
                if not isinstance(outcome, dict):
                    outcome = {}
                pname = outcome.get("profile")
                if pname:
                    state.profile_name = pname
                    store_wiz.save(state)
                    try:
                        profile = modelctl.load_profile(pname)
                    except Exception:
                        pass
        return templates.TemplateResponse(request=request, name="wizard_analyze.html",
                                          context=ctx(request, w=state, profile=profile))

    @app.post("/add/{wizard_id}/analyze")
    def wizard_analyze_next(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        # The import/pull job may still be running (its result populates
        # profile_name) -- advancing to plans before it lands in a redirect
        # loop back to /add, silently dropping the user out of the wizard.
        # Stay on analyze (which re-checks the job and shows progress)
        # instead of forcing the user to restart the whole wizard.
        if not state.profile_name and state.download_job_id:
            job = store.get(state.download_job_id)
            if not (job and job.get("status") == "done" and job.get("outcome")):
                return RedirectResponse(f"/add/{wizard_id}/analyze", status_code=303)
        state.advance("plans")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/plans", status_code=303)

    @app.get("/add/{wizard_id}/plans", response_class=HTMLResponse)
    def wizard_plans(request: Request, wizard_id: str):
        from .wizard import WizardStore
        import modelctl_plans
        import modelctl_hardware
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        if not state.profile_name:
            return RedirectResponse(f"/add/{wizard_id}/analyze", status_code=303)
        profile = modelctl.load_profile(state.profile_name)
        snap = modelctl_hardware.capture_hardware_snapshot()
        plans = modelctl_plans.compile_launch_plans(profile, snap)
        return templates.TemplateResponse(request=request, name="wizard_plans.html",
                                          context=ctx(request, w=state, plans=plans,
                                                      profile=profile))

    @app.post("/add/{wizard_id}/plans")
    async def wizard_plans_submit(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        form = await request.form()
        action = form.get("action", "next")
        state.selected_plan_id = form.get("plan_id", "")
        if action == "test":
            state.advance("test")
        elif action == "register":
            state.advance("register")
        else:
            state.advance("test")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/{state.step}", status_code=303)

    @app.get("/add/{wizard_id}/test", response_class=HTMLResponse)
    def wizard_test(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state or not state.profile_name:
            return RedirectResponse("/add", status_code=303)
        return templates.TemplateResponse(request=request, name="wizard_test.html",
                                          context=ctx(request, w=state))

    @app.post("/add/{wizard_id}/test")
    def wizard_test_submit(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        # "Run Smoke Test" and "Continue to Registration" both POST here.
        # Submitting unconditionally queued a SECOND real llama-server
        # benchmark when the user clicked continue after a test -- two
        # servers for the same model, racing for VRAM.
        from .wizard import outcome_from_job
        if state.test_job_id:
            outcome = outcome_from_job(store, state.test_job_id)
            state.record_outcome("test", outcome.get("ok", False),
                                 job_id=state.test_job_id,
                                 status=outcome.get("status", ""),
                                 error=outcome.get("error", ""))
            if outcome.get("status") in ("queued", "running"):
                # Still running: stay put rather than advancing over it.
                store_wiz.save(state)
                return RedirectResponse(f"/add/{wizard_id}/test", status_code=303)
            # Fold the measured numbers into wizard state. The test job's
            # outcome carries them (plan test: the tune run dict; smoke
            # test: tok_per_s), but nothing copied them over, so the done
            # page reported every tested registration as "not measured".
            if outcome.get("ok"):
                data = outcome.get("data") or {}
                measured = {k: data[k] for k in
                            ("generation_tps", "prompt_tps",
                             "load_seconds", "cache_state") if data.get(k)}
                if not measured and data.get("tok_per_s"):
                    measured = {"generation_tps": data["tok_per_s"]}
                if measured:
                    state.measured = measured
                    if state.selected_plan_id:
                        state.test_observations[state.selected_plan_id] = measured
            state.advance("register")
            store_wiz.save(state)
            return RedirectResponse(f"/add/{wizard_id}/register", status_code=303)

        # Test the specific plan the user picked on the previous step (real
        # launch + measured throughput), not just a generic smoke test of
        # whatever the profile's default config happened to be.
        if state.selected_plan_id:
            job_id = mutate.submit_plan_test(runner, state.profile_name, state.selected_plan_id)
        else:
            job_id = mutate.submit_smoke_test(runner, state.profile_name)
        state.test_job_id = job_id
        state.clear_outcome("test")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/test", status_code=303)

    @app.get("/add/{wizard_id}/register", response_class=HTMLResponse)
    def wizard_register(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state or not state.profile_name:
            return RedirectResponse("/add", status_code=303)
        profile = None
        try:
            profile = modelctl.load_profile(state.profile_name)
        except Exception:
            pass
        return templates.TemplateResponse(request=request, name="wizard_register.html",
                                          context=ctx(request, w=state, profile=profile))

    @app.post("/add/{wizard_id}/register")
    def wizard_register_submit(request: Request, wizard_id: str):
        from .wizard import WizardStore
        from modelctl_services import plan_service
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        if not state.profile_name:
            state.set_error("no profile to register")
            store_wiz.save(state)
            return RedirectResponse(f"/add/{wizard_id}/register", status_code=303)

        # The recorded outcome alone is not enough: submitting a test
        # clears it, so navigating straight to this URL while the test's
        # llama-server is still up slipped past the gate and raced it for
        # VRAM. Consult the live job store first.
        if state.test_job_id and not state.registration_complete:
            live = store.get(state.test_job_id)
            if live and live.get("status") in ("queued", "running"):
                state.set_error("the plan test is still running -- "
                                "wait for it to finish before registering")
                store_wiz.save(state)
                return RedirectResponse(f"/add/{wizard_id}/register",
                                        status_code=303)

        # A plan whose test failed (or never ran) must not be registered
        # and reported as verified on the done page. The wizard already
        # records structured step outcomes; this is the gate that reads
        # them. `register_untested` is the explicit opt-out.
        blocking = state.blocking_reason("test")
        if blocking and not state.registration_complete:
            import urllib.parse as _urlparse
            qs = _urlparse.parse_qs(request.url.query or "")
            if "1" not in qs.get("register_untested", []):
                state.set_error(
                    f"not registering: {blocking}. Re-run the test, or "
                    "confirm registering it untested.")
                store_wiz.save(state)
                return RedirectResponse(f"/add/{wizard_id}/register",
                                        status_code=303)

        # Apply the plan the user compared/tested so the profile actually
        # launches with it, not whatever config import_local() or the pull
        # wizard happened to default to.
        if state.selected_plan_id:
            result = plan_service.apply_plan(state.profile_name,
                                             state.selected_plan_id)
            if not result.ok:
                # Registration failed, but the profile itself is still
                # saved and valid -- say so, and stay on the register step
                # with a retry rather than reporting success on the done
                # page. This previously set
                # registration_complete=True regardless and walked on.
                state.registration_error = (
                    result.messages[0] if result.messages
                    else "failed to apply the selected plan")
                state.record_outcome("register", ok=False,
                                     messages=result.messages,
                                     error=state.registration_error)
                state.set_error(state.registration_error)
                state.registration_complete = False
                store_wiz.save(state)
                return RedirectResponse(f"/add/{wizard_id}/register",
                                        status_code=303)

        # Record what was actually registered, so the done page can show
        # the command identity rather than just a name.
        try:
            profile = modelctl.load_profile(state.profile_name)
            cmd, _ok, _msgs = modelctl.canonical_launch_command(profile)
            state.command_fingerprint = cmd.command_fingerprint
        except Exception:
            state.command_fingerprint = ""

        # Warm-load through llama-swap to verify.
        job_id = mutate.submit_load(runner, state.profile_name)
        state.registration_complete = True
        state.registration_error = ""
        state.record_outcome("register", ok=True, job_id=job_id,
                             messages=[f"registered '{state.profile_name}'"])
        base = modelctl.LLAMA_SWAP_BASE_URL.rstrip("/")
        state.endpoint = f"{base}/chat/completions"
        state.clear_error()
        state.advance("done")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/done", status_code=303)

    @app.get("/add/{wizard_id}/done", response_class=HTMLResponse)
    def wizard_done(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        # The done page answers "what did I actually get?": endpoint,
        # selected plan, command fingerprint, and the measured result the
        # choice was based on.
        plan_label = ""
        try:
            if state.selected_plan_id and state.profile_name:
                import modelctl_plans
                profile = modelctl.load_profile(state.profile_name)
                for pl in modelctl_plans.compile_launch_plans(profile):
                    if pl.id == state.selected_plan_id:
                        plan_label = pl.label
                        break
        except Exception:
            pass
        measured = state.measured or (
            state.test_observations.get(state.selected_plan_id, {})
            if state.selected_plan_id else {})
        return templates.TemplateResponse(request=request, name="wizard_done.html",
                                          context=ctx(request, w=state,
                                                      plan_label=plan_label,
                                                      measured=measured))

    # ---- jobs -----------------------------------------------------------
    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_list(request: Request):
        return templates.TemplateResponse(request=request, name="jobs.html", context=ctx(
            request, jobs=store.list()))

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail(request: Request, job_id: str, back: str = "/"):
        job = store.get(job_id)
        return templates.TemplateResponse(request=request, name="job_detail.html", context=ctx(
            request, job=job, back=_safe_back(back)))

    @app.get("/jobs/{job_id}/fragment", response_class=HTMLResponse)
    def job_fragment(request: Request, job_id: str, back: str = "/"):
        job = store.get(job_id)
        return templates.TemplateResponse(request=request, name="job_fragment.html", context=ctx(
            request, job=job, back=_safe_back(back)))

    @app.post("/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        runner.cancel(job_id)
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.get("/events/jobs/{job_id}")
    async def sse_job(request: Request, job_id: str):
        from .jobs import sse_job_stream
        return StreamingResponse(
            sse_job_stream(store, job_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- test + bench ---------------------------------------------------
    @app.post("/profiles/{name}/test")
    def profile_test(name: str):
        job_id = mutate.submit_smoke_test(runner, name)
        return RedirectResponse(f"/jobs/{job_id}?back=/", status_code=303)

    @app.post("/profiles/{name}/bench")
    async def profile_bench(request: Request, name: str):
        # Optional overrides: SSD-mmap models generate well under 1 tok/s,
        # so the default 256x3 takes the better part of an hour there --
        # a smaller budget measures the same steady state.
        form = await request.form()
        try:
            max_tokens = max(1, min(4096, int(form.get("max_tokens", 256))))
        except (TypeError, ValueError):
            max_tokens = 256
        try:
            runs = max(1, min(10, int(form.get("runs", 3))))
        except (TypeError, ValueError):
            runs = 3
        job_id = mutate.submit_bench(runner, name, max_tokens=max_tokens,
                                     runs=runs)
        return RedirectResponse(f"/jobs/{job_id}?back=/", status_code=303)

    # ---- runtime ---------------------------------------------------------
    @app.get("/runtime", response_class=HTMLResponse)
    def runtime_page(request: Request):
        state = _runtime_state()
        cache_stats = {}
        for mid, m in state.items():
            profile = None
            try:
                profile = modelctl.load_profile(mid)
            except Exception:
                # llama-swap may run models with no modelctl profile.
                pass
            if not profile:
                continue
            mc = profile.get("moe_cache", {})
            if mc.get("mode", "off") == "off":
                continue
            port = m.get("port")
            if not port:
                continue
            stats = _scrape_moe_cache_metrics(port)
            if stats:
                cache_stats[mid] = stats
        # Which runtime models have a modelctl profile: the template only
        # links "configure" for these -- llama-swap can run models modelctl
        # doesn't manage, and linking those 404'd.
        profile_names = {p.stem for p in modelctl.PROFILES_DIR.glob("*.json")}
        return templates.TemplateResponse(request=request, name="runtime.html", context=ctx(
            request, runtime=state, cache_stats=cache_stats,
            managed_names=profile_names,
            reset_ok=request.query_params.get("reset_ok", ""),
            reset_error=request.query_params.get("reset_error", "")))

    @app.get("/api/runtime")
    def api_runtime():
        return _runtime_state()

    @app.get("/api/runtime/models/{name:path}")
    def api_runtime_model(name: str):
        rt = _swap_client().model_state(name)
        return rt

    @app.get("/runtime/logs/{name:path}", response_class=HTMLResponse)
    def runtime_logs(request: Request, name: str):
        client = _swap_client()
        try:
            logs_data = client.logs(model_id=name)
            log_text = logs_data if isinstance(logs_data, str) else json.dumps(logs_data, indent=2)
        except ModelctlSwapError as e:
            log_text = f"error: {e.message}"
        # The template reads the per-model runtime_state() shape (pid,
        # port, registered, state_class); model_state() returns only
        # {state, worker}, so the load button and pid/port never rendered.
        rt = client.runtime_state().get(name) or {
            "model_id": name, "state": "unavailable", "registered": False,
            "running": False, "pid": None, "port": None, "started": None,
            "state_class": "",
        }
        return templates.TemplateResponse(request=request, name="runtime_logs.html", context=ctx(
            request, name=name, log_text=log_text, rt=rt))

    @app.get("/api/runtime/logs/{name:path}", response_class=PlainTextResponse)
    def api_runtime_logs(name: str):
        client = _swap_client()
        try:
            logs_data = client.logs(model_id=name)
            return logs_data if isinstance(logs_data, str) else json.dumps(logs_data, indent=2)
        except ModelctlSwapError as e:
            return JSONResponse({"error": e.code, "message": e.message}, status_code=502)

    @app.post("/models/{name:path}/load")
    def model_load(name: str):
        job_id = mutate.submit_load(runner, name)
        return RedirectResponse(f"/jobs/{job_id}?back=/runtime", status_code=303)

    @app.post("/models/{name:path}/unload")
    def model_unload(name: str):
        job_id = mutate.submit_unload(runner, name)
        return RedirectResponse(f"/jobs/{job_id}?back=/runtime", status_code=303)

    @app.post("/models/{name:path}/restart")
    def model_restart(name: str):
        job_id = mutate.submit_restart(runner, name)
        return RedirectResponse(f"/jobs/{job_id}?back=/runtime", status_code=303)

    @app.post("/models/{name:path}/cache/reset")
    def model_cache_reset(name: str):
        rt = _runtime_state().get(name, {})
        port = rt.get("port")
        # Report the outcome instead of always 303-ing: a failed or
        # skipped reset was indistinguishable from a successful one, so
        # a measurement taken right after could be on a cache that was
        # never actually cleared.
        if not port:
            return RedirectResponse(
                "/runtime?reset_error=" + quote(f"{name} is not running"),
                status_code=303)
        try:
            import urllib.request
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/cache/reset", method="POST")
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            return RedirectResponse(
                "/runtime?reset_error=" + quote(f"{name}: {e}"),
                status_code=303)
        return RedirectResponse("/runtime?reset_ok=" + quote(name),
                                status_code=303)

    @app.post("/runtime/unload-all")
    def runtime_unload_all():
        job_id = mutate.submit_unload_all(runner)
        return RedirectResponse(f"/jobs/{job_id}?back=/runtime", status_code=303)

    # ---- hardware --------------------------------------------------------
    @app.get("/hardware", response_class=HTMLResponse)
    def hardware_page(request: Request, saved: str = ""):
        import modelctl_hardware
        snap = modelctl_hardware.capture_hardware_snapshot()
        settings = modelctl_hardware.load_settings()
        return templates.TemplateResponse(request=request, name="hardware.html", context=ctx(
            request, snap=snap, settings=settings, saved=saved))

    @app.post("/hardware/save")
    async def hardware_save(request: Request):
        import modelctl_hardware
        import modelctl_fsutil
        form = await request.form()
        # The whole read-modify-write holds the state lock: the storage
        # calibration job writes the same hardware.json from the mutation
        # lane, and an unlocked save here could drop its measurement (or
        # ours) depending on who wrote last.
        with modelctl_fsutil.state_lock():
            settings = modelctl_hardware.load_settings()
            dev_settings = settings.setdefault("devices", {})
            for key, val in form.items():
                if key.startswith("reserve_"):
                    dev = key.removeprefix("reserve_")
                    ds = dev_settings.setdefault(dev, {})
                    try:
                        ds["reserve_bytes"] = int(val) * 1073741824
                    except ValueError:
                        pass
                elif key.startswith("role_"):
                    dev = key.removeprefix("role_")
                    ds = dev_settings.setdefault(dev, {})
                    ds["role"] = val
                elif key.startswith("enabled_"):
                    pass  # handled below -- unchecked checkboxes are absent
                elif key.startswith("bw_"):
                    dev = key.removeprefix("bw_")
                    ds = dev_settings.setdefault(dev, {})
                    try:
                        ds["memory_bandwidth_gbs_override"] = float(val)
                    except ValueError:
                        pass
            snap = modelctl_hardware.capture_hardware_snapshot(settings)
            for g in snap.gpus:
                ds = dev_settings.setdefault(g.device, {})
                ds["enabled"] = f"enabled_{g.device}" in form
            if "ram_reserve" in form:
                try:
                    settings.setdefault("ram", {})["reserve_bytes"] = int(form["ram_reserve"]) * 1073741824
                except ValueError:
                    pass
            modelctl_hardware.save_settings(settings)
        return RedirectResponse("/hardware?saved=1", status_code=303)

    @app.post("/hardware/calibrate")
    def hardware_calibrate():
        job_id = mutate.submit_calibrate_storage(runner)
        return RedirectResponse(f"/jobs/{job_id}?back=/hardware", status_code=303)

    @app.get("/api/hardware")
    def api_hardware():
        import modelctl_hardware
        snap = modelctl_hardware.capture_hardware_snapshot()
        return {
            "fingerprint": snap.fingerprint,
            "gpus": [asdict(g) for g in snap.gpus],
            "ram_total_bytes": snap.ram_total_bytes,
            "ram_available_bytes": snap.ram_available_bytes,
            "ram_reserve_bytes": snap.ram_reserve_bytes,
            "storage": [asdict(s) for s in snap.storage],
            "backend_fingerprints": snap.backend_fingerprints,
        }

    @app.get("/api/hardware/settings")
    def api_hardware_settings():
        import modelctl_hardware
        return modelctl_hardware.load_settings()

    # ---- settings --------------------------------------------------------
    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, saved: str = "", rejected: str = ""):
        import modelctl_diagnostics
        import modelctl_hardware
        defaults = modelctl.load_defaults()
        try:
            storage = modelctl_hardware.capture_hardware_snapshot().storage
        except Exception:
            storage = ()
        return templates.TemplateResponse(request=request, name="settings.html", context=ctx(
            request, defaults=defaults, saved=saved, rejected=rejected,
            models_dir=str(modelctl.DEFAULT_MODELS_DIR),
            profiles_dir=str(modelctl.PROFILES_DIR),
            llama_server=modelctl.LLAMA_SERVER_BIN,
            llama_swap_config=str(modelctl.LLAMA_SWAP_CONFIG_PATH),
            llama_swap_service=modelctl.LLAMA_SWAP_SERVICE_NAME,
            llama_swap_base_url=modelctl.LLAMA_SWAP_BASE_URL,
            manifest=modelctl_diagnostics.manifest_status(),
            capabilities=modelctl_diagnostics.capability_report(),
            environment=modelctl_diagnostics.environment_report(),
            storage=storage,
        ))

    @app.get("/api/diagnostics")
    def api_diagnostics():
        import modelctl_diagnostics
        status = modelctl_diagnostics.manifest_status()
        return {
            "manifest": {
                "path": status.path, "error": status.error,
                "present": status.present, "ok": status.ok,
                "content": status.manifest,
                "modelctl_commit": status.modelctl_commit,
                "submodule_pinned": status.submodule_pinned,
                "submodule_checked_out": status.submodule_checked_out,
                "working_tree_dirty": status.working_tree_dirty,
                "mismatches": list(status.mismatches),
                "notes": list(status.notes),
            },
            "capabilities": modelctl_diagnostics.capability_report(),
            "environment": modelctl_diagnostics.environment_report(),
        }

    @app.get("/settings/support-bundle")
    def support_bundle():
        import modelctl_diagnostics
        data = modelctl_diagnostics.support_bundle_bytes()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return Response(
            content=data, media_type="application/zip",
            headers={"content-disposition":
                     f'attachment; filename="modelctl-support-{stamp}.zip"'})

    @app.post("/settings/save")
    async def settings_save(request: Request):
        # Validation and coercion live in settings_service. The
        # route used to swallow a bad integer with `except ValueError:
        # pass`, so a mistyped context length silently kept the old value
        # with no indication anything had been rejected.
        from modelctl_services import settings_service
        form = await request.form()
        result = settings_service.update_defaults(
            {k: str(v) for k, v in form.items()})
        if result.warnings:
            from urllib.parse import quote
            return RedirectResponse(
                f"/settings?saved=1&rejected={quote('; '.join(result.warnings))}",
                status_code=303)
        return RedirectResponse("/settings?saved=1", status_code=303)

    # ---- launch plans ----------------------------------------------------
    @app.get("/profiles/{name}/plans", response_class=HTMLResponse)
    def plans_page(request: Request, name: str):
        import modelctl_plans
        import modelctl_hardware
        import modelctl_runtime
        import modelctl_evidence
        import modelctl_launch
        p = modelctl.load_profile(name)
        snap = modelctl_hardware.capture_hardware_snapshot()
        plans = modelctl_plans.compile_launch_plans(p, snap)
        rdb = modelctl_runtime.RuntimeDB()
        failures = rdb.failures_for_profile(name)

        # Resolving the backend here is what lets a plan card name the build
        # its claims and cache eligibility belong to. Without it the page
        # shows the same estimate for a stock binary and the fork, which is
        # exactly the confusion the evidence module exists to remove.
        try:
            backend = modelctl_launch.resolve_backend(p)
        except Exception:
            backend = None

        # Resolved before the observations, so staleness can be judged on the
        # effective launch environment and the capabilities the binary
        # actually reported -- not only on hardware and backend identity.
        # When resolution fails the identity degrades to exactly
        # the two checks this page did before.
        observations = rdb.observations_for_profile(
            name, identity=modelctl_runtime.ObservationIdentity.current(
                snapshot=snap, backend=backend,
                profile_name=p.get("backend", "llama-cpp")))

        # `explain` collects the ranker's own words for every plan the
        # experimental guardrail demoted, so the trace below reports what
        # actually happened rather than a rephrasing that could drift.
        explain = {}
        policy = None
        ranked = []
        try:
            policy = modelctl_plans.policy_for_profile(p)
            ranked = modelctl_plans.rank_plans(
                plans, policy, observations, failures, explain=explain)
            ranked_ids = [pl.id for pl, _score in ranked]
        except Exception:
            ranked_ids = []

        evidence = modelctl_evidence.build_plan_evidence(
            p, plans, observations, failures, backend=backend,
            ranked_ids=ranked_ids)
        trace = modelctl_evidence.build_decision_trace(
            evidence, ranked=ranked, policy=policy, observations=observations,
            explain=explain, backend=backend,
            hardware_fingerprint=snap.fingerprint)
        return templates.TemplateResponse(request=request, name="plans.html", context=ctx(
            request, p=p, plans=plans, profile_name=name,
            observations=observations,
            groups=modelctl_evidence.group_evidence(evidence),
            evidence=evidence, backend=backend, trace=trace))

    @app.get("/api/profiles/{name}/plans")
    def api_plans(name: str):
        import modelctl_plans
        import modelctl_hardware
        p = modelctl.load_profile(name)
        snap = modelctl_hardware.capture_hardware_snapshot()
        plans = modelctl_plans.compile_launch_plans(p, snap)
        return [{"id": pl.id, "label": pl.label, "source": pl.source,
                 "argv": list(pl.argv), "claim": asdict(pl.claim),
                 "estimated": pl.estimated, "warnings": list(pl.warnings),
                 "decision_data": pl.decision_data} for pl in plans]

    @app.get("/api/profiles/{name}/plans/{plan_id}")
    def api_plan_detail(name: str, plan_id: str):
        import modelctl_plans
        import modelctl_hardware
        p = modelctl.load_profile(name)
        snap = modelctl_hardware.capture_hardware_snapshot()
        plans = modelctl_plans.compile_launch_plans(p, snap)
        for pl in plans:
            if pl.id == plan_id:
                return {"id": pl.id, "label": pl.label, "source": pl.source,
                        "argv": list(pl.argv), "env": pl.env,
                        "claim": asdict(pl.claim), "estimated": pl.estimated,
                        "warnings": list(pl.warnings),
                        "decision_data": pl.decision_data}
        return JSONResponse({"error": "plan not found"}, status_code=404)

    # ---- runtime policy --------------------------------------------------
    @app.post("/profiles/{name}/runtime-policy")
    async def runtime_policy_save(request: Request, name: str):
        form = await request.form()
        mode = form.get("mode", "fixed")
        profile = modelctl.load_profile(name)
        if mode == "managed":
            import modelctl_backends
            try:
                modelctl_backends.get_backend(profile.get("backend", "llama-cpp"))
            except modelctl_backends.BackendError as e:
                return templates.TemplateResponse(
                    request=request, name="error.html",
                    context=ctx(request, message=str(e)), status_code=400)
        if mode != "managed":
            runtime = None
        else:
            runtime = {
                "mode": "managed",
                "objective": form.get("objective", "balanced"),
                "pinned_plan_id": form.get("pinned_plan_id") or None,
                "allow_fallback": "allow_fallback" in form,
                "allow_untested": "allow_untested" in form,
                "minimum_context": int(form["minimum_context"]) if form.get("minimum_context") else None,
                "maximum_cpu_bytes": (int(form["maximum_cpu_gib"]) * (1 << 30)
                                      if form.get("maximum_cpu_gib") else None),
                "maximum_storage_tier": int(form.get("maximum_storage_tier", 3)),
                "disabled_plan_ids": json.loads(form.get("disabled_plan_ids", "[]")),
            }
        job_id = mutate.submit_runtime_policy(runner, name, runtime)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}/plans", status_code=303)

    @app.post("/profiles/{name}/plans/{plan_id}/select")
    def plan_select(name: str, plan_id: str):
        job_id = mutate.submit_plan_select(runner, name, plan_id)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}/plans", status_code=303)

    @app.post("/profiles/{name}/plans/{plan_id}/disable")
    def plan_disable(name: str, plan_id: str):
        job_id = mutate.submit_plan_select(runner, name, plan_id, disable=True)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}/plans", status_code=303)

    @app.post("/profiles/{name}/plans/{plan_id}/enable")
    def plan_enable(name: str, plan_id: str):
        from modelctl_services import plan_service

        def fn(ctx):
            result = plan_service.enable_plan(name, plan_id)
            if not result.ok:
                raise RuntimeError(result.messages[0] if result.messages
                                  else f"enable failed for plan {plan_id}")
            ctx.log(f"re-enabled plan {plan_id}")
            return {"enabled": plan_id}
        job_id = runner.submit("mutation", f"enable plan {plan_id}", fn,
                               payload={"name": name, "plan_id": plan_id})
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}/plans", status_code=303)

    @app.get("/api/profiles/{name}/runtime-policy")
    def api_runtime_policy(name: str):
        p = modelctl.load_profile(name)
        return p.get("runtime") or {"mode": "fixed"}

    # ---- plan testing + tuning --------------------------------------------
    @app.post("/profiles/{name}/plans/{plan_id}/test")
    def plan_test(name: str, plan_id: str):
        job_id = mutate.submit_plan_test(runner, name, plan_id)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}/plans", status_code=303)

    @app.post("/profiles/{name}/tune")
    async def profile_tune(request: Request, name: str):
        form = await request.form()
        objective = form.get("objective", "balanced")
        ids = [v for k, v in form.multi_items() if k == "plan_id"]
        job_id = mutate.submit_autotune(runner, name, objective=objective,
                                        candidate_ids=ids or None)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}/plans", status_code=303)

    @app.get("/profiles/{name}/history", response_class=HTMLResponse)
    def profile_history(request: Request, name: str):
        import modelctl_runtime
        import modelctl_tune
        runs = modelctl_runtime.RuntimeDB().plan_runs_for(name)
        # The page has to answer "what limited this run?", which
        # is a judgement over several counters. Making it here keeps the
        # template free of the reasoning and gives one definition of the
        # answer.
        rows = []
        for r in runs:
            d = dict(r)
            label, why = modelctl_tune.classify_bottleneck(d)
            d["bottleneck"] = label
            d["bottleneck_why"] = why
            rows.append(d)
        return templates.TemplateResponse(request=request, name="history.html", context=ctx(
            request, name=name, runs=rows))

    @app.get("/api/profiles/{name}/history")
    def api_history(name: str):
        import modelctl_runtime
        return modelctl_runtime.RuntimeDB().plan_runs_for(name)

    # ---- managed routing matrix -------------------------------------------
    @app.get("/runtime/routing", response_class=HTMLResponse)
    def routing_page(request: Request):
        import modelctl_matrix
        try:
            config = modelctl.yaml.safe_load(
                modelctl.LLAMA_SWAP_CONFIG_PATH.read_text()) or {}
        except (OSError, modelctl.yaml.YAMLError):
            # A missing or unparsable router config is a state to show,
            # not a 500.
            config = {}
        generated = modelctl_matrix.generate_matrix()
        merged = modelctl_matrix.merge_matrix(config.get("matrix"), generated)
        preview = modelctl.yaml.safe_dump({"matrix": merged}, sort_keys=False)
        existing_txt = modelctl.yaml.safe_dump({"matrix": config.get("matrix")}, sort_keys=False)
        return templates.TemplateResponse(request=request, name="routing.html", context=ctx(
            request, generated=generated, preview=preview, existing=existing_txt))

    @app.post("/runtime/routing/apply")
    def routing_apply():
        job_id = mutate.submit_matrix_apply(runner)
        return RedirectResponse(f"/jobs/{job_id}?back=/runtime/routing", status_code=303)

    @app.post("/runtime/routing/rollback")
    def routing_rollback():
        def fn(ctx):
            import glob, shutil
            backups = sorted(glob.glob(str(modelctl.LLAMA_SWAP_CONFIG_PATH) + ".bak-matrix-*"))
            if not backups:
                raise RuntimeError("no matrix backup found")
            shutil.copy2(backups[-1], modelctl.LLAMA_SWAP_CONFIG_PATH)
            ctx.log(f"restored {backups[-1]}")
            import subprocess
            subprocess.run(["systemctl", "--user", "restart", "llama-swap.service"],
                           capture_output=True, timeout=60)
            return {"restored": backups[-1]}
        job_id = runner.submit("mutation", "rollback managed matrix", fn)
        return RedirectResponse(f"/jobs/{job_id}?back=/runtime/routing", status_code=303)

    # ---- reservations + runtime events -----------------------------------
    @app.get("/api/reservations")
    def api_reservations():
        import modelctl_runtime
        rdb = modelctl_runtime.RuntimeDB()
        return rdb.get_reservations()

    @app.get("/api/runtime/events")
    def api_runtime_events(profile: str = None, limit: int = 50):
        import modelctl_runtime
        rdb = modelctl_runtime.RuntimeDB()
        return rdb.get_events(profile_name=profile, limit=limit)

    # ---- JSON API -------------------------------------------------------
    @app.get("/api/profiles")
    def api_profiles():
        return [p for p in profiles()]

    @app.get("/api/profiles/{name}")
    def api_profile(name: str):
        return modelctl.load_profile(name)

    @app.post("/api/profiles/{name}")
    async def api_update(name: str, request: Request):
        updates = await request.json()
        job_id = mutate.submit_edit(runner, name, updates)
        return {"job": job_id}

    @app.get("/api/tiers/{name}")
    def api_tiers(name: str):
        # plan carries "warnings" and the machine-readable "admission"
        # record (requested / assumed / chosen, per-device math); the
        # extras ride alongside so API consumers see the same gate and
        # input provenance the HTML page shows.
        _p, plan, inputs, source, gate = _plan_for(name)
        if plan is None:
            return {"plan": None}
        return {**plan, "planning_inputs": inputs,
                "planning_inputs_source": source, "gate": gate}

    @app.post("/api/tiers/{name}/apply")
    def api_tiers_apply(name: str, accept_tier_change: bool = False):
        return {"job": mutate.submit_tier_apply(
            runner, name, accept_tier_change=accept_tier_change)}

    @app.post("/api/pull/{repo_id:path}")
    def api_pull(repo_id: str, quant: str = ""):
        # Advanced/automation shortcut: a zero-config pull as one job, for
        # scripts that do not want the step-by-step wizard. It goes through
        # the same submit_pull service as everything else -- it owns no
        # validation or registration logic of its own. Browser users go
        # through /add.
        # Dedup: two concurrent pulls of one repo download into the same
        # destination paths and mint a duplicate `name-2` profile.
        for j in store.list():
            if (j.get("type") == "pull"
                    and j.get("status") in ("queued", "running")):
                try:
                    payload = json.loads(j.get("payload") or "{}")
                except ValueError:
                    payload = {}
                if payload.get("repo_id") == repo_id:
                    return {"job": j["id"], "deduplicated": True}
        return {"job": mutate.submit_pull(runner, repo_id, quant_label=quant or None)}

    @app.get("/api/jobs")
    def api_jobs():
        return store.list()

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        job = store.get(job_id)
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        return job

    # ---- v2 console (SPA) ------------------------------------------------
    # The Vite/Preact build under modelctl/console/dist, committed so the
    # running system never needs node. One SSE stream feeds operate and
    # jobs; the typed JSON endpoints are the same shapes, on demand.

    @app.get("/api/v2/events")
    async def api_v2_events():
        return StreamingResponse(
            telemetry.sse_stream(collector, interval=tick_interval,
                                 max_seconds=tick_max_seconds),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/v2/models")
    def api_v2_models():
        try:
            runtime = collector._runtime()
        except Exception:
            runtime = {}
        return collector._model_rows(runtime)

    @app.get("/api/v2/jobs")
    def api_v2_jobs():
        return collector.job_rows()

    @app.post("/api/v2/jobs/{job_id}/cancel")
    def api_v2_job_cancel(job_id: str):
        # Typed cancel for the SPA's optimistic UI: the answer says whether
        # the cancel actually took, so the client can either keep its
        # optimistic state or loudly un-happen it. Never a redirect.
        job = store.get(job_id)
        if not job:
            return JSONResponse({"error": "not found"}, status_code=404)
        if job["status"] not in ("queued", "running"):
            return {"id": job_id, "status": job["status"], "cancelled": False,
                    "reason": f"job already {job['status']}"}
        if not job.get("cancellable", 1):
            return {"id": job_id, "status": job["status"], "cancelled": False,
                    "reason": "this job is not cancellable"}
        runner.cancel(job_id)
        job = store.get(job_id)
        cancelled = job["status"] == "cancelled"
        return {"id": job_id, "status": job["status"], "cancelled": cancelled,
                "reason": None if cancelled else
                f"server kept it {job['status']}"}

    # ---- v2 model hub ----------------------------------------------------
    from . import hub

    @app.get("/api/v2/models/{name}")
    def api_v2_model_detail(name: str):
        p = modelctl.load_profile(name)
        try:
            rt = _runtime_state().get(name)
        except Exception:
            rt = None
        try:
            inventory = modelctl.get_gpu_inventory()
        except Exception:
            inventory = []
        return hub.model_detail(p, runtime_row=rt, inventory=inventory)

    @app.get("/api/v2/models/{name}/plans")
    def api_v2_model_plans(name: str):
        return hub.plan_rows(modelctl.load_profile(name))

    @app.get("/api/v2/models/{name}/history")
    def api_v2_model_history(name: str, limit: int = 50):
        modelctl.load_profile(name)  # 404 for unknown models, not []
        return hub.history_rows(name, limit=max(1, min(200, limit)))

    @app.get("/api/v2/models/{name}/logtail")
    def api_v2_model_logtail(name: str, lines: int = 120):
        p = modelctl.load_profile(name)
        return hub.log_tail(p, swap_client=_swap_client(),
                            lines=max(10, min(1000, lines)))

    def _draft_budgets_from_query(params):
        budgets = {}
        for key, val in params.items():
            if key.startswith("budget_bytes."):
                try:
                    budgets[key.split(".", 1)[1]] = int(val)
                except ValueError:
                    continue
        return budgets or None

    @app.get("/api/v2/models/{name}/admission")
    def api_v2_model_admission(request: Request, name: str):
        # Draft values ride as query params (?ctx=..., ?moe_mode=...,
        # ?budget_bytes.SYCL0=...) so the configure form can show the
        # planner's answer for what the user is ABOUT to save. Nothing
        # here writes.
        params = request.query_params
        ctx_val = None
        if params.get("ctx"):
            try:
                ctx_val = int(params["ctx"])
            except ValueError:
                return JSONResponse({"error": "ctx must be an integer"},
                                    status_code=422)
        return hub.admission_preview(
            name, ctx=ctx_val,
            budgets_bytes=_draft_budgets_from_query(params),
            moe_mode=params.get("moe_mode") or None)

    @app.post("/api/v2/models/{name}/config")
    async def api_v2_model_config_save(request: Request, name: str):
        """The typed configure form's save: gate first, then jobs.

        A structural change (placement, cache budgets) answers 409 with
        the gate until the client re-sends accept_structural=true -- the
        explicit confirm the spec requires. Field updates and cache
        budgets ride the same mutation lane the old console used."""
        p = modelctl.load_profile(name)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=422)
        updates = body.get("updates") or {}
        unknown = [k for k in updates
                   if k not in hub.CONFIG_FIELDS + ["enabled"]]
        if unknown:
            return JSONResponse(
                {"error": f"unknown fields: {', '.join(sorted(unknown))}"},
                status_code=422)
        budgets = body.get("budgets_bytes")
        if budgets is not None:
            try:
                budgets = {str(d): int(v) for d, v in budgets.items()}
            except (AttributeError, TypeError, ValueError):
                return JSONResponse({"error": "budgets_bytes must map device "
                                     "-> integer bytes"}, status_code=422)
        moe_mode = body.get("moe_mode")
        gate = hub.classify_config_save(p, updates, budgets_bytes=budgets,
                                        moe_mode=moe_mode)
        if gate["requires_accept"] and not body.get("accept_structural"):
            return JSONResponse({"error": "structural change requires confirm",
                                 "gate": gate}, status_code=409)
        jobs = {"config": None, "moe_cache": None}
        config_updates = {k: str(v) if not isinstance(v, bool) else v
                          for k, v in updates.items()}
        if config_updates:
            jobs["config"] = mutate.submit_edit(runner, name, config_updates)
        if budgets is not None or moe_mode is not None:
            mc = json.loads(json.dumps(p.get("moe_cache") or {}))
            mc.setdefault("gpu", {}).setdefault("budgets_bytes", {})
            if budgets is not None:
                mc["gpu"]["budgets_bytes"] = {d: v for d, v in budgets.items()
                                              if v > 0}
            if moe_mode is not None:
                mc["mode"] = str(moe_mode)
            jobs["moe_cache"] = mutate.submit_moe_cache(runner, name, mc)
        return {"jobs": jobs, "gate": gate}

    # ---- v2 add wizard ---------------------------------------------------

    def _wizard_or_404(wizard_id):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        return store_wiz, state

    async def _json_body(request):
        """(body, None) or (None, 422 response) -- a malformed JSON body
        is the caller's mistake, not a traceback in the journal."""
        try:
            body = await request.json()
        except Exception:
            return None, JSONResponse({"error": "invalid JSON body"},
                                      status_code=422)
        if not isinstance(body, dict):
            return None, JSONResponse({"error": "body must be a JSON object"},
                                      status_code=422)
        return body, None

    def _wizard_json(state):
        return hub.wizard_detail(state, store)

    def _resolve_wizard_profile(state, store_wiz):
        """Fold a finished download job's outcome into profile_name, the
        same way the old analyze page does, so the SPA sees the profile as
        soon as the job lands."""
        if state.profile_name or not state.download_job_id:
            return
        job = store.get(state.download_job_id)
        if job and job.get("status") == "done" and job.get("outcome"):
            try:
                outcome = json.loads(job["outcome"])
            except (ValueError, TypeError):
                outcome = {}
            if isinstance(outcome, dict) and outcome.get("profile"):
                state.profile_name = outcome["profile"]
                store_wiz.save(state)

    @app.get("/api/v2/wizards")
    def api_v2_wizards():
        from .wizard import WizardStore
        return [hub.wizard_summary(s) for s in WizardStore().list_active()]

    @app.post("/api/v2/wizards")
    def api_v2_wizard_create():
        from .wizard import WizardState, WizardStore
        state = WizardState()
        WizardStore().save(state)
        return _wizard_json(state)

    @app.get("/api/v2/hf/search")
    def api_v2_hf_search(q: str = ""):
        if not q:
            return {"results": [], "error": ""}
        try:
            return {"results": modelctl.search_models(q), "error": ""}
        except Exception as e:
            return {"results": [], "error": f"search failed: {e}"}

    @app.get("/api/v2/wizard/{wizard_id}")
    def api_v2_wizard_get(wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        _resolve_wizard_profile(state, store_wiz)
        # Refresh only while the download outcome can still change: an
        # unconditional refresh+save on every read races concurrent POST
        # advances and needlessly bumps updated_at (record_outcome stamps
        # a fresh `at` even when nothing else moved).
        o = state.outcome("download")
        if state.download_job_id and (
                not o or o.get("status") in ("", "queued", "running")):
            before = json.dumps(state.to_dict(), sort_keys=True)
            _refresh_download_outcome(state)
            if json.dumps(state.to_dict(), sort_keys=True) != before:
                store_wiz.save(state)
        return _wizard_json(state)

    @app.post("/api/v2/wizard/{wizard_id}/source")
    async def api_v2_wizard_source(request: Request, wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        body, bad = await _json_body(request)
        if bad:
            return bad
        source_type = body.get("source_type") or ""
        state.source_type = source_type
        state.clear_error()
        if source_type == "hf_repo":
            repo_id = (body.get("repo_id") or "").strip()
            if not repo_id or "/" not in repo_id.strip("/"):
                state.set_error(
                    "enter a Hugging Face repo id like org/model-name"
                    if not repo_id else
                    f"'{repo_id}' is not a repo id -- expected org/model-name")
                store_wiz.save(state)
                return JSONResponse({"error": state.errors[-1]["message"],
                                     "state": _wizard_json(state)},
                                    status_code=422)
            state.repo_id = repo_id
            state.advance("inspect")
        elif source_type == "local_file":
            state.local_path = body.get("local_path") or ""
            from modelctl_services import acquisition_service
            check = acquisition_service.verify_local_gguf(state.local_path)
            state.source_verification = {
                "ok": check.ok,
                "messages": list(check.messages),
                "warnings": list(check.warnings),
                "data": dict(check.data),
            }
            if not check.ok:
                state.set_error(check.messages[0] if check.messages
                                else "the selected file is not usable")
                store_wiz.save(state)
                return JSONResponse({"error": state.errors[-1]["message"],
                                     "state": _wizard_json(state)},
                                    status_code=422)
            state.advance("download")
            _submit_download(state)
        else:
            state.set_error("pick a source: Hugging Face repository or local file")
            store_wiz.save(state)
            return JSONResponse({"error": state.errors[-1]["message"],
                                 "state": _wizard_json(state)},
                                status_code=422)
        store_wiz.save(state)
        return _wizard_json(state)

    @app.get("/api/v2/wizard/{wizard_id}/inspect")
    def api_v2_wizard_inspect(wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        if not state.repo_id:
            return JSONResponse({"error": "no repository selected yet"},
                                status_code=409)
        try:
            contents = modelctl.get_repo_contents(state.repo_id)
            return {"contents": contents, "error": ""}
        except Exception as e:
            return {"contents": None,
                    "error": f"could not fetch {state.repo_id}: {e}"}

    @app.post("/api/v2/wizard/{wizard_id}/inspect")
    async def api_v2_wizard_inspect_submit(request: Request, wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        body, bad = await _json_body(request)
        if bad:
            return bad
        state.selected_quant = body.get("quant") or ""
        state.advance("download")
        _submit_download(state)
        store_wiz.save(state)
        return _wizard_json(state)

    @app.post("/api/v2/wizard/{wizard_id}/download/next")
    def api_v2_wizard_download_next(wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        _refresh_download_outcome(state)
        if not state.can_advance("download"):
            reason = state.blocking_reason("download")
            store_wiz.save(state)
            return JSONResponse({"error": reason,
                                 "state": _wizard_json(state)},
                                status_code=409)
        state.clear_error()
        state.advance("analyze")
        store_wiz.save(state)
        return _wizard_json(state)

    @app.get("/api/v2/wizard/{wizard_id}/analyze")
    def api_v2_wizard_analyze(wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        _resolve_wizard_profile(state, store_wiz)
        profile = None
        if state.profile_name:
            try:
                profile = modelctl.load_profile(state.profile_name)
            except Exception:
                profile = None
        analysis = None
        if profile:
            analysis = hub.analyze_model(profile)
            # Persist what the header said: the register step's ctx max
            # comes from here, not from a client-side copy.
            state.analysis = analysis or {}
            store_wiz.save(state)
        return {"profile": (hub.model_detail(profile) if profile else None),
                "analysis": analysis}

    @app.post("/api/v2/wizard/{wizard_id}/analyze/next")
    def api_v2_wizard_analyze_next(wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        _resolve_wizard_profile(state, store_wiz)
        if not state.profile_name:
            return JSONResponse(
                {"error": "the acquisition job has not produced a profile yet",
                 "state": _wizard_json(state)}, status_code=409)
        state.advance("plans")
        store_wiz.save(state)
        return _wizard_json(state)

    @app.get("/api/v2/wizard/{wizard_id}/plans")
    def api_v2_wizard_plans(wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        if not state.profile_name:
            return JSONResponse({"error": "no profile yet"}, status_code=409)
        profile = modelctl.load_profile(state.profile_name)
        return {"plans": hub.plan_rows(profile),
                "selected_plan_id": state.selected_plan_id}

    @app.post("/api/v2/wizard/{wizard_id}/plans")
    async def api_v2_wizard_plans_submit(request: Request, wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        body, bad = await _json_body(request)
        if bad:
            return bad
        state.selected_plan_id = body.get("plan_id") or ""
        action = body.get("action") or "test"
        state.advance("register" if action == "register" else "test")
        store_wiz.save(state)
        return _wizard_json(state)

    @app.post("/api/v2/wizard/{wizard_id}/test/run")
    def api_v2_wizard_test_run(wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        if not state.profile_name:
            return JSONResponse({"error": "no profile to test"}, status_code=409)
        if state.test_job_id:
            live = store.get(state.test_job_id)
            if live and live.get("status") in ("queued", "running"):
                return JSONResponse(
                    {"error": "a test is already running",
                     "state": _wizard_json(state)}, status_code=409)
        if state.selected_plan_id:
            job_id = mutate.submit_plan_test(runner, state.profile_name,
                                             state.selected_plan_id)
        else:
            job_id = mutate.submit_smoke_test(runner, state.profile_name)
        state.test_job_id = job_id
        state.clear_outcome("test")
        store_wiz.save(state)
        return _wizard_json(state)

    @app.post("/api/v2/wizard/{wizard_id}/test/next")
    def api_v2_wizard_test_next(wizard_id: str):
        from .wizard import outcome_from_job
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        if state.test_job_id:
            outcome = outcome_from_job(store, state.test_job_id)
            if outcome.get("status") in ("queued", "running"):
                return JSONResponse(
                    {"error": "the test is still running",
                     "state": _wizard_json(state)}, status_code=409)
            state.record_outcome("test", outcome.get("ok", False),
                                 job_id=state.test_job_id,
                                 status=outcome.get("status", ""),
                                 error=outcome.get("error", ""))
            if outcome.get("ok"):
                data = outcome.get("data") or {}
                measured = {k: data[k] for k in
                            ("generation_tps", "prompt_tps",
                             "load_seconds", "cache_state") if data.get(k)}
                if not measured and data.get("tok_per_s"):
                    measured = {"generation_tps": data["tok_per_s"]}
                if measured:
                    state.measured = measured
                    if state.selected_plan_id:
                        state.test_observations[state.selected_plan_id] = measured
        state.advance("register")
        store_wiz.save(state)
        return _wizard_json(state)

    @app.get("/api/v2/wizard/{wizard_id}/register")
    def api_v2_wizard_register(request: Request, wizard_id: str):
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        if not state.profile_name:
            return JSONResponse({"error": "no profile to register"},
                                status_code=409)
        profile = modelctl.load_profile(state.profile_name)
        analysis = state.analysis or hub.analyze_model(profile)
        # The budget fields are one input per GPU; without the inventory
        # the register form has no devices to offer budgets for.
        try:
            inventory = modelctl.get_gpu_inventory()
        except Exception:
            inventory = []
        params = request.query_params
        ctx_val = None
        if params.get("ctx"):
            try:
                ctx_val = int(params["ctx"])
            except ValueError:
                ctx_val = None
        try:
            admission = hub.admission_preview(
                state.profile_name, ctx=ctx_val,
                budgets_bytes=_draft_budgets_from_query(params),
                moe_mode=params.get("moe_mode") or None)
        except Exception as e:
            admission = {"plan": None, "planning_inputs": None,
                         "planning_inputs_source": "", "gate": None,
                         "error": str(e)}
        return {"profile": hub.model_detail(profile, inventory=inventory),
                "analysis": analysis,
                "admission": admission,
                "measured": state.measured,
                "selected_plan_id": state.selected_plan_id,
                "test_gate": {"blocking_reason": state.blocking_reason("test"),
                              "outcome": state.outcome("test")}}

    @app.post("/api/v2/wizard/{wizard_id}/register")
    async def api_v2_wizard_register_submit(request: Request, wizard_id: str):
        from modelctl_services import plan_service, profile_service
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state:
            return JSONResponse({"error": "wizard not found"}, status_code=404)
        if not state.profile_name:
            return JSONResponse({"error": "no profile to register"},
                                status_code=409)
        body, bad = await _json_body(request)
        if bad:
            return bad

        # Same two gates as the old register handler: a live test job wins
        # over any recorded outcome, and an untested/failed test needs the
        # explicit opt-out.
        if state.test_job_id and not state.registration_complete:
            live = store.get(state.test_job_id)
            if live and live.get("status") in ("queued", "running"):
                return JSONResponse(
                    {"error": "the plan test is still running -- wait for it "
                     "to finish before registering"}, status_code=409)
        blocking = state.blocking_reason("test")
        if blocking and not state.registration_complete \
                and not body.get("register_untested"):
            return JSONResponse(
                {"error": f"not registering: {blocking}. Re-run the test, or "
                 "confirm registering it untested.",
                 "requires_register_untested": True}, status_code=409)

        # ctx is validated against the header the analyze step read; the
        # form can't outrun the model's trained maximum. A wizard that
        # never rendered the analyze page has no persisted analysis --
        # read the header now rather than skipping the check.
        analysis = state.analysis
        if not analysis:
            try:
                analysis = hub.analyze_model(
                    modelctl.load_profile(state.profile_name)) or {}
            except Exception:
                analysis = {}
        ctx_val = body.get("ctx")
        if ctx_val is not None:
            try:
                ctx_val = int(ctx_val)
            except (TypeError, ValueError):
                return JSONResponse({"error": "ctx must be an integer"},
                                    status_code=422)
            model_max = analysis.get("model_max_ctx")
            if model_max and ctx_val > int(model_max):
                return JSONResponse(
                    {"error": f"context length {ctx_val} is above the model's "
                     f"trained maximum ({model_max}, read from the GGUF "
                     "header at analyze)"}, status_code=422)
            if ctx_val < 512:
                return JSONResponse(
                    {"error": "context length is below the 512-token minimum "
                     "the server accepts"}, status_code=422)

        if state.selected_plan_id:
            result = plan_service.apply_plan(state.profile_name,
                                             state.selected_plan_id)
            if not result.ok:
                state.registration_error = (
                    result.messages[0] if result.messages
                    else "failed to apply the selected plan")
                state.record_outcome("register", ok=False,
                                     messages=result.messages,
                                     error=state.registration_error)
                state.set_error(state.registration_error)
                state.registration_complete = False
                store_wiz.save(state)
                return JSONResponse({"error": state.registration_error,
                                     "state": _wizard_json(state)},
                                    status_code=502)

        # Field overrides on top of the applied plan, through the same
        # service the profile editor uses.
        updates = {}
        if ctx_val is not None:
            updates["ctx"] = str(ctx_val)
        for field in ("flash_attn", "ttl"):
            if body.get(field) is not None:
                updates[field] = str(body[field])
        if updates:
            result = profile_service.update_config(state.profile_name, updates)
            if not result.ok:
                msg = (result.messages[0] if result.messages
                       else "failed to apply register-form overrides")
                return JSONResponse({"error": msg}, status_code=422)

        budgets = body.get("budgets_bytes")
        moe_mode = body.get("moe_mode")
        if budgets is not None or moe_mode is not None:
            profile = modelctl.load_profile(state.profile_name)
            mc = json.loads(json.dumps(profile.get("moe_cache") or {}))
            mc.setdefault("gpu", {}).setdefault("budgets_bytes", {})
            if budgets is not None:
                try:
                    mc["gpu"]["budgets_bytes"] = {
                        str(d): int(v) for d, v in budgets.items()
                        if int(v) > 0}
                except (AttributeError, TypeError, ValueError):
                    return JSONResponse({"error": "budgets_bytes must map "
                                         "device -> integer bytes"},
                                        status_code=422)
            if moe_mode is not None:
                mc["mode"] = str(moe_mode)
            profile["moe_cache"] = mc
            import modelctl_capabilities
            binary = profile.get("binary") or modelctl.LLAMA_SERVER_BIN
            caps = (modelctl_capabilities.probe_backend(binary)
                    if binary else None)
            msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
            errors = [m for lvl, m in msgs if lvl == "error"]
            if errors:
                return JSONResponse(
                    {"error": "moe_cache validation failed: " +
                     "; ".join(errors)}, status_code=422)
            modelctl.save_profile(profile)
            modelctl.generate_artifacts(profile)
            modelctl.sync_all_backends(restart_router=True,
                                       restart_openarc=True)

        try:
            profile = modelctl.load_profile(state.profile_name)
            cmd, _ok, _msgs = modelctl.canonical_launch_command(profile)
            state.command_fingerprint = cmd.command_fingerprint
        except Exception:
            state.command_fingerprint = ""

        job_id = mutate.submit_load(runner, state.profile_name)
        state.registration_complete = True
        state.registration_error = ""
        state.record_outcome("register", ok=True, job_id=job_id,
                             messages=[f"registered '{state.profile_name}'"])
        base = modelctl.LLAMA_SWAP_BASE_URL.rstrip("/")
        state.endpoint = f"{base}/chat/completions"
        state.clear_error()
        state.advance("done")
        store_wiz.save(state)
        return _wizard_json(state)

    @app.post("/api/v2/wizard/{wizard_id}/retry/{step}")
    def api_v2_wizard_retry(wizard_id: str, step: str):
        """Single-shot retry: re-issue the step's own job with identical
        params (the wizard state already holds them); nothing else about
        the wizard moves."""
        from .wizard import STEPS
        store_wiz, state = _wizard_or_404(wizard_id)
        if not state or step not in STEPS:
            return JSONResponse({"error": "wizard or step not found"},
                                status_code=404)
        state.clear_outcome(step)
        state.clear_error()
        if step == "download":
            state.download_job_id = ""
            state.download_complete = False
            _submit_download(state)
        elif step == "test":
            state.test_job_id = ""
        elif step == "register":
            state.registration_complete = False
            state.registration_error = ""
        state.advance(step)
        store_wiz.save(state)
        return _wizard_json(state)

    @app.post("/api/v2/wizard/{wizard_id}/delete")
    def api_v2_wizard_delete(wizard_id: str):
        from .wizard import WizardStore
        WizardStore().delete(wizard_id)
        return {"deleted": wizard_id}

    @app.get("/v2")
    def v2_root_redirect():
        return RedirectResponse("/v2/", status_code=307)

    @app.get("/v2/{path:path}")
    def v2_spa(path: str):
        # Static file when it exists, index.html for every route the SPA
        # owns (history-mode router). resolve() + prefix check keeps ../
        # traversal inside dist.
        if not CONSOLE_DIST.exists():
            return PlainTextResponse(
                "console build missing (modelctl/console/dist)", status_code=503)
        candidate = (CONSOLE_DIST / path).resolve() if path else CONSOLE_DIST
        try:
            inside = candidate.is_relative_to(CONSOLE_DIST.resolve())
        except AttributeError:
            inside = str(candidate).startswith(str(CONSOLE_DIST.resolve()))
        if path and inside and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(CONSOLE_DIST / "index.html")

    return app


# No module-level create_app(): importing this module used to construct a
# JobStore against the live web_jobs.db (flipping the running service's
# jobs to 'interrupted') and spawn six worker threads -- from `modelctl
# web url`, from the test suite, from anything that touched the module.
# Entry points call create_app() explicitly.
