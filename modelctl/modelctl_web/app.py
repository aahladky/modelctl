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
from fastapi.templating import Jinja2Templates

import modelctl
import modelctl_errors

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


def _set_session_cookie(response, token):
    """The one place session-cookie flags are decided.

    Login and token rotation both issue this cookie; when they disagreed
    about httponly/samesite the rotated session was the weaker one.
    """
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, samesite="strict",
        secure=os.environ.get("MODELCTL_WEB_SECURE_COOKIE", "") == "1")


def _safe_next(target):
    """An internal path to return to after login, or the console root.

    `next` is reflected into the form and followed on success; a
    "javascript:..." or "//evil.example" value there would execute in the
    console's origin with the session cookie attached, able to drive any
    mutating API.
    """
    target = str(target or "/v2/")
    if not target.startswith("/") or target.startswith("//"):
        return "/v2/"
    return target


def _fetch_json(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


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
    # Two server-rendered pages survive the phase-3 cutover: login (you
    # cannot be inside the SPA before authenticating) and the last-resort
    # error page. No filters -- the pages that needed them are gone.
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    app = FastAPI(title="modelctl-web", docs_url=None, redoc_url=None)
    app.state.token = token
    app.state.store = store
    app.state.runner = runner

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
        # Minimal context on purpose: this page must not be able to throw
        # while reporting that something else threw.
        return templates.TemplateResponse(
            request=request, name="error.html",
            context={"heading": "something went wrong",
                     "message": f"{type(exc).__name__}: {exc}"},
            status_code=500)

    scratch = scratch_safe_mode()

    @app.middleware("http")
    async def auth(request: Request, call_next):
        # /login stays reachable in scratch mode too: authenticating is
        # how a scratch walk gets in, and it mutates nothing server-side.
        if request.url.path in ("/login", "/healthz"):
            return await call_next(request)
        supplied = (
            request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            or request.cookies.get(COOKIE_NAME, ""))
        # app.state, not the closure variable: token rotation replaces it
        # in-process, and a stale closure would keep honouring the old
        # token until the next restart.
        if not supplied or not secrets.compare_digest(supplied, app.state.token):
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

    # ---- auth pages -----------------------------------------------------
    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz():
        return "ok"

    def _login_page(request, next, error, status_code=200):
        return templates.TemplateResponse(
            request=request, name="login.html", status_code=status_code,
            context={"next": next, "error": error,
                     "token_path": str(TOKEN_PATH)})

    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request, next: str = "/v2/"):
        return _login_page(request, _safe_next(next), "")

    # Global (not per-IP): this is a single-user console; the goal is to
    # blunt unattended guessing, not to referee concurrent users.
    login_failures = {"count": 0, "last": 0.0}

    @app.post("/login")
    def login(request: Request, next: str = Form("/v2/"), token_field: str = Form("")):
        next = _safe_next(next)
        now = time.monotonic()
        if login_failures["count"] >= 5 and now - login_failures["last"] < 30.0:
            return _login_page(request, next,
                               "Too many failed attempts -- wait 30 seconds.",
                               status_code=429)
        if not token_field or not secrets.compare_digest(token_field,
                                                         app.state.token):
            login_failures["count"] += 1
            login_failures["last"] = now
            # Render the form with the error instead of redirecting: a 307
            # redirect used to re-POST the same body in a loop, and the old
            # template had no error slot at all.
            return _login_page(request, next, "Wrong token.", status_code=401)
        login_failures["count"] = 0
        # 303 turns the follow-up into a GET; the default 307 re-POSTed to
        # `next`, which both broke the login flow (405 on /) and let a
        # crafted ?next= re-POST a fresh login into a mutating endpoint.
        resp = RedirectResponse(next, status_code=303)
        _set_session_cookie(resp, app.state.token)
        return resp

    @app.get("/logout")
    def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME)
        return resp

    # ---- dashboard ------------------------------------------------------
    # ---- first run / setup ----------------------------------------------
    @app.get("/api/setup")
    def api_setup():
        import modelctl_setup
        status = modelctl_setup.probe_setup(probe_backend=True)
        return {"ready": status.ready, "first_run": status.first_run,
                "checks": [asdict(c) for c in status.checks]}

    # ---- wizard + planner helpers ----------------------------------------
    # These three lived among the old server-rendered routes and are shared
    # with the /api/v2 surface; the phase-3 demolition took the routes and
    # would have taken the helpers with them.

    def _plan_for(name):
        import modelctl_tiers
        p = modelctl.load_profile(name)
        # Stored-first inputs and the same plan path submit_tier_apply
        # uses, so the preview is exactly what apply would compute.
        plan, inputs, source = modelctl.plan_tiers_for_profile(p)
        gate = (modelctl_tiers.tier_change_gate(p, plan)
                if plan is not None else None)
        return p, plan, inputs, source, gate

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

    # ---- old-console redirects (phase-3 cutover) ------------------------
    # The server-rendered console is gone; these keep every URL it ever
    # published pointing at the /v2 surface that replaced it. 301, not 302:
    # the move is permanent and bookmarks/history should be rewritten.
    #
    # GET only. A 301 turns a POST into a GET, and the /api/v2 replacements
    # take JSON where the old form handlers took form encoding, so an old
    # POST cannot be redirected into a working call -- those routes are
    # simply gone and answer 405 (method not allowed on the surviving GET)
    # or 404. Registration order matters: Starlette matches in order, so
    # the specific paths come before the {name}-shaped ones.

    def _moved(target):
        return RedirectResponse(target, status_code=301)

    def _wizard_for_repo(repo_id: str):
        from .wizard import WizardState, WizardStore
        state = WizardState()
        state.source_type = "hf_repo"
        state.repo_id = repo_id
        state.advance("inspect")
        WizardStore().save(state)
        return state

    @app.get("/")
    def moved_root():
        return _moved("/v2/")

    # setup folded into settings: readiness is diagnostics, and the
    # four-domain IA has no separate setup page.
    @app.get("/setup")
    def moved_setup():
        return _moved("/v2/settings")

    @app.get("/settings/support-bundle")
    def moved_support_bundle():
        return _moved("/api/v2/settings/support-bundle")

    @app.get("/settings")
    def moved_settings():
        return _moved("/v2/settings")

    @app.get("/hardware")
    def moved_hardware():
        return _moved("/v2/settings")

    @app.get("/tiers")
    def moved_tiers():
        return _moved("/v2/models")

    @app.get("/runtime/routing")
    def moved_routing():
        # No v2 equivalent: the managed llama-swap routing matrix was a
        # domain the new IA does not have. Settings is where install-level
        # configuration now lives, so that is where the link lands.
        return _moved("/v2/settings")

    @app.get("/runtime/logs/{name:path}")
    def moved_runtime_logs(name: str):
        return _moved(f"/v2/models/{quote(name, safe='')}")

    @app.get("/runtime")
    def moved_runtime():
        return _moved("/v2/")

    @app.get("/jobs")
    def moved_jobs():
        return _moved("/v2/jobs")

    # The SPA has no per-job URL; every old job link lands on the list,
    # where the job is visible in the running/queued/history tables.
    @app.get("/jobs/{rest:path}")
    def moved_job(rest: str):
        return _moved("/v2/jobs")

    @app.get("/events/jobs/{job_id}")
    def moved_job_stream(job_id: str):
        # The per-job SSE stream became the one console-wide tick stream.
        return _moved("/api/v2/events")

    # These two minted a wizard as a side effect and only then redirected
    # into it, so a plain 301 would drop the repo id. Kept as thin shims
    # over the same helper the SPA's create+source pair uses.
    @app.get("/pull/{repo_id:path}")
    def moved_pull_repo(repo_id: str):
        state = _wizard_for_repo(repo_id)
        return RedirectResponse(f"/v2/add/{quote(state.wizard_id, safe='')}",
                                status_code=303)

    @app.get("/add/start/{repo_id:path}")
    def moved_add_start(repo_id: str):
        state = _wizard_for_repo(repo_id)
        return RedirectResponse(f"/v2/add/{quote(state.wizard_id, safe='')}",
                                status_code=303)

    @app.get("/pull")
    def moved_pull(q: str = ""):
        # The old page carried its search term into the wizard; keep the
        # hand-off rather than landing a bookmarked search on an empty
        # page. /v2/add reads ?q= and pre-fills the search box.
        return _moved(f"/v2/add?q={quote(q, safe='')}" if q else "/v2/add")

    @app.get("/import")
    def moved_import():
        return _moved("/v2/add")

    @app.get("/add/{wizard_id}/{step:path}")
    def moved_wizard_step(wizard_id: str, step: str):
        return _moved(f"/v2/add/{quote(wizard_id, safe='')}")

    @app.get("/add")
    def moved_add():
        return _moved("/v2/add")

    # Every old per-profile page is a section of the one model page.
    @app.get("/profiles/{name}/run.sh")
    def moved_profile_runsh(name: str):
        return _moved(f"/v2/models/{quote(name, safe='')}")

    @app.get("/profiles/{name}/{section:path}")
    def moved_profile_section(name: str, section: str):
        return _moved(f"/v2/models/{quote(name, safe='')}")

    @app.get("/profiles/{name}")
    def moved_profile(name: str):
        return _moved(f"/v2/models/{quote(name, safe='')}")

    @app.get("/api/runtime")
    def api_runtime():
        return _runtime_state()

    @app.get("/api/runtime/models/{name:path}")
    def api_runtime_model(name: str):
        rt = _swap_client().model_state(name)
        return rt

    @app.get("/api/runtime/logs/{name:path}", response_class=PlainTextResponse)
    def api_runtime_logs(name: str):
        client = _swap_client()
        try:
            logs_data = client.logs(model_id=name)
            return logs_data if isinstance(logs_data, str) else json.dumps(logs_data, indent=2)
        except ModelctlSwapError as e:
            return JSONResponse({"error": e.code, "message": e.message}, status_code=502)

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
    @app.get("/api/profiles/{name}/runtime-policy")
    def api_runtime_policy(name: str):
        p = modelctl.load_profile(name)
        return p.get("runtime") or {"mode": "fixed"}

    # ---- plan testing + tuning --------------------------------------------
    @app.get("/api/profiles/{name}/history")
    def api_history(name: str):
        import modelctl_runtime
        return modelctl_runtime.RuntimeDB().plan_runs_for(name)

    # ---- managed routing matrix -------------------------------------------
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

    # ---- /v2 settings ----------------------------------------------------
    # Typed surface for the settings that already exist and already have a
    # backing implementation; see modelctl_web/settings.py for the scope
    # rule. Writes reuse the CLI's own service functions, so the console
    # and `modelctl defaults` cannot disagree about validation.

    @app.get("/api/v2/settings")
    def api_v2_settings():
        from . import settings as settings_mod
        return settings_mod.overview(TOKEN_PATH)

    @app.post("/api/v2/settings/capabilities/reprobe")
    def api_v2_reprobe():
        # The escape hatch the old /setup page owned: drop every cached
        # capability verdict so the next probing read rebuilds them.
        # Recovering from a poisoned entry otherwise means hand-deleting
        # files in the state dir.
        import modelctl_capabilities
        modelctl_capabilities.clear_cache()
        return {"ok": True, "message": "capability cache cleared; the next "
                                       "read reprobes the backend"}

    # Runtime actions for the operate page. These replace the old console's
    # POST /models/{name}/... routes one-for-one -- same mutate.submit_*
    # calls, same job lane -- but answer with the job id as JSON instead of
    # a 303 into a job page that no longer exists. {name:path} because
    # llama-swap model ids can contain a slash.
    @app.post("/api/v2/models/{name:path}/load")
    def api_v2_model_load(name: str):
        return {"job_id": mutate.submit_load(runner, name)}

    @app.post("/api/v2/models/{name:path}/unload")
    def api_v2_model_unload(name: str):
        return {"job_id": mutate.submit_unload(runner, name)}

    @app.post("/api/v2/settings/defaults")
    async def api_v2_settings_defaults(request: Request):
        from . import settings as settings_mod
        body = await request.json()
        updates = body.get("updates") if isinstance(body, dict) else None
        if not isinstance(updates, dict):
            return JSONResponse(
                {"error": "expected {\"updates\": {field: value}}"},
                status_code=400)
        return settings_mod.update_defaults(updates)

    @app.post("/api/v2/settings/hardware")
    async def api_v2_settings_hardware(request: Request):
        from . import settings as settings_mod
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected an object"},
                                status_code=400)
        return settings_mod.update_hardware(body)

    @app.post("/api/v2/settings/hardware/calibrate")
    def api_v2_settings_calibrate():
        # Storage calibration is a measurement job, not a setting: it goes
        # on the mutation lane like every other write and the page follows
        # it on the shared job stream.
        job_id = mutate.submit_calibrate_storage(runner)
        return {"job_id": job_id}

    @app.get("/api/v2/settings/support-bundle")
    def api_v2_support_bundle():
        import modelctl_diagnostics
        data = modelctl_diagnostics.support_bundle_bytes()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return Response(
            content=data, media_type="application/zip",
            headers={"content-disposition":
                     f'attachment; filename="modelctl-support-{stamp}.zip"'})

    @app.post("/api/v2/settings/token/rotate")
    async def api_v2_settings_rotate_token(request: Request):
        from . import settings as settings_mod
        body = await request.json()
        # Rotation signs out every other session; it is destructive enough
        # to need the same explicit confirm a structural profile change
        # gets, so an accidental POST cannot lock the operator's phone out.
        if not (isinstance(body, dict) and body.get("confirm") is True):
            return JSONResponse(
                {"error": "rotation needs an explicit confirm",
                 "gate": {"changes": [
                     "every signed-in session except this one is signed out",
                     "any script or bookmark holding the old token stops"
                     " working until it is updated",
                 ]}}, status_code=409)
        result = settings_mod.rotate_token(TOKEN_PATH)
        if not result["ok"]:
            return JSONResponse({"error": result["error"]}, status_code=400)
        app.state.token = result["token"]
        # The rotating session keeps working: re-issue its cookie with the
        # new value. The token itself is never returned in the body -- it
        # would land in the browser's memory and in any logging proxy.
        resp = JSONResponse({"ok": True, "token_path": str(TOKEN_PATH),
                             "message": "token rotated; this session was "
                                        "re-signed, others must sign in again"})
        _set_session_cookie(resp, result["token"])
        return resp

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
