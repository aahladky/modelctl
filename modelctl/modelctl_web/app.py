"""FastAPI application for the modelctl web console.

Reads are direct calls into modelctl (concurrent); every write goes through
the single JobRunner worker (see jobs.py). Auth: one shared token, checked
as Bearer header, ?token= query param, or the login cookie.
"""
import json
import os
import secrets
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import modelctl
import modelctl_vram

from . import mutate
from .jobs import JobRunner, JobStore, STATE_DIR

TOKEN_PATH = STATE_DIR / "web_token"
COOKIE_NAME = "modelctl_web_token"
TEMPLATE_DIR = Path(__file__).parent / "templates"

from .swap import LlamaSwapClient, ModelctlSwapError

LLAMA_SWAP_BASE = os.environ.get("MODELCTL_LLAMA_SWAP_BASE_URL",
                                 "http://127.0.0.1:9292/v1/")
LLAMA_SWAP_ROOT = LLAMA_SWAP_BASE.rsplit("/v1/", 1)[0]


def load_or_create_token():
    env = os.environ.get("MODELCTL_WEB_TOKEN")
    if env:
        return env
    try:
        return TOKEN_PATH.read_text().strip()
    except OSError:
        pass
    token = secrets.token_hex(16)
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token + "\n")
    os.chmod(TOKEN_PATH, 0o600)
    return token


def _fetch_json(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _scrape_moe_cache_metrics(port):
    """Fetch /metrics from a running model and extract moe_cache stats.

    Label-shape agnostic: the device label may be absent, reordered, or
    accompanied by extra labels -- parse the label block generically and
    pull device out of it instead of requiring exactly one device="...".
    Metrics without a device label land under the "" key.
    """
    import re as _re
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3) as r:
            text = r.read().decode()
        stats = {}
        for line in text.splitlines():
            if not line.startswith("llamacpp:moe_cache_"):
                continue
            m = _re.match(r"llamacpp:moe_cache_(\w+)(\{[^}]*\})?\s+(\S+)", line)
            if not m:
                continue
            key, labels, val = m.group(1), m.group(2) or "", m.group(3)
            dev = ""
            dm = _re.search(r'device="([^"]*)"', labels)
            if dm:
                dev = dm.group(1)
            stats.setdefault(dev, {})[key] = val
        return stats or None
    except Exception:
        return None


def create_app(token=None, store=None, runner=None):
    token = token or load_or_create_token()
    store = store or JobStore()
    runner = runner or JobRunner(store)
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    templates.env.filters["fmt_size"] = modelctl._format_size
    templates.env.filters["fromjson"] = json.loads

    app = FastAPI(title="modelctl-web", docs_url=None, redoc_url=None)
    app.state.token = token
    app.state.store = store
    app.state.runner = runner
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")),
              name="static")

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path in ("/login", "/healthz") or request.url.path.startswith("/static"):
            return await call_next(request)
        supplied = (
            request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            or request.cookies.get(COOKIE_NAME, ""))
        if supplied != token:
            if request.url.path.startswith("/api/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return RedirectResponse(f"/login?next={request.url.path}")
        if request.method == "POST":
            origin = (request.headers.get("origin")
                      or request.headers.get("referer") or "")
            if origin:
                from urllib.parse import urlparse
                if urlparse(origin).netloc != request.headers.get("host", ""):
                    return JSONResponse({"error": "cross-origin POST rejected"},
                                        status_code=403)
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
        d = {"request": request}
        d.update(kw)
        return d

    # ---- template filters ------------------------------------------------
    def _fmt_elapsed(start_ts):
        if not start_ts:
            return ""
        s = max(0, int(time.time() - start_ts))
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

    @app.post("/login")
    def login(next: str = Form("/"), token_field: str = Form("")):
        if token_field != token:
            return RedirectResponse("/login")
        if not next.startswith("/") or next.startswith("//"):
            next = "/"
        resp = RedirectResponse(next)
        resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict",
                        secure=os.environ.get("MODELCTL_WEB_SECURE_COOKIE", "") == "1")
        return resp

    @app.get("/logout")
    def logout():
        resp = RedirectResponse("/login")
        resp.delete_cookie(COOKIE_NAME)
        return resp

    # ---- dashboard ------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
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
        caps = None
        binary = p.get("binary") or modelctl.LLAMA_SERVER_BIN
        if binary:
            try:
                from modelctl_capabilities import get_cached_capabilities
                caps = get_cached_capabilities(binary)
            except Exception:
                pass
        args = modelctl.build_server_args(p, capabilities=caps)
        messages = [f"{lvl}: {msg}" for lvl, msg in
                    modelctl.preflight_moe_cache(p, capabilities=caps)
                    if lvl != "ok"]
        return templates.TemplateResponse(request=request, name="profile_edit.html", context=ctx(
            request, p=p, fields=EDIT_FIELDS, saved=saved,
            run_sh=" \\\n  ".join(args), messages=messages,
            capabilities=caps))

    @app.post("/profiles/{name}")
    async def profile_save(request: Request, name: str):
        form = await request.form()
        updates = {}
        for f in EDIT_FIELDS:
            if f in form:
                updates[f] = str(form[f])
        for key in ("enabled",):
            updates[key] = key in form
        job_id = mutate.submit_edit(runner, name, updates)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}", status_code=303)

    @app.post("/profiles/{name}/moe-cache")
    async def profile_moe_cache_save(request: Request, name: str):
        form = await request.form()
        # Parse flat form fields into nested moe_cache dict.
        mc = {"gpu": {"budgets_bytes": {}}, "ram": {}, "storage": {},
              "prefill": {}, "decode": {}, "prefetch": {}}
        for key, val in form.items():
            val = str(val)
            if key == "mode":
                mc["mode"] = val
            elif key.startswith("gpu.budgets_bytes."):
                dev = key.split(".", 2)[2]
                if dev == "_new_device":
                    continue
                try:
                    mc["gpu"]["budgets_bytes"][dev] = int(val)
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
        # Strip empty budgets.
        mc["gpu"]["budgets_bytes"] = {k: v for k, v in mc["gpu"]["budgets_bytes"].items() if v > 0}
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
        return " \\\n  ".join(modelctl.build_server_args(p))

    # ---- tier planner ---------------------------------------------------
    def _plan_for(name):
        import modelctl_tiers
        p = modelctl.load_profile(name)
        inventory = modelctl.get_gpu_inventory()
        d = modelctl.load_defaults()
        primary = modelctl.resolve_primary_gpu(inventory, d)
        plan = modelctl_tiers.plan_tiers(p, inventory, d["vram_limit_pct"], primary)
        return p, plan

    @app.get("/profiles/{name}/tiers", response_class=HTMLResponse)
    def tiers_one(request: Request, name: str):
        p, plan = _plan_for(name)
        return templates.TemplateResponse(request=request, name="tiers.html", context=ctx(
            request, p=p, plan=plan, current=p.get("config", {})))

    @app.post("/profiles/{name}/tiers/apply")
    def tiers_apply(name: str):
        job_id = mutate.submit_tier_apply(runner, name)
        return RedirectResponse(f"/jobs/{job_id}?back=/profiles/{name}/tiers",
                                status_code=303)

    @app.get("/tiers", response_class=HTMLResponse)
    def tiers_all(request: Request):
        import modelctl_tiers
        inventory = modelctl.get_gpu_inventory()
        d = modelctl.load_defaults()
        primary = modelctl.resolve_primary_gpu(inventory, d)
        plans = []
        for p in profiles():
            if not p.get("model_path"):
                continue
            plan = modelctl_tiers.plan_tiers(p, inventory, d["vram_limit_pct"], primary)
            plans.append({"name": p["name"], "plan": plan})
        return templates.TemplateResponse(request=request, name="tiers_all.html", context=ctx(
            request, plans=plans))

    # ---- pull wizard ----------------------------------------------------
    @app.get("/pull", response_class=HTMLResponse)
    def pull_form(request: Request, q: str = ""):
        results = modelctl.search_models(q) if q else []
        return templates.TemplateResponse(request=request, name="pull.html", context=ctx(
            request, q=q, results=results))

    @app.get("/pull/{repo_id:path}", response_class=HTMLResponse)
    def pull_repo(request: Request, repo_id: str):
        import modelctl_tiers
        contents = modelctl.get_repo_contents(repo_id)
        inventory = modelctl.get_gpu_inventory()
        d = modelctl.load_defaults()
        budget = 0
        if inventory:
            primary = modelctl.resolve_primary_gpu(inventory, d)
            prim = next((g for g in inventory if g["device"] == primary), inventory[0])
            budget = prim["total_bytes"] * d["vram_limit_pct"] / 100.0
        rec = modelctl_tiers.recommend_quant_group(
            contents["quant_groups"], budget, int(d["ctx"]))
        return templates.TemplateResponse(request=request, name="pull_repo.html", context=ctx(
            request, repo_id=repo_id, contents=contents, rec=rec,
            budget=budget))

    @app.post("/pull/{repo_id:path}")
    def pull_start(repo_id: str, quant: str = Form("")):
        job_id = mutate.submit_pull(runner, repo_id, quant_label=quant or None)
        return RedirectResponse(f"/jobs/{job_id}?back=/pull", status_code=303)

    # ---- local import ---------------------------------------------------
    @app.get("/import", response_class=HTMLResponse)
    def import_form(request: Request):
        return templates.TemplateResponse(request=request, name="import.html", context=ctx(
            request))

    @app.post("/import")
    def import_start(file_path: str = Form(...), name: str = Form(""),
                     copy: bool = Form(False)):
        job_id = mutate.submit_import_local(runner, file_path,
                                            name=name or None, copy=copy)
        return RedirectResponse(f"/jobs/{job_id}?back=/import", status_code=303)

    # ---- add-model wizard -----------------------------------------------
    @app.get("/add", response_class=HTMLResponse)
    def wizard_list(request: Request):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        active = store_wiz.list_active()
        return templates.TemplateResponse(request=request, name="wizard_list.html",
                                          context=ctx(request, wizards=active))

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
        if source_type == "hf_repo":
            state.repo_id = form.get("repo_id", "")
            state.advance("inspect")
        elif source_type == "local_file":
            state.local_path = form.get("local_path", "")
            state.advance("download")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/{state.step}", status_code=303)

    @app.get("/add/{wizard_id}/inspect", response_class=HTMLResponse)
    def wizard_inspect(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state or not state.repo_id:
            return RedirectResponse("/add", status_code=303)
        contents = modelctl.get_repo_contents(state.repo_id)
        return templates.TemplateResponse(request=request, name="wizard_inspect.html",
                                          context=ctx(request, w=state, contents=contents))

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
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/download", status_code=303)

    @app.get("/add/{wizard_id}/download", response_class=HTMLResponse)
    def wizard_download(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        # For local files, skip download and go straight to analysis.
        if state.source_type == "local_file" and state.local_path:
            if not state.download_complete:
                job_id = mutate.submit_import_local(runner, state.local_path)
                state.download_job_id = job_id
                state.download_complete = True
                store_wiz.save(state)
            return templates.TemplateResponse(request=request, name="wizard_download.html",
                                              context=ctx(request, w=state, is_local=True))
        # For HF repos, start the pull job.
        if state.repo_id and not state.download_complete and not state.download_job_id:
            job_id = mutate.submit_pull(runner, state.repo_id,
                                        quant_label=state.selected_quant or None)
            state.download_job_id = job_id
            store_wiz.save(state)
        return templates.TemplateResponse(request=request, name="wizard_download.html",
                                          context=ctx(request, w=state, is_local=False))

    @app.post("/add/{wizard_id}/download")
    def wizard_download_next(request: Request, wizard_id: str):
        from .wizard import WizardStore
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        state.advance("analyze")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/analyze", status_code=303)

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
            if job and job.get("status") == "done" and job.get("result"):
                pname = job["result"].get("profile")
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
        if not state or not state.profile_name:
            return RedirectResponse("/add", status_code=303)
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
        job_id = mutate.submit_smoke_test(runner, state.profile_name)
        state.test_job_id = job_id
        state.advance("register")
        store_wiz.save(state)
        return RedirectResponse(f"/add/{wizard_id}/register", status_code=303)

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
        store_wiz = WizardStore()
        state = store_wiz.load(wizard_id)
        if not state:
            return RedirectResponse("/add", status_code=303)
        # Warm-load through llama-swap to verify.
        if state.profile_name:
            job_id = mutate.submit_load(runner, state.profile_name)
            state.registration_complete = True
            state.endpoint = f"/v1/models"
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
        return templates.TemplateResponse(request=request, name="wizard_done.html",
                                          context=ctx(request, w=state))

    # ---- jobs -----------------------------------------------------------
    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_list(request: Request):
        return templates.TemplateResponse(request=request, name="jobs.html", context=ctx(
            request, jobs=store.list()))

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_detail(request: Request, job_id: str, back: str = "/"):
        job = store.get(job_id)
        return templates.TemplateResponse(request=request, name="job_detail.html", context=ctx(
            request, job=job, back=back))

    @app.get("/jobs/{job_id}/fragment", response_class=HTMLResponse)
    def job_fragment(request: Request, job_id: str, back: str = "/"):
        job = store.get(job_id)
        return templates.TemplateResponse(request=request, name="job_fragment.html", context=ctx(
            request, job=job, back=back))

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
    def profile_bench(name: str):
        job_id = mutate.submit_bench(runner, name)
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
            except (Exception, SystemExit):
                # llama-swap may run models with no modelctl profile;
                # load_profile sys.exit()s on those (not an Exception).
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
        return templates.TemplateResponse(request=request, name="runtime.html", context=ctx(
            request, runtime=state, cache_stats=cache_stats))

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
        rt = client.model_state(name)
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
        if port:
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/cache/reset", method="POST")
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass
        return RedirectResponse("/runtime", status_code=303)

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
        form = await request.form()
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
        import modelctl_hardware as _hw
        snap = _hw.capture_hardware_snapshot(settings)
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
    def settings_page(request: Request, saved: str = ""):
        defaults = modelctl.load_defaults()
        return templates.TemplateResponse(request=request, name="settings.html", context=ctx(
            request, defaults=defaults, saved=saved,
            models_dir=str(modelctl.DEFAULT_MODELS_DIR),
            profiles_dir=str(modelctl.PROFILES_DIR),
            llama_server=modelctl.LLAMA_SERVER_BIN,
            llama_swap_config=str(modelctl.LLAMA_SWAP_CONFIG_PATH),
            llama_swap_service=modelctl.LLAMA_SWAP_SERVICE_NAME,
            llama_swap_base_url=modelctl.LLAMA_SWAP_BASE_URL,
        ))

    @app.post("/settings/save")
    async def settings_save(request: Request):
        form = await request.form()
        defaults = modelctl.load_defaults()
        for key in ("device", "split_mode", "tensor_split", "cache_type_k",
                     "cache_type_v", "flash_attn", "mtp", "primary_gpu"):
            if key in form:
                defaults[key] = str(form[key])
        for key in ("ctx", "ttl", "vram_limit_pct"):
            if key in form:
                try:
                    defaults[key] = int(form[key])
                except ValueError:
                    pass
        modelctl.save_defaults(defaults)
        return RedirectResponse("/settings?saved=1", status_code=303)

    # ---- launch plans ----------------------------------------------------
    @app.get("/profiles/{name}/plans", response_class=HTMLResponse)
    def plans_page(request: Request, name: str):
        import modelctl_plans
        import modelctl_hardware
        p = modelctl.load_profile(name)
        snap = modelctl_hardware.capture_hardware_snapshot()
        plans = modelctl_plans.compile_launch_plans(p, snap)
        import modelctl_runtime
        observations = modelctl_runtime.RuntimeDB().observations_for_profile(
            name, snap.fingerprint,
            snap.backend_fingerprints.get(p.get("backend", "llama-cpp"), ""))
        return templates.TemplateResponse(request=request, name="plans.html", context=ctx(
            request, p=p, plans=plans, profile_name=name, observations=observations))

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
        runs = modelctl_runtime.RuntimeDB().plan_runs_for(name)
        return templates.TemplateResponse(request=request, name="history.html", context=ctx(
            request, name=name, runs=runs))

    @app.get("/api/profiles/{name}/history")
    def api_history(name: str):
        import modelctl_runtime
        return modelctl_runtime.RuntimeDB().plan_runs_for(name)

    # ---- managed routing matrix -------------------------------------------
    @app.get("/runtime/routing", response_class=HTMLResponse)
    def routing_page(request: Request):
        import modelctl_matrix
        config = modelctl.yaml.safe_load(modelctl.LLAMA_SWAP_CONFIG_PATH.read_text()) or {}
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
        _p, plan = _plan_for(name)
        return plan

    @app.post("/api/tiers/{name}/apply")
    def api_tiers_apply(name: str):
        return {"job": mutate.submit_tier_apply(runner, name)}

    @app.post("/api/pull/{repo_id:path}")
    def api_pull(repo_id: str, quant: str = ""):
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

    return app


app = create_app()
