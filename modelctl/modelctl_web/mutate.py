"""Mutation helpers: every write path in the console submits through here.

Functions accept a JobContext (ctx) with logging, progress, and cancellation
support.  The submit_* helpers route jobs to the appropriate lane.
"""
import json
import time

import modelctl


def submit_edit(runner, name, updates):
    """Apply config/profile field updates, regenerate, sync."""
    from modelctl_services import profile_service

    def fn(ctx):
        result = profile_service.update_config(name, updates, ctx=ctx)
        if not result.ok:
            raise RuntimeError(result.messages[0] if result.messages
                              else f"edit failed for '{name}'")
        return {"name": name, "messages": result.messages}
    return runner.submit("edit", f"edit {name}", fn, payload={"name": name, "updates": updates},
                         lane="mutation")


def submit_tier_apply(runner, name):
    """Compute the tier plan for one profile and apply it."""
    import modelctl_tiers

    def fn(ctx):
        profile = modelctl.load_profile(name)
        inventory = modelctl.get_gpu_inventory()
        d = modelctl.load_defaults()
        primary = modelctl.resolve_primary_gpu(inventory, d)
        plan = modelctl_tiers.plan_tiers(profile, inventory, d["vram_limit_pct"], primary,
                                          cache_request=profile.get("moe_cache"))
        if plan is None:
            raise RuntimeError(f"couldn't analyze model layout for '{name}'")
        modelctl_tiers.apply_plan_cache_budgets(profile, plan, log=ctx.log)
        cfg = profile.get("config", {})
        cfg.update(plan["config"])
        profile["config"] = cfg
        if not profile.get("env"):
            env = modelctl._env_from_scripts()
            if env:
                profile["env"] = env
                ctx.log("populated env from env script")
        modelctl.save_profile(profile)
        modelctl.generate_artifacts(profile)
        modelctl.sync_all_backends(restart_router=True, restart_openarc=True)
        for label, gib, desc in plan["layout"]:
            ctx.log(f"{label}: {gib:.1f} GiB  {desc}")
        for w in plan["warnings"]:
            ctx.log(f"WARNING: {w}")
        return {"tier": plan["tier"], "config": plan["config"],
                "warnings": plan["warnings"]}
    return runner.submit("tier-apply", f"tier apply {name}", fn,
                         payload={"name": name}, lane="mutation")


def submit_pull(runner, repo_id, quant_label=None, want_mtp=True):
    """Zero-config pull as a background job, with download progress."""
    def fn(ctx):
        from pathlib import Path
        current = {"file": None}

        def progress(event, name):
            if event == "file_start":
                current["file"] = name
                ctx.log(f"downloading {name}")
                ctx.set_progress(detail=name)
            elif event == "file_done":
                ctx.log(f"done {name}")

        def poller():
            while True:
                time.sleep(3)
                if not current["file"]:
                    job = ctx.store.get(ctx.job_id)
                    if not job or job["status"] != "running":
                        return
                    continue
                target = Path(modelctl.DEFAULT_MODELS_DIR) / repo_id / current["file"]
                total = modelctl._remote_file_size(repo_id, current["file"])
                try:
                    have = target.stat().st_size if target.exists() else 0
                except OSError:
                    have = 0
                if total:
                    ctx.set_progress(
                        min(0.99, have / total),
                        f"{current['file']}: "
                        f"{modelctl._format_size(have)} / "
                        f"{modelctl._format_size(total)}")
                job = ctx.store.get(ctx.job_id)
                if not job or job["status"] != "running":
                    return

        import threading
        t = threading.Thread(target=poller, daemon=True)
        t.start()
        ctx.raise_if_cancelled()
        profile = modelctl.pull_model(repo_id, quant_label=quant_label,
                                      want_mtp=want_mtp, progress_cb=progress)
        current["file"] = None
        if profile is None:
            raise RuntimeError(f"pull of {repo_id} failed -- see job log")
        ctx.log(f"profile '{profile['name']}' saved and synced")
        return {"profile": profile["name"], "config": profile["config"]}
    return runner.submit("pull", f"pull {repo_id}", fn,
                         payload={"repo_id": repo_id, "quant": quant_label},
                         lane="download")


def submit_import_local(runner, file_path, name=None, copy=False):
    """Import a local GGUF file as a profile.

    Verification (GGUF magic, shard completeness, readability, duplicate
    identity, destination space) happens in acquisition_service before a
    profile is created, so the CLI import path gets the same checks.
    """
    from modelctl_services import acquisition_service

    def fn(ctx):
        result = acquisition_service.import_local(
            file_path, name=name, copy=copy, resync=True, ctx=ctx)
        for m in result.messages:
            ctx.log(m)
        for w in result.warnings:
            ctx.log(f"warning: {w}")
        if not result.ok:
            raise RuntimeError("; ".join(result.messages)
                               or f"import of {file_path} failed")
        return dict(result.data, warnings=result.warnings)
    return runner.submit("import", f"import {file_path}", fn,
                         payload={"file_path": file_path, "name": name},
                         lane="mutation")


def submit_smoke_test(runner, name):
    def fn(ctx):
        ctx.raise_if_cancelled()
        res = modelctl.smoke_test_profile(name, proc_register=ctx.register_process)
        for m in res["messages"]:
            ctx.log(m)
        if res.get("tok_per_s"):
            ctx.log(f"{res['tok_per_s']:.1f} tok/s")
        ctx.log("PASS" if res["ok"] else f"not ok ({res['stage']})")
        if not res["ok"] and res["stage"] not in ("ovms",):
            raise RuntimeError(f"smoke test failed at stage {res['stage']}")
        return res
    return runner.submit("smoke", f"smoke test {name}", fn,
                         payload={"name": name}, lane="benchmark")


def submit_bench(runner, name, max_tokens=256, runs=3):
    import subprocess as _sp
    import os

    def fn(ctx):
        cmd = modelctl.build_speed_command(name, max_tokens=max_tokens, runs=runs)
        ctx.log(" ".join(cmd))
        ctx.raise_if_cancelled()
        proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
                         preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None)
        ctx.register_process(proc)
        out_lines = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            out_lines.append(line)
            ctx.log(line)
            ctx.raise_if_cancelled()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"speed.py exited {proc.returncode}")
        return {"output_tail": out_lines[-5:]}
    return runner.submit("bench", f"benchmark {name}", fn,
                         payload={"name": name}, lane="benchmark")


def submit_load(runner, name):
    from modelctl_services import runtime_service

    def fn(ctx):
        result = runtime_service.load_model(name, ctx=ctx)
        if not result.ok:
            raise RuntimeError(result.messages[0] if result.messages
                              else f"load failed for '{name}'")
        return {"loaded": result.loaded, "response_ok": result.response_ok,
                "elapsed_s": result.elapsed_s}
    return runner.submit("load", f"load {name}", fn,
                         payload={"name": name}, lane="runtime")


def submit_unload(runner, name):
    from modelctl_services import runtime_service

    def fn(ctx):
        result = runtime_service.unload_model(name, ctx=ctx)
        if not result.ok:
            raise RuntimeError(result.messages[0] if result.messages
                              else f"unload failed for '{name}'")
        return {"unloaded": name}
    return runner.submit("unload", f"unload {name}", fn,
                         payload={"name": name}, lane="runtime")


def submit_restart(runner, name):
    from modelctl_services import runtime_service

    def fn(ctx):
        result = runtime_service.restart_model(name, ctx=ctx)
        if not result.ok:
            raise RuntimeError(result.messages[0] if result.messages
                              else f"restart failed for '{name}'")
        return {"loaded": result.loaded, "response_ok": result.response_ok,
                "elapsed_s": result.elapsed_s}
    return runner.submit("restart", f"restart {name}", fn,
                         payload={"name": name}, lane="runtime")


def submit_unload_all(runner):
    from modelctl_services import runtime_service

    def fn(ctx):
        result = runtime_service.unload_all(ctx=ctx)
        if not result.ok:
            raise RuntimeError(result.messages[0] if result.messages
                              else "unload all failed")
        return {"unloaded": result.models}
    return runner.submit("unload-all", "unload all models", fn,
                         lane="runtime")


def submit_runtime_policy(runner, name, runtime):
    def fn(ctx):
        ctx.log(f"runtime policy for '{name}': " + json.dumps(runtime))
        modelctl.update_runtime_policy(name, runtime)
        ctx.log("profile saved, artifacts regenerated, synced")
        return {"name": name, "runtime": runtime}
    return runner.submit("mutation", f"runtime policy {name}", fn,
                         payload={"name": name, "runtime": runtime})


def submit_plan_select(runner, name, plan_id, disable=False):
    from modelctl_services import plan_service

    def fn(ctx):
        result = (plan_service.disable_plan(name, plan_id) if disable
                  else plan_service.select_plan(name, plan_id))
        if not result.ok:
            raise RuntimeError(result.messages[0] if result.messages
                              else f"plan {'disable' if disable else 'select'} failed for '{name}'")
        for m in result.messages:
            ctx.log(m)
        profile = modelctl.load_profile(name)
        return {"name": name, "runtime": profile.get("runtime", {})}
    label = "disable" if disable else "select"
    return runner.submit("mutation", f"{label} plan {plan_id} on {name}", fn,
                         payload={"name": name, "plan_id": plan_id,
                                  "disable": disable})


def submit_plan_test(runner, name, plan_id):
    def fn(ctx):
        import modelctl_tune
        run = modelctl_tune.test_launch_plan(
            name, plan_id, log=ctx.log, proc_register=ctx.register_process,
            cancel_check=ctx.is_cancelled)
        if not run.get("success"):
            raise RuntimeError(f"plan test failed: {run.get('failure_class')}")
        return run
    return runner.submit("benchmark", f"plan test {name}/{plan_id[:8]}", fn,
                         payload={"name": name, "plan_id": plan_id})


def submit_autotune(runner, name, objective="balanced", candidate_ids=None):
    def fn(ctx):
        import modelctl_tune
        res = modelctl_tune.autotune_profile(
            name, objective=objective, candidate_ids=candidate_ids,
            log=ctx.log, proc_register=ctx.register_process,
            cancel_check=ctx.is_cancelled)
        return res
    return runner.submit("benchmark", f"autotune {name}", fn,
                         payload={"name": name, "objective": objective})


def submit_calibrate_storage(runner):
    def fn(ctx):
        from modelctl_services import hardware_service
        file_path = hardware_service.pick_calibration_file()
        if not file_path:
            raise RuntimeError("no .gguf file found under the models directory to calibrate against")
        ctx.log(f"calibrating sequential read using {file_path} ...")
        result = hardware_service.calibrate_storage(file_path)
        if not result.ok:
            raise RuntimeError(result.messages[0] if result.messages
                              else "storage calibration failed")
        for m in result.messages:
            ctx.log(m)
        return {"messages": result.messages}
    return runner.submit("calibrate-storage", "calibrate storage", fn, lane="mutation")


def submit_matrix_apply(runner):
    """Apply the managed routing matrix.

    The backup/write/restart/health-check/rollback sequence lives in
    routing_service; this only submits it and translates the
    result into the job outcome. The rollback_status is preserved in the
    payload, because "the apply failed" and "the apply failed and the old
    config could not be restored" need different responses.
    """
    from modelctl_services import routing_service

    def fn(ctx):
        result = routing_service.apply_matrix(ctx=ctx)
        if not result.ok:
            raise RuntimeError("; ".join(result.messages)
                               or "matrix apply failed")
        return {**result.data, "rollback_status": result.rollback_status}
    return runner.submit("mutation", "apply managed matrix", fn)


def submit_moe_cache(runner, name, moe_cache):
    """Save moe_cache section for a profile, regenerate artifacts, and sync.

    Validates against probed backend capabilities first (fail closed):
    error-level results (e.g. manual mode on an incapable or unprobed
    binary) reject the save instead of writing a profile whose flags the
    backend cannot honor."""
    def fn(ctx):
        profile = modelctl.load_profile(name)
        profile["moe_cache"] = moe_cache
        ctx.log(f"updating moe_cache for '{name}': mode={moe_cache.get('mode', 'off')}")
        import modelctl_capabilities
        binary = profile.get("binary") or modelctl.LLAMA_SERVER_BIN
        caps = modelctl_capabilities.probe_backend(binary) if binary else None
        msgs = modelctl.preflight_moe_cache(profile, capabilities=caps)
        for lvl, msg in msgs:
            if lvl == "warning":
                ctx.log(f"warning: {msg}")
        errors = [msg for lvl, msg in msgs if lvl == "error"]
        if errors:
            raise RuntimeError("moe_cache validation failed: " + "; ".join(errors))
        modelctl.save_profile(profile)
        modelctl.generate_artifacts(profile)
        modelctl.sync_all_backends(restart_router=True, restart_openarc=True)
        ctx.log("saved, artifacts regenerated, synced")
        return {"name": name, "moe_cache": moe_cache}
    return runner.submit("moe-cache", f"moe cache {name}", fn,
                         payload={"name": name, "moe_cache": moe_cache},
                         lane="mutation")
