"""Mutation helpers: every write path in the console submits through here.

Functions accept a JobContext (ctx) with logging, progress, and cancellation
support.  The submit_* helpers route jobs to the appropriate lane.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import modelctl
from modelctl_services import acquisition_service

from . import pull_progress


def _sync_backends(ctx):
    """Sync configs and tell the truth about the router reload.

    Returns False when the config was written but llama-swap was NOT
    reloaded -- the caller must surface that, or the job reports a
    placement as live while the old config is still serving.
    """
    out = modelctl.sync_all_backends(restart_router=True, restart_openarc=True)
    if isinstance(out, dict) and out.get("restarted") is False:
        ctx.log("WARNING: config written but llama-swap was NOT reloaded "
                f"({out.get('restart_error') or 'reload failed'}) -- the "
                "previous config is still serving until the router restarts")
        return False
    return True


def submit_sync(runner):
    """Re-run the backend sync on demand -- the one-click retry for a job
    that reported "config written but the router was NOT reloaded".

    Fails (job goes red, reason in the error) when the reload fails
    again, so job history keeps telling the truth instead of stacking
    green "done" rows on top of a router still serving the old config."""
    def fn(ctx):
        out = modelctl.sync_all_backends(restart_router=True,
                                         restart_openarc=True)
        profiles = out.get("profiles") if isinstance(out, dict) else out
        ctx.log(f"synced {profiles} profile config(s)")
        if isinstance(out, dict) and out.get("restarted") is False:
            raise RuntimeError(
                "config written but llama-swap was NOT reloaded "
                f"({out.get('restart_error') or 'reload failed'})")
        return {"profiles": profiles,
                "router_reloaded": out.get("restarted")
                if isinstance(out, dict) else None}
    return runner.submit("sync", "sync backends", fn, lane="mutation")


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


def _apply_tier_plan(ctx, profile, plan, inputs, source, accept_tier_change):
    """Write one planner layout onto a profile, gate first.

    Shared by every apply path -- the automatic replan and the operator's
    device selection -- because they differ only in how the plan was
    computed. Two copies of this would be two answers to "what did the
    console just do to my profile".

    The caller owns the in-memory profile and may stage its own fields on
    it beforehand; a gated refusal returns without saving, so anything
    staged is discarded with it.
    """
    import time

    import modelctl_tiers

    ctx.log(f"machine snapshot: {source}")
    for w in plan["warnings"]:
        ctx.log(f"WARNING: {w}")
    gate = modelctl_tiers.tier_change_gate(profile, plan)
    if gate["requires_accept"] and not accept_tier_change:
        ctx.log("NOT APPLIED: this recalculation changes level, split, or "
                "placement beyond pins:")
        for c in gate["changes"]:
            ctx.log(f"  {c}")
        ctx.log("re-apply with 'accept placement change' checked; the "
                "profile is untouched.")
        return {"applied": False, "requires_accept_tier_change": True,
                "changes": gate["changes"], "tier": plan["tier"],
                "warnings": plan["warnings"]}
    if modelctl_tiers.record_planning_inputs(
            profile, inputs, recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S")):
        ctx.log(f"machine snapshot recorded ({source})")
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
    reloaded = _sync_backends(ctx)
    for label, gib, desc in plan["layout"]:
        ctx.log(f"{label}: {gib:.1f} GiB  {desc}")
    warnings = list(plan["warnings"])
    if not reloaded:
        warnings.append("the router was not reloaded -- the previous "
                        "placement is still serving")
    return {"applied": True, "tier": plan["tier"],
            "config": plan["config"], "warnings": warnings,
            "router_reloaded": reloaded,
            "admission": plan.get("admission")}


def submit_tier_apply(runner, name, accept_tier_change=False):
    """Compute the tier plan for one profile and apply it.

    Planner inputs resolve stored-first (recorded on first apply); a
    replan that changes tier, split, or placement beyond P5 pins is NOT
    applied unless accept_tier_change is set -- the job reports the diff
    and leaves the profile untouched."""
    def fn(ctx):
        profile = modelctl.load_profile(name)
        plan, inputs, source = modelctl.plan_tiers_for_profile(profile)
        if plan is None:
            raise RuntimeError(f"couldn't analyze model layout for '{name}'")
        return _apply_tier_plan(ctx, profile, plan, inputs, source,
                                accept_tier_change)
    return runner.submit("tier-apply", f"auto-place {name}", fn,
                         payload={"name": name,
                                  "accept_tier_change": accept_tier_change},
                         lane="mutation")


def submit_placement_apply(runner, name, selection, accept_tier_change=False):
    """Apply the placement the operator chose: which devices this model may
    use, and how much of each.

    The same job as submit_tier_apply, planned through the selection
    instead of through the machine as it stands, plus one addition: the
    selection is recorded on the profile. Reopening the placement screen
    has to show what is set to run, and reconstructing that from the
    emitted -ot rules would be a second reader of placement beside the
    planner -- the failure this whole path exists to avoid.

    The recorded PLANNING inputs stay unfiltered on purpose. They are the
    machine snapshot; writing the operator's ceilings into them would make
    every later automatic replan believe it is running on a smaller
    computer than it is.
    """
    import modelctl_plans

    def fn(ctx):
        profile = modelctl.load_profile(name)
        inputs, source = modelctl.resolve_planning_inputs(profile)
        plan = modelctl_plans.plan_for_selection(profile, selection,
                                                 inputs=inputs)
        if plan is None:
            raise RuntimeError(f"couldn't analyze model layout for '{name}'")
        profile.setdefault("planning", {})["selection"] = dict(selection or {})
        return _apply_tier_plan(ctx, profile, plan, inputs, source,
                                accept_tier_change)
    return runner.submit("placement-apply", f"place {name}", fn,
                         payload={"name": name, "selection": selection,
                                  "accept_tier_change": accept_tier_change},
                         lane="mutation")


def _cleanup_partial_downloads(ctx, repo_id, filenames):
    """Remove the half-written artifacts of a failed or cancelled pull.

    Only files THIS pull reported downloading are touched. A file whose
    size matches the repo's copy completed before the failure and is
    kept; when the remote size is unreachable the file is kept too --
    deleting a possibly-complete multi-GB download over an offline
    hiccup is worse than leaving a partial for the next pull's size
    check to catch. hf_hub_download's .incomplete spool files are always
    safe to drop: nothing is ever served from them, they only hold disk.
    """
    repo_dir = Path(modelctl.DEFAULT_MODELS_DIR) / repo_id
    for filename in filenames:
        spool_dir = (repo_dir / ".cache" / "huggingface" / "download"
                     / filename).parent
        for spool in spool_dir.glob(Path(filename).name + ".*.incomplete"):
            try:
                spool.unlink()
                ctx.log(f"removed download spool file {spool.name}")
            except OSError:
                pass
        target = repo_dir / filename
        try:
            if not target.exists():
                continue
            size = target.stat().st_size
        except OSError:
            continue
        expected = modelctl._remote_file_size(repo_id, filename)
        if expected is None:
            ctx.log(f"kept {filename}: its repo size can't be checked "
                    "right now, so it may be complete")
            continue
        if size == expected:
            ctx.log(f"kept {filename}: it finished downloading before "
                    "the failure")
            continue
        try:
            target.unlink()
            ctx.log(f"removed partial download {filename} "
                    f"({modelctl._format_size(size)} of "
                    f"{modelctl._format_size(expected)})")
        except OSError as e:
            ctx.log(f"couldn't remove partial download {filename}: {e}")


def _run_pull_subprocess(ctx, repo_id, quant_label=None, want_mtp=True,
                         current=None):
    """Run `modelctl pull --yes` as a killable child process.

    The old in-process hf_hub_download could not be interrupted: cancel
    marked the job cancelled while the download kept writing to disk and
    the profile still appeared. A registered process group dies with the
    cancel, mid-file, for real.

    `current` is the SAME dict submit_pull's byte-progress poller reads;
    writing the active filename here is what makes the progress bar move.
    """
    import os
    argv = [sys.executable, str(Path(modelctl.__file__).resolve()), "pull",
            repo_id, "--yes"]
    if quant_label:
        argv += ["--quant", quant_label]
    if not want_mtp:
        argv += ["--no-mtp"]
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, start_new_session=True,
        cwd=str(Path(modelctl.__file__).resolve().parent),
        # hf_hub_download's tqdm bar would arrive as hundreds of \r-lines
        # of job-log spam; the CLI's own per-file lines carry the progress.
        #
        # PYTHONUNBUFFERED because those per-file lines are the operator's
        # only live view of a running pull, and a plain CPython child
        # writing to a pipe block-buffers them: every line arrived at once
        # when the process exited, describing downloads that had already
        # finished. It also makes current["file"] land at file start
        # instead of at the end, which is what labels the progress bar.
        env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1",
             "PYTHONUNBUFFERED": "1"})
    ctx.register_process(proc)
    profile_name = None
    attempted = []
    if current is None:
        current = {"file": None}
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            ctx.log(line)
            m = re.search(r"downloading (\S+) ->", line)
            if m:
                current["file"] = m.group(1)
                attempted.append(m.group(1))
                ctx.set_progress(detail=m.group(1))
            m = re.search(r"Profile '([^']+)' created", line)
            if m:
                profile_name = m.group(1)
        rc = proc.wait(timeout=60)
    except BaseException:
        # Cancel usually surfaces as EOF + nonzero exit below, but a
        # reader/wait error must clean up the same way.
        _cleanup_partial_downloads(ctx, repo_id, attempted)
        raise
    if rc != 0:
        _cleanup_partial_downloads(ctx, repo_id, attempted)
        raise RuntimeError(f"pull of {repo_id} failed (exit {rc}) -- see job log")
    if not profile_name:
        # Belt and braces: the CLI's closing line is the contract, but if
        # output was cut short, the newest profile for this repo is it.
        newest = None
        for pf in modelctl.PROFILES_DIR.glob("*.json"):
            try:
                data = json.loads(pf.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("repo_id") == repo_id:
                stamp = pf.stat().st_mtime
                if newest is None or stamp > newest[0]:
                    newest = (stamp, data.get("name"))
        if newest:
            profile_name = newest[1]
    if not profile_name:
        raise RuntimeError(f"pull of {repo_id} finished but no profile was "
                           "found for it -- see job log")
    return {"profile": profile_name}


def submit_pull(runner, repo_id, quant_label=None, want_mtp=True,
                needed_bytes=0, needed_files=None):
    """Pull as a background job: disk preflight, live byte progress, and
    a download that cancel can actually stop.

    needed_bytes/needed_files describe the chosen quant and come from
    repo metadata at submit time. Both are absent when that lookup failed
    -- the pull still runs, it just cannot draw a bar, and says so.
    """
    def fn(ctx):
        space = acquisition_service.check_destination_space(needed_bytes)
        for m in space.messages:
            ctx.log(m)
        if not space.ok:
            raise RuntimeError(space.messages[0] if space.messages
                               else "not enough disk space for this download")

        # The subprocess reader writes the active filename here; the
        # poller reads it for the detail line only. The fraction comes
        # from bytes on disk, so a filename that has not arrived yet no
        # longer freezes the bar at 0 for the whole download.
        current = {"file": None}

        def job_still_running():
            job = ctx.store.get(ctx.job_id)
            return bool(job) and job["status"] == "running"

        poller = pull_progress.PullProgressPoller(
            repo_id, needed_files, needed_bytes,
            on_progress=ctx.set_progress,
            should_continue=job_still_running,
            label_fn=lambda: current["file"],
            on_error=lambda msg: ctx.log(
                f"progress reporting stopped ({msg}); the download itself "
                "is unaffected"))
        if not poller.can_measure:
            ctx.log("this download's size is unknown, so the progress bar "
                    "will not move -- the per-file lines below are the "
                    "progress")
        poller.start()
        try:
            ctx.raise_if_cancelled()
            out = _run_pull_subprocess(ctx, repo_id, quant_label=quant_label,
                                       want_mtp=want_mtp, current=current)
        finally:
            # Stop on the way out of every path, and wait: a cancelled or
            # failed pull otherwise leaves a thread still writing
            # progress onto a job that has already finished.
            poller.shutdown()
        current["file"] = None
        ctx.log(f"profile '{out['profile']}' saved and synced")
        return {"profile": out["profile"]}
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
        if not res["ok"]:
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
        ctx.log(f"placement mode for '{name}': " + json.dumps(runtime))
        modelctl.update_runtime_policy(name, runtime)
        ctx.log("profile saved, artifacts regenerated, synced")
        return {"name": name, "runtime": runtime}
    return runner.submit("mutation", f"placement mode {name}", fn,
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


def submit_plan_enable(runner, name, plan_id):
    """Undo a disable: put a plan back in the ranked candidate set.

    select/disable had submit_* helpers and enable did not -- it lived
    inline in the old console's route and went with it in the phase-3
    demolition. Same service call, same lane, now reachable from the one
    place plan actions are submitted."""
    from modelctl_services import plan_service

    def fn(ctx):
        result = plan_service.enable_plan(name, plan_id)
        if not result.ok:
            raise RuntimeError(result.messages[0] if result.messages
                              else f"enable failed for plan {plan_id}")
        ctx.log(f"re-enabled plan {plan_id}")
        return {"enabled": plan_id}
    return runner.submit("mutation", f"enable plan {plan_id} on {name}", fn,
                         payload={"name": name, "plan_id": plan_id})


def submit_remove(runner, name):
    """Delete a profile and its artifacts, then resync the backends.

    cmd_remove, not profile_service.remove_profile: the service function
    calls modelctl.remove_profile, which does not exist, so it raises
    AttributeError on every call. cmd_remove is what the old console
    deleted through and what `modelctl remove` still runs."""
    def fn(ctx):
        ctx.log(f"removing profile '{name}' and its artifacts")
        modelctl.cmd_remove(type("A", (), {"name": name, "no_hermes": True,
                                           "no_router_restart": False})())
        ctx.log("backends resynced")
        return {"removed": name}
    return runner.submit("remove", f"remove {name}", fn,
                         payload={"name": name}, lane="mutation")


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
                         payload={"name": name, "plan_id": plan_id},
                         lane="benchmark")


def submit_autotune(runner, name, objective="balanced", candidate_ids=None):
    def fn(ctx):
        import modelctl_tune
        res = modelctl_tune.autotune_profile(
            name, objective=objective, candidate_ids=candidate_ids,
            log=ctx.log, proc_register=ctx.register_process,
            cancel_check=ctx.is_cancelled)
        return res
    # lane="benchmark", not the default change queue: a multi-minute
    # autotune launches real servers, and on the serialized change queue
    # it blocked every profile save, config edit and sync for its whole
    # duration -- the head-of-line blocking the lanes exist to prevent.
    return runner.submit("benchmark", f"autotune {name}", fn,
                         payload={"name": name, "objective": objective},
                         lane="benchmark")


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


def submit_fleet_budget(runner, node, device, budget_bytes):
    """Set one fleet device's operator budget.

    Thin on purpose: every rule about what a budget may be -- the
    ceiling, the lock, which stored-input plans the change stales --
    lives in modelctl_fleet.set_device_budget, which is also what
    `modelctl fleet set-budget` calls. A second copy of the ceiling
    check on this side is a second copy that can drift from the one that
    actually guards the write.

    The route ahead of this refuses an over-ceiling budget synchronously
    so the operator gets the number to ask for instead of a failed job;
    the setter re-checks it under the lock, because between that check
    and this write another writer may have lowered the cap.
    """
    def fn(ctx):
        import modelctl_fleet
        result = modelctl_fleet.set_device_budget(node, device, budget_bytes)
        gib = budget_bytes / (1 << 30)
        if result["changed"]:
            ctx.log(f"{node}/{device} budget -> {gib:.2f} GiB "
                    f"(ceiling {result['ceiling_bytes'] / (1 << 30):.2f} GiB, "
                    f"{result['ceiling_basis']})")
        else:
            ctx.log(f"{node}/{device} budget already {gib:.2f} GiB; "
                    f"nothing written")
        for p in result["staled_profiles"]:
            ctx.log(f"stale machine snapshot: {p['name']} -- its stored plan "
                    f"was built against the old budget; replan to use the new "
                    f"one")
        if not result["staled_profiles"]:
            ctx.log("no stored-input plans depended on the old budget")
        return result
    return runner.submit("fleet-budget", f"budget {node}/{device}", fn,
                         payload={"node": node, "device": device,
                                  "budget_bytes": int(budget_bytes)},
                         lane="mutation")


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
        reloaded = _sync_backends(ctx)
        ctx.log("saved, artifacts regenerated, synced")
        return {"name": name, "moe_cache": moe_cache,
                "router_reloaded": reloaded}
    return runner.submit("moe-cache", f"moe cache {name}", fn,
                         payload={"name": name, "moe_cache": moe_cache},
                         lane="mutation")
