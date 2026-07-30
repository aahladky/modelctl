"""Managed worker for modelctl.

This is the entry point that llama-swap invokes for managed profiles:

    modelctl _worker PROFILE_NAME --port PORT

The worker:
1. Loads the profile and captures a hardware snapshot.
2. Compiles and ranks launch plans.
3. Iterates through feasible plans:
   a. Acquires a resource reservation.
   b. Launches the backend on the assigned port.
   c. Polls readiness.
   d. Marks reservation active.
   e. Supervises until exit.
4. Falls back to the next plan on failure.
5. Cleans up reservations on exit.

Signal forwarding: SIGTERM/SIGINT are forwarded to the child process
group so that llama-swap unload requests terminate the backend cleanly.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import modelctl
import modelctl_hardware
import modelctl_plans
import modelctl_runtime
import modelctl_vram


def _readiness_url(port):
    return f"http://127.0.0.1:{port}/health"


def _wait_ready(url, timeout=120):
    """Poll the backend readiness URL until 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(1)
    return False


def _build_command(profile, plan, port, binary=None):
    """Build the backend command: resolved binary + plan argv, with the
    assigned port injected. plan.argv comes from build_server_args, which
    deliberately starts at --model (no binary -- that's preflight's job).
    `binary` skips preflight (tests, or a caller that already resolved it)."""
    if binary is None:
        ok, effective_bin, _env, messages = modelctl.preflight(profile)
        if not ok:
            raise RuntimeError(f"preflight failed: {'; '.join(messages)}")
        binary = effective_bin
    argv = [binary] + list(plan.argv)
    # Replace or add --port
    new_argv = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--port" and i + 1 < len(argv):
            new_argv.extend(["--port", str(port)])
            skip_next = True
        elif arg.startswith("--port="):
            new_argv.append(f"--port={port}")
        else:
            new_argv.append(arg)
    # If --port wasn't in the plan args, add it
    if "--port" not in " ".join(new_argv):
        new_argv.extend(["--port", str(port)])
    return new_argv


def _forward_signal(proc, signum):
    """Forward a signal to the child process group."""
    try:
        if hasattr(os, "killpg") and hasattr(proc, "pgid"):
            os.killpg(proc.pgid, signum)
        elif proc.poll() is None:
            proc.send_signal(signum)
    except (ProcessLookupError, OSError):
        pass


def worker_main(profile_name, port):
    """Main worker loop. Returns exit code."""
    profile = modelctl.load_profile(profile_name)
    import modelctl_backends
    try:
        backend = modelctl_backends.get_backend(profile.get("backend", "llama-cpp"))
    except modelctl_backends.BackendError as e:
        print(f"modelctl-worker: {e} -- set the profile back to fixed mode.",
              file=sys.stderr)
        return 1
    policy = modelctl_plans.DEFAULT_POLICY
    # Load policy overrides from profile if present
    rt = profile.get("runtime", {})
    if rt:
        policy = modelctl_plans.RuntimePolicy(
            objective=rt.get("objective", policy.objective),
            pinned_plan_id=rt.get("pinned_plan_id"),
            allow_fallback=rt.get("allow_fallback", policy.allow_fallback),
            allow_untested=rt.get("allow_untested", policy.allow_untested),
            minimum_context=rt.get("minimum_context", policy.minimum_context),
            maximum_cpu_bytes=rt.get("maximum_cpu_bytes", policy.maximum_cpu_bytes),
            maximum_storage_tier=rt.get("maximum_storage_tier", policy.maximum_storage_tier),
        )

    snap = modelctl_hardware.capture_hardware_snapshot()
    plans = modelctl_plans.compile_launch_plans(profile, snap)
    rdb = modelctl_runtime.RuntimeDB()
    observations = rdb.observations_for_profile(
        profile_name, snap.fingerprint,
        snap.backend_fingerprints.get(profile.get("backend", "llama-cpp"), ""))
    failures = rdb.failures_for_profile(profile_name)
    ranked = modelctl_plans.rank_plans(plans, policy, observations, failures)

    disabled = set(rt.get("disabled_plan_ids", []))
    if disabled:
        ranked = [(p, s) for p, s in ranked if p.id not in disabled]

    # Live feasibility: drop plans whose claim doesn't fit current free
    # resources (minus configured reserves and other workers' pending
    # claims). Admission itself is re-checked atomically at reservation time.
    rdb = modelctl_runtime.RuntimeDB()
    budgets = {g.device: max(0, g.free_bytes - g.reserve_bytes)
               for g in modelctl_hardware.enabled_gpus(snap)}
    budgets["RAM"] = max(0, snap.ram_available_bytes - snap.ram_reserve_bytes)
    pending_by_res = {}
    for cl in rdb.pending_claims(exclude_pid=os.getpid()):
        for dev, b in cl.get("vram_bytes", {}).items():
            pending_by_res[dev] = pending_by_res.get(dev, 0) + b
        pending_by_res["RAM"] = pending_by_res.get("RAM", 0) + cl.get("ram_bytes", 0)

    def _effective(res):
        return budgets.get(res, 0) - pending_by_res.get(res, 0)

    feasible = []
    for plan, score in ranked:
        fits = all(b <= _effective(dev) for dev, b in plan.claim.vram_bytes.items()) \
               and plan.claim.ram_bytes <= _effective("RAM")
        if fits:
            feasible.append((plan, score))
        else:
            rdb.record_event("plan_infeasible", profile_name,
                             {"plan_id": plan.id,
                              "claim": {**{d: b for d, b in plan.claim.vram_bytes.items()},
                                        "RAM": plan.claim.ram_bytes},
                              "effective": {r: _effective(r) for r in budgets}})
    ranked = feasible

    if not ranked:
        print(f"modelctl-worker: no feasible plans for '{profile_name}'", file=sys.stderr)
        return 1

    # Pinned plan goes first; alternatives stay as fallback unless the
    # policy explicitly disables fallback.
    if policy.pinned_plan_id and not policy.allow_fallback:
        ranked = [(p, s) for p, s in ranked if p.id == policy.pinned_plan_id]
        if not ranked:
            print(f"modelctl-worker: pinned plan {policy.pinned_plan_id} not feasible "
                  f"and fallback disabled", file=sys.stderr)
            return 1

    rdb.record_event("plan_generation_started", profile_name,
                     {"plans": len(plans), "ranked": len(ranked)})

    my_pid = os.getpid()

    for plan, score in ranked:
        claim_dict = {
            "vram_bytes": dict(plan.claim.vram_bytes),
            "ram_bytes": plan.claim.ram_bytes,
            "storage_mode": plan.claim.storage_mode,
        }

        rdb.record_event("plan_selected", profile_name,
                         {"plan_id": plan.id, "label": plan.label,
                          "source": plan.source, "score": score})

        # Re-probe free memory immediately before admission: another worker
        # may have gone ACTIVE since our startup snapshot, and active
        # allocations don't show up in pending claims (they're visible only
        # in the driver's free-memory report).
        try:
            fresh = {d["device"]: d["free_bytes"]
                     for d in modelctl.get_gpu_inventory()}
            for dev in budgets:
                if dev == "RAM":
                    budgets[dev] = max(0, modelctl_vram.system_ram_available()
                                       - snap.ram_reserve_bytes)
                elif dev in fresh:
                    reserve = next((g.reserve_bytes for g in snap.gpus
                                    if g.device == dev), 0)
                    budgets[dev] = max(0, fresh[dev] - reserve)
        except Exception:
            pass  # keep the startup-snapshot budgets on probe failure
        reservation = rdb.acquire_reservation(
            profile_name, plan.id, claim_dict, my_pid, budgets=budgets)
        if reservation is None:
            rdb.record_event("reservation_conflict", profile_name,
                             {"plan_id": plan.id})
            if not policy.allow_fallback:
                return 1
            continue

        rdb.record_event("reservation_acquired", profile_name,
                         {"plan_id": plan.id, "reservation_id": reservation["id"]})

        run = {"profile_name": profile_name, "plan_id": plan.id,
               "hardware_fingerprint": snap.fingerprint,
               "backend_fingerprint": snap.backend_fingerprints.get(
                   profile.get("backend", "llama-cpp"), ""),
               "started_at": time.time(), "success": False}
        try:
            cmd = backend.build_command(profile, plan, port)

            # Provenance: record the command and backend details so
            # the history UI can explain "why this command?"
            import modelctl_capabilities
            binary = profile.get("binary") or modelctl.LLAMA_SERVER_BIN
            caps = modelctl_capabilities.probe_backend(binary)
            run["command_argv"] = cmd
            run["binary_path"] = binary
            run["binary_fingerprint"] = modelctl_vram.file_fingerprint(binary)
            run["capability_schema"] = caps.get("schema", 0)
            run["claim"] = {"vram_bytes": dict(plan.claim.vram_bytes),
                            "ram_bytes": plan.claim.ram_bytes,
                            "storage_mode": plan.claim.storage_mode}
            run["decision"] = plan.decision_data or {}
            print(f"modelctl-worker: launching plan '{plan.label}' "
                  f"(id={plan.id}) on port {port}", file=sys.stderr)
            print(f"modelctl-worker: cmd: {' '.join(cmd)}", file=sys.stderr)

            # Launch in a new process group for clean signal forwarding
            preexec = os.setpgrp if hasattr(os, "setpgrp") else None
            child_env = backend.effective_environment(profile, plan)
            proc = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=child_env,
                preexec_fn=preexec,
            )

            if hasattr(proc, "pid") and hasattr(os, "setpgrp"):
                try:
                    proc.pgid = os.getpgid(proc.pid)
                except OSError:
                    proc.pgid = proc.pid

            rdb.update_reservation(reservation["id"], state="starting")
            rdb.record_event("backend_started", profile_name,
                             {"plan_id": plan.id, "pid": proc.pid})

            # Set up signal forwarding
            def _handle_term(signum, frame):
                _forward_signal(proc, signum)

            prev_term = signal.getsignal(signal.SIGTERM)
            prev_int = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGTERM, _handle_term)
            signal.signal(signal.SIGINT, _handle_term)

            try:
                # Wait for readiness
                ready_timeout = rt.get("health_timeout", 120)
                ready_t0 = time.time()
                if _wait_ready(backend.readiness_url(profile, port), timeout=ready_timeout):
                    run["load_seconds"] = round(time.time() - ready_t0, 2)
                    run["success"] = True
                    rdb.update_reservation(reservation["id"], state="active")
                    rdb.record_event("backend_ready", profile_name,
                                     {"plan_id": plan.id, "pid": proc.pid})

                    # Supervise until child exits
                    exit_code = proc.wait()
                    run["exit_code"] = exit_code

                    rdb.record_event("model_unloaded", profile_name,
                                     {"plan_id": plan.id, "exit_code": exit_code})
                    return exit_code
                else:
                    # Health timeout
                    run["failure_class"] = "health_timeout"
                    run["exit_code"] = proc.poll()
                    _forward_proc_terminate(proc)
                    proc.wait(timeout=10)
                    rdb.record_event("backend_failed", profile_name,
                                     {"plan_id": plan.id, "reason": "health_timeout"})
                    print(f"modelctl-worker: plan '{plan.label}' failed: "
                          "health timeout", file=sys.stderr)
            finally:
                signal.signal(signal.SIGTERM, prev_term)
                signal.signal(signal.SIGINT, prev_int)
                rdb.release_reservation(reservation["id"])
                run["finished_at"] = time.time()
                rdb.record_plan_run(run)
                run["_recorded"] = True

        except Exception as e:
            run["failure_class"] = run.get("failure_class") or "backend_crash"
            if not run.get("_recorded"):
                run["finished_at"] = time.time()
                rdb.release_reservation(reservation["id"])
                rdb.record_plan_run(run)
            rdb.record_event("backend_failed", profile_name,
                             {"plan_id": plan.id, "error": str(e)})
            print(f"modelctl-worker: plan '{plan.label}' error: {e}",
                  file=sys.stderr)

        if not policy.allow_fallback:
            return 1

    print(f"modelctl-worker: all plans exhausted for '{profile_name}'",
          file=sys.stderr)
    return 1


def _forward_proc_terminate(proc):
    """Terminate a process group, with escalation to SIGKILL."""
    try:
        if hasattr(os, "killpg") and hasattr(proc, "pgid"):
            os.killpg(proc.pgid, signal.SIGTERM)
        elif proc.poll() is None:
            proc.terminate()
    except (ProcessLookupError, OSError):
        return
    deadline = time.time() + 5
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        if hasattr(os, "killpg") and hasattr(proc, "pgid"):
            os.killpg(proc.pgid, signal.SIGKILL)
        elif proc.poll() is None:
            proc.kill()
    except (ProcessLookupError, OSError):
        pass


def cmd_worker(args):
    """CLI entry point for `modelctl _worker`."""
    sys.exit(worker_main(args.profile_name, args.port))


def register_subcommand(subparsers):
    """Register the _worker subcommand (hidden from help)."""
    p = subparsers.add_parser("_worker", add_help=False)
    p.add_argument("profile_name")
    p.add_argument("--port", type=int, required=True)
    p.set_defaults(func=cmd_worker)
