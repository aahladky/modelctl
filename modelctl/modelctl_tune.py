"""Plan testing and autotuning (Milestone 6).

test_launch_plan: launch one plan on a temporary port, measure load time,
peak resources, and generation throughput, and persist the PlanRun.
autotune_profile: run test_launch_plan over a bounded candidate set, skipping
plans already validated under the current hardware/backend fingerprints.

Measured results feed rank_plans via RuntimeDB.observations_for_profile --
theory proposes, measurement disposes.
"""
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import modelctl
import modelctl_hardware
import modelctl_plans
import modelctl_runtime
import modelctl_vram

_TEST_PROMPT = ("Summarize why the sky is blue in three sentences.")
_WARMUP_PROMPT = ("Write a short greeting.")


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _post_json(url, body, timeout):
    import json
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_stream(url, body, timeout):
    """Streaming request returning (ttft_seconds, total_seconds, usage_dict).
    TTFT = time to the first SSE data chunk, not the full response."""
    import json
    req = urllib.request.Request(
        url, data=json.dumps({**body, "stream": True}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if ttft is None and raw.startswith(b"data:") and b"[DONE]" not in raw:
                ttft = time.time() - t0
            if raw.startswith(b"data:") and b'"usage"' in raw:
                try:
                    usage = json.loads(raw[5:].strip()).get("usage", {})
                except Exception:
                    pass
    return ttft, time.time() - t0, usage


class _Sampler:
    """1 Hz peak sampler: per-device VRAM free, child RSS, system RAM available."""

    def __init__(self, pid):
        self.pid = pid
        self.peak_vram = {}
        self.peak_ram = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self):
        baseline = {}
        try:
            for d in modelctl.get_gpu_inventory():
                baseline[d["device"]] = d["total_bytes"] - d["free_bytes"]
        except Exception:
            pass
        self.baseline_vram = baseline
        while not self._stop.is_set():
            try:
                for d in modelctl.get_gpu_inventory():
                    used = (d["total_bytes"] - d["free_bytes"]) - baseline.get(d["device"], 0)
                    self.peak_vram[d["device"]] = max(
                        self.peak_vram.get(d["device"], 0), used)
                rss = self._child_rss()
                self.peak_ram = max(self.peak_ram, rss)
            except Exception:
                pass
            self._stop.wait(1.0)

    def _child_rss(self):
        total = 0
        try:
            for f in os.listdir("/proc"):
                if not f.isdigit():
                    continue
                try:
                    with open(f"/proc/{f}/stat") as fh:
                        if int(fh.read().rsplit(")", 1)[-1].split()[2]) != self.pid:
                            continue
                    with open(f"/proc/{f}/status") as fh:
                        for line in fh:
                            if line.startswith("VmRSS:"):
                                total += int(line.split()[1]) * 1024
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            pass
        return total


def _wait_ready(port, proc, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, time.time() - (deadline - timeout)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True, time.time() - (deadline - timeout)
        except Exception:
            pass
        time.sleep(2)
    return False, time.time() - (deadline - timeout)


def test_launch_plan(profile_name, plan_id, log=print, prompt=None,
                     max_tokens=128, runs=2, binary=None,
                     proc_register=None, cancel_check=None,
                     warmup_tokens=0):
    """Measure one launch plan and persist the resulting PlanRun dict.

    When warmup_tokens > 0, a warmup generation runs first to fill the
    expert cache and OS page cache.  The measured runs then capture
    warm-cache performance.  The run dict includes:
      cache_state: "cold" (no warmup) or "warm" (after warmup)
      warmup_generation_tps: speed during the warmup phase (if any)
    """
    profile = modelctl.load_profile(profile_name)
    snap = modelctl_hardware.capture_hardware_snapshot()
    plans = modelctl_plans.compile_launch_plans(profile, snap)
    plan = next((p for p in plans if p.id == plan_id), None)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found for '{profile_name}'")

    rdb = modelctl_runtime.RuntimeDB()
    started = time.time()
    run = {"profile_name": profile_name, "plan_id": plan_id,
           "hardware_fingerprint": snap.fingerprint,
           "backend_fingerprint": snap.backend_fingerprints.get(
               profile.get("backend", "llama-cpp"), ""),
           "started_at": started, "success": False, "log_path": ""}

    claim = {"vram_bytes": dict(plan.claim.vram_bytes),
             "ram_bytes": plan.claim.ram_bytes,
             "storage_mode": plan.claim.storage_mode}
    budgets = {g.device: max(0, g.free_bytes - g.reserve_bytes)
               for g in modelctl_hardware.enabled_gpus(snap)}
    budgets["RAM"] = max(0, snap.ram_available_bytes - snap.ram_reserve_bytes)
    reservation = rdb.acquire_reservation(profile_name, plan_id, claim, os.getpid(),
                                          budgets=budgets)
    if reservation is None:
        run["failure_class"] = "reservation_conflict"
        run["finished_at"] = time.time()
        rdb.record_plan_run(run)
        raise RuntimeError("reservation conflict -- another worker holds the resources")

    port = _free_port()
    proc = None
    try:
        cmd = modelctl_worker._build_command(profile, plan, port, binary=binary)
        log(f"launching: {' '.join(cmd[:6])} ...")
        log_path = modelctl.PROFILES_DIR / profile_name / f"plan-test-{plan_id[:8]}.log"
        run["log_path"] = str(log_path)
        child_env = dict(os.environ)
        child_env.update(plan.env or {})
        logf = open(log_path, "w")
        preexec = os.setpgrp if hasattr(os, "setpgrp") else None
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                env=child_env, preexec_fn=preexec)
        if proc_register:
            proc_register(proc)

        with _Sampler(proc.pid) as sampler:
            ready, load_s = _wait_ready(port, proc)
            if not ready:
                run["failure_class"] = "health_timeout" if proc.poll() is None else "backend_crash"
                run["exit_code"] = proc.poll()
                log(f"backend failed to become ready (class={run['failure_class']})")
                return run

            run["load_seconds"] = round(load_s, 2)
            log(f"ready in {load_s:.1f}s, benchmarking {runs}x{max_tokens} tok ...")

            actual_ctx = None
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5) as r:
                    import json as _json
                    props = _json.loads(r.read())
                actual_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            except Exception:
                pass

            # Warmup phase: fill expert cache and OS page cache before
            # measuring.  When warmup_tokens is 0, the run is "cold".
            run["cache_state"] = "cold"
            run["warmup_generation_tps"] = None
            if warmup_tokens and warmup_tokens > 0:
                log(f"warmup: {warmup_tokens} tokens ...")
                warmup_resp = _post_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": _WARMUP_PROMPT}],
                     "max_tokens": warmup_tokens, "temperature": 0},
                    timeout=600)
                wt = warmup_resp.get("timings", {})
                if wt.get("predicted_per_second"):
                    run["warmup_generation_tps"] = round(wt["predicted_per_second"], 2)
                    log(f"warmup gen {run['warmup_generation_tps']} tok/s")
                run["cache_state"] = "warm"

            prompt_tps, gen_tps, ttfts = [], [], []
            for i in range(runs):
                if cancel_check and cancel_check():
                    run["failure_class"] = "cancelled"
                    raise InterruptedError("plan test cancelled")
                ttft, _total, _usage = _post_stream(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": prompt or _TEST_PROMPT}],
                     "max_tokens": max_tokens, "temperature": 0},
                    timeout=600)
                if ttft:
                    ttfts.append(ttft)
                resp = _post_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": prompt or _TEST_PROMPT}],
                     "max_tokens": max_tokens, "temperature": 0},
                    timeout=600)
                t = resp.get("timings", {})
                if t.get("prompt_per_second"):
                    prompt_tps.append(t["prompt_per_second"])
                if t.get("predicted_per_second"):
                    gen_tps.append(t["predicted_per_second"])

            run["success"] = True
            run["actual_context"] = actual_ctx or plan.claim.expected_context
            run["ttft_seconds"] = round(min(ttfts), 3) if ttfts else None
            run["prompt_tps"] = round(sum(prompt_tps) / len(prompt_tps), 2) if prompt_tps else None
            run["generation_tps"] = round(sum(gen_tps) / len(gen_tps), 2) if gen_tps else None
            log(f"gen {run['generation_tps']} tok/s, prompt {run['prompt_tps']} tok/s")

        run["peak_vram_bytes"] = sampler.peak_vram
        run["peak_ram_bytes"] = sampler.peak_ram
        return run
    except Exception as e:
        run["failure_class"] = "unknown"
        run["details"] = {"error": str(e)}
        raise
    finally:
        run["finished_at"] = time.time()
        if proc is not None:
            try:
                pg = os.getpgid(proc.pid)
                os.killpg(pg, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        rdb.release_reservation(reservation["id"])
        rdb.record_plan_run(run)


def autotune_profile(profile_name, objective="balanced", candidate_ids=None,
                     log=print, retest=False, max_tokens=128, runs=2, binary=None,
                     proc_register=None, cancel_check=None):
    """Test a bounded set of candidate plans and report measured ranking.
    Never changes the profile or selects a winner automatically."""
    profile = modelctl.load_profile(profile_name)
    snap = modelctl_hardware.capture_hardware_snapshot()
    plans = modelctl_plans.compile_launch_plans(profile, snap)
    rdb = modelctl_runtime.RuntimeDB()
    existing = rdb.observations_for_profile(
        profile_name, snap.fingerprint,
        snap.backend_fingerprints.get(profile.get("backend", "llama-cpp"), ""))

    candidates = []
    for p in plans:
        if cancel_check and cancel_check():
            raise InterruptedError("autotune cancelled")
        if candidate_ids and p.id not in candidate_ids:
            continue
        obs = existing.get(p.id)
        if obs and not obs.get("stale") and not retest:
            log(f"skip {p.label} -- already validated on this hardware "
                f"({obs['generation_tps']} tok/s)")
            continue
        candidates.append(p)

    results = []
    for p in candidates:
        log(f"testing plan {p.label} ...")
        try:
            run = test_launch_plan(profile_name, p.id, log=log,
                                   max_tokens=max_tokens, runs=runs, binary=binary,
                                   proc_register=proc_register, cancel_check=cancel_check)
        except Exception as e:
            log(f"plan {p.label} failed: {e}")
            continue
        results.append(run)

    policy = modelctl_plans.DEFAULT_POLICY
    scored = sorted(
        (r for r in results if r.get("success")),
        key=lambda r: -(r.get("generation_tps") or 0))
    return {"tested": len(results), "skipped_validated": len(plans) - len(candidates),
            "results": results, "best": scored[0]["plan_id"] if scored else None}


# Imported lazily by worker too; keep at end to avoid a cycle at module import.
import modelctl_worker  # noqa: E402
