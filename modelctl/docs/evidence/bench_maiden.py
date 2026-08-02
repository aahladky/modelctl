#!/usr/bin/env python3
"""Run the maiden night-lane jobs against the deployed build-sycl binary.

Drives the real measurement through `modelctl_paired.run_paired` and
`modelctl_nightlane.run_job`, so the alternation, the per-pair deltas,
the sign test and the per-run load traces are the ones under test in the
suite -- not a second implementation that happens to agree.

Commands are assembled as argv, not from a saved profile, because the
122B has never had one: every measurement of it in this tree (P1,
hybrid, determinism) was assembled the same way. The argv below is
copied out of the 2026-08-01 determinism record's own `argv` fields, so
this reproduces that protocol rather than re-deriving it.

Nothing here writes a profile, touches llama-swap, or restarts anything.
Each run gets a fresh server on a scratch port and is torn down by PID
with the port confirmed free afterwards.

Usage:
  bench_maiden.py --job <id> [--pairs N] [--runs N] [--out DIR]
  bench_maiden.py --smoke          # one C1 run, to prove the harness
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path("/home/aaron/workspace/moe-serving")
sys.path.insert(0, str(REPO / "modelctl"))

import modelctl_load        # noqa: E402
import modelctl_nightlane   # noqa: E402

BIN_DIR = REPO / "llama.cpp/build-sycl/bin"
SERVER = BIN_DIR / "llama-server"

# The 2026-08-01 record's model was Qwen3.5-122B-A10B-UD-IQ1_M. That file
# was deleted from ~/models at 01:32 on 2026-08-02, mid-battery -- the
# first twenty runs here used it, the rest could not. `--model` exists so
# the target is an argument rather than a constant: a benchmark harness
# that hardcodes a path cannot be repointed when the path stops existing.
MODEL_122B = ("/home/aaron/models/unsloth/Qwen3.5-122B-A10B-GGUF/"
              "Qwen3.5-122B-A10B-UD-IQ1_M.gguf")
MODEL = "/home/aaron/models/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
PORT = 18147

# Verbatim from the 2026-08-01 evidence record's protocol.prompt.
PROMPT = ("Explain, in plain terms, how a mixture-of-experts transformer "
          "decides which experts to activate for a given token, and why "
          "expert reuse across consecutive tokens matters for serving "
          "performance on machines whose expert weights stream from disk.")

WARMUP_TOKENS = 32
MEASURED_TOKENS = 128

def base_argv(model):
    return ["--model", model] + BASE_ARGV[2:]


BASE_ARGV = [
    "--model", MODEL, "--flash-attn", "auto", "--jinja", "--parallel", "1",
    "-ngl", "999", "-c", "4096", "--split-mode", "layer",
    "--tensor-split", "8,3", "--cache-type-k", "q8_0",
    "--cache-type-v", "q4_0", "--metrics", "--fit", "off",
    "--device", "SYCL0,SYCL1", "-ot", "ffn_.*_exps=CPU",
    "--no-warmup", "--ubatch-size", "128", "--port", str(PORT),
]
CACHE_ARGV = [
    "--moe-cache-bytes", "SYCL0=4294967296", "--moe-cache-policy", "slru",
    "--moe-cache-admission-misses", "2",
    "--moe-cache-prefill-admission", "off",
]

# name -> (extra argv, extra env). Determinism is the only thing that
# varies between the two arms of a paired job.
CONDITIONS = {
    "C1": ([], {}),
    "C2": (CACHE_ARGV, {"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "1"}),
    "C3": (CACHE_ARGV + ["--moe-hybrid-mode", "on"],
           {"GGML_OP_OFFLOAD_MOE_MIN_BATCH": "1"}),
}


# --- environment ---------------------------------------------------------

def server_env(extra):
    """The launch environment, with oneAPI reachable.

    A SYCL build without the oneAPI LD_LIBRARY_PATH SIGABRTs on this box
    (oneAPI 2026.1, since the Jul 26-27 upgrade), so this is not optional
    and must not be left to whatever the shell happens to have.
    """
    import modelctl
    env = dict(os.environ)
    env.pop("GGML_SYCL_DETERMINISTIC", None)
    modelctl.ensure_binary_ld_library_path(env, str(SERVER))
    for cand in modelctl._probe_env_candidates():
        if cand and cand.get("LD_LIBRARY_PATH"):
            env["LD_LIBRARY_PATH"] = (env.get("LD_LIBRARY_PATH", "") + ":"
                                      + cand["LD_LIBRARY_PATH"])
            break
    env.update({k: str(v) for k, v in extra.items()})
    # The floor does not move, whatever a caller passed.
    floor = env.get("GGML_OP_OFFLOAD_MIN_BATCH")
    if floor is not None and int(floor) < 32:
        raise SystemExit(f"refusing to run: GGML_OP_OFFLOAD_MIN_BATCH={floor}")
    return env


# --- server lifecycle ----------------------------------------------------

def port_free(port=PORT):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_port_free(port=PORT, seconds=300):
    for _ in range(seconds):
        if port_free(port):
            return True
        time.sleep(1)
    return False


def get(path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}",
                                timeout=timeout) as r:
        return r.read().decode()


# A 128-token decode takes seconds to a couple of minutes on any
# condition here. The original 3600 s meant a hung server was
# indistinguishable from a slow one for a full hour -- which is exactly
# what happened to C3 run 2 on 2026-08-02: it decoded 100 of 128 tokens,
# went idle, and burned 34 minutes of wall clock before anyone looked.
# A benchmark harness has to be able to tell "stuck" from "slow".
COMPLETION_TIMEOUT = 600


def post(path, body, timeout=COMPLETION_TIMEOUT):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def completion(n_predict, n_probs=0):
    body = {"prompt": PROMPT, "n_predict": n_predict, "temperature": 0.0,
            "top_k": 1, "seed": 42, "cache_prompt": False, "id_slot": 0,
            "return_tokens": True}
    if n_probs:
        body.update({"n_probs": n_probs, "post_sampling_probs": False})
    return post("/completion", body)


_METRIC = re.compile(r'^(\S+?)(\{.*\})?\s+([0-9.eE+-]+)$')


def scrape_metrics():
    try:
        text = get("/metrics", timeout=15)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = _METRIC.match(line.strip())
        if not m:
            continue
        key = m.group(1) + (m.group(2) or "")
        try:
            out[key] = float(m.group(3))
        except ValueError:
            pass
    return out


def parse_cache_geometry(log_text):
    """Effective cache budget = slots x per-slot bytes, from the server's
    own geometry line. The 2026-08-01 battery recorded only the REQUESTED
    --moe-cache-bytes, which is the gap this exists to close."""
    out = {}
    for line in log_text.splitlines():
        if "moe_cache" not in line:
            continue
        g = re.search(r"gate=(\d+)\s+up=(\d+)\s+down=(\d+)", line)
        if g:
            out["slot_gate_bytes"] = int(g.group(1))
            out["slot_up_bytes"] = int(g.group(2))
            out["slot_down_bytes"] = int(g.group(3))
            out["slot_bytes"] = sum(int(g.group(i)) for i in (1, 2, 3))
        s = re.search(r"(\d+)\s+slots", line)
        if s:
            out["slots"] = int(s.group(1))
    if "slots" in out and "slot_bytes" in out:
        out["effective_cache_budget_bytes"] = out["slots"] * out["slot_bytes"]
    return out


def one_run(condition, deterministic, log_path, label="", model=None):
    """One fresh-server measurement. Returns the metrics dict."""
    extra_argv, extra_env = CONDITIONS[condition]
    env = server_env({**extra_env, "GGML_SYCL_DETERMINISTIC": deterministic})
    is_cache = condition in ("C2", "C3")

    model = model or MODEL
    # Checked here, not left to the server: "the file is gone" and "the
    # server crashed" produce the same rc=1 and want different responses,
    # and confusing the two cost this battery fifteen runs.
    if not Path(model).exists():
        raise RuntimeError(f"model file does not exist: {model}")

    if not wait_port_free():
        raise RuntimeError(f"port {PORT} never became free")

    argv = [str(SERVER)] + base_argv(model) + extra_argv
    started = time.time()
    log = open(log_path, "w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            env=env, preexec_fn=os.setpgrp)
    result = {"condition": condition, "deterministic": deterministic,
              "label": label, "argv": argv,
              "env_overrides": {k: env[k] for k in
                                ("GGML_SYCL_DETERMINISTIC",
                                 "GGML_OP_OFFLOAD_MOE_MIN_BATCH")
                                if k in env},
              "started": started, "pid": proc.pid}
    try:
        # Health at 1 Hz -- liveness checked from the first second, so a
        # server that dies on launch is reported now, not in ten minutes.
        healthy = False
        for _ in range(1800):
            if proc.poll() is not None:
                log.flush()
                tail = Path(log_path).read_text()[-3000:]
                raise RuntimeError(
                    f"server exited rc={proc.returncode} during load; "
                    f"stderr tail:\n{tail}")
            try:
                if json.loads(get("/health", timeout=2)).get("status") == "ok":
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not healthy:
            raise RuntimeError("server never became healthy")
        result["load_seconds"] = time.time() - started

        warm = completion(WARMUP_TOKENS)
        result["warmup"] = {
            "tokens": warm.get("tokens"),
            "gen_tok_s": (warm.get("timings") or {}).get("predicted_per_second"),
        }

        if is_cache:
            m = scrape_metrics()
            learning = m.get('llamacpp:moe_cache_learning{device="SYCL0"}')
            misses = m.get('llamacpp:moe_cache_misses_total{device="SYCL0"}', 0)
            result["engagement"] = {"learning": learning, "misses_total": misses,
                                    "engaged": bool(learning == 0 and misses > 0)}
            try:
                post("/cache/reset", {}, timeout=60)
                result["cache_reset"] = "ok"
            except Exception as e:
                result["cache_reset"] = f"{type(e).__name__}: {e}"

        measured = completion(MEASURED_TOKENS, n_probs=8)
        timings = measured.get("timings") or {}
        result["measured"] = {
            "tokens": measured.get("tokens"),
            "predicted_n": timings.get("predicted_n"),
            "prompt_n": timings.get("prompt_n"),
            "stop_type": measured.get("stop_type"),
        }
        result["generation_tps"] = timings.get("predicted_per_second")
        result["prompt_tps"] = timings.get("prompt_per_second")

        after = scrape_metrics()
        result["metrics_after_measured"] = {
            k: v for k, v in after.items() if "moe_cache" in k}
        if is_cache:
            result["hit_ratio"] = after.get(
                'llamacpp:moe_cache_hit_ratio{device="SYCL0"}')
            result["slots_used"] = after.get(
                'llamacpp:moe_cache_slots_used{device="SYCL0"}')
            result["slots"] = after.get(
                'llamacpp:moe_cache_slots{device="SYCL0"}')
        return result
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=300)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait(timeout=120)
        log.close()
        result["exit_code"] = proc.returncode
        result["port_free_after"] = wait_port_free()
        text = Path(log_path).read_text()
        result.update(parse_cache_geometry(text))
        result["finished"] = time.time()


# --- the jobs ------------------------------------------------------------

def load_snapshot():
    s = modelctl_load.sample_load()
    return {"loadavg_1m": s.loadavg_1m, "loadavg_5m": s.loadavg_5m,
            "mem_available_bytes": s.mem_available_bytes}


def run_paired_job(job_id, condition, pairs, outdir, model):
    import modelctl_paired as mp
    a = mp.Condition(name="deterministic-on",
                     config={"GGML_SYCL_DETERMINISTIC": "1",
                             "condition": condition})
    b = mp.Condition(name="deterministic-off",
                     config={"GGML_SYCL_DETERMINISTIC": "0",
                             "condition": condition})

    def measure(cond, pair, slot):
        det = "1" if cond.name == "deterministic-on" else "0"
        label = f"{job_id}-p{pair}-s{slot}-{cond.name}"
        print(f"    [{time.strftime('%H:%M:%S')}] {label} "
              f"(load {load_snapshot()['loadavg_1m']})", flush=True)
        run = one_run(condition, det, outdir / f"{label}.log", label,
                      model=model)
        print(f"      -> gen {run['generation_tps']:.4f} tok/s, "
              f"load {run['load_seconds']:.1f}s", flush=True)
        return run

    comparison = mp.run_paired(a, b, measure, pairs=pairs,
                               metric="generation_tps", load_interval=5.0)
    return comparison.to_dict()


def run_battery(job_id, conditions, runs, outdir, model):
    out = {}
    for condition in conditions:
        entries = []
        for i in range(runs):
            label = f"{job_id}-{condition}-r{i}"
            print(f"    [{time.strftime('%H:%M:%S')}] {label} "
                  f"(load {load_snapshot()['loadavg_1m']})", flush=True)
            rec = modelctl_load.LoadRecorder(interval=5.0)
            entry = {"run": i}
            rec.start()
            try:
                entry["metrics"] = one_run(condition, "1",
                                           outdir / f"{label}.log", label,
                                           model=model)
                print(f"      -> gen {entry['metrics']['generation_tps']:.4f} "
                      f"tok/s", flush=True)
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
                print(f"      -> FAILED {entry['error'][:200]}", flush=True)
            finally:
                entry["load"] = rec.stop().summary()
            entries.append(entry)
        out[condition] = entries
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    outdir = Path(args.out or (REPO / "modelctl/docs/evidence/runs"))
    outdir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("smoke: one C1 run, deterministic on")
        run = one_run("C1", "1", outdir / "smoke.log", "smoke",
                      model=args.model)
        print(json.dumps({k: v for k, v in run.items()
                          if k not in ("argv", "metrics_after_measured",
                                       "measured", "warmup")}, indent=2))
        print("warmup tokens:", (run.get("warmup") or {}).get("tokens"))
        return 0

    started = load_snapshot()
    print(f"start load: {started}")
    if args.job in ("determinism-cost-c1-static-2026-08-02",
                    "determinism-cost-c2-cache-2026-08-02"):
        condition = "C1" if "c1-static" in args.job else "C2"
        record = run_paired_job(args.job, condition, args.pairs, outdir,
                                args.model)
    elif args.job == "re-anchor-c1-c2-c3-2026-08-02":
        record = run_battery(args.job, ("C1", "C2", "C3"), args.runs,
                             outdir, args.model)
    else:
        raise SystemExit(f"unknown job: {args.job}")

    payload = {"job": args.job, "load_at_start": started,
               "load_at_end": load_snapshot(),
               "binary": str(SERVER), "model": args.model, "record": record}
    path = outdir / f"{args.job}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
