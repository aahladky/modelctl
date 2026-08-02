#!/usr/bin/env python3
"""Laguna-S-2.1 fleet baseline: static placement, RPC arms, maiden protocol.

Every arm launches its OWN llama-server on a scratch port from argv copied
out of llama-swap's `laguna-s2.1` entry, and tears it down by PID with the
port confirmed free. Nothing here touches llama-swap, OVMS, the console,
remote-hands, any saved profile, or laguna's live entry.

Protocol is the maiden one, verbatim from bench_maiden.py: fixed prompt,
greedy, seed 42, cache_prompt=false, 32-token warmup, 128-token measured
decode.

Arms:
  REF   laguna's own pinned binary, live argv, no RPC. Same-protocol
        reference -- the registered 14.20 anchor was taken through
        llama-swap with speed.py at 256 tokens, so it is not directly
        comparable to a 128-token maiden run. This arm makes the control
        gate an apples-to-apples comparison.
  R1    the new RPC-enabled binary, live argv, NO --rpc flag.
  R2    R1 + rpc-cuda0 carrying the derived expert blocks.
  R3    R2 + rpc-cpu0 carrying the remaining expert blocks.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/aaron/workspace/moe-serving")
LANE = Path("/home/aaron/workspace/.lanes/fleet-laguna-baseline")
sys.path.insert(0, str(REPO / "modelctl"))
import modelctl_load  # noqa: E402

NEW_BIN = REPO / "llama.cpp/build-sycl-rpc/bin/llama-server"
PINNED_BIN = Path("/home/aaron/src/llama.cpp-laguna/build-sycl/bin/llama-server")
MODEL = ("/home/aaron/models/unsloth/Laguna-S-2.1-GGUF/UD-IQ4_NL/"
         "Laguna-S-2.1-UD-IQ4_NL-00001-of-00003.gguf")
PORT = 9501                       # inside the lane's 9500-9509 block

RIG_IFACE = "enp13s0"
LAPTOP = "192.168.0.76"
LAPTOP_IFACE = "enx6c6e072468a7"
EP_CUDA0 = f"{LAPTOP}:50052"
EP_CPU0 = f"{LAPTOP}:50053"

# Verbatim from bench_maiden.py / the 2026-08-01 evidence record.
PROMPT = ("Explain, in plain terms, how a mixture-of-experts transformer "
          "decides which experts to activate for a given token, and why "
          "expert reuse across consecutive tokens matters for serving "
          "performance on machines whose expert weights stream from disk.")
WARMUP_TOKENS = 32
MEASURED_TOKENS = 128
COMPLETION_TIMEOUT = 600
LOAD_TIMEOUT = 2400               # 55 GiB off the btrfs pool

# laguna-s2.1's live -ot rule, split so RPC rules can be inserted ahead of
# the CPU catch-all. Copied out of ~/services/llama-swap/config.yaml.
OT_HEAD = ("blk.(1[0-9]|[1-9]).ffn_.*_shexp=SYCL0,"
           "blk.2[0-8].ffn_.*_shexp=SYCL1,"
           "ffn_.*_shexp=SYCL0,"
           "blk.(1[0-9]|[1-9]).ffn_.*_exps=SYCL0,"
           "blk.2[0-8].ffn_.*_exps=SYCL1")
OT_TAIL = "ffn_.*_exps=CPU"

# Derived in step 4 from GGUF metadata; see gguf-placement.json.
R2_BLOCKS = "blk.(29|3[0-7]).ffn_.*_exps"          # 9 blocks, 9.5977 GiB
R3_BLOCKS = "blk.(3[89]|4[0-7]).ffn_.*_exps"       # 10 blocks, 11.250 GiB


def live_argv(binary, port):
    """laguna-s2.1's live argv, verbatim except --port."""
    return [
        str(binary), "--port", str(port), "--model", MODEL,
        "--flash-attn", "auto", "--jinja", "--parallel", "1",
        "-ngl", "999", "-c", "64000", "--split-mode", "layer",
        "--tensor-split", "22,10", "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0", "--fit", "off",
        "--device", "SYCL0,SYCL1",
        "-ot", f"{OT_HEAD},{OT_TAIL}",
    ]


def arm_argv(arm, port=PORT):
    if arm == "REF":
        return live_argv(PINNED_BIN, port)
    argv = live_argv(NEW_BIN, port)
    if arm == "R1":
        return argv
    i_dev = argv.index("--device")
    i_ts = argv.index("--tensor-split")
    i_ot = argv.index("-ot")
    if arm == "R2":
        argv[i_dev + 1] = "SYCL0,SYCL1,RPC0"
        argv[i_ts + 1] = "22,10,0"
        argv[i_ot + 1] = (f"{OT_HEAD},{R2_BLOCKS}=RPC0[{EP_CUDA0}],{OT_TAIL}")
        return argv[:1] + ["--rpc", EP_CUDA0] + argv[1:]
    if arm == "R3":
        argv[i_dev + 1] = "SYCL0,SYCL1,RPC0,RPC1"
        argv[i_ts + 1] = "22,10,0,0"
        argv[i_ot + 1] = (f"{OT_HEAD},{R2_BLOCKS}=RPC0[{EP_CUDA0}],"
                          f"{R3_BLOCKS}=RPC0[{EP_CPU0}],{OT_TAIL}")
        return argv[:1] + ["--rpc", f"{EP_CUDA0},{EP_CPU0}"] + argv[1:]
    raise ValueError(arm)


# --- environment ---------------------------------------------------------

def server_env(binary):
    """oneAPI on LD_LIBRARY_PATH -- a SYCL build SIGABRTs without it."""
    env = json.load(open(LANE / "agent-env.json"))
    for key in ("HOME", "USER", "LOGNAME", "PATH"):
        if key in os.environ and key != "PATH":
            env[key] = os.environ[key]
    bindir = str(Path(binary).parent)
    env["LD_LIBRARY_PATH"] = ":".join([
        bindir, "/opt/intel/oneapi/redist/lib",
        "/opt/intel/oneapi/umf/1.0/lib", env.get("LD_LIBRARY_PATH", "")])
    env["SYCL_CACHE_PERSISTENT"] = "0"
    env.pop("GGML_SYCL_DETERMINISTIC", None)
    floor = env.get("GGML_OP_OFFLOAD_MIN_BATCH")
    if floor is not None and int(floor) < 32:
        raise SystemExit(f"refusing: GGML_OP_OFFLOAD_MIN_BATCH={floor}")
    return env


# --- counters ------------------------------------------------------------

def rig_counters():
    out = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        if k in ("MemAvailable", "MemFree", "Cached"):
            out[f"rig_{k}_kB"] = int(v.split()[0])
    for d in ("rx_bytes", "tx_bytes"):
        p = f"/sys/class/net/{RIG_IFACE}/statistics/{d}"
        out[f"rig_{d}"] = int(open(p).read().strip())
    return out


def laptop_counters(timeout=15):
    cmd = ("grep -E '^(MemAvailable|MemFree|Cached):' /proc/meminfo; "
           f"cat /sys/class/net/{LAPTOP_IFACE}/statistics/rx_bytes; "
           f"cat /sys/class/net/{LAPTOP_IFACE}/statistics/tx_bytes")
    try:
        p = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
             f"aaron@{LAPTOP}", cmd],
            capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return {"laptop_error": p.stderr.strip()[:200]}
        lines = p.stdout.split()
        out = {}
        vals = p.stdout.splitlines()
        for line in vals:
            if ":" in line:
                k, _, v = line.partition(":")
                out[f"laptop_{k.strip()}_kB"] = int(v.split()[0])
        nums = [int(x) for x in vals if x.strip().isdigit()]
        if len(nums) >= 2:
            out["laptop_rx_bytes"], out["laptop_tx_bytes"] = nums[0], nums[1]
        return out
    except Exception as e:
        return {"laptop_error": f"{type(e).__name__}: {e}"}


def both_counters():
    c = rig_counters()
    c.update(laptop_counters())
    return c


def delta(before, after, keys):
    return {k: after[k] - before[k] for k in keys
            if k in before and k in after}


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
    with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return r.read().decode()


def post(path, body, timeout=COMPLETION_TIMEOUT):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def completion(n_predict):
    return post("/completion", {
        "prompt": PROMPT, "n_predict": n_predict, "temperature": 0.0,
        "top_k": 1, "seed": 42, "cache_prompt": False, "id_slot": 0,
        "return_tokens": True})


def one_run(arm, log_path, label):
    argv = arm_argv(arm)
    env = server_env(argv[0])
    if not Path(MODEL).exists():
        raise RuntimeError(f"model missing: {MODEL}")
    if not wait_port_free():
        raise RuntimeError(f"port {PORT} never became free")

    result = {"arm": arm, "label": label, "argv": argv,
              "binary": argv[0], "before": both_counters()}
    started = time.time()
    log = open(log_path, "w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                            env=env, preexec_fn=os.setpgrp)
    result["pid"] = proc.pid
    try:
        # Liveness from the first second: a server that dies on launch is
        # reported now, not after the load timeout.
        healthy = False
        for _ in range(LOAD_TIMEOUT):
            if proc.poll() is not None:
                log.flush()
                raise RuntimeError(
                    f"server exited rc={proc.returncode} during load; tail:\n"
                    + Path(log_path).read_text()[-3000:])
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
        result["after_load"] = both_counters()

        warm = completion(WARMUP_TOKENS)
        result["warmup_gen_tps"] = (warm.get("timings")
                                    or {}).get("predicted_per_second")

        net_before = both_counters()
        t0 = time.time()
        measured = completion(MEASURED_TOKENS)
        result["measured_wall_s"] = time.time() - t0
        net_after = both_counters()

        timings = measured.get("timings") or {}
        result["generation_tps"] = timings.get("predicted_per_second")
        result["prompt_tps"] = timings.get("prompt_per_second")
        result["predicted_n"] = timings.get("predicted_n")
        result["prompt_n"] = timings.get("prompt_n")
        result["stop_type"] = measured.get("stop_type")
        result["measured_tokens"] = measured.get("tokens")
        result["net_before_measured"] = net_before
        result["net_after_measured"] = net_after
        result["net_delta_measured"] = delta(net_before, net_after, [
            "rig_rx_bytes", "rig_tx_bytes",
            "laptop_rx_bytes", "laptop_tx_bytes"])
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
                result["required_sigkill"] = True
            except (ProcessLookupError, OSError):
                pass
            proc.wait(timeout=120)
        log.close()
        result["exit_code"] = proc.returncode
        result["port_free_after"] = wait_port_free()
        result["after"] = both_counters()
        result["finished"] = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="R1")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=str(LANE / "runs"))
    ap.add_argument("--tag", default="bench")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]

    records = []
    # Alternating arm order: run r0 of every arm, then r1 of every arm...
    for i in range(args.runs):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for arm in order:
            label = f"{args.tag}-{arm}-r{i}"
            print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)
            rec = modelctl_load.LoadRecorder(interval=5.0)
            entry = {"run": i, "arm": arm, "label": label}
            rec.start()
            try:
                entry["metrics"] = one_run(arm, outdir / f"{label}.log", label)
                m = entry["metrics"]
                print(f"    gen {m['generation_tps']:.4f} tok/s  "
                      f"prompt {m['prompt_tps']:.2f} tok/s  "
                      f"load {m['load_seconds']:.1f}s  "
                      f"net rig rx/tx "
                      f"{m['net_delta_measured'].get('rig_rx_bytes')}/"
                      f"{m['net_delta_measured'].get('rig_tx_bytes')}",
                      flush=True)
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
                print(f"    FAILED {entry['error'][:400]}", flush=True)
            finally:
                entry["load"] = rec.stop().summary()
            records.append(entry)
            path = outdir / f"{args.tag}.json"
            path.write_text(json.dumps(records, indent=2, default=str) + "\n")

    print(f"\nwrote {outdir / (args.tag + '.json')}")
    for arm in arms:
        vals = [r["metrics"]["generation_tps"] for r in records
                if r["arm"] == arm and "metrics" in r]
        if vals:
            print(f"  {arm}: {[round(v, 4) for v in vals]}  "
                  f"mean {sum(vals) / len(vals):.4f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
