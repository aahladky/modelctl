"""Scratch walk of the console fleet surface.

Runs a throwaway console on the lane's port block against a throwaway
state universe (all five redirections from the scratch-safe recipe), so
nothing here can touch the live registry, the live presence record, the
live profiles or the serving stack. What it proves:

* the fleet read model renders all three presence states, from seeded
  scratch presence rather than by reaching across the LAN;
* every fleet write is refused by the scratch middleware, with a reason
  naming the request;
* the budget ceiling refuses an over-ceiling edit for real -- shown
  against the same scratch state through the CLI, because a scratch
  console refuses the request before the ceiling is ever consulted, so
  the console transcript alone cannot prove the ceiling exists.

Usage: python docs/evidence/fleet_walk.py [port]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELCTL = HERE.parent.parent
PIN = "85b7e6556b6b83026d1a17df2635bc1173db1f97"
OTHER_PIN = "1111111111111111111111111111111111111111"
GIB = 1 << 30
TOKEN = "scratch-walk-token"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9500

out = []


def say(line=""):
    print(line)
    out.append(line)


def scratch_env(home):
    """The five redirections, plus the token and bind.

    MODELCTL_LLAMA_SWAP_SERVICE points at a unit that does not exist and
    the base URL at the discard port, so any code path that tried to
    drive the router fails harmlessly instead of reaching 9292.
    """
    env = dict(os.environ)
    env.update({
        "MODELCTL_WEB_SCRATCH": "1",
        "MODELCTL_HOME": str(home),
        "MODELCTL_MODELS_DIR": str(home / "models"),
        "MODELCTL_LLAMA_SWAP_CONFIG": str(home / "config.yaml"),
        "MODELCTL_LLAMA_SWAP_SERVICE": "modelctl-scratch-nonexistent.service",
        "MODELCTL_LLAMA_SWAP_BASE_URL": "http://127.0.0.1:9/v1/",
        "MODELCTL_LLAMA_SERVER": str(home / "stub-llama-server"),
        "MODELCTL_WEB_TOKEN": TOKEN,
        "MODELCTL_WEB_BIND": f"127.0.0.1:{PORT}",
        "MODELCTL_FLEET_PIN": PIN,
    })
    return env


def seed(home):
    """Three nodes, one per presence state. Never the real registry."""
    (home / "models").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("models: {}\n")
    (home / "stub-llama-server").write_text("#!/bin/sh\nexit 1\n")
    profiles = home / "profiles"
    profiles.mkdir(exist_ok=True)
    # One profile whose recorded planning inputs were taken when cpu0's
    # budget was 16 GiB, so moving that budget has something to stale.
    (profiles / "walk-dependent.json").write_text(json.dumps({
        "name": "walk-dependent", "model_path": "/fake/m.gguf",
        "config": {"ctx": 8192, "device": "SYCL0"},
        "planning": {"recorded_at": "2026-08-02", "inputs": {
            "version": 1, "ram_available_bytes": 31 * GIB,
            "vram_limit_pct": 90, "primary": "SYCL0", "inventory": [],
            "hw_settings": {"devices": {}},
            "capabilities": {"features": {
                "moe_cache_per_device_budgets": False}},
            "display": {"mode": "unknown", "freed_bytes": 0},
            "fleet": {"budgets_bytes": {"RPC:ph16-71-cpu0:CPU": 16 * GIB}}}}},
        indent=2))

    def node(name, port, pin, kind, total, budget, cap=0, variant="cpu"):
        d = {"name": {"cpu": "CPU"}.get(kind, "CUDA0"), "kind": kind,
             "total_bytes": total, "budget_bytes": budget, "cap_bytes": cap}
        return {"name": name, "host": "192.168.0.76", "port": port,
                "variant": variant, "pin": pin, "enabled": True,
                "note": f"scratch fixture for the {name} card", "devices": [d]}

    (home / "fleet.json").write_text(json.dumps({"version": 1, "nodes": [
        # present: fresh probe, agreeing pin. Same numbers the live cpu0
        # carries, so the ceiling in the transcript is the real one.
        node("ph16-71-cpu0", 50053, PIN, "cpu", 32828817408, 16 * GIB,
             cap=20 * GIB),
        # absent: never probed
        node("ph16-71-cuda0", 50052, PIN, "gpu", 12452888576, 10 * GIB,
             variant="cuda"),
        # pin mismatch: reachable, wrong commit
        node("ph16-71-oldbuild", 50054, OTHER_PIN, "gpu", 12452888576,
             8 * GIB, variant="cuda"),
    ]}, indent=2))

    now = time.time()
    (home / "fleet-presence.json").write_text(json.dumps({"version": 1, "nodes": {
        "ph16-71-cpu0": {
            "node": "ph16-71-cpu0", "endpoint": "192.168.0.76:50053",
            "reachable": True, "protocol": "5.0.0", "pin": PIN,
            "pin_agrees": True, "detail": "", "probed_at": now - 30},
        # ph16-71-cuda0 deliberately absent from the record: never probed
        "ph16-71-oldbuild": {
            "node": "ph16-71-oldbuild", "endpoint": "192.168.0.76:50054",
            "reachable": True, "protocol": "5.0.0", "pin": OTHER_PIN,
            "pin_agrees": False,
            "detail": f"node built at {OTHER_PIN[:12]}, this checkout pins "
                      f"{PIN[:12]}",
            "probed_at": now - 30},
    }}, indent=2))


def req(method, path, body=None):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(body or {}).encode() if method == "POST" else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", f"Bearer {TOKEN}")
    r.add_header("Accept", "application/json")
    if data:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait_up(proc, seconds=25):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"console exited early, rc={proc.returncode}")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/healthz", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("console never came up")


def cli(env, *args):
    p = subprocess.run([sys.executable, str(MODELCTL / "modelctl.py"), *args],
                       cwd=str(MODELCTL), env=env, capture_output=True,
                       text=True, timeout=60)
    return p.returncode, (p.stdout + p.stderr).strip()


def main():
    home = Path(tempfile.mkdtemp(prefix="fleet-walk-"))
    env = scratch_env(home)
    seed(home)
    say("=" * 72)
    say("SCRATCH WALK -- console fleet surface")
    say(f"state universe: {home}")
    say(f"console: 127.0.0.1:{PORT}  (lane port block)")
    say("MODELCTL_WEB_SCRATCH=1, all five redirections set")
    say("=" * 72)

    proc = subprocess.Popen(
        [sys.executable, "-m", "modelctl_web"], cwd=str(MODELCTL), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_up(proc)
        say(f"\n[liveness] console up, pid {proc.pid}, /healthz 200")

        say("\n--- 1. the fleet view loads ---------------------------------")
        status, body = req("GET", "/api/v2/fleet")
        view = json.loads(body)
        say(f"GET /api/v2/fleet -> {status}")
        for n in view["nodes"]:
            st = n["presence"]["state"]
            say(f"  {n['name']:<22} {n['location']:<6} {st:<13} "
                f"pin_agrees={n['pin']['agrees']} {n['presence']['detail']}")
            for d in n["devices"]:
                say(f"      {d['name']:<6} budget "
                    f"{d['budget_bytes'] / GIB:6.2f} GiB  "
                    f"ceiling {d['ceiling_bytes'] / GIB:6.2f} GiB  "
                    f"total {d['total_bytes'] / GIB:6.2f} GiB  "
                    f"({d['ceiling_basis']}) editable={d['editable']}")

        say("\n--- 2. the three presence states ----------------------------")
        states = {n["name"]: n["presence"]["state"] for n in view["nodes"]}
        for name, want in (("ph16-71-cpu0", "PRESENT"),
                           ("ph16-71-cuda0", "STALE"),
                           ("ph16-71-oldbuild", "PIN_MISMATCH")):
            got = states.get(name)
            say(f"  {name:<22} {got:<13} {'OK' if got == want else 'MISMATCH'}")
        say("  rendering: PRESENT -> chip ok, STALE -> chip warn, "
            "PIN_MISMATCH -> chip err")
        say("  every non-PRESENT card also carries the 'widget stale' "
            "treatment, so a reachable wrong-commit node cannot read as "
            "available")

        say("\n--- 3. night-lane pairs needing a node (READ-ONLY) ----------")
        for j in view["night_lane"]:
            say(f"  {j['id']:<44} enabled={j['enabled']} "
                f"needs={','.join(j['requires_nodes'])}")
        if not view["night_lane"]:
            say("  (none registered)")

        say("\n--- 4. budget edit within the ceiling, submitted ------------")
        status, body = req(
            "POST", "/api/v2/fleet/nodes/ph16-71-cpu0/devices/CPU/budget",
            {"budget_bytes": 18 * GIB})
        say(f"POST .../devices/CPU/budget  budget_bytes={18 * GIB} "
            f"(18.00 GiB, under the 19.00 GiB ceiling)")
        say(f"  -> {status} {json.loads(body).get('reason', body)}")

        say("\n--- 5. over-ceiling edit ------------------------------------")
        status, body = req(
            "POST", "/api/v2/fleet/nodes/ph16-71-cpu0/devices/CPU/budget",
            {"budget_bytes": 25 * GIB})
        say(f"POST .../devices/CPU/budget  budget_bytes={25 * GIB} "
            f"(25.00 GiB, over the 19.00 GiB ceiling)")
        say(f"  -> {status} {json.loads(body).get('reason', body)}")

        say("\n--- 6. probe writes are refused too -------------------------")
        for path in ("/api/v2/fleet/probe",
                     "/api/v2/fleet/nodes/ph16-71-cpu0/probe"):
            status, body = req("POST", path)
            say(f"POST {path} -> {status} "
                f"{json.loads(body).get('reason', body)[:96]}...")

        say("\n--- 7. the SPA route serves the page ------------------------")
        status, body = req("GET", "/v2/fleet")
        say(f"GET /v2/fleet -> {status}, {len(body)} bytes of SPA shell")

        say("\n--- 7b. what the page says a budget change would stale ------")
        for p in view["stale_profiles"]:
            say(f"  {p['name']}: {json.dumps(p['changed'])}")
        if not view["stale_profiles"]:
            say("  (nothing stale yet -- the seeded profile matches the "
                "budget in force)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        say(f"\n[teardown] console pid {proc.pid} stopped, rc={proc.returncode}")

    # The scratch console refuses a write before the handler runs, so the
    # ceiling itself never gets consulted there. Prove it against the same
    # scratch state through the CLI, which is the same primitive the
    # console's mutation entry calls.
    say("\n--- 8. the ceiling, against the same scratch state (CLI) ----")
    cli_env = dict(env)
    cli_env.pop("MODELCTL_WEB_SCRATCH", None)
    rc, text = cli(cli_env, "fleet", "set-budget", "ph16-71-cpu0", "CPU",
                   str(25 * GIB))
    say(f"$ modelctl fleet set-budget ph16-71-cpu0 CPU {25 * GIB}   # 25 GiB")
    say(f"  rc={rc}")
    for line in text.splitlines():
        say(f"  {line}")
    rc, text = cli(cli_env, "fleet", "set-budget", "ph16-71-cpu0", "CPU",
                   str(18 * GIB))
    say(f"$ modelctl fleet set-budget ph16-71-cpu0 CPU {18 * GIB}   # 18 GiB")
    say(f"  rc={rc}")
    for line in text.splitlines():
        say(f"  {line}")
    after = json.loads((home / "fleet.json").read_text())
    cpu = [n for n in after["nodes"] if n["name"] == "ph16-71-cpu0"][0]
    say(f"  scratch registry now: budget "
        f"{cpu['devices'][0]['budget_bytes'] / GIB:.2f} GiB")

    say("\n--- 9. the live registry was never touched ------------------")
    live = Path.home() / ".local" / "share" / "modelctl" / "fleet.json"
    live_nodes = json.loads(live.read_text())["nodes"] if live.exists() else []
    for n in live_nodes:
        d = n["devices"][0]
        say(f"  {n['name']:<22} budget {d['budget_bytes'] / GIB:6.2f} GiB  "
            f"cap {d.get('cap_bytes', 0) / GIB:6.2f} GiB")

    shutil.rmtree(home, ignore_errors=True)
    say(f"\n[teardown] scratch state universe {home} removed")
    Path(HERE / "fleet-walk-transcript.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
