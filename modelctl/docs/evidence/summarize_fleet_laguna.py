#!/usr/bin/env python3
"""Build the report tables from the run JSON."""
import json
from pathlib import Path

LANE = Path("/home/aaron/workspace/.lanes/fleet-laguna-baseline")
RUNS = LANE / "runs"

FILES = [("control", "control.json"), ("baseline", "baseline.json"),
         ("r2probe", "r2probe.json"), ("r3probe", "r3probe.json"),
         ("smoke", "smoke.json")]

recs = []
for tag, fn in FILES:
    p = RUNS / fn
    if p.exists():
        for r in json.loads(p.read_text()):
            r["_tag"] = tag
            recs.append(r)

ok = [r for r in recs if "metrics" in r]
bad = [r for r in recs if "metrics" not in r]

print("=" * 100)
print("PER-RUN TABLE  (battery runs first; probe runs marked)")
print("=" * 100)
hdr = (f"{'label':<20} {'arm':<5} {'gen tok/s':>10} {'prompt t/s':>11} "
       f"{'load s':>8} {'wall s':>8} {'pred_n':>7} {'stop':>7}")
print(hdr)
print("-" * len(hdr))
for r in ok:
    m = r["metrics"]
    print(f"{r['label']:<20} {r['arm']:<5} {m['generation_tps']:>10.4f} "
          f"{m['prompt_tps']:>11.2f} {m['load_seconds']:>8.1f} "
          f"{m['measured_wall_s']:>8.1f} {str(m['predicted_n']):>7} "
          f"{str(m['stop_type']):>7}")

print()
print("=" * 100)
print("NETWORK BYTES ACROSS THE MEASURED DECODE (both ends)")
print("=" * 100)
hdr2 = (f"{'label':<20} {'rig rx':>13} {'rig tx':>13} "
        f"{'laptop rx':>13} {'laptop tx':>13}")
print(hdr2)
print("-" * len(hdr2))
for r in ok:
    d = r["metrics"].get("net_delta_measured", {})
    print(f"{r['label']:<20} {d.get('rig_rx_bytes', '-'):>13,} "
          f"{d.get('rig_tx_bytes', '-'):>13,} "
          f"{d.get('laptop_rx_bytes', '-'):>13,} "
          f"{d.get('laptop_tx_bytes', '-'):>13,}")

print()
print("=" * 100)
print("MemAvailable, BOTH MACHINES, before -> after run (GiB)")
print("=" * 100)
hdr3 = (f"{'label':<20} {'rig before':>11} {'rig aload':>10} {'rig after':>10} "
        f"{'lap before':>11} {'lap aload':>10} {'lap after':>10}")
print(hdr3)
print("-" * len(hdr3))


def gib(d, k):
    v = d.get(k)
    return f"{v / 1048576:.2f}" if isinstance(v, int) else "-"


for r in ok:
    m = r["metrics"]
    b, al, a = m.get("before", {}), m.get("after_load", {}), m.get("after", {})
    print(f"{r['label']:<20} {gib(b,'rig_MemAvailable_kB'):>11} "
          f"{gib(al,'rig_MemAvailable_kB'):>10} {gib(a,'rig_MemAvailable_kB'):>10} "
          f"{gib(b,'laptop_MemAvailable_kB'):>11} "
          f"{gib(al,'laptop_MemAvailable_kB'):>10} "
          f"{gib(a,'laptop_MemAvailable_kB'):>10}")

print()
print("=" * 100)
print("RIG LOAD TRACE PER RUN (modelctl_load sampler, 5 s interval)")
print("=" * 100)
hdr4 = (f"{'label':<20} {'samples':>8} {'load1 min':>10} {'load1 mean':>11} "
        f"{'load1 max':>10} {'memavail min GiB':>17}")
print(hdr4)
print("-" * len(hdr4))
for r in ok:
    ld = r.get("load") or {}
    l1 = ld.get("loadavg_1m") or {}
    ma = ld.get("mem_available_bytes") or ld.get("mem_available_gib") or {}
    mn = ma.get("min")
    if isinstance(mn, (int, float)) and mn > 1e6:
        mn = mn / 2**30
    print(f"{r['label']:<20} {str(ld.get('samples','-')):>8} "
          f"{l1.get('min','-'):>10} {l1.get('mean','-'):>11} "
          f"{l1.get('max','-'):>10} "
          f"{(f'{mn:.2f}' if isinstance(mn,(int,float)) else '-'):>17}")

print()
print("=" * 100)
print("ARM SUMMARY (battery runs only -- probe runs excluded)")
print("=" * 100)
battery = {"control": ["R1", "REF"], "baseline": ["R2", "R3"]}
means = {}
for tag, arms in battery.items():
    for arm in arms:
        vals = [r["metrics"]["generation_tps"] for r in ok
                if r["_tag"] == tag and r["arm"] == arm]
        loads = [r["metrics"]["load_seconds"] for r in ok
                 if r["_tag"] == tag and r["arm"] == arm]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        means[arm] = mean
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {arm:<5} n={len(vals)}  runs={[round(v,4) for v in vals]}")
        print(f"        mean {mean:.4f} tok/s  sd {sd:.4f}  "
              f"load mean {sum(loads)/len(loads):.1f}s")

print()
print("=" * 100)
print("DELTAS")
print("=" * 100)
ANCHOR, REOBS = 14.20, 13.71
for arm in ("REF", "R1", "R2", "R3"):
    if arm not in means:
        continue
    m = means[arm]
    line = (f"  {arm:<5} {m:8.4f}  vs anchor 14.20: {(m/ANCHOR-1)*100:+7.2f}%"
            f"   vs 13.71 re-obs: {(m/REOBS-1)*100:+7.2f}%")
    if "R1" in means:
        line += f"   vs R1: {(m/means['R1']-1)*100:+7.2f}%"
    print(line)
if "REF" in means and "R1" in means:
    print(f"\n  R1 vs REF (same protocol, same argv, pinned vs new binary): "
          f"{(means['R1']/means['REF']-1)*100:+.2f}%")

if bad:
    print("\nFAILED RUNS:")
    for r in bad:
        print(f"  {r['label']}: {r.get('error')}")
else:
    print("\nNo failed runs.")
