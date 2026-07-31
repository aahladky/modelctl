#!/usr/bin/env python
"""Offload-threshold sweep on Qwen3.5-122B-A10B Q4_K_M.

Four conditions through modelctl_acceptance.run_matrix(), so placement,
preconditions, provenance and D2's storage counters all come from the
harness rather than from a bespoke script.

Placement is DERIVED for this quant, not inherited: the tier planner was
asked what fits, and its answer (experts 0-15 on SYCL0, 16-20 on SYCL1,
21-47 to mmap) is pinned into the profile so every condition shares it.
The run this replaces inherited a split tuned for a quant 2.2x smaller and
had to be thrown away.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/aaron/workspace/moe-serving/modelctl")

import modelctl_acceptance as acc
import modelctl_hardware
import modelctl_launch
import modelctl_profiles

MODEL = ("/home/aaron/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/"
         "Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf")
# Derived with the 4+2 GiB cache budget RESERVED, not for the model
# alone: weights 23.1 GiB + cache 4.0 = 27.1 of 30.3 free on SYCL0
# (89%), 6.1 + 2.0 = 8.1 of 11.9 on SYCL1 (68%). The previous attempt
# derived for the model only and the cache pushed SYCL0 past its
# capacity, losing the device mid-decode.
OT = (r"blk\.(?:[0-9]|1[0-2])\.ffn_.*_exps=SYCL0,"
      r"blk\.1[3-5]\.ffn_.*_exps=SYCL1,"
      r"ffn_.*_exps=CPU")
CELLS = ("offload-A-default", "offload-B-global-1",
         "offload-D-moe-1", "offload-E-moe-1-no-cache")
# Written next to this script so the sweep and its result travel together.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "offload-sweep-results.json")


def main():
    profile = modelctl_profiles.normalize_profile({
        "name": "q4km-offload-sweep",
        "backend": "llama-cpp",
        "model_path": MODEL,
        # Pinned explicitly: resolve_backend() otherwise picks the default
        # binary, which is the PRE-SYNC build-sycl. That build has no
        # MoE-specific threshold, so the gated cells correctly refused to
        # run -- the guard worked, the profile was wrong.
        "binary": ("/home/aaron/workspace/moe-serving/llama.cpp/"
                   "build-sycl-f/bin/llama-server"),
        "config": {"device": "SYCL0,SYCL1", "split_mode": "layer",
                   "tensor_split": "8,3", "ctx": 8192, "fit": "off",
                   "extra": "-ot " + OT},
    })
    snap = modelctl_hardware.capture_hardware_snapshot()
    backend = modelctl_launch.resolve_backend(profile)
    print(f"binary {backend.binary}", flush=True)
    print(f"caps moe_offload_threshold_control="
          f"{backend.capabilities.get('features', {}).get('moe_offload_threshold_control')}",
          flush=True)

    started = time.time()
    results = acc.run_matrix(
        profile, snapshot=snap, backend=backend, include_heavy=True,
        only=set(CELLS),
        # 128 tokens per measured run, a warmup pass first so every
        # condition starts from a comparable cache and page-cache state,
        # and two measured runs so the spread is visible.
        max_tokens=128, runs=2, warmup_tokens=64,
        log=lambda *a: print(*a, flush=True))

    rows = []
    for r in results:
        if r.cell.name not in CELLS:
            continue
        rows.append({"cell": r.cell.name, "status": r.status,
                     "reason": r.reason, "elapsed_s": r.elapsed_seconds,
                     **(r.measurement or {})})
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n=== sweep finished in {time.time() - started:.0f}s ===", flush=True)
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
