# RPC node enablement — ph16-71 joins the fleet

Build-only pass. No model benchmarks, no timed runs; the only generation
that happened is the 8-token smoke below, on a 1.4 MB synthetic fixture.
Raw values only — no reading of them.

Date: 2026-08-02. Rig: `192.168.0.184` (Fedora 44). Node: `ph16-71` /
`laptop`, `192.168.0.76`.

## 0. Baseline re-verification

`moe-review/fleet-ph16-71.md` re-checked with one-liners. Every fact
matches: kernel `7.0.0-28-generic`, 32 threads, 30 GiB RAM, RTX 4080
Laptop 12282 MiB / driver 595.71.05 / sm_89, nvcc 12.4 (V12.4.131),
`/` on nvme0n1p2 937G with 695G free, LAN `enx6c6e072468a7` at 2500Mb/s
full duplex link up, ProtonVPN `protonvpn` up at 10.2.0.2/32,
thunderbolt `domain0` present. Nothing materially changed.

Two facts the baseline did not record, both load-bearing here:

* **No passwordless sudo on ph16-71.** `sudo -n true` → "interactive
  authentication is required". This is what forced user-scope systemd
  units and what left the ufw rules unapplied.
* **No RPC-capable llama-server on the rig.** All of `~/tmp/build-*`
  and `~/src/llama.cpp-laguna/build-sycl` have zero RPC symbols, and
  `llama-server` is not on PATH. The smoke needed a client, so one was
  built (§2).

## 1. Fork builds on the laptop, at the pin

Two independent clones of `85b7e6556` (shallow, shipped rig→laptop;
the laptop cannot ssh back to the rig, and `gitea` does not resolve
there). Both built with **gcc-13**: CUDA 12.4's `host_config.h` errors
on `__GNUC__ > 13` and the laptop defaults to gcc-15. Same compiler for
both so the pair differs only in backend.

| | CUDA | CPU |
|---|---|---|
| commit | `85b7e6556b6b83026d1a17df2635bc1173db1f97` | same |
| `git describe` | `85b7e65` | `85b7e65` |
| sha256 | `5e912c4ed299655c9a82ba312b2e0e3c1f2d4071f2e4cbf5f71c5638bf238b6a` | `bda7a51d17806dd8a9e9c196e23517f3f1370c117fc44188b0d6aaf22ae818f2` |
| backend libs | base, cpu, **cuda**, rpc | base, cpu, rpc |
| cmake | 4.2.3 | 4.2.3 |
| cc | gcc-13 (Ubuntu 13.4.0-10ubuntu1) 13.4.0 | same |

Device probe (`ggml-rpc-server -d __probe__`, a deliberate unknown-device
argument that makes it enumerate and exit):

    cuda: CUDA0: NVIDIA GeForce RTX 4080 Laptop GPU (11876 MiB, 11681 MiB free)
          CPU:   13th Gen Intel(R) Core(TM) i9-13900HX (31308 MiB, 31308 MiB free)
    cpu:  CPU:   13th Gen Intel(R) Core(TM) i9-13900HX (31308 MiB, 31308 MiB free)

`ggml-rpc-server` links only ggml, never libcommon: it has **no
`--version`** and no embedded build-info. Version reporting is therefore
split — the wire HELLO gives the protocol version, and a
`build-manifest.json` written next to each binary records the commit.
Both manifests report the pin.

## 2. Client build on the rig (not in the work order; §5 needs it)

CPU-only + `GGML_RPC=ON`, from the same pinned source, at
`/home/aaron/tmp/build-rpc-client`. Kept off the live SYCL stack's
libraries. Confirms `--rpc SERVERS` and links `libggml-rpc.so.0`.
Nothing in `~/services/`, systemd, docker, llama-swap or OVMS was
touched.

## 3. Units

`rpc-cuda0.service` (50052, `-d CUDA0 -t 4 -c`) and `rpc-cpu0.service`
(50053, `-d CPU -t 20 -c`), both `-H 192.168.0.76`. Copies in
`docs/fleet/`.

**User units, not system units** — no root available. Linger is already
enabled and cgroup v2 delegates `cpu memory pids` to the user manager,
so the caps are enforced, read back off the running units:

    rpc-cuda0: enabled active MemoryMax=21474836480 CPUQuotaPerSecUSec=20s Nice=10
    rpc-cpu0:  enabled active MemoryMax=21474836480 CPUQuotaPerSecUSec=20s Nice=10

Bind, from `ss -ltnp` on the node — LAN address only, not `0.0.0.0`, not
the tunnel:

    LISTEN 192.168.0.76:50052  users:(("ggml-rpc-server",pid=1472070,...))
    LISTEN 192.168.0.76:50053  users:(("ggml-rpc-server",pid=1472071,...))

Tensor cache on the laptop NVMe via `LLAMA_CACHE=/home/aaron/rpc/cache/{cuda0,cpu0}/`
(`/` is nvme0n1p2; `/home` is not a separate mount).

## 4. ufw — NOT APPLIED

Requires root; none available unattended. Rules written to
`docs/fleet/ph16-71-ufw.sh`, unapplied.

Measured current state, not inferred: a listener bound to
`192.168.0.76:50052` on the node was reached from the rig with ufw
active, so the present policy does not restrict these ports. **Any host
on `192.168.0.0/24` can currently reach both, and `ggml-rpc-server` has
no authentication.**

systemd `IPAddressAllow`/`IPAddressDeny` was tested as a root-free
substitute and **is not enforced for `--user` units** on this box: a
transient unit with `IPAddressDeny=any` and `IPAddressAllow=192.168.0.184`
still connected to `1.1.1.1:53`. The properties are accepted silently.

## 5. Planner — fallback plan diff

Same profile, node registered but never probed (absent) vs node probed
present. Unified diff of the whole compiled plan set:

    === node ABSENT: 6 plans
    === node PRESENT: 7 plans
    @@ -12 +12,3 @@
         --model .../tiny-rpc-smoke.gguf --flash-attn off --jinja --parallel 1 -ngl 999 -c 512 --split-mode layer --tensor-split 28,12 --cache-type-k f16 --cache-type-v f16 --device SYCL0,SYCL1
    +f893695adc5da26c  [fleet-rpc]  layers 6-7 on 192.168.0.76:50052
    +    --model .../tiny-rpc-smoke.gguf --flash-attn off --jinja --parallel 1 -ngl 999 -c 512 --rpc 192.168.0.76:50052 -ot blk\.(6|7)\.=RPC0[192.168.0.76:50052] --cache-type-k f16 --cache-type-v f16

    lines removed or changed from the local plan set: 0

Purely additive. The same property is asserted three ways in
`test_fleet_rpc.py` (no registry vs absent node; unreachable node;
node on a different commit).

### One thing that bit, recorded because it is not guessable

`-ot` takes a **buffer type**, not a device name, and the two are
numbered differently. Device names are global across `--rpc` endpoints;
buffer type names are per-endpoint and carry the endpoint in brackets.
Two single-device nodes give:

    Available buffer types:
      CPU
      RPC0[192.168.0.76:50052]
      RPC0[192.168.0.76:50053]

— both `RPC0` — while their device names are `RPC0` and `RPC1`. Passing
the device name fails with `unknown buffer type`. First smoke attempt
died exactly this way.

## 6. Smoke — tiny fixture only

No local GGUF is small enough to be a fixture (every one is a vocab-only
stub or a shard of a 17–400 GB model), so a 1.4 MB llama-arch model was
synthesized with the repo's own `gguf-py`: 8 layers, n_embd 64, F32,
seeded and therefore byte-reproducible. Generator:
`docs/fleet/make-tiny-fixture.py`. Output:
`/home/aaron/models/fixtures/tiny-rpc-smoke.gguf`.

Its vocabulary is printable ASCII rather than byte tokens on purpose: a
random-weight model emits arbitrary byte sequences, which is invalid
UTF-8, and llama-server's content parser 500s on the response before any
comparison can happen. The prompt is passed as token ids for the same
class of reason — a natural-language prompt tokenizes to nothing against
a synthetic vocabulary and the server rejects the empty input.

Launched through `modelctl_launch.launch_command_for_profile` — the same
object the worker, `run.sh`, the llama-swap entry and the browser
preview derive from — under a scratch `MODELCTL_HOME`:

    /home/aaron/tmp/build-rpc-client/bin/llama-server --model /home/aaron/models/fixtures/tiny-rpc-smoke.gguf \
      --flash-attn off --jinja --parallel 1 -ngl 999 -c 512 \
      --rpc 192.168.0.76:50052 -ot blk\.(6|7)\.=RPC0[192.168.0.76:50052] \
      --cache-type-k f16 --cache-type-v f16 --port 18111

    command_fingerprint=be61342c13dc7e60  valid=True

Result:

    health after 0.50s: True
    run 1: tokens=[120, 120, 120, 120, 120, 120, 52, 254]
    run 2: tokens=[120, 120, 120, 120, 120, 120, 52, 254]
    token-identical across runs: True

Device log on the node during the run (`journalctl --user -u rpc-cuda0`):

    13 client connections accepted from the rig
    5 CUDA graph-compute events on the node's 4080, e.g.
    2026-08-02T00:03:11-04:00 laptop ggml-rpc-server[1472070]: ggml_backend_cuda_graph_compute: CUDA graph warmup complete

The client-side log shows no tensor-placement lines at default
verbosity; the node-side journal is the evidence that the remote
executed, and it is the stronger of the two anyway.

## 7. Registry state

Both endpoints enrolled in `$MODELCTL_HOME/fleet.json` and probed live:

    ph16-71-cuda0  192.168.0.76:50052  reachable=true protocol=5.0.0 pin_agrees=true
    ph16-71-cpu0   192.168.0.76:50053  reachable=true protocol=5.0.0 pin_agrees=true

Declared budgets (operator ceilings, not measurements): CUDA0 10 GiB of
11.6 GiB present; CPU 16 GiB, under the unit's own `MemoryMax=20G`.

## 8. Night lane

Two pairs pre-registered in `modelctl/night-lane.json`, both
`enabled: false`, neither scheduled: `ornith-rpc-criterion-2026-08-02`
and `qwen122b-remote-experts-hypothesis-2026-08-02`. Criteria and
measured quantities are fixed in the same commit that adds them; no
expected outcome is recorded, and a test enforces that.

## 9. Tests

`1286 passed, 11 skipped` (was 1234 before this pass; +52 new).

One pre-existing flake surfaced once under `-n auto` and is **not** from
this work: `test_modelctl_web.py::TestJobLanes::test_lane_survives_bookkeeping_failure`
emitted a `PytestUnraisableExceptionWarning` from a temp directory
cleaned up while its deliberately-readonly job DB was still being
written by the lane's daemon thread. It did not reproduce in three
consecutive full-suite runs afterwards (`1286 passed, 11 skipped, 28
warnings` each), nor in three runs of that file alone. The test passes
either way; the warning is cleanup noise from that test's own design.
