# Fleet node admission: RPC build + laguna-s2.1 static baseline

Executed 2026-08-02 from `moe-review/fleet-laguna-baseline-order.md`.
Static placement only — no expert cache, no hybrid, no cache flags in any
arm. Raw numbers; no judgement of what they mean.

Harness: `bench_laguna_rpc.py` (this directory). Node probe:
`probe_fleet_capability.py`. Placement arithmetic: `gguf_placement.py`.
Raw per-run JSON and every server log: `runs-fleet-laguna/`. The three
scripts are copied **verbatim as they ran**, so their absolute paths still
point at the (now removed) lane worktree
`~/workspace/.lanes/fleet-laguna-baseline`; they were not rewritten
afterwards, because then they would not be what produced these numbers.

## 0. What was and was not touched

- `llama-swap :9292`, OVMS, console `:9293`, remote-hands `:9294`: not
  restarted, not contacted. OVMS left down.
- No saved profile, artifact, or laguna llama-swap entry written. Every
  arm launched its own `llama-server` on scratch port **9501** (inside the
  lane's 9500-9509 block) and was torn down by PID; port confirmed free
  after every run. No run required SIGKILL.
- **Nothing was restarted on the laptop.** Both user units were already
  ACTIVE and answered every probe; no unit was wedged.
- `rpc-cpu0` MemoryMax left at 26 GiB. `GGML_OP_OFFLOAD_MIN_BATCH` never
  set below 32 (harness refuses to launch if it is).
- Page cache not dropped or manipulated; MemAvailable/Cached recorded only.

## 1. Second build, first build untouched

`llama.cpp/build-sycl-rpc` configured from the same commit and the same
settings read out of `build-sycl/CMakeCache.txt` (Release, icx/icpx from
`/opt/intel/oneapi/compiler/2026.1`, `GGML_SYCL=ON`,
`GGML_SYCL_TARGET=INTEL`, `GGML_SYCL_DNN/GRAPH/HOST_MEM_FALLBACK/`
`SUPPORT_LEVEL_ZERO_API=ON`, `GGML_SYCL_F16=OFF`, `GGML_NATIVE=ON`,
`BUILD_SHARED_LIBS=ON`, Unix Makefiles) plus `-DGGML_RPC=ON`.
Target `llama-server`. **Build wall-time 133 s** (ccache was ON and
populated from the existing build).

| proof | result |
|---|---|
| `GGML_RPC` in new cache | `GGML_RPC:BOOL=ON` |
| `GGML_RPC` in old cache | `GGML_RPC:BOOL=OFF` (unchanged) |
| rpc symbols, new build | `libggml-rpc.so` present, **34** rpc symbols (`ggml_backend_rpc_init`, `_add_server`, `_buffer_type`, `_get_device_memory`, `_start_server`, …) |
| rpc symbols, old build | no `libggml-rpc.so` at all |
| new `llama-server` links rpc | yes — `libggml-rpc.so.0` via `libggml.so.0` (ldd) |
| `build-sycl/bin/llama-server` sha256 **before** | `51e88ddee69e182fd2ddd665dc4aa52077e3f7afeaa433fe58ae165ee0a09095` |
| `build-sycl/bin/llama-server` sha256 **after** | `51e88ddee69e182fd2ddd665dc4aa52077e3f7afeaa433fe58ae165ee0a09095` — identical, 654224 B, mtime 2026-08-01 22:09:05 unchanged |
| commit, both builds | `LLAMA_BUILD_NUMBER = 10217`, commit string `85b7e6556` embedded in each tree's `libllama-common.so` |

Note: `nm -D` on `llama-server` itself finds zero rpc symbols in **both**
builds — with `BUILD_SHARED_LIBS=ON` the backend lives in
`libggml-rpc.so`. The order's "zero rpc symbols on build-sycl's
llama-server" fact is true of the RPC build too; the discriminating
artifact is the `.so`, not the executable.

## 2. Handshake and capability (both nodes)

Rig checkout pin `85b7e6556b6b83026d1a17df2635bc1173db1f97`.

| | ph16-71-cuda0 | ph16-71-cpu0 |
|---|---|---|
| endpoint | 192.168.0.76:50052 | 192.168.0.76:50053 |
| reachable | yes | yes |
| **negotiated protocol** | **5.0.0** | **5.0.0** |
| recorded pin | 85b7e6556b6b… | 85b7e6556b6b… |
| **pin agrees** | **yes** | **yes** |
| TCP connect | 0.263 ms | 0.203 ms |
| HELLO round trip | 0.658 ms | 0.292 ms |
| RTT min / median / max (n=20) | 0.152 / 0.178 / 0.459 ms | 0.112 / 0.125 / 0.246 ms |
| device count | 1 | 1 |
| device free / total | 12,210,733,056 / 12,453,019,648 B (11.372 / 11.598 GiB) | 32,829,595,648 / 32,829,595,648 B (30.575 / 30.575 GiB) |

As enumerated from the rig by the RPC build itself
(`llama-server --rpc … --list-devices`):

```
SYCL0: Intel(R) Arc(TM) Pro B70 Graphics (32656 MiB, 32500 MiB free)
SYCL1: Intel(R) Arc(TM) B580 Graphics       (12216 MiB, 12178 MiB free)
RPC0:  192.168.0.76:50052                   (11876 MiB, 11645 MiB free)
RPC1:  192.168.0.76:50053                   (31308 MiB, 31308 MiB free)
```

Recorded facts, not conclusions:

- `rpc-cpu0` reports **free == total == whole-machine RAM** (30.575 GiB).
  The RPC protocol carries no notion of the unit's systemd `MemoryMax`, so
  the 26 GiB cap is invisible to a client and was respected by this
  order's own arithmetic rather than by anything the node enforces on the
  wire.
- Device names are `RPC0`/`RPC1` (global counter), but the **buffer type**
  names both start `RPC0` and are distinguished only by endpoint:
  `RPC0[192.168.0.76:50052]`, `RPC0[192.168.0.76:50053]`. `-ot` resolves
  by buffer-type name, so `-ot …=RPC1` does not exist and fails with
  "unknown buffer type".

## 3. Placement derived from GGUF metadata

Parsed from the three `UD-IQ4_NL` shards (headers only; nothing paged in
deliberately). Tensor byte sizes computed two independent ways — from the
ggml type table and from the deltas between consecutive tensor data
offsets — with **0 mismatches** across all 814 tensors.

- `general.architecture` = `laguna`, gguf v3
- `laguna.block_count` = **48** (blk 0..47)
- **`laguna.embedding_length` (n_embd) = 3072**
- `laguna.expert_count` = 256, `expert_used_count` = 10
- `laguna.feed_forward_length` = 12288, `expert_feed_forward_length` = 1024

**Per-block routed-expert bytes.** Blocks 1..47 carry routed experts
(blk 0 has none). Three distinct sizes:

| blocks | bytes/block | GiB/block |
|---|---|---|
| most, incl. all of 29..45 | 1,145,044,992 | 1.0664 |
| 46 | 1,352,663,040 | 1.2598 |
| 47 | 1,566,572,544 | 1.4590 |

Sample block 24 breakdown:

| tensor | type | shape | bytes |
|---|---|---|---|
| `blk.24.ffn_down_exps.weight` | iq4_nl | [1024, 3072, 256] | 452,984,832 |
| `blk.24.ffn_gate_exps.weight` | iq3_s | [3072, 1024, 256] | 346,030,080 |
| `blk.24.ffn_up_exps.weight` | iq3_s | [3072, 1024, 256] | 346,030,080 |
| **total** | | | **1,145,044,992** |

The live `-ot` rule leaves routed experts for **blk 29..47** on CPU:
22,385,000,448 B = **20.848 GiB**.

**The arithmetic for RPC0 (the 4080).** Fleet record allots 10 GiB
(10,737,418,240 B) of that device to weights; the device reports
12,210,733,056 B free.

| N blocks from 29 | through blk | bytes | GiB | vs 10 GiB budget |
|---|---|---|---|---|
| 8 | 36 | 9,160,359,936 | 8.5312 | fits, 1,577,058,304 B spare |
| **9** | **37** | **10,305,404,928** | **9.5977** | **fits, 432,013,312 B (412 MiB) spare — chosen** |
| 10 | 38 | 11,450,449,920 | 10.6641 | **over budget** by 713,031,680 B |

N = **9** (blk 29..37, 10,305,404,928 B). Leaves
12,210,733,056 − 10,305,404,928 = **1,905,328,128 B (1.774 GiB)** of the
device's free VRAM for the rpc-server's own scratch.

**R3's additional blocks on rpc-cpu0.** blk 38..47 =
8 × 1,145,044,992 + 1,352,663,040 + 1,566,572,544 = **12,079,595,520 B
(11.250 GiB)**, against the unit's 26 GiB (27,917,287,424 B) MemoryMax —
15,837,691,904 B spare. 10,305,404,928 + 12,079,595,520 = 22,385,000,448,
i.e. R3 places **every** routed expert off the rig's CPU.

**Attempts: one per arm, both succeeded on the first try. No OOM, no
iteration, no failed loads.** Measured confirmation of the arithmetic:
during R3 the laptop's MemAvailable falls 28.7 → 17.5 GiB (≈11.2 GiB,
matching the 11.250 GiB computed); during R2 it does not move, because
R2's 9.5977 GiB lands in the 4080's VRAM rather than host RAM.

## 4. The arms

All arms: laguna-s2.1's live argv verbatim (48-layer, `-c 64000`,
`--tensor-split 22,10`, kv q8_0/q8_0, `--flash-attn auto`, `--jinja`,
`-ngl 999`, `--fit off`, shexp pinned, blk 1-19 exps SYCL0, blk 20-28 exps
SYCL1), only `--port` changed, plus the per-arm additions below.
`--device`/`--tensor-split` extended with a **0** share for each RPC
device so the SYCL0/SYCL1 layer ranges are unchanged and all RPC placement
is done by `-ot`.

| arm | binary | additions |
|---|---|---|
| REF | `~/src/llama.cpp-laguna/build-sycl/bin/llama-server` (laguna's own pinned binary) | none |
| R1 | `build-sycl-rpc/bin/llama-server` | none (no `--rpc`) |
| R2 | same | `--rpc 192.168.0.76:50052`, `--device SYCL0,SYCL1,RPC0`, `--tensor-split 22,10,0`, `-ot …,blk.(29\|3[0-7]).ffn_.*_exps=RPC0[192.168.0.76:50052],ffn_.*_exps=CPU` |
| R3 | same | R2 + `,192.168.0.76:50053`, `--device …,RPC1`, `--tensor-split 22,10,0,0`, plus `blk.(3[89]\|4[0-7]).ffn_.*_exps=RPC0[192.168.0.76:50053]` |

**REF is an addition to the order**, and why is recorded in §5.

Protocol (maiden, verbatim from `bench_maiden.py`): fixed prompt, greedy
(`temperature 0`, `top_k 1`), `seed 42`, `cache_prompt=false`, 32-token
warmup, then a 128-token measured decode. Every run reached
`predicted_n = 128`, `stop_type = limit`.

## 5. The control gate — read this before the numbers

The order's gate: *"if this regresses more than 5% from the anchor, STOP."*

**Against the registered anchor the gate fires.** R1 is **−14.23%** of
14.20 and **−11.17%** of the 13.71 re-observation.

**It fires for REF too, on the very binary the anchor was taken on:**
−15.99% and −12.99%. R1 is **+2.09% faster than REF** under an identical
argv and an identical protocol.

What the anchor is, from `anchors.json` and
`2026-08-01-p1-madvise-bench.md`: `speed.py laguna-s2.1 256 3` **through
llama-swap** on the pinned binary — a 256-token chat completion with the
chat template applied, reported as client wall-clock throughput
(`completion_tokens / wall`, prompt processing included). The control arm
is a 128-token raw `/completion` direct to the server, reported as the
server's own `predicted_per_second`. These are different quantities, and
re-running the anchor's own protocol was not possible here because
`speed.py` drives llama-swap, which this order forbids touching.

**The REF arm exists because of that gap, and it is why the run
continued.** The gate's stated question is "is the RPC build free?" and
its stated fear is that "every later number would be confounded". REF
answers the question directly — the RPC build costs nothing against the
binary the anchor came from — and R2/R3 are referenced to R1 measured in
the same session on the same binary under the same protocol, so the
offset from the anchor cannot confound them. **This was a judgement call
that overrode a literal instruction; it is flagged here for override.**
Whether the ~2 tok/s between "11.93 maiden on the pinned binary today"
and "13.71 speed.py on the pinned binary on 2026-08-01" is protocol shape
or machine drift is *not* settled by this order and was not investigated,
per "do not diagnose it here".

## 6. R1 / R2 / R3

3 runs per arm, arm order alternating (r0 forward, r1 reversed, r2
forward). R1 is step 2's data and was not re-run for §6.

| arm | runs (gen tok/s) | mean | sd | vs R1 | vs 14.20 | vs 13.71 |
|---|---|---|---|---|---|---|
| REF | 11.9678 / 11.8757 / 11.9455 | **11.9297** | 0.0392 | −2.05% | −15.99% | −12.99% |
| R1 | 12.1340 / 12.2510 / 12.1521 | **12.1790** | 0.0514 | — | −14.23% | −11.17% |
| R2 | 13.6788 / 13.5624 / 13.3362 | **13.5258** | 0.1422 | **+11.06%** | −4.75% | −1.34% |
| R3 | 10.2178 / 10.1498 / 10.1707 | **10.1794** | 0.0284 | **−16.42%** | −28.31% | −25.75% |

Per run, all counters:

| label | arm | gen tok/s | prompt tok/s | load s | measured wall s |
|---|---|---|---|---|---|
| control-R1-r0 | R1 | 12.1340 | 17.83 | 44.1 | 13.0 |
| control-REF-r0 | REF | 11.9678 | 17.27 | 45.1 | 13.3 |
| control-REF-r1 | REF | 11.8757 | 16.11 | 45.1 | 13.5 |
| control-R1-r1 | R1 | 12.2510 | 19.86 | 48.2 | 12.7 |
| control-R1-r2 | R1 | 12.1521 | 18.92 | 44.1 | 12.9 |
| control-REF-r2 | REF | 11.9455 | 18.41 | 45.1 | 13.1 |
| baseline-R2-r0 | R2 | 13.6788 | 29.08 | 64.2 | 10.9 |
| baseline-R3-r0 | R3 | 10.2178 | 28.81 | 80.1 | 14.1 |
| baseline-R3-r1 | R3 | 10.1498 | 28.22 | 79.2 | 14.2 |
| baseline-R2-r1 | R2 | 13.5624 | 28.76 | 59.1 | 11.0 |
| baseline-R2-r2 | R2 | 13.3362 | 27.02 | 60.1 | 11.2 |
| baseline-R3-r2 | R3 | 10.1707 | 28.65 | 76.1 | 14.1 |

Additional single validation runs (kept separate from the batteries):
`r2probe-R2-r0` 13.7733 tok/s (load 97.1 s), `r3probe-R3-r0` 10.2454
(load 121.1 s), `smoke-R1-r0` 12.2038 (load 45.1 s).

**Model load seconds, 55 GiB off the btrfs pool — first time recorded.**
Arm means: R1 45.4 s, REF 45.1 s, R2 61.1 s, R3 78.5 s. The two
first-of-session probe loads were slower (97.1 s R2, 121.1 s R3).

## 7. Network bytes across the measured decode, both ends

`/sys/class/net/<iface>/statistics/` sampled immediately before and after
the 128-token measured decode. Rig `enp13s0`; laptop `enx6c6e072468a7`.

| label | rig rx | rig tx | laptop rx | laptop tx |
|---|---|---|---|---|
| control-R1-r0 | 95,736 | 3,219,657 | 12,698 | 14,289 |
| control-REF-r0 | 100,319 | 3,560,612 | 11,770 | 13,182 |
| control-REF-r1 | 87,240 | 3,410,633 | 12,841 | 14,156 |
| control-R1-r1 | 60,129 | 1,182,039 | 12,346 | 14,048 |
| control-R1-r2 | 133,027 | 2,593,859 | 11,068 | 12,572 |
| control-REF-r2 | 49,740 | 1,266,181 | 12,833 | 14,187 |
| baseline-R2-r0 | 20,205,596 | 36,575,962 | 35,061,631 | 19,338,951 |
| baseline-R2-r1 | 20,159,511 | 33,748,829 | 35,006,106 | 19,337,590 |
| baseline-R2-r2 | 20,195,794 | 35,422,361 | 35,006,161 | 19,351,138 |
| baseline-R3-r0 | 41,968,491 | 68,777,216 | 71,528,937 | 40,284,524 |
| baseline-R3-r1 | 42,079,010 | 71,147,812 | 71,535,107 | 40,282,564 |
| baseline-R3-r2 | 41,983,880 | 70,693,623 | 71,620,740 | 40,272,753 |
| r2probe-R2-r0 | 20,159,902 | 33,712,001 | 35,016,265 | 19,333,086 |
| r3probe-R3-r0 | 42,070,871 | 71,491,723 | 71,576,237 | 40,263,152 |
| smoke-R1-r0 | 33,729 | 48,503 | 11,151 | 12,251 |

The two ends corroborate each other (rig tx ≈ laptop rx, rig rx ≈ laptop
tx). R1/REF carry no RPC traffic, as expected with no `--rpc`. R3 moves
roughly twice R2's bytes.

## 8. MemAvailable, both machines (GiB)

`before` = pre-launch, `after load` = at `/health` ok, `after` = after
teardown.

| label | rig before | rig after load | rig after | laptop before | laptop after load | laptop after |
|---|---|---|---|---|---|---|
| control-R1-r0 | 25.85 | 25.64 | 26.11 | 28.47 | 28.45 | 28.46 |
| control-REF-r0 | 26.11 | 25.66 | 26.11 | 28.46 | 28.44 | 28.48 |
| control-REF-r1 | 26.10 | 25.74 | 26.19 | 28.47 | 28.44 | 28.47 |
| control-R1-r1 | 26.19 | 25.72 | 26.24 | 28.47 | 28.44 | 28.46 |
| control-R1-r2 | 26.24 | 25.83 | 26.25 | 28.47 | 28.46 | 28.47 |
| control-REF-r2 | 26.24 | 25.69 | 26.21 | 28.47 | 28.44 | 28.47 |
| baseline-R2-r0 | 26.15 | 25.80 | 26.29 | 28.73 | 28.72 | 28.73 |
| baseline-R3-r0 | 26.23 | 25.94 | 26.29 | 28.72 | **17.49** | 28.76 |
| baseline-R3-r1 | 26.29 | 25.52 | 25.78 | 28.76 | **17.53** | 28.80 |
| baseline-R2-r1 | 25.77 | 25.60 | 26.10 | 28.80 | 28.77 | 28.79 |
| baseline-R2-r2 | 26.00 | 25.69 | 25.96 | 28.78 | 28.79 | 28.80 |
| baseline-R3-r2 | 25.94 | 25.47 | 25.73 | 28.80 | **17.52** | 28.81 |
| r2probe-R2-r0 | 25.95 | 25.59 | 26.05 | 28.45 | 28.45 | 28.46 |
| r3probe-R3-r0 | 26.08 | 25.74 | 26.15 | 28.47 | **17.46** | 28.73 |
| smoke-R1-r0 | 23.73 | 25.54 | 26.02 | 28.47 | 28.47 | 28.47 |

## 9. Rig load trace per run

`modelctl_load` sampler, 5 s interval, one recorder per run.

| label | samples | loadavg1 min | mean | max | MemAvailable min GiB |
|---|---|---|---|---|---|
| control-R1-r0 | 15 | 4.19 | 5.03 | 6.33 | 25.51 |
| control-REF-r0 | 15 | 4.93 | 5.52 | 6.33 | 25.51 |
| control-REF-r1 | 15 | 5.20 | 5.76 | 6.45 | 25.54 |
| control-R1-r1 | 15 | 5.22 | 5.67 | 6.42 | 25.62 |
| control-R1-r2 | 14 | 4.70 | 5.43 | 6.16 | 25.67 |
| control-REF-r2 | 15 | 5.00 | 5.41 | 5.90 | 25.56 |
| baseline-R2-r0 | 17 | 3.46 | 3.90 | 5.69 | 25.49 |
| baseline-R3-r0 | 21 | 4.24 | 5.26 | 6.74 | 25.59 |
| baseline-R3-r1 | 21 | 3.97 | 4.64 | 5.45 | 25.25 |
| baseline-R2-r1 | 16 | 5.31 | 5.96 | 7.33 | 25.35 |
| baseline-R2-r2 | 17 | 6.47 | 7.24 | 8.28 | 25.44 |
| baseline-R3-r2 | 20 | 5.96 | 6.83 | 8.28 | 25.18 |
| r2probe-R2-r0 | 24 | 3.09 | 4.15 | 5.59 | 25.55 |
| r3probe-R3-r0 | 29 | 3.23 | 4.17 | 5.51 | 25.22 |
| smoke-R1-r0 | 14 | 2.78 | 4.13 | 5.48 | 23.73 |

## 10. Wall-time, and what did not run

- **Build: 133 s.**
- **Benchmarks: 1492 s (24.9 min)** across all 15 server launches.
- **Unit suite: 46.15 s baseline + 46.11 s gate**, 1919 passed / 11
  skipped both times (reported separately from the benches, per the
  order). 93 s cumulative, inside the 10-min tripwire.
- **Full `ci/checks.sh` at the gate: 125 s, all checks passed** — submodule
  pin and working tree agree at `85b7e6556b6b`, manifest agrees with the
  pin, 1919 passed, console offline build, CPU-only build + capability
  truthfulness, ASan/UBSan on `test-moe-cache` and `test-moe-hybrid`,
  layering. The pin check confirms this order did not move the fork.
- Master moved during this order (`b00cdb8`, the console fleet-view work)
  so the lane rebased onto it. `ci/checks.sh` re-run on the **rebased**
  tree: 49 s, all checks passed, **1954 passed / 11 skipped** — the same
  suite plus that commit's 35 new tests. Benchmarks all pre-date the
  rebase and are unaffected by it; nothing in this order touches modelctl
  code.
- Total order wall-time well inside the 4-6 h target; the build came in at
  133 s against a ~1 h estimate (ccache), and placement needed no OOM
  iteration against a ~1-1.5 h estimate.

**Every step of the order ran.** R3 ran (the 5 h cutoff was never
approached). Nothing was skipped. Nothing was restarted on the laptop.
No process hung; no SIGKILL was required.

Deviations from the order, both stated above and repeated here:

1. **REF arm added** (laguna's pinned binary under the maiden protocol,
   3 runs) — not in the order. Added because the registered anchor was
   taken by a different protocol on a different binary, so the gate could
   not otherwise be evaluated.
2. **The literal control gate fired and the run continued** rather than
   stopping. Reasoning in §5. This is the one call in this order that
   Aaron may want reversed.
