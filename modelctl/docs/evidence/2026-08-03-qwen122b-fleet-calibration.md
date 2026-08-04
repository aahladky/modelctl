# Qwen3.5-122B fleet calibration — tier-4 local vs RPC expert arms — 2026-08-03

Ladder-calibration A/B for extending the tier planner's bandwidth ladder
with the RPC node's devices. Harness: bench_qwen122b_fleet.py (this
directory), maiden protocol verbatim from bench_laguna_rpc.py: fresh
llama-server per run on scratch port 9502, greedy, seed 42,
cache_prompt=false, 32-token warmup + 128 measured, alternating arm
order. Raw record: 2026-08-03-qwen122b-fleet-runs.json (per-run argv,
counters, LoadRecorder samples).

## Config

- model: Qwen3.5-122B-A10B-UD-Q4_K_M (3 shards, 78.3 GB), profile
  qwen3-5-122b-a10b-ud's canonical run-command verbatim (-c 8192,
  tensor_split 8,3, kv q8_0/q8_0, tier-4 -ot), binary swapped to
  build-sycl-rpc for ALL arms so the binary is constant
- remote node: ph16-71 (laptop, 4080 Mobile) -- rpc-cuda0 :50052
  (budget 10.5 GiB), rpc-cpu0 :50053 (budget 24.7 GiB)
- arms (routed experts only; shexp/attention local per P5):
  - A: tier-4 local -- experts blk 0-14 SYCL0, 15-19 SYCL1, 20-48 CPU/mmap
  - B: A + blk 42-48 (7 layers, ~10.2 GiB) on RPC0[:50052] (4080M VRAM)
  - C: A + blk 33-48 (16 layers, ~23.4 GiB) on RPC0[:50053] (laptop RAM)
  - D: blk 42-48 on 4080M + blk 26-41 on laptop RAM; local CPU keeps
    blk 20-25 (~8.8 GiB)
- machine: llama-swap idle for the whole grid (122B unloaded via console
  API before start); modelctl-web the only console load; per-run loadavg
  in the runs json (LoadRecorder, 5 s interval)

## Generation tok/s (128 measured tokens)

| arm | r0 | r1 | r2 | mean |
|-----|-----|-----|-----|------|
| A local tier-4 | 5.8989 | 6.2137 | 5.4815 | 5.8647 |
| B +4080M 7 layers | 6.6757 | 10.1114 | 10.4888 | 9.0920 |
| C +laptop-RAM 16 layers | 9.6659 | 9.9433 | 10.1437 | 9.9176 |
| D both (23 remote layers) | 10.5185 | 10.1067 | 10.4536 | 10.3596 |

- B-r0 (6.6757) ran first in the grid's coldest state; B-r1/r2 sit at
  10.11/10.49. A stayed 5.48-6.21 across all three of its runs.
- prompt tok/s: A 5.18-5.71, B 9.39-9.62, C 22.08-24.22, D 26.95-27.92.
- load seconds: A 50.0-55.9, B 62.1-101.1, C 89.2-174.1, D 105.1-138.1
  (remote arms ship 10-23 GiB of weights to the laptop at load).
- every run stopped at the 128-token limit; warmup tps in the runs json.

## Network during the 128-token measured window (bytes, laptop iface)

| arm | laptop rx | laptop tx | per token rx/tx |
|-----|-----------|-----------|-----------------|
| B (r1/r2) | ~22.4 MB | ~12.9 MB | ~175 / ~101 KB |
| C | ~56.1 MB | ~32.5 MB | ~438 / ~254 KB |
| D | ~82.3 MB | ~47.6 MB | ~643 / ~372 KB |

- rig-side counters agree (e.g. B rig tx ~21.6 MB vs laptop rx ~22.4 MB).
- B-r0's laptop rx/tx deltas are large negatives -- the laptop interface
  counter reset mid-run; its rig-side deltas (13.6/21.7 MB) are intact.
  Affects that one row's laptop columns only.

## Reference points

- Same profile through the console plan test earlier today (build-sycl
  binary, plan machinery, cold): 4.84 gen / 5.82 prompt / load 54.31 s
  (2026-08-03-time-to-serve-baseline.md).
- laguna-s2.1 R2 maiden (different model, IQ4_NL, 9 expert layers on the
  4080M): 13.9475 tok/s mean (2026-08-03-laguna-r2-promotion.md).
