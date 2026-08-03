# Laguna R2 promotion + maiden re-anchor — 2026-08-03

Promotion applied on Aaron's word ("do them all"): laguna-s2.1 moved off
the old private fork onto the repo's build-sycl-rpc binary at the
ratified pin f3e7141dd048, with the laptop 4080 in the loop over RPC.

## Config applied (profile laguna-s2.1)

- binary: /home/aaron/workspace/moe-serving/llama.cpp/build-sycl-rpc/bin/llama-server
- tensor_split: 22,10,0 (split_mode layer, ctx 64000, kv q8_0/q8_0)
- extra: `--rpc 192.168.0.76:50052 --fit off --device SYCL0,SYCL1,RPC0
  -ot <shexp pins>,blk\.(29|3[0-7])\.ffn_.*_exps=RPC0[192.168.0.76:50052],ffn_.*_exps=CPU`
- NOTE: `--rpc` must precede `--device` in the argv -- the first sync had
  it after, and llama-server died at parse with "invalid device: RPC0"
  (the RPC backend registers its devices when --rpc is parsed).

## Maiden runs (bench_laguna_rpc.py --arms R2 --runs 5)

Fresh server per run on a scratch port, greedy, seed 42, 32-token warmup
+ 128 measured, router idle, modelctl-web the only other console load.
Raw record: 2026-08-03-laguna-r2-runs.json (this directory).

| run | gen tok/s | prompt tok/s | load s |
|-----|-----------|--------------|--------|
| r0  | 13.7401   | 25.59        | 66.4   |
| r1  | 13.9909   | 27.96        | 58.2   |
| r2  | 13.9620   | 28.40        | 59.2   |
| r3  | 13.6709   | 28.24        | 58.2   |
| r4  | 14.3735   | 28.21        | 60.2   |

mean generation 13.9475 tok/s. Rig->laptop RPC traffic ~33.6 MB tx /
~20.3 MB rx per run (measured deltas in the json record).

## After

- Served end-to-end through llama-swap: completion answered, worker
  fingerprint b10220-f3e7141dd.
- ~/src/llama.cpp-laguna deleted (1.5 GiB) after grep of live profiles
  and config.yaml showed zero references outside dated .bak files.
- Old anchor laguna-s2.1-canary (14.2 via speed.py on the deleted
  binary) is void for gating; laguna-s2.1-r2-maiden replaces it.
