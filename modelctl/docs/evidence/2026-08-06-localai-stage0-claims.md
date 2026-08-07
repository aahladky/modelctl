# LocalAI Stage 0 — claims 1-4 — 2026-08-06

Stage 0 of the "fork LocalAI as the moe-serving front end" spike. Four claims,
proven or measured before any fork is created. Companion to
`2026-08-06-localai-stage0-m1-rebase-cost.md`, which measures the rebase cost.

Nothing was landed on the llama.cpp fork. All source changes live in a
disposable LocalAI clone at `~/workspace/localai-spike/`.

## Config

- LocalAI v4.8.1 (`8052c950`), built from source: `local-ai` 158 MB with the
  React UI embedded; Go 1.26.5
- Backend: LocalAI's `grpc-server` built against the **merged tree**
  (our MoE fork `f3e7141dd` merged onto LocalAI's llama.cpp pin `221f0f635`),
  `BUILD_TYPE=sycl_f32` flags per `backend/cpp/llama-cpp/Makefile:74-78`
- Direct-arm binary: `llama-server` from the *same* merged tree
  (`~/.cache/modelctl/ci/build-sync-221f0f635-clean/bin/llama-server`), so the
  only variable between arms is the serving harness
- Model: Qwen3.5-35B-A3B-UD-IQ4_XS, 16.29 GiB
- Machine: 2x Intel Arc (SYCL0/SYCL1), 28 threads, 31 GiB RAM, oneAPI 2026.1.
  llama-swap empty, no other inference process (a stray `tiny-moe`
  llama-server was stopped by the operator before the measurement).

## Two source changes required, both in LocalAI

1. **`find_package(Protobuf CONFIG REQUIRED)` -> module mode.** Fedora 44 ships
   protobuf 3.19.6 with only `protobuf.pc` and CMake's `FindProtobuf` module;
   there is no `protobuf-config.cmake`, so CONFIG mode hard-fails before
   anything compiles. Module mode defines the same `protobuf::libprotobuf`
   target the file links. Verified: `gRPC` 1.48.4 CONFIG resolves independently,
   and `gRPC::grpc++`, `gRPC::grpc++_reflection`, `absl::flags` all exist.
   The gRPC version gap (1.48.4 vs LocalAI's expected 1.65.0) caused **no** API
   trouble -- `grpc-server.cpp` uses only long-stable surface.
2. **MoE cache configure call in `grpc-server.cpp`'s `LoadModel`.** Upstream
   llama-server configures the cache inside `llama_server()`
   (`tools/server/server.cpp:172`); LocalAI's backend supplies its own `main()`
   and never calls it. Without the patch `--moe-cache-bytes` parses cleanly
   into `params` and is then silently dropped: no cache constructed,
   `stats_json` returns `[]`, no error anywhere. The patch reports either way
   (`CONFIGURED` / `REQUESTED BUT NO CAPABLE BACKEND`) because the defect it
   fixes was a silent one.

## Claim 1 -- LocalAI runs our fork's binary: PASS

`LOCALAI_EXTERNAL_GRPC_BACKENDS=moe-sycl:<path>` registers the binary with no
OCI image. LocalAI spawned it and served a real chat completion: 68 s including
the 16.29 GiB load, 8 tokens returned. Backend links `libggml-sycl.so.0` and
`libsycl.so.9`; 18 `moe_cache` symbols present.

## Claim 2 -- placement flags survive the passthrough: PASS

Options reach `common_params_parse` as real argv. Verified two ways:

- flags present in the backend invocation: `-ngl:99`, `-ot:blk\..*_exps=CPU`,
  `--moe-cache-bytes:SYCL0=8589934592`, `--moe-cache-policy:slru`
- **physically**: backend RSS 15.16 GiB of a 16.29 GiB model, i.e. ~93 % of the
  weights resident in host RAM. That only happens if
  `-ot blk\..*_exps=CPU` was applied; had it been dropped the weights would sit
  in VRAM.

Syntax note: value-carrying flags use `flag:value`; everything after the first
colon becomes the next argv entry, so regexes containing `.` `\` `=` `,` pass
through intact.

## Claim 3 -- the MoE cache engages: PASS (configured), with a caveat

`[llama-cpp] MoE expert cache CONFIGURED: budget=0 policy=slru admission=2
prefill=0 hybrid=0 per_device=1`

`CONFIGURED` means `procs.available()` was true -- the SYCL backend registered
the cache procs and accepted our config. `budget=0 per_device=1` is correct for
the map form (`SYCL0=...`): the scalar budget stays 0 and the per-device map
carries the entry.

**Caveat:** configured is not exercised. `MOE_CACHE_STATS []` -- zero hits, zero
misses across every measured run. The direct arm's `/metrics` likewise reported
no cache counters. This is consistent with the existing anchors (c1-static-35b
38.983 vs c2-cache-35b 24.822): on this model the cache is a loss and does not
engage under static placement. Testing cache *behaviour* through LocalAI needs a
model in the 50-70 GiB band, which is the Laguna case that does not fit this
machine's 31 GiB of host RAM.

## Claim 4 -- throughput parity: MEASURED, but on an unrepresentative config

**Read the limitation before the number.** This was run with routed experts
forced to system RAM (`-ot blk\..*_exps=CPU`) on a 16.29 GiB model that fits
entirely in the B70's 32 GiB. Nobody would run this model that way -- it would
go fully GPU-resident under stock llama.cpp with no `-ot` override at all.

That choice was made to exercise the MoE cache, and it failed at that too: the
cache never engaged (0 hits / 0 misses, see claim 3). So the configuration
bought no cache coverage and introduced a confound, because CPU-resident
experts are precisely what makes generation sensitive to thread count -- and
thread count turned out to be the single largest term in the gap.

The figures below therefore describe a configuration matching neither real
case: the 35B does not need CPU offload, and laguna, which does, will not fit
this machine's 31 GiB of host RAM. They are NOT a general statement about
LocalAI's serving path. The test that would answer that -- same pairs, model
fully GPU-resident, no `-ot` -- was not run (operator's call, 2026-08-06:
claims 1-3 answered the question that mattered).

8 alternating pairs, fresh server per run, order alternated per pair. Both arms:
threads 14, ctx 8192, `-ngl 99`, `-ot blk\..*_exps=CPU`, greedy, temp 0,
seed 42, `cache_prompt=false`, 32-token warmup + 128 measured.
LocalAI arm additionally `fit_params:false`, `kv_unified:false`,
`cache_idle_slots:false` (see elimination chain).

| pair | order | ours engine | localai engine | ours client | localai client | localai cache hits |
|---|---|---|---|---|---|---|
| 0 | ours-first | 37.43 | 28.45 | 34.57 | 26.79 | 0 |
| 1 | localai-first | 32.94 | 27.94 | 30.73 | 26.39 | 0 |
| 2 | ours-first | 38.22 | 20.25 | 35.66 | 19.12 | 0 |
| 3 | localai-first | 37.11 | 22.51 | 34.72 | 21.17 | 0 |
| 4 | ours-first | 36.01 | 22.46 | 33.37 | 21.12 | 0 |
| 5 | localai-first | 35.14 | 22.12 | 32.79 | 20.85 | 0 |
| 6 | ours-first | 35.84 | 18.93 | 33.53 | 17.65 | 0 |
| 7 | localai-first | 35.71 | 22.04 | 32.99 | 20.76 | 0 |

| metric | ours mean | localai mean | median delta | median relative | sign test |
|---|---|---|---|---|---|
| engine_tps | 36.05 | 23.09 | -13.61 | -37.8 % | n=8, positive=0, p=0.0078 |
| client_tps | 33.54 | -- | -12.24 | -36.5 % | n=8, positive=0, p=0.0078 |

Spread: ours 32.94-38.22 (16 %), localai 18.93-28.45 (50 %). The LocalAI path is
both slower and markedly less consistent run to run.

`engine_tps` is llama.cpp's own `print_timings` generation rate and **excludes**
transport; `client_tps` is client-observed wall clock and includes it. Both move
together, so this is not gRPC transport overhead.

### Elimination chain

Each step measured, not assumed:

| configuration | localai engine_tps | note |
|---|---|---|
| as-shipped defaults | 4.81 | LocalAI resolved `n_threads=28` vs the direct arm's 14 |
| threads pinned 14 = 14 | 23.31 | thread mismatch was ~half the apparent gap |
| + `kv_unified:false`, `cache_idle_slots:false` | 27.25 (single run) | LocalAI flips both on by default |
| 8-pair paired mean | 23.09 | the single run above was an optimistic outlier |

Eliminated as a cause: **batch geometry.** Running the direct arm with
LocalAI's `-b 512 -ub 512` cost only 4.5 % (37.84 -> 36.13), nowhere near the
gap.

Roughly 25-38 % remains unexplained after matching every knob identified.
Not attributed here -- attribution needs profiling this did not do.

## Not covered

- No profiling of the residual gap. Candidates untested: `n_ctx_checkpoints`,
  `checkpoint_min_step`, `cont_batching`, the gRPC server's own inference loop.
- Cache *behaviour* through LocalAI untested (see claim 3 caveat).
- Single model, single quant, single machine, n=8.
- The -4.34 / -5.0 deltas in pairs 0-1 versus -16 to -18 later suggests a
  time-varying factor (thermal, page cache, or backend state) that this design
  does not isolate. Pair order was alternated, so it is not arm-order bias.
- LocalAI captured only ~20 lines of backend stderr per load: llama.cpp's own
  model-load diagnostics, including the tensor placement report, do not reach
  the operator. Placement had to be established by measuring process RSS.
