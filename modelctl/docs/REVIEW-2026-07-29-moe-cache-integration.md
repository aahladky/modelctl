# Review packet — MoE expert cache + modelctl integration (2026-07-29)

Everything below is committed and pushed to the local gitea
(`http://localhost:3000/moe-serving`, org **moe-serving**):

| Repo | Branch | Range | Notes |
|---|---|---|---|
| moe-serving/llama.cpp | `feature/sycl-moe-expert-cache` | `f5f930be8..f42f2fe4e` (10 commits) | fork of ggerganov/llama.cpp @ `6f4f53f2b` |
| moe-serving/modelctl | `master` | `17860b2..57e775f` (15 commits) | pins the fork as a submodule at `f0750e8a4` |

Suggested reading order: §1 (what shipped), §3 (the one critical bug),
§5 (integration evidence), §6 (open issues).

---

## 1. What this is

A per-GPU MoE expert weight cache for SYCL llama.cpp, plus the modelctl
control-plane work to plan for it, launch it, and observe it from the
web console. Target workload: models too big for VRAM, experts spilling
to RAM (the web-first plan's CPU-spill case).

Architecture (fork): a fixed pool of USM device slots per SYCL device,
keyed by (layer, expert, projection), SLRU eviction with
probationary/protected segments, second-miss admission, prefill
protection. Entry point is a **global scheduler hook**
(`ggml-backend.cpp:1644`) that intercepts the selective host→device
expert copies upstream already does for CPU-resident experts; hits fill
`input_cpy` device-to-device, misses promote then fill. Server adds
`--moe-cache-bytes/-policy/-admission-misses/-prefill-admission`,
Prometheus `llamacpp:moe_cache_*` metrics, `/cache/reset`, and a
`--modelctl-capabilities` probe that modelctl consumes.

## 2. llama.cpp fork — commit inventory

Feature work (pre-existing, reviewed as a batch):
- `f5f930be8` probe + cache skeleton
- `241371396` wire into mul_mat_id, CLI args, lazy init
- `250391098` metrics endpoint, reset, stats JSON fix
- `de47d5967` M4: SLRU, admission, prefill protection, projection-aware slots
- `d6f00dfbb` F1–F5 fixes
- `496157935` revert of first hook attempt (contiguous-buffer gaps)
- `a64c80eee` Laguna bench (superseded numbers, see BENCH doc)
- `67ab58096` F0: scheduler hook re-done (fills input_cpy on every copy)

Review-driven fixes (this round):
- `f0750e8a4` **fix review round 2** — see §3
- `f42f2fe4e` drop the self-diffing 7,419-line patch snapshot

## 3. The critical bug (C1) and other round-2 fixes

**C1 — wrong weights on every up/down cache hit.** Slots store gate/up/
down at different offsets, but `lookup()` returned the slot base. 2 of 3
MoE matmuls per layer computed with gate weights. Throughput benchmarks
couldn't see it; output quality silently degraded as hit rate climbed.
Fixed by returning `base + projection_offset` from both `lookup()` and
`promote_projection()`. **Verified post-fix by bit-identical outputs
(§5).**

Also in `f0750e8a4`:
- M1: mutex over cache state — `/cache/reset` and `/metrics` no longer
  race the compute thread.
- M2: lazy init moved into the hook (parses the original tensor name);
  pure-CPU-expert configs now initialize.
- M3: hook registration gated on budget > 0 — zero overhead when off.
- M4: `ggml_backend_is_sycl()` check before the context cast.
- M5: inline `mul_mat_id` cache path removed (hook is sole entry);
  obsolete fused-decode gating removed (fused kernel re-enabled).
- Minors: padding tail on hits, `cpu_expert_calls` wired / dead counters
  removed, cache freed on backend free, promote geometry validation,
  padding only on group tail, probe reports implemented features true,
  non-SYCL builds link via weak-symbol stubs.

Build: `cmake --build build-sycl --target llama-server` clean (oneAPI
2026.1, Release); no new warnings in touched files.

## 4. modelctl — commit inventory

Milestones (rewritten clean; original `dfc495e` had swept 47k files of
workspace junk — venv, arduino binaries, orphan gitlinks — into git;
history rewritten, junk untracked not deleted, `backup/pre-cleanup`
branch retained locally):
- `17860b2` M2: cache-aware tier planning (reserve cache budget before
  static experts)
- `d2b6f86` M5: web telemetry (cache metrics, reset, variants)
- `bb76106` gitignore for stray workspace dirs
- `db5c6aa` **budget semantics fix**: fork's `--moe-cache-bytes` is a
  uniform per-GPU budget → runtime emits `max()` not `sum()`; planner
  reserves uniformly; tier-apply writes effective budgets back to the
  profile so plan and runtime can't diverge; structured flags dedupe
  raw `extra` flags
- `4d5aecf` tests for M2/M5
- `52dc4ce` restored the web-first plan doc (was never committed)
- `0d878d2` llama.cpp submodule pin + combined-project README
- `d12b981`/`422fc5b`/`57e775f` bench doc (recovered from commit
  messages; latest results; post-fix validation)
- `28efbb0` runtime.html slots gauge key
- Integration-test fixes — see §5 bug list: `3fb880d`, `7a23fda`,
  `18a6fb9`, `9f0d276`

Tests: 382 passing (`test_modelctl`, `test_modelctl_vram`,
`test_modelctl_tiers`, `test_modelctl_capabilities`,
`test_modelctl_web`), incl. new coverage for cache_request planning,
variant feasibility, metrics parsing/reset, probe env fallback.

## 5. Integration test evidence (live, web-driven)

Setup: `qwen3-5-122b-a10b-ud` (122B-A10B, IQ1_M), 19/48 expert layers
forced CPU via `-ot`, 1 GiB cache, SLRU, admission=1. Driven through the
modelctl web API (`:9293`) and llama-swap (`:9292`).

**Correctness A/B (the C1 verdict):**
| comparison | content | reasoning |
|---|---|---|
| cache-off vs cache-on (long prompts) | **identical, all** | 1/3 diverged |
| cache-off vs cache-off (control) | **identical, all** | 1/3 diverged |

The divergence rate is identical with and without cache → baseline SYCL
nondeterminism, not cache corruption. All runs temp=0, fixed seed.

**Cache activity (proof it engages):** 6,003 hits / 57,012 misses,
441/441 slots, evictions active, `hit_ratio 0.096`. `/cache/reset`
zeroed counters (direct and via the web UI proxy). `/runtime` page
renders `SYCL0: hit 0.096 (441/441)`.

**Throughput here: neutral** (prompt ~130–210 t/s, gen ~22 t/s either
way) — 1 GiB thrashes on this geometry. Laguna's +487%/+169% does not
generalize; budget sizing is per-model.

**Bugs found by the testing itself** (all fixed, tested, pushed):
1. `3fb880d` — probe ran SYCL binaries without oneAPI env → crash →
   cached "unsupported" → cache variants silently never generated.
2. `3fb880d` — variant feasibility used full-model-vs-one-card, killing
   variants for exactly the spill models the cache targets; now KV +
   overhead + budget. Plus variant label collision fix.
3. `7a23fda` — `--metrics` never emitted → `/metrics` 501 → telemetry
   permanently blank.
4. `18a6fb9` — `/runtime` crashed the whole service: `load_profile`
   raises `SystemExit` (not `Exception`) for llama-swap models with no
   profile (`fast-7b`).
5. `9f0d276` — llama-swap `/running` has no `port` key (it's in
   `proxy`); scrape never fired.

## 6. Open issues / risks (not fixed)

- **Cache only engages for batches ≥ ~32 tokens** (upstream
  op-offload threshold). Single-user decode (batch 1) with CPU-resident
  experts always runs on CPU, uncached. The F0 "+169% gen" claim is
  inconsistent with this — re-check how it was measured. Today the cache
  is a prefill/batched-decode accelerator only.
- **Poolside fork hazard**: `~/src/llama.cpp-laguna` carries pre-C1-fix
  cache code yet its probe claims cache support. Laguna profiles must
  keep `moe_cache.mode: off` (done) until the round-2 fixes are ported
  or the fork is retired. Also: the new fork cannot load `laguna` arch —
  the pin is still required for the model itself.
- **Non-variant plans emit cache flags without a capability check** — a
  pinned binary lacking support would fail at launch on unknown flags.
- **`POST /api/profiles/{name}` rejects nested `config`** — flat keys
  only ("ignored unknown field: config"); API footgun.
- `load_profile`'s `sys.exit(1)` remains a kill-the-server hazard in
  every other web handler that takes a user-supplied profile name; only
  the `/runtime` iterator was hardened.
- `repomix-output.xml` untracked in modelctl/; `backup/pre-cleanup`
  branch to delete when comfortable; submodule pin is 2 commits behind
  the fork tip (advance after you're happy with validation).
- Cache hit ratio at 1 GiB is poor on qwen3-5-122b (thrash). Default
  budget guidance probably belongs in the planner (per-model working-set
  estimate), not hardcoded fractions.

## 7. How to re-verify quickly

```
# fork build + probe
cd ~/workspace/llama.cpp && cmake --build build-sycl -j --target llama-server
source llama-sycl-env.sh && ./build-sycl/bin/llama-server --modelctl-capabilities

# modelctl tests
cd ~/workspace/modelctl && .venv/bin/python -m unittest \
  test_modelctl test_modelctl_vram test_modelctl_tiers \
  test_modelctl_capabilities test_modelctl_web

# live check (services already running fixed code)
curl localhost:9292/v1/models          # llama-swap
curl -H "Authorization: Bearer $(cat ~/.local/share/modelctl/web_token)" \
  localhost:9293/runtime               # cache column on qwen3-5-122b row
```
