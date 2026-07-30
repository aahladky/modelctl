# Task 0.4/0.5 — Architecture corrections and post-sync build matrix

Applied on top of the Task 0.3 port (`sync/moe-cache-upstream-2026-07-30`,
now at `db07b9eb7` in the worktree at
`llama.cpp/.claude/worktrees/upstream-sync-phase0`). Two new commits:

- `5c2a5feee` — sycl: rework MoE cache ownership and expose it via a versioned backend API
- `db07b9eb7` — common, server: consume the MoE cache via the backend API, fix probe truthfulness

## Task 0.4 — architecture corrections

1. **Weak globals → versioned backend-registry API.** The cache was
   configured via `extern volatile` globals declared
   `__attribute__((weak))` so non-SYCL builds would still link. That's
   fragile specifically for `GGML_BACKEND_DL=ON`: a weak `extern` in the
   main executable does not resolve to a symbol that exists only in a
   separately `dlopen`'d backend `.so`. Replaced with procs exposed
   through `ggml_backend_reg_get_proc_address` — the same mechanism
   upstream already uses for the meta-backend's `comm_init`/`comm_free`
   and the CPU backend's `set_n_threads`. New typedefs and a config
   struct in `ggml-backend.h`; `ggml_backend_moe_cache_configure_v1` and
   friends implemented in `ggml-sycl.cpp`; a small resolver
   (`tools/server/moe-cache-iface.h`) used by both `server.cpp` and
   `server-context.cpp`. **Verified, not just asserted:** built a
   `GGML_BACKEND_DL=ON` variant and confirmed the dynamically-loaded
   `libggml-sycl.so` correctly resolves and the capability probe reports
   the cache as available — see build matrix below.
2. **Truthful, backend-derived capabilities.** `--modelctl-capabilities`
   inferred "cache implemented" from `has_sycl = !sycl_devices.empty()`,
   which is only truthful today because `CMakeLists.txt` unconditionally
   compiles `GGML_MOE_EXPERT_CACHE` into every SYCL build. Replaced with a
   direct query — does any registered backend resolve
   `"ggml_backend_moe_cache_configure_v1"`? Also fixed the
   `moe_cache_backend` field to report the actual resolving backend's
   name (`ggml_backend_reg_name`) instead of a hardcoded `"SYCL"` string.
3. **Cache ownership tied to a safe lifetime model.** The cache was a raw
   one-pointer-per-device array (`g_moe_cache_instances`), deleted
   whenever *any* context on that device freed — a second context on the
   same device (main + draft/MTP) would be left with a dangling pointer
   the moment the first context unloaded. Replaced with a device-level
   registry of `std::weak_ptr<moe_expert_cache>` guarded by a mutex; each
   context holds its own `std::shared_ptr` (`moe_cache_shared` in
   `common.hpp`). Chose ref-counted-shared-by-contexts over
   context-owned because upstream already treats per-device resources
   this way (`ggml_sycl_device_info` via `dpct::dev_mgr::instance()`, a
   device-keyed singleton independent of any one context) — this
   followed an existing pattern rather than inventing one, so it didn't
   need a stop-and-ask. **Left explicitly open, because it can't be
   settled by reading code alone:** whether a second context's compute
   thread submitting through a cache created against the first context's
   queue is safe on this hardware. That's exactly what Task 0.6's
   multi-context correctness runs are for — flagged, not asserted either
   way.
4. **Allocation.** Added a null-pointer check after `sycl::malloc_device`
   (the existing `try/catch` only covers the throwing-on-failure case).
   Did **not** switch to `ggml_sycl_pool_alloc` (the pool wrapper used
   elsewhere in this file, e.g. the tensor-parallelism comm buffers):
   that wrapper is bound to one context's `ggml_sycl_pool`, which would
   reintroduce exactly the context-lifetime coupling item 3 just removed.
   Raw USM device allocation is the right tool for a device-scoped
   resource; the pool allocator is for context-scoped scratch buffers.
   Documented in the commit message rather than forcing a wrong-fit
   change.
5. **Scheduler interception point / batch-size assumption.**
   Re-confirmed (originally established during the Task 0.3 port): the
   cache hook in `ggml_backend_sched_compute_splits`'s `copy_experts`
   lambda is byte-identical to the feature's original base and untouched
   by `dee2a846b` (which lives in `ggml_backend_sched_backend_id_from_cur`,
   a different function). The `moe_cache_min_batch: 32` constraint is a
   capability-response constant, not logic derived from scheduler
   internals, so it isn't affected by upstream's scheduler changes
   either. No code change needed; assumption still holds.
6. **MTP/main-model coexistence.** Read `0324696b8` ("fit: count nextn
   (MTP) blocks in n_gpu_layers"): it only changes `common/fit.cpp`'s
   VRAM-fit layer-count heuristic (`hp_ngl`), which the cache never
   consults — the cache sizes its layer-index array to a fixed generous
   upper bound (256) and identifies layers by parsing tensor names
   (`"blk.N.ffn_*_exps"`), not by any layer-count total. No interaction,
   no fix needed. Whether MTP/NextN tensor naming could ever collide
   layer numbers with the main model is a model-format question outside
   this cache's control and outside what reading code here can answer —
   consistent with Task 0.6 needing a real "main plus draft/MTP context"
   run.
7. **`moe_hybrid_cpu_miss` stays `false`.** Confirmed: `moe-hybrid.cpp`'s
   dispatch scaffolding remains unreferenced by any call site; nothing in
   this pass changed that.

**Bonus fix found while rewriting the configure path:**
`cfg.prefill_admit` was never set from the server's configuration at all
in the pre-existing code — there was no field for it in the old ad hoc
global set, so `--moe-cache-prefill-admission on` was silently a no-op
despite the capability response claiming `moe_cache_prefill_policy: true`.
Now threaded through `ggml_backend_moe_cache_config.prefill_admit`.

**Minor, intentional behavior change:** `post_cache_reset` used to key
"not supported" off `g_moe_cache_budget_bytes == 0` (now encapsulated,
not externally readable). It now returns `NOT_SUPPORTED` only when no
backend implements the cache at all, and a legitimate `200` with
`reset_devices: 0` when the feature exists but nothing was ever
initialized — arguably more correct, since "not supported" should
describe the build/backend, not whether a model happened to load MoE
tensors yet.

## Task 0.5 — post-sync build matrix

| Variant | Flags | Result | Capability check |
|---|---|---|---|
| CPU-only | `GGML_SYCL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF` | Builds clean | All `moe_*` features `false`, `moe_cache_backend: ""` — confirmed |
| SYCL, cache compiled, `GGML_BACKEND_DL=OFF` | same as pre-sync `build-sycl` | Builds clean | Identical JSON to pre-sync baseline (see `2026-07-30-port.md`), still identical after the 0.4 changes |
| SYCL, `GGML_BACKEND_DL=ON` | `-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BACKEND_DL=ON` | Builds clean | **Dynamically-loaded** `libggml-sycl.so` correctly resolves the cache proc; capability probe reports `moe_weight_transfer_cache: true` on both SYCL0/SYCL1 exactly as the static build does — this is the direct validation of item 1's whole rationale |
| Debug/assertion-enabled | `CMAKE_BUILD_TYPE=Debug`, SYCL, `GGML_BACKEND_DL=OFF` | `ggml-sycl` target builds clean (100%, only pre-existing benign `dpct/helper.hpp` warnings) | Not run (per Task 0.5's own scope, compile-only is sufficient) |

CPU-only capability response (full):

```json
{
  "schema": 2, "backend": "llama.cpp",
  "devices": [{"type": "CPU", "name": "CPU", "index": 0, "features": {"moe_weight_transfer_cache": false}}],
  "features": {
    "moe_weight_transfer_cache": false, "moe_hybrid_cpu_miss": false,
    "moe_cache_metrics": false, "moe_cache_prefill_policy": false,
    "moe_cache_reset": false, "moe_cache_prefetch": false
  },
  "constraints": {"moe_cache_backend": "", "moe_cache_min_batch": 32, ...}
}
```

This satisfies the roadmap's hard acceptance bar: "The CPU-only
capability response must report every SYCL/cache capability as false."

## Not done (explicitly out of scope here)

Full CPU-variant matrix beyond one representative build, running the
Debug binary, and the deterministic correctness matrix (real model,
actual hit/miss/eviction/reset/multi-context behavior) — that's Task
0.6, still open, and is what will actually answer the two questions this
pass could only flag: fused-topk-MoE on/off equivalence, and cross-queue
cache sharing safety between two contexts on one device.
