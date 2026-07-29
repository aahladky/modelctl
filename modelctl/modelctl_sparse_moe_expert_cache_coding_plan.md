# modelctl Sparse-MoE Expert Caching and SSD-Backed Inference Plan

**Status:** Proposed implementation plan  
**Target repository:** [aahladky/modelctl](https://github.com/aahladky/modelctl)  
**Prepared:** July 28, 2026  
**Primary target hardware:** heterogeneous Intel SYCL GPUs, system RAM, and NVMe-backed model storage

---

## 1. Executive decision

Implement sparse-MoE expert caching **alongside `modelctl`, not inside its Python process**.

`modelctl` should remain the control plane:

- inspect GGUF structure;
- understand available GPUs, RAM, and storage;
- generate candidate launch plans;
- reserve memory for dynamic expert caches;
- select and validate a compatible inference binary;
- expose configuration and status in the web console;
- launch, test, benchmark, compare, and persist results.

A cache-capable `llama.cpp` fork should remain the data plane:

- observe router-selected expert IDs during inference;
- keep persistent expert slots in GPU memory;
- map `(layer, expert)` identities to those slots;
- execute cache hits on the GPU;
- execute misses on the CPU without forcing a synchronous PCIe copy;
- use RAM and mmap-backed storage as lower tiers;
- expose cache metrics to `modelctl`.

This boundary matches the current repository. `modelctl` already has GGUF-aware tier planning, per-profile binary pins, managed launch plans, resource claims, hardware/backend fingerprints, plan testing, persistent observations, fallback, and a FastAPI/HTMX web console. The new feature should extend those abstractions rather than create a second runtime-management path.

### Recommended delivery strategy

1. **Integrate an experimental backend contract and measurement path first.**
2. **Add cache-aware launch plans and resource claims.**
3. **Implement the persistent expert cache in a SYCL `llama.cpp` fork.**
4. **Use mmap and the Linux page cache as the initial SSD/RAM tier.**
5. **Only add explicit RAM-cache and direct-I/O management after measured evidence shows the kernel page cache is the bottleneck.**
6. **Add expert prediction and prefetching only after the cache is correct and useful without prediction.**

The first production-worthy version should prioritize correctness, observability, and reproducibility over a clever eviction algorithm.

---

## 2. Why sparse MoE is the right target

Dense model decoding generally needs to read almost every model weight for every generated token. Streaming dense weights from SSD therefore remains limited by storage bandwidth.

Sparse MoE models are different:

- routed experts account for a large share of total model bytes;
- only a small subset is selected for each token;
- shared weights, attention, routers, norms, embeddings, and KV cache can remain resident;
- expert use is often skewed and temporally correlated;
- inactive experts can live in slower memory tiers.

The useful hierarchy is:

```text
GPU VRAM
├── attention, routers, embeddings, norms
├── shared experts
├── KV cache and compute buffers
├── optional statically pinned routed experts
└── persistent dynamic routed-expert cache

System RAM
├── CPU-executable expert weights
├── page cache for mmap-backed GGUF data
└── pinned staging buffers, where useful

NVMe storage
└── complete GGUF / cold expert backing
```

The critical rule is:

> SSD access must be a relatively rare cache-miss path, not a required operation for every expert at every token.

Relevant work:

- [llama.cpp expert-cache RFC #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- [llama.cpp two-tier expert-cache issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- [MoE-Infinity](https://arxiv.org/abs/2401.14361)
- [Fiddler](https://arxiv.org/abs/2402.07033)
- [FlashMoE](https://arxiv.org/abs/2601.17063)
- [Fate](https://arxiv.org/abs/2502.12224)
- [KTransformers heterogeneous expert execution](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)

---

## 3. Current `modelctl` capabilities to preserve

The current codebase already provides most of the control-plane foundation.

### 3.1 GGUF-aware tier planning

`modelctl_tiers.py` currently:

- distinguishes four placement tiers;
- uses mmap for Tier 4 SSD-backed loading;
- detects MoE layouts;
- separates fixed/non-expert tensors from routed-expert tensors;
- assigns whole routed-expert layers to GPUs in bandwidth order;
- sends remaining expert layers to CPU;
- accounts for SYCL-specific multi-device and tensor-override behavior.

Source:

- [`modelctl_tiers.py`](https://github.com/aahladky/modelctl/blob/master/modelctl_tiers.py)

### 3.2 Per-profile binary selection and arbitrary server arguments

Profiles can pin a custom `llama-server` binary, and `build_server_args()` already emits structured settings plus raw `extra` flags.

Sources:

- [`modelctl.py` preflight and profile schema](https://github.com/aahladky/modelctl/blob/master/modelctl.py)
- [`build_server_args()`](https://github.com/aahladky/modelctl/blob/master/modelctl.py)

### 3.3 Managed launch plans

`modelctl_plans.py` already defines:

- `ResourceClaim`;
- `LaunchPlan`;
- deterministic plan IDs;
- candidate generation;
- resource estimation;
- policy filtering;
- observation-aware ranking.

Source:

- [`modelctl_plans.py`](https://github.com/aahladky/modelctl/blob/master/modelctl_plans.py)

### 3.4 Hardware and backend fingerprinting

`modelctl_hardware.py` already records:

- GPU capacities and reserves;
- bandwidth estimates;
- RAM availability and reserve;
- storage descriptors;
- backend binary identity;
- a hardware fingerprint that invalidates stale measurements.

Source:

- [`modelctl_hardware.py`](https://github.com/aahladky/modelctl/blob/master/modelctl_hardware.py)

### 3.5 Test, benchmark, and persistent observations

`modelctl_tune.py` and `modelctl_runtime.py` already support:

- launching a plan on a temporary port;
- waiting for health;
- measuring load time, TTFT, prompt speed, generation speed, peak VRAM, and peak RAM;
- persisting results by plan ID and environment fingerprints;
- ranking plans from measured performance rather than estimates alone.

Sources:

- [`modelctl_tune.py`](https://github.com/aahladky/modelctl/blob/master/modelctl_tune.py)
- [`modelctl_runtime.py`](https://github.com/aahladky/modelctl/blob/master/modelctl_runtime.py)

### 3.6 Managed worker and fallback

`modelctl_worker.py` already:

- compiles and ranks plans;
- checks live resource feasibility;
- acquires reservations;
- launches through a backend adapter;
- falls back after failures;
- supervises the child process.

Sources:

- [`modelctl_worker.py`](https://github.com/aahladky/modelctl/blob/master/modelctl_worker.py)
- [`modelctl_backends.py`](https://github.com/aahladky/modelctl/blob/master/modelctl_backends.py)

### 3.7 Web console

The web application already exposes:

- hardware settings;
- profile editing;
- tier planning;
- launch-plan comparison;
- runtime policy;
- plan testing and autotuning;
- history and runtime status.

Sources:

- [`modelctl_web/app.py`](https://github.com/aahladky/modelctl/blob/master/modelctl_web/app.py)
- [`plans.html`](https://github.com/aahladky/modelctl/blob/master/modelctl_web/templates/plans.html)

---

## 4. Scope and non-goals

## 4.1 In scope

- Persistent GPU cache slots for routed experts.
- Hybrid GPU-hit / CPU-miss execution.
- Cache-aware VRAM and RAM planning.
- mmap-backed cold expert storage.
- Separate prefill and decode cache policy.
- Per-device cache budgets on heterogeneous GPUs.
- Cache metrics, benchmark collection, and plan ranking.
- Web configuration, visibility, and testing.
- Static pinning plus dynamic caching.
- Stock-backend fallback when cache support is unavailable.
- A clear backend feature/capability contract.
- Cold-cache and warm-cache reproducible benchmarks.
- Later support for expert prefetch and managed RAM caching.

## 4.2 Explicitly out of scope for the first implementation

- A Python implementation of expert routing or tensor execution.
- A FUSE filesystem or `LD_PRELOAD` interception layer.
- Direct SSD-to-GPU I/O as a first milestone.
- Repacking every GGUF into a new custom format.
- ML-based cache replacement before simpler policies are measured.
- Expert pruning or approximate expert skipping.
- Training a predictor before basic caching is proven.
- Cross-request batching optimization.
- Replacing `llama-swap`.
- Replacing the existing launch-plan and runtime-policy system.

---

## 5. Target architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│ modelctl control plane                                               │
│                                                                      │
│ GGUF analysis  Hardware snapshot  Backend capability probe           │
│       │               │                    │                         │
│       └─────────────── Launch-plan compiler ──────────────────────┐  │
│                                                                  │  │
│ Resource claims  Cache budgets  CLI/config generation            │  │
│                                                                  ▼  │
│ Web UI / CLI ── test / tune / select / launch / observe / compare   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                    command line + env + metrics contract
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│ cache-capable llama.cpp fork                                         │
│                                                                      │
│ router output → expert IDs → cache lookup                            │
│                          ├─ hit  → GPU expert slot                    │
│                          └─ miss → CPU expert execution               │
│                                      │                               │
│                              mmap / RAM backing                       │
│                                                                      │
│ cache admission + eviction + async copies + metric export            │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.1 Runtime execution rule

For each MoE layer during decode:

1. Compute router selection.
2. Partition selected experts into cache hits and misses.
3. Run hits against persistent GPU slots.
4. Run misses directly against CPU-resident or mmap-backed expert data.
5. Combine expert outputs.
6. Update cache policy state.
7. Optionally promote eligible misses asynchronously.
8. Continue without making a miss require an immediate synchronous GPU transfer.

This follows the central lesson of the current `llama.cpp` RFC: a cache miss should not automatically become a blocking PCIe copy on the critical path.

Source:

- [llama.cpp RFC #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)

---

## 6. Configuration and schema design

Do not leave this feature permanently buried in `config.extra`. Raw flags are acceptable only for the first integration spike.

Introduce a versioned, structured profile section:

```json
{
  "profile_version": 2,
  "name": "ornith-397b-q4",
  "backend": "llama-cpp",
  "binary": "/home/aaron/workspace/llama.cpp-moe-cache/build-sycl/bin/llama-server",

  "config": {
    "device": "",
    "split_mode": "layer",
    "tensor_split": "2,1",
    "ctx": 32768,
    "cache_type_k": "q8_0",
    "cache_type_v": "q4_0",
    "flash_attn": "on",
    "fit": "off",
    "extra": ""
  },

  "moe_cache": {
    "mode": "auto",
    "gpu": {
      "budgets_bytes": {
        "SYCL0": 10737418240,
        "SYCL1": 4294967296
      },
      "policy": "slru",
      "probationary_fraction": 0.2,
      "admission_misses": 2,
      "pin_shared_experts": true,
      "pin_static_experts": []
    },
    "ram": {
      "mode": "page_cache",
      "budget_bytes": 51539607552,
      "mlock_hot_set": false
    },
    "storage": {
      "mode": "mmap",
      "readahead": "adaptive",
      "release_cold_pages": false
    },
    "prefill": {
      "admit_to_gpu_cache": false,
      "protect_decode_entries": true
    },
    "decode": {
      "admit_to_gpu_cache": true,
      "miss_execution": "cpu"
    },
    "prefetch": {
      "enabled": false,
      "method": "none",
      "max_overfetch_ratio": 1.5
    }
  },

  "runtime": {
    "mode": "managed",
    "objective": "fastest_generation",
    "allow_fallback": true,
    "allow_untested": false,
    "minimum_context": 8192,
    "maximum_storage_tier": 3
  }
}
```

### 6.1 Semantics

- `mode = off`: never generate expert-cache plans.
- `mode = auto`: generate a bounded set of cache sizes and policies.
- `mode = manual`: generate exactly the requested cache arrangement.
- `gpu.budgets_bytes`: dynamic cache reservation, not total model placement.
- `ram.mode = page_cache`: rely on mmap plus kernel page cache.
- `ram.mode = managed`: later explicit expert-level RAM cache.
- `decode.miss_execution = cpu`: required for the initial hybrid design.
- `prefill.admit_to_gpu_cache = false`: prevents long prompts from flooding the decode cache.
- `prefetch.enabled = false`: keeps Milestone 1 measurable and debuggable.

### 6.2 Backward compatibility

- Profiles without `profile_version` or `moe_cache` behave exactly as they do now.
- Migration should be additive and lazy.
- `load_profile()` should normalize missing fields in memory.
- Saving an edited profile can write `profile_version: 2`.
- Existing `extra` flags must remain untouched unless they conflict with planner-owned flags.
- Experimental raw cache flags should be detected and either imported into structured settings or clearly marked as unmanaged.

---

## 7. Backend capability contract

A custom binary must be distinguishable from stock `llama.cpp`.

Do not infer support solely from a filename or a fragile `--help` substring search.

### 7.1 Proposed capability probe

Add a fork-specific command:

```bash
llama-server --modelctl-capabilities
```

Example JSON:

```json
{
  "schema": 1,
  "backend": "llama.cpp",
  "build": "b12345",
  "devices": ["CPU", "SYCL0", "SYCL1"],
  "features": {
    "moe_expert_cache": true,
    "moe_cache_sycl": true,
    "moe_hybrid_cpu_miss": true,
    "moe_cache_metrics": true,
    "moe_cache_prefill_policy": true,
    "moe_cache_mmap_advice": false,
    "moe_cache_prefetch": false
  },
  "cli": {
    "cache_bytes": "--moe-cache-bytes",
    "cache_policy": "--moe-cache-policy",
    "admission_misses": "--moe-cache-admission-misses",
    "prefill_admission": "--moe-cache-prefill-admission"
  }
}
```

The exact proposed flag names are placeholders until the fork settles. `modelctl` should consume the advertised mapping instead of hard-coding every experimental flag forever.

### 7.2 Capability cache

Create:

```text
~/.local/share/modelctl/backend_capabilities/
└── <binary-fingerprint>.json
```

Invalidate when:

- binary content hash changes;
- `--version` changes;
- probe schema changes;
- the file is removed.

### 7.3 Failure behavior

- Cache requested + capability absent: preflight error for a fixed cache plan.
- Cache requested + managed mode + fallback allowed: cache plan is filtered; stock plans remain.
- Capability probe crashes: classify as `capability_probe_failed`.
- Binary supports cache but not SYCL cache: reject GPU-cache plans on SYCL.
- Binary supports metrics but no hybrid misses: mark plan experimental and disabled by default.

### 7.4 Files

Add:

```text
modelctl_capabilities.py
test_modelctl_capabilities.py
```

Extend:

```text
modelctl.py
modelctl_backends.py
modelctl_hardware.py
```

---

## 8. `modelctl` control-plane implementation

## 8.1 `modelctl_vram.py`: expose cache geometry

The existing GGUF layout parser should be extended to return enough information to size cache slots accurately.

Add fields such as:

```python
{
    "is_moe": True,
    "block_count": 61,
    "routed_expert_count": 128,
    "experts_used_count": 8,
    "shared_expert_count": 1,
    "expert_bytes_per_layer": {0: ..., 1: ...},
    "expert_bytes_each_per_layer": {0: ..., 1: ...},
    "expert_tensor_names": {
        0: {
            "gate": "...",
            "up": "...",
            "down": "..."
        }
    },
    "expert_quant_types": {...}
}
```

### Tasks

- Add architecture-aware detection of routed versus shared experts.
- Determine exact bytes per individual expert, not only per expert layer.
- Detect unequal expert sizes rather than assuming uniformity.
- Expose top-k / active-expert count when present in GGUF metadata.
- Preserve the existing dictionary API for compatibility.
- Add a typed helper:

```python
def moe_cache_geometry(model_path) -> MoeCacheGeometry | None:
    ...
```

### Acceptance criteria

- Ornith’s routed-expert bytes reconcile with total GGUF tensor bytes.
- Per-layer sum of individual expert bytes equals existing expert-layer bytes.
- Shared experts are never counted as dynamic cache candidates.
- Unknown MoE architectures fail conservatively rather than misclassifying dense tensors.

---

## 8.2 `modelctl_hardware.py`: represent cache and storage capabilities

Extend hardware snapshots.

Current `StorageSnapshot` is minimal. Add:

```python
@dataclass(frozen=True)
class StorageSnapshot:
    path: str
    kind: str
    allow_mmap: bool
    filesystem: str = ""
    block_device: str = ""
    measured_read_bytes_per_s: int | None = None
    direct_io_supported: bool | None = None
    numa_node: int | None = None
```

Add optional GPU transfer information:

```python
@dataclass(frozen=True)
class GpuSnapshot:
    ...
    pcie_generation: int | None = None
    measured_h2d_bytes_per_s: int | None = None
```

### Tasks

- Add a web-configurable storage entry for the model directory or RAID mount.
- Add an optional read-only benchmark job; never run it automatically during normal page loads.
- Include storage policy changes in the hardware fingerprint.
- Include custom binary capability fingerprint in the backend fingerprint.
- Add an explicit per-device `dynamic_cache_reserve_bytes` setting or keep it per profile; do not silently use all free VRAM.

### Initial recommendation

Keep cache budget per profile and use hardware settings only for hard device reserves. This prevents one model’s tuning from globally reserving 10 GiB on every launch.

---

## 8.3 `modelctl_tiers.py`: reserve dynamic cache before static assignment

Current MoE placement assigns whole routed-expert layers into remaining GPU budgets. Dynamic caching needs a reserved slot pool before that assignment.

Change the budget equation from:

```text
GPU usable
− fixed tensors
− KV cache
− compute reserve
= static expert-layer budget
```

to:

```text
GPU usable
− fixed tensors
− KV cache
− compute reserve
− dynamic expert-cache reserve
− transfer/staging reserve
= static expert-layer budget
```

### New planner inputs

```python
def plan_tiers(
    profile,
    inventory,
    vram_limit_pct,
    primary,
    *,
    cache_request=None,
    capabilities=None,
):
    ...
```

### Hybrid static/dynamic placement

The planner should be allowed to produce:

```text
SYCL0
├── fixed tensors share
├── statically pinned expert layers
└── 10 GiB dynamic expert cache

SYCL1
├── fixed tensors share
├── statically pinned expert layers
└── 4 GiB dynamic expert cache

CPU/RAM/SSD
└── all other experts
```

### Static pinning policy

Initially support:

- shared experts: always resident with fixed tensors;
- user-selected routed experts: optional;
- whole-layer static placement: preserve current behavior;
- no automatic per-expert hot-set pinning until traces exist.

Later, measured traces can create a static hot-set plan.

### Load-mode migration

Current upstream `llama.cpp` exposes `--load-mode` values including `mmap`, `mlock`, `mmap+mlock`, and `dio`, alongside `--cpu-moe` and `--n-cpu-moe`.

Plan to migrate planner-owned storage flags:

```text
Tier 3: --load-mode none or an explicitly validated resident mode
Tier 4: --load-mode mmap
```

Do not remove legacy `--no-mmap` handling until the minimum supported `llama.cpp` version is established.

Source:

- [llama.cpp server arguments](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

### Tests

Extend `test_modelctl_tiers.py` with:

- cache reserve reduces static expert placement;
- cache budget cannot consume fixed/KV reserve;
- multi-GPU per-device cache budgets;
- unknown cache device rejected;
- cache disabled leaves current output unchanged;
- Tier 3/Tier 4 storage-mode migration;
- shared experts excluded from dynamic pool;
- too-small cache budget yields warning, not malformed flags.

---

## 8.4 `modelctl_plans.py`: generate and claim cache variants

The launch-plan compiler is the best home for cache strategy alternatives.

### Candidate sources

Add:

```text
F. moe-cache-small
G. moe-cache-balanced
H. moe-cache-large
I. moe-cache-manual
J. moe-cache-static-hotset   (later)
K. moe-cache-prefetch        (later, experimental)
```

### Bounded auto variants

For each capable GPU, derive cache budgets from available space after fixed tensors, KV, and reserve:

```text
small:    20% of remaining VRAM
balanced: 50% of remaining VRAM
large:    80% of remaining VRAM
```

Apply floors and caps:

- minimum useful slot count: configurable, initially 4 experts;
- never reduce compute reserve;
- never exceed user hard budget;
- do not generate duplicate slot counts;
- maximum 3 automatic cache plans per base placement;
- preserve total candidate count limits.

### Extend `ResourceClaim`

Recommended compatible change:

```python
@dataclass(frozen=True)
class ResourceClaim:
    vram_bytes: dict
    ram_bytes: int
    storage_mode: str
    expected_context: int | None
    breakdown: dict = field(default_factory=dict)
```

Example breakdown:

```json
{
  "vram": {
    "SYCL0": {
      "fixed": 12884901888,
      "kv": 4294967296,
      "static_experts": 6442450944,
      "dynamic_expert_cache": 10737418240,
      "staging": 536870912,
      "reserve": 1610612736
    }
  },
  "ram": {
    "expert_backing": 51539607552,
    "staging": 1073741824
  }
}
```

`vram_bytes` remains the total reservation used by the worker and matrix logic.

### Stable plan identity

Include normalized cache configuration in `_plan_id()`:

```json
{
  "moe_cache": {
    "enabled": true,
    "budgets": {"SYCL0": 10737418240},
    "policy": "slru",
    "admission_misses": 2,
    "prefill_admission": false,
    "miss_execution": "cpu"
  }
}
```

A policy or cache-size change must create a different plan ID.

### Capability filtering

`compile_launch_plans()` should accept or probe backend capabilities:

```python
compile_launch_plans(
    profile,
    hardware,
    include_experimental=False,
    capabilities=None,
)
```

- Stock binary: only existing plans.
- Cache-capable binary: existing plans plus cache variants.
- Experimental prefetch capability: variants only when `include_experimental=True`.
- Manual cache requested but unsupported: produce a plan warning or preflight failure, not silent downgrade.

### Ranking

Add optional measured fields:

- `cache_hit_rate_decode`;
- `ssd_read_bytes_per_token`;
- `cpu_miss_fraction`;
- `p95_token_latency`;
- `warm_generation_tps`;
- `cold_generation_tps`.

Objectives:

```text
fastest_generation:
  prioritize warm decode t/s, then p95 token latency

balanced:
  reward generation speed and cache hit rate;
  penalize RAM, SSD bytes/token, and cold-start cost

lowest_storage_io:
  minimize SSD bytes/token subject to minimum speed

fastest_cold_start:
  prioritize first-run TTFT and cold decode
```

Do not rank on cache hit rate alone. A high hit rate can still be slower if synchronization overhead is bad.

---

## 8.5 `modelctl.py`: profile, preflight, and argument emission

### Tasks

- Add profile schema normalization for `moe_cache`.
- Add CLI editing support.
- Add preflight capability checks.
- Emit cache flags from structured settings.
- Preserve raw `extra` for unrelated flags.
- Detect conflicting cache flags in `extra`.
- Add `modelctl show` output for cache configuration.
- Add `modelctl verify` validation for cache geometry and binary support.

### Argument builder

Do not hard-code fork flags in `build_server_args()` if the capability response supplies names.

Add:

```python
def build_moe_cache_args(profile, plan=None, capabilities=None) -> list[str]:
    ...
```

Then:

```python
args.extend(build_moe_cache_args(profile, plan, capabilities))
```

For managed plans, plan-specific values override profile defaults.

### Proposed experimental command rendering

Illustrative only:

```bash
llama-server \
  --model model.gguf \
  --load-mode mmap \
  --cpu-moe \
  --moe-cache-device SYCL0 \
  --moe-cache-bytes 10737418240 \
  --moe-cache-policy slru \
  --moe-cache-admission-misses 2 \
  --moe-cache-prefill-admission off \
  --moe-cache-miss-execution cpu
```

### Preflight messages

Examples:

```text
OK: binary supports MoE expert cache schema 1
OK: SYCL expert-cache backend available for SYCL0,SYCL1
OK: hybrid CPU miss execution available
WARNING: prefetch requested but unsupported; prefetch plan omitted
ERROR: dynamic cache needs 10.0 GiB on SYCL0, only 7.2 GiB remains after fixed/KV/reserve
```

---

## 8.6 `modelctl_backends.py`: expand adapter responsibility

The adapter interface already owns backend-specific command construction, readiness, environment, and failure classification. Extend it with optional methods:

```python
class LlamaCppAdapter:
    def probe_capabilities(self, binary) -> dict:
        ...

    def collect_metrics(self, base_url) -> dict:
        ...

    def validate_plan(self, profile, plan, capabilities) -> list[str]:
        ...

    def classify_failure(self, exit_code, log_tail):
        ...
```

Add failure classes:

```text
unsupported_moe_cache
cache_allocation_failed
cache_geometry_invalid
cache_sync_failure
capability_probe_failed
storage_io_failure
```

The generic worker should not know llama.cpp metric names.

---

## 8.7 `modelctl_runtime.py`: persist cache observations

Use `details_json` for the first spike, then promote fields used by ranking to typed columns.

### Phase A: no migration beyond details JSON

```json
{
  "moe_cache": {
    "hits": 10522,
    "misses": 918,
    "hit_rate": 0.9197,
    "evictions": 312,
    "promotions": 475,
    "gpu_expert_calls": 10522,
    "cpu_expert_calls": 918,
    "ssd_read_bytes": 1887436800,
    "major_faults": 441,
    "h2d_bytes": 7516192768,
    "prefill_hit_rate": 0.24,
    "decode_hit_rate": 0.94,
    "slot_count": 96,
    "cache_bytes": 10737418240
  }
}
```

### Phase B: typed columns

Add only fields needed for filtering/ranking:

```sql
ALTER TABLE plan_runs ADD COLUMN warm_generation_tps REAL;
ALTER TABLE plan_runs ADD COLUMN cold_generation_tps REAL;
ALTER TABLE plan_runs ADD COLUMN p95_token_latency_ms REAL;
ALTER TABLE plan_runs ADD COLUMN moe_decode_hit_rate REAL;
ALTER TABLE plan_runs ADD COLUMN storage_read_bytes INTEGER;
ALTER TABLE plan_runs ADD COLUMN cpu_expert_fraction REAL;
```

Implement idempotent migrations with `PRAGMA table_info`.

### Observation freshness

Cache observations are stale when any of these change:

- hardware fingerprint;
- backend binary fingerprint;
- model fingerprint;
- cache policy;
- cache size;
- storage policy;
- quantization;
- context/KV configuration.

Most are already covered by plan ID and fingerprints; ensure model fingerprint remains part of the plan normalization.

---

## 8.8 `modelctl_tune.py`: benchmark cache behavior correctly

Current tests average two generation runs. Expert caching needs explicit cold and warm phases.

### New benchmark protocol

1. Unload previous backend.
2. Optionally drop only model-file page cache for a controlled cold-storage test.
3. Launch plan.
4. Record load time.
5. Run a **prefill stress prompt**.
6. Run a **cold decode** sample.
7. Run enough decode tokens to warm the expert cache.
8. Run a **warm decode** sample.
9. Run a second prompt from a different domain to test cache adaptability.
10. Collect backend metrics before and after each phase.
11. Persist phase-separated results.

### Safety around cache dropping

Do not run global:

```bash
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

from the web service.

Prefer one of:

- `posix_fadvise(..., POSIX_FADV_DONTNEED)` in a small helper against the model file;
- fork-provided cache reset endpoint;
- copy/read a controlled test model;
- label the run “warm OS cache” when a true cold test cannot be established.

### Prompt sets

Use at least:

- short general prompt;
- long prefill prompt;
- code prompt;
- technical prose prompt.

Reason: expert locality can vary by domain and prompt structure.

### Metrics

Collect:

```text
load seconds
TTFT
prompt t/s
cold generation t/s
warm generation t/s
p50/p95 inter-token latency
peak VRAM by device
peak RSS
major/minor page faults
process read bytes
SSD read bytes, where attributable
cache hit rate by phase
evictions and promotions
CPU/GPU expert calls
H2D bytes
slot utilization
```

Linux process sources can include:

- `/proc/<pid>/io`;
- `/proc/<pid>/stat`;
- `/proc/<pid>/status`;
- `resource.getrusage()` where appropriate.

The backend’s own metrics remain authoritative for expert-level events.

### Acceptance criteria

- Benchmark output explicitly states whether OS page cache was cold, warm, or unknown.
- Warm and cold results are never averaged into one number.
- A plan cannot be marked validated if its metric endpoint is required but missing.
- Output quality is sanity-checked, not only speed-tested.
- Baseline stock plan and cache plan use the same prompt, context, quant, and token count.

---

## 8.9 `modelctl_worker.py`: keep runtime changes small

The worker should remain generic.

Required changes:

- ask the backend adapter to validate the selected plan;
- use adapter capability data already associated with the plan;
- optionally scrape final metrics on graceful shutdown;
- record cache-allocation failures distinctly;
- preserve current fallback behavior.

Do not put cache-slot logic in the worker.

---

## 8.10 `modelctl_matrix.py`: account for dynamic cache reservations

A dynamic cache is committed VRAM, even if slots are empty at startup.

Because `ResourceClaim.vram_bytes` includes cache reservation, existing coexistence logic should mostly work automatically.

Add UI/debug breakdown so users can see why two models no longer coexist:

```text
ornith:
  SYCL0 total 29.4 GiB
    fixed/KV/static 19.4 GiB
    dynamic expert cache 10.0 GiB

helper-model:
  SYCL0 total 6.8 GiB

combined: 36.2 GiB > budget 30.4 GiB
```

Add a policy option later to shrink a cache when a helper model is resident. Do not implement elastic cache resizing in the first version unless the runtime supports safe resizing.

---

## 9. Web-console implementation

The feature should be fully controllable and understandable through the web UI.

## 9.1 Profile page

Add an “MoE expert cache” card to `profile_edit.html`:

- detected model type;
- routed/shared expert counts;
- expert bytes per layer;
- cache mode: off / auto / manual;
- per-GPU cache budget;
- policy;
- admission threshold;
- prefill admission;
- miss execution;
- RAM mode;
- storage mode;
- prefetch toggle, disabled unless supported;
- binary capability status.

Do not put every experimental knob in the default view. Use an expandable advanced section.

## 9.2 Plan page

Extend `plans.html` columns or details:

```text
cache
  SYCL0: 10.0 GiB / 96 slots
  policy: SLRU
  prefill admission: off
  miss path: CPU

measured
  cold: 1.9 t/s
  warm: 4.8 t/s
  decode hit: 93.7%
  SSD: 7.2 MiB/token
```

Add plan filters:

- cache / no cache;
- validated only;
- stock / experimental backend;
- storage tier;
- cache size.

## 9.3 Runtime page

Add live cache cards when the backend exports metrics:

- hit rate;
- slots used / total;
- CPU miss fraction;
- SSD read rate;
- promotions and evictions;
- current generation speed;
- cache warmup curve.

HTMX polling every 2–5 seconds is sufficient. Do not create a high-frequency telemetry system.

## 9.4 Benchmark history

Add phase-aware comparison:

```text
plan                 cold t/s   warm t/s   hit rate   SSD GiB   p95 ms
stock cpu-moe          2.1        2.3         —         3.8      520
cache 4 GiB            2.0        3.4       79%         1.9      350
cache 10 GiB           1.9        4.9       94%         0.6      210
```

## 9.5 New routes

Suggested routes:

```text
GET  /api/profiles/{name}/moe-layout
GET  /api/backends/{fingerprint}/capabilities
GET  /api/runtime/models/{name}/moe-cache
POST /profiles/{name}/moe-cache
POST /profiles/{name}/moe-cache/reset
POST /profiles/{name}/moe-cache/benchmark
```

Cache reset must be capability-gated and should not be exposed for a loaded production request without a warning.

## 9.6 Files

Extend:

```text
modelctl_web/app.py
modelctl_web/mutate.py
modelctl_web/templates/profile_edit.html
modelctl_web/templates/plans.html
modelctl_web/templates/runtime.html
modelctl_web/templates/history.html
modelctl_web/static/*
test_modelctl_web.py
```

Potential new partials:

```text
modelctl_web/templates/_moe_cache_form.html
modelctl_web/templates/_moe_cache_metrics.html
```

---

## 10. `llama.cpp` fork implementation

This is the technically difficult portion.

## 10.1 Initial runtime feature set

The first cache-capable fork should support:

- CPU-resident/mmap-backed routed experts;
- one persistent expert-cache allocation per GPU;
- fixed-size slots;
- `(layer, expert) -> slot` mapping;
- GPU execution for hits;
- CPU execution for misses;
- result merge;
- SLRU or LRU policy;
- admission after N misses;
- prefill admission disabled;
- metrics;
- clean fallback to stock execution when disabled.

Do not begin with:

- prediction;
- multi-layer lookahead;
- direct I/O;
- variable-size compaction;
- online ML eviction;
- cache resizing.

## 10.2 Core data structures

Illustrative C++ design:

```cpp
struct moe_expert_key {
    int32_t layer;
    int32_t expert;
};

struct moe_cache_slot {
    moe_expert_key key;
    bool valid;
    bool protected_segment;
    uint64_t last_access_tick;
    uint64_t access_count;
    sycl::event ready_event;
    void * device_ptr;
};

struct moe_cache_layer_index {
    std::vector<int32_t> expert_to_slot;
};

struct moe_cache_stats {
    std::atomic<uint64_t> hits;
    std::atomic<uint64_t> misses;
    std::atomic<uint64_t> evictions;
    std::atomic<uint64_t> promotions;
    std::atomic<uint64_t> h2d_bytes;
    std::atomic<uint64_t> cpu_expert_calls;
    std::atomic<uint64_t> gpu_expert_calls;
};
```

Avoid hash maps in the inner lookup where expert IDs are dense. A per-layer vector indexed by expert ID is simpler and faster.

## 10.3 Slot geometry

A “slot” must hold all tensors needed to execute one routed expert for one layer.

For a typical gated MLP expert:

```text
gate projection
up projection
down projection
quantization metadata / scales
alignment padding
```

Possible approaches:

1. **Packed contiguous slot**
   - copy all tensors into one allocation;
   - custom kernels use offsets;
   - best long-term control.

2. **Per-tensor slot pools**
   - one slot index shared across gate/up/down pools;
   - easier integration with existing tensor kernels;
   - more allocations and bookkeeping.

Use per-tensor pools first if it minimizes invasive changes.

## 10.4 Cache hit path

1. Router produces selected expert IDs.
2. Lookup each `(layer, expert)` in the layer index.
3. Build compact hit IDs remapped to slot IDs.
4. Dispatch existing or adapted `MUL_MAT_ID` kernels against the cache buffer.
5. Record the slot’s last access without forcing a queue-wide synchronization.
6. Return GPU result buffer.

## 10.5 Cache miss path

1. Build miss ID list.
2. Execute misses using the existing CPU expert path.
3. Produce CPU miss result.
4. In parallel, optionally schedule promotion if admission policy accepts the expert.
5. Merge hit and miss contributions.

The merge can occur:

- on GPU after copying the comparatively small CPU result tensor; or
- on CPU if that is already the natural destination.

Copying output activations is vastly cheaper than copying full expert weights.

This is the key Fiddler-style principle: use CPU compute to reduce weight movement.

Source:

- [Fiddler](https://arxiv.org/abs/2402.07033)

## 10.6 Admission policy

Start with:

```text
Policy: SLRU
Probationary segment: 20%
Protected segment: 80%
Admission: second miss
Prefill admission: disabled
```

Why:

- a one-off expert does not evict a genuinely hot entry;
- prefill cannot flood protected decode entries;
- it is deterministic and easy to test;
- current llama.cpp experiments report better mixed prefill/decode behavior than plain LRU.

Source:

- [llama.cpp issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)

Also implement plain LRU as a reference baseline.

## 10.7 SYCL memory management

Use the existing SYCL backend rather than creating a separate Level Zero backend.

Relevant upstream documentation:

- [llama.cpp SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)

Requirements:

- persistent device allocations for cache buffers;
- pinned/USM host buffers only where they improve measured transfer behavior;
- asynchronous `queue.memcpy`;
- per-slot readiness events;
- no `queue.wait()` in the per-layer steady-state hit path;
- no global synchronization for cache metadata;
- device-local cache pools on each GPU;
- correct behavior when the same model spans SYCL0 and SYCL1.

### Synchronization rule

A slot may be visible in the lookup table only after its copy-ready event has completed, or the consuming kernel must explicitly depend on that event.

Never mark a slot resident before the data is usable.

### Multi-GPU rule

Each device has its own cache namespace:

```text
(device, layer, expert) -> slot
```

An expert cached on SYCL0 is not a hit on SYCL1.

## 10.8 CPU execution

Use the existing quantized CPU kernels where possible.

Benchmark:

```text
CPU execute in place
versus
copy weights to GPU + execute
```

The miss policy should remain fixed to CPU for Milestone 1. Later, a cost model may choose per expert or per batch.

KTransformers demonstrates the broader viability of hot GPU experts plus cold CPU experts:

- [KT-Kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- [DeepSeek tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md)

## 10.9 Prefill/decode separation

Maintain separate counters and policy state.

### Prefill

- default: no GPU cache admission;
- existing protected entries remain;
- CPU/grouped expert execution;
- optional readahead into system page cache;
- no aggressive promotion.

### Decode

- admission enabled;
- cache hit path active;
- miss execution on CPU;
- second-use promotion;
- SLRU updates.

The backend needs a reliable way to know whether it is processing prefill or decode. Do not infer solely from batch size if llama.cpp already exposes execution-phase context.

## 10.10 mmap and page-cache support

The initial storage path is:

```text
GGUF mmap → Linux page cache → CPU expert kernels
```

Use current upstream load-mode support:

- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

Possible later mmap hints:

- `POSIX_MADV_WILLNEED` for likely reused experts;
- `POSIX_MADV_DONTNEED` only after explicit RAM-tier eviction;
- careful range alignment to page boundaries.

Do not call `DONTNEED` merely because an expert leaves VRAM. It may still be valuable in RAM.

## 10.11 Metrics contract

Expose Prometheus metrics under the existing metrics endpoint or a dedicated JSON endpoint.

Suggested names:

```text
llama_moe_cache_hits_total{device,layer}
llama_moe_cache_misses_total{device,layer}
llama_moe_cache_evictions_total{device}
llama_moe_cache_promotions_total{device}
llama_moe_cache_slots{device}
llama_moe_cache_slots_used{device}
llama_moe_cache_bytes{device}
llama_moe_cache_hit_ratio{device,phase}
llama_moe_cpu_expert_calls_total{layer}
llama_moe_gpu_expert_calls_total{device,layer}
llama_moe_h2d_bytes_total{device}
llama_moe_storage_fault_bytes_total
llama_moe_cache_sync_wait_seconds_total{device}
```

Add process-lifetime totals plus phase-resettable counters.

The most important diagnostic metric may be:

```text
cache synchronization wait time
```

A cache with a 98% hit rate can still lose if every layer waits on the host.

---

## 11. Explicit RAM cache: later milestone

The initial page-cache approach may be sufficient. If not, implement a managed RAM tier.

## 11.1 Trigger for this work

Only proceed if benchmarks show one or more:

- major page faults remain high after warmup;
- the kernel evicts frequently reused expert pages;
- prefill destroys the decode working set;
- RSS/page-cache behavior is too unpredictable;
- SSD read bytes/token remain high despite good VRAM policy.

## 11.2 Managed RAM cache design

```text
GPU slot cache
    ↓ miss
Managed RAM expert cache
    ↓ miss
mmap / pread from SSD
```

The RAM cache should store complete expert objects, not arbitrary 4 KiB pages.

Data:

```cpp
struct ram_expert_entry {
    moe_expert_key key;
    void * ptr;
    size_t bytes;
    uint64_t last_access;
    uint64_t access_count;
    bool pinned_for_cpu_execution;
};
```

Support:

- byte-budgeted eviction;
- SLRU or frequency/recency scoring;
- asynchronous read;
- CPU execution directly from cache format;
- optional page release on eviction.

FlashMoE is the relevant research reference for SSD-backed expert caching and recency/frequency-aware replacement:

- [FlashMoE](https://arxiv.org/abs/2601.17063)

---

## 12. Prefetching: later milestone

Prefetching should be added only after:

- hit/miss execution is correct;
- cache metrics are trusted;
- synchronization overhead is understood;
- warm-cache speedup is established.

## 12.1 Stage 1: cheap heuristic prefetch

- readahead recently used experts;
- prefetch an expert into RAM, not necessarily VRAM;
- no overfetch beyond a small configured ratio.

## 12.2 Stage 2: cross-layer prediction

Fate predicts next-layer expert use from adjacent gate inputs and is the most relevant reference:

- [Fate](https://arxiv.org/abs/2502.12224)

Implement behind a feature flag:

```text
--moe-prefetch cross-layer
```

Metrics:

```text
prediction precision
prediction recall
bytes prefetched
useful prefetched bytes
wasted prefetched bytes
latency hidden
```

## 12.3 Admission by confidence

```text
high confidence:
  prefetch to GPU slot if free or low-cost victim exists

medium confidence:
  prefetch to managed RAM / page cache

low confidence:
  no action
```

Never measure predictor accuracy without also measuring overfetch bytes and end-to-end latency.

---

## 13. Testing strategy

## 13.1 Unit tests: `modelctl`

Add:

```text
test_modelctl_capabilities.py
test_modelctl_moe_cache.py
```

Extend:

```text
test_modelctl.py
test_modelctl_vram.py
test_modelctl_tiers.py
test_modelctl_plans.py
test_modelctl_runtime.py
test_modelctl_tune.py
test_modelctl_worker.py
test_modelctl_web.py
test_modelctl_matrix.py
```

### Key test groups

#### Schema

- old profile loads unchanged;
- new cache profile round-trips;
- malformed budgets rejected;
- unknown devices rejected;
- conflicting raw flags reported.

#### GGUF geometry

- routed/shared expert distinction;
- per-expert byte accounting;
- unequal expert sizes;
- unsupported naming fails safely.

#### Planning

- fixed tensors fit before cache;
- cache reserve included in resource claim;
- static and dynamic expert allocation coexist;
- plan IDs change with cache policy;
- unsupported binary does not produce cache plans;
- manual unsupported cache plan fails clearly;
- fallback plans remain.

#### Runtime DB

- detail metrics persist;
- typed migration is idempotent;
- stale fingerprints invalidate observations;
- ranking consumes warm/cold fields correctly.

#### Web

- cache form saves;
- plan page shows budget and measured metrics;
- unsupported controls are disabled;
- POST mutation goes through job runner;
- auth/CSRF behavior remains unchanged.

## 13.2 Runtime unit tests: `llama.cpp` fork

- slot lookup;
- admission threshold;
- LRU behavior;
- SLRU promotion/demotion;
- prefill protection;
- eviction correctness;
- event readiness;
- counter accuracy;
- cache reset;
- multi-device namespace;
- zero-slot behavior;
- one-slot thrash behavior.

## 13.3 Numerical correctness

For each supported model family:

1. Run cache disabled with deterministic sampling.
2. Run cache enabled with deterministic sampling.
3. Compare logits or generated tokens within quant/backend tolerance.
4. Force all hits.
5. Force all misses.
6. Force alternating hit/miss.
7. Force eviction every token.
8. Test multi-expert top-k merging.
9. Test shared expert plus routed experts.
10. Test prefill-to-decode transition.

A performance optimization that changes expert weighting or merge order incorrectly is not acceptable.

## 13.4 Fault injection

- failed device allocation;
- failed async copy;
- backend event error;
- mmap read error;
- truncated model file;
- unsupported quant kernel;
- GPU reset/device lost;
- metrics endpoint unavailable;
- worker termination during promotion;
- simultaneous unload and benchmark.

The safe fallback is cache-disabled execution or process failure with a specific class, not silent corrupted output.

---

## 14. Benchmark matrix

Use the same model, context, prompt set, and quant across variants.

### Baselines

1. Current Tier 4 mmap plan.
2. `--cpu-moe` or equivalent without dynamic cache.
3. Current static expert-layer placement.
4. Cache-capable binary with cache disabled, to measure fork overhead.

### Cache variants

5. LRU, small.
6. LRU, balanced.
7. SLRU, balanced.
8. SLRU + second-miss admission.
9. SLRU + prefill admission disabled.
10. Large cache.
11. Multi-GPU cache.
12. Later: RAM managed tier.
13. Later: prefetch.

### Required report

```text
backend/build
model fingerprint
hardware fingerprint
storage mount/device
load mode
OS-cache state
context
prompt tokens
generated tokens
cache bytes/slots
policy
cold prompt t/s
cold generation t/s
warm generation t/s
p95 token latency
decode hit rate
CPU miss fraction
SSD read bytes/token
H2D bytes/token
peak VRAM
peak RAM
output correctness
```

### Success threshold for first runtime cache

A first version is worth keeping if, on at least one representative oversized MoE model:

- cache-disabled fork is within 5% of stock performance;
- warm generation improves by at least 25%;
- cold performance does not regress by more than 15%;
- no output divergence beyond accepted backend tolerance;
- steady-state synchronization wait is not the dominant token time;
- metrics explain the speedup.

Do not require a spectacular 10× result to declare the architecture valid.

---

## 15. Milestone plan

## Milestone 0 — Baseline and reproducibility

### Deliverables

- Record current Ornith command/profile.
- Add benchmark metadata for OS-cache state.
- Capture `/proc/<pid>/io`, page faults, VRAM, RSS.
- Establish stock Tier 4 and static-placement baselines.
- Add one long-prefill and one decode-focused test prompt.

### Files

```text
modelctl_tune.py
modelctl_runtime.py
modelctl_web/templates/history.html
tests
```

### Exit criteria

- Repeated runs produce explainable cold/warm differences.
- Current performance can be reproduced from a saved plan.

---

## Milestone 1 — Structured cache configuration and capability gating

### Deliverables

- `moe_cache` profile schema.
- Capability probe module.
- Preflight validation.
- Raw experimental flag emission from structured config.
- Web form and plan display.
- No planner-generated cache variants yet.

### Files

```text
modelctl_capabilities.py          new
test_modelctl_capabilities.py     new
modelctl.py
modelctl_backends.py
modelctl_web/app.py
modelctl_web/templates/profile_edit.html
test_modelctl.py
test_modelctl_web.py
```

### Exit criteria

- A custom fork can be pinned and launched entirely from the web UI.
- Stock binaries cannot accidentally receive unsupported flags.
- Existing profiles and tests remain unchanged.

---

## Milestone 2 — Cache-aware planning and claims

### Deliverables

- Per-expert geometry.
- Cache reserve in tier planning.
- Automatic small/balanced/large variants.
- Resource-claim breakdown.
- Plan IDs include cache settings.
- Matrix and reservation logic account for cache bytes.

### Files

```text
modelctl_vram.py
modelctl_tiers.py
modelctl_plans.py
modelctl_matrix.py
modelctl_worker.py
modelctl_web/templates/plans.html
tests
```

### Exit criteria

- The planner never overcommits VRAM by forgetting cache slots.
- Web plans clearly distinguish static expert bytes from dynamic cache bytes.
- Unsupported binaries generate no cache variants.

---

## Milestone 3 — llama.cpp cache skeleton, CPU misses

### Deliverables

- Persistent SYCL slot pool.
- Dense per-layer lookup.
- LRU policy.
- Cache-hit GPU execution.
- CPU miss execution.
- Output merge.
- Metrics.
- Cache disabled by default.

### Exit criteria

- Forced-hit, forced-miss, and mixed paths match baseline output.
- Cache-disabled fork overhead is under 5%.
- No per-layer global queue wait on the hit-only path.

---

## Milestone 4 — Admission, SLRU, and prefill protection

### Deliverables

- SLRU.
- Second-miss admission.
- Prefill admission disabled.
- Phase-separated metrics.
- Cache reset support.

### Exit criteria

- Mixed prefill/decode hit rate improves over plain LRU.
- Long prefill no longer flushes protected decode entries.
- Warm decode improvement meets the initial success threshold.

---

## Milestone 5 — Integrated tuning and web telemetry

### Deliverables

- Backend metric scraping.
- Cold/warm benchmark protocol.
- Cache-aware ranking.
- Live runtime cache page.
- History comparison.
- Autotune cache sizes.

### Exit criteria

- A user can create, benchmark, compare, select, and launch a cache plan entirely from the web console.
- Measurement history invalidates after binary/hardware/model changes.
- The selected plan is based on measured performance.

---

## Milestone 6 — mmap advice and storage optimization

### Deliverables

- Per-expert readahead.
- Page-fault and storage-byte attribution.
- Optional page release after managed-tier eviction.
- Storage benchmark/settings.

### Exit criteria

- SSD optimization shows measurable benefit beyond the cache alone.
- No regression on warm page-cache workloads.

---

## Milestone 7 — Managed RAM tier

### Deliverables

- Byte-budgeted complete-expert RAM cache.
- Asynchronous SSD reads.
- CPU execution directly from managed cache.
- RAM eviction metrics.
- Web controls.

### Exit criteria

- Lower major-fault rate or lower SSD bytes/token than page-cache mode.
- Improvement justifies complexity on the target machine.

---

## Milestone 8 — Expert prediction and prefetch

### Deliverables

- Heuristic prefetch.
- Cross-layer predictor behind experimental flag.
- Confidence-aware destination.
- Prediction/overfetch metrics.

### Exit criteria

- End-to-end token latency improves.
- Useful-prefetch ratio is high enough to justify bandwidth.
- No cache-hit gain is accepted if total performance regresses.

---

## 16. Suggested branch and repository strategy

Keep `modelctl` and the runtime fork in separate repositories or clearly separate remotes.

```text
aahladky/modelctl
  feature/moe-cache-control-plane

aahladky/llama.cpp
  feature/sycl-moe-expert-cache
```

Pin the runtime fork by:

- binary path;
- binary fingerprint;
- upstream base commit;
- fork commit;
- capability schema.

Add a `docs/runtime-forks.md` file in `modelctl` describing reproducible builds.

Do not vendor all of `llama.cpp` into `modelctl`.

### Documentation files

```text
docs/superpowers/specs/2026-07-28-moe-expert-cache-design.md
docs/superpowers/plans/2026-07-28-moe-expert-cache-implementation.md
docs/runtime-forks.md
docs/benchmarks/moe-cache-methodology.md
```

---

## 17. Initial issue breakdown

### Control plane

1. Add capability probe and cache.
2. Add profile schema normalization.
3. Add web cache configuration form.
4. Add GGUF per-expert geometry.
5. Add dynamic-cache resource breakdown.
6. Add cache-aware tier planning.
7. Add cache launch-plan variants.
8. Include cache settings in stable plan ID.
9. Persist cache metrics in plan-run details.
10. Add cold/warm benchmark phases.
11. Add runtime metric scraper.
12. Add plan/history UI.
13. Add cache-aware ranking.
14. Add matrix explanation for cache reservations.

### Runtime fork

15. Add feature probe command.
16. Add cache configuration parsing.
17. Allocate persistent SYCL slot pools.
18. Implement expert-to-slot indexing.
19. Implement forced-hit test mode.
20. Implement forced-miss test mode.
21. Add GPU hit execution.
22. Add CPU miss execution.
23. Merge hit/miss outputs.
24. Add LRU.
25. Add metrics.
26. Remove steady-state queue-wide waits.
27. Add SLRU.
28. Add second-miss admission.
29. Add prefill protection.
30. Add cache reset.
31. Validate multi-GPU.
32. Add mmap advice experiment.
33. Add explicit RAM tier only if justified.
34. Add prefetch only if justified.

---

## 18. Principal risks and mitigations

## 18.1 Synchronization erases the cache benefit

**Risk:** high nominal hit rate but slower decoding.

**Mitigation:**

- instrument wait time;
- use per-slot events;
- avoid host readback and queue-wide waits;
- measure cache-disabled fork overhead;
- keep misses on CPU.

## 18.2 Cache accounting causes VRAM OOM

**Risk:** static planner consumes memory intended for slots or staging.

**Mitigation:**

- include every reserved byte in `ResourceClaim`;
- add breakdown;
- reserve cache before static expert assignment;
- validate after backend allocation;
- classify cache allocation failures.

## 18.3 Prefill destroys decode locality

**Risk:** long prompts fill cache with one-use experts.

**Mitigation:**

- prefill admission off;
- SLRU protected segment;
- second-use admission;
- separate phase metrics.

## 18.4 Kernel page cache hides or distorts SSD results

**Risk:** “SSD” benchmark is actually RAM.

**Mitigation:**

- report OS-cache state;
- collect major faults and process read bytes;
- separate cold and warm runs;
- avoid unsupported performance claims.

## 18.5 Model architecture naming varies

**Risk:** wrong expert tensor classification.

**Mitigation:**

- architecture-specific parsing;
- exact byte reconciliation;
- conservative failure;
- fixture coverage for multiple GGUF families.

## 18.6 Fork maintenance burden

**Risk:** large invasive patch becomes impossible to rebase.

**Mitigation:**

- narrow feature probe and metrics contract;
- isolate cache code;
- avoid broad scheduler rewrites;
- track upstream RFCs;
- keep control-plane integration independent of exact flag names.

## 18.7 CPU miss path becomes the bottleneck

**Risk:** hit rate is insufficient or CPU kernels are weak.

**Mitigation:**

- measure per-layer misses;
- increase cache where useful;
- static-pin highly reused layers;
- optimize quantized CPU kernels;
- later add cost-based transfer-versus-compute decisions.

## 18.8 SSD wear concern is misunderstood or exaggerated

**Risk:** excessive reads produce needless concern; writes are the main endurance issue, but continuous I/O still matters for thermals and contention.

**Mitigation:**

- expose read bytes/token and read rate;
- avoid unnecessary cache-file writes;
- measure temperature/throttling externally where needed;
- prioritize locality over raw streaming.

---

## 19. Definition of done

The feature is complete when:

1. A sparse MoE profile can enable expert caching through the web UI.
2. `modelctl` verifies that the pinned binary supports the requested features.
3. Launch plans reserve dynamic cache VRAM accurately.
4. Stock, static-offload, and dynamic-cache plans coexist and can be compared.
5. The runtime performs GPU cache hits and CPU misses correctly.
6. Long prefill does not destroy the protected decode cache.
7. Cache metrics are visible live and stored in benchmark history.
8. Benchmarks distinguish cold storage, warm page cache, and warm expert cache.
9. Autotuning can compare cache sizes and policies.
10. Managed runtime can select a validated plan and fall back safely.
11. Existing profiles and non-MoE models behave exactly as before.
12. The cache-disabled fork is within 5% of stock performance.
13. At least one representative oversized MoE shows a meaningful warm-decode improvement.
14. No correctness regression is observed in forced-hit, forced-miss, eviction, and multi-GPU tests.
15. Documentation includes build, configuration, benchmark, and troubleshooting instructions.

---

## 20. Recommended first coding sprint

The best first sprint is entirely in `modelctl` plus a minimal stub fork.

### Sprint deliverables

- `modelctl_capabilities.py`;
- profile `moe_cache` schema;
- structured argument generation;
- web form;
- capability-gated cache plan;
- `ResourceClaim.breakdown`;
- cold/warm benchmark labels;
- stub `llama-server --modelctl-capabilities`;
- stub cache metrics endpoint;
- no real cache yet.

### Why this first

It proves the integration boundary before expensive runtime work. It also means that once the SYCL cache begins producing useful results, every experiment is immediately:

- reproducible;
- resource-accounted;
- benchmarked;
- visible in the web UI;
- tied to a binary fingerprint;
- comparable with the existing Tier 4 baseline.

That prevents the runtime experiment from becoming a pile of hand-written command lines and unverifiable “felt faster” results.

---

## 21. Source map

### Current `modelctl`

- Repository and architecture:  
  https://github.com/aahladky/modelctl
- Tier planner:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_tiers.py
- Main profile/preflight/argument code:  
  https://github.com/aahladky/modelctl/blob/master/modelctl.py
- Launch plans and claims:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_plans.py
- Runtime database:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_runtime.py
- Plan testing and autotuning:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_tune.py
- Managed worker:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_worker.py
- Backend adapters:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_backends.py
- Hardware snapshots/fingerprints:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_hardware.py
- Web application:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_web/app.py
- Launch-plan UI:  
  https://github.com/aahladky/modelctl/blob/master/modelctl_web/templates/plans.html

### `llama.cpp`

- Server flags, load modes, CPU-MoE options:  
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- SYCL backend:  
  https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
- Expert-cache RFC, hybrid hit/miss execution:  
  https://github.com/ggml-org/llama.cpp/discussions/24528
- Two-tier GPU/RAM cache experiment and policy notes:  
  https://github.com/ggml-org/llama.cpp/issues/20757
- On-demand activated-expert feature request:  
  https://github.com/ggml-org/llama.cpp/issues/11532
- CPU-MoE placement discussion:  
  https://github.com/ggml-org/llama.cpp/discussions/22183

### Research and reference implementations

- MoE-Infinity, activation-aware caching and prefetch:  
  https://arxiv.org/abs/2401.14361
- Fiddler, CPU/GPU orchestration and CPU miss execution:  
  https://arxiv.org/abs/2402.07033
- Fiddler code:  
  https://github.com/efeslab/fiddler
- FlashMoE, SSD-backed expert caching:  
  https://arxiv.org/abs/2601.17063
- Fate, cross-layer expert prediction:  
  https://arxiv.org/abs/2502.12224
- Pre-attention expert prediction:  
  https://arxiv.org/abs/2511.10676
- KTransformers:  
  https://github.com/kvcache-ai/ktransformers
- KTransformers heterogeneous hot/cold expert path:  
  https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md
- KTransformers DeepSeek architecture tutorial:  
  https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md

---

## 22. Final recommendation

Build this as a **cache-aware managed runtime feature**, not as a one-off Tier 4 flag bundle.

The current `modelctl` repository is already structurally ready:

- GGUF analysis identifies the expensive expert portion;
- the tier planner can reserve and place memory;
- launch plans can represent alternative cache budgets;
- hardware and backend fingerprints can invalidate stale tests;
- the runtime database can store measurements;
- the worker can select and fall back;
- the web UI can expose the whole process.

The missing work is a clean backend contract and a correct SYCL expert-cache implementation.

The practical order is:

```text
measurement
→ capability contract
→ cache-aware planning
→ persistent SYCL slots
→ hybrid CPU misses
→ SLRU/prefill protection
→ web telemetry/autotuning
→ storage advice
→ managed RAM tier
→ prediction/prefetch
```

That sequence produces useful checkpoints and avoids spending months optimizing SSD movement before proving that expert residency and scheduling are correct.
