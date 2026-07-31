# Re-review: modelctl and llama.cpp

**Date:** July 31, 2026  
**Code reviewed:** the local-first `modelctl` project and its pinned `llama.cpp` runtime fork, using the GitHub repositories as convenient review mirrors  
**Project goal:** A personally operated, web-first, hardware-aware local model management and serving platform, optimized for this machine and for sparse MoE models spanning VRAM, system RAM, and SSD-backed mmap storage.

## Executive assessment

The project has advanced substantially since the previous review. It now has a credible control-plane architecture rather than a collection of launch scripts:

- a browser-first product direction;
- canonical backend resolution and launch-command objects;
- persistent plans, observations, reservations, and runtime events;
- storage-topology and cold/warm benchmark concepts;
- a persistent add-model workflow;
- a versioned runtime capability interface;
- a recently synchronized llama.cpp fork;
- real-hardware measurements showing that MoE offload policy must be model- and placement-specific.

The central architectural direction remains correct. The project does not need a rewrite.

The most important remaining issues have shifted. Earlier runtime-contract problems—hardcoded cache capabilities and weak-symbol server integration—have largely been addressed. The leading blockers are now:

1. **Resource reservations do not consistently represent peak use.**
   Cache VRAM is tracked separately but omitted from several admission paths, while mmap-addressed bytes are still treated as resident RAM in managed serving.
2. **The final launched environment is not fully part of command identity.**
   Plan-specific environment overrides can change offload behavior after the canonical command and capability identity have already been produced.
3. **The real SYCL cache integration still assumes simplified expert geometry and has unresolved shared-device queue ownership.**
4. **The current runtime still does not implement true GPU-hit/CPU-miss hybrid execution.**
   It remains an experimental expert-weight transfer cache.
5. **The browser is first-class in scope, but the product surface still has duplicated acquisition paths and risky default LAN exposure.**

The next development cycle should prioritize control-plane truthfulness and operational consistency before expanding the cache policy or adding prediction.


## Project scope: local-first, not public-product-first

The GitHub repositories are review mirrors and convenient source snapshots. They are not the product, the primary installation channel, or the center of the project.

This changes the priority weighting:

### What matters

- correctness on the actual local hardware;
- a reliable browser-operated model lifecycle;
- accurate VRAM, RAM, mmap, and storage accounting;
- reproducible known-good runtime builds;
- useful measurements tied to the exact model, binary, driver, and environment;
- safe recovery from failed experiments;
- maintainability for the future owner of the same system;
- practical improvement in oversized-MoE inference.

### What does not deserve near-term effort

- public packaging polish;
- contributor onboarding;
- generic multi-user permissions;
- broad portability across arbitrary Linux systems;
- stable external APIs;
- release-page maintenance;
- public plugin ecosystems;
- preserving every legacy workflow;
- GitHub presentation work that does not improve local operation.

The project should remain understandable and reproducible, but it does not need to behave like a commercial or community-supported distribution.

A local capability that works reliably on the target Arc GPUs, RAM configuration, storage array, and selected backend is more valuable than a generic abstraction that works nowhere particularly well.

## Implementation-agent directive

This review is intended to drive implementation, not generate another round of planning documents.

Claude Code or another coding agent working from this document should follow these rules:

1. **Do the task.**  
   Inspect the relevant code, edit it, add or update tests, run them, and report the result. Do not stop after producing a plan unless the requested output is specifically a plan.

2. **Stay on the critical path.**  
   Work in the priority order in this review. Do not create side projects, broad refactors, new frameworks, plugin systems, public packaging work, or unrelated cleanup.

3. **Use the existing architecture.**  
   Prefer the current service, planner, launch, runtime, database, and web patterns. Introduce a new abstraction only when the current code cannot express the required behavior cleanly.

4. **Explain plainly.**  
   Use concrete language:
   - name the file;
   - name the function or class;
   - state what is wrong;
   - state what changed;
   - state which test proves it.

   Avoid phrases such as “semantic orchestration layer,” “capability-aware execution substrate,” or other jargon when “the worker was using the wrong VRAM total” is clearer.

5. **Do not repeatedly ask for permission.**  
   Make reasonable implementation decisions and proceed. Ask only when a genuinely missing decision would cause destructive data loss, incompatible behavior, or a major product-direction change.

6. **Do not drag the user through internal deliberation.**  
   The user does not need a tour of every possible design. Choose the simplest sound implementation consistent with the existing code and complete it.

7. **Finish vertical slices.**  
   A task is not complete when a helper exists. Follow it through every real path:
   - planner;
   - preview;
   - worker;
   - reservation;
   - launch;
   - runtime event;
   - web display;
   - test.

8. **Treat helper-only tests as insufficient.**  
   Add at least one integration test covering the actual production path whenever a bug concerns launch behavior, reservations, capabilities, persistence, or the web workflow.

9. **Preserve working local behavior.**  
   Do not rewrite stable components merely because another design is theoretically cleaner. The target is a dependable local system, not architectural novelty.

10. **Use hardware evidence.**  
    Do not declare an optimization successful from hit rate, code elegance, or plausible generated text. Compare it against the appropriate local baseline and record correctness and performance.

11. **Handle blockers productively.**  
    When hardware, oneAPI, a large model, or a live service is unavailable:
    - complete all code and deterministic tests that can be done locally;
    - provide the exact remaining hardware validation command;
    - state the expected pass condition;
    - continue with other non-blocked work.

12. **Report concisely.**  
    The normal completion report should contain:
    - files changed;
    - behavior fixed or added;
    - tests run and results;
    - one clearly stated remaining blocker, if any.

A useful default instruction for Claude Code is:

> Implement the next incomplete priority from this review. Inspect the existing code first, then edit the production path and tests. Keep the solution narrow and consistent with current architecture. Do not create a new roadmap, speculate about unrelated improvements, or stop after describing the solution. Use plain English in comments and the final report. Continue until the task is implemented and the relevant tests pass, unless a concrete hardware-only blocker prevents completion.

## Current project status

### What is now convincingly in place

#### Web-first positioning

The current modelctl documentation explicitly treats the browser as the primary management interface, with the CLI retained for bootstrap, automation, diagnostics, and recovery.

The web application includes:

- first-run setup;
- token authentication;
- profile and model inventory;
- Hugging Face and local-file acquisition;
- a persistent add-model workflow;
- hardware, plans, tuning, history, routing, runtime, and jobs surfaces;
- background operations with cancellation;
- runtime load, unload, restart, logs, and cache controls.

This is enough product surface that future work should strengthen a single browser lifecycle rather than add more disconnected pages or CLI-only features.

#### Canonical launch architecture

`ResolvedBackend` and `LaunchCommand` provide the correct architectural center for launch behavior. Backend resolution performs binary selection, preflight, environment construction, and capability probing; command construction validates profile requirements and delegates backend-specific argument generation.

That is the right direction for ensuring that:

- the UI preview;
- tests;
- managed workers;
- llama-swap configuration;
- and direct launches

all describe the same executable operation.

#### Hardware and observation model

The repository has a sophisticated resource and observation foundation:

- per-device VRAM claims;
- static weight, KV, overhead, and cache decomposition;
- RAM-resident versus mmap-addressed concepts;
- persistent runtime reservations;
- hardware and backend fingerprints;
- load, TTFT, prompt, generation, memory, page-fault, and storage-I/O observations;
- plan testing and fallback behavior.

The system is therefore close to being able to select plans from evidence rather than theory.

#### Runtime capability contract

The llama.cpp fork now uses a versioned backend procedure interface for MoE-cache capabilities and control. Cache support is discovered from the loaded backend rather than being universally hardcoded. Hybrid CPU-miss execution is now reported as unavailable, which accurately describes the implementation.

This is a major improvement over the previous review.

#### Upstream synchronization

The pinned fork has recently been synchronized onto a late-July upstream base. It is no longer approximately a month behind master. A small amount of normal post-sync drift remains, but the new integration cadence is functioning.

The fork should remain pinned and promoted only after hardware acceptance, rather than following master continuously.

## Blocking control-plane findings

### 1. Peak cache VRAM is not admitted consistently

`ResourceClaim` correctly separates:

- ordinary `vram_bytes`;
- and additional `vram_cache_bytes`.

Its device breakdown treats peak use as the sum of both. The planner also builds a per-device cache map.

However, several managed paths still use only `claim.vram_bytes`:

- worker feasibility checks;
- pending resource claims;
- runtime reservation acquisition;
- runtime event and run records;
- coexistence matrix calculations;
- RuntimeDB admission totals.

This means a cache-enabled plan can be admitted as though its dynamic cache reservation does not exist. Two individually valid plans may then be allowed to coexist despite their combined peak cache allocations exceeding available VRAM.

This is a direct violation of the hardware-aware serving goal.

#### Required change

Add one canonical method, for example:

```python
ResourceClaim.vram_admission_bytes()
```

It should return the peak per-device map:

```text
static weights + KV + execution overhead + reserved dynamic cache
```

Every feasibility, reservation, matrix, scheduling, and UI capacity calculation must use that method.

Keep the decomposed fields for explanation and telemetry, but never reconstruct admission semantics independently in callers.

#### Required integration test

Create a plan with a nonzero expert-cache budget, pass it through:

```text
planner
→ worker feasibility
→ RuntimeDB reservation
→ coexistence matrix
→ runtime event
```

Assert that every stage uses the same peak per-device values.

### 2. mmap-addressed model bytes are still treated as hard resident RAM in managed serving

The resource model distinguishes:

- total CPU-addressed model bytes;
- resident RAM;
- mmap/page-cache-backed bytes.

The direct plan-test path already reserves actual resident bytes and avoids charging the entire mmap model as permanently resident RAM.

The managed worker, RuntimeDB admission, and coexistence matrix still use `claim.ram_bytes`. For an SSD-backed oversized MoE, that value can represent the full CPU-addressed model even though much of it is mmap-backed and demand-paged.

The result is an inconsistent product:

- a plan may successfully run through browser testing;
- the same plan may be rejected by managed serving;
- matrix calculations may falsely conclude that models cannot coexist.

This blocks the exact RAM/SSD use case the project is intended to support.

#### Required change

Define canonical RAM admission semantics, for example:

```python
ResourceClaim.ram_admission_bytes()
```

A reasonable first policy is:

```text
explicit resident allocation
+ staging/working buffers
+ configured page-cache safety reserve
```

The complete mmap-addressed model size should remain visible as an informational and storage-pressure quantity, not a hard resident-RAM reservation.

Later, page-cache policy can become adaptive, but the first requirement is consistency.

### 3. Final plan environment is not fully included in launch identity

Backend resolution builds a substantial environment and produces an environment fingerprint. Capability probing is tied primarily to the binary.

After command construction, worker and test paths can still apply `plan.env` and adjust library paths. Those overrides may materially change runtime behavior.

This is not theoretical. Current hardware measurements show that changing `GGML_OP_OFFLOAD_MOE_MIN_BATCH` can reverse the performance ranking depending on whether the model is storage-bound or already fits in VRAM.

Two plans that differ only by a final environment override can therefore:

- share a command/environment identity;
- reuse observations that should be distinct;
- be capability-probed under a different environment than they launch under;
- show a preview that is not a complete representation of the final process.

#### Required change

`LaunchCommand` must contain the complete immutable launch environment.

The sequence should be:

1. Resolve binary.
2. Resolve base environment and runtime libraries.
3. Apply backend defaults.
4. Apply plan-specific overrides.
5. Probe the exact binary under the relevant environment.
6. Validate capabilities.
7. build argv;
8. fingerprint binary, argv, and a normalized whitelist of behavior-affecting environment variables.
9. Launch without any caller-side environment mutation.

Variables such as the MoE offload threshold, visible devices, oneAPI paths, backend selection, and allocation behavior should be part of the normalized identity.

## Runtime findings

### 4. The cache remains a transfer cache, not hybrid RAM/SSD execution

The runtime now truthfully reports hybrid CPU-miss execution as unsupported.

On a cache miss, the current path still relies on the normal host-to-device weight transfer and GPU execution. It does not yet:

- partition routed rows into GPU hits and CPU misses;
- execute misses directly on CPU-accessible weights;
- merge CPU and GPU outputs;
- avoid synchronous transfer for misses;
- provide a storage-aware miss path.

The cache can still be useful for prompt processing or sufficiently batched decode, but it does not yet fulfill the primary interactive oversized-MoE goal.

It should remain:

- disabled by default;
- labeled experimental;
- generated as an optional measured plan;
- selected only after a benchmark proves a benefit for that model and placement.

### 5. Real runtime cache geometry still assumes equal projections

The cache policy class can represent different gate, up, and down sizes, and host-side tests cover policy behavior.

The real lazy-initialization path still derives all three projection sizes from the first expert tensor and allocates slots on that assumption. Fused `gate_up_exps` layouts and models with unequal projection geometry can therefore be misclassified or rejected by guards.

The implementation has generalized policy code but not generalized runtime integration.

#### Required change

Either:

1. collect exact per-layer projection geometry before cache creation; or
2. initially key cached allocations independently by tensor/projection identity.

The second design may be simpler and safer for an experimental runtime because it avoids pretending that all architectures expose exactly three equal components.

Add integration fixtures for:

- separate gate/up/down tensors with unequal sizes;
- fused gate/up tensors;
- architecture-specific tensor naming;
- shared experts;
- partial cacheability.

### 6. Shared device-cache queue ownership remains unresolved

The cache is shared at device scope, which is directionally better than one independent full-budget cache per context. However, allocation and command submission remain tied to the context/queue that first initializes the shared cache.

That leaves unresolved behavior for:

- main plus draft/MTP contexts;
- multiple loaded models;
- context destruction order;
- queue lifetime;
- reset and metrics during unload.

The source itself acknowledges the need for real multi-context testing.

#### Required change

Make queue and ownership semantics explicit:

- device-owned cache and device-owned submission abstraction; or
- ref-counted cache with per-context queue submission that cannot outlive the context.

A main-model-plus-MTP test is more valuable than a generic synthetic two-context test because it exercises the intended product configuration.

### 7. Runtime build provenance remains incomplete

The binary hash gives modelctl a useful internal identity, but runtime capability metadata still reports incomplete build provenance, including placeholder or empty commit/compiler information.

Support bundles and the web UI should be able to answer:

- exact upstream base;
- fork commit;
- build type;
- compiler;
- enabled backends;
- dynamic/static backend mode;
- relevant compile flags.

Populate this from CMake-generated build metadata. Do not rely solely on repository manifests external to the binary.

## Web-product findings

### 8. Acquisition still has competing paths

The persistent `/add` workflow is the correct primary product flow. Legacy pull/import routes still coexist with it.

That creates uncertainty about which path owns:

- validation;
- job state;
- local-file verification;
- profile defaults;
- post-download analysis;
- plan generation;
- registration;
- warm-load testing.

#### Required change

Make `/add` the sole primary navigation entry.

Legacy routes should either:

- redirect into a pre-populated add workflow; or
- be clearly labeled as advanced/quick operations and call the same service layer.

There should be one authoritative acquisition operation, regardless of surface.

### 9. Network exposure should match the local operating model

This is a trusted-user local control plane, not a hardened multi-tenant service.

If the web UI is intentionally used across a trusted home LAN, binding to all interfaces may be a reasonable personal default. It should still be explicit in setup and easy to understand, because the application can launch processes and change runtime configuration.

This is a lower priority than resource-accounting and runtime-correctness work. A minimal improvement is sufficient:

- show the selected bind address clearly;
- distinguish loopback-only from LAN-accessible mode;
- warn when LAN access uses plain HTTP;
- preserve strong token authentication;
- document that untrusted-network exposure is unsupported.

There is no current need to build a general-purpose identity, permissions, or multi-tenant security system.

### 10. GitHub and public-repository polish are not project priorities

The public repositories exist mainly to make the code easy to inspect and review. Public clone instructions may be corrected when convenient, but public onboarding CI, release packaging, contributor workflows, and repository presentation should not displace work on the local system.

Repository metadata matters only where it supports:

- reproducing the current local installation;
- preserving a known-good runtime pair;
- reviewing changes;
- recovering from a failed upgrade.

## Local reproducibility and runtime maintenance

### Integration manifest drift

The integration manifest’s modelctl commit does not match the current modelctl HEAD. This may be intentional if it represents the last locally hardware-validated pair, but the field name does not make that distinction clear.

Use explicit fields such as:

```yaml
validated_modelctl_commit:
validated_llama_commit:
current_modelctl_commit:
upstream_base:
validation_report:
```

The UI and local support bundle should indicate when the running code is newer than the last validated pair. This is for future-you and rollback safety, not public release management.

### Upstream maintenance

The recent sync resolves the previous month-behind concern. Continue with:

- weekly drift reporting;
- focused review of scheduler, SYCL, server lifecycle, model loading, MTP, and memory changes;
- an integration branch every two to four weeks;
- immediate integration for relevant correctness fixes;
- promotion only after CPU, SYCL, correctness, modelctl integration, and hardware performance gates pass.

Do not make modelctl follow unpinned upstream master.

## Recommended priority order

### P0 — Resource and launch truth

1. Add canonical peak-VRAM admission including cache.
2. Add canonical resident-RAM admission for mmap plans.
3. Use both methods in planner, worker, matrix, RuntimeDB, tuning, runtime events, and UI.
4. Move all effective environment construction into immutable `LaunchCommand`.
5. Probe and fingerprint the exact final launch environment.
6. Add end-to-end launch-contract tests.

### P1 — Local workflow consolidation and runtime hardening

1. Consolidate acquisition around `/add`.
2. Make LAN exposure explicit enough for the intended trusted-network use.
3. Complete runtime build provenance and known-good local rollback metadata.
4. Fix real projection geometry integration.
5. Resolve shared-cache queue ownership.
6. Add main-plus-MTP and multi-model lifecycle tests.
7. Remove or redirect duplicate local workflows that create maintenance ambiguity.

### P2 — Primary MoE runtime goal

1. Partition selected expert work into GPU hits and misses.
2. Execute cache hits on GPU.
3. Execute misses directly on CPU-accessible RAM/mmap weights.
4. Merge outputs deterministically.
5. Cover batch-one interactive decode.
6. Add asynchronous promotion.
7. Record transfer, CPU-compute, GPU-compute, merge, page-fault, and storage-wait costs.
8. Compare against static mmap/RAM baselines before automatic selection.

### P3 — Evidence-driven automation

After the above is trustworthy:

- rank plans from matching hardware/backend observations;
- distinguish cold-page-cache, warm-page-cache, and warm-expert-cache results;
- invalidate stale recommendations on binary, driver, model, storage, or environment changes;
- display the selection reason and rejected alternatives in the browser.

## Updated local-first definition of done

The project does not need to be a polished public distribution. It fulfills its primary goal when its owner can:

1. Open the browser and import or discover an oversized MoE.
2. See exact VRAM, resident-RAM, mmap, and storage requirements.
3. Generate a bounded set of safe placement candidates.
4. Test them under explicit cold and warm conditions.
5. Verify correctness and measured performance.
6. Select or automatically accept a plan based on matching evidence.
7. Register and serve it through llama-swap.
8. Observe where weights are resident and where time is spent.
9. Unload, reload, fall back, and recover without terminal intervention.
10. Run interactive decode where GPU cache hits and CPU/RAM/mmap misses coexist without synchronous miss transfers.

## Final assessment

The project is now much closer to a coherent local system than it was during the prior review.

The web and control-plane architecture are strong. Recent runtime-interface and upstream-sync work removed several serious blockers. The project’s largest immediate risk is no longer missing functionality; it is inconsistent accounting between otherwise well-designed components.

Do not spend the next cycle turning the repository into a public software product. Spend it making the local machine behave predictably, making the browser workflow authoritative, and proving the MoE strategies with real measurements.

Fix peak-VRAM admission, mmap RAM semantics, and final launch identity before adding more cache sophistication. Those changes will make the platform trustworthy for the hardware it already supports.

Then focus runtime effort on the feature that actually completes the original vision: true batch-one hybrid expert execution across GPU VRAM and CPU-accessible RAM/mmap storage.
