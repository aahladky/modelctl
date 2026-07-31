# Architecture

The stable boundaries and principles of this stack. What each module
does is in `AGENTS.md`; how to operate the system is in
[operations.md](operations.md); runtime features are documented under
[runtime/](runtime/).

## Control plane / data plane

**modelctl (Python) is the control plane.** It inspects GGUF
structure; understands GPUs, RAM, and storage; generates candidate
launch plans; reserves memory (including dynamic expert-cache
budgets); selects and validates a compatible inference binary;
launches, tests, benchmarks, compares, and persists results; and
exposes all of it in the web console.

**The llama.cpp fork (`../llama.cpp`, pinned submodule) is the data
plane.** It observes router-selected expert IDs during inference,
keeps persistent expert slots in GPU memory, serves cache hits from
device memory, optionally executes misses on CPU (hybrid mode), and
exposes cache metrics.

The line is hard in both directions: no tensor execution or expert
routing in Python, no product policy in the fork. They talk through
exactly two interfaces:

1. the capability probe
   (`llama-server --modelctl-capabilities`, schema in
   [runtime/backend-capability-schema-v2.md](runtime/backend-capability-schema-v2.md));
2. the launched command line plus environment, plus Prometheus
   `/metrics` and `/cache/reset` at runtime.

## Principles

- **Web-first, not web-only.** The browser is the product entry
  point; the CLI is fully supported for bootstrap, automation,
  diagnostics, and recovery. New user-facing capability lands in the
  console first.
- **llama-swap remains the front door.** The public OpenAI-compatible
  endpoint (`127.0.0.1:9292`) and model selection by request ID belong
  to llama-swap; modelctl manages its config, never replaces it.
- **Measurements outrank estimates.** Planners generate candidates;
  launch and benchmark results decide. The system distinguishes
  estimated-to-fit / tested-successfully / tested-on-current-build /
  failed / stale-because-environment-changed.
- **Automatic decisions must be explainable.** Anything chosen
  automatically carries a decision trace: what was selected, why, and
  what was rejected on which grounds.
- **Bounded plan search.** Candidate generation is deterministic and
  deliberately limited (roughly 5–12 meaningful candidates), not a
  sweep of hundreds of variations.
- **Experimental features fail closed.** A cache or hybrid flag can
  only reach a binary whose capability probe affirmatively advertises
  the feature; unknown or unprobed binaries get the conservative
  path. Experimental plans additionally require a measured margin
  over the non-experimental baseline before being preferred.
- **One canonical launch path.** Every preview, artifact, smoke test,
  managed worker, and llama-swap entry derives from the same resolved
  backend and `LaunchCommand` (`modelctl_launch.py`). Command
  identity includes the binary, environment, and capability
  fingerprint, so what was tested is what launches.
- **Cold and warm measurements are never conflated.** A measurement
  records which caches were warm; cold means cold.
- **Additive, lazy migrations.** Profiles are plain JSON; profiles
  without new fields load identically to before
  (`normalize_profile()` fills defaults).

## Resource accounting

`ResourceClaim` (`modelctl_plans.py`) is the single admission
currency. VRAM decomposes into static weights + KV + overhead +
expert-cache reservation per device; the cache reservation uses the
runtime's uniform per-GPU budget on every device the profile's budget
map names. CPU-side bytes are split into resident RAM (mmap off)
versus mmap-addressed bytes whose residency is the page cache's
decision — reporting mmap-addressed bytes as required RAM is what
makes an oversized MoE look unservable. Storage-bound plans record
the backing block device so benchmark attribution can tell
storage-bound from compute-bound results.

## Web console

FastAPI + HTMX, thin over the same application services the CLI uses
(`modelctl_services/`). Reads run concurrently; long-running work
becomes background jobs in lanes (SQLite-backed), with profile and
config mutations serialized on the single-worker `mutation` lane
because profiles and the llama-swap config are plain files.
