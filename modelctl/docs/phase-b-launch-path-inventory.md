# Phase B — command-construction inventory

**Date:** 2026-07-30
**Roadmap:** `modelctl-task-by-task-roadmap-2026-07-30.md`, Phase B (Tasks B1–B4)

Phase B is an integration audit, not another abstraction. The canonical
types already existed; the question was whether every production path
actually went through them. It did not.

## The one path

```text
resolve_backend(profile)                    # binary + env + capabilities, once
  └─ modelctl.preflight(auto_fix=True)      # binary resolution, oneAPI env
  └─ modelctl_capabilities.probe_backend()  # --modelctl-capabilities

build_launch_command(profile, plan, backend, port)
  └─ modelctl.preflight_moe_cache()         # fail-closed cache validation
  └─ backends.get_backend(name).build_command()
       └─ modelctl_worker._build_command()  # [binary] + plan.argv + --port
            └─ plan.argv  ← modelctl_plans._make_plan()
                              └─ modelctl.build_server_args()

launch_command_for_profile(profile, port=None)
  └─ modelctl_plans.current_profile_plan()  # "this profile as saved"
  └─ build_launch_command(...)
```

`modelctl.canonical_launch_command(profile, port=None)` is the thin
wrapper the rendering surfaces call; it returns
`(LaunchCommand, ok, messages)` with the legacy message strings the CLI
and job logs already print.

## Inventory of call sites (as found)

| # | Site | Purpose | Went through canonical path? | Defect |
|---|---|---|---|---|
| 1 | `modelctl_plans._make_plan` | plan argv | yes — *is* the argv generator | — |
| 2 | `modelctl_worker.worker_main` | managed worker launch | yes | validation check was hand-rolled |
| 3 | `modelctl_tune.test_launch_plan` | plan-test process | yes | validation check was hand-rolled |
| 4 | `modelctl.render_llama_swap_entry` | llama-swap config | **no** | own `preflight()` + `build_server_args()` |
| 5 | `modelctl.generate_artifacts` | generated `run.sh` | **no** | own `preflight()` + `build_server_args()` |
| 6 | `modelctl.smoke_test_profile` | smoke test | **no** | own `preflight()` + `build_server_args()` |
| 7 | `modelctl_web.app.profile_edit` | browser command preview | **no** | probed the binary the profile *names*, not the one preflight resolves |
| 8 | `modelctl_web.app.profile_runsh` | browser `run.sh` preview | **no** | no capabilities passed **at all**, no binary in the output |

Sites 4–8 were five independent reconstructions of the command *after*
validation — exactly what Task B1 prohibits. All five now call
`canonical_launch_command()`.

`modelctl_backends.LlamaCppAdapter.build_command` and
`modelctl_worker._build_command` remain as internal adapter
implementation, which B1 permits; no caller reaches them without going
through backend resolution first.

## Defects this surfaced

1. **Preflight ran twice per launch, and the two runs could disagree.**
   `resolve_backend()` called `preflight()` for the binary, then
   `build_launch_command()` called it again for the messages. Auto-fix
   searches the filesystem, so the second call was free to land on a
   different binary than the argv had been built around (Task B4).

2. **The worker launched with an environment no artifact exported.**
   `resolve_backend()` sourced the first oneAPI env script it found for
   *every* llama-cpp profile; `preflight()` sourced one only for SYCL
   profiles missing `LD_LIBRARY_PATH`. The generated `run.sh` used
   preflight's answer and the worker used the other one. Environment
   resolution is now preflight's alone.

3. **`ResolvedBackend.environment` is the full process environment**, so
   the rendering surfaces could not use it — writing it into `run.sh` or
   the llama-swap `env:` block would publish every inherited variable.
   Split into `environment` (launch) and `environment_overrides`
   (render); `test_launch_truth.TestArtifactsExportOnlyProfileEnvironment`
   holds the line.

4. **The browser preview probed the wrong binary.** `profile_edit` read
   cached capabilities for `profile["binary"] or LLAMA_SERVER_BIN`, which
   `resolve_backend()`'s own docstring notes "frequently doesn't exist or
   doesn't support the device". The preview could therefore show cache
   flags the launch would drop, or vice versa.

5. **`/profiles/{name}/run.sh` passed no capabilities**, making it the one
   surface with no fail-closed gate on experimental flags.

6. **`capability_digest` was a dead column.** `plan_runs` has carried it
   since Task 1.6 and nothing ever wrote it, so no observation could be
   staled by a runtime whose *reported* capabilities changed. Added
   `modelctl_capabilities.capability_fingerprint()` and wired it through
   `ResolvedBackend` into both the worker and plan-test runs.

7. **`build_launch_command(port=None)` produced `--port None`.** Nothing
   hit it because every caller passed a port, but the rendering surfaces
   need a portless argv. `port=None` now means "no `--port` at all", which
   is what lets the six surfaces compare equal.

## Acceptance

`test_launch_truth.py` runs against a real fake binary on disk rather
than a mocked probe — the failure mode being tested is a path that
resolves a *different* binary or skips the probe, which mocking hides.

- `TestStockBinaryNeverGetsCacheFlags` — B2 acceptance. A stock upstream
  `llama-server` plus a profile with every cache setting enabled in
  `manual` mode: no `--moe-*` flag appears in the canonical command, the
  plan argv, the generated `run.sh`, the llama-swap entry, or the browser
  preview; the command is `is_valid == False`; the smoke test refuses to
  launch and the llama-swap entry reports `ok == False`.
- `TestCommandEqualityAcrossSurfaces` — B3. All six surfaces share one
  `command_fingerprint`, and the two shell-rendered surfaces re-tokenize
  to the same argv. Port is the only permitted difference.
- `TestObservationProvenance` — B3 provenance: binary, environment, and
  capability fingerprints are all populated and behave correctly under
  change.

## Still open in Phase B

- `LaunchCommand.raise_for_errors()` exists and the worker/plan-test/
  smoke-test/artifact/llama-swap paths call it or `is_valid`. The web
  *plan* pages surface `validation` but the plans page does not yet render
  the messages (Task B2's "expose all validation messages in web plan and
  runtime pages" is partially done — `profile_edit` now receives
  `validation` and `warnings` in its template context, but
  `profile_edit.html` does not render them yet).
- Automatic planning omitting unsupported cache candidates is handled by
  `compile_launch_plans(include_experimental=...)` plus the fail-closed
  `build_moe_cache_args`, but has no dedicated test at the plan-selection
  level.
