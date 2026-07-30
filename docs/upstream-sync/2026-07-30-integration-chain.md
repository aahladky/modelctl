# Task 0.8 — modelctl integration chain validation

Validates the full chain against the ported+corrected runtime
(`sync/moe-cache-upstream-2026-07-30`, commit `2dbf94801`,
`build-sycl-sync/bin/llama-server`) using a throwaway profile
(`moe-sync-test`, pinned `binary`, pointed at the Task 0.6 tiny synthetic
MoE model, `enabled: false`). No real profile, `llama-swap` config, or
router/web systemd unit was touched. The profile and its capability-cache
entry were deleted after testing; `modelctl list` and `ps aux` confirm no
trace remains.

## Chain results

| Link | Result |
|---|---|
| Runtime capability probe | ✅ Live probe (`--modelctl-capabilities`) returns schema-2 JSON; matches cached read exactly |
| Capability normalization | ✅ `get_cached_capabilities` (no subprocess) returns identical content to a live `probe_backend` call |
| Plan generation / preflight | ✅ `preflight()` resolves the pinned binary, auto-fixes missing `LD_LIBRARY_PATH` for this session only (does not mutate the saved profile) |
| **Cache validation (fail-closed)** | ✅ **and found a real gap** — see below |
| Browser preview / CLI preview | ✅ `build_server_args()` (the single function backing both preview surfaces) produces a command containing only supported flags |
| Plan test | ✅ `cmd_test` → `smoke_test_profile` launches, health-checks, and runs a completion; see caveat below |
| Managed worker / llama-swap command | ✅ `render_llama_swap_entry()` renders `effective_bin --port ${PORT} <same args>`, byte-identical (modulo port) to the preview/test command |
| Launch | ✅ Direct invocation of the exact modelctl-generated command loads the model and serves a completion (`tokens_evaluated: 48, tokens_predicted: 16`) |
| Metrics | ✅ `/metrics` (with `--metrics`, which `build_server_args` includes) shows real cache activity: 48 misses, 46 evictions, 48 promotions, 2/2 slots used, 1,572,864 H2D bytes |
| Unload/reload | ✅ Clean `terminate()`/`poll()==0` shutdown, then a full second launch on the same port completed a request identically |
| Binary fingerprint changes after sync | ✅ **Confirmed via the real mechanism** — see below |
| Submodule pointer not advanced | ✅ **Confirmed** — see below |

## A real gap found: fail-closed validation would reject the live production profile today

`preflight_moe_cache()` correctly fails closed: it blocks (`mode=manual` →
error, not just a warning) when `moe_cache.decode.miss_execution == "cpu"`
and the backend reports `moe_hybrid_cpu_miss: false` (which it does, and
should, everywhere in this project right now). This is exactly the
fail-closed behavior Task B2 asks for, and it worked correctly for a
throwaway profile constructed with that setting.

**But** the real production profile `qwen3.5-122b-iq1m`
(`~/.local/share/modelctl/profiles/qwen3.5-122b-iq1m.json`) has this exact
field set: `"decode": {"miss_execution": "cpu", "admit_to_gpu_cache": true}`.
If `cmd_test`, `render_llama_swap_entry`, or any other path that calls
`preflight_moe_cache` were run against that real profile today, it would
**block** with `error: decode.miss_execution=cpu requested but backend
lacks moe_hybrid_cpu_miss` — even though the profile is currently running
fine in production via a statically pre-generated `run.sh` that predates
this validation and never goes through it. This is precisely the kind of
canonical-path inconsistency Phase B (Task B1) exists to hunt down: the
static, once-generated artifact and the live preflight path disagree about
whether this profile's config is valid. `miss_execution: cpu` doesn't
currently do anything anyway (hybrid CPU-miss execution doesn't exist yet
per Task 0.4/G), so the config value itself is stale/aspirational, not
presently harmful — but it will actively block that profile's next
`modelctl regen`/plan-test/web-wizard-retest until either the field is
corrected or the validation is loosened for the not-yet-implemented case.
Did not touch the real profile to fix this (out of scope, and risky to
edit production config as a side effect of an integration test) — flagging
for whoever owns Phase B/G next.

## Fingerprint mechanism: real, and confirmed working

`modelctl_capabilities._binary_fingerprint()` computes a full SHA-256 of
the binary's file contents (truncated to 16 hex chars) as the on-disk
cache key (`backend_capabilities/<fingerprint>.json`) — this is a genuine
content-based fingerprint, not the empty `build.commit` string inside the
capability JSON payload itself (which Task 0.1's baseline already correctly
flagged as always empty — that's a *runtime self-report* gap, separate
from this *modelctl-side caching* mechanism, which works correctly):

```
pre-sync  build-sycl/bin/llama-server:      7cc5d96af8216629
post-sync build-sycl-sync/bin/llama-server: 0a57af55eb73fb32   (differs)
```

Each binary gets its own separate cache file; probing the new binary did
not disturb or overwrite the old one's cached observation, and the new
entry was deleted along with the throwaway profile's other artifacts after
testing — no stale-vs-fresh confusion is possible since they're keyed
independently by content hash.

## Submodule pointer: confirmed not advanced

```
$ git submodule status
 e7af6cf1996cb3d850963213b043d876ffe959ea llama.cpp (moe-cache-pre-upstream-2026-07-30)
```

Still the pre-sync tag/commit. Task 0.9 (promotion) has not happened, as
expected — this task only exercises the integration chain against the
`sync/...` branch's build artifacts directly, it doesn't touch the pinned
submodule commit.

## Test-harness caveat (not a modelctl bug)

`cmd_test`'s smoke test requests chat-formatted output and a structured
("peg-native") response; the Task 0.6 tiny synthetic model has *untrained,
random* weights, so its raw token output is gibberish that the structured-
output parser correctly rejects with a 500. This is expected given random
weights, not a chain defect — confirmed by driving the exact same
modelctl-generated command directly against the raw `/completion` endpoint
(bypassing the chat/structured-output parser), which succeeded and showed
real cache metrics as documented above.
