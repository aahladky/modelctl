# LocalAI Phase 0 -- verify Stage 1 -- 2026-08-06

Phase 0 of the Stages 2-4 breakdown in `moe-review/localai-fork-plan.md`:
build what Stage 1 wrote, run it against modelctl, walk one acquisition, and
give the wizard an e2e spec. Stage 1 landed 1475 lines over 12 files and none
of it had ever executed.

It executes now. The build is green, the sidecar round-trips, the SSE tick
arrives, and one acquisition walks end to end. Seven payload fields the page
read did not exist in modelctl's answers; all seven are fixed and pinned by
specs that were confirmed to fail against the pre-fix page.

## Config

- Fork: `~/workspace/localai-fork`, branch `moe-serving`, `7bd3e91e`
  (`git describe` = `v4.8.1-4-g7bd3e91e`)
- Go 1.26.5, node v24.18.0, npm 11.16.0, Chrome 
  `/usr/bin/google-chrome` (via `PLAYWRIGHT_CHROMIUM_PATH`)
- LocalAI bound `127.0.0.1:9295`, `LOCALAI_DISABLE_GALLERY_ENDPOINT=true`
- Sidecar: live modelctl-web on `:9293` for the read-only checks; a hermetic
  scratch modelctl-web on `:9296` for the walk (see Safety)
- Machine: 28 threads, 31 GiB RAM. Load average 2.79 at the end of the run;
  llama-swap and OVMS untouched throughout, no model resident.

## Ports, confirmed rather than assumed

`ss -ltn` at the start of the session:

| Port | Holder |
|---|---|
| 9292 | llama-swap |
| 9293 | modelctl-web |
| 9294 | modelctl-remote-hands (python, pid 749671) |
| 9295 | free -- taken by this run |

This confirms the plan's correction: 9294 was never available, and the stale
`sidecar/client.go` comment naming it has been fixed.

## Build

`PATH="$(go env GOPATH)/bin:$PATH" make build` -- rc 0, `local-ai`
158 MB. `go build ./core/...` rc 0, `go vet ./core/sidecar/... 
./core/http/routes/...` rc 0.

**Finding: `make build` alone does not build this fork's UI.** The `react-ui`
target skips when `core/http/react-ui/dist` exists *at all* (Makefile:135) --
presence, not staleness -- and `core/http/app.go:48` embeds that directory with
`//go:embed`. The first build printed "react-ui dist already exists, skipping
build" and produced a 158 MB binary carrying a dist that predates every Stage 1
page. Rebuilding the bundle and relinking produced a 210 MB binary; the 52 MB
difference is how much UI was missing.

So the Phase 0 command is two steps, not one:

```
cd core/http/react-ui && npm run build
PATH="$(go env GOPATH)/bin:$PATH" go build -o local-ai ./cmd/local-ai
```

Upstream knows: their own `test-ui` target force-rebuilds the dist for exactly
this reason. The Makefile is not edited here -- it is an upstream file and the
guard is a deliberate container-build optimisation. The cost is a documented
recipe rather than permanent rebase surface.

## Sidecar reachability and the tick

All read-only, against the **live** modelctl on :9293.

| Call | Result |
|---|---|
| `GET /api/modelctl/status` | `{"available":true,"base_url":"http://127.0.0.1:9293","reason":""}` |
| `GET /api/modelctl/models` | 15 rows, first `gemma4-26b-mtp` |
| `GET /api/modelctl/wizards` | `[]` |
| `GET /api/modelctl/hf/search?q=qwen3+moe+gguf` | 15 results, keys `repo_id, downloads, likes, is_gguf, has_mtp, contents` |

`GET /api/modelctl/events` -- HTTP 200, `Content-Type: text/event-stream`,
`X-Accel-Buffering: no`. Ticks measured from stream open:

```
arrival (s): 0.67, 3.36, 6.04, 8.72
gaps    (s): 2.69, 2.68, 2.68
```

The plan expected "~2s". The configured interval *is* 2.0 s
(`app.py:141`); the observed period is 2.68 s because
`TelemetryCollector.snapshot()` runs before each yield and its probes (GPU,
llama-swap, jobs) cost ~0.68 s on this machine. **The tick period is interval
plus collection cost, not interval.** Anything that later gates on tick
punctuality should use the measured period.

Proxying works as designed: unbuffered, flushed per chunk, and the stream is
alive through LocalAI rather than pointed at :9293 by the browser.

## The walk

One acquisition through `/api/modelctl/*` -- same paths, bodies and order the
page issues. Two branches, because they answer different questions.

**Branch A -- Hugging Face source, stopped at inspect.** Going further would
submit a multi-gigabyte pull; the only question here is the shape of
`contents`.

```
POST /wizards                  200  step=source
POST /wizard/{id}/source       200  step=inspect   repo_id=unsloth/Qwen3.5-9B-GGUF
GET  /wizard/{id}/inspect      200  contents keys: mmproj_files, mtp_files, quant_groups
                                    quant_groups: 22 entries
                                    entry keys: files, label, sharded, total_size
                                    entry[0]: {"label":"Qwen3.5-9B-BF16",
                                               "files":["Qwen3.5-9B-BF16.gguf"],
                                               "sharded":false,
                                               "total_size":17920697312}
POST /wizard/{id}/delete       200  {"deleted":..., "cancelled_jobs":...}
```

**Branch B -- local GGUF, walked to `done`.** Source was a 4096-byte tiny GGUF
fixture (`write_tiny_gguf` from `test_console_phase2.py`) in the scratch models
dir.

```
POST /wizards                  200  step=source
POST /wizard/{id}/source       200  step=download   (local_file -> import job)
GET  /wizard/{id}              200  download job status=done progress=1.0
                                    profile_name=tiny-fixture download_complete=true
                                    job keys: cancellable, created, detail, error,
                                      finished, id, lane, progress, result_tail,
                                      router_reloaded, started, status, title, type
                                    job.message = null   <-- key the page was reading
POST /wizard/{id}/download/next 200 step=analyze
GET  /wizard/{id}/analyze      200  analysis keys: arch, block_count, embedding_length,
                                      expert_count, is_moe, kv_bytes_per_token,
                                      model_max_ctx, name, weight_bytes
                                    model_max_ctx=2048
                                    n_ctx_train=None  context_length=None  <-- keys read
POST /wizard/{id}/analyze/next 200  step=plans
GET  /wizard/{id}/plans        200  6 plans
                                    keys: admission, category, category_label, disabled,
                                      estimated, id, label, measured, pinned, reason,
                                      source, stale, tested, warnings
                                    p.id=5ff4976ad81706b5   p.plan_id=None
                                    estimated={"total_vram":1078263808,"ram":0,
                                               "mmap":0,"context":32768}
                                    measured=null   category_label="safe baseline"
POST /wizard/{id}/plans        200  step=test  selected_plan_id=5ff4976ad81706b5
POST /wizard/{id}/test/run     200  test_job_id=7961b2fa1548
     (test job settles)             status=failed  error="plan test failed: backend_crash..."
POST /wizard/{id}/test/next    200  step=register  measured={}
                                    step_gates.test.blocking_reason="the test step failed: ..."
POST /wizard/{id}/register     409  {"error":"not registering: the test step failed: ...",
      {"ctx":1024}                    "requires_register_untested":true}
POST /wizard/{id}/register     200  step=done
      {"ctx":1024,                    endpoint="http://127.0.0.1:9/v1/chat/completions"
       "register_untested":true}      registration_error=""
POST /wizard/{id}/register     422  {"error":"context length 999999 is above the model's
      {"ctx":999999,...}               trained maximum (2048, read from the GGUF header
                                       at analyze)"}
```

The test job's `backend_crash` is the scratch stub `llama-server` exiting 1, by
design -- it makes the failure path cheap to walk. It is not a fork defect, and
it is why the 409 gate could be exercised for real.

The 409 body carries `error` and `requires_register_untested` but **no
`state`**. The page's `fail()` guards on `err.body?.state` before using it, so
this degrades correctly; it is recorded because the page's comment implies a
`state` always rides along, and for this response it does not.

## Seven payload disagreements, all found and fixed

Every row was confirmed against a real modelctl answer above, not inferred.

| # | Page read | modelctl sends | Effect before the fix |
|---|---|---|---|
| 1 | `contents.quants \|\| contents.files` | `contents.quant_groups` | **Hard block.** A repo with 22 quantisations rendered "No GGUF quantisations found"; the HF branch could not advance past inspect. |
| 2 | `q.quant \|\| q.name \|\| q.filename` | `q.label` | Fell through to `String(q)` = `[object Object]`, both as the visible name and as the POSTed value -- which `_quant_download` matches against `label`, so it would resolve to no files. |
| 3 | `q.size_gib \|\| q.size` | `q.total_size` (bytes) | The one number the choice is about was never shown. |
| 4 | `job.message` | `job.detail` (+ `job.error`) | Downloads narrated nothing; a failed job showed a bare status badge with the reason discarded. |
| 5 | `analysis.n_ctx_train \|\| analysis.context_length` | `analysis.model_max_ctx` | The trained ceiling was never displayed, so the operator learned it from a 422. |
| 6 | `<Facts data={plan}/>` (skips nested objects) | numbers live in `estimated` / `measured` | **The compare-plans step rendered six rows of booleans and no VRAM figure** -- the one screen whose whole job is "will these weights fit these devices". |
| 7 | "Skip to register" re-posted `plans` | `POST test/next` folds the run in | A measurement the operator waited for was discarded, **and a failed one slipped past the gate that makes registering untested a deliberate act** (proven above: `test/next` sets `blocking_reason`, and register then 409s). |

Fixes are in `core/http/react-ui/src/pages/AddModel.jsx`, plus one
`.am-tag` rule in `add-model.css` and the corrected port comment in
`core/sidecar/client.go`.

Two behaviours were added rather than corrected, both because the walk exposed
a silence:

- A `contents` payload with no `quant_groups` key now reports a payload
  disagreement instead of an empty repository. Those are different facts.
- `mmproj` / MTP companion files are now named on the inspect step.
  `cmd_pull` skips them (`modelctl.py:1500`), so a vision model acquired here
  arrives without its projector. That is pre-existing modelctl behaviour, left
  alone and now merely visible.

## The spec, and proof it is not vacuous

`core/http/react-ui/e2e/add-model.spec.js` -- 8 tests, house style
(`coverage-fixtures.js`, `page.route()` stubs, mock backend). Every fixture in
it is a payload captured from the walk above, so the file is the contract
rather than a set of shapes invented to go green.

Red-then-green was run, because a spec that has only ever passed proves
nothing:

| Page under test | Result |
|---|---|
| Pre-fix `AddModel.jsx` (reverted, dist rebuilt, `ui-test-server` relinked) | **7 failed, 1 passed** |
| Fixed `AddModel.jsx` | **8 passed** (6.9 s) |

The one that passes in both states is "an unreachable sidecar degrades to a
named state" -- that behaviour was already correct, and the test pins it
against regression rather than against a bug.

## The rest of the suite: 127 tests Stage 1 orphaned

Running the whole suite for the first time since the fork surfaced a Stage 1
consequence the plan did not account for. The fork-discipline table scores
UNROUTE at "None" rebase cost, which is true of *merges* and not true of the
*test suite*: the gallery and importer specs still drive `/app/models` and
`/app/import-model`, which now redirect to `/app/add`.

Playwright collects **435 tests in 66 files**. Nine of those files are wholly
about the two unrouted surfaces, and they hold **127 tests**:

| Spec file | Tests |
|---|---|
| `models-gallery.spec.js` | 82 |
| `import-form-ux-batch-b.spec.js` | 14 |
| `import-form-ux-batch-d.spec.js` | 8 |
| `import-form-ux-batch-e.spec.js` | 8 |
| `import-form-ux-batch-a.spec.js` | 6 |
| `models-recommended-panel.spec.js` | 3 |
| `discover-height.spec.js` | 2 |
| `discover-search-focus.spec.js` | 2 |
| `import-form-ux-batch-f.spec.js` | 2 |

That they are *all* orphaned is established structurally rather than by tally,
which is the stronger claim: every `page.goto` across those nine files targets
`/app/import-model` (35 call sites) or `/app/models` (4), with one
`about:blank` used by a localStorage helper. There is no navigation to a page
this fork still routes, so no test in them can pass. The partial run agreed --
zero passes across all nine before it was stopped.

A suite that is permanently red is a suite nobody reads, and it would mask a
real regression in Phases 1-3 exactly when those phases start adding pages.

Resolved the same way the fork resolves the components themselves: the files
stay on disk untouched, and they are dropped from the run in ONE place, a
`testIgnore` list in `playwright.config.js`. A file we never touch merges for
free on every upstream release; a deletion is a conflict to re-read and
re-discard forever. Re-routing the gallery some day means deleting a line
there, not restoring files.

`navigation.spec.js` is the exception and gets a real edit: six of its seven
tests pass, and the failing one asserts the sidebar names "Discover" pointing
at `/app/models`. This fork made that change deliberately, so the assertion is
updated to "Add model" / `/app/add` rather than ignored. Confined to one test.

Collection after the change: **308 tests in 57 files**, of which 8 are the new
wizard specs and 0 are from the ignored nine (435 - 127 = 308, verified with
`playwright test --list`). The suite runs green in **45.8 s** where the
unignored version spent half an hour accumulating 30-second timeouts.

## The suite was writing to the live planning service

Found by accident while cleaning up, which is the worrying part. After the
first full run, live modelctl on :9293 held an acquisition wizard it had not
held before -- `f875eae735e2`, `step=source`, created 22:15:24 -- and
`/tmp/ui-test-server.log` records the test server fetching that exact id:

```
22:26:26  GET /api/modelctl/wizards            200
22:26:26  GET /api/modelctl/wizard/f875eae735e2 200
```

The path: `ui-test-server` registers the modelctl proxy like any other LocalAI
instance, and `sidecar.FromEnv()` falls back to `DefaultBaseURL` when
`LOCALAI_MODELCTL_URL` is unset -- which on this machine is the real service.
Every orphaned spec navigating to `/app/models` or `/app/import-model` now
lands on `/app/add` through the redirect, and none of them stub
`/api/modelctl/*`, so `AddModel` booted, found no open wizard, and POSTed one
into production. Exactly one, because the boot logic resumes an open wizard
rather than minting a second -- which is also why this was easy to miss.

Ignoring those specs stopped it happening, but by accident: the next spec to
reach `/app/add` unstubbed would do it again. The fix is a `webServer.env`
entry pointing the test server's sidecar at port 9 (discard), so an unstubbed
call fails as unreachable -- a state the UI already renders honestly -- instead
of quietly succeeding against something it must never touch.

The stray wizard was empty (no jobs, no profile) and was deleted through the
documented abandon endpoint (`{"deleted":"f875eae735e2","cancelled_jobs":[]}`).
Verified after the fix: a full guarded run leaves the live service at 0 wizards
and 15 models, its `/api/modelctl/*` calls answer 503, and the suite is still
304 passed / 4 skipped / 0 failed.

This is the only contamination of the live install found in this phase, and it
came from the test suite, not from the walk -- the walk was contained by
design.

## Rendered, not just round-tripped

The specs prove component logic against captured payloads and the walk proves
the payloads. Neither proves a person can open the page, so the real running
fork (9295 -> scratch modelctl 9296) was driven in Chrome:

```
/app/add cold      heading "Add a model"
                   rail: SOURCE INSPECT DOWNLOAD ANALYSE PLANS MEASURE REGISTER DONE
HF search          15 results rendered
inspect, real repo 22 quant radios
                   Qwen3.5-9B-BF16    16.7 GB
                   Qwen3.5-9B-IQ4_NL   5 GB
                   Qwen3.5-9B-IQ4_XS   4.8 GB
                   "This repository also ships 3 companion file(s) (mmproj / MTP)..."
                   occurrences of "No GGUF quantisations found": 0
console errors     none
page errors        none
```

22 radios against 22 `quant_groups` from the API walk, on the same repository:
the fix holds through the browser, not only through the stubs.

One incidental property worth writing down: **`waitUntil: 'networkidle'` never
fires on this page.** The SSE stream is an open connection by design, so the
network is never idle. Any future automation of these pages must wait on
`domcontentloaded` plus a selector.

Screenshots are not committed (they are large and regenerable); the recipe is
`playwright.chromium` with `executablePath=/usr/bin/google-chrome` against a
running instance.

## Safety

The walk registers a profile and submits a load. On the live install that path
writes llama-swap's config and restarts the service, which is forbidden. So the
mutating branch ran against a **hermetic scratch modelctl-web**, not the live
one: `MODELCTL_WEB_SCRATCH=1` plus all five redirections
(`MODELCTL_HOME`, `MODELCTL_MODELS_DIR`, `MODELCTL_LLAMA_SWAP_CONFIG`,
`MODELCTL_LLAMA_SWAP_SERVICE` -> a nonexistent unit,
`MODELCTL_LLAMA_SWAP_BASE_URL` -> `http://127.0.0.1:9/v1/`), with
`MODELCTL_LLAMA_SERVER` a stub. The launcher refuses to start if
`scratch_missing_redirections()` is non-empty, so hermeticity is enforced
rather than intended.

Containment verified after the walk:

- every write landed under the scratch root (`home/profiles/tiny-fixture.json`,
  `home/wizards/*.json`, `home/web_jobs.db`, `swap/config.yaml`)
- live modelctl still reports 15 models and 0 wizards
- `systemctl --user is-active llama-swap modelctl-web` -> `active active`
- the registered endpoint came back as `http://127.0.0.1:9/v1/chat/completions`
  -- the dead port, which is the redirection proving the live swap URL was
  never in play

Nothing under `~/services/`, `~/models/` or the live state dir was written.
Undo for everything in this section is deleting the scratch directory.

The one thing that *did* touch the live install came from the opposite
direction -- the e2e suite, not the walk. See "The suite was writing to the
live planning service" above: found, fixed, and the stray record removed.

## Not covered

- The browser walk stopped at the inspect step. Download, analyse, plans,
  measure and register have been walked over HTTP and rendered from stubs, but
  not clicked through in a browser end to end.
- The download step was exercised only via the local-file import path. A real
  multi-gigabyte HF pull, its progress bar and its cancel path are unwalked.
- The test step only ever failed (stub backend). A *successful* measurement
  folding `generation_tps` into `state.measured` is unproven end to end.
- `mmproj` / MTP acquisition remains unsolved, only visible.
- The quant list is a 20rem scroll container (`.am-list`), so 22 entries are
  cut mid-row with no visual signal that the list continues. It scrolls and
  works; it does not say it scrolls. Noted, not fixed -- Phase 1 revisits this
  page family.
- Phase 0 says nothing about placement, fleet or cache pages; those are
  Phases 1-3.
