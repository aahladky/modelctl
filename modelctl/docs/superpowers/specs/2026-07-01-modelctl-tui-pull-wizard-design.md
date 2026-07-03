# modelctl TUI pull wizard — design

Date: 2026-07-01
Status: approved, not yet implemented

## Context

`modelctl.py` pulls GGUF models from Hugging Face and generates llama-swap /
router-mode / Ollama configs from saved profiles. The interactive parts of
this (`cmd_pull`, `prompt_config`) are a sequential chain of `input()`
prompts. This session's earlier work deliberately split business logic out
of those flows into pure, print-free functions (`search_models()`,
`get_repo_contents()`, `router_status()`, etc.) specifically so a future TUI
could reuse them without rewriting logic — see their docstrings.

This spec covers the first concrete TUI surface: a wizard that replaces
`cmd_pull`'s prompt chain with a proper interactive flow. It's step 3 of a
larger, informally-scoped "should modelctl grow a TUI" direction; the
dashboard/search/profile-management screens that would eventually surround
this wizard don't exist yet and are explicitly out of scope here.

## Decisions made during brainstorming

These were resolved via clarifying questions before this design was
written; recorded here so the reasoning doesn't get lost:

- **Standalone app, not embedded in a shell.** No dashboard/search/status
  app shell exists yet. Building the wizard as its own `App` avoids
  committing to shell architecture before there's enough surface (a second
  or third screen) to inform it. It gets absorbed into a larger app later
  if one gets built.
- **Search is step one**, not a separate prerequisite. `modelctl pull --tui`
  with no arguments starts at search; `search_models()` was already
  TUI-ready, so this cost little to include and makes the wizard usable
  standalone rather than requiring `modelctl search` first.
- **One profile per wizard run.** Today's CLI supports picking multiple
  quants in one `pull` and sharing config across them. The wizard does not
  replicate this — running the wizard twice for two quants from the same
  repo is an acceptable cost for a simpler state machine on the first pass.
- **Download step is a spinner + status text, not a progress bar.** A real
  byte-level progress bar requires wiring a progress callback through
  `hf_hub_download` and pushing thread-safe updates into the UI. Deferred;
  a background worker with a busy indicator is enough to not freeze the UI
  during a multi-GB download.
- **Auto-apply at the end, matching today's CLI.** The wizard's last step
  saves the profile, generates artifacts, syncs both backend configs
  (restarting the router if its preset changed), and syncs Hermes — same as
  `cmd_pull` today, with no separate confirm/apply gate.
- **Preflight issues warn, they don't block.** If `preflight()` can't fully
  resolve the binary/env (missing llama-server, missing SYCL env vars), the
  wizard shows a warning but still lets you finish and save the profile —
  matching current CLI behavior. Fixing the underlying issue and running
  `modelctl regen <name>` later remains the recovery path.
- **One `Screen` per step**, navigated via `push_screen`/`pop_screen`, over
  a single screen with a reactive step variable. Chosen for testability
  (each step screen can be driven independently via Textual's `Pilot`
  harness) and to keep each screen's responsibility narrow, consistent with
  how the rest of `modelctl.py` was factored this session. The "shared
  breadcrumb" benefit of the single-screen alternative is recovered with a
  small reusable `StepIndicator` widget mounted in each screen instead.

## Screen flow

```
Search -> Pick quant -> [Vision/MTP, only if repo has them] -> Configure
   -> Name profile -> Download -> Save & sync -> Done
```

`Vision/MTP` is conditionally skipped, same as today's CLI conditionally
skips its mmproj/MTP prompts when a repo has none.

After `Done`, the app exits — consistent with "one profile per wizard run"
above. There is no in-app "pull another" loop; a second profile is a fresh
`modelctl pull --tui` invocation.

## Architecture

New file: `modelctl_tui.py`, alongside `modelctl.py`. It imports from
`modelctl.py` and adds **no new business logic** — every screen calls an
existing pure function. Seven screens cover the eight-step flow above:
`SummaryScreen` performs the save/sync on mount and then displays the
"Done" state itself, rather than the flow's last two boxes being separate
navigable screens.

- `SearchScreen` — `search_models()`
- `QuantPickScreen` — `get_repo_contents()`, `group_files()` (via
  `get_repo_contents`)
- `VisionMtpScreen` — reads `mmproj_files`/`mtp_files` already fetched by
  `get_repo_contents()` in the previous screen
- `ConfigureScreen` — `load_defaults()` for pre-fill, `preflight()` to
  validate on submit
- `NameScreen` — `next_unique_profile_name()`, `slugify()`,
  `strip_quant_from_label()` for the profile name; also collects
  `dest_dir`, pre-filled from `DEFAULT_MODELS_DIR` and editable as a second
  field on this same screen (today's CLI prompts for it separately; the
  wizard folds it in here rather than giving it its own screen, since it's
  almost always accepted as-is)
- `DownloadScreen` — `download_if_needed()`, run via a Textual `@work`
  background worker so a multi-GB download doesn't freeze the UI
- `SummaryScreen` — `save_profile()`, `generate_artifacts()`,
  `sync_all_backends()`, `sync_hermes_custom_providers()` (identical
  sequence to `cmd_pull`'s tail today), then displays the result

`textual` is imported lazily inside the `--tui` code path in `cmd_pull`, so
it does not become a hard dependency for the rest of modelctl. If it's not
installed and `--tui` is passed, fail with a clear message naming the
package to install rather than a bare `ImportError` traceback.

Entry point: `modelctl pull --tui` (no `repo_id` argument in TUI mode,
since search is step one).

## Components

**`WizardState`** — a plain dataclass carried through screen constructors:

```python
@dataclass
class WizardState:
    repo_id: str | None = None
    quant_group: dict | None = None       # one entry from get_repo_contents()["quant_groups"]
    mmproj_choice: dict | None = None     # one entry from mmproj_files, or None
    mtp_choice: dict | None = None        # one entry from mtp_files, or None
    dest_dir: str = ""
    config: dict | None = None            # same shape prompt_config() returns today
    profile_name: str = ""
    warnings: list = field(default_factory=list)
```

**`StepIndicator`** — a small reusable widget (`Static` subclass) rendering
the breadcrumb (e.g. `Search > Pick quant > Configure > ...` with the
current step highlighted), mounted at the top of every screen's `compose()`.
Takes the current step name and the static step list as constructor args —
no shared mutable state, just a label.

**Screen transition logic** — factored as a standalone pure function (not a
method tangled into a Screen class) so it's testable without instantiating
Textual at all:

```python
def next_screen_after(current: str, state: WizardState) -> str:
    """Given the step just completed and the wizard state so far, return
    the name of the next step. Only place the Vision/MTP skip logic lives."""
```

This keeps "does Vision/MTP get skipped when the repo has neither" a plain
unit test, not a Pilot-driven UI test.

## Data flow

Linear pipeline — the same shape as `cmd_pull`'s local variables today
(`repo_id`, `chosen_groups`, `mmproj_chosen`, `mtp_chosen`, `shared_config`,
`name`), just spread across screens via `WizardState` instead of one
function's sequential locals. No new data model, no new persistence format
— the wizard produces the exact same profile JSON `cmd_pull` does today.

## Error handling

- **Search/network failure** (`SearchScreen`): show an inline error, let
  the user retry the query. Does not lose wizard state (there isn't any
  yet at this step).
- **Bad repo** (`QuantPickScreen`, `get_repo_contents()` raises or returns
  no quant groups): show an error with a "back to search" action.
- **Preflight issues** (`ConfigureScreen`, on submit): non-blocking warning
  banner, per the decision above. Warnings are also carried into
  `WizardState.warnings` so `SummaryScreen` can display them again in the
  final summary rather than losing them once dismissed.
- **Download failure** (`DownloadScreen`): stop and show an error with a
  retry action. Do not advance to `SummaryScreen` with a profile pointing
  at a file that isn't actually on disk.
- **Uncaught exceptions**: rely on Textual's built-in crash screen rather
  than building custom top-level exception handling — acceptable default
  for a first version.

## Testing

- The existing 62 tests in `test_modelctl.py` are untouched — the wizard
  never modifies `modelctl.py`'s logic, only calls it.
- New tests in a `test_modelctl_tui.py`, using Textual's `Pilot` test
  harness (`async with app.run_test() as pilot: ...`) to drive
  `PullWizardApp` screen-by-screen, with the same
  `mock.patch.object(modelctl, "search_models", ...)`-style mocking
  already used throughout `test_modelctl.py` for the underlying functions.
- `next_screen_after()` gets plain (non-Pilot) unit tests covering: repo
  with both mmproj and MTP, repo with neither (skip straight to
  Configure), repo with only one of the two.
- No new tests needed for the pure functions the wizard calls — they're
  already covered where they're defined.

## Out of scope (deliberately deferred)

- Multi-quant selection / batch profile creation in one wizard run.
- Byte-level download progress bar.
- Any app shell, dashboard, or navigation between this wizard and other
  future screens (profile list/edit, router status, etc.).
- Replacing or deprecating the existing `input()`-based `cmd_pull` — it
  stays as the non-`--tui` default.
