# modelctl TUI Pull Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `modelctl.py`'s `input()`-chain pull flow with an optional Textual TUI wizard (`modelctl pull --tui`) that produces the exact same profile JSON, reusing existing pure functions with no new business logic.

**Architecture:** One new file, `modelctl_tui.py`, holding a `WizardState` dataclass, a pure `next_screen_after()` transition function, a `StepIndicator` widget, seven `Screen` subclasses (Search, QuantPick, VisionMtp, Configure, Name, Download, Summary), and a `PullWizardApp`. `modelctl.py` gets a `--tui` flag on the `pull` subcommand that lazily imports and launches it. Every screen calls existing functions from `modelctl.py` (`search_models`, `get_repo_contents`, `preflight`, `download_if_needed`, `save_profile`, `generate_artifacts`, `sync_all_backends`, `sync_hermes_custom_providers`) — this file adds interaction only.

**Tech Stack:** Python 3, Textual (new dependency, lazily imported), the existing `modelctl.py`/`test_modelctl.py` (unittest + `unittest.mock`).

**Spec:** `docs/superpowers/specs/2026-07-01-modelctl-tui-pull-wizard-design.md`

---

## Before you start

Install Textual in the dev environment (not a hard runtime dependency for the rest of modelctl — see Task 3):

```bash
pip install textual --break-system-packages
python3 -c "import textual; print(textual.__version__)"
```

All file paths below are relative to the modelctl repo root (`/home/aaron/workspace` on the host).

---

### Task 1: `WizardState` and `next_screen_after()`

The only piece of new logic that isn't a Textual widget — a pure function, so it's the natural place to start and needs no UI framework to test.

**Files:**
- Create: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing tests**

Create `test_modelctl_tui.py`:

```python
import unittest

from modelctl_tui import WizardState, next_screen_after


class TestNextScreenAfter(unittest.TestCase):
    def test_search_always_goes_to_quant(self):
        state = WizardState()
        self.assertEqual(next_screen_after("search", state), "quant")

    def test_quant_goes_to_vision_mtp_when_repo_has_mmproj(self):
        state = WizardState(repo_contents={"mmproj_files": [{"name": "mmproj-F16.gguf"}], "mtp_files": []})
        self.assertEqual(next_screen_after("quant", state), "vision_mtp")

    def test_quant_goes_to_vision_mtp_when_repo_has_mtp(self):
        state = WizardState(repo_contents={"mmproj_files": [], "mtp_files": [{"name": "model-mtp.gguf"}]})
        self.assertEqual(next_screen_after("quant", state), "vision_mtp")

    def test_quant_skips_vision_mtp_when_repo_has_neither(self):
        state = WizardState(repo_contents={"mmproj_files": [], "mtp_files": []})
        self.assertEqual(next_screen_after("quant", state), "configure")

    def test_quant_skips_vision_mtp_when_repo_contents_missing(self):
        # Defensive: repo_contents should always be set by the time this
        # runs, but don't crash if it isn't.
        state = WizardState(repo_contents=None)
        self.assertEqual(next_screen_after("quant", state), "configure")

    def test_vision_mtp_goes_to_configure(self):
        state = WizardState()
        self.assertEqual(next_screen_after("vision_mtp", state), "configure")

    def test_full_chain_after_configure(self):
        state = WizardState()
        self.assertEqual(next_screen_after("configure", state), "name")
        self.assertEqual(next_screen_after("name", state), "download")
        self.assertEqual(next_screen_after("download", state), "summary")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `ModuleNotFoundError: No module named 'modelctl_tui'` (file doesn't exist yet)

- [ ] **Step 3: Write the minimal implementation**

Create `modelctl_tui.py`:

```python
"""
modelctl_tui - Textual-based interactive wizard for `modelctl pull --tui`.

This file adds interaction only. Every screen calls an existing pure
function from modelctl.py (search_models, get_repo_contents, preflight,
download_if_needed, save_profile, generate_artifacts, sync_all_backends,
sync_hermes_custom_providers) -- no business logic is duplicated here.
See docs/superpowers/specs/2026-07-01-modelctl-tui-pull-wizard-design.md.
"""
from dataclasses import dataclass, field

import modelctl

STEP_ORDER = ["search", "quant", "vision_mtp", "configure", "name", "download", "summary"]


@dataclass
class WizardState:
    """Carried through wizard screens via constructor args. Mirrors the
    shape of cmd_pull's local variables today -- no new persistence format,
    the wizard produces the same profile JSON cmd_pull does."""
    repo_id: str | None = None
    repo_contents: dict | None = None  # get_repo_contents() result, set once a repo is picked
    quant_group: dict | None = None    # one entry from repo_contents["quant_groups"]
    mmproj_choice: dict | None = None  # one entry from repo_contents["mmproj_files"], or None
    mtp_choice: dict | None = None     # one entry from repo_contents["mtp_files"], or None
    dest_dir: str = str(modelctl.DEFAULT_MODELS_DIR)
    config: dict | None = None         # same shape prompt_config() returns today
    profile_name: str = ""
    warnings: list = field(default_factory=list)


def next_screen_after(current: str, state: WizardState) -> str:
    """Given the step just completed and the wizard state so far, return
    the name of the next step. Only place the Vision/MTP skip logic lives."""
    idx = STEP_ORDER.index(current)
    nxt = STEP_ORDER[idx + 1]
    if nxt == "vision_mtp":
        contents = state.repo_contents or {}
        has_extras = bool(contents.get("mmproj_files")) or bool(contents.get("mtp_files"))
        if not has_extras:
            return STEP_ORDER[idx + 2]  # skip straight to "configure"
    return nxt
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK` (7 tests pass)

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Add WizardState and next_screen_after() for the TUI pull wizard

Pure logic only, no Textual dependency yet -- the Vision/MTP skip logic
lives in one place and is unit-testable without a UI test harness."
```

---

### Task 2: `--tui` flag and lazy Textual import

Wire the entry point before building any screens, so later tasks have somewhere to plug in and the "textual not installed" failure mode is handled early.

**Files:**
- Modify: `modelctl.py` (argparse `pull` subcommand, `cmd_pull`)
- Test: `test_modelctl.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl.py`:

```python
class TestPullTuiFlag(unittest.TestCase):
    def test_tui_flag_makes_repo_id_optional(self):
        parser = modelctl.build_arg_parser()
        args = parser.parse_args(["pull", "--tui"])
        self.assertTrue(args.tui)
        self.assertIsNone(args.repo_id)

    def test_repo_id_still_required_without_tui(self):
        parser = modelctl.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["pull"])

    def test_cmd_pull_dispatches_to_tui_when_flag_set(self):
        args = argparse.Namespace(tui=True, repo_id=None, no_hermes=False)
        with mock.patch.object(modelctl, "run_pull_wizard") as mock_wizard:
            modelctl.cmd_pull(args)
        mock_wizard.assert_called_once()
```

Add `import argparse` to the top of `test_modelctl.py` if not already present (it isn't — check first with `grep -n "^import argparse" test_modelctl.py`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl.TestPullTuiFlag -v`
Expected: `AttributeError: module 'modelctl' has no attribute 'build_arg_parser'` (the parser is currently built inline in `main()`, not its own function)

- [ ] **Step 3: Write the minimal implementation**

In `modelctl.py`, find the `main()` function (it currently builds the parser inline). Extract parser construction into its own function so it's testable without calling `sys.exit` via `main()`'s error paths, then add the `--tui` flag and make `repo_id` conditionally required:

```python
def build_arg_parser():
    parser = argparse.ArgumentParser(prog="modelctl")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="search Hugging Face for models")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--limit", type=int, default=15)
    p_search.add_argument("--tag", action="append", choices=["mtp"],
                           help="filter to repos matching this tag (repeatable); currently only 'mtp' is supported")
    p_search.add_argument("--min-gb", type=float, default=None,
                           help="only show repos with at least one quant at or above this size in GB")
    p_search.add_argument("--max-gb", type=float, default=None,
                           help="only show repos with at least one quant at or below this size in GB")
    p_search.add_argument("--sort", default="downloads", choices=["downloads", "likes", "lastModified"],
                           help="sort order for results (default: downloads)")
    p_search.set_defaults(func=cmd_search)

    p_pull = sub.add_parser("pull", help="pull a model from a HF repo and configure it")
    p_pull.add_argument("repo_id", nargs="?", default=None,
                         help="required unless --tui is passed (the wizard starts at search)")
    p_pull.add_argument("--tui", action="store_true",
                         help="use the interactive Textual wizard instead of the input()-based prompts")
    p_pull.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_pull.add_argument("--no-router-restart", action="store_true",
                         help="don't restart the router-mode systemd service after updating its preset")
    p_pull.set_defaults(func=cmd_pull)

    p_list = sub.add_parser("list", help="list saved profiles")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show a saved profile's JSON")
    p_show.add_argument("name")
    p_show.set_defaults(func=cmd_show)

    p_edit = sub.add_parser("edit", help="re-prompt config for a saved profile")
    p_edit.add_argument("name")
    p_edit.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_edit.add_argument("--no-router-restart", action="store_true",
                         help="don't restart the router-mode systemd service after updating its preset")
    p_edit.set_defaults(func=cmd_edit)

    p_regen = sub.add_parser("regen", help="regenerate artifacts from a saved profile")
    p_regen.add_argument("name")
    p_regen.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_regen.add_argument("--no-router-restart", action="store_true",
                          help="don't restart the router-mode systemd service after updating its preset")
    p_regen.set_defaults(func=cmd_regen)

    p_sync = sub.add_parser("sync", help="push all profiles into the live llama-swap config.yaml")
    p_sync.add_argument("--no-hermes", action="store_true", help="don't update Hermes Agent config")
    p_sync.add_argument("--hermes-dry-run", action="store_true", help="show what Hermes config would change")
    p_sync.add_argument("--no-router-restart", action="store_true",
                         help="don't restart the router-mode systemd service after updating its preset")
    p_sync.set_defaults(func=cmd_sync)

    p_defaults = sub.add_parser("defaults", help="configure default runtime settings for new profiles")
    p_defaults.add_argument("--show", action="store_true", help="show current defaults and exit")
    p_defaults.set_defaults(func=cmd_defaults)

    p_test = sub.add_parser("test", help="actually launch a profile and verify it generates correctly")
    p_test.add_argument("name")
    p_test.add_argument("--timeout", type=int, default=300, help="seconds to wait for health (default 300)")
    p_test.set_defaults(func=cmd_test)

    p_router = sub.add_parser("router", help="manage the router-mode llama-server (status, load/unload)")
    router_sub = p_router.add_subparsers(dest="router_command", required=True)

    p_router_status = router_sub.add_parser("status", help="show loaded/unloaded models and GPU placement")
    p_router_status.set_defaults(func=cmd_router_status)

    p_router_load = router_sub.add_parser("load", help="load a model now instead of waiting for a request")
    p_router_load.add_argument("name")
    p_router_load.set_defaults(func=cmd_router_load)

    p_router_unload = router_sub.add_parser("unload", help="unload a model now to free its GPU memory")
    p_router_unload.add_argument("name")
    p_router_unload.set_defaults(func=cmd_router_unload)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if getattr(args, "command", None) == "pull" and not args.tui and args.repo_id is None:
        parser.error("repo_id is required unless --tui is passed")

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except EOFError:
        print("\nInput closed unexpectedly. Cancelled.")
        sys.exit(130)
```

Delete the old inline parser-construction block from `main()` (everything between `parser = argparse.ArgumentParser(prog="modelctl")` and `args = parser.parse_args()` that you just moved into `build_arg_parser()`).

Now update `cmd_pull` to dispatch to the wizard. Find its current signature (`def cmd_pull(args):`) and add a branch at the very top:

```python
def cmd_pull(args):
    if getattr(args, "tui", False):
        run_pull_wizard()
        return

    repo_id = args.repo_id
    print(f"Fetching file list for {repo_id} ...")
    # ... rest of the existing function is unchanged
```

Add the lazy-import wrapper near the bottom of `modelctl.py`, above `def main():`:

```python
def run_pull_wizard():
    """Launch the Textual pull wizard (modelctl pull --tui). Imports
    textual lazily so it never becomes a hard dependency for the rest of
    modelctl -- only --tui users need it installed."""
    try:
        from modelctl_tui import PullWizardApp
    except ImportError:
        print(
            "The --tui wizard requires the 'textual' package, which isn't installed.\n"
            "Install it with: pip install textual",
            file=sys.stderr,
        )
        sys.exit(1)
    PullWizardApp().run()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl.TestPullTuiFlag test_modelctl -v 2>&1 | tail -20`
Expected: all tests pass, including the full existing suite (run `python3 -m unittest test_modelctl -v 2>&1 | grep -E "^Ran|^OK|^FAILED"` to confirm nothing else broke from the `main()` refactor)

- [ ] **Step 5: Commit**

```bash
git add modelctl.py test_modelctl.py
git commit -m "Add --tui flag to pull, extract build_arg_parser(), lazy textual import

repo_id becomes optional when --tui is passed (the wizard starts at
search). Parser construction pulled out of main() into build_arg_parser()
so it's testable without invoking sys.exit via argparse's error paths.
run_pull_wizard() imports textual lazily and fails with a clear message
if it's not installed, so textual stays optional for everyone not using
--tui."
```

---

### Task 3: `PullWizardApp` skeleton + `StepIndicator`

Get something that actually boots under Textual's `Pilot` test harness before building real screens on top of it.

**Files:**
- Modify: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
import unittest

from textual.widgets import Static

from modelctl_tui import PullWizardApp, StepIndicator, WizardState, next_screen_after


class TestStepIndicator(unittest.TestCase):
    def test_renders_all_steps_with_current_marked(self):
        indicator = StepIndicator(current="quant")
        rendered = str(indicator.render())
        for step in ["search", "quant", "vision_mtp", "configure", "name", "download", "summary"]:
            self.assertIn(step, rendered)


class TestPullWizardAppBoots(unittest.IsolatedAsyncioTestCase):
    async def test_app_starts_on_search_screen(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            self.assertEqual(app.screen.STEP, "search")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui.TestStepIndicator test_modelctl_tui.TestPullWizardAppBoots -v`
Expected: `ImportError: cannot import name 'PullWizardApp'` (and `StepIndicator`)

- [ ] **Step 3: Write the minimal implementation**

Add to `modelctl_tui.py`:

```python
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Static


class StepIndicator(Static):
    """Breadcrumb shown at the top of every wizard screen, e.g.
    'search > [quant] > vision_mtp > configure > name > download > summary'
    with the current step bracketed. Takes the current step name as a
    constructor arg -- no shared mutable state, just a label."""

    def __init__(self, current: str):
        self.current = current
        super().__init__()

    def render(self) -> str:
        parts = [f"[{s}]" if s == self.current else s for s in STEP_ORDER]
        return " > ".join(parts)


class SearchScreen(Screen):
    """Placeholder for Task 4."""
    STEP = "search"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="search")


class PullWizardApp(App):
    """Entry point for `modelctl pull --tui`. Pushes SearchScreen first;
    each subsequent screen is pushed by the previous one via
    next_screen_after(), carrying a shared WizardState forward."""

    def on_mount(self) -> None:
        self.state = WizardState()
        self.push_screen(SearchScreen())
```

Note: this originally used a `name = "search"` class attribute and `app.screen.STEP` for lookup, but Task 3's code review caught that this shadows Textual's built-in read-only `DOMNode.name` property (backed by `self._name`) rather than setting it properly — harmless until something touches `.name` expecting the real property, but a footgun about to be copied into 6 more screens. Fixed to use a distinct `STEP` class attribute instead (see Task 3's actual commit `b4b3a7f`, "Fix name-property shadowing and weak StepIndicator test"). Every `name = "<step>"` / `app.screen.STEP` reference in the remaining tasks below should read `STEP = "<step>"` / `app.screen.STEP` instead — the plan text wasn't fully swept for this after the fix landed, so treat `STEP` as the correct attribute throughout, not `name`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK` (all tests from Task 1 plus these 2 pass)

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Add PullWizardApp skeleton and StepIndicator widget

App boots to a placeholder SearchScreen under Textual's Pilot test
harness. Real screen implementations follow in subsequent tasks -- this
just proves the app/screen/state wiring works before building on it."
```

---

### Task 4: `SearchScreen`

**Files:**
- Modify: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
from unittest import mock

from textual.widgets import Input, ListView


class TestSearchScreen(unittest.IsolatedAsyncioTestCase):
    async def test_typing_query_and_pressing_enter_shows_results(self):
        fake_results = [
            {"repo_id": "unsloth/Qwen3.5-35B-A3B-GGUF", "downloads": 138881, "likes": 854,
             "is_gguf": True, "has_mtp": False, "contents": {"quant_groups": [], "mmproj_files": [], "mtp_files": []}},
        ]
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.search_models", return_value=fake_results) as mock_search:
            async with app.run_test() as pilot:
                await pilot.click("#search-input")
                await pilot.press(*"qwen3.5", "enter")
                await pilot.pause()
                mock_search.assert_called_once()
                results_view = app.screen.query_one("#search-results", ListView)
                self.assertEqual(len(results_view.children), 1)

    async def test_selecting_a_result_stores_repo_id_and_contents(self):
        fake_results = [
            {"repo_id": "unsloth/Qwen3.5-35B-A3B-GGUF", "downloads": 1, "likes": 1,
             "is_gguf": True, "has_mtp": False, "contents": {"quant_groups": [{"label": "Q4_K_M", "files": ["a.gguf"], "sharded": False, "total_size": 100}], "mmproj_files": [], "mtp_files": []}},
        ]
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.search_models", return_value=fake_results):
            async with app.run_test() as pilot:
                await pilot.click("#search-input")
                await pilot.press(*"qwen3.5", "enter")
                await pilot.pause()
                await pilot.click("ListItem")
                await pilot.pause()
                self.assertEqual(app.state.repo_id, "unsloth/Qwen3.5-35B-A3B-GGUF")
                self.assertEqual(app.state.repo_contents, fake_results[0]["contents"])
                self.assertEqual(app.screen.STEP, "quant")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui.TestSearchScreen -v`
Expected: `AssertionError` / `NoMatches` errors — `SearchScreen` doesn't have an `#search-input` or `#search-results` yet (still the Task 3 placeholder)

- [ ] **Step 3: Write the minimal implementation**

Replace the placeholder `SearchScreen` in `modelctl_tui.py`:

```python
from textual.containers import Vertical
from textual.widgets import Input, ListView, ListItem, Label

import modelctl


class SearchScreen(Screen):
    STEP = "search"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="search")
        yield Input(placeholder="Search Hugging Face...", id="search-input")
        yield ListView(id="search-results")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        results = modelctl.search_models(query, limit=15, enrich=True)
        self._results = results
        results_view = self.query_one("#search-results", ListView)
        results_view.clear()
        for r in results:
            label = f"{r['repo_id']} ({r['downloads']:,} downloads)"
            results_view.append(ListItem(Label(label)))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = self.query_one("#search-results", ListView).index
        chosen = self._results[index]
        self.app.state.repo_id = chosen["repo_id"]
        self.app.state.repo_contents = chosen["contents"]
        next_step = next_screen_after("search", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())
```

Add a registry mapping step names to screen classes near the bottom of the screen definitions (this gets extended in each subsequent task -- for now it only needs "search" filled in, plus a placeholder for "quant" so the reference resolves):

```python
class QuantPickScreen(Screen):
    """Placeholder for Task 5."""
    STEP = "quant"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="quant")


SCREENS_BY_NAME = {
    "search": SearchScreen,
    "quant": QuantPickScreen,
}
```

Remove the old placeholder `SearchScreen` class body (keep only the new one above) and make sure `PullWizardApp.on_mount` still references `SearchScreen()` — no change needed there since the class name is the same.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK`, all tests pass

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Implement SearchScreen: query input, results list, repo selection

Calls modelctl.search_models() directly (already pure/TUI-ready from
earlier work). Selecting a result stores repo_id + repo_contents on
WizardState and advances via next_screen_after() -- QuantPickScreen is
still a placeholder, filled in next."
```

---

### Task 5: `QuantPickScreen`

**Files:**
- Modify: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
class TestQuantPickScreen(unittest.IsolatedAsyncioTestCase):
    def _state_with_contents(self, mmproj=None, mtp=None):
        return WizardState(
            repo_id="unsloth/Qwen3.5-35B-A3B-GGUF",
            repo_contents={
                "quant_groups": [
                    {"label": "Q4_K_M", "files": ["a.gguf"], "sharded": False, "total_size": 100},
                    {"label": "Q5_K_M", "files": ["b.gguf"], "sharded": False, "total_size": 120},
                ],
                "mmproj_files": mmproj or [],
                "mtp_files": mtp or [],
            },
        )

    async def test_lists_quant_groups_from_state(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = self._state_with_contents()
            await app.push_screen(QuantPickScreen())
            await pilot.pause()
            options = app.screen.query_one("#quant-options", ListView)
            self.assertEqual(len(options.children), 2)

    async def test_picking_quant_with_no_extras_skips_to_configure(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = self._state_with_contents()  # no mmproj/mtp
            await app.push_screen(QuantPickScreen())
            await pilot.pause()
            await pilot.click("ListItem")
            await pilot.pause()
            self.assertEqual(app.state.quant_group["label"], "Q4_K_M")
            self.assertEqual(app.screen.STEP, "configure")

    async def test_picking_quant_with_mmproj_goes_to_vision_mtp(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = self._state_with_contents(mmproj=[{"name": "mmproj-F16.gguf", "size": 500}])
            await app.push_screen(QuantPickScreen())
            await pilot.pause()
            await pilot.click("ListItem")
            await pilot.pause()
            self.assertEqual(app.screen.STEP, "vision_mtp")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui.TestQuantPickScreen -v`
Expected: fails — `#quant-options` doesn't exist yet, screen doesn't advance (still the Task 4 placeholder)

- [ ] **Step 3: Write the minimal implementation**

Replace the placeholder `QuantPickScreen` in `modelctl_tui.py`:

```python
class QuantPickScreen(Screen):
    STEP = "quant"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="quant")
        yield ListView(id="quant-options")

    def on_mount(self) -> None:
        groups = (self.app.state.repo_contents or {}).get("quant_groups", [])
        self._groups = groups
        options = self.query_one("#quant-options", ListView)
        for g in groups:
            size = modelctl._format_size(g.get("total_size"))
            options.append(ListItem(Label(f"{g['label']} ({size})")))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = self.query_one("#quant-options", ListView).index
        self.app.state.quant_group = self._groups[index]
        next_step = next_screen_after("quant", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())


class VisionMtpScreen(Screen):
    """Placeholder for Task 6."""
    STEP = "vision_mtp"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="vision_mtp")


class ConfigureScreen(Screen):
    """Placeholder for Task 7."""
    STEP = "configure"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="configure")
```

Update `SCREENS_BY_NAME` to include the new placeholders:

```python
SCREENS_BY_NAME = {
    "search": SearchScreen,
    "quant": QuantPickScreen,
    "vision_mtp": VisionMtpScreen,
    "configure": ConfigureScreen,
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Implement QuantPickScreen: list quant groups, pick one, branch on extras

Reuses _format_size() from modelctl.py. Uses next_screen_after() to
decide whether VisionMtpScreen or ConfigureScreen comes next -- both are
still placeholders, VisionMtpScreen filled in next."
```

---

### Task 6: `VisionMtpScreen`

**Files:**
- Modify: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
class TestVisionMtpScreen(unittest.IsolatedAsyncioTestCase):
    async def test_lists_mmproj_and_mtp_options_with_skip(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = WizardState(repo_contents={
                "mmproj_files": [{"name": "mmproj-F16.gguf", "size": 500}],
                "mtp_files": [{"name": "model-mtp.gguf", "size": 300}],
            })
            await app.push_screen(VisionMtpScreen())
            await pilot.pause()
            mmproj_options = app.screen.query_one("#mmproj-options", ListView)
            mtp_options = app.screen.query_one("#mtp-options", ListView)
            # +1 each for the "skip" entry
            self.assertEqual(len(mmproj_options.children), 2)
            self.assertEqual(len(mtp_options.children), 2)

    async def test_picking_mmproj_and_skipping_mtp_advances_to_configure(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = WizardState(repo_contents={
                "mmproj_files": [{"name": "mmproj-F16.gguf", "size": 500}],
                "mtp_files": [],
            })
            await app.push_screen(VisionMtpScreen())
            await pilot.pause()
            await pilot.click("#mmproj-options ListItem")
            await pilot.click("#continue-button")
            await pilot.pause()
            self.assertEqual(app.state.mmproj_choice["name"], "mmproj-F16.gguf")
            self.assertIsNone(app.state.mtp_choice)
            self.assertEqual(app.screen.STEP, "configure")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui.TestVisionMtpScreen -v`
Expected: fails — `#mmproj-options` etc. don't exist yet (still the Task 5 placeholder)

- [ ] **Step 3: Write the minimal implementation**

Replace the placeholder `VisionMtpScreen` in `modelctl_tui.py`:

```python
from textual.widgets import Button

SKIP_LABEL = "(skip)"


class VisionMtpScreen(Screen):
    STEP = "vision_mtp"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="vision_mtp")
        contents = self.app.state.repo_contents or {}
        self._mmproj_files = contents.get("mmproj_files", [])
        self._mtp_files = contents.get("mtp_files", [])
        yield Label("Vision (mmproj):")
        yield ListView(id="mmproj-options")
        yield Label("MTP draft head:")
        yield ListView(id="mtp-options")
        yield Button("Continue", id="continue-button")

    def on_mount(self) -> None:
        mmproj_view = self.query_one("#mmproj-options", ListView)
        mmproj_view.append(ListItem(Label(SKIP_LABEL)))
        for f in self._mmproj_files:
            mmproj_view.append(ListItem(Label(f"{f['name']} ({modelctl._format_size(f.get('size'))})")))

        mtp_view = self.query_one("#mtp-options", ListView)
        mtp_view.append(ListItem(Label(SKIP_LABEL)))
        for f in self._mtp_files:
            mtp_view.append(ListItem(Label(f"{f['name']} ({modelctl._format_size(f.get('size'))})")))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "continue-button":
            return
        mmproj_index = self.query_one("#mmproj-options", ListView).index
        if mmproj_index and mmproj_index > 0:
            self.app.state.mmproj_choice = self._mmproj_files[mmproj_index - 1]
        mtp_index = self.query_one("#mtp-options", ListView).index
        if mtp_index and mtp_index > 0:
            self.app.state.mtp_choice = self._mtp_files[mtp_index - 1]
        next_step = next_screen_after("vision_mtp", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())
```

Note: `ListView.index` defaults to `0` (not `None`) once a list has items and nothing's been explicitly clicked in some Textual versions — the `and mmproj_index > 0` check treats index `0` (the "(skip)" row) the same as "nothing picked," which is correct here since row 0 IS skip.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Implement VisionMtpScreen: optional mmproj + MTP file pick

Both lists include a leading '(skip)' entry, matching cmd_pull's today's
optional-prompt behavior (blank input = skip). Continue button confirms
both picks at once and advances to ConfigureScreen."
```

---

### Task 7: `ConfigureScreen`

This is the biggest form — mirrors `prompt_config()`'s field set and runs `preflight()` on submit.

**Files:**
- Modify: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
class TestConfigureScreen(unittest.IsolatedAsyncioTestCase):
    async def test_prefills_from_defaults(self):
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto",
            "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults):
            async with app.run_test() as pilot:
                app.state = WizardState()
                await app.push_screen(ConfigureScreen())
                await pilot.pause()
                ctx_input = app.screen.query_one("#config-ctx", Input)
                self.assertEqual(ctx_input.value, "32768")

    async def test_submit_stores_config_and_advances(self):
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto",
            "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults), \
             mock.patch("modelctl_tui.modelctl.preflight", return_value=(True, "llama-server", {}, [])):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x", quant_group={"label": "Q4_K_M", "files": ["a.gguf"]},
                )
                await app.push_screen(ConfigureScreen())
                await pilot.pause()
                await pilot.click("#submit-config")
                await pilot.pause()
                self.assertEqual(app.state.config["ctx"], "32768")
                self.assertEqual(app.screen.STEP, "name")

    async def test_preflight_warning_does_not_block_submit(self):
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto",
            "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults), \
             mock.patch("modelctl_tui.modelctl.preflight",
                         return_value=(False, None, {}, ["ERROR: llama-server not found"])):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x", quant_group={"label": "Q4_K_M", "files": ["a.gguf"]},
                )
                await app.push_screen(ConfigureScreen())
                await pilot.pause()
                await pilot.click("#submit-config")
                await pilot.pause()
                self.assertIn("ERROR: llama-server not found", app.state.warnings)
                self.assertEqual(app.screen.STEP, "name")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui.TestConfigureScreen -v`
Expected: fails — `#config-ctx`, `#submit-config` don't exist yet (still the Task 5 placeholder)

- [ ] **Step 3: Write the minimal implementation**

Replace the placeholder `ConfigureScreen` in `modelctl_tui.py`:

```python
class ConfigureScreen(Screen):
    STEP = "configure"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="configure")
        d = modelctl.load_defaults()
        self._defaults = d
        yield Label("Device (blank = use split strategy):")
        yield Input(value=d.get("device", ""), id="config-device")
        yield Label("Split mode:")
        yield Input(value=d["split_mode"], id="config-split-mode")
        yield Label("Tensor split:")
        yield Input(value=d["tensor_split"], id="config-tensor-split")
        yield Label("Context length:")
        yield Input(value=str(d["ctx"]), id="config-ctx")
        yield Label("KV cache quant:")
        yield Input(value=d["kv_quant"], id="config-kv-quant")
        yield Label("Flash attention:")
        yield Input(value=d["flash_attn"], id="config-flash-attn")
        yield Label("llama-swap idle TTL (seconds):")
        yield Input(value=str(d["ttl"]), id="config-ttl")
        yield Label("Multi-token prediction (on/off):")
        yield Input(value=d.get("mtp", modelctl.DEFAULT_MTP), id="config-mtp")
        yield Label("Extra llama-server flags (optional):")
        yield Input(value="", id="config-extra")
        yield Static("", id="preflight-warning")
        yield Button("Continue", id="submit-config")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "submit-config":
            return
        config = {
            "device": self.query_one("#config-device", Input).value,
            "split_mode": self.query_one("#config-split-mode", Input).value,
            "tensor_split": self.query_one("#config-tensor-split", Input).value,
            "ctx": self.query_one("#config-ctx", Input).value,
            "kv_quant": self.query_one("#config-kv-quant", Input).value,
            "flash_attn": self.query_one("#config-flash-attn", Input).value,
            "ttl": self.query_one("#config-ttl", Input).value,
            "mtp": self.query_one("#config-mtp", Input).value,
            "extra": self.query_one("#config-extra", Input).value,
        }
        self.app.state.config = config

        # Build a throwaway profile dict just for preflight() -- same
        # shape cmd_pull assembles, minus fields preflight doesn't read.
        probe_profile = {
            "name": "preflight-probe",
            "model_path": (self.app.state.quant_group or {}).get("files", [""])[0],
            "mmproj_path": (self.app.state.mmproj_choice or {}).get("name"),
            "config": config,
        }
        ok, _, _, messages = modelctl.preflight(probe_profile, auto_fix=True)
        if not ok or messages:
            self.app.state.warnings.extend(messages)
            self.query_one("#preflight-warning", Static).update("\n".join(messages))

        next_step = next_screen_after("configure", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())


class NameScreen(Screen):
    """Placeholder for Task 8."""
    STEP = "name"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="name")
```

Update `SCREENS_BY_NAME`:

```python
SCREENS_BY_NAME = {
    "search": SearchScreen,
    "quant": QuantPickScreen,
    "vision_mtp": VisionMtpScreen,
    "configure": ConfigureScreen,
    "name": NameScreen,
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Implement ConfigureScreen: full runtime config form + preflight check

Same field set as prompt_config(), pre-filled from load_defaults().
preflight() issues are non-blocking per the spec -- shown in a warning
area and appended to WizardState.warnings so SummaryScreen can surface
them again later, but submission always advances."
```

---

### Task 8: `NameScreen`

**Files:**
- Modify: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
class TestNameScreen(unittest.IsolatedAsyncioTestCase):
    async def test_prefills_name_from_quant_label_and_default_dest_dir(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.next_unique_profile_name", side_effect=lambda s: s):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                )
                await app.push_screen(NameScreen())
                await pilot.pause()
                name_input = app.screen.query_one("#profile-name", Input)
                self.assertEqual(name_input.value, "model")
                dest_input = app.screen.query_one("#dest-dir", Input)
                self.assertEqual(dest_input.value, str(modelctl.DEFAULT_MODELS_DIR))

    async def test_submit_stores_name_and_dest_dir_and_advances(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.next_unique_profile_name", side_effect=lambda s: s):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                )
                await app.push_screen(NameScreen())
                await pilot.pause()
                await pilot.click("#submit-name")
                await pilot.pause()
                self.assertEqual(app.state.profile_name, "model")
                self.assertEqual(app.screen.STEP, "download")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui.TestNameScreen -v`
Expected: fails — `#profile-name`, `#dest-dir` don't exist yet (still the Task 7 placeholder)

- [ ] **Step 3: Write the minimal implementation**

Replace the placeholder `NameScreen` in `modelctl_tui.py`:

```python
class NameScreen(Screen):
    STEP = "name"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="name")
        label = (self.app.state.quant_group or {}).get("label", "")
        clean = modelctl.strip_quant_from_label(label)
        default_name = modelctl.next_unique_profile_name(modelctl.slugify(clean))
        yield Label("Profile name:")
        yield Input(value=default_name, id="profile-name")
        yield Label("Download directory:")
        yield Input(value=self.app.state.dest_dir, id="dest-dir")
        yield Button("Continue", id="submit-name")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "submit-name":
            return
        self.app.state.profile_name = self.query_one("#profile-name", Input).value.strip()
        self.app.state.dest_dir = self.query_one("#dest-dir", Input).value.strip()
        next_step = next_screen_after("name", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())


class DownloadScreen(Screen):
    """Placeholder for Task 9."""
    STEP = "download"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="download")
```

Update `SCREENS_BY_NAME`:

```python
SCREENS_BY_NAME = {
    "search": SearchScreen,
    "quant": QuantPickScreen,
    "vision_mtp": VisionMtpScreen,
    "configure": ConfigureScreen,
    "name": NameScreen,
    "download": DownloadScreen,
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Implement NameScreen: profile name + dest_dir, both editable

Pre-fills name via strip_quant_from_label() + slugify() +
next_unique_profile_name(), same as cmd_pull today. dest_dir is folded
into this screen rather than getting its own, per the spec -- it's
pre-filled from DEFAULT_MODELS_DIR and almost always accepted as-is."
```

---

### Task 9: `DownloadScreen`

Background worker so a multi-GB download doesn't freeze the UI.

**Files:**
- Modify: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
class TestDownloadScreen(unittest.IsolatedAsyncioTestCase):
    async def test_downloads_model_file_and_advances_on_success(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.download_if_needed", return_value="/models/model-Q4_K_M.gguf") as mock_dl:
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"], "sharded": False},
                    dest_dir="/models",
                )
                await app.push_screen(DownloadScreen())
                await pilot.pause()
                await pilot.pause()  # let the worker complete
                mock_dl.assert_called_once_with("repo/x", "model-Q4_K_M.gguf", modelctl.Path("/models"))
                self.assertEqual(app.state.__dict__.get("model_path"), "/models/model-Q4_K_M.gguf")
                self.assertEqual(app.screen.STEP, "summary")

    async def test_download_failure_shows_retry_and_does_not_advance(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.download_if_needed", side_effect=OSError("network drop")):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"], "sharded": False},
                    dest_dir="/models",
                )
                await app.push_screen(DownloadScreen())
                await pilot.pause()
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "download")
                retry_button = app.screen.query_one("#retry-download", Button)
                self.assertFalse(retry_button.disabled)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui.TestDownloadScreen -v`
Expected: fails — `download_if_needed` never called, no `#retry-download` widget (still the Task 8 placeholder)

- [ ] **Step 3: Write the minimal implementation**

Replace the placeholder `DownloadScreen` in `modelctl_tui.py`. Add `from pathlib import Path` and `from textual import work` to the top imports if not already present:

```python
from pathlib import Path

from textual import work


class DownloadScreen(Screen):
    STEP = "download"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="download")
        yield Static("Downloading...", id="download-status")
        yield Button("Retry", id="retry-download", disabled=True)

    def on_mount(self) -> None:
        self.run_download()

    @work(thread=True)
    def run_download(self) -> None:
        state = self.app.state
        dest = Path(state.dest_dir)
        try:
            parts = state.quant_group["files"]
            primary_path = None
            for part in parts:
                path = modelctl.download_if_needed(state.repo_id, part, dest)
                if primary_path is None:
                    primary_path = path
            if state.mmproj_choice:
                state.mmproj_choice["local_path"] = modelctl.download_if_needed(
                    state.repo_id, state.mmproj_choice["name"], dest)
            if state.mtp_choice:
                state.mtp_choice["local_path"] = modelctl.download_if_needed(
                    state.repo_id, state.mtp_choice["name"], dest)
        except Exception as e:
            self.app.call_from_thread(self._on_failure, str(e))
            return
        state.model_path = primary_path
        self.app.call_from_thread(self._on_success)

    def _on_success(self) -> None:
        next_step = next_screen_after("download", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())

    def _on_failure(self, message: str) -> None:
        self.query_one("#download-status", Static).update(f"Download failed: {message}")
        self.query_one("#retry-download", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "retry-download":
            return
        self.query_one("#retry-download", Button).disabled = True
        self.query_one("#download-status", Static).update("Downloading...")
        self.run_download()


class SummaryScreen(Screen):
    """Placeholder for Task 10."""
    STEP = "summary"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="summary")
```

Update `SCREENS_BY_NAME`:

```python
SCREENS_BY_NAME = {
    "search": SearchScreen,
    "quant": QuantPickScreen,
    "vision_mtp": VisionMtpScreen,
    "configure": ConfigureScreen,
    "name": NameScreen,
    "download": DownloadScreen,
    "summary": SummaryScreen,
}
```

Add `model_path: str | None = None` to `WizardState` in the Task 1 dataclass (edit it directly — it doesn't need its own TDD cycle, it's a one-line dataclass field addition covered by this task's own tests):

```python
@dataclass
class WizardState:
    repo_id: str | None = None
    repo_contents: dict | None = None
    quant_group: dict | None = None
    mmproj_choice: dict | None = None
    mtp_choice: dict | None = None
    dest_dir: str = str(modelctl.DEFAULT_MODELS_DIR)
    config: dict | None = None
    profile_name: str = ""
    model_path: str | None = None
    warnings: list = field(default_factory=list)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Implement DownloadScreen: background worker, spinner, retry on failure

Runs download_if_needed() for the model file (all shards if split) plus
mmproj/MTP if chosen, in a @work(thread=True) worker so the UI stays
responsive. Failure stops here with a retry button rather than advancing
to a summary for a profile whose files aren't actually on disk."
```

---

### Task 10: `SummaryScreen`

**Files:**
- Modify: `modelctl_tui.py`
- Test: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
class TestSummaryScreen(unittest.IsolatedAsyncioTestCase):
    async def test_saves_generates_and_syncs_on_mount(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.save_profile") as mock_save, \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True) as mock_gen, \
             mock.patch("modelctl_tui.modelctl.sync_all_backends") as mock_sync, \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers") as mock_hermes, \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                mock_save.assert_called_once()
                mock_gen.assert_called_once()
                mock_sync.assert_called_once()
                mock_hermes.assert_called_once()
                status = app.screen.query_one("#summary-status", Static)
                self.assertIn("model", str(status.render()))

    async def test_shows_warnings_if_any_were_collected(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.save_profile"), \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends"), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                    warnings=["ERROR: llama-server not found"],
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                status = app.screen.query_one("#summary-status", Static)
                self.assertIn("ERROR: llama-server not found", str(status.render()))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m unittest test_modelctl_tui.TestSummaryScreen -v`
Expected: fails — none of `save_profile`/`generate_artifacts`/`sync_all_backends`/`sync_hermes_custom_providers` get called, `#summary-status` doesn't exist (still the Task 9 placeholder)

- [ ] **Step 3: Write the minimal implementation**

Replace the placeholder `SummaryScreen` in `modelctl_tui.py`:

```python
class SummaryScreen(Screen):
    STEP = "summary"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="summary")
        yield Static("Saving...", id="summary-status")
        yield Button("Done", id="done-button")

    def on_mount(self) -> None:
        state = self.app.state
        profile = {
            "name": state.profile_name,
            "repo_id": state.repo_id,
            "file": state.quant_group["label"],
            "model_path": state.model_path,
            "mmproj_path": (state.mmproj_choice or {}).get("local_path"),
            "mtp_path": (state.mtp_choice or {}).get("local_path"),
            "config": state.config,
            "env": modelctl.capture_env_passthrough(),
        }
        modelctl.save_profile(profile)
        modelctl.generate_artifacts(profile)
        modelctl.sync_all_backends()
        modelctl.sync_hermes_custom_providers()

        lines = [f"Saved profile '{state.profile_name}'."]
        if state.warnings:
            lines.append("Warnings:")
            lines.extend(f"  {w}" for w in state.warnings)
        self.query_one("#summary-status", Static).update("\n".join(lines))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "done-button":
            self.app.exit()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m unittest test_modelctl_tui -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Implement SummaryScreen: save, generate, sync, display result

Same save_profile/generate_artifacts/sync_all_backends/
sync_hermes_custom_providers sequence cmd_pull runs today, auto-applied
on mount per the spec (no separate confirm gate). Collected preflight
warnings are shown again here since ConfigureScreen's banner is easy to
miss/dismiss. Done button exits the app -- per the spec, there's no
in-app 'pull another' loop."
```

---

### Task 11: End-to-end wiring test

All seven screens exist; prove the full chain works together in one Pilot-driven test, not just screen-by-screen.

**Files:**
- Modify: `test_modelctl_tui.py`

- [ ] **Step 1: Write the failing test**

Add to `test_modelctl_tui.py`:

```python
class TestFullWizardFlow(unittest.IsolatedAsyncioTestCase):
    async def test_search_to_summary_no_extras(self):
        fake_results = [{
            "repo_id": "repo/x", "downloads": 1, "likes": 1, "is_gguf": True, "has_mtp": False,
            "contents": {
                "quant_groups": [{"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"], "sharded": False, "total_size": 100}],
                "mmproj_files": [], "mtp_files": [],
            },
        }]
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto", "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.search_models", return_value=fake_results), \
             mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults), \
             mock.patch("modelctl_tui.modelctl.preflight", return_value=(True, "llama-server", {}, [])), \
             mock.patch("modelctl_tui.modelctl.next_unique_profile_name", side_effect=lambda s: s), \
             mock.patch("modelctl_tui.modelctl.download_if_needed", return_value="/models/model-Q4_K_M.gguf"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]), \
             mock.patch("modelctl_tui.modelctl.save_profile") as mock_save, \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends"), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"):
            async with app.run_test() as pilot:
                # search -> quant (no vision_mtp since repo has neither)
                await pilot.click("#search-input")
                await pilot.press(*"x", "enter")
                await pilot.pause()
                await pilot.click("ListItem")
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "quant")

                # quant -> configure (skips vision_mtp)
                await pilot.click("ListItem")
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "configure")

                # configure -> name
                await pilot.click("#submit-config")
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "name")

                # name -> download
                await pilot.click("#submit-name")
                await pilot.pause()

                # download -> summary (worker runs in background)
                await pilot.pause()
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "summary")

                mock_save.assert_called_once()
                saved_profile = mock_save.call_args[0][0]
                self.assertEqual(saved_profile["repo_id"], "repo/x")
                self.assertEqual(saved_profile["model_path"], "/models/model-Q4_K_M.gguf")
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `python3 -m unittest test_modelctl_tui.TestFullWizardFlow -v`

If every prior task's screens are implemented correctly, this should already pass — it exercises only code that exists by now, no new implementation. If it fails, the failure points at exactly which screen-to-screen handoff has a bug (wrong widget ID, wrong `next_screen_after` call, etc.); fix that screen's code, not this test.

- [ ] **Step 3: Fix any integration bugs found**

There's no new production code to write for this task if Tasks 4-10 were done correctly — this step exists to catch integration issues that per-screen tests miss (e.g., a widget ID typo'd differently between the screen that sets it and the screen that reads it). Fix in place if anything surfaces.

- [ ] **Step 4: Run the full test suite to verify everything passes together**

Run: `python3 -m unittest test_modelctl_tui test_modelctl -v 2>&1 | tail -10`
Expected: `OK`, both test modules pass

- [ ] **Step 5: Commit**

```bash
git add test_modelctl_tui.py
git commit -m "Add end-to-end Pilot test for the full pull wizard flow

Search -> quant -> configure (vision_mtp skipped, repo has neither) ->
name -> download -> summary, asserting the final saved profile has the
right repo_id and model_path. Exercises only existing screen code --
this is an integration check, not new implementation."
```

---

### Task 12: Manual smoke test against the real Hugging Face API and live backends

Everything above is mocked. Confirm the wizard actually works end to end against real data before calling this done, following the same pattern used for every other feature this session (real HF API calls, real llama-swap/router sync, real preflight against the actual llama-server binary).

**Files:** none (verification only, no code changes expected unless this surfaces a real bug)

- [ ] **Step 1: Confirm textual is installed on the host** (not just the dev sandbox)

```bash
/home/aaron/.hf-cli/venv/bin/python3 -c "import textual; print(textual.__version__)"
```

If missing: `/home/aaron/.hf-cli/venv/bin/pip install textual`

- [ ] **Step 2: Run the wizard against a real, known-good repo**

```bash
cd /home/aaron/workspace
/home/aaron/.hf-cli/venv/bin/python3 modelctl.py pull --tui
```

Search for `qwen3.5-35b-a3b-ud` (or another repo already known to work from this session's earlier `search`/`pull` testing), walk through the full flow, and let it actually download and save a profile — or `Ctrl-C` out before the download step if you don't want to pull a real multi-GB file for this smoke test, and instead verify Steps 3-4 below with a small/already-cached model.

- [ ] **Step 3: Verify the resulting profile matches what CLI `pull` would have produced**

```bash
cat ~/.local/share/modelctl/profiles/<name-you-chose>.json
```

Confirm it has the same shape as any existing profile (`name`, `repo_id`, `file`, `model_path`, `mmproj_path`, `mtp_path`, `config`, `env`) — compare field-by-field against `cat ~/.local/share/modelctl/profiles/Qwythos-9B-Q4.json` as a known-good reference.

- [ ] **Step 4: Verify it actually synced**

```bash
grep -A3 "^  <name-you-chose>:" ~/llama-swap/config.yaml
grep "\[<name-you-chose>\]" ~/llama-router/router.preset.ini
```

Both should show the new profile.

- [ ] **Step 5: If anything's wrong, fix it and re-run the affected task's tests before re-testing manually**

Do not patch around a real bug found here without adding or updating a test that would have caught it — same standard as every other fix this session.

- [ ] **Step 6: Commit if any fixes were needed**

```bash
git add modelctl_tui.py test_modelctl_tui.py
git commit -m "Fix <specific bug found during live smoke test>"
```

(Skip this commit if Step 5 found nothing to fix.)

---

## Plan self-review notes

- **Spec coverage:** Standalone app (Task 3's `PullWizardApp`), search as step one (Task 4), one profile per run (implicit throughout — no multi-select anywhere), spinner not progress bar (Task 9's `Static` status text, no byte counter), auto-apply (Task 10's `on_mount`), non-blocking preflight (Task 7), one-Screen-per-step with `WizardState` + `next_screen_after` (Tasks 1, 3-10), `dest_dir` folded into `NameScreen` (Task 8), exits after Done with no loop (Task 10's `done-button` handler calls `self.app.exit()`) — every decision in the spec has a task.
- **Out-of-scope items confirmed absent:** no multi-quant selection, no progress bar, no app shell/dashboard, `cmd_pull`'s non-`--tui` path untouched except for the dispatch branch added in Task 2.
- **Type/naming consistency checked:** `WizardState` field names (`repo_id`, `repo_contents`, `quant_group`, `mmproj_choice`, `mtp_choice`, `dest_dir`, `config`, `profile_name`, `model_path`, `warnings`) are used identically across Tasks 1 and 4-10. `SCREENS_BY_NAME` keys match `STEP_ORDER` entries and each `Screen.name` class attribute throughout.
