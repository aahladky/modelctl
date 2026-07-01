"""
modelctl_tui - Textual-based interactive wizard for `modelctl pull --tui`.

This file adds interaction only. Every screen calls an existing pure
function from modelctl.py (search_models, get_repo_contents, preflight,
download_if_needed, save_profile, generate_artifacts, sync_all_backends,
sync_hermes_custom_providers) -- no business logic is duplicated here.
See docs/superpowers/specs/2026-07-01-modelctl-tui-pull-wizard-design.md.
"""
from dataclasses import dataclass, field

from textual import work
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Input, Label, ListItem, ListView, Static

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
    """First wizard screen: search Hugging Face and pick a repo."""
    STEP = "search"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="search")
        yield Input(placeholder="Search Hugging Face...", id="search-input")
        yield Static("", id="search-status")
        yield ListView(id="search-results")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        self.query_one("#search-status", Static).update("Searching...")
        self.run_search(query)

    @work(thread=True)
    def run_search(self, query: str) -> None:
        """Runs search_models() on a worker thread -- it can make up to
        15+ sequential HTTP round-trips against the real Hugging Face API
        (list_models + get_repo_contents per hit when enrich=True), which
        would otherwise freeze the whole Textual event loop. Any exception
        is caught here so a network hiccup can't take down the app."""
        try:
            results = modelctl.search_models(query, limit=15, enrich=True)
        except Exception as e:
            self.app.call_from_thread(self._on_search_failure, str(e))
            return
        self.app.call_from_thread(self._on_search_success, results)

    def _on_search_success(self, results: list) -> None:
        self._results = results
        self.query_one("#search-status", Static).update("" if results else "No results.")
        results_view = self.query_one("#search-results", ListView)
        results_view.clear()
        for r in results:
            label = f"{r['repo_id']} ({r['downloads']:,} downloads)"
            results_view.append(ListItem(Label(label)))

    def _on_search_failure(self, message: str) -> None:
        self.query_one("#search-status", Static).update(f"Search failed: {message}")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = self.query_one("#search-results", ListView).index
        chosen = self._results[index]
        self.app.state.repo_id = chosen["repo_id"]
        self.app.state.repo_contents = chosen["contents"]
        next_step = next_screen_after("search", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())


class QuantPickScreen(Screen):
    """Second wizard screen: pick a quant group from the repo contents
    already fetched by SearchScreen. No network I/O here -- just reads
    self.app.state.repo_contents -- so no background worker is needed."""
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


SCREENS_BY_NAME = {
    "search": SearchScreen,
    "quant": QuantPickScreen,
    "vision_mtp": VisionMtpScreen,
    "configure": ConfigureScreen,
}


class PullWizardApp(App):
    """Entry point for `modelctl pull --tui`. Pushes SearchScreen first;
    each subsequent screen is pushed by the previous one via
    next_screen_after(), carrying a shared WizardState forward."""

    def on_mount(self) -> None:
        self.state = WizardState()
        self.push_screen(SearchScreen())
