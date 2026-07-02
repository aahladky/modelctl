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
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

import modelctl

STEP_ORDER = ["search", "quant", "vision_mtp", "configure", "name", "download", "summary"]

SKIP_LABEL = "(skip)"


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
    """Third wizard screen: optionally pick a vision projector (mmproj) file
    and/or an MTP draft-head file from the repo contents already fetched by
    SearchScreen. No network I/O here, so no background worker is needed.

    Unlike QuantPickScreen, this screen has two independent optional choices
    confirmed together by a single Continue button rather than one selection
    advancing immediately -- selecting either list just highlights a row.
    Both lists list the real files first followed by a trailing "(skip)"
    entry; a selected index that falls within the real-files range is a
    real choice, while the trailing "(skip)" row or no selection at all
    (ListView.index is None until something is clicked) both mean "no
    choice", matching cmd_pull's existing blank-input-to-skip behavior for
    these optional prompts."""
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
        # "(skip)" is listed LAST, not first -- pilot.click()'s default
        # ListItem selector resolves to the first DOM match, so a leading
        # skip row would make "click a real result" tests always hit skip
        # instead. This ordering is load-bearing for click-based testing,
        # not just a UI preference -- see commit 2e578e6 for the empirical
        # verification (ListView.index defaults to None, not 0, on this
        # Textual version).
        mmproj_view = self.query_one("#mmproj-options", ListView)
        for f in self._mmproj_files:
            mmproj_view.append(ListItem(Label(f"{f['name']} ({modelctl._format_size(f.get('size'))})")))
        mmproj_view.append(ListItem(Label(SKIP_LABEL)))

        mtp_view = self.query_one("#mtp-options", ListView)
        for f in self._mtp_files:
            mtp_view.append(ListItem(Label(f"{f['name']} ({modelctl._format_size(f.get('size'))})")))
        mtp_view.append(ListItem(Label(SKIP_LABEL)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "continue-button":
            return
        # ListView.index is None until a row has been explicitly clicked
        # (verified empirically against the installed Textual version,
        # 8.2.8 -- it does NOT default to 0). The trailing "(skip)" row
        # sits at index == len(files), one past the real choices, so any
        # index within range(len(files)) is a real pick and everything
        # else (None, or the skip row itself) means "no choice".
        mmproj_index = self.query_one("#mmproj-options", ListView).index
        if mmproj_index is not None and mmproj_index < len(self._mmproj_files):
            self.app.state.mmproj_choice = self._mmproj_files[mmproj_index]
        mtp_index = self.query_one("#mtp-options", ListView).index
        if mtp_index is not None and mtp_index < len(self._mtp_files):
            self.app.state.mtp_choice = self._mtp_files[mtp_index]
        next_step = next_screen_after("vision_mtp", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())


# preflight() reports these two exact prefixes when a model/mmproj file
# doesn't exist on disk yet. At the ConfigureScreen stage of the wizard,
# probe_profile's paths are bare repo filenames (download happens later, in
# Task 9's DownloadScreen), so these are guaranteed false positives here --
# filtered out at the ConfigureScreen call site only, not inside preflight()
# itself, since preflight() is also used post-download where these checks
# are meaningful.
FILE_NOT_FOUND_PREFIXES = (
    "ERROR: model file not found on disk:",
    "ERROR: mmproj file not found on disk:",
)


class ConfigureScreen(Screen):
    """Fourth wizard screen: runtime config for the profile, mirroring
    prompt_config()'s field set exactly, pre-filled from load_defaults().
    On submit, runs preflight() synchronously -- like cmd_test,
    render_llama_swap_entry, and render_router_preset already do elsewhere
    in modelctl.py, none of which use a worker for it. preflight() only
    does local filesystem existence checks and (at most) a fast
    `--list-devices` subprocess call to resolve the llama-server binary;
    it does no network I/O, so it doesn't need the SearchScreen-style
    background-worker treatment reserved for slow/blocking calls.

    Any preflight issues (errors OR warnings) are surfaced in a warning
    area and appended to WizardState.warnings for SummaryScreen to show
    again later, but per the wizard design they are non-blocking --
    submitting always advances to the next step.

    This screen has nine label/input pairs plus the step indicator,
    warning area, and submit button -- comfortably taller than a typical
    80x24 terminal at Textual's default bordered Input height (3 rows
    each). CSS below collapses each Input to a single borderless row so
    the whole form (and, critically, the Continue button) fits within a
    standard-size viewport without requiring the user -- or a test
    Pilot, which can't click widgets scrolled out of view -- to scroll."""
    STEP = "configure"

    DEFAULT_CSS = """
    ConfigureScreen Input {
        height: 1;
        border: none;
    }
    """

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
        yield Input(value=str(d["ctx"]), id="config-ctx", type="integer")
        yield Label("KV cache quant:")
        yield Input(value=d["kv_quant"], id="config-kv-quant")
        yield Label("Flash attention:")
        yield Input(value=d["flash_attn"], id="config-flash-attn")
        yield Label("llama-swap idle TTL (seconds):")
        yield Input(value=str(d["ttl"]), id="config-ttl", type="integer")
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
        # Filter out file-existence errors -- probe_profile's model_path/mmproj_path
        # are bare repo filenames at this point in the wizard (download hasn't
        # happened yet, that's DownloadScreen's job in Task 9), so these two
        # specific messages are guaranteed false positives here, not real signal.
        # Everything else preflight() reports (binary resolution, SYCL env
        # warnings) IS meaningful at this stage and should still surface.
        messages = [m for m in messages if not m.startswith(FILE_NOT_FOUND_PREFIXES)]
        if messages:
            self.app.state.warnings.extend(messages)
            self.query_one("#preflight-warning", Static).update("\n".join(messages))

        next_step = next_screen_after("configure", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())


class NameScreen(Screen):
    """Fifth wizard screen: profile name and download directory. dest_dir
    is folded into this screen rather than getting its own -- it's
    pre-filled from DEFAULT_MODELS_DIR (via WizardState.dest_dir, set in
    Task 1) and almost always accepted as-is, so it doesn't warrant a
    dedicated step. No network/disk I/O here, just reads self.app.state
    and generates a default string, so no background worker is needed."""
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


SCREENS_BY_NAME = {
    "search": SearchScreen,
    "quant": QuantPickScreen,
    "vision_mtp": VisionMtpScreen,
    "configure": ConfigureScreen,
    "name": NameScreen,
    "download": DownloadScreen,
}


class PullWizardApp(App):
    """Entry point for `modelctl pull --tui`. Pushes SearchScreen first;
    each subsequent screen is pushed by the previous one via
    next_screen_after(), carrying a shared WizardState forward."""

    def on_mount(self) -> None:
        self.state = WizardState()
        self.push_screen(SearchScreen())
