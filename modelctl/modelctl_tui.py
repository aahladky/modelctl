"""
modelctl_tui - Textual-based interactive wizard for `modelctl pull --tui`.

This file adds interaction only. Every screen calls an existing pure
function from modelctl.py (search_models, get_repo_contents, preflight,
download_if_needed, save_profile, generate_artifacts, sync_all_backends,
sync_hermes_custom_providers) -- no business logic is duplicated here.
"""
import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

import modelctl

STEP_ORDER = ["search", "quant", "vision_mtp", "configure", "name", "download", "summary"]

SKIP_LABEL = "(skip)"


def _int_or(value, default):
    """Coerce a form field to int, falling back to `default`.

    The wizard's ctx/ttl inputs are text; storing them as strings put
    `-c ''` into the generated run.sh when a field was cleared, and left
    every consumer doing int() on a value that might be "".
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


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
    model_path: str | None = None      # local path of the primary model file, set by DownloadScreen
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

    @work(thread=True, exclusive=True)
    def run_search(self, query: str) -> None:
        """Runs search_models() on a worker thread -- it can make up to
        15+ sequential HTTP round-trips against the real Hugging Face API
        (list_models + get_repo_contents per hit when enrich=True), which
        would otherwise freeze the whole Textual event loop. Any exception
        is caught here so a network hiccup can't take down the app.

        exclusive=True (matching DownloadScreen/SummaryScreen, both fixed
        for this same bug class already): a user retyping and resubmitting
        a search before the first one resolves could otherwise interleave
        two searches, with on_list_view_selected indexing into a
        self._results array that might not match what's visually
        displayed. The stale search's worker is cancelled instead."""
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
    self.app.state.repo_contents -- so no background worker is needed.

    Defensive case (design spec): a repo with zero real quant groups
    (get_repo_contents() raised upstream, or the repo genuinely has no
    selectable model files -- only mmproj/MTP files, say) must not render
    a silent empty ListView with no way out. When that happens, this
    screen shows an error message and lets Escape pop back to a fresh
    SearchScreen instead of leaving the user stuck (Ctrl+C only)."""
    STEP = "quant"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="quant")
        yield Static("", id="quant-error")
        yield ListView(id="quant-options")

    def on_mount(self) -> None:
        groups = (self.app.state.repo_contents or {}).get("quant_groups", [])
        self._groups = groups
        if not groups:
            self.query_one("#quant-error", Static).update(
                "No selectable model files found in this repo. Press Escape to search again."
            )
            return
        options = self.query_one("#quant-options", ListView)
        for g in groups:
            size = modelctl._format_size(g.get("total_size"))
            options.append(ListItem(Label(f"{g['label']} ({size})")))

    def on_key(self, event) -> None:
        if event.key == "escape" and not self._groups:
            self.app.pop_screen()
            self.app.push_screen(SCREENS_BY_NAME["search"]())

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
# probe_profile's paths are bare repo filenames (download happens later,
# in DownloadScreen), so these are guaranteed false positives here --
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

    This screen has ten label/input pairs plus the step indicator,
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
        yield Label("K cache quant:")
        yield Input(value=d["cache_type_k"], id="config-cache-type-k")
        yield Label("V cache quant:")
        yield Input(value=d["cache_type_v"], id="config-cache-type-v")
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
            "ctx": _int_or(self.query_one("#config-ctx", Input).value, 8192),
            "cache_type_k": self.query_one("#config-cache-type-k", Input).value,
            "cache_type_v": self.query_one("#config-cache-type-v", Input).value,
            "flash_attn": self.query_one("#config-flash-attn", Input).value,
            "ttl": _int_or(self.query_one("#config-ttl", Input).value, 3600),
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
        # happened yet, that's DownloadScreen's job), so these two
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
    pre-filled from DEFAULT_MODELS_DIR (via WizardState.dest_dir) and
    almost always accepted as-is, so it doesn't warrant a
    dedicated step. No network/disk I/O here, just reads self.app.state
    and generates a default string, so no background worker is needed."""
    STEP = "name"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="name")
        label = (self.app.state.quant_group or {}).get("label", "")
        clean = modelctl.strip_quant_from_label(label)
        self._default_name = modelctl.next_unique_profile_name(modelctl.slugify(clean))
        self._default_dest_dir = self.app.state.dest_dir
        yield Label("Profile name:")
        yield Input(value=self._default_name, id="profile-name")
        yield Label("Download directory:")
        yield Input(value=self._default_dest_dir, id="dest-dir")
        yield Button("Continue", id="submit-name")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "submit-name":
            return
        # Mirrors cmd_pull's `input(...).strip() or name_default` fallback:
        # save_profile() does zero validation, so a blank name would write
        # to PROFILES_DIR / ".json" and a blank dest_dir would silently
        # resolve to the current working directory via Path("").
        self.app.state.profile_name = (
            self.query_one("#profile-name", Input).value.strip() or self._default_name
        )
        self.app.state.dest_dir = (
            self.query_one("#dest-dir", Input).value.strip() or self._default_dest_dir
        )
        next_step = next_screen_after("name", self.app.state)
        self.app.push_screen(SCREENS_BY_NAME[next_step]())


class DownloadScreen(Screen):
    """Sixth wizard screen: actually pulls the chosen quant group's file(s)
    (all shards if the group is split) plus mmproj/MTP extras if chosen, via
    modelctl.download_if_needed() -- which is itself the only place skip-if-
    already-present logic lives. Runs on a worker thread (SearchScreen's
    established pattern) since a multi-GB model download would otherwise
    freeze the whole Textual event loop. Any exception stops progress here
    with a retry button rather than silently advancing to a summary for a
    profile whose files aren't actually on disk."""
    STEP = "download"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="download")
        yield Static("Downloading...", id="download-status")
        yield Button("Retry", id="retry-download", disabled=True)

    def on_mount(self) -> None:
        self.run_download()

    @work(thread=True, exclusive=True)
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
    """Seventh and final wizard screen: assembles the profile dict (same
    shape cmd_pull builds today), then runs the exact save/generate/sync
    sequence cmd_pull runs after a pull -- save_profile, generate_artifacts,
    sync_all_backends, sync_hermes_custom_providers -- auto-applied with no
    separate confirm gate, per the wizard spec.

    Runs on a worker thread (SearchScreen/DownloadScreen's established
    pattern), NOT synchronously in on_mount like ConfigureScreen's
    preflight() call. The two look superficially similar (both call into
    modelctl.py from a screen's on_mount/submit handler) but they differ in
    a way that matters: preflight() only does local filesystem
    existence checks and, at most, a fast `--list-devices` probe. This
    screen's sync_all_backends() -> sync_router_preset() calls
    restart_router_service() -- a real `subprocess.run(["systemctl",
    "--user", "restart", ...], timeout=30)` -- whenever the router preset
    content actually changed. Reaching this screen at all means a brand-new
    profile was just saved, so the preset content changing (and the restart
    firing) is the common case here, not a rare edge. systemd unit restarts
    have no guaranteed-fast bound the way a llama-server --list-devices
    probe does; the code itself budgets up to 30 seconds before giving up.
    Running that synchronously on Textual's event-loop thread would freeze
    the whole UI for however long systemctl takes, which is exactly the
    class of latency that got SearchScreen's HTTP calls and
    DownloadScreen's multi-GB transfers worker treatment. Using
    exclusive=True from the start here (rather than adding it in a
    follow-up review, as happened for DownloadScreen) since
    there's no legitimate reason for two sync sequences to ever race on
    this screen."""
    STEP = "summary"

    def compose(self) -> ComposeResult:
        yield StepIndicator(current="summary")
        yield Static("Saving...", id="summary-status")
        yield Button("Done", id="done-button")

    def on_mount(self) -> None:
        self.run_sync()

    @work(thread=True, exclusive=True)
    def run_sync(self) -> None:
        """Runs the save/generate/sync sequence on a worker thread (see
        class docstring for why). Every step is wrapped in try/except so a
        real failure here -- disk full, permissions, a network hiccup during
        Hermes sync -- resolves to a status update via call_from_thread
        instead of propagating out of the worker thread, which Textual's
        Worker (default exit_on_error=True) would otherwise turn into a full
        app crash (app._handle_exception() exits the app, tears down the
        alt-screen, dumps a raw traceback -- see code review on 0d6ab1b).

        sync_all_backends()/sync_hermes_custom_providers() report some
        failures (e.g. restart_router_service()'s systemctl failures) by
        printing to stdout/stderr rather than raising or returning a value --
        those prints vanish silently while Textual has stdout/stderr
        redirected, unless we capture them ourselves. redirect_stdout/
        redirect_stderr around just those two calls (not save_profile/
        generate_artifacts, which communicate failure via exceptions) catches
        that output so it can be shown to the user instead of lost."""
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
        captured = io.StringIO()
        saved = False
        try:
            modelctl.save_profile(profile)
            saved = True
            modelctl.generate_artifacts(profile)
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                modelctl.sync_all_backends(restart_router=not self.app.no_router_restart)
                if not self.app.no_hermes:
                    modelctl.sync_hermes_custom_providers()
        except Exception as e:
            self.app.call_from_thread(self._on_failure, str(e), captured.getvalue(), saved)
            return
        self.app.call_from_thread(self._on_success, captured.getvalue())

    def _on_success(self, sync_output: str) -> None:
        state = self.app.state
        lines = [f"Saved profile '{state.profile_name}'."]
        if state.warnings:
            lines.append("Warnings:")
            lines.extend(f"  {w}" for w in state.warnings)
        if sync_output.strip():
            lines.append("Sync details:")
            lines.append(sync_output.strip())
        self.query_one("#summary-status", Static).update("\n".join(lines))

    def _on_failure(self, message: str, sync_output: str, saved: bool) -> None:
        state = self.app.state
        if saved:
            lines = [
                f"Profile '{state.profile_name}' was saved, but a later step failed: {message}",
                "The saved profile may not be fully synced to llama-swap/router/Hermes yet.",
            ]
        else:
            lines = [f"Nothing was saved -- save/sync failed: {message}"]
        if sync_output.strip():
            lines.append("Partial sync output:")
            lines.append(sync_output.strip())
        self.query_one("#summary-status", Static).update("\n".join(lines))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "done-button":
            self.app.exit()


SCREENS_BY_NAME = {
    "search": SearchScreen,
    "quant": QuantPickScreen,
    "vision_mtp": VisionMtpScreen,
    "configure": ConfigureScreen,
    "name": NameScreen,
    "download": DownloadScreen,
    "summary": SummaryScreen,
}


class PullWizardApp(App):
    """Entry point for `modelctl pull --tui`. Pushes SearchScreen first;
    each subsequent screen is pushed by the previous one via
    next_screen_after(), carrying a shared WizardState forward.

    no_hermes/no_router_restart mirror the CLI's --no-hermes/
    --no-router-restart flags (see cmd_pull), threaded through from
    run_pull_wizard() so SummaryScreen can honor them when it calls
    sync_all_backends()/sync_hermes_custom_providers() -- previously these
    flags were accepted by argparse but silently ignored under --tui."""

    def __init__(self, no_hermes: bool = False, no_router_restart: bool = False):
        super().__init__()
        self.no_hermes = no_hermes
        self.no_router_restart = no_router_restart

    def on_mount(self) -> None:
        self.state = WizardState()
        self.push_screen(SearchScreen())
