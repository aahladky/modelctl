"""
modelctl_tui - Textual-based interactive wizard for `modelctl pull --tui`.

This file adds interaction only. Every screen calls an existing pure
function from modelctl.py (search_models, get_repo_contents, preflight,
download_if_needed, save_profile, generate_artifacts, sync_all_backends,
sync_hermes_custom_providers) -- no business logic is duplicated here.
See docs/superpowers/specs/2026-07-01-modelctl-tui-pull-wizard-design.md.
"""
from dataclasses import dataclass, field

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Static

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
