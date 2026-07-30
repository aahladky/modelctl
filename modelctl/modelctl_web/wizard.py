"""Persistent wizard state for the add-model workflow.

Survives browser reloads and service restarts. Each wizard has a
unique ID, tracks its current step, and stores enough state to
resume without repeating downloads.

Public API:
    WizardState class
    WizardStore (persistence)
"""
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

STATE_DIR = Path(os.environ.get(
    "MODELCTL_HOME", Path.home() / ".local" / "share" / "modelctl"))
WIZARD_DIR = STATE_DIR / "wizards"

# Wizard steps in order.
STEPS = [
    "source",       # Choose HF repo or local file
    "inspect",      # Review files, quants, companions
    "download",     # Download or reference files
    "analyze",      # GGUF analysis
    "plans",        # Plan comparison
    "test",         # Benchmark selected plans
    "register",     # Register with llama-swap
    "done",         # Complete
]


@dataclass
class WizardState:
    """Persistent state for one add-model wizard instance."""
    wizard_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    step: str = "source"

    # Source selection
    source_type: str = ""  # "hf_repo" or "local_file"
    repo_id: str = ""
    local_path: str = ""

    # File selection
    selected_files: list = field(default_factory=list)
    selected_quant: str = ""
    mmproj_file: str = ""
    mtp_file: str = ""

    # Download state
    download_job_id: str = ""
    download_complete: bool = False
    downloaded_paths: list = field(default_factory=list)

    # Analysis
    analysis: dict = field(default_factory=dict)

    # Plans
    candidate_plan_ids: list = field(default_factory=list)
    selected_plan_id: str = ""

    # Test results
    test_job_id: str = ""
    test_observations: dict = field(default_factory=dict)

    # Registration
    profile_name: str = ""
    registration_complete: bool = False
    endpoint: str = ""

    # Errors
    errors: list = field(default_factory=list)

    def advance(self, to_step: str):
        """Move to a later step."""
        if to_step in STEPS:
            self.step = to_step
            self.updated_at = time.time()

    def set_error(self, message: str):
        """Record an error."""
        self.errors.append({"time": time.time(), "message": message})
        self.updated_at = time.time()

    def clear_error(self):
        self.errors = []
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WizardState":
        # Filter to known fields.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class WizardStore:
    """Persistent storage for wizard state."""

    def __init__(self, base_dir: Path = WIZARD_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, wizard_id: str) -> Path:
        return self.base_dir / f"{wizard_id}.json"

    def save(self, state: WizardState):
        """Persist wizard state to disk."""
        state.updated_at = time.time()
        self._path(state.wizard_id).write_text(
            json.dumps(state.to_dict(), indent=2))

    def load(self, wizard_id: str) -> WizardState | None:
        """Load wizard state from disk."""
        path = self._path(wizard_id)
        if not path.exists():
            return None
        try:
            return WizardState.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def delete(self, wizard_id: str):
        """Remove wizard state."""
        path = self._path(wizard_id)
        if path.exists():
            path.unlink()

    def list_active(self, max_age: float = 86400) -> list[WizardState]:
        """List recently active wizards."""
        cutoff = time.time() - max_age
        results = []
        for p in self.base_dir.glob("*.json"):
            try:
                state = WizardState.from_dict(json.loads(p.read_text()))
                if state.updated_at > cutoff and state.step != "done":
                    results.append(state)
            except Exception:
                pass
        results.sort(key=lambda s: s.updated_at, reverse=True)
        return results

    def cleanup(self, max_age: float = 604800):
        """Remove wizards older than max_age (default 7 days)."""
        cutoff = time.time() - max_age
        for p in self.base_dir.glob("*.json"):
            try:
                state = WizardState.from_dict(json.loads(p.read_text()))
                if state.updated_at < cutoff:
                    p.unlink()
            except Exception:
                pass
