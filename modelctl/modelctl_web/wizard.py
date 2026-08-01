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

import modelctl_fsutil
from modelctl_paths import STATE_DIR

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
    # Set when registration failed. The profile is left valid and saved;
    # this is what the register page shows alongside its retry action.
    registration_error: str = ""

    # Verification of the chosen source, recorded before any profile is
    # created (GGUF magic, shard completeness, duplicates, free space).
    source_verification: dict = field(default_factory=dict)
    # Hugging Face metadata / checksums for the selected files, when the
    # source published any.
    source_metadata: dict = field(default_factory=dict)

    # Provenance of what was actually registered.
    command_fingerprint: str = ""
    measured: dict = field(default_factory=dict)

    # Structured per-step outcomes, keyed by step name:
    #   {ok, job_id, status, messages, warnings, error, at}
    # Recorded from the job store's structured result rather than scraped
    # out of a log, so a retry can tell "never ran" from "ran and failed".
    step_outcomes: dict = field(default_factory=dict)

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

    # --- structured step outcomes ---------------------------------------

    def record_outcome(self, step: str, ok: bool, job_id: str = "",
                       status: str = "", messages=None, warnings=None,
                       error: str = ""):
        """Record what happened on a step."""
        self.step_outcomes[step] = {
            "ok": bool(ok),
            "job_id": job_id or "",
            "status": status or ("done" if ok else "failed"),
            "messages": list(messages or []),
            "warnings": list(warnings or []),
            "error": error or "",
            "at": time.time(),
        }
        self.updated_at = time.time()
        return self.step_outcomes[step]

    def outcome(self, step: str) -> dict:
        return self.step_outcomes.get(step) or {}

    def clear_outcome(self, step: str):
        """Forget a step's result so it can be retried from scratch."""
        self.step_outcomes.pop(step, None)
        self.updated_at = time.time()

    def step_failed(self, step: str) -> bool:
        o = self.outcome(step)
        return bool(o) and not o.get("ok") and o.get("status") != "running"

    def step_running(self, step: str) -> bool:
        return self.outcome(step).get("status") in ("queued", "running")

    def blocking_reason(self, step: str) -> str:
        """Why the wizard must not move past `step` yet, or "".

        Advancing over a running job produces a wizard that looks finished
        while its download is still going; advancing over a failed one
        produces a profile built on a step that never succeeded.
        """
        if self.step_running(step):
            return f"the {step} step is still running"
        if self.step_failed(step):
            o = self.outcome(step)
            detail = o.get("error") or "; ".join(o.get("messages") or [])
            return f"the {step} step failed{': ' + detail if detail else ''}"
        return ""

    def can_advance(self, step: str) -> bool:
        return not self.blocking_reason(step)


    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WizardState":
        # Filter to known fields.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


def outcome_from_job(store, job_id: str) -> dict:
    """Read a job's structured result out of the job store.

    The wizard used to infer success by looking at the log text. Jobs
    already persist a structured outcome, a status and an error; using
    them means a retry can distinguish "never started", "still running",
    "failed", and "succeeded with warnings".
    """
    if not job_id:
        return {}
    job = store.get(job_id)
    if not job:
        return {"ok": False, "status": "missing", "job_id": job_id,
                "error": "job record is gone"}
    status = job.get("status", "")
    data = {}
    raw = job.get("outcome")
    if raw:
        try:
            data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            data = {}
        # Legacy rows can hold a JSON-encoded scalar rather than an object;
        # treating one as a dict 500'd the wizard download page.
        if not isinstance(data, dict):
            data = {}
    return {
        "ok": status == "done",
        "status": status,
        "job_id": job_id,
        "data": data,
        "warnings": list(data.get("warnings") or []),
        "error": job.get("error", "") or "",
    }


class WizardStore:
    """Persistent storage for wizard state."""

    def __init__(self, base_dir: Path | None = None):
        # Resolved at call time (not bound as a default) so tests can
        # patch WIZARD_DIR and route handlers pick the patched dir up.
        self.base_dir = base_dir if base_dir is not None else WIZARD_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, wizard_id: str) -> Path:
        return self.base_dir / f"{wizard_id}.json"

    def save(self, state: WizardState):
        """Persist wizard state to disk."""
        state.updated_at = time.time()
        modelctl_fsutil.atomic_write_text(
            self._path(state.wizard_id),
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
