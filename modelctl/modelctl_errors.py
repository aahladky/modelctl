"""Structured validation messages and failure classes for modelctl.

Replaces free-form preflight text with machine-readable results that
the web UI, fallback policy, and decision traces can act on.

Public API:
    ValidationMessage dataclass
    ValidationResult (list of ValidationMessage with helpers)
"""
from dataclasses import dataclass
from typing import Literal


class ProfileNotFoundError(Exception):
    """Raised when a named profile has no file on disk.

    load_profile() raises this instead of exiting so that long-lived
    callers (web job lanes, route handlers, services) survive a missing
    profile; the CLI translates it to a friendly message and exit code 1
    in exactly one place (main()).
    """

    def __init__(self, name: str):
        self.name = name
        super().__init__(
            f"No profile named '{name}'. Run `modelctl list` to see saved profiles.")


@dataclass(frozen=True)
class ValidationMessage:
    """A single validation result attached to a launch path, plan, or profile."""
    code: str
    severity: Literal["info", "warning", "error"]
    summary: str
    detail: str = ""
    recovery: str = ""
    field: str | None = None

    def as_tuple(self):
        """Backward-compatible (severity, summary) tuple."""
        return (self.severity, self.summary)


# Well-known validation codes.  Consumers should switch on these, not
# on English text.
CODES = {
    # Binary / backend
    "binary_missing",
    "binary_not_executable",
    "backend_probe_unsupported",
    "backend_feature_missing",
    "backend_device_missing",
    "backend_probe_stale",
    # Cache configuration
    "invalid_cache_budget",
    "cache_feature_unsupported",
    "cache_hybrid_unsupported",
    "cache_prefetch_unsupported",
    # Resources
    "insufficient_vram",
    "insufficient_ram",
    "storage_path_unavailable",
    "model_file_missing",
    # Arguments
    "invalid_backend_argument",
    "reservation_conflict",
    # Runtime
    "health_timeout",
    "backend_crash",
    "numerical_mismatch",
    # General
    "experimental_feature",
}


def errors_only(messages: list) -> list:
    """Return only error-severity messages."""
    return [m for m in messages if m.severity == "error"]


def warnings_only(messages: list) -> list:
    """Return only warning-severity messages."""
    return [m for m in messages if m.severity == "warning"]


def has_errors(messages: list) -> bool:
    """True if any message is an error."""
    return any(m.severity == "error" for m in messages)


def max_severity(messages: list) -> str:
    """Return the highest severity in the list: error > warning > info."""
    for level in ("error", "warning", "info"):
        if any(m.severity == level for m in messages):
            return level
    return "info"


def from_preflight_tuples(tuples: list) -> list:
    """Convert legacy (severity, message) tuples to ValidationMessage objects.

    Used during migration from the old preflight_moe_cache() output.
    """
    result = []
    for severity, summary in tuples:
        # Try to infer a code from the summary text.
        code = "unknown"
        lower = summary.lower()
        if "moe_weight_transfer_cache" in lower or "moe_expert_cache" in lower:
            code = "cache_feature_unsupported"
        elif "moe_cache_sycl" in lower:
            code = "backend_feature_missing"
        elif "moe_hybrid_cpu_miss" in lower or "hybrid" in lower:
            code = "cache_hybrid_unsupported"
        elif "prefetch" in lower:
            code = "cache_prefetch_unsupported"
        elif "binary" in lower or "backend" in lower:
            code = "backend_probe_unsupported"
        result.append(ValidationMessage(
            code=code,
            severity=severity,
            summary=summary,
        ))
    return result
