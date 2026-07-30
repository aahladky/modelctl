"""Benchmark mode definitions and safety for modelctl.

Defines explicit cold and warm benchmark modes so the UI cannot
label a page-cache-warm run as an SSD-cold benchmark.

Public API:
    BenchmarkMode enum
    validate_mode(mode, context) -> list[str]
    describe_mode(mode) -> str
"""
from enum import Enum
from dataclasses import dataclass


class BenchmarkMode(Enum):
    """Explicit benchmark modes for plan testing.

    Each mode describes what cache state is assumed and how
    to achieve it.
    """
    NATURAL = "natural"
    """No cache manipulation. Record current state as-is."""

    PROCESS_COLD = "process-cold"
    """New backend process, OS page cache unchanged.
    Achieved by: starting a fresh process. Does NOT guarantee
    the model file isn't in page cache from a prior run."""

    PAGE_CACHE_WARM = "page-cache-warm"
    """Run a controlled warmup before measurement.
    Achieved by: loading the model once, discarding results,
    then measuring the second load."""

    EXPERT_CACHE_WARM = "expert-cache-warm"
    """Warmup until MoE cache counters stabilize or a token
    budget is reached. Requires cache metrics support."""

    STORAGE_COLD = "storage-cold"
    """Evict model file from page cache before measurement.
    Requires explicit consent and a clear scope. May require
    elevated privileges."""


@dataclass(frozen=True)
class BenchmarkContext:
    """Context for benchmark mode validation."""
    has_cache_metrics: bool = False
    can_drop_caches: bool = False
    model_path: str = ""
    storage_info: object = None  # StorageInfo if available


def describe_mode(mode: BenchmarkMode) -> str:
    """Human-readable description of a benchmark mode."""
    descriptions = {
        BenchmarkMode.NATURAL:
            "No cache manipulation. Measures current state as-is. "
            "Results may include page cache or expert cache from prior runs.",
        BenchmarkMode.PROCESS_COLD:
            "Fresh backend process. Does NOT guarantee the model file "
            "is evicted from OS page cache.",
        BenchmarkMode.PAGE_CACHE_WARM:
            "Controlled warmup: loads the model once before measuring. "
            "Ensures page cache is warm.",
        BenchmarkMode.EXPERT_CACHE_WARM:
            "Warmup until MoE cache counters stabilize. Requires cache "
            "metrics support in the backend.",
        BenchmarkMode.STORAGE_COLD:
            "Evicts model file from OS page cache before measuring. "
            "Requires explicit consent and may need elevated privileges.",
    }
    return descriptions.get(mode, "Unknown mode")


def validate_mode(mode: BenchmarkMode, ctx: BenchmarkContext) -> list[str]:
    """Validate that a benchmark mode is achievable in the current context.

    Returns a list of warnings/issues. Empty list means the mode is valid.
    """
    issues = []

    if mode == BenchmarkMode.EXPERT_CACHE_WARM and not ctx.has_cache_metrics:
        issues.append(
            "expert-cache-warm requires cache metrics support in the backend. "
            "The backend does not report moe_cache_metrics."
        )

    if mode == BenchmarkMode.STORAGE_COLD and not ctx.can_drop_caches:
        issues.append(
            "storage-cold requires elevated privileges to drop page cache. "
            "Run with appropriate permissions or use a different mode."
        )

    if mode == BenchmarkMode.STORAGE_COLD and ctx.storage_info:
        if ctx.storage_info.filesystem in ("nfs", "cifs", "smbfs"):
            issues.append(
                f"storage-cold on {ctx.storage_info.filesystem} may not be "
                "effective. Remote filesystems often have their own caching."
            )

    return issues


def label_for_result(mode: BenchmarkMode, validated: bool) -> str:
    """Generate a label for a benchmark result.

    When cold state cannot be guaranteed, the label includes
    'cold_unverified' to prevent misinterpretation.
    """
    if mode == BenchmarkMode.PROCESS_COLD and not validated:
        return "process-cold (page cache unverified)"
    if mode == BenchmarkMode.STORAGE_COLD and not validated:
        return "cold_unverified"
    return mode.value
