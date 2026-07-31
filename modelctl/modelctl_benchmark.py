"""Benchmark mode definitions, execution, and safety for modelctl.

Defines explicit cold and warm benchmark modes so the UI cannot
label a page-cache-warm run as an SSD-cold benchmark.

The modes are controlled procedures, not labels. The part
that matters is verification: a "storage-cold" run that did not actually
evict anything is reported as `cold_unverified`, because a cold label on
a warm measurement corrupts every later comparison that trusts it.

Residency is measured with mincore(2) over an mmap of the model file, and
eviction uses posix_fadvise(POSIX_FADV_DONTNEED), which is scoped to that
file and needs no privileges -- unlike /proc/sys/vm/drop_caches, which
empties the whole machine's cache and requires root.

Public API:
    BenchmarkMode enum
    validate_mode(mode, context) -> list[str]
    describe_mode(mode) -> str
    label_for_result(mode, validated) -> str
    page_cache_residency(paths) -> Residency | None
    evict_from_page_cache(paths) -> bool
    prepare_mode(mode, context) -> ModePreparation
    expert_cache_stabilized(samples, ...) -> bool
"""
import ctypes
import ctypes.util
import os
from enum import Enum
from dataclasses import dataclass, field


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
    # Storage-cold destroys the page cache for this model, making the next
    # real request slow. It must never happen because a default said so.
    consent_storage_cold: bool = False
    # Extra files (shards, mmproj, draft) that must be evicted together;
    # evicting one shard of five proves nothing.
    companion_paths: tuple = ()


@dataclass(frozen=True)
class Residency:
    """How much of a file set is currently in the OS page cache."""
    resident_bytes: int
    total_bytes: int

    @property
    def fraction(self) -> float:
        return (self.resident_bytes / self.total_bytes) if self.total_bytes else 0.0


@dataclass
class ModePreparation:
    """The outcome of putting the machine into a mode's required state."""
    mode: object
    # Whether the mode's precondition was actually achieved. False does not
    # abort the run -- it changes the label the result is recorded under.
    achieved: bool = True
    label: str = ""
    messages: list = field(default_factory=list)
    # Counters proving what happened, so the claim can be audited later.
    evidence: dict = field(default_factory=dict)
    # Set when the mode cannot be attempted at all (no consent, etc.).
    refused: bool = False


_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _libc():
    try:
        return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except (OSError, TypeError):
        return None


def _file_residency(path):
    """(resident_bytes, total_bytes) for one file, or None if unmeasurable."""
    libc = _libc()
    if libc is None or not hasattr(libc, "mincore"):
        return None
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return (0, 0)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        PROT_READ, MAP_SHARED = 1, 1
        addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if not addr or addr == ctypes.c_void_p(-1).value:
            return None
        try:
            npages = (size + _PAGE_SIZE - 1) // _PAGE_SIZE
            vec = (ctypes.c_ubyte * npages)()
            if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                return None
            # mincore sets bit 0 on resident pages. Counting in C over the
            # raw bytes keeps this fast for a 100 GiB model, where the
            # vector is tens of millions of entries.
            buf = bytes(vec)
            resident_pages = sum(buf.count(v) for v in range(1, 256, 2))
            return (min(resident_pages * _PAGE_SIZE, size), size)
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    except (OSError, ValueError, AttributeError):
        return None
    finally:
        os.close(fd)


def page_cache_residency(paths) -> Residency | None:
    """How much of a model (all its files) is in the page cache right now.

    Returns None when residency cannot be measured at all, which is
    different from "nothing is resident" and must not be reported as cold.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    resident = total = 0
    measured = False
    for p in paths:
        if not p:
            continue
        got = _file_residency(str(p))
        if got is None:
            continue
        measured = True
        resident += got[0]
        total += got[1]
    return Residency(resident, total) if measured else None


def evict_from_page_cache(paths) -> bool:
    """Drop a model's pages from the page cache, scoped to its own files.

    posix_fadvise(DONTNEED) only drops clean, unreferenced pages -- if a
    server still has the file mmap'd, this legitimately does nothing.
    That is exactly why callers must verify residency afterwards rather
    than trusting the call's return value.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    any_ok = False
    for p in paths:
        if not p:
            continue
        try:
            fd = os.open(str(p), os.O_RDONLY)
        except OSError:
            continue
        try:
            # DONTNEED skips dirty pages, so a file that was just written
            # stays fully cached and the eviction silently does nothing.
            # Flushing first is what makes the call effective on a model
            # that was recently downloaded.
            try:
                os.fsync(fd)
            except OSError:
                pass
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            any_ok = True
        except (OSError, AttributeError):
            pass
        finally:
            os.close(fd)
    return any_ok


# Eviction counts as effective only if residency actually dropped to at
# most this fraction. A model still 80% cached is not cold by any useful
# definition.
_COLD_RESIDENCY_CEILING = 0.10


def prepare_mode(mode: BenchmarkMode, ctx: BenchmarkContext) -> ModePreparation:
    """Put the machine into the state `mode` claims, and report honestly.

    Never raises and never silently downgrades: when a precondition cannot
    be met the run still proceeds, but `achieved` is False and `label`
    carries the qualification, so the stored result cannot later be read
    as something it was not.
    """
    paths = [p for p in (ctx.model_path, *ctx.companion_paths) if p]
    prep = ModePreparation(mode=mode, label=mode.value)

    if mode == BenchmarkMode.NATURAL:
        # Records the current state and claims nothing about it.
        before = page_cache_residency(paths)
        if before:
            prep.evidence["residency_fraction"] = round(before.fraction, 4)
            prep.evidence["resident_bytes"] = before.resident_bytes
        prep.messages.append(
            "natural: no cache manipulation; the result describes whatever "
            "state the machine was already in")
        return prep

    if mode == BenchmarkMode.PROCESS_COLD:
        before = page_cache_residency(paths)
        if before:
            prep.evidence["residency_fraction"] = round(before.fraction, 4)
            prep.evidence["resident_bytes"] = before.resident_bytes
            # A fresh process says nothing about the page cache; measuring
            # it turns "unknown" into a number.
            prep.messages.append(
                f"process-cold: fresh process, but "
                f"{before.fraction * 100:.0f}% of the model is already in "
                f"the page cache")
            prep.achieved = before.fraction <= _COLD_RESIDENCY_CEILING
        else:
            prep.achieved = False
            prep.messages.append(
                "process-cold: page-cache residency could not be measured")
        prep.label = label_for_result(mode, prep.achieved)
        return prep

    if mode == BenchmarkMode.PAGE_CACHE_WARM:
        # The caller performs the load/unload cycle; this records the
        # starting point so "warm" can be checked rather than assumed.
        before = page_cache_residency(paths)
        if before:
            prep.evidence["residency_fraction_before"] = round(before.fraction, 4)
        prep.messages.append(
            "page-cache-warm: run a full load and unload before measuring")
        return prep

    if mode == BenchmarkMode.EXPERT_CACHE_WARM:
        if not ctx.has_cache_metrics:
            # Without counters there is no way to know the cache warmed,
            # so the result must not claim it did.
            prep.achieved = False
            prep.label = "expert-cache-warm (unverified)"
            prep.messages.append(
                "expert-cache-warm: the backend does not report "
                "moe_cache_metrics, so stabilisation cannot be verified")
            return prep
        prep.messages.append(
            "expert-cache-warm: warm until cache counters stabilise or the "
            "token budget is spent")
        return prep

    if mode == BenchmarkMode.STORAGE_COLD:
        if not ctx.consent_storage_cold:
            prep.refused = True
            prep.achieved = False
            prep.label = "cold_unverified"
            prep.messages.append(
                "storage-cold requires explicit opt-in: it evicts this "
                "model from the page cache, making the next real request "
                "slow")
            return prep

        before = page_cache_residency(paths)
        evicted = evict_from_page_cache(paths)
        after = page_cache_residency(paths)

        if before:
            prep.evidence["residency_fraction_before"] = round(before.fraction, 4)
        if after:
            prep.evidence["residency_fraction_after"] = round(after.fraction, 4)
        prep.evidence["evict_call_succeeded"] = evicted

        if after is None:
            prep.achieved = False
            prep.messages.append(
                "storage-cold: residency could not be measured after "
                "eviction, so coldness is unverified")
        elif after.fraction <= _COLD_RESIDENCY_CEILING:
            prep.achieved = True
            prep.messages.append(
                f"storage-cold: evicted; {after.fraction * 100:.1f}% of the "
                f"model remains cached")
        else:
            # The call can succeed while dropping nothing -- a running
            # server holding the mapping keeps its pages referenced.
            prep.achieved = False
            prep.messages.append(
                f"storage-cold: eviction did not take effect — "
                f"{after.fraction * 100:.0f}% of the model is still cached "
                f"(is the model still loaded?)")
        prep.label = label_for_result(mode, prep.achieved)
        return prep

    return prep


def expert_cache_stabilized(samples, tolerance: float = 0.01,
                            required: int = 3) -> bool:
    """Have the MoE cache hit-rate samples settled?

    `samples` is a sequence of hit ratios in observation order. Stability
    means the last `required` samples all sit within `tolerance` of their
    mean; a still-climbing hit rate is not warm yet.
    """
    window = [s for s in (samples or []) if s is not None][-required:]
    if len(window) < required:
        return False
    mean = sum(window) / len(window)
    return all(abs(s - mean) <= tolerance for s in window)


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
        fs = ctx.storage_info.filesystem
        if fs in ("nfs", "cifs", "smbfs"):
            issues.append(
                f"storage-cold on {fs} may not be "
                "effective. Remote filesystems often have their own caching."
            )
        if fs in ("tmpfs", "ramfs"):
            # On a RAM-backed filesystem the cached pages *are* the file.
            # There is no storage layer to go cold, so the mode can never
            # be achieved -- and must never be claimed.
            issues.append(
                f"storage-cold is impossible on {fs}: the file lives in RAM, "
                "so there is no storage read to measure."
            )

    if mode == BenchmarkMode.STORAGE_COLD and not ctx.consent_storage_cold:
        issues.append(
            "storage-cold needs explicit opt-in: it evicts this model from "
            "the page cache, so the next real request pays a full reload."
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
