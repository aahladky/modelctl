"""Concurrent machine load, sampled beside a measurement and recorded with it.

The project rule is that a benchmark records raw numbers, the exact
config, and the concurrent machine load. The third was the one that got
left out, and it is exactly what voided the "cost of determinism"
figures in the 2026-08-01 determinism record: those -8.01% / -14.87% /
-15.20% deltas were read off two blocks of runs taken across a battery
whose loadavg ranged 2.63 to 17.15. The numbers are real; what they
measure is not only the knob.

So load is sampled here as a first-class part of a run rather than
noted once in prose afterwards. Two properties matter:

  * Sampling is cheap enough to run in a thread beside a decode loop --
    everything comes from /proc, nothing shells out, nothing touches a
    GPU. A load recorder that perturbs the thing it measures is worse
    than none.
  * What could not be read is recorded as unreadable, never as zero. An
    unread loadavg is not an idle machine, and a summary that quietly
    reports 0.0 for it would make the worst runs look like the best.
"""
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

MEMINFO = Path("/proc/meminfo")
LOADAVG = Path("/proc/loadavg")


@dataclass(frozen=True)
class LoadSample:
    """One instant of machine state. A None field was not readable."""
    at: float
    loadavg_1m: float | None = None
    loadavg_5m: float | None = None
    loadavg_15m: float | None = None
    runnable: int | None = None       # currently-runnable tasks
    procs: int | None = None          # total tasks
    mem_available_bytes: int | None = None
    mem_cached_bytes: int | None = None
    swap_free_bytes: int | None = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def _read_loadavg():
    try:
        parts = LOADAVG.read_text().split()
    except (OSError, ValueError):
        return {}
    out = {}
    try:
        out["loadavg_1m"] = float(parts[0])
        out["loadavg_5m"] = float(parts[1])
        out["loadavg_15m"] = float(parts[2])
    except (IndexError, ValueError):
        pass
    try:
        runnable, _, procs = parts[3].partition("/")
        out["runnable"] = int(runnable)
        out["procs"] = int(procs)
    except (IndexError, ValueError):
        pass
    return out


_MEMINFO_KEYS = {
    "MemAvailable:": "mem_available_bytes",
    "Cached:": "mem_cached_bytes",
    "SwapFree:": "swap_free_bytes",
}


def _read_meminfo():
    try:
        text = MEMINFO.read_text()
    except OSError:
        return {}
    out = {}
    for line in text.splitlines():
        field_name = _MEMINFO_KEYS.get(line.split(" ", 1)[0])
        if field_name is None:
            continue
        try:
            out[field_name] = int(line.split()[1]) * 1024
        except (IndexError, ValueError):
            pass
    return out


def sample_load(now=None) -> LoadSample:
    """One sample. Never raises: a benchmark must not die because /proc
    hiccuped, and a missing field is more honest than a substituted one."""
    values = {}
    values.update(_read_loadavg())
    values.update(_read_meminfo())
    return LoadSample(at=(time.time() if now is None else now), **values)


def _stats(values):
    """min/mean/max over the readable values, or None when there are none."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return {"min": min(present), "max": max(present),
            "mean": sum(present) / len(present), "n": len(present)}


@dataclass
class LoadTrace:
    """The samples taken across one run, plus how to state them."""
    samples: list = field(default_factory=list)
    interval: float = 0.0
    started: float | None = None
    finished: float | None = None
    # Samples the recorder was asked for but could not take, e.g. because
    # it was stopped mid-interval. Recorded so `n` is never read as
    # "this is the whole run" when it isn't.
    missed: int = 0

    def add(self, sample):
        self.samples.append(sample)

    def summary(self) -> dict:
        """Load across the run, in the fields a comparison is read against.

        `samples: 0` is a real and reportable answer -- a run too short to
        sample is a run whose load is unknown, and the summary says so
        rather than implying an idle machine.
        """
        out = {
            "samples": len(self.samples),
            "interval_seconds": self.interval,
            "started": self.started,
            "finished": self.finished,
            "missed": self.missed,
        }
        for name in ("loadavg_1m", "loadavg_5m", "loadavg_15m", "runnable",
                     "procs", "mem_available_bytes", "mem_cached_bytes",
                     "swap_free_bytes"):
            stat = _stats([getattr(s, name) for s in self.samples])
            if stat is not None:
                out[name] = stat
        if not self.samples:
            out["note"] = ("no load samples were taken: the concurrent load "
                           "during this run is unknown, not zero")
        return out

    def to_dict(self) -> dict:
        return {"summary": self.summary(),
                "samples": [s.to_dict() for s in self.samples]}


class LoadRecorder:
    """Samples load in a background thread for the duration of a run.

    Used as a context manager so a run that raises still gets its trace
    closed and its samples kept -- a failed run's load is often the
    reason it failed, and discarding it loses the evidence.
    """

    def __init__(self, interval=5.0, sampler=sample_load, clock=time.time):
        self.interval = float(interval)
        self._sampler = sampler
        self._clock = clock
        self.trace = LoadTrace(interval=self.interval)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return self
        self.trace.started = self._clock()
        # One sample immediately: a run shorter than the interval would
        # otherwise produce an empty trace, which is the case where the
        # load is least likely to be noticed and most likely to matter.
        self.trace.add(self._sampler())
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="load-recorder")
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.trace.add(self._sampler())
            except Exception:
                # Sampling must never take the run down with it. Count it
                # so the summary can say the trace has holes.
                self.trace.missed += 1

    def stop(self) -> LoadTrace:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval + 5)
            self._thread = None
        self.trace.finished = self._clock()
        return self.trace

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def comparable_load(traces, loadavg_tolerance=1.0) -> dict:
    """Were these runs taken under the same machine?

    Answers with the observed spread and the tolerance it was checked
    against, never with a verdict on the measurements themselves. The
    caller decides what an incomparable load means for its own numbers;
    this only refuses to let the question go unasked.
    """
    means = []
    for t in traces or []:
        summary = t.summary() if hasattr(t, "summary") else (t or {})
        stat = summary.get("loadavg_1m")
        if isinstance(stat, dict) and stat.get("mean") is not None:
            means.append(stat["mean"])
    if len(means) < 2:
        return {"checked": False,
                "reason": "fewer than two runs carry a readable loadavg",
                "runs_with_load": len(means)}
    spread = max(means) - min(means)
    return {
        "checked": True,
        "runs_with_load": len(means),
        "loadavg_1m_min": min(means),
        "loadavg_1m_max": max(means),
        "loadavg_1m_spread": spread,
        "tolerance": loadavg_tolerance,
        "within_tolerance": spread <= loadavg_tolerance,
    }
