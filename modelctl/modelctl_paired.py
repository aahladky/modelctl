"""Paired benchmarking: alternate two conditions, compare inside each pair.

Two conditions measured as two blocks -- five runs of A, then five of B --
are comparable only if the machine did not change between the blocks. On
this rig it does. The determinism record's "cost of determinism"
(-8.01% / -14.87% / -15.20%) was read off exactly that shape, across a
battery whose loadavg ranged 2.63 to 17.15, so the block difference
contains both the knob and whatever else the machine was doing that hour.
Those figures are void, and this module is what replaces the method that
produced them.

Pairing removes the drift. A and B run back-to-back inside one pair, so
whatever the machine is doing during pair 3 it does to both arms of pair
3, and the comparison is the DELTA WITHIN each pair. The pairs, not the
runs, are the sample.

Pair order alternates -- (A,B), (B,A), (A,B), ... -- because "back to
back" is not symmetric: the second slot inherits the first's thermal and
page-cache state. Fixing the order would fold that asymmetry into every
delta with the same sign; alternating cancels it across pairs instead.
`schedule()` is a separate, trivially checkable function for that reason.

Nothing here decides anything. It reports every raw run, every per-pair
delta with its sign, and a two-sided exact sign test over those signs.
The sign test is a statistic -- how consistently the deltas agreed in
direction, and how surprising that consistency would be from coin flips.
It is not a verdict, there is no threshold in this module, and no
function returns a winner. Aaron reads the numbers.

Public API:
    Condition, Run, Pair, PairedComparison, SignTest
    schedule(pairs) -> list[tuple[int, int]]
    sign_test(deltas) -> SignTest
    run_paired(a, b, measure, pairs, ...) -> PairedComparison
"""
import math
import statistics
import time
from dataclasses import dataclass, field

import modelctl_load


@dataclass(frozen=True)
class Condition:
    """One arm. `config` and `env` are recorded verbatim into the result:
    a paired delta is only meaningful next to what the two arms actually
    differed by, and reconstructing that afterwards from prose is how the
    C1 naming collision happened in the first place."""
    name: str
    label: str = ""
    config: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)

    def to_dict(self):
        return {"name": self.name, "label": self.label or self.name,
                "config": dict(self.config), "env": dict(self.env)}


@dataclass
class Run:
    """One measurement of one condition inside one pair."""
    condition: str
    pair: int
    slot: int                       # 0 = ran first in its pair, 1 = second
    metrics: dict = field(default_factory=dict)
    load: dict = field(default_factory=dict)
    started: float | None = None
    finished: float | None = None
    error: str = ""

    @property
    def ok(self):
        return not self.error

    def to_dict(self):
        return {"condition": self.condition, "pair": self.pair,
                "slot": self.slot, "metrics": dict(self.metrics),
                "load": dict(self.load), "started": self.started,
                "finished": self.finished, "error": self.error}


@dataclass
class Pair:
    """Two runs that share a moment. `order` is what actually ran, not
    what was planned -- a resumed or retried pair must not claim the
    schedule's order."""
    index: int
    order: tuple = ()
    runs: dict = field(default_factory=dict)     # condition name -> Run

    @property
    def complete(self):
        return len(self.runs) == 2 and all(r.ok for r in self.runs.values())

    def value(self, condition, metric):
        run = self.runs.get(condition)
        if run is None or not run.ok:
            return None
        v = run.metrics.get(metric)
        return v if isinstance(v, (int, float)) else None

    def delta(self, a, b, metric):
        """b - a for this pair, or None if either arm is missing.

        An incomplete pair is dropped from the statistics and kept in the
        raw record. Silently substituting a mean for the missing arm --
        or dropping the pair from the record as well -- would both make
        the sample look better behaved than it was.
        """
        av, bv = self.value(a, metric), self.value(b, metric)
        if av is None or bv is None:
            return None
        return bv - av

    def to_dict(self):
        return {"index": self.index, "order": list(self.order),
                "runs": {k: r.to_dict() for k, r in self.runs.items()}}


@dataclass(frozen=True)
class SignTest:
    """A two-sided exact sign test over the per-pair delta signs.

    `p_value` is the probability of a split at least this lopsided from
    fair coin flips. It carries no threshold and implies no decision;
    `ties` are deltas of exactly zero, which the test excludes from `n`
    because a zero delta is evidence of neither direction.
    """
    n: int                  # non-tied pairs -- the sample the test used
    positive: int
    negative: int
    ties: int
    p_value: float | None

    def to_dict(self):
        return {"n": self.n, "positive": self.positive,
                "negative": self.negative, "ties": self.ties,
                "p_value": self.p_value,
                "note": ("two-sided exact sign test over per-pair delta "
                         "signs; ties excluded from n; no threshold is "
                         "applied here")}


def sign_test(deltas) -> SignTest:
    """Exact two-sided sign test. `deltas` may contain None (dropped)."""
    values = [d for d in (deltas or []) if d is not None]
    positive = sum(1 for d in values if d > 0)
    negative = sum(1 for d in values if d < 0)
    ties = sum(1 for d in values if d == 0)
    n = positive + negative
    if n == 0:
        # No non-tied pair: there is nothing for the test to be about,
        # and a p-value of 1.0 here would read as a measured result
        # rather than an absence of data.
        return SignTest(0, positive, negative, ties, None)
    k = max(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return SignTest(n, positive, negative, ties, min(1.0, 2.0 * tail))


def schedule(pairs: int):
    """Slot order per pair: (0,1) on even pairs, (1,0) on odd ones.

    Returned as indices into the (a, b) tuple so the alternation can be
    read and tested without running anything.
    """
    return [((0, 1) if i % 2 == 0 else (1, 0)) for i in range(max(0, int(pairs)))]


@dataclass
class PairedComparison:
    """The whole record: both arms, every run, every delta, the sign test."""
    a: Condition
    b: Condition
    metric: str
    pairs: list = field(default_factory=list)
    started: float | None = None
    finished: float | None = None
    notes: list = field(default_factory=list)

    def deltas(self):
        """Per-pair b - a, with None kept in place so a delta's position
        still identifies its pair."""
        return [p.delta(self.a.name, self.b.name, self.metric)
                for p in self.pairs]

    def relative_deltas(self):
        """Per-pair (b - a) / a. None where a is missing or zero."""
        out = []
        for p in self.pairs:
            av = p.value(self.a.name, self.metric)
            d = p.delta(self.a.name, self.b.name, self.metric)
            out.append(None if (d is None or not av) else d / av)
        return out

    def sign_test(self):
        return sign_test(self.deltas())

    def complete_pairs(self):
        return [p for p in self.pairs if p.complete]

    def incomplete_pairs(self):
        return [p for p in self.pairs if not p.complete]

    def load_comparability(self, tolerance=1.0):
        """Did the two arms see the same machine? Per pair, not overall --
        pairing's whole claim is that the arms inside a pair shared a
        moment, so that is the level the claim has to be checked at."""
        out = []
        for p in self.pairs:
            summaries = [r.load for r in p.runs.values() if r.load]
            check = modelctl_load.comparable_load(summaries, tolerance)
            check["pair"] = p.index
            out.append(check)
        return out

    def to_dict(self):
        deltas = self.deltas()
        present = [d for d in deltas if d is not None]
        return {
            "metric": self.metric,
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "delta_convention": f"delta = {self.b.name} - {self.a.name}",
            "pairs_run": len(self.pairs),
            "pairs_complete": len(self.complete_pairs()),
            "pairs_incomplete": [p.index for p in self.incomplete_pairs()],
            "per_pair": [
                {"pair": p.index, "order": list(p.order),
                 self.a.name: p.value(self.a.name, self.metric),
                 self.b.name: p.value(self.b.name, self.metric),
                 "delta": p.delta(self.a.name, self.b.name, self.metric),
                 "sign": _sign_of(p.delta(self.a.name, self.b.name, self.metric))}
                for p in self.pairs],
            "relative_deltas": self.relative_deltas(),
            # Median, not mean: the statistic that goes with a sign test,
            # and the one a single stalled run cannot move on its own.
            # It is a description of the deltas, not a decision from them.
            "median_delta": statistics.median(present) if present else None,
            "sign_test": self.sign_test().to_dict(),
            "load_comparability": self.load_comparability(),
            "runs": [r.to_dict() for p in self.pairs
                     for r in p.runs.values()],
            "started": self.started,
            "finished": self.finished,
            "notes": list(self.notes),
        }


def _sign_of(delta):
    if delta is None:
        return None
    return 0 if delta == 0 else (1 if delta > 0 else -1)


def run_paired(a: Condition, b: Condition, measure, pairs=5, metric="generation_tps",
               load_interval=5.0, between=None, should_stop=None,
               recorder_factory=None, clock=time.time) -> PairedComparison:
    """Run `pairs` alternating pairs of (a, b) and return the whole record.

    `measure(condition, pair_index, slot) -> dict` performs one run and
    returns its metrics; it is injected so the alternation, the load
    recording and the statistics are testable without a GPU, and so the
    caller keeps ownership of how a condition is actually launched.

    A run that raises does not abort the comparison: its pair is recorded
    incomplete, with the exception text, and excluded from the sign test.
    Aborting would discard the pairs already taken, and the failure of one
    arm is itself part of the record.
    """
    comparison = PairedComparison(a=a, b=b, metric=metric, started=clock())
    arms = (a, b)
    factory = recorder_factory or (
        lambda: modelctl_load.LoadRecorder(interval=load_interval))

    for index, order in enumerate(schedule(pairs)):
        if should_stop is not None and should_stop():
            comparison.notes.append(
                f"stopped before pair {index}: {len(comparison.pairs)} of "
                f"{pairs} pairs were run")
            break
        pair = Pair(index=index, order=tuple(arms[i].name for i in order))
        for slot, arm_index in enumerate(order):
            condition = arms[arm_index]
            recorder = factory()
            started = clock()
            run = Run(condition=condition.name, pair=index, slot=slot,
                      started=started)
            recorder.start()
            try:
                run.metrics = measure(condition, index, slot) or {}
            except Exception as e:
                run.error = f"{type(e).__name__}: {e}"
            finally:
                trace = recorder.stop()
                run.load = trace.summary()
                run.finished = clock()
            pair.runs[condition.name] = run
            if between is not None and slot == 0:
                between()
        comparison.pairs.append(pair)

    comparison.finished = clock()
    return comparison
