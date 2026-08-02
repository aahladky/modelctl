"""Anchor registry: a reference measurement, and what produced it.

An anchor is a number later work is read against. Its failure mode is
that it outlives the conditions it was taken under. The 122B battery in
the 2026-08-01 determinism record is a set of real numbers, but it was
taken across a loadavg spanning 2.63 to 17.15; anything compared against
it inherits that contamination without anything in the comparison saying
so.

So an anchor here is never a bare value. It carries the fingerprint of
what produced it -- build commit, profile hash, environment, driver --
and a battery re-runs an anchor exactly when that fingerprint no longer
matches the machine in front of it. Re-running an anchor whose
fingerprint still holds costs hours and produces a second sample of the
same thing; re-using one whose fingerprint has moved is the more
expensive mistake, so staleness is decided field by field and the
differing fields are named.

Two things a fingerprint check cannot express, both first-class here:

  * `void` -- the measurement itself was bad, whatever the conditions
    were. "The machine changed" and "that number should never have been
    recorded" are different facts, and an anchor voided by hand re-runs
    with its reason attached rather than being quietly deleted.
  * `always_run` -- laguna-s2.1's canary. It is not a value to compare
    against; it exists to notice that the machine moved under everything
    else, which is precisely the case where its fingerprint still
    matches. A fingerprint gate would skip it exactly when it matters.

The registry lives in the repo next to night-lane.json, version
controlled and diffable, for the same reason: an anchor that can change
without a diff is not a reference.

Public API:
    Fingerprint, Anchor
    load_anchors / save_anchors / anchor_by_id
    profile_hash, env_hash, driver_identity, current_fingerprint
    staleness(anchor, fingerprint) -> list[str]
    plan_battery(anchors, fingerprint) -> BatteryPlan
"""
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "anchors.json"

# The environment variables that change what a number means. Deliberately
# a whitelist: hashing the whole environment would stale every anchor on
# an unrelated shell variable, and hashing nothing would let an ambient
# GGML_OP_OFFLOAD_MOE_MIN_BATCH change the runtime silently -- the same
# failure modelctl_launch's identity_environment() exists to prevent.
FINGERPRINTED_ENV = (
    "GGML_SYCL_DETERMINISTIC",
    "GGML_SYCL_ENABLE_DNN",
    "GGML_SYCL_FA_ONEDNN",
    "GGML_OP_OFFLOAD_MIN_BATCH",
    "GGML_OP_OFFLOAD_MOE_MIN_BATCH",
    "GGML_MOE_CACHE_MMAP_ADVISE",
    "ONEAPI_DEVICE_SELECTOR",
    "LD_LIBRARY_PATH",
)


@dataclass(frozen=True)
class Fingerprint:
    """What an anchor's value is only true of.

    Every field defaults to "" meaning *unknown*, and unknown never
    matches -- an anchor recorded without a build commit cannot be shown
    to still apply, so it is stale rather than assumed current.
    """
    build_commit: str = ""
    profile_hash: str = ""
    env_hash: str = ""
    driver: str = ""

    def to_dict(self):
        return {"build_commit": self.build_commit,
                "profile_hash": self.profile_hash,
                "env_hash": self.env_hash, "driver": self.driver}

    @staticmethod
    def from_dict(d):
        d = d or {}
        return Fingerprint(build_commit=d.get("build_commit", ""),
                           profile_hash=d.get("profile_hash", ""),
                           env_hash=d.get("env_hash", ""),
                           driver=d.get("driver", ""))

    def differences(self, other) -> list:
        """Field-by-field, in words. Named fields, not a single digest:
        "the driver moved" and "the binary moved" call for different
        work, and a combined hash cannot tell them apart."""
        if other is None:
            return ["no current fingerprint to compare against"]
        out = []
        for name in ("build_commit", "profile_hash", "env_hash", "driver"):
            mine, theirs = getattr(self, name), getattr(other, name)
            if not mine:
                out.append(f"{name} was not recorded with this anchor")
            elif not theirs:
                out.append(f"{name} is unknown on this machine")
            elif mine != theirs:
                out.append(f"{name}: anchored at {mine}, now {theirs}")
        return out


@dataclass(frozen=True)
class Anchor:
    """One reference measurement."""
    id: str
    condition: str                 # the config in words, e.g. "C1 static ..."
    metric: str = "generation_tps"
    unit: str = "tok/s"
    value: float | None = None     # the headline figure, if one was taken
    runs: tuple = ()               # every raw run behind it
    fingerprint: Fingerprint = field(default_factory=Fingerprint)
    recorded: str = ""             # ISO date
    source: str = ""               # the evidence record it came from
    load: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # hit ratio, budgets, ...
    void: bool = False
    void_reason: str = ""
    always_run: bool = False
    note: str = ""

    def to_dict(self):
        return {"id": self.id, "condition": self.condition,
                "metric": self.metric, "unit": self.unit, "value": self.value,
                "runs": list(self.runs),
                "fingerprint": self.fingerprint.to_dict(),
                "recorded": self.recorded, "source": self.source,
                "load": dict(self.load), "extra": dict(self.extra),
                "void": self.void, "void_reason": self.void_reason,
                "always_run": self.always_run, "note": self.note}

    @staticmethod
    def from_dict(d):
        return Anchor(
            id=d["id"], condition=d.get("condition", ""),
            metric=d.get("metric", "generation_tps"),
            unit=d.get("unit", "tok/s"), value=d.get("value"),
            runs=tuple(d.get("runs", [])),
            fingerprint=Fingerprint.from_dict(d.get("fingerprint")),
            recorded=d.get("recorded", ""), source=d.get("source", ""),
            load=d.get("load", {}) or {}, extra=d.get("extra", {}) or {},
            void=bool(d.get("void", False)),
            void_reason=d.get("void_reason", ""),
            always_run=bool(d.get("always_run", False)),
            note=d.get("note", ""))


# --- registry io ---------------------------------------------------------

def load_anchors(path=None) -> list:
    p = path or REGISTRY_PATH
    try:
        raw = json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return []
    return [Anchor.from_dict(a) for a in raw.get("anchors", [])]


def save_anchors(anchors, path=None):
    p = Path(path or REGISTRY_PATH)
    payload = {"version": 1, "anchors": [a.to_dict() for a in anchors]}
    p.write_text(json.dumps(payload, indent=2) + "\n")


def anchor_by_id(anchor_id, anchors=None, path=None):
    for a in (anchors if anchors is not None else load_anchors(path)):
        if a.id == anchor_id:
            return a
    return None


def void_anchor(anchor, reason) -> Anchor:
    """Mark an anchor's value unusable, keeping the value and its
    provenance. The number stays readable because the record of a
    measurement that should not be trusted is still evidence."""
    return replace(anchor, void=True, void_reason=reason)


# --- fingerprint construction -------------------------------------------

def _digest(text) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def profile_hash(profile) -> str:
    """Stable digest of the launch-relevant parts of a profile.

    Only fields that change what the runtime does: renaming a profile or
    editing its description must not stale a battery, and changing its
    context or its cache budget must.
    """
    if not profile:
        return ""
    config = profile.get("config", {}) or {}
    relevant = {
        "model_path": profile.get("model_path", ""),
        "backend": profile.get("backend", ""),
        "config": {k: config.get(k) for k in sorted(config)
                   if k not in ("description", "label")},
        "moe_cache": profile.get("moe_cache", {}) or {},
    }
    return _digest(json.dumps(relevant, sort_keys=True, separators=(",", ":"),
                              default=str))


def env_hash(env=None) -> str:
    """Digest of FINGERPRINTED_ENV as seen in `env` (default os.environ).

    An unset variable is hashed as unset rather than as its default: the
    default is a property of the binary, which build_commit already
    covers, and conflating them would hide a build whose default moved.
    """
    source = os.environ if env is None else env
    items = [(k, source.get(k)) for k in FINGERPRINTED_ENV]
    return _digest(json.dumps(items, sort_keys=True, separators=(",", ":")))


_DRIVER_CACHE = None


def driver_identity(runner=subprocess.run) -> str:
    """The GPU compute-runtime version, or "" when it cannot be read.

    "" propagates into the fingerprint as unknown, which stales rather
    than matches -- an anchor that cannot prove its driver is the same
    driver has to be re-taken.
    """
    global _DRIVER_CACHE
    if _DRIVER_CACHE is not None:
        return _DRIVER_CACHE
    version = ""
    try:
        p = runner(["rpm", "-q", "--queryformat", "%{VERSION}-%{RELEASE}",
                    "intel-compute-runtime"],
                   capture_output=True, text=True, timeout=10)
        if p.returncode == 0 and p.stdout.strip():
            version = "intel-compute-runtime-" + p.stdout.strip()
    except Exception:
        version = ""
    _DRIVER_CACHE = version
    return version


def current_fingerprint(build_commit="", profile=None, env=None,
                        driver=None) -> Fingerprint:
    return Fingerprint(
        build_commit=build_commit or "",
        profile_hash=profile_hash(profile),
        env_hash=env_hash(env),
        driver=driver_identity() if driver is None else driver)


# --- staleness and battery planning -------------------------------------

def staleness(anchor, fingerprint) -> list:
    """Why this anchor cannot be reused as-is. Empty means reusable.

    Order matters: `void` is checked first, because a bad measurement is
    a reason to re-run regardless of whether the conditions still match,
    and reporting "conditions unchanged" about a voided number would be
    true and useless.
    """
    reasons = []
    if anchor.void:
        reasons.append("voided: " + (anchor.void_reason or "no reason recorded"))
    if anchor.always_run:
        reasons.append(
            "exempt from fingerprint reuse: this anchor always runs")
    if anchor.value is None and not anchor.runs:
        reasons.append("no value has ever been recorded for this anchor")
    reasons.extend(anchor.fingerprint.differences(fingerprint))
    return reasons


def needs_run(anchor, fingerprint) -> bool:
    return bool(staleness(anchor, fingerprint))


@dataclass
class BatteryPlan:
    """What a battery would do, before it does it."""
    to_run: list = field(default_factory=list)     # (anchor, reasons)
    reusable: list = field(default_factory=list)   # anchors kept as-is

    def to_dict(self):
        return {
            "to_run": [{"id": a.id, "condition": a.condition,
                        "reasons": r} for a, r in self.to_run],
            "reusable": [{"id": a.id, "condition": a.condition,
                          "value": a.value, "unit": a.unit,
                          "recorded": a.recorded} for a in self.reusable],
            "runs": len(self.to_run),
            "reused": len(self.reusable),
        }


def plan_battery(anchors, fingerprint) -> BatteryPlan:
    """Split a battery into what must be measured and what can be reused."""
    plan = BatteryPlan()
    for a in anchors or []:
        reasons = staleness(a, fingerprint)
        if reasons:
            plan.to_run.append((a, reasons))
        else:
            plan.reusable.append(a)
    return plan


def record(anchor, value, runs=(), fingerprint=None, recorded="", source="",
           load=None, extra=None) -> Anchor:
    """Replace an anchor's value with a fresh measurement.

    Clears `void`: the whole point of re-running a voided anchor is that
    the new number is not the old one. `always_run` survives -- it is a
    property of the anchor's role, not of any measurement.
    """
    return replace(
        anchor,
        value=value,
        runs=tuple(runs),
        fingerprint=fingerprint if fingerprint is not None else anchor.fingerprint,
        recorded=recorded or anchor.recorded,
        source=source or anchor.source,
        load=dict(load or {}),
        extra=dict(extra if extra is not None else anchor.extra),
        void=False,
        void_reason="",
    )
