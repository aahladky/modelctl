"""Pre-registered night-lane jobs.

A pre-registration is a comparison declared -- with its criterion --
BEFORE it runs. That ordering is the whole value: once numbers exist,
the temptation is to pick the reading that flatters the change, and a
criterion written afterwards is not a criterion. So this registry lives
in the repo (version-controlled, diffable) rather than in the mutable
state directory, and the criterion field is written in the same commit
that adds the job.

Everything here is `enabled: false`. Pre-registering is not scheduling:
these describe runs that are ready to be started deliberately, by a
person, on a quiet machine. Nothing in modelctl reads this registry and
launches anything -- `enabled_jobs()` exists so a future scheduler has a
single gate to consult, and today it returns nothing.

The registry deliberately records no expected outcome and no verdict.
Per the project rule, benchmarks record raw numbers, exact config, and
concurrent machine load; the reading is Aaron's.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "night-lane.json"


@dataclass(frozen=True)
class Arm:
    """One side of a comparison.

    `profile` names the modelctl profile; `overrides` are the config
    deltas that make this arm what it is. An arm that needs a fleet node
    says so in `requires_nodes` -- the runner must refuse rather than
    silently measure a local fallback and file it as a remote result.
    """
    name: str
    profile: str
    overrides: dict = field(default_factory=dict)
    requires_nodes: tuple = ()
    note: str = ""

    def to_dict(self):
        return {"name": self.name, "profile": self.profile,
                "overrides": self.overrides,
                "requires_nodes": list(self.requires_nodes),
                "note": self.note}

    @staticmethod
    def from_dict(d):
        return Arm(name=d["name"], profile=d["profile"],
                   overrides=d.get("overrides", {}),
                   requires_nodes=tuple(d.get("requires_nodes", [])),
                   note=d.get("note", ""))


@dataclass(frozen=True)
class NightLaneJob:
    id: str
    title: str
    question: str        # what this run is asked to answer
    criterion: str       # the decision rule, fixed before any numbers
    measures: tuple      # exactly which quantities get recorded
    arms: tuple
    enabled: bool = False
    registered: str = ""  # ISO date the pre-registration was committed
    note: str = ""

    def to_dict(self):
        return {"id": self.id, "title": self.title, "question": self.question,
                "criterion": self.criterion, "measures": list(self.measures),
                "arms": [a.to_dict() for a in self.arms],
                "enabled": self.enabled, "registered": self.registered,
                "note": self.note}

    @staticmethod
    def from_dict(d):
        return NightLaneJob(
            id=d["id"], title=d["title"], question=d["question"],
            criterion=d["criterion"], measures=tuple(d.get("measures", [])),
            arms=tuple(Arm.from_dict(a) for a in d.get("arms", [])),
            enabled=bool(d.get("enabled", False)),
            registered=d.get("registered", ""), note=d.get("note", ""))

    @property
    def required_nodes(self) -> set:
        out = set()
        for a in self.arms:
            out.update(a.requires_nodes)
        return out


def load_jobs(path=None) -> list:
    p = path or REGISTRY_PATH
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    return [NightLaneJob.from_dict(j) for j in raw.get("jobs", [])]


def save_jobs(jobs, path=None):
    p = path or REGISTRY_PATH
    payload = {"version": 1, "jobs": [j.to_dict() for j in jobs]}
    p.write_text(json.dumps(payload, indent=2) + "\n")


def enabled_jobs(path=None) -> list:
    """The gate a scheduler consults. Empty by design, today."""
    return [j for j in load_jobs(path) if j.enabled]


def job_by_id(job_id, path=None):
    for j in load_jobs(path):
        if j.id == job_id:
            return j
    return None


def blocking_reasons(job, usable_node_names=()) -> list:
    """Why this job cannot run right now.

    A pre-registered job that needs ph16-71 must not quietly run without
    it: the local-fallback plan is byte-identical to a fleet-free launch,
    so the run would produce a perfectly valid number for the wrong
    question.
    """
    reasons = []
    if not job.enabled:
        reasons.append("pre-registered but not enabled")
    missing = sorted(job.required_nodes - set(usable_node_names))
    if missing:
        reasons.append(f"fleet nodes not usable: {', '.join(missing)}")
    return reasons
