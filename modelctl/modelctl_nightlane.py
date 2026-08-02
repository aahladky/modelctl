"""Pre-registered night-lane jobs, and the lane that runs them.

A pre-registration is a comparison declared -- with its criterion --
BEFORE it runs. That ordering is the whole value: once numbers exist,
the temptation is to pick the reading that flatters the change, and a
criterion written afterwards is not a criterion. So this registry lives
in the repo (version-controlled, diffable) rather than in the mutable
state directory, and the criterion field is written in the same commit
that adds the job.

`enabled` is the gate, and it means "queued", not "scheduled". An
enabled job is one a person has released to run the next time the
machine is quiet; a disabled one is a declaration only. Enabling is a
diff, which is the point -- nothing here starts running because a
default changed.

The lane half of this module (`window_state`, `dispatch_due`) is what
turns a queued job into a running one, and it refuses far more often
than it dispatches:

  * The window is gated on llama-swap holding no models AND the load
    being below a ceiling. A benchmark taken beside a live model is a
    measurement of the two of them together, and the whole reason the
    2026-08-01 battery is void is that nobody checked.
  * A job whose arms need a fleet node the machine cannot reach is
    blocked, never silently run local -- the local-fallback plan is
    byte-identical to a fleet-free launch, so it would answer a
    different question with a perfectly valid number.
  * Dispatch is explicit. Importing this module starts nothing; a
    caller has to hand it a job manager and ask.

Every dispatched run carries its own load trace and files its evidence
under docs/evidence/ with a one-line summary, because a night run has no
witness and the record is all there will be in the morning.

The registry deliberately records no expected outcome and no verdict.
Per the project rule, benchmarks record raw numbers, exact config, and
concurrent machine load; the reading is Aaron's.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "night-lane.json"
EVIDENCE_DIR = Path(__file__).resolve().parent / "docs" / "evidence"
SUMMARY_PATH = EVIDENCE_DIR / "night-lane-log.md"

# The lane runs on the benchmark lane of the existing job store: one
# worker, so two night jobs can never contend for the GPUs, and the
# console's jobs page renders it already.
LANE = "benchmark"

# Above this 1-minute load average the machine is not quiet enough to
# measure on. The rig idles around 0.1-1.0; the void battery ran at a
# mean of 8.99. The ceiling sits above idle and far below that.
DEFAULT_LOADAVG_CEILING = 1.5


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


# How a job's arms are measured. The mode is part of the
# pre-registration because it decides what the numbers can support: a
# `paired` job's answer is a set of within-pair deltas, a `block` job's
# is two means, and swapping one for the other after the fact is exactly
# the substitution that voided the determinism cost figures.
MODES = ("paired", "block", "battery", "reproducibility")


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
    mode: str = "block"
    # Pairs for `paired`, runs per arm for everything else. Fixed here so
    # the sample size is part of the pre-registration and cannot be
    # extended until the numbers come out the way someone wanted.
    pairs: int = 0
    runs: int = 0
    metric: str = "generation_tps"

    def to_dict(self):
        out = {"id": self.id, "title": self.title, "question": self.question,
               "criterion": self.criterion, "measures": list(self.measures),
               "arms": [a.to_dict() for a in self.arms],
               "enabled": self.enabled, "registered": self.registered,
               "note": self.note}
        # Fields added after the first pre-registrations were committed are
        # emitted only when they carry something. A registry entry is a
        # record of what someone declared; re-serializing an untouched job
        # with a new field defaulted in makes the diff claim an edit that
        # nobody made.
        for key, value, default in (("mode", self.mode, "block"),
                                    ("pairs", self.pairs, 0),
                                    ("runs", self.runs, 0),
                                    ("metric", self.metric, "generation_tps")):
            if value != default:
                out[key] = value
        return out

    @staticmethod
    def from_dict(d):
        return NightLaneJob(
            id=d["id"], title=d["title"], question=d["question"],
            criterion=d["criterion"], measures=tuple(d.get("measures", [])),
            arms=tuple(Arm.from_dict(a) for a in d.get("arms", [])),
            enabled=bool(d.get("enabled", False)),
            registered=d.get("registered", ""), note=d.get("note", ""),
            mode=d.get("mode", "block"), pairs=int(d.get("pairs", 0)),
            runs=int(d.get("runs", 0)),
            metric=d.get("metric", "generation_tps"))

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


# --- the window ----------------------------------------------------------

@dataclass(frozen=True)
class WindowState:
    """Is the machine quiet enough to measure on, and how do we know?

    `open` is never returned on its own: `reasons` says what closed it
    and `observed` records the readings the decision was made from, so a
    night that dispatched nothing can be explained in the morning
    without re-deriving the machine's state from logs.
    """
    open: bool
    reasons: tuple = ()
    observed: dict = field(default_factory=dict)

    def to_dict(self):
        return {"open": self.open, "reasons": list(self.reasons),
                "observed": dict(self.observed)}


def window_state(swap_client=None, load_sample=None,
                 loadavg_ceiling=DEFAULT_LOADAVG_CEILING) -> WindowState:
    """The gate: llama-swap holding nothing AND the load below the ceiling.

    Both halves fail closed. An unreachable llama-swap is not an idle
    one -- it might be mid-restart with a model about to land -- and an
    unreadable loadavg is not a quiet machine. In either case the window
    stays shut and says which reading it could not take.
    """
    reasons = []
    observed = {}

    if swap_client is None:
        # Built from MODELCTL_LLAMA_SWAP_BASE_URL, not from the client's
        # own hardcoded default: that env var is one of the five
        # redirections a scratch walk sets, and a lane that ignored it
        # would poll the live service while claiming to be scratch-safe.
        # Same /v1/-stripping the web app does -- /running sits at the
        # service root, not under /v1/.
        from modelctl_web.app import LLAMA_SWAP_ROOT
        from modelctl_web.swap import LlamaSwapClient
        swap_client = LlamaSwapClient(base_url=LLAMA_SWAP_ROOT)
    try:
        running = sorted(m for m in swap_client.running_model_ids() if m)
        observed["llama_swap_running"] = running
        if running:
            reasons.append(
                "llama-swap is holding " + ", ".join(running))
    except Exception as e:
        observed["llama_swap_error"] = f"{type(e).__name__}: {e}"
        reasons.append(
            "llama-swap could not be reached, so it cannot be shown idle")

    if load_sample is None:
        import modelctl_load
        load_sample = modelctl_load.sample_load()
    loadavg = getattr(load_sample, "loadavg_1m", None)
    observed["loadavg_1m"] = loadavg
    observed["loadavg_ceiling"] = loadavg_ceiling
    if loadavg is None:
        reasons.append("loadavg could not be read, so it cannot be shown low")
    elif loadavg > loadavg_ceiling:
        reasons.append(
            f"loadavg(1m) {loadavg:.2f} is above the {loadavg_ceiling:.2f} "
            f"ceiling")

    return WindowState(open=not reasons, reasons=tuple(reasons),
                       observed=observed)


# --- dispatch ------------------------------------------------------------

@dataclass
class Dispatch:
    """What one pass over the lane did, and what it declined to do."""
    window: WindowState
    submitted: list = field(default_factory=list)   # (job_id, store_job_id)
    skipped: list = field(default_factory=list)     # (job_id, reasons)

    def to_dict(self):
        return {"window": self.window.to_dict(),
                "submitted": [{"job": j, "store_job": s}
                              for j, s in self.submitted],
                "skipped": [{"job": j, "reasons": r} for j, r in self.skipped]}


def due_jobs(jobs=None, usable_node_names=(), path=None):
    """(runnable, skipped) -- enabled jobs split by whether they can run.

    Returns skipped jobs with their reasons rather than filtering them
    away: a night lane that quietly runs three of five pre-registrations
    and reports three results looks like it ran everything.
    """
    runnable, skipped = [], []
    for job in (jobs if jobs is not None else load_jobs(path)):
        reasons = blocking_reasons(job, usable_node_names)
        if reasons:
            skipped.append((job, reasons))
        else:
            runnable.append(job)
    return runnable, skipped


def dispatch_due(manager, runner, jobs=None, usable_node_names=(),
                 window=None, path=None, limit=None) -> Dispatch:
    """Submit every due job to the job store's benchmark lane.

    `runner(job, ctx) -> dict` performs one job; it is injected so this
    module owns the gate and the bookkeeping and nothing else. The
    benchmark lane has a single worker, so submitting several at once
    still runs them one at a time -- which is required, not incidental:
    two benchmarks sharing the GPUs measure each other.

    Submits nothing at all when the window is shut. A partially-open
    window is not a thing: the machine either was quiet for the run or
    the run does not count.
    """
    state = window if window is not None else window_state()
    result = Dispatch(window=state)
    runnable, skipped = due_jobs(jobs, usable_node_names, path)
    result.skipped = [(j.id, r) for j, r in skipped]

    if not state.open:
        for job in runnable:
            result.skipped.append(
                (job.id, ["the measurement window is shut: "
                          + "; ".join(state.reasons)]))
        return result

    for job in runnable[:limit] if limit else runnable:
        store_job_id = manager.submit(
            "nightlane", job.title,
            lambda ctx, _job=job: runner(_job, ctx),
            payload={"night_lane_job": job.id, "mode": job.mode,
                     "window": state.to_dict()},
            lane=LANE)
        result.submitted.append((job.id, store_job_id))
    if limit and len(runnable) > limit:
        for job in runnable[limit:]:
            result.skipped.append(
                (job.id, [f"not dispatched this pass: limit of {limit} "
                          f"submissions was reached"]))
    return result


# --- arms into runnable profiles -----------------------------------------

# The global op-offload floor. Below 32 there is a known correctness bug
# on this hardware, so no arm may set it lower -- checked here rather than
# trusted to review because the night lane runs with nobody watching, and
# a floor violation at 03:00 produces numbers that look fine.
OFFLOAD_MIN_BATCH_FLOOR = 32


def arm_violations(arm) -> list:
    """Hard rules an arm's overrides break. Empty means it may run."""
    out = []
    env = (arm.overrides or {}).get("env", {}) or {}
    raw = env.get("GGML_OP_OFFLOAD_MIN_BATCH")
    if raw is not None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            out.append(
                f"GGML_OP_OFFLOAD_MIN_BATCH is {raw!r}, which is not a number")
        else:
            if value < OFFLOAD_MIN_BATCH_FLOOR:
                out.append(
                    f"GGML_OP_OFFLOAD_MIN_BATCH={value} is below the "
                    f"{OFFLOAD_MIN_BATCH_FLOOR} floor: known correctness bug "
                    f"on this hardware")
    return out


def arm_profile(arm, profile):
    """The profile this arm actually runs, with its overrides applied.

    Built in memory and never saved. Two arms of a comparison have to
    differ by exactly what the registry says they differ by, and routing
    that through the saved profile store would both mutate the user's
    profiles and let an edit between arms change the answer.
    """
    if arm_violations(arm):
        raise ValueError("; ".join(arm_violations(arm)))
    import copy
    out = copy.deepcopy(profile)
    overrides = dict(arm.overrides or {})
    env = overrides.pop("env", None)
    if env:
        existing = {e.split("=", 1)[0]: e for e in (out.get("env") or [])
                    if "=" in e}
        for key, value in env.items():
            existing[key] = f"{key}={value}"
        out["env"] = [existing[k] for k in sorted(existing)]
    config = dict(out.get("config") or {})
    for key, value in overrides.items():
        if key.startswith("_"):          # runner directives, not config
            continue
        config[key] = value
    out["config"] = config
    return out


# --- evidence ------------------------------------------------------------

def evidence_path(job_id, date, directory=None) -> Path:
    return Path(directory or EVIDENCE_DIR) / f"{date}-nightlane-{job_id}.json"


def file_evidence(job, record, date, directory=None,
                  summary_path=None) -> Path:
    """Write one run's full record, and append its one-line summary.

    The full record is JSON because it is machine-written and nobody
    will be awake to read it; the summary line exists so the morning
    question ("did anything run?") is answered by one file rather than
    by opening every record.
    """
    directory = Path(directory or EVIDENCE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = evidence_path(job.id, date, directory)
    payload = {
        "night_lane_job": job.to_dict(),
        "filed": date,
        "record": record,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    append_summary(summary_line(job, record, date, path),
                   summary_path or (directory / SUMMARY_PATH.name))
    return path


def summary_line(job, record, date, path=None) -> str:
    """One line: what ran, under what load, and where the numbers are.

    Deliberately carries no reading of the result. The line says the run
    happened and where to look; anything more would be this module
    judging, which it does not do.
    """
    load = (record or {}).get("load_summary") or {}
    stat = load.get("loadavg_1m") if isinstance(load, dict) else None
    if isinstance(stat, dict) and stat.get("mean") is not None:
        load_text = (f"loadavg(1m) {stat['min']:.2f}-{stat['max']:.2f} "
                     f"mean {stat['mean']:.2f}")
    else:
        load_text = "load not recorded"
    where = f" — {Path(path).name}" if path else ""
    status = (record or {}).get("status", "recorded")
    return (f"- {date} `{job.id}` ({job.mode}) {status}; {load_text}{where}")


def append_summary(line, summary_path=None):
    p = Path(summary_path or SUMMARY_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(
            "# Night-lane run log\n\n"
            "One line per dispatched run. Numbers live in the per-run JSON "
            "beside this file; nothing here is a reading of a result.\n\n")
    with p.open("a") as fh:
        fh.write(line.rstrip("\n") + "\n")


def today(clock=time.time) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(clock()))


# --- running a job -------------------------------------------------------

# How long a night job waits for the GPU lock before refusing. The
# window gate already asked for a quiet machine, so a held lock means
# somebody's bench is running right now: a minute covers the gap between
# a lane releasing the lock and this job asking, and refusing after that
# is correct -- the job is queued, not scheduled, and tomorrow night is
# fine. Waiting instead would put a benchmark inside a window that was
# checked an hour earlier.
GPU_LOCK_WAIT_SECONDS = 60.0


def run_job(job, measure, ctx=None, load_interval=5.0, clock=time.time,
            gpu_lock=None, gpu_lock_wait=GPU_LOCK_WAIT_SECONDS):
    """Execute one pre-registered job and return its record.

    `measure(arm, index, slot) -> dict` performs one run of one arm. It is
    injected because launching a server is not this module's business and
    because a night lane whose orchestration can only be tested with real
    GPUs does not get tested.

    Every run happens holding the machine-wide GPU lock
    (modelctl_lanes.gpu_lock), which is the same lock `modelctl lane
    gpu-lock` takes. The single-worker benchmark lane already stops two
    night jobs from overlapping; the lock is what stops a night job and
    a session's bench in a parallel lane from measuring each other, and
    those two schedulers know nothing about one another.

    The record it returns is what `file_evidence` writes. It carries the
    job's own criterion verbatim: a result read months later without the
    rule it was judged by is a number looking for a story.
    """
    record = {
        "job_id": job.id, "mode": job.mode, "metric": job.metric,
        "criterion": job.criterion, "started": clock(), "status": "recorded",
    }
    violations = [v for a in job.arms for v in arm_violations(a)]
    if violations:
        record["status"] = "refused"
        record["violations"] = violations
        record["finished"] = clock()
        return record

    if job.mode == "paired" and len(job.arms) != 2:
        # Refused before the lock: a job that cannot run should not make
        # anything else wait to find that out.
        record["status"] = "refused"
        record["violations"] = [
            f"paired mode needs exactly two arms, this job has "
            f"{len(job.arms)}"]
        record["finished"] = clock()
        return record

    if gpu_lock is None:
        import modelctl_lanes
        gpu_lock = modelctl_lanes.gpu_lock
    try:
        with gpu_lock(timeout=gpu_lock_wait, note=f"night lane {job.id}"):
            return _run_measurements(job, measure, record, ctx, load_interval,
                                     clock)
    except Exception as e:
        # LockBusy is the only expected one, and it is a refusal rather
        # than an error: nothing ran, so there is nothing to file but the
        # reason. Caught by name rather than by class to keep this module
        # a leaf of modelctl_lanes' exception types.
        if type(e).__name__ != "LockBusy":
            raise
        record["status"] = "refused"
        record["violations"] = [
            f"the GPU lock is held by another run ({e}); nothing was measured"]
        record["finished"] = clock()
        return record


def _run_measurements(job, measure, record, ctx, load_interval, clock):
    if job.mode == "paired":
        import modelctl_paired
        a, b = (modelctl_paired.Condition(
            name=arm.name, label=arm.note,
            config=dict(arm.overrides or {})) for arm in job.arms)
        by_name = {arm.name: arm for arm in job.arms}
        comparison = modelctl_paired.run_paired(
            a, b,
            lambda condition, pair, slot: measure(
                by_name[condition.name], pair, slot),
            pairs=job.pairs, metric=job.metric,
            load_interval=load_interval,
            should_stop=(ctx.is_cancelled if ctx is not None else None),
            clock=clock)
        record["comparison"] = comparison.to_dict()
        record["load_summary"] = _merge_load(
            r["load"] for r in record["comparison"]["runs"])
    else:
        import modelctl_load
        arms = {}
        for arm in job.arms:
            runs = []
            for index in range(job.runs):
                if ctx is not None and ctx.is_cancelled():
                    record.setdefault("notes", []).append(
                        f"cancelled during {arm.name} run {index}")
                    break
                recorder = modelctl_load.LoadRecorder(interval=load_interval)
                entry = {"run": index, "started": clock()}
                recorder.start()
                try:
                    entry["metrics"] = measure(arm, index, 0) or {}
                except Exception as e:
                    entry["error"] = f"{type(e).__name__}: {e}"
                finally:
                    entry["load"] = recorder.stop().summary()
                    entry["finished"] = clock()
                runs.append(entry)
            arms[arm.name] = {"note": arm.note,
                              "overrides": dict(arm.overrides or {}),
                              "runs": runs}
        record["arms"] = arms
        record["load_summary"] = _merge_load(
            r["load"] for arm in arms.values() for r in arm["runs"])

    record["finished"] = clock()
    return record


def _merge_load(summaries):
    """Job-wide load, folded from the per-run summaries.

    The per-run traces stay in the record; this is only for the one-line
    summary. It is explicitly a fold of readings that were taken per run,
    not a single battery-wide measurement -- which is the distinction the
    2026-08-01 battery did not make.
    """
    mins, maxes, means, samples = [], [], [], 0
    for summary in summaries:
        stat = (summary or {}).get("loadavg_1m")
        if not isinstance(stat, dict):
            continue
        mins.append(stat["min"])
        maxes.append(stat["max"])
        means.append(stat["mean"])
        samples += stat.get("n", 0)
    if not means:
        return {"note": "no run carried a readable load trace"}
    return {"loadavg_1m": {"min": min(mins), "max": max(maxes),
                           "mean": sum(means) / len(means), "n": samples},
            "note": "folded from per-run traces, not a battery-wide sample"}


def default_measure(profile_name, plan_id, job):
    """A `measure` built on modelctl's own fresh-server launch path.

    `test_launch_plan` takes a profile_override, which is what lets each
    arm run its own configuration without saving a throwaway profile --
    the same mechanism the hardware acceptance matrix uses.
    """
    import modelctl
    base = modelctl.load_profile(profile_name)

    def measure(arm, index, slot):
        import modelctl_tune
        run = modelctl_tune.test_launch_plan(
            profile_name, plan_id, profile_override=arm_profile(arm, base),
            max_tokens=128, runs=1, warmup_tokens=32)
        if not run.get("success"):
            raise RuntimeError(
                f"{arm.name} run {index} failed: {run.get('failure_class')}")
        return run
    return measure
