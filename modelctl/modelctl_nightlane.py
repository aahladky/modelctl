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

The lane half of this module (`observe`, `cleanup_pass`, `dispatch_due`)
turns a queued job into a running one. It does not decide whether the
machine deserves to be measured:

  * **A benchmark is never gated on conditions.** There was a loadavg
    ceiling here, and it skipped jobs -- including when the loadavg
    could not be read at all, on the reasoning that an unread load
    "cannot be shown low". Fail-closed both ways, on a rig that runs
    lanes all day: the 2026-08-02 night pair nearly did not happen, and
    a comparison that does not run answers nothing. So conditions are
    now *recorded*, per run, as a load trace. Recording is what lets a
    paired design survive a noisy machine; refusing only produces
    silence.
  * **Make room, then run.** Every job is preceded by a cleanup pass
    that sweeps orphaned lane scratch and reaps llama-servers no living
    launcher owns, and records MemAvailable and loadavg either side of
    it. Cleanup frees junk -- never the page cache, which is part of
    the measurement.
  * **One form of waiting, and it is the GPU lock.** Two benchmarks on
    one GPU is garbage rather than noise, so a job waits for the lock;
    it does not skip. A wait that runs out is reported as a failure
    naming the holder, never as a quiet skip.
  * A job whose arms need a fleet node the machine cannot reach is
    blocked, never silently run local -- the local-fallback plan is
    byte-identical to a fleet-free launch, so it would answer a
    different question with a perfectly valid number. That is a
    statement about the question, not about the machine's mood.
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
import os
import signal
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


# --- conditions ----------------------------------------------------------

@dataclass(frozen=True)
class Conditions:
    """What the machine looked like. Readings only, no verdict.

    This is what is left of `window_state` after the gate came out. The
    same readings are taken; none of them decides anything. A field that
    is None was not readable, and `unreadable` says which and why --
    never substituted with a zero, and never turned into a refusal,
    because "the loadavg could not be read" is a fact about /proc and
    the old lane treated it as a fact about the machine being busy.
    """
    at: float = 0.0
    loadavg_1m: float | None = None
    loadavg_5m: float | None = None
    mem_available_bytes: int | None = None
    llama_swap_running: tuple = ()
    unreadable: tuple = ()

    def to_dict(self):
        return {"at": self.at, "loadavg_1m": self.loadavg_1m,
                "loadavg_5m": self.loadavg_5m,
                "mem_available_bytes": self.mem_available_bytes,
                "llama_swap_running": list(self.llama_swap_running),
                "unreadable": list(self.unreadable)}


def observe(swap_client=None, load_sample=None, clock=time.time) -> Conditions:
    """Take every reading the old window took, and judge none of them."""
    unreadable = []

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
        running = tuple(sorted(m for m in swap_client.running_model_ids() if m))
    except Exception as e:
        running = ()
        unreadable.append(f"llama-swap: {type(e).__name__}: {e}")

    if load_sample is None:
        import modelctl_load
        load_sample = modelctl_load.sample_load()
    loadavg = getattr(load_sample, "loadavg_1m", None)
    if loadavg is None:
        unreadable.append("loadavg: /proc/loadavg could not be read")
    mem = getattr(load_sample, "mem_available_bytes", None)
    if mem is None:
        unreadable.append("mem_available: /proc/meminfo could not be read")

    return Conditions(at=clock(), loadavg_1m=loadavg,
                      loadavg_5m=getattr(load_sample, "loadavg_5m", None),
                      mem_available_bytes=mem, llama_swap_running=running,
                      unreadable=tuple(unreadable))


# --- the cleanup pass ----------------------------------------------------
# What the machine hoards between night runs: lane build trees nothing
# deleted, and llama-servers whose launcher died mid-bench. Both are
# junk, and junk is the only thing a cleanup pass is allowed to free --
# never the page cache, which on the ornith pair is part of the
# measurement rather than something in its way.

SERVER_PROCESS_NAMES = ("llama-server",)


def _proc_info(pid, proc_root="/proc"):
    """name/ppid/start time/cgroup for one pid, or None if it is gone."""
    base = os.path.join(str(proc_root), str(pid))
    try:
        with open(os.path.join(base, "stat")) as fh:
            stat = fh.read()
    except OSError:
        return None
    # comm is parenthesised and may itself contain spaces and brackets,
    # so everything positional is read after the LAST ')'.
    start, end = stat.find("("), stat.rfind(")")
    if start < 0 or end < start:
        return None
    fields = stat[end + 2:].split()
    try:
        ppid = int(fields[1])
        starttime = fields[19]
    except (IndexError, ValueError):
        return None
    try:
        with open(os.path.join(base, "cgroup")) as fh:
            cgroup = fh.read().strip()
    except OSError:
        cgroup = ""
    return {"pid": int(pid), "comm": stat[start + 1:end], "ppid": ppid,
            "starttime": starttime, "cgroup": cgroup}


def orphaned_servers(proc_root="/proc", names=SERVER_PROCESS_NAMES):
    """llama-server processes that no living launcher owns.

    Two conditions, both required, both read rather than assumed:

      * the parent is PID 1. llama-swap holds its servers as children
        and so does a lane's bench, so a reparented server is one whose
        launcher has died;
      * its own cgroup is not a systemd service. "ppid is 1" is also
        true of every process systemd itself started, and that check
        alone would let a 03:00 cleanup pass kill the serving stack.

    Never a live one is the entire requirement: this runs unattended,
    and the cost of one false positive is the rig.
    """
    out = []
    try:
        pids = sorted(int(n) for n in os.listdir(proc_root) if n.isdigit())
    except OSError:
        return out
    for pid in pids:
        info = _proc_info(pid, proc_root)
        if info is None or info["comm"] not in names:
            continue
        if info["ppid"] != 1:
            continue
        if service_unit(info["cgroup"]):
            continue
        out.append(info)
    return out


def service_unit(cgroup):
    """The systemd service unit owning a cgroup, or "" if none does.

    The LEAF path component decides it, never a substring: every user
    process on this machine sits under `user@1000.service`, so `".service"
    in cgroup` is true of all of them and a reaper using it would spare
    everything, forever, while looking like it worked.
    """
    for line in (cgroup or "").splitlines():
        leaf = line.rpartition(":")[2].rstrip("/").rsplit("/", 1)[-1]
        if leaf.endswith(".service"):
            return leaf
    return ""


def _same_process(info, proc_root="/proc"):
    """Is this pid still the process we decided about?"""
    now = _proc_info(info["pid"], proc_root)
    return bool(now and all(now[k] == info[k]
                            for k in ("comm", "ppid", "starttime")))


def reap_orphaned_servers(proc_root="/proc", killer=os.kill, wait=15.0,
                          poll=0.5, clock=time.monotonic, sleep=time.sleep):
    """Terminate the orphans; return one record per process signalled.

    Every signal is preceded by re-reading the identity the decision was
    made from (name, parent, start time). PID reuse is not theoretical
    on a machine that spawns servers all night, and the window between
    listing a pid and signalling it is exactly where a reused one would
    land.
    """
    reaped = []
    for info in orphaned_servers(proc_root):
        if not _same_process(info, proc_root):
            continue
        entry = {"pid": info["pid"], "comm": info["comm"],
                 "cgroup": info["cgroup"]}
        try:
            killer(info["pid"], signal.SIGTERM)
        except OSError as e:
            reaped.append({**entry, "outcome": f"SIGTERM failed: {e}"})
            continue
        deadline = clock() + wait
        while _same_process(info, proc_root) and clock() < deadline:
            sleep(poll)
        entry["outcome"] = "SIGTERM"
        if _same_process(info, proc_root):
            try:
                killer(info["pid"], signal.SIGKILL)
                entry["outcome"] = f"SIGKILL after {wait:.0f}s"
            except OSError as e:
                entry["outcome"] = f"SIGKILL failed: {e}"
        reaped.append(entry)
    return reaped


def _sweep_lane_scratch():
    import modelctl_lanes
    return modelctl_lanes.sweep_scratch_orphans()


def cleanup_pass(observer=None, sweep=None, reap=None) -> dict:
    """Free what the machine is hoarding, and record both sides of it.

    This is what stands where the loadavg ceiling used to. The ceiling
    read a busy machine and refused; this makes the machine less busy
    and then measures anyway, which is the same information put to use
    instead of to a veto.

    Nothing in here can stop a job. A cleanup that fails records why and
    the run continues -- a new refusal path smuggled in as error
    handling would be the same bug wearing a different hat.
    """
    observer = observer or observe
    sweep = sweep or _sweep_lane_scratch
    reap = reap or reap_orphaned_servers

    before = observer()
    result = {"before": before.to_dict(), "scratch_removed": [],
              "processes_reaped": [], "errors": []}
    for key, action in (("scratch_removed", sweep),
                        ("processes_reaped", reap)):
        try:
            result[key] = list(action() or [])
        except Exception as e:
            result["errors"].append(f"{key}: {type(e).__name__}: {e}")
    after = observer()
    result["after"] = after.to_dict()
    result["scratch_bytes"] = sum(int(d.get("bytes") or 0)
                                  for d in result["scratch_removed"])
    if before.mem_available_bytes is None or after.mem_available_bytes is None:
        result["mem_available_delta_bytes"] = None
    else:
        result["mem_available_delta_bytes"] = (after.mem_available_bytes
                                               - before.mem_available_bytes)
    return result


# --- dispatch ------------------------------------------------------------

@dataclass
class Dispatch:
    """What one pass over the lane did, and what it declined to do."""
    conditions: Conditions = field(default_factory=Conditions)
    submitted: list = field(default_factory=list)   # (job_id, store_job_id)
    skipped: list = field(default_factory=list)     # (job_id, reasons)

    def to_dict(self):
        return {"conditions": self.conditions.to_dict(),
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
                 conditions=None, path=None, limit=None) -> Dispatch:
    """Submit every due job to the job store's benchmark lane.

    `runner(job, ctx) -> dict` performs one job; it is injected so this
    module owns the bookkeeping and nothing else. The benchmark lane has
    a single worker, so submitting several at once still runs them one
    at a time -- which is required, not incidental: two benchmarks
    sharing the GPUs measure each other.

    The machine's condition is read once here and carried in the job's
    payload, because knowing what the queue looked like at 03:00 is
    worth having in the morning. It is not consulted: nothing about the
    readings can stop a submission.
    """
    state = conditions if conditions is not None else observe()
    result = Dispatch(conditions=state)
    runnable, skipped = due_jobs(jobs, usable_node_names, path)
    result.skipped = [(j.id, r) for j, r in skipped]

    for job in runnable[:limit] if limit else runnable:
        store_job_id = manager.submit(
            "nightlane", job.title,
            lambda ctx, _job=job: runner(_job, ctx),
            payload={"night_lane_job": job.id, "mode": job.mode,
                     "conditions": state.to_dict()},
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

# How long a night job waits for the GPU lock. It waits; it does not
# refuse. This is the one thing the lane still blocks on, because two
# benchmarks on one GPU produce garbage rather than noise -- unlike a
# busy machine, which produces a real number under a recorded load. A
# night's worth of waiting outlasts any bench a session might be
# running, and a wait that does run out is a failure with the holder
# named, not a skip: something is holding the lock that should not be,
# and that is worth waking up to.
GPU_LOCK_WAIT_SECONDS = 6 * 3600.0


def run_job(job, measure, ctx=None, load_interval=5.0, clock=time.time,
            gpu_lock=None, gpu_lock_wait=GPU_LOCK_WAIT_SECONDS,
            cleanup=None):
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

    Inside the lock and before the first measurement, `cleanup` frees
    what the machine is hoarding and records MemAvailable and loadavg on
    both sides of it. Inside, so that the numbers describe the machine
    the run actually got.

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
                                     clock, cleanup)
    except Exception as e:
        # LockBusy is the only expected one. It is a failure, not a
        # skip: the job waited hours for a lock somebody else never
        # released, and the holder is in the message so the morning
        # knows who. Caught by name rather than by class to keep this
        # module a leaf of modelctl_lanes' exception types.
        if type(e).__name__ != "LockBusy":
            raise
        record["status"] = "failed"
        record["error"] = (
            f"the GPU lock was still held after {gpu_lock_wait:.0f}s, so "
            f"nothing was measured: {e}")
        record["finished"] = clock()
        return record


def _run_measurements(job, measure, record, ctx, load_interval, clock,
                      cleanup=None):
    record["cleanup"] = (cleanup or cleanup_pass)()
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
