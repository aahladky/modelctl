"""Plan testing and autotuning (Milestone 6).

test_launch_plan: launch one plan on a temporary port, measure load time,
peak resources, and generation throughput, and persist the PlanRun.
autotune_profile: run test_launch_plan over a bounded candidate set, skipping
plans already validated under the current hardware/backend fingerprints.

Measured results feed rank_plans via RuntimeDB.observations_for_profile --
theory proposes, measurement disposes.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import modelctl
import modelctl_hardware
import modelctl_plans
import modelctl_runtime
import modelctl_vram

_TEST_PROMPT = ("Summarize why the sky is blue in three sentences.")
_WARMUP_PROMPT = ("Write a short greeting.")

# Measurement prompts, rotated one per request.
#
# Repeating a single prompt is not a valid throughput measurement on this
# stack: llama.cpp reuses the KV cache for an identical prefix, so the
# second request skips prompt processing entirely, and graph reuse skips
# redundant expert restaging -- which is why `moe_cache_hits_total` stays
# flat across identical repeated requests (scripts/moe-cache-correctness
# records the same effect). A single prompt at temperature 0 also routes to
# one fixed expert sequence, so a MoE cache sees one access pattern rather
# than a representative spread.
#
# Four of them, each over 32 tokens so the expert staging path genuinely
# re-executes, and distinct in subject so they route differently.
_MEASURE_PROMPTS = (
    "Explain how a modern solid-state drive differs from a spinning hard "
    "disk, covering access latency, wear levelling, the role of the flash "
    "translation layer, and why sequential and random access patterns "
    "perform so differently on each of the two technologies in practice.",
    "Describe the water cycle from evaporation through condensation and "
    "precipitation, and explain what role ocean currents, prevailing winds "
    "and mountain ranges each play in moving moisture between different "
    "regions of the planet over the course of a typical year.",
    "Summarise the main differences between the Baroque and Classical "
    "periods in European music, mentioning representative composers from "
    "each, the typical forms and structures that dominated them, and how "
    "the size and instrumentation of orchestras changed between the two.",
    "Walk through how an optimising compiler turns source code into a "
    "running program, from lexing and parsing through intermediate "
    "representation, optimisation passes and code generation, and explain "
    "where the assembler and the linker fit into that overall sequence.",
)


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _post_json(url, body, timeout):
    import json
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_stream(url, body, timeout):
    """Streaming request returning (ttft_seconds, total_seconds, usage_dict).
    TTFT = time to the first SSE data chunk, not the full response."""
    import json
    req = urllib.request.Request(
        url, data=json.dumps({**body, "stream": True}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    ttft = None
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if ttft is None and raw.startswith(b"data:") and b"[DONE]" not in raw:
                ttft = time.time() - t0
            if raw.startswith(b"data:") and b'"usage"' in raw:
                try:
                    usage = json.loads(raw[5:].strip()).get("usage", {})
                except Exception:
                    pass
    return ttft, time.time() - t0, usage


def _short_device_name(device):
    """"/dev/nvme1n1p1" -> "nvme1n1p1", as /proc/diskstats names it."""
    if not device:
        return ""
    return str(device).rsplit("/", 1)[-1]


def _disk_read_bytes(device):
    """Cumulative bytes read from one block device, or None if unknown.

    /proc/diskstats field 6 is sectors read; sectors are 512 bytes for
    this interface regardless of the device's physical sector size.
    """
    if not device:
        return None
    try:
        with open("/proc/diskstats") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) > 5 and parts[2] == device:
                    return int(parts[5]) * 512
    except (OSError, ValueError, IndexError):
        pass
    return None


# A run has to touch at least this much before "the storage was busy" is a
# claim worth making; below it, the reads are startup noise.
_STORAGE_ACTIVE_BYTES = 64 << 20


def classify_storage_activity(read_bytes, major_faults, storage_mode,
                              elapsed_seconds=None):
    """Say what the storage actually did, from counters rather than config.

    `mmap=true` says the model *can* be paged from disk, not that it was:
    a fully page-cached model reads nothing and takes no major faults.
    Inferring "SSD streaming" from the mount option is the specific error
    this exists to prevent, so every answer here is backed by a counter.

    Returns (label, explanation).
    """
    read_bytes = read_bytes or 0
    major_faults = major_faults or 0

    if read_bytes < _STORAGE_ACTIVE_BYTES and major_faults < 1000:
        if storage_mode == "mmap":
            return ("page-cache-served",
                    "mmap is in use but almost nothing was read from disk — "
                    "the model was already in the page cache")
        return ("no-storage-activity",
                "the run did essentially no storage reads")

    if major_faults >= 1000 and read_bytes >= _STORAGE_ACTIVE_BYTES:
        detail = (f"{major_faults} major faults and "
                  f"{modelctl._format_size(read_bytes)} read from disk")
        if elapsed_seconds:
            detail += (f" ({modelctl._format_size(read_bytes / elapsed_seconds)}/s)")
        return ("storage-backed", "weights were paged in during the run: " + detail)

    if read_bytes >= _STORAGE_ACTIVE_BYTES:
        return ("bulk-read",
                f"{modelctl._format_size(read_bytes)} read from disk with few "
                f"major faults — consistent with the model load itself rather "
                f"than paging during generation")

    return ("fault-heavy",
            f"{major_faults} major faults but little data read — "
            f"unusual; check for memory pressure")


def classify_bottleneck(run):
    """What limited this run: storage, CPU, H2D transfer, or GPU compute?

    Every branch is justified by a counter the run actually
    recorded, and the answer is "unknown" whenever the evidence does not
    support one -- a confident wrong answer here sends someone off to
    optimise the wrong thing.

    `run` is a plan_run dict (or sqlite row). Returns (label, why).
    """
    def num(key, default=0):
        try:
            value = run[key] if key in run.keys() else run.get(key, default)
        except (AttributeError, TypeError, KeyError, IndexError):
            value = run.get(key, default) if hasattr(run, "get") else default
        return value if value is not None else default

    gen_read = num("read_bytes_generation")
    major = num("major_faults")
    gen_tps = num("generation_tps")
    ram_bytes = 0
    try:
        claim = run.get("claim_json") or run.get("claim") or {}
        if isinstance(claim, str):
            claim = json.loads(claim)
        ram_bytes = claim.get("ram_bytes", 0) or 0
    except Exception:
        ram_bytes = 0

    # Storage: bytes are still being pulled from disk while generating.
    if gen_read >= 256 << 20:
        return ("storage",
                f"{modelctl._format_size(gen_read)} was read from disk during "
                f"generation — the model is being paged in as it runs")
    if major >= 10_000 and gen_read >= 64 << 20:
        return ("storage",
                f"{major} major faults during the run with "
                f"{modelctl._format_size(gen_read)} read — weights are "
                f"faulting in from storage")

    # CPU: weights live in RAM and are not being re-read, so the host is
    # doing that part of the FFN every token.
    if ram_bytes >= 4 << 30 and gen_read < 64 << 20:
        return ("cpu",
                f"{modelctl._format_size(ram_bytes)} of weights sit on the "
                f"host and are not being re-read from disk — the CPU is "
                f"computing over them each token")

    # H2D: the expert cache is missing often enough that every token drags
    # weights across PCIe.
    #
    # Metrics live in details["cache_metrics"] as {device: {name: value}}
    # (scrape_moe_cache_metrics), persisted through details_json. This
    # branch used to read a flat run["cache_metrics"] with numeric
    # hits/lookups keys -- a shape nothing ever produced -- so every
    # transfer-bound run fell through and was reported, confidently, as
    # GPU-bound.
    cache = {}
    try:
        details = run["details_json"] if "details_json" in run.keys() \
            else run.get("details_json") or run.get("details") or {}
    except (AttributeError, TypeError, KeyError, IndexError):
        details = run.get("details") if hasattr(run, "get") else {}
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except ValueError:
            details = {}
    if isinstance(details, dict):
        cache = details.get("cache_metrics") or {}
    hits = lookups = 0
    if isinstance(cache, dict):
        for per_device in cache.values():
            if not isinstance(per_device, dict):
                continue
            def _n(name):
                try:
                    return float(per_device.get(name, 0) or 0)
                except (TypeError, ValueError):
                    return 0.0
            hits += _n("hits_total") or _n("hits")
            lookups += (_n("lookups_total") or _n("lookups")
                        or (_n("hits_total") or _n("hits"))
                        + (_n("misses_total") or _n("misses")))
    if lookups:
        hit_ratio = hits / lookups
        if hit_ratio < 0.5:
            return ("h2d",
                    f"expert cache hit rate is {hit_ratio * 100:.0f}% — most "
                    f"tokens transfer expert weights over PCIe")

    if gen_tps:
        return ("gpu",
                f"no significant storage reads, host weights, or cache misses "
                f"— generation at {gen_tps:.1f} t/s is GPU-bound")

    return ("unknown",
            "not enough evidence in this run to attribute a bottleneck")


class _Sampler:
    """1 Hz peak sampler: per-device VRAM free, child RSS, system RAM
    available, and cumulative storage read bytes / page faults across the
    launched process group -- distinguishes compute speed from
    active storage reads."""

    def __init__(self, pid, storage_device=""):
        self.pid = pid
        self.peak_vram = {}
        self.baseline_vram = {}
        # VRAM after the process exits, so a leak (or a device that never
        # released) is visible rather than being read as peak usage.
        self.final_vram = {}
        self.peak_ram = 0
        self.peak_pss = 0
        self.read_bytes = 0
        self.read_syscalls = 0
        self.minor_faults = 0
        self.major_faults = 0
        # Block-device counters for the disk the model lives on, so
        # storage throughput is attributed to the right device rather than
        # to whatever else the machine happened to be doing.
        self.storage_device = _short_device_name(storage_device)
        self._disk_start = None
        self.disk_read_bytes = 0
        self.started_at = 0.0
        self.finished_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.started_at = time.time()
        self._disk_start = _disk_read_bytes(self.storage_device)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join(timeout=3)
        self.finished_at = time.time()
        end = _disk_read_bytes(self.storage_device)
        if self._disk_start is not None and end is not None:
            self.disk_read_bytes = max(0, end - self._disk_start)
        try:
            for d in modelctl.get_gpu_inventory():
                self.final_vram[d["device"]] = (
                    (d["total_bytes"] - d["free_bytes"])
                    - self.baseline_vram.get(d["device"], 0))
        except Exception:
            pass

    def io_snapshot(self):
        """Current cumulative counters -- callers diff two snapshots to get
        a phase's contribution (e.g. warmup vs. measured generation)."""
        return {"read_bytes": self.read_bytes,
                "read_syscalls": self.read_syscalls,
                "minor_faults": self.minor_faults,
                "major_faults": self.major_faults}

    def rates(self):
        """Derived rates alongside the raw counters.

        The raw counters are what get stored; these are for display, and
        are computed here so every surface derives them the same way.
        """
        elapsed = max(1e-6, (self.finished_at or time.time()) - (self.started_at or 0))
        return {
            "elapsed_seconds": elapsed,
            "read_bytes_per_second": self.read_bytes / elapsed,
            "disk_read_bytes_per_second": self.disk_read_bytes / elapsed,
            "major_faults_per_second": self.major_faults / elapsed,
        }

    def _run(self):
        baseline = {}
        try:
            for d in modelctl.get_gpu_inventory():
                baseline[d["device"]] = d["total_bytes"] - d["free_bytes"]
        except Exception:
            pass
        self.baseline_vram = baseline
        while not self._stop.is_set():
            try:
                for d in modelctl.get_gpu_inventory():
                    used = (d["total_bytes"] - d["free_bytes"]) - baseline.get(d["device"], 0)
                    self.peak_vram[d["device"]] = max(
                        self.peak_vram.get(d["device"], 0), used)
                rss, pss = self._child_memory()
                self.peak_ram = max(self.peak_ram, rss)
                self.peak_pss = max(self.peak_pss, pss)
                read_bytes, minor, major, syscr = self._child_io()
                self.read_bytes = read_bytes
                self.minor_faults = minor
                self.major_faults = major
                self.read_syscalls = syscr
            except Exception:
                pass
            self._stop.wait(1.0)

    def _child_memory(self):
        """(RSS, PSS) summed across the process group.

        PSS divides shared pages by the number of sharers, so a main model
        and its draft sharing mmap'd pages are not counted twice. It comes
        from smaps_rollup, which is not always readable; 0 means unknown,
        not zero.
        """
        rss = pss = 0
        try:
            for f in os.listdir("/proc"):
                if not f.isdigit():
                    continue
                try:
                    with open(f"/proc/{f}/stat") as fh:
                        if int(fh.read().rsplit(")", 1)[-1].split()[2]) != self.pid:
                            continue
                    with open(f"/proc/{f}/status") as fh:
                        for line in fh:
                            if line.startswith("VmRSS:"):
                                rss += int(line.split()[1]) * 1024
                                break
                except (OSError, ValueError, IndexError):
                    continue
                try:
                    with open(f"/proc/{f}/smaps_rollup") as fh:
                        for line in fh:
                            if line.startswith("Pss:"):
                                pss += int(line.split()[1]) * 1024
                                break
                except (OSError, ValueError, IndexError):
                    pass
        except OSError:
            pass
        return rss, pss

    def _child_io(self):
        """Cumulative read_bytes (/proc/<pid>/io, bytes actually fetched
        from the storage layer -- not rchar, which includes page-cache
        hits) and minflt/majflt (/proc/<pid>/stat), summed across every
        process in the launched process group (same pgrp-match as
        _child_rss -- catches draft-model/worker subprocesses too)."""
        read_bytes = minor = major = syscr = 0
        try:
            for f in os.listdir("/proc"):
                if not f.isdigit():
                    continue
                try:
                    with open(f"/proc/{f}/stat") as fh:
                        parts = fh.read().rsplit(")", 1)[-1].split()
                    if int(parts[2]) != self.pid:
                        continue
                    minor += int(parts[7])   # minflt
                    major += int(parts[9])   # majflt
                except (OSError, ValueError, IndexError):
                    continue
                try:
                    with open(f"/proc/{f}/io") as fh:
                        for line in fh:
                            if line.startswith("read_bytes:"):
                                read_bytes += int(line.split()[1])
                            elif line.startswith("syscr:"):
                                # Read syscall count separates "many small
                                # reads" from "few large ones" at the same
                                # byte total.
                                syscr += int(line.split()[1])
                except OSError:
                    pass
        except OSError:
            pass
        return read_bytes, minor, major, syscr



def _classify_failure_from_log(profile, exit_code, log_path, tail_bytes=8192):
    """Failure class for a backend that died or never became ready.

    Reads the tail of the run's log and asks the backend adapter to
    classify it (modelctl_backends.*.classify_failure), so permanent
    failures get a class that plan ranking can suppress rather than the
    generic backend_crash it retries.
    """
    tail = ""
    try:
        with open(log_path, "rb") as f:
            try:
                f.seek(-tail_bytes, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        pass
    try:
        import modelctl_backends
        adapter = modelctl_backends.get_backend(
            (profile or {}).get("backend", "llama-cpp"))
        return adapter.classify_failure(exit_code, tail)
    except Exception:
        return "health_timeout" if exit_code is None else "backend_crash"


def _wait_ready(port, proc, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, time.time() - (deadline - timeout)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True, time.time() - (deadline - timeout)
        except Exception:
            pass
        time.sleep(2)
    return False, time.time() - (deadline - timeout)


def _prompt_sequence(prompt):
    """Prompts to rotate through the measured requests.

    A caller passing one string keeps the old single-prompt behaviour,
    because some callers (a user testing a specific prompt) mean exactly
    that. Passing a sequence, or nothing, gets the rotated set.
    """
    if prompt is None:
        return _MEASURE_PROMPTS
    if isinstance(prompt, str):
        return (prompt,)
    seq = tuple(p for p in prompt if p)
    return seq or _MEASURE_PROMPTS


# A "cold" run whose model is still more resident than this fraction is
# not cold; it is relabeled warm rather than recorded as a measurement of
# something it did not measure.
_COLD_RESIDENCY_MAX = 0.05


def _effective_cache_state(warmed_up: bool, plan_cache_bytes: int,
                           residency_at_start) -> str:
    """The honest cache state of a measurement (P3: cold page cache, warm
    page cache, and warm expert cache are three different things).

      "cold"        no warmup, and the model's pages were verified evicted
      "warm"        page cache holds the model (after warmup, or because a
                    requested-cold run's eviction did not take -- another
                    process maps the file, or the store is RAM-backed)
      "warm-expert" warmed up AND the plan reserves a GPU expert cache, so
                    the warmup also filled that

    residency_at_start is the page-cache fraction measured after the
    pre-launch eviction attempt, or None when it could not be measured
    (residency unknown => trust the eviction attempt, the conservative
    reading for ranking purposes since cold numbers rank lower)."""
    if warmed_up:
        return "warm-expert" if (plan_cache_bytes or 0) > 0 else "warm"
    if residency_at_start is not None and residency_at_start > _COLD_RESIDENCY_MAX:
        return "warm"
    return "cold"


def _denial_message(denial):
    """Plain words for a reservation denial: a plan that does not fit the
    machine must not read as a collision with a worker that does not
    exist (the 2026-08-03 baseline shipped exactly that confusion)."""
    if not denial:
        return "reservation conflict -- another worker holds the resources"

    def gib(n):
        return f"{n / (1 << 30):.1f} GiB"

    if denial.get("code") in ("insufficient_vram", "insufficient_ram"):
        msg = (f"this plan needs {gib(denial['need_bytes'])} on "
               f"{denial['resource']} but only "
               f"{gib(denial['budget_bytes'])} is free")
        if denial.get("pending_bytes"):
            who = ", ".join(denial.get("holders") or []) or "another worker"
            msg += (f" ({gib(denial['pending_bytes'])} already claimed "
                    f"by {who})")
        return msg
    holders = ", ".join(denial.get("holders") or []) or "another worker"
    return f"reservation conflict -- {holders} holds the resources"


def test_launch_plan(profile_name, plan_id, log=print, prompt=None,
                     max_tokens=128, runs=2, binary=None,
                     proc_register=None, cancel_check=None,
                     warmup_tokens=0, profile_override=None):
    """Measure one launch plan and persist the resulting PlanRun dict.

    When warmup_tokens > 0, a warmup generation runs first to fill the
    expert cache and OS page cache.  The measured runs then capture
    warm-cache performance.  For a cold run (warmup_tokens == 0) the
    model's pages are evicted from the page cache before launch and the
    residual residency is recorded; if eviction does not take, the run is
    honestly labeled warm instead of pretending to be cold.  The run dict
    includes:
      cache_state: "cold", "warm" (page cache), or "warm-expert"
                   (warmed up with a GPU expert cache reserved)
      warmup_generation_tps: speed during the warmup phase (if any)

    profile_override supplies the profile dict directly instead of loading
    it by name. The hardware acceptance matrix varies config per
    cell over one fixture model, and saving a dozen throwaway profiles to
    do that would pollute the user's profile store.
    """
    profile = profile_override or modelctl.load_profile(profile_name)
    snap = modelctl_hardware.capture_hardware_snapshot()
    plans = modelctl_plans.compile_launch_plans(profile, snap)
    plan = next((p for p in plans if p.id == plan_id), None)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found for '{profile_name}'")

    rdb = modelctl_runtime.RuntimeDB()
    started = time.time()
    run = {"profile_name": profile_name, "plan_id": plan_id,
           "hardware_fingerprint": snap.fingerprint,
           "backend_fingerprint": snap.backend_fingerprints.get(
               profile.get("backend", "llama-cpp"), ""),
           "started_at": started, "success": False, "log_path": ""}

    # Canonical admission values: VRAM includes the dynamic expert-cache
    # budget, RAM counts resident bytes rather than mmap-addressed ones.
    # The rationale (a 71 GiB Q4_K_M on mmap needs 0 bytes resident and
    # was being denied against 26.5 GiB available) lives on
    # ResourceClaim.ram_admission_bytes(); this path must charge the same
    # numbers the managed worker does, or a plan that passes browser
    # testing gets rejected by managed serving.
    claim = {"vram_bytes": plan.claim.vram_admission_bytes(),
             "ram_bytes": plan.claim.ram_admission_bytes(),
             "storage_mode": plan.claim.storage_mode,
             "vram_cache_bytes": dict(plan.claim.vram_cache_bytes),
             "ram_resident_bytes": plan.claim.ram_resident_bytes,
             "mmap_bytes": plan.claim.mmap_bytes}
    budgets = modelctl_hardware.reservation_budgets(snap)
    reservation, denial = rdb.acquire_reservation_verdict(
        profile_name, plan_id, claim, os.getpid(), budgets=budgets)
    if reservation is None:
        run["failure_class"] = (denial or {}).get("code",
                                                  "reservation_conflict")
        run["finished_at"] = time.time()
        rdb.record_plan_run(run)
        raise RuntimeError(_denial_message(denial))

    port = _free_port()
    proc = None
    try:
        # Canonical launch path: same ResolvedBackend + LaunchCommand as the
        # managed worker, including fail-closed cache validation.
        import modelctl_launch
        resolved = modelctl_launch.resolve_backend(profile, binary_override=binary)
        launch = modelctl_launch.build_launch_command(profile, plan,
                                                      backend=resolved, port=port)
        if not launch.is_valid:
            run["failure_class"] = "preflight_failed"
            run["details"] = {"validation": [v.summary for v in launch.errors]}
            log("launch validation failed: "
                + "; ".join(v.summary for v in launch.errors))
            return run
        cmd = list(launch.argv)
        run["command_argv"] = cmd
        run["command_fingerprint"] = launch.command_fingerprint
        run["binary_path"] = launch.backend.binary
        run["binary_fingerprint"] = launch.backend.binary_fingerprint
        run["environment_fingerprint"] = launch.environment_fingerprint
        run["capability_schema"] = launch.backend.capabilities.get("schema", 0)
        run["capability_digest"] = launch.backend.capability_fingerprint
        log(f"launching: {' '.join(cmd[:6])} ...")
        log_path = modelctl.PROFILES_DIR / profile_name / f"plan-test-{plan_id[:8]}.log"
        run["log_path"] = str(log_path)
        # launch.environment is complete (plan env and library paths
        # included) and is what the fingerprint describes -- no caller-side
        # environment assembly.
        child_env = dict(launch.environment)
        # A cold run means COLD: evict the model's pages before launch and
        # measure what remains, instead of trusting that nothing else has
        # touched the file. Best-effort -- eviction legitimately fails when
        # another server maps the file or the store is RAM-backed, and the
        # residual residency then relabels the run warm (see
        # _effective_cache_state).
        residency_at_start = None
        if not warmup_tokens:
            try:
                import modelctl_benchmark
                import modelctl_vram
                shards = modelctl_vram._shard_paths(
                    profile.get("model_path", "")) or []
                shards = [str(s) for s in shards]
                if shards:
                    modelctl_benchmark.evict_from_page_cache(shards)
                    res = modelctl_benchmark.page_cache_residency(shards)
                    if res is not None:
                        residency_at_start = round(res.fraction, 4)
                        run["details"] = {
                            **run.get("details", {}),
                            "page_cache_residency_at_start": residency_at_start}
            except Exception as e:
                log(f"page-cache eviction skipped: {e}")

        # The per-profile directory is created by generate_artifacts(), but a
        # plan test must not require artifacts to have been generated first:
        # testing a plan before registering anything is the normal wizard
        # order, and this crashed with FileNotFoundError instead of running.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "w")
        preexec = os.setpgrp if hasattr(os, "setpgrp") else None
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                env=child_env, preexec_fn=preexec)
        if proc_register:
            proc_register(proc)

        with _Sampler(proc.pid,
                      storage_device=plan.claim.storage_device) as sampler:
            ready, load_s = _wait_ready(port, proc)
            if not ready:
                run["exit_code"] = proc.poll()
                # Classify from the log tail rather than defaulting to a
                # generic crash: the suppression classes
                # (unsupported_architecture, invalid_argument) only exist
                # if something produces them, and nothing did -- so a
                # plan that can never work was retried on every managed
                # launch forever instead of being permanently suppressed.
                run["failure_class"] = _classify_failure_from_log(
                    profile, run["exit_code"], log_path)
                log(f"backend failed to become ready (class={run['failure_class']})")
                return run

            run["load_seconds"] = round(load_s, 2)
            log(f"ready in {load_s:.1f}s, benchmarking {runs}x{max_tokens} tok ...")

            actual_ctx = None
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5) as r:
                    import json as _json
                    props = _json.loads(r.read())
                actual_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            except Exception:
                pass

            # Warmup phase: fill expert cache and OS page cache before
            # measuring.  When warmup_tokens is 0, the run is "cold".
            io_at_ready = sampler.io_snapshot()
            warmed_up = False
            run["warmup_generation_tps"] = None
            if warmup_tokens and warmup_tokens > 0:
                log(f"warmup: {warmup_tokens} tokens ...")
                warmup_resp = _post_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": _WARMUP_PROMPT}],
                     "max_tokens": warmup_tokens, "temperature": 0},
                    timeout=600)
                wt = warmup_resp.get("timings", {})
                if wt.get("predicted_per_second"):
                    run["warmup_generation_tps"] = round(wt["predicted_per_second"], 2)
                    log(f"warmup gen {run['warmup_generation_tps']} tok/s")
                warmed_up = True
            run["cache_state"] = _effective_cache_state(
                warmed_up, plan.claim.cache_bytes, residency_at_start)
            if not warmed_up and run["cache_state"] == "warm":
                log(f"requested a cold run but {residency_at_start:.0%} of "
                    "the model is still in the page cache -- recording it "
                    "as warm")

            io_after_warmup = sampler.io_snapshot()
            run["read_bytes_warmup"] = io_after_warmup["read_bytes"] - io_at_ready["read_bytes"]

            # Keep the warmup measurement separate from the measured TPS but
            # not lost: persist it in details alongside the cache_state column.
            run["details"] = {**run.get("details", {}),
                              "warmup_generation_tps": run["warmup_generation_tps"]}

            # Rotate a distinct prompt into every request, including the
            # streaming TTFT call and the timings call within one iteration
            # -- those two were previously the same prompt back to back, so
            # the second measured a warm KV cache rather than the plan.
            prompts = _prompt_sequence(prompt)
            issued = 0

            prompt_tps, gen_tps, ttfts = [], [], []
            for i in range(runs):
                if cancel_check and cancel_check():
                    run["failure_class"] = "cancelled"
                    raise InterruptedError("plan test cancelled")
                ttft, _total, _usage = _post_stream(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user",
                                   "content": prompts[issued % len(prompts)]}],
                     "max_tokens": max_tokens, "temperature": 0},
                    timeout=600)
                issued += 1
                if ttft:
                    ttfts.append(ttft)
                resp = _post_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"messages": [{"role": "user",
                                   "content": prompts[issued % len(prompts)]}],
                     "max_tokens": max_tokens, "temperature": 0},
                    timeout=600)
                issued += 1
                t = resp.get("timings", {})
                if t.get("prompt_per_second"):
                    prompt_tps.append(t["prompt_per_second"])
                if t.get("predicted_per_second"):
                    gen_tps.append(t["predicted_per_second"])

            run["success"] = True
            # Scraped before the server is torn down, because afterwards
            # there is no way to tell whether the cache served anything.
            # A throughput number without this cannot distinguish "the
            # cache did not help" from "the cache never ran" -- the exact
            # confusion behind two earlier misleading results. None means
            # not recorded, never zero.
            cache_metrics = modelctl.scrape_moe_cache_metrics(port)
            # Which measurement protocol produced these numbers. Rotated
            # prompts and a single repeated prompt are not comparable --
            # the repeated one reuses the KV cache -- and rows recorded
            # before this existed have no key here at all.
            run["details"] = {**run.get("details", {}),
                              "prompt_count": len(prompts),
                              "requests_issued": issued,
                              "cache_metrics": cache_metrics}
            run["actual_context"] = actual_ctx or plan.claim.expected_context
            run["ttft_seconds"] = round(min(ttfts), 3) if ttfts else None
            run["prompt_tps"] = round(sum(prompt_tps) / len(prompt_tps), 2) if prompt_tps else None
            run["generation_tps"] = round(sum(gen_tps) / len(gen_tps), 2) if gen_tps else None
            log(f"gen {run['generation_tps']} tok/s, prompt {run['prompt_tps']} tok/s")

        run["peak_vram_bytes"] = sampler.peak_vram
        run["peak_ram_bytes"] = sampler.peak_ram
        run["peak_pss_bytes"] = sampler.peak_pss
        run["baseline_vram_bytes"] = sampler.baseline_vram
        run["final_vram_bytes"] = sampler.final_vram
        io_at_end = sampler.io_snapshot()
        run["read_bytes"] = io_at_end["read_bytes"]
        run["read_bytes_generation"] = io_at_end["read_bytes"] - io_after_warmup["read_bytes"]
        run["major_faults"] = io_at_end["major_faults"]
        run["minor_faults"] = io_at_end["minor_faults"]
        run["read_syscalls"] = io_at_end["read_syscalls"]
        run["disk_read_bytes"] = sampler.disk_read_bytes
        run["storage_device"] = sampler.storage_device
        # Raw counters are what get stored; the rates are derived here so
        # every surface derives them identically.
        run["rates"] = sampler.rates()
        # What the storage actually did, from counters -- never from the
        # fact that mmap was enabled.
        label, why = classify_storage_activity(
            io_at_end["read_bytes"], io_at_end["major_faults"],
            plan.claim.storage_mode,
            elapsed_seconds=run["rates"]["elapsed_seconds"])
        run["storage_activity"] = label
        run["storage_activity_detail"] = why
        log(f"storage: {label} — {why}")
        return run
    except Exception as e:
        run["failure_class"] = run.get("failure_class") or "unknown"
        run["details"] = {**run.get("details", {}), "error": str(e)}
        raise
    finally:
        run["finished_at"] = time.time()
        if proc is not None:
            try:
                pg = os.getpgid(proc.pid)
                os.killpg(pg, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        rdb.release_reservation(reservation["id"])
        rdb.record_plan_run(run)


def autotune_profile(profile_name, objective="balanced", candidate_ids=None,
                     log=print, retest=False, max_tokens=128, runs=2, binary=None,
                     proc_register=None, cancel_check=None):
    """Test a bounded set of candidate plans and report measured ranking.
    Never changes the profile or selects a winner automatically."""
    profile = modelctl.load_profile(profile_name)
    snap = modelctl_hardware.capture_hardware_snapshot()
    plans = modelctl_plans.compile_launch_plans(profile, snap)
    rdb = modelctl_runtime.RuntimeDB()
    existing = rdb.observations_for_profile(
        profile_name, snap.fingerprint,
        snap.backend_fingerprints.get(profile.get("backend", "llama-cpp"), ""))

    candidates = []
    skipped_validated = 0
    for p in plans:
        if cancel_check and cancel_check():
            raise InterruptedError("autotune cancelled")
        if candidate_ids and p.id not in candidate_ids:
            continue
        obs = existing.get(p.id)
        if obs and not obs.get("stale") and not retest:
            tps = obs.get("generation_tps")
            log(f"skip {p.label} -- already validated on this hardware "
                + (f"({tps} tok/s)" if tps else "(no throughput recorded)"))
            skipped_validated += 1
            continue
        candidates.append(p)

    results = []
    for p in candidates:
        log(f"testing plan {p.label} ...")
        try:
            run = test_launch_plan(profile_name, p.id, log=log,
                                   max_tokens=max_tokens, runs=runs, binary=binary,
                                   proc_register=proc_register, cancel_check=cancel_check)
        except Exception as e:
            log(f"plan {p.label} failed: {e}")
            continue
        results.append(run)

    policy = modelctl_plans.DEFAULT_POLICY
    scored = sorted(
        (r for r in results if r.get("success")),
        key=lambda r: -(r.get("generation_tps") or 0))
    # Counted where the skip happens: len(plans) - len(candidates) also
    # counted plans excluded by a candidate_ids filter, so a run testing
    # 2 of 10 selected plans reported 8 "already validated".
    return {"tested": len(results), "skipped_validated": skipped_validated,
            "results": results, "best": scored[0]["plan_id"] if scored else None}


# Imported lazily by worker too; keep at end to avoid a cycle at module import.
import modelctl_worker  # noqa: E402
