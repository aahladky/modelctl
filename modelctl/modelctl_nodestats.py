"""Per-host telemetry for the fleet's remote nodes.

Separate from modelctl_fleet on purpose. That module is the registry and
the placement contract -- what nodes exist, what may be spent on each,
whether their build agrees with the commit this checkout pins. This one
answers a different question, "what is that machine doing right now",
and answers it the only way another box can be asked: over ssh.

Four rules this module's shape comes from.

* ONE ssh round trip per host, never one per unit. Both of ph16-71's
  units live on the same laptop, and opening two connections to ask two
  questions of one machine is two handshakes and two chances to hang.
* The command is a fixed template. Unit names are the only interpolated
  values and they are validated against UNIT_RE before they are
  interpolated; the host is passed as an ssh argv element, never as a
  shell word. Nothing arriving over HTTP reaches either.
* Every field is optional, and a field that could not be read is
  ABSENT -- never zero. "the GPU is 0% busy" and "we could not ask the
  GPU" are different facts, and a card that prints the first when it
  means the second is lying at a glance. The renderer draws an em dash
  for absent; that contract starts here, by leaving the key out.
* Nothing here runs in the tick request path. NodeStatsPoller owns a
  timer thread and the tick reads its cache, because one hung ssh must
  not stall the front page.

Why ssh and not the RPC socket: the protocol does have a memory query
(GET_DEVICE_MEMORY, cmd 11) and it does answer over the probe socket.
It is deliberately not wired in here, because for the CPU node its
answer is not a reading -- the CPU backend hardcodes `*free = *total`
(llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp, ggml_backend_cpu_device_get_memory).
A number that is always "all of it" rendered next to numbers that are
measurements is the same lie as the zero above. ssh is the source of
truth for this surface.
"""
import re
import subprocess
import threading
import time

# The one shape a systemd unit name may have before it is interpolated
# into the remote command. Deliberately narrower than systemd's own
# grammar: this is an allowlist guarding a shell string, not a parser.
UNIT_RE = re.compile(r"^[A-Za-z0-9._@-]+$")

# BatchMode: never prompt -- a poller that blocks on a passphrase prompt
# hangs its thread until the hard timeout instead of failing fast.
# accept-new: trust a first-seen host key but still refuse a CHANGED one,
# which is the half of StrictHostKeyChecking worth keeping on a LAN.
SSH_FLAGS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
             "-o", "StrictHostKeyChecking=accept-new")

# How often the background thread polls, and the age past which a cached
# reading stops being served as current. The gap between them is
# deliberate: three missed polls before the card goes stale, so one
# dropped ssh does not flicker the front page.
POLL_INTERVAL_SECONDS = 10.0
STALE_AFTER_SECONDS = 30.0

MIB = 1 << 20
KIB = 1024

# The one presence state that means "the planner will use it". Spelled
# here rather than imported from modelctl_web.fleet for the reason
# node_rows() takes its state function as an argument: this module sits
# below the web package, not beside it. The two spellings are one string
# and test_console_nodestats pins them together.
PRESENT = "PRESENT"


def unit_for_node(node):
    """The systemd user unit serving one registry node.

    Derived rather than stored, because the registry file is not ours to
    add fields to. The mapping is the one recorded in
    modelctl/docs/fleet/README.md -- node `ph16-71-cuda0` is served by
    `rpc-cuda0.service` on port 50052, `ph16-71-cpu0` by
    `rpc-cpu0.service` on 50053 -- and the rule that reproduces it is the
    node name's last dash-separated segment.

    Returns "" when the derived name would not survive UNIT_RE, so a
    node named in some future scheme reports no unit (and renders as
    absent) rather than smuggling its name into a shell string.
    """
    suffix = str(getattr(node, "name", "") or "").rsplit("-", 1)[-1]
    if not suffix:
        return ""
    unit = f"rpc-{suffix}.service"
    return unit if UNIT_RE.match(unit) else ""


def poll_targets(nodes):
    """{host: [unit, ...]} for the nodes worth an ssh round trip.

    Grouped by host, not by node: that grouping is the whole reason one
    connection can answer for both units. A node with no host recorded
    is not polled at all -- it still renders, from presence and budget,
    which is all the registry knows about it.
    """
    targets = {}
    for node in nodes or []:
        host = str(getattr(node, "host", "") or "").strip()
        if not host:
            continue
        unit = unit_for_node(node)
        units = targets.setdefault(host, [])
        if unit and unit not in units:
            units.append(unit)
    return targets


def _remote_script(units):
    """The fixed command template, with validated unit names in it.

    Section markers rather than a JSON payload on the far side: the node
    is a stock laptop and this must not depend on python (or jq) being
    installed there. Every section is best-effort -- a probe that fails
    prints nothing under its marker and the field comes back absent,
    which is exactly the rendering it deserves.
    """
    lines = [
        'echo "@@load"; cat /proc/loadavg 2>/dev/null',
        'echo "@@mem"; grep -E "^(MemTotal|MemAvailable):" /proc/meminfo '
        '2>/dev/null',
        'echo "@@nproc"; nproc 2>/dev/null',
    ]
    for unit in units:
        lines.append(
            f'echo "@@unit {unit}"; systemctl --user show {unit} '
            f'-p MemoryCurrent,MemoryMax,ActiveState 2>/dev/null')
    # Absent nvidia-smi is not an error: the cpu-only node answers every
    # other section and simply has no GPU block.
    lines.append(
        'echo "@@gpu"; if command -v nvidia-smi >/dev/null 2>&1; then '
        'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,'
        'temperature.gpu --format=csv,noheader,nounits 2>/dev/null; fi')
    lines.append('echo "@@end"')
    return "\n".join(lines)


def _int_or_none(text):
    """systemd prints "[not set]" and "infinity"; both mean "no reading"."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def _float_or_none(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def parse_host_output(text):
    """Marker-delimited remote output -> the optional-field dict.

    Split out from the subprocess call so the parse is testable without
    faking a process, and so a garbled section can only cost its own
    field: every value goes through _int_or_none/_float_or_none and a
    line that does not parse leaves its key absent.
    """
    out = {"units": {}}
    section = None
    unit = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line.startswith("@@"):
            marker = line[2:].strip()
            if marker.startswith("unit "):
                section, unit = "unit", marker[5:].strip()
                out["units"].setdefault(unit, {})
            else:
                section, unit = marker, None
            continue
        if not line:
            continue
        if section == "load":
            one = _float_or_none(line.split()[0])
            if one is not None:
                out["load1"] = one
        elif section == "mem":
            parts = line.split()
            if len(parts) >= 2:
                kb = _int_or_none(parts[1])
                if kb is not None:
                    if line.startswith("MemTotal:"):
                        out["mem_total_bytes"] = kb * KIB
                    elif line.startswith("MemAvailable:"):
                        out["mem_available_bytes"] = kb * KIB
        elif section == "nproc":
            n = _int_or_none(line)
            if n is not None:
                out["nproc"] = n
        elif section == "unit" and unit:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key == "ActiveState":
                out["units"][unit]["active_state"] = value
            elif key == "MemoryCurrent":
                n = _int_or_none(value)
                if n is not None:
                    out["units"][unit]["memory_bytes"] = n
            elif key == "MemoryMax":
                n = _int_or_none(value)
                if n is not None:
                    out["units"][unit]["memory_max_bytes"] = n
        elif section == "gpu":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                used, total = _int_or_none(parts[0]), _int_or_none(parts[1])
                util, temp = _int_or_none(parts[2]), _int_or_none(parts[3])
                gpu = {}
                if used is not None:
                    gpu["used_bytes"] = used * MIB
                if total is not None:
                    gpu["total_bytes"] = total * MIB
                if util is not None:
                    gpu["util_pct"] = util
                if temp is not None:
                    gpu["temp_c"] = temp
                if gpu:
                    out["gpu"] = gpu
    return out


def poll_host(host, units, timeout=8, runner=None):
    """One ssh round trip -> this host's telemetry, or a reason it failed.

    Never raises. Every caller of this is either a timer thread (where an
    exception is a silently dead poller) or a test, and the failure of a
    remote machine is data this surface has to render, not an error
    condition for the console.

    `runner` is the subprocess boundary, injected by the tests: nothing
    in the suite opens an ssh connection.
    """
    units = list(units or [])
    bad = [u for u in units if not UNIT_RE.match(str(u))]
    if bad:
        # Rejected before the template is built, so a name that fails the
        # allowlist is never interpolated even into a string we then
        # discard.
        return {"host": host, "ok": False,
                "error": f"unit name rejected: {', '.join(map(str, bad))}"}
    if not str(host or "").strip():
        return {"host": host, "ok": False, "error": "no ssh host recorded"}

    argv = ["timeout", str(int(timeout)), "ssh", *SSH_FLAGS, str(host),
            _remote_script(units)]
    run = runner or subprocess.run
    try:
        # The coreutils `timeout` is the hard stop the order asks for;
        # subprocess's own timeout is the backstop for the case where ssh
        # ignores its signal, and is deliberately the looser of the two.
        proc = run(argv, capture_output=True, text=True, timeout=timeout + 2)
    except subprocess.TimeoutExpired:
        return {"host": host, "ok": False,
                "error": f"ssh timed out after {timeout}s"}
    except OSError as e:
        return {"host": host, "ok": False, "error": str(e) or "ssh failed"}

    rc = getattr(proc, "returncode", 1)
    text = getattr(proc, "stdout", "") or ""
    if rc == 124:
        # coreutils timeout's own exit status.
        return {"host": host, "ok": False,
                "error": f"ssh timed out after {timeout}s"}
    if rc != 0 and "@@end" not in text:
        detail = (getattr(proc, "stderr", "") or "").strip().splitlines()
        return {"host": host, "ok": False,
                "error": detail[-1] if detail else f"ssh exited {rc}"}

    stats = parse_host_output(text)
    stats["host"] = host
    stats["ok"] = True
    stats["error"] = None
    return stats


class NodeStatsPoller:
    """Background per-host poller with a last-good cache.

    The cache is what the tick reads. It holds the last SUCCESSFUL
    reading and the monotonic instant it was taken, so a run of failures
    ages the number the operator is looking at instead of blanking it --
    and `ok` goes false the moment that age passes the threshold, so an
    aged number can never pass for a current one.
    """

    def __init__(self, targets_fn=None, poll_fn=None,
                 interval=POLL_INTERVAL_SECONDS,
                 stale_after=STALE_AFTER_SECONDS, clock=time.monotonic):
        self._targets_fn = targets_fn or _registry_targets
        self._poll_fn = poll_fn or poll_host
        self._interval = interval
        self._stale_after = stale_after
        self._clock = clock
        self._lock = threading.Lock()
        self._hosts = {}     # host -> last good stats dict
        self._at = {}        # host -> monotonic stamp of that reading
        self._error = None   # reason the last sweep failed, if it did
        self._thread = None
        self._stop = threading.Event()

    # -- lifecycle -------------------------------------------------------
    def start(self):
        """Start the timer thread once. Safe to call repeatedly."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop,
                                            name="node-stats-poller",
                                            daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            self.poll_once()
            # wait(), not sleep(): a shutdown interrupts the pause rather
            # than riding it out, same contract as the SSE generator.
            if self._stop.wait(self._interval):
                return

    def poll_once(self):
        """One sweep over every target host. Used by the thread and tests."""
        try:
            targets = self._targets_fn()
        except Exception as e:            # registry unreadable
            with self._lock:
                self._error = str(e) or "fleet registry unreadable"
            return
        errors = []
        for host, units in sorted(targets.items()):
            result = self._poll_fn(host, units)
            now = self._clock()
            with self._lock:
                if result.get("ok"):
                    self._hosts[host] = result
                    self._at[host] = now
                else:
                    errors.append(f"{host}: {result.get('error') or 'failed'}")
        with self._lock:
            # Drop hosts that left the registry, so a de-registered node
            # cannot keep answering out of the cache.
            live = set(targets)
            for gone in [h for h in self._hosts if h not in live]:
                self._hosts.pop(gone, None)
                self._at.pop(gone, None)
            self._error = "; ".join(errors) if errors else None

    # -- read ------------------------------------------------------------
    def snapshot(self):
        """{stats, age_seconds, ok} -- the cache, never a network call."""
        now = self._clock()
        with self._lock:
            stats = {h: dict(v) for h, v in self._hosts.items()}
            ages = [now - t for t in self._at.values()]
            error = self._error
        age = max(ages) if ages else None
        ok = bool(stats) and error is None \
            and age is not None and age <= self._stale_after
        out = {"stats": stats, "age_seconds": age, "ok": ok}
        if error:
            out["error"] = error
        elif stats and age is not None and age > self._stale_after:
            out["error"] = (f"last reading is {int(age)}s old, past the "
                            f"{int(self._stale_after)}s threshold")
        elif not stats:
            out["error"] = "no node has answered yet"
        return out


def _registry_targets():
    """Which hosts to poll: the fleet registry, read fresh each sweep.

    Lazy import so this module stays importable (and testable) on a box
    with no registry at all.
    """
    import modelctl_fleet
    return poll_targets(modelctl_fleet.load_fleet())


def node_rows(nodes, presence_state_fn, presence, stats, now=None):
    """Registry + presence + telemetry -> the rows the card renders.

    One row per REGISTERED remote node, whether or not it answered:
    presence and budget come from the registry and are always knowable,
    telemetry is attached where a poll succeeded. A node the poller has
    never reached is therefore a row with an em dash where its numbers
    would be, which is the honest rendering -- not a missing row, which
    would read as "no such node".

    `presence_state_fn` is passed in rather than imported. The tri-state
    lives in modelctl_web.fleet, and a core module reaching up into the
    web package to borrow it would invert the dependency the layering
    check exists to keep one-way.
    """
    stats = stats or {}
    rows = []
    for node in nodes or []:
        probe = (presence or {}).get(node.name)
        state, _detail = presence_state_fn(probe, now=now)
        host = str(getattr(node, "host", "") or "")
        unit = unit_for_node(node)
        host_stats = stats.get(host) or {}
        unit_stats = (host_stats.get("units") or {}).get(unit) or {}
        device = node.devices[0] if node.devices else None
        kind = (device.kind if device else node.variant) or ""
        row = {
            "name": node.name,
            "kind": "gpu" if kind == "gpu" else "cpu",
            "host": host,
            "unit": unit,
            "device": device.name if device else "",
            "budget_bytes": int(device.budget_bytes) if device else None,
            "presence": state,
            "present": state == PRESENT,
            "pin_agrees": bool(probe.pin_agrees) if probe else False,
            "protocol": (probe.protocol if probe else "") or "",
            "polled": bool(host),
            "active_state": unit_stats.get("active_state"),
            "unit_memory_bytes": unit_stats.get("memory_bytes"),
            "unit_memory_max_bytes": unit_stats.get("memory_max_bytes"),
            "host_load1": host_stats.get("load1"),
            "host_nproc": host_stats.get("nproc"),
            "host_mem_total_bytes": host_stats.get("mem_total_bytes"),
            "host_mem_available_bytes": host_stats.get("mem_available_bytes"),
        }
        # The GPU block belongs to the host, but only a gpu-kind node has
        # any business rendering it: attaching it to the cpu node's row
        # would put the laptop's 4080 on a card about a cgroup.
        gpu = host_stats.get("gpu") if row["kind"] == "gpu" else None
        if gpu:
            row["gpu_used_bytes"] = gpu.get("used_bytes")
            row["gpu_total_bytes"] = gpu.get("total_bytes")
            row["gpu_util_pct"] = gpu.get("util_pct")
            row["gpu_temp_c"] = gpu.get("temp_c")
        rows.append(row)
    return rows


def summarize(rows, snap):
    """Rows + poller snapshot -> the `node_stats` block of one tick.

    The summary counters live here rather than in the renderer so the
    front page and any future consumer agree on what "pins agree" means
    without each deciding for itself.
    """
    protocols = sorted({r["protocol"] for r in rows if r["protocol"]})
    return {
        "nodes": rows,
        "age_seconds": snap.get("age_seconds"),
        "ok": bool(snap.get("ok")),
        "present": sum(1 for r in rows if r["present"]),
        # A single disagreeing pin makes the whole summary line say so: a
        # mismatched node answers a handshake in milliseconds and is
        # unusable, and that has to be visible from the front page
        # without navigating to fleet.
        "pins_agree": all(r["pin_agrees"] for r in rows) if rows else True,
        "protocol": protocols[0] if len(protocols) == 1 else "",
    }
