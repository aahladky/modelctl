"""Fleet registry: remote llama.cpp RPC nodes as optional planning targets.

A fleet node is another machine running `ggml-rpc-server` from the same
pinned llama.cpp commit this checkout builds. The rig attaches one with
`--rpc host:port`; llama.cpp then registers its devices as RPC0, RPC1,
... in the order the endpoints were passed, and any tensor can be placed
on them with a normal -ot rule.

Three things this module is deliberately strict about:

* **Optional, never required.** Nothing here may make a launch that
  worked without a node stop working. The planner adds RPC variants on
  top of the local plan set; with every node absent the compiled plans
  are byte-identical to a fleet-free checkout. That property has a test.

* **No network I/O during planning.** `compile_launch_plans` is
  documented deterministic and side-effect free, and the suite is
  hermetic. Reachability is probed by an explicit `probe_node()` call
  (CLI, cron, the operator), which *records* what it saw; planning reads
  that stored presence record and never opens a socket. A node that has
  never been probed is treated as absent, not as present.

* **Budgets are operator-declared ceilings, not measurements.** A node
  reports how much memory its device has; what we are willing to use is
  a separate number in the registry. Admission charges the declared
  budget through the same `_usable_vram_map` / `_admission_overflow`
  path local GPUs go through, so an over-budget remote placement is
  refused at plan time exactly like an over-budget local one.

`ggml-rpc-server` links only ggml -- never libcommon -- so it has no
`--version` and no embedded build-info. Two different things therefore
answer two different questions:

* the wire HELLO handshake gives the RPC *protocol* version, which is
  what governs whether client and server can talk at all;
* the *pin* (which commit the node's binary was built from) is recorded
  in the registry when the node is enrolled, because the protocol
  carries no such field. `pin_agrees` compares it to the commit this
  checkout pins, and a node whose pin disagrees is reported, not used.
"""
import json
import os
import socket
import struct
import time
from dataclasses import dataclass, field, replace

import modelctl_fsutil
import modelctl_paths

FLEET_PATH = modelctl_paths.STATE_DIR / "fleet.json"
PRESENCE_PATH = modelctl_paths.STATE_DIR / "fleet-presence.json"

# Wire constants, read off ggml/src/ggml-rpc/ggml-rpc.cpp at the pin.
# RPC_CMD_HELLO is 14 and static_assert'd to stay 14 upstream, precisely
# so an old client can still introduce itself to a newer server.
RPC_CMD_HELLO = 14
RPC_CONN_CAPS_SIZE = 24          # transport.h
_HELLO_RSP_SIZE = 4 + RPC_CONN_CAPS_SIZE   # major/minor/patch/padding + caps

# A presence record older than this is stale: the planner treats the node
# as absent rather than trusting a reading from an hour ago. Deliberately
# generous -- the cost of a stale "present" is one refused launch that
# falls back, and the cost of a stale "absent" is a lost optimization.
PRESENCE_TTL_SECONDS = 900


@dataclass(frozen=True)
class FleetDevice:
    """One device a node offers, plus what we are willing to use of it.

    Three different numbers, and conflating any two of them is a bug:

    * `total_bytes` is what the node reports the device physically has;
    * `cap_bytes` is the hard limit the node's own OS enforces on the
      unit serving this device -- systemd `MemoryMax` for a cpu unit. It
      is operator-recorded, not probed: reading it means asking another
      machine's service manager, and nothing in the planning path may
      shell out over SSH. Zero means "not recorded", which reads as "the
      device total is the only limit we know of";
    * `budget_bytes` is what we are willing to place, which admission
      charges. It is the only one of the three an operator edits here.

    A budget above the cap is not a policy, it is a mistake that becomes
    an OOM kill on the far side of a 2.5GbE link -- `device_ceiling`
    exists so the setter can refuse it before it is written.
    """
    name: str            # remote-side device name as the client sees it
    kind: str            # "gpu" | "cpu"
    total_bytes: int     # what the node reports it has
    budget_bytes: int    # operator ceiling; admission never exceeds this
    cap_bytes: int = 0   # OS-enforced limit on the serving unit (0 = unknown)

    def to_dict(self):
        return {"name": self.name, "kind": self.kind,
                "total_bytes": self.total_bytes,
                "budget_bytes": self.budget_bytes,
                "cap_bytes": self.cap_bytes}

    @staticmethod
    def from_dict(d):
        return FleetDevice(
            name=d["name"], kind=d.get("kind", "gpu"),
            total_bytes=int(d.get("total_bytes", 0)),
            budget_bytes=int(d.get("budget_bytes", 0)),
            cap_bytes=int(d.get("cap_bytes", 0)))


@dataclass(frozen=True)
class FleetNode:
    name: str            # "ph16-71"
    host: str            # LAN address; never a VPN address
    port: int
    variant: str         # "cuda" | "cpu" -- which build serves this port
    pin: str             # commit the node's ggml-rpc-server was built from
    devices: tuple = ()
    enabled: bool = True
    note: str = ""

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def to_dict(self):
        return {"name": self.name, "host": self.host, "port": self.port,
                "variant": self.variant, "pin": self.pin,
                "enabled": self.enabled, "note": self.note,
                "devices": [d.to_dict() for d in self.devices]}

    @staticmethod
    def from_dict(d):
        return FleetNode(
            name=d["name"], host=d["host"], port=int(d["port"]),
            variant=d.get("variant", ""), pin=d.get("pin", ""),
            devices=tuple(FleetDevice.from_dict(x)
                          for x in d.get("devices", [])),
            enabled=bool(d.get("enabled", True)),
            note=d.get("note", ""))


@dataclass(frozen=True)
class NodeProbe:
    """What one probe of one node actually observed."""
    node: str
    endpoint: str
    reachable: bool
    protocol: str = ""       # "5.0.0" from the HELLO response
    pin: str = ""            # registry-recorded build commit
    pin_agrees: bool = False
    detail: str = ""
    probed_at: float = 0.0

    def to_dict(self):
        return {"node": self.node, "endpoint": self.endpoint,
                "reachable": self.reachable, "protocol": self.protocol,
                "pin": self.pin, "pin_agrees": self.pin_agrees,
                "detail": self.detail, "probed_at": self.probed_at}

    @staticmethod
    def from_dict(d):
        return NodeProbe(
            node=d.get("node", ""), endpoint=d.get("endpoint", ""),
            reachable=bool(d.get("reachable")),
            protocol=d.get("protocol", ""), pin=d.get("pin", ""),
            pin_agrees=bool(d.get("pin_agrees")),
            detail=d.get("detail", ""),
            probed_at=float(d.get("probed_at", 0.0)))

    @property
    def usable(self) -> bool:
        """Reachable AND running the commit this checkout pins.

        A node on a different commit is not a planning target: the whole
        point of the pin is that both ends of an RPC graph agree on the
        kernels. It still shows up in status output, flagged.
        """
        return self.reachable and self.pin_agrees


def admission_key(node_name: str, device_name: str) -> str:
    """The device key a remote device is admitted under.

    Namespaced so it can never collide with a local device id ("SYCL0",
    "CUDA0"): a remote budget must not silently satisfy a claim that was
    computed against a local card.
    """
    return f"RPC:{node_name}:{device_name}"


def is_remote_key(device_key: str) -> bool:
    return str(device_key).startswith("RPC:")


# --- registry -------------------------------------------------------

def load_fleet(path=None) -> list:
    """Registered nodes, or [] when no registry exists.

    Never raises on a malformed file: a corrupt registry must degrade to
    "no fleet" (which is always safe -- everything falls back to local)
    rather than break every launch path that consults it.
    """
    p = path or FLEET_PATH
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(raw, dict):
        # Valid JSON of the wrong shape ("[]", "null", "42") reached
        # `raw.get` and raised AttributeError, straight through the
        # "never raises" promise above and out of every caller that
        # trusts it -- including both budget builders, where it would
        # fail a routing or launch job with a traceback instead of
        # degrading to local-only budgets.
        return []
    nodes = []
    for entry in raw.get("nodes", []):
        try:
            nodes.append(FleetNode.from_dict(entry))
        except (KeyError, TypeError, ValueError):
            continue
    return nodes


def save_fleet(nodes, path=None):
    p = path or FLEET_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "nodes": [n.to_dict() for n in nodes]}
    modelctl_fsutil.atomic_write_text(
        p, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def enabled_nodes(nodes=None) -> list:
    return [n for n in (load_fleet() if nodes is None else nodes)
            if n.enabled]


def node_by_name(name, nodes=None):
    for n in (load_fleet() if nodes is None else nodes):
        if n.name == name:
            return n
    return None


# --- the pin --------------------------------------------------------

def local_pin() -> str:
    """The llama.cpp commit this checkout pins.

    MODELCTL_FLEET_PIN overrides it -- tests set it, and so can an
    operator driving a checkout whose submodule is mid-bump. Otherwise
    read the superproject's recorded submodule pin with git, which is
    the same number `git ls-tree HEAD llama.cpp` shows.
    """
    override = os.environ.get("MODELCTL_FLEET_PIN")
    if override:
        return override.strip()
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "-C", repo, "ls-tree", "HEAD", "llama.cpp"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            # "160000 commit <sha>\tllama.cpp"
            return out.stdout.split()[2]
    except Exception:
        pass
    return ""


# --- the wire probe -------------------------------------------------

def _default_connect(host, port, timeout):
    return socket.create_connection((host, port), timeout=timeout)


def _recv_exactly(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"connection closed after {len(buf)} of {n} bytes")
        buf += chunk
    return buf


def hello(sock) -> tuple:
    """Perform the RPC HELLO handshake; return (major, minor, patch).

    Wire format, from ggml-rpc.cpp:
        request : cmd (1 byte) | size (8 bytes LE) | payload
        response:               size (8 bytes LE) | payload

    We advertise all-zero conn_caps: that means "no transport upgrades
    offered", so the server answers with its own caps and the socket
    stays a plain one. A probe has no business negotiating a shared
    memory transport it will never use.
    """
    req = bytes([RPC_CMD_HELLO]) + struct.pack("<Q", RPC_CONN_CAPS_SIZE) \
        + bytes(RPC_CONN_CAPS_SIZE)
    sock.sendall(req)
    (size,) = struct.unpack("<Q", _recv_exactly(sock, 8))
    if size != _HELLO_RSP_SIZE:
        raise ValueError(
            f"HELLO response size {size}, expected {_HELLO_RSP_SIZE} -- "
            "the node is not a matching ggml-rpc-server")
    body = _recv_exactly(sock, size)
    major, minor, patch = body[0], body[1], body[2]
    return major, minor, patch


def probe_node(node, expected_pin=None, timeout=3.0, connect=None,
               now=None) -> NodeProbe:
    """Reachability + protocol version + pin agreement for one node.

    `connect` is injectable so the tests can drive this against a fake
    socket without opening one; production passes nothing and gets a
    real TCP connection.
    """
    connect = connect or _default_connect
    expected = local_pin() if expected_pin is None else expected_pin
    stamp = time.time() if now is None else now
    agrees = bool(node.pin) and bool(expected) and node.pin == expected

    sock = None
    try:
        sock = connect(node.host, node.port, timeout)
        major, minor, patch = hello(sock)
    except Exception as e:
        return NodeProbe(
            node=node.name, endpoint=node.endpoint, reachable=False,
            pin=node.pin, pin_agrees=agrees,
            detail=f"{type(e).__name__}: {e}", probed_at=stamp)
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    detail = ""
    if not node.pin:
        detail = "node has no recorded build pin; enroll it with one"
    elif not expected:
        detail = "this checkout's llama.cpp pin could not be determined"
    elif not agrees:
        detail = (f"node built at {node.pin[:12]}, this checkout pins "
                  f"{expected[:12]}")
    return NodeProbe(
        node=node.name, endpoint=node.endpoint, reachable=True,
        protocol=f"{major}.{minor}.{patch}", pin=node.pin,
        pin_agrees=agrees, detail=detail, probed_at=stamp)


def probe_fleet(nodes=None, **kw) -> list:
    return [probe_node(n, **kw) for n in (load_fleet() if nodes is None
                                          else nodes)]


# --- presence: the stored planning input ----------------------------

def load_presence(path=None) -> dict:
    """{node_name: NodeProbe} as last recorded. Missing file -> {}."""
    p = path or PRESENCE_PATH
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for name, entry in (raw.get("nodes") or {}).items():
        try:
            out[name] = NodeProbe.from_dict(entry)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def save_presence(probes, path=None):
    p = path or PRESENCE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1,
               "nodes": {pr.node: pr.to_dict() for pr in probes}}
    modelctl_fsutil.atomic_write_text(
        p, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def refresh_presence(nodes=None, **kw) -> list:
    """Probe every node and record the result. The one network call."""
    probes = probe_fleet(nodes, **kw)
    save_presence(probes)
    return probes


def ensure_fresh_presence(nodes=None, now=None, ttl=PRESENCE_TTL_SECONDS,
                          **kw):
    """Refresh presence for enabled nodes whose record is missing or
    stale, then return the usable node list.

    Planning entry points call this so a plan set is never silently
    fleet-blind: the 2026-08-03 baseline compiled local-only plans for a
    122B purely because nobody had opened the fleet page inside the
    presence TTL. Bounded work: one probe (3 s TCP hello) per stale
    node, no sockets at all while records are fresh. Fresh records are
    preserved on save, mirroring the single-node probe route.
    """
    enabled = enabled_nodes(nodes)
    if not enabled:
        return []
    presence = load_presence()
    stamp = time.time() if now is None else now
    stale = [n for n in enabled
             if n.name not in presence
             or stamp - presence[n.name].probed_at > ttl]
    if stale:
        recorded = dict(presence)
        for node in stale:
            try:
                recorded[node.name] = probe_node(node, **kw)
            except Exception:
                continue
        save_presence(list(recorded.values()))
        presence = recorded
    return usable_nodes(enabled, presence=presence, now=now, ttl=ttl)


def usable_nodes(nodes=None, presence=None, now=None,
                 ttl=PRESENCE_TTL_SECONDS) -> list:
    """Enabled nodes whose last recorded probe says present, agreeing, fresh.

    This is what planning consults. It opens no socket: a node nobody has
    probed is absent as far as the planner is concerned, which is the
    fail-safe direction -- the plan set simply stays local.
    """
    nodes = enabled_nodes(nodes)
    presence = load_presence() if presence is None else presence
    stamp = time.time() if now is None else now
    out = []
    for n in nodes:
        pr = presence.get(n.name)
        if pr is None or not pr.usable:
            continue
        if ttl is not None and stamp - pr.probed_at > ttl:
            continue
        out.append(n)
    return out


# --- budgets into the existing admission machinery ------------------

def admission_budgets(nodes=None, presence=None, now=None) -> dict:
    """{admission_key: budget_bytes} for every device on every usable node.

    Merged into `_usable_vram_map` so a remote placement is admitted --
    or refused -- by exactly the code path that admits a local one.
    """
    out = {}
    for n in usable_nodes(nodes, presence, now):
        for d in n.devices:
            if d.budget_bytes > 0:
                out[admission_key(n.name, d.name)] = int(d.budget_bytes)
    return out


def budget_input(nodes=None) -> dict:
    """{admission_key: budget_bytes} for every ENABLED node's devices.

    The planning-input twin of `admission_budgets`, and deliberately not
    the same function. Admission asks "what may this plan spend right
    now", so it reads presence and a node nobody can reach contributes
    nothing. A recorded planning input asks "what were the operator's
    declared ceilings when this plan was built", which must not move
    because a laptop was closed: presence in the record would make every
    probe look like an input change and every profile permanently
    drifted. So this reads the registry only.
    """
    out = {}
    for n in enabled_nodes(nodes):
        for d in n.devices:
            out[admission_key(n.name, d.name)] = int(d.budget_bytes)
    return out


# --- the ceiling, and the one writer of a budget --------------------

# What a budget may not claim, over and above the OS-enforced cap. The
# ggml-rpc-server process is not only its tensor buffers: it has a
# resident set of its own (0.4-1 GiB observed on ph16-71), it copies
# tensors through send/recv staging buffers, and on a cpu unit it shares
# the cgroup with nothing else, so overshooting is an OOM kill rather
# than a swap-out. A flat floor covers the small devices, the fraction
# covers the large ones.
HEADROOM_FRACTION = 0.05
HEADROOM_MIN_BYTES = 1 << 30


def runtime_headroom(base_bytes: int) -> int:
    return max(HEADROOM_MIN_BYTES, int(base_bytes * HEADROOM_FRACTION))


def device_ceiling(device) -> tuple:
    """(ceiling_bytes, basis) -- the largest budget this device may hold.

    The basis is named because a refusal that says only "too big" leaves
    the operator guessing which of three numbers to go look at:

    * a cpu device is limited by the systemd `MemoryMax` on its unit,
      because that is what will actually kill it, and the machine's RAM
      total is irrelevant next to a cgroup limit;
    * a gpu device is limited by what the node reports the card has;
    * a cpu device with no recorded cap falls back to the reported
      total and says so -- a guess would silently authorize a budget the
      cgroup will refuse at load time.
    """
    if device.kind == "cpu" and device.cap_bytes > 0:
        base, basis = device.cap_bytes, "systemd MemoryMax"
    elif device.kind == "cpu":
        base, basis = device.total_bytes, "reported total (no MemoryMax recorded)"
    else:
        base, basis = device.total_bytes, "reported device total"
    return max(0, base - runtime_headroom(base)), basis


def _find(nodes, node_name, device_name) -> tuple:
    for i, n in enumerate(nodes):
        if n.name != node_name:
            continue
        for j, d in enumerate(n.devices):
            if d.name == device_name:
                return i, j
        raise KeyError(
            f"node '{node_name}' has no device '{device_name}' "
            f"(it has: {', '.join(d.name for d in n.devices) or 'none'})")
    raise KeyError(f"no fleet node named '{node_name}'")


def _replace_device(nodes, i, j, **changes) -> list:
    node = nodes[i]
    devices = list(node.devices)
    devices[j] = replace(devices[j], **changes)
    out = list(nodes)
    out[i] = replace(node, devices=tuple(devices))
    return out


def set_device_budget(node_name, device_name, budget_bytes, path=None) -> dict:
    """Set one device's operator budget. The only writer of that number.

    Read-modify-write under `state_lock`, so two concurrent setters
    serialize instead of each writing a whole registry computed from the
    same stale read (which would silently drop one of the two edits --
    the registry is one JSON document, not a row per device).

    Refuses a budget over `device_ceiling`, naming the ceiling and its
    basis. Returns the change, including the profiles whose recorded
    planning inputs no longer match the fleet budgets in force -- a
    budget IS a planning input, so moving one stales every stored-input
    plan built against the old number exactly as a hardware change does.
    """
    budget_bytes = int(budget_bytes)
    if budget_bytes < 0:
        raise ValueError(f"budget must not be negative, got {budget_bytes}")
    with modelctl_fsutil.state_lock():
        nodes = load_fleet(path)
        i, j = _find(nodes, node_name, device_name)
        device = nodes[i].devices[j]
        ceiling, basis = device_ceiling(device)
        if budget_bytes > ceiling:
            raise ValueError(
                f"budget {_gib(budget_bytes)} GiB exceeds the ceiling for "
                f"{node_name}/{device_name}: {_gib(ceiling)} GiB "
                f"({basis} {_gib(_ceiling_base(device))} GiB minus "
                f"{_gib(runtime_headroom(_ceiling_base(device)))} GiB runtime "
                f"headroom). Raise the cap on the node first, or ask for less.")
        previous = device.budget_bytes
        if budget_bytes != previous:
            save_fleet(_replace_device(nodes, i, j,
                                       budget_bytes=budget_bytes), path)
        staled = stale_input_profiles()
    return {"node": node_name, "device": device_name,
            "previous_bytes": previous, "budget_bytes": budget_bytes,
            "ceiling_bytes": ceiling, "ceiling_basis": basis,
            "changed": budget_bytes != previous,
            "staled_profiles": staled}


def set_device_cap(node_name, device_name, cap_bytes, path=None) -> dict:
    """Record the OS-enforced limit on the unit serving one device.

    Separate from `set_device_budget` because the cap and the budget are
    different facts with different owners: the cap is what the node's
    service manager enforces (an operator changed it over there, this
    only writes down what it now says), the budget is what we choose to
    spend under it. Recording a cap never moves a budget -- but a cap
    BELOW the budget in force is refused, because writing it would leave
    a registry whose own setter could not reproduce it.
    """
    cap_bytes = int(cap_bytes)
    if cap_bytes < 0:
        raise ValueError(f"cap must not be negative, got {cap_bytes}")
    with modelctl_fsutil.state_lock():
        nodes = load_fleet(path)
        i, j = _find(nodes, node_name, device_name)
        device = nodes[i].devices[j]
        ceiling, _ = device_ceiling(replace(device, cap_bytes=cap_bytes))
        if device.budget_bytes > ceiling:
            raise ValueError(
                f"cap {_gib(cap_bytes)} GiB would put the ceiling at "
                f"{_gib(ceiling)} GiB, below {node_name}/{device_name}'s "
                f"budget of {_gib(device.budget_bytes)} GiB; lower the "
                f"budget first")
        previous = device.cap_bytes
        if cap_bytes != previous:
            save_fleet(_replace_device(nodes, i, j, cap_bytes=cap_bytes), path)
    return {"node": node_name, "device": device_name,
            "previous_bytes": previous, "cap_bytes": cap_bytes,
            "ceiling_bytes": ceiling, "changed": cap_bytes != previous}


def _ceiling_base(device) -> int:
    if device.kind == "cpu" and device.cap_bytes > 0:
        return device.cap_bytes
    return device.total_bytes


def _gib(n) -> str:
    return f"{n / (1 << 30):.2f}"


def stale_input_profiles(live=None) -> list:
    """Profiles whose recorded planning inputs disagree with the fleet
    budgets in force now.

    Stored planning inputs are sticky on purpose -- that is the whole
    point of recording them -- so a budget change does not rewrite any
    profile. What it does is make the record disagree with the machine,
    and this is the list of profiles where it now does. A profile whose
    record predates fleet budgets makes no claim and is not stale; a
    profile with no recorded inputs at all has nothing to invalidate.
    """
    import modelctl
    import modelctl_tiers
    live = budget_input() if live is None else live
    out = []
    for profile in modelctl._read_all_profiles():
        stored = modelctl_tiers.stored_planning_inputs(profile)
        if not stored:
            continue
        recorded = modelctl_tiers.planning_input_fleet_budgets(stored)
        if recorded is None:
            continue  # recorded before budgets were an input: no claim
        drift = {k: (recorded.get(k), live.get(k))
                 for k in set(recorded) | set(live)
                 if recorded.get(k) != live.get(k)}
        if drift:
            out.append({"name": profile.get("name", ""),
                        "changed": {k: {"recorded": a, "live": b}
                                    for k, (a, b) in sorted(drift.items())}})
    return out


# --- placement ------------------------------------------------------

def contiguous_range(first_layer: int, last_layer: int) -> str:
    """An -ot regex for the ROUTED EXPERTS of layers first..last inclusive.

    Routed experts only, never the whole layer. A layer also holds norm
    and attention tensors, and the RPC buffer cannot run those ops, so a
    whole-layer pattern aborts llama.cpp at load:

        ggml-backend.cpp:934: pre-allocated tensor
        (blk.42.attn_norm.weight) in a buffer (RPC0[192.168.0.76:50052])
        that cannot run the operation (NONE)

    -- observed live 2026-08-04, and it blocked every plan this function
    feeds. `ffn_.*_exps` is the calibration-proven placement, and the
    same rule modelctl_tiers emits for the ladder plans that do serve.
    It excludes shared experts (`_shexp`) on purpose: those run every
    token and belong with the layer's attention on the local card.

    The layer set is spelled as an explicit alternation of indices
    rather than a clever numeric range pattern: `blk\\.(1[6-9]|2[0-3])\\.`
    is the kind of thing that silently also matches blk.160 on a model
    with more layers. An alternation of exact indices, anchored on the
    trailing dot, cannot.
    """
    if last_layer < first_layer:
        raise ValueError(
            f"empty layer range {first_layer}..{last_layer}")
    idx = "|".join(str(i) for i in range(first_layer, last_layer + 1))
    return rf"blk\.({idx})\.ffn_.*_exps=%s"


def placement_args(rpc_config: dict) -> list:
    """llama-server tokens for a plan's RPC placement, or [].

    Shape of rpc_config (carried in the plan's merged config under
    "rpc", so it flows through the one build_server_args path):

        {"endpoints": ["192.168.0.76:50052"],
         "placements": [{"buffer_type": "RPC0[192.168.0.76:50052]",
                         "first_layer": 16, "last_layer": 23}]}

    Endpoint order is load-bearing: it decides the device numbering, and
    the buffer-type names in `placements` are only meaningful against
    this same list.
    """
    endpoints = list(rpc_config.get("endpoints") or [])
    if not endpoints:
        return []
    args = ["--rpc", ",".join(endpoints)]
    for pl in rpc_config.get("placements") or []:
        pattern = contiguous_range(
            int(pl["first_layer"]), int(pl["last_layer"]))
        args.extend(["-ot", pattern % pl["buffer_type"]])
    return args


def device_name_for(endpoints, endpoint, device_index=0) -> str:
    """The client-side RPC *device* name for one endpoint.

    Device ids are assigned GLOBALLY in --rpc order, walking each
    endpoint's own device list; with one device per endpoint (our units
    serve exactly one each) that is just the endpoint's position. This
    is what --device would take.
    """
    base = 0
    for ep in endpoints:
        if ep == endpoint:
            return f"RPC{base + device_index}"
        base += 1
    raise ValueError(f"endpoint {endpoint} not in {endpoints}")


def buffer_type_for(endpoint, device_index=0) -> str:
    """The client-side RPC *buffer type* name -- what -ot actually takes.

    Not the same string as the device name, and the difference bites:
    buffer-type numbering is per-endpoint, not global, so attaching two
    nodes gives you two buffer types both called RPC0, disambiguated
    only by the bracketed endpoint::

        Available buffer types:
          CPU
          RPC0[192.168.0.76:50052]
          RPC0[192.168.0.76:50053]

    while their device names are RPC0 and RPC1. Passing a device name to
    -ot fails with "unknown buffer type".
    """
    return f"RPC{device_index}[{endpoint}]"
