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
    """One device a node offers, plus what we are willing to use of it."""
    name: str            # remote-side device name as the client sees it
    kind: str            # "gpu" | "cpu"
    total_bytes: int     # what the node reports it has
    budget_bytes: int    # operator ceiling; admission never exceeds this

    def to_dict(self):
        return {"name": self.name, "kind": self.kind,
                "total_bytes": self.total_bytes,
                "budget_bytes": self.budget_bytes}

    @staticmethod
    def from_dict(d):
        return FleetDevice(
            name=d["name"], kind=d.get("kind", "gpu"),
            total_bytes=int(d.get("total_bytes", 0)),
            budget_bytes=int(d.get("budget_bytes", 0)))


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


# --- placement ------------------------------------------------------

def contiguous_range(first_layer: int, last_layer: int) -> str:
    """An -ot regex matching exactly the layers first..last inclusive.

    Spelled as an explicit alternation of indices rather than a clever
    numeric range pattern: `blk\\.(1[6-9]|2[0-3])\\.` is the kind of
    thing that silently also matches blk.160 on a model with more
    layers. An alternation of exact indices, anchored on the trailing
    dot, cannot.
    """
    if last_layer < first_layer:
        raise ValueError(
            f"empty layer range {first_layer}..{last_layer}")
    idx = "|".join(str(i) for i in range(first_layer, last_layer + 1))
    return rf"blk\.({idx})\.=%s"


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
