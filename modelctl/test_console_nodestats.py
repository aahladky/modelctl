"""The remote-node telemetry poller and the operate page's fleet card.

Two questions, the same two the rest of the console surface is held to.

"Does an absent reading render as absent?" -- this module's whole reason
to be careful is that a card of live numbers about another machine has
exactly one catastrophic failure mode, which is printing a plausible
number when it could not ask. Every test here that covers a missing
field asserts the key is ABSENT or None, never 0.

"Can the suite open a connection?" -- no. The subprocess boundary is
faked in every test, and the one test that checks the rejection path for
a bad unit name asserts the runner was never called at all: a name that
fails the allowlist must not reach the command string even in a form
that is then discarded.

Carries its own isolation so single-file runs never touch real state.
"""
import os
import subprocess
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

_TMP = TemporaryDirectory()
os.environ.setdefault("MODELCTL_HOME", _TMP.name)

import modelctl_fleet as fleet
import modelctl_nodestats as ns
from modelctl_web import fleet as fleetview
from modelctl_web import telemetry

GIB = 1 << 30
MIB = 1 << 20
PIN = "85b7e6556b6b83026d1a17df2635bc1173db1f97"
OTHER_PIN = "0" * 40

# What the far side actually printed on 2026-08-02, trimmed to the
# sections the template asks for. Verbatim rather than invented: the
# parser's job is to read this exact shape.
GOOD = """@@load
0.00 0.15 0.34 1/616 1691767
@@mem
MemTotal:       32060152 kB
MemAvailable:   29849272 kB
@@nproc
32
@@unit rpc-cuda0.service
ActiveState=active
MemoryCurrent=299642880
MemoryMax=21474836480
@@unit rpc-cpu0.service
ActiveState=active
MemoryCurrent=1011712
MemoryMax=27917287424
@@gpu
232, 12282, 0, 40
@@end
"""

# The same laptop with no nvidia-smi installed: the marker is printed,
# the section is empty. Not an error -- the cpu-only case.
NO_NVIDIA = """@@load
0.00 0.15 0.34 1/616 1691767
@@mem
MemTotal:       32060152 kB
MemAvailable:   29849272 kB
@@nproc
32
@@unit rpc-cpu0.service
ActiveState=active
MemoryCurrent=1011712
MemoryMax=27917287424
@@gpu
@@end
"""


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def runner_for(stdout="", stderr="", returncode=0, record=None):
    """A stand-in for subprocess.run that records how it was called."""
    def run(argv, **kw):
        if record is not None:
            record.append((argv, kw))
        return FakeProc(stdout, stderr, returncode)
    return run


def cuda_node(pin=PIN, host="192.168.0.76"):
    return fleet.FleetNode(
        name="ph16-71-cuda0", host=host, port=50052, variant="cuda", pin=pin,
        devices=(fleet.FleetDevice(name="CUDA0", kind="gpu",
                                   total_bytes=12452888576,
                                   budget_bytes=10 * GIB),))


def cpu_node(pin=PIN, host="192.168.0.76"):
    return fleet.FleetNode(
        name="ph16-71-cpu0", host=host, port=50053, variant="cpu", pin=pin,
        devices=(fleet.FleetDevice(name="CPU", kind="cpu",
                                   total_bytes=32828817408,
                                   budget_bytes=16 * GIB,
                                   cap_bytes=27917287424),))


def probe(node, reachable=True, pin_agrees=True, at=None, protocol="5.0.0"):
    import time
    return fleet.NodeProbe(node=node, endpoint="x", reachable=reachable,
                           protocol=protocol, pin=PIN, pin_agrees=pin_agrees,
                           probed_at=time.time() if at is None else at)


class TestParse(unittest.TestCase):
    """The parser, against the shape the machine really prints."""

    def test_full_good_response(self):
        got = ns.parse_host_output(GOOD)
        self.assertEqual(got["load1"], 0.00)
        self.assertEqual(got["mem_total_bytes"], 32060152 * 1024)
        self.assertEqual(got["mem_available_bytes"], 29849272 * 1024)
        self.assertEqual(got["nproc"], 32)
        self.assertEqual(got["gpu"], {"used_bytes": 232 * MIB,
                                      "total_bytes": 12282 * MIB,
                                      "util_pct": 0, "temp_c": 40})
        self.assertEqual(got["units"]["rpc-cuda0.service"], {
            "active_state": "active", "memory_bytes": 299642880,
            "memory_max_bytes": 21474836480})
        # The number the registry's note contradicted: 26.00 GiB, not 20G.
        self.assertEqual(
            got["units"]["rpc-cpu0.service"]["memory_max_bytes"],
            27917287424)

    def test_missing_nvidia_smi_is_not_an_error_and_yields_no_gpu(self):
        got = ns.parse_host_output(NO_NVIDIA)
        self.assertNotIn("gpu", got)
        # every other section still parsed
        self.assertEqual(got["nproc"], 32)
        self.assertEqual(got["load1"], 0.00)

    def test_unreadable_field_is_absent_never_zero(self):
        """systemd's "[not set]" / "infinity" are not readings."""
        got = ns.parse_host_output(
            "@@unit rpc-cpu0.service\n"
            "ActiveState=active\n"
            "MemoryCurrent=[not set]\n"
            "MemoryMax=infinity\n"
            "@@end\n")
        unit = got["units"]["rpc-cpu0.service"]
        self.assertEqual(unit["active_state"], "active")
        self.assertNotIn("memory_bytes", unit)
        self.assertNotIn("memory_max_bytes", unit)
        self.assertIsNone(unit.get("memory_bytes"))

    def test_absent_sections_leave_no_keys(self):
        got = ns.parse_host_output("@@end\n")
        for key in ("load1", "mem_total_bytes", "mem_available_bytes",
                    "nproc", "gpu"):
            self.assertNotIn(key, got, f"{key} must be absent, not zero")

    def test_partial_gpu_line_does_not_invent_zeros(self):
        got = ns.parse_host_output("@@gpu\n232, [N/A], 0, 40\n@@end\n")
        self.assertNotIn("total_bytes", got["gpu"])
        self.assertEqual(got["gpu"]["used_bytes"], 232 * MIB)


class TestPollHost(unittest.TestCase):
    """The subprocess boundary: one round trip, fixed template, no shell."""

    def test_one_round_trip_for_two_units(self):
        calls = []
        out = ns.poll_host("192.168.0.76",
                           ["rpc-cuda0.service", "rpc-cpu0.service"],
                           runner=runner_for(GOOD, record=calls))
        self.assertTrue(out["ok"])
        self.assertEqual(len(calls), 1, "one ssh per host, not per unit")
        argv = calls[0][0]
        # argv, not a shell string: the host can never be a shell word.
        self.assertEqual(argv[0], "timeout")
        self.assertIn("ssh", argv)
        self.assertIn("192.168.0.76", argv)
        self.assertIn("-o", argv)
        self.assertIn("BatchMode=yes", argv)
        script = argv[-1]
        self.assertIn("rpc-cuda0.service", script)
        self.assertIn("rpc-cpu0.service", script)

    def test_bad_unit_name_is_rejected_before_any_process_runs(self):
        for bad in ["rpc-cpu0.service; rm -rf /", "$(id)", "a b", "a`id`",
                    "a|b", "a&b"]:
            calls = []
            out = ns.poll_host("192.168.0.76", [bad],
                               runner=runner_for(GOOD, record=calls))
            self.assertFalse(out["ok"], bad)
            self.assertIn("rejected", out["error"])
            self.assertEqual(calls, [],
                             f"a rejected unit name ({bad!r}) still ran a "
                             f"process")

    def test_ssh_timeout_reports_not_ok(self):
        def boom(argv, **kw):
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        out = ns.poll_host("192.168.0.76", ["rpc-cpu0.service"], runner=boom)
        self.assertFalse(out["ok"])
        self.assertIn("timed out", out["error"])

    def test_coreutils_timeout_exit_status_reports_not_ok(self):
        out = ns.poll_host("192.168.0.76", ["rpc-cpu0.service"],
                           runner=runner_for("", returncode=124))
        self.assertFalse(out["ok"])
        self.assertIn("timed out", out["error"])

    def test_unreachable_host_carries_ssh_stderr(self):
        out = ns.poll_host(
            "192.168.0.76", ["rpc-cpu0.service"],
            runner=runner_for("", stderr="ssh: connect to host port 22: "
                                         "No route to host", returncode=255))
        self.assertFalse(out["ok"])
        self.assertIn("No route to host", out["error"])

    def test_no_host_is_not_polled(self):
        calls = []
        out = ns.poll_host("", ["rpc-cpu0.service"],
                           runner=runner_for(GOOD, record=calls))
        self.assertFalse(out["ok"])
        self.assertEqual(calls, [])


class TestTargets(unittest.TestCase):
    def test_units_derived_from_node_names(self):
        self.assertEqual(ns.unit_for_node(cuda_node()), "rpc-cuda0.service")
        self.assertEqual(ns.unit_for_node(cpu_node()), "rpc-cpu0.service")

    def test_derived_unit_must_survive_the_allowlist(self):
        weird = fleet.FleetNode(name="node-with a space", host="h", port=1,
                                variant="cpu", pin=PIN)
        self.assertEqual(ns.unit_for_node(weird), "")

    def test_one_host_two_units(self):
        targets = ns.poll_targets([cuda_node(), cpu_node()])
        self.assertEqual(targets, {"192.168.0.76": ["rpc-cuda0.service",
                                                    "rpc-cpu0.service"]})

    def test_node_without_ssh_host_is_not_polled(self):
        self.assertEqual(ns.poll_targets([cpu_node(host="")]), {})


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class TestPoller(unittest.TestCase):
    """The cache and its staleness contract. No thread is started here."""

    def poller(self, clock, poll_fn, nodes=None):
        nodes = nodes if nodes is not None else [cuda_node(), cpu_node()]
        return ns.NodeStatsPoller(
            targets_fn=lambda: ns.poll_targets(nodes),
            poll_fn=poll_fn, clock=clock)

    def test_good_poll_is_served_as_ok(self):
        clock = Clock()
        p = self.poller(clock, lambda h, u: ns.poll_host(
            h, u, runner=runner_for(GOOD)))
        p.poll_once()
        snap = p.snapshot()
        self.assertTrue(snap["ok"])
        self.assertAlmostEqual(snap["age_seconds"], 0.0)
        self.assertEqual(snap["stats"]["192.168.0.76"]["nproc"], 32)

    def test_age_past_the_threshold_flips_ok_false(self):
        clock = Clock()
        p = self.poller(clock, lambda h, u: ns.poll_host(
            h, u, runner=runner_for(GOOD)))
        p.poll_once()
        clock.t += ns.STALE_AFTER_SECONDS - 1
        self.assertTrue(p.snapshot()["ok"], "not stale yet")
        clock.t += 2
        snap = p.snapshot()
        self.assertFalse(snap["ok"])
        self.assertIn("old", snap["error"])
        # The last good numbers are still served: the card ages them
        # under the stale treatment rather than blanking.
        self.assertEqual(snap["stats"]["192.168.0.76"]["nproc"], 32)

    def test_failed_poll_keeps_last_good_but_is_not_ok(self):
        clock = Clock()
        results = [ns.poll_host("h", [], runner=runner_for(GOOD))]

        def poll_fn(host, units):
            if results:
                return results.pop()
            return {"host": host, "ok": False, "error": "ssh timed out after 8s"}
        p = self.poller(clock, poll_fn)
        p.poll_once()
        self.assertTrue(p.snapshot()["ok"])
        p.poll_once()                       # now failing
        snap = p.snapshot()
        self.assertFalse(snap["ok"])
        self.assertIn("timed out", snap["error"])
        self.assertIn("192.168.0.76", snap["stats"])

    def test_never_polled_is_not_ok(self):
        p = self.poller(Clock(), lambda h, u: {"ok": False, "error": "x"})
        snap = p.snapshot()
        self.assertFalse(snap["ok"])
        self.assertIsNone(snap["age_seconds"])

    def test_deregistered_host_leaves_the_cache(self):
        clock = Clock()
        nodes = [cuda_node()]
        p = ns.NodeStatsPoller(targets_fn=lambda: ns.poll_targets(nodes),
                               poll_fn=lambda h, u: ns.poll_host(
                                   h, u, runner=runner_for(GOOD)),
                               clock=clock)
        p.poll_once()
        self.assertIn("192.168.0.76", p.snapshot()["stats"])
        nodes.clear()
        p.poll_once()
        self.assertEqual(p.snapshot()["stats"], {})


class TestRows(unittest.TestCase):
    """Registry + presence + telemetry, as the card consumes it."""

    def rows(self, nodes=None, presence=None, text=GOOD):
        nodes = nodes if nodes is not None else [cuda_node(), cpu_node()]
        stats = {"192.168.0.76": ns.poll_host(
            "192.168.0.76", ["rpc-cuda0.service", "rpc-cpu0.service"],
            runner=runner_for(text))}
        if presence is None:
            presence = {n.name: probe(n.name) for n in nodes}
        return ns.node_rows(nodes, fleetview.presence_state, presence, stats)

    def test_gpu_row_carries_vram_and_its_unit(self):
        gpu = self.rows()[0]
        self.assertEqual(gpu["kind"], "gpu")
        self.assertEqual(gpu["device"], "CUDA0")
        self.assertEqual(gpu["gpu_used_bytes"], 232 * MIB)
        self.assertEqual(gpu["gpu_total_bytes"], 12282 * MIB)
        self.assertEqual(gpu["gpu_util_pct"], 0)
        self.assertEqual(gpu["gpu_temp_c"], 40)
        self.assertEqual(gpu["unit_memory_max_bytes"], 21474836480)

    def test_cpu_row_carries_the_cgroup_and_the_host(self):
        cpu = self.rows()[1]
        self.assertEqual(cpu["kind"], "cpu")
        self.assertEqual(cpu["unit_memory_bytes"], 1011712)
        self.assertEqual(cpu["unit_memory_max_bytes"], 27917287424)
        self.assertEqual(cpu["host_nproc"], 32)
        self.assertEqual(cpu["host_load1"], 0.00)
        self.assertEqual(cpu["host_mem_total_bytes"], 32060152 * 1024)

    def test_gpu_block_is_not_attached_to_the_cpu_node(self):
        """One laptop, two nodes: the 4080 belongs to the cuda row only."""
        cpu = self.rows()[1]
        for key in ("gpu_used_bytes", "gpu_total_bytes", "gpu_util_pct",
                    "gpu_temp_c"):
            self.assertNotIn(key, cpu)

    def test_unpolled_node_still_renders_presence_and_budget(self):
        rows = ns.node_rows([cpu_node(host="")], fleetview.presence_state,
                            {"ph16-71-cpu0": probe("ph16-71-cpu0")}, {})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertFalse(row["polled"])
        self.assertTrue(row["present"])
        self.assertEqual(row["budget_bytes"], 16 * GIB)
        # ...and nothing it could not have measured
        self.assertIsNone(row["host_load1"])
        self.assertIsNone(row["unit_memory_bytes"])

    def test_unreached_node_is_a_row_with_absent_numbers_not_a_missing_row(self):
        rows = ns.node_rows([cuda_node(), cpu_node()],
                            fleetview.presence_state,
                            {}, {})           # nothing ever answered
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNone(row["host_load1"])
            self.assertIsNone(row["unit_memory_bytes"])
            self.assertIsNone(row["unit_memory_max_bytes"])
            self.assertFalse(row["present"])

    def test_present_constant_agrees_with_the_web_read_model(self):
        """Two spellings of one string; this is what keeps them one."""
        self.assertEqual(ns.PRESENT, fleetview.PRESENT)


class TestSummary(unittest.TestCase):
    def summarize(self, nodes, presence):
        rows = ns.node_rows(nodes, fleetview.presence_state, presence, {})
        return ns.summarize(rows, {"age_seconds": 1.0, "ok": True})

    def test_counts_present_nodes(self):
        nodes = [cuda_node(), cpu_node()]
        block = self.summarize(nodes, {n.name: probe(n.name) for n in nodes})
        self.assertEqual(block["present"], 2)
        self.assertTrue(block["pins_agree"])
        self.assertEqual(block["protocol"], "5.0.0")

    def test_one_mismatched_pin_makes_the_summary_say_so(self):
        nodes = [cuda_node(), cpu_node()]
        presence = {n.name: probe(n.name) for n in nodes}
        presence["ph16-71-cuda0"] = probe("ph16-71-cuda0", pin_agrees=False)
        block = self.summarize(nodes, presence)
        self.assertFalse(block["pins_agree"])
        # A pin-mismatched node is UP and unusable: it must not count as
        # present either.
        self.assertEqual(block["present"], 1)

    def test_no_remote_nodes_is_an_empty_block_not_an_error(self):
        block = ns.summarize([], {"age_seconds": None, "ok": False})
        self.assertEqual(block["nodes"], [])
        self.assertEqual(block["present"], 0)
        self.assertTrue(block["pins_agree"])


class TestTickSurface(unittest.TestCase):
    """The block on the tick, and the degradation path it reuses."""

    def collector(self, **kw):
        return telemetry.TelemetryCollector(
            inventory_fn=lambda: [], meminfo_fn=lambda: {
                "total_bytes": 1, "available_bytes": 1, "used_bytes": 0},
            runtime_fn=lambda: {}, profiles_fn=lambda: [],
            swap_probe_fn=lambda: {
                "swap": {"ok": True, "latency_ms": 1, "detail": ""},
                "api": {"ok": True, "latency_ms": 1, "detail": ""}},
            **kw)

    def test_tick_carries_the_block(self):
        nodes = [cuda_node(), cpu_node()]
        poller = ns.NodeStatsPoller(
            targets_fn=lambda: ns.poll_targets(nodes),
            poll_fn=lambda h, u: ns.poll_host(h, u, runner=runner_for(GOOD)),
            clock=Clock())
        poller.poll_once()
        with mock.patch.object(fleet, "load_fleet", lambda: nodes), \
             mock.patch.object(fleet, "load_presence",
                               lambda: {n.name: probe(n.name) for n in nodes}):
            snap = self.collector(node_poller=poller).snapshot()
        self.assertNotIn("node_stats", snap["errors"])
        block = snap["node_stats"]
        self.assertTrue(block["ok"])
        self.assertEqual(len(block["nodes"]), 2)
        self.assertEqual(block["present"], 2)

    def test_ssh_failure_populates_tick_errors(self):
        nodes = [cpu_node()]

        def boom(argv, **kw):
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        poller = ns.NodeStatsPoller(
            targets_fn=lambda: ns.poll_targets(nodes),
            poll_fn=lambda h, u: ns.poll_host(h, u, runner=boom),
            clock=Clock())
        poller.poll_once()
        with mock.patch.object(fleet, "load_fleet", lambda: nodes), \
             mock.patch.object(fleet, "load_presence", lambda: {}):
            snap = self.collector(node_poller=poller).snapshot()
        # Reuses the existing per-region path -- no parallel mechanism.
        self.assertIn("node_stats", snap["errors"])
        self.assertIn("timed out", snap["errors"]["node_stats"])
        self.assertFalse(snap["node_stats"]["ok"])

    def test_no_registry_means_no_card_and_no_error(self):
        with mock.patch.object(fleet, "load_fleet", lambda: []), \
             mock.patch.object(fleet, "load_presence", lambda: {}):
            snap = self.collector().snapshot()
        self.assertEqual(snap["node_stats"]["nodes"], [])
        self.assertNotIn("node_stats", snap["errors"])

    def test_unreadable_registry_is_an_error_not_a_silent_empty(self):
        def boom():
            raise OSError("registry is a directory")
        with mock.patch.object(fleet, "load_fleet", boom):
            snap = self.collector().snapshot()
        self.assertIn("node_stats", snap["errors"])
        self.assertIn("registry", snap["errors"]["node_stats"])

    def test_snapshot_never_starts_a_poll_inline(self):
        """The tick reads a cache. It must not ssh on the request path."""
        nodes = [cpu_node()]
        calls = []
        poller = ns.NodeStatsPoller(
            targets_fn=lambda: ns.poll_targets(nodes),
            poll_fn=lambda h, u: (calls.append(h), {"ok": False,
                                                    "error": "x"})[1],
            clock=Clock())
        with mock.patch.object(fleet, "load_fleet", lambda: nodes), \
             mock.patch.object(fleet, "load_presence", lambda: {}), \
             mock.patch.object(poller, "start", lambda: None):
            self.collector(node_poller=poller).snapshot()
        self.assertEqual(calls, [], "the tick path polled a host inline")


if __name__ == "__main__":
    unittest.main()
