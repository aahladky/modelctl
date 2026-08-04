"""Fleet RPC nodes: registry, wire probe, and optional planner targets.

Hermetic by construction. The only thing here that would otherwise touch
a network is `probe_node`, and its socket factory is injected -- these
tests drive it against an in-memory fake that speaks the same framing
`ggml-rpc-server` does, so the assertions are about the real wire format
without a real wire.

The load-bearing test in this file is
`TestFallbackIsByteIdentical`: with every node absent, the compiled plan
set must be indistinguishable from a checkout that has no fleet at all.
An "optional" target that quietly perturbs the local plans is not
optional.
"""
import inspect
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import modelctl_fleet as fleet
import modelctl_plans


PIN = "85b7e6556b6b83026d1a17df2635bc1173db1f97"
OTHER_PIN = "0000000000000000000000000000000000000000"
UNSET = object()  # "argument not supplied", where None means something


def a_node(name="ph16-71", host="192.168.0.76", port=50052, pin=PIN,
           device="CUDA0", budget=8 << 30, total=12 << 30, enabled=True):
    return fleet.FleetNode(
        name=name, host=host, port=port, variant="cuda", pin=pin,
        enabled=enabled,
        devices=(fleet.FleetDevice(name=device, kind="gpu",
                                   total_bytes=total, budget_bytes=budget),))


class FakeSocket:
    """Speaks the ggml-rpc framing well enough to answer one HELLO.

    request : cmd (1 byte) | size (8 bytes LE) | payload
    response:               size (8 bytes LE) | payload
    """

    def __init__(self, major=5, minor=0, patch=0, truncate=False,
                 wrong_size=False):
        self.version = (major, minor, patch)
        self.truncate = truncate
        self.wrong_size = wrong_size
        self.sent = b""
        self._out = b""
        self.closed = False

    def sendall(self, data):
        self.sent += data
        cmd = data[0]
        assert cmd == fleet.RPC_CMD_HELLO, f"unexpected command {cmd}"
        (size,) = struct.unpack("<Q", data[1:9])
        assert size == fleet.RPC_CONN_CAPS_SIZE
        body = bytes(self.version) + b"\x00" + bytes(fleet.RPC_CONN_CAPS_SIZE)
        declared = 7 if self.wrong_size else len(body)
        self._out = struct.pack("<Q", declared) + body

    def recv(self, n):
        chunk, self._out = self._out[:n], self._out[n:]
        if self.truncate:
            # Hand back the length prefix, then hang up before the body --
            # the failure a node killed mid-handshake produces.
            self._out = b""
            self.truncate = False
        return chunk

    def close(self):
        self.closed = True


class FleetStateBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fleet-test-")
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        self.fleet_path = Path(self.tmp) / "fleet.json"
        self.presence_path = Path(self.tmp) / "fleet-presence.json"
        for attr, val in (("FLEET_PATH", self.fleet_path),
                          ("PRESENCE_PATH", self.presence_path)):
            p = mock.patch.object(fleet, attr, val)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {"MODELCTL_FLEET_PIN": PIN})
        p.start()
        self.addCleanup(p.stop)


class TestRegistryRoundTrip(FleetStateBase):
    def test_save_then_load_preserves_every_field(self):
        node = a_node()
        fleet.save_fleet([node])
        (back,) = fleet.load_fleet()
        self.assertEqual(back, node)
        self.assertEqual(back.endpoint, "192.168.0.76:50052")

    def test_a_corrupt_registry_reads_as_no_fleet(self):
        # A broken registry must not break every launch path that
        # consults it -- "no fleet" is always a safe answer.
        self.fleet_path.write_text("{not json")
        self.assertEqual(fleet.load_fleet(), [])

    def test_valid_json_of_the_wrong_shape_reads_as_no_fleet(self):
        """"Never raises" has to mean it. A top-level list or null is
        valid JSON, so it survived the decode and then met `raw.get` --
        an AttributeError out of a loader every budget builder calls
        without a guard of its own."""
        for text in ("[]", "null", "42", '"nodes"'):
            with self.subTest(text=text):
                self.fleet_path.write_text(text)
                self.assertEqual(fleet.load_fleet(), [])

    def test_a_malformed_node_is_skipped_not_fatal(self):
        self.fleet_path.write_text(json.dumps(
            {"nodes": [{"host": "h"}, a_node().to_dict()]}))
        nodes = fleet.load_fleet()
        self.assertEqual([n.name for n in nodes], ["ph16-71"])

    def test_admission_keys_cannot_collide_with_a_local_device(self):
        key = fleet.admission_key("ph16-71", "CUDA0")
        self.assertEqual(key, "RPC:ph16-71:CUDA0")
        self.assertTrue(fleet.is_remote_key(key))
        self.assertFalse(fleet.is_remote_key("CUDA0"))


class TestWireProbe(FleetStateBase):
    def test_hello_sends_the_documented_framing(self):
        sock = FakeSocket()
        major, minor, patch = fleet.hello(sock)
        self.assertEqual((major, minor, patch), (5, 0, 0))
        self.assertEqual(sock.sent[0], 14)              # RPC_CMD_HELLO
        self.assertEqual(struct.unpack("<Q", sock.sent[1:9])[0], 24)
        # All-zero caps: a probe advertises no transport upgrade.
        self.assertEqual(sock.sent[9:], bytes(24))

    def test_reachable_node_at_the_pin_is_usable(self):
        pr = fleet.probe_node(a_node(), connect=lambda h, p, t: FakeSocket())
        self.assertTrue(pr.reachable)
        self.assertEqual(pr.protocol, "5.0.0")
        self.assertTrue(pr.pin_agrees)
        self.assertTrue(pr.usable)

    def test_a_refused_connection_is_reported_not_raised(self):
        def refuse(h, p, t):
            raise ConnectionRefusedError("nothing listening")
        pr = fleet.probe_node(a_node(), connect=refuse)
        self.assertFalse(pr.reachable)
        self.assertFalse(pr.usable)
        self.assertIn("ConnectionRefusedError", pr.detail)

    def test_a_node_on_a_different_commit_is_reachable_but_not_usable(self):
        pr = fleet.probe_node(a_node(pin=OTHER_PIN),
                              connect=lambda h, p, t: FakeSocket())
        self.assertTrue(pr.reachable)
        self.assertFalse(pr.pin_agrees)
        self.assertFalse(pr.usable)
        self.assertIn("this checkout pins", pr.detail)

    def test_a_wrong_sized_hello_response_is_not_a_matching_server(self):
        pr = fleet.probe_node(a_node(),
                              connect=lambda h, p, t: FakeSocket(wrong_size=True))
        self.assertFalse(pr.reachable)
        self.assertIn("not a matching ggml-rpc-server", pr.detail)

    def test_a_closed_socket_mid_response_does_not_hang_or_raise(self):
        pr = fleet.probe_node(a_node(),
                              connect=lambda h, p, t: FakeSocket(truncate=True))
        self.assertFalse(pr.reachable)

    def test_the_socket_is_closed_even_when_the_handshake_fails(self):
        sock = FakeSocket(wrong_size=True)
        fleet.probe_node(a_node(), connect=lambda h, p, t: sock)
        self.assertTrue(sock.closed)


class TestPresenceIsTheStoredPlanningInput(FleetStateBase):
    def test_planning_never_opens_a_socket(self):
        fleet.save_fleet([a_node()])

        def explode(*a, **k):
            raise AssertionError("planning opened a socket")

        with mock.patch.object(fleet.socket, "create_connection", explode):
            self.assertEqual(fleet.usable_nodes(), [])

    def test_an_unprobed_node_is_absent_not_present(self):
        fleet.save_fleet([a_node()])
        self.assertEqual(fleet.usable_nodes(), [])

    def test_a_recorded_present_node_is_usable(self):
        fleet.save_fleet([a_node()])
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket(), now=1000.0)
        names = [n.name for n in fleet.usable_nodes(now=1000.0)]
        self.assertEqual(names, ["ph16-71"])

    def test_a_stale_presence_record_expires_to_absent(self):
        fleet.save_fleet([a_node()])
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket(), now=1000.0)
        late = 1000.0 + fleet.PRESENCE_TTL_SECONDS + 1
        self.assertEqual(fleet.usable_nodes(now=late), [])

    def test_a_disabled_node_is_never_usable_however_present(self):
        fleet.save_fleet([a_node(enabled=False)])
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket(), now=1000.0)
        self.assertEqual(fleet.usable_nodes(now=1000.0), [])

    def test_budgets_are_the_declared_ceiling_not_what_the_node_reports(self):
        fleet.save_fleet([a_node(budget=8 << 30, total=12 << 30)])
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket(), now=1000.0)
        budgets = fleet.admission_budgets(now=1000.0)
        self.assertEqual(budgets, {"RPC:ph16-71:CUDA0": 8 << 30})


class TestContiguousPlacement(unittest.TestCase):
    def _rx(self, first, last):
        import re
        return re.compile(
            (fleet.contiguous_range(first, last) % "RPC0").rsplit("=", 1)[0])

    def test_range_matches_only_the_named_layers(self):
        rx = self._rx(16, 19)
        for good in (16, 17, 18, 19):
            self.assertTrue(rx.search(f"blk.{good}.ffn_gate_exps.weight"), good)
        for bad in (15, 20, 160, 1):
            self.assertFalse(rx.search(f"blk.{bad}.ffn_gate_exps.weight"), bad)

    def test_range_places_routed_experts_only(self):
        """A whole-layer placement aborts llama.cpp at load.

        The RPC buffer cannot run the norm/attention ops, so a pattern
        catching every tensor in a layer dies at ggml-backend.cpp:934
        ("pre-allocated tensor (blk.42.attn_norm.weight) in a buffer
        (RPC0[...]) that cannot run the operation") -- observed live
        2026-08-04 on plan 47ac2f5ab7bcb0c1. Only routed experts may
        cross the wire: the same rule modelctl_tiers already emits for
        the ladder plans, and the one the 122B fleet calibration proved.
        """
        rx = self._rx(16, 19)
        for good in ("blk.16.ffn_gate_exps.weight",
                     "blk.17.ffn_up_exps.weight",
                     "blk.19.ffn_down_exps.weight"):
            self.assertTrue(rx.search(good), good)
        for bad in ("blk.16.attn_norm.weight", "blk.16.attn_q.weight",
                    "blk.17.ffn_norm.weight", "blk.18.attn_output.weight",
                    "blk.19.ffn_gate_shexp.weight"):
            self.assertFalse(rx.search(bad), bad)

    def test_an_empty_range_is_refused(self):
        with self.assertRaises(ValueError):
            fleet.contiguous_range(10, 9)

    def test_placement_args_are_the_canonical_tokens(self):
        args = fleet.placement_args({
            "endpoints": ["192.168.0.76:50052"],
            "placements": [{"buffer_type": "RPC0[192.168.0.76:50052]",
                            "first_layer": 2, "last_layer": 3}]})
        self.assertEqual(args[:2], ["--rpc", "192.168.0.76:50052"])
        self.assertEqual(args[2], "-ot")
        self.assertEqual(
            args[3],
            r"blk\.(2|3)\.ffn_.*_exps=RPC0[192.168.0.76:50052]")

    def test_no_endpoints_emits_nothing(self):
        self.assertEqual(fleet.placement_args({"endpoints": []}), [])

    def test_device_names_are_global_across_endpoints(self):
        eps = ["a:1", "b:2", "c:3"]
        self.assertEqual(fleet.device_name_for(eps, "a:1"), "RPC0")
        self.assertEqual(fleet.device_name_for(eps, "b:2"), "RPC1")
        self.assertEqual(fleet.device_name_for(eps, "c:3"), "RPC2")

    def test_buffer_types_are_per_endpoint_not_global(self):
        # The distinction -ot cares about, and the one that produced
        # "unknown buffer type" when this first went over the wire:
        # attaching two single-device nodes gives two buffer types BOTH
        # called RPC0, told apart only by the bracketed endpoint --
        # while their device names are RPC0 and RPC1.
        self.assertEqual(fleet.buffer_type_for("192.168.0.76:50052"),
                         "RPC0[192.168.0.76:50052]")
        self.assertEqual(fleet.buffer_type_for("192.168.0.76:50053"),
                         "RPC0[192.168.0.76:50053]")
        eps = ["192.168.0.76:50052", "192.168.0.76:50053"]
        self.assertEqual(fleet.device_name_for(eps, eps[1]), "RPC1")
        self.assertNotEqual(fleet.buffer_type_for(eps[1]),
                            fleet.device_name_for(eps, eps[1]))


class TestClaimReattribution(unittest.TestCase):
    """Remotely-placed bytes leave the local card and land on the node key."""

    def _claim(self):
        return modelctl_plans.ResourceClaim(
            vram_bytes={"CUDA0": 10 << 30}, ram_bytes=0, storage_mode="mmap",
            expected_context=8192,
            vram_static_bytes={"CUDA0": 8 << 30},
            vram_kv_bytes={"CUDA0": 1 << 30},
            vram_overhead_bytes={"CUDA0": 1 << 30})

    def test_moved_bytes_are_not_charged_twice(self):
        merged = {"rpc": {"admission": {"RPC:n:CUDA0": 4 << 30}}}
        out = modelctl_plans._apply_rpc_placement(self._claim(), merged)
        self.assertEqual(out.vram_static_bytes["CUDA0"], 4 << 30)
        self.assertEqual(out.vram_bytes["CUDA0"], 6 << 30)
        self.assertEqual(out.vram_bytes["RPC:n:CUDA0"], 4 << 30)

    def test_kv_and_overhead_stay_local(self):
        merged = {"rpc": {"admission": {"RPC:n:CUDA0": 8 << 30}}}
        out = modelctl_plans._apply_rpc_placement(self._claim(), merged)
        # All static weight moved; KV + overhead remain on the local card.
        self.assertEqual(out.vram_static_bytes["CUDA0"], 0)
        self.assertEqual(out.vram_bytes["CUDA0"], 2 << 30)

    def test_a_claim_with_no_rpc_config_is_returned_untouched(self):
        c = self._claim()
        self.assertIs(modelctl_plans._apply_rpc_placement(c, {}), c)


class TestClientBufferNamesFoldIntoAdmissionKeys(unittest.TestCase):
    """The same remote bytes must not appear under two device names.

    A tier-ladder plan writes its rpc placement into config["extra"] as
    -ot rules, so `_make_claim` has already attributed those bytes to the
    llama.cpp-side buffer name ("RPC0[host:port]", or the bare "RPC0"
    from --device) before `_apply_rpc_placement` runs. Left alongside the
    admission key, that second name is an unbudgeted device: no budget
    map contains it, so `acquire_reservation_verdict` reads
    `budgets.get("RPC0[...]", 0)` -> 0 and denies the plan.

    Observed live 2026-08-04 on the 122B tier-4 ladder plan, whose claim
    carried RPC0[..:50052] 9.52 GiB and RPC0[..:50053] 15.36 GiB beside
    the real RPC:ph16-71-cuda0:CUDA0 / RPC:ph16-71-cpu0:CPU keys.
    """

    # placements is EMPTY on purpose: plan_tiers emits the ladder's
    # placement as -ot rules in config["extra"] and leaves this list
    # empty, so the fold cannot be driven off it. Anyone "improving" the
    # predicate to read the authoritative placements set instead would
    # silently un-fix this -- with these tests still green if the fixture
    # pretended otherwise.
    LADDER = {"rpc": {"admission": {"RPC:n:CUDA0": 6 << 30},
                      "placements": []}}

    def _ladder_claim(self):
        # SYCL0 is the LARGEST static holder, matching the live 122B
        # shape (SYCL0 24.88 GiB vs alias 9.52/24.47). That ordering is
        # what makes the double-debit visible: the un-fixed loop sorts by
        # descending static and eats the local card first. With the alias
        # largest, the un-fixed loop exhausts itself on the alias and the
        # local card survives by luck -- a fixture that proves nothing.
        return modelctl_plans.ResourceClaim(
            vram_bytes={"SYCL0": 8 << 30, "RPC0[h:1]": 6 << 30, "RPC0": 0},
            ram_bytes=0, storage_mode="mmap", expected_context=8192,
            vram_static_bytes={"SYCL0": 8 << 30, "RPC0[h:1]": 6 << 30},
            vram_kv_bytes={"SYCL0": 0}, vram_overhead_bytes={"SYCL0": 0})

    def test_no_client_side_name_survives(self):
        out = modelctl_plans._apply_rpc_placement(self._ladder_claim(),
                                                  self.LADDER)
        self.assertNotIn("RPC0[h:1]", out.vram_bytes)
        self.assertNotIn("RPC0", out.vram_bytes)
        self.assertEqual(out.vram_bytes["RPC:n:CUDA0"], 6 << 30)

    def test_every_remaining_key_is_local_or_an_admission_key(self):
        out = modelctl_plans._apply_rpc_placement(self._ladder_claim(),
                                                  self.LADDER)
        for dev in out.vram_bytes:
            if dev.startswith("RPC"):
                self.assertTrue(fleet.is_remote_key(dev), dev)

    def test_the_local_card_keeps_bytes_it_never_gave_up(self):
        # The -ot rule had already moved these 6 GiB off SYCL0. Debiting
        # the local card a second time would understate the local claim
        # and let an oversized plan pass local admission.
        out = modelctl_plans._apply_rpc_placement(self._ladder_claim(),
                                                  self.LADDER)
        self.assertEqual(out.vram_bytes["SYCL0"], 8 << 30)
        self.assertEqual(out.vram_static_bytes["SYCL0"], 8 << 30)

    def test_the_single_rung_path_still_debits_the_local_card(self):
        # _compile_rpc_plans does NOT write -ot into extra, so no client
        # name exists and the old re-attribution must be unchanged.
        claim = modelctl_plans.ResourceClaim(
            vram_bytes={"CUDA0": 10 << 30}, ram_bytes=0, storage_mode="mmap",
            expected_context=8192,
            vram_static_bytes={"CUDA0": 8 << 30},
            vram_kv_bytes={"CUDA0": 1 << 30},
            vram_overhead_bytes={"CUDA0": 1 << 30})
        out = modelctl_plans._apply_rpc_placement(
            claim, {"rpc": {"admission": {"RPC:n:CUDA0": 4 << 30}}})
        self.assertEqual(out.vram_static_bytes["CUDA0"], 4 << 30)
        self.assertEqual(out.vram_bytes["CUDA0"], 6 << 30)
        self.assertEqual(out.vram_bytes["RPC:n:CUDA0"], 4 << 30)


class TestAdmissionRefusesOverBudget(FleetStateBase):
    """Per-node budgets ride the existing admission machinery."""

    def _usable_with_fleet(self, budget):
        fleet.save_fleet([a_node(budget=budget)])
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket())
        return fleet.admission_budgets()

    def test_remote_budget_appears_in_the_same_usable_map(self):
        self._usable_with_fleet(8 << 30)
        import modelctl_hardware
        snap = modelctl_hardware.HardwareSnapshot(
            captured_at=0.0, fingerprint="x", gpus=(), ram_total_bytes=0,
            ram_available_bytes=0, ram_reserve_bytes=0, storage=(),
            backend_fingerprints={})
        usable = modelctl_plans._usable_vram_map(snap)
        self.assertEqual(usable.get("RPC:ph16-71:CUDA0"), 8 << 30)

    def test_a_claim_over_the_node_budget_overflows(self):
        usable = {"RPC:ph16-71:CUDA0": 4 << 30}
        claim = modelctl_plans.ResourceClaim(
            vram_bytes={"RPC:ph16-71:CUDA0": 6 << 30}, ram_bytes=0,
            storage_mode="mmap", expected_context=8192,
            vram_overhead_bytes={"RPC:ph16-71:CUDA0": 2 << 30})
        over = modelctl_plans._admission_overflow(claim, usable)
        self.assertIn("RPC:ph16-71:CUDA0", over)

    def test_a_claim_inside_the_node_budget_is_admitted(self):
        usable = {"RPC:ph16-71:CUDA0": 8 << 30}
        claim = modelctl_plans.ResourceClaim(
            vram_bytes={"RPC:ph16-71:CUDA0": 3 << 30}, ram_bytes=0,
            storage_mode="mmap", expected_context=8192,
            vram_overhead_bytes={"RPC:ph16-71:CUDA0": 2 << 30})
        self.assertEqual(
            modelctl_plans._admission_overflow(claim, usable), {})


class TestReservationBudgetsCarryTheFleet(FleetStateBase):
    """A remote placement is charged against the node's declared budget.

    The gap this pins (2026-08-04): `vram_admission_bytes()` carries
    remote keys, but every reservation budget map was built from local
    cards plus RAM alone. `budgets.get(dev, 0)` therefore read 0 for each
    fleet device and EVERY rpc plan was refused -- "needs 10.2 GiB on
    RPC:ph16-71-cuda0:CUDA0 but only 0.0 GiB is free" -- at plan test and
    at managed serve alike, so the ladder could emit rung plans that
    nothing was ever able to run.

    Local cards contribute live free bytes minus their reserve; a remote
    device contributes the operator's declared ceiling, because we cannot
    see another machine's free bytes and the ceiling is what that machine
    agreed to lend.
    """

    def _snap(self):
        import modelctl_hardware
        gpu = modelctl_hardware.GpuSnapshot(
            device="SYCL0", name="local", total_bytes=16 << 30,
            free_bytes=8 << 30, reserve_bytes=1 << 30, enabled=True,
            role="primary", bandwidth_gbs=None)
        return modelctl_hardware.HardwareSnapshot(
            captured_at=0.0, fingerprint="fp", gpus=(gpu,),
            ram_total_bytes=64 << 30, ram_available_bytes=32 << 30,
            ram_reserve_bytes=2 << 30, storage=(), backend_fingerprints={})

    def _present_node(self, budget):
        fleet.save_fleet([a_node(budget=budget)])
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket())

    def _budgets(self):
        import modelctl_hardware
        return modelctl_hardware.reservation_budgets(self._snap())

    def setUp(self):
        super().setUp()
        import modelctl_runtime
        self.rdb = modelctl_runtime.RuntimeDB(Path(self.tmp) / "rt.db")

    def _verdict(self, claim_vram):
        return self.rdb.acquire_reservation_verdict(
            "fixture", "p1", {"vram_bytes": claim_vram, "ram_bytes": 0},
            os.getpid(), budgets=self._budgets())

    def test_map_carries_local_free_ram_and_the_remote_ceiling(self):
        self._present_node(8 << 30)
        budgets = self._budgets()
        self.assertEqual(budgets["SYCL0"], 7 << 30)
        self.assertEqual(budgets["RAM"], 30 << 30)
        self.assertEqual(budgets["RPC:ph16-71:CUDA0"], 8 << 30)

    def test_an_absent_node_contributes_nothing(self):
        fleet.save_fleet([a_node(budget=8 << 30)])  # never probed
        self.assertNotIn("RPC:ph16-71:CUDA0", self._budgets())

    def test_a_remote_claim_inside_the_budget_is_reserved(self):
        self._present_node(8 << 30)
        res, denial = self._verdict(
            {"SYCL0": 2 << 30, "RPC:ph16-71:CUDA0": 6 << 30})
        self.assertIsNone(denial)
        self.assertIsNotNone(res)

    def test_a_remote_claim_over_the_budget_is_a_capacity_denial(self):
        self._present_node(4 << 30)
        res, denial = self._verdict({"RPC:ph16-71:CUDA0": 6 << 30})
        self.assertIsNone(res)
        self.assertEqual(denial["code"], "insufficient_vram")
        self.assertEqual(denial["resource"], "RPC:ph16-71:CUDA0")
        self.assertEqual(denial["need_bytes"], 6 << 30)
        self.assertEqual(denial["budget_bytes"], 4 << 30)
        # A pure capacity shortfall with nobody else holding anything --
        # the distinction f94c9e9 exists to make.
        self.assertEqual(denial["holders"], [])


class TestMatrixBudgetsCarryTheFleet(FleetStateBase):
    """An rpc-planned profile keeps its llama-swap route.

    The gap this pins (2026-08-04): `profile_claim` returns the PINNED
    plan's claim for a managed profile, so an rpc-planned model claims
    `RPC:<node>:<device>` keys -- while `_budgets` was built from local
    cards plus RAM alone. `_fits` then read `budgets.get(remote_key, 0)`
    -> 0, and `generate_matrix` dropped the model as "exceeds budgets
    alone", losing its `mc_` route at the next
    `routing_service.apply_matrix` (which rewrites config.yaml and
    restarts llama-swap). Measured on the live rig for
    qwen3-5-122b-a10b-ud: claim SYCL0 25.90, SYCL1 8.86,
    RPC:ph16-71-cuda0:CUDA0 9.52, RPC:ph16-71-cpu0:CPU 24.47 GiB against
    budgets naming only SYCL0, SYCL1 and RAM.

    The merged map is `budget_input` -- declared ceilings for ENABLED
    nodes -- and deliberately neither of its siblings.
    `admission_budgets` is presence-gated, so a closed laptop would
    delete a model's route; `reservation_budgets` is free-based. Routing
    must not flap with a lid or with desktop load, which is the same
    reason the local half is total-based.
    """

    INVENTORY = [{"device": "SYCL0", "total_bytes": 20 << 30}]
    DEFAULTS = {"vram_limit_pct": 90}
    LOCAL_BUDGET = (20 << 30) * (90 / 100.0)

    def setUp(self):
        super().setUp()
        import modelctl
        import modelctl_hardware
        self.profiles = Path(self.tmp) / "profiles"
        self.profiles.mkdir()
        for target, attr, val in (
                (modelctl_hardware, "load_settings",
                 lambda: {"devices": {}, "ram": {"reserve_bytes": 0}}),
                (modelctl_hardware, "_system_ram", lambda: 64 << 30),
                (modelctl, "PROFILES_DIR", self.profiles)):
            p = mock.patch.object(target, attr, val)
            p.start()
            self.addCleanup(p.stop)

    def _budgets(self):
        import modelctl_matrix
        return modelctl_matrix._budgets(self.INVENTORY, self.DEFAULTS)

    def test_a_declared_remote_ceiling_is_a_matrix_budget(self):
        fleet.save_fleet([a_node(budget=8 << 30)])
        self.assertEqual(self._budgets()["RPC:ph16-71:CUDA0"], 8 << 30)

    def test_a_closed_laptop_does_not_move_the_matrix(self):
        """Presence-independent on purpose: were this map presence-gated,
        every probe that missed would delete an rpc-planned model's route
        and the next one would put it back."""
        fleet.save_fleet([a_node(budget=8 << 30)])
        absent = self._budgets()  # never probed
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket())
        self.assertEqual(self._budgets(), absent)

    def test_a_disabled_node_is_not_lent(self):
        fleet.save_fleet([a_node(budget=8 << 30, enabled=False)])
        self.assertNotIn("RPC:ph16-71:CUDA0", self._budgets())

    def test_the_local_half_stays_total_based(self):
        fleet.save_fleet([a_node(budget=8 << 30)])
        budgets = self._budgets()
        self.assertEqual(budgets["SYCL0"], self.LOCAL_BUDGET)
        self.assertEqual(budgets["RAM"], 64 << 30)

    def test_a_claim_inside_the_declared_ceiling_fits(self):
        import modelctl_matrix
        fleet.save_fleet([a_node(budget=8 << 30)])
        claim = {"m": {"SYCL0": 10 << 30, "RPC:ph16-71:CUDA0": 6 << 30}}
        self.assertTrue(
            modelctl_matrix._fits(claim, ["m"], self._budgets()))

    def test_a_claim_over_the_declared_ceiling_still_does_not_fit(self):
        import modelctl_matrix
        fleet.save_fleet([a_node(budget=8 << 30)])
        claim = {"m": {"SYCL0": 10 << 30, "RPC:ph16-71:CUDA0": 12 << 30}}
        self.assertFalse(
            modelctl_matrix._fits(claim, ["m"], self._budgets()))

    def _generate(self, claim):
        """generate_matrix over one profile with a recorded claim.

        The claim is injected rather than computed: producing a real one
        needs a managed profile, a pinned plan and a hardware probe, all
        of which this test would then be asserting about instead of the
        thing it is here for. The shape is the live 122B's.
        """
        import modelctl_matrix
        import modelctl_runtime
        (self.profiles / "planned.json").write_text(json.dumps(
            {"name": "planned", "model_path": "/fake/m.gguf",
             "config": {"ctx": 8192}}))

        class FakeRdb:
            def plan_runs_for(self, name, limit=10):
                return []

        with mock.patch.object(modelctl_matrix, "profile_claim",
                               lambda p, inv, d=None: dict(claim)), \
                mock.patch.object(modelctl_runtime, "RuntimeDB", FakeRdb):
            return modelctl_matrix.generate_matrix(self.INVENTORY,
                                                   self.DEFAULTS)

    def test_an_rpc_planned_profile_keeps_its_managed_route(self):
        fleet.save_fleet([a_node(budget=8 << 30)])
        out = self._generate({"SYCL0": 10 << 30, "RPC:ph16-71:CUDA0": 6 << 30})
        self.assertEqual(out["excluded"], [])
        self.assertIn("mc_planned", out["sets"])

    def test_a_profile_that_really_is_too_big_is_still_excluded(self):
        """The merge must not turn the routing gate into a rubber stamp."""
        fleet.save_fleet([a_node(budget=8 << 30)])
        out = self._generate({"SYCL0": 10 << 30, "RPC:ph16-71:CUDA0": 12 << 30})
        self.assertNotIn("mc_planned", out["sets"])
        self.assertEqual([e["reason"] for e in out["excluded"]],
                         ["exceeds budgets alone"])


class TestBothChargingPathsUseTheSharedBuilder(unittest.TestCase):
    """The reservation map is built in ONE place, at every call site.

    The 2026-08-04 bug was never in the map builder -- it was that
    `modelctl_tune.test_launch_plan` and `modelctl_worker.worker_main`
    each built their own local-only map, and both forgot the fleet. A
    test that only exercises `reservation_budgets` directly stays green
    when either call site is re-inlined, which is exactly how the bug
    arrived the first time.

    A STRUCTURAL pin, deliberately: driving either function to its
    reservation call needs a profile, a compiled plan set, a hardware
    probe and a spawned server, so a behavioural test here would assert
    mostly about its own mocks. What actually needs guarding is narrow --
    that neither function rebuilds the map by hand -- and reading the
    source says exactly that and nothing more.
    """

    INLINE = "g.free_bytes - g.reserve_bytes"
    SHARED = "modelctl_hardware.reservation_budgets(snap)"

    def _assert_delegates(self, fn):
        src = inspect.getsource(fn)
        self.assertIn(self.SHARED, src,
                      f"{fn.__qualname__} must charge the shared builder")
        self.assertNotIn(self.INLINE, src,
                         f"{fn.__qualname__} rebuilt the budget map by hand; "
                         "a local-only map denies every remote claim")

    def test_the_plan_test_path_delegates(self):
        import modelctl_tune
        self._assert_delegates(modelctl_tune.test_launch_plan)

    def test_the_managed_worker_path_delegates(self):
        import modelctl_worker
        self._assert_delegates(modelctl_worker.worker_main)


class TestFallbackIsByteIdentical(FleetStateBase):
    """With the node absent, the plan set matches a fleet-free checkout.

    This is the whole "optional target" claim, stated as an assertion.
    """

    def _profile(self):
        return {
            "name": "fixture", "backend": "llama-cpp",
            "model_path": "/nonexistent/fixture.gguf",
            "config": {"ctx": 4096, "flash_attn": "on", "device": "CUDA0",
                       "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                       "fit": "off"},
        }

    def _snapshot(self):
        import modelctl_hardware
        return modelctl_hardware.HardwareSnapshot(
            captured_at=0.0, fingerprint="fp", gpus=(), ram_total_bytes=64 << 30,
            ram_available_bytes=32 << 30, ram_reserve_bytes=0, storage=(),
            backend_fingerprints={})

    def _compile(self):
        return modelctl_plans.compile_launch_plans(
            self._profile(), self._snapshot())

    def test_no_registry_and_absent_node_compile_identically(self):
        without = [(p.label, p.argv, p.id) for p in self._compile()]

        # Same checkout, node registered and enabled -- but never probed,
        # so presence says absent.
        fleet.save_fleet([a_node()])
        absent = [(p.label, p.argv, p.id) for p in self._compile()]

        self.assertEqual(without, absent)

    def test_an_unreachable_node_adds_no_plans(self):
        fleet.save_fleet([a_node()])
        baseline = [(p.label, p.argv, p.id) for p in self._compile()]

        def refuse(h, p, t):
            raise ConnectionRefusedError("down")
        fleet.refresh_presence(connect=refuse)

        self.assertEqual([(p.label, p.argv, p.id) for p in self._compile()],
                         baseline)

    def test_a_node_on_the_wrong_commit_adds_no_plans(self):
        fleet.save_fleet([a_node(pin=OTHER_PIN)])
        baseline_no_presence = [(p.label, p.argv, p.id)
                                for p in self._compile()]
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket())
        self.assertEqual([(p.label, p.argv, p.id) for p in self._compile()],
                         baseline_no_presence)

    def test_no_rpc_tokens_leak_into_local_plans(self):
        fleet.save_fleet([a_node()])
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket())
        for plan in self._compile():
            if plan.source == "fleet-rpc":
                continue
            self.assertNotIn("--rpc", plan.argv, plan.label)


class TestRpcPlansAppearWhenTheNodeIs(FleetStateBase):
    """The present-node path, driven off a synthetic GGUF layout.

    The layout is faked rather than read from a real model: what is under
    test is the placement arithmetic and argv, not the GGUF parser.
    """

    # An 8-layer MoE model: per layer, 1 GiB of routed experts plus 0.25
    # GiB of attention/norms, over 1 GiB of embeddings/output. Only the
    # expert bytes can cross the wire, so the two numbers are kept
    # DIFFERENT on purpose -- a sizer that reaches for whole-layer bytes
    # reads 1.25 GiB per layer here and the arithmetic tests catch it.
    # Shaped to gguf_model_layout's full contract so _make_claim sees the
    # same keys a real parse would hand it.
    LAYOUT = {"arch": "fixture", "meta": {}, "block_count": 8,
              "is_moe": True, "weight_bytes": 11 << 30,
              "non_expert_bytes": 3 << 30, "other_bytes": 1 << 30,
              "layer_bytes": 2 << 30,
              "expert_bytes_per_layer": {i: 1 << 30 for i in range(8)},
              "has_shexp": False, "unknown_type_tensors": 0}

    def _profile(self):
        return {
            "name": "fixture", "backend": "llama-cpp",
            "model_path": "/nonexistent/fixture.gguf",
            "config": {"ctx": 4096, "flash_attn": "on", "device": "CUDA0",
                       "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                       "fit": "off"},
        }

    def _layout_with(self, expert_bytes_per_layer):
        """LAYOUT with a different expert map, invariants kept.

        gguf_model_layout always yields weight_bytes == non_expert_bytes +
        sum(expert_bytes_per_layer). A derived fixture that changes the
        expert map without restating weight_bytes drifts out of that
        contract and would mask a real bug the day sizing cross-checks it.
        """
        return dict(self.LAYOUT,
                    expert_bytes_per_layer=dict(expert_bytes_per_layer),
                    weight_bytes=(self.LAYOUT["non_expert_bytes"]
                                  + sum(expert_bytes_per_layer.values())))

    def test_the_fixtures_satisfy_the_gguf_layout_contract(self):
        for layout in (self.LAYOUT,
                       self._layout_with({i: 1 << 30 for i in range(6)})):
            self.assertEqual(
                layout["weight_bytes"],
                layout["non_expert_bytes"]
                + sum(layout["expert_bytes_per_layer"].values()))
            self.assertEqual(layout["non_expert_bytes"],
                             layout["other_bytes"] + layout["layer_bytes"])

    def _split_profile(self):
        """A profile shaped like the live 122B: an explicit two-card split,
        with --device carried in `extra` because build_server_args drops
        config["device"] whenever a tensor split is configured."""
        p = self._profile()
        p["config"] = {**p["config"], "device": "", "split_mode": "layer",
                       "tensor_split": "8,3",
                       "extra": "--fit off --device SYCL0,SYCL1"}
        return p

    def _rpc_plans(self, budget, layout=None, profile=None):
        fleet.save_fleet([a_node(budget=budget)])
        fleet.refresh_presence(connect=lambda h, p, t: FakeSocket())
        import modelctl_hardware
        snap = modelctl_hardware.HardwareSnapshot(
            captured_at=0.0, fingerprint="fp", gpus=(), ram_total_bytes=64 << 30,
            ram_available_bytes=32 << 30, ram_reserve_bytes=0, storage=(),
            backend_fingerprints={})
        with mock.patch("os.path.exists", return_value=True), \
             mock.patch("modelctl_vram.gguf_model_layout",
                        return_value=layout or self.LAYOUT):
            plans = modelctl_plans._compile_rpc_plans(
                profile or self._profile(), snap, None)
        return plans

    def test_a_present_node_yields_a_trailing_contiguous_range(self):
        # 8 layers, 1 GiB each. A 3 GiB budget takes the last 3.
        (plan,) = self._rpc_plans(3 << 30)
        rpc = plan.decision_data["rpc"]
        self.assertEqual((rpc["first_layer"], rpc["last_layer"]), (5, 7))
        self.assertEqual(rpc["node"], "ph16-71")
        self.assertEqual(rpc["client_device"], "RPC0")

    def test_the_argv_carries_the_endpoint_and_the_range(self):
        (plan,) = self._rpc_plans(3 << 30)
        argv = list(plan.argv)
        self.assertIn("--rpc", argv)
        self.assertEqual(argv[argv.index("--rpc") + 1], "192.168.0.76:50052")
        # -ot takes the buffer type, endpoint and all -- not "RPC0".
        self.assertIn(
            r"blk\.(5|6|7)\.ffn_.*_exps=RPC0[192.168.0.76:50052]", argv)

    def test_the_range_never_swallows_every_layer(self):
        # A budget far larger than the model still leaves layers local.
        (plan,) = self._rpc_plans(999 << 30)
        rpc = plan.decision_data["rpc"]
        self.assertGreater(rpc["first_layer"], 0)
        self.assertEqual(rpc["layers"], 7)

    def test_a_budget_too_small_for_one_layer_yields_no_plan(self):
        self.assertEqual(self._rpc_plans(1 << 20), [])

    def test_the_plan_is_labelled_by_where_the_layers_went(self):
        # "experts of", not "layers": each layer's attention and norms
        # stay local, and the plans page is read by someone deciding
        # what this plan actually does.
        (plan,) = self._rpc_plans(3 << 30)
        self.assertEqual(plan.label,
                         "experts of layers 5-7 on 192.168.0.76:50052")

    def test_the_remote_claim_is_charged_to_the_node_key(self):
        (plan,) = self._rpc_plans(3 << 30)
        self.assertIn("RPC:ph16-71:CUDA0", plan.claim.vram_bytes)

    def test_the_plan_warns_that_the_local_plans_remain(self):
        (plan,) = self._rpc_plans(3 << 30)
        self.assertTrue(any("local-only" in w for w in plan.warnings))

    # --- sizing follows what actually crosses the wire ----------------
    #
    # Only routed experts move, so only routed-expert bytes may be spent
    # against the remote budget or charged to the node's admission key.
    # Sizing by whole-layer bytes was wrong in BOTH directions: it bought
    # too few layers, and -- because _apply_rpc_placement debits the
    # local card by exactly what it says went remote -- it understated
    # the local claim by the attention/norm bytes that never left.

    def test_the_charge_is_the_expert_bytes_that_actually_move(self):
        # 1 GiB of routed experts per layer, so 3 layers charge 3 GiB --
        # not the 3.75 GiB those layers weigh in full.
        (plan,) = self._rpc_plans(3 << 30)
        self.assertEqual(plan.decision_data["rpc"]["claimed_bytes"], 3 << 30)
        self.assertEqual(plan.config["rpc"]["admission"],
                         {"RPC:ph16-71:CUDA0": 3 << 30})

    def test_sizing_uses_each_layers_own_expert_bytes(self):
        # Layer 7 carries twice the experts of the rest, so a 3 GiB
        # budget buys layers 6-7 (2 + 1), not three layers.
        layout = self._layout_with(
            {**{i: 1 << 30 for i in range(7)}, 7: 2 << 30})
        (plan,) = self._rpc_plans(3 << 30, layout=layout)
        rpc = plan.decision_data["rpc"]
        self.assertEqual((rpc["first_layer"], rpc["last_layer"]), (6, 7))
        self.assertEqual(rpc["claimed_bytes"], 3 << 30)

    def test_layers_without_routed_experts_are_not_placed(self):
        # Trailing dense layers hold nothing the RPC buffer may execute,
        # so the placement ends at the last layer that has experts.
        layout = self._layout_with({i: 1 << 30 for i in range(6)})
        (plan,) = self._rpc_plans(3 << 30, layout=layout)
        rpc = plan.decision_data["rpc"]
        self.assertEqual((rpc["first_layer"], rpc["last_layer"]), (3, 5))

    def test_a_dense_model_yields_no_rpc_plan(self):
        # No routed experts means an expert-only -ot rule would place
        # nothing at all; a plan whose remote range is empty is a plan
        # that pays the wire cost for no work. The whole-layer rule this
        # replaced would have emitted one -- and it aborted at load.
        dense = dict(self.LAYOUT, is_moe=False, expert_bytes_per_layer={},
                     non_expert_bytes=11 << 30, layer_bytes=10 << 30)
        self.assertEqual(self._rpc_plans(3 << 30, layout=dense), [])

    # --- the RPC device has to be in --device to receive anything ------
    #
    # llama.cpp only offloads to devices named in --device, and every
    # fleet config that has ever loaded (the tier-4 ladder, the laguna R2
    # promotion) lists the RPC client there with a ZERO tensor-split
    # share -- the -ot rule decides what lands on it, so it must take no
    # part of the automatic layer split. These plans omitted it entirely.

    def test_the_rpc_client_joins_an_existing_device_list(self):
        (plan,) = self._rpc_plans(3 << 30, profile=self._split_profile())
        self.assertIn("--device SYCL0,SYCL1,RPC0", plan.config["extra"])

    def test_the_rpc_client_takes_a_zero_share_of_an_existing_split(self):
        (plan,) = self._rpc_plans(3 << 30, profile=self._split_profile())
        self.assertEqual(plan.config["tensor_split"], "8,3,0")

    def test_the_device_list_reaches_the_argv_after_rpc(self):
        # --rpc must precede --device: the RPC backend registers its
        # devices when --rpc is parsed (laguna R2 gotcha).
        (plan,) = self._rpc_plans(3 << 30, profile=self._split_profile())
        argv = list(plan.argv)
        self.assertIn("--device", argv)
        self.assertEqual(argv[argv.index("--device") + 1],
                         "SYCL0,SYCL1,RPC0")
        self.assertLess(argv.index("--rpc"), argv.index("--device"))

    def test_a_single_local_device_gains_a_split_so_rpc_gets_no_layers(self):
        # The fixture profile names one card and no split at all. Left
        # alone, two devices with no --tensor-split let llama.cpp spread
        # layers onto the RPC device automatically, on top of the -ot rule.
        (plan,) = self._rpc_plans(3 << 30)
        self.assertIn("--device CUDA0,RPC0", plan.config["extra"])
        self.assertEqual(plan.config["tensor_split"], "1,0")
        self.assertEqual(plan.config["split_mode"], "layer")

    def test_split_positions_always_match_the_device_list(self):
        for profile in (self._profile(), self._split_profile()):
            (plan,) = self._rpc_plans(3 << 30, profile=profile)
            argv = list(plan.argv)
            devices = argv[argv.index("--device") + 1].split(",")
            split = argv[argv.index("--tensor-split") + 1].split(",")
            self.assertEqual(len(devices), len(split))
            self.assertEqual(split[-1], "0", "the RPC client takes no share")
            self.assertTrue(devices[-1].startswith("RPC"))

    def test_a_split_that_does_not_match_the_devices_yields_no_plan(self):
        # Three shares for two cards: the positional mapping is already
        # broken, so appending a fourth would place the remote device
        # against the wrong share. Fail closed rather than guess -- the
        # same stance RPC plans take on a non-RPC build.
        profile = self._split_profile()
        profile["config"] = {**profile["config"], "tensor_split": "8,3,1"}
        self.assertEqual(self._rpc_plans(3 << 30, profile=profile), [])

    def test_a_device_list_already_naming_the_client_is_not_doubled(self):
        profile = self._split_profile()
        profile["config"] = {
            **profile["config"], "tensor_split": "8,3,0",
            "extra": "--fit off --device SYCL0,SYCL1,RPC0"}
        (plan,) = self._rpc_plans(3 << 30, profile=profile)
        self.assertEqual(plan.config["extra"].count("RPC0"), 1)
        self.assertEqual(plan.config["tensor_split"], "8,3,0")

    def test_an_already_listed_client_with_a_real_share_yields_no_plan(self):
        # Present in --device is not enough: a nonzero share means the
        # automatic layer split ALSO hands the RPC device attention and
        # norm tensors it cannot run, which is the whole failure this
        # function exists to prevent. Accepting it because the name was
        # already there would smuggle the defect back in.
        profile = self._split_profile()
        profile["config"] = {
            **profile["config"], "tensor_split": "8,3,2",
            "extra": "--fit off --device SYCL0,SYCL1,RPC0"}
        self.assertEqual(self._rpc_plans(3 << 30, profile=profile), [])

    def test_an_already_listed_client_with_no_split_yields_no_plan(self):
        # No --tensor-split at all means llama.cpp spreads layers across
        # every named device by its own rule -- including the RPC one.
        profile = self._split_profile()
        profile["config"] = {
            **profile["config"], "tensor_split": "", "split_mode": "",
            "extra": "--fit off --device SYCL0,SYCL1,RPC0"}
        self.assertEqual(self._rpc_plans(3 << 30, profile=profile), [])

    def test_the_equals_form_of_device_is_recognised(self):
        # llama.cpp accepts --device=X, and _make_claim already parses
        # both forms; a second parser in the same file that saw only the
        # space form would miss the list and append a CONFLICTING second
        # --device token instead of extending the first.
        profile = self._split_profile()
        profile["config"] = {**profile["config"], "device": "CUDA0",
                             "extra": "--fit off --device=SYCL0,SYCL1"}
        (plan,) = self._rpc_plans(3 << 30, profile=profile)
        extra = plan.config["extra"]
        self.assertEqual(extra.count("--device"), 1)
        self.assertIn("--device SYCL0,SYCL1,RPC0", extra)
        self.assertEqual(plan.config["tensor_split"], "8,3,0")


class TestActiveRemoteClaimsStillCharge(unittest.TestCase):
    """A remote ceiling is not freed by the model already using it.

    The gap this pins (2026-08-04): admission counted only pending and
    starting claims. That is sound for a LOCAL card -- its budget is
    driver-reported free bytes, so a model that finished loading has
    already subtracted itself -- but a remote device contributes a fixed
    declared ceiling that no reading ever reduces. Two managed models
    could therefore both be admitted against one laptop's 10.5 GiB, and
    overshoot on another machine is an OOM kill, not a swap-out.

    So an ACTIVE reservation charges its `RPC:` keys and nothing else.
    Charging its local keys too would double-count against free bytes
    and deny launches that genuinely fit.
    """

    NODE = "RPC:ph16-71:CUDA0"
    NODE_CPU = "RPC:ph16-71:CPU"

    def setUp(self):
        import modelctl_runtime
        self.runtime = modelctl_runtime
        self.tmp = tempfile.mkdtemp(prefix="reservation-active-")
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))
        self.rdb = modelctl_runtime.RuntimeDB(Path(self.tmp) / "rt.db")
        self.pid = os.getpid()
        self.budgets = {"SYCL0": 8 << 30, "RAM": 16 << 30,
                        self.NODE: 10 << 30, self.NODE_CPU: 24 << 30}

    def _hold(self, vram, ram=0, state="active", pid=None, name="holder",
              rdb=None):
        """Another worker's reservation, advanced to `state`."""
        rdb = self.rdb if rdb is None else rdb
        with mock.patch.object(self.runtime, "_pid_alive", return_value=True):
            res = rdb.acquire_reservation(
                name, "hp", {"vram_bytes": vram, "ram_bytes": ram},
                self.pid + 1 if pid is None else pid, budgets=self.budgets)
        self.assertIsNotNone(res)
        rdb.update_reservation(res["id"], state=state)
        return res

    def _verdict(self, vram, ram=0, budgets=UNSET, alive=True, pid=None,
                 rdb=None):
        # `None` is a meaningful budgets value -- it selects the legacy
        # overlap rule -- so "not supplied" needs a sentinel of its own.
        rdb = self.rdb if rdb is None else rdb
        with mock.patch.object(self.runtime, "_pid_alive", return_value=alive):
            return rdb.acquire_reservation_verdict(
                "second", "sp", {"vram_bytes": vram, "ram_bytes": ram},
                self.pid if pid is None else pid,
                budgets=self.budgets if budgets is UNSET else budgets)

    def test_an_active_remote_claim_denies_a_second_model(self):
        self._hold({self.NODE: 7 << 30})
        res, denial = self._verdict({self.NODE: 5 << 30})
        self.assertIsNone(res)
        self.assertEqual(denial["code"], "insufficient_vram")
        self.assertEqual(denial["resource"], self.NODE)

    def test_the_denial_carries_the_active_bytes_and_holder(self):
        self._hold({self.NODE: 7 << 30})
        _res, denial = self._verdict({self.NODE: 5 << 30})
        self.assertEqual(denial["pending_bytes"], 7 << 30)
        self.assertEqual(denial["budget_bytes"], 10 << 30)
        self.assertEqual(denial["holders"], ["holder"])

    def test_a_remote_claim_that_still_fits_beside_it_is_admitted(self):
        self._hold({self.NODE: 7 << 30})
        res, denial = self._verdict({self.NODE: 3 << 30})
        self.assertIsNone(denial)
        self.assertIsNotNone(res)

    def test_remote_cpu_bytes_are_charged_the_same_way(self):
        self._hold({self.NODE_CPU: 20 << 30})
        res, denial = self._verdict({self.NODE_CPU: 8 << 30})
        self.assertIsNone(res)
        self.assertEqual(denial["resource"], self.NODE_CPU)

    def test_an_active_local_claim_is_not_charged_twice(self):
        # The local card's budget is already free-bytes, so charging an
        # active local claim on top of it would refuse a launch that fits.
        self._hold({"SYCL0": 6 << 30})
        res, denial = self._verdict({"SYCL0": 6 << 30})
        self.assertIsNone(denial)
        self.assertIsNotNone(res)

    def test_active_local_ram_is_not_charged_twice(self):
        # Same argument: ram_available_bytes already reflects a running
        # process. Remote RAM rides an `RPC:<node>:CPU` vram key instead.
        self._hold({}, ram=12 << 30)
        res, denial = self._verdict({}, ram=12 << 30)
        self.assertIsNone(denial)
        self.assertIsNotNone(res)

    def test_a_purely_local_active_claim_is_not_listed_as_a_holder(self):
        self._hold({"SYCL0": 6 << 30})
        _res, denial = self._verdict({"SYCL0": 9 << 30})
        self.assertEqual(denial["holders"], [])

    def test_a_mixed_active_claim_charges_only_its_remote_half(self):
        self._hold({"SYCL0": 6 << 30, self.NODE: 7 << 30})
        # Local half ignored: 6 GiB against an 8 GiB free-bytes budget.
        res, denial = self._verdict({"SYCL0": 6 << 30})
        self.assertIsNone(denial, denial)
        self.assertIsNotNone(res)
        # Remote half still charged.
        _res, denial = self._verdict({self.NODE: 5 << 30})
        self.assertEqual(denial["resource"], self.NODE)

    def test_a_dead_owner_frees_the_remote_ceiling(self):
        self._hold({self.NODE: 7 << 30})
        res, denial = self._verdict({self.NODE: 9 << 30}, alive=False)
        self.assertIsNone(denial)
        self.assertIsNotNone(res)

    def test_our_own_active_reservation_does_not_deny_us(self):
        self._hold({self.NODE: 7 << 30}, pid=self.pid)
        res, denial = self._verdict({self.NODE: 9 << 30})
        self.assertIsNone(denial)
        self.assertIsNotNone(res)

    def test_an_active_row_charges_remote_bytes_beside_local_ram(self):
        # The realistic shape: a rung plan holds laptop VRAM and local
        # resident RAM at once. The remote half charges, the RAM does not.
        self._hold({self.NODE: 7 << 30}, ram=12 << 30)
        res, denial = self._verdict({}, ram=12 << 30)
        self.assertIsNone(denial, denial)
        self.assertIsNotNone(res)
        _res, denial = self._verdict({self.NODE: 5 << 30})
        self.assertEqual(denial["resource"], self.NODE)

    def test_two_active_holders_on_one_device_add_up(self):
        self._hold({self.NODE: 4 << 30}, name="first")
        self._hold({self.NODE: 4 << 30}, name="second-holder",
                   pid=self.pid + 2)
        _res, denial = self._verdict({self.NODE: 3 << 30})
        self.assertEqual(denial["pending_bytes"], 8 << 30)
        self.assertEqual(sorted(denial["holders"]),
                         ["first", "second-holder"])

    def test_a_pending_local_and_an_active_remote_charge_together(self):
        self._hold({"SYCL0": 5 << 30}, state="pending", name="pend")
        self._hold({self.NODE: 7 << 30}, name="act", pid=self.pid + 2)
        _res, denial = self._verdict({"SYCL0": 4 << 30})
        self.assertEqual(denial["resource"], "SYCL0")
        _res, denial = self._verdict({self.NODE: 5 << 30})
        self.assertEqual(denial["resource"], self.NODE)

    def _charge_pending_state(self, state):
        rdb = self.runtime.RuntimeDB(Path(self.tmp) / f"{state}.db")
        self._hold({"SYCL0": 6 << 30}, ram=12 << 30, state=state, rdb=rdb)
        _res, denial = self._verdict({"SYCL0": 6 << 30}, rdb=rdb)
        self.assertEqual(denial["resource"], "SYCL0")
        _res, denial = self._verdict({}, ram=12 << 30, rdb=rdb)
        self.assertEqual(denial["resource"], "RAM")

    def test_a_pending_row_still_charges_everything(self):
        self._charge_pending_state("pending")

    def test_a_starting_row_still_charges_everything(self):
        self._charge_pending_state("starting")

    def test_an_unreadable_row_does_not_wedge_the_gate(self):
        # An active row lives as long as the model it serves, so a claim
        # nobody can price must not raise on every admission for hours.
        # It contributes nothing, the way a malformed fleet file does.
        held = self._hold({self.NODE: 7 << 30})
        for junk in ('[]', 'null', '42', 'not json',
                     '{"vram_bytes": null}',
                     '{"vram_bytes": {"RPC:ph16-71:CUDA0": "lots"}}'):
            with self.subTest(claim=junk):
                with self.rdb._conn() as c:
                    c.execute("UPDATE reservations SET claim_json=? "
                              "WHERE id=?", (junk, held["id"]))
                res, denial = self._verdict({self.NODE: 9 << 30})
                self.assertIsNone(denial, denial)
                self.assertIsNotNone(res)
                self.rdb.release_reservation(res["id"])

    def test_an_unpriceable_row_is_not_named_as_a_holder(self):
        # It charges nothing, so a denial caused by someone else must not
        # accuse it of holding the device.
        junk = self._hold({self.NODE: 7 << 30}, name="junk-row")
        with self.rdb._conn() as c:
            c.execute("UPDATE reservations SET claim_json=? WHERE id=?",
                      ('{"vram_bytes": {"RPC:ph16-71:CUDA0": "lots"}}',
                       junk["id"]))
        self._hold({self.NODE: 7 << 30}, name="real-holder", pid=self.pid + 2)
        _res, denial = self._verdict({self.NODE: 5 << 30})
        self.assertEqual(denial["pending_bytes"], 7 << 30)
        self.assertEqual(denial["holders"], ["real-holder"])

    def test_a_negative_count_cannot_subtract_from_occupancy(self):
        # The one malformed direction that would ADMIT too much rather
        # than deny too much.
        junk = self._hold({self.NODE: 7 << 30}, name="junk-row")
        with self.rdb._conn() as c:
            c.execute("UPDATE reservations SET claim_json=? WHERE id=?",
                      ('{"vram_bytes": {"RPC:ph16-71:CUDA0": -9663676416}}',
                       junk["id"]))
        self._hold({self.NODE: 7 << 30}, name="real-holder", pid=self.pid + 2)
        _res, denial = self._verdict({self.NODE: 5 << 30})
        self.assertEqual(denial["pending_bytes"], 7 << 30)

    def test_charged_bytes_is_the_gate_view_not_the_pending_view(self):
        # What the worker's pre-filter must ask. `pending_claims` cannot
        # see an active row, so a filter built on it called remote plans
        # feasible that the gate then denied.
        self._hold({"SYCL0": 6 << 30, self.NODE: 7 << 30}, ram=12 << 30)
        with mock.patch.object(self.runtime, "_pid_alive", return_value=True):
            charged = self.rdb.charged_bytes(exclude_pid=self.pid)
            invisible = self.rdb.pending_claims(exclude_pid=self.pid)
        self.assertEqual(charged.get(self.NODE), 7 << 30)
        self.assertNotIn("SYCL0", charged)
        self.assertNotIn("RAM", charged)
        self.assertEqual(invisible, [])

    def test_charged_bytes_excludes_our_own_pid(self):
        self._hold({self.NODE: 7 << 30}, pid=self.pid)
        with mock.patch.object(self.runtime, "_pid_alive", return_value=True):
            self.assertEqual(self.rdb.charged_bytes(exclude_pid=self.pid), {})

    def test_the_legacy_no_budget_path_still_ignores_active_rows(self):
        # Without a budget map the rule is "any device overlap conflicts",
        # which is what lets llama-swap hand a local card between models.
        # Broadening it to active rows would deadlock that handover.
        self._hold({"SYCL0": 6 << 30})
        res, denial = self._verdict({"SYCL0": 6 << 30}, budgets=None)
        self.assertIsNone(denial)
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()
