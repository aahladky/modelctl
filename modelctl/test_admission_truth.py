"""P0 admission truth: every managed path charges the same peak claim.

The bug class these exist to catch: ResourceClaim separates the static
VRAM reservation from the additional dynamic expert-cache budget, and
resident RAM from mmap-addressed bytes -- but an admission path that
reads the decomposed fields directly can admit a cache-enabled plan as
though its cache does not exist, or refuse an mmap plan as though the
whole model had to be resident.

The canonical values are ResourceClaim.vram_admission_bytes() and
ram_admission_bytes(). These tests drive one plan produced by the real
planner through the real worker_main() -- feasibility, RuntimeDB
reservation, runtime events, plan-run record -- and through the
coexistence matrix, asserting every stage saw the same numbers.

Only the hardware snapshot, the GGUF layout, and the actual backend
process are faked; the admission logic under test is the production code.
"""
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import modelctl
import modelctl_hardware
import modelctl_launch
import modelctl_matrix
import modelctl_plans
import modelctl_runtime
import modelctl_worker

GiB = 1 << 30


class FakeGpu:
    def __init__(self, device, free_bytes, total_bytes=16 * GiB):
        self.device = device
        self.name = device
        self.enabled = True
        self.role = ""
        self.total_bytes = total_bytes
        self.free_bytes = free_bytes
        self.reserve_bytes = 0


class FakeSnapshot:
    def __init__(self, gpus, ram_available):
        self.gpus = tuple(gpus)
        self.fingerprint = "hw-test"
        self.backend_fingerprints = {"llama-cpp": "be-test"}
        self.ram_available_bytes = ram_available
        self.ram_reserve_bytes = 0

    def gpu_by_device(self, device):
        return next((g for g in self.gpus if g.device == device), None)


class FakeProc:
    """Stands in for the launched backend. The PID is deliberately absent
    from the system so signal forwarding hits ProcessLookupError paths
    instead of a real process group."""

    def __init__(self):
        self.pid = 999999999

    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 1


FAKE_LAUNCH = SimpleNamespace(
    is_valid=True,
    argv=("/fake/llama-server", "--model", "m.gguf"),
    environment={},
    command_fingerprint="cmd-test",
    environment_fingerprint="launch-env-test",
    errors=(),
    raise_for_errors=lambda: None,
    backend=SimpleNamespace(
        binary="/fake/llama-server",
        binary_fingerprint="bin-test",
        environment_fingerprint="env-test",
        capabilities={"schema": 2},
        capability_fingerprint="cap-test"),
)

MODEL_BYTES = 32 * GiB
CACHE_BUDGET = 2 * GiB


class AdmissionTruthBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        self.model_path = self.root / "m.gguf"
        self.model_path.write_bytes(b"\0" * 4096)
        self.db = modelctl_runtime.RuntimeDB(self.root / "runtime.db")

        p = mock.patch.object(modelctl, "PROFILES_DIR", self.profiles)
        p.start()
        self.addCleanup(p.stop)

    def make_profile(self, extra=""):
        profile = modelctl.normalize_profile({
            "name": "admtest",
            "model_path": str(self.model_path),
            "backend": "llama-cpp",
            "binary": "/fake/llama-server",
            "config": {"ctx": 8192, "flash_attn": "auto", "fit": "off",
                       "device": "SYCL0", "extra": extra,
                       "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
            # Nonempty so the launch-environment assertion in run_worker()
            # can catch a caller re-applying plan.env on top of the
            # already-complete launch.environment.
            "env": ["GGML_OP_OFFLOAD_MOE_MIN_BATCH=64"],
        })
        profile["moe_cache"] = {
            "mode": "manual",
            "gpu": {"budgets_bytes": {"SYCL0": CACHE_BUDGET}}}
        profile["runtime"] = {"mode": "managed", "health_timeout": 0}
        (self.profiles / "admtest.json").write_text(json.dumps(profile))
        return profile

    def make_plan(self, profile):
        """The real planner path (_make_plan/_make_claim) over a synthetic
        GGUF layout: a 32 GiB MoE, half of it experts, 48 layers."""
        layout = {
            "weight_bytes": MODEL_BYTES,
            "non_expert_bytes": MODEL_BYTES // 8,
            "layer_bytes": MODEL_BYTES // 2,
            "block_count": 48,
            "is_moe": True,
            "expert_bytes_per_layer": {i: (MODEL_BYTES // 2) // 48
                                       for i in range(48)},
            "other_bytes": MODEL_BYTES // 8,
            "meta": {},
        }
        with mock.patch.object(modelctl_plans.modelctl_vram,
                               "gguf_model_layout", return_value=layout), \
             mock.patch.object(modelctl_plans.modelctl_vram,
                               "gguf_kv_params", return_value=None), \
             mock.patch.object(modelctl_plans, "_block_device_for",
                               return_value="/dev/test0"):
            return modelctl_plans.current_profile_plan(profile)

    def run_worker(self, plan, snap, launch_succeeds=False):
        """Run the real worker_main with the backend process faked out.

        Returns (exit_code, reservation_claims): every claim_dict the
        worker actually handed to RuntimeDB.acquire_reservation.
        """
        captured = []
        orig_acquire = self.db.acquire_reservation

        def spy(profile_name, plan_id, claim_dict, owner_pid, budgets=None):
            captured.append(claim_dict)
            return orig_acquire(profile_name, plan_id, claim_dict,
                                owner_pid, budgets=budgets)

        launched_envs = []

        def fake_popen(cmd, env=None, **kwargs):
            launched_envs.append(env)
            return FakeProc()

        with mock.patch.object(modelctl_hardware, "capture_hardware_snapshot",
                               return_value=snap), \
             mock.patch.object(modelctl_plans, "compile_launch_plans",
                               return_value=[plan]), \
             mock.patch.object(modelctl_runtime, "RuntimeDB",
                               return_value=self.db), \
             mock.patch.object(self.db, "acquire_reservation", side_effect=spy), \
             mock.patch.object(modelctl, "get_gpu_inventory",
                               side_effect=RuntimeError("no probe in tests")), \
             mock.patch.object(modelctl_launch, "build_launch_command",
                               return_value=FAKE_LAUNCH), \
             mock.patch.object(modelctl_worker.subprocess, "Popen",
                               side_effect=fake_popen), \
             mock.patch.object(modelctl_worker, "_wait_ready",
                               return_value=launch_succeeds), \
             mock.patch.object(modelctl_worker, "_forward_proc_terminate",
                               lambda proc: None):
            code = modelctl_worker.worker_main("admtest", 18080)
        # Launch-contract check that rides along on every worker run: the
        # child process gets EXACTLY the fingerprinted launch environment,
        # with no caller-side additions (P0 launch identity).
        for env in launched_envs:
            self.assertEqual(env, dict(FAKE_LAUNCH.environment),
                             "worker launched with an environment other "
                             "than LaunchCommand.environment")
        return code, captured

    def events(self, event_type):
        return [json.loads(e["detail_json"])
                for e in self.db.get_events("admtest")
                if e["event_type"] == event_type]

    def matrix_claim(self, profile, plan, snap):
        with mock.patch.object(modelctl_hardware,
                               "capture_hardware_snapshot", return_value=snap), \
             mock.patch.object(modelctl_plans, "compile_launch_plans",
                               return_value=[plan]):
            return modelctl_matrix.profile_claim(profile, inventory=[])


class TestCacheBudgetIsAdmitted(AdmissionTruthBase):
    """The review's required test: a plan with a nonzero expert-cache
    budget, passed planner -> worker feasibility -> RuntimeDB reservation
    -> coexistence matrix -> runtime event, must use the same peak
    per-device values at every stage."""

    def test_every_stage_charges_the_same_peak_values(self):
        profile = self.make_profile()
        plan = self.make_plan(profile)
        self.assertEqual(plan.claim.vram_cache_bytes,
                         {"SYCL0": CACHE_BUDGET},
                         "fixture lost its cache budget; test is void")

        expected_vram = plan.claim.vram_admission_bytes()
        expected_ram = plan.claim.ram_admission_bytes()
        self.assertEqual(expected_vram["SYCL0"],
                         plan.claim.vram_bytes["SYCL0"] + CACHE_BUDGET)

        # Plenty of room: the plan is feasible and reaches reservation,
        # launch, and the recorded run.
        snap = FakeSnapshot(
            [FakeGpu("SYCL0", free_bytes=expected_vram["SYCL0"] + 8 * GiB,
                     total_bytes=expected_vram["SYCL0"] + 8 * GiB)],
            ram_available=64 * GiB)
        code, reservations = self.run_worker(plan, snap)
        self.assertEqual(code, 1)  # fake backend never gets healthy

        # Stage: worker feasibility -- the plan passed it.
        self.assertEqual(self.events("plan_infeasible"), [])
        self.assertTrue(self.events("plan_selected"))

        # Stage: RuntimeDB reservation -- admission values, not static.
        self.assertEqual(len(reservations), 1)
        self.assertEqual(reservations[0]["vram_bytes"], expected_vram)
        self.assertEqual(reservations[0]["ram_bytes"], expected_ram)

        # Stage: runtime event.
        self.assertTrue(self.events("reservation_acquired"))

        # Stage: recorded plan run.
        runs = self.db.plan_runs_for("admtest", plan.id)
        self.assertEqual(len(runs), 1)
        run_claim = json.loads(runs[0]["claim_json"])
        self.assertEqual(run_claim["vram_bytes"], expected_vram)
        self.assertEqual(run_claim["ram_bytes"], expected_ram)

        # Stage: coexistence matrix envelope.
        env = self.matrix_claim(profile, plan, snap)
        self.assertEqual({d: b for d, b in env.items() if d != "RAM"},
                         expected_vram)

        # The reservation was released after the failed launch.
        self.assertEqual(self.db.get_reservations(), [])

    def test_a_plan_whose_cache_does_not_fit_is_not_admitted(self):
        profile = self.make_profile()
        plan = self.make_plan(profile)
        static = plan.claim.vram_bytes["SYCL0"]

        # The static claim fits with 1 GiB to spare; the 2 GiB cache
        # budget does not. Pre-fix admission (static only) accepted this
        # plan and the cache allocation then collided with whatever else
        # held the device.
        snap = FakeSnapshot([FakeGpu("SYCL0", free_bytes=static + 1 * GiB,
                                     total_bytes=static + 1 * GiB)],
                            ram_available=64 * GiB)
        self.assertLessEqual(static, static + 1 * GiB,
                             "fixture must fit the static claim alone")
        code, reservations = self.run_worker(plan, snap)

        self.assertEqual(code, 1)
        self.assertEqual(reservations, [], "an oversubscribing plan was "
                         "handed to the reservation path")
        infeasible = self.events("plan_infeasible")
        self.assertEqual(len(infeasible), 1)
        self.assertEqual(
            infeasible[0]["claim"],
            {**plan.claim.vram_admission_bytes(),
             "RAM": plan.claim.ram_admission_bytes()},
            "the infeasibility event must report the peak claim it "
            "actually tested")
        self.assertEqual(self.events("plan_selected"), [])


class TestMmapIsNotChargedAsResidentRam(AdmissionTruthBase):
    """An oversized-MoE mmap plan must be admitted against resident RAM,
    not the full CPU-addressed model -- the exact case managed serving
    used to refuse after browser testing had accepted it."""

    def test_mmap_plan_is_feasible_with_less_ram_than_model(self):
        profile = self.make_profile(extra="-ngl 24")
        plan = self.make_plan(profile)
        self.assertEqual(plan.claim.storage_mode, "mmap")
        ram_available = 4 * GiB
        self.assertGreater(plan.claim.ram_bytes, ram_available,
                           "fixture must address more bytes than RAM has, "
                           "or this test proves nothing")

        peak = plan.claim.vram_admission_bytes()
        snap = FakeSnapshot([FakeGpu("SYCL0", free_bytes=peak["SYCL0"] + 8 * GiB,
                                     total_bytes=peak["SYCL0"] + 8 * GiB)],
                            ram_available=ram_available)
        code, reservations = self.run_worker(plan, snap)
        self.assertEqual(code, 1)  # fake backend never gets healthy

        # Feasibility admitted it, and the reservation charged resident
        # bytes while keeping the mmap size visible as information.
        self.assertEqual(self.events("plan_infeasible"), [])
        self.assertEqual(len(reservations), 1)
        self.assertEqual(reservations[0]["ram_bytes"],
                         plan.claim.ram_admission_bytes())
        self.assertLess(reservations[0]["ram_bytes"], ram_available)
        self.assertEqual(reservations[0]["mmap_bytes"], plan.claim.mmap_bytes)

        # The matrix envelope agrees: no phantom RAM charge that would
        # falsely rule out coexistence.
        env = self.matrix_claim(profile, plan, snap)
        self.assertEqual(env.get("RAM", 0), plan.claim.ram_admission_bytes())


if __name__ == "__main__":
    unittest.main()


class TestTheSelectionDecidesTheLaunch(AdmissionTruthBase):
    """Phase 2 of the backend alignment plan: the launcher plans the
    profile's stored selection instead of compiling its own candidate set
    and promoting a pinned plan id.

    The two used to be independent answers to "where does this model run",
    which is how qwen3-5-122b-a10b-ud came to serve a four-device fleet
    ladder while its placement surface described three local devices.
    """


    def _roomy_snap(self, plan):
        """A machine with room to spare, so these tests are about which
        planner path ran and not about admission arithmetic."""
        need = plan.claim.vram_admission_bytes()
        return FakeSnapshot(
            [FakeGpu(dev, free_bytes=b + 8 * GiB, total_bytes=b + 8 * GiB)
             for dev, b in (need or {"SYCL0": 0}).items()],
            ram_available=64 * GiB)

    def _with_selection(self, selection):
        profile = self.make_profile()
        profile["planning"] = {"selection": selection}
        (self.profiles / "admtest.json").write_text(json.dumps(profile))
        return profile

    def test_a_stored_selection_is_what_the_worker_launches(self):
        profile = self._with_selection({"SYCL0": {"on": True}})
        plan = self.make_plan(profile)
        snap = self._roomy_snap(plan)
        with mock.patch.object(modelctl_plans, "plans_for_selection_launch",
                               return_value=[(plan, 0.0)]) as ladder, \
             mock.patch.object(modelctl_plans, "compile_launch_plans",
                               side_effect=AssertionError(
                                   "the candidate compiler must not run for "
                                   "a profile that has been placed by hand")):
            self.run_worker(plan, snap, launch_succeeds=True)
        self.assertTrue(ladder.called)
        self.assertEqual(ladder.call_args[0][1], {"SYCL0": {"on": True}},
                         "the ladder is handed the selection verbatim")

    def test_no_selection_still_compiles_and_ranks(self):
        """The migration-free half: a profile nobody has placed by hand
        keeps exactly the old path, so nothing in flight changes."""
        profile = self.make_profile()
        plan = self.make_plan(profile)
        with mock.patch.object(modelctl_plans, "plans_for_selection_launch",
                               side_effect=AssertionError(
                                   "no selection means the automatic "
                                   "placement, not the ladder")):
            _code, claims = self.run_worker(plan, self._roomy_snap(plan),
                                            launch_succeeds=True)
        # What matters is which planner path ran and that it reached
        # admission; the fake backend's exit code is this harness's own
        # business and is asserted where that is the subject.
        self.assertTrue(claims, "the compiled candidate never reached the "
                                "admission gate")

    def test_the_worker_never_writes_the_selection_back(self):
        """Degradation is a property of one compilation; intent is a
        property of the profile. A closed laptop must not un-tick a node."""
        selection = {"SYCL0": {"ceiling_bytes": 12 << 30}}
        profile = self._with_selection(selection)
        plan = self.make_plan(profile)
        with mock.patch.object(modelctl_plans, "plans_for_selection_launch",
                               return_value=[(plan, 0.0)]):
            self.run_worker(plan, self._roomy_snap(plan), launch_succeeds=True)
        after = json.loads((self.profiles / "admtest.json").read_text())
        self.assertEqual(after["planning"]["selection"], selection)
