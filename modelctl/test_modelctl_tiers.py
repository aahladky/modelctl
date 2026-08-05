import re
import shlex
import unittest
from pathlib import Path

import modelctl_tiers
import modelctl_vram

GIB = 1 << 30

B70 = {"device": "SYCL0", "name": "Intel(R) Arc(TM) Pro B70 Graphics",
       "total_bytes": 32656 * (1 << 20), "free_bytes": 0}
B580 = {"device": "SYCL1", "name": "Intel(R) Arc(TM) B580 Graphics",
        "total_bytes": 12216 * (1 << 20), "free_bytes": 0}
INVENTORY = [B70, B580]


def moe_layout(n_layers, per_layer_gib, non_expert_gib=4.0, arch="testmoe",
               first_layer=1, has_shexp=False):
    layers = {i: int(per_layer_gib * GIB)
              for i in range(first_layer, first_layer + n_layers)}
    block_count = first_layer + n_layers
    meta = {"general.architecture": arch, f"{arch}.block_count": block_count}
    weights = sum(layers.values()) + int(non_expert_gib * GIB)
    return {"arch": arch, "meta": meta, "block_count": block_count,
            "is_moe": True, "weight_bytes": weights,
            "non_expert_bytes": int(non_expert_gib * GIB),
            "other_bytes": GIB, "layer_bytes": int((non_expert_gib - 1) * GIB),
            "expert_bytes_per_layer": layers, "has_shexp": has_shexp,
            "unknown_type_tensors": 0}


def ot_rules(extra):
    """The -ot rules of an extra-flags string as [(pattern, target)], in
    order, after shlex processing (what llama-server actually receives)."""
    toks = shlex.split(extra)
    rules = []
    for i, tok in enumerate(toks):
        if tok == "-ot" and i + 1 < len(toks):
            for part in toks[i + 1].split(","):
                pattern, _, target = part.rpartition("=")
                rules.append((pattern, target))
    return rules


def ot_first_match(extra, tensor_name):
    """Mimic llama.cpp --override-tensor: the first rule (in flag order)
    whose regex search-matches the tensor name decides its placement;
    None means no override (default layer-split placement)."""
    for pattern, target in ot_rules(extra):
        if re.search(pattern, tensor_name):
            return target
    return None


def dense_layout(n_layers, per_layer_gib, other_gib=2.0, arch="testdense"):
    layer_bytes = int(per_layer_gib * GIB) * n_layers
    other = int(other_gib * GIB)
    meta = {"general.architecture": arch, f"{arch}.block_count": n_layers}
    return {"arch": arch, "meta": meta, "block_count": n_layers,
            "is_moe": False, "weight_bytes": layer_bytes + other,
            "non_expert_bytes": layer_bytes + other, "other_bytes": other,
            "layer_bytes": layer_bytes, "expert_bytes_per_layer": {},
            "unknown_type_tensors": 0}


def profile(ctx=8192, extra=""):
    # tiny ctx keeps heuristic KV (no attention fields in fake metas) small
    return {"config": {"ctx": ctx, "cache_type_k": "q8_0",
                       "cache_type_v": "q8_0", "extra": extra},
            "model_path": "/fake/model.gguf"}


class TestRangeRegex(unittest.TestCase):
    def assert_range(self, a, b):
        rx = re.compile("^(?:" + modelctl_tiers.range_regex(a, b) + ")$")
        for n in range(max(0, a - 3), b + 4):
            self.assertEqual(bool(rx.match(str(n))), a <= n <= b,
                             f"range {a}-{b}, value {n}")

    def test_ranges(self):
        for a, b in [(0, 0), (5, 5), (0, 9), (1, 9), (9, 10), (1, 19),
                     (20, 28), (29, 47), (0, 47), (1, 47), (0, 99),
                     (95, 105), (1, 128)]:
            self.assert_range(a, b)

    def test_layers_regex_runs(self):
        rx = re.compile("^(?:" + modelctl_tiers.layers_regex([1, 2, 3, 7, 8, 20]) + ")$")
        for n, want in [(1, True), (2, True), (3, True), (4, False),
                        (7, True), (8, True), (9, False), (20, True), (21, False)]:
            self.assertEqual(bool(rx.match(str(n))), want, n)


class TestSplitExtraFlags(unittest.TestCase):
    def test_strips_placement_keeps_rest(self):
        placement, other = modelctl_tiers.split_extra_flags(
            "-ot 'ffn_.*_exps=CPU' -ngl 10 --no-mmap --ubatch-size 128 --fit off")
        self.assertIn("-ot", placement)
        self.assertIn("-ngl", placement)
        self.assertIn("--no-mmap", placement)
        self.assertIn("--fit", placement)
        self.assertEqual(other, ["--ubatch-size", "128"])

    def test_empty(self):
        self.assertEqual(modelctl_tiers.split_extra_flags(""), ([], []))
        self.assertEqual(modelctl_tiers.split_extra_flags(None), ([], []))


class TestPlanTiers(unittest.TestCase):
    def test_tier1_fits_primary(self):
        plan = modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=dense_layout(10, 0.5))
        self.assertEqual(plan["tier"], 1)
        self.assertEqual(plan["config"]["device"], "SYCL0")
        self.assertEqual(plan["config"]["extra"], "")

    def test_tier2_splits_gpus(self):
        # ~36 GiB dense: over the primary card alone, under both combined
        plan = modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=dense_layout(30, 1.0))
        self.assertEqual(plan["tier"], 2)
        self.assertEqual(plan["config"]["split_mode"], "layer")
        # The split ratio comes from USABLE bytes (limit_pct, reserves and
        # any cache reservation applied), not raw card capacity: llama.cpp
        # distributes the model by these weights, so a raw-capacity ratio
        # ("8,3" here) over-filled whichever card had the bigger haircut.
        # 90% of 32 GiB -> 29, 90% of 12 GiB -> 11.
        self.assertEqual(plan["config"]["tensor_split"], "29,11")

    def test_moe_spill_assigns_fastest_first(self):
        # Laguna-like: 47 layers x 1.1 GiB experts, 4 GiB non-expert
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=8192), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=moe_layout(47, 1.1))
        self.assertIn(plan["tier"], (3, 4))
        extra = plan["config"]["extra"]
        # multi-device: needs split layer, explicit --device, --fit off
        self.assertEqual(plan["config"]["split_mode"], "layer")
        self.assertIn("--device SYCL0,SYCL1", extra)
        self.assertIn("--fit off", extra)
        # -ot: specific ranges before the CPU catch-all (first-match-wins)
        ot_pos = {tag: extra.index(tag) for tag in
                  ("=SYCL0", "=SYCL1", "=CPU")}
        self.assertLess(ot_pos["=SYCL0"], ot_pos["=CPU"])
        self.assertLess(ot_pos["=SYCL1"], ot_pos["=CPU"])
        # every expert layer appears in a specific range or under the catch-all
        toks = extra.split()
        ot = toks[toks.index("-ot") + 1]
        covered = set()
        has_catchall = False
        for part in ot.split(","):
            pattern = part.rsplit("=", 1)[0]
            # planner emits doubled backslashes for shlex; undo to get the regex
            pattern = pattern.replace("\\\\", "\\")
            m = re.match(r"blk\\\.(.*)\\\.ffn_", pattern)
            if not m:
                if pattern.startswith("ffn_"):
                    has_catchall = True
                continue
            rrx = re.compile("^(?:" + m.group(1) + ")$")
            for n in range(1, 48):
                if rrx.match(str(n)):
                    covered.add(n)
        if has_catchall:
            covered |= set(range(1, 48))
        self.assertEqual(covered, set(range(1, 48)))

    def test_moe_gpu_layers_respect_bandwidth_order(self):
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=8192), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=moe_layout(47, 1.1))
        rows = {label: (gib, desc) for label, gib, desc in plan["layout"]}
        # B70 gets more expert GiB than the B580
        self.assertGreater(rows["SYCL0"][0], rows["SYCL1"][0])
        # CPU holds the remainder
        self.assertIn("CPU", rows)

    def test_tier4_when_exceeding_ram(self):
        # Ornith-like: 60 layers x 2.9 GiB experts -- way past RAM
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=4096), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=moe_layout(60, 2.9))
        self.assertEqual(plan["tier"], 4)
        self.assertNotIn("--no-mmap", plan["config"]["extra"])
        self.assertTrue(any("SSD" in w for w in plan["warnings"]))

    def test_tier3_gets_no_mmap(self):
        # MoE past combined VRAM but comfortably inside GPU+RAM
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=8192), INVENTORY, 90, "SYCL0", ram_available=64 * GIB,
            layout=moe_layout(40, 1.2))
        self.assertEqual(plan["tier"], 3)
        self.assertIn("--no-mmap", plan["config"]["extra"])

    def test_dense_spill_computes_ngl(self):
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=4096), INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=dense_layout(60, 1.5))
        self.assertIn(plan["tier"], (3, 4))
        m = re.search(r"-ngl (\d+)", plan["config"]["extra"])
        self.assertIsNotNone(m)
        ngl = int(m.group(1))
        self.assertGreater(ngl, 0)
        self.assertLess(ngl, 60)

    def test_replan_strips_old_placement_flags(self):
        p = profile(extra="-ot 'ffn_.*_exps=CPU' -ngl 10 --no-mmap --ubatch-size 128")
        plan = modelctl_tiers.plan_tiers(
            p, INVENTORY, 90, "SYCL0", ram_available=25 * GIB,
            layout=moe_layout(47, 1.1))
        extra = plan["config"]["extra"]
        self.assertIn("--ubatch-size 128", extra)
        self.assertNotIn("-ngl 10", extra)
        self.assertEqual(extra.count("--no-mmap"), 1 if "--no-mmap" in extra else 0)

    def test_none_on_empty_layout(self):
        self.assertIsNone(modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0", layout=None))


class TestP5PinRules(unittest.TestCase):
    """P5 hard rule (landscape RQ7): shared experts and leading dense
    layers are pinned to a GPU by explicit -ot rules; only routed experts
    (_exps) ever flow to the CPU/offload tiers."""

    def _plan(self, layout, ram_gib=25, ctx=8192):
        return modelctl_tiers.plan_tiers(
            profile(ctx=ctx), INVENTORY, 90, "SYCL0",
            ram_available=ram_gib * GIB, layout=layout)

    def test_shexp_always_pinned_to_a_gpu(self):
        extra = self._plan(moe_layout(48, 1.1, first_layer=0,
                                      has_shexp=True))["config"]["extra"]
        for layer in (0, 19, 20, 25, 26, 47):  # GPU- and CPU-expert layers
            for stem in ("ffn_gate_shexp", "ffn_up_shexp",
                         "ffn_down_shexp", "ffn_gate_inp_shexp"):
                target = ot_first_match(extra, f"blk.{layer}.{stem}.weight")
                self.assertIn(target, ("SYCL0", "SYCL1"),
                              f"blk.{layer}.{stem} must be pinned to a GPU")

    def test_routed_experts_still_flow_to_tiers(self):
        extra = self._plan(moe_layout(48, 1.1, first_layer=0,
                                      has_shexp=True))["config"]["extra"]
        # per the greedy assignment: 0-19 -> SYCL0, 20-25 -> SYCL1, rest CPU
        self.assertEqual(
            ot_first_match(extra, "blk.0.ffn_gate_exps.weight"), "SYCL0")
        self.assertEqual(
            ot_first_match(extra, "blk.25.ffn_up_exps.weight"), "SYCL1")
        self.assertEqual(
            ot_first_match(extra, "blk.40.ffn_down_exps.weight"), "CPU")

    def test_dense_leading_layers_pinned_to_primary(self):
        # DeepSeek/GLM pattern: blk.0-2 dense, experts from blk.3
        extra = self._plan(moe_layout(61, 2.9, first_layer=3, has_shexp=True),
                           ctx=4096)["config"]["extra"]
        for layer in (0, 1, 2):
            for stem in ("ffn_gate", "ffn_up", "ffn_down", "ffn_norm"):
                self.assertEqual(
                    ot_first_match(extra, f"blk.{layer}.{stem}.weight"),
                    "SYCL0", f"dense blk.{layer}.{stem} must pin to primary")
        # the [0-2] range must not swallow two-digit layers (blk.20 etc.)
        self.assertEqual(
            ot_first_match(extra, "blk.20.ffn_gate_exps.weight"), "CPU")

    def test_no_shexp_rules_without_shexp(self):
        extra = self._plan(moe_layout(47, 1.1))["config"]["extra"]
        self.assertNotIn("_shexp", extra)

    def test_no_dense_pin_when_experts_start_at_layer_zero(self):
        extra = self._plan(moe_layout(48, 1.1, first_layer=0,
                                      has_shexp=True))["config"]["extra"]
        self.assertFalse(
            any(p.endswith("ffn_.*") for p, _ in ot_rules(extra)),
            "no dense-leading pin expected when every layer routes")

    def test_pins_precede_expert_rules(self):
        extra = self._plan(moe_layout(61, 2.9, first_layer=3, has_shexp=True),
                           ctx=4096)["config"]["extra"]
        rules = ot_rules(extra)
        kinds = ["dense" if p.endswith("ffn_.*")
                 else "shexp" if p.endswith("_shexp")
                 else "exps" for p, _ in rules]
        self.assertEqual(kinds, sorted(
            kinds, key=["dense", "shexp", "exps"].index),
            "pin rules must come before routed-expert rules")
        self.assertEqual(rules[-1], ("ffn_.*_exps", "CPU"))


class TestP5PlacementSnapshots(unittest.TestCase):
    """Frozen planner output for the affected MoE profiles.

    Hermetic mirrors: tests never read the real GGUFs, so each snapshot
    runs on a synthetic layout encoding the profile's documented shape --
    laguna-s2.1 (qwen35moe pattern: shexp, MoE from layer 0, RAM-resident
    tier 3) and ornith-397b (DeepSeek pattern: shexp + dense blk.0-2,
    SSD tier 4). Any intended planner change must re-freeze these."""

    def test_laguna_class_snapshot(self):
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=8192), INVENTORY, 90, "SYCL0",
            ram_available=31 * GIB,
            layout=moe_layout(48, 1.1, first_layer=0, has_shexp=True))
        self.assertEqual(plan["tier"], 3)
        self.assertEqual(plan["warnings"], [])
        self.assertEqual(plan["config"], {
            "device": "", "split_mode": "layer", "tensor_split": "8,3",
            "extra": r"--fit off --device SYCL0,SYCL1 "
                     r"-ot blk\\.(?:[0-9]|1[0-9])\\.ffn_.*_shexp=SYCL0,"
                     r"blk\\.2[0-5]\\.ffn_.*_shexp=SYCL1,"
                     r"ffn_.*_shexp=SYCL0,"
                     r"blk\\.(?:[0-9]|1[0-9])\\.ffn_.*_exps=SYCL0,"
                     r"blk\\.2[0-5]\\.ffn_.*_exps=SYCL1,"
                     r"ffn_.*_exps=CPU --no-mmap"})

    def test_ornith_class_snapshot(self):
        plan = modelctl_tiers.plan_tiers(
            profile(ctx=4096), INVENTORY, 90, "SYCL0",
            ram_available=25 * GIB,
            layout=moe_layout(61, 2.9, first_layer=3, has_shexp=True))
        self.assertEqual(plan["tier"], 4)
        self.assertEqual(plan["config"], {
            "device": "", "split_mode": "layer", "tensor_split": "8,3",
            "extra": r"--fit off --device SYCL0,SYCL1 "
                     r"-ot blk\\.[0-2]\\.ffn_.*=SYCL0,"
                     r"blk\\.[3-9]\\.ffn_.*_shexp=SYCL0,"
                     r"blk\\.1[0-1]\\.ffn_.*_shexp=SYCL1,"
                     r"ffn_.*_shexp=SYCL0,"
                     r"blk\\.[3-9]\\.ffn_.*_exps=SYCL0,"
                     r"blk\\.1[0-1]\\.ffn_.*_exps=SYCL1,"
                     r"ffn_.*_exps=CPU"})


if __name__ == "__main__":
    unittest.main()


class TestRecommendQuantGroup(unittest.TestCase):
    def _groups(self):
        return [
            {"label": "model-BF16", "total_size": 54 * GIB, "sharded": True},
            {"label": "model-Q8_0", "total_size": 28 * GIB, "sharded": False},
            {"label": "model-Q6_K", "total_size": 22 * GIB, "sharded": False},
            {"label": "model-Q4_K_M", "total_size": 16 * GIB, "sharded": False},
            {"label": "model-IQ2_XXS", "total_size": 9 * GIB, "sharded": False},
            {"label": "imatrix_unsloth", "total_size": 350 * (1 << 20), "sharded": False},
        ]

    def test_picks_largest_that_fits(self):
        # 28.7 GiB budget: Q8_0 (28 + ~3.8 overhead/KV) doesn't fit, Q6_K does
        rec = modelctl_tiers.recommend_quant_group(
            self._groups(), int(28.7 * GIB), 64000)
        self.assertEqual(rec["group"]["label"], "model-Q6_K")
        self.assertTrue(rec["fits"])

    def test_excludes_imatrix(self):
        rec = modelctl_tiers.recommend_quant_group(
            [{"label": "imatrix_unsloth", "total_size": 1 << 20}],
            30 * GIB, 64000)
        self.assertIsNone(rec)

    def test_fallback_is_smallest(self):
        rec = modelctl_tiers.recommend_quant_group(
            self._groups(), 2 * GIB, 64000)
        self.assertEqual(rec["group"]["label"], "model-IQ2_XXS")
        self.assertFalse(rec["fits"])

    def test_tiny_model_takes_bf16(self):
        groups = [{"label": "m-BF16", "total_size": 2 * GIB},
                  {"label": "m-Q4_K_M", "total_size": 1 * GIB}]
        rec = modelctl_tiers.recommend_quant_group(groups, 30 * GIB, 32768)
        self.assertEqual(rec["group"]["label"], "m-BF16")


class TestAutoCtx(unittest.TestCase):
    """auto_ctx: largest CTX_STEP fitting weights+exact KV+overhead."""
    def _write_model(self, context_length=262144):
        from test_modelctl_vram import gguf_bytes
        from tempfile import TemporaryDirectory
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "m.gguf"
        path.write_bytes(gguf_bytes({
            "general.architecture": (8, "testarch"),
            "testarch.block_count": (4, 32),
            "testarch.context_length": (4, context_length),
            "testarch.embedding_length": (4, 512),
            "testarch.attention.head_count": (4, 8),
            "testarch.attention.head_count_kv": (4, 4),
            "testarch.attention.key_length": (4, 128),
            "testarch.attention.value_length": (4, 128),
        }))
        return str(path)

    # KV/token here: 32 layers * 4 heads * 128 dims * 2 (K+V) * 1.0625 (q8_0)
    #   = 34816 B -> 131072 ctx ~= 4.25 GiB, 65536 ~= 2.1 GiB, 32768 ~= 1.1 GiB

    def test_picks_largest_fitting_step(self):
        rec = modelctl_tiers.auto_ctx(self._write_model(), 6 * GIB, "q8_0")
        self.assertEqual(rec["ctx"], 131072)
        self.assertTrue(rec["fits"])

    def test_drops_to_affordable_step(self):
        # 131072 (4.25 GiB KV) and 65536 (2.1 GiB) exceed 3 GiB with the
        # 1 GiB overhead; 32768 (1.06 GiB) fits.
        rec = modelctl_tiers.auto_ctx(self._write_model(), 3 * GIB, "q8_0")
        self.assertEqual(rec["ctx"], 32768)

    def test_model_max_caps(self):
        rec = modelctl_tiers.auto_ctx(self._write_model(context_length=32768),
                                      100 * GIB, "q8_0")
        self.assertEqual(rec["ctx"], 32768)
        self.assertEqual(rec["ctx"], rec["model_max"])

    def test_floor_when_nothing_fits(self):
        rec = modelctl_tiers.auto_ctx(self._write_model(), 1 << 20, "q8_0")
        self.assertEqual(rec["ctx"], modelctl_tiers.AUTO_CTX_FLOOR)
        self.assertFalse(rec["fits"])


class TestCacheRequest(unittest.TestCase):
    """cache_request: the planner reserves a UNIFORM per-GPU cache budget
    (the fork applies one --moe-cache-bytes to every device) before static
    expert placement."""

    def _req(self, budgets, mode="auto"):
        return {"mode": mode, "gpu": {"budgets_bytes": budgets}}

    def _plan(self, cache_request=None, **kw):
        args = dict(profile=profile(ctx=kw.pop("ctx", 8192)),
                    inventory=INVENTORY, limit_pct=90, primary="SYCL0",
                    ram_available=kw.pop("ram", 64 * GIB),
                    layout=kw.pop("layout", moe_layout(40, 1.2)))
        return modelctl_tiers.plan_tiers(cache_request=cache_request, **args)

    def test_budget_subtracted_shrinks_gpu_share(self):
        no_cache = self._plan()
        with_cache = self._plan(self._req({"SYCL0": 8 * GIB}))
        # uniform budget reserved on EVERY participating device
        self.assertEqual(with_cache["cache_budgets"],
                         {"SYCL0": 8 * GIB, "SYCL1": 8 * GIB})
        rows_nc = {l: g for l, g, _ in no_cache["layout"]}
        rows_wc = {l: g for l, g, _ in with_cache["layout"]}
        # reserved cache pushes more expert layers to CPU
        self.assertGreater(rows_wc["CPU"], rows_nc["CPU"])

    def test_per_device_budgets_collapse_to_uniform_max(self):
        plan = self._plan(self._req({"SYCL0": 4 * GIB, "SYCL1": 6 * GIB}))
        self.assertEqual(plan["cache_budgets"],
                         {"SYCL0": 6 * GIB, "SYCL1": 6 * GIB})
        self.assertEqual(plan["analysis"]["cache_budgets_gib"],
                         {"SYCL0": 6.0, "SYCL1": 6.0})

    def test_oversize_budget_shrinks_cache_loudly(self):
        # B580 usable is ~10.7 GiB; 20 GiB can't be reserved there, and the
        # fork has no per-device budgets -- so the uniform budget SHRINKS
        # to what the smallest device can give (never emitted as-is, never
        # disabled outright: the fallback-to-a-smaller-cache contract).
        req = self._req({"SYCL0": 20 * GIB})
        plan = self._plan(req)
        smallest_usable = int(B580["total_bytes"] * 0.9)
        self.assertEqual(plan["cache_budgets"],
                         {"SYCL0": smallest_usable, "SYCL1": smallest_usable})
        self.assertTrue(any("fall back to a smaller cache" in w
                            for w in plan["warnings"]))
        # Machine-readable degradation record: requested vs chosen.
        degr = plan["admission"]["degradations"]
        self.assertTrue(any(d["action"] == "shrink_cache"
                            and d["requested_bytes"] == 20 * GIB
                            and d["chosen_bytes"] == smallest_usable
                            for d in degr))
        # The degraded plan actually admits on every device it uses.
        self.assertTrue(plan["admission"]["fits"])

    def test_admission_report_present_and_fits(self):
        plan = self._plan(self._req({"SYCL0": 8 * GIB}))
        adm = plan["admission"]
        self.assertTrue(adm["fits"])
        for dev, row in adm["devices"].items():
            self.assertLessEqual(row["demand_bytes"], row["usable_bytes"], dev)
            self.assertEqual(row["demand_bytes"],
                             row["weights_bytes"] + row["kv_bytes"]
                             + row["overhead_bytes"]
                             + row["pinned_expert_bytes"] + row["cache_bytes"]
                             + row["compute_reserve_bytes"], dev)

    def test_layout_has_cache_rows(self):
        plan = self._plan(self._req({"SYCL0": 8 * GIB}))
        rows = {l: (g, d) for l, g, d in plan["layout"]}
        self.assertIn("SYCL0 (cache)", rows)
        self.assertIn("SYCL1 (cache)", rows)
        self.assertEqual(rows["SYCL0 (cache)"][0], 8.0)
        self.assertIn("cache", rows["SYCL0 (cache)"][1])

    def test_off_zero_and_absent_budgets(self):
        for req in (None,
                    self._req({}, mode="off"),
                    self._req({"SYCL0": 0}),
                    self._req({})):
            plan = self._plan(req)
            self.assertIsNone(plan["cache_budgets"], req)
            # docstring contract: the key is always present, value or None
            self.assertIn("cache_budgets_gib", plan["analysis"])
            self.assertIsNone(plan["analysis"]["cache_budgets_gib"], req)


RUNG_GPU = {"name": "RPC:ph16-71-cuda0:CUDA0",
            "endpoint": "192.168.0.76:50052",
            "local_device_index": 0, "kind": "gpu",
            "budget_bytes": 10 * GIB}
RUNG_CPU = {"name": "RPC:ph16-71-cpu0:CPU",
            "endpoint": "192.168.0.76:50053",
            "local_device_index": 0, "kind": "cpu",
            "budget_bytes": 24 * GIB}


class TestSelectInputs(unittest.TestCase):
    """A device selection is nothing more than filtered planner inputs.

    This is what lets the console ask the PLANNER where weights land
    instead of re-deriving placement in the browser: ticking a device off
    removes it from the inventory (or zeroes the RAM), and dragging a
    ceiling lowers that device's budget. One placement implementation,
    one answer, so the screen cannot disagree with what launches.
    """

    def test_an_unticked_gpu_leaves_the_inventory(self):
        inv, ram, rungs = modelctl_tiers.select_inputs(
            INVENTORY, 30 * GIB, [RUNG_GPU, RUNG_CPU],
            {"SYCL1": {"on": False}}, 90)
        self.assertEqual([d["device"] for d in inv], ["SYCL0"])
        self.assertEqual(ram, 30 * GIB)
        self.assertEqual(len(rungs), 2)

    def test_unticked_memory_is_zero_not_absent(self):
        """RAM is not a list entry that can be dropped -- it is a scalar
        the planner always reads, so "off" has to be nothing available."""
        _inv, ram, _rungs = modelctl_tiers.select_inputs(
            INVENTORY, 30 * GIB, [], {"RAM": {"on": False}}, 90)
        self.assertEqual(ram, 0)

    def test_an_unticked_rung_is_not_offered(self):
        _inv, _ram, rungs = modelctl_tiers.select_inputs(
            INVENTORY, 30 * GIB, [RUNG_GPU, RUNG_CPU],
            {RUNG_CPU["name"]: {"on": False}}, 90)
        self.assertEqual([r["name"] for r in rungs], [RUNG_GPU["name"]])

    def test_a_ceiling_lowers_what_the_planner_may_spend(self):
        """A GPU ceiling is expressed in the planner's own units: it reads
        limit_pct of total_bytes, so the ceiling scales the total it is
        given rather than being carried as a separate concept it would
        have to learn."""
        inv, _ram, _rungs = modelctl_tiers.select_inputs(
            INVENTORY, 30 * GIB, [], {"SYCL0": {"ceiling_bytes": 9 * GIB}}, 90)
        sycl0 = [d for d in inv if d["device"] == "SYCL0"][0]
        self.assertAlmostEqual(sycl0["total_bytes"] * 0.90 / GIB, 9.0, places=2)

    def test_a_ceiling_can_only_take_room_away(self):
        """Never grant more than the device physically has: a ceiling
        above capacity is the same as no ceiling at all."""
        inv, ram, rungs = modelctl_tiers.select_inputs(
            INVENTORY, 30 * GIB, [RUNG_CPU],
            {"SYCL0": {"ceiling_bytes": 999 * GIB},
             "RAM": {"ceiling_bytes": 999 * GIB},
             RUNG_CPU["name"]: {"ceiling_bytes": 999 * GIB}}, 90)
        self.assertEqual([d for d in inv if d["device"] == "SYCL0"][0]["total_bytes"],
                         [d for d in INVENTORY if d["device"] == "SYCL0"][0]["total_bytes"])
        self.assertEqual(ram, 30 * GIB)
        self.assertEqual(rungs[0]["budget_bytes"], RUNG_CPU["budget_bytes"])

    def test_an_empty_selection_changes_nothing(self):
        """No selection means the machine as it is -- the automatic
        placement, which is what the operator sees before touching it."""
        inv, ram, rungs = modelctl_tiers.select_inputs(
            INVENTORY, 30 * GIB, [RUNG_GPU], {}, 90)
        self.assertEqual(inv, INVENTORY)
        self.assertEqual(ram, 30 * GIB)
        self.assertEqual(rungs, [RUNG_GPU])

    def test_the_filtered_inputs_actually_plan(self):
        """End to end: the filter feeds plan_tiers unchanged, so a
        selection produces a real placement and not a shape error."""
        inv, ram, rungs = modelctl_tiers.select_inputs(
            INVENTORY, 40 * GIB, [RUNG_GPU, RUNG_CPU],
            {RUNG_GPU["name"]: {"on": False}}, 90)
        plan = modelctl_tiers.plan_tiers(
            profile(), inv, 90, "SYCL0", ram_available=ram,
            layout=moe_layout(60, 1.1, has_shexp=True), remote_rungs=rungs)
        extra = plan["config"]["extra"]
        self.assertNotIn(RUNG_GPU["endpoint"], extra)
        self.assertIn(RUNG_CPU["endpoint"], extra)


class TestPlanForSelection(unittest.TestCase):
    """The one entry point the placement UI needs: a device selection in,
    the planner's own placement out. Injected inputs keep it hermetic --
    nothing here reads the machine or a GGUF."""

    def _inputs(self, ram_gib=40):
        return {"inventory": INVENTORY, "vram_limit_pct": 90,
                "primary": "SYCL0", "ram_available_bytes": ram_gib * GIB}

    def _plan(self, selection, rungs=None, ram_gib=40):
        import modelctl_plans
        return modelctl_plans.plan_for_selection(
            profile(), selection, inputs=self._inputs(ram_gib),
            remote_rungs=rungs if rungs is not None else [RUNG_GPU, RUNG_CPU],
            layout=moe_layout(60, 1.1, has_shexp=True))

    def test_no_selection_is_the_automatic_placement(self):
        """What the machine would do on its own -- what the operator sees
        before touching anything."""
        plan = self._plan({})
        extra = plan["config"]["extra"]
        self.assertIn(RUNG_GPU["endpoint"], extra)
        self.assertIn(RUNG_CPU["endpoint"], extra)

    def test_turning_a_machine_off_moves_its_work(self):
        both = self._plan({})
        without = self._plan({RUNG_GPU["name"]: {"on": False},
                              RUNG_CPU["name"]: {"on": False}})
        self.assertIn(RUNG_CPU["endpoint"], both["config"]["extra"])
        self.assertNotIn(RUNG_CPU["endpoint"], without["config"]["extra"])
        # the weights did not evaporate: with the laptop gone they fall
        # back to the host, which is what makes the tier no better
        self.assertGreaterEqual(without["tier"], both["tier"])

    def test_a_ceiling_shrinks_what_that_device_takes(self):
        rungs = [RUNG_CPU]
        full = self._plan({}, rungs=rungs)
        capped = self._plan({RUNG_CPU["name"]: {"ceiling_bytes": 4 * GIB}},
                            rungs=rungs)

        def remote_gib(plan):
            rows = [r for r in plan["layout"] if RUNG_CPU["name"] in r[0]]
            return rows[0][1] if rows else 0.0

        self.assertGreater(remote_gib(full), remote_gib(capped))
        self.assertLessEqual(remote_gib(capped), 4.0)

    def test_an_unticked_card_is_absent_from_the_plan(self):
        plan = self._plan({"SYCL1": {"on": False}})
        placement = plan["config"].get("device", "") + plan["config"]["extra"]
        self.assertNotIn("SYCL1", placement)


class TestRemoteRungs(unittest.TestCase):
    """RPC devices join the expert-placement ladder. Ranks and the wire
    floor come from the 2026-08-03 122B calibration
    (docs/evidence/2026-08-03-qwen122b-fleet-calibration.md): remote GPU
    above local RAM, remote CPU above SSD but below local RAM."""

    def _plan(self, ram_gib, rungs, n_layers=60):
        return modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0",
            ram_available=ram_gib * GIB,
            layout=moe_layout(n_layers, 1.1, has_shexp=True),
            remote_rungs=rungs)

    def _remote_layer_count(self, extra, endpoint, layout):
        n = 0
        for layer in layout["expert_bytes_per_layer"]:
            tgt = ot_first_match(extra, f"blk.{layer}.ffn_gate_exps.weight")
            if tgt and endpoint in tgt:
                n += 1
        return n

    def test_cpu_share_that_fits_ram_is_held_resident(self):
        """Weights do not go to the SSD while RAM is free.

        Rungs absorb the bulk, so what is left for the host is small --
        4.4 GiB against 40 GiB of RAM. It streamed off the NVMe anyway,
        because --no-mmap was gated on the tier LABEL (tier 4 = 'needs
        SSD streaming') rather than on whether this plan's own CPU share
        actually fits. Measured on the live 122B: 72 GB read from disk
        and 25312 major faults for a share that fits in memory.
        """
        plan = self._plan(40, [RUNG_GPU, RUNG_CPU])
        self.assertEqual(plan["tier"], 4)
        cpu_share = [r for r in plan["layout"] if r[0] == "CPU"][0][1]
        self.assertLess(cpu_share, 40)
        self.assertIn("--no-mmap", plan["config"]["extra"])

    def test_cpu_share_bigger_than_ram_stays_mmapped(self):
        """The other direction, and the reason this is a fit check and
        not 'always --no-mmap': mmap degrades to slow paging, while a
        --no-mmap run that does not fit is an OOM kill."""
        plan = self._plan(2, [RUNG_GPU, RUNG_CPU])
        cpu_share = [r for r in plan["layout"] if r[0] == "CPU"][0][1]
        self.assertGreater(cpu_share, 2)
        self.assertNotIn("--no-mmap", plan["config"]["extra"])

    def test_tier4_offers_every_rung_before_ssd(self):
        layout = moe_layout(60, 1.1, has_shexp=True)
        plan = self._plan(8, [RUNG_GPU, RUNG_CPU])
        self.assertEqual(plan["tier"], 4)
        extra = plan["config"]["extra"]
        self.assertIn("--rpc", extra)
        self.assertLess(extra.index("--rpc"), extra.index("--device"))
        n_gpu = self._remote_layer_count(extra, "50052", layout)
        n_cpu = self._remote_layer_count(extra, "50053", layout)
        self.assertGreater(n_gpu, 0)
        self.assertGreater(n_cpu, 0)
        self.assertLessEqual(n_gpu * 1.1 * GIB, RUNG_GPU["budget_bytes"])
        self.assertLessEqual(n_cpu * 1.1 * GIB, RUNG_CPU["budget_bytes"])
        # a CPU tail remains and stays last (first-match catch-all)
        self.assertIn("ffn_.*_exps=CPU", extra)
        # tensor split carries a zero share per remote client device
        self.assertTrue(plan["config"]["tensor_split"].endswith(",0,0"),
                        plan["config"]["tensor_split"])
        toks = shlex.split(extra)
        devlist = toks[toks.index("--device") + 1]
        self.assertIn("RPC0", devlist.split(","))
        self.assertIn("RPC1", devlist.split(","))

    def test_shexp_never_goes_remote(self):
        layout = moe_layout(60, 1.1, has_shexp=True)
        plan = self._plan(8, [RUNG_GPU, RUNG_CPU])
        extra = plan["config"]["extra"]
        for layer in layout["expert_bytes_per_layer"]:
            tgt = ot_first_match(extra, f"blk.{layer}.ffn_gate_exps_shexp.weight")
            if tgt is not None:
                self.assertNotIn("RPC", tgt, f"layer {layer} shexp remote")

    def test_tier3_offers_gpu_rungs_only(self):
        plan = self._plan(60, [RUNG_GPU, RUNG_CPU])
        self.assertEqual(plan["tier"], 3)
        extra = plan["config"]["extra"]
        self.assertIn("50052", extra)
        self.assertNotIn("50053", extra)

    def test_admission_charges_the_rung_not_a_local_card(self):
        layout = moe_layout(60, 1.1, has_shexp=True)
        plan = self._plan(8, [RUNG_GPU, RUNG_CPU])
        adm = plan["config"]["rpc"]["admission"]
        n_gpu = self._remote_layer_count(plan["config"]["extra"], "50052",
                                         layout)
        self.assertEqual(adm["RPC:ph16-71-cuda0:CUDA0"],
                         n_gpu * int(1.1 * GIB))
        self.assertIn("RPC:ph16-71-cpu0:CPU", adm)
        self.assertTrue(plan["admission"]["fits"])

    def test_network_load_warning_present(self):
        plan = self._plan(8, [RUNG_GPU, RUNG_CPU])
        self.assertTrue(any("network" in w for w in plan["warnings"]))
        self.assertTrue(any("RPC-enabled build" in w
                            for w in plan["warnings"]))

    def test_wire_floor_skips_tiny_placements(self):
        # local GPUs absorb 26 layers; the single leftover layer
        # (~1.1 GiB) is under the 2 GiB wire floor, so no rung is used
        plan = self._plan(8, [RUNG_GPU, RUNG_CPU], n_layers=27)
        self.assertNotIn("--rpc", plan["config"]["extra"])

    def test_no_rungs_is_byte_identical_to_the_old_planner(self):
        import json as _json
        a = modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0", ram_available=8 * GIB,
            layout=moe_layout(60, 1.1))
        b = modelctl_tiers.plan_tiers(
            profile(), INVENTORY, 90, "SYCL0", ram_available=8 * GIB,
            layout=moe_layout(60, 1.1), remote_rungs=[])
        self.assertEqual(_json.dumps(a, sort_keys=True, default=str),
                         _json.dumps(b, sort_keys=True, default=str))


class TestPlacementInvariant(unittest.TestCase):
    """No weight byte streams off the SSD while an enabled device still
    has room to hold it.

    The rule, in Aaron's words on 2026-08-04: the SSD is overflow of last
    resort, never a default. It broke three separate ways that day and all
    three were fixed, but it was only ever probed by hand -- this is the
    standing form, so the next planner change cannot put it back quietly.

    The slack is the planner's real granularity, not a fudge factor.
    Experts move a whole layer at a time, so a device holding less than
    one layer of headroom has nowhere to put another one. Everything else
    -- the per-card compute reserve, the KV, the cache -- is already
    charged inside demand_bytes, so what is compared here is genuinely
    free space.

    Ceilings need no special case: select_inputs applies them by shrinking
    the device, so `usable_bytes` is already the room the OPERATOR left,
    and a capped device that spills is obeying an instruction rather than
    breaking the rule.
    """

    PER_LAYER_GIB = 1.1

    # (name, n_layers, ram_gib, rungs, selection)
    SCENARIOS = [
        ("everything on, model fits", 20, 40, [RUNG_GPU, RUNG_CPU], {}),
        ("big model, plenty of RAM", 60, 40, [RUNG_GPU, RUNG_CPU], {}),
        ("big model, thin RAM", 60, 8, [RUNG_GPU, RUNG_CPU], {}),
        ("big model, no laptop", 60, 8, [], {}),
        ("huge model, no laptop", 120, 8, [], {}),
        ("huge model, laptop on", 120, 8, [RUNG_GPU, RUNG_CPU], {}),
        ("laptop switched off", 60, 8, [RUNG_GPU, RUNG_CPU],
         {RUNG_GPU["name"]: {"on": False}, RUNG_CPU["name"]: {"on": False}}),
        ("second card capped", 60, 8, [RUNG_GPU, RUNG_CPU],
         {"SYCL1": {"ceiling_bytes": 2 * GIB}}),
        ("host memory capped", 60, 40, [RUNG_GPU, RUNG_CPU],
         {"RAM": {"ceiling_bytes": 2 * GIB}}),
        ("host memory switched off", 60, 40, [RUNG_GPU, RUNG_CPU],
         {"RAM": {"on": False}}),
        ("one card only", 60, 8, [], {"SYCL1": {"on": False}}),
    ]

    def _plan(self, n_layers, ram_gib, rungs, selection):
        import modelctl_plans
        inputs = {"inventory": INVENTORY, "vram_limit_pct": 90,
                  "primary": "SYCL0",
                  "ram_available_bytes": int(ram_gib * GIB)}
        plan = modelctl_plans.plan_for_selection(
            profile(), selection, inputs=inputs, remote_rungs=rungs,
            layout=moe_layout(n_layers, self.PER_LAYER_GIB, has_shexp=True))
        # The rungs the planner was ALLOWED to use. Read back through the
        # production filter rather than re-derived, so "enabled" means here
        # exactly what it meant to the planner.
        _inv, _ram, kept = modelctl_tiers.select_inputs(
            INVENTORY, int(ram_gib * GIB), rungs, selection, 90)
        return plan, kept

    def _streaming_host_gib(self, plan):
        """GiB the host holds off the SSD, or 0 when nothing streams.

        --no-mmap is the whole difference: the planner decides it by fit,
        so a tier-4 plan whose remainder fits is holding it resident.
        """
        if "--no-mmap" in plan["config"]["extra"].split():
            return 0.0
        for label, gib, _detail in plan["layout"]:
            if label == "CPU":
                return gib
        return 0.0

    def test_nothing_streams_while_an_enabled_device_has_room(self):
        layer_bytes = self.PER_LAYER_GIB * GIB
        for name, n_layers, ram_gib, rungs, selection in self.SCENARIOS:
            with self.subTest(scenario=name):
                plan, kept = self._plan(n_layers, ram_gib, rungs, selection)
                spilled = self._streaming_host_gib(plan)
                if not spilled:
                    continue
                for dev, row in plan["admission"]["devices"].items():
                    free = row["usable_bytes"] - row["demand_bytes"]
                    self.assertLess(
                        free, layer_bytes,
                        f"{name}: {spilled:.1f} GiB streams off the SSD while "
                        f"{dev} still has {free / GIB:.2f} GiB free -- room "
                        f"for another {self.PER_LAYER_GIB} GiB expert layer")
                used = (plan["config"].get("rpc") or {}).get("admission", {})
                for rung in kept:
                    free = rung["budget_bytes"] - used.get(rung["name"], 0)
                    self.assertLess(
                        free, layer_bytes,
                        f"{name}: {spilled:.1f} GiB streams off the SSD while "
                        f"{rung['name']} still has {free / GIB:.2f} GiB of its "
                        "declared budget unused")

    def test_nothing_streams_while_the_host_could_have_held_it(self):
        """The RAM leg of the same rule. Weights land on the SSD only
        because memory could not take them -- never while it could."""
        for name, n_layers, ram_gib, rungs, selection in self.SCENARIOS:
            with self.subTest(scenario=name):
                plan, _kept = self._plan(n_layers, ram_gib, rungs, selection)
                spilled = self._streaming_host_gib(plan)
                if not spilled:
                    continue
                budget = plan["analysis"]["ram_budget_gib"]
                self.assertGreater(
                    spilled, budget,
                    f"{name}: {spilled:.1f} GiB streams off the SSD although "
                    f"the RAM budget is {budget:.1f} GiB -- it should have "
                    "been held resident")

    def test_the_invariant_can_actually_fail(self):
        """A guard that never fires is not a guard. At least one scenario
        must reach the streaming branch, or both cases above would pass on
        an empty loop no matter what the planner did."""
        streamed = [name for name, n, r, rungs, sel in self.SCENARIOS
                    if self._streaming_host_gib(
                        self._plan(n, r, rungs, sel)[0])]
        self.assertTrue(streamed, "no scenario spills -- the invariant "
                                  "cases are vacuous")
