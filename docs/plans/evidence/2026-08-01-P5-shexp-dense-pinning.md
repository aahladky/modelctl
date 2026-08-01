# P5 — planner rule: pin shared experts and dense early layers

Date: 2026-08-01. Branch `agent/P5` off `staging`. Backlog item P5;
landscape doc RQ7 / steal-list item 5 ("shared-expert / dense-layer static
pinning as a hard rule").

## What changed

`modelctl place --tiers` (the only path that emits expert-offload `-ot`
rules) now pins shared experts and leading dense layers to a GPU with
explicit override rules, emitted ahead of every other rule so that no CPU
catch-all — the planner's own, or anything appended later in a profile's
extra flags — can ever take them. Only routed experts (`_exps`) may flow
to the CPU/offload tiers.

Emitted rule order in `_plan_moe_spill` (first-match-wins in llama.cpp):

1. `blk\.<leading-dense-range>\.ffn_.*=<primary>` — leading dense layers
   (GLM/DeepSeek pattern, e.g. blk.0-2; computed as the layers below the
   first routed-expert layer). Omitted when experts start at layer 0.
2. `blk\.<range>\.ffn_.*_shexp=<dev>` per GPU device — each layer's
   shared experts co-located with that layer's routed experts (the
   shared-expert output sums with the routed output, so the FFN input
   activation is already shipped there; no extra hop).
3. `ffn_.*_shexp=<primary>` — shexp of CPU-spilled layers (and any other
   shexp) pinned to the primary GPU.
4. `blk\.<range>\.ffn_.*_exps=<dev>` per GPU device (unchanged).
5. `ffn_.*_exps=CPU` catch-all (unchanged).

Supporting change: `modelctl_vram.gguf_model_layout` now reports
`has_shexp` (matched by `blk.N.ffn_.*_shexp.weight`, covering
`ffn_gate/up/down_shexp` and `ffn_gate_inp_shexp` per the fork's
gguf-py constants). Shexp bytes were already — deliberately — counted in
the non-expert (GPU-resident) set; that accounting is unchanged, so the
budget math did not move.

## Why (and what was already true)

Before this change the placement was only safe by regex accident: the CPU
catch-all `ffn_.*_exps` happens not to match `_shexp` or dense-FFN tensor
names, so shexp and dense layers stayed GPU-resident by default
placement, not by rule. The P5 hard rule makes the guarantee structural
and explicit, per landscape RQ7: shared experts are active every token
(worst possible cache clients), and dense early layers never route.
Runtime-wise this also guarantees the transfer cache and any future
offload tier only ever see routed-expert traffic from planner-emitted
configs.

Known approximation (documented in the code): the pinned bytes are part
of the fixed set the budget math spreads across cards by capacity ratio;
the pins concentrate a small slice (a few hundred MB on real models) onto
named devices. That divergence is absorbed by the per-card 1.5 GiB
compute reserve. Leading dense runs are 1–3 layers on the known
architectures; a hypothetical model with a very long dense prefix would
widen the divergence — noted, not handled.

## Affected profiles and snapshots

The two MoE profiles are the affected set:

- **laguna-s2.1** (qwen35moe pattern: shexp incl. `ffn_gate_inp_shexp`,
  MoE from layer 0 — per the 2026-07-30 real-model cache-activation
  report — RAM-resident tier 3): snapshot
  `TestP5PlacementSnapshots.test_laguna_class_snapshot`. No dense pin
  (experts start at blk.0); shexp pins mirror the expert assignment;
  shexp catch-all to SYCL0; `--no-mmap` retained.
- **ornith-397b** (DeepSeek-pattern per the P5 spec: shexp + dense
  blk.0-2, SSD tier 4): snapshot
  `TestP5PlacementSnapshots.test_ornith_class_snapshot`. Dense pin
  `blk\.[0-2]\.ffn_.*=SYCL0` first, then shexp pins, then expert rules.

Hermeticity: the suite must not read the real GGUFs, so each snapshot
runs the planner on a synthetic layout encoding the profile's documented
shape (layer count / per-layer expert bytes / shexp / first expert
layer), frozen to the exact emitted config dict. Any intended planner
change must consciously re-freeze them.

Behavioral tests besides the snapshots (`TestP5PinRules`): a
llama.cpp-faithful first-match evaluator asserts, against real tensor
names, that every shexp tensor resolves to a GPU on both GPU- and
CPU-expert layers; that routed experts still flow to their tiers; that
`blk\.[0-2]` cannot swallow two-digit layers; that no shexp/dense rules
appear for models without them; and that pin rules precede expert rules
with the CPU catch-all last. `TestGgufModelLayout` covers the
`has_shexp` flag and confirms shexp bytes stay non-expert.

## Gate results

- Placement snapshot test per affected profile: **green** (2 snapshots +
  6 pin-rule tests + 2 layout tests, all new).
- modelctl suite: **green** — full `unittest discover` in the worktree
  with the main checkout's venv python: 1114 tests, OK (skipped=11),
  218 s. All 26 pre-existing tiers tests pass unmodified.
- Fork tests: not run — no fork code touched (modelctl-only change; the
  emitted flags are consumed by stock `-ot` parsing).

## Consumer audit (no drift)

- `modelctl_plans._make_claim` probes `-ot` rules with
  `blk.N.ffn_gate_exps.weight` names for routed-expert layers only; the
  new pin patterns cannot match those probes (dense pins cover only
  expert-less layers; `_shexp$` patterns don't match `_exps` names), so
  VRAM claims are unchanged.
- The "CPU offload" placement labels in `modelctl_matrix`,
  `modelctl_plans`, and `modelctl_web/app.py` key on `exps=CPU`, which
  the catch-all still emits verbatim.
- `split_extra_flags` strips all `-ot` on re-plan, so the new rules are
  idempotent across re-planning, and stale pins cannot accrete.
- `modelctl_baseline` / `modelctl_acceptance` build their own
  routed-only (`ffn_.*_exps\.`) baseline configs for the three-condition
  protocol; they never emit shexp/dense offload and are out of P5 scope.

## Rollout

No profile is rewritten by this commit: pins appear the next time
`modelctl place --tiers --apply` runs on a MoE profile. Existing saved
placements keep working (they were shexp-safe by default placement).
Rollback = revert the commit; integration-manifest.json is untouched.
