# BACKLOG — plain todo list

Numbered so items are easy to dispatch by name ("work on P4"). Update
your item's line when you start/finish; add a short note of where the
work lives (branch, files). Details on why these exist:
docs/research/2026-08-01-moe-offloading-landscape.md.

## The list

P1 — madvise hints for the SSD/mmap tier. CODE DONE on branch agent/P1
(WILLNEED after decode steps, DONTNEED on eviction, new test-mmap-advise
passing incl. sanitizer build). Still needs: run the ornith-397b bench
per the testing protocol, write the raw numbers down for Aaron. Baseline
to compare against: 0.37 tok/s static.

P2 — add a --moe-route-trace FILE flag to the fork's llama-server:
stream (token, layer, expert ids) per step, near-zero overhead when off.
This feeds P3. IN PROGRESS (session on agent/P2).

P3 — offline predictor study (Python only, needs P2's traces): implement
the ETH pre-attention linear probe (arXiv 2511.10676), measure top-k
recall vs the real router on laguna + a Qwen3.5-MoE, compare against a
simple "reuse last token's experts" baseline. Write the recall table up
for Aaron. The study picks WHICH predictor drives prefetch, not whether
prefetch gets built — it does (P12).

P4 — reorder same-expert tokens to compute contiguously in the
prefill/batch path (idea from arXiv 2511.14102). Outputs must stay
bit-identical; report the before/after throughput numbers. Wait until
P2's session frees the fork build dirs.

P5 — modelctl placement: always pin shared experts (shexp) and dense
early layers to GPU; only routed experts go through cache/offload.
IN PROGRESS (session running).

P6 — draft a comment for llama.cpp issue #20757 presenting the fork as
the C++ implementation the author asked for. Draft only — Aaron posts
it. IN PROGRESS (session running).

P7 — Phase G, the big one: hit/miss-aware decode dispatch (cached
experts compute on GPU, missed experts compute on CPU via activation
copy — Fiddler, arXiv 2402.07033). Write a short design note first
(how it engages at batch-1 without the global threshold, test plan,
off-by-default flag, rollback), then implement. Full test battery plus
a laguna benchmark with raw numbers for Aaron; the hope is beating
14.2 tok/s by a decent margin.

P8 — the B580's cache never engages (scheduler picks the first capable
backend). Fix it or rip out the dead config, with a short written
reason either way. Doesn't have to wait for P7.

P9 — batch "ticket" mode in modelctl: submit a prompt file as a job
against the SSD-tier model, results + stats land in a directory.

P10 — RAM upgrade memo: board's max DDR5, price 96/128GB kits, what it
does to the ornith on-disk tail. Research only; Aaron decides.
IN PROGRESS (session running).

P11 — console overhaul. First: teardown + proposed layout + 2-3 HTML
mockup directions (IN PROGRESS, session running; brief has the
details). Then: build a route-walk/wizard/cancel test harness BEFORE
restructuring, then restructure.

P12 — build prefetch into the fork (the moe_cache_prefetch capability):
asynchronously stage predicted experts so transfer overlaps compute.
Predictor = P3's winner; fall back to "reuse last token's experts" for
models without a trained probe. Literature: ETH probe 93-97% accuracy
(arXiv 2511.10676), ProMoE ~2x decode (2410.22134). Report hit-rate and
throughput numbers raw.

P13 — pinned-RAM tier-2 backing store + double-buffered async H2D
staging, per llama.cpp #20757's design (tinyserve PoC: 12-14 tok/s on
an 8GB GPU for GPT-OSS-120B). Misses copy from pinned memory
overlapping GPU compute instead of pageable mmap. Directly targets
RAM/SSD-bound serving.

P14 — mixed-precision expert tiers (HOBBIT, arXiv 2411.01433, up to
9.93x decode reported): cold experts stored/served at lower precision,
hot experts at full quant. Report speed AND output-quality numbers raw.

## Questions for Aaron

(none right now)

## Aaron's own list

- Decide whether agents may push branches to Gitea (currently push is
  blocked; everything is local).
- Post the P6 comment to #20757 if it reads well.
- Look at the P1 bench numbers when they exist.
