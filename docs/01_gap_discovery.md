# 01 - How we found the gap (the research trail)

This document is the *procedure*: how the team went from the open project brief to a specific, testable
gap, what was tried and rejected, and how the claim was checked. Everything here is traceable to a file in
this repository or in the original hand-off (`docs/handoff/`, `llmc/out/`).

## 1. The brief

P07 asks for: (1) a **roofline** characterisation of the target operator, (2) an evaluation of **existing
on-chip memory configurations** against it to find the dominant bottleneck at different operating points,
(3) a **proposed on-chip hierarchy** (SRAM sizing, tiling, bandwidth allocation) aimed at that bottleneck,
and (4) a simulated improvement in **throughput and energy within a fixed area budget**.

## 2. Screening: eleven candidate mismatches, each as a falsifiable hypothesis

Instead of picking a topic and looking for results, the team listed places where a common accelerator
design choice might not match a common LLM workload, and tested each one quickly (analytically or in the
LLMCompass simulator), then checked the literature. Sources: the team's gap write-up (P07, 1 Nov 2026)
and the project ledger `llmc/out/loop/`.

| # | Candidate mismatch | What we found | Outcome |
|---|---|---|---|
| 1 | Equal-area SRAM split for shared-prefix decode | Split is flat; a 25% spread was a softmax cost-model artifact | No lever; GA100 already on the plateau |
| 2 | Sliding-window KV pinned on-chip (gpt-oss, Gemma 3) | at most 2% where it fits | Refuted |
| 3 | Recurrent-state traffic in SSM hybrids | Real: 19-45% of step bytes | Already published (ReplaySSM, Pimba) |
| 4 | Diffusion-LLM decode becomes compute-bound | Real | Already published (Optimus, CDLM, DART) |
| 5 | MLA per-core row budget / split flip | Retracted after an independent check (latent counted twice); only 64 KB/core hurts | Dropped |
| 6 | Head-dim-256 prefill tiles | Fit at 128 KB/core and above | Refuted |
| 7 | SRAM split changes energy | at most 4.6% | Refuted |
| 8 | 32-64 KB local buffer breaks MLA in LLMCompass | at most 4% | Refuted |
| 9 | SRAM-resident chips: per-request footprint | Real | Mostly published (LIMINAL, SHIP) |
| 10 | Unfused attention erodes MLA/MTP traffic advantage (3-4x) | Real | Mechanism known (FLAT); kept as backup |
| **11** | **Scratchpad vs cache for prefix-cached decode (M1)** | **Real: 1.1-1.5x per decode step at measured bandwidth** | **Selected** |

A second screening round (recorded in the C4 hand-off) added: logits round-trips for huge vocabularies
(3% or less, refuted), head-dim vs 128-lane layouts (covered in practice), dispatch order and refcount-aware
pinning (partly covered), prefix sharing in hybrid models (partly covered), and **write-heavy hybrid decode
(C4)**.

### Decision M1 vs C4

C4 (hybrid SSM models write 10-34% of DRAM bytes, against the 5% that read-optimised memory designs assume)
had a stronger novelty argument, but it is a **DRAM-controller** problem evaluated with a DRAM simulator
(Ramulator). The brief asks for **on-chip SRAM** design, a roofline, an area budget and energy. C4 has no
accelerator model, no area model and no on-chip hierarchy question, so the team switched back to M1 on
fit-to-brief grounds. C4 is kept as a documented alternative; it is not part of this repository's results.

## 3. The M1 gap in one paragraph

With vLLM/SGLang **prefix caching** and an **ordinary (not prefix-aware) decode kernel**, many requests'
block tables point at the *same physical prefix blocks*. Prefix caching therefore saves memory *capacity*.
On a software-managed **scratchpad** accelerator it saves **no decode bandwidth**: every request's cores DMA
the shared prefix from HBM again, so shared bytes cross the HBM interface B times per step. A hardware-managed
**cache** recovers part of that, then becomes limited by **global-buffer bandwidth**, because every core is
still served separately. The missing piece is a mechanism that fetches a shared block once and **multicasts**
it to every core that needs it.

## 4. How the claim was built and checked (chronological)

1. **Trace model** (`llmc/studies/m1_prefix_cache.py`, `m1b_prefix_cache_jitter.py`, `m1c_threeway.py`):
   a trace-driven model of one layer's attention with 128 cores, 8 KB blocks and a fully-associative buffer.
   Result: ideal HBM-traffic reduction 3.3-43x; reuse is *temporal* (cores read the same blocks nearly in
   lockstep), so the conditions are core drift, replacement policy and dispatch order, not just capacity.
2. **Independent review** (`llmc/out/gap_review.md`): 13 issues were applied; the qualitative finding survived,
   the quantitative cache-path speedups were cut by 10-25%, and one validation claim was dropped because it was
   circular (a bandwidth was calibrated to the very measurement it was compared with).
3. **Prior-work check** (`docs/handoff/prior_work_check.md`, 9 Nov 2026): the contrast between a scratchpad and a
   cache for prefix-cached decode is not stated in the sources read; the nearest hardware work (MeshKV)
   deduplicates across *heads within a request*, not across *requests sharing a prefix*; prefix-aware
   *software* kernels (Hydragen, FlashInfer cascade, PAT) remove the redundant reads on any hardware. These
   software kernels are the main objection to the hardware argument (see `05_novelty_and_related_work.md`).
4. **Event-driven simulator `gbsim`** replaced the trace model so that timing (bandwidth, latency, queueing)
   is modelled, not only hit counts. It was validated (`02_methodology.md`, section 6) *before* any gap number
   was used.
5. **Experiments E1-E6** (the team's X1-X5, re-organised here) produced the gap evidence.
6. **This work** (E5, E7, E8, E9): found that the many-prefix failure of the cache is mostly a *dispatch
   artifact*, built the roofline/bottleneck analysis, proposed and evaluated the time-window multicast
   design, and tested it against core drift.

## 5. What we corrected along the way (honesty log)

| Earlier claim | Correction | Where |
|---|---|---|
| "A cache gets nothing with 16 distinct prefixes (1.00x)" | True only when sharers are interleaved in dispatch; with group-major dispatch the cache recovers (E5) | `results/data/e5_dispatch_order.csv` |
| 4.4 TB/s buffer bandwidth "validates" against ChunkAttention | The value was calibrated from that measurement; agreement is by construction | `docs/handoff/GAP.md` |
| gbsim "cache" speedups | Upper bound for a conventional cache: a one-off GPU L2 comparison (Tesla M60, archived) reached a median 55% of gbsim's predicted gain | `docs/handoff/GAP.md` |
| MLA split flip (project A) | Retracted: the latent was counted twice | ledger |
