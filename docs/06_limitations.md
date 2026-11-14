# 06 - Limitations and what we did not verify

**Model scope**
* gbsim models **one attention layer** with an aggregate global-buffer bandwidth. No NoC topology, bank conflicts,
  real CTA scheduling or launch overheads (whole-step numbers use LLMCompass core time without launch overhead).
* The score matrix is not modelled (gbsim's naive kernel is fused per work item); LLMCompass's unfused attention
  moves 3-6% more bytes (validation check 1).
* The "cache" is an **upper bound for a conventional cache** (perfect miss merging); a GPU L2 comparison on a Tesla
  M60 reached a median 55% of the predicted gain. This one-off experiment is archived and **not reproduced here**.
* Latencies are not in LLMCompass; defaults are of the order of published A100 microbenchmarks and are hidden by the
  prefetch window (validation check 4). At double latency the cache path becomes latency-bound unless the window grows.

**Design evaluation**
* The window design is **evaluated in simulation only**. Its table is costed in entries and KB, not in silicon area
  or timing closure. The recent-multicast table is assumed to be an ideal associative structure.
* Window multicast is conceptually close to MeshKV's duplicate suppression (a short time window, multicast through
  the NoC). Our differences are cross-request sharing keyed by *physical block*, windows long enough to span core
  drift, and an evaluation at the global buffer; whether a reviewer sees this as incremental is a risk (see
  `05_novelty_and_related_work.md`).
* Drift is modelled as fixed per-core speed variation plus random start skew (E9). Real drift sources (cache
  conflicts, DRAM refresh, scheduler effects) are richer.

**Numbers**
* **Energy constants are estimates** (HBM 3.9, buffer 0.5, network 0.3 pJ/bit). Bytes per level are the hard numbers.
* Area of cache bookkeeping (0.1-0.5%) comes from the hand-off and needs an LLMCompass clone; not regenerated here.
* Whole-step numbers use the shipped LLMCompass step times (`llmc/out/prefix_results.csv`), not regenerated here.

**Literature**
* The prior-work check was a web search and reading exercise on 9 Nov 2026; absence of evidence is not proof of
  absence. The authors of this repository did not re-read every cited paper. The three starting-point papers named in
  the brief (MLX, SMOOTH, PADE) were not used as quantitative baselines.
* Statements about the TPU Ragged Paged Attention kernel (that it does not deduplicate shared pages) are inferred
  from its algorithm, not stated by its authors.
* **Small-array area is quantised.** LLMCompass's SRAM area model rounds small arrays up (its own tag arrays are non-monotonic in size), so the multicast table's 0.05-0.11 mm2 is an upper bound at the right order of magnitude, not a layout result. Logic, routing and lookup timing are not modelled.
