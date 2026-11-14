# 00 - Overview: the whole project in plain words

## The problem in one minute
LLM serving systems (vLLM, SGLang) notice that many requests start with the same text (a system prompt, retrieved documents)
and store that shared text's KV cache **once**. That saves memory. But while generating each new token, every request
still has to *read* its whole KV cache, including the shared part, from HBM. On an accelerator with a software-managed
**scratchpad** (TPU-style), each core copies its own data, so the same shared block is read from HBM once **per request**.
Prefix caching therefore saves memory capacity but **no bandwidth**, and decode is bandwidth-bound.

## Three on-chip memory organisations
1. **Scratchpad:** every core fetches for itself. HBM-bound.
2. **Cache (like a GPU L2):** repeated reads hit on chip, so HBM traffic falls. But every core is still served separately, so the
   **global buffer's bandwidth** becomes the limit.
3. **Multicast buffer:** fetch a shared block once and **deliver it to all cores that want it at once.** This is the missing piece.
   The *oracle* is the unreachable best case of this idea; it tells us how much room there is.

## What we did
1. **Found the gap** by screening 11 candidate hardware/workload mismatches and keeping the one that survived tests and a literature check (`01_gap_discovery.md`).
2. **Built and validated a simulator** (`gbsim`, an event-driven timing model) before using any number (`02_methodology.md`).
3. **Evaluated existing designs** (E1-E7): speedups, bottlenecks, a roofline, whole-step results at equal chip area.
4. **Corrected a common claim** (E5): a cache's failure with many prefixes is mostly a dispatch-order effect.
5. **Proposed a design** (E8-E11): a time-window multicast buffer, tested for window size, table size, core drift, bandwidth needs and silicon area.
6. **Wrote it up** with reproducible code, tests and a PDF report (`report/M1_final_report.pdf`).

## Key results (all computed from `results/data/`, see `docs/04_results.md`)

* **Scratchpad:** prefix caching saves capacity, not bandwidth. HBM traffic is unchanged at every operating point; the scratchpad is HBM-bound everywhere.
* **Cache:** 2.0-3.3x attention speedup, but the global buffer runs 96-99% busy; a multicast bound sits at 5.3-8.2x (B ≥ 32, P ≥ 4K).
* **Correction:** the cache's collapse with 16 prefixes (1.00x) is a dispatch artifact: 2.82x when sharers are dispatched together.
* **Proposed design (time-window multicast buffer):** reaches the multicast bound when the per-layer prefix fits (P ≤ 4K), with a 0.3-3 µs window and at most 2,159 table entries; stays at ≥ 97% of the bound under heavy core drift when the window is sized for it; with half the buffer bandwidth it still beats a full-bandwidth cache (5.5x vs 3.3x at B=128).

* **Area:** the 48 MB global buffer is 113.7 mm2 at 7 nm; the multicast table costs at most 0.103 mm2 (0.09% of it), and all added storage together is 0.24 mm2 (0.21%). The design fits inside the existing area budget.
* **Where it does not work:** prefixes larger than the buffer (P=16K) and interleaved dispatch. A prefix-aware software kernel reaches about the same whole-step speedup, so the hardware case is transparency and bandwidth provisioning, not raw speed.

## Your folder, file by file
See the repository map in `README.md`. In one line each: `gbsim/` is the simulator; `experiments/eN_*.py` are the experiments (one question each);
`results/data` holds raw numbers, `results/figures` the plots, `results/tables` the tables; `analysis/` turns data into numbers and figures;
`report/` builds the PDF; `docs/` is the written project; `tests/` protects the results; `llmc/` is the earlier LLMCompass study.

## Talking points for the presentation
* **One sentence:** "Prefix caching saves memory but not bandwidth on a scratchpad accelerator; a cache is capped by its global-buffer bandwidth; a time-window multicast buffer, with sharing-aware dispatch, closes the gap when the prefix fits and lets the buffer run at half the bandwidth."
* **Show:** Figure 1 (three organisations), Figure 2 (roofline), Figure 4 (baseline), Figure 7 (the correction), Figure 9 (window sweep), Figure 12 (drift), Figure 13 (bandwidth).
* **Be upfront about:** simulation only; the cache is an upper bound; the design fails for oversized prefixes; a software kernel competes.

## Questions you will probably be asked, and honest answers
* **"Why hardware if a prefix-aware software kernel exists?"** The software kernel matches the oracle at whole-step level (Table 7). We do not claim hardware is faster. Hardware works with *ordinary* kernels, with dynamic or radix-tree sharing nobody tells the kernel about, and lets the buffer be provisioned with much less bandwidth (E10).
* **"Isn't this just MeshKV?"** Close in using a time window. Different in sharing *across requests* by physical block, windows of microseconds (not a dozen cycles) to span core drift, and evaluation at the global buffer. We rate novelty medium-low to medium.
* **"Is your simulator right?"** It reproduces LLMCompass's HBM bytes exactly and its time within 6-8%; reproduces the earlier trace model's misses exactly; and is insensitive to latency and prefetch assumptions within stated limits (`02_methodology.md`, section 6). Its cache assumes perfect miss merging, so it is an upper bound.
* **"What if the prefix does not fit?"** Then the multicast table cannot help (P=16K): the gap is capacity. Our pinned variant is mixed (helps B=32, hurts B=128). We state this as a limit.
* **"What if cores drift apart?"** A short fixed window degrades; a window sized for the drift (1-3 µs) holds within 97% of the bound in our drift model (E9).
* **"What did you do versus the hand-off?"** The hand-off contained the simulator, X1-X5 and the validation. We re-organised the experiments, reproduced them exactly, corrected the many-prefix claim, added the roofline, the drift model, the window design and its experiments, and the report and tests.
