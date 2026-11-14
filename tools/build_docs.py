"""Writes the data-driven documents: docs/00_overview.md, docs/04_results.md and the README key-results block.
    python tools/build_docs.py          (run after experiments + tables; `make docs`)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from analysis import facts as N  # noqa: E402

SP = "attn_speedup_vs_scratchpad"
h = N.headline()
e8 = h["e8_summary"].set_index(["batch", "prefix"])
pin = h["e8_pin"].set_index(["batch", "prefix"])
d9 = pd.read_csv(N.DATA / "e9_drift.csv")


def frac(b, pol):
    d = d9[d9.batch == b]
    a = d[d.policy == pol].merge(d[d.policy == "oracle-multicast"][["jitter", "skew_s", "seed", SP]], on=["jitter", "skew_s", "seed"], suffixes=("", "_o"))
    f = a[SP] / a[SP + "_o"]
    return f.min() * 100, f.mean() * 100


def e10(b, pol, tb):
    d = h["e10"]
    return d[(d.batch == b) & (d.policy == pol) & (d.gb_TBps.round(1) == tb)].speedup_vs_baseline_chip.iloc[0]


f128, f32 = frac(128, "cache-wmc-3000"), frac(32, "cache-wmc-1000")
area = pd.read_csv(N.DATA / "e11_area.csv") if (N.DATA / "e11_area.csv").exists() else None
if area is not None:
    gb_mm2 = area.iloc[0].area_mm2
    tab = area[area.component.str.startswith("Multicast table")]
    added_mm2 = tab.area_mm2.max() + area[~area.component.str.startswith("Multicast table")].iloc[1:].area_mm2.sum()
    AREA_LINE = (f"\n* **Area:** the 48 MB global buffer is {gb_mm2:.1f} mm2 at 7 nm; the multicast table costs at most "
                 f"{tab.area_mm2.max():.3f} mm2 ({tab.pct_of_gb.max():.2f}% of it), and all added storage together is "
                 f"{added_mm2:.2f} mm2 ({100 * added_mm2 / gb_mm2:.2f}%). The design fits inside the existing area budget.")
else:
    AREA_LINE = ""
KEY = f"""## Key results (all computed from `results/data/`, see `docs/04_results.md`)

* **Scratchpad:** prefix caching saves capacity, not bandwidth. HBM traffic is unchanged at every operating point; the scratchpad is HBM-bound everywhere.
* **Cache:** {h['cache_min']:.1f}-{h['cache_max']:.1f}x attention speedup, but the global buffer runs {h['cache_gbutil_min'] * 100:.0f}-{h['cache_gbutil_max'] * 100:.0f}% busy; a multicast bound sits at {h['oracle_big_min']:.1f}-{h['oracle_big_max']:.1f}x (B ≥ 32, P ≥ 4K).
* **Correction:** the cache's collapse with 16 prefixes ({h['cache_16_rr']:.2f}x) is a dispatch artifact: {h['cache_16_gm']:.2f}x when sharers are dispatched together.
* **Proposed design (time-window multicast buffer):** reaches the multicast bound when the per-layer prefix fits (P ≤ 4K), with a 0.3-3 µs window and at most {int(h['e8_summary'].table95.max()):,} table entries; stays at ≥ {f128[0]:.0f}% of the bound under heavy core drift when the window is sized for it; with half the buffer bandwidth it still beats a full-bandwidth cache ({e10(128, 'cache-wmc-3000', 3.6):.1f}x vs {e10(128, 'cache', 7.2):.1f}x at B=128).
{AREA_LINE}
* **Where it does not work:** prefixes larger than the buffer (P=16K) and interleaved dispatch. A prefix-aware software kernel reaches about the same whole-step speedup, so the hardware case is transparency and bandwidth provisioning, not raw speed.
"""

OVERVIEW = f"""# 00 - Overview: the whole project in plain words

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

{KEY}
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
* **"What if cores drift apart?"** A short fixed window degrades; a window sized for the drift (1-3 µs) holds within {f128[0]:.0f}% of the bound in our drift model (E9).
* **"What did you do versus the hand-off?"** The hand-off contained the simulator, X1-X5 and the validation. We re-organised the experiments, reproduced them exactly, corrected the many-prefix claim, added the roofline, the drift model, the window design and its experiments, and the report and tests.
"""

RESULTS = f"""# 04 - Results

All tables below are generated from `results/data/*.csv` by `analysis/facts.py` (also saved in `results/tables/`);
figures are in `results/figures/` (PNG and vector PDF). Figure fN is Figure N+1 in the report.

## E1 - Baseline: scratchpad vs cache vs multicast bound  (figure f3)
{N.to_md(N.t_baseline())}

* Cache: {h['cache_min']:.1f}-{h['cache_max']:.1f}x; multicast on merged misses: {h['mcfill_big_min']:.1f}-{h['mcfill_big_max']:.1f}x; oracle: {h['oracle_big_min']:.1f}-{h['oracle_big_max']:.1f}x (B ≥ 32, P ≥ 4K).
* Cache buffer utilisation {h['cache_gbutil_min'] * 100:.0f}-{h['cache_gbutil_max'] * 100:.0f}%, HBM {h['cache_hbmutil_min'] * 100:.0f}-{h['cache_hbmutil_max'] * 100:.0f}%.

## E2, E3 - Capacity and bandwidth  (figures f5, f4)
Deduplication needs concurrency, not capacity; the cache path scales with buffer bandwidth. Raw data: `e2_capacity.csv`, `e3_gb_bandwidth.csv`.

## E4 - Realistic sharing
Long private suffixes shrink everything (cache {h['cache_longsuffix']:.1f}x, oracle {h['oracle_longsuffix_min']:.1f}-{h['oracle_longsuffix_max']:.1f}x at a 4K suffix). Second model (Qwen2.5-7B): `e4_model_qwen.csv`.

## E5 - Dispatch order  (figure f6)
{N.to_md(N.t_dispatch())}

## E6 - Whole decode step at equal area  (figure f7)
{N.to_md(N.t_step())}

## E7 - Roofline and bottleneck map  (figures f1, f2)
FLOP per KV byte {h['flop_per_kv_byte']:.0f}; HBM roof {h['hbm_roof_tflops']:.1f} TFLOP/s; buffer roof without multicast {h['gb_roof_nomulticast_tflops']:.1f}; kernel ceiling {h['kernel_ceiling_tflops']:.1f}; peak {h['peak_tflops']:.1f}; HBM ridge {h['ridge_hbm_flop_per_byte']:.0f} FLOP/B.
Scratchpad: {h['bottleneck_scratchpad']}. Cache: {h['bottleneck_cache']}. Multicast designs mostly leave no bandwidth resource above 80% busy.

## E8 - Window multicast  (figures f8, f9, f10)
{N.to_md(N.t_window())}

Bypass pinning (design v2) is mixed:

{N.to_md(N.t_pin())}

## E9 - Drift robustness  (figure f11)
{N.to_md(N.t_drift())}

At B=128 the 3 µs window stays at ≥ {f128[0]:.1f}% of the oracle (mean {f128[1]:.1f}%) over every tested drift and seed; at B=32 the 1 µs window at ≥ {f32[0]:.1f}% (mean {f32[1]:.1f}%).

## E10 - Bandwidth allocation  (figure f12) and energy (figure f13)
{N.to_md(N.t_bandwidth())}

Energy values are estimates (bytes are exact): `results/data/e8_window_multicast.csv`, columns `E_*_uJ`.

## E11 - Silicon area of the added storage (needs an LLMCompass clone)
{N.to_md(N.t_area()) if (N.DATA / "e11_area.csv").exists() else "_(not run: clone LLMCompass next to this folder, then `make experiments`)_"}

Areas come from LLMCompass's own 7 nm SRAM model (`cost_model.calc_cache_sram_area_mm2`). The model quantises small
arrays, so the table figures are an upper bound at the right order of magnitude, not a layout result. Logic, routing
and lookup timing are not modelled.

## Validation
{N.to_md(N.t_validation())}
"""

NOVELTY = """# 05 - Novelty and related work

## What is claimed
1. **Characterisation.** On a software-managed scratchpad accelerator, prefix caching gives capacity but no decode bandwidth; a hardware-managed cache recovers part of it and is then capped by global-buffer bandwidth; a multicast bound sits well above. The comparison, its conditions and the roofline view are, in the sources we read, not stated.
2. **Correction.** A cache's collapse with many distinct prefixes is mostly a dispatch-order artifact (E5).
3. **A quantified mechanism.** A time-window multicast buffer, with its window, table size, drift robustness, bandwidth saving and two documented failure regimes (E8-E10).

## What is not claimed
Prefix caching, multicast hardware, merged-miss hardware (MSHR) and prefix-aware software kernels are all prior work. We do not claim hardware beats a prefix-aware software kernel.

## Status of each related idea
| Work / idea | What it does | Relation to this project |
|---|---|---|
| vLLM PagedAttention, SGLang RadixAttention | Prefix caching: shared physical KV pages | The workload we study |
| ChunkAttention (2402.15220), PAT (2511.22333), Hydragen (2402.05099) | GPU L2 partly dedups shared pages; prefix-aware kernels | GPU-side observations; software baselines |
| FlashInfer cascade, FlashForge (2505.17694), TyphoonMLA (2509.21081) | Prefix-aware software kernels (read shared KV once) | The main competitor: matches the oracle at step level (E6) |
| **MeshKV (2609.19207)** | NoC multicast with duplicate suppression in a ~12-cycle window, mainly across heads | **Closest hardware work.** Ours: across requests, keyed by physical block, microsecond windows, evaluated at the global buffer |
| DCO (2512.07312) | Shared cache over scratchpads in LLM accelerators | Cache-vs-scratchpad argument, not KV sharing |
| TPU Ragged Paged Attention (2604.15464) | Per-sequence page DMA into VMEM | Consistent with the scratchpad premise (inferred from its algorithm, not stated) |
| PRESERVE (2501.08192), MoSKA (2511.06010) | Prefetch; software | Adjacent |
| Buffets and related multicast storage idioms | Explicit multicast / merged requests in accelerator memory | General prior art for multicast delivery |

Source: `docs/handoff/prior_work_check.md` (9 Nov 2026; search and reading). We did not re-read every paper; absence of evidence is not proof of absence.

## Honest rating
**Medium-low to medium.** A careful characterisation, a correction of a stated failure, and a quantified mechanism with explicit limits. The nearest hardware work (MeshKV) uses a time window too, and a software kernel matches the speed, so the contribution is transparency, bandwidth provisioning and understanding, not a new phenomenon.

## Risk and the answer to it
*Risk:* a reviewer sees extending MeshKV's window to physical-block tags as trivial. *Answer:* the cross-request case needs block-table indirection and windows long enough to span core drift; we show window length must scale with batch and drift (E8, E9), that dispatch order is part of the design (E8b), and where it fails (oversized prefixes).
"""

(ROOT / "docs" / "00_overview.md").write_text(OVERVIEW)
(ROOT / "docs" / "04_results.md").write_text(RESULTS)
(ROOT / "docs" / "05_novelty_and_related_work.md").write_text(NOVELTY)
rd = (ROOT / "README.md").read_text()
a, b = rd.split("<!--RESULTS-->")[0], rd.split("<!--RESULTS-->")[-1]
if "<!--/RESULTS-->" in rd:
    a = rd.split("<!--RESULTS-->")[0]
    b = rd.split("<!--/RESULTS-->")[-1]
(ROOT / "README.md").write_text(a + "<!--RESULTS-->\n" + KEY + "<!--/RESULTS-->" + b)
print("wrote docs/00_overview.md, 04_results.md, 05_novelty_and_related_work.md and the README results block")
