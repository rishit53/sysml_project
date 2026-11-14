# 03 - Experiment catalogue

Every experiment is one script in `experiments/` and writes one CSV (two for E4/E7) to `results/data/`.
`python experiments/run_all.py` runs them in order; `--only e1 e5` runs a subset.

| ID | Script | Question | Output | Figure |
|---|---|---|---|---|
| E1 | `e1_baseline.py` | How much bandwidth does each organisation recover, and how far is that from the best possible? (B x P x dispatch order x design) | `e1_baseline.csv` | F3 |
| E2 | `e2_buffer_capacity.py` | Does the cache need capacity or concurrency? (11-67 MB, shared-pin) | `e2_capacity.csv` | F5 |
| E3 | `e3_buffer_bandwidth.py` | Is the cache path bandwidth-capped? (3.3 / 4.4 / 7.2 TB/s) | `e3_gb_bandwidth.csv` | F4 |
| E4 | `e4_realistic_sharing.py` | Do the conclusions survive several prefixes, long private suffixes, a second model? | `e4_sharing.csv`, `e4_model_qwen.csv` | - |
| E5 | `e5_dispatch_order.py` | Is the many-prefix collapse a hardware limit or a scheduling artifact? | `e5_dispatch_order.csv` | F6 |
| E6 | `e6_step_equal_area.py` | After adding everything outside attention, at equal area, how does hardware compare with a prefix-aware software kernel? | `e6_step_equal_area.csv` | F7 |
| E7 | `e7_roofline.py` | What limits each design at each operating point? (roofline + bottleneck map) | `e7_roofline.csv`, `e7_roofline_constants.csv` | F1, F2 |
| E8 | `e8_window_multicast.py` | Proposed design: how large a window and table close the gap to the oracle? Also several prefixes and the pinned variant (v2) | `e8_window_multicast.csv` | F8, F9, F10, F13 |
| E9 | `e9_drift_robustness.py` | Does the design survive core drift (speed variation, start skew)? | `e9_drift.csv` | F11 |
| E11 | `e11_area.py` | Silicon area of the storage the design adds (needs an LLMCompass clone) | `e11_area.csv` | - |
| E10 | `e10_bandwidth_allocation.py` | How much global-buffer bandwidth does multicast save? (1.8-7.2 TB/s) | `e10_bandwidth_allocation.csv` | F12, F13 (energy, from E8) |

Figure file `fN_*` is **Figure N+1 in the report** (f0 is the schematic of the three organisations). Figure code is `analysis/figures.py`; all numbers quoted in
documents come from `analysis/facts.py`, which reads the CSVs.

## Run times (laptop, 16 workers)

E1-E6 about 1 minute together on a laptop; E7 instant; E8 and E10 under a minute; E9 a few minutes (about 20 minutes in total on one core). Finished simulations are cached in `results/.cache/`, so an interrupted run resumes.

## How to read a result file

Common columns: `policy`, `batch`, `prefix`, `suffix`, `n_prefixes`, `order`, `time_s` (layer time),
`attn_speedup_vs_scratchpad`, `dram_bytes`, `gb_read_bytes`, `gb_fill_bytes`, `dram_dedup_x`, `hbm_util`,
`gb_util`, `hits`, `misses`, `merges`, `E_*_uJ` (estimated energy per layer), `max_live` (design table entries).
