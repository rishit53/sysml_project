# 04 - Results

All tables below are generated from `results/data/*.csv` by `analysis/facts.py` (also saved in `results/tables/`);
figures are in `results/figures/` (PNG and vector PDF). Figure fN is Figure N+1 in the report.

## E1 - Baseline: scratchpad vs cache vs multicast bound  (figure f3)
| Batch | Prefix | Cache | Cache + mcfill | Oracle bound |
|---|---|---|---|---|
| 8 | 1K | 2.02× | 2.04× | 2.04× |
| 8 | 4K | 2.54× | 2.56× | 2.56× |
| 8 | 16K | 2.77× | 2.79× | 2.79× |
| 32 | 1K | 2.58× | 2.81× | 3.25× |
| 32 | 4K | 3.12× | 3.75× | 5.26× |
| 32 | 16K | 3.21× | 5.35× | 6.50× |
| 128 | 1K | 2.83× | 2.90× | 4.38× |
| 128 | 4K | 3.27× | 3.43× | 7.22× |
| 128 | 16K | 3.21× | 5.36× | 8.19× |

* Cache: 2.0-3.3x; multicast on merged misses: 3.4-5.4x; oracle: 5.3-8.2x (B ≥ 32, P ≥ 4K).
* Cache buffer utilisation 96-99%, HBM 22-27%.

## E2, E3 - Capacity and bandwidth  (figures f5, f4)
Deduplication needs concurrency, not capacity; the cache path scales with buffer bandwidth. Raw data: `e2_capacity.csv`, `e3_gb_bandwidth.csv`.

## E4 - Realistic sharing
Long private suffixes shrink everything (cache 1.5x, oracle 1.8-1.9x at a 4K suffix). Second model (Qwen2.5-7B): `e4_model_qwen.csv`.

## E5 - Dispatch order  (figure f6)
| Prefixes | Dispatch | Cache | Cache + mcfill | Oracle bound |
|---|---|---|---|---|
| 1 | group-major | 3.27× | 3.43× | 7.22× |
| 1 | request-major | 3.27× | 3.43× | 7.22× |
| 4 | group-major | 3.14× | 3.80× | 5.30× |
| 4 | request-major | 2.58× | 3.40× | 6.93× |
| 16 | group-major | 2.82× | 4.48× | 4.48× |
| 16 | request-major | 1.00× | 1.00× | 4.33× |

## E6 - Whole decode step at equal area  (figure f7)
| Batch | Prefix | Cache | Cache + mcfill | Oracle bound | Prefix-aware kernel |
|---|---|---|---|---|---|
| 32 | 4K | 1.58× | 1.66× | 1.78× | 1.80× |
| 32 | 16K | 2.29× | 2.99× | 3.26× | 3.45× |
| 128 | 4K | 2.31× | 2.37× | 3.37× | 3.16× |
| 128 | 16K | 2.86× | 4.31× | 5.85× | 6.03× |

## E7 - Roofline and bottleneck map  (figures f1, f2)
FLOP per KV byte 4; HBM roof 8.2 TFLOP/s; buffer roof without multicast 28.9; kernel ceiling 75.8; peak 369.6; HBM ridge 37 FLOP/B.
Scratchpad: HBM (9 of 9 points). Cache: global buffer (8 of 9 points). Multicast designs mostly leave no bandwidth resource above 80% busy.

## E8 - Window multicast  (figures f8, f9, f10)
| Batch | Prefix | mcfill | best window | oracle | window for 95% of oracle | table entries |
|---|---|---|---|---|---|---|
| 32 | 1K | 2.81× | 3.25× | 3.25× | 300 ns | 261 |
| 32 | 4K | 3.75× | 5.25× | 5.26× | 300 ns | 261 |
| 32 | 16K | 5.35× | 5.35× | 6.50× | none | - |
| 128 | 1K | 2.90× | 4.38× | 4.38× | 1000 ns | 716 |
| 128 | 4K | 3.43× | 7.22× | 7.22× | 3000 ns | 2,159 |
| 128 | 16K | 5.36× | 5.36× | 8.19× | none | - |

Bypass pinning (design v2) is mixed:

| Batch | Prefix | mcfill | window (3 µs) | window + pinning | oracle | HBM reduction: window / +pin / oracle |
|---|---|---|---|---|---|---|
| 32 | 1K | 2.81× | 3.25× | 3.25× | 3.25× | 4.4× / 4.4× / 4.4× |
| 32 | 4K | 3.75× | 5.25× | 5.25× | 5.26× | 11.3× / 11.3× / 11.3× |
| 32 | 16K | 5.35× | 5.35× | 6.22× | 6.50× | 13.0× / 18.1× / 21.7× |
| 128 | 1K | 2.90× | 4.38× | 4.38× | 4.38× | 4.8× / 4.8× / 4.8× |
| 128 | 4K | 3.43× | 7.14× | 7.14× | 7.22× | 15.1× / 15.1× / 15.1× |
| 128 | 16K | 5.36× | 5.36× | 4.98× | 8.19× | 13.0× / 6.9× / 43.3× |

## E9 - Drift robustness  (figure f11)
| Core speed variation | Cache | mcfill | window 300 ns | window 1 µs | oracle |
|---|---|---|---|---|---|
| ±0% | 3.27× | 3.43× | 4.86× | 6.57× | 7.22× |
| ±10% | 3.27× | 3.43× | 4.27× | 5.96× | 7.28× |
| ±25% | 3.28× | 3.43× | 3.95× | 4.94× | 7.50× |
| ±50% | 3.27× | 3.44× | 3.74× | 4.14× | 7.63× |

At B=128 the 3 µs window stays at ≥ 97.3% of the oracle (mean 98.5%) over every tested drift and seed; at B=32 the 1 µs window at ≥ 99.4% (mean 99.9%).

## E10 - Bandwidth allocation  (figure f12) and energy (figure f13)
| Batch | Buffer TB/s | Scratchpad | Cache | Window multicast (ours) | Oracle |
|---|---|---|---|---|---|
| 32 | 1.8 | 0.44× | 0.81× | 3.69× | 3.70× |
| 32 | 3.6 | 0.88× | 1.61× | 4.95× | 4.96× |
| 32 | 5.4 | 1.00× | 2.41× | 5.25× | 5.25× |
| 32 | 7.2 | 1.00× | 3.11× | 5.25× | 5.26× |
| 128 | 1.8 | 0.44× | 0.82× | 4.05× | 4.07× |
| 128 | 3.6 | 0.88× | 1.65× | 5.54× | 5.76× |
| 128 | 5.4 | 1.00× | 2.47× | 6.53× | 7.22× |
| 128 | 7.2 | 1.00× | 3.26× | 7.13× | 7.22× |

Energy values are estimates (bytes are exact): `results/data/e8_window_multicast.csv`, columns `E_*_uJ`.

## E11 - Silicon area of the added storage (needs an LLMCompass clone)
| Component | Entries | SRAM | Area (mm²) | % of global buffer |
|---|---|---|---|---|
| Global buffer (48 MB) - the reference | - | 48 MB | 113.7060 | 100.000% |
| Cache tags (one per 8 KB physical block) | 6,144 | 33.8 KB | 0.0841 | 0.074% |
| MSHRs (256 entries, tag + core bitmap) | 256 | 5.0 KB | 0.0490 | 0.043% |
| Multicast table, B=32 P=4K (300 ns window) | 261 | 2.0 KB | 0.0733 | 0.064% |
| Multicast table, B=128 P=1K (1 us window) | 716 | 5.6 KB | 0.0549 | 0.048% |
| Multicast table, B=128 P=4K (3 us window) | 2,159 | 16.9 KB | 0.0598 | 0.053% |
| Multicast table, largest in the sweep (B=128, 10000 ns window) | 3,712 | 29.0 KB | 0.1028 | 0.090% |

Areas come from LLMCompass's own 7 nm SRAM model (`cost_model.calc_cache_sram_area_mm2`). The model quantises small
arrays, so the table figures are an upper bound at the right order of magnitude, not a layout result. Logic, routing
and lookup timing are not modelled.

## Validation
| Check | Result |
|---|---|
| 1. Scratchpad vs LLMCompass attention | HBM bytes identical in 9 of 9 cases; time error -8.3% to -6.0% |
| 2. Zero-time mode vs earlier trace model | misses identical in 10 of 10 cases |
| 4. Sensitivity to latency / prefetch window | largest change 10% across all variants (see docs/02_methodology.md) |
