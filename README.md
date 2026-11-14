# Prefix-cached decode on accelerators: scratchpad vs cache vs multicast global buffer

Course project **P07 - On-chip memory hierarchy design for LLM inference accelerators** (SysML).
Simulation study of how an accelerator's on-chip memory organisation determines whether *prefix caching* saves decode
**bandwidth** (not just capacity), and a proposed **time-window multicast global buffer**.

<!--RESULTS-->
## Key results (all computed from `results/data/`, see `docs/04_results.md`)

* **Scratchpad:** prefix caching saves capacity, not bandwidth. HBM traffic is unchanged at every operating point; the scratchpad is HBM-bound everywhere.
* **Cache:** 2.0-3.3x attention speedup, but the global buffer runs 96-99% busy; a multicast bound sits at 5.3-8.2x (B ≥ 32, P ≥ 4K).
* **Correction:** the cache's collapse with 16 prefixes (1.00x) is a dispatch artifact: 2.82x when sharers are dispatched together.
* **Proposed design (time-window multicast buffer):** reaches the multicast bound when the per-layer prefix fits (P ≤ 4K), with a 0.3-3 µs window and at most 2,159 table entries; stays at ≥ 97% of the bound under heavy core drift when the window is sized for it; with half the buffer bandwidth it still beats a full-bandwidth cache (5.5x vs 3.3x at B=128).

* **Area:** the 48 MB global buffer is 113.7 mm2 at 7 nm; the multicast table costs at most 0.103 mm2 (0.09% of it), and all added storage together is 0.24 mm2 (0.21%). The design fits inside the existing area budget.
* **Where it does not work:** prefixes larger than the buffer (P=16K) and interleaved dispatch. A prefix-aware software kernel reaches about the same whole-step speedup, so the hardware case is transparency and bandwidth provisioning, not raw speed.
<!--/RESULTS-->

## Start here

| If you want to... | Open |
|---|---|
| Read the final report | `report/M1_final_report.pdf` |
| Understand the whole project in plain words | `docs/00_overview.md` |
| See how we found the gap and what we rejected | `docs/01_gap_discovery.md` |
| See the simulator, specs, metrics, validation | `docs/02_methodology.md` |
| See every experiment, its script and its output | `docs/03_experiments.md` |
| See all results and figures | `docs/04_results.md`, `results/figures/` |
| See what is novel and what is not | `docs/05_novelty_and_related_work.md` |
| See the limits of the study | `docs/06_limitations.md` |

## Reproduce everything

```bash
pip install -r requirements.txt       # pandas, numpy, matplotlib, reportlab
make all                              # validate -> experiments E1-E9 -> tables -> figures -> report -> tests -> status
make status                           # at any time: what has run, headline numbers, what is left
```

Single stages: `make validate | experiments | tables | figures | report | test`. Without `make`:
`python -m gbsim.validate`, `python experiments/run_all.py [--only e1 e5] [--workers N]`, `python analysis/facts.py`,
`python analysis/figures.py`, `python report/build_report.py`, `python tests/test_regression.py`.
Everything is deterministic. gbsim needs only pandas; the area model and LLMCompass step times need an LLMCompass
clone (optional, see `requirements.txt`).

## Repository map

```
m1/
├── README.md                     this file
├── Makefile, requirements.txt
├── docs/                         the written project (read in numbered order)
│   ├── 00_overview.md  01_gap_discovery.md  02_methodology.md  03_experiments.md
│   ├── 04_results.md   05_novelty_and_related_work.md  06_limitations.md
│   └── handoff/                  the team's original gap write-up and prior-work check (unchanged)
├── gbsim/                        THE SIMULATOR (event-driven timing model of one decode-attention layer)
│   ├── config.py                 hardware + model parameters (GA100 template, Llama-3-8B, Qwen2.5-7B)
│   ├── workload.py               shared-prefix workload, block tables, dispatch orders
│   ├── policies.py               memory designs: scratchpad, cache, shared-pin, mcfill, WINDOW MULTICAST (ours), oracle
│   ├── sim.py                    the engine (HBM / buffer / port queues, drift model)
│   ├── energy.py  area.py        energy per level (estimates); cache bookkeeping area (needs LLMCompass)
│   └── validate.py               correctness checks against LLMCompass and the earlier trace model
├── experiments/                  ONE SCRIPT PER EXPERIMENT -> results/data/*.csv
│   ├── e1_baseline.py  e2_buffer_capacity.py  e3_buffer_bandwidth.py  e4_realistic_sharing.py
│   ├── e5_dispatch_order.py  e6_step_equal_area.py  e7_roofline.py
│   ├── e8_window_multicast.py  e9_drift_robustness.py  e10_bandwidth_allocation.py
│   ├── e11_area.py               silicon area of the added storage (needs an LLMCompass clone)
│   └── run_all.py  _common.py
├── analysis/                     facts.py (every quoted number), figures.py + style.py (all figures)
├── results/
│   ├── data/                     raw experiment CSVs
│   ├── figures/                  PNG (300 dpi) + PDF (vector), F0-F13
│   ├── tables/                   generated tables (md + csv)
│   ├── validation/               validation check outputs
│   └── reference/                the team's original results (regression baseline)
├── report/                       build_report.py -> M1_final_report.pdf
├── tests/test_regression.py      reproduces the original results; checks simulator and design properties
├── tools/                        status.py (make status), build_docs.py (make docs)
└── llmc/                         earlier LLMCompass-based study: trace models, equal-area splits, step times, screening ledger
```

## Provenance

Inherited from the team's hand-off: `gbsim` core (config, workload, sim, policies except the window design),
`llmc/`, experiments X1-X5 (re-organised here as E1-E6) and the validation checks. Added in this work: the window-multicast
policy, the drift model, dispatch order `group-major`, E5, E7, E8, E9, the analysis and figure code, the report, the tests
and this structure. The regression test shows that the re-organised experiments reproduce the original results exactly.
