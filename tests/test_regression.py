"""Regression and behaviour tests.   python tests/test_regression.py      (or: pytest tests/)

1. The refactored experiments reproduce the team's original (shipped) results exactly.
2. Properties the simulator and the new design must satisfy.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _common as C  # noqa: E402

PAIRS = {"e1_baseline.csv": "x1_x3_main.csv", "e2_capacity.csv": "x2_capacity.csv", "e3_gb_bandwidth.csv": "x3_gb_bandwidth.csv",
         "e4_sharing.csv": "x4_sharing.csv", "e4_model_qwen.csv": "x4_model_qwen.csv", "e6_step_equal_area.csv": "x5_step_equal_area.csv"}


def test_matches_shipped_results():
    checked = 0
    for new, ref in PAIRS.items():
        if not (C.DATA / new).exists():
            continue
        a, b = pd.read_csv(C.DATA / new), pd.read_csv(ROOT / "results" / "reference" / ref)
        assert a.shape[0] == b.shape[0], f"{new}: row count {a.shape[0]} != {b.shape[0]}"
        for c in b.columns:
            if c in a.columns and pd.api.types.is_numeric_dtype(b[c]):
                x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                m = np.isfinite(x) & np.isfinite(y) & (y != 0)
                assert not m.any() or np.max(np.abs(x[m] - y[m]) / np.abs(y[m])) < 1e-9, f"{new}: column {c} differs"
        checked += 1
    print(f"  matches the shipped results: {checked} tables identical")


def sim1(policy, **kw):
    return C.job(dict(exp="t", policy=policy, batch=32, prefix=1024, **kw))


def test_scratchpad_saves_no_bandwidth():
    assert sim1("scratchpad")["dram_dedup_x"] == 1.0


def test_lockstep_is_independent_of_seed():
    assert sim1("cache", seed=0)["time_s"] == sim1("cache", seed=7)["time_s"]


def test_drift_is_deterministic_and_matters():
    a, b, c = sim1("cache", jitter=0.3, seed=1), sim1("cache", jitter=0.3, seed=1), sim1("cache", jitter=0.3, seed=2)
    assert a["time_s"] == b["time_s"] and a["time_s"] != c["time_s"]


def test_window_zero_matches_mcfill_and_buffer_reads_fall_with_window():
    m, w0 = sim1("cache-mcfill"), sim1("cache-wmc-0")
    assert abs(w0["time_s"] / m["time_s"] - 1) < 0.02
    reads = [sim1(f"cache-wmc-{w}")["gb_read_bytes"] for w in ("0", "100", "1000", "inf")]
    assert all(x >= y for x, y in zip(reads, reads[1:])), reads
    assert reads[-1] >= sim1("oracle-multicast")["gb_read_bytes"]


def test_oracle_is_an_upper_bound():
    t = {p: sim1(p)["time_s"] for p in ("scratchpad", "cache", "cache-mcfill", "cache-wmc-300", "oracle-multicast")}
    assert t["oracle-multicast"] <= min(t["cache"], t["cache-mcfill"], t["cache-wmc-300"]) * 1.001 and t["cache"] < t["scratchpad"]


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); n += 1
            print(f"PASS {name}")
    print(f"ALL {n} TESTS PASS")
