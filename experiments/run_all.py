"""Run experiments in order and write results/data/*.csv.

    python experiments/run_all.py                    # everything (e1..e10)
    python experiments/run_all.py --only e1 e5       # selected experiments
    python experiments/run_all.py --workers 8
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ORDER = ["e1_baseline", "e2_buffer_capacity", "e3_buffer_bandwidth", "e4_realistic_sharing", "e5_dispatch_order",
         "e6_step_equal_area", "e7_roofline", "e8_window_multicast", "e9_drift_robustness", "e10_bandwidth_allocation", "e11_area"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="experiment ids, e.g. e1 e5 (default: all)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    todo = [m for m in ORDER if not a.only or m.split("_")[0] in a.only]
    for m in todo:
        t = time.time()
        print(f"== {m}", flush=True)
        importlib.import_module(m).main(a.workers)
        print(f"   done in {time.time() - t:.0f}s\n", flush=True)


if __name__ == "__main__":
    main()
