#!/usr/bin/env python3
"""Project status: what has been run, the headline numbers, what is left.   python tools/status.py   (make status)"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import facts as N  # noqa: E402

DATA = ROOT / "results" / "data"
EXPS = [("E1", "baseline: scratchpad vs cache vs bound", "e1_baseline.csv"), ("E2", "buffer capacity", "e2_capacity.csv"),
        ("E3", "buffer bandwidth", "e3_gb_bandwidth.csv"), ("E4", "several prefixes, long suffix, 2nd model", "e4_sharing.csv"),
        ("E5", "dispatch order (correction)", "e5_dispatch_order.csv"), ("E6", "whole step, equal area", "e6_step_equal_area.csv"),
        ("E7", "roofline + bottleneck map", "e7_roofline.csv"), ("E8", "window-multicast design", "e8_window_multicast.csv"),
        ("E9", "drift robustness", "e9_drift.csv"), ("E10", "buffer bandwidth allocation", "e10_bandwidth_allocation.csv"),
        ("E11", "silicon area of the added storage", "e11_area.csv")]


def headline(i, h):
    try:
        f = {"E1": lambda: f"cache {h['cache_min']:.1f}-{h['cache_max']:.1f}x | oracle {h['oracle_big_min']:.1f}-{h['oracle_big_max']:.1f}x (B>=32, P>=4K)",
             "E4": lambda: f"cache at 16 prefixes, interleaved: {h['cache_16prefix_roundrobin']:.2f}x",
             "E5": lambda: f"B=128, 16 prefixes: cache {h['cache_16_rr']:.2f}x -> {h['cache_16_gm']:.2f}x with group-major",
             "E7": lambda: f"scratchpad: {h['bottleneck_scratchpad']}; cache: {h['bottleneck_cache']}",
             "E8": lambda: "window needed for 95% of oracle (B,P): " + ", ".join(f"({r.batch},{N.BP[r.prefix]}) {'none' if r.win95 != r.win95 else format(r.win95, 'g') + 'ns'}" for r in h['e8_summary'].itertuples()),
             "E9": lambda: "window-300ns speedup at B=128, jitter 0/10/25/50%: " + "/".join(f"{h['e9'][(h['e9'].batch == 128) & (h['e9'].skew_s == 0) & (h['e9'].jitter == j) & (h['e9'].policy == 'cache-wmc-300')].attn_speedup_vs_scratchpad.iloc[0]:.1f}" for j in (0.0, 0.1, 0.25, 0.5))}
        return f[i]()
    except Exception:
        return ""


def main():
    h = N.headline()
    print(f"PROJECT STATUS  {ROOT}   {time.strftime('%Y-%m-%d %H:%M')}\n")
    v = ROOT / "results" / "validation.txt"
    print(f"{'check':5s} {'validation':38s} {'DONE' if v.exists() else 'MISSING':8s} {'ALL PASS' if v.exists() and 'ALL PASS' in v.read_text() else ''}")
    for i, d, f in EXPS:
        p = DATA / f
        print(f"{i:5s} {d:38s} {'DONE' if p.exists() else 'MISSING':8s} {time.strftime('%m-%d %H:%M', time.localtime(p.stat().st_mtime)) if p.exists() else '-':12s} {headline(i, h) if p.exists() else ''}")
    nf = len(list((ROOT / "results" / "figures").glob("*.png")))
    rp = ROOT / "report" / "M1_final_report.pdf"
    print(f"\nfigures: {nf}/14   tables: {len(list((ROOT / 'results' / 'tables').glob('*.md')))}   report PDF: {'yes' if rp.exists() else 'NOT BUILT (make report)'}")
    print("\nREMAINING (manual): claims ledger check against cited papers | team review of report text | slides + rehearsal")


if __name__ == "__main__":
    main()
