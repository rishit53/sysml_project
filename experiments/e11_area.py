"""E11 -- Silicon area of the proposed design (needs an LLMCompass clone next to m1/).

The brief asks for an improvement within a fixed area budget. E6 already holds the budget fixed by
using the six equal-area SRAM splits (about 179 mm2 each, from llmc/splits.py). This experiment costs
the storage the proposed design ADDS, with LLMCompass's own 7 nm SRAM area model
(cost_model.calc_cache_sram_area_mm2), so the new hardware can be stated in mm2 rather than in entries.

Costed:
  * the 48 MB global buffer itself (the reference);
  * the cache bookkeeping a block-tagged buffer already needs (tags + MSHRs, from gbsim/area.py);
  * the recent-multicast table at the design points of E8, at ENTRY_BYTES per entry
    (a 32-bit physical block tag + a 16-bit window timer + a valid bit, rounded up to 8 B).

Output: results/data/e11_area.csv   (component, entries, bytes, area_mm2, pct_of_gb)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

DATA = ROOT / "results" / "data"
ENTRY_BYTES = 8
GB_MB = 48
# (label, batch, prefix, policy) design points from E8; the last is the worst case in the whole sweep
POINTS = [("Multicast table, B=32 P=4K (300 ns window)", 32, 4096, "cache-wmc-300"),
          ("Multicast table, B=128 P=1K (1 us window)", 128, 1024, "cache-wmc-1000"),
          ("Multicast table, B=128 P=4K (3 us window)", 128, 4096, "cache-wmc-3000")]


def area_model():
    """Returns (overhead dict from gbsim.area, f(bytes) -> mm2). Raises if LLMCompass is absent."""
    from gbsim import area as A
    o = A.overhead(GB_MB)                      # also performs the LLMCompass path setup
    from cost_model.cost_model import calc_cache_sram_area_mm2, find_logic_sram_transistor_density
    _, bitcell = find_logic_sram_transistor_density("7nm")
    return o, (lambda n: calc_cache_sram_area_mm2(int(n), bitcell))


def main(workers=None):
    try:
        o, mm2 = area_model()
    except Exception as e:                      # noqa: BLE001 - any import/path failure means no clone
        print(f"  skipped: needs an LLMCompass clone next to m1/ ({type(e).__name__}: {e})", flush=True)
        return
    gb = o["gb_mm2"]
    rows = [dict(component=f"Global buffer ({GB_MB} MB) - the reference", entries=0, bytes=GB_MB * 2**20, area_mm2=gb, pct_of_gb=100.0),
            dict(component="Cache tags (one per 8 KB physical block)", entries=GB_MB * 2**20 // 8192, bytes=o["tag_KB"] * 1024, area_mm2=o["tag_mm2"], pct_of_gb=100 * o["tag_mm2"] / gb),
            dict(component="MSHRs (256 entries, tag + core bitmap)", entries=256, bytes=o["mshr_KB"] * 1024, area_mm2=o["mshr_mm2"], pct_of_gb=100 * o["mshr_mm2"] / gb)]
    f = DATA / "e8_window_multicast.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d.exp == "e8a"]
        pts = list(POINTS)
        w = d[d.policy.str.startswith("cache-wmc-") & (d.window_ns < float("inf"))]
        if len(w):
            r = w.loc[w.max_live.idxmax()]
            pts.append((f"Multicast table, largest in the sweep (B={r.batch}, {r.window_ns:g} ns window)", r.batch, r.prefix, r.policy))
        for label, b, p, pol in pts:
            q = d[(d.batch == b) & (d.prefix == p) & (d.policy == pol)]
            if q.empty:
                continue
            n = int(q.max_live.iloc[0])
            a = mm2(n * ENTRY_BYTES)
            rows.append(dict(component=label, entries=n, bytes=n * ENTRY_BYTES, area_mm2=a, pct_of_gb=100 * a / gb))
    t = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    t.to_csv(DATA / "e11_area.csv", index=False)
    print(f"  wrote results/data/e11_area.csv ({len(t)} rows)")
    pd.set_option("display.width", 200)
    print(t.assign(area_mm2=t.area_mm2.round(5), pct_of_gb=t.pct_of_gb.round(4)).to_string(index=False))


if __name__ == "__main__":
    main()
