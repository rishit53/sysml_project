"""Validation of gbsim before any gap number is used.

  python -m gbsim.validate          checks 1, 2, 4 (seconds to a minute)

1. Scratchpad baseline vs LLMCompass "unshared" (prefix-cached, naive kernel) attention, GA100:
   DRAM KV bytes must match the analytical count exactly; per-layer time within 15%.
   (LLMCompass runs the attention unfused, so its bytes also include the score matrix,
   ~3-6% here; gbsim models a per-item fused kernel. Both are reported.)
2. Functional: in zero-time lockstep mode with a fully-associative LRU and window 1, gbsim's
   DRAM misses equal the earlier trace model's (llmc/studies/m1_prefix_cache.py) exactly.
3. (Removed from the handoff: a one-off GPU comparison, archived in
   archive/m1_gpu_anchor_2026-11-09.tar.gz. Its conclusion is in docs/GAP.md: gbsim's "cache"
   assumes perfect MSHR merging, so its numbers are an upper bound for a conventional cache.)
4. Sensitivity of the timing assumptions LLMCompass does not pin down (latencies, prefetch window):
   attention time moves by < 10% for 0.5x latency and window 6 / 16. Reported, not gated: 2x latency
   at window 8 (the cache path becomes latency-bound unless the window grows, which the last row shows).
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gbsim import policies as P, sim  # noqa: E402
from gbsim.config import HW, LLAMA3_8B  # noqa: E402
from gbsim.workload import Sharing, Workload  # noqa: E402

OUT = ROOT / "results" / "validation"
RESULTS = ROOT / "llmc" / "out" / "prefix_results.csv"


def check1():
    d = pd.read_csv(RESULTS)
    d = d[(d.design == "GA100") & (d.split_id.isin(["shipped", "L192_G48"])) & (d.variant == "unshared")
          & (d.group == "attention")]
    rows = []
    for (b, p), g in d.groupby(["batch", "prefix"]):
        if p > 16384:
            continue
        g = g.drop_duplicates("operator")
        llmc_s = g.core_s.sum() / 32
        llmc_bytes = g.sim_bytes.sum() / 32
        r = sim.layer(HW(), "scratchpad", batch=b, prefix=p, suffix=256)
        kv = b * LLAMA3_8B.n_kv * (p + 256) * 2 * LLAMA3_8B.d_head * 2
        rows.append(dict(batch=b, prefix=p, llmc_us=1e6 * llmc_s, gbsim_us=1e6 * r["time_s"],
                         time_err=r["time_s"] / llmc_s - 1, kv_bytes=kv, gbsim_dram=r["dram_bytes"],
                         llmc_bytes=llmc_bytes, kv_share_of_llmc_bytes=kv / llmc_bytes))
    t = pd.DataFrame(rows)
    ok = (t.gbsim_dram == t.kv_bytes).all() and (t.time_err.abs() < 0.15).all()
    return t, ok


def check2():
    spec = importlib.util.spec_from_file_location("old", ROOT / "llmc" / "studies" / "m1_prefix_cache.py")
    sys.path.insert(0, str(ROOT / "llmc"))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rows = []
    for b, p, cap in [(8, 1024, 11), (32, 4096, 11), (32, 16384, 27), (128, 4096, 27), (128, 16384, 67)]:
        for order in ("request-major", "head-major"):
            acc, miss, _ = old.simulate(b, p, 256, cap, order, "lru")
            hw = HW().with_(window=1, gb_MB=cap)
            r = sim.run(hw, P.Cache(cap * 2**20, LLAMA3_8B.block_bytes, None),
                        Workload(LLAMA3_8B, Sharing(b, p, 256), order), zero_time=True)
            rows.append(dict(batch=b, prefix=p, gb_MB=cap, order=order, old_misses=miss, gbsim_misses=r["misses"],
                             accesses=acc, gbsim_accesses=r["accesses"]))
    t = pd.DataFrame(rows)
    return t, bool((t.old_misses == t.gbsim_misses).all() and (t.accesses == t.gbsim_accesses).all())


def check4():
    base = HW()
    variants = {"default": base, "2x latency": base.with_(hbm_lat=2 * base.hbm_lat, gb_lat=2 * base.gb_lat),
                "0.5x latency": base.with_(hbm_lat=base.hbm_lat / 2, gb_lat=base.gb_lat / 2),
                "window 6": base.with_(window=6), "window 16": base.with_(window=16),
                "2x latency, window 16": base.with_(hbm_lat=2 * base.hbm_lat, gb_lat=2 * base.gb_lat, window=16)}
    rows = []
    for name, hw in variants.items():
        for pol in ("scratchpad", "cache"):
            for b, p in [(32, 4096), (128, 16384)]:
                r = sim.layer(hw, pol, batch=b, prefix=p)
                rows.append(dict(variant=name, policy=pol, batch=b, prefix=p, time_us=1e6 * r["time_s"],
                                 dram_dedup_x=r["dram_dedup_x"]))
    t = pd.DataFrame(rows)
    ref = t[t.variant == "default"].set_index(["policy", "batch", "prefix"]).time_us
    t["vs_default"] = [x.time_us / ref[(x.policy, x.batch, x.prefix)] - 1 for x in t.itertuples()]
    gated = t[t.variant.isin(["0.5x latency", "window 6", "window 16"])]
    return t, bool((gated.vs_default.abs() < 0.10).all())


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    checks = [("check1_llmcompass", check1), ("check2_trace_model", check2), ("check4_sensitivity", check4)]
    all_ok = True
    for name, fn in checks:
        t, ok = fn()
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n== {name}: {'PASS' if ok else 'FAIL'}")
        print(t.round(3).to_string(index=False))
        all_ok &= ok
    print("\nALL PASS" if all_ok else "\nSOME CHECKS FAILED (see above)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
