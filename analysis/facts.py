"""Single source of truth for every number quoted in the docs and the report.

All values are computed from results/data/*.csv (nothing is typed by hand).
    python analysis/facts.py          # prints the headline numbers and writes results/tables/*.md|csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.style import DATA, ROOT, TABLES  # noqa: E402

SP = "attn_speedup_vs_scratchpad"
BP = {1024: "1K", 4096: "4K", 16384: "16K"}


def rd(n):
    return pd.read_csv(DATA / n)


def have(*names):
    return all((DATA / n).exists() for n in names)


def _main_pts():
    a = rd("e1_baseline.csv")
    return a[(a.order == "request-major") & (a.n_prefixes == 1) & (a.suffix == 256)]


def headline():
    """Dict of headline numbers (only the ones whose experiments have been run)."""
    h = {}
    if have("e1_baseline.csv"):
        a = _main_pts()
        big = a[(a.batch >= 32) & (a.prefix >= 4096)]
        c = a[a.policy == "cache"][SP]
        h.update(cache_min=c.min(), cache_max=c.max())
        for p, k in (("cache", "cache"), ("cache-mcfill", "mcfill"), ("oracle-multicast", "oracle")):
            s = big[big.policy == p][SP]
            h[f"{k}_big_min"], h[f"{k}_big_max"] = s.min(), s.max()
        cb = big[big.policy == "cache"]
        h.update(cache_gbutil_min=cb.gb_util.min(), cache_gbutil_max=cb.gb_util.max(), cache_hbmutil_min=cb.hbm_util.min(),
                 cache_hbmutil_max=cb.hbm_util.max(), cache_dedup_min=cb.dram_dedup_x.min(), cache_dedup_max=cb.dram_dedup_x.max())
        sp = a[a.policy == "scratchpad"]
        h["scratchpad_dedup_max"] = sp.dram_dedup_x.max()
    if have("e4_sharing.csv"):
        s = rd("e4_sharing.csv")
        q = s[(s.policy == "cache") & (s.n_prefixes == 16) & (s.suffix == 256)][SP]
        h["cache_16prefix_roundrobin"] = q.mean()
        long_ = s[(s.suffix == 4096) & (s.n_prefixes == 1)]
        h["cache_longsuffix"] = long_[long_.policy == "cache"][SP].mean()
        h["oracle_longsuffix_min"], h["oracle_longsuffix_max"] = (long_[long_.policy == "oracle-multicast"][SP].min(), long_[long_.policy == "oracle-multicast"][SP].max())
    if have("e5_dispatch_order.csv"):
        d = rd("e5_dispatch_order.csv")
        d = d[(d.batch == 128) & (d.n_prefixes == 16) & (d.suffix == 256)]
        g = lambda p, o: d[(d.policy == p) & (d.order == o)][SP].iloc[0]  # noqa: E731
        h.update(cache_16_rr=g("cache", "request-major"), cache_16_gm=g("cache", "group-major"), mcfill_16_gm=g("cache-mcfill", "group-major"),
                 oracle_16_gm=g("oracle-multicast", "group-major"), oracle_16_rr=g("oracle-multicast", "request-major"))
    if have("e7_roofline_constants.csv"):
        h.update(rd("e7_roofline_constants.csv").set_index("name").value.to_dict())
        e7 = rd("e7_roofline.csv")
        for p in ("scratchpad", "cache", "cache-mcfill", "oracle-multicast"):
            v = e7[e7.policy == p].bottleneck.value_counts()
            h[f"bottleneck_{p}"] = f"{v.index[0]} ({v.iloc[0]} of {v.sum()} points)"
    if have("e8_window_multicast.csv"):
        d = rd("e8_window_multicast.csv")
        a = d[d.exp == "e8a"]
        rows = []
        for (b, p), g in a.groupby(["batch", "prefix"]):
            orc, mc = g[g.policy == "oracle-multicast"][SP].iloc[0], g[g.policy == "cache-mcfill"][SP].iloc[0]
            w = g[g.policy.str.startswith("cache-wmc-") & (g.window_ns < float("inf"))].sort_values("window_ns")
            ok = w[w[SP] >= 0.95 * orc]
            rows.append(dict(batch=b, prefix=p, oracle=orc, mcfill=mc, best=w[SP].max(), best_window=w.loc[w[SP].idxmax(), "window_ns"],
                             win95=ok.window_ns.min() if len(ok) else np.nan, table95=ok.max_live.iloc[0] if len(ok) else np.nan))
        h["e8_summary"] = pd.DataFrame(rows)
    if have("e8_window_multicast.csv"):
        d = rd("e8_window_multicast.csv")
        a = d[d.exp == "e8a"]
        rows = []
        for (b, p), g in a.groupby(["batch", "prefix"]):
            q = g.set_index("policy")
            rows.append(dict(batch=b, prefix=p, mcfill=q.loc["cache-mcfill", SP], wmc3000=q.loc["cache-wmc-3000", SP], pin3000=q.loc["cache-wmcpin-3000", SP],
                             oracle=q.loc["oracle-multicast", SP], dedup_wmc=q.loc["cache-wmc-3000", "dram_dedup_x"], dedup_pin=q.loc["cache-wmcpin-3000", "dram_dedup_x"],
                             dedup_oracle=q.loc["oracle-multicast", "dram_dedup_x"]))
        h["e8_pin"] = pd.DataFrame(rows)
    if have("e10_bandwidth_allocation.csv"):
        h["e10"] = rd("e10_bandwidth_allocation.csv")
    if have("e9_drift.csv"):
        d = rd("e9_drift.csv")
        h["e9"] = d.groupby(["batch", "skew_s", "jitter", "policy"])[SP].mean().reset_index()
    return h


# ---------------------------------------------------------------- tables (DataFrames of strings)
def t_specs():
    from gbsim.config import HW, LLAMA3_8B, QWEN25_7B
    hw = HW()
    rows = [("Compute cores", f"{hw.cores}"), ("Core clock", f"{hw.freq / 1e9:.2f} GHz"),
            ("Peak MAC rate", f"{hw.cores * hw.mac_per_cycle * hw.freq * 2 / 1e12:.1f} TFLOP/s"),
            ("HBM bandwidth / loaded latency", f"{hw.hbm_bw / 1e12:.3f} TB/s / {hw.hbm_lat * 1e9:.0f} ns"),
            ("Global buffer capacity", f"{hw.gb_MB:g} MB"),
            ("Global buffer bandwidth / latency", f"{hw.gb_bw / 1e12:.2f} TB/s / {hw.gb_lat * 1e9:.0f} ns"),
            ("Outstanding block requests per core", f"{hw.window}"), ("Per-core port limit", "none (LLMCompass has no per-core limit)"),
            ("Model (main)", f"{LLAMA3_8B.name}: {LLAMA3_8B.n_layers} layers, {LLAMA3_8B.n_q} query / {LLAMA3_8B.n_kv} KV heads, d_head {LLAMA3_8B.d_head}, fp16 KV"),
            ("Model (second)", f"{QWEN25_7B.name}: {QWEN25_7B.n_layers} layers, {QWEN25_7B.n_q} / {QWEN25_7B.n_kv} heads"),
            ("Paged-KV block", f"16 tokens x K,V x d_head = {LLAMA3_8B.block_bytes // 1024} KB per KV head"),
            ("Workload", "B requests share a prefix of P tokens; 256 private tokens each; ordinary (not prefix-aware) kernel")]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def t_baseline():
    a = _main_pts()
    a = a[a.order == "request-major"]
    t = a.pivot_table(index=["batch", "prefix"], columns="policy", values=SP)[["cache", "cache-mcfill", "oracle-multicast"]]
    return pd.DataFrame({"Batch": [b for b, _ in t.index], "Prefix": [BP[p] for _, p in t.index],
                         "Cache": t["cache"].map("{:.2f}×".format).values, "Cache + mcfill": t["cache-mcfill"].map("{:.2f}×".format).values,
                         "Oracle bound": t["oracle-multicast"].map("{:.2f}×".format).values})


def t_dispatch():
    d = rd("e5_dispatch_order.csv")
    d = d[(d.batch == 128) & (d.suffix == 256)]
    t = d.pivot_table(index=["n_prefixes", "order"], columns="policy", values=SP)
    cols = ["cache", "cache-mcfill", "oracle-multicast"]
    return pd.DataFrame({"Prefixes": [k for k, _ in t.index], "Dispatch": [o for _, o in t.index],
                         **{n: t[c].map("{:.2f}×".format).values for n, c in zip(["Cache", "Cache + mcfill", "Oracle bound"], cols)}})


def t_step():
    d = rd("e6_step_equal_area.csv")
    d = d[d.split_id == "L192_G48"]
    rows = []
    for (b, p), g in d.groupby(["batch", "prefix"]):
        f = lambda pol: g[g.policy == pol].step_speedup_vs_scratchpad.iloc[0]  # noqa: E731
        rows.append([b, BP[p], f"{f('cache'):.2f}×", f"{f('cache-mcfill'):.2f}×", f"{f('oracle-multicast'):.2f}×",
                     f"{g[g.policy == 'scratchpad'].sw_kernel_speedup_vs_scratchpad.iloc[0]:.2f}×"])
    return pd.DataFrame(rows, columns=["Batch", "Prefix", "Cache", "Cache + mcfill", "Oracle bound", "Prefix-aware kernel"])


def t_window():
    h = headline()["e8_summary"]
    return pd.DataFrame({"Batch": h.batch, "Prefix": h.prefix.map(BP), "mcfill": h.mcfill.map("{:.2f}×".format),
                         "best window": h.best.map("{:.2f}×".format), "oracle": h.oracle.map("{:.2f}×".format),
                         "window for 95% of oracle": h.win95.map(lambda v: "none" if np.isnan(v) else f"{v:g} ns"),
                         "table entries": h.table95.map(lambda v: "-" if np.isnan(v) else f"{int(v):,}")})


def t_pin():
    h = headline()["e8_pin"]
    return pd.DataFrame({"Batch": h.batch, "Prefix": h.prefix.map(BP), "mcfill": h.mcfill.map("{:.2f}×".format), "window (3 µs)": h.wmc3000.map("{:.2f}×".format),
                         "window + pinning": h.pin3000.map("{:.2f}×".format), "oracle": h.oracle.map("{:.2f}×".format),
                         "HBM reduction: window / +pin / oracle": [f"{a:.1f}× / {b:.1f}× / {c:.1f}×" for a, b, c in zip(h.dedup_wmc, h.dedup_pin, h.dedup_oracle)]})


def t_bandwidth():
    d = headline()["e10"]
    rows = []
    for (b, tb), g in d.groupby(["batch", "gb_TBps"]):
        f = lambda p: g[g.policy == p].speedup_vs_baseline_chip.iloc[0]  # noqa: E731
        rows.append([b, f"{tb:g}", f"{f('scratchpad'):.2f}×", f"{f('cache'):.2f}×", f"{f('cache-wmc-3000'):.2f}×", f"{f('oracle-multicast'):.2f}×"])
    return pd.DataFrame(rows, columns=["Batch", "Buffer TB/s", "Scratchpad", "Cache", "Window multicast (ours)", "Oracle"])


def t_drift():
    g = headline()["e9"]
    g = g[(g.batch == 128) & (g.skew_s == 0)]
    cols = ["cache", "cache-mcfill", "cache-wmc-300", "cache-wmc-1000", "oracle-multicast"]
    t = g.pivot_table(index="jitter", columns="policy", values=SP)[cols]
    return pd.DataFrame({"Core speed variation": [f"±{int(j * 100)}%" for j in t.index],
                         "Cache": t["cache"].map("{:.2f}×".format).values, "mcfill": t["cache-mcfill"].map("{:.2f}×".format).values,
                         "window 300 ns": t["cache-wmc-300"].map("{:.2f}×".format).values, "window 1 µs": t["cache-wmc-1000"].map("{:.2f}×".format).values,
                         "oracle": t["oracle-multicast"].map("{:.2f}×".format).values})


def t_area():
    a = pd.read_csv(DATA / "e11_area.csv")
    sz = lambda v: f"{v / 1024:,.1f} KB" if v < 2**20 else f"{v / 2**20:,.0f} MB"  # noqa: E731
    return pd.DataFrame({"Component": a.component,
                         "Entries": a.entries.map(lambda v: "-" if not v else f"{int(v):,}"),
                         "SRAM": a.bytes.map(sz),
                         "Area (mm²)": a.area_mm2.map("{:.4f}".format),
                         "% of global buffer": a.pct_of_gb.map("{:.3f}%".format)})


def t_validation():
    v = ROOT / "results" / "validation"
    rows = []
    if (v / "check1_llmcompass.csv").exists():
        c = pd.read_csv(v / "check1_llmcompass.csv")
        rows.append(("1. Scratchpad vs LLMCompass attention", f"HBM bytes identical in {int((c.kv_bytes == c.gbsim_dram).sum())} of {len(c)} cases; time error {c.time_err.min() * 100:.1f}% to {c.time_err.max() * 100:.1f}%"))
    if (v / "check2_trace_model.csv").exists():
        c = pd.read_csv(v / "check2_trace_model.csv")
        rows.append(("2. Zero-time mode vs earlier trace model", f"misses identical in {int((c.old_misses == c.gbsim_misses).sum())} of {len(c)} cases"))
    if (v / "check4_sensitivity.csv").exists():
        c = pd.read_csv(v / "check4_sensitivity.csv")
        rows.append(("4. Sensitivity to latency / prefetch window", f"largest change {c.vs_default.abs().max() * 100:.0f}% across all variants (see docs/02_methodology.md)"))
    return pd.DataFrame(rows, columns=["Check", "Result"])


def to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


TABLES_FN = {"area": t_area, "pin": t_pin, "bandwidth": t_bandwidth, "specs": t_specs, "baseline": t_baseline, "dispatch": t_dispatch, "step": t_step, "window": t_window, "drift": t_drift, "validation": t_validation}


def export():
    TABLES.mkdir(parents=True, exist_ok=True)
    for n, f in TABLES_FN.items():
        try:
            df = f()
        except (FileNotFoundError, KeyError, IndexError):
            continue
        df.to_csv(TABLES / f"{n}.csv", index=False)
        (TABLES / f"{n}.md").write_text(to_md(df) + "\n")
        print(f"  table {n}")


if __name__ == "__main__":
    for k, v in headline().items():
        if not isinstance(v, pd.DataFrame):
            print(f"{k:28s} {v:.4g}" if isinstance(v, (int, float, np.floating)) else f"{k:28s} {v}")
    export()
