"""All report figures, generated from results/data/*.csv.   python analysis/figures.py [name ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from analysis import style as S  # noqa: E402

SP = "attn_speedup_vs_scratchpad"
BP = {1024: "1K", 4096: "4K", 16384: "16K"}


def rd(n):
    return pd.read_csv(S.DATA / n)


def times(ax, axis="y"):
    f = (lambda v, _: f"{v:g}×")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(plt.FuncFormatter(f))


# ------------------------------------------------------------------ F0: three organisations
def f0_architectures():
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.9))
    titles = ["(a) Scratchpad", "(b) Cache", "(c) Window-multicast buffer (ours)"]
    notes = ["Shared block: read from HBM B times,\nand delivered B times", "Shared block: read from HBM once,\nbut delivered B times (buffer-bound)",
             "Shared block: read from HBM once,\nand broadcast once per time window"]
    cx = [1.4, 3.8, 6.2, 8.6]

    def box(ax, x, y, w, h, txt, fc, ec, fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arr(ax, x1, y1, x2, y2, c="#555555", lw=1.3):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=lw, color=c, shrinkA=0, shrinkB=0))

    for ax, t, n in zip(axs, titles, notes):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.grid(False)
        ax.set_title(t, loc="left", fontsize=10)
        box(ax, 0.6, 8.4, 8.8, 1.2, "HBM (prefix stored once)", "#E8EEF7", "#0072B2", 8.5, True)
        for x in cx:
            box(ax, x - 0.8, 0.9, 1.6, 1.1, "core", "#F4F4F4", "#8C8C8C", 8)
        ax.text(5, 0.1, n, ha="center", va="bottom", fontsize=8.2, color="#333333")
    # (a) per-core DMA straight from HBM, one staging buffer per core
    ax = axs[0]
    for x in cx:
        box(ax, x - 0.8, 3.7, 1.6, 1.1, "private\nstaging", "#EAF2FA", "#0072B2", 7.5)
        arr(ax, x, 8.4, x, 4.8, "#0072B2", 1.6)
        arr(ax, x, 3.7, x, 2.0)
    # (b) one read into a shared cache, then B reads out
    ax = axs[1]
    box(ax, 0.6, 4.6, 8.8, 1.8, "Global buffer (cache)\nhits served to each core separately", "#FDF0D5", "#E69F00", 8.2, True)
    arr(ax, 5, 8.4, 5, 6.4, "#E69F00", 2.0)
    for x in cx:
        arr(ax, x, 4.6, x, 2.0, "#E69F00", 1.6)
    # (c) cache with recent-multicast table and one broadcast
    ax = axs[2]
    box(ax, 0.6, 4.6, 8.8, 1.8, "", "#FBE3D6", "#D55E00")
    ax.text(2.2, 5.5, "Global buffer", fontsize=8.2, fontweight="bold", ha="center", va="center")
    box(ax, 4.0, 4.85, 5.1, 1.3, "recent-multicast table\n(block tag, window timer)", "#FFFFFF", "#D55E00", 7.6)
    arr(ax, 5, 8.4, 5, 6.4, "#D55E00", 2.0)
    ax.plot([1.4, 8.6], [3.2, 3.2], color="#D55E00", lw=2.4, solid_capstyle="round")
    arr(ax, 5, 4.6, 5, 3.3, "#D55E00", 2.0)
    for x in cx:
        arr(ax, x, 3.2, x, 2.0, "#D55E00", 1.4)
    ax.text(9.7, 3.2, "broadcast", fontsize=7, ha="right", va="bottom", color="#D55E00")
    fig.tight_layout()
    S.savefig(fig, "f0_architectures")


# ------------------------------------------------------------------ F1: roofline
def f1_roofline():
    d, k = rd("e7_roofline.csv"), rd("e7_roofline_constants.csv").set_index("name").value
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, (xc, bw, ttl) in zip(axs, (("I_dram", k.hbm_TBps * 1e12, "(a) against HBM bytes"), ("I_gb", k.gb_TBps * 1e12, "(b) against global-buffer bytes"))):
        x = np.logspace(-0.4, 2.8, 300)
        ax.plot(x, np.minimum(bw * x, k.kernel_ceiling_tflops * 1e12) / 1e12, color="#222222", lw=1.8, zorder=2)
        ax.axhline(k.peak_tflops, color="#888888", ls=":", lw=1)
        ax.text(0.5, k.peak_tflops * 1.05, f"peak MAC rate {k.peak_tflops:.0f} TFLOP/s", fontsize=7.5, color="#666666")
        ax.text(60, k.kernel_ceiling_tflops * 1.08, f"kernel ceiling {k.kernel_ceiling_tflops:.0f} TFLOP/s\n(GQA uses 1/4 of the array)", fontsize=7.5, color="#222222", va="bottom", ha="center")
        for p in ("scratchpad", "cache", "cache-mcfill", "oracle-multicast"):
            s = d[d.policy == p]
            ax.scatter(s[xc], s.perf_TFLOPs, s=38, c=S.color(p), marker=S.MARK[p], edgecolor="white", linewidth=0.6, label=S.LABEL[p], zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.4, 700); ax.set_ylim(1.5, 600)
        ax.set_xlabel("arithmetic intensity (FLOP per byte)"); ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.5)
    axs[0].set_ylabel("achieved performance (TFLOP/s)")
    axs[0].text(0.6, 3.1, f"HBM roof\n{k.hbm_TBps:.2f} TB/s", fontsize=7.5, rotation=34, color="#222222")
    axs[1].text(0.6, 3.1, f"buffer roof\n{k.gb_TBps:.1f} TB/s", fontsize=7.5, rotation=34, color="#222222")
    axs[0].annotate("scratchpad sits on the HBM roof:\n4 FLOP per byte, no reuse", xy=(4.0, 8.2), xytext=(0.55, 40), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#0072B2", lw=0.9), color="#0072B2")
    axs[1].annotate("caches sit on the buffer roof:\nevery core still reads the block", xy=(3.5, 26), xytext=(0.5, 110), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#E69F00", lw=0.9), color="#B07800")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    S.savefig(fig, "f1_roofline")


# ------------------------------------------------------------------ F2: bottleneck map
def f2_bottleneck():
    d = rd("e7_roofline.csv")
    pol = ["scratchpad", "cache", "cache-mcfill", "oracle-multicast"]
    pts = [(b, p) for b in (8, 32, 128) for p in (1024, 4096, 16384)]
    rc = {"HBM": "#BDD7EE", "global buffer": "#FBD9A6", "compute": "#BFE3C6", "latency/queueing": "#E3E3E3"}
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, pol_ in enumerate(pol):
        for j, (b, p) in enumerate(pts):
            r = d[(d.policy == pol_) & (d.batch == b) & (d.prefix == p)].iloc[0]
            ax.add_patch(plt.Rectangle((j, len(pol) - 1 - i), 1, 1, fc=rc[r.bottleneck], ec="white", lw=2.5))
            ax.text(j + .5, len(pol) - 1 - i + .5, f"{r[SP]:.1f}×\n{r.max_util * 100:.0f}% busy", ha="center", va="center", fontsize=7.8)
    ax.set_xlim(0, len(pts)); ax.set_ylim(0, len(pol)); ax.grid(False)
    ax.set_xticks(np.arange(len(pts)) + .5, [f"B={b}\nP={BP[p]}" for b, p in pts])
    ax.set_yticks(np.arange(len(pol)) + .5, [S.LABEL[x] for x in pol[::-1]])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Limiting resource per design and operating point (cell: attention speedup, busiest resource)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in rc.values()], list(rc), ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    S.savefig(fig, "f2_bottleneck_map")


# ------------------------------------------------------------------ F3: baseline speedups
def f3_baseline():
    d = rd("e1_baseline.csv")
    d = d[(d.order == "request-major") & (d.n_prefixes == 1) & (d.suffix == 256)]
    pol = ["cache", "cache-mcfill", "oracle-multicast"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, b in zip(axs, (8, 32, 128)):
        for i, p in enumerate((1024, 4096, 16384)):
            for j, pol_ in enumerate(pol):
                v = d[(d.batch == b) & (d.prefix == p) & (d.policy == pol_)][SP].iloc[0]
                x = i + (j - 1) * 0.26
                ax.bar(x, v, 0.24, color=S.color(pol_), label=S.LABEL[pol_] if (b == 8 and i == 0) else None)
                ax.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=7)
        ax.axhline(1, color=S.COLOR["scratchpad"], ls="--", lw=1.2)
        ax.set_xticks(range(3), [f"P={BP[p]}" for p in (1024, 4096, 16384)])
        ax.set_title(f"batch B={b}"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 9.4)
    axs[0].set_ylabel("attention speedup vs scratchpad (×)")
    axs[0].text(-0.42, 1.15, "scratchpad = 1×", fontsize=7.5, color=S.COLOR["scratchpad"], ha="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    S.savefig(fig, "f3_baseline_speedup")


# ------------------------------------------------------------------ F4: buffer bandwidth
def f4_bandwidth():
    d = rd("e3_gb_bandwidth.csv")
    pol = ["cache", "cache-mcfill", "oracle-multicast"]
    cases = [(32, 4096), (32, 16384), (128, 4096), (128, 16384)]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.3), sharey=True)
    for ax, (b, p) in zip(axs, cases):
        for pol_ in pol:
            s = d[(d.batch == b) & (d.prefix == p) & (d.policy == pol_)].sort_values("gb_TBps")
            ax.plot(s.gb_TBps, s[SP], marker=S.MARK[pol_], color=S.color(pol_), lw=1.8, ms=5, label=S.LABEL[pol_])
        ax.axhline(1, color=S.COLOR["scratchpad"], ls="--", lw=1.2)
        ax.set_xticks([3.3, 4.4, 7.2]); ax.set_title(f"B={b}, P={BP[p]}"); ax.set_xlabel("buffer bandwidth (TB/s)")
    axs[0].set_ylabel("attention speedup vs scratchpad (×)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    S.savefig(fig, "f4_buffer_bandwidth")


# ------------------------------------------------------------------ F5: buffer capacity
def f5_capacity():
    d = rd("e2_capacity.csv")
    cases = [(32, 4096), (32, 16384), (128, 4096), (128, 16384)]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.3), sharey=False)
    for ax, (b, p) in zip(axs, cases):
        for pol_ in ("cache", "cache-sharedpin"):
            s = d[(d.batch == b) & (d.prefix == p) & (d.policy == pol_)].sort_values("gb_MB")
            ax.plot(s.gb_MB, s.dram_dedup_x, marker=S.MARK[pol_], color=S.color(pol_), lw=1.8, ms=5, label=S.LABEL[pol_])
        ax.set_xticks([11, 27, 48, 67]); ax.set_title(f"B={b}, P={BP[p]}"); ax.set_xlabel("global buffer (MB)")
    axs[0].set_ylabel("HBM traffic reduction (×)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    S.savefig(fig, "f5_buffer_capacity")


# ------------------------------------------------------------------ F6: dispatch order and many prefixes
def f6_dispatch():
    d = rd("e5_dispatch_order.csv")
    d = d[(d.batch == 128) & (d.suffix == 256)]
    pol = ["cache", "cache-mcfill", "oracle-multicast"]
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=True)
    for ax, o, ttl in zip(axs, ("request-major", "group-major"), ("(a) sharers interleaved (vLLM-like dispatch)", "(b) sharers dispatched together")):
        for pol_ in pol:
            s = d[(d.order == o) & (d.policy == pol_)].sort_values("n_prefixes")
            ax.plot(s.n_prefixes, s[SP], marker=S.MARK[pol_], color=S.color(pol_), lw=1.9, ms=5.5, label=S.LABEL[pol_])
        ax.axhline(1, color=S.COLOR["scratchpad"], ls="--", lw=1.2)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16], ["1", "4", "16"]); ax.set_title(ttl)
        ax.set_xlabel("distinct shared prefixes in the batch")
    c = d[(d.order == "request-major") & (d.policy == "cache") & (d.n_prefixes == 16)][SP].iloc[0]
    g = d[(d.order == "group-major") & (d.policy == "cache") & (d.n_prefixes == 16)][SP].iloc[0]
    axs[0].annotate(f"cache gains nothing ({c:.2f}×)", xy=(16, c), xytext=(2.0, 1.9), fontsize=8, arrowprops=dict(arrowstyle="-", color="#B07800", lw=0.9), color="#B07800")
    axs[1].annotate(f"cache recovers to {g:.2f}×", xy=(16, g), xytext=(1.6, 1.4), fontsize=8, arrowprops=dict(arrowstyle="-", color="#B07800", lw=0.9), color="#B07800")
    axs[0].set_ylabel("attention speedup vs scratchpad (×)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    S.savefig(fig, "f6_dispatch_order")


# ------------------------------------------------------------------ F7: whole decode step
def f7_step():
    d = rd("e6_step_equal_area.csv")
    d = d[d.split_id == "L192_G48"]
    pts = [(32, 4096), (32, 16384), (128, 4096), (128, 16384)]
    series = [("cache", "cache", S.color("cache")), ("cache-mcfill", "cache-mcfill", S.color("cache-mcfill")),
              ("oracle-multicast", "oracle-multicast", S.color("oracle-multicast")), ("sw", "software-kernel", S.color("software-kernel"))]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for j, (key, lab, c) in enumerate(series):
        vals = []
        for b, p in pts:
            g = d[(d.batch == b) & (d.prefix == p)]
            vals.append(g[g.policy == "scratchpad"].sw_kernel_speedup_vs_scratchpad.iloc[0] if key == "sw" else g[g.policy == key].step_speedup_vs_scratchpad.iloc[0])
        x = np.arange(len(pts)) + (j - 1.5) * 0.2
        ax.bar(x, vals, 0.19, color=c, label=S.LABEL[lab])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.06, f"{v:.1f}", ha="center", fontsize=7)
    ax.axhline(1, color=S.COLOR["scratchpad"], ls="--", lw=1.2)
    ax.set_xticks(range(len(pts)), [f"B={b}, P={BP[p]}" for b, p in pts]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("whole decode-step speedup vs scratchpad (×)")
    ax.legend(ncol=2, loc="upper left")
    S.savefig(fig, "f7_step_speedup")


# ------------------------------------------------------------------ F8/F9/F10: the design
WIN = ["0", "30", "100", "300", "1000", "3000", "10000", "inf"]


def f8_window():
    d = rd("e8_window_multicast.csv")
    d = d[d.exp == "e8a"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
    for r, b in enumerate((32, 128)):
        for c, p in enumerate((1024, 4096, 16384)):
            ax = axs[r, c]
            g = d[(d.batch == b) & (d.prefix == p)]
            y = [g[g.policy == f"cache-wmc-{w}"][SP].iloc[0] for w in WIN]
            ax.plot(range(len(WIN)), y, marker="o", ms=4.5, lw=2, color=S.WMC["300"], label="Window multicast (ours)", zorder=3)
            for pol_, ls in (("cache", "-"), ("cache-mcfill", "--"), ("oracle-multicast", "--")):
                ax.axhline(g[g.policy == pol_][SP].iloc[0], color=S.color(pol_), ls=ls, lw=1.4, label=S.LABEL[pol_])
            for w, xi in (("1000", 4), ("3000", 5)):
                q = g[g.policy == f"cache-wmcpin-{w}"]
                if len(q):
                    ax.scatter([xi], q[SP], marker="D", s=34, color=S.PIN, zorder=4, label="Window multicast + pinning (ours, v2)" if (w == "1000" and r == 0 and c == 0) else None)
            ax.set_title(f"B={b}, P={BP[p]}")
            ax.set_xticks(range(len(WIN)), ["0", "30", "100", "300", "1k", "3k", "10k", "∞"])
            if c == 0:
                ax.set_ylabel("attention speedup (×)")
            if r == 1:
                ax.set_xlabel("multicast window (ns)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    S.savefig(fig, "f8_window_sweep")


def f9_table_cost():
    d = rd("e8_window_multicast.csv")
    d = d[(d.exp == "e8a") & d.policy.str.startswith("cache-wmc-") & (d.window_ns != float("inf"))]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (b, p), g in d.groupby(["batch", "prefix"]):
        g = g[g.window_ns > 0].sort_values("window_ns")
        ax.plot(g.window_ns, g.max_live, marker="o", ms=4, lw=1.6, label=f"B={b}, P={BP[p]}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("multicast window (ns)"); ax.set_ylabel("peak table entries")
    sec = ax.secondary_yaxis("right", functions=(lambda v: v * 8 / 1024, lambda v: v * 1024 / 8))
    sec.set_ylabel("table size at 8 B per entry (KB)")
    ax.legend(ncol=2, loc="upper left", fontsize=7.5)
    ax.set_title("Cost of the design: table occupancy")
    S.savefig(fig, "f9_table_cost")


def f10_multiprefix():
    d = rd("e8_window_multicast.csv")
    d = d[d.exp == "e8b"]
    pol = ["cache", "cache-mcfill", "cache-wmc-300", "cache-wmc-1000", "cache-wmc-inf", "oracle-multicast"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for ax, k in zip(axs, (4, 16)):
        for j, pol_ in enumerate(pol):
            for i, o in enumerate(("request-major", "group-major")):
                v = d[(d.n_prefixes == k) & (d.order == o) & (d.policy == pol_)][SP].iloc[0]
                x = i + (j - 2.5) * 0.14
                ax.bar(x, v, 0.13, color=S.color(pol_), label=S.label(pol_) if (k == 4 and i == 0) else None)
                ax.text(x, v + 0.05, f"{v:.1f}", ha="center", fontsize=6.5)
        ax.axhline(1, color=S.COLOR["scratchpad"], ls="--", lw=1.1)
        ax.set_xticks([0, 1], ["interleaved\n(request-major)", "sharers together\n(group-major)"]); ax.set_title(f"{k} distinct prefixes, B=128, P=4K"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("attention speedup vs scratchpad (×)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5)
    fig.tight_layout()
    S.savefig(fig, "f10_multi_prefix_design")


def f11_drift():
    d = rd("e9_drift.csv")
    pol = ["cache", "cache-mcfill", "cache-wmc-100", "cache-wmc-300", "cache-wmc-1000", "cache-wmc-3000", "oracle-multicast"]
    g = d.groupby(["batch", "skew_s", "jitter", "policy"])[SP].mean().reset_index()
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for r, b in enumerate((32, 128)):
        for c, sk in enumerate((0.0, 2e-6)):
            ax = axs[r, c]
            for pol_ in pol:
                s = g[(g.batch == b) & (g.skew_s == sk) & (g.policy == pol_)].sort_values("jitter")
                ax.plot(s.jitter * 100, s[SP], marker=S.MARK.get(pol_, "o"), ms=4.5, lw=1.8, color=S.color(pol_), label=S.label(pol_))
            ax.set_title(f"B={b}, P=4K, start skew {'none' if sk == 0 else '2 µs'}")
            ax.set_xticks([0, 10, 25, 50])
            if c == 0:
                ax.set_ylabel("attention speedup (×)")
            if r == 1:
                ax.set_xlabel("per-core speed variation (± %)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    S.savefig(fig, "f11_drift_robustness")


def f12_bandwidth_allocation():
    d = rd("e10_bandwidth_allocation.csv")
    pol = ["scratchpad", "cache", "cache-wmc-3000", "oracle-multicast"]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, b in zip(axs, (32, 128)):
        for pol_ in pol:
            s = d[(d.batch == b) & (d.policy == pol_)].sort_values("gb_TBps")
            ax.plot(s.gb_TBps, s.speedup_vs_baseline_chip, marker=S.MARK.get(pol_, "o"), ms=5.5, lw=2.0 if "wmc" in pol_ else 1.6, color=S.color(pol_), label=S.label(pol_))
        c = d[(d.batch == b) & (d.policy == "cache") & (d.gb_TBps.round(1) == 7.2)].speedup_vs_baseline_chip.iloc[0]
        ax.axhline(c, color=S.COLOR["cache"], ls=":", lw=1.2)
        ax.text(1.75, c + 0.12, f"cache at full bandwidth: {c:.1f}×", fontsize=7.5, color="#B07800")
        ax.set_xticks([1.8, 3.6, 5.4, 7.2]); ax.set_title(f"B={b}, P=4K"); ax.set_xlabel("global-buffer bandwidth (TB/s)")
    axs[0].set_ylabel("speedup vs the unmodified chip\n(scratchpad at 7.2 TB/s)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    fig.tight_layout()
    S.savefig(fig, "f12_bandwidth_allocation")


def f13_energy():
    d = rd("e8_window_multicast.csv")
    d = d[(d.exp == "e8a") & (d.prefix == 4096)]
    pol = ["scratchpad", "cache", "cache-mcfill", "cache-wmc-3000", "oracle-multicast"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, b in zip(axs, (32, 128)):
        bot = np.zeros(len(pol))
        for col, nm, c in (("E_dram_uJ", "HBM", "#4C78A8"), ("E_gb_uJ", "global buffer", "#F2A541"), ("E_noc_uJ", "buffer-to-core network", "#9AA5B1")):
            v = np.array([d[(d.batch == b) & (d.policy == p)][col].iloc[0] for p in pol]) / 1e3
            ax.bar(range(len(pol)), v, bottom=bot, color=c, label=nm, width=0.62)
            bot += v
        for i, t in enumerate(bot):
            ax.text(i, t + bot.max() * 0.02, f"{t:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(pol)), ["scratch-\npad", "cache", "cache +\nmcfill", "window 3 µs\n(ours)", "oracle"]); ax.grid(axis="x", visible=False)
        ax.set_title(f"B={b}, P=4K (estimated)"); ax.set_ylabel("attention energy per layer (mJ)")
    axs[0].legend(loc="upper right")
    fig.tight_layout()
    S.savefig(fig, "f13_energy")


ALL = [f0_architectures, f1_roofline, f2_bottleneck, f3_baseline, f4_bandwidth, f5_capacity, f6_dispatch, f7_step,
       f8_window, f9_table_cost, f10_multiprefix, f11_drift, f12_bandwidth_allocation, f13_energy]


def main(names):
    S.setup()
    for f in ALL:
        if not names or f.__name__.split("_")[0] in names or f.__name__ in names:
            try:
                f()
            except FileNotFoundError as e:
                print(f"  skipped {f.__name__}: missing {Path(str(e)).name} (run its experiment first)")


if __name__ == "__main__":
    main(sys.argv[1:])
