"""Analysis of out/prefix_results.csv (prefix_run.py).

    python prefix_analysis.py day1    # go/no-go gate
    python prefix_analysis.py sweep   # G-A1 split flip, G-A2 knee, G-A3 bound shift + figures

Latency is reported two ways: `latency_s` includes LLMCompass's A100 kernel
launch overheads (21 us per matmul, 12 us softmax, 45 us norm/elementwise);
`core_s` excludes them. The shared variant has more kernels, so overheads
penalise it; the gate uses core_s and both are printed.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common

RIDGE = 369.6e12 / 2.048e12  # GA100 FLOP/B
COLORS = {"unshared": "#2a78d6", "shared": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PT = ["batch", "prefix", "suffix"]


def load():
    df = pd.read_csv(common.OUT / "prefix_results.csv")
    return df[df.operator != "INFEASIBLE"].copy()


def per_point(df, keys):
    g = df.groupby(keys)
    out = g[["latency_s", "core_s", "sim_bytes", "derived_bytes", "flops"]].sum()
    att = df[df.group == "attention"].groupby(keys)[["latency_s", "core_s", "sim_bytes", "flops"]].sum()
    out = out.join(att, rsuffix="_attn")
    return out.reset_index()


def day1():
    df = load()
    df = df[df.split_id == "shipped"]
    t = per_point(df, ["design", "variant"] + PT)
    pd.set_option("display.width", 200)
    show = t.assign(
        step_ms=1e3 * t.latency_s, step_core_ms=1e3 * t.core_s,
        attn_ms=1e3 * t.latency_s_attn, attn_core_ms=1e3 * t.core_s_attn,
        attn_GB=t.sim_bytes_attn / 1e9, attn_FLOP_per_B=t.flops_attn / t.sim_bytes_attn,
        attn_share=t.core_s_attn / t.core_s,
    )
    print(show[["design", "variant"] + PT + ["step_ms", "step_core_ms", "attn_ms", "attn_core_ms",
                                             "attn_GB", "attn_FLOP_per_B", "attn_share"]]
          .round(3).to_string(index=False))
    for (d, b, p, s), g in t.groupby(["design"] + PT):
        u = g[g.variant == "unshared"].iloc[0]
        sh = g[g.variant == "shared"].iloc[0]
        speed = u.core_s_attn / sh.core_s_attn
        speed_ovh = u.latency_s_attn / sh.latency_s_attn
        bred = u.sim_bytes_attn / sh.sim_bytes_attn
        share = u.core_s_attn / u.core_s
        step = u.core_s / sh.core_s
        ok = speed >= 3 and bred >= 3 and share >= 0.2
        print(f"\n{d} B={b} P={p} S={s}: attention speedup {speed:.2f}x core "
              f"({speed_ovh:.2f}x with launch overhead), DRAM bytes {bred:.2f}x fewer, "
              f"attention = {share:.0%} of unshared step, step speedup {step:.2f}x core")
        print(f"GATE (>=3x attn speed, >=3x bytes, attn >=20% of step): {'PASS' if ok else 'FAIL'}")


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def sweep():
    df = load()
    ga = df[(df.design == "GA100") & (df.split_id != "shipped")]
    t = per_point(ga, ["split_id", "local_KB", "gb_MB", "variant"] + PT)
    pd.set_option("display.width", 220)

    # G-A1: best split per variant, and the cost of using the other variant's best split
    rows = []
    for (b, p, s), g in t.groupby(PT):
        r = dict(batch=b, prefix=p, suffix=s)
        best = {}
        for v in ("unshared", "shared"):
            gv = g[g.variant == v].set_index("split_id")
            if gv.empty:
                continue
            for metric in ("core_s", "latency_s"):
                best[(v, metric)] = gv[metric].idxmin()
            r[f"{v}_best"] = best[(v, "core_s")]
            r[f"{v}_spread_pct"] = 100 * (gv.core_s.max() / gv.core_s.min() - 1)
            r[f"{v}_ga100_gap_pct"] = 100 * (gv.core_s["L192_G48"] / gv.core_s.min() - 1)
        if ("unshared", "core_s") in best and ("shared", "core_s") in best:
            gs = g[g.variant == "shared"].set_index("split_id").core_s
            r["shared_loss_on_unshared_best_pct"] = 100 * (gs[best[("unshared", "core_s")]] / gs.min() - 1)
            gu = g[g.variant == "unshared"].set_index("split_id").core_s
            r["unshared_loss_on_shared_best_pct"] = 100 * (gu[best[("shared", "core_s")]] / gu.min() - 1)
        rows.append(r)
    flip = pd.DataFrame(rows)
    flip.to_csv(common.OUT / "prefix_split_flip.csv", index=False)
    print("G-A1 split flip (core time, whole decode step):")
    print(flip.round(2).to_string(index=False))

    # G-A2: shared attention vs global buffer size
    knee = t.pivot_table(index=["variant"] + PT, columns="gb_MB", values="core_s_attn") * 1e3
    print("\nG-A2 attention core ms vs global buffer MB:")
    print(knee.round(3).to_string())
    kb = t.pivot_table(index=["variant"] + PT, columns="gb_MB", values="sim_bytes_attn") / 1e9
    print("\nG-A2 attention simulated DRAM GB vs global buffer MB:")
    print(kb.round(3).to_string())

    # G-A3: arithmetic intensity of attention vs ridge, at the GA100 split
    base = t[t.split_id == "L192_G48"].copy()
    base["attn_flop_per_byte"] = base.flops_attn / base.sim_bytes_attn
    base["attn_util"] = base.flops_attn / (base.core_s_attn * 369.6e12)
    print("\nG-A3 attention intensity on GA100 (ridge %.0f FLOP/B):" % RIDGE)
    print(base[["variant"] + PT + ["attn_flop_per_byte", "attn_util", "core_s_attn", "core_s"]]
          .round(4).to_string(index=False))

    _figures(t, base)


def _figures(t, base):
    # Fig 1: step core latency vs per-core SRAM split, small multiples over (B, P)
    pts = t[PT].drop_duplicates().sort_values(PT).values.tolist()
    ncol = 4
    nrow = (len(pts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), facecolor=SURFACE, squeeze=False)
    for ax, (b, p, s) in zip(axes.flat, pts):
        g = t[(t.batch == b) & (t.prefix == p) & (t.suffix == s)]
        for v, c in COLORS.items():
            gv = g[g.variant == v].sort_values("local_KB")
            if gv.empty:
                continue
            rel = gv.core_s / gv.core_s.min()
            ax.plot(gv.local_KB, rel, color=c, lw=2, marker="o", ms=4, label=v)
        _style(ax, f"B={b}, prefix {p // 1024}K")
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 192, 256, 384, 512])
        ax.set_xticklabels(["64", "128", "192", "256", "384", "512"])
        ax.ticklabel_format(axis="y", useOffset=False)
    for ax in axes.flat[len(pts):]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.supxlabel("per-core SRAM KB (global buffer shrinks to keep area equal)", color=MUTED, fontsize=8)
    fig.supylabel("decode step time / best split", color=MUTED, fontsize=8)
    fig.suptitle("Equal-area SRAM split vs decode step time (Llama-3-8B, GA100 area)", color=INK,
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(common.OUT / "prefix_split_curves.png", dpi=160)

    # Fig 2: attention roofline position
    fig, ax = plt.subplots(figsize=(5.5, 3.6), facecolor=SURFACE)
    for v, c in COLORS.items():
        g = base[base.variant == v]
        ax.scatter(g.attn_flop_per_byte, g.flops_attn / g.core_s_attn / 1e12, color=c, s=36,
                   label=v, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for _, r in g.iterrows():
            if r.batch == 128 and v == "shared":
                ax.annotate(f"{int(r.prefix) // 1024}K", (r.attn_flop_per_byte, r.flops_attn / r.core_s_attn / 1e12),
                            fontsize=6, color=MUTED, xytext=(3, 3), textcoords="offset points")
    ax.plot([1, RIDGE, 1e3], [2.048, 369.6, 369.6], color=MUTED, lw=1)
    ax.axvline(64, color=MUTED, lw=0.8, ls=":")
    ax.text(64, 3, " d_h/2 = 64: unfused cap", color=MUTED, fontsize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Decode attention on GA100 roofline (all B, P)")
    ax.set_xlabel("FLOP per simulated DRAM byte", color=MUTED, fontsize=8)
    ax.set_ylabel("achieved TFLOP/s", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(common.OUT / "prefix_roofline.png", dpi=160)


if __name__ == "__main__":
    {"day1": day1, "sweep": sweep}[sys.argv[1]]()
