"""Shared plotting style: one colour per design everywhere, clean axes, vector + 300 dpi output."""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS, TABLES = ROOT / "results" / "data", ROOT / "results" / "figures", ROOT / "results" / "tables"

# Colour-blind-safe (Okabe-Ito based). Window-multicast (our design) uses the warm vermillion family.
COLOR = {"scratchpad": "#0072B2", "cache": "#E69F00", "cache-sharedpin": "#999999", "cache-mcfill": "#CC79A7",
         "oracle-multicast": "#009E73", "software-kernel": "#332288"}
WMC = {"0": "#F5B895", "30": "#F0A078", "100": "#E8834F", "300": "#D55E00", "1000": "#B04A00", "3000": "#8A3A00",
       "10000": "#6B2D00", "inf": "#4A1F00"}
LABEL = {"scratchpad": "Scratchpad (software-managed)", "cache": "Cache (hardware-managed)",
         "cache-sharedpin": "Cache, shared-pin", "cache-mcfill": "Cache + multicast on merged misses",
         "oracle-multicast": "Oracle multicast (upper bound)", "software-kernel": "Prefix-aware software kernel"}
MARK = {"scratchpad": "o", "cache": "s", "cache-mcfill": "D", "oracle-multicast": "^", "cache-sharedpin": "v"}


PIN = "#6A3D9A"


def color(p):
    if p.startswith("cache-wmcpin-"):
        return PIN
    return WMC[p.split("-")[-1]] if p.startswith("cache-wmc-") else COLOR[p]


def label(p):
    if p.startswith("cache-wmcpin-"):
        w = p.split("-")[-1]
        return "Window multicast + pinning (ours), " + ("unbounded window" if w == "inf" else f"{int(float(w)):,} ns window")
    if p.startswith("cache-wmc-"):
        w = p.split("-")[-1]
        return "Window multicast (ours), " + ("unbounded window" if w == "inf" else f"{int(float(w)):,} ns window")
    return LABEL[p]


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 130,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})


def savefig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png")
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)
    print(f"  figure {name}")
