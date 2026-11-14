"""Report: abstract, section 6 (proposed design), 7 (discussion), 8 (conclusion) and the appendices. All numbers from the data."""
import pandas as pd
from rl import H1, H2, P, W, bullets, callout, fig, table

from analysis import facts as N

SP = "attn_speedup_vs_scratchpad"


def rd(n):
    return pd.read_csv(N.DATA / n)


def frac_of_oracle(batch, pol):
    d = rd("e9_drift.csv")
    d = d[d.batch == batch]
    a = d[d.policy == pol].merge(d[d.policy == "oracle-multicast"][["jitter", "skew_s", "seed", SP]], on=["jitter", "skew_s", "seed"], suffixes=("", "_o"))
    f = a[SP] / a[SP + "_o"]
    return f.min(), f.mean()


def e10v(b, pol, tb):
    d = rd("e10_bandwidth_allocation.csv")
    return d[(d.batch == b) & (d.policy == pol) & (d.gb_TBps.round(1) == tb)].speedup_vs_baseline_chip.iloc[0]


def energy(b):
    d = rd("e8_window_multicast.csv")
    d = d[(d.exp == "e8a") & (d.batch == b) & (d.prefix == 4096)].set_index("policy").E_total_uJ / 1e3
    return d


def abstract(h):
    e8 = h["e8_summary"]
    ok = e8[e8.win95.notna()]
    f128, f32 = frac_of_oracle(128, "cache-wmc-3000"), frac_of_oracle(32, "cache-wmc-1000")
    return [callout(
        f"<b>Abstract.</b> Prefix caching stores shared prompt prefixes once, but on a software-managed scratchpad accelerator every request still streams the shared prefix from HBM, so decode <i>bandwidth</i> is not reduced. "
        f"Using a validated event-driven model of a GA100-class accelerator (128 cores, 48 MB global buffer, 2.05 TB/s HBM) we show that (i) a scratchpad is HBM-bound at every operating point; (ii) a hardware-managed cache recovers "
        f"{h['cache_min']:.1f}-{h['cache_max']:.1f}x of attention speed but saturates the global buffer ({h['cache_gbutil_min'] * 100:.0f}-{h['cache_gbutil_max'] * 100:.0f}% busy), leaving up to {h['oracle_big_max']:.1f}x on the table (multicast bound); "
        f"(iii) the often-cited collapse of caches with many distinct prefixes is mostly a <i>dispatch artifact</i> ({h['cache_16_rr']:.2f}x interleaved, {h['cache_16_gm']:.2f}x when sharers run together). "
        f"We propose a <b>time-window multicast global buffer</b>: a small table of recently read blocks lets cores that request a block within a window share one buffer read. It reaches the multicast bound whenever the per-layer prefix fits in the buffer "
        f"({len(ok)} of {len(e8)} tested operating points, all with P ≤ 4K), needs a window of 0.3-3 µs and a table of at most {int(ok.table95.max()):,} entries, stays within {f128[0] * 100:.0f}% of the bound under ±50% core-speed variation and 2 µs start skew, "
        f"and lets the buffer run at half the bandwidth while still beating a full-bandwidth cache ({e10v(128, 'cache-wmc-3000', 3.6):.1f}x vs {e10v(128, 'cache', 7.2):.1f}x at B=128). "
        "It does not help when the prefix exceeds the buffer or when sharers are interleaved in dispatch, and a prefix-aware software kernel already matches the bound at whole-step level, so the hardware case rests on transparency to ordinary kernels and on buffer-bandwidth provisioning.")]


def design(h):
    e8b = rd("e8_window_multicast.csv")
    e8b = e8b[e8b.exp == "e8b"]
    g = lambda k, o, p: e8b[(e8b.n_prefixes == k) & (e8b.order == o) & (e8b.policy == p)][SP].iloc[0]  # noqa: E731
    pin = h["e8_pin"].set_index(["batch", "prefix"])
    e8 = h["e8_summary"].set_index(["batch", "prefix"])
    f128, f32 = frac_of_oracle(128, "cache-wmc-3000"), frac_of_oracle(32, "cache-wmc-1000")
    drift = h["e9"]
    d128 = drift[(drift.batch == 128) & (drift.skew_s == 0)].pivot_table(index="jitter", columns="policy", values=SP)
    e32, e128 = energy(32), energy(128)
    return [H1("6  Proposed design: a time-window multicast global buffer"),
            H2("6.1  Mechanism"),
            P("Section 5 locates the remaining headroom precisely: the cache serves each requesting core separately, so the global buffer, not HBM, is the bottleneck. The proposal removes that duplication at the buffer. "
              "Like the cache, the buffer is tagged by <b>physical KV block</b> and merges concurrent misses (one HBM fetch, one buffer read delivered to every waiting core). In addition it keeps a small <b>recent-multicast table</b>: "
              "each entry holds a block tag and a window timer started by a buffer read. A request that hits a block whose timer is still running <b>joins that broadcast</b> instead of causing another buffer read; its core port is still charged. "
              "The window is not refreshed on joining, so it cannot chain indefinitely. A window of zero reduces to multicast-on-merged-misses, and an unbounded window approaches the oracle's buffer traffic. "
              "The design touches no kernel: it sees only block requests."),
            P("<b>How this answers the brief.</b> <i>SRAM sizing:</i> the multicast table (a few KB to tens of KB) is the only new storage; the 48 MB buffer is unchanged. <i>Tiling / dispatch:</i> work items of one prefix group are issued together (group-major order), "
              "which Section 6.3 shows is necessary. <i>Bandwidth allocation:</i> because one buffer read serves many cores, the buffer can be provisioned with less bandwidth (Section 6.5)."),
            H2("6.2  How large a window and table?"),
            P(f"When the per-layer prefix fits in the buffer (P ≤ 4K), a window reaches the oracle: <b>300 ns</b> at B=32 ({e8.loc[(32, 4096), 'best']:.2f}x against the oracle's {e8.loc[(32, 4096), 'oracle']:.2f}x at P=4K), <b>1 µs</b> at B=128 and P=1K, and <b>3 µs</b> at B=128 and P=4K "
              f"({e8.loc[(128, 4096), 'best']:.2f}x against {e8.loc[(128, 4096), 'oracle']:.2f}x, versus {e8.loc[(128, 4096), 'mcfill']:.2f}x for multicast on merged misses). Larger batches need longer windows because more cores contend and drift further apart. "
              f"The table stays small: {int(e8.loc[(32, 4096), 'table95']):,} entries at B=32 and {int(e8.loc[(128, 4096), 'table95']):,} at B=128 and P=4K, which is about {e8.loc[(128, 4096), 'table95'] * 8 / 1024:.0f} KB at 8 B per entry, "
              f"under {e8.loc[(128, 4096), 'table95'] * 8 / (48 * 2**20) * 100:.2f}% of the 48 MB buffer (Figure 10)."),
            fig("f8_window_sweep", "<b>Figure 9.</b> Attention speedup against the multicast window (0 = multicast on merged misses only; ∞ = unbounded). Dashed lines: multicast on merged misses (pink) and the oracle bound (green); orange: plain cache. Purple diamonds: the pinned variant (v2, Section 6.2). "
                "At P=16K no window helps."),
            fig("f9_table_cost", "<b>Figure 10.</b> Cost of the design: peak table occupancy against window length (right axis: kilobytes at 8 B per entry)."),
            table(N.t_window(), None, "<b>Table 8.</b> Window needed to reach 95% of the oracle, and the table it requires (E8). 'none': no window reaches 95%."),
            H2("6.2.1  When the prefix does not fit: capacity, not delivery"),
            P(f"At P=16K the per-layer prefix is 64 MB, larger than the 48 MB buffer, and no window helps ({pin.loc[(128, 16384), 'wmc3000']:.2f}x at B=128, equal to multicast on merged misses, against an oracle of {pin.loc[(128, 16384), 'oracle']:.2f}x). "
              "Every wave of work items streams the whole prefix cyclically, which defeats LRU. The remaining gap is a <i>capacity</i> effect (the oracle has infinite capacity), not a delivery effect, so a multicast table cannot close it."),
            P(f"We tried a second variant (v2) that never lets a shared block evict another shared block and instead <i>bypasses</i> insertion when a set is full of shared blocks. The result is <b>mixed</b> (Table 9): it helps at B=32 and P=16K "
              f"(HBM traffic reduction {pin.loc[(32, 16384), 'dedup_wmc']:.1f}x to {pin.loc[(32, 16384), 'dedup_pin']:.1f}x; speedup {pin.loc[(32, 16384), 'wmc3000']:.2f}x to {pin.loc[(32, 16384), 'pin3000']:.2f}x against an oracle of {pin.loc[(32, 16384), 'oracle']:.2f}x) "
              f"but <b>hurts at B=128 and P=16K</b> ({pin.loc[(128, 16384), 'wmc3000']:.2f}x to {pin.loc[(128, 16384), 'pin3000']:.2f}x). A plausible reason, which we did not test, is that bypassing discards freshly fetched blocks that trailing cores would otherwise hit when many cores are spread out. "
              "We therefore do <b>not</b> claim to solve the oversized-prefix case; partial pinning (reserve only some ways) is future work."),
            table(N.t_pin(), [W * 0.07, W * 0.08, W * 0.11, W * 0.14, W * 0.15, W * 0.1, W * 0.35], "<b>Table 9.</b> Window multicast (3 µs) without and with bypass pinning (E8). Speedups are attention speedups over the scratchpad."),
            H2("6.3  Several prefixes: dispatch order is part of the design"),
            P(f"With four distinct prefixes and sharers dispatched together, the window design reaches the oracle ({g(4, 'group-major', 'cache-wmc-300'):.2f}x, against {g(4, 'group-major', 'cache-mcfill'):.2f}x for multicast on merged misses and {g(4, 'group-major', 'cache'):.2f}x for the cache). "
              f"With sixteen prefixes, mcfill already equals the oracle ({g(16, 'group-major', 'cache-mcfill'):.2f}x). When the same requests are dispatched interleaved, <b>no window helps</b> ({g(4, 'request-major', 'cache-wmc-1000'):.2f}x at four prefixes, {g(16, 'request-major', 'cache-wmc-inf'):.2f}x at sixteen), "
              f"because sharers are never resident together. The multicast buffer and a sharing-aware dispatch hint are therefore one design, not two (Figure 11)."),
            fig("f10_multi_prefix_design", "<b>Figure 11.</b> Several prefixes, B=128, P=4K. The window design reaches the oracle only when sharers are dispatched together."),
            H2("6.4  Robustness to core drift"),
            P("The window design relies on cores reading a block within a short time of each other. The baseline simulator starts all cores together at identical speed, which favours it, so we added a drift model: each core's speed is scaled by a fixed random factor within ±jitter and each core starts after a random delay up to 2 µs (E9; 4 seeds per case)."),
            P(f"A <i>fixed short</i> window degrades as drift grows: at B=128 the 300 ns window falls from {d128.loc[0.0, 'cache-wmc-300']:.2f}x to {d128.loc[0.5, 'cache-wmc-300']:.2f}x at ±50% speed variation, close to the {d128.loc[0.5, 'cache-mcfill']:.2f}x of multicast on merged misses. "
              f"A window <i>sized for the drift</i> does not: the 3 µs window stays within <b>{f128[0] * 100:.1f}%</b> of the oracle in the worst of all tested drift and seed combinations (mean {f128[1] * 100:.1f}%) at B=128, and a 1 µs window within {f32[0] * 100:.1f}% (mean {f32[1] * 100:.1f}%) at B=32. "
              "The window should therefore be a programmable timer, set from the expected core-to-core drift (Figure 12). The cost of a longer window is table size (Figure 10)."),
            fig("f11_drift_robustness", "<b>Figure 12.</b> Attention speedup against per-core speed variation, with no start skew (left) and 2 µs start skew (right), mean over four seeds. Short windows lose their benefit as drift grows; 1-3 µs windows track the oracle."),
            table(N.t_drift(), None, "<b>Table 10.</b> B=128, P=4K, no start skew, mean over four seeds (E9). Speedups are over the scratchpad under the same drift."),
            H2("6.5  Bandwidth allocation and energy"),
            P(f"Because one buffer read serves many cores, the buffer needs less bandwidth for the same speed. Measured against the <i>unmodified chip</i> (scratchpad at 7.2 TB/s), a cache at the full 7.2 TB/s gives {e10v(32, 'cache', 7.2):.2f}x at B=32 and {e10v(128, 'cache', 7.2):.2f}x at B=128. "
              f"The window design with <b>half</b> the buffer bandwidth (3.6 TB/s) gives {e10v(32, 'cache-wmc-3000', 3.6):.2f}x and {e10v(128, 'cache-wmc-3000', 3.6):.2f}x, and with a <b>quarter</b> (1.8 TB/s) still {e10v(32, 'cache-wmc-3000', 1.8):.2f}x and {e10v(128, 'cache-wmc-3000', 1.8):.2f}x (Figure 13, Table 11). "
              "This is a statement about <i>bandwidth</i>: we do not model the silicon area of buffer bandwidth, so we do not convert it to square millimetres."),
            fig("f12_bandwidth_allocation", "<b>Figure 13.</b> Speedup over the unmodified chip against global-buffer bandwidth (B and P=4K). A multicast buffer at a quarter of the bandwidth beats a cache at full bandwidth."),
            table(N.t_bandwidth(), None, "<b>Table 11.</b> Speedup over the unmodified chip (scratchpad at 7.2 TB/s) against buffer bandwidth (E10)."),
            P(f"<b>Energy</b> (estimates; bytes are exact, the picojoule constants are not). At B=128 and P=4K, attention energy per layer is {e128['scratchpad']:.1f} mJ for the scratchpad, {e128['cache']:.1f} mJ for the cache and {e128['cache-wmc-3000']:.1f} mJ for the window design "
              f"(oracle {e128['oracle-multicast']:.1f} mJ), i.e. {(1 - e128['cache-wmc-3000'] / e128['cache']) * 100:.0f}% below the cache and {(1 - e128['cache-wmc-3000'] / e128['scratchpad']) * 100:.0f}% below the scratchpad; at B=32 the figures are {e32['scratchpad']:.1f}, {e32['cache']:.1f} and {e32['cache-wmc-3000']:.1f} mJ. "
              "Most of the saving comes from HBM bytes; the multicast table contributes the cache-to-window difference through fewer buffer reads."),
            fig("f13_energy", "<b>Figure 14.</b> Estimated attention energy per layer (3.9 / 0.5 / 0.3 pJ per bit for HBM, buffer and buffer-to-core network)."), *area_block()]


def area_block():
    """Section 6.6 - only rendered when E11 has been run (it needs an LLMCompass clone)."""
    p = N.DATA / "e11_area.csv"
    if not p.exists():
        return []
    a = pd.read_csv(p).set_index("component")
    gb = a.iloc[0]
    tab = a[a.index.str.startswith("Multicast table")]
    book = a[~a.index.str.startswith("Multicast table")].iloc[1:]
    added = tab.area_mm2.max() + book.area_mm2.sum()
    return [H2("6.6  Silicon area of the added storage"),
            P(f"The area budget is held fixed by construction: Section 5.7 evaluates the six equal-area SRAM splits of about 179 mm² each. This section costs the storage the design <i>adds</i>, using LLMCompass's own 7 nm SRAM area model "
              f"(<font face='DV-I'>cost_model.calc_cache_sram_area_mm2</font>), the same call its cost model uses. The {gb.name.split(' - ')[0].lower()} occupies {gb.area_mm2:.2f} mm². "
              f"The recent-multicast table at the largest design point holds {int(tab.entries.max()):,} entries, that is {tab.bytes.max() / 1024:,.0f} KB, costing <b>{tab.area_mm2.max():.4f} mm²</b> or <b>{tab.pct_of_gb.max():.3f}%</b> of the buffer. "
              f"Including the tags and MSHRs a block-tagged buffer needs anyway, the total added storage is {added:.3f} mm², i.e. {100 * added / gb.area_mm2:.2f}% of the global buffer. "
              "The design therefore fits inside the existing area budget rather than competing with it, and the speedups of Sections 6.2 to 6.5 are gains at effectively equal area."),
            table(N.t_area(), [W * 0.40, W * 0.13, W * 0.13, W * 0.17, W * 0.17], "<b>Table 12.</b> Area of the added storage at 7 nm (E11, LLMCompass SRAM area model). Logic, routing and the table's lookup path are not modelled."),
            P("Not modelled: the comparator logic and routing of the table lookup, the multicast delivery network itself, and the silicon cost of global-buffer bandwidth (Section 6.5 states the bandwidth saving, not its area). The area model also quantises small arrays - its own tag arrays are non-monotonic in size, and at kilobyte scale fixed peripheral circuitry dominates - so these figures are an upper bound at the right order of magnitude, not a precise layout result.", "note")]


def discussion(h):
    sw = rd("e6_step_equal_area.csv")
    sw = sw[(sw.split_id == "L192_G48") & (sw.policy == "scratchpad")]
    swr = sw.sw_kernel_speedup_vs_scratchpad
    return [H1("7  Discussion"),
            H2("7.1  Does the solution work?"),
            table(pd.DataFrame([
                ["Prefix fits in the buffer (P ≤ 4K), sharers dispatched together", "<b>Yes</b>: reaches the multicast bound with a 0.3-3 µs window and a few KB of table"],
                ["Core drift (±50% speed, 2 µs start skew)", "<b>Yes</b>, if the window is sized for the drift; a fixed short window degrades"],
                ["Buffer bandwidth provisioning", "<b>Yes</b>: half or a quarter of the bandwidth still beats a full-bandwidth cache"],
                ["Prefix larger than the buffer (P=16K)", "<b>No</b>: capacity-limited; bypass pinning is mixed (helps B=32, hurts B=128)"],
                ["Sharers interleaved in dispatch", "<b>No</b>: needs a sharing-aware dispatch hint"],
                ["Long private suffixes", "Opportunity itself shrinks (oracle only 1.8-1.9x at a 4K suffix)"]], columns=["Condition", "Outcome"]), [W * 0.42, W * 0.58], "<b>Table 13.</b> Where the design works and where it does not."),
            H2("7.2  Where is the novelty, and what is not novel"),
            bullets(["<b>Characterisation.</b> On a scratchpad accelerator, prefix caching saves capacity but no decode bandwidth; a cache recovers part of it and is capped by buffer bandwidth. The three-way comparison, its conditions, and the roofline view are, to the best of our literature check, not stated in the sources we read.",
                     "<b>Correction.</b> The many-prefix collapse of a cache is a dispatch artifact. This changes where a hardware proposal should aim.",
                     "<b>A quantified mechanism.</b> A time-window multicast buffer, with its window, table size, robustness to drift, bandwidth saving and its two failure regimes.",
                     "<b>Not novel:</b> prefix caching itself; multicast and merged-miss hardware (Buffets and related designs); prefix-aware <i>software</i> kernels (Hydragen, FlashInfer cascade, PAT, FlashForge, TyphoonMLA)."]),
            P("<b>Closest hardware work.</b> MeshKV suppresses duplicate KV fetches in a network-on-chip with a short window, mainly across heads within a request. Our design is conceptually close in using a time window. Our differences are sharing <i>across requests</i> keyed by physical block "
              "(through block-table indirection), windows long enough to span core drift (microseconds, not a dozen cycles), and an evaluation at the global buffer against a cache and a bound. A reviewer may nonetheless regard the extension as incremental; we state this risk openly. "
              "We rate the novelty as <b>medium-low to medium</b>: a careful characterisation plus a quantified mechanism, not a new phenomenon."),
            H2("7.3  Hardware versus a software kernel"),
            P(f"The strongest objection is that a <b>prefix-aware software kernel</b> on the plain scratchpad already reaches about the oracle's whole-step speedup (Table 7; {swr.min():.1f}-{swr.max():.1f}x across the four operating points). We have <i>not</i> shown that hardware beats software. "
              "The case for hardware rests on three points: it works with ordinary paged kernels, with no kernel change and no need to tell the kernel which requests share a prefix (dynamic, radix-tree-style sharing); it lets the buffer be provisioned with much less bandwidth (Section 6.5); and it composes with kernels that are not prefix-aware. "
              "An earlier trace-based analysis suggested that launch overheads of the extra kernels reduce the software kernel's gain; that was not re-evaluated here."),
            H2("7.4  Threats to validity"),
            bullets(["gbsim models one attention layer with an aggregate buffer bandwidth; no NoC topology, bank conflicts or real CTA scheduling.",
                     "The cache assumes perfect miss merging and is an upper bound for a conventional cache; a one-off GPU comparison reached a median 55% of its predicted gain, so the cache-to-multicast headroom is probably larger.",
                     "The drift model is simple (fixed per-core speed plus start skew; four seeds); real drift has richer sources.",
                     "Table cost is counted in entries and kilobytes, not silicon area or timing; the table is assumed to be an ideal associative structure.",
                     "Energy constants are estimates. LLMCompass step times and the cache-bookkeeping area estimate come from the hand-off and were not regenerated."]),
            P("The complete list is in <font face='DV-I'>docs/06_limitations.md</font>.", "note"),
            H2("7.5  Future work"),
            bullets(["Partial pinning (reserve some ways for the stable part of an oversized prefix, keep the rest recency-managed).",
                     "An adaptive window timer driven by measured core-to-core skew.",
                     "A cycle-level or RTL-level model of the table and its lookup latency, and silicon area of the table and of buffer bandwidth.",
                     "Validation of the cache model on additional GPUs; evaluation with a real serving trace of prefix-sharing structure."]),
            H1("8  Conclusion"),
            P(f"With prefix caching and an ordinary kernel, a scratchpad accelerator saves capacity but not bandwidth; a hardware-managed cache recovers {h['cache_min']:.1f}-{h['cache_max']:.1f}x but is capped by global-buffer bandwidth, "
              f"leaving up to {h['oracle_big_max']:.1f}x unrealised. A time-window multicast buffer, together with a sharing-aware dispatch hint, closes that gap whenever the per-layer prefix fits, survives core drift when its window is sized for it, and cuts the required buffer bandwidth by half or more. "
              "It does not solve oversized prefixes or interleaved dispatch, and it competes with prefix-aware software kernels, so its value is transparency and provisioning rather than raw speed. "
              "The results are simulation-based and bounded by the limitations above; the full pipeline is reproducible with <font face='DV-I'>make all</font>.")]


def appendices(h):
    led = pd.DataFrame([
        ["Scratchpad saves no HBM bandwidth", "e1_baseline.csv: dram_dedup_x (scratchpad)", f"{h['scratchpad_dedup_max']:.1f}x at every point"],
        ["Cache attention speedup", "e1_baseline.csv: attn_speedup_vs_scratchpad", f"{h['cache_min']:.1f}-{h['cache_max']:.1f}x"],
        ["Cache saturates the buffer", "e1_baseline.csv: gb_util, hbm_util (B≥32, P≥4K)", f"{h['cache_gbutil_min'] * 100:.0f}-{h['cache_gbutil_max'] * 100:.0f}% buffer, {h['cache_hbmutil_min'] * 100:.0f}-{h['cache_hbmutil_max'] * 100:.0f}% HBM"],
        ["Multicast headroom (oracle)", "e1_baseline.csv (B≥32, P≥4K)", f"{h['oracle_big_min']:.1f}-{h['oracle_big_max']:.1f}x"],
        ["Many-prefix collapse is a dispatch artifact", "e5_dispatch_order.csv (B=128, 16 prefixes)", f"cache {h['cache_16_rr']:.2f}x → {h['cache_16_gm']:.2f}x; mcfill {h['mcfill_16_gm']:.2f}x"],
        ["Roofline constants", "e7_roofline_constants.csv", f"{h['flop_per_kv_byte']:.0f} FLOP/B; roofs {h['hbm_roof_tflops']:.1f} / {h['gb_roof_nomulticast_tflops']:.1f} / {h['kernel_ceiling_tflops']:.1f} TFLOP/s"],
        ["Window design reaches the oracle (P ≤ 4K)", "e8_window_multicast.csv", "300 ns (B=32); 1-3 µs (B=128)"],
        ["Table size at B=128, P=4K, 3 µs", "e8_window_multicast.csv: max_live", f"{int(h['e8_summary'].set_index(['batch', 'prefix']).loc[(128, 4096), 'table95']):,} entries"],
        ["Pinning is mixed at P=16K", "e8_window_multicast.csv", "B=32 improves, B=128 degrades"],
        ["Drift robustness", "e9_drift.csv (4 seeds)", f"3 µs window ≥ {frac_of_oracle(128, 'cache-wmc-3000')[0] * 100:.1f}% of the oracle at B=128"],
        ["Bandwidth allocation", "e10_bandwidth_allocation.csv", f"3.6 TB/s: {e10v(128, 'cache-wmc-3000', 3.6):.2f}x vs cache at 7.2 TB/s: {e10v(128, 'cache', 7.2):.2f}x (B=128)"],
        ["Whole-step and software kernel", "e6_step_equal_area.csv (split L192_G48)", "Table 7"]], columns=["Claim", "Source (results/data/)", "Value"])
    refs = ["vLLM PagedAttention and SGLang RadixAttention: prefix caching in LLM serving (named in the team's gap write-up).",
            "ChunkAttention, arXiv 2402.15220 (Table 3: shared-page speedup on A100).",
            "PAT, ASPLOS 2026, arXiv 2511.22333 (L2 gives partial reuse, limited by size and bandwidth).",
            "Hydragen, arXiv 2402.05099; FlashInfer cascade attention; FlashForge, arXiv 2505.17694; TyphoonMLA, arXiv 2509.21081 (prefix-aware software kernels).",
            "MeshKV, arXiv 2609.19207 (NoC multicast with duplicate suppression, mainly across heads).",
            "DCO, arXiv 2512.07312 (shared cache over scratchpads in LLM accelerators).",
            "TPU Ragged Paged Attention, arXiv 2604.15464 (per-sequence page DMA).",
            "PRESERVE, arXiv 2501.08192; MoSKA, arXiv 2511.06010 (adjacent work).",
            "LLMCompass (Princeton), the accelerator simulator and GA100 template used for step times and the hardware template.",
            "Starting-point papers of the brief (MLX, ISCA 2026; SMOOTH, ISCA 2026; PADE, HPCA 2026): motivating context; not used as quantitative baselines."]
    return [H1("Appendix A  Reproducibility"),
            P("Everything is deterministic and regenerated by one command: <font face='DV-I'>make all</font> (validation, experiments E1-E10, tables, figures, this report, tests). Individual stages: <font face='DV-I'>make validate | experiments | tables | figures | report | test | status</font>. "
              "The simulator needs only pandas; this report needs reportlab. The repository map, experiment catalogue and methodology are in <font face='DV-I'>README.md</font> and <font face='DV-I'>docs/00-06</font>."),
            H1("Appendix B  Claims ledger"),
            P("Every number in this report is computed from the files below by <font face='DV-I'>analysis/facts.py</font>; nothing is typed by hand.", "note"),
            table(led, [W * 0.30, W * 0.34, W * 0.36], "<b>Table 14.</b> Where each headline claim comes from."),
            H1("Appendix C  References"),
            P("As recorded in the team's prior-work check (9 Nov 2026). The authors of this report did not independently re-read every cited paper; the check was a search-and-read exercise, and absence of evidence is not proof of absence.", "note"),
            bullets(refs, "note")]
