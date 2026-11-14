"""Report sections 1-5 (introduction, background, research trail, methodology, baseline evaluation) and appendices."""
import pandas as pd
from rl import H1, H2, P, W, bullets, fig, table

from analysis import facts as N

SP = "attn_speedup_vs_scratchpad"


def rd(n):
    return pd.read_csv(N.DATA / n)


def intro():
    return [H1("1  Introduction"),
            P("LLM serving systems such as vLLM and SGLang <b>cache shared prompt prefixes</b>: system prompts, retrieval context, agent instructions and parallel samples give many concurrent requests the same opening tokens, and "
              "prefix caching stores that part of the KV cache once. This saves memory <i>capacity</i> and prefill compute. Decoding, however, is memory-bandwidth-bound: every step each request reads its whole KV cache. "
              "With an ordinary attention kernel each request reads the shared prefix pages itself, so the shared bytes are fetched B times per step."),
            P("This project asks how the <b>on-chip memory organisation</b> of an inference accelerator determines whether that duplicated traffic reaches HBM, and what a better organisation looks like. "
              "It follows the P07 brief step by step:"),
            table(pd.DataFrame([["Roofline characterisation of the operator", "Section 5.1 (E7, Figure 2)"],
                                ["Evaluate existing on-chip configurations, find the dominant bottleneck", "Sections 5.2-5.7 (E1-E7, Figures 3-8)"],
                                ["Propose an on-chip design: sizing, tiling/dispatch, bandwidth allocation", "Section 6 (E8, E10; Figures 9-14)"],
                                ["Simulate throughput and energy at a fixed area budget", "Sections 5.7, 6.5 (E6, E10; Figures 8, 13, 14)"],
                                ["Robustness and limits of the result", "Sections 6.4, 7 (E9, Figure 12)"]], columns=["Brief requirement", "Where"]), [W * 0.62, W * 0.38]),
            P("<b>Contributions.</b>"),
            bullets(["A timing-level characterisation, on one accelerator template, of three on-chip organisations for prefix-cached decode: a software-managed scratchpad, a hardware-managed cache and a multicast bound, with the conditions under which each works.",
                     "A <b>correction</b> of a frequently stated failure: a cache's collapse with many distinct prefixes is mostly a <i>dispatch-order artifact</i>, not a hardware limit.",
                     "A roofline and bottleneck map showing which resource limits each design at each operating point.",
                     "A <b>time-window multicast global buffer</b>, evaluated against the multicast bound, a prefix-aware software kernel, and core drift, with its table size and energy.",
                     "A reproducible, validated simulation framework (gbsim) with one script per experiment and regression tests."])]


def background():
    return [H1("2  Background: three ways to organise on-chip memory"),
            P("<b>Scratchpad</b> accelerators (the LLMCompass GA100 template, TPU VMEM/CMEM, most NPUs) move data with per-core DMA. Nothing in the hardware notices that sixteen cores want the same physical block at the same moment, "
              "so prefix caching cuts capacity but not decode traffic. <b>Cache</b>-managed buffers (GPU L2) serve repeats: concurrent work items walk the same prefix blocks almost in lockstep, so a block is re-requested within a short window and "
              "later requests hit. HBM traffic falls, but every core is still served separately from the buffer, so <i>buffer</i> bandwidth becomes the limit. A <b>multicast</b> buffer would coalesce concurrent requests for one physical block, "
              "fetch it once and deliver it to every requesting core at once. Figure 1 summarises the three."),
            fig("f0_architectures", "<b>Figure 1.</b> Where a shared prefix block is read, for four of the 128 cores. Each organisation removes one level of duplicate reads; only the multicast buffer reads a shared block once from HBM and once from the buffer.")]


def trail():
    sc = pd.DataFrame([
        ["1", "Equal-area SRAM split for shared-prefix decode", "Split is flat (a 25% spread was a softmax cost-model artifact)", "No lever"],
        ["2", "Sliding-window KV pinned on-chip", "at most 2% where it fits", "Refuted"],
        ["3", "Recurrent-state traffic in SSM hybrids", "Real: 19-45% of step bytes", "Already published"],
        ["4", "Diffusion-LLM decode becomes compute-bound", "Real", "Already published"],
        ["5", "MLA per-core row budget / split flip", "Retracted after an independent check (latent counted twice)", "Dropped"],
        ["6-8", "Head-dim-256 prefill tiles; SRAM split vs energy; small local buffer vs MLA", "each at most 4.6%", "Refuted"],
        ["9", "SRAM-resident chips: per-request footprint", "Real", "Mostly published"],
        ["10", "Unfused attention erodes MLA/MTP advantage", "Real", "Known mechanism; backup"],
        ["<b>11</b>", "<b>Scratchpad vs cache for prefix-cached decode</b>", "<b>Real: 1.1-1.5x per step at measured bandwidth</b>", "<b>Selected</b>"]],
        columns=["#", "Candidate mismatch", "Finding", "Outcome"])
    return [H1("3  How we found the gap"),
            P("The topic was not chosen first and justified afterwards. The team listed places where a common accelerator design choice might not match a common LLM workload, posed each as a <b>falsifiable hypothesis</b>, "
              "tested it quickly (analytically or in the LLMCompass simulator) and checked the literature. Table 2 summarises the first round; a second round added logits round-trips (refuted), layout questions (covered in practice) and "
              "write-heavy decode in hybrid SSM models (C4)."),
            table(sc, [W * 0.07, W * 0.40, W * 0.38, W * 0.15], "<b>Table 2.</b> Screening of candidate mismatches (team ledger, <font face='DV-I'>llmc/out/loop/</font>)."),
            H2("3.1  Why M1 and not C4"),
            P("C4 (hybrid SSM models write 10-34% of DRAM bytes where read-optimised memory designs assume 5%) had the stronger novelty argument. It is, however, a DRAM-controller question evaluated with a DRAM simulator, while the brief asks for "
              "<b>on-chip SRAM</b> design, a roofline, an area budget and energy. C4 has no accelerator model, no area model and no on-chip hierarchy question, so we returned to M1 on fit-to-brief grounds and kept C4 only as a documented alternative."),
            H2("3.2  How the M1 claim was built and checked"),
            bullets(["<b>Trace model</b> (128 cores, 8 KB blocks, fully-associative buffer): ideal HBM-traffic reduction 3.3-43x; reuse is temporal, so the conditions are drift, replacement policy and dispatch order, not capacity alone.",
                     "<b>Independent review</b>: 13 issues were applied; the qualitative finding survived, quantitative cache-path speedups were cut by 10-25%, and one circular validation claim was removed.",
                     "<b>Prior-work check</b> (9 Nov 2026): the scratchpad-vs-cache contrast for prefix-cached decode was not found stated; the nearest hardware work (MeshKV) deduplicates across heads within a request, not across requests; prefix-aware <i>software</i> kernels (Hydragen, FlashInfer cascade, PAT) already remove redundant reads on any hardware.",
                     "<b>Event-driven simulator gbsim</b> replaced the trace model so that bandwidth, latency and queueing are modelled, and was validated before any gap number was used (Section 4.5).",
                     "<b>This work</b> re-organised the experiments (E1-E9), found and corrected the many-prefix artifact, added the roofline analysis, proposed and evaluated the window-multicast design, and tested it against core drift."]),
            P("Every step above is traceable to a file in the repository (Appendix A); the full trail is in <font face='DV-I'>docs/01_gap_discovery.md</font>.", "note")]


def method(h):
    val = N.t_validation() if (N.ROOT / "results" / "validation").exists() else pd.DataFrame([["(run make validate)", ""]], columns=["Check", "Result"])
    return [H1("4  Methodology"),
            H2("4.1  System and workload"),
            P("We model the <b>LLMCompass GA100 template</b> and study one decode-attention layer, which dominates KV traffic at serving batch sizes. Whole-step numbers (E6) add LLMCompass's time for everything outside attention. "
              "The workload is B concurrent requests sharing a prefix of P tokens, plus 256 private tokens each, decoded by an <i>ordinary</i> (not prefix-aware) GQA kernel: one work item per (request, KV head)."),
            table(N.t_specs(), [W * 0.34, W * 0.66], "<b>Table 3.</b> Hardware and workload parameters (generated from <font face='DV-I'>gbsim/config.py</font>)."),
            H2("4.2  Simulator"),
            P("<b>gbsim</b> is an event-driven timing model. Each core runs one work item at a time with up to eight block requests in flight. HBM, the global buffer and each core port are first-in-first-out bandwidth servers followed by a "
              "fixed latency, so contention, queueing and core drift emerge from the model. A <i>policy</i> decides for every block request whether it hits, misses (HBM fetch) or merges with an in-flight fetch, and how many buffer bytes its delivery costs. "
              "Dispatch order decides which work item a free core takes next: request-major (vLLM-like), head-major, or group-major (sharers together)."),
            H2("4.3  Designs compared"),
            bullets(["<b>Scratchpad</b>: per-core DMA, no reuse (baseline for every speedup).",
                     "<b>Cache</b>: set-associative LRU over physical blocks with miss merging. It assumes <i>perfect</i> merging, so it is an upper bound for a conventional cache.",
                     "<b>Cache + mcfill</b>: one buffer read serves all cores merged on a miss.",
                     "<b>Window multicast (ours)</b>: cache + mcfill + a small table of recently read blocks; a request for a block whose buffer read began less than a <i>window</i> ago joins that broadcast.",
                     "<b>Oracle multicast</b>: an upper bound, not a design. Each distinct block is read from HBM once and from the buffer once, whatever the timing."]),
            H2("4.4  Metrics"),
            P("Attention speedup is relative to the scratchpad with the same workload, order and drift. HBM traffic reduction is accessed bytes divided by HBM bytes. Utilisation is busy time over total time. "
              "Energy applies 3.9 / 0.5 / 0.3 pJ per bit to HBM, buffer and buffer-to-core network bytes; <b>these constants are estimates, the byte counts are the hard numbers</b>. "
              "Design cost is the peak number of table entries, converted to KB at 8 B per entry."),
            H2("4.5  Validation"),
            P("The simulator was validated before any gap number was used. A one-off GPU comparison (Tesla M60, archived, not reproduced here) showed that the real cache reached a median 55% of gbsim's predicted gain, so gbsim's cache numbers are an upper bound and the true cache-to-multicast headroom is likely larger."),
            table(val, [W * 0.38, W * 0.62], "<b>Table 4.</b> Validation checks (<font face='DV-I'>make validate</font>, details in docs/02_methodology.md)."),
            P("Regression tests (<font face='DV-I'>make test</font>) additionally confirm that the re-organised experiments reproduce the team's original results to 1e-9, that the scratchpad saves no HBM bytes, "
              "that drift is deterministic for a given seed, that a zero window equals mcfill, that buffer reads fall monotonically with the window, and that the oracle is an upper bound.")]


def baseline(h):
    e3, e2, e4q = rd("e3_gb_bandwidth.csv"), rd("e2_capacity.csv"), rd("e4_model_qwen.csv")
    lo = e3[(e3.policy == "cache") & e3.gb_TBps.isin([3.3, 4.4])][SP]
    ded = e2[e2.prefix == 4096].pivot_table(index="gb_MB", columns="policy", values="dram_dedup_x", aggfunc="mean")
    sp67 = e2[(e2.gb_MB == 67) & (e2.prefix == 16384)].pivot_table(index="batch", columns="policy", values="dram_dedup_x")
    sp48 = e2[(e2.gb_MB == 48) & (e2.prefix == 16384)].pivot_table(index="batch", columns="policy", values="dram_dedup_x")
    q = e4q[e4q.policy == "cache"][SP], e4q[e4q.policy == "oracle-multicast"][SP]
    out = [H1("5  Evaluating existing on-chip configurations"),
           H2("5.1  Roofline"),
           P(f"Decode attention performs only <b>{h['flop_per_kv_byte']:.0f} FLOP per KV byte</b> (4 x g x 16 x d_head FLOPs per 8 KB block, g = 4 query heads per KV head), far to the left of any ridge point. "
             f"A scratchpad, which fetches every byte from HBM, therefore sits exactly on the HBM roof at {h['hbm_roof_tflops']:.1f} TFLOP/s. A design that serves every core from the buffer but does not multicast is capped at "
             f"{h['gb_roof_nomulticast_tflops']:.1f} TFLOP/s by the buffer bandwidth, a gain of {h['gb_roof_nomulticast_tflops'] / h['hbm_roof_tflops']:.1f}x. "
             f"The kernel's own compute ceiling is {h['kernel_ceiling_tflops']:.1f} TFLOP/s, not the {h['peak_tflops']:.0f} TFLOP/s peak, because a GQA group of four query rows fills only a quarter of the 16-high array and the softmax adds vector work. "
             f"Against that ceiling the HBM ridge is at {h['ridge_hbm_flop_per_byte']:.0f} FLOP/B: a design becomes compute-bound only when it cuts HBM traffic by more than about {h['ridge_hbm_flop_per_byte'] / h['flop_per_kv_byte']:.0f}x."),
           fig("f1_roofline", "<b>Figure 2.</b> Roofline of decode attention (GA100 template, Llama-3-8B), plotted against HBM bytes (a) and global-buffer bytes (b). Each marker is one batch/prefix operating point. Scratchpad points lie on the HBM roof; cache points lie on the buffer roof; only multicast moves points toward the compute ceiling."),
           H2("5.2  Bottleneck map"),
           P(f"Figure 3 labels the limiting resource at each of the nine operating points. The scratchpad is <b>{h['bottleneck_scratchpad']}</b>-limited; the cache is <b>{h['bottleneck_cache']}</b>-limited. "
             "The multicast designs mostly leave no bandwidth resource above 80% busy: their limit moves to latency, queueing and available parallelism. Only the oracle at the largest operating points reaches the HBM or compute limit."),
           fig("f2_bottleneck_map", "<b>Figure 3.</b> Limiting resource per design and operating point; cells show attention speedup and the utilisation of the busiest resource. B=8 has only 64 work items for 128 cores, so its labels reflect too little parallelism rather than a design limit."),
           H2("5.3  Baseline: how much do the existing organisations recover?"),
           P(f"With request-major (vLLM-like) dispatch, a hardware-managed cache gives <b>{h['cache_min']:.1f}-{h['cache_max']:.1f}x</b> attention speedup over the scratchpad and removes {h['cache_dedup_min']:.1f}-{h['cache_dedup_max']:.1f}x of the HBM traffic at B ≥ 32 and P ≥ 4K. "
             f"At those points the buffer runs at <b>{h['cache_gbutil_min'] * 100:.0f}-{h['cache_gbutil_max'] * 100:.0f}%</b> utilisation while HBM is only {h['cache_hbmutil_min'] * 100:.0f}-{h['cache_hbmutil_max'] * 100:.0f}% busy, so the cache gain stays near the 7.2 : 2.05 bandwidth ratio. "
             f"The oracle multicast bound reaches <b>{h['oracle_big_min']:.1f}-{h['oracle_big_max']:.1f}x</b> there, and multicasting only merged misses (mcfill) gets {h['mcfill_big_min']:.1f}-{h['mcfill_big_max']:.1f}x. "
             "At B=8 there is no headroom: the cache already reaches the bound. The scratchpad removes no HBM traffic at any point (HBM bytes equal accessed bytes)."),
           fig("f3_baseline_speedup", "<b>Figure 4.</b> Attention speedup over the scratchpad (request-major dispatch, one shared prefix, 256 private tokens)."),
           table(N.t_baseline(), None, "<b>Table 5.</b> Attention speedup over the scratchpad, GA100 template, Llama-3-8B (E1)."),
           H2("5.4  The cache is capped by buffer bandwidth, and needs concurrency, not capacity"),
           P(f"At buffer bandwidths of 3.3-4.4 TB/s (the measured A100 shared-page throughput and a calibrated value) the cache gain falls to <b>{lo.min():.1f}-{lo.max():.1f}x</b> (Figure 5). "
             f"Capacity matters little: an 11 MB buffer already removes {ded.loc[11, 'cache']:.1f}x of the HBM traffic at P=4K against {ded.loc[48, 'cache']:.1f}x at 48 MB, because almost all deduplication comes from "
             f"<i>merging misses that are in flight at the same time</i>. Refcount-aware replacement (shared-pin) equals LRU up to 48 MB ({sp48.loc[32, 'cache']:.1f}x and {sp48.loc[128, 'cache']:.1f}x at P=16K) and helps only at 67 MB "
             f"({sp67.loc[32, 'cache-sharedpin']:.1f}x and {sp67.loc[128, 'cache-sharedpin']:.1f}x) (Figure 6)."),
           fig("f4_buffer_bandwidth", "<b>Figure 5.</b> Attention speedup against global-buffer bandwidth. The cache path scales with the buffer; the oracle does not need it."),
           fig("f5_buffer_capacity", "<b>Figure 6.</b> HBM traffic reduction against buffer capacity (LRU cache vs shared-pin replacement)."),
           H2("5.5  Realistic sharing"),
           P(f"Longer private suffixes shrink everything: at a 4K suffix the cache gives {h['cache_longsuffix']:.1f}x and the oracle only {h['oracle_longsuffix_min']:.1f}-{h['oracle_longsuffix_max']:.1f}x, so the opportunity depends on how much of each context is shared. "
             f"A second model (Qwen2.5-7B, GQA 28/4) behaves the same way: cache {q[0].min():.1f}-{q[0].max():.1f}x, oracle {q[1].min():.1f}-{q[1].max():.1f}x."),
           H2("5.6  Correction: the many-prefix collapse is a dispatch artifact"),
           P(f"With 16 distinct prefixes, requests are assigned to prefix groups round-robin. Under request-major dispatch the requests that share a prefix are as far apart in time as possible and the 16 per-layer prefixes do not fit in the buffer, "
             f"so the cache gets <b>{h['cache_16_rr']:.2f}x</b>: nothing. This looked like a hardware limit. When the same workload is dispatched <b>group-major</b>, so that sharers run together, the cache recovers to <b>{h['cache_16_gm']:.2f}x</b> and the existing mcfill "
             f"multicast reaches <b>{h['mcfill_16_gm']:.2f}x</b>, equal to the oracle's {h['oracle_16_gm']:.2f}x. The failure is a scheduling effect, and the correct statement is: <i>a cache fails when sharers are interleaved in dispatch</i>."),
           fig("f6_dispatch_order", "<b>Figure 7.</b> Many distinct prefixes, B=128, P=4K, 256 private tokens. (a) Sharers interleaved: the cache gains nothing at 16 prefixes. (b) Sharers dispatched together: the cache recovers and mcfill reaches the oracle."),
           table(N.t_dispatch(), None, "<b>Table 6.</b> Attention speedup over the same-order scratchpad, B=128, P=4K (E5)."),
           P("Consequence for the hardware argument: handling many prefixes is a scheduling problem that software can solve. The case for hardware must rest on the regime where many cores read <i>the same</i> prefix, which Section 6 addresses.", "note"),
           H2("5.7  Whole decode step, equal area, and the software alternative"),
           P("Adding everything outside attention (from LLMCompass, for the same area split) dilutes the attention gain (Figure 8, Table 7). Across the six equal-area SRAM splits the step speedup changes by at most about 0.14x for the cache and 0.4x for cache + mcfill, "
             "so <b>how the area is split is not the lever; the mechanism is</b>. Importantly, a <b>prefix-aware software kernel</b> on the plain scratchpad already reaches about the oracle's step speedup, which is the strongest objection to any hardware proposal: "
             "hardware must offer something a cascade kernel does not."),
           fig("f7_step_speedup", "<b>Figure 8.</b> Whole decode-step speedup over the scratchpad at the GA100 SRAM configuration, including the prefix-aware software kernel (LLMCompass core time, no launch overhead)."),
           table(N.t_step(), None, "<b>Table 7.</b> Whole decode-step speedup over the scratchpad (E6, split L192_G48).")]
    return out
