"""M1c: three-way comparison per decode step (Llama-3-8B, GA100, S = 256).

  scratchpad + naive kernel   : LLMCompass 'unshared' (prefix read B times from HBM)
  scratchpad + prefix kernel  : LLMCompass 'shared' (Hydragen/cascade, project A)
  cache      + naive kernel   : same naive work items, global buffer as a cache;
                                attention time = max(KV DRAM misses / HBM,
                                (KV accesses + score-matrix bytes) / global-buffer bw)
                                + the softmax op's LLMCompass time (unfused, as in
                                the scratchpad baseline); misses from the trace model
                                (head-major order, plain LRU, jitter 0.3, 48 MB)
Global-buffer bandwidth sensitivity: LLMCompass GA100 7.2 TB/s; 4.4 TB/s
ratio-calibrated from ChunkAttention (arXiv 2402.15220 Table 3: fully shared
context, MHA, A100-80GB, unchanged PagedAttention 2.1-2.3x faster), so the
resulting ~2x attention speedup is by construction, not a validation; 3.3 TB/s
= ChunkAttention's measured absolute shared-page throughput.
Non-attention time is LLMCompass core_s (no launch overhead), same for all three.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import common
from studies.m1b_prefix_cache_jitter import run as cache_run

HBM = 2.048e12
# 4.4 TB/s = ChunkAttention's measured shared/unshared speedup (mean 2.16) x HBM peak
# (a ratio calibration, so the ~2x attention speedup it gives is by construction);
# 3.3 TB/s = ChunkAttention's measured absolute shared-page throughput (3.2-3.5 TB/s).
BW = {"gbuf 7.2 TB/s": 5120 * 1.41e9, "ratio-calibrated 4.4 TB/s": 4.4e12, "measured-abs 3.3 TB/s": 3.3e12}
BLK = 8192


def main():
    d = pd.read_csv(common.OUT / "prefix_results.csv")
    d = d[(d.design == "GA100") & (d.split_id.isin(["shipped", "L192_G48"])) & (d.operator != "INFEASIBLE")]
    rows = []
    for (b, p), g in d.groupby(["batch", "prefix"]):
        if p not in (1024, 4096, 16384):
            continue
        g = g.drop_duplicates(["variant", "operator"])
        u, s = g[g.variant == "unshared"], g[g.variant == "shared"]
        if u.empty or s.empty:
            continue
        rest = u[u.group != "attention"].core_s.sum()
        att_u = u[u.group == "attention"].core_s.sum()
        mm = u[(u.group == "attention") & (u.operator != "attn_softmax")]
        softmax_s = u[u.operator == "attn_softmax"].core_s.sum()
        att_s = s[s.group == "attention"].core_s.sum()
        acc, miss, dist = cache_run(b, p, 256, 48, 0.3)
        layers = 32
        kv_access = layers * acc * BLK
        score_bytes = max(mm.sim_bytes.sum() - kv_access, 0)  # scores written + read by the matmuls
        for name, gbw in BW.items():
            att_c = max(layers * miss * BLK / HBM, (kv_access + score_bytes) / gbw) + softmax_s
            rows.append(dict(batch=b, prefix=p, gb_bw=name,
                             step_scratch_naive_ms=1e3 * (rest + att_u),
                             step_scratch_prefixkernel_ms=1e3 * (rest + att_s),
                             step_cache_naive_ms=1e3 * (rest + att_c),
                             attn_cache_vs_scratch_x=att_u / att_c,
                             step_cache_vs_scratch_x=(rest + att_u) / (rest + att_c),
                             step_kernel_vs_scratch_x=(rest + att_u) / (rest + att_s),
                             dram_dedup_x=acc / miss,
                             cache_share_of_kernel_gain=((rest + att_u) - (rest + att_c)) / max((rest + att_u) - (rest + att_s), 1e-12)))
    t = pd.DataFrame(rows)
    t.to_csv(common.OUT / "loop" / "m1c_threeway.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
