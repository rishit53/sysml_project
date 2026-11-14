"""E7 -- Roofline and bottleneck map (analysis of E1; no new simulations).

For each design and operating point (request-major dispatch, one prefix, 256 private tokens):
  FLOPs per KV block   = 4 * g * 16 tokens * d_head              (QK^T and PV; g query heads per KV head)
  intensity vs HBM     = FLOPs / DRAM bytes
  intensity vs buffer  = FLOPs / (global-buffer read + fill bytes)
  roofs                = HBM bandwidth; global-buffer bandwidth; kernel compute ceiling (GQA fills only
                         g/array_h of the array, plus softmax vector work) and the peak MAC rate
  limiting resource    = the busiest of HBM, global buffer, compute (busy time / total time);
                         'latency/queueing' if none is more than 80% busy.
Output: results/data/e7_roofline.csv, results/data/e7_roofline_constants.csv
"""
import _common as C
import numpy as np
import pandas as pd
from gbsim.config import BLOCK_TOKENS, HW, LLAMA3_8B

POL = ["scratchpad", "cache", "cache-mcfill", "oracle-multicast"]


def constants():
    hw, M = HW(), LLAMA3_8B
    flops_blk = 4 * M.g * BLOCK_TOKENS * M.d_head
    return dict(flops_per_block=flops_blk, flop_per_kv_byte=flops_blk / M.block_bytes,
                peak_tflops=hw.cores * hw.mac_per_cycle * hw.freq * 2 / 1e12,
                kernel_ceiling_tflops=hw.cores * flops_blk / hw.block_compute_s(M) / 1e12,
                hbm_TBps=hw.hbm_bw / 1e12, gb_TBps=hw.gb_bw / 1e12,
                hbm_roof_tflops=flops_blk / M.block_bytes * hw.hbm_bw / 1e12,
                gb_roof_nomulticast_tflops=flops_blk / M.block_bytes * hw.gb_bw / 1e12,
                ridge_hbm_flop_per_byte=hw.cores * flops_blk / hw.block_compute_s(M) / hw.hbm_bw)


def main(workers=None):
    k = constants()
    d = pd.read_csv(C.DATA / "e1_baseline.csv")
    d = d[(d.order == "request-major") & (d.n_prefixes == 1) & (d.suffix == 256) & d.policy.isin(POL)].copy()
    d["flops"] = d.accesses * k["flops_per_block"]
    d["perf_TFLOPs"] = d.flops / d.time_s / 1e12
    d["I_dram"] = d.flops / d.dram_bytes
    d["I_gb"] = d.flops / (d.gb_read_bytes + d.gb_fill_bytes)
    d["compute_util"] = d.compute_floor_s / d.time_s
    util = d[["hbm_util", "gb_util", "compute_util"]].to_numpy()
    d["max_util"] = util.max(1)
    d["bottleneck"] = np.where(util.max(1) < 0.80, "latency/queueing", np.array(["HBM", "global buffer", "compute"])[util.argmax(1)])
    cols = ["policy", "batch", "prefix", C.SPEEDUP, "perf_TFLOPs", "I_dram", "I_gb", "hbm_util", "gb_util",
            "compute_util", "max_util", "bottleneck"]
    C.save(d[cols], "e7_roofline.csv")
    C.save(pd.DataFrame(k.items(), columns=["name", "value"]), "e7_roofline_constants.csv")
    print({a: round(b, 2) for a, b in k.items()})
    print(d.pivot_table(index=["batch", "prefix"], columns="policy", values="bottleneck", aggfunc="first")[POL].to_string())


if __name__ == "__main__":
    C.cli(main)
