"""E10 -- Bandwidth allocation: how much global-buffer bandwidth does multicast save?

Multicast delivers a shared block with one buffer read instead of one per core, so the buffer needs
less bandwidth for the same speed. Question: at B in {32,128}, P=4K, how does each design's speed depend
on the buffer bandwidth (1.8-7.2 TB/s), measured against the existing chip (scratchpad at 7.2 TB/s)?
Output: results/data/e10_bandwidth_allocation.csv
        attn_speedup_vs_scratchpad       : vs the scratchpad at the SAME buffer bandwidth
        speedup_vs_baseline_chip         : vs the scratchpad at 7.2 TB/s (the unmodified chip)
"""
import _common as C

POL = ["scratchpad", "cache", "cache-wmc-3000", "oracle-multicast"]
BW = (1.8, 3.6, 5.4, 7.2)


def main(workers=None):
    J = [dict(exp="e10", policy=x, batch=b, prefix=4096, hw=dict(gb_bw=tb * 1e12)) for b in (32, 128) for tb in BW for x in POL]
    d = C.add_speedup(C.run_jobs(J, workers), ["batch", "prefix", "gb_TBps"])
    base = d[(d.policy == "scratchpad") & (d.gb_TBps.round(1) == 7.2)].set_index("batch").time_s
    d["speedup_vs_baseline_chip"] = [base[r.batch] / r.time_s for r in d.itertuples()]
    C.save(d, "e10_bandwidth_allocation.csv")
    print(d.pivot_table(index=["batch", "gb_TBps"], columns="policy", values="speedup_vs_baseline_chip")[POL].round(2).to_string())


if __name__ == "__main__":
    C.cli(main)
