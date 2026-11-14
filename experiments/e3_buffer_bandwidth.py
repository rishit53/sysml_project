"""E3 -- Global-buffer bandwidth: is the cache path bandwidth-capped?

Question: how does each design scale when the global-buffer bandwidth is 3.3, 4.4 or 7.2 TB/s
(3.3 = A100 shared-page throughput measured by ChunkAttention; 7.2 = the GA100 template)?
Output: results/data/e3_gb_bandwidth.csv
"""
import _common as C


def jobs():
    return [dict(exp="e3", policy=x, batch=b, prefix=p, hw=dict(gb_bw=tb * 1e12))
            for b in (32, 128) for p in (4096, 16384) for tb in (3.3, 4.4, 7.2)
            for x in ("scratchpad", "cache", "cache-mcfill", "oracle-multicast")]


def main(workers=None):
    d = C.add_speedup(C.run_jobs(jobs(), workers), ["batch", "prefix", "gb_TBps"])
    C.save(d, "e3_gb_bandwidth.csv")
    print(d.pivot_table(index=["batch", "prefix", "gb_TBps"], columns="policy", values=C.SPEEDUP).round(2).to_string())


if __name__ == "__main__":
    C.cli(main)
