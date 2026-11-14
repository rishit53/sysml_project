"""E2 -- Global-buffer capacity: does the cache need size or concurrency?

Question: how does DRAM deduplication change with the buffer size (11-67 MB, the sizes of the six
equal-area SRAM splits), and does refcount-aware replacement (shared-pin) help?
Output: results/data/e2_capacity.csv
"""
import _common as C


def jobs():
    return [dict(exp="e2", policy=x, batch=b, prefix=p, hw=dict(gb_MB=mb))
            for b in (32, 128) for p in (4096, 16384) for mb in (11, 27, 48, 67)
            for x in ("cache", "cache-sharedpin")]


def main(workers=None):
    d = C.run_jobs(jobs(), workers)
    C.save(d, "e2_capacity.csv")
    print(d.pivot_table(index=["batch", "prefix", "gb_MB"], columns="policy", values="dram_dedup_x").round(1).to_string())


if __name__ == "__main__":
    C.cli(main)
