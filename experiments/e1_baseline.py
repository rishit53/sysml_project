"""E1 -- Baseline: scratchpad vs cache vs multicast bound (attention, one layer).

Question: with prefix caching and an ordinary attention kernel, how much decode bandwidth does each
on-chip memory organisation recover, and how far is that from the best possible (the oracle)?
Grid: batch {8,32,128} x shared-prefix length {1K,4K,16K} x dispatch order x policy; one shared
prefix, 256 private tokens per request, Llama-3-8B, GA100 template (48 MB buffer, 7.2 TB/s).
Output: results/data/e1_baseline.csv   (speedup column: attn_speedup_vs_scratchpad)
"""
import _common as C

POL = ("scratchpad", "cache", "cache-sharedpin", "cache-mcfill", "oracle-multicast")


def jobs():
    return [dict(exp="e1", policy=x, batch=b, prefix=p, order=o)
            for b in (8, 32, 128) for p in (1024, 4096, 16384)
            for o in ("request-major", "head-major") for x in POL]


def main(workers=None):
    d = C.add_speedup(C.run_jobs(jobs(), workers), ["batch", "prefix", "order"])
    C.save(d, "e1_baseline.csv")
    t = d[d.order == "request-major"].pivot_table(index=["batch", "prefix"], columns="policy", values=C.SPEEDUP)
    print(t[list(POL)].round(2).to_string())


if __name__ == "__main__":
    C.cli(main)
