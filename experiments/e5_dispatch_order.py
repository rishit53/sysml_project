"""E5 -- Dispatch order: is the multi-prefix collapse a hardware limit or a scheduling artifact?

E4 assigns requests to prefix groups round-robin and dispatches request-major, which puts the
requests that share a prefix as far apart in time as possible. Here the same workload is also
dispatched group-major (sharers run together, as a sharing-aware scheduler would).
Output: results/data/e5_dispatch_order.csv   (speedups are vs the scratchpad with the same order)
"""
import _common as C

POL = ("scratchpad", "cache", "cache-mcfill", "oracle-multicast")


def main(workers=None):
    J = [dict(exp="e5", policy=x, batch=b, prefix=4096, suffix=s, n_prefixes=k, order=o)
         for b in (32, 128) for k in (1, 4, 16) for s in (256, 4096)
         for o in ("request-major", "group-major") for x in POL]
    d = C.add_speedup(C.run_jobs(J, workers), ["batch", "suffix", "n_prefixes", "order"])
    C.save(d, "e5_dispatch_order.csv")
    t = d[d.suffix == 256].pivot_table(index=["batch", "n_prefixes", "order"], columns="policy", values=C.SPEEDUP)
    print(t[list(POL)].round(2).to_string())


if __name__ == "__main__":
    C.cli(main)
