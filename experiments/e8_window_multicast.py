"""E8 -- Proposed design: time-window multicast global buffer (gbsim.policies.CacheWindowMulticast, and the v2 variant
with bypass pinning, CacheWindowMulticastPin, aimed at prefixes larger than the buffer).

Design: the global buffer keeps a small table of recently read blocks; a request for a block whose
buffer read began less than `window` ago joins that broadcast instead of causing another buffer read.
Question: how large a window closes the gap between cache-mcfill and the oracle, and how big is the table?
(a) single shared prefix: batch {32,128} x prefix {1K,4K,16K} x window {0,30,100,300,1K,3K,10K ns, inf}
(b) several prefixes (batch 128, prefix 4K): k {4,16} x dispatch order {request-major, group-major}
Output: results/data/e8_window_multicast.csv  (speedups are vs the scratchpad with the same workload)
        max_live = peak table entries (blocks tracked at once); E_*_uJ = ESTIMATED energy per layer
"""
import _common as C

BASE = ["scratchpad", "cache", "cache-mcfill", "oracle-multicast"]
WINDOWS_NS = ["0", "30", "100", "300", "1000", "3000", "10000", "inf"]
KEYS = ["batch", "prefix", "suffix", "n_prefixes", "order"]


def jobs():
    wmc = [f"cache-wmc-{w}" for w in WINDOWS_NS]
    pin = [f"cache-wmcpin-{w}" for w in ("1000", "3000")]
    J = [dict(exp="e8a", policy=x, batch=b, prefix=p) for b in (32, 128) for p in (1024, 4096, 16384) for x in BASE + wmc + pin]
    sel = BASE + ["cache-wmc-300", "cache-wmc-1000", "cache-wmc-inf"]
    J += [dict(exp="e8b", policy=x, batch=128, prefix=4096, n_prefixes=k, order=o)
          for k in (4, 16) for o in ("request-major", "group-major") for x in sel]
    return J


def main(workers=None):
    d = C.add_speedup(C.run_jobs(jobs(), workers), KEYS)
    d["window_ns"] = d.policy.map(lambda s: float(s.split("-")[-1]) if s.startswith("cache-wmc") else float("nan"))
    C.save(d, "e8_window_multicast.csv")
    a = d[d.exp == "e8a"]
    print(a.pivot_table(index=["batch", "prefix"], columns="policy", values=C.SPEEDUP)[BASE + [f"cache-wmc-{w}" for w in WINDOWS_NS]].round(2).to_string())


if __name__ == "__main__":
    C.cli(main)
