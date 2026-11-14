"""E9 -- Robustness of the window design to core drift.

The window design relies on cores reading the same block within a short time of each other. The
baseline simulator starts all cores together with identical speed (lockstep), which favours it.
Here each core gets a fixed random speed factor within +-jitter and a random start delay within skew,
(see gbsim.sim.run), averaged over seeds, to see how quickly the benefit disappears.
Grid: batch {32,128}, prefix 4K, jitter {0,10,25,50%}, start skew {0, 2 us}, 4 seeds (1 for the lockstep case).
Output: results/data/e9_drift.csv   (speedups are vs the scratchpad under the same drift and seed)
"""
import _common as C

POL = ["scratchpad", "cache", "cache-mcfill", "cache-wmc-100", "cache-wmc-300", "cache-wmc-1000",
       "cache-wmc-3000", "oracle-multicast"]


def jobs():
    J = []
    for b in (32, 128):
        for jit in (0.0, 0.10, 0.25, 0.50):
            for sk in (0.0, 2e-6):
                for seed in ((0,) if (jit == 0 and sk == 0) else (0, 1, 2, 3)):
                    J += [dict(exp="e9", policy=x, batch=b, prefix=4096, jitter=jit, skew_s=sk, seed=seed) for x in POL]
    return J


def main(workers=None):
    d = C.add_speedup(C.run_jobs(jobs(), workers), ["batch", "jitter", "skew_s", "seed"])
    C.save(d, "e9_drift.csv")
    g = d.groupby(["batch", "jitter", "skew_s", "policy"])[C.SPEEDUP].mean().unstack("policy")[POL]
    print(g.round(2).to_string())


if __name__ == "__main__":
    C.cli(main)
