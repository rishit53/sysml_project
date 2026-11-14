"""M1b: does cache-based prefix dedup survive realistic core drift?

Same single-layer attention trace as m1_prefix_cache.py, but work items do not
progress in lockstep: each active item has its own speed (blocks per tick)
drawn from [1 - j, 1 + j] (core contention, variable start, NoC arbitration),
and new items start when a core frees up. Readers of the same prefix block
spread out over time, so the cache must hold the blocks between the leading
and trailing reader. Sweep jitter j and global-buffer capacity.
Deterministic seed for reproducibility.
"""
import random
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import common

NKV, BLOCK_TOK, BLK_BYTES, CORES = 8, 16, 8192, 128


def run(b, p, s, cap_mb, jitter, order="head-major", seed=0):
    rng = random.Random(seed)
    pb, sb = p // BLOCK_TOK, s // BLOCK_TOK
    nblk = pb + sb
    items = [(r, h) for h in range(NKV) for r in range(b)] if order == "head-major" \
        else [(r, h) for r in range(b) for h in range(NKV)]
    queue = iter(items)
    cap = int(cap_mb * 2**20 // BLK_BYTES)
    lru = OrderedDict()
    acc = miss = 0

    def new(it):
        return dict(it=it, pos=0.0, done=0, speed=rng.uniform(1 - jitter, 1 + jitter))

    active = [new(it) for it in [next(queue, None) for _ in range(CORES)] if it is not None]
    while active:
        nxt = []
        for a in active:
            a["pos"] += a["speed"]
            r, h = a["it"]
            while a["done"] < min(int(a["pos"]), nblk):
                i = a["done"]
                blk = ("P", h, i) if i < pb else ("S", r, h, i - pb)
                acc += 1
                if blk in lru:
                    lru.move_to_end(blk)
                else:
                    miss += 1
                    lru[blk] = True
                    if len(lru) > cap:
                        lru.popitem(last=False)
                a["done"] += 1
            if a["done"] < nblk:
                nxt.append(a)
            else:
                it = next(queue, None)
                if it is not None:
                    nxt.append(new(it))
        active = nxt
    distinct = NKV * pb + b * NKV * sb
    return acc, miss, distinct


def main():
    rows = []
    for b, p in ((32, 4096), (128, 4096), (32, 16384), (128, 16384)):
        for j in (0.0, 0.1, 0.3, 0.5):
            for cap in (2, 11, 27, 48):
                acc, miss, dist = run(b, p, 256, cap, j)
                rows.append(dict(batch=b, prefix=p, jitter=j, gb_MB=cap,
                                 reduction_x=acc / miss, ideal_x=acc / dist,
                                 recovered=(acc - miss) / (acc - dist)))
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(common.OUT / "loop" / "m1b_prefix_cache_jitter.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.pivot_table(index=["batch", "prefix", "jitter"], columns="gb_MB", values="recovered").round(2).to_string())


if __name__ == "__main__":
    main()
