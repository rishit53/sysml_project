"""Mechanism iteration M1: cache-managed vs scratchpad global buffer for prefix-cached decode.

With vLLM/SGLang prefix caching, B requests' KV block tables point at the SAME
physical prefix blocks. A plain (non-prefix-aware) decode attention kernel
processes work items (request, kv_head) and streams each item's blocks. A
software-managed scratchpad global buffer (LLMCompass template, TPU-style
CMEM/VMEM staging) gives no reuse: the prefix is read B times. A hardware-
managed cache could dedup reads transparently, without Hydragen/cascade kernel
changes -- if capacity, replacement policy and work-item order cooperate.

Trace-driven model of ONE layer's attention operator:
  - Llama-3-8B: 8 KV heads, head dim 128, fp16 -> K+V per token per head 512 B;
    paged blocks of 16 tokens -> 8 KB per (head, block).
  - 128 cores; each active core streams one work item block by block; accesses
    interleave round-robin across active items; finished items are replaced
    from the queue (request-major or head-major order).
  - Global buffer of C MB as a fully-associative cache over 8 KB blocks.
    Policies: LRU, and LRU + pin-shared (blocks with refcount > 1, i.e. prefix
    blocks, are evicted only after private blocks).
DRAM bytes = misses x 8 KB. Compared with scratchpad (every access misses)
and ideal (each distinct block once).

Hypothesis HM1: a cache-managed global buffer recovers >= 80% of the ideal
prefix dedup when the per-layer prefix fits, but plain LRU collapses (near 0%
hits) once it does not, in request-major order; a prefix-pinning policy or
head-major order keeps most of the benefit. Effect on attention DRAM bytes
>= 3x at B = 32, P = 4K -- a memory-mechanism difference, not a capacity one.
"""
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import common

NKV, BLOCK_TOK, BLK_BYTES, CORES = 8, 16, 8192, 128


def trace(b, p, s, order):
    """Yield (block_id, shared) for one attention op, interleaved over 128 cores."""
    pb, sb = p // BLOCK_TOK, s // BLOCK_TOK
    items = [(r, h) for r in range(b) for h in range(NKV)] if order == "request-major" \
        else [(r, h) for h in range(NKV) for r in range(b)]
    queue = iter(items)
    active = []
    for _ in range(CORES):
        it = next(queue, None)
        if it is None:
            break
        active.append([it, 0])
    n_blocks = pb + sb
    while active:
        nxt = []
        for slot in active:
            (r, h), i = slot
            if i < pb:
                yield ("P", h, i), True
            else:
                yield ("S", r, h, i - pb), False
            slot[1] += 1
            if slot[1] < n_blocks:
                nxt.append(slot)
            else:
                it = next(queue, None)
                if it is not None:
                    nxt.append([it, 0])
        active = nxt


def simulate(b, p, s, cap_mb, order, policy):
    cap = int(cap_mb * 2**20 // BLK_BYTES)
    shared_lru, priv_lru = OrderedDict(), OrderedDict()
    misses = accesses = 0
    for blk, shared in trace(b, p, s, order):
        accesses += 1
        lru = shared_lru if (shared and policy == "pin-shared") else priv_lru
        if blk in shared_lru:
            shared_lru.move_to_end(blk)
            continue
        if blk in priv_lru:
            priv_lru.move_to_end(blk)
            continue
        misses += 1
        lru[blk] = True
        while len(shared_lru) + len(priv_lru) > cap:
            if priv_lru:
                priv_lru.popitem(last=False)
            else:
                shared_lru.popitem(last=False)
    distinct = NKV * (p // BLOCK_TOK) + b * NKV * (s // BLOCK_TOK)
    return accesses, misses, distinct


def main():
    rows = []
    for b in (8, 32, 128):
        for p in (1024, 4096, 16384):
            s = 256
            for cap in (11, 27, 48, 67):
                for order in ("request-major", "head-major"):
                    for policy in ("lru", "pin-shared"):
                        acc, miss, dist = simulate(b, p, s, cap, order, policy)
                        rows.append(dict(batch=b, prefix=p, suffix=s, gb_MB=cap, order=order, policy=policy,
                                         prefix_per_layer_MB=NKV * p * 512 / 2**20,
                                         scratchpad_GB=acc * BLK_BYTES / 1e9, cache_GB=miss * BLK_BYTES / 1e9,
                                         ideal_GB=dist * BLK_BYTES / 1e9,
                                         dedup_recovered=(acc - miss) / max(acc - dist, 1),
                                         reduction_x=acc / miss))
                        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(common.OUT / "loop" / "m1_prefix_cache.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.pivot_table(index=["batch", "prefix", "gb_MB"], columns=["order", "policy"],
                        values="reduction_x").round(2).to_string())


if __name__ == "__main__":
    main()
