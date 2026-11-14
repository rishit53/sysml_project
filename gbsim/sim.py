"""Event-driven timing model of one decode-attention layer's on-chip memory system.

Per core: one work item at a time; up to `window` outstanding block requests (prefetch);
blocks are consumed in order, each taking block_compute_s. When an item finishes, the core
takes the next item from the dispatch queue (so core drift emerges from contention).

Shared resources are FIFO bandwidth servers (reserve-on-arrival):
  DRAM            hbm_bw, then hbm_lat
  global buffer   gb_bw, aggregate (reads to cores + fill writes x fill_cost)
  core port       port_bw per core, then gb_lat
Request path: hit -> GB -> port -> core;  miss -> DRAM -> fill (GB) -> GB -> port -> core;
merge -> waits for the in-flight fill, then GB -> port -> core.

run() returns times, bytes per level and utilisations for one layer.
"""
import heapq
import itertools
import random

from . import policies as P
from .config import HW, LLAMA3_8B
from .workload import Sharing, Workload


def run(hw: HW, policy, wl: Workload, model=LLAMA3_8B, zero_time=False, jitter=0.0, skew_s=0.0, seed=0):
    """Simulate one layer. zero_time=True: infinite bandwidth / zero latency / equal compute
    (cores advance in lockstep; used to check functional hit counts against the trace model).
    jitter: each core's compute time is scaled by a fixed factor drawn uniformly from 1 +- jitter
            (heterogeneous core speed, so cores drift apart; 0 = identical cores).
    skew_s: each core starts after a random delay in [0, skew_s] (0 = all cores start together).
    seed:   random seed for the two draws above (results are deterministic for a given seed)."""
    if isinstance(policy, str):
        policy = P.make(policy, hw, model)
    B = model.block_bytes
    t_c = hw.block_compute_s(model, wl.rows)
    if zero_time:
        hbm_s = gb_s = port_s = hbm_lat = gb_lat = 0.0
        t_c = 1.0
    else:
        hbm_s, gb_s = B / hw.hbm_bw, B / hw.gb_bw
        port_s = 0.0 if hw.port_bw == float("inf") else B / hw.port_bw
        hbm_lat, gb_lat = hw.hbm_lat, hw.gb_lat

    rng = random.Random(seed)
    t_core = [t_c * (1 + jitter * (2 * rng.random() - 1)) for _ in range(hw.cores)] if (jitter and not zero_time) else [t_c] * hw.cores
    ev, seq = [], itertools.count()
    push = lambda t, kind, a, b=None: heapq.heappush(ev, (t, next(seq), kind, a, b))  # noqa: E731
    hbm_free = gb_free = 0.0
    hbm_busy = gb_busy = 0.0
    port_free = [0.0] * hw.cores
    st = dict(dram_bytes=0, gb_read_bytes=0, gb_fill_bytes=0, port_bytes=0, hits=0, misses=0, merges=0)
    mshr = {}
    items = iter(wl.items)
    n = wl.n_blocks
    pb = wl.pb
    # per core: [p0, s0, shared, next_issue, arrived, computed, busy]
    cores = [None] * hw.cores
    done_t = 0.0

    def start_item(c, now):
        it = next(items, None)
        if it is None:
            cores[c] = None
            return
        p0, s0, sh = wl.blocks(it)
        cores[c] = [p0, s0, sh, 0, 0, 0, False]
        issue(c, now)

    def deliver(c, blk, now, charge=True):
        nonlocal gb_free, gb_busy
        rb = policy.gb_read_bytes(blk, c, B) if charge else 0
        st["gb_read_bytes"] += rb
        st["port_bytes"] += B
        s = gb_s * rb / B
        t = max(now, gb_free) + s
        gb_free, gb_busy = t, gb_busy + s
        t2 = max(t, port_free[c]) + port_s
        port_free[c] = t2
        push(t2 + gb_lat, "arrive", c)

    def issue(c, now):
        nonlocal hbm_free, hbm_busy
        cs = cores[c]
        while cs[3] < n and cs[3] - cs[5] < hw.window:
            j = cs[3]
            cs[3] += 1
            if j < pb:
                blk, shared = cs[0] + j, cs[2]
            else:
                blk, shared = cs[1] + j - pb, False
            r = policy.access(blk, shared, c, now)
            if r == "hit":
                st["hits"] += 1
                deliver(c, blk, now)
            elif policy.merges_misses and blk in mshr:
                st["merges"] += 1
                mshr[blk].append(c)
            else:
                st["misses"] += 1
                st["dram_bytes"] += B
                t = max(now, hbm_free) + hbm_s
                hbm_free, hbm_busy = t, hbm_busy + hbm_s
                key = blk if policy.merges_misses else (blk, c, next(seq))
                mshr[key] = [c]
                push(t + hbm_lat, "fill", key, shared)

    def try_compute(c, now):
        cs = cores[c]
        if not cs[6] and cs[4] > cs[5]:
            cs[6] = True
            push(now + t_core[c], "computed", c)

    for c in range(hw.cores):
        if skew_s and not zero_time:
            push(skew_s * rng.random(), "start", c)
        else:
            start_item(c, 0.0)

    while ev:
        now, _, kind, a, b = heapq.heappop(ev)
        if kind == "fill":
            blk = a if not isinstance(a, tuple) else a[0]
            policy.fill(blk, b, now)
            fb = B * hw.fill_cost
            st["gb_fill_bytes"] += fb
            s = gb_s * hw.fill_cost
            gb_free = max(now, gb_free) + s
            gb_busy += s
            mc = getattr(policy, "multicast_merged", False)
            for i, c in enumerate(mshr.pop(a)):
                deliver(c, blk, gb_free, charge=(i == 0 or not mc))
        elif kind == "arrive":
            cores[a][4] += 1
            try_compute(a, now)
        elif kind == "start":
            start_item(a, now)
        else:  # computed
            cs = cores[a]
            cs[5] += 1
            cs[6] = False
            done_t = max(done_t, now)
            if cs[5] == n:
                start_item(a, now)
            else:
                issue(a, now)
                try_compute(a, now)

    T = done_t
    acc = wl.accesses()
    return dict(policy=policy.name, time_s=T, accesses=acc, distinct=wl.distinct_blocks(),
                **st, dram_dedup_x=acc * B / max(st["dram_bytes"], 1),
                hbm_util=hbm_busy / T if T else 0, gb_util=gb_busy / T if T else 0,
                compute_floor_s=acc * t_c / hw.cores if not zero_time else 0)


def layer(hw, policy, model=LLAMA3_8B, batch=32, prefix=4096, suffix=256, n_prefixes=1,
          order="request-major", qhead=False, ways=16, jitter=0.0, skew_s=0.0, seed=0):
    pol = P.make(policy, hw, model, ways) if isinstance(policy, str) else policy
    wl = Workload(model, Sharing(batch, prefix, suffix, n_prefixes), order, qhead)
    r = run(hw, pol, wl, model, jitter=jitter, skew_s=skew_s, seed=seed)
    r.update(hw=hw.name, model=model.name, batch=batch, prefix=prefix, suffix=suffix, n_prefixes=n_prefixes,
             order=order, qhead=qhead, gb_MB=hw.gb_MB, gb_TBps=hw.gb_bw / 1e12, ways=ways, jitter=jitter, skew_s=skew_s, seed=seed)
    return r
