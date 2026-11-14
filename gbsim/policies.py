"""Global-buffer policies. THIS IS THE PLUG-IN POINT FOR THE SOLUTION.

A policy decides, for each block request from a core:
  access(blk, shared, core, now) -> "hit" | "miss" | "merge"
      hit   : block is resident; served from the global buffer
      miss  : fetch from DRAM (sim calls fill() when the data arrives)
      merge : an in-flight DRAM fetch of the same block will also serve this core (MSHR)
  fill(blk, shared, now)          : DRAM data arrived; insert / evict as the policy wishes
  gb_read_bytes(blk, core, nbytes): global-buffer read bytes charged for delivering blk to core
                                    (multicast/coalescing designs charge less for repeats)
  merges_misses                   : whether "merge" may be returned
  multicast_merged                : if True, one global-buffer read delivers a filled block to ALL
                                    cores merged on that miss (each core's port is still charged)

Shipped:
  Scratchpad       software-managed staging (LLMCompass / TPU-style DMA per request): no reuse
  Cache            hardware-managed, set-associative LRU over physical blocks, MSHR merging
  CacheSharedPin   Cache, but blocks referenced by >1 request (prefix) are evicted last
  CacheWindowMulticast    OUR DESIGN v1: time-window multicast table (see class docstring)
  CacheWindowMulticastPin OUR DESIGN v2: v1 + bypass pinning for prefixes larger than the buffer
  OracleMulticast  UPPER BOUND, not a design: infinite capacity, each distinct block is read from
                   DRAM once and from the global buffer once, whatever the timing
"""
from collections import OrderedDict


class GlobalBuffer:
    name = "base"
    merges_misses = False

    def access(self, blk, shared, core, now):
        raise NotImplementedError

    def fill(self, blk, shared, now):
        pass

    def gb_read_bytes(self, blk, core, nbytes):
        return nbytes


class Scratchpad(GlobalBuffer):
    name = "scratchpad"

    def access(self, blk, shared, core, now):
        return "miss"


class Cache(GlobalBuffer):
    """Set-associative LRU. ways=None -> fully associative."""
    name = "cache"
    merges_misses = True

    def __init__(self, capacity_bytes, block_bytes, ways=16, merge=True):
        self.lines = max(1, int(capacity_bytes // block_bytes))
        self.ways = self.lines if ways is None else min(ways, self.lines)
        self.n_sets = max(1, self.lines // self.ways)
        self.sets = [OrderedDict() for _ in range(self.n_sets)]
        self.merges_misses = merge
        if not merge:
            self.name = "cache-nomshr"

    def _set(self, blk):
        # multiplicative hash so that consecutive block ids spread over sets
        return self.sets[(blk * 2654435761 >> 7) % self.n_sets] if self.n_sets > 1 else self.sets[0]

    def access(self, blk, shared, core, now):
        s = self._set(blk)
        if blk in s:
            s.move_to_end(blk)
            return "hit"
        return "miss"

    def _victim(self, s):
        return next(iter(s))

    def fill(self, blk, shared, now):
        s = self._set(blk)
        if blk in s:
            s.move_to_end(blk)
            return
        if len(s) >= self.ways:
            del s[self._victim(s)]
        s[blk] = shared


class CacheSharedPin(Cache):
    name = "cache-sharedpin"

    def _victim(self, s):
        for b, shared in s.items():  # LRU order
            if not shared:
                return b
        return next(iter(s))


class CacheMulticastFill(Cache):
    """EXAMPLE plug-in (shows the interface; not the team's solution): Cache whose MSHR delivers a
    filled block to all merged cores with one global-buffer read. Later hits are still per core."""
    name = "cache-mcfill"
    multicast_merged = True


class OracleMulticast(GlobalBuffer):
    name = "oracle-multicast"
    merges_misses = True
    multicast_merged = True

    def __init__(self):
        self.resident, self.delivered = set(), set()

    def access(self, blk, shared, core, now):
        return "hit" if blk in self.resident else "miss"

    def fill(self, blk, shared, now):
        self.resident.add(blk)

    def gb_read_bytes(self, blk, core, nbytes):
        if blk in self.delivered:
            return 0
        self.delivered.add(blk)
        return nbytes


class CacheWindowMulticast(Cache):
    """OUR DESIGN (v1): cache + time-window multicast.

    The global buffer keeps a small table of blocks that were read recently. A request for a block
    whose global-buffer read began less than `window_s` ago joins that broadcast instead of causing
    another global-buffer read (its core port is still charged). The window is not refreshed when a
    request joins, so it cannot chain forever. `max_live` is the peak number of table entries.
    window_s = 0 reduces to cache-mcfill; window_s = inf approaches the oracle's global-buffer traffic."""
    multicast_merged = True

    def __init__(self, capacity_bytes, block_bytes, ways=16, window_s=1e-6):
        super().__init__(capacity_bytes, block_bytes, ways)
        self.window_s = window_s
        self.name = "cache-wmc-" + ("inf" if window_s == float("inf") else f"{window_s * 1e9:g}")
        self.live = OrderedDict()      # blk -> time of the global-buffer read that opened its window
        self.max_live = 0              # peak table occupancy (entries)
        self._now = 0.0

    def access(self, blk, shared, core, now):
        self._now = now
        return super().access(blk, shared, core, now)

    def fill(self, blk, shared, now):
        self._now = now
        super().fill(blk, shared, now)

    def gb_read_bytes(self, blk, core, nbytes):
        now, w = self._now, self.window_s
        while self.live:                                  # drop expired windows (oldest first)
            b0 = next(iter(self.live))
            if now - self.live[b0] > w:
                del self.live[b0]
            else:
                break
        t = self.live.get(blk)
        if t is not None and now - t <= w:
            return 0                                      # joins the broadcast already in flight
        self.live[blk] = now
        self.live.move_to_end(blk)
        self.max_live = max(self.max_live, len(self.live))
        return nbytes


class CacheWindowMulticastPin(CacheWindowMulticast):
    """OUR DESIGN (v2): window multicast + bypass pinning for prefixes larger than the buffer.

    Plain LRU thrashes when the shared working set exceeds the buffer and is scanned cyclically (every wave
    of work items streams the whole prefix). Here shared blocks are never evicted by other shared blocks: when a
    set holds only shared blocks, a newly fetched shared block is delivered to its requesting cores but NOT
    inserted (bypass), so the resident part of the prefix stays stable and keeps hitting. Private blocks are
    evicted first."""

    def __init__(self, capacity_bytes, block_bytes, ways=16, window_s=1e-6):
        super().__init__(capacity_bytes, block_bytes, ways, window_s)
        self.name = self.name.replace("cache-wmc-", "cache-wmcpin-")

    def fill(self, blk, shared, now):
        self._now = now
        s = self._set(blk)
        if blk in s:
            s.move_to_end(blk)
            return
        if len(s) >= self.ways:
            victim = next((b for b, sh in s.items() if not sh), None)
            if victim is None:
                if shared:
                    return                      # bypass: keep the resident shared blocks
                victim = next(iter(s))
            del s[victim]
        s[blk] = shared


def make(name, hw, model, ways=16):
    cap = hw.gb_MB * 2**20
    if name == "scratchpad":
        return Scratchpad()
    if name == "cache":
        return Cache(cap, model.block_bytes, ways)
    if name == "cache-nomshr":
        return Cache(cap, model.block_bytes, ways, merge=False)
    if name == "cache-fa":
        return Cache(cap, model.block_bytes, None)
    if name == "cache-sharedpin":
        return CacheSharedPin(cap, model.block_bytes, ways)
    if name == "cache-mcfill":
        return CacheMulticastFill(cap, model.block_bytes, ways)
    if name == "oracle-multicast":
        return OracleMulticast()
    if name.startswith("cache-wmcpin-"):   # cache-wmcpin-<window in ns>
        return CacheWindowMulticastPin(cap, model.block_bytes, ways, float(name.split("-")[-1]) * 1e-9)
    if name.startswith("cache-wmc-"):   # cache-wmc-<window in ns>, e.g. cache-wmc-300, cache-wmc-inf
        return CacheWindowMulticast(cap, model.block_bytes, ways, float(name.split("-")[-1]) * 1e-9)
    raise ValueError(name)


POLICIES = ("scratchpad", "cache", "cache-sharedpin", "oracle-multicast")
