"""Decode-attention work items for prefix-cached serving with an ordinary (not prefix-aware) kernel.

Every request has a block table per KV head. With prefix caching (vLLM / SGLang), requests in
the same prefix group point at the SAME physical prefix blocks; their suffix blocks are private.
The kernel processes one work item at a time per core and streams the item's blocks.

  item = (request, kv_head)           GQA-grouped kernel (FlashDecoding / FlashInfer style), g rows
  item = (request, q_head)  ("qhead")  one query head per item, reads its KV head (vLLM v1 style), 1 row

Dispatch orders:
  request-major  all heads of request 0, then request 1, ... (CUDA blockIdx.x = head fastest; vLLM-like)
  head-major     all requests of head 0, then head 1, ...
  group-major    all requests of prefix group 0 (request-major inside the group), then group 1, ...
                 i.e. a sharing-aware scheduler that runs sharers together. With n_prefixes = 1 it is
                 identical to request-major.
"""
from dataclasses import dataclass

from .config import BLOCK_TOKENS


@dataclass(frozen=True)
class Sharing:
    batch: int = 32
    prefix: int = 4096        # shared prefix tokens per request
    suffix: int = 256         # private tokens per request
    n_prefixes: int = 1       # requests are assigned round-robin to this many prefix groups

    def group(self, r):
        return r % self.n_prefixes


class Workload:
    def __init__(self, model, sharing, order="request-major", qhead=False):
        self.m, self.s, self.order, self.qhead = model, sharing, order, qhead
        self.pb = -(-sharing.prefix // BLOCK_TOKENS)
        self.sb = -(-sharing.suffix // BLOCK_TOKENS)
        self.n_blocks = self.pb + self.sb
        heads = model.n_q if qhead else model.n_kv
        rs = range(sharing.batch)
        if order == "request-major":
            self.items = [(r, h) for r in rs for h in range(heads)]
        elif order == "head-major":
            self.items = [(r, h) for h in range(heads) for r in rs]
        elif order == "group-major":
            self.items = [(r, h) for k in range(sharing.n_prefixes) for r in rs
                          if sharing.group(r) == k for h in range(heads)]
        else:
            raise ValueError(order)
        self.rows = 1 if qhead else model.g
        self._suffix_base = sharing.n_prefixes * model.n_kv * self.pb
        sizes = [sum(1 for r in rs if sharing.group(r) == k) for k in range(sharing.n_prefixes)]
        self._shared_group = [n > 1 for n in sizes]

    def blocks(self, item):
        """(prefix_start, suffix_start, prefix_is_shared): block j of the item is
        prefix_start + j for j < pb, else suffix_start + (j - pb)."""
        r, h = item
        kvh = h // self.m.g if self.qhead else h
        grp = self.s.group(r)
        p0 = (grp * self.m.n_kv + kvh) * self.pb
        s0 = self._suffix_base + (r * self.m.n_kv + kvh) * self.sb
        return p0, s0, self._shared_group[grp]

    def accesses(self):
        return len(self.items) * self.n_blocks

    def distinct_blocks(self):
        return self.s.n_prefixes * self.m.n_kv * self.pb + self.s.batch * self.m.n_kv * self.sb
