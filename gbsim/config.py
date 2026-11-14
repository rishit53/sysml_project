"""Hardware and model parameters for gbsim.

`HW.from_llmcompass(specs)` reads an LLMCompass architecture template (e.g. GA100, or an
equal-area split from llmc/splits.py), so the simulator and LLMCompass describe the same chip.
Latencies are not in LLMCompass; the defaults are of the order of published A100
microbenchmarks and are hidden by the per-core prefetch window (see validate.py).
"""
from dataclasses import dataclass, replace

BLOCK_TOKENS = 16


@dataclass(frozen=True)
class Model:
    name: str = "Llama-3-8B"
    n_layers: int = 32
    n_q: int = 32
    n_kv: int = 8
    d_head: int = 128
    word: int = 2  # fp16 KV

    @property
    def g(self):
        return self.n_q // self.n_kv

    @property
    def block_bytes(self):
        """One paged block of one KV head: 16 tokens x (K + V) x d_head."""
        return BLOCK_TOKENS * 2 * self.d_head * self.word


LLAMA3_8B = Model()
QWEN25_7B = Model("Qwen2.5-7B", n_layers=28, n_q=28, n_kv=4, d_head=128)


@dataclass(frozen=True)
class HW:
    name: str = "GA100"
    cores: int = 128
    freq: float = 1.41e9
    hbm_bw: float = 2.048e12          # B/s
    hbm_lat: float = 500e-9           # s, loaded DRAM latency
    gb_bw: float = 5120 * 1.41e9      # B/s, global buffer aggregate read+write bandwidth
    gb_lat: float = 150e-9            # s, global buffer -> core
    gb_MB: float = 48
    port_bw: float = float("inf")     # B/s per core (LLMCompass has no per-core limit)
    fill_cost: float = 1.0            # GB bandwidth charged per filled byte (DRAM -> GB write)
    # outstanding block requests per core (prefetch depth). 8 x 8 KB = 64 KB of the 192 KB per-core
    # SRAM; covers the bandwidth-delay product (56 GB/s per-core GB share x ~0.65 us = ~37 KB,
    # Little's law), as a tuned kernel would. At 4 the cache path is latency-bound (validate.py check 4).
    window: int = 8
    mac_per_cycle: int = 4 * 16 * 16  # per core: sublanes x array
    array_h: int = 16
    vec_flop_per_cycle: int = 4 * 32 * 4
    softmax_flop_per_elem: int = 112  # LLMCompass softmax vector cost per element (softmax.py)

    @classmethod
    def from_llmcompass(cls, specs, **kw):
        d = specs["device"]
        core = d["compute_chiplet"]["core"]
        sa, vu, io = core["systolic_array"], core["vector_unit"], d["io"]
        f = d["frequency_Hz"]
        hbm = io["memory_channel_active_count"] * io["pin_count_per_channel"] * io["bandwidth_per_pin_bit"] / 8
        return cls(name=specs.get("name", "llmcompass"), cores=d["compute_chiplet"]["core_count"], freq=f,
                   hbm_bw=hbm, gb_bw=io["global_buffer_bandwidth_per_cycle_byte"] * f, gb_MB=io["global_buffer_MB"],
                   mac_per_cycle=core["sublane_count"] * sa["array_width"] * sa["array_height"] * sa["mac_per_cycle"],
                   array_h=sa["array_height"],
                   vec_flop_per_cycle=core["sublane_count"] * vu["vector_width"] * vu["flop_per_cycle"], **kw)

    def with_(self, **kw):
        return replace(self, **kw)

    def block_compute_s(self, model, rows=None):
        """Core time to consume one KV block for one work item (QK, softmax, PV over 16 tokens).
        rows = query rows per item (g for a GQA-grouped kernel, 1 for one query head)."""
        rows = model.g if rows is None else rows
        macs = 2 * rows * BLOCK_TOKENS * model.d_head
        util = min(1.0, rows / self.array_h)
        cycles = macs / (self.mac_per_cycle * util) + rows * BLOCK_TOKENS * self.softmax_flop_per_elem / self.vec_flop_per_cycle
        return cycles / self.freq
