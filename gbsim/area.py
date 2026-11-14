"""Area of the bookkeeping a cache-managed global buffer needs, with LLMCompass's own SRAM area
model (cost_model.calc_cache_sram_area_mm2), so cache designs can be compared at equal area.

Tags are per physical KV block (8 KB), not per 128 B line, so the overhead is small:
  tag entry  = tag + valid + LRU + refcount bits (default 32 + 1 + 4 + 8 = 45 bits)
  MSHR entry = block tag + waiter bitmap over the cores (default 32 + 128 bits)

    python -m gbsim.area
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def overhead(gb_MB=48, block_bytes=8192, tag_bits=45, mshr_entries=256, mshr_bits=32 + 128, total_sram_mm2=None):
    sys.path.insert(0, str(ROOT / "llmc"))
    import common
    common.setup()
    from cost_model.cost_model import calc_cache_sram_area_mm2, find_logic_sram_transistor_density
    _, bitcell = find_logic_sram_transistor_density("7nm")
    tag_B = gb_MB * 2**20 // block_bytes * tag_bits / 8
    mshr_B = mshr_entries * mshr_bits / 8
    a_tag = calc_cache_sram_area_mm2(int(tag_B), bitcell)
    a_mshr = calc_cache_sram_area_mm2(int(mshr_B), bitcell)
    a_gb = calc_cache_sram_area_mm2(gb_MB * 2**20, bitcell)
    return dict(gb_MB=gb_MB, tag_KB=tag_B / 1024, mshr_KB=mshr_B / 1024, tag_mm2=a_tag, mshr_mm2=a_mshr,
                gb_mm2=a_gb, overhead_vs_gb=(a_tag + a_mshr) / a_gb)


if __name__ == "__main__":
    for mb in (11, 27, 39, 48, 62, 67):
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in overhead(mb).items()})
