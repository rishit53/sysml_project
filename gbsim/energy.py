"""Energy per level of the hierarchy: ESTIMATES. Bytes per level (from sim.run) are the hard
numbers; swap these constants for your own (e.g. CACTI / a 7 nm SRAM compiler) if you have them.

  HBM2e access     3.9 pJ/bit  HBM2 figure, O'Connor et al., "Fine-Grained DRAM", MICRO 2017
  global-buffer    0.5 pJ/bit  large on-chip SRAM read/write; estimate scaled from Horowitz,
                               ISSCC 2014 (45 nm: ~1.5 pJ/bit for a 1 MB array) to 7 nm
  NoC transport    0.3 pJ/bit  global buffer -> core over a few mm of wire; estimate
"""
PJ_PER_BIT = {"dram": 3.9, "gb": 0.5, "noc": 0.3}


def attention_energy_J(r, pj=PJ_PER_BIT):
    """r: dict from sim.run (one layer). Returns joules per layer by level and total."""
    e = dict(dram=r["dram_bytes"] * 8 * pj["dram"] * 1e-12,
             gb=(r["gb_read_bytes"] + r["gb_fill_bytes"]) * 8 * pj["gb"] * 1e-12,
             noc=r["port_bytes"] * 8 * pj["noc"] * 1e-12)
    e["total"] = sum(e.values())
    return e
