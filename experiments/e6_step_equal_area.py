"""E6 -- Whole decode step at equal area, against a prefix-aware software kernel.

Question: once everything outside attention is included, how much of the attention gain survives,
across the six equal-area SRAM splits (the cores/buffer trade-off at fixed area), and how does it
compare with simply running a prefix-aware kernel (Hydragen/cascade style) on the scratchpad?
Decode step = LLMCompass non-attention time for that split + 32 x gbsim attention layer.
Software baseline: LLMCompass 'shared' variant (prefix-aware kernel), core time without launch overhead.
Output: results/data/e6_step_equal_area.csv   (energy columns are ESTIMATES, see gbsim/energy.py)
"""
import _common as C
import pandas as pd

POL = ("scratchpad", "cache", "cache-mcfill", "oracle-multicast")


def step_table(att):
    d = pd.read_csv(C.ROOT / "llmc" / "out" / "prefix_results.csv")
    d = d[(d.design == "GA100") & (d.suffix == 256)]
    rows = []
    for x in att.itertuples():
        g = d[(d.split_id == x.split_id) & (d.batch == x.batch) & (d.prefix == x.prefix)]
        u, s = g[g.variant == "unshared"], g[g.variant == "shared"]
        if u.empty or s.empty:
            continue
        rest = u[u.group != "attention"].core_s.sum()
        rows.append(dict(split_id=x.split_id, gb_MB=x.gb_MB, batch=x.batch, prefix=x.prefix, policy=x.policy,
                         attn_ms=32e3 * x.time_s, rest_ms=1e3 * rest, step_ms=1e3 * rest + 32e3 * x.time_s,
                         sw_prefix_kernel_step_ms=1e3 * s.core_s.sum(), llmc_unshared_step_ms=1e3 * u.core_s.sum(),
                         attn_energy_mJ=32e-3 * x.E_total_uJ, attn_dram_GB=32 * x.dram_bytes / 1e9))
    t = pd.DataFrame(rows)
    base = t[t.policy == "scratchpad"].set_index(["split_id", "batch", "prefix"]).step_ms
    t["step_speedup_vs_scratchpad"] = [base[(r.split_id, r.batch, r.prefix)] / r.step_ms for r in t.itertuples()]
    t["sw_kernel_speedup_vs_scratchpad"] = [base[(r.split_id, r.batch, r.prefix)] / r.sw_prefix_kernel_step_ms
                                           for r in t.itertuples()]
    return t


def main(workers=None):
    splits = pd.read_csv(C.ROOT / "llmc" / "out" / "splits.csv")
    J = [dict(exp="e6", split_id=sp.split_id, policy=x, batch=b, prefix=p, hw=dict(gb_MB=sp.global_buffer_MB))
         for sp in splits.itertuples() for b in (32, 128) for p in (4096, 16384) for x in POL]
    t = step_table(C.run_jobs(J, workers))
    C.save(t, "e6_step_equal_area.csv")
    g = t[t.split_id == "L192_G48"]
    print(g.pivot_table(index=["batch", "prefix"], columns="policy", values="step_speedup_vs_scratchpad")[list(POL)].round(2).to_string())


if __name__ == "__main__":
    C.cli(main)
