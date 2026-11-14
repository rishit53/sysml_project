"""Equal-area (per-core SRAM, global buffer) splits for GA100 -> out/splits.csv.

Area is computed with LLMCompass's own calc_cache_sram_area_mm2, called the
same way cost_model.py calls it: once per core for the local buffer, once for
physical_global_buffer_MB. The global buffer is searched in whole MB because
template_to_system does not cast MB * 2**20 to int.
"""
import csv

import common
from cost_model.cost_model import calc_cache_sram_area_mm2, find_logic_sram_transistor_density
from designs import load_specs

LOCAL_KB = [64, 128, 192, 256, 384, 512]
TOL = 0.01


def sram_area_mm2(specs):
    cc = specs["device"]["compute_chiplet"]
    _, bitcell = find_logic_sram_transistor_density(cc["process_node"])
    local = calc_cache_sram_area_mm2(cc["core"]["SRAM_KB"] * 2**10, bitcell)
    gbuf = calc_cache_sram_area_mm2(
        specs["device"]["io"]["physical_global_buffer_MB"] * 2**20, bitcell
    )
    return cc["physical_core_count"] * local + gbuf


def solve(design="GA100", local_kbs=LOCAL_KB):
    base = load_specs(design)
    a0 = sram_area_mm2(base)
    rows = []
    for kb in local_kbs:
        err, mb = min(
            (abs(sram_area_mm2(load_specs(design, kb, mb)) - a0), mb) for mb in range(1, 256)
        )
        area = sram_area_mm2(load_specs(design, kb, mb))
        rel = (area - a0) / a0
        if abs(rel) > TOL:
            print(f"skip {kb} KB: best global buffer {mb} MB is {rel:+.2%} off")
            continue
        rows.append(
            dict(
                split_id=f"L{kb}_G{mb}",
                local_KB=kb,
                global_buffer_MB=mb,
                area_mm2=round(area, 3),
                area_err_pct=round(100 * rel, 3),
                is_baseline=int(kb == 192 and mb == 48),
            )
        )
    return a0, rows


def load(path=common.OUT / "splits.csv"):
    with open(path) as f:
        return list(csv.DictReader(f))


def design_variants(design_names, split_sel):
    """Yield (design, split_id, specs). split 'shipped' = the config as published."""
    for d in design_names:
        if split_sel == "none":
            yield d, "shipped", load_specs(d)
            continue
        assert d == "GA100", "equal-area splits are defined for GA100 only"
        for r in load():
            if split_sel != "all" and r["split_id"] not in split_sel:
                continue
            yield d, r["split_id"], load_specs(d, int(r["local_KB"]), int(r["global_buffer_MB"]))


if __name__ == "__main__":
    a0, rows = solve()
    with open(common.OUT / "splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"GA100 SRAM area A0 = {a0:.3f} mm^2")
    for r in rows:
        print(r)
