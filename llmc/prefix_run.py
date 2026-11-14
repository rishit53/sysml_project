"""Shared-prefix decode study: (design x split x variant x B x P x S) -> CSV.

One row per operator per point. Each unique (design, op shape) is simulated
once and cached in out/op_cache_v2.jsonl with raw latency, launch overhead
and simulated DRAM bytes (traffic.py).

    python prefix_run.py --preset day1
    python prefix_run.py --preset sweep
    python prefix_run.py --designs GA100 --splits all --batch 32 --prefix 4096 --suffix 256

Runs at nice 19 with at most 32 workers and refuses to start above load 40.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

CALLER_CWD = Path(os.getcwd())  # common chdirs into LLMCompass on import

import common
import designs
import llama as L
import ops
from ops import op_key
from splits import design_variants

COLUMNS = [
    "design", "split_id", "local_KB", "gb_MB", "variant", "batch", "prefix", "suffix",
    "operator", "group", "count", "latency_s", "core_s", "overhead_s",
    "sim_bytes", "derived_bytes", "flops", "roofline_s", "roofline_shortcut",
]
KEY = ["design", "split_id", "variant", "batch", "prefix", "suffix", "operator"]

PRESETS = {
    "day1": dict(designs=["GA100"], splits="none", points=[(32, 4096, 256)]),
    "sweep": dict(
        designs=["GA100"], splits="all",
        points=[(b, p, 256) for b in (8, 32, 128) for p in (1024, 4096, 16384, 65536)],
    ),
    "designs": dict(
        designs=["generation_system", "prefilling_system"], splits="none",
        points=[(b, p, 256) for b in (8, 32, 128) for p in (1024, 4096, 16384, 65536)],
    ),
}

_SYSTEMS = {}


def _simulate(task):
    key, specs_json, op = task
    dkey = key.split("|", 1)[0]
    if dkey not in _SYSTEMS:
        _SYSTEMS[dkey] = designs.make_system(json.loads(specs_json))
    t0 = time.time()
    raw, ovh, nbytes = ops.simulate_detailed(op, _SYSTEMS[dkey].device)
    return key, raw, ovh, nbytes, time.time() - t0


def load_cache(path):
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                cache[r["key"]] = r
    return cache


def run(design_names, split_sel, points, variants, workers, out_csv, cache_path):
    workers = common.be_polite(workers)
    cache = load_cache(cache_path)
    jobs, rows = [], []
    for d, split_id, specs in design_variants(design_names, split_sel):
        dkey = designs.specs_key(specs)
        local_kb, gb_mb = designs.sram_of(specs)
        cap = designs.capacity_bytes(specs)
        for b, p, s in points:
            for v in variants:
                base = dict(design=d, split_id=split_id, local_KB=local_kb, gb_MB=gb_mb,
                            variant=v, batch=b, prefix=p, suffix=s)
                if L.footprint_bytes(b, p, s) > cap:
                    rows.append(dict(base, operator="INFEASIBLE", group="", count=0,
                                     latency_s=float("nan"), core_s=float("nan"),
                                     overhead_s=float("nan"), sim_bytes=L.footprint_bytes(b, p, s),
                                     derived_bytes=0, flops=0, roofline_s=float("nan"),
                                     roofline_shortcut=0))
                    continue
                layer_ops = [(op, op.count * L.N_LAYERS) for op in L.build(v, b, p, s)]
                layer_ops.append((L.lm_head(b), 1))
                jobs.append((base, dkey, specs, layer_ops))

    tasks = {}
    for _, dkey, specs, layer_ops in jobs:
        specs_json = json.dumps(specs, sort_keys=True)
        for op, _ in layer_ops:
            k = op_key(dkey, op)
            if k not in cache and k not in tasks:
                tasks[k] = (k, specs_json, op)
    print(f"{len(jobs)} points, {len(tasks)} new operator simulations "
          f"({len(cache)} cached), {workers} workers, nice {os.nice(0)}", flush=True)

    if tasks:
        done = 0
        ctx = mp.get_context("spawn")
        with open(cache_path, "a") as fc, ctx.Pool(workers, maxtasksperchild=50) as pool:
            for k, raw, ovh, nbytes, secs in pool.imap_unordered(_simulate, tasks.values()):
                rec = {"key": k, "raw_s": raw, "overhead_s": ovh, "sim_bytes": nbytes,
                       "sim_wall_s": round(secs, 1)}
                cache[k] = rec
                fc.write(json.dumps(rec) + "\n")
                fc.flush()
                done += 1
                print(f"[{done}/{len(tasks)}] {secs:7.1f}s {k}", flush=True)

    systems = {}
    for base, dkey, specs, layer_ops in jobs:
        if dkey not in systems:
            systems[dkey] = designs.make_system(specs)
        dev = systems[dkey].device
        for op, n in layer_ops:
            c = cache[op_key(dkey, op)]
            rows.append(dict(
                base, operator=op.name, group=op.group, count=n,
                latency_s=(c["raw_s"] + c["overhead_s"]) * n,
                core_s=c["raw_s"] * n,
                overhead_s=c["overhead_s"] * n,
                sim_bytes=c["sim_bytes"] * n,
                derived_bytes=ops.offchip_bytes(op) * n,
                flops=ops.flops(op) * n,
                roofline_s=ops.roofline_bound(op, dev) * n,
                roofline_shortcut=int(ops.roofline_shortcut(op)),
            ))
    merge_write(rows, out_csv)
    return rows


def merge_write(rows, out_csv):
    merged = {}
    if out_csv.exists():
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                merged[tuple(str(r[k]) for k in KEY)] = r
    for r in rows:
        merged[tuple(str(r[k]) for k in KEY)] = r
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(merged.values())
    print(f"wrote {len(rows)} rows -> {out_csv} ({len(merged)} total)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS)
    p.add_argument("--designs", nargs="+", choices=list(designs.DESIGNS))
    p.add_argument("--splits", nargs="+", help="'all', 'none' (shipped configs) or split ids")
    p.add_argument("--batch", type=int, nargs="+")
    p.add_argument("--prefix", type=int, nargs="+")
    p.add_argument("--suffix", type=int, nargs="+", default=[256])
    p.add_argument("--variants", nargs="+", default=list(L.VARIANTS), choices=L.VARIANTS)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", default=str(common.OUT / "prefix_results.csv"))
    p.add_argument("--cache", default=str(common.OUT / "op_cache_v2.jsonl"))
    a = p.parse_args(argv)

    cfg = dict(PRESETS[a.preset]) if a.preset else dict(designs=None, splits="none", points=None)
    if a.designs:
        cfg["designs"] = a.designs
    if a.splits:
        cfg["splits"] = a.splits[0] if a.splits[0] in ("all", "none") else a.splits
    if a.batch and a.prefix:
        cfg["points"] = [(b, pp, s) for b in a.batch for pp in a.prefix for s in a.suffix]
    if not (cfg["designs"] and cfg["points"]):
        p.error("give --preset or --designs, --batch and --prefix")
    return run(cfg["designs"], cfg["splits"], cfg["points"], a.variants, a.workers,
               CALLER_CWD / a.out, CALLER_CWD / a.cache)


if __name__ == "__main__":
    main()
