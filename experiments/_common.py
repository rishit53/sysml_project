"""Shared helpers for the experiment scripts: paths, the parallel simulation runner, speedup columns.

Every experiment is a list of simulation jobs (plain dicts). A job names a policy, a workload and
optionally hardware overrides; `job()` runs one gbsim layer simulation and returns a result row.
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from gbsim import policies as P  # noqa: E402
from gbsim import sim  # noqa: E402
from gbsim.config import HW, LLAMA3_8B, QWEN25_7B  # noqa: E402
from gbsim.energy import attention_energy_J  # noqa: E402

DATA = ROOT / "results" / "data"
GA100 = HW()
MODELS = {m.name: m for m in (LLAMA3_8B, QWEN25_7B)}
SPEEDUP = "attn_speedup_vs_scratchpad"


def job(j):
    """Run one simulation. Keys: exp, policy, optional model / hw (HW overrides) / split_id, and the
    sim.layer arguments (batch, prefix, suffix, n_prefixes, order, jitter, skew_s, seed, ...)."""
    j = dict(j)
    exp, name = j.pop("exp"), j.pop("policy")
    model = MODELS[j.pop("model", LLAMA3_8B.name)]
    hw = GA100.with_(**j.pop("hw", {}))
    split = j.pop("split_id", "shipped")
    pol = P.make(name, hw, model, j.get("ways", 16))
    r = sim.layer(hw, pol, model=model, **j)
    r.update(exp=exp, split_id=split, **{f"E_{k}_uJ": v * 1e6 for k, v in attention_energy_J(r).items()})
    r["policy"] = name
    r["max_live"] = getattr(pol, "max_live", 0)
    return r


def _code_hash():
    h = hashlib.sha1()
    for f in sorted((ROOT / "gbsim").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def run_jobs(jobs, workers=None, fresh=False):
    """Run simulations in parallel. Every finished simulation is cached in results/.cache/<hash of the
    gbsim source>.jsonl, so an interrupted run resumes where it stopped and a changed simulator never
    reuses stale results. fresh=True ignores the cache."""
    workers = workers or min(16, os.cpu_count() or 1)
    os.nice(19)
    cache = ROOT / "results" / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    path, done = cache / f"{_code_hash()}.jsonl", {}
    if path.exists() and not fresh:
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["k"]] = rec["r"]
            except (ValueError, KeyError):
                pass                      # truncated last line of an interrupted run
    keys = [json.dumps(j, sort_keys=True) for j in jobs]
    todo = [(k, j) for k, j in zip(keys, jobs) if k not in done]
    print(f"  {len(jobs)} simulations on {workers} worker(s), {len(jobs) - len(todo)} cached", flush=True)
    if todo:
        with ProcessPoolExecutor(workers) as ex, open(path, "a") as f:
            for (k, _), r in zip(todo, ex.map(job, [j for _, j in todo], chunksize=1)):
                f.write(json.dumps({"k": k, "r": r}, default=float) + "\n")
                f.flush()
                done[k] = r
    return pd.DataFrame([done[k] for k in keys])


def add_speedup(t, keys):
    """Attention speedup of every row over the scratchpad row with the same `keys`."""
    base = t[t.policy == "scratchpad"].set_index(keys).time_s
    t[SPEEDUP] = [base.get(tuple(getattr(r, k) for k in keys), float("nan")) / r.time_s for r in t.itertuples()]
    return t


def save(df, name):
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / name, index=False)
    print(f"  wrote results/data/{name} ({len(df)} rows)", flush=True)


def cli(main):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None, help="parallel workers (default: min(16, cpus))")
    main(ap.parse_args().workers)
