"""Path setup for using the unmodified LLMCompass checkout.

LLMCompass imports its own top-level `utils` and opens files relative to its
root (systolic-array look-up tables, SCALE-Sim temp files), so every process
that touches it must chdir there and put it first on sys.path. Import this
module before any LLMCompass import.
"""
import os
import sys
from pathlib import Path

P07 = Path(__file__).resolve().parent
LLMC = Path(os.environ.get("LLMCOMPASS_ROOT", P07.parents[1] / "LLMCompass"))  # default: next to m1/
OUT = P07 / "out"


def setup():
    os.chdir(LLMC)
    for p in (LLMC, P07):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    # SCALE-Sim fallback writes per-pid configs here; missing in a fresh clone.
    (LLMC / "systolic_array_model" / "temp").mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)


MAX_LOAD = 40.0  # refuse to start simulations above this 1-min load average


def be_polite(workers, max_workers=32):
    """Shared-server guard: lowest CPU priority, capped workers, load check.

    Niceness is inherited by the worker processes the pool spawns.
    """
    if os.nice(0) < 19:
        os.nice(19 - os.nice(0))
    load1 = os.getloadavg()[0]
    if load1 > MAX_LOAD:
        sys.exit(f"server load {load1:.1f} > {MAX_LOAD}; not starting simulations")
    return min(workers, max_workers)


if LLMC.exists():  # gbsim-only use (validate, experiments) works without LLMCompass
    setup()
