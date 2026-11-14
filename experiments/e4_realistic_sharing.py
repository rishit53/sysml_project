"""E4 -- Realistic sharing: several prefixes, longer private suffixes, a second model.

Question: do the E1 conclusions survive when the batch has k distinct prefixes (requests assigned
round-robin to prefix groups, vLLM-like dispatch), longer private suffixes, or a different model
(Qwen2.5-7B, GQA 28/4)?
Output: results/data/e4_sharing.csv, results/data/e4_model_qwen.csv
"""
import _common as C
from gbsim.config import QWEN25_7B

POL3 = ("scratchpad", "cache", "oracle-multicast")


def main(workers=None):
    J = [dict(exp="e4", policy=x, batch=b, prefix=4096, suffix=s, n_prefixes=k)
         for b in (32, 128) for k in (1, 4, 16) for s in (256, 1024, 4096) for x in POL3]
    d = C.add_speedup(C.run_jobs(J, workers), ["batch", "suffix", "n_prefixes"])
    C.save(d, "e4_sharing.csv")
    J = [dict(exp="e4", model=QWEN25_7B.name, policy=x, batch=b, prefix=p)
         for b in (32, 128) for p in (4096, 16384) for x in POL3]
    q = C.add_speedup(C.run_jobs(J, workers), ["batch", "prefix"])
    C.save(q, "e4_model_qwen.csv")
    print(d[d.suffix == 256].pivot_table(index=["batch", "n_prefixes"], columns="policy", values=C.SPEEDUP)[list(POL3)].round(2).to_string())


if __name__ == "__main__":
    C.cli(main)
