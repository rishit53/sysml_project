# 02 - Methodology

## 1. System under study

An LLM-inference accelerator modelled on the **LLMCompass GA100 template**: 128 cores, a shared on-chip
**global buffer**, HBM2e. We study **one decode-attention layer** (the operator that dominates KV traffic at
serving batch sizes). Whole-step numbers add LLMCompass's time for everything outside attention (E6).

| Parameter | Value |
|---|---|
| Cores / clock | 128 / 1.41 GHz |
| Peak MAC rate | 369.6 TFLOP/s |
| HBM bandwidth / loaded latency | 2.048 TB/s / 500 ns |
| Global buffer capacity / bandwidth / latency | 48 MB / 7.22 TB/s (read + fill, aggregate) / 150 ns |
| Prefetch depth | 8 outstanding block requests per core (64 KB of the 192 KB per-core SRAM) |
| Model | Llama-3-8B: 32 layers, 32 query / 8 KV heads (GQA group g=4), d_head 128, fp16 KV; second model Qwen2.5-7B (28/4 heads) |
| Paged KV block | 16 tokens x (K and V) x d_head x 2 B = 8 KB per KV head |

(`gbsim/config.py` is the authority; `results/tables/specs.md` is generated from it.)

## 2. Workload

`B` concurrent requests; each has the same **shared prefix of P tokens** (system prompt, RAG context) plus
**256 private tokens** (unless stated). With prefix caching, all requests' block tables point at the same
physical prefix blocks. The attention kernel is **ordinary**: one work item per (request, KV head), as in a
GQA-grouped decode kernel. Work items are handed to free cores in a **dispatch order**:

* `request-major` - all heads of request 0, then request 1... (vLLM-like);
* `head-major` - all requests of head 0, then head 1...;
* `group-major` - all requests of prefix group 0, then group 1... (a sharing-aware scheduler; added in this work).

With `k` distinct prefixes, requests are assigned to prefix groups round-robin.

## 3. The simulator (`gbsim/`)

An **event-driven timing model** of one layer's memory system.

* Each core runs one work item at a time and keeps up to 8 block requests in flight; blocks are consumed in
  order, each taking the time the core needs to compute QK, softmax and PV on 16 tokens.
* **HBM**, the **global buffer** and each **core port** are FIFO bandwidth servers (a request reserves service
  time on arrival), followed by a fixed latency. Contention, queueing and core drift therefore emerge from the
  model.
* A **policy** (`gbsim/policies.py`) decides, for every block request, `hit`, `miss` (fetch from HBM) or `merge`
  (join an in-flight fetch), and how many global-buffer bytes the delivery costs.
* Output per layer: time, bytes at each level, hits/misses/merges, HBM/buffer utilisation, compute floor.

### Designs compared

| Design | Description |
|---|---|
| Scratchpad | software-managed staging (TPU / LLMCompass style): per-core DMA, no reuse |
| Cache | hardware-managed set-associative LRU over physical blocks with miss merging (MSHR); assumes *perfect* merging, so it is an **upper bound** for a conventional cache |
| Cache + shared-pin | as Cache, blocks referenced by more than one request are evicted last |
| Cache + mcfill | as Cache; one buffer read serves all cores merged on a miss (example multicast) |
| **Window multicast (ours)** | as Cache + mcfill, plus a small table of recently read blocks: a request for a block whose buffer read began less than a *window* ago joins that broadcast (no extra buffer read; its core port is still charged) |
| Oracle multicast | **upper bound, not a design**: infinite capacity; each distinct block is read from HBM once and from the buffer once, whatever the timing |

### Drift model (E9, `gbsim.sim.run`)

* `jitter`: each core's compute time is multiplied by a fixed factor drawn uniformly from 1 +/- jitter;
* `skew_s`: each core starts after a random delay in [0, skew];
* `seed`: fixes both draws; results are deterministic for a given seed. Default (0, 0) is the lockstep case
  used by E1-E8 and reproduces the original results exactly (regression test).

## 4. Metrics

* **Attention speedup** over the scratchpad with the same workload, dispatch order and drift (E1-E5, E8, E9).
* **Whole decode-step speedup** (E6): step = LLMCompass non-attention time + 32 x gbsim attention layer, against
  the scratchpad and against LLMCompass's *prefix-aware kernel* ("shared" variant, the software baseline).
* **HBM traffic reduction** (`dram_dedup_x` = accessed bytes / HBM bytes).
* **Utilisation** of HBM, buffer and compute (busy time / total time).
* **Energy** (estimates, `gbsim/energy.py`): 3.9 pJ/bit HBM, 0.5 pJ/bit global buffer, 0.3 pJ/bit buffer-to-core
  network, applied to the simulated bytes. **Bytes per level are the hard numbers; the pJ constants are
  estimates.**
* **Design cost**: peak entries of the recent-multicast table (`max_live`), converted to KB at 8 B per entry
  (a 32-bit block tag, a 16-bit window timer and a valid bit fit in 7 B; 8 B is rounded up). Tag/MSHR area of the
  cache itself was estimated in the hand-off at 0.1-0.5% of buffer area with LLMCompass's SRAM model
  (`python -m gbsim.area`; needs an LLMCompass clone, not regenerated in this repository).

## 5. Roofline (E7)

FLOPs per KV block = 4 x g x 16 x d_head = 32,768 for a block of 8 KB, i.e. **4 FLOP per KV byte**: decode
attention is deep in the memory-bound region. Roofs: HBM (2.048 TB/s), global buffer (7.22 TB/s), the *kernel*
compute ceiling (GQA fills g/array_height = 1/4 of a 16-high array, plus softmax vector work: 75.8 TFLOP/s) and
the peak MAC rate (369.6 TFLOP/s). A point's **limiting resource** is the busiest of HBM, buffer and compute;
if none is more than 80% busy it is labelled *latency/queueing* (a catch-all that also covers too little
parallelism, for example B=8 has only 64 work items for 128 cores).

## 6. Validation (before any gap number is used)

`python -m gbsim.validate` (results in `results/validation/`, summarised in `results/tables/validation.md`):

1. **Scratchpad vs LLMCompass's unshared attention**: HBM KV bytes must match exactly; time within 15%
   (LLMCompass runs the attention unfused, so it also moves the score matrix).
2. **Zero-time lockstep mode vs the earlier trace model**: miss counts must be identical in all cases.
3. *(Removed from the hand-off)* a one-off GPU comparison on a Tesla M60, archived: every measured gain lay
   between "no merging" and "perfect merging", with the real cache reaching a median 55% (17-101%) of gbsim's
   prediction. Consequence: gbsim's cache is an upper bound, so the real cache-to-multicast headroom is likely
   larger.
4. **Sensitivity** to latency and prefetch depth: attention time moves by less than 10% for half latency and
   prefetch 6 / 16; at double latency the cache path becomes latency-bound unless the prefetch depth grows
   (reported, not gated).

Additional tests (`tests/test_regression.py`): the refactored experiments reproduce the original results to 1e-9;
the scratchpad saves no HBM bytes; lockstep is independent of the seed; drift is deterministic; window 0 equals
mcfill; buffer reads fall monotonically with the window; the oracle is an upper bound.

## 7. Reproducibility

Everything is deterministic. `make all` regenerates validation, E1-E9, tables, figures, the PDF report and runs
the tests. The simulator needs only pandas; the report needs reportlab.
