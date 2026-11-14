# Gap card (M1): on a scratchpad accelerator, prefix caching saves capacity but no decode bandwidth. A cache recovers part of it; coalescing/multicast is the missing mechanism.

*Revised after an independent check (13 issues applied). The qualitative finding survived; the quantitative cache-path speedups were cut by 10–25%.*

## Setting
- **Workload:** Llama-3-8B decode with vLLM/SGLang prefix caching. B requests' paged-KV block tables point at the same physical prefix blocks (prefix P, plus 256 unique tokens each). The attention kernel is ordinary (not prefix-aware); each work item is one (request, KV head), as in a GQA-grouped kernel.
- **Hardware:** the LLMCompass GA100 template: 128 cores, a 48 MB global buffer, HBM2e at 2.05 TB/s.
- **Compared:**
  - a pure software-managed **scratchpad** with per-core DMA (LLMCompass template; TPU-style staging);
  - a hardware-managed **cache** global buffer (GPU-L2-like);
  - the scratchpad running a **prefix-aware kernel** (Hydragen/cascade, project A);
  - coalescing/multicast, argued analytically.

## Evidence
### 1. Reuse is temporal; the conditions are policy and order, not just capacity
Trace model: `studies/m1_prefix_cache.py` and `m1b_prefix_cache_jitter.py`. Single-layer attention, 8 KB (KV-head, 16-token) blocks, 128 cores interleaving work items, fully associative cache.
- **Ideal reduction in attention DRAM bytes:** 3.3–43× (B = 8–128, P = 1K–16K).
- **Head-major order, plain LRU, with progress drift** (drift tested only for this order and policy). Concurrent readers keep the reuse distance short:
  - with ±10–50% per-core drift, an 11–27 MB cache gets most of the dedup, but the residual can still be large. At B=32, P=16K, drift ±50%, 11 MB: 3.2× reduction against 21.7× ideal, i.e. 6.8× more DRAM bytes than ideal.
  - with no drift, even 2 MB is ideal; with ≥ ±10% drift, 2 MB recovers only 21–65% of the dedup at P=16K.
- **Request-major order** (vLLM-like dispatch) caps dedup at 16 requests per wave: 13× at 16K against 43× ideal.
  - Shared-priority LRU restores full dedup only once the per-layer prefix fits: 27 MB at 4K, 67 MB at 16K.
  - It does **not** restore it on GA100's 48 MB at 16K (per-layer prefix 64 MB).
  - A true pin/bypass policy is untested.

### 2. Decode-step speedups (`m1c_threeway.py`)
Cache-path attention time = max(KV misses / HBM, (KV accesses + score-matrix bytes) / global-buffer bandwidth) + the unfused softmax time, matching the scratchpad baseline's unfused attention. Steps use LLMCompass core time without launch overhead; the non-attention time is identical for all three variants.

Global-buffer bandwidth is taken at three values:
- **3.3 TB/s:** ChunkAttention's measured absolute shared-page throughput on A100 (arXiv 2402.15220, Table 3: fully shared context, MHA, A100-80GB).
- **4.4 TB/s:** a ratio calibration, i.e. their measured 2.16× mean speedup × HBM peak. The ≈ 2× attention speedup it gives is therefore **by construction, not a validation**.
- **7.2 TB/s:** LLMCompass GA100's global-buffer bandwidth.

| B / P | cache + naive kernel (3.3 ; 4.4 ; 7.2 TB/s) | scratchpad + prefix kernel |
|---|---|---|
| 8 / 4K | 1.10× ; 1.14× ; 1.20× | 1.23× |
| 32 / 4K | 1.26× ; 1.40× ; 1.62× | 1.88× |
| 32 / 16K | 1.43× ; 1.73× ; 2.31× | 3.69× |
| 128 / 4K | 1.43× ; 1.73× ; 2.30× | 3.38× |
| 128 / 16K | 1.52× ; 1.93× ; 2.83× | 6.52× |

All speedups are relative to the scratchpad running the naive kernel.
- **Global-buffer bandwidth caps the cache path,** because every access is still served from the global buffer.
- **The cache captures most of the kernel's gain only at small B and P** (B ≤ 32, P ≤ 4K). Elsewhere it captures about half or less.
- **Launch overheads change the picture.** With LLMCompass's per-op overheads included, the prefix kernel's extra ops cut its gain at 32/4K from 1.88× to 1.23×, so the cache path (≈ 1.27×) matches it.
- **Coalescing/multicast** (each block delivered once to all requesting cores) would close most of the remaining gap. The naive kernel's GEMV-shaped compute (M = 4 rows on a 16×16 array) and its unfused softmax would remain. This is analytical only.

## Known vs. new
**Known; cite these.** On GPUs, L2 partly deduplicates physically shared prefix reads:
- ChunkAttention Table 3: measured 2.1–2.3× with a fully shared context, and ≈ 1.4× with 50% shared;
- PAT, ASPLOS'26 (arXiv 2511.22333) §3.2/§8.3: "L2 offers partial reuse … limited by size and bandwidth";
- Hydragen §4.1.

Adjacent work:
- MeshKV (arXiv 2609.19207): NoC multicast with duplicate suppression, mainly across heads.
- DCO (arXiv 2512.07312): a shared cache over scratchpads for LLM accelerators, but not KV sharing.

**Not found in the sources read:**
- (b) on a pure scratchpad running a non-prefix-aware kernel, prefix caching gives capacity savings but no decode-bandwidth savings. Existing scratchpad accelerators and kernels lack a sharing mechanism: e.g. the TPU ragged paged-attention kernel (arXiv 2604.15464) DMAs each sequence's pages separately, per our literature check (not independently verified).
- (c) the conditions:
  - concurrency-driven reuse distance vs core drift;
  - replacement policy (shared-priority LRU fails when the per-layer prefix exceeds the buffer);
  - work-item and wave order;
  - the global-buffer-bandwidth cap that makes coalescing/multicast the next mechanism.

**Novelty: medium-low to medium.** Frame it as "a memory-architecture characterisation of prefix-cached decode: scratchpad vs cache vs coalescing", building on the GPU observations above.

## Caveats
1. **Analytical and trace-driven.** LLMCompass cannot model a cache, so cache-path time is a bandwidth bound.
2. **Simplified trace:**
   - fully associative LRU;
   - jitter is a fixed per-item speed (no resync), and all items start together;
   - no real CTA or split-K scheduling;
   - vLLM's query-head-fastest dispatch reads each KV head from 4 CTAs (4× the on-chip accesses modelled here).
3. **Bandwidth calibration** comes from one A100 MHA measurement.
4. **Tag and coalescing costs** (area, energy) are not modelled; multicast is not simulated.

## Map to the problem statement and to SMOOTH/MLX/PADE-style fixes
- **Roofline and bottleneck:** naive prefix-cached decode is HBM-bound on a scratchpad, and global-buffer-bandwidth-bound on a cache.
- **Existing configurations:** scratchpad (LLMCompass/TPU-like) vs cache (GPU-like).
- **Fix direction for Weeks 3–5** (not done here): a hardware-managed, physical-block-tagged global buffer with coalescing/multicast to cores and a shared-block pin/bypass policy, in the family of SMOOTH's hardware-managed on-chip memory and MeshKV's multicast. Evaluate it against GA100 at fixed area, with energy.
