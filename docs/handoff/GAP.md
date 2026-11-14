# Gap M1: prefix-cached decode on a scratchpad accelerator saves capacity, not bandwidth

Status 2026-11-09. Everything below is simulation (gbsim + LLMCompass), with the checks listed under "How far to trust the numbers". Numbers come from `out/x*.csv`.

## Claim

**The setting.** vLLM/SGLang **prefix caching** with an **ordinary (not prefix-aware) decode-attention kernel**. B requests' block tables point at the **same physical prefix blocks**.

**The three-part claim:**
1. **Scratchpad.** On an LLMCompass/TPU-style software-managed scratchpad accelerator, prefix caching saves memory capacity but **no DRAM bandwidth**: every request still DMAs the prefix.
2. **Cache.** A hardware-managed (cache) global buffer recovers part of the bandwidth. It then becomes **bound by global-buffer bandwidth**, and gets **nothing** once several distinct prefixes are interleaved.
3. **Missing mechanism.** Delivering one fetched block to every core that needs it (coalescing/multicast), plus sharing-aware placement and dispatch, would recover the rest. That is the solution space for the team.

## Evidence (GA100 template: 128 cores, 48 MB global buffer at 7.2 TB/s, HBM2e 2.05 TB/s; Llama-3-8B, suffix 256)

### X1. Scratchpad: capacity saved, bandwidth not
- **DRAM traffic is unchanged.** With the scratchpad, attention DRAM bytes equal B × (prefix + suffix) KV exactly (`dram_dedup_x = 1.0` everywhere). Validation check 1 confirms this against LLMCompass.
- **Capacity is saved.** Prefix caching stores (B−1) × P × 128 KiB less KV. For B=32 and P=4K that is 16.6 GB.

### X2. Cache-managed global buffer (`x1_x3_main.csv`, vLLM-like request-major dispatch)
- **Gain:** attention speedup **2.0–3.3×** over scratchpad; DRAM dedup 3.3–15.1×.
- **How the dedup happens.** Almost all of it comes from **merging misses that are in flight at the same time** (cores reading the same block at once), not from capacity hits. So the cache needs concurrency rather than size.
- **Capacity** (`x2_capacity.csv`):
  - 11 MB loses dedup at P=4K: 8.5× against 11.3–15.1× at ≥ 27 MB.
  - Refcount-aware SharedPin equals LRU at ≤ 48 MB. It helps only at 67 MB with P=16K: 13 → 18.9× at B=32, and 13.4 → 27.9× at B=128.

### X3. The cache path is capped by global-buffer bandwidth (`x1_x3_main.csv`, `x3_gb_bandwidth.csv`)
- **Saturation.** At B ≥ 32 the cache path runs the global buffer at **96–99% utilisation** for P ≥ 4K (89–97% at P=1K), while HBM sits at 22–58%. The cache gain stays near the 7.2 : 2.05 bandwidth ratio. At 3.3–4.4 TB/s it falls to 1.8–2.0× (against a scratchpad at the same bandwidth).
- **Headroom.** The oracle multicast bound (each block read once from DRAM and once from the global buffer) reaches **5.3–8.2×** at B ≥ 32 with P ≥ 4K. That is **2.0–2.5× above the cache** at B=32/P=16K, B=128/P=4K and B=128/P=16K.
  - At B=8 there is no headroom: the cache already reaches the bound.
- **Example plug-in.** Multicasting only the merged misses (`cache-mcfill`) gets 3.4–5.4× at B ≥ 32 with P ≥ 4K. It closes most of the gap at P=16K, and little of it at B=128/P=4K, where dedup comes from per-core hits.

### X4. Realistic sharing (`x4_sharing.csv`, `x4_model_qwen.csv`)
- **Longer private suffixes shrink everything.** At suffix 4K: cache 1.5×, oracle 1.8–1.9×.
- **Many prefixes break the cache.** With **16 distinct prefixes**, a cache gets **1.00×** at both B=32 and B=128: the sharers are not scheduled together, and the 16 per-layer prefixes do not fit. The oracle still gets 1.3–4.3×.
  - This is the clearest failure of "just use a cache".
- **A second model.** Qwen2.5-7B (GQA 28/4) behaves the same way: cache 3.0–3.4×, oracle 4.4–6.9×.

### X5. Whole decode step, equal area, energy (`x5_step_equal_area.csv`)

At the GA100 SRAM configuration (split L192_G48), speedup over scratchpad:

| B, P | cache (upper bound) | cache + mcfill | oracle multicast | prefix-aware software kernel |
|---|---|---|---|---|
| 32, 4K | 1.58× | 1.66× | 1.78× | 1.80× |
| 32, 16K | 2.29× | 2.99× | 3.26× | 3.45× |
| 128, 4K | 2.31× | 2.37× | 3.37× | 3.16× |
| 128, 16K | 2.86× | 4.31× | 5.85× | 6.03× |

- **Equal-area splits.** Across the 6 equal-area SRAM splits, step speedups change by at most ~0.14× for the cache and ~0.4× for cache + mcfill, and almost not at all for the oracle. The split is not the lever; the mechanism is. This matches the earlier project-A result.
- **Bookkeeping area.** Tags plus MSHRs for a block-granular cache cost **0.1–0.5%** of global-buffer area (`python -m gbsim.area`).
- **Attention energy per step** (estimates, `gbsim/energy.py`):
  - B=32/4K: 0.76 J scratchpad → 0.17 J cache → 0.11 J oracle;
  - B=128/16K: 11.6 J → 2.5 J → 0.9 J.

## How far to trust the numbers
- **Validation passed** (`python -m gbsim.validate`, `out/validation/`):
  1. Scratchpad vs LLMCompass's unshared attention: DRAM bytes exact; time −6% to −8%, because LLMCompass also moves the score matrix.
  2. Zero-time mode reproduces the earlier trace model's misses exactly (10 cases).
  3. Results move ≤ 3% for prefetch window 6–16 and for half latency. Double latency at the default window moves the cache path by +6% / +10%.
- **"Cache" is an upper bound for a conventional cache.** gbsim assumes perfect MSHR merging. A one-off comparison on a real GPU L2 (Tesla M60, archived, not part of this handoff) found:
  - every measured gain fell between "no merging" and "perfect merging";
  - the real cache achieved a median **55%** (range 17–101%) of gbsim's predicted gain.

  So the real cache-to-multicast headroom is likely **larger** than shown here.
- **Not modelled:**
  - the score matrix (gbsim's naive kernel is fused per item);
  - NoC topology (global-buffer bandwidth is aggregate; a per-core port limit is available as `port_bw`);
  - bank conflicts;
  - real CTA scheduling;
  - launch overheads (step times use LLMCompass core time).
- **Energy constants are estimates; bytes per level are the hard numbers.**

## Prior work (checked 2026-11-09; see `prior_work_check.md`)
- **Known, cite these:**
  - A GPU L2 partly dedups shared prefix reads: ChunkAttention (2402.15220, Table 3), PAT (ASPLOS'26, 2511.22333), Hydragen.
  - **Prefix-aware software kernels** remove the redundant reads on any hardware: Hydragen, FlashInfer cascade, PAT, FlashForge, TyphoonMLA (NPU).
- **Nearest hardware work:** MeshKV (2609.19207) suppresses duplicate KV fetches in the NoC **across heads within a request**, not across requests sharing a prefix.
- **Scratchpad premise:** the TPU Ragged Paged Attention kernel (2604.15464) gathers pages per sequence and never mentions prefix sharing. That it does no dedup is inferred from its algorithm, not stated.
- **Not found:** hardware dedup or multicast of prefix-cached KV across requests in an accelerator global buffer or NoC.

## Known vs new, stated honestly
- **The objection to answer.** The software kernel reaches about the oracle's speedup (X5), so the team must say why hardware. The arguments:
  1. It needs no kernel change: ordinary paged kernels are the common path.
  2. It handles dynamic or partial (radix-tree) sharing that a cascade kernel must be told about.
  3. It works across many concurrent prefixes, where the cache fails (X4).
  4. It avoids the extra kernel launches. The earlier analysis showed launch overhead cut the software kernel's step gain at B=32/4K from 1.88× to 1.23×.
- **New:**
  - on a scratchpad accelerator, prefix caching gives capacity but no bandwidth;
  - a timing-level comparison of scratchpad, cache and multicast on an accelerator template;
  - the global-buffer-bandwidth cap on the cache path;
  - the multi-prefix collapse;
  - the tiny bookkeeping area.
- **Novelty: medium-low to medium.** It is a characterisation plus a mechanism direction, suitable for the course project.

## Solution directions for the team (not done)
- **Multicast/coalescing global buffer.** One read serves all cores requesting a block within a time window, extending `cache-mcfill` beyond in-flight misses.
- **Sharing-aware dispatch or pinning** for many prefixes (the X4 failure).
- **Refcount-aware replacement** sized to the per-layer prefix.

Evaluate each as a `GlobalBuffer` subclass against scratchpad, cache and the oracle (see README), at equal area (`gbsim/area.py`), with energy.
