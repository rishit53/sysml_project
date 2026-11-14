# Prior-work check for M1 (2026-11-09; web search and reading only)

1. **TPU Ragged Paged Attention** (arXiv 2604.15464, https://arxiv.org/html/2604.15464): the premise holds, by inference.
   - The kernel gathers KV pages from HBM per sequence through the page table, with one DMA per block (§2.3, §2.4.1, §3.3, Algorithm 1).
   - "Prefix", "shared pages" and "dedup" do not appear in the paper. So a page shared by N sequences is DMA'd N times. This follows from the algorithm; the paper does not state it.
   - No TPU/Pallas or Ascend paged-attention kernel with DMA-level dedup was found. That is absence of evidence, not proof.
2. **MeshKV** (arXiv 2609.19207, ICCD 2026, https://arxiv.org/html/2609.19207): it dedups across heads, not requests.
   - It merges KV fetches for the same segment that arrive within a 12-cycle window, and multicasts them through NoC trees ("Mare then suppresses redundant head copies").
   - It has no mechanism for physically shared prefix-cache blocks across requests.
   - Risk: a reviewer may see extending it to tag by physical block as trivial. Answer with the cross-request differences: block-table indirection, longer arrival windows, different cores at different steps.
3. **Software kernels that already remove the redundant reads.** Cite them, and scope M1 to ordinary kernels:
   - Hydragen (2402.05099);
   - FlashInfer cascade;
   - PAT (ASPLOS'26, 2511.22333);
   - FlashForge (2505.17694);
   - TyphoonMLA (ICLR'26, 2509.21081): "shared parts of the KV-cache are read only once from HBM" (MLA only).
4. **Not found:** hardware dedup, coalescing or multicast of prefix-cached KV blocks across requests in an accelerator global buffer or NoC.
   - Adjacent: PRESERVE (2501.08192, prefetch only); MoSKA (2511.06010, software); PIM-CACHE (content-dedup in PIM; secondary source only).
