# Gap card: shared-prefix decode attention and on-chip memory at fixed area

Workload: Llama-3-8B (GQA, 32 query heads / 8 KV heads, head dim 128), fp16, decode, B requests sharing a P-token prefix plus 256 unique tokens each.
Hardware: LLMCompass GA100 (1 device), as shipped plus 6 equal-area splits of per-core SRAM vs global buffer (64/67 … 512/11 KB/MB, within 1% of GA100's SRAM area).
Variants:
- **unshared:** prefix-cached serving (vLLM/SGLang). The prefix is stored once but read once per request.
- **shared:** Hydragen/cascade-style. The B requests' queries attend to the prefix in one wide matmul per KV head (M = B·4), followed by a per-request suffix pass and an LSE merge.

Times are LLMCompass "core" times without the A100 launch-overhead constants (those are kept separately in `prefix_results.csv`). Every number below was re-derived from the CSVs by an independent reviewer (`gap_review.md`, 18 corrections applied).

## 1. Sharing is a large win in the simulator (day-one gate: PASS)

| Point | Attention (unshared → shared) | Attention DRAM bytes | Whole decode step |
|---|---|---|---|
| B=32, P=4K | 9.70 → 1.61 ms (6.0×) | 19.4 → 2.8 GB (6.9×) | 17.2 → 9.2 ms (1.88×) |
| B=128, P=64K¹ | 588 → 59 ms (10.0×) | 1173 → 82 GB (14.3×) | 596 → 67 ms (8.9×) |

¹ B=128/64K uses the L192_G48 grid point, which has the same SRAM configuration as GA100; the shipped-design run covers only B=32/4K.

Unshared attention is 56% of the step at B=32/4K, so the effect matters end to end. With the A100 launch-overhead constants included, the shared variant's extra kernels cut the B=32/4K attention gain from 6.0× to 1.76×.

## 2. The equal-area SRAM split is NOT the lever: GA100 sits on a plateau

**Simulated, operator-at-a-time.** Spread of the decode step time across the 6 splits (`prefix_split_flip.csv`):

| Prefix | Unshared | Shared |
|---|---|---|
| 1K–4K | ≤ 0.7% | ~0.8% (max 0.81%) |
| 16K | ~0.7% (max 0.71%) | 0.1–4.3% |
| 64K | 2.2–2.9% | 6.6–25.5% (B=8 / 32 / 128) |

- **No split flip.** At every point both variants pick the same or a nearly equivalent best split: 192/48 or 128/62 wherever the spread exceeds 1%, and 64/67 at B=8 with P ≤ 16K, where spreads are ≤ 0.12%. Using the other variant's best split costs at most 1.3%.
- **The 64K spread is a softmax cost-model artifact, not a memory effect.** At B=128/64K, prefix softmax takes 30.5–30.8 ms at 48–62 MB, 47.5 ms at 11 MB and 39.2 ms at 64 KB per core, yet its DRAM time is a constant 16.8 ms in every split.
  - LLMCompass's Softmax holds whole 64K-token rows per L2 tile (⌊GB/128 KB⌋ rows), with no L2 double-buffering and one extra fill wave per tile. Below 128 KB per core it splits each row and charges a full extra reduction pass.
  - Both variants pay the same absolute penalty (~17 ms). In unshared decode it is diluted by 556 ms of KV streaming (2.9% of the step). Sharing removes the streaming, so the same penalty becomes 25.5%.
  - The prefix matmuls vary by at most 8% (only prefix_scores, on the 64 KB split). Simulated DRAM bytes are flat, except +5% attention bytes on 64 KB/67 MB from prefix_scores re-reads.

**Analytical, fused attention** (`fused_model.py`, peak rates, suffix included). No equal-area split beats GA100 at any (B, P), whether global-buffer bandwidth is held at GA100's 7.2 TB/s or scaled with capacity as LLMCompass's own DSE does (`dse.py`: 5120·MB/40, anchored at GA100). Only the extremes lose:
- 64 KB per core becomes global-buffer-bound at large B with fixed bandwidth: +3% / +8% at P=16K for B=128 / 256, and +4.6% at P=64K for B=64.
- 512 KB/11 MB becomes global-buffer-bound with scaled bandwidth: +23–29% at P=16K and +9–24% at P=64K for B ≤ 64.

So the split defines a **safe window** (roughly 128–384 KB per core), and GA100 is inside it.

## 3. After sharing, the bottleneck in an operator-at-a-time accelerator is the materialised score matrix

When attention runs operator by operator (LLMCompass, and accelerators without attention fusion), the B·g × P scores make four DRAM round trips: scores write, softmax read and write, values read. Per KV head the intensity is 4MdP / (4Pd + 8MP), which tends to **d_h/2 = 64 FLOP/B** for large M with fp16 scores (d_h/4 = 32 with fp32 scores). That is below GA100's ridge of 180.
- Simulated: shared attention reaches 54 FLOP/B at B=128/64K and runs at 20% of peak. That is about 70% of its 30% memory-roof bound (59 ms against a 40 ms byte floor); the rest is the unfused softmax's serialised vector work. Unshared stays at 3.8 FLOP/B (`prefix_roofline.png`).
- So on an operator-at-a-time accelerator, prefix sharing never reaches the compute-bound regime that GPU and NPU kernels report (Hydragen, MoSKA, TyphoonMLA). It stalls on the memory roof at ≤ d_h/2 FLOP/B because of score traffic.

## 4. Fused tiling is the lever, and it is bounded by per-core SRAM and global-buffer bandwidth

- **Bound shift.** Keeping scores on-chip makes the shared-prefix pass DRAM-bound up to B=32 and compute-bound from B=64 (B·g ≥ 180). Including the 256-token unique suffix, total attention becomes compute-bound only at P=64K (B ≥ 64) or P=16K (B=256).
- **Size of the gain** (`fused_vs_unfused.csv`, B ∈ {8, 32, 128}, P ∈ {4K, 16K, 64K}, attention per step):
  - fused bound against the unfused **roofline bound**: **1.4–3.7×**;
  - fused bound against the **simulated** unfused time: 1.6–5.0×;
  - fused against unshared: 6–49×.
- **Per-core SRAM caps the query block.** A fused tile holds Q, the fp32 output accumulator, the running max/sum, an fp32 score tile and a K/V tile. That allows m = 58 / 136 / 214 / 292 / 448 / 604 query rows at 64 / 128 / 192 / 256 / 384 / 512 KB per core.
  - m uses the smallest K/V tile (n=16) and keeps all state in per-core SRAM. Holding O/S in a register file, as GPU kernels do, would raise m.
  - When B·g > m, each KV head's prefix is re-streamed from the global buffer once per core group, ⌈B·g/m⌉ times.
  - At 64 KB per core with fixed bandwidth and P=64K, that re-streaming plus the suffix traffic exceeds compute time at B=64 (1.05×) and needs 95% of compute time at B=128 and B=256.
- **Requirement direction.** Under fusion, prefix sharing removes the reason LLMCompass gives for local SRAM not mattering in decode (IO-boundness), and the per-core working set grows with B·g, whereas ordinary decode needs about g·d per core.
  - At GA100's global-buffer bandwidth this still changes latency nowhere inside the safe window.
  - The cost shows up only through global-buffer bandwidth, which the area model does not charge for.

## 5. Pinning the shared prefix on-chip across steps (E1): small

From `e1_pinning_bound.csv` (B=32, 8 MB kept for tiles):
- For Llama-3-8B the prefix never fits, even at 1K (128 MB).
- For a 1B model it fits only at 1K, saving 1.2% of step traffic with fp16 weights and 3.6% with int4.
- The best case is Qwen2.5-0.5B, int4, 4K prefix: 12.6%, and only on splits with ≥ 56 MB of global buffer (64/67 and 128/62).

## The gap, stated honestly

> **On an accelerator that executes attention operator-at-a-time, the dominant bottleneck after prefix sharing is the materialised B·g×P score matrix, not KV bandwidth.**
> - Shared-prefix decode attention stalls on the memory roof at ≤ d_h/2 = 64 FLOP/B, below GA100's ridge of 180. In LLMCompass, the softmax over this matrix accounts for most of the split-dependent cost (a simulator-heuristic U-shape, with the worst split up to 25.5% slower than the best at a 64K prefix).
> - Re-sizing the equal-area SRAM split does not help: GA100 already sits on the plateau.
> - The lever is the **tiling strategy**: fused shared-prefix tiles, worth 1.4–3.7× on attention bound-to-bound. Fused tiles bring two sizing constraints: per-core SRAM must hold a query block that grows with B·g, and global-buffer bandwidth must cover ⌈B·g/m⌉ prefix re-streams.
>
> What is already known and what is not:
> - The bound shift under fusion is known on GPUs and on an Ascend NPU (Hydragen, MoSKA, TyphoonMLA).
> - Prefix-aware tiling under on-chip capacity limits is known for GPU shared memory (PAT).
> - Neither the unfused residual nor the per-core/global-buffer sizing of fused shared-prefix tiles at equal area has been characterised for accelerators. The decode DSE work (LLMCompass, SMOOTH, MemExplorer) does not consider prefix sharing.

**Novelty: low-to-medium; fine for a course project if framed as "we characterise the accelerator-side consequences of a known software technique".**

Already known, not ours:
- sharing raises arithmetic intensity (Hydragen, MoSKA, TyphoonMLA);
- fusion removes score traffic, and materialised scores add Θ(M·P) traffic (FlashAttention IO analysis);
- on-chip capacity caps the shared-prefix query tile, and splitting it re-reads the prefix KV (PAT, GPU shared memory).

Ours:
- applying FlashAttention's IO bound to shared-prefix decode, which gives an unfused cap of d_h/2 = 64 FLOP/B below GA100's ridge (the regime an operator-at-a-time simulator such as LLMCompass reports);
- showing that the equal-area SRAM split is not the lever (GA100 is on a plateau; the fused-tile window is about 128–384 KB per core);
- quantifying the per-core query-tile cap and its global-buffer bandwidth cost across equal-area splits of an accelerator template.

**How it maps to the problem statement:** a roofline characterisation of the target operator (§3); evaluation of existing on-chip memory configurations at fixed area across sequence length and batch (§2); the dominant bottleneck per operating point (§§3–4). The natural proposal for Weeks 3–5 (**not done here**) is a fused shared-prefix tiling with per-core SRAM and global-buffer bandwidth sized from §4. Its improvement over GA100 is bounded at 1.4–3.7× on attention. Energy still needs a bytes × pJ model.

**Caveats:**
1. Section 2's simulated sensitivity is a cost-model artifact of LLMCompass's Softmax mapping (whole-row L2 tiles sized by global-buffer capacity, no L2 double-buffering, one fill wave per tile, a full reduction pass when a row is split), not a memory effect. Softmax DRAM time is the same 16.8 ms in every split.
2. Section 4 is analytical, at peak rates, with lockstep re-streaming from the global buffer (no on-chip multicast).
3. LLMCompass has no fused-attention operator, so fused numbers are bounds, not simulations.

Files (in `p07/out/`):
- results: `prefix_results.csv`, `prefix_split_flip.csv`, `prefix_sweep_analysis.txt`
- figures: `prefix_split_curves.png`, `prefix_roofline.png`
- analytical models: `fused_model.csv`, `fused_model.txt`, `fused_vs_unfused.csv`, `e1_pinning_bound.csv`
- reviews: `novelty_report.md`, `novelty_addendum.md`, `gap_review.md`
