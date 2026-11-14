# Gap card review: rulings on out/prefix_gap.md

I spot-checked every issue against the cited sources and accepted all of them; there are no rejections. I recomputed the B=128/64K points in `prefix_results.csv` with pandas. I also read the cited code in `softmax.py`, `dse.py` (lines 200-210) and `fused_model.py`, plus `prefix_split_flip.csv`, `e1_pinning_bound.csv`, `fused_model.csv` and both novelty files. Where two reviewers raised the same problem, I merged them into one fix.

## ACCEPTED

1. **[wrong] Sec 1 table, row 2 (line 16).** Shared attention core time at L192_G48 is 58.81 ms; the minimum across splits is 58.54 ms. No figure is 57 ms. The shipped split only has the B=32/4K point.
   Fix: change the row to `| B=128, P=64K¹ | 588 → 59 ms (10.0×) | 1173 → 82 GB (14.3×) | 596 → 67 ms (8.9×) |`. Add under the table: "¹ B=128/64K uses the L192_G48 grid point (same SRAM configuration as GA100); the shipped run covers only B=32/4K."

2. **[wrong] Sec 2, "No split flip" bullet (line 30).** At B=8 with P=1K, 4K and 16K, both variants' best split is L64_G67.
   Fix: "**No "split flip".** At every point both variants pick the same or a nearly equivalent best split: 192/48 or 128/62 wherever the spread exceeds 1%, and 64/67 at B=8 with P ≤ 16K, where spreads are ≤ 0.12%. Using the other variant's best split costs at most 1.3%."

3. **[minor] Sec 2 table (line 26-27).** The maxima are 0.810% and 0.711%.
   Fix: "≤ 0.8%" becomes "~0.8% (max 0.81%)"; "16K Unshared ≤ 0.7%" becomes "~0.7% (max 0.71%)".

4. **[overstated] Sec 2, cause bullets and Caveat 1 (lines 32-35, 76).** I confirmed the mechanism in `softmax.py`:
   - `l2_tile_N=N` and `l2_tile_M = l2_size//(N·2)`, with `is_l2_double_buffering=False`.
   - Read, compute and write are summed for each L2 tile.
   - Each tile costs `(ceil(l1_tiles/cores)+1)` waves.
   - A split row adds a `log2(N/l1_tile_N)·reduction` term.

   The softmax DRAM bytes are identical in every split. On the matmul side, prefix_values is constant at 10.53 ms, while prefix_scores goes from 15.07 to 16.32 ms and from 21.51 to 25.77 GB (+20%) on L64_G67. Total attention bytes go from 82.14 to 86.40 GB (+5.2%).

   Fix for line 32: "**Cause: LLMCompass's softmax cost model over the B·g × P score matrix**, not the matmuls and not DRAM traffic."

   Fix for line 33: "...prefix softmax takes 30.5–30.8 ms at 48–62 MB, 47.5 ms at 11 MB and 39.2 ms at 64 KB per core. Its DRAM time is a constant 16.8 ms in every split. The variation is vector-compute wave quantisation: each L2 tile must hold whole 64K-token rows (⌊GB/128 KB⌋ rows) and pays one extra fill wave, with no L2 double-buffering. Below 128 KB per core, the row is split and charged a full extra reduction pass per tile."

   Fix for line 34: "The prefix matmuls vary by at most 8% (only prefix_scores, on the 64 KB split), and simulated DRAM bytes are flat except on the 64 KB / 67 MB split (+5% attention bytes, from prefix_scores re-reads)."

   Add a bullet: "The 128/62 (shared) vs 192/48 (unshared) best-split difference at 64K is a power-of-two boundary effect. A 64K-token prefix row is exactly 128 KB and fits a 128 KB core. A P+S row (131,584 B) does not, so it is split and pays the reduction (39.1 vs 30.5 ms at L128_G62)."

   Replace Caveat 1 with: "Section 2's sensitivity is a cost-model artifact of LLMCompass's Softmax mapping (whole-row L2 tiles sized by global-buffer capacity, no L2 double-buffering, one fill wave per tile, a full reduction pass when a row is split), not a memory effect. Softmax DRAM time is the same 16.8 ms in every split."

5. **[overstated] Sec 2, line 35, and the 'Ours' list (line 73).** Unshared attn_softmax runs 31.27 → 48.31 ms and shared prefix_softmax runs 30.49 → 47.47 ms, so both have the same ~17 ms absolute spread. LLMCompass overlaps nothing, so the unshared softmax is not "hidden".
   Fix for line 35: "Both variants pay the same absolute softmax split penalty (~17 ms at B=128, P=64K). In unshared decode it is diluted by 556 ms of KV streaming (2.9% of the step). Sharing removes that streaming, so the same penalty becomes 25.5% and the softmax becomes over half of attention time."
   In 'Ours', replace "the split sensitivity appearing only after sharing" with "the softmax split penalty becoming a large relative share of the step after sharing (a simulator-heuristic result)".

6. **[overstated] Sec 3 roofline sentence (line 40).** 54.28 × 2.048 / 369.6 = 30.1% of peak is the memory roof. The achieved 20.5% is about 68% of that roof. The time is 58.8 ms against a byte floor of 40.1 ms.
   Fix: "Simulated: shared attention reaches 54 FLOP/B at B=128/64K and runs at 20% of peak, about 70% of its 30% memory-roof bound (59 ms against a 40 ms byte floor). The rest is the unfused softmax's serialised vector work (LLMCompass charges 112 vector FLOPs per element, compute added to IO). Unshared stays at 3.8 FLOP/B (`prefix_roofline.png`)."

7. **[minor] Sec 3 cap (line 39).**
   Fix: "...tends to **d_h/2 = 64 FLOP/B** for large M with fp16 scores (d_h/4 = 32 with fp32 scores)."

8. **[wrong] Sec 4, line 48.** `fused_model.csv` gives the ratio 669.30/743.67 = 0.900 at B=256, and 0.900 at B=128 as well. At B=64, compute_us equals l2_us exactly (11.6199 = 11.6199); `max()` breaks the tie toward compute.
   Fix: "At 64 KB per core the broadcast needs 90% of the compute time at B=128 and B=256, and exactly 100% (a tie) at B=64."

9. **[overstated] Sec 4, line 49.** `designs` holds global-buffer bandwidth at 5120 B/clk on every split. LLMCompass's `dse.py` (lines 200-210) instead scales it as 5120·MB//40.
   Fix: "**At a fixed 7.2 TB/s global-buffer bandwidth the broadcast reaches parity with compute (64 KB, B=64) but never exceeds it**, so time is flat across splits at peak rates. If global-buffer bandwidth scales with capacity, as in LLMCompass's DSE, the 512 KB / 11 MB split instead becomes global-buffer-bound at B ≤ 32 for P=64K."

10. **[overstated] Sec 4 bound shift (line 45).** `layer_time` in `fused_model.py` has no suffix term. The suffix adds B·1 MiB per layer, which is 32.8 µs at B=64.
    Fix: "Keeping scores on-chip makes the shared-prefix attention pass DRAM-bound up to B=32 and **compute-bound from B=64** (B·g ≥ 180, as predicted). Including the 256-token unique suffix, total attention becomes compute-bound only at P=64K (B ≥ 64) or P=16K (B=256)."

11. **[minor] Sec 4 working set (line 46).** Every row has kv_tile=16, and all state is charged to SRAM.
    Append: "m uses the smallest K/V tile (n=16) and keeps all state in per-core SRAM. Holding O/S in a register file, as GPU kernels do, would raise m."

12. **[unsupported] Sec 4, "reverses the LLMCompass-paper conclusion" (line 50).** In `fused_model.csv`, `t_us.nunique()` is 1 for every (B, P), so latency never depends on the split. The novelty report stated this point only as a hypothesis.
    Fix: "Under fusion, prefix sharing removes the reason LLMCompass gives for local SRAM not mattering in decode (IO-boundness), and the per-core working set grows with B·g. At GA100's global-buffer bandwidth this still does not change latency anywhere in our grid. It would bind only if global-buffer bandwidth were lower (see above). The cost appears only through global-buffer bandwidth, which the area model does not charge for."

13. **[minor] Sec 5, line 56.** The CSV value is 3.647%.
    Fix: "3.6% with int4".

14. **[wrong] Sec 5, line 57.** The CSV value is 12.646%. Also, e1() tests only 48 MB and 67 MB. A 48 MB prefix needs 56 MB including the 8 MB reserve, which also fits the 62 MB buffer.
    Fix: "The best case is Qwen2.5-0.5B, int4, 4K prefix: 12.6%, and only on splits with a global buffer of at least 56 MB (64 KB / 67 MB and 128 KB / 62 MB)."

15. **[overstated] Gap box, unfused bullet (line 62).** Over the best split at 64 KB, prefix_scores adds 1.25 ms and ffn_down adds 0.06 ms. The 25.5% figure is (max−min)/min.
    Fix: "...Attention is capped near d_h/2 FLOP/B. In LLMCompass, the softmax over this matrix is the dominant split-dependent cost (about 87–100% of the spread; the same softmax varies in unshared decode, but there it is < 3% of the step). This is a simulator-heuristic result: a U-shape with the worst split up to 25.5% slower than the best step at a 64K prefix."

16. **[overstated] Gap box, prior-work line (line 65).** The addendum and novelty report record that TyphoonMLA runs on Ascend and has a roofline there, and that MoSKA is a hardware venue (CAL).
    Fix: "The bound shift under fusion is known on GPUs and on an Ascend NPU (Hydragen, MoSKA, TyphoonMLA). Prefix-aware tiling under on-chip capacity limits is known for GPU shared memory (PAT). Neither the unfused residual nor the equal-area per-core/global-buffer split has been characterised for prefix-shared decode. Decode DSE without prefix sharing: LLMCompass, SMOOTH, MemExplorer."

17. **[overstated] 'Already known' / 'Ours' (lines 69-73).** The PAT smem constraint is already cited for G-A2 in `novelty_report.md`. The d_h/2 cap follows directly from FlashAttention's IO analysis.
    Add to 'Already known':
    - "that on-chip capacity caps the shared-prefix query tile and that splitting it re-reads the prefix KV (PAT, GPU shared memory)"
    - "that materialised scores add Θ(M·P) traffic (FlashAttention IO analysis)"

    Replace 'Ours' with: "applying FlashAttention's IO bound to shared-prefix decode, which gives an unfused cap of d_h/2 = 64 FLOP/B below GA100's ridge (what an operator-at-a-time simulator such as LLMCompass reports); the softmax split penalty becoming a large relative share of the step after sharing (simulator-heuristic); quantifying the per-core query-tile cap and its global-buffer bandwidth cost across equal-area per-core/global splits of an accelerator template."

18. **[minor] `fused_model.py` docstring (lines 10-14).** This fix is to the code, not the card. The model charges `l2 = NKV·qblocks·kv_head`, so it does not multicast on the global-buffer-to-core link, and `gb_MB` is never used.
    Fix the docstring to: "Assuming the qblocks groups stream the prefix in lockstep, each chunk is fetched from DRAM once into the global buffer and read qblocks times by different cores (no on-chip multicast). Global-buffer capacity is assumed sufficient for the staged chunks and does not enter the model."
    In card line 47, change "broadcast from the global buffer" to "re-streamed from the global buffer (once per core group)".

## REJECTED

None. Every cited number and code line matched what I found. The only adjustments were merging duplicate fixes and noting that issue 18 is a code fix rather than a card fix.