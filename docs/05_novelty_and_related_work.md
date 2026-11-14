# 05 - Novelty and related work

## What is claimed
1. **Characterisation.** On a software-managed scratchpad accelerator, prefix caching gives capacity but no decode bandwidth; a hardware-managed cache recovers part of it and is then capped by global-buffer bandwidth; a multicast bound sits well above. The comparison, its conditions and the roofline view are, in the sources we read, not stated.
2. **Correction.** A cache's collapse with many distinct prefixes is mostly a dispatch-order artifact (E5).
3. **A quantified mechanism.** A time-window multicast buffer, with its window, table size, drift robustness, bandwidth saving and two documented failure regimes (E8-E10).

## What is not claimed
Prefix caching, multicast hardware, merged-miss hardware (MSHR) and prefix-aware software kernels are all prior work. We do not claim hardware beats a prefix-aware software kernel.

## Status of each related idea
| Work / idea | What it does | Relation to this project |
|---|---|---|
| vLLM PagedAttention, SGLang RadixAttention | Prefix caching: shared physical KV pages | The workload we study |
| ChunkAttention (2402.15220), PAT (2511.22333), Hydragen (2402.05099) | GPU L2 partly dedups shared pages; prefix-aware kernels | GPU-side observations; software baselines |
| FlashInfer cascade, FlashForge (2505.17694), TyphoonMLA (2509.21081) | Prefix-aware software kernels (read shared KV once) | The main competitor: matches the oracle at step level (E6) |
| **MeshKV (2609.19207)** | NoC multicast with duplicate suppression in a ~12-cycle window, mainly across heads | **Closest hardware work.** Ours: across requests, keyed by physical block, microsecond windows, evaluated at the global buffer |
| DCO (2512.07312) | Shared cache over scratchpads in LLM accelerators | Cache-vs-scratchpad argument, not KV sharing |
| TPU Ragged Paged Attention (2604.15464) | Per-sequence page DMA into VMEM | Consistent with the scratchpad premise (inferred from its algorithm, not stated) |
| PRESERVE (2501.08192), MoSKA (2511.06010) | Prefetch; software | Adjacent |
| Buffets and related multicast storage idioms | Explicit multicast / merged requests in accelerator memory | General prior art for multicast delivery |

Source: `docs/handoff/prior_work_check.md` (9 Nov 2026; search and reading). We did not re-read every paper; absence of evidence is not proof of absence.

## Honest rating
**Medium-low to medium.** A careful characterisation, a correction of a stated failure, and a quantified mechanism with explicit limits. The nearest hardware work (MeshKV) uses a time window too, and a software kernel matches the speed, so the contribution is transparency, bandwidth provisioning and understanding, not a new phenomenon.

## Risk and the answer to it
*Risk:* a reviewer sees extending MeshKV's window to physical-block tags as trivial. *Answer:* the cross-request case needs block-table indirection and windows long enough to span core drift; we show window length must scale with batch and drift (E8, E9), that dispatch order is part of the design (E8b), and where it fails (oversized prefixes).
