| Parameter | Value |
|---|---|
| Compute cores | 128 |
| Core clock | 1.41 GHz |
| Peak MAC rate | 369.6 TFLOP/s |
| HBM bandwidth / loaded latency | 2.048 TB/s / 500 ns |
| Global buffer capacity | 48 MB |
| Global buffer bandwidth / latency | 7.22 TB/s / 150 ns |
| Outstanding block requests per core | 8 |
| Per-core port limit | none (LLMCompass has no per-core limit) |
| Model (main) | Llama-3-8B: 32 layers, 32 query / 8 KV heads, d_head 128, fp16 KV |
| Model (second) | Qwen2.5-7B: 28 layers, 28 / 4 heads |
| Paged-KV block | 16 tokens x K,V x d_head = 8 KB per KV head |
| Workload | B requests share a prefix of P tokens; 256 private tokens each; ordinary (not prefix-aware) kernel |
