"""Llama-3-8B decode step with a prompt prefix shared by all B requests.

Each request = P shared-prefix tokens + S unique-suffix tokens of KV.

- "unshared": prefix-cached serving (vLLM / SGLang RadixAttention). The
  prefix KV is stored once but every request's attention reads it, so it is
  streamed B times per layer per step.
- "shared": prefix-aware attention (Hydragen / FlashInfer cascade). All B
  requests' queries go against the prefix in one wide matmul per KV head,
  then a per-request suffix pass, then an LSE merge of the two partials.

GQA: 32 query heads, 8 KV heads (group g = 4), head dim 128. Stand-ins as in
the MiniCPM study: GeLU for SiLU, LayerNorm for RMSNorm, RoPE omitted. The
LSE merge is modelled as one element-wise op over both partial outputs.
"""
from ops import bmm, mm, vec

WORD = 2
N_LAYERS = 32
HIDDEN = 4096
NQ, NKV, DH = 32, 8, 128
G = NQ // NKV
FFN = 14336
VOCAB = 128256

LAYER_PARAMS = (
    HIDDEN * NQ * DH * 2  # q, o
    + HIDDEN * NKV * DH * 2  # k, v
    + 3 * HIDDEN * FFN  # gate, up, down
)
TOTAL_PARAMS = N_LAYERS * LAYER_PARAMS + 2 * VOCAB * HIDDEN  # untied embeddings
WEIGHT_BYTES = TOTAL_PARAMS * WORD
KV_BYTES_PER_TOKEN = N_LAYERS * NKV * DH * 2 * WORD  # 131,072 (K and V)
VARIANTS = ("unshared", "shared")


def _linears(b):
    return [
        vec("attn_norm", "layernorm", [b, HIDDEN], "norm"),
        mm("qkv_proj", [b, HIDDEN], [HIDDEN, (NQ + 2 * NKV) * DH], "linear"),
        mm("o_proj", [b, NQ * DH], [NQ * DH, HIDDEN], "linear"),
        vec("ffn_norm", "layernorm", [b, HIDDEN], "norm"),
        mm("ffn_gate_up", [b, HIDDEN], [HIDDEN, 2 * FFN], "linear"),
        vec("ffn_act", "gelu", [b, FFN], "act"),
        mm("ffn_down", [b, FFN], [FFN, HIDDEN], "linear"),
    ]


def _per_request_attn(name, b, n):
    """Each request's g query heads against its own n KV tokens, per KV head."""
    return [
        bmm(f"{name}_scores", [b * NKV, G, DH], [b * NKV, DH, n], "attention"),
        vec(f"{name}_softmax", "softmax", [b * NKV, G, n], "attention"),
        bmm(f"{name}_values", [b * NKV, G, n], [b * NKV, n, DH], "attention"),
    ]


def attention(variant, b, p, s):
    if variant == "unshared":
        return _per_request_attn("attn", b, p + s)
    assert variant == "shared", variant
    ops = [
        # all b requests' g heads share one KV head's prefix: M = b * g
        bmm("prefix_scores", [NKV, b * G, DH], [NKV, DH, p], "attention"),
        vec("prefix_softmax", "softmax", [NKV, b * G, p], "attention"),
        bmm("prefix_values", [NKV, b * G, p], [NKV, p, DH], "attention"),
    ]
    if s:
        ops += _per_request_attn("suffix", b, s)
        ops.append(vec("lse_merge", "gelu", [2 * b * NQ * DH], "attention"))
    return ops


def build(variant, b, p, s):
    """Ops of one layer (count = per-layer repeats); LM head returned separately."""
    return _linears(b) + attention(variant, b, p, s)


def lm_head(b):
    return mm("lm_head", [b, HIDDEN], [HIDDEN, VOCAB], "lm_head")


def footprint_bytes(b, p, s):
    # The prefix is stored once in both variants (prefix caching).
    return WEIGHT_BYTES + KV_BYTES_PER_TOKEN * (p + b * s)
