"""Operator descriptions that the blocks emit and the runner simulates.

An Op is plain data (picklable, hashable), so the runner can dedupe identical
simulations across points and send them to worker processes.
"""
import json
from dataclasses import dataclass
from math import prod

WORD = 2  # fp16 bytes

def op_key(dkey, op):
    """Cache key for one simulated op on one design (design hash, kind, shapes)."""
    return f"{dkey}|{op.kind}|{json.dumps(op.shapes)}"


KINDS = ("matmul", "bmm", "softmax", "layernorm", "gelu")
GROUPS = ("linear", "attention", "norm", "act", "lm_head")


@dataclass(frozen=True)
class Op:
    name: str
    kind: str  # one of KINDS
    shapes: tuple  # input shapes, tuple of tuples
    group: str  # one of GROUPS
    count: int = 1  # repeats per layer

    def __post_init__(self):
        assert self.kind in KINDS, self.kind
        assert self.group in GROUPS, self.group
        n_in = 2 if self.kind in ("matmul", "bmm") else 1
        assert len(self.shapes) == n_in, (self.name, self.shapes)


def mm(name, a, b, group, count=1):
    return Op(name, "matmul", (tuple(a), tuple(b)), group, count)


def bmm(name, a, b, group, count=1):
    return Op(name, "bmm", (tuple(a), tuple(b)), group, count)


def vec(name, kind, x, group, count=1):
    return Op(name, kind, (tuple(x),), group, count)


def _mnk(op):
    a, b = op.shapes
    if op.kind == "matmul":
        return 1, prod(a[:-1]), a[-1], b[-1]
    return prod(a[:-2]), a[-2], a[-1], b[-1]


def flops(op):
    if op.kind in ("matmul", "bmm"):
        bs, m, k, n = _mnk(op)
        return 2 * bs * m * k * n
    elems = prod(op.shapes[0])
    # Rough vector FLOPs per element; these ops are memory-bound anyway.
    return elems * {"softmax": 5, "layernorm": 7, "gelu": 8}[op.kind]


def offchip_bytes(op):
    """Operator-at-a-time traffic: every input and output crosses HBM once."""
    if op.kind in ("matmul", "bmm"):
        bs, m, k, n = _mnk(op)
        return bs * (m * k + k * n + m * n) * WORD
    return 2 * prod(op.shapes[0]) * WORD


def roofline_shortcut(op):
    """LLMCompass heuristic modes skip the tiled simulation when M or N is 1."""
    if op.kind not in ("matmul", "bmm"):
        return False
    _, m, _, n = _mnk(op)
    return m == 1 or n == 1


def roofline_bound(op, device):
    cm = device.compute_module
    peak = cm.total_systolic_array_flops if op.kind in ("matmul", "bmm") else cm.total_vector_flops
    return max(flops(op) / peak, offchip_bytes(op) / device.io_module.bandwidth)


def overhead_s(op, device):
    oh = device.compute_module.overhead
    return {"matmul": oh.matmul, "bmm": oh.matmul, "softmax": oh.softmax,
            "layernorm": oh.layernorm, "gelu": oh.gelu}[op.kind]


def simulate_detailed(op, device, compile_mode="heuristic-GPU"):
    """(raw latency without launch overhead, launch overhead, simulated DRAM bytes).

    Matmuls go through traffic.py, which reproduces LLMCompass's latency and
    replays its L2 loop for bytes. Vector ops have no L2 reuse model, so their
    bytes are the derived input + output size.
    """
    import traffic

    if op.kind == "matmul":
        a, b = op.shapes
        raw, nbytes = traffic.simulate_mm(prod(a[:-1]), a[-1], b[-1], device, compile_mode)
    elif op.kind == "bmm":
        bs, m, k, n = _mnk(op)
        raw, nbytes = traffic.simulate_bmm(bs, m, k, n, device, compile_mode)
    else:
        raw = simulate(op, device, compile_mode) - overhead_s(op, device)
        nbytes = offchip_bytes(op)
    return raw, overhead_s(op, device), nbytes


def simulate(op, device, compile_mode="heuristic-GPU"):
    """Simulated latency of one instance, including LLMCompass's launch overhead."""
    from software_model.gelu import GeLU
    from software_model.layernorm import LayerNorm
    from software_model.matmul import BatchedMatmul, Matmul
    from software_model.softmax import Softmax
    from software_model.utils import Tensor, data_type_dict

    fp16 = data_type_dict["fp16"]
    cls, overhead = {
        "matmul": (Matmul, device.compute_module.overhead.matmul),
        "bmm": (BatchedMatmul, device.compute_module.overhead.matmul),
        "softmax": (Softmax, device.compute_module.overhead.softmax),
        "layernorm": (LayerNorm, device.compute_module.overhead.layernorm),
        "gelu": (GeLU, device.compute_module.overhead.gelu),
    }[op.kind]
    obj = cls(fp16)
    obj(*[Tensor(list(s), fp16) for s in op.shapes])
    return obj.compile_and_simulate(device, compile_mode) + overhead
