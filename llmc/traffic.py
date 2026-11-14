"""Simulated off-chip (HBM) bytes for LLMCompass matmuls.

LLMCompass reports only latency. Its L2 (global buffer) loop in
Matmul.simulate reuses an operand only between consecutive L2 tiles, so a
smaller global buffer can mean more DRAM re-reads. This module re-runs the
same compile step to get the chosen mapping, then replays that loop counting
bytes instead of cycles. LLMCompass itself is not modified.

simulate_mm / simulate_bmm return (latency_s_without_overhead, dram_bytes)
and reproduce Matmul / BatchedMatmul.compile_and_simulate latencies exactly.
"""
from math import ceil

import common  # noqa: F401
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict

FP16 = data_type_dict["fp16"]


def _tile(i, n_tiles, full, t):
    return full - t * (n_tiles - 1) if i == n_tiles - 1 else t


def l2_replay_bytes(M, N, K, mapping, word):
    """Byte-count mirror of the tile loop in Matmul.simulate."""
    tm, tn, tk = mapping.l2_tile_M, mapping.l2_tile_N, mapping.l2_tile_K
    nm, nn, nk = ceil(M / tm), ceil(N / tn), ceil(K / tk)

    def dims(m, n, k):
        return _tile(m, nm, M, tm), _tile(n, nn, N, tn), _tile(k, nk, K, tk)

    m0, n0, k0 = dims(0, 0, 0)
    total = (m0 * k0 + k0 * n0) * word
    pm = pn = pk = 0
    for m, n, k in Matmul.generate_tile_loops(nm, nn, nk, mapping.l2_loop_order):
        if m == 0 and n == 0 and k == 0:
            continue
        a, b, c = dims(m, n, k)
        if m == pm and k == pk:
            total += c * b * word  # K_N tile
        elif n == pn and k == pk:
            total += a * c * word  # M_K tile
        else:
            total += (a * c + c * b) * word
        if k > 0 and not (m == pm and n == pn):
            total += a * b * word  # partial sums read back
        if not (m == pm and n == pn):
            pa, pb, _ = dims(pm, pn, pk)
            total += pa * pb * word  # previous output tile written
        pm, pn, pk = m, n, k
    la, lb, _ = dims(nm - 1, nn - 1, nk - 1)
    total += la * lb * word  # last output tile written
    return total


def simulate_mm(M, K, N, device, compile_mode="heuristic-GPU"):
    mm = Matmul(FP16)
    mm(Tensor([M, K], FP16), Tensor([K, N], FP16))
    lat = mm.compile_and_simulate(device, compile_mode)
    word = FP16.word_size
    if mm.best_mapping is None:  # M == 1 or N == 1 roofline shortcut
        return lat, (M * K + K * N + M * N) * word
    return lat, l2_replay_bytes(M, N, K, mm.best_mapping, word)


def simulate_bmm(bs, M, K, N, device, compile_mode="heuristic-GPU"):
    """Mirror of BatchedMatmul.compile_and_simulate (min of its two paths)."""
    lat1, bytes1 = simulate_mm(M, K, N, device, compile_mode)
    lat1, bytes1 = lat1 * bs, bytes1 * bs
    lat2, bytes2 = simulate_mm(M, K * bs, N, device, compile_mode)
    extra = (bs - 1) * M * N * FP16.word_size
    lat2 += extra / device.io_module.bandwidth
    bytes2 += extra
    return (lat1, bytes1) if lat1 <= lat2 else (lat2, bytes2)
