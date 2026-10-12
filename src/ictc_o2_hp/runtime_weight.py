"""Compact strict-FP32 path-weight reduction using immutable path metadata.

The metadata holds the original sparse Fischer coefficients and routing.
Path identity is a program index; it no longer expands every path into one
large branch-dispatched kernel. Channel and row tiles bound the live state.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def weight_metadata(keys, paths):
    integers, coefficients = [], []
    for k1, k2, k3, weight_start, terms in paths:
        if len(terms) > 4:
            raise ValueError("the O(2) sparse CG path requires more than four terms")
        row = []
        for key in (k1, k2, k3):
            row.extend(keys[key][:4])
        row.append(weight_start)
        padded = (*terms, *((0, 0, 0, 0.0),) * (4 - len(terms)))
        for i, j, k, _ in padded:
            row.extend((i, j, k))
        integers.append(row)
        coefficients.append([term[3] for term in padded])
    return (torch.tensor(integers, dtype=torch.int32),
            torch.tensor(coefficients, dtype=torch.float32))


@triton.jit
def _meta_read(ptr, edge, channel, mask, start, mul, dim, frequency,
               WIDTH: tl.constexpr, C: tl.constexpr,
               FULL: tl.constexpr, LAYOUT: tl.constexpr):
    if LAYOUT == 0:
        base = edge * WIDTH + start + channel * dim
        stride = 1
    elif LAYOUT == 1:
        base = edge * WIDTH + start + channel
        stride = mul
    else:
        base = edge * WIDTH + start + tl.where(
            frequency != 0, (channel // C) * 2 * C + channel % C, channel)
        stride = tl.where(frequency != 0, C, 1)
    raw0 = tl.load(ptr + base, mask=mask, other=0.0)
    raw1 = tl.load(ptr + base + stride, mask=mask & (dim == 2), other=0.0)
    if FULL:
        odd = (frequency != 0) & (channel >= C)
        return tl.where(odd, -raw1, raw0), tl.where(odd, raw0, raw1)
    return raw0, raw1


@triton.jit
def _backward_weight_runtime(
    left, right, grad_output, partial, path_info, path_coefficients,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    WEIGHT_COUNT: tl.constexpr, FULL: tl.constexpr, LAYOUT: tl.constexpr,
    BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    p = tl.program_id(1)
    info = path_info + p * 25
    start1, mul1 = tl.load(info), tl.load(info + 1)
    dim1, freq1 = tl.load(info + 2), tl.load(info + 3)
    start2, mul2 = tl.load(info + 4), tl.load(info + 5)
    dim2, freq2 = tl.load(info + 6), tl.load(info + 7)
    start3, mul3 = tl.load(info + 8), tl.load(info + 9)
    dim3, freq3 = tl.load(info + 10), tl.load(info + 11)
    weight_start = tl.load(info + 12)
    edge = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    channel = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (edge[:, None] < E) & (channel[None, :] < mul3)
    channel1 = tl.where(channel >= mul1, channel - mul1, channel)
    channel2 = tl.where(channel >= mul2, channel - mul2, channel)
    a0, a1 = _meta_read(left, edge[:, None], channel1[None, :], mask,
                        start1, mul1, dim1, freq1, WIDTH, C, FULL, LAYOUT)
    b0, b1 = _meta_read(right, edge[:, None], channel2[None, :], mask,
                        start2, mul2, dim2, freq2, WIDTH, C, FULL, LAYOUT)
    g0, g1 = _meta_read(grad_output, edge[:, None], channel[None, :], mask,
                        start3, mul3, dim3, freq3, WIDTH, C, FULL, LAYOUT)
    product = tl.full((BLOCK_E, BLOCK_C), 0.0, tl.float32)
    for term in tl.static_range(4):
        i = tl.load(info + 13 + term * 3)
        j = tl.load(info + 14 + term * 3)
        k = tl.load(info + 15 + term * 3)
        coefficient = tl.load(path_coefficients + p * 4 + term)
        ai, bj, g = tl.where(i == 0, a0, a1), tl.where(j == 0, b0, b1), tl.where(k == 0, g0, g1)
        product = product + coefficient * ai * bj * g
    contribution = tl.sum(product, axis=0)
    tl.store(partial + tl.program_id(0) * WEIGHT_COUNT + weight_start + channel,
             contribution, mask=channel < mul3)
