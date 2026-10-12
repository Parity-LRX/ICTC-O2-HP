"""Fused FP32 CUDA execution of the r5 O(2) paths and their first VJP.

The source of the CG coefficients and channel routing is the reference r5
operator.  Triton handles only the edge-local arithmetic; PyTorch keeps the
public tuple layout, parameter, and ordinary autograd interface.
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl

from .fast_o2 import PackedFastPathWeightedTensorProductO2
from .runtime_weight import _backward_weight_runtime, weight_metadata
from . import __version__

# Inductor can cache the lowering of a custom op by its operator name. Give
# each release its own internal namespace so upgraded bodies cannot reuse an
# older release's lowering. Public Python API and checkpoint keys stay stable.
_OP_NAMESPACE = "ictc_o2_hp_v" + __version__.replace(".", "_")


_BLOCK = int(os.environ.get("O2_TRITON_BLOCK", "128"))
if _BLOCK not in (64, 128, 256, 512):
    raise ValueError("O2_TRITON_BLOCK must be 64, 128, 256, or 512")
_WEIGHT_TILE_SETTING = os.environ.get("O2_WEIGHT_TILE", "auto")
if _WEIGHT_TILE_SETTING not in ("auto", "0", "32", "64", "128", "256"):
    raise ValueError("O2_WEIGHT_TILE must be auto, 0, 32, 64, 128, or 256")
_WEIGHT_CHANNEL_TILE = int(os.environ.get("O2_WEIGHT_CHANNEL_TILE", "64"))
if _WEIGHT_CHANNEL_TILE not in (0, 16, 32, 64, 128, 256):
    raise ValueError("O2_WEIGHT_CHANNEL_TILE must be 0, 16, 32, 64, 128, or 256")
_WEIGHT_SCHEDULE = os.environ.get("O2_WEIGHT_SCHEDULE", "grouped")
if _WEIGHT_SCHEDULE not in ("runtime", "static", "grouped"):
    raise ValueError("O2_WEIGHT_SCHEDULE must be runtime, static, or grouped")
_RUNTIME_EDGE_TILE = int(os.environ.get("O2_RUNTIME_EDGE_TILE", "16"))
if _RUNTIME_EDGE_TILE not in (16, 32, 64, 128, 256):
    raise ValueError("O2_RUNTIME_EDGE_TILE must be 16, 32, 64, 128, or 256")
_WEIGHT_WARPS = int(os.environ.get("O2_WEIGHT_WARPS", "4"))
if _WEIGHT_WARPS not in (1, 2, 4, 8):
    raise ValueError("O2_WEIGHT_WARPS must be 1, 2, 4, or 8")
_FORWARD_PAIR = os.environ.get("O2_FORWARD_PAIR", "0") == "1"
_FORWARD_SPLIT = os.environ.get("O2_FORWARD_SPLIT", "0") == "1"
_FORWARD_FUSED = os.environ.get("O2_FORWARD_FUSED", "0") == "1"
_FORWARD_DEFERRED_MODE = os.environ.get("O2_FORWARD_DEFERRED", "auto")
if _FORWARD_DEFERRED_MODE not in ("auto", "0", "1"):
    raise ValueError("O2_FORWARD_DEFERRED must be auto, 0, or 1")
_FORWARD_WARPS = int(os.environ.get("O2_FORWARD_WARPS", "4"))
if _FORWARD_WARPS not in (1, 2, 4, 8):
    raise ValueError("O2_FORWARD_WARPS must be 1, 2, 4, or 8")
_LOW_ORDER_FUSED_MODE = os.environ.get("O2_LOW_ORDER_FUSED", "auto")
if _LOW_ORDER_FUSED_MODE not in ("auto", "0", "1"):
    raise ValueError("O2_LOW_ORDER_FUSED must be auto, 0, or 1")
_LOW_ORDER_FORWARD_BLOCK = int(os.environ.get("O2_LOW_ORDER_FORWARD_BLOCK", "256"))
if _LOW_ORDER_FORWARD_BLOCK not in (64, 128, 256, 512):
    raise ValueError("O2_LOW_ORDER_FORWARD_BLOCK must be 64,128,256,or512")
_BACKWARD_INPUT_FUSED_MODE = os.environ.get("O2_BACKWARD_INPUT_FUSED", "auto")
if _BACKWARD_INPUT_FUSED_MODE not in ("auto", "0", "1"):
    raise ValueError("O2_BACKWARD_INPUT_FUSED must be auto, 0, or 1")


def _use_backward_input_fused(keys: tuple, full: bool) -> bool:
    frequency_count = len(keys) - (2 if full else 1)
    return (_BACKWARD_INPUT_FUSED_MODE == "1" or (
        _BACKWARD_INPUT_FUSED_MODE == "auto"
        and (frequency_count <= 3 or (full and frequency_count <= 6))))


def _use_low_order_fused(keys: tuple, full: bool) -> bool:
    frequency_count = len(keys) - (2 if full else 1)
    # The execution strategy depends on the representation, never its row
    # count. All positive N use the same kernels and fixed per-CTA tile sizes.
    return (_LOW_ORDER_FUSED_MODE != "0" and frequency_count <= 3
            and not (_FORWARD_PAIR or _FORWARD_SPLIT or _FORWARD_FUSED))


def _use_forward_deferred(keys: tuple, full: bool) -> bool:
    frequency_count = len(keys) - (2 if full else 1)
    if frequency_count < 1:
        return False
    if _FORWARD_DEFERRED_MODE == "1":
        return True
    return (
        _FORWARD_DEFERRED_MODE == "auto"
        and ((full and frequency_count >= 4)
             or (not full and keys[0][1] >= 64 and frequency_count >= 4))
        and not (_FORWARD_PAIR or _FORWARD_SPLIT or _FORWARD_FUSED)
    )


def _weight_tile_for(keys: tuple, full: bool) -> int:
    if _WEIGHT_TILE_SETTING != "auto":
        return int(_WEIGHT_TILE_SETTING)
    frequency_count = len(keys) - (2 if full else 1)
    return 64 if full and frequency_count >= 4 else 0


@triton.jit
def _read_canonical(
    ptr, edge, channel, mask,
    KEY: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, FULL: tl.constexpr,
):
    start: tl.constexpr = KEYS.value[KEY][0]
    multiplicity: tl.constexpr = KEYS.value[KEY][1]
    dim: tl.constexpr = KEYS.value[KEY][2]
    is_frequency: tl.constexpr = KEYS.value[KEY][3]
    layout: tl.constexpr = KEYS.value[KEY][4]
    if layout == 0:
        base = edge * WIDTH + start + channel * dim
        component_stride: tl.constexpr = 1
    elif layout == 1:
        base = edge * WIDTH + start + channel
        component_stride: tl.constexpr = multiplicity
    elif is_frequency:
        base = edge * WIDTH + start + (channel // C) * dim * C + channel % C
        component_stride: tl.constexpr = C
    else:
        base = edge * WIDTH + start + channel
        component_stride: tl.constexpr = 1
    raw0 = tl.load(ptr + base, mask=mask, other=0)
    if dim == 1:
        return raw0, raw0 * 0.0
    raw1 = tl.load(ptr + base + component_stride, mask=mask, other=0)
    if FULL and is_frequency:
        odd = channel >= C
        return tl.where(odd, -raw1, raw0), tl.where(odd, raw0, raw1)
    return raw0, raw1


@triton.jit
def _write_raw(
    ptr, edge, channel, value0, value1, mask,
    KEY: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, FULL: tl.constexpr,
):
    start: tl.constexpr = KEYS.value[KEY][0]
    multiplicity: tl.constexpr = KEYS.value[KEY][1]
    dim: tl.constexpr = KEYS.value[KEY][2]
    is_frequency: tl.constexpr = KEYS.value[KEY][3]
    layout: tl.constexpr = KEYS.value[KEY][4]
    if layout == 0:
        base = edge * WIDTH + start + channel * dim
        component_stride: tl.constexpr = 1
    elif layout == 1:
        base = edge * WIDTH + start + channel
        component_stride: tl.constexpr = multiplicity
    elif is_frequency:
        base = edge * WIDTH + start + (channel // C) * dim * C + channel % C
        component_stride: tl.constexpr = C
    else:
        base = edge * WIDTH + start + channel
        component_stride: tl.constexpr = 1
    if dim == 1:
        tl.store(ptr + base, value0, mask=mask)
    elif FULL and is_frequency:
        odd = channel >= C
        tl.store(ptr + base, tl.where(odd, value1, value0), mask=mask)
        tl.store(ptr + base + component_stride,
                 tl.where(odd, -value0, value1), mask=mask)
    else:
        tl.store(ptr + base, value0, mask=mask)
        tl.store(ptr + base + component_stride, value1, mask=mask)


@triton.jit
def _forward_path(
    left, right, weight, edge, channel, mask,
    WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    OUT_KEY: tl.constexpr, P: tl.constexpr, FULL: tl.constexpr,
):
    key1: tl.constexpr = PATHS.value[OUT_KEY][P][0]
    key2: tl.constexpr = PATHS.value[OUT_KEY][P][1]
    weight_start: tl.constexpr = PATHS.value[OUT_KEY][P][3]
    a0, a1 = _read_canonical(
        left, edge, channel % KEYS.value[key1][1], mask,
        key1, WIDTH, C, KEYS, FULL,
    )
    b0, b1 = _read_canonical(
        right, edge, channel % KEYS.value[key2][1], mask,
        key2, WIDTH, C, KEYS, FULL,
    )
    path0 = tl.full(channel.shape, 0.0, tl.float32)
    path1 = tl.full(channel.shape, 0.0, tl.float32)
    for t in tl.static_range(len(PATHS.value[OUT_KEY][P][4])):
        i = PATHS.value[OUT_KEY][P][4][t][0]
        j = PATHS.value[OUT_KEY][P][4][t][1]
        k = PATHS.value[OUT_KEY][P][4][t][2]
        coefficient = PATHS.value[OUT_KEY][P][4][t][3]
        ai = tl.where(i == 0, a0, a1)
        bj = tl.where(j == 0, b0, b1)
        term = coefficient * ai * bj
        path0 = path0 + tl.where(k == 0, term, 0.0)
        path1 = path1 + tl.where(k == 1, term, 0.0)
    w = tl.load(weight + weight_start + channel, mask=mask, other=0)
    return w * path0, w * path1


@triton.jit
def _forward_key(
    left, right, weight, output,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    OUT_KEY: tl.constexpr, FULL: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    out_mul: tl.constexpr = KEYS.value[OUT_KEY][1]
    mask = offsets < E * out_mul
    edge = offsets // out_mul
    channel = offsets % out_mul
    accum0 = tl.full((BLOCK,), 0.0, tl.float32)
    accum1 = tl.full((BLOCK,), 0.0, tl.float32)
    for p in tl.static_range(len(PATHS.value[OUT_KEY])):
        path0, path1 = _forward_path(
            left, right, weight, edge, channel, mask,
            WIDTH, C, KEYS, PATHS, OUT_KEY, p, FULL,
        )
        accum0 = accum0 + path0
        accum1 = accum1 + path1
    _write_raw(
        output, edge, channel, accum0, accum1, mask,
        OUT_KEY, WIDTH, C, KEYS, FULL,
    )


@triton.jit
def _forward_all(
    left, right, weight, output,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    FULL: tl.constexpr, BLOCK: tl.constexpr,
):
    output_key = tl.program_id(1)
    for key in tl.static_range(len(KEYS.value)):
        if output_key == key:
            _forward_key(
                left, right, weight, output,
                E, WIDTH, C, KEYS, PATHS, key, FULL, BLOCK,
            )


@triton.jit
def _read_copy(
    ptr, edge, channel, mask,
    KEY: tl.constexpr, COPY: tl.constexpr,
    WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr,
):
    start: tl.constexpr = KEYS.value[KEY][0]
    multiplicity: tl.constexpr = KEYS.value[KEY][1]
    dim: tl.constexpr = KEYS.value[KEY][2]
    is_frequency: tl.constexpr = KEYS.value[KEY][3]
    layout: tl.constexpr = KEYS.value[KEY][4]
    raw_channel = channel + COPY * C if is_frequency else channel
    if layout == 0:
        base = edge * WIDTH + start + raw_channel * dim
        component_stride: tl.constexpr = 1
    elif layout == 1:
        base = edge * WIDTH + start + raw_channel
        component_stride: tl.constexpr = multiplicity
    elif is_frequency:
        base = edge * WIDTH + start + (raw_channel // C) * dim * C + raw_channel % C
        component_stride: tl.constexpr = C
    else:
        base = edge * WIDTH + start + raw_channel
        component_stride: tl.constexpr = 1
    raw0 = tl.load(ptr + base, mask=mask, other=0)
    if dim == 1:
        return raw0, raw0 * 0.0
    raw1 = tl.load(ptr + base + component_stride, mask=mask, other=0)
    if is_frequency and COPY == 1:
        return -raw1, raw0
    return raw0, raw1


@triton.jit
def _write_copy(
    ptr, edge, channel, value0, value1, mask,
    KEY: tl.constexpr, COPY: tl.constexpr,
    WIDTH: tl.constexpr, C: tl.constexpr, KEYS: tl.constexpr,
):
    start: tl.constexpr = KEYS.value[KEY][0]
    multiplicity: tl.constexpr = KEYS.value[KEY][1]
    layout: tl.constexpr = KEYS.value[KEY][4]
    if layout == 0:
        base = edge * WIDTH + start + (channel + COPY * C) * 2
        component_stride: tl.constexpr = 1
    elif layout == 1:
        base = edge * WIDTH + start + channel + COPY * C
        component_stride: tl.constexpr = multiplicity
    else:
        base = edge * WIDTH + start + COPY * 2 * C + channel
        component_stride: tl.constexpr = C
    if COPY == 1:
        tl.store(ptr + base, value1, mask=mask)
        tl.store(ptr + base + component_stride, -value0, mask=mask)
    else:
        tl.store(ptr + base, value0, mask=mask)
        tl.store(ptr + base + component_stride, value1, mask=mask)


@triton.jit
def _forward_path_pair(
    left, right, weight, edge, channel, mask,
    WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    OUT_KEY: tl.constexpr, P: tl.constexpr,
):
    key1: tl.constexpr = PATHS.value[OUT_KEY][P][0]
    key2: tl.constexpr = PATHS.value[OUT_KEY][P][1]
    weight_start: tl.constexpr = PATHS.value[OUT_KEY][P][3]
    a00, a01 = _read_copy(left, edge, channel, mask, key1, 0, WIDTH, C, KEYS)
    a10, a11 = _read_copy(left, edge, channel, mask, key1, 1, WIDTH, C, KEYS)
    b00, b01 = _read_copy(right, edge, channel, mask, key2, 0, WIDTH, C, KEYS)
    b10, b11 = _read_copy(right, edge, channel, mask, key2, 1, WIDTH, C, KEYS)
    p00 = tl.full(channel.shape, 0.0, tl.float32)
    p01 = tl.full(channel.shape, 0.0, tl.float32)
    p10 = tl.full(channel.shape, 0.0, tl.float32)
    p11 = tl.full(channel.shape, 0.0, tl.float32)
    for t in tl.static_range(len(PATHS.value[OUT_KEY][P][4])):
        i = PATHS.value[OUT_KEY][P][4][t][0]
        j = PATHS.value[OUT_KEY][P][4][t][1]
        k = PATHS.value[OUT_KEY][P][4][t][2]
        coefficient = PATHS.value[OUT_KEY][P][4][t][3]
        even_term = coefficient * tl.where(i == 0, a00, a01) * tl.where(j == 0, b00, b01)
        odd_term = coefficient * tl.where(i == 0, a10, a11) * tl.where(j == 0, b10, b11)
        p00 = p00 + tl.where(k == 0, even_term, 0.0)
        p01 = p01 + tl.where(k == 1, even_term, 0.0)
        p10 = p10 + tl.where(k == 0, odd_term, 0.0)
        p11 = p11 + tl.where(k == 1, odd_term, 0.0)
    w0 = tl.load(weight + weight_start + channel, mask=mask, other=0)
    w1 = tl.load(weight + weight_start + C + channel, mask=mask, other=0)
    return w0 * p00, w0 * p01, w1 * p10, w1 * p11


@triton.jit
def _forward_key_pair(
    left, right, weight, output,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    OUT_KEY: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < E * C
    edge = offsets // C
    channel = offsets % C
    acc00 = tl.full((BLOCK,), 0.0, tl.float32)
    acc01 = tl.full((BLOCK,), 0.0, tl.float32)
    acc10 = tl.full((BLOCK,), 0.0, tl.float32)
    acc11 = tl.full((BLOCK,), 0.0, tl.float32)
    for p in tl.static_range(len(PATHS.value[OUT_KEY])):
        p00, p01, p10, p11 = _forward_path_pair(
            left, right, weight, edge, channel, mask,
            WIDTH, C, KEYS, PATHS, OUT_KEY, p,
        )
        acc00 = acc00 + p00
        acc01 = acc01 + p01
        acc10 = acc10 + p10
        acc11 = acc11 + p11
    _write_copy(output, edge, channel, acc00, acc01, mask, OUT_KEY, 0, WIDTH, C, KEYS)
    _write_copy(output, edge, channel, acc10, acc11, mask, OUT_KEY, 1, WIDTH, C, KEYS)


@triton.jit
def _forward_all_pair(
    left, right, weight, output,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr, BLOCK: tl.constexpr,
):
    output_key = tl.program_id(1)
    for key in tl.static_range(len(KEYS.value)):
        if output_key == key:
            if KEYS.value[key][3]:
                _forward_key_pair(
                    left, right, weight, output,
                    E, WIDTH, C, KEYS, PATHS, key, BLOCK,
                )
            else:
                _forward_key(
                    left, right, weight, output,
                    E, WIDTH, C, KEYS, PATHS, key, True, BLOCK,
                )


@triton.jit
def _forward_all_fused(
    left, right, weight, output,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    FULL: tl.constexpr, BLOCK: tl.constexpr,
):
    for key in tl.static_range(len(KEYS.value)):
        _forward_key(
            left, right, weight, output,
            E, WIDTH, C, KEYS, PATHS, key, FULL, BLOCK,
        )


@triton.jit
def _forward_all_fused_pair(
    left, right, weight, output,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr, BLOCK: tl.constexpr,
):
    for key in tl.static_range(len(KEYS.value)):
        if KEYS.value[key][3]:
            _forward_key_pair(
                left, right, weight, output,
                E, WIDTH, C, KEYS, PATHS, key, BLOCK,
            )
        else:
            _forward_key(
                left, right, weight, output,
                E, WIDTH, C, KEYS, PATHS, key, True, BLOCK,
            )


@triton.jit
def _forward_frequencies_deferred(
    left, right, weight, output,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    FULL: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    multiplicity: tl.constexpr = KEYS.value[2 if FULL else 1][1]
    mask = offsets < E * multiplicity
    edge = offsets // multiplicity
    channel = offsets % multiplicity
    values0 = ()
    values1 = ()
    for key in tl.static_range(2 if FULL else 1, len(KEYS.value)):
        accum0 = tl.full((BLOCK,), 0.0, tl.float32)
        accum1 = tl.full((BLOCK,), 0.0, tl.float32)
        for path in tl.static_range(len(PATHS.value[key])):
            value0, value1 = _forward_path(
                left, right, weight, edge, channel, mask,
                WIDTH, C, KEYS, PATHS, key, path, FULL,
            )
            accum0 += value0
            accum1 += value1
        values0 += (accum0,)
        values1 += (accum1,)
    for index in tl.static_range(len(KEYS.value) - (2 if FULL else 1)):
        _write_raw(
            output, edge, channel, values0[index], values1[index], mask,
            index + (2 if FULL else 1), WIDTH, C, KEYS, FULL,
        )


@triton.jit
def _backward_input_path(
    left, right, weight, grad_output, edge, channel, mask,
    WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    IN_KEY: tl.constexpr, P: tl.constexpr, FULL: tl.constexpr,
):
    key1: tl.constexpr = PATHS.value[IN_KEY][P][0]
    key2: tl.constexpr = PATHS.value[IN_KEY][P][1]
    key3: tl.constexpr = PATHS.value[IN_KEY][P][2]
    weight_start: tl.constexpr = PATHS.value[IN_KEY][P][3]
    input_mul: tl.constexpr = KEYS.value[IN_KEY][1]
    output_mul: tl.constexpr = KEYS.value[key3][1]
    copies: tl.constexpr = 1 + FULL * (input_mul == C) * (output_mul == 2 * C)
    dl0 = tl.full(channel.shape, 0.0, tl.float32)
    dl1 = tl.full(channel.shape, 0.0, tl.float32)
    dr0 = tl.full(channel.shape, 0.0, tl.float32)
    dr1 = tl.full(channel.shape, 0.0, tl.float32)
    for copy in tl.static_range(copies):
        out_channel = channel + copy * C
        active = mask & (out_channel < output_mul)
        a0, a1 = _read_canonical(
            left, edge, out_channel % KEYS.value[key1][1], active,
            key1, WIDTH, C, KEYS, FULL,
        )
        b0, b1 = _read_canonical(
            right, edge, out_channel % KEYS.value[key2][1], active,
            key2, WIDTH, C, KEYS, FULL,
        )
        g0, g1 = _read_canonical(
            grad_output, edge, out_channel, active,
            key3, WIDTH, C, KEYS, FULL,
        )
        w = tl.load(weight + weight_start + out_channel, mask=active, other=0)
        for t in tl.static_range(len(PATHS.value[IN_KEY][P][4])):
            i = PATHS.value[IN_KEY][P][4][t][0]
            j = PATHS.value[IN_KEY][P][4][t][1]
            k = PATHS.value[IN_KEY][P][4][t][2]
            coefficient = PATHS.value[IN_KEY][P][4][t][3]
            g = tl.where(k == 0, g0, g1)
            common = coefficient * w * g
            if key1 == IN_KEY:
                bj = tl.where(j == 0, b0, b1)
                dl0 = dl0 + tl.where(i == 0, common * bj, 0.0)
                dl1 = dl1 + tl.where(i == 1, common * bj, 0.0)
            if key2 == IN_KEY:
                ai = tl.where(i == 0, a0, a1)
                dr0 = dr0 + tl.where(j == 0, common * ai, 0.0)
                dr1 = dr1 + tl.where(j == 1, common * ai, 0.0)
    return dl0, dl1, dr0, dr1


@triton.jit
def _backward_input_key(
    left, right, weight, grad_output, grad_left, grad_right,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    IN_KEY: tl.constexpr, FULL: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    input_mul: tl.constexpr = KEYS.value[IN_KEY][1]
    mask = offsets < E * input_mul
    edge = offsets // input_mul
    channel = offsets % input_mul
    dl0 = tl.full((BLOCK,), 0.0, tl.float32)
    dl1 = tl.full((BLOCK,), 0.0, tl.float32)
    dr0 = tl.full((BLOCK,), 0.0, tl.float32)
    dr1 = tl.full((BLOCK,), 0.0, tl.float32)
    for p in tl.static_range(len(PATHS.value[IN_KEY])):
        path_dl0, path_dl1, path_dr0, path_dr1 = _backward_input_path(
            left, right, weight, grad_output, edge, channel, mask,
            WIDTH, C, KEYS, PATHS, IN_KEY, p, FULL,
        )
        dl0 = dl0 + path_dl0
        dl1 = dl1 + path_dl1
        dr0 = dr0 + path_dr0
        dr1 = dr1 + path_dr1
    _write_raw(
        grad_left, edge, channel, dl0, dl1, mask,
        IN_KEY, WIDTH, C, KEYS, FULL,
    )
    _write_raw(
        grad_right, edge, channel, dr0, dr1, mask,
        IN_KEY, WIDTH, C, KEYS, FULL,
    )


@triton.jit
def _backward_input_all(
    left, right, weight, grad_output, grad_left, grad_right,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    FULL: tl.constexpr, BLOCK: tl.constexpr,
):
    input_key = tl.program_id(1)
    for key in tl.static_range(len(KEYS.value)):
        if input_key == key:
            _backward_input_key(
                left, right, weight, grad_output, grad_left, grad_right,
                E, WIDTH, C, KEYS, PATHS, key, FULL, BLOCK,
            )


@triton.jit
def _backward_input_all_fused(
    left, right, weight, grad_output, grad_left, grad_right,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    FULL: tl.constexpr, BLOCK: tl.constexpr,
):
    # Use the identical per-key adjoint, allowing shared loads to be reused.
    # Each key retains its own multiplicity/mask, including scalar blocks.
    for key in tl.static_range(len(KEYS.value)):
        _backward_input_key(
            left, right, weight, grad_output, grad_left, grad_right,
            E, WIDTH, C, KEYS, PATHS, key, FULL, BLOCK,
        )


@triton.jit
def _backward_weight_path(
    left, right, grad_output, grad_weight, edge, channel,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    P: tl.constexpr, FULL: tl.constexpr, BLOCK_E: tl.constexpr,
):
    key1: tl.constexpr = PATHS.value[P][0]
    key2: tl.constexpr = PATHS.value[P][1]
    key3: tl.constexpr = PATHS.value[P][2]
    weight_start: tl.constexpr = PATHS.value[P][3]
    output_mul: tl.constexpr = KEYS.value[key3][1]
    active = (edge < E) & (channel < output_mul)
    a0, a1 = _read_canonical(
        left, edge, channel % KEYS.value[key1][1], active,
        key1, WIDTH, C, KEYS, FULL,
    )
    b0, b1 = _read_canonical(
        right, edge, channel % KEYS.value[key2][1], active,
        key2, WIDTH, C, KEYS, FULL,
    )
    g0, g1 = _read_canonical(
        grad_output, edge, channel, active,
        key3, WIDTH, C, KEYS, FULL,
    )
    product = tl.full((BLOCK_E,), 0.0, tl.float32)
    for t in tl.static_range(len(PATHS.value[P][4])):
        i = PATHS.value[P][4][t][0]
        j = PATHS.value[P][4][t][1]
        k = PATHS.value[P][4][t][2]
        coefficient = PATHS.value[P][4][t][3]
        ai = tl.where(i == 0, a0, a1)
        bj = tl.where(j == 0, b0, b1)
        g = tl.where(k == 0, g0, g1)
        product = product + coefficient * ai * bj * g
    gradient = tl.sum(product, axis=0)
    tl.store(grad_weight + weight_start + channel, gradient, mask=channel < output_mul)


@triton.jit
def _backward_weight(
    left, right, grad_output, grad_weight,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    KEYS: tl.constexpr, PATHS: tl.constexpr,
    FULL: tl.constexpr, BLOCK_E: tl.constexpr,
):
    channel = tl.program_id(0)
    path_index = tl.program_id(1)
    edge = tl.arange(0, BLOCK_E)
    for p in tl.static_range(len(PATHS.value)):
        if path_index == p:
            _backward_weight_path(
                left, right, grad_output, grad_weight, edge, channel,
                E, WIDTH, C, KEYS, PATHS, p, FULL, BLOCK_E,
            )


@triton.jit
def _backward_weight_tile_path(
    left, right, grad_output, partial, edge, channel,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    WEIGHT_COUNT: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
    P: tl.constexpr, FULL: tl.constexpr, BLOCK_E: tl.constexpr,
):
    key1: tl.constexpr = PATHS.value[P][0]
    key2: tl.constexpr = PATHS.value[P][1]
    key3: tl.constexpr = PATHS.value[P][2]
    weight_start: tl.constexpr = PATHS.value[P][3]
    output_mul: tl.constexpr = KEYS.value[key3][1]
    active = (edge[:, None] < E) & (channel[None, :] < output_mul)
    a0, a1 = _read_canonical(
        left, edge[:, None], channel[None, :] % KEYS.value[key1][1], active,
        key1, WIDTH, C, KEYS, FULL,
    )
    b0, b1 = _read_canonical(
        right, edge[:, None], channel[None, :] % KEYS.value[key2][1], active,
        key2, WIDTH, C, KEYS, FULL,
    )
    g0, g1 = _read_canonical(
        grad_output, edge[:, None], channel[None, :], active,
        key3, WIDTH, C, KEYS, FULL,
    )
    product = tl.full((BLOCK_E, channel.shape[0]), 0.0, tl.float32)
    for t in tl.static_range(len(PATHS.value[P][4])):
        i = PATHS.value[P][4][t][0]
        j = PATHS.value[P][4][t][1]
        k = PATHS.value[P][4][t][2]
        coefficient = PATHS.value[P][4][t][3]
        ai = tl.where(i == 0, a0, a1)
        bj = tl.where(j == 0, b0, b1)
        g = tl.where(k == 0, g0, g1)
        product = product + coefficient * ai * bj * g
    contribution = tl.sum(product, axis=0)
    tl.store(
        partial + tl.program_id(0) * WEIGHT_COUNT + weight_start + channel,
        contribution, mask=channel < output_mul,
    )


@triton.jit
def _backward_weight_tiled(
    left, right, grad_output, partial,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    WEIGHT_COUNT: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
    FULL: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    channel = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    path_index = tl.program_id(1)
    for p in tl.static_range(len(PATHS.value)):
        if path_index == p:
            _backward_weight_tile_path(
                left, right, grad_output, partial, edge, channel,
                E, WIDTH, C, WEIGHT_COUNT, KEYS, PATHS, p, FULL, BLOCK_E,
            )


@triton.jit
def _backward_weight_group_path(
    a0, a1, right, grad_output, partial, edge, channel,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    WEIGHT_COUNT: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
    FIRST_KEY: tl.constexpr, P: tl.constexpr, FULL: tl.constexpr,
    BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    key2: tl.constexpr = PATHS.value[FIRST_KEY][P][1]
    key3: tl.constexpr = PATHS.value[FIRST_KEY][P][2]
    weight_start: tl.constexpr = PATHS.value[FIRST_KEY][P][3]
    output_mul: tl.constexpr = KEYS.value[key3][1]
    active = (edge[:, None] < E) & (channel[None, :] < output_mul)
    b0, b1 = _read_canonical(
        right, edge[:, None], channel[None, :] % KEYS.value[key2][1],
        active, key2, WIDTH, C, KEYS, FULL)
    g0, g1 = _read_canonical(
        grad_output, edge[:, None], channel[None, :], active,
        key3, WIDTH, C, KEYS, FULL)
    product = tl.full((BLOCK_E, BLOCK_C), 0.0, tl.float32)
    for t in tl.static_range(len(PATHS.value[FIRST_KEY][P][4])):
        i = PATHS.value[FIRST_KEY][P][4][t][0]
        j = PATHS.value[FIRST_KEY][P][4][t][1]
        k = PATHS.value[FIRST_KEY][P][4][t][2]
        coefficient = PATHS.value[FIRST_KEY][P][4][t][3]
        ai = tl.where(i == 0, a0, a1)
        bj = tl.where(j == 0, b0, b1)
        g = tl.where(k == 0, g0, g1)
        product = product + coefficient * ai * bj * g
    contribution = tl.sum(product, axis=0)
    tl.store(partial + tl.program_id(0) * WEIGHT_COUNT + weight_start + channel,
             contribution, mask=channel < output_mul)


@triton.jit
def _backward_weight_grouped(
    left, right, grad_output, partial,
    E: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr,
    WEIGHT_COUNT: tl.constexpr, KEYS: tl.constexpr, PATHS: tl.constexpr,
    FULL: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    channel = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    group = tl.program_id(1)
    for first_key in tl.static_range(len(KEYS.value)):
        if group == first_key:
            a0, a1 = _read_canonical(
                left, edge[:, None], channel[None, :] % KEYS.value[first_key][1],
                (edge[:, None] < E), first_key, WIDTH, C, KEYS, FULL)
            for p in tl.static_range(len(PATHS.value[first_key])):
                _backward_weight_group_path(
                    a0, a1, right, grad_output, partial, edge, channel,
                    E, WIDTH, C, WEIGHT_COUNT, KEYS, PATHS, first_key, p, FULL,
                    BLOCK_E, BLOCK_C)


@triton.jit
def _finish_backward_weight(
    partial, grad_weight,
    WEIGHT_COUNT: tl.constexpr, TILES: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_W: tl.constexpr,
):
    lane = tl.arange(0, BLOCK_T)
    weight_index = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    values = tl.full((BLOCK_T, BLOCK_W), 0.0, tl.float32)
    for start in range(0, TILES, BLOCK_T):
        tile = start + lane
        values += tl.load(
            partial + tile[:, None] * WEIGHT_COUNT + weight_index[None, :],
            mask=(tile[:, None] < TILES) & (weight_index[None, :] < WEIGHT_COUNT),
            other=0.0,
        )
    result = tl.sum(values, axis=0)
    tl.store(grad_weight + weight_index, result, mask=weight_index < WEIGHT_COUNT)


@triton.jit
def _finish_backward_weight_chunk(
    partial, reduced,
    WEIGHT_COUNT: tl.constexpr, TILES: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_W: tl.constexpr,
):
    chunk = tl.program_id(1)
    tile = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    weight_index = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    values = tl.load(
        partial + tile[:, None] * WEIGHT_COUNT + weight_index[None, :],
        mask=(tile[:, None] < TILES) & (weight_index[None, :] < WEIGHT_COUNT),
        other=0.0,
    )
    result = tl.sum(values, axis=0)
    tl.store(reduced + chunk * WEIGHT_COUNT + weight_index, result,
             mask=weight_index < WEIGHT_COUNT)


_METADATA: dict[int, tuple[tuple, tuple, int, bool]] = {}


@torch.library.triton_op(_OP_NAMESPACE + "::o2_forward", mutates_args={})
def _o2_forward_op(
    left: torch.Tensor,
    right: torch.Tensor,
    weight: torch.Tensor,
    path_info: torch.Tensor,
    path_coefficients: torch.Tensor,
    metadata_id: int,
) -> torch.Tensor:
    keys, paths, channels, full = _METADATA[metadata_id]
    edge_count, width = left.shape
    output = torch.empty_like(left)
    max_multiplicity = max(key[1] for key in keys)
    paths_by_output = tuple(
        tuple(path for path in paths if path[2] == output_key)
        for output_key in range(len(keys))
    )
    if _use_forward_deferred(keys, full):
        scalar_count = 2 if full else 1
        for key_index in range(scalar_count):
            torch.library.wrap_triton(_forward_key)[
                (triton.cdiv(edge_count * keys[key_index][1], _BLOCK),)
            ](
                left, right, weight, output,
                edge_count, width, channels, keys, paths_by_output,
                key_index, full, _BLOCK, num_warps=_FORWARD_WARPS,
            )
        torch.library.wrap_triton(_forward_frequencies_deferred)[
            (triton.cdiv(edge_count * max_multiplicity, _BLOCK),)
        ](
            left, right, weight, output,
            edge_count, width, channels, keys, paths_by_output,
            full, _BLOCK, num_warps=_FORWARD_WARPS,
        )
    elif _FORWARD_FUSED or _use_low_order_fused(keys, full):
        forward_block = (_LOW_ORDER_FORWARD_BLOCK
                         if _use_low_order_fused(keys, full) else _BLOCK)
        if full and _FORWARD_PAIR:
            torch.library.wrap_triton(_forward_all_fused_pair)[
                (triton.cdiv(edge_count * channels, forward_block),)
            ](
                left, right, weight, output,
                edge_count, width, channels, keys, paths_by_output, forward_block,
                num_warps=_FORWARD_WARPS,
            )
        else:
            torch.library.wrap_triton(_forward_all_fused)[
                (triton.cdiv(edge_count * max_multiplicity, forward_block),)
            ](
                left, right, weight, output,
                edge_count, width, channels, keys, paths_by_output,
                full, forward_block, num_warps=_FORWARD_WARPS,
            )
    elif _FORWARD_SPLIT:
        for key_index, key in enumerate(keys):
            if full and _FORWARD_PAIR and key[3]:
                torch.library.wrap_triton(_forward_key_pair)[
                    (triton.cdiv(edge_count * channels, _BLOCK),)
                ](
                    left, right, weight, output,
                    edge_count, width, channels,
                    keys, paths_by_output, key_index, _BLOCK,
                    num_warps=_FORWARD_WARPS,
                )
            else:
                torch.library.wrap_triton(_forward_key)[
                    (triton.cdiv(edge_count * key[1], _BLOCK),)
                ](
                    left, right, weight, output,
                    edge_count, width, channels,
                    keys, paths_by_output, key_index, full, _BLOCK,
                    num_warps=_FORWARD_WARPS,
                )
    elif full and _FORWARD_PAIR:
        torch.library.wrap_triton(_forward_all_pair)[
            (triton.cdiv(edge_count * channels, _BLOCK), len(keys))
        ](
            left, right, weight, output,
            edge_count, width, channels, keys, paths_by_output, _BLOCK,
            num_warps=_FORWARD_WARPS,
        )
    else:
        torch.library.wrap_triton(_forward_all)[
            (triton.cdiv(edge_count * max_multiplicity, _BLOCK), len(keys))
        ](
            left, right, weight, output,
            edge_count, width, channels, keys, paths_by_output,
            full, _BLOCK, num_warps=_FORWARD_WARPS,
        )
    return output


@torch.library.triton_op(_OP_NAMESPACE + "::o2_backward", mutates_args={})
def _o2_backward_op(
    left: torch.Tensor,
    right: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    path_info: torch.Tensor,
    path_coefficients: torch.Tensor,
    metadata_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys, paths, channels, full = _METADATA[metadata_id]
    edge_count, width = left.shape
    if edge_count == 0:
        return torch.empty_like(left), torch.empty_like(right), torch.zeros_like(weight)
    grad_output = grad_output.contiguous()
    grad_left = torch.empty_like(left)
    grad_right = torch.empty_like(right)
    paths_by_input = tuple(
        tuple(
            path for path in paths if path[0] == input_key or path[1] == input_key
        )
        for input_key in range(len(keys))
    )
    max_multiplicity = max(key[1] for key in keys)
    if _use_backward_input_fused(keys, full):
        torch.library.wrap_triton(_backward_input_all_fused)[
            (triton.cdiv(edge_count * max_multiplicity, _BLOCK),)
        ](left, right, weight, grad_output, grad_left, grad_right,
          edge_count, width, channels, keys, paths_by_input, full, _BLOCK)
    else:
        torch.library.wrap_triton(_backward_input_all)[
            (triton.cdiv(edge_count * max_multiplicity, _BLOCK), len(keys))
        ](left, right, weight, grad_output, grad_left, grad_right,
          edge_count, width, channels, keys, paths_by_input, full, _BLOCK)
    grad_weight = torch.empty_like(weight)
    weight_tile = (_RUNTIME_EDGE_TILE if _WEIGHT_SCHEDULE in ("runtime", "grouped")
                   else _weight_tile_for(keys, full))
    if weight_tile:
        edge_tiles = triton.cdiv(edge_count, weight_tile)
        channel_tile = (_WEIGHT_CHANNEL_TILE or 32) if _WEIGHT_SCHEDULE in ("runtime", "grouped") else (
            _WEIGHT_CHANNEL_TILE or triton.next_power_of_2(max_multiplicity))
        partial = torch.empty(
            (edge_tiles, weight.numel()), device=weight.device, dtype=weight.dtype,
        )
        if _WEIGHT_SCHEDULE == "grouped":
            paths_by_first = tuple(tuple(path for path in paths if path[0] == key)
                                   for key in range(len(keys)))
            torch.library.wrap_triton(_backward_weight_grouped)[
                (edge_tiles, len(keys), triton.cdiv(max_multiplicity, channel_tile))
            ](
                left, right, grad_output, partial,
                edge_count, width, channels, weight.numel(), keys, paths_by_first,
                full, weight_tile, channel_tile, num_warps=_WEIGHT_WARPS,
            )
        elif _WEIGHT_SCHEDULE == "runtime":
            torch.library.wrap_triton(_backward_weight_runtime)[
                (edge_tiles, len(paths), triton.cdiv(max_multiplicity, channel_tile))
            ](
                left, right, grad_output, partial, path_info, path_coefficients,
                edge_count, width, channels, weight.numel(), full, keys[0][4],
                weight_tile, channel_tile, num_warps=_WEIGHT_WARPS,
            )
        else:
            torch.library.wrap_triton(_backward_weight_tiled)[
                (edge_tiles, len(paths), triton.cdiv(max_multiplicity, channel_tile))
            ](
                left, right, grad_output, partial,
                edge_count, width, channels, weight.numel(), keys, paths,
                full, weight_tile, channel_tile,
            )
        # Use the same two-stage FP32 reduction for every positive N. Bounded
        # 1024-row chunks avoid spills at high N without an N-based dispatch.
        chunks = triton.cdiv(edge_tiles, 1024)
        reduced = torch.empty((chunks, weight.numel()),
                              device=weight.device, dtype=weight.dtype)
        torch.library.wrap_triton(_finish_backward_weight_chunk)[
            (triton.cdiv(weight.numel(), 32), chunks)
        ](partial, reduced, weight.numel(), edge_tiles, 1024, 32)
        torch.library.wrap_triton(_finish_backward_weight)[
            (triton.cdiv(weight.numel(), 32),)
        ](reduced, grad_weight, weight.numel(), chunks,
          32, 32)
    else:
        torch.library.wrap_triton(_backward_weight)[(max_multiplicity, len(paths))](
            left, right, grad_output, grad_weight,
            edge_count, width, channels, keys, paths,
            full, triton.next_power_of_2(edge_count), num_warps=8,
        )
    return grad_left, grad_right, grad_weight


def _torch_forward_flat(left, right, weight, metadata_id):
    """Differentiable reference formula used only for higher derivatives."""
    keys, paths, channels, full = _METADATA[metadata_id]
    batch = left.shape[0]

    def split_canonical(value):
        result = []
        for start, multiplicity, dim, is_frequency, layout in keys:
            block = value[:, start:start + multiplicity * dim]
            if layout == 0:
                block = block.reshape(batch, multiplicity, dim)
            elif layout == 1:
                block = block.reshape(batch, dim, multiplicity).transpose(1, 2)
            elif is_frequency:
                block = block.reshape(batch, multiplicity // channels, dim, channels)
                block = block.permute(0, 1, 3, 2).reshape(batch, multiplicity, dim)
            else:
                block = block.reshape(batch, multiplicity, dim)
            if full and is_frequency:
                even, odd = block.split(channels, dim=1)
                block = torch.cat((
                    even, torch.stack((-odd[..., 1], odd[..., 0]), dim=-1),
                ), dim=1)
            result.append(block)
        return result

    first = split_canonical(left)
    second = split_canonical(right)
    output = [None] * len(keys)

    def select_copies(value, multiplicity):
        if value.shape[1] < multiplicity:
            return torch.cat((value, value), dim=1)
        return value[:, :multiplicity]

    for key1, key2, key3, weight_start, terms in paths:
        multiplicity = keys[key3][1]
        a = select_copies(first[key1], multiplicity)
        b = select_copies(second[key2], multiplicity)
        components = [None] * keys[key3][2]
        for i, j, k, coefficient in terms:
            term = coefficient * a[..., i] * b[..., j]
            components[k] = term if components[k] is None else components[k] + term
        value = torch.stack(components, dim=-1)
        value = value * weight[weight_start:weight_start + multiplicity].view(
            1, multiplicity, 1,
        )
        output[key3] = value if output[key3] is None else output[key3] + value

    restored = []
    for index, (_start, multiplicity, dim, is_frequency, layout) in enumerate(keys):
        value = output[index]
        if full and is_frequency:
            even, odd = value.split(channels, dim=1)
            value = torch.cat((
                even, torch.stack((odd[..., 1], -odd[..., 0]), dim=-1),
            ), dim=1)
        if layout == 1:
            value = value.transpose(1, 2)
        elif layout == 2 and is_frequency:
            value = value.reshape(batch, multiplicity // channels, channels, dim)
            value = value.permute(0, 1, 3, 2)
        restored.append(value.reshape(batch, -1))
    return torch.cat(restored, dim=-1)


def _o2_backward_setup_context(ctx, inputs, output):
    left, right, weight, grad_output, _path_info, _path_coefficients, metadata_id = inputs
    ctx.save_for_backward(left, right, weight, grad_output)
    ctx.metadata_id = metadata_id


def _o2_double_backward(ctx, dd_left, dd_right, dd_weight):
    left, right, weight, grad_output = ctx.saved_tensors
    selected = tuple(
        (index, cotangent) for index, cotangent in enumerate(
            (dd_left, dd_right, dd_weight)
        ) if cotangent is not None
    )
    if not selected:
        return None, None, None, None, None, None, None
    with torch.enable_grad():
        inputs = tuple(
            value.detach().requires_grad_(True)
            for value in (left, right, weight, grad_output)
        )
        value = _torch_forward_flat(*inputs[:3], ctx.metadata_id)
        first_grads = torch.autograd.grad(
            value, inputs[:3], inputs[3], create_graph=True,
        )
        second_grads = torch.autograd.grad(
            tuple(first_grads[index] for index, _ in selected), inputs,
            grad_outputs=tuple(cotangent for _, cotangent in selected),
            allow_unused=True,
        )
    return *second_grads, None, None, None


_o2_backward_op.register_autograd(
    _o2_double_backward, setup_context=_o2_backward_setup_context,
)


def _o2_setup_context(ctx, inputs, output):
    left, right, weight, path_info, path_coefficients, metadata_id = inputs
    ctx.save_for_backward(left, right, weight, path_info, path_coefficients)
    ctx.metadata_id = metadata_id


def _o2_autograd_backward(ctx, grad_output):
    left, right, weight, path_info, path_coefficients = ctx.saved_tensors
    return (
        *_o2_backward_op(
            left, right, weight, grad_output,
            path_info, path_coefficients,
            ctx.metadata_id,
        ),
        None, None, None,
    )


_o2_forward_op.register_autograd(
    _o2_autograd_backward, setup_context=_o2_setup_context
)


class TritonPathWeightedTensorProductO2(PackedFastPathWeightedTensorProductO2):
    """Exact r5 path family with fused edge arithmetic and first-order VJP."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        key_index = {key: index for index, key in enumerate(self.keys)}
        keys = []
        offset = 0
        for block in self.blocks:
            keys.append((
                offset, block.multiplicity, block.irrep_dimension,
                int(block.frequency > 0), 0,
            ))
            offset += block.multiplicity * block.irrep_dimension
        self._triton_keys = tuple(keys)
        paths = []
        for path_index, (key1, key2, key3) in enumerate(self.paths):
            terms = tuple(
                (i, j, k, coefficient)
                for k, component in enumerate(self._sparse_cg[path_index])
                for i, j, coefficient in component
            )
            paths.append((
                key_index[key1], key_index[key2], key_index[key3],
                self._weight_offsets[path_index][0], terms,
            ))
        self._triton_paths = tuple(paths)
        info, coefficients = weight_metadata(self._triton_keys, self._triton_paths)
        self.register_buffer("_weight_path_info", info, persistent=False)
        self.register_buffer("_weight_path_coefficients", coefficients, persistent=False)
        self._triton_metadata_id = len(_METADATA)
        _METADATA[self._triton_metadata_id] = (
            self._triton_keys, self._triton_paths,
            self.channels, self.parity == "full",
        )
        self.forward_deferred = _use_forward_deferred(
            self._triton_keys, self.parity == "full"
        )

    def forward(self, left, right):
        if len(left) != len(self.keys) or len(right) != len(self.keys):
            raise ValueError("one tensor per representation block is required")
        if (
            any(value.device.type != "cuda" or value.dtype != torch.float32
                for value in (*left, *right))
            or self.weight.dtype != torch.float32
            or self.weight.device != left[0].device
        ):
            return super().forward(left, right)
        batch_shape = left[0].shape[:-2]
        left_flat = torch.cat(
            [value.reshape(-1, block.multiplicity * block.irrep_dimension)
             for value, block in zip(left, self.blocks)], dim=-1,
        )
        right_flat = torch.cat(
            [value.reshape(-1, block.multiplicity * block.irrep_dimension)
             for value, block in zip(right, self.blocks)], dim=-1,
        )
        output_flat = _o2_forward_op(
            left_flat, right_flat, self.weight,
            self._weight_path_info, self._weight_path_coefficients,
            self._triton_metadata_id,
        )
        result = []
        for block, (start, _mul, _dim, _freq, _layout) in zip(self.blocks, self._triton_keys):
            width = block.multiplicity * block.irrep_dimension
            result.append(
                output_flat[..., start:start + width].reshape(
                    *batch_shape, block.multiplicity, block.irrep_dimension,
                )
            )
        return tuple(result)


class FlatTritonPathWeightedTensorProductO2(TritonPathWeightedTensorProductO2):
    """The original Cartesian r5 map on flat channel-major tensors."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if (left.ndim < 1 or left.shape[-1] != self.representation_dimension
                or right.shape != left.shape):
            raise ValueError("flat O(2) inputs must have the configured width")
        if (left.device.type != "cuda" or right.device != left.device
                or left.dtype != torch.float32 or right.dtype != torch.float32
                or self.weight.dtype != torch.float32
                or self.weight.device != left.device):
            split_left = tuple(
                left[..., start:start + block.multiplicity * block.irrep_dimension]
                .reshape(*left.shape[:-1], block.multiplicity, block.irrep_dimension)
                for block, (start, _mul, _dim, _freq, _layout) in zip(self.blocks, self._triton_keys)
            )
            split_right = tuple(
                right[..., start:start + block.multiplicity * block.irrep_dimension]
                .reshape(*right.shape[:-1], block.multiplicity, block.irrep_dimension)
                for block, (start, _mul, _dim, _freq, _layout) in zip(self.blocks, self._triton_keys)
            )
            return torch.cat(
                [value.flatten(start_dim=-2) for value in super().forward(split_left, split_right)],
                dim=-1,
            )
        output = _o2_forward_op(
            left.reshape(-1, self.representation_dimension).contiguous(),
            right.reshape(-1, self.representation_dimension).contiguous(),
            self.weight,
            self._weight_path_info, self._weight_path_coefficients,
            self._triton_metadata_id,
        )
        return output.reshape(left.shape)


class XuLayoutTritonPathWeightedTensorProductO2(TritonPathWeightedTensorProductO2):
    """The r5 map with contiguous channel lanes in each real O(2) irrep.

    Positive-frequency blocks use ``[component, channel]`` in canonical
    mode and ``[copy, component, channel]`` in full-parity mode. This is the
    component-major storage order also used by EquivariantX's O(2) product.
    The full-parity second-copy Cartesian basis still requires a J map when
    comparing physical data with EquivariantX.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = 1 if self.parity == "canonical" else 2
        self._triton_keys = tuple((*key[:4], layout) for key in self._triton_keys)
        _METADATA[self._triton_metadata_id] = (
            self._triton_keys, self._triton_paths,
            self.channels, self.parity == "full",
        )

    def _unpack(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_shape = value.shape[:-1]
        pieces = []
        for block, (start, multiplicity, dim, _frequency, _layout) in zip(
            self.blocks, self._triton_keys
        ):
            part = value[..., start:start + multiplicity * dim]
            if dim == 1:
                part = part.reshape(*batch_shape, multiplicity, 1)
            elif self.parity == "full":
                part = part.reshape(
                    *batch_shape, multiplicity // self.channels, dim, self.channels
                ).transpose(-1, -2).reshape(*batch_shape, multiplicity, dim)
            else:
                part = part.reshape(
                    *batch_shape, dim, multiplicity
                ).transpose(-1, -2)
            pieces.append(part)
        return tuple(pieces)

    def _pack(self, blocks: tuple[torch.Tensor, ...]) -> torch.Tensor:
        pieces = []
        for block, spec in zip(blocks, self.blocks):
            batch_shape = block.shape[:-2]
            width = spec.multiplicity * spec.irrep_dimension
            if spec.irrep_dimension == 1:
                part = block.reshape(*batch_shape, width)
            elif self.parity == "full":
                part = block.reshape(
                    *batch_shape, spec.parity_copies,
                    self.channels, spec.irrep_dimension,
                ).transpose(-1, -2).reshape(*batch_shape, width)
            else:
                part = block.transpose(-1, -2).reshape(*batch_shape, width)
            pieces.append(part)
        return torch.cat(pieces, dim=-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if (left.ndim < 1 or left.shape[-1] != self.representation_dimension
                or right.shape != left.shape):
            raise ValueError("flat O(2) inputs must have the configured width")
        if (left.device.type != "cuda" or right.device != left.device
                or left.dtype != torch.float32 or right.dtype != torch.float32
                or self.weight.dtype != torch.float32
                or self.weight.device != left.device):
            return self._pack(TritonPathWeightedTensorProductO2.forward(
                self, self._unpack(left), self._unpack(right)
            ))
        output = _o2_forward_op(
            left.reshape(-1, self.representation_dimension).contiguous(),
            right.reshape(-1, self.representation_dimension).contiguous(),
            self.weight,
            self._weight_path_info, self._weight_path_coefficients,
            self._triton_metadata_id,
        )
        return output.reshape(left.shape)
