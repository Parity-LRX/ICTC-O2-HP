"""Fused uvu forward and VJP without row-by-path-by-u-by-v intermediates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import __version__
from .uvu_o2 import torch_uvu_forward

_NAMESPACE = "ictc_o2_hp_v" + __version__.replace(".", "_") + "_uvu"
_METADATA = []
_STATIC = []
_SPLITS = 16


def _required_gradients(values, declared):
    """Use the active autograd target graph; retain declared roles in tracing."""
    if torch.compiler.is_compiling():
        return tuple(declared)
    from torch.fx.experimental.proxy_tensor import get_proxy_mode
    if (get_proxy_mode() is not None or torch._C._are_functorch_transforms_active()
            or not hasattr(torch._C, "_will_engine_execute_node")):
        return tuple(declared)
    needed = []
    for value, active in zip(values, declared):
        if not active:
            needed.append(False)
            continue
        try:
            edge = torch.autograd.graph.get_gradient_edge(value)
            needed.append(bool(torch._C._will_engine_execute_node(edge.node)))
        except (RuntimeError, AttributeError, AssertionError):
            needed.append(True)
    return tuple(needed)


def register_metadata(metadata, coefficient_values):
    first, second, output, paths, u, v = metadata
    records = []
    for p, (i, j, k) in enumerate(paths):
        a, b, out = first[i], second[j], output[k]
        records.append((a.offset, b.offset, out.offset, a.dim, b.dim, out.dim, p * u * v))
    order1 = sorted(range(len(paths)), key=lambda p: paths[p][0])
    order2 = sorted(range(len(paths)), key=lambda p: paths[p][1])
    for entries, order, role in ((output, range(len(paths)), 2),
                                 (first, order1, 0), (second, order2, 1)):
        cursor = 0
        for index, entry in enumerate(entries):
            begin = cursor
            while cursor < len(paths) and paths[order[cursor]][role] == index:
                cursor += 1
            records.append((entry.offset, entry.dim, begin, cursor, 0, 0, 0))
    identifier = len(_METADATA)
    _METADATA.append(metadata)
    sparse = []
    for p, (i, j, k) in enumerate(paths):
        terms = tuple((a, b, c, float(coefficient_values[p][c * 4 + a * 2 + b]))
                      for c in range(output[k].dim)
                      for a in range(first[i].dim) for b in range(second[j].dim)
                      if abs(coefficient_values[p][c * 4 + a * 2 + b]) > 1e-12)
        sparse.append((i, j, k, terms))
    _STATIC.append((tuple((entry.offset, entry.dim) for entry in first),
                    tuple((entry.offset, entry.dim) for entry in second), tuple(sparse)))
    return (identifier, torch.tensor(records, dtype=torch.int32),
            torch.tensor(order1, dtype=torch.int32), torch.tensor(order2, dtype=torch.int32))


@triton.jit
def _path(info, coefficients, p):
    xoff = tl.load(info + p * 7)
    yoff = tl.load(info + p * 7 + 1)
    outoff = tl.load(info + p * 7 + 2)
    dx = tl.load(info + p * 7 + 3)
    dy = tl.load(info + p * 7 + 4)
    dout = tl.load(info + p * 7 + 5)
    woff = tl.load(info + p * 7 + 6)
    c0 = tl.load(coefficients + p * 8)
    c1 = tl.load(coefficients + p * 8 + 1)
    c2 = tl.load(coefficients + p * 8 + 2)
    c3 = tl.load(coefficients + p * 8 + 3)
    c4 = tl.load(coefficients + p * 8 + 4)
    c5 = tl.load(coefficients + p * 8 + 5)
    c6 = tl.load(coefficients + p * 8 + 6)
    c7 = tl.load(coefficients + p * 8 + 7)
    return xoff, yoff, outoff, dx, dy, dout, woff, c0, c1, c2, c3, c4, c5, c6, c7


@triton.jit
def _forward(x, y, w, info, coefficients, out, N: tl.constexpr,
             WX: tl.constexpr, WY: tl.constexpr, U: tl.constexpr, V: tl.constexpr,
             P: tl.constexpr, WNUM: tl.constexpr, SHARED: tl.constexpr,
             B: tl.constexpr, BV: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    row, u = offsets // U, offsets % U
    mask = row < N
    entry = P + tl.program_id(1)
    outoff = tl.load(info + entry * 7)
    dout = tl.load(info + entry * 7 + 1)
    begin = tl.load(info + entry * 7 + 2)
    end = tl.load(info + entry * 7 + 3)
    v = tl.arange(0, BV)
    acc0, acc1 = tl.full((B,), 0, tl.float32), tl.full((B,), 0, tl.float32)
    for p in range(begin, end):
        xo, yo, oo, dx, dy, do, wo, c0, c1, c2, c3, c4, c5, c6, c7 = _path(info, coefficients, p)
        a0 = tl.load(x + row * WX + xo + u * dx, mask, 0)
        a1 = tl.load(x + row * WX + xo + u * dx + 1, mask & (dx == 2), 0)
        b0 = tl.load(y + row[:, None] * WY + yo + v[None, :] * dy,
                     mask[:, None] & (v[None, :] < V), 0)
        b1 = tl.load(y + row[:, None] * WY + yo + v[None, :] * dy + 1,
                     mask[:, None] & (v[None, :] < V) & (dy == 2), 0)
        weight_offset = wo + u[:, None] * V + v[None, :]
        if not SHARED:
            weight_offset += row[:, None] * WNUM
        weights = tl.load(w + weight_offset, mask[:, None] & (v[None, :] < V), 0)
        r0, r1 = tl.sum(weights * b0, 1), tl.sum(weights * b1, 1)
        acc0 += a0 * (c0 * r0 + c1 * r1) + a1 * (c2 * r0 + c3 * r1)
        acc1 += a0 * (c4 * r0 + c5 * r1) + a1 * (c6 * r0 + c7 * r1)
    tl.store(out + row * WX + outoff + u * dout, acc0, mask)
    tl.store(out + row * WX + outoff + u * dout + 1, acc1, mask & (dout == 2))


@triton.jit
def _input_grad(x, y, w, g, info, coefficients, order, dest, N: tl.constexpr,
                WX: tl.constexpr, WY: tl.constexpr, U: tl.constexpr, V: tl.constexpr,
                P: tl.constexpr, NO: tl.constexpr, NX: tl.constexpr,
                WNUM: tl.constexpr, SHARED: tl.constexpr, FIRST: tl.constexpr,
                B: tl.constexpr, BR: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    if FIRST:
        row, channel = offsets // U, offsets % U
        entry = P + NO + tl.program_id(1)
        width = WX
        reduction = V
    else:
        row, channel = offsets // V, offsets % V
        entry = P + NO + NX + tl.program_id(1)
        width = WY
        reduction = U
    mask = row < N
    offset = tl.load(info + entry * 7)
    dim = tl.load(info + entry * 7 + 1)
    begin = tl.load(info + entry * 7 + 2)
    end = tl.load(info + entry * 7 + 3)
    r = tl.arange(0, BR)
    active = mask[:, None] & (r[None, :] < reduction)
    acc0, acc1 = tl.full((B,), 0, tl.float32), tl.full((B,), 0, tl.float32)
    for q in range(begin, end):
        p = tl.load(order + q)
        xo, yo, oo, dx, dy, do, wo, c0, c1, c2, c3, c4, c5, c6, c7 = _path(info, coefficients, p)
        if FIRST:
            b0 = tl.load(y + row[:, None] * WY + yo + r[None, :] * dy, active, 0)
            b1 = tl.load(y + row[:, None] * WY + yo + r[None, :] * dy + 1, active & (dy == 2), 0)
            wi = wo + channel[:, None] * V + r[None, :]
            if not SHARED:
                wi += row[:, None] * WNUM
            weights = tl.load(w + wi, active, 0)
            b0, b1 = tl.sum(weights * b0, 1), tl.sum(weights * b1, 1)
            g0 = tl.load(g + row * WX + oo + channel * do, mask, 0)
            g1 = tl.load(g + row * WX + oo + channel * do + 1, mask & (do == 2), 0)
            acc0 += g0 * (c0 * b0 + c1 * b1) + g1 * (c4 * b0 + c5 * b1)
            acc1 += g0 * (c2 * b0 + c3 * b1) + g1 * (c6 * b0 + c7 * b1)
        else:
            a0 = tl.load(x + row[:, None] * WX + xo + r[None, :] * dx, active, 0)
            a1 = tl.load(x + row[:, None] * WX + xo + r[None, :] * dx + 1, active & (dx == 2), 0)
            g0 = tl.load(g + row[:, None] * WX + oo + r[None, :] * do, active, 0)
            g1 = tl.load(g + row[:, None] * WX + oo + r[None, :] * do + 1, active & (do == 2), 0)
            wi = wo + r[None, :] * V + channel[:, None]
            if not SHARED:
                wi += row[:, None] * WNUM
            weights = tl.load(w + wi, active, 0)
            b0 = a0 * (c0 * g0 + c4 * g1) + a1 * (c2 * g0 + c6 * g1)
            b1 = a0 * (c1 * g0 + c5 * g1) + a1 * (c3 * g0 + c7 * g1)
            acc0 += tl.sum(weights * b0, 1)
            acc1 += tl.sum(weights * b1, 1)
    tl.store(dest + row * width + offset + channel * dim, acc0, mask)
    tl.store(dest + row * width + offset + channel * dim + 1, acc1, mask & (dim == 2))


@triton.jit
def _weight_shared(x, y, g, info, coefficients, partial, N: tl.constexpr,
                   WX: tl.constexpr, WY: tl.constexpr, U: tl.constexpr, V: tl.constexpr,
                   WNUM: tl.constexpr, SPLITS: tl.constexpr, BUV: tl.constexpr, BN: tl.constexpr):
    p, split = tl.program_id(1), tl.program_id(2)
    uv = tl.program_id(0) * BUV + tl.arange(0, BUV)
    u, v = uv // V, uv % V
    e = tl.arange(0, BN)
    xo, yo, oo, dx, dy, do, wo, c0, c1, c2, c3, c4, c5, c6, c7 = _path(info, coefficients, p)
    acc = tl.full((BUV, BN), 0, tl.float32)
    for start in range(split * BN, N, SPLITS * BN):
        row = start + e
        active = (uv[:, None] < U * V) & (row[None, :] < N)
        a0 = tl.load(x + row[None, :] * WX + xo + u[:, None] * dx, active, 0)
        a1 = tl.load(x + row[None, :] * WX + xo + u[:, None] * dx + 1, active & (dx == 2), 0)
        b0 = tl.load(y + row[None, :] * WY + yo + v[:, None] * dy, active, 0)
        b1 = tl.load(y + row[None, :] * WY + yo + v[:, None] * dy + 1, active & (dy == 2), 0)
        g0 = tl.load(g + row[None, :] * WX + oo + u[:, None] * do, active, 0)
        g1 = tl.load(g + row[None, :] * WX + oo + u[:, None] * do + 1, active & (do == 2), 0)
        r0 = a0 * (c0 * g0 + c4 * g1) + a1 * (c2 * g0 + c6 * g1)
        r1 = a0 * (c1 * g0 + c5 * g1) + a1 * (c3 * g0 + c7 * g1)
        acc += r0 * b0 + r1 * b1
    tl.store(partial + split * WNUM + wo + uv, tl.sum(acc, 1), uv < U * V)


@triton.jit
def _weight_unshared(x, y, g, info, coefficients, dest, N: tl.constexpr,
                     WX: tl.constexpr, WY: tl.constexpr, U: tl.constexpr, V: tl.constexpr,
                     WNUM: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    row, uv = offsets // (U * V), offsets % (U * V)
    u, v = uv // V, uv % V
    active = row < N
    p = tl.program_id(1)
    xo, yo, oo, dx, dy, do, wo, c0, c1, c2, c3, c4, c5, c6, c7 = _path(info, coefficients, p)
    a0 = tl.load(x + row * WX + xo + u * dx, active, 0)
    a1 = tl.load(x + row * WX + xo + u * dx + 1, active & (dx == 2), 0)
    b0 = tl.load(y + row * WY + yo + v * dy, active, 0)
    b1 = tl.load(y + row * WY + yo + v * dy + 1, active & (dy == 2), 0)
    g0 = tl.load(g + row * WX + oo + u * do, active, 0)
    g1 = tl.load(g + row * WX + oo + u * do + 1, active & (do == 2), 0)
    r0 = a0 * (c0 * g0 + c4 * g1) + a1 * (c2 * g0 + c6 * g1)
    r1 = a0 * (c1 * g0 + c5 * g1) + a1 * (c3 * g0 + c7 * g1)
    tl.store(dest + row * WNUM + wo + uv, r0 * b0 + r1 * b1, active)


@triton.jit
def _reduce_weight(partial, dest, WNUM: tl.constexpr, SPLITS: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    split = tl.arange(0, SPLITS)
    value = tl.load(partial + split[:, None] * WNUM + i[None, :], i[None, :] < WNUM, 0)
    tl.store(dest + i, tl.sum(value, 0), i < WNUM)


def _forward_into(left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor,
                info: torch.Tensor, coefficients: torch.Tensor,
                first_order: torch.Tensor, second_order: torch.Tensor,
                metadata_id: int, shared: bool, result: torch.Tensor) -> None:
    first, second, output, paths, u, v = _METADATA[metadata_id]
    rows, wx, wy = left.shape[0], left.shape[1], right.shape[1]
    if rows:
        if v == 1:
            from .uvu_single import _fused_forward
            first_meta, second_meta, static_paths = _STATIC[metadata_id]
            torch.library.wrap_triton(_fused_forward)[(triton.cdiv(rows * u, 128),)](
                left, right, weight, result, rows, wx, wy, u, len(paths) * u, shared,
                first_meta, second_meta, static_paths, 128, triton.next_power_of_2(len(first)),
                num_warps=4, enable_fp_fusion=False,
            )
        elif v <= 8:
            from .uvu_fused import _forward as _fused_forward
            first_meta, second_meta, static_paths = _STATIC[metadata_id]
            torch.library.wrap_triton(_fused_forward)[(triton.cdiv(rows * u, 128),)](
                left, right, weight, result, rows, wx, wy, u, v, len(paths) * u * v, shared,
                first_meta, second_meta, static_paths, 128, triton.next_power_of_2(len(first)),
                triton.next_power_of_2(v), num_warps=4, enable_fp_fusion=False,
            )
        else:
            block = 128 if v <= 8 else 32
            torch.library.wrap_triton(_forward)[(triton.cdiv(rows * u, block), len(output))](
                left, right, weight, info, coefficients, result, rows, wx, wy, u, v,
                len(paths), len(paths) * u * v, shared, block, triton.next_power_of_2(v),
                num_warps=4, enable_fp_fusion=False,
            )


@torch.library.triton_op(_NAMESPACE + "::forward", mutates_args={})
def uvu_forward(left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor,
                info: torch.Tensor, coefficients: torch.Tensor,
                first_order: torch.Tensor, second_order: torch.Tensor,
                metadata_id: int, shared: bool) -> torch.Tensor:
    result = torch.empty_like(left)
    _forward_into(left, right, weight, info, coefficients, first_order, second_order,
                  metadata_id, shared, result)
    return result


@torch.library.triton_op(_NAMESPACE + "::forward_into", mutates_args={"result"})
def uvu_forward_into(left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor,
                     info: torch.Tensor, coefficients: torch.Tensor,
                     first_order: torch.Tensor, second_order: torch.Tensor,
                     metadata_id: int, shared: bool, result: torch.Tensor) -> None:
    _forward_into(left, right, weight, info, coefficients, first_order, second_order,
                  metadata_id, shared, result)


def _backward_into(left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor,
                 grad_output: torch.Tensor, info: torch.Tensor, coefficients: torch.Tensor,
                 first_order: torch.Tensor, second_order: torch.Tensor,
                 metadata_id: int, shared: bool, needed: int,
                 gx: torch.Tensor, gy: torch.Tensor, gw: torch.Tensor) -> None:
    first, second, output, paths, u, v = _METADATA[metadata_id]
    rows, wx, wy, p = left.shape[0], left.shape[1], right.shape[1], len(paths)
    wnum = p * u * v
    if not rows:
        gw.zero_()
        return
    if v == 1 and not shared and 128 <= u <= 256 and needed & 6 == 6:
        from .uvu_single import _joint
        first_meta, second_meta, static_paths = _STATIC[metadata_id]
        partial_y = right.new_empty((len(first), rows, wy))
        for i in range(len(first)):
            selected = tuple(index for index, path in enumerate(paths) if path[0] == i)
            torch.library.wrap_triton(_joint)[(rows,)](
                left, right, weight, grad_output, gx, partial_y, gw, rows, wx, wy,
                u, wnum, first_meta, second_meta, static_paths, i, selected,
                1, triton.next_power_of_2(u), triton.next_power_of_2(wy), bool(needed & 1),
                num_warps=4 if u <= 128 else 8, enable_fp_fusion=False,
            )
        torch.sum(partial_y, dim=0, out=gy)
        return
    if v == 1:
        from .uvu_single import _fused_left, _right
        first_meta, second_meta, static_paths = _STATIC[metadata_id]
        if needed & 1:
            torch.library.wrap_triton(_fused_left)[(triton.cdiv(rows * u, 128),)](
                right, weight, grad_output, gx, rows, wx, wy, u, wnum, shared,
                first_meta, second_meta, static_paths, 128, triton.next_power_of_2(len(first)),
                num_warps=4, enable_fp_fusion=False,
            )
        for j in range(len(second) if needed & 2 else 0):
            selected = tuple(index for index, path in enumerate(paths) if path[1] == j)
            torch.library.wrap_triton(_right)[(triton.cdiv(rows, 2),)](
                left, weight, grad_output, gy, rows, wx, wy, u, wnum, shared,
                first_meta, second_meta, static_paths, j, selected, 2, triton.next_power_of_2(u),
                num_warps=4, enable_fp_fusion=False,
            )
    elif v <= 8:
        from .uvu_fused import _left as _fused_left, _right
        first_meta, second_meta, static_paths = _STATIC[metadata_id]
        if needed & 1:
            torch.library.wrap_triton(_fused_left)[(triton.cdiv(rows * u, 128),)](
                right, weight, grad_output, gx, rows, wx, wy, u, v, wnum, shared,
                first_meta, second_meta, static_paths, 128, triton.next_power_of_2(len(first)),
                triton.next_power_of_2(v), num_warps=4, enable_fp_fusion=False,
            )
        for j in range(len(second) if needed & 2 else 0):
            selected = tuple(index for index, path in enumerate(paths) if path[1] == j)
            torch.library.wrap_triton(_right)[(triton.cdiv(rows, 2), 1)](
                left, weight, grad_output, gy, rows, wx, wy, u, v, wnum, shared,
                first_meta, second_meta, static_paths, j, selected, 2,
                triton.next_power_of_2(u), triton.next_power_of_2(v),
                num_warps=4, enable_fp_fusion=False,
            )
    else:
        block = 128 if v <= 8 else 32
        if needed & 1:
            torch.library.wrap_triton(_input_grad)[(triton.cdiv(rows * u, block), len(first))](
                left, right, weight, grad_output, info, coefficients, first_order, gx,
                rows, wx, wy, u, v, p, len(output), len(first), wnum, shared, True,
                block, triton.next_power_of_2(v), num_warps=4, enable_fp_fusion=False,
            )
        if needed & 2:
            torch.library.wrap_triton(_input_grad)[(triton.cdiv(rows * v, 32), len(second))](
                left, right, weight, grad_output, info, coefficients, second_order, gy,
                rows, wx, wy, u, v, p, len(output), len(first), wnum, shared, False,
                32, triton.next_power_of_2(u), num_warps=4, enable_fp_fusion=False,
            )
    if not needed & 4:
        return
    if shared:
        partial = weight.new_empty((_SPLITS, wnum))
        torch.library.wrap_triton(_weight_shared)[(triton.cdiv(u * v, 32), p, _SPLITS)](
            left, right, grad_output, info, coefficients, partial,
            rows, wx, wy, u, v, wnum, _SPLITS, 32, 128,
            num_warps=4, enable_fp_fusion=False,
        )
        torch.library.wrap_triton(_reduce_weight)[(triton.cdiv(wnum, 128),)](
            partial, gw, wnum, _SPLITS, 128, num_warps=4,
        )
    elif v == 1:
        from .uvu_single import _weights
        first_meta, second_meta, static_paths = _STATIC[metadata_id]
        for k in range(len(output)):
            selected = tuple(index for index, path in enumerate(paths) if path[2] == k)
            if selected:
                torch.library.wrap_triton(_weights)[(triton.cdiv(rows * u, 128),)](
                    left, right, grad_output, gw, rows, wx, wy, u, wnum,
                    first_meta, second_meta, static_paths, selected, 128,
                    num_warps=4, enable_fp_fusion=False,
                )
    else:
        torch.library.wrap_triton(_weight_unshared)[(triton.cdiv(rows * u * v, 128), p)](
            left, right, grad_output, info, coefficients, gw,
            rows, wx, wy, u, v, wnum, 128, num_warps=4, enable_fp_fusion=False,
        )


@torch.library.triton_op(_NAMESPACE + "::backward", mutates_args={})
def uvu_backward(left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor,
                 grad_output: torch.Tensor, info: torch.Tensor, coefficients: torch.Tensor,
                 first_order: torch.Tensor, second_order: torch.Tensor,
                 metadata_id: int, shared: bool, needed: int = 7
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = left.shape[0]
    paths, u, v = _METADATA[metadata_id][3:]
    gx = torch.empty_like(left) if needed & 1 else left.new_empty(0)
    gy = torch.empty_like(right) if needed & 2 else right.new_empty(0)
    gw = (torch.empty_like(weight) if shared else left.new_empty((rows, len(paths)*u*v))) if needed & 4 else left.new_empty(0)
    _backward_into(left, right, weight, grad_output, info, coefficients,
                   first_order, second_order, metadata_id, shared, needed, gx, gy, gw)
    return gx, gy, gw


@torch.library.triton_op(_NAMESPACE + "::backward_into", mutates_args={"gx", "gy", "gw"})
def uvu_backward_into(left: torch.Tensor, right: torch.Tensor, weight: torch.Tensor,
                      grad_output: torch.Tensor, info: torch.Tensor, coefficients: torch.Tensor,
                      first_order: torch.Tensor, second_order: torch.Tensor,
                      metadata_id: int, shared: bool, needed: int,
                      gx: torch.Tensor, gy: torch.Tensor, gw: torch.Tensor) -> None:
    _backward_into(left, right, weight, grad_output, info, coefficients,
                   first_order, second_order, metadata_id, shared, needed, gx, gy, gw)


def _forward_setup(ctx, inputs, output):
    left, right, weight, info, coefficients, first_order, second_order, metadata_id, shared = inputs
    ctx.save_for_backward(left, right, weight, info, coefficients, first_order, second_order)
    ctx.metadata_id, ctx.shared = metadata_id, shared


def _backward(ctx, grad_output):
    left, right, weight, info, coefficients, first_order, second_order = ctx.saved_tensors
    needed = _required_gradients((left, right, weight), ctx.needs_input_grad[:3])
    mask = sum(int(active) << index for index, active in enumerate(needed))
    grads = uvu_backward(left, right, weight, grad_output.contiguous(), info, coefficients,
                         first_order, second_order, ctx.metadata_id, ctx.shared, mask)
    return *(value if active else None for value, active in zip(grads, needed)), None, None, None, None, None, None


uvu_forward.register_autograd(_backward, setup_context=_forward_setup)


def _backward_setup(ctx, inputs, output):
    left, right, weight, grad_output, info, coefficients, first_order, second_order, metadata_id, shared, needed = inputs
    ctx.save_for_backward(left, right, weight, grad_output, coefficients)
    ctx.metadata_id, ctx.shared = metadata_id, shared
    ctx.set_materialize_grads(False)


def _double_backward(ctx, dd_left, dd_right, dd_weight):
    left, right, weight, grad_output, coefficients = ctx.saved_tensors
    selected = [(i, cot) for i, cot in enumerate((dd_left, dd_right, dd_weight)) if cot is not None]
    if not selected:
        return (None,) * 11
    with torch.enable_grad():
        operands = tuple(value.detach().requires_grad_(True)
                         for value in (left, right, weight, grad_output))
        value = torch_uvu_forward(*operands[:3], coefficients,
                                  _METADATA[ctx.metadata_id], ctx.shared)
        gradients = torch.autograd.grad(value, operands[:3], operands[3], create_graph=True)
        result = torch.autograd.grad(
            tuple(gradients[i] for i, _ in selected), operands,
            grad_outputs=tuple(cot for _, cot in selected), allow_unused=True,
        )
    return *result, None, None, None, None, None, None, None


uvu_backward.register_autograd(_double_backward, setup_context=_backward_setup)
