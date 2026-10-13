"""Bounded radial-affine/TP execution using cuBLAS and fused TP adjoints."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .uvu_triton import _NAMESPACE, uvu_forward, uvu_backward

_MAX_ROWS = 16384
_WEIGHT_ELEMENTS = 2**30  # At most 4 GiB of FP32 values in one weight block.


def _block_rows(weight_numel):
    capacity = max(1, _WEIGHT_ELEMENTS // weight_numel)
    return min(_MAX_ROWS, 1 << (capacity.bit_length() - 1))


def _block_forward(left, angular, radial, alpha, bias, info, coefficients,
                    first_order, second_order, metadata_id, has_bias, keep_weights):
    weights = F.linear(radial, alpha, bias if has_bias else None)
    value = uvu_forward(left, angular, weights, info, coefficients,
                        first_order, second_order, metadata_id, False)
    return value, weights if keep_weights else alpha.new_empty(0)


def _block_backward(left, angular, radial, alpha, bias, cached_weights, grad_output, info, coefficients,
                     first_order, second_order, metadata_id, has_bias):
    weights = cached_weights if cached_weights.numel() else F.linear(radial, alpha, bias if has_bias else None)
    gx, gy, gw = uvu_backward(left, angular, weights, grad_output, info, coefficients,
                              first_order, second_order, metadata_id, False)
    gr = gw @ alpha
    ga = gw.T @ radial
    gb = gw.sum(0) if has_bias else torch.empty_like(bias)
    return gx, gy, gr, ga, gb


# Compile the complete block so the scheduler can reuse buffers and release
# generated weights before allocating their adjoints. All N use the same cap.
_compiled_forward = torch.compile(_block_forward, fullgraph=True, dynamic=False)
_compiled_backward = torch.compile(_block_backward, fullgraph=True, dynamic=False)


@torch.library.custom_op(_NAMESPACE + "::radial_stream_forward", mutates_args={})
def radial_forward(left: torch.Tensor, angular: torch.Tensor, radial: torch.Tensor,
                    alpha: torch.Tensor, bias: torch.Tensor, info: torch.Tensor,
                    coefficients: torch.Tensor, first_order: torch.Tensor,
                    second_order: torch.Tensor, metadata_id: int, has_bias: bool,
                    save_first: bool) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.empty_like(left)
    first_weights = alpha.new_empty((0, alpha.shape[0])) if save_first else alpha.new_empty(0)
    block_rows = _block_rows(alpha.shape[0])
    with torch.no_grad():
        for start in range(0, left.shape[0], block_rows):
            end = min(start + block_rows, left.shape[0])
            value, block_weights = _compiled_forward(left[start:end], angular[start:end], radial[start:end],
                                       alpha, bias, info, coefficients, first_order, second_order,
                                       metadata_id, has_bias, save_first and start == 0)
            result[start:end].copy_(value)
            if save_first and start == 0:
                first_weights = block_weights
    return result, first_weights


@radial_forward.register_fake
def _fake_forward(left, angular, radial, alpha, bias, info, coefficients,
                   first_order, second_order, metadata_id, has_bias, save_first):
    cache = alpha.new_empty((min(_block_rows(alpha.shape[0]), left.shape[0]), alpha.shape[0])) if save_first else alpha.new_empty(0)
    return torch.empty_like(left), cache


@torch.library.custom_op(_NAMESPACE + "::radial_stream_backward", mutates_args={})
def radial_backward(left: torch.Tensor, angular: torch.Tensor, radial: torch.Tensor,
                     alpha: torch.Tensor, bias: torch.Tensor, cached_weights: torch.Tensor,
                     grad_output: torch.Tensor,
                     info: torch.Tensor, coefficients: torch.Tensor,
                     first_order: torch.Tensor, second_order: torch.Tensor,
                     metadata_id: int, has_bias: bool
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gx, gy, gr = torch.empty_like(left), torch.empty_like(angular), torch.empty_like(radial)
    ga, gb = torch.zeros_like(alpha), torch.zeros_like(bias)
    block_rows = _block_rows(alpha.shape[0])
    with torch.no_grad():
        for start in range(0, left.shape[0], block_rows):
            end = min(start + block_rows, left.shape[0])
            hidden = radial[start:end]
            dx, dy, dr, da, db = _compiled_backward(
                left[start:end], angular[start:end], hidden, alpha, bias,
                cached_weights if start == 0 else alpha.new_empty(0), grad_output[start:end],
                info, coefficients, first_order, second_order, metadata_id, has_bias,
            )
            gx[start:end].copy_(dx)
            gy[start:end].copy_(dy)
            gr[start:end].copy_(dr)
            ga.add_(da)
            if has_bias:
                gb.add_(db)
            del dx, dy, dr, da, db
    return gx, gy, gr, ga, gb


@radial_backward.register_fake
def _fake_backward(left, angular, radial, alpha, bias, cached_weights, grad_output, info, coefficients,
                    first_order, second_order, metadata_id, has_bias):
    return tuple(torch.empty_like(value) for value in (left, angular, radial, alpha, bias))


def _forward_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:9], output[1])
    ctx.metadata_id, ctx.has_bias = inputs[9:11]
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, grad_output, ignored_cache_grad):
    x, y, r, a, b, info, coefficients, order1, order2, cached = ctx.saved_tensors
    grads = radial_backward(x, y, r, a, b, cached, grad_output.contiguous(), info, coefficients,
                            order1, order2, ctx.metadata_id, ctx.has_bias)
    return *grads, None, None, None, None, None, None, None


radial_forward.register_autograd(_backward, setup_context=_forward_setup)


def _backward_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:11])
    ctx.metadata_id, ctx.has_bias = inputs[11:]
    ctx.set_materialize_grads(False)


def _double_backward(ctx, dx, dy, dr, da, db):
    x, y, r, a, b, cached, g, info, coefficients, order1, order2 = ctx.saved_tensors
    accumulated = [None] * 6

    def add_term(operands, original, has_bias):
        tx, ty, tr, ta, tb = operands
        grads = radial_backward(tx, ty, tr, ta, tb, ta.new_empty(0), g, info, coefficients,
                                order1, order2, ctx.metadata_id, has_bias)
        value, unused_cache = radial_forward(tx, ty, tr, ta, tb, info, coefficients,
                                              order1, order2, ctx.metadata_id, has_bias, False)
        for index in original:
            term = grads[index]
            accumulated[index] = term if accumulated[index] is None else accumulated[index] + term
        accumulated[5] = value if accumulated[5] is None else accumulated[5] + value

    if dx is not None:
        add_term((dx.contiguous(), y, r, a, b), (1, 2, 3, 4), ctx.has_bias)
    if dy is not None:
        add_term((x, dy.contiguous(), r, a, b), (0, 2, 3, 4), ctx.has_bias)
    if dr is not None:
        add_term((x, y, dr.contiguous(), a, b.new_empty(0)), (0, 1, 3), False)
    if da is not None or db is not None:
        probe_a = da.contiguous() if da is not None else torch.zeros_like(a)
        probe_b = db.contiguous() if db is not None else b.new_empty(0)
        add_term((x, y, r, probe_a, probe_b), (0, 1, 2), db is not None and ctx.has_bias)
    return *accumulated[:5], None, accumulated[5], None, None, None, None, None, None


radial_backward.register_autograd(_double_backward, setup_context=_backward_setup)
