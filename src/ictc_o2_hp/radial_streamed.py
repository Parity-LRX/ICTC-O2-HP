"""Bounded radial-affine/TP execution using cuBLAS and fused TP adjoints."""

from __future__ import annotations

from functools import lru_cache
from types import FunctionType

import torch
import torch.nn.functional as F

from .uvu_triton import _NAMESPACE, _METADATA, uvu_forward_into, uvu_backward_into, _required_gradients

_MAX_ROWS = 16384
_WEIGHT_ELEMENTS = 2**30  # At most 4 GiB of FP32 values in one weight block.


def _block_rows(weight_numel, metadata_id):
    # Joint adjoints keep weights and their adjoints live together. Bound their
    # combined storage with half-size blocks for the qualified channel range.
    channels = _METADATA[metadata_id][4]
    joint = 128 <= channels <= 256
    capacity = max(1, (_WEIGHT_ELEMENTS // (2 if joint else 1)) // weight_numel)
    return min(_MAX_ROWS // (2 if joint else 1), 1 << (capacity.bit_length() - 1))


def _block_forward(left, angular, radial, alpha, bias, info, coefficients,
                    first_order, second_order, metadata_id, has_bias, keep_weights, dest):
    weights = F.linear(radial, alpha, bias if has_bias else None)
    uvu_forward_into(left, angular, weights, info, coefficients,
                     first_order, second_order, metadata_id, False, dest)
    return weights if keep_weights else alpha.new_empty(0)


def _block_backward(left, angular, radial, alpha, bias, cached_weights, grad_output, info, coefficients,
                     first_order, second_order, metadata_id, has_bias, needed, gx, gy):
    weights = (cached_weights if cached_weights.numel() else F.linear(radial, alpha, bias if has_bias else None)) if needed & 3 else alpha.new_empty(0)
    tp_needed = (needed & 3) | (4 if needed & 28 else 0)
    gw = left.new_empty((left.shape[0], alpha.shape[0])) if tp_needed & 4 else left.new_empty(0)
    uvu_backward_into(left, angular, weights, grad_output, info, coefficients,
                      first_order, second_order, metadata_id, False, tp_needed, gx, gy, gw)
    # Equivalent transposed products give cuBLAS a different output layout.
    # Leave the block adjoints as views; the caller copies/accumulates them.
    gr = (alpha.T @ gw.T).T if needed & 4 else radial.new_empty(0)
    ga = (radial.T @ gw).T if needed & 8 else alpha.new_empty(0)
    gb = gw.sum(0) if has_bias and needed & 16 else bias.new_empty(0)
    return gr, ga, gb


# Compile complete blocks with bounded temporary lifetimes. Joint TP adjoints
# keep local weights and their gradients live together. All N use the same cap.
def _compile_block(function, key):
    # Dynamo's specialization limit belongs to a code object. Different TP
    # plans/adjoint roles have independent graphs; do not raise a global limit.
    name = function.__name__ + "_" + "_".join(map(str, key))
    isolated = FunctionType(function.__code__.replace(co_name=name), function.__globals__, name)
    isolated.__module__ = function.__module__
    return torch.compile(isolated, fullgraph=True, dynamic=False)


@lru_cache(maxsize=256)
def _compiled_forward(metadata_id, keep_weights):
    return _compile_block(_block_forward, (metadata_id, int(keep_weights)))


@lru_cache(maxsize=256)
def _compiled_backward(metadata_id, needed):
    return _compile_block(_block_backward, (metadata_id, needed))


@torch.library.custom_op(_NAMESPACE + "::radial_stream_forward", mutates_args={})
def radial_forward(left: torch.Tensor, angular: torch.Tensor, radial: torch.Tensor,
                    alpha: torch.Tensor, bias: torch.Tensor, info: torch.Tensor,
                    coefficients: torch.Tensor, first_order: torch.Tensor,
                    second_order: torch.Tensor, metadata_id: int, has_bias: bool,
                    save_first: bool) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.empty_like(left)
    first_weights = alpha.new_empty((0, alpha.shape[0])) if save_first else alpha.new_empty(0)
    block_rows = _block_rows(alpha.shape[0], metadata_id)
    with torch.no_grad():
        for start in range(0, left.shape[0], block_rows):
            end = min(start + block_rows, left.shape[0])
            keep_weights = save_first and start == 0
            block_weights = _compiled_forward(metadata_id, keep_weights)(left[start:end], angular[start:end], radial[start:end],
                                       alpha, bias, info, coefficients, first_order, second_order,
                                       metadata_id, has_bias, keep_weights, result[start:end])
            if save_first and start == 0:
                first_weights = block_weights
    return result, first_weights


@radial_forward.register_fake
def _fake_forward(left, angular, radial, alpha, bias, info, coefficients,
                   first_order, second_order, metadata_id, has_bias, save_first):
    cache = alpha.new_empty((min(_block_rows(alpha.shape[0], metadata_id), left.shape[0]), alpha.shape[0])) if save_first else alpha.new_empty(0)
    return torch.empty_like(left), cache


@torch.library.custom_op(_NAMESPACE + "::radial_stream_backward", mutates_args={})
def radial_backward(left: torch.Tensor, angular: torch.Tensor, radial: torch.Tensor,
                     alpha: torch.Tensor, bias: torch.Tensor, cached_weights: torch.Tensor,
                     grad_output: torch.Tensor,
                     info: torch.Tensor, coefficients: torch.Tensor,
                     first_order: torch.Tensor, second_order: torch.Tensor,
                     metadata_id: int, has_bias: bool, needed: int = 31
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gx = torch.empty_like(left) if needed & 1 else left.new_empty(0)
    gy = torch.empty_like(angular) if needed & 2 else angular.new_empty(0)
    gr = torch.empty_like(radial) if needed & 4 else radial.new_empty(0)
    # Match each block's physical layout and convert only the final sum.
    ga = torch.zeros_like(alpha.T, memory_format=torch.contiguous_format).T if needed & 8 else alpha.new_empty(0)
    gb = torch.zeros_like(bias) if needed & 16 else bias.new_empty(0)
    block_rows = _block_rows(alpha.shape[0], metadata_id)
    with torch.no_grad():
        for start in range(0, left.shape[0], block_rows):
            end = min(start + block_rows, left.shape[0])
            hidden = radial[start:end]
            dr, da, db = _compiled_backward(metadata_id, needed)(
                left[start:end], angular[start:end], hidden, alpha, bias,
                cached_weights if start == 0 else alpha.new_empty(0), grad_output[start:end],
                info, coefficients, first_order, second_order, metadata_id, has_bias, needed,
                gx[start:end] if needed & 1 else gx, gy[start:end] if needed & 2 else gy,
            )
            if needed & 4: gr[start:end].copy_(dr)
            if needed & 8: ga.add_(da)
            if has_bias and needed & 16:
                gb.add_(db)
            del dr, da, db
    return gx, gy, gr, ga.contiguous(), gb


@radial_backward.register_fake
def _fake_backward(left, angular, radial, alpha, bias, cached_weights, grad_output, info, coefficients,
                    first_order, second_order, metadata_id, has_bias, needed=31):
    return tuple(torch.empty_like(value) if needed & (1 << index) else value.new_empty(0)
                 for index, value in enumerate((left, angular, radial, alpha, bias)))


def _forward_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:9], output[1])
    ctx.metadata_id, ctx.has_bias = inputs[9:11]
    ctx.mark_non_differentiable(output[1])


def _backward(ctx, grad_output, ignored_cache_grad):
    x, y, r, a, b, info, coefficients, order1, order2, cached = ctx.saved_tensors
    needed = _required_gradients((x, y, r, a, b), ctx.needs_input_grad[:5])
    mask = sum(int(active) << index for index, active in enumerate(needed))
    grads = radial_backward(x, y, r, a, b, cached, grad_output.contiguous(), info, coefficients,
                            order1, order2, ctx.metadata_id, ctx.has_bias, mask)
    return *(value if active else None for value, active in zip(grads, needed)), None, None, None, None, None, None, None


radial_forward.register_autograd(_backward, setup_context=_forward_setup)


def _backward_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:11])
    ctx.metadata_id, ctx.has_bias, ctx.first_needed = inputs[11:]
    ctx.set_materialize_grads(False)


def _double_backward(ctx, dx, dy, dr, da, db):
    x, y, r, a, b, cached, g, info, coefficients, order1, order2 = ctx.saved_tensors
    needed = _required_gradients((x, y, r, a, b, g),
                                (*ctx.needs_input_grad[:5], ctx.needs_input_grad[6]))
    accumulated = [None] * 6

    def add_term(operands, original, has_bias):
        tx, ty, tr, ta, tb = operands
        selected = tuple(index for index in original if needed[index])
        if selected:
            mask = sum(1 << index for index in selected)
            grads = radial_backward(tx, ty, tr, ta, tb, ta.new_empty(0), g, info, coefficients,
                                    order1, order2, ctx.metadata_id, has_bias, mask)
        for index in selected:
            term = grads[index]
            accumulated[index] = term if accumulated[index] is None else accumulated[index] + term
        if needed[5]:
            value, unused_cache = radial_forward(tx, ty, tr, ta, tb, info, coefficients,
                                                  order1, order2, ctx.metadata_id, has_bias, False)
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
    return *accumulated[:5], None, accumulated[5], None, None, None, None, None, None, None


radial_backward.register_autograd(_double_backward, setup_context=_backward_setup)
