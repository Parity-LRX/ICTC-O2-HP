"""Fuse a radial network's final affine layer with a V=1 O(2) product."""

from __future__ import annotations

import torch

from .uvu_o2 import ReferenceUvuTensorProduct, UvuTensorProduct


class RadialUvuTensorProduct(torch.nn.Module):
    """Equivalent to ``tp(left, angular, radial_out(radial_hidden))``.

    ``tp`` must use one second-input channel and external per-edge weights.
    The supplied Linear and its parameters are reused without reordering.
    CUDA FP32 combines the affine projection with TP and its adjoints using
    bounded edge blocks. Saved generated weights are limited to the first
    block; local weight adjoints are consumed immediately. Other dtypes/devices
    and a reference TP use the original differentiable composition.
    """

    def __init__(self, tp: ReferenceUvuTensorProduct, radial_out: torch.nn.Linear):
        super().__init__()
        if not isinstance(tp, ReferenceUvuTensorProduct):
            raise TypeError("tp must be an O(2) uvu tensor product")
        if tp.channels_in2 != 1 or tp.shared_weights or tp.internal_weights:
            raise ValueError("radial fusion requires V=1 and external per-edge TP weights")
        if not isinstance(radial_out, torch.nn.Linear):
            raise TypeError("radial_out must be the final torch.nn.Linear of the radial network")
        if radial_out.out_features != tp.weight_numel:
            raise ValueError("radial_out.out_features must equal tp.weight_numel")
        if radial_out.in_features < 1:
            raise ValueError("radial_out must have at least one input feature")
        self.tp = tp
        self.radial_out = radial_out

    def forward(self, left, angular, radial_hidden):
        if left.ndim < 1 or left.shape[-1] != self.tp.input1_dimension:
            raise ValueError("left has the wrong final dimension")
        if angular.shape != (*left.shape[:-1], self.tp.input2_dimension):
            raise ValueError("angular must have the same batch shape and configured dimension")
        if radial_hidden.shape != (*left.shape[:-1], self.radial_out.in_features):
            raise ValueError("radial_hidden must have the same batch shape and affine input dimension")
        if not left.is_floating_point():
            raise TypeError("radial uvu operands must be real floating-point tensors")
        operands = [angular, radial_hidden, self.radial_out.weight, self.tp._coefficients]
        if self.radial_out.bias is not None:
            operands.append(self.radial_out.bias)
        if any(value.device != left.device or value.dtype != left.dtype for value in operands):
            raise ValueError("all operands and modules must have the same device and dtype")
        if (left.device.type != "cuda" or left.dtype != torch.float32
                or not isinstance(self.tp, UvuTensorProduct)):
            return self.tp(left, angular, self.radial_out(radial_hidden))
        from .radial_streamed import radial_forward
        bias = self.radial_out.bias
        result, cached_weights = radial_forward(
            left.reshape(-1, self.tp.input1_dimension).contiguous(),
            angular.reshape(-1, self.tp.input2_dimension).contiguous(),
            radial_hidden.reshape(-1, self.radial_out.in_features).contiguous(),
            self.radial_out.weight.contiguous(),
            bias.contiguous() if bias is not None else self.radial_out.weight.new_empty(0),
            self.tp._path_info, self.tp._coefficients,
            self.tp._first_order, self.tp._second_order,
            self.tp._kernel_metadata_id, bias is not None,
            torch.is_grad_enabled() and (left.requires_grad or angular.requires_grad),
        )
        return result.reshape(*left.shape[:-1], self.tp.representation_dimension)
