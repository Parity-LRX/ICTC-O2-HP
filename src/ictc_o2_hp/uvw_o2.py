"""Shared-weight Cartesian O(2) UVW with three independent channel axes.

First candidate: tiled Triton channel projection, ordinary differentiable Torch
angular assembly. This is not a claim of a single fused generic UVW launch.
"""
import math
import torch

from .uvu_o2 import ReferenceUvuTensorProduct, _integer, _representation


class ReferenceUvwTensorProduct(ReferenceUvuTensorProduct):
    connection_mode = "uvw"

    def __init__(self, channels_in1, channels_in2, channels_out, m_max, *,
                 parity="full", parity_in2=None, m_max_in2=None,
                 normalization="component", path_normalization="none",
                 internal_weights=True, shared_weights=True):
        if not shared_weights:
            raise NotImplementedError("UVW v1 supports shared weights only")
        super().__init__(channels_in1, channels_in2, m_max, parity=parity,
                         parity_in2=parity_in2, m_max_in2=m_max_in2,
                         normalization=normalization, path_normalization="none",
                         internal_weights=False, shared_weights=True)
        self.channels_out = _integer(channels_out, "channels_out", 1)
        self.blocks_out, self.entries_out, self.representation_dimension = _representation(
            self.channels_out, self.m_max, self.parity)
        if path_normalization not in ("none", "element", "path"):
            raise ValueError("Invalid path_normalization")
        self.path_normalization = path_normalization
        if path_normalization != "none":
            counts = [sum(p[2] == k for p in self.paths) for k in range(len(self.entries_out))]
            values = torch.tensor(self._coefficient_values, dtype=torch.float64).reshape(-1, 2, 2, 2)
            for p, (_, _, k) in enumerate(self.paths):
                values[p] /= math.sqrt(self.channels * self.channels_in2 * counts[k])
            self._coefficient_values = tuple(tuple(r) for r in values.reshape(-1, 8).tolist())
            self._coefficients.copy_(values)
        self.weight_numel = len(self.paths) * self.channels * self.channels_in2 * self.channels_out
        self.weight_shape = (self.weight_numel,)
        self.internal_weights = bool(internal_weights)
        del self.weight
        if self.internal_weights:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel) * 0.02)
        else:
            self.register_buffer("weight", torch.empty(0))

    def weight_view_for_path(self, path, weight=None):
        if not 0 <= path < len(self.paths):
            raise IndexError("Invalid UVW path")
        if weight is None:
            weight = self.weight
        width = self.channels * self.channels_in2 * self.channels_out
        return weight[path * width:(path + 1) * width].reshape(
            self.channels, self.channels_in2, self.channels_out)

    def _contract(self, x, y, weight, coefficients):
        return torch.einsum("nua,nvb,uvw,cab->nwc", x, y, weight, coefficients)

    def forward(self, left, right, weight=None):
        weight = self._prepare(left, right, weight)
        batch = left.shape[:-1]
        left = left.reshape(-1, self.input1_dimension)
        right = right.reshape(-1, self.input2_dimension)
        rows = left.shape[0]
        values = [None] * len(self.entries_out)
        for p, (i, j, k) in enumerate(self.paths):
            a, b, out = self.entries_in1[i], self.entries_in2[j], self.entries_out[k]
            x = left[:, a.offset:a.offset + self.channels * a.dim].reshape(rows, self.channels, a.dim)
            y = right[:, b.offset:b.offset + self.channels_in2 * b.dim].reshape(rows, self.channels_in2, b.dim)
            cg = self._coefficients[p, :out.dim, :a.dim, :b.dim]
            part = self._contract(x, y, self.weight_view_for_path(p, weight), cg)
            values[k] = part if values[k] is None else values[k] + part
        return torch.cat([v.reshape(rows, -1) for v in values], -1).reshape(
            *batch, self.representation_dimension)


class UvwTensorProduct(ReferenceUvwTensorProduct):
    def _contract(self, x, y, weight, coefficients):
        if not x.is_cuda or x.dtype != torch.float32:
            return super()._contract(x, y, weight, coefficients)
        from .uvw_triton import channel_matmul
        rows, u, da = x.shape
        _, v, db = y.shape
        w = weight.shape[-1]
        # Project v while keeping independent u and w. Never materialize N*U*V*W.
        projected = channel_matmul(y.transpose(1, 2).reshape(rows * db, v),
                                   weight.permute(1, 0, 2).reshape(v, u * w))
        projected = projected.reshape(rows, db, u, w)
        return torch.einsum("nua,nbuw,cab->nwc", x, projected, coefficients)
