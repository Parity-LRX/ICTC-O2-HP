"""O(2) products preserving the first input channel and summing the second."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from . import o2_math
from .native_o2 import canonical_blocks, full_parity_blocks


class UvuEntry(NamedTuple):
    key: tuple[str, int]
    copy: int
    offset: int
    dim: int


def _integer(value, name, minimum):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _representation(channels, order, parity):
    if parity not in ("canonical", "full"):
        raise ValueError("parity must be canonical or full")
    blocks = (canonical_blocks if parity == "canonical" else full_parity_blocks)(
        order, channels
    )
    entries, cursor = [], 0
    for block in blocks:
        for copy in range(block.parity_copies):
            entries.append(UvuEntry(
                block.key, copy, cursor + copy * channels * block.irrep_dimension,
                block.irrep_dimension,
            ))
        cursor += block.multiplicity * block.irrep_dimension
    return blocks, tuple(entries), cursor


def _basis(entry):
    if entry.copy:
        return torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    return torch.eye(entry.dim, dtype=torch.float64)


def torch_uvu_forward(left, right, weight, coefficients, metadata, shared):
    """Factor the v reduction before angular coupling, avoiding an outer product."""
    first, second, output, paths, channels, channels_in2 = metadata
    rows = left.shape[0]
    values = [None] * len(output)
    for p, (i, j, k) in enumerate(paths):
        a, b, out = first[i], second[j], output[k]
        x = left[:, a.offset:a.offset + channels * a.dim].reshape(rows, channels, a.dim)
        y = right[:, b.offset:b.offset + channels_in2 * b.dim].reshape(
            rows, channels_in2, b.dim
        )
        start = p * channels * channels_in2
        w = weight[..., start:start + channels * channels_in2]
        if shared:
            projected = torch.einsum("uv,nvj->nuj", w.reshape(channels, channels_in2), y)
        else:
            projected = torch.einsum(
                "nuv,nvj->nuj", w.reshape(rows, channels, channels_in2), y
            )
        cg = coefficients[p, :out.dim, :a.dim, :b.dim]
        contribution = torch.einsum("nui,nuj,kij->nuk", x, projected, cg)
        values[k] = contribution if values[k] is None else values[k] + contribution
    return torch.cat([value.reshape(rows, channels * entry.dim)
                      for entry, value in zip(output, values)], dim=-1)


class ReferenceUvuTensorProduct(torch.nn.Module):
    """Flat Cartesian O(2) uvu product, implemented with differentiable PyTorch.

    The first input and output have C channels per representation copy. The
    second input has V channels per copy. Every legal entry-to-entry coupling,
    including all full-parity copy combinations, has a (C, V) weight matrix.
    Output frequencies are truncated at m_max; the second input can use its
    own cutoff and parity. Flattened blocks use [copy, channel, component].
    """

    connection_mode = "uvu"

    def __init__(
        self, channels, channels_in2, m_max, *, parity="full", parity_in2=None,
        m_max_in2=None, normalization="component", path_normalization="none",
        internal_weights=None, shared_weights=True,
    ):
        super().__init__()
        self.channels = _integer(channels, "channels", 1)
        self.channels_in2 = _integer(channels_in2, "channels_in2", 1)
        self.m_max = _integer(m_max, "m_max", 0)
        self.m_max_in2 = _integer(
            m_max if m_max_in2 is None else m_max_in2, "m_max_in2", 0
        )
        self.parity = parity
        self.parity_in2 = parity if parity_in2 is None else parity_in2
        if normalization not in ("component", "norm"):
            raise ValueError("normalization must be component or norm")
        if path_normalization not in ("none", "element", "path"):
            raise ValueError("path_normalization must be none, element or path")
        self.normalization, self.path_normalization = normalization, path_normalization
        self.shared_weights = bool(shared_weights)
        self.internal_weights = (self.shared_weights if internal_weights is None
                                 else bool(internal_weights))
        if self.internal_weights and not self.shared_weights:
            raise ValueError("internal_weights requires shared_weights=True")
        self.blocks, self.entries_in1, self.input1_dimension = _representation(
            self.channels, self.m_max, self.parity
        )
        self.blocks_in2, self.entries_in2, self.input2_dimension = _representation(
            self.channels_in2, self.m_max_in2, self.parity_in2
        )
        self.entries_out = self.entries_in1
        self.representation_dimension = self.input1_dimension
        self.keys = tuple(block.key for block in self.blocks)
        paths, coefficients = [], []
        for k, out in enumerate(self.entries_out):
            start = len(paths)
            for i, first in enumerate(self.entries_in1):
                for j, second in enumerate(self.entries_in2):
                    if out.key not in o2_math.o2_valid_output_keys(first.key, second.key):
                        continue
                    cg = o2_math.o2_cg_tensor(first.key, second.key, out.key,
                                             dtype=torch.float64)
                    norm = cg.norm()
                    cg = cg * (math.sqrt(out.dim) if normalization == "component" else 1) / norm
                    cg = torch.einsum("abk,ai,bj,lk->ijl", cg, _basis(first),
                                      _basis(second), _basis(out).T)
                    padded = torch.zeros(2, 2, 2, dtype=torch.float64)
                    padded[:out.dim, :first.dim, :second.dim] = cg.permute(2, 0, 1)
                    paths.append((i, j, k))
                    coefficients.append(padded)
            count = len(paths) - start
            if path_normalization != "none":
                # All entries of input2 have the same V. Thus element and
                # path normalization coincide for this uniform-multiplicity API.
                scale = math.sqrt(self.channels_in2 * count)
                for p in range(start, len(paths)):
                    coefficients[p] = coefficients[p] / scale
        self.paths = tuple(paths)
        self.weight_numel = len(paths) * self.channels * self.channels_in2
        self.weight_shape = (self.weight_numel,)
        self._metadata = (self.entries_in1, self.entries_in2, self.entries_out,
                          self.paths, self.channels, self.channels_in2)
        source = torch.stack(coefficients)
        self._coefficient_values = tuple(tuple(row) for row in source.reshape(-1, 8).tolist())
        self.register_buffer("_coefficients", source.to(torch.get_default_dtype()), persistent=False)
        if self.internal_weights:
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel) * 0.02)
        else:
            self.register_buffer("weight", torch.empty(0))

    @property
    def parameter_count(self):
        return self.weight.numel() if self.internal_weights else 0

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # Recover the original f64 coefficients when converting a f32 module
        # to f64, rather than retaining a prior cast's rounding error.
        self._coefficients.copy_(torch.tensor(
            self._coefficient_values, dtype=self._coefficients.dtype,
            device=self._coefficients.device,
        ).reshape(-1, 2, 2, 2))
        return self

    def _prepare(self, left, right, weight):
        if left.ndim < 1 or left.shape[-1] != self.input1_dimension:
            raise ValueError(f"left must end in input1_dimension={self.input1_dimension}")
        if (right.ndim < 1 or right.shape[-1] != self.input2_dimension
                or right.shape[:-1] != left.shape[:-1]):
            raise ValueError("right must have the same batch shape and configured input2_dimension")
        if weight is None:
            if not self.internal_weights:
                raise ValueError("external weights are required")
            weight = self.weight
        expected = (self.weight_numel,) if self.shared_weights else (*left.shape[:-1], self.weight_numel)
        if weight.shape != expected:
            raise ValueError(f"weight shape must be {expected}")
        if not left.is_floating_point():
            raise TypeError("uvu inputs and weights must be real floating-point tensors")
        if any(value.device != left.device or value.dtype != left.dtype
               for value in (right, weight, self._coefficients)):
            raise ValueError("both inputs, weights and module must have the same device and dtype")
        return weight

    def weight_view_for_path(self, path, weight=None):
        if not 0 <= path < len(self.paths):
            raise IndexError("invalid uvu path index")
        if weight is None:
            if not self.internal_weights:
                raise ValueError("external weights are required")
            weight = self.weight
        start = path * self.channels * self.channels_in2
        return weight[..., start:start + self.channels * self.channels_in2].reshape(
            *weight.shape[:-1], self.channels, self.channels_in2
        )

    def forward(self, left, right, weight=None):
        weight = self._prepare(left, right, weight)
        flat_weight = weight if self.shared_weights else weight.reshape(-1, self.weight_numel)
        result = torch_uvu_forward(
            left.reshape(-1, self.input1_dimension), right.reshape(-1, self.input2_dimension),
            flat_weight, self._coefficients, self._metadata, self.shared_weights,
        )
        return result.reshape(*left.shape[:-1], self.representation_dimension)


class UvuTensorProduct(ReferenceUvuTensorProduct):
    """Memory-bounded FP32 CUDA uvu kernels, with a PyTorch dtype/CPU fallback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .uvu_triton import register_metadata
        metadata_id, info, first_order, second_order = register_metadata(
            self._metadata, self._coefficient_values
        )
        self._kernel_metadata_id = metadata_id
        self.register_buffer("_path_info", info, persistent=False)
        self.register_buffer("_first_order", first_order, persistent=False)
        self.register_buffer("_second_order", second_order, persistent=False)

    def forward(self, left, right, weight=None):
        weight = self._prepare(left, right, weight)
        if left.device.type != "cuda" or left.dtype != torch.float32:
            return super().forward(left, right, weight)
        from .uvu_triton import uvu_forward
        flat_weight = weight if self.shared_weights else weight.reshape(-1, self.weight_numel)
        result = uvu_forward(
            left.reshape(-1, self.input1_dimension).contiguous(),
            right.reshape(-1, self.input2_dimension).contiguous(), flat_weight.contiguous(),
            self._path_info, self._coefficients, self._first_order, self._second_order,
            self._kernel_metadata_id, self.shared_weights,
        )
        return result.reshape(*left.shape[:-1], self.representation_dimension)
