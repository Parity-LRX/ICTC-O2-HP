"""Standalone full-parity O(2) operator adapter.

The timed path acts directly on two-dimensional O(2) irreps.  It contains no
O(3) carrier, edge-frame construction, Wigner rotation, projection, or lift.

For m > 0, O(2) has one real two-dimensional irrep.  The familiar ``me`` and
``mo`` labels are equivalent copies with different reflection bases.  This
module retains both copies explicitly: the even copy uses the source basis and
the odd copy uses a fixed J basis transform.  The underlying CG contraction is
therefore performed once in one canonical O(2) basis without pretending the
two copies are inequivalent irreps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


O2Key = tuple[str, int]


@dataclass(frozen=True)
class RepresentationBlock:
    token: str
    key: O2Key
    frequency: int
    irrep_dimension: int
    multiplicity: int
    parity_copies: int


def full_parity_blocks(max_frequency: int, channels: int) -> tuple[RepresentationBlock, ...]:
    if max_frequency < 0:
        raise ValueError("max_frequency must be non-negative")
    if channels <= 0:
        raise ValueError("channels must be positive")
    blocks = [
        RepresentationBlock("0e", ("scalar", 1), 0, 1, channels, 1),
        RepresentationBlock("0o", ("scalar", -1), 0, 1, channels, 1),
    ]
    blocks.extend(
        RepresentationBlock(str(m), ("freq", m), m, 2, 2 * channels, 2)
        for m in range(1, max_frequency + 1)
    )
    return tuple(blocks)


def canonical_blocks(max_frequency: int, channels: int) -> tuple[RepresentationBlock, ...]:
    """One real O(2) copy at each positive frequency, with the even scalar."""
    if max_frequency < 0 or channels <= 0:
        raise ValueError("invalid frequency or channels")
    return (
        RepresentationBlock("0e", ("scalar", 1), 0, 1, channels, 1),
        *(
            RepresentationBlock(f"{m}", ("freq", m), m, 2, channels, 1)
            for m in range(1, max_frequency + 1)
        ),
    )


class FullParityPathWeightedTensorProductO2(torch.nn.Module):
    """Depthwise path-weighted O(2) tensor product with full parity copies.

    Multiplicity contractions use a deterministic cyclic map from each output
    channel/copy to one channel/copy on each input.  This keeps the operator
    linear in edge count and approximately linear in the full-parity width,
    matching the path-weighted benchmark boundary instead of introducing a
    cubic fully-connected multiplicity cost.
    """

    def __init__(
        self,
        o2_module: Any,
        channels: int,
        max_frequency: int,
        *,
        parity: str = "full",
        normalization: str = "component",
        internal_compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.o2 = o2_module
        self.channels = int(channels)
        self.max_frequency = int(max_frequency)
        if parity not in ("canonical", "full"):
            raise ValueError(parity)
        self.parity = parity
        self.blocks = (
            canonical_blocks(self.max_frequency, self.channels)
            if parity == "canonical"
            else full_parity_blocks(self.max_frequency, self.channels)
        )
        self.keys = tuple(block.key for block in self.blocks)
        self.block_by_key = {block.key: block for block in self.blocks}
        self.internal_compute_dtype = internal_compute_dtype
        self.normalization = normalization

        paths: list[tuple[O2Key, O2Key, O2Key]] = []
        active = set(self.keys)
        for key1 in self.keys:
            for key2 in self.keys:
                for key3 in o2_module.o2_valid_output_keys(key1, key2):
                    if key3 in active:
                        paths.append((key1, key2, key3))
        self.paths = tuple(paths)

        self.weights = torch.nn.ParameterList()
        self._cg_names: list[str] = []
        self._index_names: list[tuple[str, str]] = []
        for path_index, (key1, key2, key3) in enumerate(self.paths):
            out_mul = self.block_by_key[key3].multiplicity
            in1_mul = self.block_by_key[key1].multiplicity
            in2_mul = self.block_by_key[key2].multiplicity
            self.weights.append(torch.nn.Parameter(torch.randn(out_mul) * 0.02))
            cg = o2_module.o2_cg_tensor(key1, key2, key3, dtype=torch.float64)
            norm = float(cg.norm().item())
            if normalization == "component" and norm > 1.0e-30:
                cg = cg * ((self.block_by_key[key3].irrep_dimension ** 0.5) / norm)
            elif normalization == "norm" and norm > 1.0e-30:
                cg = cg / norm
            cg_name = f"cg_{path_index}"
            i1_name = f"input1_index_{path_index}"
            i2_name = f"input2_index_{path_index}"
            self.register_buffer(cg_name, cg.contiguous(), persistent=True)
            self.register_buffer(i1_name, torch.arange(out_mul) % in1_mul, persistent=False)
            self.register_buffer(i2_name, torch.arange(out_mul) % in2_mul, persistent=False)
            self._cg_names.append(cg_name)
            self._index_names.append((i1_name, i2_name))

        j = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
        self.register_buffer("odd_to_even", j.transpose(0, 1).contiguous(), persistent=True)
        self.register_buffer("even_to_odd", j.contiguous(), persistent=True)

    @property
    def representation_dimension(self) -> int:
        return sum(block.multiplicity * block.irrep_dimension for block in self.blocks)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def representation_spec(self) -> list[dict[str, int | str]]:
        rows: list[dict[str, int | str]] = []
        for block in self.blocks:
            if block.frequency == 0:
                labels = block.token
            else:
                labels = (
                    str(block.frequency)
                    if self.parity == "canonical"
                    else f"{block.frequency}e+{block.frequency}o"
                )
            rows.append(
                {
                    "labels": labels,
                    "frequency": block.frequency,
                    "irrep_dimension": block.irrep_dimension,
                    "multiplicity": block.multiplicity,
                    "parity_copies": block.parity_copies,
                }
            )
        return rows

    def _canonicalize(self, key: O2Key, value: torch.Tensor) -> torch.Tensor:
        if key[0] != "freq" or self.parity == "canonical":
            return value
        even, odd = value.split(self.channels, dim=-2)
        transform = self.odd_to_even.to(device=value.device, dtype=value.dtype)
        odd_canonical = torch.einsum("...ci,ij->...cj", odd, transform)
        return torch.cat((even, odd_canonical), dim=-2)

    def _restore_copy_bases(self, key: O2Key, value: torch.Tensor) -> torch.Tensor:
        if key[0] != "freq" or self.parity == "canonical":
            return value
        even, odd = value.split(self.channels, dim=-2)
        transform = self.even_to_odd.to(device=value.device, dtype=value.dtype)
        odd_basis = torch.einsum("...ci,ij->...cj", odd, transform)
        return torch.cat((even, odd_basis), dim=-2)

    def forward(
        self,
        left: tuple[torch.Tensor, ...],
        right: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(left) != len(self.keys) or len(right) != len(self.keys):
            raise ValueError("one tensor per representation block is required")
        left_by_key = {
            key: self._canonicalize(key, value) for key, value in zip(self.keys, left)
        }
        right_by_key = {
            key: self._canonicalize(key, value) for key, value in zip(self.keys, right)
        }
        sample = left[0]
        batch_shape = sample.shape[:-2]
        output = {
            block.key: torch.zeros(
                *batch_shape,
                block.multiplicity,
                block.irrep_dimension,
                dtype=sample.dtype,
                device=sample.device,
            )
            for block in self.blocks
        }
        compute_dtype = self.internal_compute_dtype
        for path_index, (key1, key2, key3) in enumerate(self.paths):
            i1_name, i2_name = self._index_names[path_index]
            index1 = getattr(self, i1_name)
            index2 = getattr(self, i2_name)
            a = left_by_key[key1].index_select(-2, index1).to(compute_dtype)
            b = right_by_key[key2].index_select(-2, index2).to(compute_dtype)
            cg = getattr(self, self._cg_names[path_index]).to(
                device=sample.device, dtype=compute_dtype
            )
            weight = self.weights[path_index].to(compute_dtype)
            value = torch.einsum("...om,...on,mnk,o->...ok", a, b, cg, weight)
            output[key3] = output[key3] + value.to(sample.dtype)
        return tuple(self._restore_copy_bases(key, output[key]) for key in self.keys)


def apply_full_parity_o2(
    module: FullParityPathWeightedTensorProductO2,
    tensors: Iterable[torch.Tensor],
    group_element: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Apply an O(2) element in the public even/odd copy bases."""
    outputs: list[torch.Tensor] = []
    determinant = torch.linalg.det(group_element)
    for block, value in zip(module.blocks, tensors):
        matrix = module.o2.o2_representation(group_element, block.key).to(
            device=value.device, dtype=value.dtype
        )
        if block.frequency == 0:
            outputs.append(torch.einsum("...ab,...cb->...ca", matrix, value))
            continue
        if module.parity == "canonical":
            outputs.append(torch.einsum("...ab,...cb->...ca", matrix, value))
            continue
        even, odd = value.split(module.channels, dim=-2)
        odd_matrix = determinant[..., None, None] * matrix
        even_out = torch.einsum("...ab,...cb->...ca", matrix, even)
        odd_out = torch.einsum("...ab,...cb->...ca", odd_matrix, odd)
        outputs.append(torch.cat((even_out, odd_out), dim=-2))
    return tuple(outputs)
