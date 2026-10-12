"""Closed-form execution of the r5 path-weighted O(2) tensor product.

This keeps r5's paths, channel rule, weights, normalization, and public parity
bases.  The small real CG tensors are expanded into scalar products once in
``__init__``; no per-edge ``einsum`` or channel gather remains in ``forward``.
The original ``native_o2.py`` is retained as the reference implementation.
"""

from __future__ import annotations

import torch

from .native_o2 import FullParityPathWeightedTensorProductO2, O2Key


class FastPathWeightedTensorProductO2(FullParityPathWeightedTensorProductO2):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        sparse_cg = []
        for name in self._cg_names:
            cg = getattr(self, name)
            path_terms = []
            for output_component in range(cg.shape[-1]):
                terms = []
                for first_component in range(cg.shape[0]):
                    for second_component in range(cg.shape[1]):
                        coefficient = float(
                            cg[first_component, second_component, output_component]
                        )
                        if abs(coefficient) > 1.0e-12:
                            terms.append(
                                (first_component, second_component, coefficient)
                            )
                path_terms.append(tuple(terms))
            sparse_cg.append(tuple(path_terms))
        self._sparse_cg = tuple(sparse_cg)

    def _path_weight(self, path_index: int) -> torch.Tensor:
        return self.weights[path_index]

    def _cg_apply(
        self, path_index: int, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        components = []
        for terms in self._sparse_cg[path_index]:
            component = None
            for i, j, coefficient in terms:
                term = a[..., i] * b[..., j]
                if coefficient != 1.0:
                    term = term * coefficient
                component = term if component is None else component + term
            if component is None:
                component = a[..., 0] * b[..., 0] * 0.0
            components.append(component)
        return (
            components[0].unsqueeze(-1)
            if len(components) == 1
            else torch.stack(components, dim=-1)
        )

    def _canonicalize_fast(self, key: O2Key, value: torch.Tensor) -> torch.Tensor:
        if key[0] != "freq" or self.parity == "canonical":
            return value
        even, odd = value.split(self.channels, dim=-2)
        rotated = torch.stack((-odd[..., 1], odd[..., 0]), dim=-1)
        return torch.cat((even, rotated), dim=-2)

    def _restore_fast(self, key: O2Key, value: torch.Tensor) -> torch.Tensor:
        if key[0] != "freq" or self.parity == "canonical":
            return value
        even, odd = value.split(self.channels, dim=-2)
        rotated = torch.stack((odd[..., 1], -odd[..., 0]), dim=-1)
        return torch.cat((even, rotated), dim=-2)

    def _finalize_outputs(
        self, result: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        return result

    def forward(
        self,
        left: tuple[torch.Tensor, ...],
        right: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(left) != len(self.keys) or len(right) != len(self.keys):
            raise ValueError("one tensor per representation block is required")
        compute_dtype = self.internal_compute_dtype
        first = {
            key: self._canonicalize_fast(key, value).to(compute_dtype)
            for key, value in zip(self.keys, left)
        }
        second = {
            key: self._canonicalize_fast(key, value).to(compute_dtype)
            for key, value in zip(self.keys, right)
        }
        output: dict[O2Key, torch.Tensor | None] = {
            key: None for key in self.keys
        }
        batch_shape = left[0].shape[:-2]
        for path_index, (key1, key2, key3) in enumerate(self.paths):
            output_block = self.block_by_key[key3]
            copies = output_block.parity_copies
            first_block = self.block_by_key[key1]
            second_block = self.block_by_key[key2]
            a = first[key1].reshape(
                *batch_shape,
                first_block.parity_copies,
                self.channels,
                first_block.irrep_dimension,
            )
            b = second[key2].reshape(
                *batch_shape,
                second_block.parity_copies,
                self.channels,
                second_block.irrep_dimension,
            )
            if first_block.parity_copies > copies:
                a = a[..., :copies, :, :]
            if second_block.parity_copies > copies:
                b = b[..., :copies, :, :]

            value = self._cg_apply(path_index, a, b)
            weight = self._path_weight(path_index).to(compute_dtype).reshape(
                *((1,) * len(batch_shape)), copies, self.channels, 1
            )
            value = value * weight
            output[key3] = value if output[key3] is None else output[key3] + value

        result = []
        for block in self.blocks:
            value = output[block.key]
            if value is None:
                value = torch.zeros(
                    *batch_shape,
                    block.parity_copies,
                    self.channels,
                    block.irrep_dimension,
                    dtype=compute_dtype,
                    device=left[0].device,
                )
            result.append(
                self._restore_fast(
                    block.key,
                    value.reshape(
                        *batch_shape, block.multiplicity, block.irrep_dimension
                    ).to(left[0].dtype),
                )
            )
        return self._finalize_outputs(tuple(result))


class PackedFastPathWeightedTensorProductO2(FastPathWeightedTensorProductO2):
    """The same arithmetic with r5's path parameters in one flat tensor."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        offsets = []
        cursor = 0
        initial = []
        for parameter in self.weights:
            next_cursor = cursor + parameter.numel()
            offsets.append((cursor, next_cursor))
            initial.append(parameter.detach())
            cursor = next_cursor
        self._weight_offsets = tuple(offsets)
        del self.weights
        self.weight = torch.nn.Parameter(torch.cat(initial))

    def _path_weight(self, path_index: int) -> torch.Tensor:
        start, end = self._weight_offsets[path_index]
        return self.weight[start:end]

    def copy_weights_to_r5(self, reference: FullParityPathWeightedTensorProductO2) -> None:
        if len(reference.weights) != len(self._weight_offsets):
            raise ValueError("r5 path count differs")
        with torch.no_grad():
            for parameter, (start, end) in zip(reference.weights, self._weight_offsets):
                parameter.copy_(self.weight[start:end])

    def load_weights_from_r5(self, reference: FullParityPathWeightedTensorProductO2) -> None:
        """Copy the original per-path parameters into the packed layout."""
        if len(reference.weights) != len(self._weight_offsets):
            raise ValueError("r5 path count differs")
        with torch.no_grad():
            for parameter, (start, end) in zip(reference.weights, self._weight_offsets):
                self.weight[start:end].copy_(parameter)


class FormulaPackedPathWeightedTensorProductO2(PackedFastPathWeightedTensorProductO2):
    """Pack the weights and factor each sparse CG map into one scalar scale."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        formulas = []
        for path_index, (key1, key2, _key3) in enumerate(self.paths):
            terms = self._sparse_cg[path_index]
            coefficients = [abs(c) for component in terms for _, _, c in component]
            scale = max(coefficients)
            if any(abs(c - scale) > 1.0e-8 for c in coefficients):
                formulas.append(None)
                continue
            signed = tuple(
                tuple((i, j, 1 if c > 0 else -1) for i, j, c in component)
                for component in terms
            )
            dim1 = self.block_by_key[key1].irrep_dimension
            dim2 = self.block_by_key[key2].irrep_dimension
            signature = tuple(
                (i, j, k, sign)
                for k, component in enumerate(signed)
                for i, j, sign in component
            )
            special = None
            for sign in (1, -1):
                if dim1 == 1 and dim2 == 2:
                    if signature == ((0, 0, 0, sign), (0, 1, 1, sign)):
                        special = ("scalar_left_identity", sign)
                    elif signature == ((0, 1, 0, -sign), (0, 0, 1, sign)):
                        special = ("scalar_left_quarter", sign)
                elif dim1 == 2 and dim2 == 1:
                    if signature == ((0, 0, 0, sign), (1, 0, 1, sign)):
                        special = ("scalar_right_identity", sign)
                    elif signature == ((1, 0, 0, -sign), (0, 0, 1, sign)):
                        special = ("scalar_right_quarter", sign)
            formulas.append((scale, signed, special))
        self._formulas = tuple(formulas)

    def _cg_apply(
        self, path_index: int, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        formula = self._formulas[path_index]
        if formula is None:
            return super()._cg_apply(path_index, a, b)
        scale, signed, special = formula
        if special is not None:
            kind, sign = special
            if kind == "scalar_left_identity":
                return (a[..., 0:1] * b) * (sign * scale)
            if kind == "scalar_right_identity":
                return (a * b[..., 0:1]) * (sign * scale)
            if kind == "scalar_left_quarter":
                rotated = torch.stack((-b[..., 1], b[..., 0]), dim=-1)
                return (a[..., 0:1] * rotated) * (sign * scale)
            rotated = torch.stack((-a[..., 1], a[..., 0]), dim=-1)
            return (rotated * b[..., 0:1]) * (sign * scale)
        components = []
        for terms in signed:
            component = None
            for i, j, sign in terms:
                product = a[..., i] * b[..., j]
                term = product if sign > 0 else -product
                component = term if component is None else component + term
            if component is None:
                return super()._cg_apply(path_index, a, b)
            components.append(component)
        result = (
            components[0].unsqueeze(-1)
            if len(components) == 1
            else torch.stack(components, dim=-1)
        )
        return result * scale


class FlatFormulaPackedPathWeightedTensorProductO2(FormulaPackedPathWeightedTensorProductO2):
    """Return tuple views of one packed output allocation."""

    def _finalize_outputs(
        self, result: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        flat = torch.cat([value.flatten(start_dim=-2) for value in result], dim=-1)
        widths = tuple(
            block.multiplicity * block.irrep_dimension for block in self.blocks
        )
        return tuple(
            value.reshape(*value.shape[:-1], block.multiplicity, block.irrep_dimension)
            for value, block in zip(torch.split(flat, widths, dim=-1), self.blocks)
        )
