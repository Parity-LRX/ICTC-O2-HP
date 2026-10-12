"""Public operators with an internal, fixed Cartesian O(2) coefficient source."""

from . import o2_math
from .triton_o2 import (
    FlatTritonPathWeightedTensorProductO2,
    TritonPathWeightedTensorProductO2,
    XuLayoutTritonPathWeightedTensorProductO2,
)


def _validate(channels, m_max):
    if int(channels) != channels or channels < 1:
        raise ValueError("channels must be a positive integer")
    if int(m_max) != m_max or m_max < 0:
        raise ValueError("m_max must be a nonnegative integer")


class TensorProduct(FlatTritonPathWeightedTensorProductO2):
    """Flat channel-major O(2) product with all path weights trainable.

    Inputs/outputs: ``(*batch, representation_dimension)``. Positive-frequency
    blocks use ``[copy, channel, component]`` in full parity and
    ``[channel, component]`` in canonical parity. The full second copy retains
    its reflection-twisted Cartesian basis.
    """

    def __init__(self, channels, m_max, *, parity="full", normalization="component"):
        _validate(channels, m_max)
        super().__init__(o2_math, channels, m_max, parity=parity, normalization=normalization)


class TupleTensorProduct(TritonPathWeightedTensorProductO2):
    """One ``(*batch, multiplicity, irrep_dimension)`` tensor per block."""

    def __init__(self, channels, m_max, *, parity="full", normalization="component"):
        _validate(channels, m_max)
        super().__init__(o2_math, channels, m_max, parity=parity, normalization=normalization)


class ComponentMajorTensorProduct(XuLayoutTritonPathWeightedTensorProductO2):
    """Flat product with ``[copy, component, channel]`` positive-frequency blocks.

    This changes storage order only; full-parity reflection-twisted bases remain
    the same as TensorProduct. Comparing with EQX also requires its basis map.
    """

    def __init__(self, channels, m_max, *, parity="full", normalization="component"):
        _validate(channels, m_max)
        super().__init__(o2_math, channels, m_max, parity=parity, normalization=normalization)
