"""Pure PyTorch tuple reference; no optional Triton import."""

from . import o2_math
from .native_o2 import FullParityPathWeightedTensorProductO2


class ReferenceTensorProduct(FullParityPathWeightedTensorProductO2):
    """Original Cartesian path algebra, with one parameter per path."""

    def __init__(self, channels, m_max, *, parity="full", normalization="component"):
        if int(channels) != channels or channels < 1:
            raise ValueError("channels must be a positive integer")
        if int(m_max) != m_max or m_max < 0:
            raise ValueError("m_max must be a nonnegative integer")
        super().__init__(o2_math, channels, m_max, parity=parity, normalization=normalization)
