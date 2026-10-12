"""ICTC-O2 HP: standalone Cartesian O(2) tensor products."""

__version__ = "0.1.5"
__all__ = [
    "TensorProduct", "TupleTensorProduct", "ComponentMajorTensorProduct",
    "ReferenceTensorProduct", "__version__",
]


def __getattr__(name):
    # Importing the package/reference does not require the optional GPU runtime.
    if name == "ReferenceTensorProduct":
        from .reference import ReferenceTensorProduct
        return ReferenceTensorProduct
    if name in {"TensorProduct", "TupleTensorProduct", "ComponentMajorTensorProduct"}:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
