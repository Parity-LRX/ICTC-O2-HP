# Public API

`TensorProduct(channels, m_max, parity="full", normalization="component")`
accepts two flat tensors of identical shape `(*batch, width)` and returns one
flat tensor. `width` is available as `representation_dimension`. `parity` is
`canonical` or `full`; `normalization` follows the original `component`/`norm`
coefficient convention.

Blocks are ordered by `tp.blocks`. Scalar blocks have shape
`(*batch, channels, 1)`. Positive-frequency blocks have shape
`(*batch, channels, 2)` for canonical parity and
`(*batch, 2*channels, 2)` for full parity. Flatten the final two dimensions of
each block and concatenate to construct the default flat layout.

`m_max` is the largest O(2) angular frequency, and includes every frequency
from 0 through `m_max`. At `m_max=0`, only scalar blocks remain. The packed
width is `C * (2*m_max + 1)` for canonical parity and twice that for full parity.

For a single SO(3) angular momentum `l`, restriction to rotations about a fixed
axis gives magnetic indices `m=-l,...,l`. In a real basis, each pair `±m` forms
one two-dimensional positive-frequency block, and `m=0` is a scalar. Thus an
SO(3) basis truncated at `l_max=L` has largest axial frequency `m_max=L`, but
this does not identify the two tensor-product operators or their channel
multiplicities. Including every `l=0,...,L` gives `L-m+1` such blocks at frequency
`m`, whereas this benchmark uses the same `C` at every frequency. Reflection
and copy conventions must also be matched separately.

`TupleTensorProduct` consumes and returns a tuple containing one such tensor per
block. `ComponentMajorTensorProduct` stores frequency blocks in
`[component, channel]` or `[copy, component, channel]` order. Storage conversion
does not rotate the full-parity second-copy basis.

`ReferenceTensorProduct` is the original pure PyTorch tuple operator with a
parameter per path. A packed operator's `copy_weights_to_r5(reference)` and
`load_weights_from_r5(reference)` transfer all weights without freezing them.

All packed operators expose trainable `weight`, `parameter_count`, `paths`,
`blocks`, `keys` and `representation_dimension`. Read-only routing buffers are
nonpersistent; checkpoint keys and weight order follow the qualified 0.1.5
implementation.

CUDA FP32 input and weights on the same device select the optimized kernels.
Both input VJPs and every path-weight VJP are computed. `torch.compile` with
`fullgraph=True, dynamic=False` is supported on the qualified runtime.

Packed operators import the optional Triton runtime. Install `.[hp]` to use
them. The reference operator and base package import need only PyTorch.
Internal custom-op names include the package version to keep compiled caches
separate across releases; serialized parameter keys retain the original order.
