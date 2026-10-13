# O(2) uvu tensor product

`UvuTensorProduct(channels, channels_in2, m_max, *, parity="full",
parity_in2=None, m_max_in2=None, normalization="component",
path_normalization="none", internal_weights=None, shared_weights=True)`

For each legal representation-entry path `p=(i,j,k)`, the operator computes

```text
output[k, u] += sum_v weight[p, u, v] * CG(input1[i, u], input2[j, v])
```

The first input and output have `channels=C` per representation copy; the second
input has `channels_in2=V`. This is the standard uvu multiplicity contraction.
`V=1` is the single-channel second-input case often used for angular geometry.
For fixed angular paths, its channel arithmetic scales as `C*V`, rather than the
`C_in1*C_in2*C_out` cost of a fully connected uvw product.

## Representations and routing

The first input and output contain every frequency from zero through `m_max`.
The second uses `m_max_in2` (default `m_max`) and `parity_in2` (default `parity`).
`canonical` contains `0e` and one copy at each positive frequency; `full`
contains both `0e` and `0o`, and two copies at each positive frequency. Positive
copies use the same reflection-twisted public Cartesian bases as the existing
operators. Flat blocks use `[copy, channel, component]` order.

Each copy is a separate representation entry. Every legal input-entry/output-
entry combination is retained, including cross-copy paths. The channel index
`u` is preserved within each entry; the second channel index `v` is summed.
This differs from the old operator's fixed cyclic copy routing, so its full-
parity path count and parameter order are different. No old checkpoint is
implicitly reinterpreted as a uvu checkpoint.

Paths are ordered by output entry, first-input entry, then second-input entry.
`tp.paths` stores entry-index triples; `entries_in1`, `entries_in2`, and
`entries_out` provide each entry's irrep key, copy index, offset and dimension.
Only requested output frequencies are produced; frequency truncation preserves
O(2) rotation and reflection equivariance.

## Inputs and weights

Both inputs have the same leading batch shape, with last dimensions given by
`input1_dimension` and `input2_dimension`. The output's last dimension is
`representation_dimension`, equal to `input1_dimension`.

There are `weight_numel = len(paths)*C*V` weights, in `[path,u,v]` order.
`weight_view_for_path(p, weight=None)` returns a `(...,C,V)` view. All path
weights remain differentiable.

- `shared_weights=True`: use a flat `(weight_numel,)` weight vector. Internal
  trainable weights are enabled by default; external weights can override them.
- `shared_weights=False`: internal weights default to disabled. Supply weights
  of shape `(*batch,weight_numel)` on each forward call, for example from a radial
  network. Internal weights with this mode are rejected.

Inputs, weights and module must have the same device and real floating-point
dtype. Noncontiguous input layouts and multidimensional batch shapes are
accepted. Batch dimensions must match exactly; implicit broadcasting is not
part of this API.

`normalization` follows the original library's Cartesian coefficient convention.
`component` gives each CG output component unit variance for independent unit-
variance input components; `norm` makes the CG tensor's Frobenius norm one.
`path_normalization="none"` adds no extra path scaling. `element` and `path`
divide each path by `sqrt(V * number_of_paths_to_that_output_entry)`; they coincide
because the API uses a uniform second-input multiplicity.

## Execution and derivatives

CUDA float32 uses fused Triton forward, both input adjoints and path-weight
adjoints. For `V<=8`, sparse compile-time coupling coefficients fuse every
output entry in the forward and every first-input entry in its adjoint. The
second-input adjoint sums all paths before a single channel reduction for each
entry, reusing the first input and output cotangent across v channels. For
V=1, per-row weight adjoints group paths by output entry, reusing feature
loads and cotangents while omitting zero CG terms. External V=1 weights with
C128–256 use joint angular/weight adjoints when both are requested. They share
the first-input/output-cotangent contraction; the left adjoint can be included
in the same launches. Joint execution keeps weights and their gradients live
together. [Radial fusion](RADIAL.md) bounds these buffers for large edge batches.
For
multi-channel inputs, the v projection precedes angular coupling. Larger V
uses runtime path metadata. Input adjoints recompute the required contractions
rather than saving row-by-path projections. Shared
weight gradients use a fixed 16-way reduction with workspace `16*weight_numel`
float32 values. No row-by-path-by-C-by-V outer product is stored.
Eager autograd evaluates only gradient roles in the active target graph.
Compilation, tracing and unsupported engine queries conservatively retain all
declared roles; parameters and paths remain trainable.

The launch policy depends on representation/channel sizes, not on the number
of batch rows. No TF32 or reduced-precision dot product is used. CPU and other
floating-point dtypes use differentiable PyTorch. Second derivatives use a
PyTorch recomputation of the backward. `torch.compile(fullgraph=True,
dynamic=False)` is qualified for first-order VJPs on the Linux runtime.
For second derivatives, keep the outer model in ordinary PyTorch; the
qualified PyTorch 2.7.1 AOT limitation is described in [radial execution](RADIAL.md).

`ReferenceUvuTensorProduct` exposes the same API using PyTorch only and does not
import Triton. Its v projection is also factored before angular coupling; it is
a mathematical/reference execution path rather than a bound on peak memory.

The existing `TensorProduct`, tuple and component-major APIs keep their original
uuu-style routing. The v0.1.5 eqx figures measure those operators and are not uvu
performance claims.

## Example

```python
import torch
from ictc_o2_hp import UvuTensorProduct

tp = UvuTensorProduct(128, 1, 6, parity="full", parity_in2="canonical").cuda()
x = torch.randn(16384, tp.input1_dimension, device="cuda", requires_grad=True)
y = torch.randn(16384, tp.input2_dimension, device="cuda", requires_grad=True)
output = torch.compile(tp, fullgraph=True, dynamic=False)(x, y)
dx, dy, dw = torch.autograd.grad(output, (x, y, tp.weight), torch.ones_like(output))
```

[examples/uvu.py](../examples/uvu.py) supplies per-row weights from a radial
network and differentiates through its parameters and radial inputs.

[RadialUvuTensorProduct](RADIAL.md) combines that final affine layer with a
V=1 TP and reduces the memory required by large per-edge weight tensors.
