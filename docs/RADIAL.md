# Radial MLP to O(2) tensor product

`RadialUvuTensorProduct(tp, radial_out)` computes the same operation as
`tp(left, angular, radial_out(radial_hidden))`. It reuses the supplied final
`torch.nn.Linear` and its parameters, preserving TP path order, normalization,
copy bases and coefficient values.

The TP uses `channels_in2=1`, `shared_weights=False` and external weights.
The angular input has one channel per representation copy. `radial_hidden`
contains invariant features from the preceding radial network; its width H
is independent of the TP's second-input channel count V.

For each path and retained channel:

```text
weight[edge,p,u] = sum_h radial_hidden[edge,h] * alpha[p,u,h] + bias[p,u]
output[edge,k,u] += weight[edge,p,u] * CG(left[edge,i,u], angular[edge,j])
```

## Execution and derivatives

CUDA float32 uses a block cap determined by TP path/channel sizes. C128–256
uses at most 8192 edges and a 2 GiB weight budget so joint TP adjoints can keep
local weights and their gradients live together. Other channels use at most
16384 edges and a 4 GiB budget. These policies are independent of total N.
The affine projection and TP run together inside one operator.
Inputs and affine parameters are saved. When TP-input adjoints may be needed,
the first bounded weight block is also saved. Backward reuses that first block
and regenerates remaining blocks only when input adjoints require their weights,
computes the TP adjoints and immediately contracts the
weight adjoint into radial-hidden and affine-parameter gradients. Generated
weight storage is bounded by the block cap, rather than growing with total N;
weight adjoints are not saved between blocks.

The complete block is compiled with bounded temporary lifetimes. Joint TP
adjoints keep local weights and their gradients live together. The main bound is proportional
to `block_cap * tp.weight_numel`, in addition to model inputs, outputs, gradients
and the first block's bounded cache. The cache is disabled for inference.
This is operator fusion with bounded temporary buffers; it still generates
local weight blocks and does not claim a single GPU kernel.

The operator differentiates both TP inputs, radial hidden features, the final
affine matrix and its optional bias. Second derivatives use multilinear
adjoint identities and the same bounded execution, rather than regenerating
a full edge-weight tensor. Gradients through earlier radial layers and through
angular/radial geometry are handled by PyTorch.

[Adjoint scheduling](RADIAL_GRAPH.md) explains actual-target graph pruning and
the shared coupling expressions used by the joint angular/weight adjoints.

Compiled first-order VJPs are qualified on PyTorch 2.7.1. Its outer AOT
autograd does not support double backward: an energy/force-loss test on an
outer `torch.compile(model)` raises that runtime error. For force-loss
training, use an ordinary outer PyTorch module containing this operator.
Its bounded affine/TP blocks remain compiled internally, and second-order
gradients through the radial trunk and angular geometry are qualified in
that execution mode.

Both inputs and radial hidden features must have the same leading batch shape;
module parameters and inputs must share dtype and device. Noncontiguous inputs
are packed when required. CPU and other dtypes use the differentiable original
composition. A `ReferenceUvuTensorProduct` also selects that composition.
The optimized CUDA path uses PyTorch's float32 matrix-multiplication settings;
qualification disables TF32 and AMP. Use the settings in the example below.

## Usage

```python
import torch
from ictc_o2_hp import UvuTensorProduct, RadialUvuTensorProduct

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

tp = UvuTensorProduct(128, 1, 6, parity="full", parity_in2="canonical",
                      shared_weights=False).cuda()
radial_trunk = torch.nn.Sequential(
    torch.nn.Linear(16, 64), torch.nn.SiLU(),
    torch.nn.Linear(64, 64), torch.nn.SiLU(),
).cuda()
radial_out = torch.nn.Linear(64, tp.weight_numel, bias=False).cuda()
fused = RadialUvuTensorProduct(tp, radial_out)

# x and angular are per-edge TP inputs; radial_basis holds 16 invariant features.
output = fused(x, angular, radial_trunk(radial_basis))
```

The preceding trunk remains an ordinary trainable PyTorch module. The wrapper
does not apply an activation after `radial_out`; any preceding activation
belongs in the trunk. Existing last-layer weights and bias retain their shapes.
In the wrapper state dictionary they appear under `radial_out`, with the TP
under `tp`. An activation or nonlinear operation after the final affine layer
cannot be moved into this wrapper unchanged.

N in this chain counts edges, not atoms. A complete message-passing layer also
includes neighbor gathering, scattering and other model operations. Those are
outside the radial-MLP/TP chain measured here.

[Measured throughput and memory](RADIAL_PERFORMANCE.md) records the qualified
whole-chain comparison with actual EquivariantX, including timing variability,
ordinary ICTC composition and maximum-shape OOM results.
