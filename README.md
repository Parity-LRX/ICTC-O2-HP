# ICTC-O2 HP

Independent private Cartesian O(2) tensor-product library. Distribution:
`ictc-o2-hp`; Python package: `ictc_o2_hp`. This project contains its own
coefficient source, reference algebra, packed operators and Triton kernels.

Repository: [Parity-LRX/ICTC-O2-HP](https://github.com/Parity-LRX/ICTC-O2-HP).

Frozen release: **0.1.5**, Git tag **v0.1.5**, stable branch `main`.
[Release layout and reproduction](docs/RELEASE.md) describes the current
source, qualified runtime and numerical evidence.
[CHANGELOG.md](CHANGELOG.md) records the current release changes.

Development version **0.1.6.dev0** adds `UvuTensorProduct`: it retains the first
input channel and sums weighted second-input channels. Its flat inputs may have
different channel counts, and weights can be shared or supplied per batch row.
See [uvu API](docs/UVU.md) and [example](examples/uvu.py). The frozen 0.1.5 tag and
the existing eqx performance cohort continue to describe the original operator.

`RadialUvuTensorProduct` combines the final radial affine layer with the
per-edge V=1 TP. It bounds generated weights and their adjoints for large edge
batches while preserving gradients through the complete radial network.
See [radial-weight fusion](docs/RADIAL.md).

The source project and Git repository are separate from the O(3) HP project and
the paper. It has no dependency on EquivariantX, MACE-ICTC HP, CHORUS or the paper
repository. Runtime import uses package-relative paths.

## Install and use

On the qualified Linux GPU runtime, install `.[hp]`. Builds and numerical work
in the current workflow must run through `sbatch` on GPU-A100 / XHPC-A100.

```python
import torch
from ictc_o2_hp import TensorProduct

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

tp = TensorProduct(128, 6, parity="full").cuda().float()
x = torch.randn(6400, tp.representation_dimension, device="cuda", dtype=torch.float32, requires_grad=True)
y = torch.randn_like(x, requires_grad=True)
compiled = torch.compile(tp, fullgraph=True, dynamic=False)
output = compiled(x, y)
dx, dy, dw = torch.autograd.grad(output, (x, y, tp.weight), torch.ones_like(output))
```

`canonical` means **canonical parity**, one positive-frequency copy.
`full` means **full parity**, two positive-frequency copies plus both scalar
parities. Both modes preserve their original Cartesian/Fischer coefficients,
cyclic channel/copy routing, normalization and trainable path weights.

GPU FP32 uses Triton; other device/dtype cases use the original PyTorch algebra.
Higher derivatives use the differentiable PyTorch fallback. Importing the
reference operator alone does not require Triton.

## API and evidence

The qualified 0.1.5 implementation uses an [execution policy independent of N](docs/EXECUTION.md).
It extends input-adjoint fusion to full-parity m_max=4–6. Seven installed-wheel
checks, 73 API contracts and the focused C128/full-parity promotion checks
passed. The final installed package also passed seven fresh public API checks.
The expanded m_max=1–6 grid is complete: N=16,384/32,768/65,536/131,072,
C32/64/96/128/256 and both parities, 240 shapes and 480 operator metrics.
Against EQX, 455/479 comparable medians are faster, 22 slower
and two equal; all 239 comparable full-VJP memory triples are lower. Slower
cases, timing variability and the isolated EQX OOM remain in the report.

See [API](docs/API.md), [example](examples/operators.py) and
[validation](docs/VALIDATION.md). [Figure interpretation](docs/FIGURES.md) explains
the throughput/memory ratio directions, IQR marks and 3D axes.
[Measured performance/memory](docs/PERFORMANCE.md)
includes the complete larger N/C256 scan and slower/OOM cases. Qualified runtime: PyTorch 2.7.1+cu128,
Triton 3.3.1, NVIDIA A100 40 GB, strict FP32, TF32/AMP disabled.

Benchmark comparator: **EquivariantX (EQX)**, `eqx.o2.TensorProduct`.
“Through the Looking-Glass” is the associated paper title. The library is named
EquivariantX in its [upstream README](https://github.com/xvzemin/tace/blob/ac3aed47f5b35d49f1f237912150f0d9dd759ae7/eqx/README.md).

## eqx comparison scripts, results and figures

[benchmarks/eqx](benchmarks/eqx/README.md) contains the current v0.1.5
comparison scripts, all 240 shapes / 480 operator metrics, raw measurements,
numerical-reference records and the final 2D/3D figures in PNG/SVG/PDF.
Throughput and allocated/reserved/sampled-NVML peak memory are retained
together, including slower cases and the isolated eqx OOM.

The directory includes batch templates for scanning, reporting and rendering.
These benchmark files are kept outside the installed library package.
[Exact qualified library dependencies](docs/runtime_constraints.txt) and
[benchmark dependencies](benchmarks/eqx/requirements-qualified.txt) record the
reproduction environment. General API tests, older versions, superseded
experiments, build products and compiler caches remain in the separate archive.

## License

Project-owned ICTC-O2 HP implementations use the same
[PolyForm Noncommercial 1.0.0](LICENSE) license as ICTC-O3-HP (formerly
MACE-ICTC-HP). **Commercial uses outside that license's permitted purposes
require separate written authorization** from the relevant rights holder;
see [commercial permission](COMMERCIAL_LICENSE.md).

The standard educational/public-research and other permitted-purpose
provisions apply unchanged. Third-party and previously granted permissive
rights retain their original scope; see [LICENSE_SCOPE.md](LICENSE_SCOPE.md)
and [NOTICE](NOTICE).
