# Release v0.1.5

The private repository is [Parity-LRX/ICTC-O2-HP](https://github.com/Parity-LRX/ICTC-O2-HP).
Stable branch: `main`. Annotated release tag: `v0.1.5`.
Distribution `ictc-o2-hp` and Python package `ictc_o2_hp` both report 0.1.5.

This repository starts with a single current-release snapshot. The eight
Python modules are identical to the qualified 0.1.5 implementation. Original
development commits, earlier source versions, tests, benchmark/plotting scripts
and generated artifacts are retained in the separate private archive.

## Source layout

| Location | Purpose |
|---|---|
| `src/ictc_o2_hp/api.py`, `__init__.py` | Public constructors and version |
| `o2_math.py`, `native_o2.py`, `reference.py` | Coefficients and reference algebra |
| `fast_o2.py`, `runtime_weight.py`, `triton_o2.py` | Packed execution and GPU kernels |
| `examples/operators.py` | Minimal forward and gradient usage example |
| `docs/` | Current API, execution, validation, performance and figure interpretation |
| `docs/runtime_constraints.txt` | Exact qualified runtime/build dependencies |
| `pyproject.toml`, `MANIFEST.in`, `NOTICE` | Packaging and private-use notice |

The installed wheel contains the library and its distribution metadata.
EquivariantX is an external comparator, rather than a package dependency.

## Qualified runtime

The measured environment uses Linux, Python 3.11.15, PyTorch 2.7.1+cu128,
Triton 3.3.1 and an NVIDIA A100 40 GB. The exact installed dependency versions
are recorded in [runtime_constraints.txt](runtime_constraints.txt). CUDA/GPU
runtime requirements must be matched when reproducing the measurements.

In the qualified environment, install the current source with
`python -m pip install --no-deps .`. PyTorch and Triton are already supplied
by that environment. The standard optional-dependency installation is
`python -m pip install '.[hp]'`.

Builds and numerical verification in the project workflow run through
`sbatch` on GPU-A100 / XHPC-A100. The Mac is used for source/document edits,
Git and file synchronization. Qualification uses strict FP32 with TF32/AMP
disabled.

## Validation and release archive

Final release job 90912 built and privately installed the wheel and source
distribution, passed package-content/version checks and seven fresh public
API checks. All eight Python source modules matched the qualified wheel.
The archive also preserves the original 73 API contracts, 60 maximum-N
reference families and complete 240-shape/480-metric scan, including throughput
and allocated/reserved/sampled-process peak memory.

Installable packages, full Git backup, environment records, validation results,
raw measurements and final figures are preserved together in the private
release archive. They are separate from this source repository. See
[validation](VALIDATION.md) and [measured performance](PERFORMANCE.md).

Keep `v0.1.5` fixed. Later numerical or public-API changes receive a new
package version, custom-op namespace and qualification. Only the current
source and documentation should be committed to this repository.
