# Release v0.1.5

This page describes the frozen v0.1.5 release. The separate `codex/uvu-mode`
development branch reports 0.1.6.dev0 and adds the [uvu operator](UVU.md).
Its qualification and timings are separate from the v0.1.5 eqx cohort.

The private repository is [Parity-LRX/ICTC-O2-HP](https://github.com/Parity-LRX/ICTC-O2-HP).
Stable branch: `main`. Annotated release tag: `v0.1.5`.
Distribution `ictc-o2-hp` and Python package `ictc_o2_hp` both report 0.1.5.

This repository starts with a single current-release snapshot. The eight
Python modules are identical to the qualified 0.1.5 implementation. Original
development commits, earlier source versions, general API tests and superseded
experiments are retained in the separate private archive. The current v0.1.5
eqx benchmark scripts, recorded results and final figures are now included in
[benchmarks/eqx](../benchmarks/eqx/README.md) on main. The release tag remains fixed.

## Source layout

| Location | Purpose |
|---|---|
| `src/ictc_o2_hp/api.py`, `__init__.py` | Public constructors and version |
| `o2_math.py`, `native_o2.py`, `reference.py` | Coefficients and reference algebra |
| `fast_o2.py`, `runtime_weight.py`, `triton_o2.py` | Packed execution and GPU kernels |
| `examples/operators.py` | Minimal forward and gradient usage example |
| `docs/` | Current API, execution, validation, performance and figure interpretation |
| `docs/runtime_constraints.txt` | Exact qualified runtime/build dependencies |
| `benchmarks/eqx/` | Current comparison scripts, raw/report data and final figures |
| `pyproject.toml`, `MANIFEST.in` | Packaging and license-file inclusion |
| `LICENSE`, `NOTICE`, `LICENSE_SCOPE.md`, `COMMERCIAL_LICENSE.md` | PolyForm Noncommercial terms, scope and separate commercial authorization |

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
release archive. The current eqx evidence is also available in this repository;
compiler caches, job logs, historical controls and installable build products
remain separate. Benchmark files are excluded from the wheel/source package. See
[validation](VALIDATION.md) and [measured performance](PERFORMANCE.md).

Keep `v0.1.5` fixed. Later numerical or public-API changes receive a new
package version, custom-op namespace and qualification. Commit current source,
documentation and the explicitly included eqx comparison evidence.

The current main branch offers project-owned code under the same unmodified
PolyForm Noncommercial 1.0.0 license as ICTC-O3-HP. Both wheel and source
distribution include all four licensing documents and the SPDX license
expression. The original release tag and archived build artifacts remain fixed;
the licensing metadata update is recorded separately from those frozen objects.
