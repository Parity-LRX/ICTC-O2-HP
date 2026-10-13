# Changelog

## 0.1.6.dev0

- Add flat `UvuTensorProduct` and `ReferenceUvuTensorProduct` operators with
  independent first/second channel counts and second-input frequency/parity.
- Enumerate all legal representation-entry/copy paths; preserve u and sum v
  with trainable `(C,V)` path matrices, shared or supplied per batch row.
- Add fused FP32 CUDA forward and full VJP, bounded shared-weight reduction
  workspace and a differentiable second-derivative fallback.
- Fuse sparse angular paths for V=1–8; reuse inputs across paths/channels and
  perform the second-input channel reduction after summing paths.
- Preserve the original operators and frozen v0.1.5 benchmark cohort.
- Add `RadialUvuTensorProduct` for final radial affine layer / per-edge TP
  fusion, bounded weight/adjoint blocks and differentiable force-loss training.
- Qualify complete radial-chain gradients, rotation/reflection covariance,
  force-loss second derivatives and all-edge/all-parameter largest-shape math;
  document throughput, VRAM and the qualified outer-AOT limitation.

## Licensing metadata - 2026-10-12

- Align project-owned O2 code with ICTC-O3-HP's unmodified PolyForm
  Noncommercial 1.0.0 and separate written commercial-permission procedure.
- Add O2-specific scope and notices, preserve standard permitted-purpose
  exceptions and third-party/previously granted rights.
- Include licensing documents and SPDX metadata in source/wheel builds using
  setuptools 77 or newer. Numerical implementation and version remain 0.1.5.

## 0.1.5 - 2026-10-12

- Extend full-parity input-adjoint fusion to m_max=4–6, with one fixed
  execution policy for every positive N of a given representation.
- Retain strict FP32, Cartesian/Fischer coefficients, normalization, routing,
  both input gradients, every trainable path weight and checkpoint order.
- Complete N=16,384/32,768/65,536/131,072, C=32/64/96/128/256, m_max=1–6,
  canonical/full parity: 240 shapes, 480 candidate metrics and 60 maximum-N
  mathematical reference families. Record throughput and all memory measures.
- Include current API/execution/validation/performance/figure documentation
  and examples in the source
  distribution; keep benchmark data, tests, logs and build products outside
  the source project and wheel.
- Freeze the current source tag and qualified runtime constraints. Matching
  installable packages, validation records and scientific evidence are
  preserved in the separate private release archive.
