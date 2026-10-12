# Changelog

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
