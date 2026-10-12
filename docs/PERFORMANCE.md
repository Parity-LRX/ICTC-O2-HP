# Measured performance and memory

Qualified version: 0.1.5. NVIDIA A100 40 GB, PyTorch 2.7.1+cu128,
Triton 3.3.1, strict FP32, TF32/AMP disabled. `N` counts independent
tensor-product rows, rather than the cost of a complete atomistic model.

## Complete measurement grid

| Axis | Measured values |
|---|---|
| N | 16,384; 32,768; 65,536; 131,072 |
| C | 32; 64; 96; 128; 256 |
| m_max | 1 through 6 |
| Parity | canonical parity; full parity |
| Operation | forward; forward + full VJP |

All 240 shapes and 480 ICTC operator measurements completed. The 60
maximum-N mathematical-reference families check every output row, both input
gradients and all path weights. Maximum normalized error is 8.10846e-7,
below the 1e-5 gate. One fixed execution policy is used across positive N
for each representation.

The comparator is EquivariantX (eqx), `eqx.o2.TensorProduct`. Of 479
comparable operator measurements, ICTC has 455 faster medians, 22 slower
and two equal. These counts combine forward and full VJP. Medians and
IQR separation describe timing variability; they are not formal
significance tests.

## Peak memory

Full VJP includes forward, both input adjoints and every trainable path-weight
gradient. Memory is measured in fresh isolated processes. PyTorch allocated
and reserved peaks include live inputs, parameters, outputs/adjoints and the
precreated VJP cotangent, including in forward-only measurements. Sampled
NVML process memory also includes the CUDA context and allocator cache.

Against eqx, all 239 comparable full-VJP cases use less peak memory in all
three recorded measures. For the 240 forward measurements:

| Measure | ICTC lower | Equal | ICTC higher |
|---|---:|---:|---:|
| Allocated peak | 160 | 0 | 80 |
| Reserved peak | 160 | 80 | 0 |
| Sampled NVML process peak | 160 | 80 | 0 |

The largest forward allocated-memory increase is 0.628418 MiB. Rounded
ratios near 1.00 do not imply exact equality. Throughput and memory must be
assessed together.

## Largest-N comparison

At N=131,072, forward has 55/60 faster medians and five slower; full VJP
has 56/59 faster medians and three slower. Full-VJP throughput ratios
(ICTC / eqx) across C=32/64/96/128/256 are:

| m_max | canonical parity | full parity |
|---:|---:|---:|
| 1 | 1.04–1.21x | 1.11–1.17x |
| 2 | 1.24–1.55x | 1.20–1.29x |
| 3 | 1.27–1.58x | 1.18–1.27x |
| 4 | 0.97–1.20x | 1.22–1.29x |
| 5 | 0.96–1.21x | 1.19–1.31x* |
| 6 | 0.89–1.11x | 1.51–1.61x** |

The three slower full-VJP medians are C32/canonical/m_max=4–6.
At C128/full/m_max=6, paired full VJP is 60.348 ms for ICTC and 91.244 ms
for eqx, a throughput ratio of 1.512x.

\* For C256/full/m_max=5/N=131,072, the joint process exceeded memory;
both libraries succeeded in separate processes. Its ratio uses isolated
timing and does not contribute a paired-IQR win.

\*\* eqx failed its isolated full VJP at C256/full/m_max=6/N=131,072.
That configuration has no finite comparison ratio and is excluded from
this range. ICTC completed it successfully.

Raw samples, measurement scripts, reports and figures remain in the
separate release archive. See [validation](VALIDATION.md),
[figure interpretation](FIGURES.md) and [release notes](RELEASE.md).
