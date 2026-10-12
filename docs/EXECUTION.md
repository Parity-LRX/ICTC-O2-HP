# Execution policy independent of input size

Version 0.1.5 uses an execution policy independent of N and full-parity
input-adjoint fusion for m_max=4–6. Qualification covers four N values,
five C values, m_max=1–6 and both parities: 240 shapes and 480 operator metrics.
The measurements use the default policy without experimental overrides.

For a fixed representation (C, m_max, parity), every positive N selects the
same forward and first-VJP kernel family and the same per-CTA tiles. N changes
the launch grid, boundary masks, buffer lengths and reduction loop extent.
The empty-input branch returns empty input adjoints and zero weight adjoints;
it is a correctness case, not a positive-N optimization threshold.

The default policy is:

| Work | Fixed policy for all positive N |
|---|---|
| m_max 0–3 forward | all-output fusion, block 256 |
| m_max 0–3 input VJP | all-input fusion, block 128 |
| Full-parity m_max 4–6 input VJP | all-input fusion, block 128 |
| Higher-order forward and other input VJPs | existing representation-dependent kernels |
| Weight contributions | grouped first-input paths, row 16/channel 64 tiles |
| Weight reduction, first stage | 1024 partial rows by 32 weights per CTA |
| Weight reduction, final stage | 32 partial rows by 32 weights per CTA, tiled loop |

Both weight-reduction stages execute even when there is only one chunk.
This can add launch/buffer overhead at small N; numerical and performance
qualification must include small N rather than silently switching it away.
The intermediate allocation is proportional to ceil(ceil(N/16)/1024) times
the packed weight count. It must be included in peak allocated/reserved and
sampled process-memory accounting.

Cartesian/Fischer coefficients, normalization, channel/copy routing and public
checkpoint keys are unchanged. Forward and first VJP remain FP32, including
all row reductions, both input gradients and every trainable path weight.
TF32/AMP are disabled for qualification. Release-specific custom-op namespaces
keep compiled lowerings separate across package versions.

Environment overrides select a fixed policy for the whole process; none
selects a strategy from N. The measured default uses the grouped schedule.

The full-parity m_max=4–6 families passed small/large-N profiler checks:
six settings, three complete correlated VJPs each, plus three small-N
mathematical-reference gates. The completed scan passed 60 maximum-N
all-row/all-weight reference families at a normalized-error gate of 1e-5.
Timing uses CUDA events outside profiling; memory uses separate fresh processes.
