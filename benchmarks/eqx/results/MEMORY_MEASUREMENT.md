# Memory measurement state

PyTorch peak allocated and reserved memory are total allocator peaks in a
fresh isolated process. They include live inputs, trainable parameters,
outputs/adjoints, routing metadata and workspace. The precreated VJP cotangent
is retained during both forward-only and full-VJP measurement. Forward calls
use `torch.no_grad()`; full VJP computes both input gradients and every path
weight gradient. Layout preparation precedes timing.

NVML process memory includes the CUDA context and allocator cache. Sampling
is requested every 50 ms; records retain observed intervals, sample counts
and sampling errors. Short transients may be missed.

The current comparison uses ICTC-O2 HP 0.1.5 and EquivariantX, strict FP32,
TF32/AMP disabled. OOM failure statistics are distinct from successful peaks.
Joint-process OOM does not establish isolated-process OOM. An isolated timing
fallback is marked and excluded from paired-IQR win counts. Exact memory
increases remain recorded even when the displayed ratio rounds to 1.00x.
