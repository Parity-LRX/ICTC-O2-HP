# Memory measurement state

Peak allocated/reserved are total PyTorch allocator peaks in a fresh isolated
benchmark process, including live inputs, trainable parameters, output/input/
weight gradients when present, routing metadata and operation workspace.
The precreated VJP cotangent is retained during both forward-only and full-VJP
measurements. Forward calls execute under torch.no_grad(). All implementation
comparisons use the same benchmark-state convention and public native layouts.
Preparation happens before timing; memory peaks count all retained allocations.

NVML measures sampled process memory, including CUDA context/allocator cache;
the requested sampling interval is50ms, with observed intervals and sample
counts in every record. Short transients may be missed. OOM records preserve
failure statistics separately; they supply no successful peak or throughput.

The current figures compare ICTC-O2 HP 0.1.5 with EquivariantX.
Exact increases are retained even if a figure rounds a ratio to1.00. An EQX
joint-process OOM is distinguished from an isolated OOM, and isolated medians
are excluded from paired-IQR win counts.
