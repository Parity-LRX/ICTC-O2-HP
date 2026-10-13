# ICTC-O2 HP radial MLP + TP vs EquivariantX

Development version 0.1.6.dev0, after joint-adjoint and requested-gradient scheduling.
A100-SXM4 40GB; PyTorch 2.7.1+cu128; Triton 3.3.1; strict FP32, TF32/AMP disabled.
Job 91260 uses a fresh process per implementation/metric, two warmups and seven
CUDA-event samples. Both complete chains use torch.compile(fullgraph=True,dynamic=False).

The chain is Linear(16,64) -> SiLU -> Linear(64,64) -> SiLU ->
Linear(64,P*C,bias=False) -> V=1 TP. First/output full parity, angular
canonical parity, m_max=6. N counts edges. Gathering/scattering and the
rest of a message-passing model are outside the measured chain.

EQX uses its actual native u1u with external per-edge weights. Every TP
path and network parameter is matched. Basis/layout conversion is outside
timing; phase differences are absorbed into the final affine parameters.
Full VJP includes forward, all three input gradients and every network parameter.

## Current comparison

Speed is EQX time divided by ICTC time; values above one favor ICTC.

| N edges | C | metric | ICTC bounded ms | EQX ms | EQX/ICTC speed | ICTC allocated GiB | EQX allocated GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 16,384 | 64 | forward | 3.822 | 4.020 | 1.05× | 1.31 | 1.57 |
| 16,384 | 64 | full_vjp | 15.101 | 14.948 | 0.99× | 2.44 | 3.88 |
| 32,768 | 128 | forward | 13.304 | 15.740 | 1.18× | 2.03 | 6.24 |
| 32,768 | 128 | full_vjp | 52.221 | 57.816 | 1.11× | 4.78 | 15.40 |
| 131,072 | 256 | forward | 105.482 | OOM | — | 8.96 | — |
| 131,072 | 256 | full_vjp | 436.218 | OOM | — | 19.39 | — |

At C128, forward is about 18% faster and full VJP about 11% faster.
The full-VJP allocated peak is 31.1% of EQX's
(EQX/ICTC memory ratio 3.22×).
C64 full VJP is about 1% slower; its speed is effectively comparable.
At N131072/C256, EQX cannot allocate the complete 31.75 GiB edge-weight
tensor. The bounded chain completes. OOM has no speed or memory ratio.

## Ordinary ICTC composition

This uses the same current TP and radial MLP with a full edge-weight tensor.
It exposes the speed/memory tradeoff of bounded execution.

| N edges | C | metric | bounded ms | ordinary ICTC ms | bounded allocated GiB | ordinary allocated GiB |
|---:|---:|---|---:|---:|---:|---:|
| 16,384 | 64 | forward | 3.822 | 3.576 | 1.31 | 1.21 |
| 16,384 | 64 | full_vjp | 15.101 | 14.724 | 2.44 | 1.44 |
| 32,768 | 128 | forward | 13.304 | 13.753 | 2.03 | 4.80 |
| 32,768 | 128 | full_vjp | 52.221 | 43.970 | 4.78 | 9.64 |
| 131,072 | 256 | forward | 105.482 | OOM | 8.96 | — |
| 131,072 | 256 | full_vjp | 436.218 | OOM | 19.39 | — |

The ordinary C128 composition has a faster full VJP, but its allocated peak
is more than twice the bounded chain's. Joint TP adjoints keep weights and
weight gradients live together; without the wrapper this also increases
memory relative to the preceding separate-adjoint implementation. C64
ordinary composition remains preferable when its smaller memory footprint
and measured speed fit the application. Both ordinary ICTC and EQX run out
of memory at N131072/C256; the bounded chain completes.

## Improvement over the preceding ICTC implementation

Job 91257 compares the frozen preceding radial implementation and current
graph optimization on the same GPU, with identical parameters and shared
inputs. Seven rounds alternate execution order after warmup. The outer model
uses ordinary PyTorch; internal blocks compile in both versions.

force_parameters computes energy, geometry gradients with create_graph=True,
then the force loss's gradients for features and every network parameter.
A coordinate Hessian is not requested by this metric. These are ICTC-to-ICTC
improvements; no EQX force-loss timing was collected.

| N edges | C | metric | preceding ICTC ms | current ICTC ms | old/current speed | old peak GiB | current peak GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 16,384 | 64 | full_vjp | 16.567 | 16.614 | 1.00× | 3.44 | 3.44 |
| 16,384 | 64 | force_parameters | 59.355 | 47.009 | 1.26× | 3.56 | 3.56 |
| 32,768 | 128 | full_vjp | 59.659 | 53.288 | 1.12× | 7.88 | 5.79 |
| 32,768 | 128 | force_parameters | 234.862 | 189.228 | 1.24× | 8.32 | 5.27 |
| 131,072 | 256 | full_vjp | 479.852 | 436.814 | 1.10× | 25.58 | 21.39 |
| 131,072 | 256 | force_parameters | 1894.013 | 1534.443 | 1.23× | 31.90 | 28.35 |

Paired peaks include both resident models and shared inputs. They measure
the controlled change between implementations, and must not be combined
with the isolated deployment-capacity numbers above. Force-loss training is
1.23–1.26× faster in these three shapes; C64 first-order full VJP is unchanged.

## Allocator and process memory

Allocated peaks include inputs, outputs, cotangents and workspaces. Reserved
peaks include allocator cache. NVML is sampled process memory, including
context/cache; sampling can miss transient peaks. The raw records preserve
sampling intervals/errors and every timing sample, including outliers.

| N edges | C | metric | implementation | allocated GiB | reserved GiB | sampled NVML GiB |
|---:|---:|---|---|---:|---:|---:|
| 16,384 | 64 | forward | fused | 1.31 | 1.40 | 1.89 |
| 16,384 | 64 | forward | eqx | 1.57 | 1.65 | 2.14 |
| 16,384 | 64 | forward | ictc_unfused | 1.21 | 1.30 | 1.79 |
| 16,384 | 64 | full_vjp | fused | 2.44 | 2.50 | 2.99 |
| 16,384 | 64 | full_vjp | eqx | 3.88 | 3.92 | 4.42 |
| 16,384 | 64 | full_vjp | ictc_unfused | 1.44 | 1.50 | 2.00 |
| 32,768 | 128 | forward | fused | 2.03 | 2.21 | 2.70 |
| 32,768 | 128 | forward | eqx | 6.24 | 6.28 | 6.77 |
| 32,768 | 128 | forward | ictc_unfused | 4.80 | 5.19 | 5.68 |
| 32,768 | 128 | full_vjp | fused | 4.78 | 5.60 | 6.09 |
| 32,768 | 128 | full_vjp | eqx | 15.40 | 15.46 | 15.96 |
| 32,768 | 128 | full_vjp | ictc_unfused | 9.64 | 9.97 | 10.47 |
| 131,072 | 256 | forward | fused | 8.96 | 9.75 | 10.24 |
| 131,072 | 256 | forward | eqx | — | — | — |
| 131,072 | 256 | forward | ictc_unfused | — | — | — |
| 131,072 | 256 | full_vjp | fused | 19.39 | 20.22 | 20.72 |
| 131,072 | 256 | full_vjp | eqx | — | — | — |
| 131,072 | 256 | full_vjp | ictc_unfused | — | — | — |

## Timing variability

Median and interquartile interval from all seven independent-chain samples.

| N edges | C | metric | implementation | median ms | Q1–Q3 ms |
|---:|---:|---|---|---:|---:|
| 16,384 | 64 | forward | fused | 3.822 | 3.811–3.876 |
| 16,384 | 64 | forward | eqx | 4.020 | 4.002–4.117 |
| 16,384 | 64 | forward | ictc_unfused | 3.576 | 3.567–3.626 |
| 16,384 | 64 | full_vjp | fused | 15.101 | 15.088–15.178 |
| 16,384 | 64 | full_vjp | eqx | 14.948 | 14.841–14.983 |
| 16,384 | 64 | full_vjp | ictc_unfused | 14.724 | 14.716–14.801 |
| 32,768 | 128 | forward | fused | 13.304 | 13.285–13.956 |
| 32,768 | 128 | forward | eqx | 15.740 | 15.411–15.798 |
| 32,768 | 128 | forward | ictc_unfused | 13.753 | 13.741–13.858 |
| 32,768 | 128 | full_vjp | fused | 52.221 | 52.021–52.318 |
| 32,768 | 128 | full_vjp | eqx | 57.816 | 57.332–58.054 |
| 32,768 | 128 | full_vjp | ictc_unfused | 43.970 | 43.650–44.161 |
| 131,072 | 256 | forward | fused | 105.482 | 105.241–105.609 |
| 131,072 | 256 | full_vjp | fused | 436.218 | 434.133–436.867 |

## Execution and numerical qualification

[Requested derivative roles and shared coupling](RADIAL_GRAPH.md) describe
the graph changes. C128–256 blocks cap at 8192 edges/2 GiB of FP32 weights;
other channels cap at 16384 edges/4 GiB. Path/channel sizes set the cap,
independently of total N. Generated block weights and their adjoints still
exist, and their lifetime is bounded. Only TP-input adjoints require weight
regeneration; unused GEMMs and second-derivative contractions are skipped.

Role-pruning qualification passed 40 checks, including all 31 first-order
radial role combinations, live cotangents, geometry Hessians and force-loss
parameter targets. The radial integration suite passed 22 checks. Generic
uvu derivatives passed 30 checks across V1/unshared, V3/shared and V17/unshared.
Joint adjoints passed 33 checks across C128/C129/C256, channel tails, full
angular parity and second derivatives. Actual EQX-chain parity passed 12
checks, including the 8193-edge block boundary. Independent dense CG checks
covered all 131072 edges and all final-affine parameters at C256; maximum
normalized error was 3.41e-6. These checks do not use sampled edges.

Installed-wheel qualification covers both new APIs, all three original APIs
and empty batches. The original APIs' values/VJPs remain bitwise identical
to v0.1.5. Force-loss double backward is qualified with ordinary outer
PyTorch and compiled internal blocks. Outer PyTorch 2.7.1 AOT does not
support double backward; see [usage](RADIAL.md).

Private raw evidence: ICTC-O2-HP-local-only/extensions/graph_20261013/
(joint_v2/results, source snapshots, scheduled harnesses and package checks).
The preceding radial cohort is preserved under extensions/radial_20261013/.
The existing v0.1.5 uuu benchmark and figures describe a separate operator.
