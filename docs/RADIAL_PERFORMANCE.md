# ICTC-O2 HP radial MLP + TP vs EquivariantX

Development version 0.1.6.dev0 with one affine-adjoint layout for every channel count.
A100-SXM4 40GB, PyTorch 2.7.1+cu128, Triton 3.3.1; strict FP32, TF32/AMP disabled.
Job 91314: fresh process per metric/implementation, two warmups, seven CUDA-event samples.
Both full chains use torch.compile(fullgraph=True,dynamic=False).

Linear(16,64) -> SiLU -> Linear(64,64) -> SiLU -> Linear(64,P*C,bias=False) -> V=1 TP.
First/output full parity; angular canonical parity; m_max=6; N counts edges.
Full VJP includes forward, all three input gradients and every network parameter.
Neighbor gathering/scattering and the rest of a message-passing model are outside this chain.

EQX uses its actual native u1u with matched paths and external per-edge weights.
Basis/layout conversion is outside timing; phase factors are absorbed into final affine parameters.

## Current comparison

Speed is EQX time divided by ICTC time; values above one favor ICTC.

| N edges | C | metric | ICTC ms | EQX ms | EQX/ICTC speed | ICTC allocated GiB | EQX allocated GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 131,072 | 256 | forward | 106.285 | OOM | — | 8.96 | — |
| 131,072 | 256 | full_vjp | 415.652 | OOM | — | 19.39 | — |
| 32,768 | 128 | forward | 14.700 | 15.672 | 1.07× | 2.03 | 6.24 |
| 32,768 | 128 | full_vjp | 51.383 | 57.472 | 1.12× | 4.78 | 15.40 |
| 16,384 | 64 | forward | 3.832 | 4.030 | 1.05× | 1.31 | 1.57 |
| 16,384 | 64 | full_vjp | 13.942 | 14.892 | 1.07× | 2.44 | 3.88 |

OOM has no speed or memory ratio. Small differences should be assessed against raw sample distributions.
All channel counts use the same hidden-adjoint GEMM, affine-adjoint GEMM and accumulation layout.
The measured changes in each shape are reported below, including any slower result.

## Improvement over preceding ICTC and the 10% VRAM budget

The control is the preceding qualified graph/joint-adjoint implementation, with only its library
namespace renamed for isolation. Both cohorts have identical path counts, architecture and FP32 settings.
Allocated and reserved peaks must each be at most 110% of the preceding chain.
All ten isolated shape/metric checks and all six paired full-VJP/force-loss checks pass.
The additional fixed-C/different-N measurements check the same library policy across edge counts.

| N edges | C | metric | previous ms | current ms | old/current speed | old GiB | current GiB | current/old memory |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 131,072 | 256 | forward | 106.453 | 106.285 | 1.00× | 8.96 | 8.96 | 1.000× |
| 131,072 | 256 | full_vjp | 440.171 | 415.652 | 1.06× | 19.39 | 19.39 | 1.000× |
| 32,768 | 128 | forward | 14.705 | 14.700 | 1.00× | 2.03 | 2.03 | 1.000× |
| 32,768 | 128 | full_vjp | 52.923 | 51.383 | 1.03× | 4.78 | 4.78 | 1.000× |
| 16,384 | 64 | forward | 3.820 | 3.832 | 1.00× | 1.31 | 1.31 | 1.000× |
| 16,384 | 64 | full_vjp | 13.567 | 13.942 | 0.97× | 2.44 | 2.44 | 1.000× |
| 32,768 | 256 | forward | 27.109 | 26.993 | 1.00× | 4.05 | 4.05 | 1.000× |
| 32,768 | 256 | full_vjp | 106.697 | 101.207 | 1.05× | 9.50 | 9.50 | 1.000× |
| 131,072 | 128 | forward | 53.406 | 53.519 | 1.00× | 4.51 | 4.51 | 1.000× |
| 131,072 | 128 | full_vjp | 216.500 | 210.365 | 1.03× | 9.79 | 9.79 | 1.000× |

At N16384/C64, isolated full-VJP time increases by 2.8%.
At N131072/C256, time decreases by 5.6%.
The uniform layout does not establish a C64 speedup; its largest measured benefit is at C256.

## Paired full VJP and force-loss training

Job 91313 uses identical parameters/shared resident inputs on the same GPU, seven alternating rounds.
The outer model uses ordinary PyTorch, with compiled internal blocks. force_parameters computes an
energy derivative with create_graph=True, then force-loss gradients for features and every network parameter.
This metric does not request a coordinate Hessian. These improvements are relative to prior ICTC, not EQX.
Paired memory includes both resident models/shared inputs and is separate from isolated capacity above.

| N edges | C | metric | previous ms | current ms | old/current speed | previous peak GiB | current peak GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 131,072 | 256 | force_parameters | 1553.848 | 1483.782 | 1.05× | 28.35 | 28.35 |
| 131,072 | 256 | full_vjp | 442.359 | 419.061 | 1.06× | 21.39 | 21.39 |
| 32,768 | 128 | force_parameters | 190.423 | 186.675 | 1.02× | 5.27 | 5.27 |
| 32,768 | 128 | full_vjp | 53.649 | 52.902 | 1.01× | 5.79 | 5.79 |
| 16,384 | 64 | force_parameters | 47.316 | 47.468 | 1.00× | 3.56 | 3.56 |
| 16,384 | 64 | full_vjp | 16.186 | 16.399 | 0.99× | 3.44 | 3.44 |

## Timing variability and allocator/process memory

All allocated/reserved peaks, isolated NVML readings and timing samples are retained. NVML is process memory
including context/cache and can miss short peaks. Reserved memory includes allocator cache.
Allocated peaks include inputs, outputs, cotangents and workspaces.

| N | C | metric | cohort | median ms | Q1–Q3 ms | allocated GiB | reserved GiB | sampled NVML GiB |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 131,072 | 256 | forward | previous | 106.453 | 106.298–106.978 | 8.96 | 9.75 | 10.24 |
| 131,072 | 256 | forward | optimized | 106.285 | 106.277–106.591 | 8.96 | 9.75 | 10.24 |
| 131,072 | 256 | full_vjp | previous | 440.171 | 439.926–440.587 | 19.39 | 20.22 | 20.72 |
| 131,072 | 256 | full_vjp | optimized | 415.652 | 413.654–416.350 | 19.39 | 20.22 | 20.72 |
| 32,768 | 128 | forward | previous | 14.705 | 14.089–14.731 | 2.03 | 2.21 | 2.70 |
| 32,768 | 128 | forward | optimized | 14.700 | 14.691–14.723 | 2.03 | 2.21 | 2.70 |
| 32,768 | 128 | forward | eqx | 15.672 | 15.562–16.068 | 6.24 | 6.28 | 6.77 |
| 32,768 | 128 | full_vjp | previous | 52.923 | 52.827–54.412 | 4.78 | 5.60 | 6.09 |
| 32,768 | 128 | full_vjp | optimized | 51.383 | 51.332–51.699 | 4.78 | 5.60 | 6.09 |
| 32,768 | 128 | full_vjp | eqx | 57.472 | 57.262–57.667 | 15.40 | 15.46 | 15.96 |
| 16,384 | 64 | forward | previous | 3.820 | 3.801–3.909 | 1.31 | 1.40 | 1.89 |
| 16,384 | 64 | forward | optimized | 3.832 | 3.818–3.921 | 1.31 | 1.40 | 1.89 |
| 16,384 | 64 | forward | eqx | 4.030 | 4.014–4.053 | 1.57 | 1.65 | 2.14 |
| 16,384 | 64 | full_vjp | previous | 13.567 | 13.395–13.670 | 2.44 | 2.50 | 2.99 |
| 16,384 | 64 | full_vjp | optimized | 13.942 | 13.923–14.632 | 2.44 | 2.50 | 2.99 |
| 16,384 | 64 | full_vjp | eqx | 14.892 | 14.856–15.105 | 3.88 | 3.92 | 4.42 |
| 32,768 | 256 | forward | previous | 27.109 | 26.908–28.919 | 4.05 | 4.42 | 4.91 |
| 32,768 | 256 | forward | optimized | 26.993 | 26.985–27.646 | 4.05 | 4.42 | 4.91 |
| 32,768 | 256 | full_vjp | previous | 106.697 | 106.613–107.038 | 9.50 | 11.19 | 11.69 |
| 32,768 | 256 | full_vjp | optimized | 101.207 | 101.189–101.227 | 9.50 | 11.19 | 11.69 |
| 131,072 | 128 | forward | previous | 53.406 | 53.152–53.411 | 4.51 | 4.88 | 5.37 |
| 131,072 | 128 | forward | optimized | 53.519 | 53.463–54.087 | 4.51 | 4.88 | 5.37 |
| 131,072 | 128 | full_vjp | previous | 216.500 | 216.185–216.869 | 9.79 | 10.11 | 10.61 |
| 131,072 | 128 | full_vjp | optimized | 210.365 | 210.308–210.788 | 9.79 | 10.11 | 10.61 |

## Implementation and qualification

[Adjoint scheduling](RADIAL_GRAPH.md) describes requested-role pruning, shared coupling and affine-adjoint layout.
Every C uses equivalent transposed GEMMs and matching parameter-gradient accumulation, converting once
after all blocks. The same layout policy applies to every N, with no channel-based layout branch.
TP kernels, coefficients, routes, all parameters, block caps and the first-weight cache remain unchanged.
Joint-channel blocks cap at 8192 edges/2 GiB FP32 weights; other channels at 16384 edges/4 GiB.

The new layout passed 40 requested-role checks, 22 radial integration checks and 33 joint/tail/second checks.
Independent dense CG validation checked all 131072 edges and all final-affine parameters at C256.
Maximum normalized error in that largest-shape check was 1.67e-6. Geometry Hessians, live cotangents
and force-loss parameters remain qualified.
Installed-wheel checks cover original/new APIs, actual EQX chain parity across the block boundary and empty batches.
Original three APIs remain bitwise identical to frozen v0.1.5. Outer PyTorch 2.7.1 AOT still lacks double backward;
force-loss training uses ordinary outer PyTorch and compiled internal blocks; see [usage](RADIAL.md).

## Prior ordinary-composition comparison

The separate job 91260 cohort also measured ordinary ICTC composition, which materializes the complete
edge-weight tensor. At N16384/C64 its full VJP was 14.724 ms and 1.44 GiB allocated, compared with
15.101 ms and 2.44 GiB for the preceding bounded wrapper. At N32768/C128 ordinary composition was
43.970 ms and 9.64 GiB, compared with 52.221 ms and 4.78 GiB for the bounded wrapper.
Ordinary composition ran out of memory at N131072/C256. Its TP/trunk are unchanged by this layout step.
These historical measurements show the bounded wrapper's speed/memory tradeoff; they were not rerun
in job 91314 and must not be divided by current times from a different cohort.

Private evidence: ICTC-O2-HP-local-only/extensions/latency_20261014/. Preceding graph cohort remains in
extensions/graph_20261013/. The existing v0.1.5 uuu benchmark/figures describe a separate operator.
