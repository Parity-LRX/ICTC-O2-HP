# ICTC-O2 HP radial MLP + TP vs EquivariantX

Version 0.1.6.dev0; one physical tensor and GEMM layout for every C and N.
A100-SXM4 40GB; PyTorch 2.7.1+cu128; Triton 3.3.1; strict FP32, TF32/AMP disabled.
Job 91436: fresh process per implementation/metric, 20 warmups and 11 CUDA-event samples.
Both complete chains use torch.compile(fullgraph=True,dynamic=False).

Linear(16,64) -> SiLU -> Linear(64,64) -> SiLU -> Linear(64,P*C,bias=False) -> V=1 TP.
First/output full parity; angular canonical parity; m_max=6; N counts edges.
Full VJP includes forward, all three inputs and every network parameter.
Gather/scatter and the rest of a message-passing model are outside this measured chain.
EQX is actual native u1u with matched paths/external weights. Basis conversion is outside timing;
phase factors are absorbed into trainable final-affine parameters.

## Current EQX comparison

Speed is EQX time / ICTC time; above one favors ICTC. OOM has no ratio.

| N edges | C | metric | ICTC ms | EQX ms | EQX/ICTC speed | ICTC allocated GiB | EQX allocated GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 131,072 | 64 | forward | 25.991 | 32.489 | 1.25× | 2.18 | 12.46 |
| 131,072 | 64 | full_vjp | 114.167 | 119.141 | 1.04× | 5.01 | 30.89 |
| 16,384 | 64 | forward | 3.660 | 4.028 | 1.10× | 0.72 | 1.57 |
| 16,384 | 64 | full_vjp | 13.159 | 15.207 | 1.16× | 1.88 | 3.88 |
| 131,072 | 256 | forward | 102.812 | OOM | — | 8.55 | — |
| 131,072 | 256 | full_vjp | 410.604 | OOM | — | 19.43 | — |
| 32,768 | 128 | forward | 12.808 | 16.042 | 1.25× | 1.83 | 6.24 |
| 32,768 | 128 | full_vjp | 50.896 | 58.402 | 1.15× | 4.81 | 15.40 |

## Increment over the preceding ICTC

The control is qualified a52398c, with only a namespace change for paired isolation.
Parameters, coefficients, normalization, paths and physical layouts are identical.
All 16 isolated and eight paired memory comparisons pass the 115% allocated/reserved limit.
Sampled NVML readings in isolated comparisons are also checked against 115%.

C64/N131072 full-VJP time decreases 7.2%;
its paired force-loss time decreases 1.5%.
C64/N16384 pure-forward time increases 17.3%.
The scheduling change primarily improves first-order training; pure inference and force-loss results are reported separately.

| N edges | C | metric | preceding ms | current ms | old/current speed | old allocated GiB | current allocated GiB | current/old memory |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 131,072 | 64 | forward | 25.750 | 25.991 | 0.99× | 2.67 | 2.18 | 0.815× |
| 131,072 | 64 | full_vjp | 123.032 | 114.167 | 1.08× | 5.44 | 5.01 | 0.920× |
| 16,384 | 64 | forward | 3.120 | 3.660 | 0.85× | 1.21 | 0.72 | 0.591× |
| 16,384 | 64 | full_vjp | 13.507 | 13.159 | 1.03× | 2.44 | 1.88 | 0.771× |
| 131,072 | 256 | forward | 102.804 | 102.812 | 1.00× | 8.55 | 8.55 | 1.000× |
| 131,072 | 256 | full_vjp | 409.289 | 410.604 | 1.00× | 19.18 | 19.43 | 1.013× |
| 32,768 | 128 | forward | 12.908 | 12.808 | 1.01× | 1.83 | 1.83 | 1.000× |
| 32,768 | 128 | full_vjp | 50.792 | 50.896 | 1.00× | 4.68 | 4.81 | 1.027× |
| 32,768 | 64 | forward | 6.272 | 6.212 | 1.01× | 1.42 | 0.93 | 0.651× |
| 32,768 | 64 | full_vjp | 29.277 | 27.778 | 1.05× | 2.87 | 2.44 | 0.848× |
| 65,536 | 64 | forward | 12.923 | 13.015 | 0.99× | 1.84 | 1.34 | 0.730× |
| 65,536 | 64 | full_vjp | 60.871 | 56.530 | 1.08× | 3.73 | 3.29 | 0.883× |
| 32,768 | 256 | forward | 25.655 | 25.843 | 0.99× | 3.64 | 3.64 | 1.000× |
| 32,768 | 256 | full_vjp | 99.500 | 100.963 | 0.99× | 9.30 | 9.55 | 1.027× |
| 131,072 | 128 | forward | 51.277 | 51.236 | 1.00× | 4.30 | 4.30 | 1.000× |
| 131,072 | 128 | full_vjp | 207.352 | 208.039 | 1.00× | 9.69 | 9.82 | 1.013× |

## Paired full VJP and force-loss training

Jobs 91432 (full VJP) and 91437 (force_parameters) alternate identical parameters/shared inputs on one GPU.
Ordinary outer PyTorch, compiled internal blocks. Both use 20 warmups/model; full VJP has 11 rounds, force loss 15.
Force loss first differentiates energy with create_graph=True, then differentiates the loss for input features
and every network parameter. This timing does not request a coordinate Hessian; no EQX force timing was measured.
Paired peaks include both resident models/shared inputs and differ from isolated deployment capacity.

| N edges | C | metric | preceding ms | current ms | old/current speed | old allocated GiB | current allocated GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 131,072 | 256 | force_parameters | 1457.202 | 1459.585 | 1.00× | 28.35 | 28.60 |
| 131,072 | 256 | full_vjp | 411.260 | 411.427 | 1.00× | 21.19 | 21.68 |
| 131,072 | 64 | force_parameters | 386.415 | 380.575 | 1.02× | 7.99 | 7.37 |
| 131,072 | 64 | full_vjp | 123.970 | 115.382 | 1.07× | 6.45 | 5.57 |
| 32,768 | 128 | force_parameters | 180.662 | 181.690 | 0.99× | 5.27 | 5.52 |
| 32,768 | 128 | full_vjp | 51.678 | 52.003 | 0.99× | 5.68 | 5.93 |
| 16,384 | 64 | force_parameters | 46.090 | 46.730 | 0.99× | 3.56 | 2.34 |
| 16,384 | 64 | full_vjp | 14.680 | 13.979 | 1.05× | 3.44 | 2.45 |

## Timing and memory details

Allocated includes live inputs/outputs/cotangents/workspaces; reserved includes allocator cache.
NVML samples include process context/cache and can miss short peaks. All timing samples and sampling errors are archived.

| N | C | metric | cohort | median ms | Q1–Q3 ms | allocated GiB | reserved GiB | sampled NVML GiB |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 131,072 | 64 | forward | previous | 25.750 | 25.689–25.928 | 2.67 | 3.43 | 3.92 |
| 131,072 | 64 | forward | optimized | 25.991 | 25.855–26.169 | 2.18 | 2.44 | 2.93 |
| 131,072 | 64 | forward | eqx | 32.489 | 32.256–32.625 | 12.46 | 12.50 | 12.99 |
| 131,072 | 64 | full_vjp | previous | 123.032 | 122.699–123.737 | 5.44 | 6.41 | 6.90 |
| 131,072 | 64 | full_vjp | optimized | 114.167 | 114.017–114.270 | 5.01 | 5.62 | 6.11 |
| 131,072 | 64 | full_vjp | eqx | 119.141 | 118.449–119.849 | 30.89 | 31.03 | 31.52 |
| 16,384 | 64 | forward | previous | 3.120 | 3.100–3.167 | 1.21 | 1.30 | 1.79 |
| 16,384 | 64 | forward | optimized | 3.660 | 3.654–3.670 | 0.72 | 0.80 | 1.29 |
| 16,384 | 64 | forward | eqx | 4.028 | 4.015–4.079 | 1.57 | 1.65 | 2.14 |
| 16,384 | 64 | full_vjp | previous | 13.507 | 13.313–13.662 | 2.44 | 2.50 | 2.99 |
| 16,384 | 64 | full_vjp | optimized | 13.159 | 12.968–13.241 | 1.88 | 2.40 | 2.89 |
| 16,384 | 64 | full_vjp | eqx | 15.207 | 15.117–15.314 | 3.88 | 3.92 | 4.42 |
| 131,072 | 256 | forward | previous | 102.804 | 102.453–102.825 | 8.55 | 9.75 | 10.24 |
| 131,072 | 256 | forward | optimized | 102.812 | 102.543–102.831 | 8.55 | 9.75 | 10.24 |
| 131,072 | 256 | full_vjp | previous | 409.289 | 407.949–409.561 | 19.18 | 20.22 | 20.72 |
| 131,072 | 256 | full_vjp | optimized | 410.604 | 409.249–411.152 | 19.43 | 20.47 | 20.96 |
| 32,768 | 128 | forward | previous | 12.908 | 12.639–12.912 | 1.83 | 2.21 | 2.70 |
| 32,768 | 128 | forward | optimized | 12.808 | 12.521–12.911 | 1.83 | 2.21 | 2.70 |
| 32,768 | 128 | forward | eqx | 16.042 | 15.941–16.122 | 6.24 | 6.28 | 6.77 |
| 32,768 | 128 | full_vjp | previous | 50.792 | 50.469–50.835 | 4.68 | 5.60 | 6.09 |
| 32,768 | 128 | full_vjp | optimized | 50.896 | 50.855–50.928 | 4.81 | 5.85 | 6.34 |
| 32,768 | 128 | full_vjp | eqx | 58.402 | 58.120–58.604 | 15.40 | 15.46 | 15.96 |
| 32,768 | 64 | forward | previous | 6.272 | 6.009–6.352 | 1.42 | 1.60 | 2.09 |
| 32,768 | 64 | forward | optimized | 6.212 | 6.151–6.378 | 0.93 | 1.11 | 1.60 |
| 32,768 | 64 | full_vjp | previous | 29.277 | 29.152–29.350 | 2.87 | 3.79 | 4.29 |
| 32,768 | 64 | full_vjp | optimized | 27.778 | 27.504–27.784 | 2.44 | 2.92 | 3.42 |
| 65,536 | 64 | forward | previous | 12.923 | 12.670–12.968 | 1.84 | 2.21 | 2.70 |
| 65,536 | 64 | forward | optimized | 13.015 | 12.675–13.108 | 1.34 | 1.72 | 2.21 |
| 65,536 | 64 | full_vjp | previous | 60.871 | 60.823–60.898 | 3.73 | 4.60 | 5.10 |
| 65,536 | 64 | full_vjp | optimized | 56.530 | 56.482–56.566 | 3.29 | 3.83 | 4.32 |
| 32,768 | 256 | forward | previous | 25.655 | 25.534–25.834 | 3.64 | 4.42 | 4.91 |
| 32,768 | 256 | forward | optimized | 25.843 | 25.553–25.875 | 3.64 | 4.42 | 4.91 |
| 32,768 | 256 | full_vjp | previous | 99.500 | 99.309–99.705 | 9.30 | 11.19 | 11.69 |
| 32,768 | 256 | full_vjp | optimized | 100.963 | 100.878–101.005 | 9.55 | 11.69 | 12.18 |
| 131,072 | 128 | forward | previous | 51.277 | 51.216–51.439 | 4.30 | 4.88 | 5.37 |
| 131,072 | 128 | forward | optimized | 51.236 | 51.215–51.251 | 4.30 | 4.88 | 5.37 |
| 131,072 | 128 | full_vjp | previous | 207.352 | 207.137–207.682 | 9.69 | 10.11 | 10.61 |
| 131,072 | 128 | full_vjp | optimized | 208.039 | 207.790–208.388 | 9.82 | 10.24 | 10.73 |

## Implementation and qualification

[Adjoint scheduling](RADIAL_GRAPH.md) retains requested-role pruning, direct TP writeback and uniform transposed GEMMs.
External V=1 joint adjoints share x/cotangent contractions for C1–256. One thread policy uses
BU=nextpow2(C), B=max(1,128//BU); layouts, raw coupling arithmetic and all trainable paths stay unchanged.
Every C/N uses ordinary blocks capped at 8192 edges/2 GiB weights. The saved first block is always
9/8 of that cap, at most 9216 edges/2.25 GiB, to avoid additional affine recomputation.
Backward consumes that exact prefix, then ordinary blocks. No N-based algorithm selector is introduced.

Jobs 91411 and 91433 cover 40 role, 22 radial, 66 joint/tail/second, four cache-boundary and 59 generic uvu checks.
The first job's small-batch checks are unchanged by the sole repair from 11/8 to 9/8 cache expansion;
the second job rechecks the new cache boundary, all-edge largest shape, and all generic APIs.
Independent dense CG checks all 131072 edges/every affine parameter at C256; max normalized error 2.94e-06.
Installed-wheel job 91439 covers original/new APIs, actual EQX chain parity at the 9217-edge cache boundary, and empty inputs.
Original three APIs retain bitwise v0.1.5 values/VJPs. Outer PyTorch 2.7.1 AOT double backward remains unsupported;
force-loss training uses ordinary outer PyTorch and compiled blocks; see [usage](RADIAL.md).

## Earlier ordinary composition

Separate job 91260: ordinary full edge-weight composition at N16384/C64 was 14.724 ms /1.44 GiB versus
15.101 ms /2.44 GiB for its bounded wrapper. N32768/C128 was 43.970 ms /9.64 GiB versus bounded 52.221 ms /4.78 GiB.
Ordinary composition OOM at N131072/C256. Those earlier-cohort times must not be divided by current times.

Private evidence: ICTC-O2-HP-local-only/extensions/schedule_20261014/. Preceding qualified cohort: extensions/writeback_20261014/.
Existing v0.1.5 uuu figures cover a separate operator.
