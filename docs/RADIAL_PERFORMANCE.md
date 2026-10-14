# ICTC-O2 HP radial MLP + TP vs EquivariantX

Version 0.1.6.dev0: direct TP output/adjoint writeback with the same layout for every C and N.
A100-SXM4 40GB; PyTorch 2.7.1+cu128; Triton 3.3.1; strict FP32, TF32/AMP disabled.
Job 91331 uses a fresh process per implementation/metric, 20 warmups and 11 CUDA-event samples.
Both complete chains use torch.compile(fullgraph=True,dynamic=False).

Linear(16,64) -> SiLU -> Linear(64,64) -> SiLU -> Linear(64,P*C,bias=False) -> V=1 TP.
First/output full parity; angular canonical parity; m_max=6; N counts edges.
Full VJP includes forward, all three inputs and every network parameter.
Gathering/scattering and the rest of a message-passing model are outside the measured chain.
EQX is its actual native u1u, with matched paths and external per-edge weights.
Basis/layout conversion is outside timing; phase factors are absorbed into final-affine parameters.

## Current EQX comparison

Speed is EQX time / ICTC time; above one favors ICTC. OOM has no ratio.

| N edges | C | metric | ICTC ms | EQX ms | EQX/ICTC speed | ICTC allocated GiB | EQX allocated GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 131,072 | 256 | forward | 101.032 | OOM | — | 8.55 | — |
| 131,072 | 256 | full_vjp | 402.970 | OOM | — | 19.18 | — |
| 16,384 | 64 | forward | 3.639 | 4.044 | 1.11× | 1.21 | 1.57 |
| 16,384 | 64 | full_vjp | 13.276 | 14.890 | 1.12× | 2.44 | 3.88 |
| 32,768 | 128 | forward | 12.569 | 15.881 | 1.26× | 1.83 | 6.24 |
| 32,768 | 128 | full_vjp | 49.548 | 57.289 | 1.16× | 4.68 | 15.40 |

## Increment over the preceding uniform-layout ICTC

The control is the preceding qualified uniform-layout implementation, with only its library namespace
renamed for paired isolation. All parameters, coefficients, normalization and routes are identical.
Every C and N keeps the same hidden/affine GEMM and accumulation layout.
All 12 isolated and six paired allocated/reserved comparisons stay within the 110% peak-VRAM limit.

| N edges | C | metric | preceding ms | current ms | preceding/current speed | old allocated GiB | current allocated GiB | current/old memory |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 131,072 | 256 | forward | 106.260 | 101.032 | 1.05× | 8.96 | 8.55 | 0.955× |
| 131,072 | 256 | full_vjp | 412.337 | 402.970 | 1.02× | 19.39 | 19.18 | 0.990× |
| 16,384 | 64 | forward | 3.659 | 3.639 | 1.01× | 1.31 | 1.21 | 0.923× |
| 16,384 | 64 | full_vjp | 13.407 | 13.276 | 1.01× | 2.44 | 2.44 | 1.000× |
| 32,768 | 128 | forward | 13.229 | 12.569 | 1.05× | 2.03 | 1.83 | 0.900× |
| 32,768 | 128 | full_vjp | 50.966 | 49.548 | 1.03× | 4.78 | 4.68 | 0.979× |
| 32,768 | 256 | forward | 26.462 | 25.399 | 1.04× | 4.05 | 3.64 | 0.900× |
| 32,768 | 256 | full_vjp | 100.550 | 97.451 | 1.03× | 9.50 | 9.30 | 0.979× |
| 131,072 | 128 | forward | 52.970 | 50.371 | 1.05× | 4.51 | 4.30 | 0.955× |
| 131,072 | 128 | full_vjp | 209.393 | 205.162 | 1.02× | 9.79 | 9.69 | 0.990× |
| 131,072 | 64 | forward | 26.571 | 25.283 | 1.05× | 2.88 | 2.67 | 0.929× |
| 131,072 | 64 | full_vjp | 123.856 | 121.272 | 1.02× | 5.54 | 5.44 | 0.982× |

## Paired full VJP and force-loss training

Jobs 91325 (full VJP) and 91330 (force_parameters) use identical parameters/shared inputs on one GPU.
The outer model uses ordinary PyTorch; internal blocks compile in both versions.
Full VJP: five warmups/model, 11 alternating rounds. Force loss: 20 warmups/model, 15 alternating rounds.
force_parameters differentiates energy with create_graph=True, then the force loss for features and
every network parameter. It does not request a coordinate Hessian. No EQX force-loss time was measured.
Paired peaks include both resident models/shared inputs and are distinct from isolated capacity above.

| N edges | C | metric | preceding ms | current ms | preceding/current speed | old peak GiB | current peak GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 131,072 | 256 | force_parameters | 1496.789 | 1466.788 | 1.02× | 28.35 | 28.35 |
| 131,072 | 256 | full_vjp | 412.855 | 403.542 | 1.02× | 21.39 | 21.19 |
| 32,768 | 128 | force_parameters | 186.551 | 182.884 | 1.02× | 5.27 | 5.27 |
| 32,768 | 128 | full_vjp | 51.867 | 50.512 | 1.03× | 5.79 | 5.68 |
| 16,384 | 64 | force_parameters | 47.890 | 46.907 | 1.02× | 3.56 | 3.56 |
| 16,384 | 64 | full_vjp | 14.794 | 14.468 | 1.02× | 3.44 | 3.44 |

## Timing variability and memory definitions

Allocated peaks include live inputs, outputs, cotangents and workspaces; reserved peaks include allocator
cache. NVML is sampled process memory, includes context/cache and can miss short peaks.
Raw records preserve every sample, sampling errors and intervals. Small differences require caution.

| N | C | metric | cohort | median ms | Q1–Q3 ms | allocated GiB | reserved GiB | sampled NVML GiB |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 131,072 | 256 | forward | previous | 106.260 | 106.250–106.318 | 8.96 | 9.75 | 10.24 |
| 131,072 | 256 | forward | optimized | 101.032 | 101.024–101.073 | 8.55 | 9.75 | 10.24 |
| 131,072 | 256 | full_vjp | previous | 412.337 | 411.493–412.697 | 19.39 | 20.22 | 20.72 |
| 131,072 | 256 | full_vjp | optimized | 402.970 | 402.457–403.248 | 19.18 | 20.22 | 20.72 |
| 16,384 | 64 | forward | previous | 3.659 | 3.366–3.809 | 1.31 | 1.40 | 1.89 |
| 16,384 | 64 | forward | optimized | 3.639 | 3.631–3.657 | 1.21 | 1.30 | 1.79 |
| 16,384 | 64 | forward | eqx | 4.044 | 4.041–4.102 | 1.57 | 1.65 | 2.14 |
| 16,384 | 64 | full_vjp | previous | 13.407 | 13.389–13.598 | 2.44 | 2.50 | 2.99 |
| 16,384 | 64 | full_vjp | optimized | 13.276 | 13.206–13.331 | 2.44 | 2.50 | 2.99 |
| 16,384 | 64 | full_vjp | eqx | 14.890 | 14.790–14.935 | 3.88 | 3.92 | 4.42 |
| 32,768 | 128 | forward | previous | 13.229 | 13.006–13.234 | 2.03 | 2.21 | 2.70 |
| 32,768 | 128 | forward | optimized | 12.569 | 12.339–12.659 | 1.83 | 2.21 | 2.70 |
| 32,768 | 128 | forward | eqx | 15.881 | 15.640–15.925 | 6.24 | 6.28 | 6.77 |
| 32,768 | 128 | full_vjp | previous | 50.966 | 50.779–50.978 | 4.78 | 5.60 | 6.09 |
| 32,768 | 128 | full_vjp | optimized | 49.548 | 49.440–49.592 | 4.68 | 5.60 | 6.09 |
| 32,768 | 128 | full_vjp | eqx | 57.289 | 57.033–57.449 | 15.40 | 15.46 | 15.96 |
| 32,768 | 256 | forward | previous | 26.462 | 26.450–26.501 | 4.05 | 4.42 | 4.91 |
| 32,768 | 256 | forward | optimized | 25.399 | 25.270–25.431 | 3.64 | 4.42 | 4.91 |
| 32,768 | 256 | full_vjp | previous | 100.550 | 100.218–100.595 | 9.50 | 11.19 | 11.69 |
| 32,768 | 256 | full_vjp | optimized | 97.451 | 97.104–97.791 | 9.30 | 11.19 | 11.69 |
| 131,072 | 128 | forward | previous | 52.970 | 52.621–52.987 | 4.51 | 4.88 | 5.37 |
| 131,072 | 128 | forward | optimized | 50.371 | 50.355–50.377 | 4.30 | 4.88 | 5.37 |
| 131,072 | 128 | full_vjp | previous | 209.393 | 209.251–209.612 | 9.79 | 10.11 | 10.61 |
| 131,072 | 128 | full_vjp | optimized | 205.162 | 204.390–205.255 | 9.69 | 10.11 | 10.61 |
| 131,072 | 64 | forward | previous | 26.571 | 26.293–26.574 | 2.88 | 3.43 | 3.92 |
| 131,072 | 64 | forward | optimized | 25.283 | 25.133–25.464 | 2.67 | 3.43 | 3.92 |
| 131,072 | 64 | full_vjp | previous | 123.856 | 123.359–123.887 | 5.54 | 6.41 | 6.90 |
| 131,072 | 64 | full_vjp | optimized | 121.272 | 121.139–121.318 | 5.44 | 6.41 | 6.90 |

## Implementation and qualification

[Adjoint scheduling](RADIAL_GRAPH.md) covers role pruning, shared coupling and the uniform GEMM layout.
TP forward and left/angular adjoints write into final block slices through internal mutation-declared
Triton operations. Existing allocation-returning uvu APIs share the raw-kernel launch helpers.
The joint angular reduction writes into its final slice. Hidden/affine GEMMs and final accumulation stay
unchanged, as do TP kernel arithmetic, block caps and first-weight caching. There is no new N dispatch.
Joint-channel blocks cap at 8192 edges/2 GiB weights; other channels at 16384 edges/4 GiB.

Source job 91328 passes 40 role checks, 22 radial integration checks, 33 joint/tail/second checks and
59 generic uvu checks. Independent dense CG checks every 131072 edge and every final-affine parameter
at C256; maximum normalized error is 1.67e-6. Geometry Hessians/live cotangents/force parameters pass.
Installed-wheel checks cover old/new APIs, actual EQX chain parity and empty batches. The three original
APIs retain bitwise v0.1.5 values/VJPs. Outer PyTorch 2.7.1 AOT double backward is unsupported; force-loss
training uses ordinary outer PyTorch with compiled internal blocks; see [usage](RADIAL.md).

## Earlier ordinary-composition evidence

Job 91260 measured ordinary ICTC composition with the complete edge-weight tensor: C64/N16384 full
VJP was 14.724 ms / 1.44 GiB allocated versus 15.101 ms / 2.44 GiB for that cohort's bounded wrapper.
C128/N32768 ordinary was 43.970 ms / 9.64 GiB versus bounded 52.221 ms / 4.78 GiB.
Ordinary composition ran out of memory at C256/N131072. This shows the speed/memory tradeoff; these
historical times must not be divided by current times from a different cohort.

Private evidence: ICTC-O2-HP-local-only/extensions/writeback_20261014/. The preceding uniform-layout
cohort remains in extensions/latency_20261014/. Existing v0.1.5 uuu figures describe a separate operator.
