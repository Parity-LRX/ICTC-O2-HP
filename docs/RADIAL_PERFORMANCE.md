# ICTC-O2 HP radial MLP + TP vs EquivariantX

Development version 0.1.6.dev0; A100-SXM4 40GB, PyTorch 2.7.1+cu128, Triton 3.3.1.
Strict FP32, TF32/AMP disabled. Two warmups and seven CUDA-event samples per metric.
Both full chains use torch.compile(fullgraph=True,dynamic=False).

The measured chain is Linear(16,64) -> SiLU -> Linear(64,64) -> SiLU ->
Linear(64,P*C,bias=False) -> V=1 TP. First/output full parity, angular
canonical parity, m_max=6. N counts edges; gathering/scattering and the
rest of a message-passing model are outside this chain.

EQX uses its actual native u1u with external per-edge weights. Every TP
path and network parameter is matched. Basis/layout conversion is outside
timing; phase differences are absorbed into the final affine parameters.
Full VJP includes forward, all three input gradients and every network parameter.

| N edges | C | metric | ICTC fusion ms | EQX ms | speed EQX/ICTC | ICTC allocated GiB | EQX allocated GiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 16,384 | 64 | forward | 3.847 | 4.094 | 1.06× | 1.31 | 1.57 |
| 16,384 | 64 | full_vjp | 15.386 | 15.478 | 1.01× | 2.44 | 3.88 |
| 32,768 | 128 | forward | 14.574 | 16.026 | 1.10× | 3.23 | 6.24 |
| 32,768 | 128 | full_vjp | 57.987 | 58.309 | 1.01× | 5.89 | 15.40 |
| 131,072 | 256 | forward | 107.785 | OOM | — | 11.35 | — |
| 131,072 | 256 | full_vjp | 485.410 | OOM | — | 21.59 | — |

OOM has no speed or memory ratio. Allocated peaks include inputs,
outputs, cotangents and workspaces. Raw records also retain reserved
and sampled NVML process peaks; the latter include context/cache and
are not exact transient peaks. Small timing differences should be
assessed against the retained sample distributions.

Forward is 6–10% faster in the two comparable shapes. Their full-VJP
differences are below 1% and do not establish a stable speed advantage.
The C128 full-VJP allocated peak is 38% of EQX's. At N131072/C256, both
EQX and the ordinary ICTC radial-MLP/TP composition run out of memory.
The bounded operator completes with a 21.59 GiB full-VJP allocated peak.

The ordinary ICTC composition remains faster at N32768/C128: 13.977 ms
forward and 51.730 ms full VJP, with a 5.66 GiB full-VJP allocated peak.
The bounded wrapper improves maximum feasible edge-batch size; it does
not improve speed or memory over that composition in every shape.

The fusion caps blocks at 16,384 edges and 4 GiB of FP32 weights.
The cap depends on path/channel sizes and is independent of N.
V=1 weight adjoints group paths by output entry and reuse feature loads.
Training retains the first bounded block's weights; remaining blocks
are regenerated in backward. Local weights and adjoints still exist,
and storage is bounded by the cap. The implementation combines compiled
cuBLAS and TP kernels inside one operator with bounded buffers.

Numerical evidence includes independent dense CG outputs and all input
and parameter gradients, rotation/reflection covariance, second VJP,
force-loss training through a nonlinear radial trunk, real EQX parity
across a block boundary and all-edge/all-parameter largest-shape checks.
The changed V=1 adjoint passed 30 focused checks, the radial operator
passed 22 checks, and actual EQX-chain parity passed 12 checks. All 131072
edges and all final-affine parameters were checked against independent
dense CG; maximum normalized error was 3.4e-6.

Installed-wheel checks cover both new APIs, the radial example and all
three original APIs, whose outputs and VJPs remain bitwise identical to
v0.1.5. Force-loss double backward is qualified with an ordinary outer
PyTorch model and compiled internal blocks. Outer PyTorch 2.7.1 AOT
compilation raises its unsupported-double-backward error; see [usage](RADIAL.md).

Private raw evidence: ICTC-O2-HP-local-only/extensions/radial_20261013/.
The existing v0.1.5 uuu benchmark and figures remain a separate cohort.
