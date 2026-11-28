# V3-driven O(2) optimization history, 24–27 November 2026

This development snapshot preserves the original optimization commits, including failed candidates, and the receipts needed to interpret them. Its archived v6 source had unresolved L7 training and L8 capture gates. The [subsequent v7 repair](../20261127-stability/README.md) passed the complete registered original gate matrix with source `13c9328`; additional bounded stability validation is recorded there separately.

## Git history and recovery

The operator work branches from upstream `96815838f470652dce3d0e9d5cf7fe2978e949ca`. All 18 subsequent operator commits retain their original IDs and parent links. Six stage branches and 19 `snapshot/*` tags are published to [LeutOrca/ICTC-O2-HP](https://github.com/LeutOrca/ICTC-O2-HP). [HISTORY.json](HISTORY.json) records every commit, tree, parent, branch and tag object.

| Stage branch | Tip | Recorded outcome |
| --- | --- | --- |
| `codex/uvw-triton-v1` | `df12a3a` | Tiled IEEE FP32 channel contractions. E/F diagnostic passed; four parameter gradients and 50 actual force-loss mixed roles failed. |
| `codex/uvw-fp32-repair-v2` | `a80f0dc` | Five common-state comparisons passed. The separate independent three-step trajectory retained 202 failed comparisons. |
| `codex/uvw-kgrad-rng-v3` | `c05bc7b` | Recursive physical layouts and RNG/diagnostic isolation. Original independent three-step gates passed; no whole-network speed or allocated-memory gain. |
| `codex/uvw-target-gemm-v4` | `8324b33` | Unconsumed K-adjoint pruning and CUDA SIMT eight-FMA weight contraction. Three-step gates passed; training wall time remained close to Torch. |
| `codex/bounded-bilinear-v5` | `dd7a3db` | Complete/tail bilinear fusion with recursive VJP and mixed second derivatives. Independent three-step gates passed. Fixed-workload wall time improved approximately 1.985× versus the matched Torch reference. |
| `codex/angular-fp32-repair-v6` | `5764f66` | Grid-axis repair and physical-stride FP32 plans for additional orders. Partial qualification only, detailed below. |

The earlier frozen stage-one baseline `e4763c76bf183499357e40cb39cc0934bb98779b` belongs to a separate research Git history. Its original history is published under [`archive/icml-o2-stage1-baseline-20261124`](https://github.com/LeutOrca/ICTC-O2-HP/tree/archive/icml-o2-stage1-baseline-20261124). It is not an ancestor of the operator branch above. The [baseline registry](evidence/baselines/icml-o2-stage1-20261124-r1/REGISTRY.json) and [verification receipt](evidence/baselines/icml-o2-stage1-20261124-r1/VERIFICATION.json) preserve the original source identities.

The code branch and snapshot tags allow checking out every operator candidate independently. Publishing an upstream PR does not transfer fork tags or the separate baseline branch to upstream automatically. Preserve those refs when retaining the historical record, and use a merge that preserves commits if upstream also needs the original commit sequence.

## Archived v6 qualification, candidate `5764f66`

These results come from the completed AMD audit **120792**, rather than the older progress notes that still describe running jobs.

| Scope | E/F / forward | Training / derivative gate |
| --- | --- | --- |
| 144 operator configurations, L1–L8 | All corresponding configurations passed | All corresponding configurations passed |
| 240 local permutation/tail/derivative regressions | Passed | Passed, 1,440 roles in total |
| Original projected L4/M2 checkpoint anchor | Passed | Independent three steps passed |
| Structural L1–L6 models | Passed | Independent three steps passed |
| Structural L7 model | Passed | Failed |
| Structural L8 model | Not captured | Not captured |

The 144 operator configurations cover UVW canonical/full parity and canonical bilinear products at N=16, 256 and 4096 in their registered inference/training modes. They are not a statement that the V3 whole network uses a full-parity architecture. Local regressions include nonzero errors within the approved gates and do not establish global bitwise equality.

At L7, the second update differs in `blocks.2.ffn.so3_linear_2.bias` and its EMA entry. Their relative L2 errors are approximately `1.0122e-5`, exceeding the unchanged `1e-5` threshold. The absolute errors are approximately `6.33e-13` and `6.33e-16`. They remain failures.

The original L8 capture ran out of memory during the second mixed-derivative diagnostic. Retry **120791** then failed before model computation: the copied local `coverage.py` shadowed Numba's `coverage` dependency and asserted that the GPU job was in `CPU-AMD`. The native, native-repeat and candidate records all contain this import error. The proposed earlier release of old `.grad` tensors therefore remains unvalidated by that retry. A Slurm `COMPLETED` status is not a numerical pass.

Direct receipts:

- [Whole-network and operator audit, 120792](evidence/runs/o2-angular-repair-a100-20261127-r1/gate-audits/120792/RESULT.json)
- [Local regression audit, 120793](evidence/runs/o2-angular-repair-a100-20261127-r1/local-audits/120793/RESULT.json)
- [L8 retry error, 120791](evidence/runs/o2-angular-repair-a100-20261127-r1/gate-captures/120791/network-L8-native/RESULT.json)
- [Actual derivative shape coverage](evidence/runs/o2-angular-repair-a100-20261127-r1/ACTUAL-COVERAGE.json)

No latency, throughput or capacity measurement has been run for `5764f66`. The measurement preparation script requires the complete registered gate matrix to pass.

## Qualified fixed-workload result, candidate `dd7a3db`

The model is the independent projected `equiformerv3-ictc-o2-canonical-min-v1`, with seven blocks and C128 at L4/M2. The comparison uses the same model, checkpoint, input, graph semantics, precision, losses and native optimizer on both backends. The E/F workload has 600 atoms and 31,512 directed edges. The training workload has 100 atoms and 5,252 edges. The two workloads have different sizes and must not be compared as inference-versus-training cost for the same input.

| Mode | Matched Torch median | `dd7a3db` median | Speedup | Torch allocated | Candidate allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| E/F | 3.620289 s | 1.823517 s | 1.9853× | 7.586998 GiB | 7.587051 GiB |
| Complete training step | 1.839475 s | 0.926420 s | 1.9856× | 11.331550 GiB | 11.343708 GiB |

Each mode/backend used three fresh processes with three warm-ups and five synchronized samples per process. Allocated memory did not improve. Reserved memory was lower in that measurement and does not establish a capacity improvement. Three-step acceptance does not establish a long-term training trajectory or public-dataset prediction accuracy.

The [bounded bilinear report](evidence/runs/o2-bounded-bilinear-a100-20261126-r1/REPORT.zh-CN.md) records the full protocol and historical failures. The [hotspot analysis](evidence/runs/o2-bounded-hotspot-a100-20261126-r1/REPORT.zh-CN.md) identifies remaining rotation, basis-transform, layout and derivative costs. Kernel-category shares use the sum of actual GPU kernel durations, not synchronized wall time.

## Evidence inventory and validation boundaries

[SOURCES.json](SOURCES.json) records the original workspace-relative path, byte count and SHA256 of every copied evidence file. The `evidence/` tree contains unchanged contracts, reports, control scripts and selected machine-readable receipts from the baseline, canonical model preparation, diagnosis and optimization runs. The original documents may contain local absolute paths and older process-status notes. Those are retained as historical evidence. The current status above follows the final audit receipts.

The copied controls preserve the original HPC environment and workspace assumptions. They are not a standalone portable V3 benchmark distribution. Raw tensor captures, checkpoints, profiler traces and large recovery archives remain in their original storage and are not included here. Original manifest/hash references remain in the copied receipts. The archived controls and reports are outside the installed library package.

Energy outputs and sums use FP64. Other arithmetic, accumulation, derivatives and error statistics use FP32. TF32, AMP, internal half precision and CUDA Graph are disabled. Each role retains both the original componentwise gate `abs(candidate-reference) <= 2e-5 + 2e-5*abs(reference)` and relative L2 `<= 1e-5`; local epsilon gates remain `<= 1e-5`. ULP is descriptive. Backends use their own true-loss cotangents and continuous three-step states, without restoring model, optimizer, EMA, scheduler or RNG inside the trajectory.

The original publication checks verified copied file hashes, preserved Git refs, the unchanged v6 operator source tree and whitespace. They did not rerun GPU qualification. L7 correction and a valid L8 retry were pending at that archived snapshot; current qualification follows the separate v7 receipts. Fresh-process performance/capacity must be remeasured for each new numerical source.
