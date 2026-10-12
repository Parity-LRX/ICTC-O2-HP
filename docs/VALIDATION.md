# Validation provenance

The qualified 0.1.5 implementation uses one execution policy across positive N
and two-stage weight reduction. Full-parity m_max=4–6 uses input-adjoint
fusion. The package passed seven installed-wheel checks and 73 API contracts.
Focused small/large-N profiler and mathematical-reference checks also passed.
See [execution policy](EXECUTION.md).

GPU arrays90864/90867 completed all N16384/32768/65536/131072, five C values,
six maximum frequencies and both parities:240 shapes/480 successful candidate
metrics. CPU report90868 confirms no missing records and60 maximum-N
original-reference families, checking every output/input-gradient row and all
weights at1e-5. Maximum normalized error is8.10846e-7. EQX has479 successful
metrics and one isolated largest-shape full-VJP OOM. All239 comparable full-VJP memory
triples are lower than EQX. Final scientific figures use this completed0.1.5
cohort, with separate data/geometry/font and header-layout audits.


## Release-package verification

Final release job 90912 completed on A100 with exit code 0:0. The wheel and
source distribution passed version/content/document-link checks. All eight
Python modules matched the previously qualified 0.1.5 wheel. Seven fresh
installed public API checks passed against the mathematical reference, with
zero reported output, input-gradient and path-weight-gradient differences.
The original 73 API contracts, 60 maximum-N reference families and complete
480-metric scan remain the numerical evidence. Current eqx measurement and
reference records are included in [benchmarks/eqx](../benchmarks/eqx/README.md);
general API-contract scripts and original release builds remain in the archive.
See [release layout and reproduction](RELEASE.md).
