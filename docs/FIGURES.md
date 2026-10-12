# Reading the performance figures

The current scan axes are independent tensor-product rows `N`, channel multiplicity
`C`, maximum O(2) frequency `m_max=1,...,6`, and canonical/full parity.
See [API](API.md) for the relation to SO(3)
angular momentum `l`; equal maximum indices do not make the operators or their
channel multiplicities identical.

## Throughput and memory together

Throughput ratios are **ICTC-O2 HP / EquivariantX**. Memory ratios are
**EquivariantX / ICTC-O2 HP**. Both ratios favour ICTC-O2 HP above 1, with the
direction printed in each figure. For example, throughput ratio 1.12 means
12% greater throughput, while memory ratio 2 means half the peak memory.
Below-1 cells are preserved, and slower timings are listed in the complete
report. Memory increases, including small persistent-buffer overheads, are
also accounted for separately; a rounded label of 1.00 is not exact equality.

Forward + full VJP measures forward execution, both input gradients and all
trainable path-weight gradients. Forward-only panels describe a different
operation. Native input-layout preparation happens before timing for both
libraries. `N` counts TP rows; these measurements do not measure the cost of a
complete atomistic model, its neighbours or layout conversion in that model.

Allocated and reserved are distinct PyTorch allocator peaks from fresh
processes. These totals include live inputs, parameters, outputs/adjoints and
the precreated VJP cotangent, which is also retained during forward-only
measurement. NVML figures show separately sampled memory of the measured
process, including its context/cache; short transients can be missed.
The strict FP32, TF32/AMP-off protocol is unchanged between figures.

## Variability and OOM

Timing medians use the recorded CUDA-event samples. A dagger, †, marks
overlapping IQRs in paired measurements; this is descriptive variability,
not a formal significance test. An asterisk, *, marks independent-process
timing used after a joint comparison process exceeded memory. Such timing
does not contribute a paired-IQR win.

OOM has no finite measured throughput or peak-memory value. In the 3D plot,
white diagonal hatching frames the missing interpolation region, with an
`eqx OOM` label inside it. The adjoining measured vertices remain visible;
the hatch does not assert that all those vertices failed. The 2D plot labels
the actual failed cell. A joint-process OOM alone
does not establish that either implementation fails in an isolated process;
the independent measurements determine that boundary.

## Publication and exploration views

The current publication style uses the completed 0.1.5 cohort only. Final
figures are included in [benchmarks/eqx/figures](../benchmarks/eqx/figures/INDEX.md):
240 shapes/480 metrics and60 reference families, with no pending measurements.
They show all four N values, m_max=1–6 and separate parity panels.
Five C values receive separate all-N 2D landscapes; joint 3D figures use
m_max, N and channel count C as their spatial axes. Ratios appear only in
colour. All-N atlases retain the complete grid. Largest-N snapshots are
explicitly identified as slices.

Throughput uses ICTC/EQX rows/s; memory uses EQX/ICTC peak occupancy. Every
relative heatmap cell and colourbar tick carries a multiplication symbol.
The 3D height axis shows actual C ticks at log2 spacing; N ticks16/32/64/128
are in units of1,024 rows. Both ratio colour scales are centred at1x, and
values above1x favour ICTC. For
example, a memory ratio of2.19x means EQX uses2.19 times the ICTC footprint.
Allocated/reserved absolute GiB atlases for both libraries remain separate
figures. Colour scales are common across N/C/parity within each metric and
operation. Tiny increases are marked even when their ratio rounds to1.00x.
Hatched EQX OOM cells have no
finite ratio; the 3D white hatch is an annotation, not interpolated data. Editable SVG and
embedded-font PDF accompany PNG.
Comparison titles are centred and identify the metric, for example
`ICTC-O2 HP vs eqx: throughput and peak memory`. A centred subtitle gives the
O(2) operator, forward/backward mode and N/C/frequency range. The release
version is a quiet corner label. Colourbars use at most four ticks; explanatory methods,
coverage counts and interpolation details are in the companion captions/data
audits. OOM labels sit inside their hatched 3D regions; 2D figures keep
one short key for timing and memory symbols. Eight-panel atlases place the
horizontal-axis title only on the bottom row, keeping neighbouring rows clear.
Canonical parity and full parity headings are centred above each 2D panel.
The typographic/layout reference is the
[Nature research figure guide](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/):
standard editable sans-serif text, visible axis units and non-overlapping
labels. Centred headings are a layout choice for these comparison plates.

The filled channel view uses one vertical m_max–C sheet per measured N, with
continuous bilinear colour interpolation between neighbouring measured C/m
settings. The connected view joins the C256, N131072 and m_max=6 boundary
faces, including C–m and C–N side walls. It exposes boundary slices; interior
cases remain visible in the companion sheets and exact heatmaps. Interpolation
preserves measured corners, stays within their value range and requires all
four source corners; it never fills missing or OOM data. The captions and
geometry audits distinguish display interpolation from extra measurements.
Channel gradients can be weak when measured ratios are nearly independent of C.
Dense bilinear source vertices now receive continuous vertex-colour interpolation
along C and across joined faces. Shared vertices are checked for identical ratios;
the operation and ratio scales do not change. The smooth surface fills are embedded
at450dpi in PDF/SVG to avoid polygon stitching seams. Labels, axes, measured points
and the OOM hatches remain vector; SVG text remains editable.

The completed 0.1.5 study uses m_max=1–6 and displays every measured N.
Current figure data, renderer and exported PNG/SVG/PDF files are included in
[benchmarks/eqx](../benchmarks/eqx/README.md). Earlier figure versions remain
in the separate release archive.
