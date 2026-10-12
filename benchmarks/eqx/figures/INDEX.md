# ICTC-O2 HP vs eqx

[Comparison scripts and complete results](../README.md) ·
[Current renderer](../scripts/render_figures.py).

Completed ICTC-O2 HP 0.1.5 grid: N=16,384/32,768/65,536/131,072;
C=32/64/96/128/256; m_max=1–6; canonical parity and full parity.
240 shapes, 480 operator metrics, 60 maximum-N reference families.

## Figures

- [Connected 3D comparison](N_channel_connected_forward_vjp.png) · [PDF](N_channel_connected_forward_vjp.pdf) · [SVG](N_channel_connected_forward_vjp.svg)
- [All measured N sheets](N_channel_layers_forward_vjp.png) · [PDF](N_channel_layers_forward_vjp.pdf)
- [C256 all-N comparison](N_landscape_forward_vjp_C256.png) · [PDF](N_landscape_forward_vjp_C256.pdf)
- [All-N throughput heatmaps](all_N_throughput_forward_vjp.png)
- [Allocated-memory ratios](all_N_allocated_forward_vjp.png)
- [Reserved-memory ratios](all_N_reserved_forward_vjp.png)
- [NVML process-memory ratios](all_N_process_forward_vjp.png)
- [ICTC absolute allocated memory](all_N_absolute_allocated_forward_vjp_candidate.png)
- [eqx absolute allocated memory](all_N_absolute_allocated_forward_vjp_equivariantx.png)

Each C also has its own all-N landscape heatmap. Forward-only counterparts,
absolute reserved memory and PNG/SVG/PDF exports are included in this folder.

## Data and methods

[All measurements](complete_available_matrix.csv), [figure captions](FIGURE_CAPTIONS.md),
[measurement state](MEMORY_MEASUREMENT.md), [data and font audit](figure_qa.json),
[sheet geometry](surface_geometry.json), [connected geometry](connected_geometry.json),
[OOM frames](oom_markers.json), [continuous-colour seams](smooth_surface_qa.json).

Throughput ratio is ICTC/eqx; memory ratio is eqx/ICTC. Both favour ICTC above1x.
The connected exterior uses boundary slices; N sheets and heatmaps retain the
interior. Continuous colour interpolation along C does not add measurements.
White diagonal-hatched frames mark interpolation cells touching EQX's actual
C256/full/m6/N131072 full-VJP OOM. The label sits inside the missing region;
no finite ratio is assigned. All 2D parity headings are centred per panel.

Final render: job90910 (0:0,1:25), frozen source
publication_v015_continuous_v13_20261012. PDF raster/SVG hatch QA:90911 (0:0).
All480 rows/6592 displayed cells,0 pending, shared-vertex ratio continuity,
bounded four-corner source interpolation, OOM source checks, embedded PDF
fonts, editable SVG text, centred headings and sparse legends passed.
PNG and rendered PDF review confirms smooth surfaces without stitching seams.
PDF/SVG embed450dpi smooth surface fills; labels, axes, measured points and
OOM hatches remain vector. Numerical records, colour scales and kernels unchanged.
