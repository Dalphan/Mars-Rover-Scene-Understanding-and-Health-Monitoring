# Reusable Figure Sources

These files support diagrams and plots used by the dissertation. Run the
commands below from the repository root.

## Quantization chart from the current chapter table

[`render_miou_energy.py`](quantization/render_miou_energy.py) reads the
20 measurements in
[Chapter 5](<../latex/Chapters/5 Evaluation of Lightweight Scene Understanding.tex>).
The source table contains the current test mIoU and GPU-energy measurements.

With Matplotlib installed in your active Python environment, render a preview:

```bash
python thesis/figures/quantization/render_miou_energy.py --output outputs/thesis_figures/quantization_miou_energy.png
```

A Windows renderer is also available as
[`render_miou_energy.ps1`](quantization/render_miou_energy.ps1).
Pass the source explicitly so it uses your checkout location:

```powershell
./thesis/figures/quantization/render_miou_energy.ps1 -Source "thesis/latex/Chapters/5 Evaluation of Lightweight Scene Understanding.tex" -Output "outputs/thesis_figures/quantization_miou_energy.png"
```

The dissertation image is
[`quantization_miou_energy.png`](../latex/Images/quantization_miou_energy.png).
Review a regenerated preview before replacing it.

## Historical plotting snapshot

[`quantization_results.csv`](quantization/quantization_results.csv) and
[`plot_quantization_tradeoff.py`](quantization/plot_quantization_tradeoff.py)
preserve an earlier curated snapshot. This CSV has a different scope from
the current chapter table.

To reproduce that historical plot with Matplotlib:

```bash
python thesis/figures/quantization/plot_quantization_tradeoff.py --output outputs/thesis_figures/quantization_historical.png
```

## Editable architecture diagram

[`segmentation_architectures_overview.svg`](segmentation/segmentation_architectures_overview.svg)
is the editable segmentation architecture overview. Dissertation-ready
images are stored in [the LaTeX image directory](../latex/Images/).
