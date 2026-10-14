# Reusable figure sources

These small source files were preserved before removing temporary project
outputs on 14 October 2026.

- `quantization/render_miou_energy.py` and `render_miou_energy.ps1` read the
  current quantization table from Chapter 5 in `thesis/latex/Chapters/`.
- `quantization/plot_quantization_tradeoff.py` and `quantization_results.csv`
  preserve the earlier curated data snapshot and its plotting script.
  This historical CSV does not replace the current chapter table.
- `segmentation/segmentation_architectures_overview.svg` is the editable
  architecture diagram formerly stored under the segmentation documentation.

The figure used by the dissertation remains in `thesis/latex/Images/`.
Chart regeneration requires the appropriate Python packages or the existing
PowerShell renderer; the former root `.venv` was removed as requested.
