# Dissertation and Presentation Materials

This directory contains the dissertation
**Lightweight Scene Understanding and Synthetic Data-Driven Damage Inspection
for Martian Rovers**, together with reusable figures and presentation materials
shared by the two code projects.

## Contents

| Directory | Contents |
| --- | --- |
| `latex/` | Main document, chapters, bibliography, images, Sapthesis resources and a compiled PDF snapshot. |
| `presentation/` | VisionLab PowerPoint template for the thesis presentation. |
| `figures/` | Editable diagrams, chart renderers and curated plotting data. |
| `research/` | Selected writing-evidence records. |

## Building the dissertation

The main source is
[the LaTeX document](<latex/Lightweight Scene Understanding and Synthetic Data-Driven Damage Inspection for Martian Rovers.tex>);
a [compiled PDF snapshot](<latex/Lightweight Scene Understanding and Synthetic Data-Driven Damage Inspection for Martian Rovers.pdf>)
is also included. The PDF is a saved research artifact; compile the sources
to incorporate later edits.

Use a LaTeX distribution with the packages declared in the main document.
The checked-in source selects pdfLaTeX and the bundled Sapthesis class.
From the repository root:

```bash
cd thesis/latex
pdflatex -interaction=nonstopmode -halt-on-error "Lightweight Scene Understanding and Synthetic Data-Driven Damage Inspection for Martian Rovers.tex"
```

Repeat compilation until references and the table of contents settle.
The current bibliography is included directly from
[the bibliography chapter](latex/Chapters/Bibliography.tex).
Compile from `thesis/latex/` so chapter, image and class paths resolve.

## Presentation and figures

Use [the VisionLab PowerPoint template](<presentation/Presentazione standard VisionLab.pptx>)
for the presentation. This directory currently contains the template;
the presentation content is prepared separately.

[The figure guide](figures/README.md) explains editable sources and chart
regeneration. Figure renderers should write to a preview location first,
before replacing an image referenced by the dissertation.

The introduction's preserved evidence record is
[`writing_evidence.json`](research/introduction/writing_evidence.json).
It records research provenance; downloaded reference copies are not included.

## Template attribution

The bundled Sapthesis class retains its original
[README and license notice](latex/README). That upstream README describes the
complete template distribution; its separate class manual is not bundled here.
University logos, third-party illustrations and presentation materials retain
their respective attribution and terms.
