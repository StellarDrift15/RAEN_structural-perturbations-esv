# Figures

Six figures, one `.tex` each. Every file is a **standalone build of a figure that
`main.tex` already draws inline** — the `tikzpicture` body is copied verbatim out
of `main.tex`, so the PDF you build here is identical to what the paper renders.

| File | Figure in the paper | What it shows |
|---|---|---|
| `fig_overview.tex` | Fig. 1 (`fig:overview`) | all five methods hanging off the one shared sampling step |
| `fig_trap.tex` | Fig. 2 (`fig:trap`) | the GSM8K #131 strategy-breaking pair and its trap answer |
| `fig_pipeline.tex` | Fig. 3 (`fig:pipeline`) | ESV's three stages, shaded by what each one costs |
| `fig_gain_headroom.tex` | Fig. 4 (`fig:gain`) | Δ accuracy vs maj@4, with the pass@4 headroom band |
| `fig_lr_criterion.tex` | Fig. 5 (`fig:lr`) | measured likelihood ratios against the Eq. (2) bars |
| `fig_cost_accuracy.tex` | Fig. 6 (`fig:cost`) | accuracy gain against extra tokens |

`preamble-figures.tex` holds the colours and marker macros the figures share.
`main.tex` carries its own copy of those definitions, because the AAAI submission
source has to be a single `.tex` file.

## Building

Needs a TeX distribution with `standalone`, `tikz` and `pifont` (TeX Live,
MiKTeX, or just drop the folder into Overleaf). From inside this folder:

```sh
sh build.sh          # macOS / Linux / Git Bash
build.bat            # Windows cmd
pdflatex fig_lr_criterion.tex    # or one at a time
```

Each run produces a `.pdf` cropped to the picture's bounding box.

## Why these are `.tex` and not `.pdf`

The AAAI-27 graphics rules ask for `.pdf`/`.png`/`.jpg` at 300 dpi or better,
with all fonts embedded, no Type 3 fonts, and no `trim`/`clip` cropping. Building
the figures from TikZ satisfies every one of those by construction rather than by
inspection:

- **Vector, so DPI does not apply.** There is no raster layer to be under 300 dpi.
- **Fonts are the document's own.** The pictures inherit `times`/`helvet`, so
  every glyph is Type 1 and `pdflatex` embeds it. Nothing is Type 3, and nothing
  needs outlining.
- **`standalone` crops to the bounding box** at build time, so `trim`/`clip` is
  never used in the paper.
- **No `\resizebox`.** Each picture is sized by its own coordinates, so no text
  is scaled away from its intended point size.

Shipping pre-rendered PDFs would lose the first two properties the moment anyone
regenerated one from a plotting library, which is the usual way Type 3 fonts get
into a submission.

## Relationship to `main.tex`

`main.tex` does **not** `\includegraphics` these files, and should not: AAAI wants
a single-file source, and inline TikZ keeps the figures in one file with the text
that cites them. This folder is for working on a figure on its own — faster to
compile, and you can eyeball it cropped — and for the case where a venue insists
on separate graphics files.

If you ever do need the `\includegraphics` route, build the PDFs here and swap the
`tikzpicture` in `main.tex` for, e.g.:

```latex
\includegraphics[width=\columnwidth]{Figures/fig_gain_headroom.pdf}
```

keeping the surrounding `figure`/`caption`/`label` as they are. Only the overview
figure is full width (`figure*`, so `width=\textwidth`); the other five are
single-column.

## Editing

Edit the figure in `main.tex` and re-extract, or edit here and paste back — but
keep the two in sync, because the paper renders the inline copy. The numbers
plotted in the last three figures all come from the committed analyser reports
under `experiments/results/*/analysis/`; if a number changes there, it has to
change in the figure, the matching appendix table and the prose that quotes it.
