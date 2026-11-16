#!/bin/sh
# Build every figure to a cropped PDF. Run from inside docs/paper/Figures/.
set -e

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Install TeX Live or MiKTeX, or build these in Overleaf." >&2
  exit 1
fi

for f in fig_*.tex; do
  echo "==> $f"
  pdflatex -interaction=nonstopmode -halt-on-error "$f" >/dev/null
done

# keep the PDFs, drop the build litter
rm -f fig_*.aux fig_*.log fig_*.out

echo
echo "Built:"
ls -1 fig_*.pdf
