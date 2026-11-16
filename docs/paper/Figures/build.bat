@echo off
REM Build every figure to a cropped PDF. Run from inside docs\paper\Figures\.

where pdflatex >nul 2>nul
if errorlevel 1 (
  echo pdflatex not found. Install MiKTeX or TeX Live, or build these in Overleaf.
  exit /b 1
)

for %%f in (fig_*.tex) do (
  echo ==^> %%f
  pdflatex -interaction=nonstopmode -halt-on-error "%%f" >nul
  if errorlevel 1 (
    echo FAILED on %%f -- see %%~nf.log
    exit /b 1
  )
)

del /q fig_*.aux fig_*.log fig_*.out 2>nul

echo.
echo Built:
dir /b fig_*.pdf
