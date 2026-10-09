#!/bin/bash
# Build the viva guide PDF.
#
#   bash docs/build_viva_guide.sh
#
# Runs xelatex twice so the table of contents and cross-references settle.
# Needs a Devanagari-capable font (Nirmala UI on Windows; change the
# \newfontfamily line in the .tex elsewhere).
set -u

cd "$(dirname "$0")"
TEX=VIVA_GUIDE.tex
PDF=VIVA_GUIDE.pdf

echo "building $TEX ..."
for pass in 1 2; do
    if ! xelatex -interaction=nonstopmode -halt-on-error "$TEX" > "build_pass$pass.log" 2>&1; then
        echo
        echo "XeLaTeX FAILED on pass $pass. Last errors:"
        grep -A3 "^!" "build_pass$pass.log" | head -40
        exit 1
    fi
done

PAGES=$(python -c "import pypdf; print(len(pypdf.PdfReader('$PDF').pages))" 2>/dev/null || echo "?")
echo
echo "  pages : $PAGES"
echo "  size  : $(du -h "$PDF" | cut -f1)"

# Missing glyphs are silent in the PDF but noisy in the log.
MISSING=$(grep -ci "missing character" build_pass2.log || true)
echo "  missing glyphs : $MISSING"

rm -f VIVA_GUIDE.aux VIVA_GUIDE.out VIVA_GUIDE.toc
[ "$MISSING" -eq 0 ] && echo "  ok: no missing characters."
