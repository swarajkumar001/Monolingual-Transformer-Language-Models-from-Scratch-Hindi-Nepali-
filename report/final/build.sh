#!/bin/bash
# Build the consolidated report and check it against the two hard rules:
# ten pages maximum, and no unwritten sections left in it.
#
#   bash report/final/build.sh
#
# Runs xelatex twice so cross-references settle. Needs a Devanagari-capable
# font -- Nirmala UI on Windows, Noto Sans Devanagari elsewhere (change the
# \newfontfamily line in the .tex if you are not on Windows).
set -u

cd "$(dirname "$0")"
TEX=final_report.tex
PDF=final_report.pdf
LIMIT=10

echo "building $TEX ..."
for pass in 1 2; do
    if ! xelatex -interaction=nonstopmode -halt-on-error "$TEX" > "build_pass$pass.log" 2>&1; then
        echo
        echo "XeLaTeX FAILED on pass $pass. Last errors:"
        grep -A3 "^!" "build_pass$pass.log" | head -40
        echo
        echo "(full log: report/final/build_pass$pass.log)"
        exit 1
    fi
done

if [ ! -f "$PDF" ]; then
    echo "no $PDF was produced"
    exit 1
fi

# Page count. pypdf is already a project dependency; a regex over the raw
# PDF miscounts because /Type /Page appears inside the page tree too.
PAGES=$(python -c "
import pypdf
print(len(pypdf.PdfReader('$PDF').pages))
" 2>/dev/null || echo 0)

echo
echo "  pages : $PAGES  (limit $LIMIT)"
SIZE=$(du -h "$PDF" | cut -f1)
echo "  size  : $SIZE"

# Unwritten sections.
TODOS=$(grep -c '\\TODO{' "$TEX" || true)
echo "  TODOs : $TODOS"

# Every headline number in the report must still match the artifact it came from.
FACTS=0
if python check_report_facts.py > facts_check.log 2>&1; then
    echo "  facts : $(head -1 facts_check.log)"
else
    echo "  facts : MISMATCH -- see report/final/facts_check.log"
    FACTS=1
fi

echo
STATUS=0
if [ "$PAGES" -gt "$LIMIT" ] 2>/dev/null; then
    echo "  OVER THE PAGE LIMIT by $((PAGES - LIMIT)). Cut before submitting."
    STATUS=1
fi
if [ "$TODOS" -gt 0 ]; then
    echo "  $TODOS section(s) still unwritten. They render in red -- search for \\TODO."
    STATUS=1
fi
if [ "$FACTS" -ne 0 ]; then
    echo "  a reported number no longer matches its source artifact. Fix before submitting."
    STATUS=1
fi
[ "$STATUS" -eq 0 ] && echo "  ready to submit: under the page limit, nothing unwritten."

rm -f final_report.aux final_report.out final_report.toc
exit $STATUS
