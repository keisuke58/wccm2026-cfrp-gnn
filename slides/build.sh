#!/bin/bash
# Build the WCCM Beamer deck and copy tex+pdf to the LUHsummer26 viewing folder.
# (real copies, NOT symlinks — PDF viewers / VSCode do not follow symlinks.)
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode wccm_beamer.tex
DEST=/home/nishioka/LUHsummer26/40_Academic/WCCM2026
cp -f wccm_beamer.pdf "$DEST/wccm_beamer.pdf"
cp -f wccm_beamer.tex "$DEST/wccm_beamer.tex"
echo "synced -> $DEST/wccm_beamer.{pdf,tex}"
