#!/bin/bash
# Build the WCCM Beamer decks and copy tex+pdf to the LUHsummer26 viewing folder.
# (real copies, NOT symlinks — PDF viewers / VSCode do not follow symlinks.)
#   WCCM2026_Nishioka_CFRP_GNN.tex : paper-faithful deck (Frontiers 2025; the official talk)
#   WCCM2026_extended.tex          : extended deck (architecture sweep + 4 variants; research)
set -e
cd "$(dirname "$0")"
DEST=/home/nishioka/LUHsummer26/40_Academic/WCCM2026
for doc in WCCM2026_extended WCCM2026_Nishioka_CFRP_GNN; do
  [ -f "$doc.tex" ] || continue
  latexmk -pdf -interaction=nonstopmode "$doc.tex"
  cp -f "$doc.pdf" "$DEST/$doc.pdf"
  cp -f "$doc.tex" "$DEST/$doc.tex"
  echo "synced -> $DEST/$doc.{pdf,tex}"
done

# pympress speaker-notes build of the paper deck: slide | notes side-by-side.
# Open WCCM2026_Nishioka_CFRP_GNN_notes.pdf in pympress -> presenter window shows notes.
if [ -f WCCM2026_Nishioka_CFRP_GNN.tex ]; then
  latexmk -pdf -interaction=nonstopmode \
    -jobname=WCCM2026_Nishioka_CFRP_GNN_notes \
    -usepretex='\def\NOTESBUILD{}' WCCM2026_Nishioka_CFRP_GNN.tex
  cp -f WCCM2026_Nishioka_CFRP_GNN_notes.pdf "$DEST/WCCM2026_Nishioka_CFRP_GNN_notes.pdf"
  echo "synced -> $DEST/WCCM2026_Nishioka_CFRP_GNN_notes.pdf (pympress)"
fi
