#!/bin/bash
# Build the WCCM Beamer decks and copy tex+pdf to the LUHsummer26 viewing folder.
# (real copies, NOT symlinks — PDF viewers / VSCode do not follow symlinks.)
#   wccm_beamer.tex        : extended deck (architecture sweep + 4 variants)
#   wccm_beamer_paper.tex  : paper-faithful deck (Frontiers 2025; supervisor-requested)
set -e
cd "$(dirname "$0")"
DEST=/home/nishioka/LUHsummer26/40_Academic/WCCM2026
for doc in wccm_beamer wccm_beamer_paper; do
  [ -f "$doc.tex" ] || continue
  latexmk -pdf -interaction=nonstopmode "$doc.tex"
  cp -f "$doc.pdf" "$DEST/$doc.pdf"
  cp -f "$doc.tex" "$DEST/$doc.tex"
  echo "synced -> $DEST/$doc.{pdf,tex}"
done

# pympress speaker-notes build of the paper deck: slide | notes side-by-side.
# Open wccm_beamer_paper_notes.pdf in pympress -> presenter window shows the notes.
if [ -f wccm_beamer_paper.tex ]; then
  latexmk -pdf -interaction=nonstopmode \
    -jobname=wccm_beamer_paper_notes \
    -usepretex='\def\NOTESBUILD{}' wccm_beamer_paper.tex
  cp -f wccm_beamer_paper_notes.pdf "$DEST/wccm_beamer_paper_notes.pdf"
  echo "synced -> $DEST/wccm_beamer_paper_notes.pdf (pympress)"
fi
