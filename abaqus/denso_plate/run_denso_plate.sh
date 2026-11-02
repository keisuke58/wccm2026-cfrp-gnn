#!/usr/bin/env bash
# run_denso_plate.sh -- Denso residual-strain plate on the Abaqus host (fifa).
# Guide: research/GUIDE_fifa_abaqus.md
#
#   bash abaqus/denso_plate/run_denso_plate.sh                 # symmetric + asymmetric (total)
#   bash abaqus/denso_plate/run_denso_plate.sh --part thermal  # any gen_denso_plate.py options
#   ABAQUS=abq2024 CPUS=8 bash abaqus/denso_plate/run_denso_plate.sh
#
# Steps: (1) write decks + references with python3, (2) run every deck in the manifest with
# Abaqus, (3) extract with Abaqus' Python, (4) compare with lamination theory (exit 1 on FAIL).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ABQ="${ABAQUS:-abaqus}"
CPUS="${CPUS:-4}"
PY="${PY:-python3}"

"$PY" "$HERE/gen_denso_plate.py" "$@"
cd "$HERE/jobs"
while read -r job; do
  [ -z "$job" ] && continue
  echo "=== $job ==="
  "$ABQ" job="$job" cpus="$CPUS" interactive double ask_delete=OFF
  "$ABQ" python "$HERE/extract_denso_plate.py" "$job"
done < manifest.txt
"$PY" "$HERE/compare_denso_plate.py" $(cat manifest.txt) --fig
