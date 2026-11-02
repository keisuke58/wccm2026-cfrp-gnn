#!/usr/bin/env bash
# run_hole_drilling.sh -- hole-drilling simulation after Tsuda et al. (2026) on the Abaqus host (fifa).
# Guide: research/GUIDE_fifa_abaqus.md, Part III
#   CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh          # full deck (154 layers, ~133k C3D8I)
#   bash abaqus/denso_plate/run_hole_drilling.sh coarse          # 77-layer check deck (~34k C3D8I)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ABQ="${ABAQUS:-abaqus}"; CPUS="${CPUS:-4}"; PY="${PY:-python3}"
if [ "${1:-}" = "coarse" ]; then
  "$PY" "$HERE/gen_hole_drilling.py" --layers 77 --na 8 --nc 20 --tag coarse; JOB=denso_hole_coarse
else
  "$PY" "$HERE/gen_hole_drilling.py"; JOB=denso_hole
fi
cd "$HERE/jobs"
"$ABQ" job="$JOB" cpus="$CPUS" interactive double ask_delete=OFF
"$ABQ" python "$HERE/extract_hole_drilling.py" "$JOB"
"$PY" "$HERE/compare_hole_drilling.py" "$JOB"
