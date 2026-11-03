#!/usr/bin/env bash
# run_hole_drilling.sh -- hole-drilling simulation after Tsuda et al. (2026) on the Abaqus host (fifa).
# Guide: research/GUIDE_fifa_abaqus.md, Part III
#   CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh          # full deck (154 layers, ~133k C3D8I)
#   bash abaqus/denso_plate/run_hole_drilling.sh coarse          # 77-layer check deck (~34k C3D8I)
#   CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh basis    # full deck, one run per eigenstrain component xx, yy, zz
#   CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh gauge    # full deck, gauge grids moved -0.2 / +0.2 mm radially
#   CPUS=8 bash abaqus/denso_plate/run_hole_drilling.sh overlap  # full deck with the ZONE set; grid shifts by overlap areas
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ABQ="${ABAQUS:-abaqus}"; CPUS="${CPUS:-4}"; PY="${PY:-python3}"
if [ "${1:-}" = "basis" ]; then
  for C in xx yy zz; do
    "$PY" "$HERE/gen_hole_drilling.py" --eig "$C" --tag "$C"
    (cd "$HERE/jobs" && "$ABQ" job="denso_hole_$C" cpus="$CPUS" interactive double ask_delete=OFF \
      && "$ABQ" python "$HERE/extract_hole_drilling.py" "denso_hole_$C" \
      && "$PY" "$HERE/compare_hole_drilling.py" "denso_hole_$C" --no-fig)
  done
  exit 0
elif [ "${1:-}" = "overlap" ]; then
  "$PY" "$HERE/gen_hole_drilling.py" --tag zone
  cd "$HERE/jobs"
  "$ABQ" job=denso_hole_zone cpus="$CPUS" interactive double ask_delete=OFF
  "$ABQ" python "$HERE/extract_hole_drilling.py" denso_hole_zone
  "$PY" "$HERE/compare_hole_drilling.py" denso_hole_zone --no-fig
  "$PY" "$HERE/gauge_overlap.py" denso_hole_zone
  exit 0
elif [ "${1:-}" = "gauge" ]; then
  for D in m02:-0.2 p02:0.2; do
    T="${D%%:*}"; V="${D#*:}"
    "$PY" "$HERE/gen_hole_drilling.py" --gauge-shift="$V" --tag "g$T"
    (cd "$HERE/jobs" && "$ABQ" job="denso_hole_g$T" cpus="$CPUS" interactive double ask_delete=OFF \
      && "$ABQ" python "$HERE/extract_hole_drilling.py" "denso_hole_g$T" \
      && "$PY" "$HERE/compare_hole_drilling.py" "denso_hole_g$T")
  done
  exit 0
elif [ "${1:-}" = "coarse" ]; then
  "$PY" "$HERE/gen_hole_drilling.py" --layers 77 --na 8 --nc 20 --tag coarse; JOB=denso_hole_coarse
else
  "$PY" "$HERE/gen_hole_drilling.py"; JOB=denso_hole
fi
cd "$HERE/jobs"
"$ABQ" job="$JOB" cpus="$CPUS" interactive double ask_delete=OFF
"$ABQ" python "$HERE/extract_hole_drilling.py" "$JOB"
"$PY" "$HERE/compare_hole_drilling.py" "$JOB"
