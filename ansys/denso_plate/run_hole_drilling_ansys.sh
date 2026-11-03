#!/usr/bin/env bash
# run_hole_drilling_ansys.sh -- hole-drilling check (Tsuda et al. 2026 inputs) in Ansys MAPDL (IKM).
# Guide: research/GUIDE_ikm_ansys.md, Part II
#   bash ansys/denso_plate/run_hole_drilling_ansys.sh coarse          # 77 layers, ~34k elements (first)
#   NP=8 bash ansys/denso_plate/run_hole_drilling_ansys.sh            # production deck, 154 layers
#   ANSYS=ansys241 ... to name the MAPDL executable
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ANS="${ANSYS:-mapdl}"; NP="${NP:-4}"; PY="${PY:-python3}"
GEN="$ROOT/abaqus/denso_plate/gen_hole_drilling.py"
if [ "${1:-}" = "coarse" ]; then
  "$PY" "$GEN" --layers 77 --na 8 --nc 20 --tag coarse --out "$HERE/jobs"; JOB=denso_hole_coarse
else
  "$PY" "$GEN" --out "$HERE/jobs"; JOB=denso_hole
fi
"$PY" "$HERE/gen_hole_drilling_ansys.py" "$HERE/jobs/$JOB.inp" --out "$HERE/jobs"
cd "$HERE/jobs"
rm -f "${JOB}_ansys_strain.csv"
"$ANS" -b -np "$NP" -j "$JOB" -i "$JOB.dat" -o "$JOB.out"
"$PY" "$HERE/gen_hole_drilling_ansys.py" --csv2json "$JOB"
"$PY" "$ROOT/abaqus/denso_plate/compare_hole_drilling.py" "$JOB" --result "${JOB}_ansys.json"
