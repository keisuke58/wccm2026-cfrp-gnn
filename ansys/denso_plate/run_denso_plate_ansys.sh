#!/usr/bin/env bash
# run_denso_plate_ansys.sh -- the Denso plate lamination check in Ansys Mechanical APDL (IKM).
# Guide: research/GUIDE_ikm_ansys.md
#
#   bash ansys/denso_plate/run_denso_plate_ansys.sh                 # symmetric + asymmetric
#   bash ansys/denso_plate/run_denso_plate_ansys.sh --part thermal  # any generator option
#   ANSYS=ansys241 NP=4 bash ansys/denso_plate/run_denso_plate_ansys.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ANS="${ANSYS:-mapdl}"; NP="${NP:-4}"; PY="${PY:-python3}"

"$PY" "$HERE/gen_denso_plate_ansys.py" "$@"
cd "$HERE/jobs"
while read -r job; do
  [ -z "$job" ] && continue
  echo "=== $job ==="
  "$ANS" -b -np "$NP" -j "$job" -i "$job.dat" -o "$job.out"
done < manifest.txt
"$PY" "$ROOT/abaqus/denso_plate/compare_denso_plate.py" --ansys --fig $(cat manifest.txt)
