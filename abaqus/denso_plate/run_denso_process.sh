#!/usr/bin/env bash
# run_denso_process.sh -- Denso PROCESS analysis on the Abaqus host (fifa):
# heat transfer -> sequentially coupled stress with the UMAT denso_cryst_umat.f.
# Guide: research/GUIDE_fifa_abaqus.md (section 5)
#
#   bash abaqus/denso_plate/run_denso_process.sh               # both cases
#   bash abaqus/denso_plate/run_denso_process.sh asymmetric    # one case
#   ABAQUS=abq2024 CPUS=8 bash abaqus/denso_plate/run_denso_process.sh
# Needs a Fortran compiler linked to Abaqus (the UMAT), unlike run_denso_plate.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ABQ="${ABAQUS:-abaqus}"; CPUS="${CPUS:-4}"; PY="${PY:-python3}"

"$PY" "$HERE/gen_denso_process.py"
cd "$HERE/jobs"
CASES="${*:-$(cat manifest_process.txt)}"
for c in $CASES; do
  echo "=== $c: heat transfer ==="
  "$ABQ" job="denso_heat_$c" cpus="$CPUS" interactive double ask_delete=OFF
  echo "=== $c: stress (UMAT) ==="
  "$ABQ" job="denso_proc_$c" user="$HERE/denso_cryst_umat.f" cpus="$CPUS" interactive double ask_delete=OFF
  "$ABQ" python "$HERE/extract_denso_process.py" "$c"
done
"$PY" "$HERE/compare_denso_process.py" $CASES --fig
