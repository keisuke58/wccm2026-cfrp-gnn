#!/usr/bin/env bash
# run_all.sh -- submit the CFRTP Abaqus jobs and post-process, on YOUR server
# (the one with an Abaqus license + a Fortran compiler linked to Abaqus).
#
# Usage (from your own SSH session on the Abaqus box):
#   git clone <this repo> && cd wccm2026-cfrp-gnn/abaqus   # or rsync it over
#   bash run_all.sh                                        # runs both jobs + summary
#   bash run_all.sh cure                                   # cure job only
#   bash run_all.sh delam                                  # delamination job only
#
# Nothing here reaches back to the sandbox; it runs entirely on your machine.
set -euo pipefail
cd "$(dirname "$0")"

ABQ="${ABAQUS:-abaqus}"          # override: ABAQUS=abq2023 bash run_all.sh
WHICH="${1:-all}"

run_cure() {
  echo "=== cure residual stress (UMAT) ==="
  "$ABQ" job=cfrtp_cure_residual user=cfrtp_cure_umat.f interactive double
  echo "  -> cfrtp_cure_residual.odb"
}
run_delam() {
  echo "=== mixed-mode delamination (built-in cohesive) ==="
  "$ABQ" job=cfrtp_delamination_mixedmode interactive double
  echo "  -> cfrtp_delamination_mixedmode.odb"
}

case "$WHICH" in
  cure)  run_cure ;;
  delam) run_delam ;;
  all)   run_cure; run_delam ;;
  *) echo "usage: bash run_all.sh [cure|delam|all]"; exit 2 ;;
esac

echo "=== post-processing (odb -> summary) ==="
"$ABQ" python postprocess.py

echo
echo "Done. Compare the printed summary with the Python seeds:"
echo "  cure  : residual sigma_xx range & warpage  vs cfrp_cure_residual_stress_fe.py / cfrtp_residual_stress_fe.py"
echo "  delam : peak reaction & delamination front vs cfrtp_delamination_2d_fe.py"
