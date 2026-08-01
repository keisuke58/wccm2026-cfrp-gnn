#!/usr/bin/env bash
# run_all.sh -- submit the CFRTP Abaqus jobs and post-process, on YOUR server
# (the one with an Abaqus license + a Fortran compiler linked to Abaqus).
#
# Usage (from your own SSH session on the Abaqus box):
#   git clone <this repo> && cd wccm2026-cfrp-gnn/abaqus   # or rsync it over
#   bash run_all.sh                                        # runs both jobs + summary
#   bash run_all.sh cure                                   # cure job only
#   bash run_all.sh delam                                  # delamination job only
#   bash run_all.sh sanity                                 # 1-element free-contraction check
#   bash run_all.sh ve                                     # viscoelastic cure job
#   bash run_all.sh crystve                                # crystallization-coupled VE job
#   bash run_all.sh crystpeek                              # carbon/PEEK crystallization validation
#   bash run_all.sh crystpeekhl                            # carbon/PEEK Hoffman-Lauritzen (physical time)
#   bash run_all.sh daikinpfa                              # Daikin fluoropolymer (PFA) HL -- PLACEHOLDER params
#   bash run_all.sh delam3d                                # 3D mixed-mode delamination
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
run_sanity() {
  echo "=== 1-element free-contraction sanity (UMAT) -- expect residual ~0 ==="
  "$ABQ" job=cfrtp_1elem_sanity user=cfrtp_cure_umat.f interactive double
  echo "  -> cfrtp_1elem_sanity.odb"
}
run_ve() {
  echo "=== cure residual stress (VISCOELASTIC UMAT) ==="
  "$ABQ" job=cfrtp_cure_residual_ve user=cfrtp_cure_umat_ve.f interactive double
  echo "  -> cfrtp_cure_residual_ve.odb"
}
run_crystve() {
  echo "=== residual stress (CRYSTALLIZATION-COUPLED VISCOELASTIC UMAT) ==="
  "$ABQ" job=cfrtp_cryst_residual_ve user=cfrtp_cryst_umat_ve.f interactive double
  echo "  -> cfrtp_cryst_residual_ve.odb"
}
run_crystpeek() {
  echo "=== carbon/PEEK crystallization VALIDATION (crystallization-coupled VE) ==="
  "$ABQ" job=cfrtp_cryst_peek_validation user=cfrtp_cryst_umat_ve.f interactive double
  echo "  -> cfrtp_cryst_peek_validation.odb"
}
run_crystpeekhl() {
  echo "=== carbon/PEEK crystallization, HOFFMAN-LAURITZEN (physical time) ==="
  "$ABQ" job=cfrtp_cryst_peek_hl user=cfrtp_cryst_umat_hl.f interactive double
  echo "  -> cfrtp_cryst_peek_hl.odb"
}
run_daikinpfa() {
  echo "=== Daikin fluoropolymer (PFA) HL crystallization -- PLACEHOLDER params ==="
  "$ABQ" job=cfrtp_daikin_pfa_hl user=cfrtp_cryst_umat_hl.f interactive double
  echo "  -> cfrtp_daikin_pfa_hl.odb"
}
run_delam3d() {
  echo "=== 3D mixed-mode delamination (COH3D8 + B-K) ==="
  "$ABQ" job=cfrtp_delamination_3d interactive double
  echo "  -> cfrtp_delamination_3d.odb"
}

case "$WHICH" in
  cure)      run_cure ;;
  delam)     run_delam ;;
  sanity)    run_sanity ;;
  ve)        run_ve ;;
  crystve)   run_crystve ;;
  crystpeek) run_crystpeek ;;
  crystpeekhl) run_crystpeekhl ;;
  daikinpfa) run_daikinpfa ;;
  delam3d)   run_delam3d ;;
  all)       run_cure; run_delam ;;
  *) echo "usage: bash run_all.sh [cure|delam|sanity|ve|crystve|crystpeek|crystpeekhl|daikinpfa|delam3d|all]"; exit 2 ;;
esac

echo "=== post-processing (odb -> summary) ==="
"$ABQ" python postprocess.py

echo
echo "Done. Compare the printed summary with the Python seeds:"
echo "  cure  : residual sigma_xx range & warpage  vs cfrp_cure_residual_stress_fe.py / cfrtp_residual_stress_fe.py"
echo "  delam : peak reaction & delamination front vs cfrtp_delamination_2d_fe.py"
