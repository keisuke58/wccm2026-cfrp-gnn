#!/usr/bin/env bash
# to_calculix.sh -- copy an Abaqus deck from gen_denso_plate.py into CalculiX (ccx 2.21) syntax,
# for a licence-free cross-check. Only keyword spellings and print requests change:
# TYPE=ORTHOTROPIC -> ORTHO, *STEP parameters dropped, EE -> ME (mechanical strain),
# U printed for the mid-plane node sets.
#   bash to_calculix.sh jobs/denso_asymmetric.inp     # -> jobs/denso_asymmetric_ccx.inp
#   ccx -i jobs/denso_asymmetric_ccx
#   (cd jobs && python3 ../compare_denso_plate.py denso_asymmetric --calculix denso_asymmetric_ccx.dat)
set -euo pipefail
in="$1"; out="${in%.inp}_ccx.inp"
sed -e 's/TYPE=ORTHOTROPIC/TYPE=ORTHO/' -e 's/^\*STEP, NAME=COOL, NLGEOM=NO/*STEP/' \
    -e 's/^ S, E, EE, THE/ S, E, ME/' -e 's/^\*EL PRINT, ELSET=ECOL, SUMMARY=NO/*EL PRINT, ELSET=ECOL/' \
    -e 's/^ EE$/ ME, S/' \
    -e 's/^\*END STEP/*NODE PRINT, NSET=MIDY\n U\n*NODE PRINT, NSET=MIDX\n U\n*END STEP/' \
    "$in" > "$out"
echo "wrote $out"
