"""to_calculix.py -- copy a deck from gen_denso_plate.py (elastic) or gen_denso_process.py
(heat transfer) into CalculiX 2.21 syntax, for a licence-free cross-check. Only keyword
spellings and print requests change; mesh, materials, loads and steps are untouched.

    python3 abaqus/denso_plate/to_calculix.py jobs/denso_asymmetric.inp       # elastic deck
    python3 abaqus/denso_plate/to_calculix.py jobs/denso_heat_asymmetric.inp  # heat deck
    ccx -i jobs/denso_asymmetric_ccx

Changes: TYPE=ORTHOTROPIC -> ORTHO; *STEP parameters NAME/NLGEOM dropped; EE -> ME
(mechanical strain) and U printed for the mid-plane node sets (elastic deck);
*HEAT TRANSFER gets DIRECT (CalculiX's fixed increments; Abaqus uses fixed increments when
DELTMX is omitted) and NT is printed for the centre column NCOL instead of written for all
nodes (heat deck). The stress deck of the process analysis is NOT convertible (UMAT).
"""
import re
import sys


def convert(src):
    s = open(src).read()
    heat = "TYPE=DC3D8" in s
    if "*USER MATERIAL" in s:
        sys.exit("process stress deck uses a UMAT: not convertible to the stock CalculiX build")
    s = s.replace("TYPE=ORTHOTROPIC", "TYPE=ORTHO")
    s = re.sub(r"^\*STEP, NAME=\w+", "*STEP", s, flags=re.M)
    s = s.replace(", NLGEOM=NO", "")
    if heat:
        s = s.replace("*HEAT TRANSFER, END=PERIOD", "*HEAT TRANSFER, DIRECT")
        s = s.replace("*OUTPUT, FIELD, FREQUENCY=1\n*NODE OUTPUT\n NT\n",
                      "*NODE PRINT, NSET=NCOL\n NT\n")
    else:
        s = s.replace(" S, E, EE, THE\n", " S, E, ME\n")
        s = s.replace("*EL PRINT, ELSET=ECOL, SUMMARY=NO\n EE\n", "*EL PRINT, ELSET=ECOL\n ME, S\n")
        s = s.replace("*END STEP", "*NODE PRINT, NSET=MIDY\n U\n*NODE PRINT, NSET=MIDX\n U\n*END STEP")
    out = src[:-4] + "_ccx.inp"
    open(out, "w").write(s)
    print("wrote", out)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        convert(f)
