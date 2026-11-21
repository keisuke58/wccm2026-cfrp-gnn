import sys
L, W, H = 0.08, 0.002, 0.00308
def bm(nx, nz, gate_mid, vent_mid):
    hdr = "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\nconvertToMeters 1;\n"
    if not (gate_mid or vent_mid):
        v = [(0,0,0),(L,0,0),(L,W,0),(0,W,0),(0,0,H),(L,0,H),(L,W,H),(0,W,H)]
        vs = " ".join("(%g %g %g)" % p for p in v)
        return hdr + f"""vertices ( {vs} );
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} 4 {nz}) simpleGrading (1 1 1) );
boundary
(
    inlet  {{ type patch; faces ( (0 4 7 3) ); }}
    outlet {{ type patch; faces ( (1 2 6 5) ); }}
    walls  {{ type wall;  faces ( (0 3 2 1) (4 5 6 7) ); }}
    sym0   {{ type symmetryPlane; faces ( (0 1 5 4) ); }}
    sym1   {{ type symmetryPlane; faces ( (3 7 6 2) ); }}
);
mergePatchPairs ();
"""
    zs = (0, H / 6, 5 * H / 6, H); ncz = (5, 22, 5)      # gate/vent = middle 2/3 of the thickness (22 of 32 cells)
    v = [p for z in zs for p in ((0,0,z),(L,0,z),(L,W,z),(0,W,z))]
    vs = " ".join("(%g %g %g)" % p for p in v)
    blocks = " ".join(f"hex ({' '.join(str(4*j+i) for i in range(8))}) ({nx} 4 {ncz[j]}) simpleGrading (1 1 1)" for j in range(3))
    xin = [f"({4*j} {4*j+4} {4*j+7} {4*j+3})" for j in range(3)]
    xout = [f"({4*j+1} {4*j+2} {4*j+6} {4*j+5})" for j in range(3)]
    inl = [xin[1]] if gate_mid else xin
    outl = [xout[1]] if vent_mid else xout
    wal = ["(0 3 2 1)", "(12 13 14 15)"] + ([xin[0], xin[2]] if gate_mid else []) + ([xout[0], xout[2]] if vent_mid else [])
    s0 = " ".join(f"({4*j} {4*j+1} {4*j+5} {4*j+4})" for j in range(3))
    s1 = " ".join(f"({4*j+3} {4*j+7} {4*j+6} {4*j+2})" for j in range(3))
    return hdr + f"""vertices ( {vs} );
blocks ( {blocks} );
boundary
(
    inlet  {{ type patch; faces ( {' '.join(inl)} ); }}
    outlet {{ type patch; faces ( {' '.join(outl)} ); }}
    walls  {{ type wall;  faces ( {' '.join(wal)} ); }}
    sym0   {{ type symmetryPlane; faces ( {s0} ); }}
    sym1   {{ type symmetryPlane; faces ( {s1} ); }}
);
mergePatchPairs ();
"""
cfg = {"E_b": (320, 32, 1, 0), "E_c": (320, 32, 0, 1), "E_d": (320, 32, 1, 1), "M1": (833, 32, 0, 0), "M2": (320, 16, 0, 0)}
name = sys.argv[1]; print(bm(*cfg[name]), end="")
