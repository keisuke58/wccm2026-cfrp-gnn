# GUIDE section 21 on fifa: insulated walls at the gate, step 1 (21 Nov 2026) -- did not pass, stopped

## Set-up (one change from the section 20 2-D strip, `../stage4_strip/`)
- The top and bottom wall faces within 2 mm of the gate (face centres x < 2 mm: 21 cell columns, 2.02 mm) moved to a
  patch `walls_gate` (`case/topoSetDict`, `case/createPatchDict`); `T` there `zeroGradient` (insulated), every other
  field as on `walls` (`case/T_walls_gate.txt`). Range decided before the run, as section 21 asks. It stands for the
  hot land of a film gate. Material, Ci, flow rate and mesh (833 x 1 x 32, one cell across, sides empty) as section 20.
- Restart at T = 0.70 s from `strip_2D_fill/0.7`. `mapFields -consistent` refuses the extra patch; createPatch keeps
  the cells, so the internal fields were copied as they are (patch values start uniform and are recomputed).
- 8 ranks, memory limit (`-l mem=16gb` and a guard), 30 min timeout (`case/s5_trial.pbs`, `case/Ars5`).

## Result of step 1: diverges sooner, at T = 0.7177 s
145 steps; max(U) steady at 0.147 m/s up to 0.7174 s, then 0.153 and 131 m/s in the next two steps, alpha.poly to
-170, floating-point exception (exit 136) after about 1.5 min (`case/maxU_per_step.txt`, `case/trial_log_tail.txt`).
The first divergence of the 2-D strip was at 0.767 s, so the insulated gate walls do not get past it: section 21
stops here (steps 2-4 not run, no stage-5 CSV).

Where: the pressure maximum is at the top corner of the gate (x = 0.05 mm, 0.05-0.15 mm from the top wall) and
rises from 10.3 to 11.2 MPa over the last 0.0013 s before the blow-up, as in section 20.
A possible reason, not tested: at 0.70 s the gate cells next to the walls had already frozen (one to two cells at the
viscosity cap, `../stage4_strip/FIFA_STAGE4.md`); switching the wall to insulated does not melt them, and the sudden
change of the wall condition at the restart may add to the instability. Starting the 2-D strip with the insulated gate
walls from T = 0 would test the set-up itself (about 7 h on 8 ranks by the section 20 rate); not started.
