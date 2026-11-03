# GUIDE section 17, stages 1 and 2 on fifa (openInjMoldSim)

## Stage 1: can it be installed? Yes, without root

| item | result |
|---|---|
| solver | [krebeljk/openInjMoldSim](https://github.com/krebeljk/openInjMoldSim), GPL-3.0, v7.2, last commit 2021-01-29 (`6ebefb1`); the gitee repository is a mirror |
| OpenFOAM | 7 (openfoam.org), required by the README |
| OpenFOAM on fifa | official image `docker.io/openfoam/openfoam7-paraview56` (2.56 GB) with rootless podman, stored under the home directory (Docker would store it on `/`, which has 11 GB free) |
| build | `Allwmake -j 8` in the container: `openInjMoldSim` and `openInjMoldSimF` (fibre orientation, Folgar-Tucker), 0 errors, 2 min 18 s |
| fibre tutorial | `tutorials/demo/dogbone_fiber` |

Workarounds needed (none changes the solver or the case):
- The image has files owned by UID 98765, outside the user's 65 536 sub-UIDs. Pulled with `ignore_chown_errors`.
- OpenFOAM's dynamicCode (`#calcEntry` in `constant/solidificationProperties`) refuses to run as root, so the case
  must run as the user. A derived image `localhost/of7-keepid` changes the owner of the UID-98765 files to 0;
  then `podman run --userns=keep-id --user <uid>` runs as the user with the shared home mounted read-write.
- Compute nodes (frontale03): podman cannot keep its store on the NFS home (overlay xattr, pivot_root) and the
  nodes have no `/etc/subuid`. It works with the store on the node's local disk (`/tmp/pm<uid>`, 150 GB free),
  `overlay.ignore_chown_errors=true`, `--network=host`, and the image loaded from a tar on the shared home.

## Stage 2: the tutorial runs; filling alone would take about 3 days on 7 cores

`dogbone_fiber` as shipped (`AllMesh`, `AllRun`, 7 MPI ranks), on the fifa login node at nice 10
(nobody else logged in):
- `AllMesh`: snappyHexMesh, 232 123 cells, about 2 min, exit 0.
- `openInjMoldSimF -parallel -fillEnd 0.98`: ran without error. Stopped by hand (the user's choice) at simulated time
  0.0104 s after 3425 s of wall time, with deltaT about 4.2e-5 s.
- At that rate the filling stage alone (to 0.45 s) would take about 74 h on 7 cores; packing and holding (to 15.05 s)
  come on top. So the tutorial is not run to the end; stage 3 needs its own time estimate (cells, filling time).

Files: `fifa_dogbone_fiber_snappyHexMesh.txt` (mesh log), `fifa_dogbone_fiber_solver_tail.txt` (last 60 lines of the
solver log). Scripts and the case are on fifa only (`~/flow_sim/`).
