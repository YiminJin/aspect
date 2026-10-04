# Fresh BP3 production candidate

`bp3_fresh.prm` is a **fresh, unqualified candidate**, not a continuation of a
historical checkpoint. Its only model data file is `fault.txt`. Build the single
maintained source package in `../plugin/`; no geometry/profile/mesh preparation
script or historical fixture directory is a runtime dependency.

The maintained mesh now retains the fine endpoint buffer used by both successful
local meshes. A's exterior slope alone is insufficient. The buffer and actual
150×50 km completion footprint are checked in the
[birth/completion report](../../bp3_birth_completion/README.md); core admission is
unchanged. The separate historical unbuffered 45° failure is not a blocker for
this 60° configuration. A full production mechanics trajectory/server run remains
unqualified, and the resource-limited startup attempts are recorded explicitly.

## Settings provenance

The latest server input was not identified. These physical settings and caps
are inherited from the tracked `bp3_150x50_filter20.prm` plus
`bp3_150x50_first_event.prm`; the corresponding files in `../aspect` were inspected
read-only. They must not be described as a verified match to the user's last
server job. The old `../aspect/doc/reconstructed_fault/bp3/bp3.prm` uses the
obsolete PhaseFieldRSF model and is not the basis of this candidate.

| Quantity | Candidate | Provenance / change |
|---|---|---|
| Box / fault | 150×50 km, original native fault anchors, peak 0.6 | Historical model; only the file path changes |
| Mature phase field / length / Gc | frozen / 20 m / 1e5 | Historical production values, not ell=4000 coarse-test values |
| G / cohesion / viscosity | 32038120320 Pa / 1e6 Pa / 1e26 Pa s | Historical production values |
| a / b / Dc / f0 | 0.010,0.025 / 0.015 / 0.008 m / 0.6 | Original material contrast retained |
| Damping / Vref / Vmin | 4624440 / 1e-6 / 1e-20 | Historical values |
| Vp / Vinit / nominal normal stress | 1e-9 m/s / 1e-9 m/s / 50 MPa | Existing BP3 loading/initialization convention |
| Normal feedback / filter / bottom | true mechanical / Helmholtz 20 m / full velocity | Preserved |
| Initial / maximum dt | 100 s / 4e6 s | Inherited; newer server caps unconfirmed |
| Maximum relative dt increase | 91.0 | Native resolved default, reported explicitly; latest server choice unknown |
| Log-state bound | **0.1, provisional** | Current unweighted core predictor; explicitly chosen from the existing documented example, not verified server input. Historical input omitted it (disabled default); **0.2 is not silently substituted** |
| Nonlinear failure | **cut timestep size; factor 0.5, provisional** | Native tested retry path; changes historical `abort program`. Additional `repeat on cutback` controller is not selected |
| Nonlinear / linear tolerance | 1e-8 / 1e-9 | Preserved; no convergence relaxation |
| Initial particle policy | regular 4×4; 12–24 | User-selected replacement of old 3×3/no-replenishment setup |
| Interpolation | native LLS via BP3 history adapter; continuous Q2 | H and all Maxwell components limited, boundary extrapolation off |
| Mesh levels | initial-global 9, minimum 1, adaptive 0 | h_fine=3.90625 m / h_coarse=1000 m; generated before particles; protected endpoint strip included |
| Profiles / heavy schedule | 0.1 m or 31557600 s, plus existing forced outputs | Inherited first-event schedule |
| Field/particle visualization / detailed CSVs | off / off | Explicit opt-in; fault profiles and full checkpoints remain |
| Checkpoint / graceful wall stop | 1800 wall seconds / 82800 seconds | Inherited |

The declared mask `true,false,false,true,true,true` documents this property
layout. The adapter derives and validates the effective H/Maxwell mask by field
names; it does not trust positional text. Startup prints resolved material,
mesh, particle, solver, state-bound and cutback settings and writes
`bp3_resolved_settings.json`. ASPECT also writes its complete parameter record.
The exact explicit differences are in
[production_changes.json](../../bp3_packaging/results/production_changes.json),
and native default-expanded candidate values in
[production_resolved.json](../../bp3_packaging/results/production_resolved.json).

## Build and validate

From the repository root, with the matching compiler/MPI/deal.II environment:

```sh
cmake -S benchmarks/reconstructed_fault/bp3/plugin \
  -B benchmarks/reconstructed_fault/bp3/build-maintained-buffered \
  -DAspect_DIR="$PWD/build-refactor-r6b" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/bp3/build-maintained-buffered -j 3
export ASPECT_SOURCE_DIR="$PWD"
mpirun -np 1 build-refactor-r6b/aspect-birth-identity-qualified --validate \
  benchmarks/reconstructed_fault/bp3/production/bp3_fresh.prm
```

For a future separately qualified server run, adjust the library, fault and
output paths and rebuild against the exact server executable stack. The new
particle-manager notification changes the C++ interface, so rebuild plugins
against the corrected executable; checkpoint payload versions remain unchanged.
The old qualified executable and `build-maintained` library are preserved.

The measured buffered mesh has **552,084 cells**, finest/coarsest edges
**3.90625/1000 m**, corresponding to **8,833,344 initial particles** at 4×4. Its production
completion footprint has half-width **1622.994616 m** at each endpoint, with
**832 aligned fine boundary faces** covering each footprint. Native smoothing
remains active. The model identity includes the full buffer policy; old unbuffered
checkpoints must use their original plugin rather than silently changing meshes.
