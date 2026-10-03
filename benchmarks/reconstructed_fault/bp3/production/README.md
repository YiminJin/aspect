# Fresh BP3 production candidate

`bp3_fresh.prm` is a **fresh, unqualified candidate**, not a continuation of a
historical checkpoint. Its only model data file is `fault.txt`. Build the single
maintained source package in `../plugin/`; no geometry/profile/mesh preparation
script or historical fixture directory is a runtime dependency.

**Do not launch the production mesh yet.** Section 3 demonstrated that the
preserved graded mesh conflicts with core automatic-completion boundary-lattice
admission. This packaging pass does not change that numerical restriction.
Parameter validation is not resolved-mesh or first-event qualification.

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
| Mesh levels | initial-global 9, minimum 1, adaptive 0 | Same intended h_fine=3.90625 m / h_coarse=1000 m; generated before particle initialization |
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

## Build and parse only

From the repository root, with the matching compiler/MPI/deal.II environment:

```sh
cmake -S benchmarks/reconstructed_fault/bp3/plugin \
  -B benchmarks/reconstructed_fault/bp3/build-maintained \
  -DAspect_DIR="$PWD/build-refactor-r6b" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/bp3/build-maintained -j 3
export ASPECT_SOURCE_DIR="$PWD"
mpirun -np 1 build-refactor-r6b/aspect-filter-derivative-qualified --validate \
  benchmarks/reconstructed_fault/bp3/production/bp3_fresh.prm
```

For a future separately qualified server run, adjust the library, fault and
output paths and rebuild against the exact server executable stack. No server
job or full production inventory was run here. The historical 542,958-cell
mesh would start with 8,687,328 particles at 4×4; that is an estimate based on
the historical count, not a measured inventory of this generated candidate.
