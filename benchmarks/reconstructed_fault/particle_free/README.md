# Particle-free history and fixed-geometry experiments

The current entry points are [STATUS.md](../../../doc/reconstructed_fault/particle_free/STATUS.md),
[BP3 server startup review](../../../doc/reconstructed_fault/particle_free/BP3_DG_STARTUP_REVIEW.md)
and the [manual server build/startup procedure](bp3_dg_first_event/README.md).
Local qualification, prepared server startup and the unqualified first-event
experiment remain distinct. The earlier
[domain-free report](../../../doc/reconstructed_fault/particle_free/PF6_DOMAIN_FREE_REPORT.md)
records the homogeneous configuration and retained legacy CPDI limitations.
Unlimited upwind DGQ2 is the experimental baseline. Production defaults are
unchanged; both WENO paths remain experiments. The earlier PF5 subset is
serial, fixed mesh/geometry/phase/particles, closed-boundary prescribed transport
with uniform surface V/Z. The later bounded uniform-Cartesian fixed-surface
transfer fixture checks serial/1/2/4-rank ownership and restart. General local AMR,
moving geometry, birth and coupled propagation require separate work.

The later [Gauss12/local report](../../../doc/reconstructed_fault/particle_free/PF6_LOCAL_REPORT.md)
adds one frozen local cell, hanging-node evaluation and restore-only MPI checks.
It retains the original Gauss6 and genuine deep particle-support failures;
coupled hanging-node phase evolution and automatic AMR remain guarded.

The selected [homogeneous mesh-data configuration](../../../doc/reconstructed_fault/particle_free/PF6_DOMAIN_FREE_REPORT.md)
removes domain-dependent setup/material projection and restores 64→256→64 at
1/2/4 ranks with 144 fixed diagnostic points. Domain-backed CPDI remains separate,
including its original deep coverage failure. The new selection is default off;
normalization/derivatives and the independent publication guard remain unchanged.
Use `run_pf6_domain_free.py` to prepare/finalize a case, or the bounded sequential
`run_pf6_domain_free_cases.py serial|negative|mpi` driver through the existing
`refactoring_boundary/run_logged.py` wrapper. Labels must be new. The serial gate
must pass before MPI. Each native case has a 180 s cap and one thread/rank.
[Portable evidence and replay](evidence/pf6_domain_free/README.md) launch no native
job during independent reassessment.

This directory's CMake, C++ observers, parameter files and independent Python
checks are retained for future development. Reports preserve measured outcomes
and limitations. Large evidence archives, plots, detailed source/run manifests,
binaries, checkpoints and raw outputs are retained locally and are not part of
the source commit. Historical report links to those local artifacts require the
original evidence package; a fresh checkout can rebuild and rerun the fixtures.
See [evidence/README.md](evidence/README.md) for the committed summary subset.

| Check | Native observer / inputs | Independent assessment |
|---|---|---|
| PF1 identities and counterexamples | None | `reference_checks.py` |
| PF2 native FE/DG source, mechanics and publication | `fixed_profile_checks.cc`, `fixed_profile_{fe,dg}.prm` | `assess_fixed_profile.py` |
| PF3 noncommitting phase functional | `phase_energy_checks.cc` | `assess_phase_energy.py` |
| PF4 prescribed material transport | `transport_checks.cc`, `transport_{4,8,16,32}.prm` | `assess_transport.py` |
| Experimental compact WENO / scaling | `limiter_checks.cc`, `stabilization_checks.cc` | `assess_limiter.py`, `assess_stabilization.py` |
| Pre-PF7 candidate-construction audit | `stabilization_checks.cc`, `run_candidate_case.py` | `assess_candidates.py`, `plot_candidates.py` |
| Native scalar EV/SUPG comparison | `scalar_method_checks.cc`, `native_method_checks.cc`, `methods-*.prm` | `assess_methods.py` |
| PF5 accepted history and filesystem restart | `pf5_checks.cc`, `pf5-*.prm` | `assess_pf5.py` |
| PF6a publication diagnosis and fixed-geometry confirmation | `pf6a_checks.cc`, `run_pf6a_reference.py` | `replay_publication.py`, `assess_publication_confirmation.py` |
| Fixed-reference transfers, trace ownership, MPI lifecycle/restart | `pf6_transfer_checks.cc`, `run_pf6_transfer.py`, `run_pf6_transfer_restarts.py` | `assess_pf6_transfer.py`, `replay_pf6_transfer.py` |
| Homogeneous mesh-data initialization/materials and restored deep remap | `pf6_transfer_checks.cc`, `run_pf6_domain_free.py` | `assess_pf6_domain_free.py`, `replay_pf6_domain_free.py` |

Build against this checkout's configured ASPECT build, from the repository root:

```sh
cmake --build build-particle-free/native -j4
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/plugin -DAspect_DIR="$PWD/build-particle-free/native" -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/plugin --target pf5_checks fixed_profile_checks transport_checks -j2
```

Use the compiler/dependency toolchain of that ASPECT build. Optional observer
targets are listed in `CMakeLists.txt`. `run_fixed_profile.py` pins the executable,
observer and expanded input into a new run directory, bounds runtime, and refuses
to overwrite a label. For example:

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf5-new-16 --input benchmarks/reconstructed_fault/particle_free/pf5-16.prm --plugin build-particle-free/plugin/libpf5_checks.release.so --postprocessor 'PF5 accepted history' --seconds 180
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py pf5-new-restart-16 --input benchmarks/reconstructed_fault/particle_free/pf5-restart-16.prm --plugin build-particle-free/plugin/libpf5_checks.release.so --postprocessor 'PF5 accepted history' --restart-from build-particle-free/runs/pf5-new-16/run/restart --seconds 180
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_pf5.py build-particle-free/runs/pf5-new-16 --restart-parent build-particle-free/runs/pf5-new-16 --restart build-particle-free/runs/pf5-new-restart-16 --output build-particle-free/pf5-new-assessment.json
```

The reports specify multi-step convergence studies, negative controls and the
remaining transfer task. These examples do not authorize additional runs or
coupled deployment. Python references require NumPy/SciPy; native tests use the
existing ASPECT/deal.II infrastructure.

The pre-PF7 comparison adds a diagnostic `Candidate construction = constrained
least squares` observer option; its default remains `mean shift`. Build
`stabilization_checks`, then use a unique label, for example:

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_candidate_case.py NEW-LABEL --construction cls --detector fu_shu --case signed_jump --divisor 1
```

`--local --mesh 4` runs the independent-candidate export fixture. The report and
[frozen formulation](../../../doc/reconstructed_fault/particle_free/WENO_CLS_FORMULATION.md)
specify the bounded 30-job matrix and exact equations. The existing detector,
thresholds, smoothness, epsilon, transport and RK schedule are held fixed; this
is a controlled Q2 adaptation, not the full 2016 method. Native sharp-pulse
improvement is measured, but temporal and transverse artifacts prevent adoption.
See [portable evidence/replay](evidence/weno_cls/README.md). No PF7 is selected.

## Retained PF0/PF1 instructions

PF2 is complete; see [PF2_REPORT.md](../../../doc/reconstructed_fault/particle_free/PF2_REPORT.md).
`fixed_profile_{fe,dg}.prm` and the observer drive three native mechanical steps
at frozen accepted localization. `run_fixed_profile.py` bounds runs and preserves
artifacts; `assess_fixed_profile.py` checks the fresh baseline and same-build
FE/DG fields, actual loads, independent source/work references and lifecycle.
The report gives exact build/run commands, gates, failed initialization evidence
and remaining particle dependencies. Use new labels; preserve existing outputs.

The paragraphs below document the retained PF0/PF1 mathematical references.

These scripts audit a retained baseline and verify mathematical identities.
They change no production source and do not implement DG transport or a new
phase functional. Reports and the PF2 proposal are under
[`doc/reconstructed_fault/particle_free/`](../../../doc/reconstructed_fault/particle_free/STATUS.md).

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/reference_checks.py --output build-particle-free/pf1-reproduction.json
python3 benchmarks/reconstructed_fault/particle_free/run_baseline.py --retained-build /home/ein/repository/aspect-fault-propagation/build-bp3-native --output build-particle-free/pf0-reproduction --seconds 180
```

Choose unused output names: existing results are never overwritten. The native
runner copies the historical executable and two matching libraries into its
output directory and expands the existing `history-uniform-q6.prm` paths into
this worktree. It does not configure/build or write in the source checkout.
Only library/output paths change. One serial process and one computational
thread are used. The local budget is 180 s for the native case and 10 s for
the mathematical checks (observed <1 s); there is no build campaign.

The native fixture deliberately performs synthetic history commits, rejected
candidate/duplicate-commit checks and restoration inside its observer. There
is no coupled solve. Its huge artificial timesteps test general Maxwell beta,
not BP3 temporal accuracy. A binary's embedded Git label need not identify its
dirty compilation inputs; see the source audit before reusing this evidence.

`reference_checks.py` depends only on NumPy and SciPy. V1 uses 70-digit Decimal
exponentials; V2 uses analytic/complex-step derivatives and a separate central
difference sweep; V3 uses independent characteristic and commutator-forcing
integrals; V4 has a closed Gaussian-overlap error and midpoint refinement.
Each numerical assertion and expected nonzero defect is saved in one JSON.
Failure returns nonzero. Norms/tolerances are in the script and PF1 report.

`evidence/` contains compact original results and exact artifact identities.
Raw native output, expanded parameters, binaries and the earlier successful
89-check development run remain ignored under `build-particle-free/`.

## PF6a: native cohesive endpoint phase/mechanics

The selected serial fixed-grid/fixed-reference-surface experiment uses the real
consistent-endpoint/reduced-Newton outer loop, independent unlimited DGQ2 history
and ordinary FE fracture, without bulk/surface advection. Production defaults
and limiter settings stay unchanged. See
[PF6A_REPORT.md](../../../doc/reconstructed_fault/particle_free/PF6A_REPORT.md)
and [portable evidence/replay](evidence/pf6a/README.md). Operators and serial
acceptance/restart pass; large normal spatial stress sensitivity remains. This
does not complete PF5 transfers or qualify propagation.

Use unique labels; the runner refuses to overwrite inputs/runs. It explicitly
sets the existing fixed-particle-advection diagnostic in each native process.

```sh
cmake --build build-particle-free/native -j4
cmake --build build-particle-free/plugin --target pf6a_checks fixed_profile_checks transport_checks pf5_checks -j2
python3 benchmarks/reconstructed_fault/particle_free/run_pf6a.py NEW-PF6A --diagnostics
python3 benchmarks/reconstructed_fault/particle_free/run_pf6a.py NEW-PF6A-DT2 --divisor 2
python3 benchmarks/reconstructed_fault/particle_free/run_pf6a.py NEW-PF6A-PARENT --divisor 2 --end-time .53125 --checkpoint
python3 benchmarks/reconstructed_fault/particle_free/run_pf6a.py NEW-PF6A-RESUME --divisor 2 --restart-from build-particle-free/runs/NEW-PF6A-PARENT/run/restart
python3 benchmarks/reconstructed_fault/particle_free/run_pf6a.py NEW-CUTBACK --schedule cutback
python3 benchmarks/reconstructed_fault/particle_free/run_pf6a.py NEW-MATCHED --schedule matched
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_pf6a.py build-particle-free/runs/NEW-PF6A --work --output build-particle-free/new-pf6a-assessment.json
```

`--normal-cells 64 --divisor 4` is the bounded spatial comparison;
`--quadrature 12 --divisor 4` isolates quadrature. `assess_pf6a.py --matrix`
expects the five reported configurations and separates common-point temporal,
spatial, quadrature and initial seed-representation differences. Its
`--contact-only` checks the consistent-Q1 free fault residual and sticking
complementarity rather than treating admissible sticking traction as a failed
equality. `verify_pf6a.py` compares all exported lifecycle state and writes raw
manager archive pairs for the native load/resave check; its default labels are
the recorded campaign labels. `plot_pf6a.py` exports the measured figure.


## Bounded PF6a spatial diagnostic

See [PF6A_SPATIAL_REPORT.md](../../../doc/reconstructed_fault/particle_free/PF6A_SPATIAL_REPORT.md)
and [portable evidence/replay](evidence/pf6a_spatial/README.md). The separate
`pf6a_spatial_checks` target uses the unchanged native PF6a core. It distinguishes
common-field evaluation from frozen native mechanical solves; it separately
compares compatible DG/accepted-FE histories and actual CPDI fracture.
`prepare_pf6a_spatial.py` recovers the compatible FE history from the retained
72-step source case and writes complete common fields. Example, from the root:

```sh
python3 benchmarks/reconstructed_fault/particle_free/prepare_pf6a_spatial.py \
  build-particle-free/runs/pf6a-final-32-4-q12 \
  build-particle-free/pf6a-spatial/common
cmake --build build-particle-free/plugin --target pf6a_spatial_checks -j2
python3 benchmarks/reconstructed_fault/particle_free/run_pf6a_spatial.py \
  UNIQUE-LABEL --normal-cells 32 --backend DG
```

The runner refuses existing labels, pins the observer/core/input, and caps each
serial run at 180 s. The selected campaign uses 32/64, then one finer 128 normal
level. `--event` imports the common physical entry at .53125 s and observes four
actual coupled steps through .5625 s; `--divisor 256` selects the bounded
half-step control. It is not a production restart or general transfer path.
`assess_pf6a_spatial.py`, `assess_pf6a_spatial_event.py` and
`plot_pf6a_spatial.py` provide independent checks and standalone figures.
The evidence replay runs these checks without launching new native jobs.
Unlimited DGQ2, all constitutive equations and production defaults remain fixed.


## Matched evolving spatial reference

[PF6A_REFERENCE_REPORT.md](../../../doc/reconstructed_fault/particle_free/PF6A_REFERENCE_REPORT.md)
records the measured 1D selection and failed single native confirmation.
`pf6a_spatial_reference.py` implements the independently checked coupled 1D
endpoint with separate mechanical/phase resolution; it reuses the retained
continuum initialization and analytic normal integration. It does not use the
historical split driver. `assess_pf6a_reference.py` separates mechanical,
phase/obstacle, seed-representation, temporal and quadrature effects through
onset and growth. `plot_pf6a_reference.py` creates standalone figures.

`run_pf6a_reference.py` is the one bounded native selection runner, retaining
all defaults/guards and the original PF6a observer. The 512 test fails its
existing publication consistency guard at step 1; `verify_pf6a_reference_native.py`
verifies startup and classifies that failure without claiming native evolution.
[Evidence/replay](evidence/pf6a_reference/README.md) preserves failed preliminary
references, the literal-table initialization diagnostic, full main 1D states,
pre-native targets/decision and native failure. No further native job is selected.

## Fixed-reference transfer qualification

[Report](../../../doc/reconstructed_fault/particle_free/PF6_TRANSFER_REPORT.md),
[declared rules](../../../doc/reconstructed_fault/particle_free/PF6_TRANSFER_FORMULATION.md)
and [portable replay](evidence/pf6_transfer/README.md) cover complete DGQ2
refinement/coarsening, admissible Q1 phase/obstacle transfer with fixed surface,
matching-cell traces, 1/2/4-rank rollback/publication and native restart. The
original strict-reference failure and a newly measured native-quadrature screen
failure remain explicit. General local AMR is not qualified.

Build `pf6_transfer_checks` against the matching core and use a fresh label:

```sh
cmake --build build-particle-free/plugin --target pf6_transfer_checks -j2
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_transfer.py prepare NEW-TRANSFER --ranks 1
```

The runner pins the binary/observer and writes flattened inputs, source hashes,
`source.patch` and `command.json`. Run its printed argv through
`refactoring_boundary/run_logged.py NEW-TRANSFER 190 timeout --signal=TERM --kill-after=5 180 ...`,
then finalize with the same label and assess:

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_transfer.py finalize NEW-TRANSFER
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_pf6_transfer.py build-particle-free/runs/NEW-TRANSFER --output build-particle-free/NEW-TRANSFER.json
```

Only this observer DSO is loaded in the transfer fixture: it reuses the test
friend `Postprocess::MomentCycle` and must not share a process with another test
DSO defining that class. Rebuild every observer used after core-header changes;
old pinned run binaries and their matching DSOs remain intact. The restart
driver takes `--help`, requires completed full-rank baselines, and runs bounded
half/checkpoint and restart cases sequentially; it does not tune any parameters.
`replay_pf6_transfer.py` verifies checksums and reassesses saved fields with NumPy
without launching ASPECT. Inputs explicitly select RK2 and set the existing
freeze environment variable; particle advection dt is zero, constitutive dt is
unchanged. Do not infer that particle infrastructure is inactive.

## Gauss12 and small local-transfer followup

[Report](../../../doc/reconstructed_fault/particle_free/PF6_LOCAL_REPORT.md),
[formulation/amendments](../../../doc/reconstructed_fault/particle_free/PF6_LOCAL_FORMULATION.md)
and [portable replay](evidence/pf6_local/README.md) define the measured scope.
`run_pf6_local.py` wraps the existing preparation/finalization runner, pins one
core/observer pair per case, and explicitly selects physical quadrature and the
optional local checks. `run_pf6_local_cases.py` orchestrates capped sequential
cases; use a fresh prefix and the existing run logger for MPI host sockets.

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_local_cases.py --help
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_local.py prepare NEW-LOCAL --local --ranks 1 --quadrature 12
```

Run the printed argv through `refactoring_boundary/run_logged.py` with the
documented 180 s native timeout, then finalize and independently assess:

```sh
python3 benchmarks/reconstructed_fault/particle_free/run_pf6_local.py finalize NEW-LOCAL
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/assess_pf6_local.py build-particle-free/runs/NEW-LOCAL --local --output build-particle-free/NEW-LOCAL.json
```

The selected patch is CellId 1_2:30 (64→67 cells), with fixed surface/particles.
The diagnostic permits an isolated cell after checking default-smoothing no-ops;
physical steps retain default smoothing. The runner omits the identity-only
256-cell roundtrip explicitly; `--deep-uniform` retains its genuine one-layer
particle-support negative control. The original transfer observer's default is
unchanged. The shared Voro closed-patch adapter has an independent small-library
counterexample in `voro_closed_patch.cc`; its actual compile/link command and
result are recorded in the evidence. Do not remove either the volume/coverage
guard or the independent Maxwell/DG consistency comparison.

`--snapshot --quadrature 6 --restart PATH` reevaluates a retained completed
checkpoint with native Gauss6/12/24/32; restore-only termination must start no
physical step. Same-count local and serial-to-four checkpoint restores use the
native lifecycle. Cross-rank RNG replay remains unavailable with population
changes disabled, even though deterministic physical histories compare exactly.
All observer use remains exclusive: do not load another `MomentCycle` test DSO
in the same process. Production defaults and limiters remain unchanged.


The bounded mature mechanical/profile-resolution matrix is complete. It adds
exact coarse-Q1 phase/exterior-lattice prolongation onto finer mechanics and
matched stationary particle controls, preserving Gauss6 and unlimited DGQ2.
See [the report](../../../doc/reconstructed_fault/particle_free/PF6_MATURE_RESOLUTION_REPORT.md)
and [portable evidence/replay](evidence/pf6_mature_resolution/README.md).
Native perturbation weak loads are distinguished from changing bulk-work mass;
backend comparisons are differences, not accuracy errors against either backend.

### Mature stationary DG acceptance and server-pilot preparation

The bounded mature lifecycle and mechanical/profile controls are recorded in
[PF6_MATURE_PILOT_REPORT.md](../../../doc/reconstructed_fault/particle_free/PF6_MATURE_PILOT_REPORT.md).
Unlimited domain-free DGQ2 remains experimental; retained particle controls are
neutral comparison evidence. `run_pf6_mature.py` supports the fixed FF reference,
1/2/4 ranks, same-mode checkpoints and default-off actual outer/nonlinear failure
fixtures. `sample_pf6_mature_dg.py` independently evaluates every Q2 coefficient;
`verify_pf6_mature_fixed_ff.py`, `verify_pf6_mature_surface.py` and
`reduce_pf6_mature_pilot.py` reproduce the mathematical/physical-state reductions.
`package_pf6_mature_pilot.py` retains source, pinned artifacts and selected raw
fields; `replay_pf6_mature_pilot.py` verifies and independently reduces the
[portable evidence](evidence/pf6_mature_pilot/README.md) without ASPECT/MPI.
The separate [bounded server package](bp3_dg_pilot/README.md) prepares inputs and
an explicit supervised launcher; preparation does not launch the full geometry.
Original failures, physical laws, limiter/filter settings and production defaults
remain. Phase evolution, transport, AMR and the full event are subsequent tasks.
