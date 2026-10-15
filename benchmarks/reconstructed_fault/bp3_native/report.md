# Stationary mature BP3 FE history

Scope: sections 1–6 of [plan.md](plan.md), including focused verification of the
implementation. The already prepared history fixture supplies these checks.
The section-7 coupled campaign and sections 8–9 are deferred.

## Derivation before implementation

The native convention is the embedded 3D plane-strain deviator
`D3(symgrad(u))`; its in-plane trace subtraction is `tr(symgrad(u))/3`.
The existing legacy particle recurrence uses the full symmetric gradient.
These agree on divergence-free fields. With nonzero divergence their difference
is the isotropic in-plane increment `-2 eta_ve div(u) I2/3` (and the missing
3D zz component). This distinction is preserved and measured separately.

Let `S = shear_sense sym(tangent ⊗ normal)`, with the normal obtained by rotating
the actual segment tangent counterclockwise. The stress is a perturbation:

`tau = 2 G D3(symgrad(d_R)) - 2 chi Z S`, `chi = h(phi)/I_h`.

For immutable geometry/profile/normalization and homogeneous matching bulk and
surface coefficients, linearity gives

`d_R_new = beta d_R_old + (eta_ve/G) u_new`,
`Z_new = beta Z_old + eta_ve V_new`,
`tau_new = beta tau_old + 2 eta_ve [D3(symgrad(u_new)) - chi V_new S]`.

Here `beta = exp(-dt G/eta)` and `eta_ve = -eta expm1(-dt G/eta)` are the existing
exact Maxwell coefficients. Q1 interpolation commutes with the Z recurrence
because these coefficients are homogeneous. Endpoint continuation is constant
for both Z and V, using the same bulk-source association as mechanics.
The mature cohesive traction is identically zero; it contains no Z information.
The cohesive P2 identity `Z = I_h tau_coh` remains independent and unchanged.
Background normal/shear tractions stay on the surface and never enter d_R or Z.
Fresh d_R and Z are zero. Every trial uses the incoming accepted state; staged
candidate samples are compared with direct Maxwell evaluation before publication.

## Baseline and provenance

Current worktree `fault-propagation` at `c33edaaeb145e5d3dce155bb815552ef80efc5f9`;
read-only reference `pf-rsf-refactor` at `8f126905c559df106f08af4079f678401fec4d06`.
The cleanup recovery correction is included in the current baseline.
Maintained plugin and fresh template already exist under
`benchmarks/reconstructed_fault/bp3/{plugin,production}`. Neither selected
worktree has `benchmarks/bp3`; the reference's maintained directory also contains
historical outputs/checkpoints. The historical first-event PRM is a restart input.
No branch merge, historical-data copy or old plugin binary is used.

Recovery found 15 modified tracked files, no staged changes, and 2346 untracked
files. The reference has no tracked changes and 63 untracked entries. Its local
revised inputs/packages remain read-only. Recent current commits already contain
the P2 split, normalization/cache cleanup and missing-export correction; none
needed reapplication. `baseline.json` retains reference input hashes. The
incoming patch, inventory and per-file hashes are in
`build-bp3-native/recovery-20261013/`. No reset, branch switch, merge, commit,
push, old-checkpoint continuation or other-worktree build was performed.

## Geometry, normalization and mechanics

Mature stress evaluation uses `project_to_bulk_source` at the actual physical
point, then the associated segment's tangent, rotated normal, shear sense and
Q1 coordinate. It uses the same map as Stokes assembly and history subtraction.
Constant endpoint values extend into the existing paired boundary-source wedges;
finite particle-to-surface admission is unchanged. The translated fixture has
Box origin (-1234.5,137.25) m, extent 2000×1000 m, a 1154.700538 m fault dipping
60°, and shear sense -1. Both top/bottom contacts pass native completion admission.

The mature branch retains BP3's general cell-interval integration and consistent
tridiagonal surface mass solve. Independent 16/32-point integration evaluates
the actual bilinear Q1 restriction and the exterior completion lattice, split
at every cell/lattice crossing. No rotated linear-ray formula, normalization
floor, lumping, profile change or dense fault-wide inverse is introduced.
Accepted nodal I_h is copied into FE state and required to remain unchanged.

Mature mechanics uses one tensor-product Gauss rule per cell, including hanging
constraints on the fixed graded mesh. Existing Stokes residual/tangent, slip
coupling, surface bulk-work assembly and work diagnostics all dispatch through
`fault_mechanical_quadrature`. The P2 horizontal panel rule and dense phase block
are excluded from mature mode; endpoint phase solving stays disabled. The
q6/q8 comparison below measures two manufactured work forms, not complete
coupled operator or physical spatial convergence.

## History lifecycle

The authoritative state is accepted FE d_R plus independent nodal Z. Particle
Maxwell values are derived samples and are never projected back into d_R.
Initialization checks zero particle perturbation stress. Trial construction
does not change accepted histories. The complete candidate, surface history,
RSF state, and derived samples are validated before terminal publication;
local FE errors are propagated collectively before any writes. The existing
solver rolls back unsuccessful prepublication mechanical trials and commits
the accepted history once. Post-publication observations read FE stress without
applying a second Maxwell increment.

The named `BP3 stationary samples` integrator stores no integration stages and
does not move particles. Inputs also select `Load balancing strategy = none`.
Runtime guards retain particle count/ID/position, mesh vertex/DoF/cell, geometry,
orientation, phase and normalization checks. Supported scope is fresh 2D,
straight fixed geometry, fixed Cartesian Box meshes including startup grading
and hanging nodes, and homogeneous matching G/eta without thermal variation or
viscosity clipping. Moving/born/deleted particles, mesh adaptation/motion,
geometry/profile changes and FE restart/conversion are rejected. The legacy
particle defaults and restart path are retained. Fixed Eulerian samples do not
establish equivalence to advected particle history.

## Changed files and recovery delta

| Files | Responsibility |
|---|---|
| `source/material_model/phase_field_fault/fe_strain.cc`, `include/aspect/material_model/phase_field_fault/fe_strain_internal.h` | Mature Z recurrence, native geometry/localization, accepted-state restrictions and samples; shared tensor-product mature quadrature. |
| `source/material_model/phase_field_fault/history.cc` | Zero mature cohesive candidates, complete FE candidate validation and terminal derived-sample publication. |
| `source/material_model/phase_field_fault/normalization.cc` | Keep mature FE requests on native BP3 normalization/completion; retain P2 normalization independently. |
| `source/material_model/phase_field_fault/constitutive.cc`, `source/material_model/phase_field_fault.cc` | Mature nodal-zero cohesive response and optional configuration restrictions/documentation; native RSF law unchanged. |
| `source/simulator/phase_field.cc`, `source/particle/property/interface.cc` | Exclude inactive dense P2 phase block; register zero internal-stage storage for the named stationary diagnostic. |
| Maintained `bp3/plugin/{bp3.cc,particle_history.cc,runtime.h,work_audit.cc}` | Stationary diagnostic integrator and its no-Stokes setup guard; accepted FE stress observations and shared quadrature. Coupled full-bottom constraint checks remain. |
| `tests/phase_field_fault_test_access.h`, `bp3_native/{CMakeLists.txt,history_checks.cc,*.prm,fault.txt}` | Focused access, translated dipping history/normalization/tensor controls and unmodified legacy controls. |
| `doc/reconstructed_fault/{current_design.md,specification.tex,CURRENT_STATUS.md}`, `bp3_native/{STATUS.md,report.md,baseline.json}` | Paired selected contract, recovery/provenance and results. |

All production numerical changes above were already present when recovery began.
Recovery corrected the test's triangulation-to-DoF iterator conversion, its
inadmissible zero-rate relaxation control, serial guard and diagnostics; finished
the builds/checks; and updated documentation. The only additional core edit is
parameter help clarifying that tangential panels belong to evolving P2. Incoming
production source edits remain intact. Repository astyle 2.04 formats the changed
fixture; broader whitespace-only changes to two older files were removed to keep
the diff confined. Formatting snapshots are retained separately.

## Actual verification

Release stack: GCC 12.4, deal.II 9.6.2, Trilinos 14.2, p4est 2.8.7, Voro++ 0.4.6.
Core and both plugins build only in `build-bp3-native`, with each plugin's
`Aspect_DIR` pointing there. Numerical controls use one rank/thread, no mechanical
solve and End time=0. Each applies six diagnostic history increments to both a
divergence-free Q2 field and a nonzero-divergence Q2 field, restores publications
afterwards, and uses nonuniform surface slip. dt/lambda is
1e-12, 1e-6, .1, 1, 4, .25. The last has zero bulk velocity and the existing
RSF minimum rate 1e-20 m/s; it is a relaxation control with that small forcing,
not an exact zero-slip RSF step. These large-Maxwell-time increments are algebraic
stress tests, not physically plausible BP3 trajectories.

| Fresh control | Cells / edge range (m) | Samples / increment, continued | Max recurrence error / original allowance | Max I_h integration error | Runtime (s) |
|---|---|---|---|---|---|
| Graded q6 | 4136 / 7.8125–125 | 215072, 13044 | .0088812 | 5.97e-16 relative | 125.76 |
| Graded q8 | 4136 / 7.8125–125 | 330880, 20052 | .0113962 | 5.97e-16 relative | 248.39 |
| Uniform q6 | 2048 / 31.25 | 106496, 212 | .0099286 | 4.85e-16 relative | 129.67 |

All 36 increments pass the unchanged 128-epsilon, pre-cancellation operand-scale
stress checks. Maximum Z relative recurrence error is 1.78e-16. Absolute stress
errors reach 42.60 Pa in the artificial O(1e14–1e15 Pa) stress controls; they
remain below 1.14% of the original arithmetic allowance. Mature cohesive traction
stays exactly zero while Z is positive. Every case rejects an invalid scratch
candidate without changing d_R/Z, rejects duplicate terminal publication, and
rejects a moved particle. Production commit independently checks every derived
particle sample against direct Maxwell evaluation before writing it.

The manufactured divergence-free fields have measured |div u|≤1.53e-25 /s;
the nonzero-divergence fields reach 3.00e-12 /s. The full-symmetric-gradient
reference shares the incoming history and isolates the one-increment difference
`-2 eta_ve div(u) I2/3`. Every point agrees within the original arithmetic
allowance. This verifies the derived tensor distinction, not a complete
FE-versus-advected-particle trajectory or a production legacy-law change.

Changing q6 to q8 leaves normalization identical. Relative changes in the
manufactured slip-work forms are 1.59e-7 (divergence-free) and 3.31e-7 (nonzero
divergence); bulk-strain forms differ ≤9.45e-15. Graded minimum I_h is
12188.324109 m and the largest exterior contribution is 5762.268084 m, so
completion is actually exercised. Uniform minimum I_h is 4841.791606 m.
This substantial coarse-profile representation difference prevents interpreting
the uniform/graded controls as continuum spatial accuracy. Each independent
quadrature checks the corresponding discrete Q1 profile; it does not validate
the continuum profile approximation.

| Other check | Actual result / provenance |
|---|---|
| Fresh final compatibility | 21,112 assertions in 55 cases pass. |
| Fresh FE resume rejection | Intended material-parser fresh-run guard fires; no checkpoint conversion. |
| Retained matching-build legacy Stage-J, two ranks | Pass, 28.17 s; includes one-owner candidate failure and unchanged committed histories. Already completed, not reapplied. |
| Retained matching-build native boundary completion | Pass, 4.07 s. Already completed, not reapplied. |
| Preserved failures | Eight earlier startup/configuration/dependency failures; two recovery recurrence attempts expose the particle-iterator fixture defect; the corrected-iterator attempt reaches five valid increments then rejects zero RSF slip. None reclassified as passing. |
| Preserved failed build attempts | Missing utilities include, const particle iterator, and duplicate Stage-I linkage; their corrections were already present at recovery. Final core/plugin builds pass. |

The three successful fresh manufactured controls total 503.82 seconds of local
run time, excluding compilation and preserved failed attempts. The q8/uniform timings include their
concurrent execution and are not a paired performance comparison. No full mesh,
first-event run, server job, evolving phase-energy solve or saddle search is run.

## Commands, identities and next step

Build commands are `cmake --build build-bp3-native -j 2`, followed by the same
command for `build-bp3-native/bp3-plugin` and `build-bp3-native/history-plugin`.
The initialized core cache records Release, installed deal.II and Voro paths;
the supplied `bp3_native/CMakeLists.txt` configures the small history plugin.
Actual run commands, bounded wall limits, environments, outcomes and logs are in
`benchmarks/reconstructed_fault/refactoring_boundary/evidence/` under:

- `bp3-native-recovery-graded-q6-final`
- `bp3-native-recovery-graded-q8`
- `bp3-native-recovery-uniform-q6`
- `bp3-native-recovery-units`
- `bp3-native-recovery-reject-resume`

For example, the completed q6 command is:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py \
  bp3-native-recovery-graded-q6-final 600 \
  env ASPECT_SOURCE_DIR=/home/ein/repository/aspect-fault-propagation \
  build-bp3-native/aspect \
  build-bp3-native/recovery-20261013/history-graded-q6-final.prm
```

The runner refuses to overwrite its logs. For reproduction select a new label
and a copied input with an unused Output directory. Recovery's runtime inputs
change only that output path relative to the source fixtures; originals and old
outputs remain preserved. Full executable/plugin/source/input hashes are in
`build-bp3-native/recovery-20261013/tested-provenance.json`; compact numerical
tables are in `assessment.json`. The final delivery rebuild changes only help
text and removes broad whitespace formatting from the numerical-control build;
its matching plugins and compatibility suite pass. Final identities are in
`delivery-provenance.json` (core `1ba2f7f43b37`, maintained plugin `4356d84eaf6a`,
fixture plugin `e5e5482723a3`). `preservation-final.json` confirms no incoming
untracked file is lost, only the three authorized BP3 fixture/report files
change, and reference/staged states are unchanged. `git diff --check` passes.
Raw results remain under this worktree's
`build-bp3-native/runs/`, outside tracked source files.

Sections 1–6 are complete. No mathematical conflict required a new assumption.
Coupled FE/particle BP3 accuracy, native FE MPI/cutback campaign, spatial friction
transition, temporal refinement, unfiltered comparison, long-run readiness and
production input preparation remain later plan items. Proposed next task:
section 7B's bounded paired stationary-history coupled dipping-fault comparison,
retaining full bottom velocity and the 20 m filter. Await that selection.
