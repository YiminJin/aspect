# PF2 fixed-profile native history experiment — 15 October 2026

**PF2 passes its selected numerical gates. Stop before PF3.** The default-off
`Bulk history representation = DG trial stress` option stores authoritative
DGQ2 trial stress and retains the existing physical cohesive elastic-slip law.
The final FE control exactly reproduces a fresh pre-change source baseline;
the same-build FE/DG comparison passes with identical accepted initialization.
This is a particle-free **bulk-history** prototype, with explicit remaining
particle dependencies. It is not evolving-phase, transport or full-model
qualification.

## Provenance and scope

Worktree `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus
this uncommitted PF2 implementation. Incoming `AGENTS.md`, PF0/PF1 evidence,
plans and manual derivation are preserved. No other checkout/build was changed;
nothing was staged, committed or pushed.

A new `build-particle-free/native` directory was configured and compiled **before
production edits**. The fresh baseline's 875 source/include/unit/CMake file
hashes all match HEAD and its production patch is empty
([source proof](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf2/baseline-source-proof.json)).
This removes PF1's retained-binary source-correspondence limitation for the
present PF2 control. It does not retroactively requalify the historical BP3 run.

| Artifact / run | Identity and result |
|---|---|
| Fresh pre-change baseline | `baseline-fe-v2`; exit 0, 17.096s; core SHA256 `89643b5eab1b1ab540906d1ab01804f71dff38bde181e7a9398eddbf45b1ce74` |
| Final FE and DG core | Both `7a4dcd559d54cd809e28575f590d835dc3d173795b5b19b817070d52f12be843` |
| Final FE and DG observer | Both `26fbf447ab722fd27ca3d104025f68870a5de8137709a98dfe334aa3128b1c1d` |
| Final FE | `fe-shared`; exit 0, 40.103s, peak child RSS222748KiB |
| Final DG | `dg-shared`; exit 0, 39.798s, peak child RSS223484KiB |
| Accepted initial phase | Saved coefficient CSVs byte-identical: SHA256 `ad7d06937963d2e1b7ebf958fc8555f39eeb0394eebe05634f03b0889b7a6bfb` |

Build: GCC 12.4.0 from `/opt/gcc/12.4.0`, Release, deal.II 9.6.2 from
`/opt/dealii/9.6-local`, Trilinos 14.2, p4est 2.8.7, Voro enabled, NetCDF enabled.
The actual linked system MPI is Open MPI 5.0.10 (`/usr/lib/libmpi.so.40`), not
the older propagation report's MPI version. Each run is serial with OMP,
deal.II and OpenBLAS thread counts1. Native controls have a 180 s hard limit;
invalid-configuration controls30s. At most two native controls overlap.
Observed diagnostic wall times include repeated trials, manufactured checks,
poisoning and rejection; they are not a production-step speed comparison.

## Fixture and accepted equations

The fixture inherits `propagation/native_fe.prm`/`recoverable.prm` physical
conventions through the existing recoverable/coupled inputs, with an explicit
small uniform mesh: unit Box, 8x16 affine Cartesian cells, Q2 velocity/Q1
pressure and phase, 16 surface segments and 3x3 fixed particles per cell.
The straight through-going horizontal fault retains full-normal association.
G=1e6Pa, eta=1e8Pa s, lambda=100s, cohesion1000Pa, Gc=26.6666666667Pa m,
ell=.15625m, Lorentz p=1, prescribed seed core.6, fixed Coulomb mu=.1 and
prescribed pressure10000Pa remain explicit. Radiation damping is zero.
There is no RSF evolution or particle transport.

Native H/CPDI initialization and its bound-constrained relaxation supply the
accepted seed; unloaded initialization gives physical tau=0 and Z=q=0.
H remains at its initialized values during this experiment. The test observer
then holds the accepted phase fixed and drives the **native coupled Stokes/
surface solve and terminal history publication**, rather than assigning V or q.
The three increments use dt=.125s at t=.125,.25,.375s. Prescribed side/boundary
velocity is `0.256*t*(t<.3)*(y-.5)` in x and zero in y: two loading increments,
then zero boundary velocity while stored history relaxes.

The enclosing ASPECT invocation performs initialization only; the observer
controls these mechanical steps without an outer phase/advection loop. Its
output records the actual physical step/time labels and commits. This is not
a normal outer-loop trajectory or a checkpoint/restart test.

At fixed localization the unchanged law is

```
beta = exp(-dt G/eta), kappa = -eta expm1(-dt G/eta)
tau_tr+ = beta tau_tr- + 2 kappa dev3(symgrad2 u+)
Z+ = beta Z- + kappa V+,       s_elas = Z/G,       q = Z/I
tau+ = tau_tr+ - 2(h/I) Z+ S.
```

The circle/spin term is absent, as in the audited native law. The compact
cohesive representation is exact under these frozen, matched-coefficient
assumptions. No redistribution is added or removed; evolving-profile and
transport commutators are outside this experiment. The DG history stores
in-plane xx,yy,xy and reconstructs zz=-(xx+yy).

## Implementation and responsibilities

| Files | PF2 responsibility |
|---|---|
| `phase_field_fault.h`, `phase_field_fault.cc` | Add the optional selector and a common `uses_independent_bulk_history()` query. Preserve the existing particle default and FE selector. Reject unsupported DG configurations |
| `phase_field_fault/dg_history.cc` | Own the scalar FE_DGQ2 DoFHandler/space, three distributed accepted tensor vectors, noncommitting source projection/Maxwell trial, physical stress reconstruction and test snapshot |
| `phase_field_fault/fe_strain_internal.h`, `fe_strain.cc` | Reuse fixed phase/geometry, analytic normalization and physical Z/I data. In DG mode the FE displacement velocity block remains exactly zero and is neither advanced nor sampled for stress. Route retained/candidate stress to DG; swap complete candidates only at terminal publication |
| `phase_field_fault/history.cc` | Bypass parent stress reads, compare derived DG samples with the separate native Maxwell recurrence, validate before writes, publish derived samples and DG state once, and identify unsupported DG checkpoint provenance |
| `phase_field_fault/normalization.cc` | Reuse the same consistent FE normalization path for both native owners |
| `surface_system_bulk_work.cc`, `assemblers/reconstructed_fault_stokes.cc` | Bypass mapped stress reads for both native owners; consume one authoritative retained physical stress in surface and bulk loads |
| `simulator/phase_field.cc`, `fe_phase_energy.cc` | Reject DG physical phase stepping/energy requests. Reuse the FE control's initial phase graph so initial CPDI relaxation has identical arithmetic |
| `tests/phase_field_fault_test_access.h`, benchmark fixture/observer/runners/checker | Narrow snapshots and source/operator/lifecycle controls, full-field/load comparison and artifact identity |

Paths in the table are beneath the existing `include/aspect/material_model/`,
`source/material_model/`, `source/reconstructed_fault/` or `source/simulator/`
owners. No new manager/backend hierarchy or generic-projector change was made.
The new nested Space groups the immutable element/DoFHandler and cell lookup so
trials can share the space without sharing mutable coefficients. Initialization,
trial projection and stress evaluation are substantial numerical operations;
the snapshot method is test-only.

Gauss3 per coordinate integrates the scalar DGQ2 mass and native affine-Q2
source products exactly. The existing mechanical rule is retained: two
tangential intersection panels per bulk cell with Gauss3 each, and six normal
points. Normalization and CPDI fracture/initialization remain separate. No
physical kernel is projected into DG, no h/I floor/cap/filter is added, and no
material/tolerance/solver parameter is tuned.

The actual specification diff adds only the selected optional PF2 contract to
`current_design.md` and `specification.tex`: ownership, frozen-localization
equations, tensor convention, quadrature and lifecycle restrictions. The
existing cohesive, Maxwell, friction and H equations/defaults are not revised.

## Measured verification

The complete [assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf2/assessment.json)
has `passed=true`. All resolved settings match except output/library paths and
the selected bulk-history owner. Tensor norms include zz and both xy entries;
bulk RMS uses actual shared JxW over1m². Scaled errors use the predeclared unit
SI scale or reference norm, whichever is larger. Field/load targets are1e-10
scaled; velocity/slip targets1e-12 on unit SI scales. Geometry, phase and I
comparisons are exact. No tolerance was loosened.

| Gate | Measured result |
|---|---|
| Fresh baseline / disabled-DG FE control | All four bulk and four surface CSVs byte-identical; all common field/energy values exactly equal |
| Initial localization | All saved phase coefficients, sampled phase and I exactly equal across final FE/DG; zero change throughout each path |
| Physical cohesive storage | DG peak q=67.24372179388826Pa, peak s_elas=.007291729801607329m; positive in all three accepted steps |
| Surface Maxwell recurrence | Maximum absolute Z defect9.095e-13Pa m at unchanged native acceptance |
| FE/DG full fields | Maximum component-scaled difference3.195e-14; full3D stress maximum point error4.550e-10Pa, maximum volume-RMS error7.416e-11Pa (scaled5.650e-15) |
| Native weak bulk/surface loads | Maximum scaled difference7.319e-13; actual bulk RHS max difference2.272e-11Pa m |
| V5 manufactured source, unloaded and nonzero accepted anchors | Ten general-beta cases/backend, dt/lambda=1e-12,1e-6,.1,1,4. DG full3D max error4.888e-11Pa, max scaled2.372e-15; FE max scaled1.201e-14; target1e-12 |
| V5 representation negative controls | Actual mapping DGQ2 scalar mass rank 9 with Gauss3, rank 4 with Gauss2; DGQ1 loses analytic source modes, L2 error5.8684e-9m/s |
| V6 actual B/G work maps | Both 946.6269045328081W/m; measured adjoint difference0. Independent quadrature differs1.592e-12W/m (~1.68e-15 scaled) |
| Particle-stress independence | Poison particle stress and all mapped stress fields by1e7–6e7Pa, clear native sample caches: exact zero change in full native weak loads, surface loads, mechanical u/p and V, at all three entries |
| V9 trial/commit/failure | Repeated trials identical; invalid candidates leave accepted FE/DG/Z unchanged. After mechanical convergence, injected invalid H rejects before publication; every accepted bulk/surface/phase/V and noninjected particle value restores exactly. Same-step retry completes; three total physical publications; duplicate commits rejected without history mutation |
| Restricted support | Changing particle position or phase is rejected in DG validation. Evolving-H/phase and restart parameter controls fail at the expected markers |
| Existing unit controls | Fresh baseline and final builds both pass 628 assertions in 8 cohesive/Maxwell cases |
| Independent PF1 compatibility |94/94 mathematical checks still pass |
| Compilation/style | Fresh full build, final incremental build/plugin, standalone new DG translation-unit syntax compilation without unity/PCH, ASPECT2.04 formatting for new C++ files, Python/JSON/whitespace checks pass |

Bulk/cohesive recoverable storage at t=.25s is43.0758343708684 and
.09459846052594J/m. These are measured same-formulation comparisons, not a new
global work/loss balance or temporal-accuracy qualification. The existing
hold-step nonlinear active-set search takes additional iterations in these
controls; its tolerances and acceptance are unchanged.

### Retained failed controls and initialization correction

The first baseline launch failed input parsing because inherited PRMs named
unloaded historical observers. The runner now expands includes and replaces
those observer declarations, retaining physical inputs. No core equation or
expected physical result was changed to resolve that harness error.

The first complete comparison's exact-initialization gate failed: sampled phi
differed 5.55e-17 and I differed 1.42e-14m. Source inspection found FE initialization
used its complete phase sparsity graph, whereas DG initially used the generic
graph. Although both assembled the same CPDI equations, CG/AMG followed different
arithmetic. DG now reuses the FE initial graph, and the unchanged exact gate
passes, including saved coefficients. This is a focused discrete-initialization
compatibility change for the new option. It is not permission to evolve DG
phase, change the seed, filter it or relax bounds/tolerances. The
[failed assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf2/failed-initialization-assessment.json)
and every failed/preliminary run remain preserved. No production numerical bug
was found requiring a constitutive correction.

## Remaining dependencies and limits

Particles remain required by `PhaseFieldHandler` initialization, native H and
CPDI/domain data; the generic surface material projection and support/admission
services; particle chemical data used during publication validation; and derived
stress/H sample output. Even unused frozen H is still validated before terminal
publication, as the failure control demonstrates. Prescribed geometry itself
can be retained without deriving it from stress particles, but these live
manager services still prevent an end-to-end particle-free run.

Only fresh serial homogeneous unloaded cohesive data, the fixed horizontal
Cartesian fixture, fixed phase/geometry/population and no transport are covered.
General initial stress is not converted from particles. Mapped/non-affine meshes,
varying coefficients, RSF evolution, spin, birth/AMR/MPI/restart and broad
stability/space/time convergence are not qualified. The old mature-memory
counterexample remains valid: this frozen cohesive experiment does not authorize
dropping mature residual history when the commutator is nonzero.

## Reproduction and next task

From this worktree, use unused run labels:

```sh
cmake -S . -B build-particle-free/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_C_COMPILER=/opt/gcc/12.4.0/bin/gcc -DDEAL_II_DIR=/opt/dealii/9.6-local -DASPECT_WITH_VORO=ON -DVORO_DIR=/home/ein/local/voro++/0.4.6
cmake --build build-particle-free/native -j4
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/plugin -DAspect_DIR=/home/ein/repository/aspect-pf-rsf-particle-free/build-particle-free/native -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-particle-free/plugin -j2
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py fe-repeat --input benchmarks/reconstructed_fault/particle_free/fixed_profile_fe.prm
python3 benchmarks/reconstructed_fault/particle_free/run_fixed_profile.py dg-repeat --input benchmarks/reconstructed_fault/particle_free/fixed_profile_dg.prm
python3 benchmarks/reconstructed_fault/particle_free/assess_fixed_profile.py --baseline build-particle-free/runs/baseline-fe-v2 --fe build-particle-free/runs/fe-repeat --dg build-particle-free/runs/dg-repeat --output build-particle-free/repeated-assessment.json
build-particle-free/native/aspect --test '[phase_field_fault_cohesive],*Maxwell*'
```

Run the two `reject_*.prm` inputs through `run_fixed_profile.py` with
`--expect-failure-marker` as recorded in their evidence. Raw binaries, fields,
expanded inputs/configs, build logs and failed states stay under
`build-particle-free/`. Compact results, artifact/source identities and selected
tables are under `benchmarks/reconstructed_fault/particle_free/evidence/pf2/`.
The full-field checker needs the local raw native outputs; compact assessments
are not substitutes for them. The pre-change baseline executable/observer remain
pinned in their original run directory; rebuilding the modified source cannot
recreate that pre-change executable.

Proposed next task: **PF3 discrete phase energy and derivatives with immutable
DG trial stress and physical Z on the same fixed surface**, explicitly choosing
the FE fracture functional, differentiating the actual projected normalization,
and comparing energy/residual/tangent against independent directional checks
and the retained CPDI functional. No phase evolution, transport or limiter work
is started here.
