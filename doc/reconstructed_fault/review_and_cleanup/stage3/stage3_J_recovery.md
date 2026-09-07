# Stage-J recovery record

The requested Stage-J recovery is complete, and the evolving feedback,
checkpoint/restart, and CPDI-regeneration gates pass without the bypass.
This record separates pre-existing implementation from recovery edits and
records evidence before solver changes. All focused verification results are listed below.

## Recoverable starting state

Checkpoint: `/tmp/aspect-stage-j-recovery-25HUvk`, based on
`8a87657bf1fbe66263596125e80fb534475b20e8`.
`working-tree.patch`, `index.patch`, `working-files.tar.gz`, and
`test-output.tar.gz` preserve the initial dirty tree and relevant test outputs.
No broad restore/reset was performed; unrelated Stage-I/J work is retained.

Removed bypasses: `phase_field_evolution_is_enabled()` and the associated
solver-scheme conditional; the design paragraphs that reinterpreted
`Evolve phase field`; the Stage-J `1e-6` timestep and linear-tolerance override.
The original phase-field tolerances are **linear 2e-7, nonlinear 1e-5**.
The evolving feedback fixture explicitly selects `Evolve phase field=true`.
The established parameter meaning is retained: false freezes H, not the
phase-field solve. Stage-J H publication now respects that selection.

Recovery files in this pass: `solver_schemes.cc`, PhaseFieldFault header/source,
`current_design.md`, `specification.tex`, and the Stage-J parameter fixture.
`git diff --check` passed. The genuine Maxwell plugin-span lookup correction
and explicit compositional-property mappings were preserved.

## Initial phase-field failure: measured cause

Isolation command (two ranks, no mechanics, unchanged accuracy):

```
mpirun -np 2 build-pf-cpdi/aspect /tmp/aspect-stage-j-recovery-25HUvk/initialization.prm
```

The original ten-iteration budget silently returned at relative residual
`3.007e-4`, above `1e-5`. With only the nonlinear diagnostic budget raised to
50, the residual decreased regularly to `2.640e-5` after iteration 12; the
next CG solve failed at 1000 iterations. Initial/final/required linear
residuals were `8.789972e10`, `4.148324e5`, and `1.757994e4`.

Temporary instrumentation captured the actual failing matrix, RHS, iterate,
particle CPDI samples, and every CG residual. The records and independent
SciPy analysis script are in the checkpoint directory. Results:

| Quantity | Measured value |
| --- | --- |
| Matrix size / stored entries | 4225 x 4225 / 196249 |
| Relative symmetry defect | exactly 0 |
| Diagonal range | -1.080453e14 to 2.026926e13 |
| Nodal phi range | -0.00349349 to 0.000580809 |
| Particle CPDI phi range | -0.00250405 to 0.000505474 |
| Negative particle samples | 7662 |
| Rational denominator range | -1197.929 to 243.749 |
| Negative reaction-curvature samples | 1116 |
| Extremal eigenvalues after absolute-diagonal scaling | -7.07549, 6.67637 |
| Independent direct-solve relative residual | 3.00e-16 |

Here m=480000 and p=1. Newton crossed the negative-phi pole close to
`-2.083e-6`. Negative degradation curvature made the symmetric Jacobian
indefinite, invalidating CG and the elliptic AMG assumption. This is not
explained by insufficient Krylov iterations, pressure constraints, or merely
unfinished initialization. Switching preconditioners would not repair the
invalid constitutive branch.

The targeted correction rejects trial residuals outside the nonsingular
degradation branch containing [0,1], collectively across MPI ranks. It does
not clip phi in the residual, clamp H, or change the equations. Finite small
negative values on the valid branch remain possible; the separate I_h
bounded-negative rule and empirical 1e-4 guard remain unchanged. The
documented last-candidate line-search fallback remains available for a
defined residual, but cannot publish a singular candidate.

Nonlinear iteration exhaustion now throws the existing
`ExcNonlinearSolverNoConvergence` and follows the selected ASPECT failure
strategy. MPI ranks make the same decision from globally reduced residuals
and admissibility. The expected-failure regression deliberately permits one
Newton iteration and selects `abort program`.

```
cmake --build build-pf-cpdi --target aspect -j4
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_nonlinear_exhaustion$' -j1
```

Result before the domain fix: **1/1 passed**, 40.78 s. The corrected isolated
initialization then converged in **20 Newton iterations**, relative residual
`9.395e-6`, with **five CG iterations at every solve**. No linear tolerance,
linear budget, AMG setting, or line-search budget was changed. A nonlinear
budget of 50 is used in the focused fixtures; the observed history justifies
more than the old ten iterations. Full logs: `initialization-domain.log`.

## Frozen later-timestep isolation

`frozen.prm` uses the same initial data, zero prescribed velocity, no Stokes
solve or material history commit, dt=0.01, and end time 0.02. The three
phase-field solves completed with 20/15/8 Newton iterations and final
relative residuals `9.395e-6`, `4.747e-6`, `1.948e-6`; CG took 5--8
iterations. Log: `frozen.log`.

The relative nonlinear tolerance is anchored to the residual at each call,
so another solve may refine an already tolerance-converged frozen state.
This documented parameter meaning has not been changed. An additional
fixed-probe residual/Jacobian regression checks timestep independence and
the intended AT1 weak residual independently of that stopping convention.

## Controlled transverse-temperature fixture

The old test searched ordinary particles with matching rounded xi. The
reconstructed fault is slightly curved; vertical particle columns therefore
do not define coincident surface coordinates. MPI gathering cannot fix that
geometric premise.

The replacement constructs two points on the actual normal through an
interior segment midpoint, using supported offsets, then verifies both
production associations (fault, segment, xi) to 1e-12 before constitutive
comparison. Remote FE evaluation supplies the local temperatures. Production
point responses provide the bulk kappa and surface slope/history intercept;
there is no test-only Maxwell law. A large diagnostic initialization interval
makes bulk thermal dependence observable. V_min keeps intercept extraction
well scaled; neither coordinate nor coefficient tolerances were relaxed.
The surface temperature is independently checked against the fixture's
linear FE temperature at the actual surface point.

Files: `tests/phase_field_fault_stage_j_temperature{,_mpi}.{cc,prm,sh}` and
their `screen-output` references. The old particle search is removed from
the evolving Stage-J fixture. The one-rank test passed in 137.40 s and the
two-rank test passed in 77.51 s.

## Advection exposed a separate particle-domain defect

Once initialization and the first real phase-field solve converged, particle
advection exposed the existing global domain-volume assertion. A controlled
perturbed-grid reproducer measured raw Voronoi area `0.99999999999998557`,
but area `0.99999068326658858` after ASPECT's vertex cleanup. The old merge
distance, `1e-6` times the domain diameter, merged genuine short edges created
by particle displacement. Independent cleanup of neighboring polygons then
left gaps. This was not particle loss or a CG/preconditioner defect.

`source/particle/particle_domain.cc` now merges only coordinate-roundoff
duplicates (`64*epsilon*coordinate_scale`), using 64-bit spatial-bin indices
to accommodate that resolution. The polygon algorithm and the existing
global volume assertion are unchanged. The same reproducer now gives area
`0.99999999999998568`. The new `[particle_domain_area]` regression constructs
actual Voronoi/CPDI domains after deterministic 1e-6 particle perturbations
and checks area against one to 1e-10, on both one and two ranks.

This is a roundoff-level geometric identity safeguard, not a new user
parameter or a change to the phase-field approximation. Three-dimensional
reconstruction and overlapping faults remain deferred.

## Feedback and history verification

The feedback fixture keeps phase-field evolution enabled, its original
linear `2e-7` and nonlinear `1e-5` tolerances, and explicitly selected RSF
timestep restriction. It requests end step two because the standard selected
timestep models can shorten the second physical step. It does not force a
timestep that violates their restrictions (in this bound-active fixture,
the RSF limit is large and convection limits the second step).

The independent Theta reference integrates the selected aging ODE with
long-double arithmetic and does not call `update_state()`. Its comparison
tolerance was **tightened** from `1e-10` to `1e-12` relative to the expected
state. Each real step must have an increment exceeding 100 comparison
tolerances. Initial Theta and H are still tested for preservation through
the artificial Maxwell initialization interval.

The observed trajectory grows summed particle H by 6.23% on the first real
step. The second phase solve responds to that H and the irreversible maximum
then remains unchanged. An initially added assertion requiring strict H
growth on *every* step was mathematically inappropriate: the approved rule
is `max(old H, candidate)`, not strict increase. The fixture now requires
observable first-step growth, particle-by-particle nondecrease, and an exact
second-step plateau. No physical loading or solver tolerance was adjusted
to manufacture a second growth event. Particle IDs are gathered before
advection so this comparison remains valid across ownership changes.

After each commit, a production assembly at a fixed constant phi probe is
compared to an independent integrated AT1 weak residual using the newly
committed particle H. Together with the separate frozen-input residual and
Jacobian test, this verifies the return edge `H_k -> phi_(k+1)` rather than
mistaking continued relative-tolerance refinement for history feedback.

Local history errors previously could throw on one owner while peers entered
a projection collective. Candidate construction now catches those errors,
selects the first failing owner, and broadcasts its diagnostic before all
ranks throw. All persistent writes still occur only in the terminal commit.
The feedback test injects invalid H on exactly one owner after step two and
checks exact preservation of all particle and surface properties, current
and committed V, and the production bulk vector. It then restores only the
test-injected input. The nonlinear rollback tests separately check history
preservation after accepted Newton iterations and a subsequent forced failure.

Checkpoint/restart coverage saves after step one, continues the uninterrupted
run through step two, checks restored state before resumed mechanics, and
compares the resumed final state to the uninterrupted run. Its fingerprint
contains particle stress/H first and second moments, every fault property,
geometry coordinate and committed V, and bulk block norms. It is not an
entry-by-entry comparison of the complete bulk/particle arrays. Norms use an
owned vector copy; taking norms of ASPECT's ghosted solution was an initial
test-fixture error, not a production checkpoint defect.

## Focused verification record

All builds use `-j4`. No complete ASPECT test suite was run. Temporary
matrix/profile/raw-volume instrumentation has been removed; the failing
matrix, CG history, analysis script and diagnostic patch remain recoverable
in the checkpoint directory.

Commands below run from the repository root. Individual CTest commands use
`ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '<pattern>' -j1`
unless otherwise noted.

| Test / pattern | Result |
| --- | --- |
| `^phase_field_fault_stage_j_temperature$` | PASS, 1 rank, 137.40 s |
| `^phase_field_fault_stage_j_temperature_mpi$` | PASS, 2 ranks, 77.51 s |
| `^phase_field_frozen_history$` | PASS, 2 ranks, final rerun 58.34 s |
| `^phase_field_fault_stage_i_rollback$` | PASS, 1 rank, final rerun 50.25 s |
| `^phase_field_fault_stage_i_rollback_mpi$` | PASS, 2 ranks, final rerun 39.36 s |
| `^phase_field_fault_stage_j$` | PASS, 2 ranks, 376.62 s, after all production fixes |
| `^phase_field_fault_stage_j_restart_create$` | PASS, 2 ranks, 334.74 s |
| `^phase_field_fault_stage_j_restart_resume$` | PASS, 2 ranks, 118.44 s, unchanged 1e-10 comparison |
| `^phase_field_particle_domains$` | PASS, 2 ranks, 55.37 s, exactly zero regeneration error |
| `^phase_field_nonlinear_exhaustion$` | PASS, 1 rank, 63.60 s; direct 2-rank invocation also failed as intended |
| `build-pf-cpdi/aspect --test '[phase_field_domain],[particle_domain_area],*cohesive*,MaxwellStress*'` | PASS, 35 assertions in 9 cases |
| `mpirun -np 2 build-pf-cpdi/aspect --test '[particle_domain_area]'` | PASS, 1 assertion in 1 case on each rank |

The final feedback/restart/CPDI batch passed **4/4**, 885.19 seconds total:

```
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_stage_j($|_restart_(create|resume)$)|^phase_field_particle_domains$' -j1
```

The complete CTest output is preserved in `final-feedback-restart-ctest.log`
in the recovery checkpoint. The frozen-input and nonlinear-rollback cases
also passed their final-executable reruns. The one-rank rollback passed in
50.25 s; a sandboxed MPI launch was blocked by socket permissions before
ASPECT started. The unchanged two-rank cases then passed outside the sandbox:

```
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_frozen_history$|^phase_field_fault_stage_i_rollback_mpi$' -j1
```

Result: **2/2 passed**, 97.71 s total. No MPI settings or test tolerances were
changed to bypass the sandbox restriction.

## Recovery-only file inventory and interfaces

This inventory is relative to the saved dirty-tree checkpoint, **not HEAD**;
earlier uncommitted Stage-I/J changes are not attributed to this recovery.
The checkpoint's `review-before` and `review-after` directories provide a
reviewable no-index comparison without altering the repository index.

| Files | Recovery change / invariant |
| --- | --- |
| `doc/reconstructed_fault/current_design.md`, `doc/reconstructed_fault/specification.tex` | Remove the bypass's reinterpretation of `Evolve phase field`; retain approved Stage-J equations and history semantics. |
| `doc/reconstructed_fault/review_and_cleanup/stage3/stage3_J_recovery.md` | Evidence, verification, scope and recovery record. |
| `include/aspect/material_model/phase_field_fault.h`, `source/material_model/phase_field_fault.cc` | Remove the bypass accessor; respect frozen-H selection; make particle-local candidate failures collective before projection/publication. |
| `source/simulator/solver_schemes.cc` | Restore unconditional phase solve in the enabled phase-field scheme, including reconstructed-fault feedback. |
| `include/aspect/phase_field.h`, `source/simulator/phase_field.cc` | Reject constitutively undefined Newton trials, correctly report nonlinear exhaustion, preserve failing CG history, expose a minimal private test seam. |
| `source/particle/particle_domain.cc` | Preserve true Voronoi edges and area after small particle displacements. |
| `tests/phase_field_fault_stage_i.prm` | Increase only the phase Newton budget to 50, retaining accuracy. |
| `tests/phase_field_fault_stage_i_rollback.cc` | Check exact particle/surface history preservation after accepted Newton updates followed by failure. |
| `tests/phase_field_fault_stage_i_rollback_mpi.cc`, `.prm`, `.sh`, `tests/phase_field_fault_stage_i_rollback_mpi/screen-output` | Two-rank counterpart of the rollback check. |
| `tests/phase_field_fault_stage_j.cc`, `.prm`, `.sh`, `tests/phase_field_fault_stage_j/screen-output` | Restore evolving feedback through two real steps; independent resolved Theta reference; observable H growth, irreversible plateau and fixed-probe return edge; one-owner failed-history preservation. |
| `tests/phase_field_fault_stage_j_restart.cc` | Shared checkpoint fingerprint/save/load and resumed-state verification. |
| `tests/phase_field_fault_stage_j_restart_create.cc`, `.prm`, `.sh`, `tests/phase_field_fault_stage_j_restart_create/screen-output` | Save after step one and continue through step two as the independent restart reference. |
| `tests/phase_field_fault_stage_j_restart_resume.cc`, `.prm`, `.sh`, `tests/phase_field_fault_stage_j_restart_resume/screen-output` | Resume the saved checkpoint, check restored state, and compare final feedback with the uninterrupted run. |
| `tests/phase_field_fault_stage_j_temperature.cc`, `.prm`, `.sh`, `tests/phase_field_fault_stage_j_temperature/screen-output` | Controlled transverse evaluation through the production geometry and constitutive paths. |
| `tests/phase_field_fault_stage_j_temperature_mpi.cc`, `.prm`, `.sh`, `tests/phase_field_fault_stage_j_temperature_mpi/screen-output` | Two-rank counterpart with identical association/coefficient tolerances. |
| `tests/phase_field_fault_test_access.h` | Remove unused test-only Maxwell wrapper; prepare the required transient surface temperature in the existing cohesive-initialization test seam. |
| `tests/phase_field_frozen_history.cc`, `.prm`, `.sh`, `tests/phase_field_frozen_history/screen-output` | Verify the independent AT1 residual and identical fixed-probe Jacobian/residual at initialization and two frozen-input later timesteps. |
| `tests/phase_field_nonlinear_exhaustion.cc`, `.prm`, `.sh`, `tests/phase_field_nonlinear_exhaustion/screen-output` | Deliberate one-iteration nonlinear exhaustion must fail before reconstruction. |
| `tests/phase_field_test_access.h` | Testing-only access to phase assembly and CPDI vertex/DoF mapping. |
| `unit_tests/particles.cc` | Perturbed-particle Voronoi area regression. |
| `unit_tests/phase_field_fault_ih.cc` | Rational constitutive-domain regression, including the disconnected lower branch. |

Public-interface changes in this recovery are limited to adding
`PhaseField::DegradationFunction::is_in_domain(double) const` and removing
the bypass-only `PhaseFieldFault::phase_field_evolution_is_enabled() const`.
The former identifies the connected rational branch containing [0,1],
subject to the physical upper bound; it is not the I_h lower clamp or the
fault activation criterion. Phase assembly's boolean admissibility result
and the friend declaration are private. No runtime parameter was added.

Remaining scope limits: all reconstructed-fault integration fixtures here
are two-dimensional and single-fault; no complete integration suite or
release build was run. Existing relative Newton stopping semantics can
permit additional refinement on a repeated solve, as explicitly checked
with the frozen-input test. The normalization fixture's low reconstruction
activation threshold bootstraps the prescribed fault; reconstruction
accuracy is not under test. No assertion is made about general 3-D Voronoi
geometry from the new 2-D conservation regression.

## Restart isolation: stale ghost positions in CPDI construction

The first complete uninterrupted calculation passed its production and
postprocessing invariants. Its shell filter initially miscounted the Stage-I
label because ASPECT pads output columns with spaces; the filter now accepts
horizontal/vertical whitespace between the label and `verified`. This does
not change any numerical assertion. The filter replay reports exactly three
lifecycle checks, three feedback checks, and two real-step commit diagnostics.

The resumed calculation preserved the saved state before solving, but its
final Maxwell-stress moment differed from the uninterrupted reference at
the unchanged `1e-10` fingerprint tolerance. Its phase Newton history also
differed (`4.879e-7` versus `4.922e-7` at the last iteration). This was not
silently accepted as solver noise.

`Particle::Manager::advance_timestep()` constructed Voronoi/CPDI domains
**before** refreshing ghost particles. Domain construction uses neighboring
ghost positions, whereas the restart path refreshes ghosts before rebuilding
domains. Thus these paths did not define the same discrete operator.

The independent two-rank `phase_field_particle_domains` regression prescribes
a small interior-preserving velocity, snapshots production volume/support/
CPDI weights and gradients after advection, and regenerates domains for the
unchanged particles. Before the fix it failed in 56.84 s: even the CPDI
support changed. At timestep zero its error was exactly zero. After moving
construction below the existing ghost exchange, both initialization and the
real timestep report **exactly zero** regeneration error. The `1e-11`
comparison tolerance is unchanged. The first successful numerical run was
followed by an output-script executable-bit correction; final CTest results
are recorded separately.

This adds `source/particle/manager.cc` and
`tests/phase_field_particle_domains.{cc,prm,sh}` plus
`tests/phase_field_particle_domains/screen-output` to the recovery-only
inventory. The change is cache-lifecycle ordering, not a new particle motion,
constitutive, or CPDI integration algorithm. It also avoids invalidating
neighbor references immediately after their construction.

Two large debug runs launched concurrently exceeded CTest's existing
600-second wall-clock limit before finishing step two. Final heavy tests
are run serially; neither that timeout nor any numerical convergence
threshold was increased. The complete uninterrupted run takes about
340 seconds when run without a competing large simulation.

The nonlinear exhaustion CTest rerun passed in 63.60 s. A direct two-rank
run also exited with the expected code 1 and printed the configured abort
diagnostic on both ranks, before reconstruction:

```
cd build-pf-cpdi/tests
mpirun -np 2 ../aspect /tmp/aspect-stage-j-recovery-25HUvk/exhaustion-mpi.prm
```

Its complete output is `exhaustion-mpi.log` in the recovery checkpoint.
The test harness only supports `EXPECT FAILURE` with one rank, so this
additional two-rank negative case is recorded as a direct invocation.
The default production nonlinear iteration budget remains ten; only the
focused parameter fixtures select 50. Exhausting a budget now follows the
configured failure strategy instead of being reported as success.

## Final initialization and feedback measurements

A read-only GDB invocation recorded all successful initial CG residuals,
without any further production instrumentation:

```
gdb -q -batch -x /tmp/aspect-stage-j-recovery-25HUvk/successful-cg-history.gdb --args build-pf-cpdi/aspect /tmp/aspect-stage-j-recovery-25HUvk/successful-initialization.prm
```

It exited normally. The complete per-CG residual histories are in
`successful-cg-history.log` and the accompanying nonlinear history is in
`successful-initialization/log.txt`. This one-rank repeat has 20 Newton
linearizations, each with five CG iterations; final/initial CG residual
ratios range from `6.04944e-8` to `9.05867e-8`, all below the unchanged
`2e-7` request. The initial nonlinear result remains `9.395e-6`.

After the ghost-ordering correction, the **main evolving Stage-J CTest
passes** in 376.62 s. Newton counts for initialization/step one/step two are
20/15/12, with final relative residuals `9.395e-6`, `4.747e-6`, and
`4.879e-7`. The two physical step sizes are `0.01` and approximately
`0.000201994` seconds. Observable changes are:

| Quantity | Step one | Step two |
| --- | --- | --- |
| Relative summed H change | 6.230e-2 | Exact plateau at every particle |
| Phase-vector increment norm | 7.336 | 10.35 |
| Maximum Theta increment / comparison tolerance | 4.545e5 | 9.182e3 |

The summed H diagnostic on step two can differ by roundoff due to particle
iteration/reduction ordering (`1.261e-16` relative in this run); the stronger
particle-ID-based comparison proves exact preservation of every H value.
The one-owner failed-history test passes with all persistent state preserved.
The complete log and filtered result are saved as `final-feedback.log` and
`final-feedback.screen-output` in the recovery checkpoint.

## Handoff

The final executable is the Debug, Voro-enabled build at
`build-pf-cpdi/aspect` (symlink to `aspect-debug`). Builds used
`cmake --build build-pf-cpdi --target aspect -j4`, with focused plugin builds
also using `-j4`. At the recovery handoff, no new commit had been made;
the original working-tree changes remained present and the index had not
been altered.

The recovery checkpoint additionally contains `recovery-only.patch`,
`recovery-only.stat`, and `final-focused-test-output.tar.gz`. These preserve
the recovery diff relative to the starting dirty tree and the final focused
outputs, separately from `working-tree.patch` and the original test-output
archive. The before/after directories can be inspected without applying a
patch to the repository. `git diff --check` passes.

No known blocker remains for the requested recovery. This is not a claim
that the complete ASPECT suite, 3-D/overlap cases, arbitrary constitutive
parameters, or a release build were verified. In particular, excluding the
observed rational pole crossing does not prove positive definiteness for
every possible material model; it fixes the measured failure and is backed
by the recorded successful linearizations for these intended fixtures.
