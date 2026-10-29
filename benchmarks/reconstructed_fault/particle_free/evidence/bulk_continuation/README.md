# P4 continuation evidence

Read `BULK_CONTINUATION_REPORT.md` for the scientific conclusion,
`SOURCE_AUDIT_P4.md` for the source/verification audit, and
`P5_DIRECTION_SPECIFICATION.md` for the proposed, unimplemented extension.
`BULK_CONTINUATION_SELECTION.md` preserves the predeclared controls/targets.
The eventual single ZIP contains these PNG/GIF, Markdown, CSV/JSON, input,
source-overlay and reproduction files. Full native records are separate;
`full-archive.json` gives their path, size and SHA256. This is an evidence
package, not a standalone ASPECT distribution or a first-event server job.

## Definitions and units

The natural-side residual gate, as before, applies to positive accepted steps;
step-zero ASPECT pressure precedes the first bulk solve and is not assigned a
new equilibrium qualification. All times are accepted physical times [s]; the continuation parent is .15875 s.
Loading ends at .175 s. Relative top/bottom horizontal displacement is
U=.018*clip(t-.05,0,.125) m. Curves against U retain multiple hold states at the
same displacement; they are not collapsed onto one value. Native work uses
actual imposed increments rather than differences of rounded output times.
Delta t denotes timestep; Delta x and Delta y denote cell sizes.

`*-history.csv` gives phase increments from each mesh's fresh initial Q1 field,
maximum accepted-step changes, integral of native |w| over area [m2/s], maximum
native-QP |w| [1/s], grip reaction [N/m], storage/work/dissipation [J/m] and the
separate BE/residual terms. In this bulk model a/k/L are strain memories and
w is a strain rate, distinct from surface slip [m]. Values are per unit
out-of-plane thickness. Trapezoidal work is half the old/new grip force times
native relative displacement increment; hold work is exactly zero.

`*-contours.csv` uses levels .001/.01/.05/.1/.3/.5 on rays at 0 and +/-45 degrees
from the original endpoint (1,.5) m. The reported front is the first exit from
the seed, not the furthest isolated point. Centerline crossings are physical
Q1 interpolations; `*-topology.csv` independently checks seed connectivity on
4/8-subdivision rasters. Raster extents have their own sampling error and are
not substituted for the Q1 crossing. Low-phase advance is process-zone advance,
not an assertion of developed crack-core propagation.

`*-profiles.csv` has common physical centers and radius a=.03125 m. The compact
kernel is K(r)=(1-r^2/a^2)^3 for r<a, zero outside. Native cell means of physical
stress/Y/|w| are averaged with cell-area weights; compatible weak fracture,
mechanical and residual forces use the same kernel with nodal lumped mass.
Each is normalized by the sum of its actual weights. Force comparisons verify
fracture-mechanical=residual. No local history is interpolated into a new
constitutive state. These are defined discrete kernel averages, not an
independently converged integration of a continuous convolution.

`fields/*-nodes.csv` contains selected physical Q1 phase, its own initial seed,
step/total increments and mass-normalized native weak forces. `*-cells.csv`
contains native Gauss8 cell means of physical deviatoric stress, Y and |w|.
The stress is reconstructed tau, not auxiliary T; zz=-(xx+yy) and both symmetric
xy terms enter the 3-D tensor metric. Reported stress L2 comparisons are of
these cell means. Native storage integrates actual quadrature-point stresses.

`*-comparison.csv` evaluates Q1 phase at common physical Gauss3 points on the
union grid, with exact Q1 squared norms and vertex peak differences. Each
mesh's own represented seed is subtracted for growth comparisons. Interval
growth subtracts each actual .15875 state; `interval_growth_defined=0` marks
zero growth at the parent, not a relative-error claim. Native common-time
comparisons and time-interpolant sampling errors are distinct. `*-sampling*`
interpolates only physical phase between saved times; it evaluates no new
constitutive state. No reflection or branch alignment is used.

`*-background.csv` records raw reaction, reaction divided by specimen width,
and excess relative to the homogeneous intact Maxwell response with the same
native displacement sequence. This removes the accounting effect of added
bulk/grip width; the intact solution is not a reference for the seeded problem.
`*-work-summary.json` and `temporal-work-uncertainty.json` distinguish the exact
BE identity from physical trapezoidal work, summed absolute per-step remainder
and cumulative Cauchy uncertainty. Initial fracture storage is not a work floor.

Stability matrices are the actual converged native tangent, physically weighted
and already condensed with native constraints. Mechanics/pressure and the local
constitutive return are eliminated for the phase Schur action. The eigenvalue
metric is Kf times lumped phase mass. Strict obstacle multipliers exclude phase
critical directions. Full matrices and eigenmodes stay in the native archive;
JSON reports eigen/linear residuals, subspace choices and feasible amplitudes.
`diagnostics/` retains the original failed full-space FD probes and corrected
critical P^T J P checks. A probe crossing a Coulomb branch is not retroactively
marked passed. No numerical tolerance is increased.

## Reproduction and provenance

Entry HEAD f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
Native v1--v6 and both observer DSOs have source/binary identities in the full
archive. The current source overlay contains this task's three edited files;
it is not the entire pre-existing implementation. Point law and history-owner
sources remain unchanged. `logged/` stores exact commands, environment,
bounded timeout, exit code and wall time, including diagnostic failures.
Native runs were sequential; elapsed times are local costs, not a performance
benchmark or server/full-event prediction.

The runner is the existing `refactoring_boundary/run_logged.py`; direct native
commands use four MPI ranks and the saved parameter files. Resume inputs copy
immutable checkpoint files and use the native post_resume_time_step signal to
reduce the pending timestep. Accepted clocks/owners and old timestep are never
rewritten. A microscopic old end-time interval remains preserved but is not
used to increase a pending interval; clean replays use native end-step stopping
and retain the original .185 s physical cap.

After extracting the full native archive under the original worktree layout,
set ASPECT_P4_ROOT to that worktree and run the `reproduction/continuation_*.py`
analysis tools. Native/rebuild reproduction additionally requires the local
ASPECT/deal.II/Trilinos/SUNDIALS/MPI toolchain recorded in the build commands.
`inputs/repository/` preserves recursive physical input includes. Original
failed runs remain separate from corrected runs and canonical merged datasets.
`SHA256SUMS`, `manifest.json` and the ZIP checksum cover the final package.

Primary datasets are `reference-continuation`, `half-continuation`,
`quarter-continuation`, `spatial-complete` and `domain-complete`.
`stable-*-common.csv` are the final common-time comparisons. Earlier
`temporal-{continuation,predictor,half-load,half-partial}.csv`,
`stable-native-*.csv` and `default-native-coarse-quarter.csv` are retained
development calculations: their peak column used Gauss-point sampling and is
superseded by the exact common-Q1-vertex peaks. Their original values are kept,
not used for final qualification. The prelaunch gate is likewise historical.
The current comparison subtracts interval parent fields directly and explicitly
marks zero interval growth, avoiding a roundoff ratio at the parent state.

The timestep study qualifies only advancement after the shared accepted .15875 s
parent. Its earlier .00125 s loading prefix is retained, not requalified at a
finer timestep. The GIF labels that parent rate with its original interval.
Fresh spatial/domain runs use their own continuum initialization and the same
whole-trajectory .00125 s sequence. Ordinary Q1 fracture integration is exact
on these affine cells; no new nonlinear constitutive-quadrature Cauchy control
is inferred from the earlier, differently scaled specimen.
