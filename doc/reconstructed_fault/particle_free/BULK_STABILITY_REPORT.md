# Bulk hold instability resolved within the tested setting

The supported loading-aware continuation and a prepublication relaxed-curvature
gate now prevent acceptance of the demonstrated hold saddle. All selected
four-rank checks pass. This closes this instability investigation for the
retained homogeneous, matched-coefficient, fixed-direction bulk model. It does
**not** pass P4's failed quantitative propagation criteria. Free direction remains
unimplemented and requires scientific review of the [P5 recommendation](P5_DIRECTION_SPECIFICATION.md).

Entry: f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
The [predeclared selection](BULK_STABILITY_SELECTION.md) and
[verification JSON](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_stability/verification.json)
record inputs, immutable checkpoint hashes, exact commands, bounds and costs.
The original [P4 report](BULK_CONTINUATION_REPORT.md), archive and failed checks remain.

## Remedy and mathematical scope

For a change in prescribed velocity, or all-zero prescribed velocities, the
supported guess is accepted phase, zero interior displacement increment and
beta-scaled accepted pressure, followed by current boundary constraints.
Unchanged nonzero loading retains the established velocity predictor. A
128-epsilon comparison of the boundary-increment norms detects roundoff-level
equality; it is not a physical acceptance tolerance. Every subsequent Newton
iteration allows phase, mechanics and constitutive response to change freely.
Stopping boundary motion does not remove stress.

For the established constant matched Maxwell coefficients,
beta=1/(1+Delta t G/eta) and

    Pi = Eb + Ec + Ef + beta*Df_step,
    Df_step = Delta t * integral(R*abs(w)).

The weighted actual tangent uses beta*G on mechanical/pressure rows and Kf on
phase rows. In those physical coordinates the relaxed phase Hessian is
Hred=D-B^T A^-1 B, with A the mixed mechanics/pressure block. Native constraints
have already produced C^T H C. Artificial constrained rows are excluded; no
contribution is condensed again. Local constitutive response is already
eliminated by the retained consistent tangent. A frozen-mechanics D check would
miss the demonstrated instability.

The numerical critical superset contains every strictly positive phase
increment and obstacle masters with normalized gradient <= the existing 1e-7
KKT tolerance. The metric is Kf times lumped master mass. A complete dense
spectrum of the relaxed operator is computed on rank zero, with at most 1024
critical masters. Positivity outside the exported floating-point margin accepts
this check. With no critical phase directions, strict obstacle multipliers and
the positive incompressible mechanical tangent establish local stability.
The mechanical engineering-shear coercivity bound here is 39999.00002 Pa.

A negative mode is labelled an instability only if its expanded physical phase
direction has a nonzero bound-feasible amplitude. An unresolved, infeasible-mode
or oversized check also prevents publication. The decision/error is broadcast
to every rank. There is one retry from the same accepted owner and obstacle
using the accepted-state guess; another failed check stops before publication.
Neither trial constructs the next history owner until the stability gate passes.

This is a numerical local second-variation safeguard for this potential and
the checked return branches, not a global-minimum, finite-jump or general
nonsmooth-cone theorem. The margin reports eigen residual, Schur asymmetry and
128*epsilon*m*max(1,Schur infinity norm); the mixed solve residual and cross
symmetry are checked separately. It is not an interval-arithmetic enclosure
of an arbitrarily ill-conditioned mixed inverse. A nonpositive result in a
linear cone superset need not supply a feasible mode; that case fails closed.
The size bound makes this unsuitable as a scalable large-growth solver.

## Measured native checks

All use the unchanged 96x128 mesh, Gauss8/source3, unlimited DGQ2 plus native FE
source memory, four ranks, Delta t=.00125 s, original loading/material inputs
and original tolerances. Short immutable checkpoints replace full trajectories.

| Check | Relaxed minimum / response | Solver / gate / process wall [s] | Linear solves |
|---|---|---:|---:|
| Loaded step 139 -> 140, .17375 -> .175 s | +.139907314547; max dphi=.08560809023 | 165.07 / 85.25 / 171.00 | 10 |
| Forced former predictor, first hold | -.208558482525; rejected dphi=.01583691082; recovered dphi=0 | 156.54 / 87.93 / 164.88 | 9 |
| Supported hold, steps 141–143 | dphi=0, w=0, strict phase obstacle | 11.23 / .00258 / 25.73 | 0 |
| Two outer rejections then accepted step 141 | all owners unchanged; deterministic retry | 10.85 / .00265 / 18.80 | 0 |
| Restart at 141, advance to 143 | bitwise identity with unsplit hold | 7.35 / .00174 / 18.32 | 0 |

The gate adds about 85–88 s when 544–574 phase masters are critical: about
52% of the loaded solver time. Strict-obstacle hold checks cost .8–1.0 ms/step.
Reported process times include restart/output; solver times include the gate
and intentionally rejected work. Runs are sequential. Peak rank memory is
1247.16 MiB for loading and 1240.85 MiB for forced recovery, versus 879.16 MiB
for the three-step supported hold. These are local costs, not server estimates.

The negative mode's phase-peak-normalized feasible amplitudes are +.000188967
and -.928397. Its eigen residual is 2.16e-13; the relaxed solve residual is
3.99e-15. The 5.07e-9 floating-point margin is far below the negative curvature.
Its value agrees with the independent archived Schur calculation within 3e-13;
the loaded value agrees within 4e-13. Publication remains at 140 for both
candidate attempts, then increases once to 141. Phase and constitutive freedoms
remain available in the successful retry.

Complete 38,535,168-byte physical-history exports compare bitwise: all 27 DG
coefficients/cell, native projection and FE memory, a/k/L/q/rate at every Gauss8
point. Phase, reconstructed full-tensor stress means, phase weak forces and
boundary reactions also match bitwise after recovery, outer retry and restart.
The loaded physical exports are bitwise identical to the prior P4 loaded state.
Publication increments once per accepted step, including after two discarded
outer trials. Every immutable input file and stationary particle fingerprint
remains unchanged (110592 particles). Native maxima across the controls are
mechanics 1.52e-15, phase KKT 1.41e-10, source discrepancy 3.90e-15 and absolute
BE-ledger error 8.64e-16 J/m, against the original 1e-9/1e-7/1e-12 and scaled
1e-10 bounds. Independent small mixed-matrix checks recover curvatures -.25,
+.5 and zero, exclude artificial rows and reject an inconsistent cross tangent.

![Actual rejected and published states](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_stability/publication-recovery.png)

The predictor was imposed only as a test starting guess. The maps show its
calculated rejected phase increment, the calculated published zero increment,
and its admissible relaxed phase mode at identical physical scales. Contours
are phi=.01/.05/.1/.3; the dashed line marks the original seed endpoint x=1 m.
No histories from the rejected state are published. This does not test global
energetic branch selection or a free direction.

The first development gate safely rejected on an overly strict entrywise
cross-symmetry test and recovered. That result and binary remain preserved.
The corrected check uses the pre-existing independent analysis's norm-relative
1e-11 symmetry criterion, retains both actual cross actions, and measures their
2.03e-15 defect. No source, physical or FD tolerance was relaxed.

## Closeout and next scientific decision

The hold-instability diagnosis, implementation and lifecycle verification are
complete within this setting. Earlier temporal growth errors (8.302% L2 versus
5%), spatial interval errors (7.442% versus 5%), 57.26% boundary-induced growth
reduction, original energy-screen failures, strict_reference_match_passed=false
and BP3 evidence remain. No new mesh, quadrature or propagation qualification
is inferred. Uniform fixed meshes/same-rank restart are checked here; the new
gate has not been qualified for hanging-node evolution, larger critical sets,
heterogeneous coefficients, transport or changing direction. The earlier fresh
loading prefix and other loading changes were not rerun.

Recommend tensor cohesive memory with changing **flow** direction, including
already loaded material, for scientific review. This preserves old memory
instead of rotating a scalar correction. The first proposed P5 task is a small
material-point return/tangent/lifecycle qualification plus fixed-direction
identity, before any directional specimen. A subsequent specimen should be
declared a finite physical specimen; current boundary sensitivity does not
support a distant-boundary interpretation. P5 is proposed, not begun.

Changed files for this task (separate from inherited local changes):

- `source/simulator/solver/bulk_cohesive_phase.cc`: continuation, gate and retry.
- `include/aspect/material_model/phase_field_fault/bulk_stability.h`: relaxed
  critical-space calculation using existing native solvers.
- `benchmarks/reconstructed_fault/particle_free/CMakeLists.txt` and
  `bulk_cohesive/{parallel.cc,stability_checks.cc,stability_qualification.py}`:
  replayable lifecycle hook, independent checks and evidence verification.
- This report, `BULK_STABILITY_SELECTION.md`, `BULK_CONTINUATION_REPORT.md`,
  `SOURCE_AUDIT.md`, `STATUS.md` and `P5_DIRECTION_SPECIFICATION.md` in this
  directory. The [exact file/hash list](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_stability/changed-files.json)
  and compact evidence archive accompany it. Full native records/checkpoints
  remain in `build-particle-free/bulk-stability`; no BP3 package is replaced.
