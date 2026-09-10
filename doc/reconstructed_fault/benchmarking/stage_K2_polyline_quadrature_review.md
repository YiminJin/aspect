# Polyline domain quadrature: bounded verification record

2026-09-09. Nonlinear verification hold; not a completed K2.2/Gate K2 result.
The geometric addendum was written before production changes:
`stage_K2_polyline_quadrature_addendum.md`. It extends the approved
domain-integrated discrete rule to actual reconstructed open 2-D polylines.

## Invariants and implementation boundary

The parent particle retains its admission, assigned fault and entire existing
domain measure. Bulk FE and particle-history inputs remain parent-P0. Surface
fields and the one-sided segment frame are evaluated at each integration
coordinate. Corner/tip continuation, finite-segment selection and the explicit
departure from the old width-filtered point convention are defined in the
addendum; no reconstruction is flattened. The straight fast path retains its
original roundoff criterion and is not used to approximate real bends.

Complementary convex partitions feed one cached geometric rule. First moments
assemble loads/support; second moments assemble mass and Jacobians. Nonlinear
responses are evaluated at each quadrature coordinate, not at an averaged
coordinate. Projections, R_Gamma, K_V, G, norms and diagnostic weak loads share
this rule. The cache is frozen during mechanics and invalidated by particle
domain or reconstructed-geometry changes. A rank-local geometry failure is
communicated before collective matrix reductions.

The installed Voro++ probe verifies assignment-copy independence, the factor
two in plane offsets, oblique/complementary cuts and tiny cuts. A 1e-13 sliver
collapses under that backend's tolerance. Existing 2-D polygons are therefore
split directly with shared edge intersections; no temporary-prism dependency
or live CPDI mutation is needed. Particle-domain construction, periodic flags,
history transfer, support, full I_h, parameters and nonlinear settings are
unchanged.

## Focused evidence

All artifacts below are relative to
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/polyline-quadrature/`.

| Check | Result | Artifact |
|---|---|---|
| Debug/Release builds, `cmake --build build-pf-cpdi -j4` | Pass | `build.log` |
| Benchmark plugin Debug/Release build, `-j4` | Pass | `plugin-build.log` |
| Installed Voro++ cut/copy probe | Pass; documented tiny-cut difference | `voro-plane-cuts.log` |
| Geometry plus Stage-I unit checks, one rank | 815 assertions / 18 cases pass | `unit-one.log` |
| Same checks, two ranks | 815 assertions / 18 cases pass on each rank | `unit-two.log` |
| First formerly blocked condensed-adiabatic fixture | Pass, 182.71 s | `first-coupling.log` |
| Python nonuniform diagnostics | 10 tests pass, 8.613 s | `analysis-tests.log` |
| Python homogeneous reference/analysis | 15 tests pass, .029 s | `reference-tests.log` |
| Selected surface/coupled/lifecycle fixtures | 14/19 pass; three timeouts, one stale golden output, one invalid stale-reference comparison; 2204.97 s | `focused-tests.log` |
| Isolated dynamic-pressure Stage-I fixture | Nonlinear iteration exhaustion; process exit 0 follows configured continue policy, **not convergence** | `stage_i_isolated.log`, `stage_i_isolated.resources.json` |
| Fresh sequential two-rank restart pair | 2/2 pass; create 340.34 s, resume 119.93 s, total 460.30 s | `restart-isolated.log`, `restart-create-passed.log`, `restart-resume-passed.log` |

The geometry tests include the captured eight-vertex near-straight fault,
genuine right-angle bends, overlapping finite-projection slabs, corner gaps,
true tips, node crossings, straight-limit recovery and 1e-13 tip cuts.
Analytic first/second moments are checked independently. Separate nonlinear
log-response tests compare orders 3/5/7; these are not claims that all possible
constitutive profiles are resolved by order 3. Existing production K_V/G
finite differences differentiate the actual fixed-quadrature residual.

The timeout log (`restart-create-timeout.log`) records a genuinely converged
first real timestep, lifecycle/feedback assertions and a new restart snapshot;
the run was advancing timestep 2 when killed by the wall-time budget. No
numerical tolerances, iteration budgets or expected outputs were changed.
An isolated rerun is required before counting restart-create as passed.

The first dependent resume restores the checkpoint and completes second-step
feedback/failure-preservation assertions, then compares with a stale final
reference: `stage-j-final-state.txt` still has modification time
2026-09-08 23:52:11 -0700, preceding this implementation. Its mismatch is not
valid evidence of a restart algorithm defect. Require a fresh completed pair.

That fresh pair subsequently **passes** with the original 600 s per-test cap,
running sequentially. It compares a newly completed uninterrupted trajectory
against the resumed trajectory, including histories, V, geometry and bulk.
The creator includes the ordinary Stage-J fixture and its timestep-zero,
two-real-step feedback and one-owner failed-history-candidate assertions.
Those assertions therefore have successful fresh coverage, although the
separate ordinary Stage-J CTest timeout was not rerun redundantly. The passing
final-state reference is retained as `restart-final-state.txt`.

The rate-and-state fixture completes its numerical solve and verification
postprocessor but fails the old golden-output comparison. That file still
contains the former ten-iteration, unconverged phase initialization and lacks
the current linear/nonlinear diagnostics. It is not silently refreshed or
counted as a harness pass. The dynamic-pressure Stage-I fixture is a separate
issue: its concurrent run reaches a roughly 5e-10 bulk residual against a
5.889e-13 target before timeout. Its zero base iterate gives zero under the
existing matrix/state precision rule. This does not establish the cause of
the remaining residual or authorize a new allowance; an isolated replay must
distinguish actual nonlinear failure from the wall-time cap.

All seven selected surface/condensed fixtures pass (surface adiabatic,
dynamic, rate-dependent and singular; condensed adiabatic one/two ranks and
dynamic). Changed-loading, temperature lifecycle and rollback each pass on
one/two ranks; linear exhaustion passes. No old expected outputs or physical
parameters have been changed to obtain these results.

The isolated dynamic-pressure replay uses `stage_i_isolated.prm`, which
includes the actual checked-in fixture, changing only library/output paths.
It has a 1200 s diagnostic wall-time bound, not a changed nonlinear budget.
Through iteration 4 it reproduces the concurrent trajectory: bulk residual
4.91662864451368e-10, velocity contribution essentially identical, scaled
continuity 1.81373964079018e-17, fixed bulk scale 5.88893781023253e-7, target
5.88893781023253e-13 and zero matrix/initial-state precision allowance. The
fresh linear residual is 2.49199525023638e-17 against 4.91662864451368e-17.
This isolates a small-bulk-residual issue; it does not yet prove assembly
cancellation versus represented-increment effects or justify a new stopping
criterion. The geometry/action checks above must not be substituted for that
small-direction diagnosis.

The isolated replay finishes after **723.85 s / 648188 KiB peak RSS**, not a
wall timeout. It exhausts the unchanged 12 nonlinear iterations. Last reported
residuals are bulk/velocity 3.47630767714197e-10, scaled continuity
8.09452794089071e-18 and projected surface zero. The fresh linear residual is
3.17288069515664e-17 (requested 3.47630767714197e-17). The fixed bulk target
remains 5.88893781023253e-13, about 590 times smaller than the last reported
bulk residual. The log explicitly warns of nonlinear failure. The configured
default continue policy produces process exit zero and the lifecycle
postprocessor's `verified` line; neither certifies convergence. Source review
confirms the coupled solver rolls back bulk/V and does not publish histories
before this policy is applied. Do not refresh a golden output to accept this
warning as the successful baseline.

**Smallest next action:** a separate bounded bulk residual-consistency audit
at this small-forcing linearization: compare the predicted affine residual
change with fresh assembly using the actually represented constrained bulk
increment, separating velocity/continuity and the frozen stress load. The
present evidence is consistent with cancellation, but does not prove its
location or exclude a small-direction inconsistency. Prefer a local evaluation
correction if demonstrated; do not adjust the geometry, V_min, tolerance or
precision allowance to hide this gate. No such production change is part of
this patch.

## Trajectory gate and resources

Short K1 and unchanged K2-64/K2-128 through 1 s were **not run** because the
nonlinear integration gate is unresolved. Inputs change only end time/output
directory. Preliminary prospective
budgets are 2--5 minutes for K1 and 5--15 minutes for K2-64; refine the K2-128
estimate from measured cost before starting it. Do not count saved point-rule
trajectory evidence as verification of this revised discrete rule.

Measure the actual accepted weak traction using `surface_weak_*.csv` from the
frozen, pre-history-publication linearization. Keep raw published parent stress
and bulk FE transfer separate. Compare changed initial projections explicitly;
initialize the independent homogeneous reference once and never reset it from
later output. Report support and actual slip normalization independently.
The provisional family omitted-fraction allowance remains 1e-4 (original
1e-6), separate from the unchanged 1e-4 actual slip-normalization requirement.

The expensive temporal campaign, true-normal-stress branch and full ASPECT
suite remain held. The previously documented bulk-history transfer concern is
not modified by this correction.

## Reproduction and handoff

From the repository root:

```sh
build-pf-cpdi/aspect --test '[fault_domain_quadrature],Stage-I*'
mpirun -np 2 build-pf-cpdi/aspect --test '[fault_domain_quadrature],Stage-I*'
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_(surface_(adiabatic_pressure|dynamic_pressure|rate_dependent|singular)|condensed_(adiabatic(_mpi)?|dynamic)|stage_i(_rate_state|_rollback(_mpi)?)?|stage_j(_temperature(_mpi)?|_restart_(create|resume))?|changed_loading(_mpi)?|linear_exhaustion)$' -j2
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_stage_j_restart_(create|resume)$' -j1
timeout 1200 build-pf-cpdi/aspect \
  benchmarks/reconstructed_fault/uniform_shear/nonuniform/polyline-quadrature/stage_i_isolated.prm
```

The polyline implementation itself is in `source/reconstructed_fault/utilities.cc`;
the manager assembles cached domain moments and propagates geometry failures
before MPI sums; surface assembly uses each integration point's own segment
frame. `unit_tests/reconstructed_fault.cc` contains the new geometric and
nonlinear accuracy cases. The installed-library probe is
`nonuniform/endpoint/verify_plane_cuts.cc`. The earlier approved straight-rule
changes to polygon retention, projections, point responses and weak diagnostics
are preserved. No production solver or history-transfer file was changed in
this extension. A final indentation-only cleanup of the new manager try-loop
does not change the tested tokens/behavior. `git diff --check` passes.

This is an uncommitted implementation with a documented verification hold,
not a finalized correctness baseline. Preserve the current tree and the
successful evidence when addressing the separate small-bulk-residual issue.
No corrected-trajectory initial projections, raw stresses, weak endpoint
tractions or support/normalization results exist yet for the revised rule.
