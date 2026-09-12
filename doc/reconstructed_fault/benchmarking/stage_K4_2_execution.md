# K4.2 approved post-transient production comparison

The user approved the scope refinement in
`stage_K4_post_transient_recommendation.md`. Assess finite-width accuracy on
[4,6] s using dt=.125 s and the saved .125/.0625 independent references.
K4.1 is **not** an all-time pass. Keep the early transient, its physical
influence, and all convergence/history/support checks from t=0. No histories
are restarted or reset at 4 s. No acceptance coefficient changes.

## Prepared cases and discriminating purpose

| Case | ell (m) | Bulk mesh | Fault elements | Purpose |
| --- | ---: | --- | ---: | --- |
| A | .15625 | 32x256 | 32 | Resolved-width production baseline |
| B | .078125 | 32x256 | 32 | Width difference on the same bulk mesh |
| C | .078125 | 32x512 | 32 | Distinguish half-width under-resolution from width physics |

All start from prescribed physical initial data independently. The K1 plugin
freezes converged initial Q1 phi through constraints on phase DoFs only; the
material's established `Evolve phase field=false` retains H. Mechanics,
Theta, cohesive traction and Maxwell histories otherwise advance normally.
Activation is explicitly .1. All material constants, K1 loading, pressure
treatment, particle density, support policy, full Ih, open topology and
solver settings are inherited unchanged. Width-dependent calibration,
support distance and initialization are recomputed, not transferred.

Expected one-rank Release resources from saved K3 timings: 6--12 minutes and
1--1.5 GiB per 256-normal case; 12--25 minutes and 2--3 GiB for C. Caps are
1200/1200/2400 s with no automatic retry. Actual timing/RSS and binary/plugin
hashes are recorded by `finite-width/production.py`. No separate performance
campaign. The cases are sequential; a failed physical/numerical guard stops
the sequence for review.

The first A attempt stopped at t=.125 s solely because the new diagnostic
mistakenly imposed a 1e-10 total particle-measure threshold. Its measured
relative difference was 1.00e-9; the saved passing K3 256-normal case also
exceeds that unapproved threshold (7.50e-10). Existing checks require positive
domains and report total measure. Restore those semantics, not a fitted new
threshold. The complete stopped attempt is retained under
`finite-width/attempt1-volume-guard/`; all original physical/convergence
checks passed. Two focused diagnostic tests cover this evidence and exact
Q2 profile reconstruction. A single explicit rerun uses unchanged parameters
and production binary; only the Python guard is corrected.

An opt-in benchmark-only export hook runs a read-only K4 guard after each
accepted state, before advancing. It verifies geometry, phase admissibility,
sequence, actual support/normalization across all exported x columns,
particle measure, weak surface balance and history updates. The existing
plugin requires genuine nonlinear success. Fresh linear residuals remain
checked in production and recorded in the run log. Neither production code
nor physical/numerical settings are changed by this hook.

The criterion remains the approved K4.2 conservative error accounting and
factor-four signal requirement. No effect is declared resolved merely because
a run exits successfully. Stop on an unexplained B--C plateau or inadequate
uncertainty; do not automatically add another mesh/fault level.

## A completed (before launching B)

A reaches all 49 accepted samples through 6 s; all guards and all eight
post-transient K1 observable checks pass. All 158 returned linear directions
pass the fresh requested residual check (largest fresh/target .990924).
Maximum [4,6] errors: actual weak q .219720 Pa, C .218789 Pa, Theta .023970 s,
slip 3.16736e-7 m, V 8.83956e-9 m/s, raw stress .230685 Pa and velocity
3.71294e-9 m/s. Initialized C0 differs by -.229713 Pa and Ih by +.00986768 m;
these errors are retained in the comparison, not subtracted.

Runtime is 966.847 s, peak RSS 1112812 KiB. This exceeds the preliminary
estimate but stays inside its 1200-s cap. Before B, update the working cost
estimate to roughly 16 minutes for B and 30--38 minutes for C, without
changing their 1200/2400-s caps. B/C remain scientifically necessary to
separate the width difference from half-width normal resolution.

## B completed (before launching C)

B reaches 6 s with all 49 state guards and 159 fresh-linear checks passing
(max fresh/target .988580). It takes 798.405 s and 869532 KiB. Containment
and normalization remain below 1e-4. All post-transient observable checks pass
except cohesive traction: max C error .991835 Pa, versus A's .218789 Pa.
This holds against both .125 and .0625 references. B's C0 error is +1.03826 Pa,
so C is needed to test normal-resolution contraction without changing initial
physical data. Raw stress error is 1.111996 Pa and velocity error 1.05657e-8 m/s.
Launch the already-planned C, not an additional mesh or solver correction.

## Execution closed

C completes in 1941.790 s (1581112 KiB), inside its 2400-s cap. All 49 guards
and 159 fresh-linear checks pass. Its cohesive error is .246654 Pa, a 4.02x
contraction from B, and it meets every post-transient K1 observable check
against both reference timesteps. The completed attribution/uncertainty
report is `stage_K4_2_result.md`; no further simulation was launched.
