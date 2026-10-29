# P4 continuation check — bounded completion, 21 October 2026

Follow-up: the [bounded stability remedy](BULK_STABILITY_REPORT.md) is now
implemented and qualified. It promotes loading-aware continuation and rejects
the demonstrated saddle before publication. The investigation is resolved in
that tested setting; all original P4 measurements/failures below remain. The
original report/specification snapshots are preserved in the P4 evidence archive.

The predictor's hold-growth root is locally unstable. Native accepted-state
continuation gives a stable no-growth hold at all three timesteps. The bounded
check is complete, but quantitative directional evolution is **not qualified**:
within-step temporal and interval spatial targets fail, and moving the right
boundary substantially changes the response. The [P5 specification](P5_DIRECTION_SPECIFICATION.md)
is prepared for scientific review; no direction, physical law or default changes.

Entry is f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
The retained bulk law has G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m2, strength drop
2000 Pa, residual resistance R=1000 Pa, ell=.046875 m, m=4 and Kf=8 Pa.
Unlimited DGQ2 plus native FE source memory, Gauss8/source3, fixed meshes,
stationary diagnostic particles, physical obstacles and native tolerances remain.
This bulk experiment is distinct from protected mature BP3/Gauss6 controls.

## What was imposed and calculated

Restart the immutable accepted step 127 at .15875 s and advance to .17875 s with
Δt=.00125/.000625/.0003125 s. The prescribed shear rate is .018/s until
.175 s, then displacement is held. Relative grip displacement ends at 2.25 mm.
All three sequences split exactly at the loading-to-hold transition. Native
end-step termination avoids an old microscopic end-time interval; its original
checkpoint/error remains preserved. Accepted clocks and old timesteps are not
edited; the existing restart signal reduces only the pending interval.

Each first-hold pair uses byte-identical incoming owners. The existing guess
extrapolates accepted velocity. Continuation instead starts with zero displacement
increment, beta-scaled pressure and accepted phase, then applies current boundary
constraints and solves the complete coupled equations. Phase and constitutive
variables remain available. Every trial reads the same accepted histories and
obstacle; publication occurs once after acceptance. This is a test-only starting
guess, not a phase freeze or a general stability-guaranteed branch selector.

## Local stability selects the tested hold continuation

With beta=1/(1+Δt G/eta), Tplus=beta(Tn+2G dev symgrad du), and fixed
Sxy=Syx=1/2, the actual return minimizes

    ||Tplus-2GaS||_3D^2/(4G) + G a^2/(2h) + R|a-beta an|.

Consequently Pi=Eb+Ec+Ef+beta*Δt*integral(R abs(w)) is an incremental potential for these
homogeneous matched coefficients. Its mechanical derivative is beta times the
physical residual; its phase derivative is physical. Weighting native mechanical/
pressure rows by beta*G and phase rows by Kf produces the symmetric tangent.
The phase Schur action eliminates mixed mechanics/pressure and the local return;
constraints are already condensed once. Artificial constrained-row diagonals
and pressure-saddle eigenvalues are excluded. Eigenvalues below use the positive
Kf-times-lumped-phase-mass metric and the admissible phase critical space.

| Δt [s] | Loaded minimum curvature | Predictor first-hold curvature | Predictor hold max dphi | Continuation hold dphi |
|---:|---:|---:|---:|---:|
| .00125 | +.139907 | -.208558 | .0158369 | 0 |
| .000625 | +.140005 | -.168661 | .0155861 | 0 |
| .0003125 | +.140077 | -.143308 | .0154274 | 0 |

The coarse predictor's frozen-mechanics phase block is positive (+6.37136),
so freezing mechanics would miss its instability. Its negative relaxed mode
admits bound-feasible perturbations in both signs. Minimum-mode residuals are
about 1e-12. Continuation has strictly positive normalized obstacle gaps
2.869e-6/1.434e-6/7.173e-7; the enlarged near-critical subspace at the smallest
step is positive too. Mechanics is coercive on the incompressible velocity space.

At the coarse first hold, Pi is 3.47532819005 J/m for continuation and
3.47532898851 J/m for the predictor. Predictor frictional expenditure is
.00534657 J/m; continuation has zero slip activity. Both pass KKT and ledger
checks. Negative curvature, rather than the small potential difference alone,
invalidates interpretation of that predictor root as a stable physical jump.
Local stability does not prove global minimality or exclude another finite jump.

For this homogeneous zero-increment hold, Maxwell scaling of stress and elastic
memory moves the return inside the friction threshold and reduces driving.
The native first-hold tensor/pressure identity agrees at roundoff. Later default
pressure reuse causes up to .0372 Pa pressure discrepancy before the unchanged
mechanical tolerance requests another solve; deviatoric discrepancies are at
most 6.27e-8 Pa. Phase/slip activity remain exactly zero. Holding displacement
has not removed stress. Other loading/material configurations can grow during a
hold; no general prohibition is imposed.

![Both native hold candidates](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation/first-hold-branches.png)

Colour shows actual phase change, physical shear-stress redistribution and weak
phase balance. Contours describe damage geometry, not auxiliary T. The original
cap support is marked. Mechanics and phase were solved together in both trials.

## Rapid loading is not a resolved propagation speed

| Δt [s] | Max loaded step dphi | Max native slip rate [1/s] | Max loaded .1-front step [mm] | Absolute physical-work defect [%] |
|---:|---:|---:|---:|---:|
| .00125 | .0856081 | .808882 | 77.606 | .0147508 |
| .000625 | .0581368 | 1.069845 | 53.401 | .00485105 |
| .0003125 | .0385435 | 1.397869 | 35.905 | .00157888 |

The maximum loaded increments shrink; a persistent loaded quasistatic jump is
not established. Rates and within-step paths remain unresolved. The predictor's
hold increment persists under halving, but belongs to an unstable root. The
selected continuation shows no hold advance at fixed imposed displacement.

At actual common times, final half/quarter interval-growth L2 differs by
3.638e-5; cell-mean full-3D stress differs by 2.064e-6 relatively. However, the
half-step physical-phase time interpolant misses a saved quarter-step state by
up to .0109288 in phase. Growth-to-date normalization reaches 8.302% L2 and
13.889% peak, failing the declared 5% target. The .1-front discrepancy is
7.654 mm, within one original cell; the weakest contour differs by 47.117 mm.
The maximum absolute discrepancy is 4.262% of final interval peak growth, a
supplementary scale, not a replacement for the failed criterion. No constitutive
history is evaluated on an interpolated state. Only the shared-parent interval
is temporally tested; its earlier loading prefix is not requalified.

![Fronts against time and displacement](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation/front-continuation.png)

Six contours separate weak process-zone advance from core growth. The original
.1 contour advances 215.04 mm from its represented initial position; .5 advances
only .927 mm. No detached components occur at the checked thresholds/states.
The [65-frame GIF](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation/continuation.gif)
shows actual quarter-step accepted states, physical stress and local slip-rate
activity on fixed scales. It does not interpolate a crack speed.

Physical work uses native boundary reactions and actual displacement increments:
Wtr=(Rn+Rn-1) Delta U/2. Summed absolute per-step residuals avoid cancellation.
At the half/quarter comparison C_E=.00157888%, C_T=.00347847%, and
B_E=C_E+C_T=.00505735% against the retained 1% work criterion. The exact BE
ledger is separate: stress, traction, signed compliance and fracture-gradient
remainders plus phase, mechanical and pressure residual-work terms are exported.
The smallest-step window has Wtr=.788556718 J/m, dE=.606178935 J/m and physical
dissipation .182390233 J/m. Phase/mechanical residual work is about 1e-12 J/m.
Small work defects also occur on the unstable predictor branch; they do not
qualify stability, spatial accuracy or continuum quadrature.

## One spatial and one boundary-distance control

Both changed meshes start freshly from the identical continuum seed table and
zero histories. Shared vertices are bitwise equal; no quadrature history is
remapped. Initial phase spans [0,.6] within physical [0,1] bounds, unloaded drift is zero and the minimum
scaled unloaded obstacle gap is .112446. The full saved-field scan corrects the
preliminary log-based refined-onset estimate.

| Mesh; width [m] | Δx / Δy [m] | Initial L2 error whole / cap [%] | First growth bracket [s] | Final .1 front x [m] | Integrated growth [m2] |
|---|---|---|---|---:|---:|
| 96x128; 1.5 | .015625 / .0078125 | .59196 / 2.05755 | (.13375,.13500] | 1.276461 | .00410291 |
| 192x128; 1.5 | .0078125 / .0078125 | .50001 / .75911 | (.12750,.12875] | 1.277998 | .00414355 |
| 112x128; 1.75 | .015625 / .0078125 | .59196 / 2.05755 | (.13375,.13500] | 1.165361 | .00175375 |

All use Δt=.00125 s and identical strain/displacement histories. Common-time
state differences in the shared-parent timestep check are small compared with
these changes; continuous-path and prefix uncertainty remain. Refinement is
along growth only, not a normal-refinement or convergence-order qualification.

The refined final growth L2/peak differences are 1.328%/3.869%, but interval
maxima are 7.442%/9.020%, with integrated-growth differences up to 5.413%.
Thus the declared spatial target fails. Final cell-mean full-tensor stress
changes .269%; the primary front differs at most 1.537 mm.

Moving the right boundary weakens growth substantially: common-domain growth
L2 differs by 94.51% of the enlarged-case growth norm, integrated growth falls
57.26%, and the .1 front differs by 111.10 mm. Added fracture storage falls
.0366368 to .0144925 J/m. There is no damage in the added strip. Raw reaction
rises 2804.67 to 3406.62 N/m because the grips are wider; reaction/width is
1869.78 versus 1946.64 Pa. Subtracting the intact Maxwell background gives
seed-excess reactions -568.08 versus -528.25 N/m. This is boundary sensitivity,
not a domain-independent result. Loaded relaxed curvatures are +.120646 on the
refined mesh and +2.34054 on the enlarged domain; both holds are locally stable
with zero phase/slip activity.

![Common-scale physical fields](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation/spatial-domain-fields.png)

The top row shows the imposed initial damage; lower rows calculate growth and
physical cell-mean shear stress at .175 s. White contours share levels/scales.
Blank area lies outside the smaller specimens. No activation region is gated.
The pronounced domain effect exceeds the final mesh difference. Companion
profile/response PNG and CSV use common physical kernels and compatible weak
force averaging; forces are not inferred from interpolated local histories.

## Source corrections, evidence and readiness

Only diagnostic/initial-guess code changes in the bulk solver and its benchmark
observers. Point law and history-owner sources are unchanged. Full-space FD
failures .0260832, 1.14085e-5 and 5.74934e-6 remain recorded against 1e-6. Tiny-phase
Coulomb branch crossing is reproduced independently; the complete critical
P^T J P check passes without changing either FD or physical tolerances. The
initial incomplete-output restriction failure also remains. See the [source audit](SOURCE_AUDIT.md)
and the archived point/action traces for the exact scope.

The first enlarged run exposed a benchmark hard-coded x=1.5 boundary diagnostic.
Its false .165726 N/m norm matches the missing-cell analytic calculation to
4.49e-13 relatively. Native boundary-face DoFs replace that coordinate test.
The same coarse loaded replay is bitwise identical in all four exports; 124
physical files of the enlarged prefix also match. Actual first-load boundary
residual is 1.29e-19 against the unchanged 1e-9 bound. The failed run/DSO remain;
this post-publication observer failure is not an outer-loop rejection.

All completed positive-step native mechanical/KKT/source/ledger, ownership,
publication and same-rank restart checks pass at existing tolerances. Frozen
particle ID/coordinate fingerprints remain unchanged. Four-rank fresh controls
cost 3984+59 s (refined load/hold) and 1898+26 s (enlarged); these are local costs,
not performance or full-event guarantees. Full solver/checkpoint records remain
separate from the compact PNG/GIF/CSV/JSON package, with manifests/checksums.

The local hold branch is understood within this scope. The tested temporal,
spatial and boundary findings do **not** support quantitative propagation or
automatic directional adoption. Native Gauss8 is unchanged; Q1 fracture
integration is exact on these affine cells, but this task adds no independent
nonlinear constitutive-quadrature bound for the new specimen. Global branch
selection, earlier-prefix error, normal resolution and further boundary
independence remain open. Historical 1% endpoint-screen failures, prior spatial
and branch errors, strict_reference_match_passed=false and BP3 are preserved.
P5 now has an explicit tensor-memory/free-flow proposal and a distinct virgin-
locking alternative. Selecting that constitutive choice and its branch rules
must precede implementation. No P5 run is begun.
