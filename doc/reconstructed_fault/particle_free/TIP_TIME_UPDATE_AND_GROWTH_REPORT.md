# Coupled midpoint and compact-loading study — 9 November 2026

**Completed bounded study; retain BE as default.** The selectable midpoint candidate is implemented and passes
point, small coupled, tangent/direct-reference and active restart checks. It has
not completed the difficult common-parent window, so it is not adopted as the
replacement for BE. The independent compact-loading screen and BE pilot both
identify boundary-dominated loading, with strong cancellation of section slip.
Neither the method nor this specimen is ready for a longer propagation campaign.
All jobs have finished. No production campaign is authorized by this report.

![Time study and loading-screen summary](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/time-update-summary.png)

[PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/time-update-summary.pdf)
and [plot generator](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/plot_time_update.py).

## Source and method

Entry: `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, `e8b6dd0912640433676a69089d2b2ad17701bcbf`.
The incoming eleven modified tip documents were preserved in `entry.patch` and
`entry-source.tar.xz`, alongside the actual source/binary identities in
[`tip_time_update/entry.json`](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/entry.json).
Protected final BE core `76e9c5c1…5641a` and its checker were copied to
`build-particle-free/bulk-tip/controls/native-bernstein-final/`; initial native,
nodal efficiency and mature reference identities were checked against the
native-Bernstein closeout. No checkout, reset, production restart or server job.

[Time integration](tip_treatment/time_integration.md) is the authoritative
continuous-equation → stage/endpoint → residual/tangent → work derivation.
`Bulk tip time update = midpoint` selects h=Δt/2 stage coordinates and exact
constant-stage-rate aging, while BE remains default. The endpoint obstacle is
imposed through the equivalent stage upper bound (1+b_n)/2. Only endpoints
publish; last-stage rates/reactions retain their explicit observation time.
Method metadata guards ordinary restart independently of the physical-law
archive. Prescribed compression, constitutive parameters, physical energy,
Bernstein FE, quadrature, endpoint bounds and residual gates are unchanged.

The only scoped nonlinear completion tests a pressure-only correction when the
inexact full saddle direction is rejected as uphill. Pressure does not enter
the primal functional or prescribed-compression law. It can therefore change
without a primal energy descent, but acceptance requires **both original full
mechanics/KKT gates**. Velocity, phase and all material histories stay fixed.
No tolerance, search shift or radiation damping is adjusted.

## Independent local and coupled checks

The point suite passes **6078 assertions in 21 cases**. Added checks directly
substitute the recovered endpoints into the midpoint equations, cover both
shear signs, intact birth, full damage, reversal and nonnegative viscous losses,
and compare stress/cohesive state against independent analytic smooth solutions.
Successive analytic Maxwell error ratios exceed 3.9; smooth prescribed-phase
cohesive state ratios exceed 3.8. The latter is not a coupled growth solution.
The nonlinear cohesive chain defect scales quadratically when accumulated over
smooth prescribed histories. The stiff Maxwell counterexample changes sign for
z>2. The integrated candidate rejects ΔtG/eta>2 without a BE fallback.

Both signed aging laws share the friction/root machinery. Independent aging
quadrature, near-zero and signed derivatives, the condensed section derivative
and the primitive derivative pass. Exact-stage aging uses its own sufficient
monotonicity condition eta_rad≥p b h/Dc.

The small active midpoint oracle has 306 free unknowns: relative Krylov/direct
error 2.525e-8, weighted symmetry error 5.205e-17; tangent action error 2.961e-10,
phase functional derivative error 5.605e-9, Lagrangian derivative error 1.383e-8.
The first accepted refined quarter-step tangent independently gives 1.002e-8.

The short BE control reproduces every common exported point/section value of
the retained native BE run **bitwise**. An active midpoint 1→2-rank restart at
accepted .075 → .08 passes unchanged-parent, rollback, deterministic retry and
single-publication checks. Compared with uninterrupted serial: max phase
2.94e-15, shear stress 1.36e-12 Pa, age 1.78e-15 s, slip 4.34e-19 m, resistance
2.27e-13 Pa. An initial checker crash was fixed: the first-pending-step rejection
probe ran before ASPECT allocated its restarted matrices. The checker now
prepares the existing matrix owners before invoking the trial. This did not
change the production lifecycle.

The final-source MPI/rejection check reproduces these tolerances. An ordinary
BE restart of its midpoint checkpoint rejects the numerical-method mismatch.
The one-step layered mature control reproduces the retained physical fields,
surface histories, background and publication counts **byte for byte**; only
peak memory differs in the storage diagnostics. Ordinary no-fault depth-average
and statistics files are byte-identical. See
[final smoke comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/final-smoke.json).

## Difficult common-parent window: qualification failure

All paths import the retained accepted step23 at **.057500000000000016 s**;
intended common endpoint is **.060000000000000019 s**. No trajectory replay.
The BE full/half references remain the completed efficiency/native evidence.
The new candidate's initial results are:

| Physical dt | Accepted substeps | Last accepted time | Failure at 80 iterations | Wall s |
|---|---:|---:|---|---:|
| .0025 | 0/1 | .0575 | mechanics 2.51e-15; KKT 6.34003e-7 | 109.8 |
| .00125 | 0/2 | .0575 | mechanics 1.62e-15; KKT 1.03390e-7 | 261.4 |
| .000625 | 2/4 | .05875 | third substep mechanics 1.01e-10; KKT 8.94164e-5 | 136.4 |

The mechanics/KKT gates remain 1e-9/1e-7. The near miss at half dt is a failure,
not a tolerated endpoint. The quarter-step path has good accepted-step work
accounts but no completed .06 state. Thus no full-window state convergence,
10%-of-added-fracture target, or cost-at-comparable-accuracy is established.
A further refinement cannot serve as a convergence reference for these failed
paths; no extra finer run is selected. This is a nonlinear qualification limit,
not evidence that the local midpoint algebra is wrong. The failed attempts and
accepted partial checkpoints remain available.

After the multiplier-only correction, `tip-time-parent-final01` repeats the
full-step failure at exactly the same KKT 6.34003e-7 (102.97 s). The correction
does not remove this qualification failure. Retained BE full/half work remainders
are .00174783606 / .000874875585 J/m, or 10.25 / 5.17 times added fracture. There
is no accepted midpoint full-window state with which to claim an improvement
over those figures. The two accepted quarter substeps have cohesive defects
about 2.90e-9 and 3.18e-9 J/m and ledger errors around 1e-15; partial success is
not a completed window.

## Old-loading localized growth: positive but bounded evidence

Identical native Bernstein seed/loading, from rest to .09 s:

| Quantity | dt=.005, 18 steps | dt=.0025, 36 steps |
|---|---:|---:|
| Wall s | 16.10 | 25.71 |
| Added fracture J/m | 7.355745e-5 | 7.455128e-5 |
| Signed physical remainder J/m | 1.294130e-6 | 3.175507e-7 |
| Accumulated absolute remainder J/m | 1.294130e-6 | 3.175585e-7 |
| Remainder / added fracture | 1.759% | .426% |
| Maximum ledger error J/m | 3.90e-14 | 4.21e-14 |
| Nonlinear corrections / outer Krylov iterations | 17 / 652 | 31 / 1144 |
| Stokes inverse calls | 670 | 1180 |
| Stage bulk viscous loss J/m | .001850269 | .001851221 |
| Stage cohesive viscous loss J/m | 1.719266e-6 | 1.731647e-6 |
| Section friction plus radiation J/m | .302283256 | .302234985 |
| Friction alone J/m (signed = absolute) | .297867252 | .297846398 |
| Radiation alone J/m (signed = absolute to roundoff) | .004416004 | .004388588 |
| Signed / absolute cohesive chain defect J/m | 1.211510e-6 / 1.211510e-6 | 3.029790e-7 / 3.029790e-7 |
| Signed / absolute mechanical residual work J/m | -8.261954e-8 / 8.261954e-8 | -1.457168e-8 / 1.457943e-8 |

Halving dt reduces physical remainder 4.075×. Endpoint differences are max phase
8.40e-6, shear-stress weighted L2 .01794 Pa m, A_xy L2 2.068e-8 m,
slip L2 2.131e-8 m^(3/2), and age L2 2.087e-5 s sqrt(m). These two grids through
changing active sets do **not** establish smooth second order for coupled growth.

An independent endpoint diagnostic holds the last stage V fixed and evaluates
R(V,theta_1), never advancing age again. Positive-phase endpoint force mismatch
falls .785→.279 Pa; conditional endpoint phase natural-map residual falls
5.144e-4→1.411e-4. These are temporal diagnostics, not extra acceptance gates or
an assertion that the unknown instantaneous endpoint velocity equals V_m.
Independent Bernstein reconstruction/stage-load checks are below 5e-16.

At dt=.005, actual SIPG weak work partitions 7.167308e-5 J/m to the predeclared
tip neighborhoods and 1.884377e-6 outside them; the global partition discrepancy
is below 3.2e-14 J/m. Fixed phase contours .001/.01/.1/.5 show **no resolved axial
advance**. A smaller energy remainder does not turn this into propagation
qualification. See the [sampled fields](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/old-loading-midpoint.png)
and [time/state data](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/old-loading-time.json).

## Compact-loading screen and pilot

Regions were fixed before inspecting results in
[`screen-regions.json`](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/screen-regions.json).
Tip neighborhoods are |x−1.2|≤.375 or |x−2.8|≤.375, with |y−1|≤.375;
seeded interior is 1.575<x<2.425 within that transverse strip. Remote means
|x−2|>1.3; boundary strips are within .25 m of any wall. These diagnostic regions
overlap. The **tip + outside-tip** weak-work partition is exhaustive and disjoint.

The fixed-phase hook now uses the actual integrated assembler-setup signal and
checks exact phase invariance. The first wrongly hooked attempt is retained and
excluded from screen evidence. Boundary nodal reactions and stage normal stress
use the solved pressure gauge; prescribed frictional compression is a distinct
input and is not silently added to mechanical pressure.

All screens use the 4×2 m domain, h=ell=.25 m, six-point cell quadrature,
native Bernstein phase, unchanged bulk/section physics, BE dt=.02 through .12 s.
Both intact and equilibrated-seed screens preserve their phase exactly. The
first compact seed screen required the multiplier-only completion described
above; its failed predecessor and diagnostic replay remain available.

| Fixed-phase screen at .12 s | Tip max Y/Kf | Remote max Y/Kf | Boundary max Y/Kf |
|---|---:|---:|---:|
| Width 1.2, intact | .419 | .344 | 9.525 |
| Width 1.2, seed | .020 | .341 | 10.091 |
| Width 2.0, intact | .235 | 1.287 | 4.026 |
| Width 2.0, seed | .042 | 1.265 | 2.483 |

The width-1.2 seed reaches boundary Y/Kf=1.122 already at .04 s, while tip
driving is .002. The **single allowed width change**, recorded before running,
widens support to 2 m: center boundary curvature falls 64%, and the imposed
profile at the nominal tip rises from .3086 to .7056; side velocities remain
zero. It reduces the wall peak but does not produce a tip-focused specimen.
At .06/.08 s the wider seed boundary ratio is .621/1.104, with tip .011/.019.
The intact response confirms that the wall concentration is not created solely
by the seed. These diagnostic ratios do not replace the phase VI.

Full spatial stress, dynamic normal stress, pressure, section rate/resistance
and nodal boundary reactions are retained in
[width 1.2 data](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/screen-width12.json)
and [width 2 data](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/screen-width20.json).
Reaction work independently matches the imposed-velocity work. Corners belong
to the constrained nodal reaction; no unique top/side continuum traction split
is claimed. Prescribed friction compression remains 1000 Pa; reported mechanical
normal stresses use the solved dynamic-pressure gauge, not that imposed value.

For the fixed seed at .12 s, width 1.2→2 reduces boundary max |tau_xy| from
3081→1999 Pa and max dynamic |normal stress| from 5697→5296 Pa. Section max
|V| falls 1.666e-4→9.103e-9 m/s, while max resistance is 574→426 Pa. The
dramatic integrated-rate reduction does not imply vanishing local fault
deformation; the pilot below resolves its sign cancellation. For the intact
screens at this time, boundary shear is 3086/2006 Pa and dynamic normal
5696/5310 Pa; the intact constraint has zero integrated fault slip.

### One pilot: boundary damage, no resolved axial propagation

The pre-recorded [pilot choice](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/pilot-choice.json)
uses width 2, amplitude factor 1.5, BE dt=.0025 and at most 24 steps to .06 s.
It starts from a fresh equilibrated phase seed, zero tau/A, age 10 s and zero
slip, not a checkpoint under another loading history. The amplitude brings the
predicted boundary onset into the bounded window. BE's timestep has retained
solver evidence, **not** adequate fracture-work accuracy. The old remote .95
stop is retained and an explicit previously-intact boundary-damage stop is added.

`tip-time-pilot02` stops at step23, **.057500000000000016 s**, in 18.41 s wall
time. Boundary phase first reaches 6.45542e-6, with boundary Y/Kf=1.28265;
remote Y/Kf=.65421 and remote phase zero. Tip Y/Kf is only .02192. Maximum
incremental phase is .00036416 near the tips and .00014610 in the seeded interior.

![Native phase contours on the actual cells](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/tip-time-pilot02-native-contours.png)

The [native contour generator](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/native_tip_contours.py)
evaluates each DGQ2 polynomial separately, displaying the .25 m mesh; its 25
plot samples per cell do not constitute mesh refinement. Exact quadratic axial
roots on both traces of y=1 show a .001 contour extension of **4.19653e-5 m**
at each end, only .000168h. The .01/.1/.5 axial roots are unchanged. This is a
weak diffuse halo and seed thickening, not resolved propagation. Incremental
damage and sampled stage Y are shown in the
[pilot field plot](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/compact-pilot.png).

Actual SIPG weak fracture work is 1.415495e-5 J/m globally: tip 5.240013e-6,
outside-tip 8.914937e-6; their sum differs from the global change by −5.23e-15.
The overlapping boundary strip contributes 8.229375e-6 and seeded interior
6.855619e-7 J/m. Thus boundary work is already the larger contribution. The
physical work remainder is **.08689265 J/m**, 6138.68 times added fracture:
BE stress loss .08679022 dominates. Its signed and accumulated absolute values
coincide. Ledger error stays below 7.74e-14, mechanics below 3.23e-11 and KKT
below 2.25e-8; those solver passes do not establish physical accuracy.

The closed-return-flow response also exposes the section closure's limitation.
Final max |V| is 3.14e-12 m/s and accumulated |slip| 1.33e-14 m, while resistance
reaches 306.4 Pa and cohesive energy is .00236452 J/m. Opposite-sign local fault
rates cancel: final sum ds(integral |w|dn−|V|)=.00393464 m²/s. Signed local
friction work accumulates **+.0166317930813 and −.0166317930789 J/m**, giving
net 2.40123e-12 J/m. Twelve sections show multiple active bands (maximum three,
diagnostic threshold max(1e-10,1e-6 max|w|)). No clipping or extra localization
is used. The section closure has nonnegative integrated friction work but does
not enforce nonnegative local f*w everywhere. Bulk cohesive history and energy
remain present. Age reaches 10.0575 s because the integrated rates are tiny.
Integrated radiation work is only 1.34e-23 J/m; the independent friction/radiation
split uses actual section weights and closes to the exported total within
1.29e-18 J/m per pilot step (8.33e-17 in the old-loading control).
This is further reason to reject the specimen as an RSF propagation control.

### Reproducibility and cost

[`closeout.json`](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_time_update/closeout.json)
records final source/core/checker identities, protected-reference rechecks,
exact commands, per-run outcomes and input/checkpoint hashes. Run inputs,
`command.json`, logs, exports and snapshots remain under
`build-particle-free/bulk-tip/runs/tip-time-*`. The retained failed setup attempts
are excluded from scientific comparisons: wrong fixed-phase signal, the
pre-allocation rejection probe, and a sandbox MPI startup failure. Original
nonlinear failures are retained, not relabeled as infrastructure failures.

The 24 bounded wrappers total **841.96 s (14.03 min)**, including failed
attempts and startup; point tests are additional inexpensive local checks.
Builds ran sequentially with at most four workers; simulations used one or two
MPI ranks and one numerical thread per rank. No heavy build overlapped a run.
No comparable-accuracy cost advantage over BE has been established. The
localized half-dt control costs 1.60× the coarse run, but neither supplies the
missing difficult-window endpoint.

To regenerate analysis without simulations, run `plot_time_update.py` and
`native_tip_contours.py tip-time-pilot02` under `benchmarks/.../bulk_tip/`.
The companion `analyze_time_update.py` and `endpoint_time_diagnostic.py` retain
the numerical reductions, exact spatial weak-work partition and conditional
endpoint diagnostic.

## Remaining boundary

BE stays default. The candidate remains experimental until the difficult-window
nonlinear solve and completed state accuracy are established. No general moving
growth, turning, branching, AMR, multiple-root RSF, stiff-crust or instability
qualification. The [cleanup inventory](TIP_DEFERRED_CLEANUP.md) lists concrete
callers and retained-test dependencies for a subsequent focused consolidation.
The next justified numerical task is a **targeted first-failure active-row and
search diagnosis** at the same retained .0575 parent, before spending on another
propagation trajectory. Preserve the independent tangent/direct oracle and the
unchanged KKT gate. Separately, a future specimen screen could test a uniform
shear background plus a compact perturbation, with compatible side velocities,
to avoid the pure compact return flow; that would be a new selected loading
experiment, not a result or authorized continuation of this task. Broad cleanup
is planned before expensive publication runs, after these bounded decisions.
Inherited losses remain 39/45 pilot frames, early temporary logs and the missing
old `bulk_continuation.zip`; this task does not recreate those campaigns.
