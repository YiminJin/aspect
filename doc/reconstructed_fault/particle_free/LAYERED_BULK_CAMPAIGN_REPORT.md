# Corrected-spin layered moving-fault campaign

**Complete, 6 November 2026.** The selected direction, time, mesh, finite-slip
law, adaptive, restart/rejection and wider-domain comparisons are finished.
No native campaign run remains active. The outcome supports local use of the
implemented boundary/spin configuration, with explicit accuracy limits below.
The user resumed the shutdown pause; historical progress notices are preserved in
`layered_bulk_campaign/documentation-before-closeout/`. They are not current
execution instructions. [Protocol](LAYERED_BULK_CAMPAIGN_PROTOCOL.md) freezes the
choices; [closeout handoff](LAYERED_BULK_CAMPAIGN_RESUME.md) records completed work.

## Findings and decision

The selected bulk-slip-subtracted spin produces a persistent finite-slip effect:
central/end cumulative-slip differences from total spin are 3.43%/4.24% on the
fine mesh. Their changes under mesh refinement are 2.61%/10.21% **of that law
difference**, not of the total slip. Normal-traction law differences remain
substantially more sensitive to space. This is a constitutive comparison, not
proof that either trajectory is an exact solution.

The principal accuracy obstacle is the bounded Q1 profile remapping. L+ profile
mass error is 6.24% at dt=.0125, 7.04% at dt=.00625, and 2.39% on the finer mesh
at dt=.00625. The integrated physical divergence-minus-boundary flux is below
4.3e-15 m² across the uniform/adaptive/domain controls. Smaller physical timesteps do not fix this numerical mass defect.
A small endpoint energy ratio does not establish profile or traction accuracy.

The layered exterior uses the correct unreacted accepted stress during transport,
then supplies compatible endpoint velocity and traction to mechanics. Its scalar
reference agrees with an independent recurrence near roundoff. This boundary
implementation is locally verified; the advected problem is not yet ready for
an unqualified BP3-like production pilot.

## Source identity and bounded implementation changes

Worktree `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `cf06525471ce89789f1ac10905dc79f3aac7562a`, with
preserved dirty source and manual work. No reset, alternate checkout or commit.
The canonical native binary is historical and deliberately not assumed to match
HEAD. Original coherent candidate `stage-c/layered-bulk/aspect-release` has SHA256
`17f7152ca5e5f78b309cfebf4586a82ac1926e1ef2b9231e225d2afdc3e525ed`.
`candidate.json` identifies its core and matching energy/homogeneous/layered plugins.

The first 144x48 attempt failed before initialization because mature DG Cartesian
admission required bitwise corner-coordinate equality. An independent mesh audit
measured at most 1.77636e-15 m mapping roundoff. `fe_strain.cc` now admits only
16 machine-epsilons times coordinate/extent scale; an explicitly sheared negative
control still rejects. This is a geometric roundoff guard, not a nonlinear or
physical tolerance change. The separate candidate SHA256 is
`812811cd41f0551ef346117ecbc86ed0e694f8cb1a0229be7b02207fdbb8834a` in
`stage-c/layered-bulk-cartesian/`. It passes 183390 assertions in 157 unit cases
and reproduces the original coarse two-step prefix bitwise. Plugins are unchanged.
The canonical binary and original candidates remain protected. Fine, adaptive
and subsequent lifecycle/domain controls use this corrected-admission candidate.

Benchmark-only additions include explicit initial shear metadata, selectable
candidate/law runners, rectangular-field reductions, common-quadrature comparisons
and the stateful layered provider. The law, native DGQ2/SSPRK3 transport, endpoint
forcing, accepted-spin lag, solver criteria, defaults and checkpoint meanings
are unchanged in this campaign. The integrator's generic “total material spin”
banner is stale; resolved `Constitutive spin` and per-step `Accepted bulk spin`
diagnostics identify the selected law. No old-law checkpoint is relabeled.

## Physical fixture and exterior ownership

G=10 Pa; eta(y)=5.5+4.5*tanh(y/.35) Pa s; top-to-bottom velocity difference
2 m/s. The mature fault starts at (-.8,-.48) to (.8,.48), with 20 fixed material
labels (requested initial spacing .1 m), compact Q1 profile
`.35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.6*x)/.6)^2)^4`.
The .6 m denominator is in y-.6*x; its initial normal-distance scale is
.6/sqrt(1+.6^2)=.51450 m, not a .6 m normal halfwidth. Growth is disabled. Label shear preload is
3.7+.15*cos(3*x_initial) Pa; normal background compression is 5 Pa. Initial
age is 10+2*sin(2*x_initial) s. Friction retains a=.025, b=.015, Dc=.008 m,
Vref=.001 m/s, mu0=.6 and radiation coefficient 1. All cases start fresh.

Domain [-6,6] x [-2,2]; U=0,+2,-2 m/s; T=.8 s. Coarse grid 96x32, h=.125 m;
fine 144x48, h=1/12 m. The factor-1.5 nonnested refinement was selected before
fine results for bounded cost. Surface labels and all physical scales are fixed.
Added translation traverses 12.8 coarse cells; total mean L+ displacement is
about 2.03 m. Curve clearance remains above 1.49 m; boundary phase is negligible.
U=0 is still a moving/deforming material fault, not a frozen geometry control.
There is no identical-physics freeze-only switch in this transported path.

The exterior scalar Maxwell recurrence enforces constant shear q and the imposed
velocity difference. It rotates accepted tau_n using accepted gamma_n, relaxes
with beta=exp(-G*dt/eta), and solves the scalar velocity-integral condition for q.
Pressure is p=tau_yy (zero pressure reference), so top traction is (q,0).
Pure advection receives tau_n on every incoming face, including local top inflow;
it must not receive the rotated/reacted endpoint twice. Exterior J_f=0, so the
background is shared by both spin laws. Initialization keeps the actual .1-s
mechanics convention and zero stored stress. Homogeneous p=0 is a distinct
reference convention and is not silently substituted here.

Provider candidates use attempted dt; only accepted-step postprocessing publishes.
Checkpoint save/load preserves accepted stress, gradient, time and step under a
provider identity. The actual accepted clock, not t/dt, defines its recurrence.
Independent 32/64/128-panel reference differences are at most 1.23e-11; the constant
viscosity limit differs by 4.15e-13. Native accepted-clock checks give stress
error <=6.22e-15, gradient <=3.14e-14 and integrated velocity <=1.53e-13.

## Completed trajectories and cost

Times are local wall seconds with four MPI ranks sequentially. No completed
physical case below retried. Costs include native initialization and output;
postprocessing is recorded separately. All layered trajectories reach .8 s.

| Case | h | dt | Steps | Wall s | Solves / linear iterations | Max cumulative slip m | Profile mass error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bulk U=0 | .125 | .0125 | 64 | 2437.21 | 449 / 14434 | .152391 | .4525% |
| Bulk U=+2 | .125 | .0125 | 64 | 2349.92 | 454 / 14625 | .155063 | 6.2405% |
| Bulk U=-2 | .125 | .0125 | 64 | 2476.63 | 451 / 14665 | .157484 | 3.2705% |
| Bulk U=+2 half dt | .125 | .00625 | 128 | 4282.29 | 776 / 24151 | .156150 | 7.0373% |
| Total U=+2 | .125 | .0125 | 64 | 2222.13 | 451 / 14525 | .150104 | 6.2279% |
| Total U=+2 half dt | .125 | .00625 | 128 | 4147.67 | 776 / 24157 | .151132 | 7.0209% |
| Bulk U=+2 fine | 1/12 | .00625 | 128 | 8826.66 | 770 / 24241 | .153315 | 2.3861% |
| Total U=+2 fine | 1/12 | .00625 | 128 | 8518.47 | 770 / 24238 | .148336 | 2.3785% |
| Bulk U=+2 adaptive | .125 | .002269–.00625 | 217 | 5876.80 | 1213 / 37639 | .156789 | 7.2572% |
| Bulk U=+2, halfwidth 8 | .125 | .0125 | 64 | 2753.56 | 456 / 14051 | .156566 | 6.2273% |

The separate homogeneous corrected-law bridge completed 32 steps to .4 s in
395.38 s. Initial fields match the retained old-law case bitwise. Final peak
stress is 2.34061 Pa; whole-domain old/new difference is .0056491 Pa m (.06076%),
96.93% of its squared norm in the interior and .1445% in the corner. Top inflow
callback count is 23814. This verifies integration of the boundary and corrected
law without repeating the earlier homogeneous matrix.

## Norms, temporal sensitivity and finite-slip law effect

Central labels are .2 <= label <= .8; ends are the complement. Final surface norms use
common line quadrature and include the full plane-strain stress metric. Stored
histories use 400 material midpoint samples and the reference curve Jacobian;
the variable-step comparison interpolates the uniform accepted history in time. Normal
traction is n.(tau-pI).n; effective compression additionally includes 5 Pa.
Induced, work-adjoint and preload-added driving shear are distinct quantities.
Absolute norms and denominator norms are retained in pair JSONs; relative numbers
below are pairwise sensitivities, not exact errors or universal acceptance gates.
Common bulk quadrature splits at the union of both Cartesian grids, including
translated/nonnested grids. A discontinuous Q2 analytical control agrees to
4.45e-16. Earlier unsplit comparisons are preserved separately.

| Final quantity | Bulk time pair central / ends | Fine bulk-vs-total central / ends | Bulk mesh pair central / ends |
|---|---:|---:|---:|
| Physical surface stress | .6998 / .8910% | 2.7549 / 2.2867% | 8.7095 / 4.9227% |
| Work-adjoint induced shear | .7014 / .4676% | 9.9865 / 2.5825% | 3.6838 / 3.0571% |
| Pressure-inclusive induced normal | .4253 / 1.0284% | .5877 / 1.6645% | 2.5281 / 6.7633% |
| Slip rate | .2805 / .5184% | 5.3384 / 6.0589% | 5.2739 / 8.0290% |
| Cumulative slip | .7091 / 1.3566% | 3.4287 / 4.2432% | 2.7753 / 4.8381% |
| Age | .3472 / .9520% | 5.0265 / 6.1697% | 5.9589 / 9.8222% |

Whole-domain stress sensitivities are .23839% in the bulk time pair, .23758% in
the total-spin time pair, .38752% for the fine law pair and 1.07060% for the bulk
mesh pair. Total-spin cumulative-slip time sensitivity is .6660%/1.0573%.
The coarse half-dt law difference is 3.4183%/4.2162% in cumulative slip, so the
observed constitutive effect persists under both timestep and mesh refinement.
Principal-direction gap-weighted RMS sensitivities are .001981 rad (time),
.008637 rad (coarse law pair) and .014607 rad (bulk mesh pair). Angles are modulo
pi and weighted by the in-plane principal gap; isotropic directions carry no
false orientation significance.

The four-run comparison defines D_h=(bulk-total) at each mesh. The norm of
D_coarse-D_fine divided by norm(D_fine) is 16.52% for whole-domain stress,
16.78%/20.88% for central/end surface stress, 5.86%/26.24% for work shear,
4.63%/20.83% for V and 2.61%/10.21% for cumulative slip. Absolute cumulative-slip
norms of the difference-of-differences are .000134416/.000434294 m sqrt(m),
against fine-law norms .00515091/.00425167 m sqrt(m). Normal-traction ratios are
69.90%/31.97%; those smaller effects are not spatially resolved. A single mesh
pair supports no asymptotic order or uniform accuracy qualification.

## Spin, localization and energy evidence

Corrected L+ coarse/half/fine sampled-max slip-spin envelopes are
.28140/.28443/.28603 rad; largest absolute label integrals are
.24128/.24168/.24970 rad. The fine maximum integrated commutator-norm tendency
is .50081 Pa. Actual chi=h(phi)/I is used. The sampled-max sum is an envelope,
not a continuum supremum or the rotation of one material trajectory. Label
integrals use the accepted parent at the old label position; native stress
rotation uses the frozen Eulerian accepted-spin field. Neither diagnostic
changes the selected accepted-gradient lag.

| Bulk L+ | Endpoint signed J/m | Accumulated absolute J/m | Absolute / active work | Trapezoidal signed / absolute J/m |
|---|---:|---:|---:|---:|
| dt=.0125 | -.00311206 | .0754080 | .269529% | +.000546914 / .00232096 |
| dt=.00625 | -.00143523 | .0385043 | .137630% | retained in case JSON / .000599813 |
| Fine dt=.00625 | -.000603139 | .0393063 | .140529% | +.00120899 / .00124435 |

The endpoint remainder roughly halves with dt, but is nearly unchanged by mesh
refinement while profile/stress sensitivity changes markedly. Signed cancellation,
trapezoidal work replacement and the unchanged endpoint scheme are reported
separately. Rotation changes orientation/traction without directly changing the
elastic norm; energy alone cannot establish correctness of the spin choice.
Geometry replay is within 2.3e-16 m and age replay within 8.9e-16 s. This verifies
the implemented updates, not their continuum accuracy.

## Direction, profile balance and finite-domain comparison

After removing U*t, coarse L+/L- phase differs 34.40%, whole stress 2.5743%,
central/end cumulative slip 2.6710%/7.8903% and V 4.8053%/12.3627%. These exceed
time sensitivity. Refined L+ versus coarse U=0 reduces phase difference from
20.47% to 6.36%, central slip from 2.66% to .658%, and central V from 5.31% to
.921%. This mixed-resolution/time comparison supports a profile-discretization
contribution; U=0 is not an exact reference, and finite fixed boundaries are not
Galilean transformed. Refined L+ versus coarse L- remains more sensitive; no
direction independence is claimed.

Initial coarse profile mass is .1941809534 m². Changes for U=0, L+, L-, L+ half-dt
and refined L+ are .000878599, .0121178, .00635069, .0136652 and .00463325 m².
Their integrated physical balance contributions are below 1.8e-15 m². Source
`profile_transport.cc` uses characteristic nodal interpolation, reverse
MacCormack correction and donor-range clamping, without a paired flux or integral
balance constraint. Bounds and transported irreversibility do not ensure
  d integral(phi)/dt = integral(phi div(v)) - boundary integral(phi v.n).
The necessary next profile improvement must preserve this material equation and
advected obstacle; global renormalization, a density remap or compensating phase
clipping would change the problem.

The residual translation-removed velocity difference for refined L+ versus U=0
was .97250% (.0463347 m²/s), against .24702% (.0127041 m²/s) under the L+ mesh
change. Pressure sensitivities were 9.30% and 7.32%. This justified the completed
halfwidth 8 versus 6 control, at matching h=.125, dt=.0125, U=2, T=.8 and all
other physical inputs fixed. It uses 128x32 cells and the same 20 labels, completing
64 steps without retries in 2753.56 s (456 solves/14051 iterations). The independent
reference retains stress/gradient/velocity-integral errors below
5.78e-15/1.78e-14/5.25e-14 and zero clock error.

On the shared [-6,6] x [-2,2] domain, wide versus original stress differs
.105726 Pa m (1.02243%), translation-removed velocity .0435347 m²/s (.84598%),
pressure .117050 Pa m (11.0575%), and phase .00184031 m (1.02632%). Initial
geometry, age and zero stored stress match; initial V already differs
.1848%/.2266% centrally/at the ends because changing the domain changes the
initial mechanical solve. This is a physical domain sensitivity, not a time
or spatial discretization error.

| Final wide/original sensitivity | Central | Ends |
|---|---:|---:|
| Surface stress | 1.3498% | 1.2617% |
| Work-adjoint induced shear | 5.4862% | 2.7873% |
| Pressure-inclusive induced normal | 1.0406% | 1.2289% |
| Slip rate | 1.7281% | 1.5265% |
| Cumulative slip | 1.0161% | 1.1454% |
| Age | 1.7144% | 1.5208% |
| Localization I | .08293% | .06020% |

The central/end cumulative-slip absolute differences are
.00161222/.00122841 m sqrt(m); geometry differences are
.00305913/.00272231 m sqrt(m). Boundary influence is measurable, with velocity
sensitivity comparable to the residual directional difference. The single
widening does not establish an infinite-domain limit or isolate all directional
error. In particular, endpoint/normal-traction uncertainty remains.

Profile mass growth barely changes: 6.24045% to 6.22731%, with sampled peak
.289376 to .289439. The wide mass increase is .01209224 m² while integrated
physical divergence-minus-flux is -1.43e-17 m². Its minimum curve clearance is
1.49870 m and maximum boundary phase 8.40e-15. Thus lateral enlargement does not
repair the profile balance defect. Adaptive mass increase is .01409202 m² versus
physical balance -4.24e-15 m². These results separate two mechanisms: numerical
profile redistribution and finite-boundary mechanical response.

The wide endpoint remainder is -.00843686 J/m signed and .0966087 J/m accumulated
absolute, .259226% of 37.2681 J/m active work; trapezoidal diagnostics are
-.00321488 signed/.00671047 absolute. These are inventories over the larger box,
so its relative energy ratio is not a same-domain improvement measure. Shared-
domain field norms and profile balance supply the controlled comparison.

Both translation directions complete with the same qualitative behavior: bounded
profiles, positive ages, stable accepted solves and nonzero remapping mass gain.
Their quantitative differences are retained. Another opposite-direction fine
run would not remove the demonstrated missing integral balance; no direction-
independence claim is made, and further directional qualification belongs after
the profile correction rather than an additional sweep of the same method.

## Adaptive policy and lifecycle

The existing reconstructed-fault controller completed 217 steps in 5876.80 s,
with 1213 linear solves/37639 iterations and no retries. It uses cap .00625 s and
predicted log-age bound .05. Actual dt spans .00226877–.00625 s; it reaches the
cap at step 169 near t=.5006. Cost is 37.2% above uniform .00625 s (71.37 minutes
versus 97.95 minutes), with 69.5% more accepted steps. Independent boundary errors
remain near roundoff: stress 4.22e-15, gradient 4.39e-14, integrated velocity
2.88e-13, and accepted-clock error zero.

Source and accepted CSVs confirm that timestep selection precedes deferred mature
DG publication: proposal for step n sees committed surface n-2 (initial surface
for the first two steps). That prediction satisfies .05 within 2.3e-16, but 105
actual steps exceed .05 and the maximum actual change is .08273. The immediate
parent prediction also reaches .08269. The lag and the unknown endpoint rate
both matter; even a fresh-parent predictor is not an endpoint acceptance bound.
No controller or transaction ordering is changed to hide this distinction.
Maximum actual age change falls from .13518 in the uniform reference; maximum
V*dt/Dc barely changes (.18609 to .18473), so the age setting is not a substitute
for a slip-based event restriction.

Adaptive versus uniform .00625 s, on the same mesh and physical initial state:

| Quantity | Final central / ends | Space-time central / ends |
|---|---:|---:|
| Surface stress | .2654 / .3732% | .4397 / .4088% |
| Work-adjoint induced shear | .2778 / .4560% | 1.8027 / .6683% |
| Pressure-inclusive induced normal | .1750 / .2347% | .1634 / .3883% |
| Slip rate | .2501 / 1.2037% | 1.4745 / 1.3500% |
| Cumulative slip | .3482 / .4942% | .7500 / .8831% |
| Age | .3361 / .3129% | 1.5117 / 1.0760% |

Final whole-domain stress differs .00464264 Pa m (.04490%). Final central/end
cumulative-slip absolute norms are .000556329/.000535343 m sqrt(m). Complete
histories are in `schedule-layered-bulk-adaptive.json`; space-time norms use
trapezoidal integration of squared surface norms, with linear interpolation of
the uniform .00625 s output. This is not solver dense output, particularly across
the initial rate transient, and no convergence order follows from this comparison.

Adaptive endpoint remainder is +.00440255 J/m signed and .0208729 J/m accumulated
absolute (.0746123% of active work). Trapezoidal diagnostics are +.000126385
signed/.000354784 absolute. Signed cancellation differs from the uniform case;
the reduction in accumulated absolute remainder must not conceal that change.
Profile mass error rises to 7.2572%, sampled peak is .285468 and maximum
normalization discrepancy .65249%. Geometry/age replay errors are 2.23e-16 m and
5.56e-17 s. This policy resolves the age transient at moderate extra cost, while
leaving the profile accuracy obstacle intact. It is a defensible local comparison
policy for this fixture, not an event or BP3 timestep qualification.

A preserved checkpoint after accepted 159 at t=.4501946708 starts continuation
at step 160 with dt=.00406655, before the cap is reached. Its files were copied
and hashed before ordinary rotating-slot overwrite. `--checkpoint-directory`
selects this bookmark while retaining the parent fixture metadata and native
checkpoint semantics. The 4→2-rank continuation completed all 58 remaining steps (160–217), no retries,
in 2576.59 s. Every resumed accepted time/step matches the uninterrupted trajectory
within 2.8e-16 s. Maximum exported bulk/surface/background component differences
are 1.79e-13/3.38e-14/1.79e-13 in their respective native units, below the
2e-10 bulk/surface and 1e-12 background comparison bounds. This checks changing
timesteps after reload as well as the later cap. The separate provider rejection
check also passes. One completed step 2 attempt is rejected and retried at half
dt; the 5 accepted steps to .05 s are bitwise identical to a fresh control with
the same accepted schedule, in bulk, surface and background exports. The
rejected/control costs are 78.39/68.31 s. Both independent background recurrences
have zero clock error and stress error below 8.4e-17. This verifies that rejected
reference candidates do not advance accepted background history.

## Qualification boundary and next experiment

The coherent boundary/reference and bulk-spin feature are implemented and locally
verified. Coupled physical comparisons establish finite-slip effects and an
explicit profile accuracy obstacle. They do not qualify production BP3, cohesive
growth/tips, AMR or arbitrary transport. Existing cohesive energy/admission failures
remain separate; this mature comparison neither repairs nor invalidates them.

This SI-scaled algorithmic fixture is not a dimensional lower-crust model.
[BP3 mapping](LAYERED_BP3_PILOT_MAPPING.md) specifies the still-required dimensional
elastic/viscous scales, tectonic loading, effective/lithostatic pressure convention,
fault geometry and actual boundary initialization. U=0 is not the planned fixed
fault. No realistic subduction, server submission or production checkpoint
continuation is included. Cleanup losses remain 39/45 native-pilot frames and
earlier temporary logs.

Preserved failures include the first non-dyadic Cartesian admission, unused GCC16
configure, initial bridge argparse error and a postprocessing list-index defect
(corrected without native reruns). The historical total-spin L0 stopped by user
SIGTERM during step 27 after accepted step 26, exit 143; it was not a solver failure.
All original outputs and protected binaries remain. No production/server run,
checkpoint conversion, additional rotation matrix or general refactor was
performed. Source/native changes in this campaign are limited to the verified
Cartesian roundoff admission fix; the benchmark provider, diagnostics and
runners supply the remaining implementation work.

**Next justified experiment:** implement bounded Q1 transport that satisfies the
integrated material equation, including its divergence source, while transporting
the irreversibility obstacle consistently. Compare it against this retained
MacCormack control using the same physical cases and fixed accepted schedules,
then test both directions and refine space/time with the same 20 labels. Preserve
joint publication/rejection/restart and the native stress/endpoint conventions.
Do not use global phase renormalization, smoothing or a smaller dt as a substitute
for the missing balance. The finite-domain sensitivity must remain in the error
budget; the present box is not an infinite-domain reference. A separately chosen
stationary layered pilot avoids this particular remapping error but still needs
its own supported stationary configuration and dimensional BP3 boundary/loading
specification. No new run is queued by this recommendation.

## Figures, reproducibility and diagnostic precision

[Evidence index](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/README.md)
links all figures and the 65-frame cell-crossing animation. In particular:

- [Profile balance, age resolution, energy and cost](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/layered-accuracy-cost.png).
- [Direction histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/layered-directions-histories.png) and [translation-removed fields](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/layered-directions-shifted-fields.png).
- [Uniform/adaptive histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/layered-timestep-histories.png) and [V, cumulative slip and age on material labels](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/layered-timestep-label-histories.png).
- [Matched fine-mesh spin/stress comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/layered-spin.png) and [cell-crossing GIF](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/layered-cell-crossing.gif).

`make_layered_bulk_figures.py` reproduces the figures serially; `figure-commands.json`
records commands and layout corrections. Raw outputs and checkpoints stay in the
existing build tree. Case commands, solver logs, reference checks, complete
histories, absolute/reference norms and final provenance are retained in the
evidence directory. All native runs/builds use at most four workers; analysis
entry points set BLAS/OpenMP to one before importing NumPy.

Postprocessing quadrature is also a limit. For L+ coarse, half-dt, fine, adaptive
and wide cases, final q12-minus-q8 storage/mechanical/
Maxwell/flux differences are near roundoff (at most 2e-12 in the reported units),
and profile mass differs by at most 2.5e-14 m². Localized friction power changes by
up to .00291 W/m, slip power by .000313 W/m and localization measure by .00390 m
in the selected checks. Finite-end localization is not polynomial within a cell;
these are sensitivity checks, not a converged integral reference. Small signed
energy differences and normalization discrepancies should not be interpreted
beyond that precision. No frozen quadrature audit or native quadrature change
was repeated. The profile mass defect is many orders above this integration
sensitivity and has a separate, identified transport mechanism.
