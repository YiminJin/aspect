# Material cleanup and bounded midpoint growth — 10 November 2026

Entry: pf-rsf-particle-free at 5f1fc6f3c, with the incoming dirty synthetic
specimen, diagnostics, documentation and storage-maintenance work preserved.
Core 5d8f9ab1…c18e3e and synthetic checker 1086b403…d70ea remain untouched.
New binaries are built under build-particle-free/material-overnight/.
The material milestone and coarse-mesh midpoint temporal screen are complete.
The fine run stopped at a scalar obstacle-solver failure. The bounded correction
passes the isolated algebraic problem and the exact failed coupled interval.
The full fine-mesh .06 s endpoint remains unavailable; this task is closed
without another long continuation.

## Scoped material inventory

| Operation | Authoritative owner / active consumers | Finding and action |
|---|---|---|
| Aging value and speed derivative | FaultFriction; mature history.cc, signed section law, bulk-tip accepted candidate | Duplicate exact/BE formulas consolidated into evaluate_age with explicit AgingMethod and value/derivative result. Preserve exact mature/full midpoint endpoint and BE tip timing. Accepted candidate no longer evaluates unused friction. Remove derivative-only public wrapper after whole-source caller search. |
| Effective RSF parameters | FaultFriction; positive-state coefficient/partials/inversion/timestep and signed law/root/primitive | One checked mixture calculation skips inactive components. Fixed-state partials and time-discrete total derivative remain distinct. |
| State partial | FaultFriction public API; no production callers found, now covered by focused checker | Confirmed nonregularized a=0 division-by-zero defect. Shared evaluator returns b/theta directly; reject stateless and malformed mixtures before indexing. No physical direct-effect restriction tightened. |
| Regularized coefficient | FaultFriction signed and positive-rate APIs | Share coefficient and fixed-state partials; signed law adds the aging chain rule. Positive-rate Vmin wrapper remains. Root and Gauss48 search primitive remain separate. |
| Maxwell factors | Tip point update/tangent, material reference viscosity, coupled stage assembly | One rational stage relaxation; retain distinct mature exponential propagator. Avoid constructing exponential coefficients when selecting BE/half-stage viscosity. |
| Cohesive accounting | Tip::account and cohesive_energy | Share inverse compliance and old cohesive storage, retaining intact A=0 and phi=1 limits. Keep rational point elimination instead of singular h(1). |
| Calibration and phase law | initialize_bulk_cohesive_history, PhaseFieldHandler, Tip, BulkCohesive::compliance | K=Gc/[(8/3)ell], m=2GK/c² agree for the admitted AT1/curvature-one control. Generic phase functions and historical strict-positive-compliance controls have different admissible endpoints; preserve those intentional owners and independent oracles. |
| Configuration | PhaseFieldFault parser | Keep early midpoint/native guard; remove identical later guard. No checkpoint/input selector change. |
| Publication / tensor metric | Ordinary simulator, material candidate, native phase owner | Single publication remains; no lifetime change. Full 3D plane-strain stress/A metrics and immutable incoming age remain. Physical duration versus midpoint half-stage stays explicit. |
| Benchmark regions | bulk_tip/checks.cc | Correct box distances using actual extent/origin; synthetic remote masks match 8×4 analysis. Separate actual-integrity stop (>1e-6 phase remote/boundary) from retained .95 driving-margin screen. Production residual unchanged. |

Two further admitted edge cases were corrected: nonregularized constant a=b=0
friction imposes no state-splitting timestep restriction (avoid 0/0), and exact
aging has the proper value at subnormal speed even when x underflows to zero.
Neither changes the regularized finite-rate specimen. Stateful/stateless and
mixture-size checks precede parameter-array access. The unused derivative-only
public aging helper was removed after a full source/header/test caller search.
No publication, tensor layout, spin, quadrature, convergence or checkpoint-law
change was needed. These fixes are separate from the behavior-preserving patch.

Calibration was traced through initialization, the phase handler, generic
`BulkCohesive::compliance`, and the endpoint-safe tip elimination. The admitted
law has K=Gc/[(8/3)ell], m=2GK/c² and h=m phi(1+phi)/(1-phi)² on 0<phi<1.
Its derivatives and full plane-strain tensor factors agree with the rational
tip formulas. Generic strict-positive-compliance controls remain intentional
alternatives; the tip update must retain its regular limits at phi=0 and phi=1.
Independent point/phase references were kept rather than redirected to the
new helper. No second history publication or competing active calibration was
found within the selected scope.

## Selected experiment order and fixed screen

Cleanup build and focused mathematics/regressions first; record a buildable patch
milestone before solver changes. Repeat the available old midpoint sentinel once.
Then M1 (.0025) and M2 (.00125), h=.125, fresh equilibrated t=0 to .06.
At most two hours of focused repair if the new midpoint specimen fails.
Spatial S1 (h=.0625) is conditional on both midpoint histories passing unchanged
gates, remote/boundary integrity, fine absolute physical remainder ≤10% of added
fracture in both windows with improvement, and approximately ≤1% phase-increment
and fracture changes. One additional temporal level is permitted only if close
and improving. Do not change this screen after observing results.

Approximate total wall budget eight hours; four aggregate build workers/MPI
ranks, one numerical thread each; two-rank baseline; substantial jobs sequential.
The synthetic specimen is not an unstable-RSF calibration. No BE halving ladder,
server submission, loading adjustment or forced visible growth.

## Cleanup milestone (completed before numerical qualification)

The stable cleanup core is `bc9d14c1e2c83dba56235e4452c5cc7f097ed7079187ae1932d54389eff99eec`.
`cleanup-milestone.json` records exact source/checker identities; the tested
binary/checker are retained in `build-particle-free/material-overnight/cleanup-baseline/`.
Ordered patches separate consolidation, edge fixes and benchmark diagnostics.
No globalization or solver change was part of the cleanup milestone.

- Full units: 189151 assertions / 169 cases pass; point controls: 6078 / 21 pass.
- Focused two-material audit passes aging at zero/tiny/finite speed, dt=0,
  exact semigroup, derivatives, a=0, inactive mixture and wrong-interface guards.
- BE .0575→.06 restart agrees with protected core: max phase 4.14e-15,
  shear stress 2.39e-12 Pa, age 2.78e-16 s, slip 7.59e-19 m. Rollback/retry passes.
- Successful midpoint .09 control: max phase 9.13e-15, stress 2.54e-9 Pa,
  age 7.11e-15 s, slip 3.69e-18 m. Coupled tangent/functional checks pass.
- Midpoint 1→2-rank restart/rejection: max phase 2.37e-15, stress 1.14e-12 Pa.
- Layered mature transport: max recorded field difference 1.07e-14.
- Old sentinel replays once: mechanics 2.51e-15, phase KKT 6.34004e-7 at
  80 iterations, against the unchanged 1e-7 gate. Its limitation remains visible.

One initial focused audit used an object compiled concurrently with the final
edge-case edit and exposed that stale build. The stable source was rebuilt
before the successful audit and all regression/qualification controls. The
failed audit and original build logs remain. Two archived reference frames
were selectively restored and SHA-verified for the comparisons; no older
campaign was rerun. Original data-loss records remain unchanged.

## Completed temporal histories

| Method | dt (s) | Added fracture (J/m) | Full signed / absolute physical remainder (J/m) | Late signed / absolute remainder (J/m) | Wall s |
|---|---:|---:|---:|---:|---:|
| BE | 0.0025 | 0.0102742695 | 0.213323993 / 0.213323993 | 0.068307573 / 0.068307573 | 652.91 |
| BE | 0.00125 | 0.0102471184 | 0.106743927 / 0.106743927 | 0.034163849 / 0.034163849 | 1235.38 |
| midpoint | 0.0025 | 0.00994792955 | 1.04274833e-05 / 1.04274833e-05 | 8.65075821e-06 / 8.65075821e-06 | 789.06 |
| midpoint | 0.00125 | 0.010194438 | 2.67734016e-06 / 2.67734016e-06 | 2.1957452e-06 / 2.1957452e-06 | 1157.95 |
| midpoint | 0.000625 | 0.0102660734 | 7.33194286e-07 / 7.33194286e-07 | 5.48121286e-07 / 5.48121286e-07 | 2043.96 |

M1/M2 pass all unchanged gates and preserve remote/boundary integrity. The
fine absolute physical remainder is .0263%/.0226% of added fracture (full/late),
but phase-increment and added-fracture changes are 2.45% and 2.42%, missing the
1% state screen. The one user-authorized quarter-step M3 is selected as a
percent-level near-miss diagnostic with a clear improving work trend. The
initial 1.5% scheduling heuristic in entry.json is superseded for this cost
decision; the actual qualification screen remains unchanged at 10% work and
1% state. M2/M3 then pass at .969% phase-increment and .698% added-fracture
change. Fine absolute work-error ratios are .00714% full and .00561% late.
This passes the practical screen for S1; it does not establish a universal
order across active-set transitions or qualify the small axial growth error.

The final M2/M3 endpoint comparison gives relative L2 differences of .0100%
for the full plane-strain stress tensor, .0185% for cohesive A, .00661% for
cumulative slip and .00547% for age. Common-time rates differ by .0216%, .0196%,
.132% and .0783% at .04, .05, .0575 and .05875 s. Rates use piecewise-linear
reconstruction at their actual stage times; no midpoint observation is relabeled
as an endpoint rate. All three midpoint runs have exactly identical initial
Gauss fields. M3 is a finer comparison, not an exact reference.

The implemented spending screen checks the .06 endpoint state and both work
windows. It does not establish a uniform-in-time 1% state bound: M2/M3 phase
increments at .04 s differ by 3.67% (absolute L2 1.318e-4 m), when the growth
increment is much smaller. The .04 max phase difference is 3.674e-4.
Both times remain in the detailed comparison; activation and weak axial growth
need their own accuracy assessment.

M3 step 87 (.054375 s) exhibits difficult active-set/search behavior and passes
at iteration 80, the existing limit. No failed coarse-history interval, hidden BE
fallback, iteration-limit increase or solver repair occurred in these three runs. Together with the
still-failing old sentinel, this is a robustness/cost limit on midpoint adoption.
The cleanup itself does not remove BE numerical work loss. The BE pair even has
smaller two-grid phase sensitivity at the tested timesteps; a better work ledger
alone is not evidence of better local phase accuracy.

Physical accounting always uses
R_phys=Wext−Delta(E_bulk+E_cohesive+E_fracture)−D_bulk−D_cohesive−D_friction−D_radiation.
Raw column `Df` already contains friction plus radiation and is subtracted once;
raw `Dc` is cohesive viscous dissipation, not the RSF length Dc. Numerical loss,
nonlinear cohesive remainder, phase residual work and mechanical work defect
are reported separately, without subtracting them and renaming the result
physical accuracy. The scalar search primitive is never physical dissipation.
[Full and late signed/absolute channels](../../../benchmarks/reconstructed_fault/particle_free/evidence/material_overnight/energy-components.csv)
and the independent split-channel checks retain these distinctions.

M3 has maximum phase increment .03678 and maximum Bernstein-coefficient
increment .05792. The full SIPG addition is assigned 67.69% to the old central
band, 27.29% to the old tip band and 5.02% to initially intact axial cells;
the initially intact transverse region is unchanged. At the end, cancellation
of signed fault deformation is .8505%; 132/384 sections have opposite signs
above the declared local-rate observation threshold, with no multiple separated
bands detected. Full-history local positive/negative resistance work is
.285525/−.00144212 J/m, summing to the section dissipation .284083 J/m.
No local sign is clipped. These are limits of the selected section closure,
not evidence of pointwise frictional positivity.

The rate-pattern detector uses max(1e-10 s^-1, 1e-6 max|w|) within each
section. Earlier multiple-band counts are not zero: M3 reaches 147/384 at
.0075 s, when max|V| is only 1.06e-10 m/s and the specimen-wide unsigned
fault-deformation integral is 6.33e-10 m²/s. The first interval has nearly
complete cancellation, but local positive/negative work is only about
±4.23e-13 J/m. Counts at these small magnitudes do not establish physical
branching. Their signed fields, section integrals and absolute work remain
available; selected early point/native-row exports are restored for review.

Both DG traces are retained. M3 axial thresholds .001/.01/.1 move
.852/.614/.117 cells, while .5 does not advance. The normal .5 threshold
widens by .0543 cells on each side within the original normal support.
This remains band broadening plus a weak, sub-cell tail; a low contour is not
a resolved advancing crack. The global error/total-fracture ratio does not
bound the much smaller axial-growth error.

## Spatial sensitivity at the shared .04 s observation

S1 uses h=.0625 m, dt=.000625 s and a fresh re-equilibration of the same
continuum obstacle. Initial tau/A/slip are zero and age is .5 on all sections;
the initial phase is not copied from the coarse mesh. Initial fracture is
2.157382603065 J/m versus 2.157351615561 J/m. At common physical coordinates,
initial phase differs by .0664% in L2 and its cell-interior gradient by .330%.
The gradient comparison excludes DG jumps; the SIPG energy includes them.

The accepted .04 observation is complete independently of S1's later progress:

| Common-coordinate quantity | Coarse/fine relative L2 difference |
|---|---:|
| Total phase | .500% |
| Phase increment from each mesh's own seed | 222.39% |
| Increment gradient, cell interiors | 139.15% |
| Physical stress, full plane strain | 3.52% |
| Cohesive tensor A | 41.91% |
| Cumulative section slip | 5.04% |
| Section age | 1.37% |
| Stage section rate | 4.21% |
| Stage section resistance | .388% |

Added full SIPG fracture is .00048764244 J/m coarse and .00014501196 J/m
fine. The denominator for increment norms is the fine increment, rather than
the much larger pre-existing seed. Total-phase agreement alone would hide
this sensitivity. Phase uses exact native Q2 reconstruction; native point
stress/A and section histories use explicitly labeled Q5 observation
interpolants at fine physical coordinates and fine JxW/ds weights. These maps
are diagnostics, not history transfer or an assumed Q2 stress representation.
Age/slip are observed at .04; rate/resistance at the identical .0396875 stage.
Manufactured polynomial checks validate the maps and analytic gradients.

This is one combined spatial/seed-discretization sensitivity, not a convergence
order or an isolated cause. Mesh refinement changes the mechanical and phase
spaces, native integration points and section sampling together. The initial
differences are reported separately and each growth energy uses its own seed.
The large evolution difference prevents treating h=.125 as resolved, even with
small temporal/energy errors. The optional .08 continuation is deferred in
favor of analyzing this sensitivity.

S1 step73 (.045→.045625) passes after eight Newton corrections, but costs
1829.79 s, with 6674 constrained Krylov iterations, 6675 Stokes solves and
707930 bound-release events over 58 active passes. Its tiny shifted-search
steps and phase residual excursions are retained. A stack sample identifies
the constrained FGMRES action as a major cost; this is not evidence that
functional cancellation alone is responsible. No tolerance, iteration cap,
loading, phase floor or solver behavior was changed. The pre-window checkpoint
after step71/.044375 is copied and SHA-verified before rolling snapshots can
overwrite it; step72 can be replayed and checked against the original exports
to recover the step73 parent.

S1 stops after accepted step73/.045625, with MPI exit1 and 8067.78 s total
wall time. Step74 reaches mechanics 3.84109e-10 and phase KKT .0120299 at
Newton2, then the **shifted scalar obstacle subproblem** exhausts 100 active
passes. This is distinct from the older outer-Newton80 sentinel. The .06 fine
endpoint is missing; no full/late-to-.06 fine qualification is claimed.

## One bounded scalar-obstacle correction attempt

The scalar CG solve used a roundoff floor of 64 eps times the full absolute
load norm. Prescribed active bound values can keep that norm large while the
unresolved free-row residual is smaller than the floor but larger than the
unchanged VI certificate. An independent diagonal three-row problem exhibits
this defect: old solver exhausts its budget at a free residual 1e-15 against
1e-16; residual correction reaches the analytic solution in one active pass.
One- and two-rank oracles pass, as does the existing midpoint/tangent control.

The candidate solves A_restricted delta=b_restricted−A_restricted x, then
updates x and enforces its already-selected active rows exactly. It uses the
same tolerance and iteration budgets, with the roundoff scale belonging to
the correction right-hand side. The final primal/dual certificate is unchanged.
This proves a numerical defect in the scalar solver, not yet the cause of every
refined-case difficulty. Actual scalar matrices are captured for old/new replay.

The original cleanup core recovers step73 from the retained step71 checkpoint.
Every exported native point field, Bernstein coefficient row and section
history is **bitwise equal** to the original S1 step73. Snapshot02 in
`material-obstacle-parent73/output/restart/` is the exact incoming parent.
The deliberately stopped preliminary probe recovered step72 bitwise; its
nonzero status is a controlled stop, not a scientific failure.

Replay of the actual second scalar subproblem confirms the stopping defect.
The old solver stalls for 100 passes with certificate 4.51746e-14 against
4.34028e-14; its absolute-load roundoff floor is 2.78828e-13. The correction
solver passes in one active pass at 4.96092e-16, with unchanged final tolerance
and iteration caps. The largest offending free row is 27687, cell 769_1:0,
shape3, centered at (2.15625,2.03125) m: coefficient .13958275, lower bound
.13935274. This row is not almost intact; the verified defect is scalar solve
accuracy rather than proof that nonlinear tip stiffness is resolved. Actual
matrix/vector/map exports and ranked row diagnostics are retained.

The coupled retry `material-obstacle-window02` accepts step74 at .04625 s in
six Newton corrections: mechanics 4.90869e-14, phase KKT 9.34664e-8, ledger
2.58525e-15, physical-work remainder approximately 4.45314e-9 J/m.
It costs 1657.85 s on two ranks. The expensive constrained mechanics and
small search steps remain; a successful repaired interval does not establish
an affordable fine trajectory. No fine .06 continuation was launched.

The actual coupled retry uses diagnostic candidate core a6fd710c…fd4a and
map checker f8352bd8…41ea. Its opt-in scalar capture was then removed from
production source. The final clean core b2300f6b…7b32 reproduces the captured
scalar certificates and passes the full unit suite (189151 assertions,
169 cases), one/two-rank analytic oracles, and short BE/midpoint controls.
The final BE interval is bitwise equal to the cleanup endpoint; midpoint
maximum phase/stress differences are 1.54e-14 / 2.54e-9 Pa. Its age/slip
maximum differences are 8.88e-15 s / 4.01e-18 m. An earlier BE test with
repeated rollback was stopped by an undersized 180 s process cap after two
converged rollback trials; the separate accepted interval completes in
24.67 s with the retained 300 s cap. No acceptance threshold was changed.
The old outer80 sentinel was repeated only before this repair and remains
unqualified under the final kernel.

## Complete spatial prefix and interpretation

All 73 accepted fine intervals have been reduced, ending at .045625 s.
Added SIPG fracture is .00043359763 J/m: .00037364707 in the old central
band, .00005995057 in the old tip band, 2.14e-13 in initially intact axial
cells, and zero in transverse intact cells. Meaningful central increment
(>1e-4) first appears at .0275 s, old-tip increment at .040625 s; no axial
increment reaches that threshold in this prefix. Maximum phase increment is
.00590778. Full-prefix signed and accumulated absolute physical remainder
are both 1.71997e-7 J/m (.03967% of added fracture); .04–.045625 gives
3.13500e-8 J/m (.01086%). These windows end at the actual prefix, not .06.

At the last common endpoint .045625 s, coarse/fine phase-increment relative
L2 difference is 257.31%, stress 3.71%, A 41.92%, slip 4.77%, age 1.39%.
Stage rate and resistance differ 3.20% / 7.87%; force differences are sensitive
to diagnostic interpolation and near-zero reaction. Coarse added fracture is
.00138927609 J/m. The finer prefix cannot establish whether late axial growth
is removed because it stops before the coarse late onset. The initial seed
and evolution sensitivity must be resolved before claiming physical growth.

## Decision and next experiment

Retain the material consolidation and the scalar residual-correction fix.
Keep BE as the default reference and midpoint selectable. Midpoint provides
useful completed coarse histories with much smaller physical-work error, but
its final endpoint screen is not a uniform-time accuracy guarantee, and the
large spatial sensitivity prevents propagation qualification. This synthetic
specimen does not demonstrate an unstable RSF event.

The next justified numerical experiment is a bounded continuation from the
repaired .04625 s fine checkpoint to .06 s with the final clean kernel,
identical method/inputs/tolerances, two ranks and a two-hour process cap with
accepted-step checkpoints. Stop on the original acceptance/integrity gates.
First verify that checkpoint's initial observations against the retained
candidate endpoint; compare the completed fine state with M3 using the existing
physical-coordinate analysis. This completes the missing spatial comparison
before any .08 extension, new mesh, loading calibration or instability campaign.
It is a future selection, not a background job left by this task.

## Reproduction, raw evidence and availability

The [evidence reading path and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/material_overnight/README.md)
identify ordered source patches, actual executables, exact inputs, checkpoints,
failed attempts and final checks. `cleanup-milestone.json` identifies the
buildable material-only baseline; `final-runtime-identities.json` identifies
current source and runtime separately. M1/M2/M3 and original S1 used the cleanup
core, not the current final executable.

`energy-summary.csv`, `energy-components.csv` and `regional-fracture-windows.csv`
retain signed and accumulated absolute interval terms for every applicable
window. The physical remainder includes every dissipation channel once; the
search primitive is never counted as physical energy. Fine-prefix windows are
explicitly labeled. `costs.csv` records numerical costs; failed parent recovery
and the repaired interval are separate costs, not a new complete S1 history.

The raw review archive includes accepted energy tables, section histories,
remaining initial/final/transition native exports, sources, generated inputs,
commands and logs. Its manifest verifies every member. Intermediate native
exports were compressed losslessly into indexed local field archives; selective
restoration verifies bytes and metadata before publication. All checkpoints
and protected executables remain local. No scientific data were discarded.
Earlier cleanup losses (39/45 pilot frames, early temporary logs and the absent
old bulk_continuation.zip) remain losses; this task has not recreated them.

Figures: [all histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/material_overnight/temporal-comparison-all-histories.png),
[native phase/increments](../../../benchmarks/reconstructed_fault/particle_free/evidence/material_overnight/midpoint-phase-all-histories.png),
[axial/normal coarse traces](../../../benchmarks/reconstructed_fault/particle_free/evidence/material_overnight/native-traces-M2-M3.png),
[common-prefix phase](../../../benchmarks/reconstructed_fault/particle_free/evidence/material_overnight/spatial-prefix-phase.png),
[common-prefix native traces](../../../benchmarks/reconstructed_fault/particle_free/evidence/material_overnight/spatial-prefix-native-traces.png).
PDF counterparts are retained. Figures label actual endpoint/stage times and
retain both native DG traces without smoothing.
