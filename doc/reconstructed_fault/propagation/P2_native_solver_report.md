# Native P2 intermediate accuracy and saddle escape

The bounded numerical task passes. Adaptive intermediate accuracy and one
balanced translation preconditioner retain the original physical KKT and energy
criteria. Three retained core entries use19.7% fewer tangent actions. Fresh
native evolution with the energy-qualified prefix reaches .53125s in **85min9s,
1684.9MiB peak RSS**, with three accepted core-growth cycles, zero healing and
all seven unaligned matching-prototype field targets passing. Core phase grows
.60000194→.60457720; cohesive traction turns downward during growth. Complete
prepublication failure/retry restores histories exactly and publishes once.

The known native saddle escapes to the matching stable prototype branch in a
separate noncommitting control. This is not fresh saddle continuation or full
angular stability. Numerical cost remains high:15.7–16.8min per fresh growth
step. One full economical native run is prepared, **27–35h estimated**, and
has not been launched. Governing energy, constitutive laws, accepted-state
ownership, physical obstacle and defaults remain fixed. No P3 or material tuning.
The original15.3446-J/m coarse-prefix energy failure stays failed.

## Fixed model settings

| Quantity | Value / units |
|---|---|
| Domain / sharp fault | [0,1]² m; y=.5 m, endpoints x=0,1 m |
| Association | Fixed full normal cross-section, half-width .50000000000001588 m |
| Mesh / fields | 16×512 cells; Q2 velocity, Q1 pressure and phase |
| Integration / particles | Six normal bulk points, intersection surface panels; native CPDI fracture; 3×3 fixed particles per cell (73728 total) |
| Surface | 64 Q1 segments; consistent normalization, analytic transverse reference |
| G / eta / lambda | 1e6 Pa / 1e8 Pa s / 100 s; homogeneous constituents |
| Viscosity bounds | 1e6–1e10 Pa s |
| c / Gc / ell / p | 1000 Pa / 26.6666666667 J/m² / .15625 m / 1 |
| m / Hc / A | 128.00000000016 / .5 Pa / 64.00000000008 Pa |
| Friction / pressure / radiation | Coulomb .1 / prescribed 10000 Pa / zero |
| Initialization | Shared relaxed nominal-core-.6 seed; relaxed peak .600001937763; fresh zero FE d_R, Z, slip and bulk stress |
| Loading | Interval mean of ux=.256 t(y−.5) m/s; top/bottom uy=0; sides uy=0, natural ux |
| Phase admissibility / boundary | Immutable accepted-phase lower bound, upper bound 1; natural phase boundary |
| Tolerances | Phase nonlinear 1e−12, linear 2e−7; mechanical nonlinear 1e−8, linear 1e−7; phase max 100 iterations, 35 line searches |
| Fresh economical onset | 58 steps to .53125 s; fine windows around yield and damage onset |


G remains a dimensional synthetic verification value. The restrictions isolate
fixed-geometry coupling; they do not qualify transport, replenishment or RSF
evolution. H+ is diagnostic; the prescribed pre-existing damage is protected by
the previous physical phase bound.

## Intermediate linear accuracy

Let d_k be the projected defect, d_0 its entry value, t_phi the unchanged final
entry target, eta_min the existing configured phase linear tolerance, and b_F
the actual free Newton load including active-column forcing. The default-false
`Adaptive endpoint linear accuracy` selects

    eta_k = min(.01, max(eta_min, .1 sqrt(d_k/max(d_0,t_phi))))
    linear_target = max(eta_k ||b_F||_2, .1 t_phi).

After a resolved saddle departure, update only the forcing reference d_0 to
max(d_0,new projected defect). A stationary saddle otherwise supplies an almost
zero scale and prevents tightening toward the new branch. The original physical
entry target remains fixed. This changes intermediate search accuracy explicitly. Mechanical tangent-action
refinement still uses eta_min=2e−7; each Stokes solve retains 1e−7. Original
CPDI reassembly and physical KKT, irreversibility, bounds, energy and publication
criteria do not change. With the option off, retain the original relative target.

On the exact normal-space shadow, adaptive accuracy reduces actions from
23/19/19 to15/13/13 for the three retained entries, but takes six updates instead
of four. This is a cost/accuracy diagnostic, not full native/tangential evidence.
Its accepted references and histories are immutable and no state is published.
The initial cap .25 passes native acceptance but costs1088.4s/179actions,
including eight updates and twelve line-search trials. Adding the one translation
correction costs873.5s/127actions, with seven updates and ten trials; a brief
diagnostic build overlapped that run, so its wall time is not a solitary-run
measurement (CPU838.9s). The final candidate cap .01 is selected by one further
cheap normal test: five updates and14/12/14actions, unchanged roots/KKT.
The final cap/block passes all three retained native entries; the earlier trials
are retained. No option is enabled by default.

## Completed retained-entry comparison

The selected cap-.01/block diagnostic completes in2489.76s (41min30s),
2471.69s user CPU and1815.78MiB peak RSS. Each imported incoming FE reference
and obstacle is immutable; no physical history is published.

| Time (s) | KKT / unchanged target (J/m) | Full phase difference | Actions | ME / tangent seconds | Updates / line trials |
|---|---|---|---|---|---|
| 0.52929688 | 2.609e-11 / 2.653e-11 | 1.176e-10 | 189 | 272.4 / 632.4 | 7 / 12 |
| 0.53027344 | 1.939e-13 / 2.667e-11 | 1.725e-10 | 194 | 254.5 / 511.8 | 7 / 11 |
| 0.53125000 | 2.932e-13 / 2.685e-11 | 1.734e-10 | 135 | 227.7 / 363.3 | 6 / 9 |

Healing is zero; normalization stays positive. Correctly constrained bulk norms
are5.78e−11–5.96e−11N/m, surface contact RMS7.08e−12–1.08e−11. The first
KKT is close to its unchanged target, and passes original CPDI reassembly.
The selected block uses189/194/135actions versus214/214/217 in the strict
retained native evolution. Total actions decrease19.7%; this is a numerical
work comparison across scoped entries versus fresh evolution, not an identical
whole-run speedup. Setup actions are included. Extra nonlinear iterations remain
a cost. The fresh onset below supplies the end-to-end measurement.

The final executable repeats the unloaded/growing/asymmetric operator regression:
normalization relative error2.66e−15, full gradient error3.72e−15, exact normal
prototype action difference1.82e−10m², independent inner repeat6.70e−12J/m,
and mixed symmetry1.02e−10J/m. Original tangential checks pass; normal FD uses
prototype-perturbed mechanics evaluated natively, while tangential FD is at fixed
endpoint. This is not an exhaustive full-native angular FD/stability campaign.

## One targeted preconditioner candidate

Recycling the preceding symmetric phase correction has almost zero overlap
with the lowest generalized mode and adds cost in the normal shadow. The normal
derivative of phase overlaps that translation-like mode by about .99996–.99998
near convergence. This mode includes mechanical and normalization feedback in
the reduced Hessian; it cannot be diagnosed from fracture stiffness alone.

The initial single global translation improves cost, but a later native search
direction has .797 tangential variance and only .011 global-translation overlap
in the same lumped diagnostic metric. The `fixed-core-selected` cost pilot is
deliberately stopped after783.22s without publishing history; it is not a failed
physical acceptance. This supports extending the same targeted correction to
independent tangential amplitudes, rather than suppressing native modes.

The default-false `Endpoint translation preconditioner` uses the explicit trial's
normal derivative, masked to the proposed free set. Early searches use one global
direction; when eta_k<.01, use seventeen disjoint tangential nodal columns.
Normalize them and evaluate their images with the actual full native action.
Only the small symmetric coarse matrix is diagonalized. Resolved positive eigenpairs define Q; unresolved/negative
coarse eigenpairs are omitted from the preconditioner only. A combined image
guard triggers the independent soft-mode check described below. With C the existing
positive search inverse, apply

    C_coarse = (I-Q H_red) C (I-H_red Q) + Q.

This SPD numerical inverse retains every native degree of freedom and the actual
Hessian. The left factor is implemented as the transpose of the right factor
for the inexact action. Its images refresh for every new phase/free set. This
small block is not a dense native/prototype Hessian. The combined image bound is
conservative for a soft direction formed by cancellation: the first block pilot
retains16/17modes and is stopped as a cost diagnostic. Positive modes below that
combined guard are now checked by a direct native action on the rotated mode;
the direct curvature must exceed four times its mode-wise action estimate and
assembled-versus-direct uncertainty. No physical criterion is relaxed.
Actual CG curvature and independent
native finite-difference stability checks remain unchanged. The obstacle predictor
continues using its original positive metric. This is not a centering condition.

## True-energy arithmetic diagnosis

The forcing-reference saddle run reaches its3600.67-s budget without passing
KKT (last scaled defect1.2879e−11 against5.4395e−13). It is not declared a
physical instability or a nonlinear stall: the residual was still decreasing,
and no history was published. Late updates are repeatedly halved by the energy
line search. The old implementation adds8700small fracture increments directly
to the large reference constant; its summation estimate exceeds the unchanged
line-search allowance. This is distinct from linear-action uncertainty.

The numerical repair sums the actual linear/quadratic CPDI energy increments
before adding the reference, using extended accumulation. Bulk/cohesive storage
and friction use extended accumulation on the same points. No integrand, spatial
rule, gradient, Hessian, energy acceptance allowance, KKT target or physical law
is changed. Old-versus-new sums are retained as diagnostics. Native verification passes: bulk-energy operator disagreement falls2.47e−12→
9.84e−14J/m with unchanged gradients/actions. The native branch residual
2.707e−9→8.487e−12→1.867e−14 (scaled) replaces the old repeated halvings.
The initial-only onset launch is stopped/preserved; final fresh onset uses this
build. Observed old/new bulk sums reach1e−12J/m; fracture sums differ4.88e−13
J/m, comparable with the original4.55e−13-J/m line allowance. The3.12e−11-J/m
addition estimate is a conservative bound, not a measured error floor.

## Fixed-entry diagnostic ownership

The replay installs a scoped FE reference and its consistent previous q/I,
using the matching retained discrete prototype data. This is explicitly diagnostic
setup, not physical initialization, checkpoint conversion or history transport.
R_a, Z_a and the previous-phase obstacle remain fixed throughout every native
trial; the minimizer runs with publication disabled. Scratch fields, reference,
clock, statistics and surface data are restored after each case. Particle history
is untouched. The first two cap-.25 controls printed an incorrect raw bulk residual because
the new diagnostic reader omitted the fault source and retained the physical
boundary lift. The corrected reader uses the production source and homogeneous
test constraints; it changes no residual assembly or physical acceptance. Those
old measurements remain labelled instrumentation failures, not mechanical
failures.

Actual native fields and residuals are compared with the retained
reference, separately from fresh native evolution.

The clearly resolved saddle control starts from the verified symmetric accepted
state at .5615234375 s and uses one diagnostic dt=1/256 s to .5654296875 s.
It retains the continuous .256 t shear ramp and applies its interval mean.
The first native diagnostic resolves negative curvature (−.28020-Pa derivative
probe, with1.14e−6-J/m action uncertainty), departs with an admissible energy
decrease, then exposes the stationary-start forcing issue above. It is stopped
as a bounded noncommitting control, and retained. The forcing-reference repair
is applied before the final diagnostic.

The prototype symmetric entry has mass-normalized curvature −.42128 Pa and
translation overlap .9999727. Constrained minimization finds a stable branch
with an 8.81e−9-J/m energy decrease. This avoids further microscopic near-release
searches. It does not demonstrate a physical jump or fresh native continuation.

## Completed native saddle escape

The resolved fixed-entry diagnostic `fixed-saddle-summation` converges in1564.14s
(26min4s),1538.97s user CPU,1807.35MiB peak RSS. Every trial uses the same incoming
state at .5615234375s and physical obstacle; diagnostic dt=1/256 to .5654296875s.
This explicitly larger diagnostic step avoids microscopic first-release gains;
it is not a fresh1/1024 trajectory or qualification of its first release.

| Measure | Native result / scope |
|---|---|
| Initial KKT / stability | Stationary; derivative-mode L2 quotient−.28020Pa, independently resolved negative |
| Energy departure | 22.8955205557656→22.8955205469501J/m; decrease8.81543e−9J/m, about19,385times the unchanged allowance |
| Final original CPDI KKT / target | 1.20164e−12 /3.48127e−11J/m |
| Full phase / tangential variation | Difference5.05e−11 from matching positive branch; actual tangential variation2.00e−13, not suppressed |
| Centroid | .500126224153m; reference .500126224139m; no reflection/recentering needed |
| Bounds / normalization | Zero healing; positive I≥149.51177m; maximum phase increment .00618032 |
| Native endpoint residuals | Bulk2.36e−10N/m; contact RMS8.91e−12N/m |
| Final known-mode curvature | L2 quotient15.99224Pa; curvature4.61855J/m versus1.17e−6-J/m action uncertainty |
| Endpoint q / interval V / candidate slip | 264.73646Pa / .14558871m/s / .03965253m; no physical publication |
| Solve work | 355actions/1010inner calls; tangent1082.1s, ME349.3s; eight updates, nine line trials |

The native endpoint matches the prototype's stable branch, and the known native
unstable direction becomes positive. This is sampled admissible stability;
it does not establish the whole native angular critical-cone spectrum. Full
native degrees are retained. The old forcing/budget failures remain preserved;
no failed state is labelled accepted. The fixed diagnostic restores its scoped
histories and does not substitute for fresh evolution.

## Fresh economical-prefix qualification

`onset-economical-resumed` completes58fresh physical steps to .53125s with the
same154-step economical prototype schedule:5×1/64 to .078125s,16×1/1024 through
yield to .09375s,26×1/64 to .5s,3×1/128 to .5234375s, then8×1/1024.
The continuous loading and its interval means remain identical. Initial phase
matches exactly; shared phase/H input hashes match all retained fresh fixtures.
Zero accepted FE history/Z is reconstructed natively, without checkpoint conversion.

The original `onset-economical-final` reveals the inherited91% timestep-growth
cap and is preserved as a fixture failure. The explicit case setting1500%
allows the specified1/1024→1/64 return; the global default91% is unchanged.
A default-empty benchmark schedule-file check verifies exact times/dt before
solving. The interrupted `onset-economical-matched` retains54accepted steps and
no final cost/recovery qualification. Its phase/surface fields reproduce exactly
in the resumed fresh run through .52734375s; this is a repeat, not restart.

| Unaligned quantity | Maximum native/prototype difference | Existing target |
|---|---:|---:|
| Full phase | 9.00e−9 | .001 |
| Cohesive traction | 9.60e−6Pa | 1Pa |
| Cumulative slip | 6.26e−10m | 4.88e−5m |
| Common-interval slip rate | 4.62e−7m/s | 8.06e−4m/s |
| Whole/local stress RMS | .0005605/.00026995Pa | 13.325Pa each |
| Bulk storage | 1.17e−7J/m | .008878J/m |

Stress is independently sampled at12off-assembly normal points in three
native tangential sections at .5,.52734375,.5283203125,.529296875,.5302734375
and .53125s. These are sampled-state comparisons, not continuous-time stress
bounds. Full phase, surface fields and storage are checked at every accepted time.
All actual published KKT targets pass. Maximum physical KKT1.869e−11J/m occurs
at step55 against2.646e−11; final2.398e−13 against2.689e−11. FE recurrence
error≤1.735e−18m, healing zero, I≥108.64890m. Tangential phase variation
1.133e−8 and normal stress .001473Pa are measured, not suppressed.

At the fixed1e−6 growth threshold, first full-field growth is bracketed by
[.52734375,.5283203125]s and first core growth by [.5283203125,.529296875]s.
Final maximum lies at y=.5m; its nodal x=.0625m reflects tiny tangential
variation, not resolved positional selection. Final center q=312.15635Pa,
cumulative slip=.03480758m, last **interval** V=.13729071m/s, I=111.31900m.
No frame is reflected/recentered. This is early widening of a prescribed band.

Complete outer failure after convergence and before publication restores all
solution histories, accepted d_R, surface q/I/V, particle properties/positions,
clock/bookkeeping and statistics exactly. All520surface components match by
bytes, including325NaN sentinels. Pre-failure phase histories genuinely differ
by6.994e−7. Step55 retries with the identical residual sequence and publishes
once. Postpublication rollback, physical restart and transported/birth histories
remain unqualified.

## Complete-prefix energy assessment

All58steps use the same documented diagnostic path: linear phase interpolation,
exact Maxwell R/Z at the fixed interval velocity, actual normalization derivatives,
and matched spatial measures. Intermediate mechanical/contact/phase imbalance
is measured. This path is a diagnostic reconstruction, not exact intermediate
equilibrium. Boundary-reaction work41.2408988J/m supplies storage6.4162193J/m,
bulk/cohesive viscous loss .00579145/.01138348J/m and friction34.8075766J/m.
Bulk/cohesive/fracture storage changes are .86088043/5.42354986/.13178900J/m.

| Residual work (J/m) | Signed cumulative | Sum absolute step | Max step | Integral absolute power |
|---|---:|---:|---:|---:|
| Mechanical | +1.40686e−4 | 1.42493e−4 | 5.11139e−5 | 1.43034e−4 |
| Friction | −3.03408e−6 | 1.23737e−4 | 6.02974e−5 | 1.23737e−4 |
| Phase | −6.56353e−5 | 6.57721e−5 | 2.88734e−5 | 7.04373e−5 |

These terms cancel substantially: external remainder(storage−reaction work+loss)
is +7.20170e−5J/m, sum absolute steps7.39269e−5. The independent identity
remainder is5.84e−13signed,3.06e−12sum absolute,3.11e−13maximum step.
This verifies accounting, not exact finite-step physical energy closure.
Bulk-rate versus boundary-reaction work differs1.40686e−4J/m; the lifting integral
is separately retained. Bulk profile work−7.77192e−5 and cohesive normalization
work−.13177692J/m are explicitly included; the latter accompanies fracture
expenditure and is not relabelled as dissipation.

Signed native/prototype residual-work differences are6.78e−7mechanical,
9.02e−13friction and9.15e−12phase J/m. Summed absolute mechanical-power difference
1.562e−6J/m reflects small native reaction/nonlinear errors, with negligible
bulk-rate work difference. The absolute phase-power discrepancy7.166e−8J/m
is traced to a sign crossing in the first growth interval: frozen matching paths
at Q8/Q16 reproduce it. Q32/Q64 differ3.23e−9J/m; signed integrals vary below
5e−16. This is time quadrature of a nonsmooth absolute value, distinct from
spatial normalization accuracy or timestep error. At the final smooth native
interval, Q8/Q16 differences≤8.31e−14J/m. No operator/tolerance is changed.
The economical prototype's full-interval residual-work/refinement gate remains
qualified; one native schedule does not establish a native convergence order.

[Phase, traction–slip, profiles, stress and storage charts](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/onset-economical-resumed-comparison.png).
[Separate signed/absolute work and profile-energy charts](../../../benchmarks/reconstructed_fault/propagation/native_solver_evidence/onset-economical-resumed-energy.png).

## Measured cost and prepared full-run decision

End-to-end5108.59s,4738.74s user CPU,30.90s system,1725380KiB peak RSS.
The one-second observer measures nonoverlapping entry-to-commit time:

| Core time (s) | Step seconds | Actions | ME/tangent seconds | Phase iterations / line trials |
|---|---:|---:|---:|---:|
| .529296875 | 940.9 | 216 | 266.1/624.3 | 7/9 |
| .5302734375 | 973.1 | 156 | 309.6/599.9 | 7/9 |
| .53125 | 1005.4 | 161 | 316.4/627.8 | 7/9 |

Complete run611actions/1517inner calls,2008.37s tangent wall including the failed
attempt;58successful cost records omit that attempt. There are262mechanical
Newton iteration reports,82reported phase iterations and33line trials on
successful solves. Native timers:574rhs assemblies/1080s,60full Stokes
assemblies/111s,27tangent setups/99.8s,1718condensed solves/1630s,
2128pullbacks/428s. These timers nest; do not sum them into a second wall total.
The remaining bottleneck is mechanical action/assembly cost. The retained-entry
19.7% action reduction is not a whole-run speedup: fresh economical and old
coarse-prefix runs have different schedules/incoming states. The measured fresh
steps remain15.7–16.8min each.

The prepared `native_solver_evidence/economical-full-planned.prm` starts fresh,
uses154exact end times, and extends the same loading/material to .625s.96fine
growing steps remain after the onset prefix. Straight extrapolation26.56h,
including prefix; allow **27–35h** for later conditioning, saddle and diagnostics,
with a proposed36h wall bound and2.5GiB memory reservation. This is a cost plan,
not a measured late trajectory or an authorization to launch it in this task.
The run is **not launched**. Before a future launch, extend the completion check
from .53125 to .625 and consistently compare either one reflected branch over
the whole history, never individual recentered frames.

Stop on unresolved descent, original KKT/contact/normalization/bounds/history
failure and preserve the entry. Do not search microscopic gains below the
arithmetic allowance. The larger diagnostic saddle does not qualify fresh first
release at1/1024s. Full native angular critical-cone stability, positional/spatial
selection, coupled restart, postpublication rejection and transport/birth remain
separate. This task supports the next native P2 verification; it does not qualify
P3 or a gradual-damage material.

## Source purposes

| File / function | Purpose |
|---|---|
| phase_field.h; phase_field.cc, SolverParameters declaration/parse | Two default-false numerical options; no new physical parameters |
| reconstructed_fault_endpoint_phase.cc, solve_reconstructed_fault_endpoint_phase | Adaptive intermediate stopping, one balanced coarse inverse, unchanged true-energy/KKT/curvature; optional private noncommitting diagnostic start |
| simulator.h; reconstructed_fault_stokes.cc, endpoint_mechanical_defects | Private test seam for actual native endpoint residuals; no new physical solver |
| fe_strain.cc, evaluate_fe_phase_energy_at / make_fe_endpoint_trial | Extended summation of the same discrete storage/friction terms; no integrand or derivative change |
| phase_field_fault.h; fe_strain.cc, set_fe_diagnostic_reference | Private scoped fixed-entry FE anchor with unchanged geometry; no checkpoint or particle conversion |
| tests/phase_field_test_access.h; phase_field_fault_test_access.h | Scoped test clock/reference and noncommitting entry access/restoration |
| native_endpoint_verification.cc | Native retained-entry/escape fields, energy, residual and immutability checks; optional full-prefix energy ledger and pre-step exact reference-schedule check |
| assess_native_solver_cases.py; assess_native_tangent_cost.py | Fixed-entry KKT, admissibility, fields, conditioning and measured cost, without fresh-evolution claims |
| assess_native_solver_onset.py | Exact physical schedule, immutable phase bounds, positive normalization, distinct pre-failure histories, successful retry and independent residual-work channels |
| check_native_solver_accuracy.py | Normal-only intermediate accuracy/slow-mode comparison, no publication |
| prepare_native_solver_cases.py; run_native_endpoint.py | Explicit fixed/fresh fixtures and unique bounded run provenance |
| monitor_native_endpoint.py | Append-only one-second cost observer, avoiding repeated whole-log reads |
| assess_native_endpoint_run.py | Matching-schedule reference argument and maxima over actual exported stress samples, with explicit sampling times; preserving the existing default comparator |
| specification.tex; current_design.md; STATUS/PLAN | Numerical algorithm and qualification scope; governing energy unchanged |

Entry hashes/strict source, fixtures and new evidence are retained under
`benchmarks/reconstructed_fault/propagation/native_solver_evidence/`.
Native logs and run JSONs retain unique tags under `native_endpoint_evidence/`.
Baseline HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6, fault-propagation,
existing worktree; GCC12.4/dealII9.6.2/Trilinos14.2/p4est2.8.7; one rank/thread.
No VTUs, material tuning, repeated full native trajectories or P3.

## Reproduction and verification scope

The final comparison uses one rank/thread with GCC12.4, deal.II9.6.2,
Trilinos14.2 and p4est2.8.7. The release executable is the existing propagation
build. Unique start/run JSON files record SHA256 of the executable, plugin,
actual copied input and shared phase/H definitions. The exact endpoint source
change against the retained task-entry snapshot is in
`native_solver_evidence/endpoint-phase-task.diff`; final source hashes are in
`source-provenance.json`. No incoming manual notation edit is removed.

```sh
python3 benchmarks/reconstructed_fault/propagation/run_native_endpoint.py onset \
  --tag economical-resumed \
  --input benchmarks/reconstructed_fault/propagation/native_solver_evidence/economical-onset-coarse.prm \
  --max-wall-seconds 5400 --force-complete-failure --failure-step 55
python3 benchmarks/reconstructed_fault/propagation/assess_native_endpoint_run.py \
  onset-economical-resumed \
  --prototype benchmarks/reconstructed_fault/propagation/native_tangent_evidence/economical-1
python3 benchmarks/reconstructed_fault/propagation/assess_native_solver_onset.py \
  onset-economical-resumed
python3 benchmarks/reconstructed_fault/propagation/assess_native_tangent_cost.py \
  onset-economical-resumed \
  --output benchmarks/reconstructed_fault/propagation/native_solver_evidence/economical-resumed-cost.json
```

The actual ASPECT compatibility command is `aspect-release --test
'*Fault*,*fault*,Stage-I*'`:21,105 assertions/54 cases pass after the final energy
summation repair. `bin/unit_tests` is the separate World Builder binary and is
not counted as ASPECT qualification. Relevant Python files compile; the standalone
FE derivation passes two LaTeX runs. Standalone `specification.tex` retains its
pre-existing undefined `\si` dependency and is not reported as compiled.
A preservation check finds all471 historical benchmark/script files in the
matched-initialization manifest unchanged and present. That manifest also hashes
production files intentionally changed by later authorized native work; those
are inventoried separately, rather than silently discarded or reported unchanged.
No new tests or full trajectories are launched to requalify unrelated evidence.
