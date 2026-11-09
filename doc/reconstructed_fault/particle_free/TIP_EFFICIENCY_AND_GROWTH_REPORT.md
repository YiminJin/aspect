# Bulk-tip efficiency and localized growth — completed 7 November 2026

1. **Repeated failed exact searches became cheaper:** progress-based search
   ordering reduces the difficult step from 200.51 s accounted work to 101.36 s
   step time, **1.98×**. The next two steps and common-parent half-step path pass.
   Bulk stress, cohesive strain, phase and slip remain close to the control;
   near-zero-rate section resistance differs by up to 35.35 Pa. This is a
   retained efficiency improvement with an explicit reaction-accuracy limit,
   not uniform equivalence of every field or proof of a unique stationary state.
2. **Halving dt halves physical work error**, from .00174784 to .000874876 J/m
   over the same .0025 s interval. The latter is still 5.17 times added fracture.
   Extrapolating this local first-order trend suggests about 103 substeps and
   2.35 hours for that interval at a provisional 10%-of-fracture budget.
   That cost is a planning estimate, not a qualified smaller-dt result.
3. **No resolved axial advance of the measured weakened front occurs** before
   the remote-intact margin is exhausted at .09 s. Additional irreversible
   damage develops in the existing halo. Remote phase remains zero, but
   max Y/Kf reaches .98149, triggering the declared .95 diagnostic stop.

## Baseline, ownership and evidence

Entry: `d481a09c1415578ec6fb148e6b9e13c80184b6c4`, branch
`pf-rsf-particle-free`, clean tracked worktree in the existing repository.
The entry core `1cc27f9c…7cac6`, checker and sources are preserved in
`build-particle-free/bulk-tip/controls/efficiency-entry/`. The final core is
`8a652f3ec0ce1efda2608db133376bfae21551aa0205a7c96f3fed8a66989115`;
physical comparisons used checker
`9b20c83a52600db345f4cb7f08340aa345f0d920556630d77227461d71e6416f`.
Their preserved pair is `controls/efficiency-polished/`. The subsequently rebuilt
checker only adds an optional zero-clock seed fracture-gradient export; its
identity and seed-only check are recorded separately. The cached executable
banner is not a source identity.

Local [evidence and manifest](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency/closeout.json)
retain source/input hashes, commands, exact binaries, failures and reduced data.
Each `build-particle-free/bulk-tip/runs/efficiency-*/command.json` records its
arguments and executable/checker hashes. Wrapped logs/records are under
`benchmarks/reconstructed_fault/refactoring_boundary/evidence/tip-efficiency-*`.
The [previous report](TIP_SOLVER_INTEGRATION_REPORT.md) remains the authority for
original fine-trajectory ancestry: this is a continuation chain, not a fresh
uninterrupted trajectory of the final executable.

Snapshot 3 of `integration-fine02/output` really contains accepted step 23 at
.057500000000000016 s. `efficiency-control02` reproduces retained
`integration-fine03` step 24 at .06000000000000002 bitwise in exported native
point/section fields and all stored energies. No substitute initialization was
used. For the half-step path, the existing post-resume signal changes only the
pending interval; the intermediate obstacle, stress/cohesive histories, age and
slip advance normally.

The ordinary simulator scheme still owns iteration and acceptance. Material
laws, condensed section closure, direct tau/A histories, DGQ2/SIPG quadrature,
BE endpoint/age convention, physical inputs and checkpoint meanings are unchanged.
Mechanics 1e-9, phase KKT 1e-7, dissipation and scaled ledger 1e-10 gates remain.
No global moving/turning/branching closure or new time integrator is introduced.

## Measured cause and retained algorithm

The control requires 28 nonlinear iterations, 19,964 outer FGMRES iterations,
19,965 Stokes calls, and 221,459 inner Stokes iterations. There are 564 active
passes, 376,478 released/reclassified coordinates, four repeated sets, 26 failed
search corrections and 960 line trials. Exact searches cost 175.56 s; shifted
searches 10.85 s; tangent evaluation 2.40 s; trial evaluation 11.67 s.
The Stokes adapter costs 111.04 s (ordinary Stokes 102.43 s inside it).
Phase AMG setup costs only 3.18 s over 558 builds. These are **nested timers**:
Stokes and AMG costs are already inside search costs and must not be summed as
independent fractions. The full control wrapper costs 202.83 s; disjoint solver
scopes account for 200.51 s. Repeated unsuccessful exact VI searches dominate.

The constrained free tangent is J=[[J_mm,J_mb],[J_bm,J_bb]], including condensed
section derivatives. The exact action stays `BulkTipCoupling::vmult`; variables
remain displacement increment, p/G and Bernstein phase, with existing mechanics
and 1/Kf phase residual scalings. The retained block-diagonal preconditioner uses
ordinary ASPECT Stokes and positive fracture-plus-mass phase AMG. FGMRES, the
strict linear tolerance and 1e-4 inner Stokes tolerance are unchanged.

`compute_bulk_tip_search_direction` now names a nonlinear search correction
unambiguously. Only search **ordering** changes:

- Start every physical step with exact Newton VI first.
- After an accepted positive-metric correction, prefer the shifted family on
  the next iterate, always restarting at the least shift in 1,10,…,1e8.
- With r=max(mechanics/1e-9,KKT/1e-7), retry exact first when
  r ≤ max(10, .1 r_last_exact). The near-gate clause preserves exact polishing.
- Exact remains fallback if all shifted searches fail. A successful exact
  correction restores exact-first order. Recompute bounds, corrections and
  functional/residual acceptance at every iterate; cache no active set/history.

The [discretization guide](tip_treatment/discretization.md) maps exact blocks,
ASPECT inverse calls, tolerances and publication. The physical owner publishes
once after the original gates; search preference and counters are not serialized.

| Candidate (all evidence retained) | Outcome |
|---|---|
| Upper triangular inverse: z_b=P_b^-1 r_b, z_m=M^-1(r_m−J_mb z_b) | 19,691 outer iterations, but 258.35 s versus 202.83 s: 27.4% slower; removed |
| Reuse successful shift=100 | 80-iteration failure, KKT .08574; 56.13 s; removed |
| Always prefer shifted family | .06 and .0625 pass, .065 fails at KKT .000367645; 138.44 s total; removed |
| Tenfold-progress retry without near-gate polishing | Three full steps pass; second half-step stalls near KKT 1.34e-7; removed |
| Progress retry plus original near-gate exact ordering | Retained; full continuation, two halves and independent checks pass |

The triangular candidate used an exact phase-only outer action for J_mb, with
active rows excluded and identity output there; fixed bound displacements were
already in the RHS. It added no Stokes application but its extra action cost
outweighed the iteration reduction. No inexact forcing policy was introduced:
measurements support avoiding failed searches, not a tolerance relaxation.

## Continuation and numerical comparison

`efficiency-final01` continues the identical parent through .065 continuously.
Times below exclude restart/setup; its complete wrapper is 196.49 s.

| Endpoint | Nonlinear iterations | Outer FGMRES | Stokes calls / inner iterations | Step time |
|---|---:|---:|---:|---:|
| .06 | 43 | 9078 | 9079 / 100725 | 101.356 s |
| .0625 | 26 | 4771 | 4772 / 52623 | 53.3445 s |
| .065 | 19 | 3631 | 3632 / 40013 | 37.9126 s |

At .06 there are 12 exact attempts, 11 failed corrections, 237 active passes,
two repeated sets and 1095 line trials. Savings come despite more nonlinear
iterations/trials. The approximate 1.98× comparison uses disjoint control scopes
versus the new step timer, not a repeated statistical timing study.

Across these endpoints, max phase difference from the retained reference is
8.49e-10, max shear-stress difference 2.61e-5 Pa, max section-rate difference
1.60e-9 m/s, max slip difference 3.99e-12 m. At .06 max Axy difference is
9.54e-12 and max age difference 4.00e-9 s. Stored-energy differences there are
−5.21e-10 bulk, +3.88e-11 cohesive, +1.72e-10 fracture J/m.
Full tensors, sections, residuals and component differences are in
[continuation comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency/final-continuation-comparison.json).

Resistance differs by **35.34594, 21.17466 and .03511 Pa** at the three endpoints;
cohesive conjugate stress also changes in nearly intact material. At the worst
.06 section x=1.629220655, V changes from 6.80882e-13 to 6.45211e-14 m/s,
and resistance from 283.5052 to 248.1592 Pa. An independent regularized RSF
calculation reproduces both to 1.14e-13 Pa; p a log(V_new/V_old) explains the
−35.34594 Pa difference. Where max(|V|)>1e-6 m/s, resistance differences stay
below 1.98e-5 Pa. This cutoff is only a reported diagnostic: no rates are clipped.
The accepted KKT tolerance does not uniformly control the near-intact reaction.
This sensitivity prevents a blanket full-state equivalence claim; uniqueness or
stability of the nonconvex physical stationary state has not been established.

## Same-parent time accuracy and cost

`efficiency-half03` takes two normal .00125 s steps from the same original parent
through .06. Its step times sum to 163.965 s (wrapper 167.78 s), versus 101.356 s
for one .0025 step. Maximum half-path mechanics/KKT/ledger residuals are
2.88e-15 / 4.09e-10 / 1.53e-15; the full step has
2.19e-14 / 7.61e-8 / 3.81e-16. Algebraic closure is not physical-work accuracy.

| Integrated component (J/m) | Full step signed | Two halves signed | Accumulated absolute |
|---|---:|---:|---|
| Independent physical remainder | .00174783606 | .000874875585 | Same as signed, both paths |
| Stress BE | .00160930094 | .000804996915 | Same as signed |
| Cohesive BE | .000126555739 | .0000632274643 | Same as signed |
| Compliance endpoint | −1.18536691e-7 | −6.22223073e-8 | Positive magnitudes |
| Fracture quadratic | 1.20979522e-5 | 6.71342931e-6 | Same as signed |
| Phase residual work | 3.41190548e-11 | 9.65688191e-13 | Same as signed |
| Mechanical residual work | 1.26759714e-13 | −1.46133106e-14 | Positive magnitudes |
| Independent ledger error | 3.80918307e-16 | 2.64265779e-15 | Same as signed |
| Added fracture | .000170567769 | .000169269812 | Storage difference |

Halving dt changes V by .2983% relative L2, cumulative slip by .01669%, age by
.0003383%; max phase difference is 4.88e-6 and shear-stress L2 difference
.04858 Pa·m. Added fracture falls .761%. Bulk/cohesive/fracture storage changes
are +2.779e-5/−5.488e-6/−1.298e-6 J/m. Force relative L2 changes 2.11%, again
strongly affected by tiny-rate sensitivity. Native weighted norms and all signed/
absolute sums are in [time comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency/time-comparison.json).

Define active work explicitly as sum dt ds f V = friction+radiation: .0193447
and .0193238 J/m. Remainder is about 9.04% and 4.53% of this scale, but 1025%
and 517% of added fracture. The measured error ratio is 1.99781. A provisional
10%-of-added-fracture target extrapolates to 103.37 substeps over .0025 s,
dt≈2.42e-5 s and 2.35 hours at the measured half-step cost per substep.
Only two levels support local first-order scaling; neither a further 50-fold
refinement nor constant cost is validated. Parent time error remains. Affordable
BE accuracy at the fracture-work scale is **not demonstrated**. A separate time-
integration derivation is justified before expensive propagation; no BE term is
subtracted to claim an energy-conserving trajectory.

[Cost/time figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency/cost-and-time.png)
and [planning calculation](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency/cost-planning.json).

## Localized growth and actual front balance

Both final-source localized controls to .08 (dt=.005 and .0025, h=.25) reproduce
retained fields, sections and stored energies bitwise. `efficiency-growth02`
restores snapshot 3 of `integration-localized-half01` at .08 with identical
physical inputs. Its four accepted steps reach .0825, .085, .0875 and .09;
remote max Y/Kf is .790740, .852003, .915588 and **.981490**, respectively.
The final accepted endpoint triggers the benchmark stop. Remote phase stays zero.
The separate remote-damage trigger is max phi>1e-12 (diagnostic roundoff threshold).
The final solver and diagnostic-control growth replay agree bitwise.

From .08 to .09, integral damage grows 1.37701e-4 m², maximum phase increment
is .000578510, and added fracture is 6.22934e-5 J/m. Maximum cumulative slip
is .000294220 m. Physical remainder is .00767750 J/m: stress BE .00733200,
cohesive BE .000338847, compliance −8.4790e-9 and fracture quadratic 6.66056e-6.
Active work is .118624 J/m. This is substantial BE error alongside small fracture
increment, despite a converged algebraic balance.

| Contour | Native x extent at both .08 and .09 (m) |
|---|---|
| phi>.001 | [.904827,3.095173] |
| phi>.01 | [1.154827,2.845173] |
| phi>.1 | [1.258441,2.741559] |
| phi>.5 | [1.542349,2.457651] |
| phi>.99 | Empty |
| H>.5 | [1.154827,2.845173] |
| H>.9 | [1.241559,2.758441] |
| H>.99 | [1.457651,2.542349] |

Here H=h/(1+h) is `Response::fraction`. Holding accepted history, current phi and
section resistance fixed, d tau_xy/d engineering-strain increment=beta G(1−H).
It measures point elimination weakening; the full coupled tangent also contains
phase and section feedback. Large H before phi≈1 does not establish propagation.
Both these mechanical contours and phase extents remain fixed at native resolution.
Gauss6 sample spacing in x ranges .016883–.059655 m on h=.25; no subcell front
accuracy is claimed. Lines connect native samples; image tiles do not smooth them.

In each front region |x−1.2|<.25 or |x−2.8|<.25, damage added since .08 is
3.64345e-5 m², and since the common zero-clock equilibrated seed 5.92185e-5 m².
At .09, max Y=4.66933 Pa, H=.988967 and |w|=.0204143/s in each region.
The actual Bernstein fracture-minus-driving residual has 40 free front rows
(max |residual|/(Kf mass)=4.82e-12) and 248 lower rows (minimum multiplier/
(Kf mass)=.00927875). This is the discrete SIPG/obstacle balance, not a comparison
of point Y with a fictitious local fracture threshold.

To include face terms, the seed-only observer supplies g_f(B_seed). Partition
actual quadratic fracture change by .5[g_f(B)+g_f(B_seed)]·(B−B_seed), assigning
rows to cells. Each front contributes 2.08016e-5 J/m; global weak work is
7.4906569046e-5 versus the energy difference 7.4906569004e-5 J/m, a 4.17e-14
closure defect. This is a specified weak-work partition, not unique point energy.
Volume-only front contributions (.00133930 J/m each) exclude SIPG faces and must
not be presented as the actual fracture-energy change.

No negative local friction work, integrated-rate cancellation or multiple active
bands appear in this four-step extension. The band diagnostic uses
|w|>max(1e-10/s,1e-6 max_section|w|); it changes no integration or physical law.
Earlier tiny opposing rates remain evidence. Full tau/A/q/w/H/Y, section V/age/
slip and fracture-gradient samples are retained in native exports.

[Front profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency/growth-profiles.png) ·
[Damage increments, rate and cohesive stress](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency/growth-increments.png) ·
[Spatial/front accounting](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_efficiency/growth.json).

## Checks, failures, preservation and next decision

- Point suite: **3319 assertions in 15 cases** pass, including bound release.
- Strict fixed-state direct/manufactured oracle: errors 3.286e-8 against 2e-5
  tolerance; weighted tangent asymmetry 5.10e-17 (318 free unknowns).
- Localized derivatives: action 2.24e-9, phase functional 6.67e-8,
  Lagrangian 1.86e-7. Active 1→2 rank restart plus rejection/retry passes;
  rollback is bitwise, final slip difference 4.34e-19 m, force 3.41e-13 Pa.
- Ordinary no-fault Stokes statistics/depth-average exports are byte-identical.
  One native DGQ2 layered mature step passes its independent reference: largest
  reported error 8.88e-16, one history/geometry publication, no retries.
- Rejected candidates and all failure outputs remain. `control01` is restore-only
  because literal .06 preceded the stored clock by roundoff. `half01` failed MPI
  socket setup before simulation; retry used the authorized wrapper. `mature01`
  lacked copied geometry and failed setup; fresh `mature02` used exact geometry.
- **21 wrapped records, 1456.59 s (24.28 min)** including failures and seed-only
  export; separate build logs span about **797 s (13.28 min)**. At most four
  build workers/two MPI ranks, one numerical thread per rank, sequential heavy
  jobs. All finished; no background simulation, server job or mature campaign.
- Protected binaries/checkpoints and prior reports remain. Deleted 39/45 pilot
  frames, early logs and the absent older `bulk_continuation.zip` remain losses;
  no historical campaign was recreated. No files are committed by this task.

Implemented: measured search ordering and compact native diagnostics. Verified:
bounded continuation, time sensitivity, operator/lifecycle and shared-solver
regressions. Physically observed: halo adjustment without resolved front advance
before remote activation threatens. Unresolved: near-intact reaction accuracy,
BE cost at fracture scale, spatial/quadrature accuracy, stability and a credible
localized propagation specimen. This is not production qualification.

The next physical experiment should first verify the intact response to compact
antisymmetric tangential loading, keeping the seed and law fixed:
v_x(top/bottom)=±.01[max(0,1−((x−2)/1.2)^2)]², v_y=0, sides fixed. Its vanishing
remote loading is intended to concentrate driving near the seed instead of
raising the whole intact specimen toward threshold; mechanical localization
must be checked, not assumed from the boundary formula. This is a concrete
proposal, **not implemented or launched**. In parallel, a separately selected
endpoint/time-integration derivation should establish a feasible work-error
budget before an expensive growth continuation. The current probe does not
prove that the constitutive model cannot propagate.
