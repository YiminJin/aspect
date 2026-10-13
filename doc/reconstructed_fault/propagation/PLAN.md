# P2 checkpoint disposition (2026-10-13)

Close the bounded P2 implementation/diagnostic task with the existing evidence;
see [P2_CLOSEOUT.md](P2_CLOSEOUT.md). Preserve reduced Newton/FGMRES, original
physical acceptance and accepted histories. Direct mechanics remains experimental,
serial and default off; quasi-Newton and unsuccessful staggered/single-pass
alternatives are inactive. Commit reusable source, fixtures, evidence and docs.
No new trajectory, arc-length, quasi-Newton or scalability work is selected.

Full native evolution and native temporal/spatial qualification, coupled restart,
transport/birth histories, broader stability/localization and P3 require a later
user selection. Earlier cost plans and suggested next tasks below are historical
context, not active authorization. No roadmap qualification is promoted here.

---
# Completed bounded P2 direct mechanical inversion (2026-10-13)

The default-off serial Endpoint direct mechanics option retains interchangeable
FGMRES and exact native surface Schur inversion with the original outer Newton,
physics, histories, bounds and final energy/KKT criteria. The paired batch and
cost-gated complete growth / retained saddle pass. No trajectory or P3 is run.

| Result | Measured evidence |
|---|---|
| Eight-RHS setup-inclusive cost | FGMRES 10.51 s; direct 6.29 s (1.67×) |
| Setup / repeat / surface size | Bulk construction + KLU 2.291 s; 65-node surface setup 2.153 s; reused solve median .03429 s; active control 64 nodes |
| Factorization-only / memory | Pure factorization timing unavailable; batch peak RSS 1789.6 MiB, retained increment 102.2 MiB; standalone entry peaks unavailable, no reruns for missing data |
| Original operators / actions | Full uncondensed checks pass; resolved action differences ≤3.13e-10 relative; independent prototype normal/translation differences ≤6.93e-9 J/m |
| Complete growth | 374.11 s versus retained 664.10 s; same 135 actions /7 iterations /9 searches; KKT 2.81e-13 J/m, zero healing, publication once |
| Known saddle | 615.71 s versus retained 1655.61 s; 316 versus 363 actions; curvature −.280202→+15.992244 Pa, same branch, no history publication |
| Operator reuse | Growth: 1 bulk /18 surface setups, 282 direct RHS; saddle: 1 /25, 649 RHS; refresh each new linearization, retain all native modes |
| Compatibility | 21,112 assertions /55 cases pass; old physics/outer Newton sources retained |

The original MPI FGMRES path and incoming manual/evidence work are preserved.
Direct inversion is qualified only for these restricted serial controls.
FGMRES complete-entry timings are retained and PMPI instrumented; the present
paired batch is the controlled cost comparison. The full trajectory, direct
MPI and broader physical stability remain unqualified. FE energy/normalization
evaluations and mechanical preparation dominate remaining cost. Stop here.

Baseline 33c0adba59132890b386f3e984b3e1650feffe44 in the existing fault-propagation
worktree. Tested executable e78821a5b12b, observer f5b7567699a8. Prior Newton
executable and serial/MPI evidence retained. Commands, input/output identities,
actual tolerances, separate cost/reuse tables, missing metrics and source purposes:
[P2_direct_mechanical_report.md](P2_direct_mechanical_report.md). Evidence:
benchmarks/reconstructed_fault/propagation/direct_mechanical_evidence/.

---
# Completed bounded P2 cheaper-coupling assessment — native screen stopped (2026-10-12)

Latest scope ends with the user's instruction to terminate the expensive current
native test and report retained data. No remaining native/rank/saddle run or full
trajectory is launched. Physics, mesh, normalization, immutable histories,
irreversibility and final KKT/energy criteria remain fixed; P3 is deferred.

| Result | Evidence / qualification |
|---|---|
| Prototype early/saddle/late QN |Pass at original KKT and admissible curvature;0.46/1.08/.50s versus1.17/3.64/1.60s |
| Four staggered correctors |Growth KKT fails; saddle tiny KKT but negative curvature, not accepted |
| Fresh QN prototype1024/2048 |.625s completed in67.34/136.51s; all7field targets pass; same-dt phase errors3.85e-8/8.14e-9 |
| Separate energy ledgers |Storage/loss/residual-work totals differ<1e-9J/m; incoming profile-cost diagnostic4.09e-7/5.80e-8J/m |
| Native1rank core6 |Interrupted at user request after2208.99s;90completed mechanical trials,169logged inner solves/5802iterations; KKT5.24e-8 versus4.20e-13 |
| Native acceptance /remaining tests |No complete state accepted/published; final fields/stability unavailable;4rank growth and1/4rank saddle not run |
| Compatibility /cleanup |21,105assertions/54cases pass; isolated interruption regression passes; launcher/ASPECT child stopped and verified absent |

QN is promising in the prototype but has no demonstrated native cost/accuracy
qualification. Retain the Newton endpoint reference. The new default-false
Endpoint quasi-Newton search changes numerical directions only; no physical
operator, parameter, bound, H law or publication rule changes. The MPI runner's
interruption handler prevents orphaned tests. Preserve all incoming MPI/manual
work and references:868source/reference hashes plus391MPI evidence/output hashes.
Baseline33c0adba59132890b386f3e984b3e1650feffe44, existing fault-propagation worktree.
Executable3257574451ef, observer8d0060654edb; commands, inputs, logs, charts,
interrupted counts and verification under cheaper_coupling_evidence/.

[Settings, algorithm, cost/energy/field tables and source purposes](P2_cheaper_coupling_report.md).
Next proposed task (not started): bounded native search-set/conditioning and
reduced directional energy-gradient diagnosis at core6. No acceptance relaxation,
new material/physical model, extra MPI campaign, full run or P3.

---
# Completed P2 single-pass / endpoint comparison (2026-10-12)

The selected prototype comparison uses identical shared initialization, matching
512-cell Q2 normal mechanics/Q1 phase, six-point bulk integration, 64-segment
consistent normalization, physical loading and immutable accepted histories.
Fresh phase-first runs at dt=1/1024 and1/2048 reach .625s; matching retained
endpoint references are reused. Native coupled/MPI source and evidence remain
unchanged; no native trajectory or task-4 mesh study is launched.

| Result | Single 1/1024 / 1/2048 |
|---|---|
| Whole prototype cost | 40.46 /71.75s; retained endpoint279.94 /383.92s |
| Maximum phase / q errors vs fine endpoint | .06261 /.05994;39.66 /36.84Pa, fail .001/1Pa targets |
| Common-interval rate / stress / storage | Fine errors1.391% /2.822% whole RMS /5.004%, fail selected targets |
| Cumulative slip | Fine .07552% error, passes .1% target |
| Incoming phase convergence / bounds / recurrence | Pass at unchanged tolerances; no healing; I≥108.64890m |
| Endpoint phase defect (splitting diagnostic) |86.1 /89.1Pa in nodal-measure scale, despite converged incoming subproblem |
| Phase release / raw external remainder | .022997 /.022035J/m; .046160 /.044216J/m; no compensating dissipation |

Single-pass is cheaper per run but does not reach the required physical accuracy
at either tested level. Retain the coupled reference; the qualified single-pass
timestep and native cost remain unresolved. Stop after this bounded comparison.
No governing change, tolerance relaxation, material tuning, native replacement,
further MPI work or P3. Details, equations, source purposes and charts:
[P2_single_pass_report.md](P2_single_pass_report.md). Evidence: single_pass_evidence/.

---
# Completed bounded P2 MPI verification: tasks 1–3 (2026-10-12)

The selected original 16×512 Q2/Q1 fixture passes fixed operators, the complete
retained core-growth entry and noncommitting known-saddle escape at 1/2/4 ranks.
Four-rank failure on rank 3 before publication restores all tested histories and
bookkeeping exactly; retry publishes once. Governing laws, defaults, physical
bounds, normalization, immutable accepted FE/surface anchors and final KKT/energy
criteria are unchanged. No substantial growth-step speedup is established.

| Check | 1 / 2 / 4 ranks |
|---|---|
| 15 fixed operators, whole suite | 133 / 105 / 79 s; max rank energy error 3.55e−15 J/m versus 4.55e−13 allowance |
| Complete growing entry | 664 / 850 / 717 s; 135 / 136 / 140 tangent actions; zero healing, original KKT passes |
| Known saddle, scoped noncommitting entry | 1656 / 2190 / 1466 s; 363 / 342 / 260 actions; same branch, known-mode curvature positive |
| Four-rank recovery | 713 s failed attempt +717 s successful retry; all tested state bytes restored |
| Compatibility / final label-only build | 21,105 assertions in 54 cases; all 15 four-rank operators/gradients unchanged |

Baseline `33c0adba59132890b386f3e984b3e1650feffe44`, branch `fault-propagation`,
existing worktree. Serial binaries and 1505 baseline files are preserved. Matched
MPI executable e9c8b1423d21 is saved as build-propagation-integration/aspect-mpi-verified-release;
final label-only build is 8fba047bd33d. Exact hashes, inputs, commands, measured
costs/communication/memory and plots are in native_mpi_evidence/ and
[P2_native_MPI_report.md](P2_native_MPI_report.md).

**Stop after task 3.** Tangential-mesh screening, further optimization and the
fresh .625-s trajectory are deferred. This qualifies the original-mesh retained
controls, not a complete MPI trajectory, full angular stability, restart,
transport/birth, physical jump selection or P3. Earlier broader authorization
below is historical and does not authorize deferred tasks.

---
# Historical MPI/cost request (narrowed to tasks 1–3) (2026-10-12)

The current user selection authorizes MPI correctness, matched 1/2/4-rank costs,
16/8/4 by 512 tangential-mesh screening and one conditional four-rank full run.
Reuse the qualified 154-step schedule. Keep normal resolution, surface segments,
particles, quadrature, physical model and final acceptance fixed. Establish
correctness before optimization; launch only after measured cost fits the
proposed 12-hour bound. If it does not, report the bottleneck and estimate.
The serial results below remain the reproducible reference, not superseded
qualifications. This authorization extends P2 only.

---
# Completed P2 native solver accuracy / escape task (2026-10-12)

The bounded task passes without changing governing laws, physical settings,
16×512Q2/Q1 discretization, accepted FE d_R/surface Z, physical phase obstacle,
true-energy search or final KKT. Two numerical options remain default false.

| Result | Evidence / scope |
|---|---|
| Retained three core entries | Original KKT passes; zero healing;518 versus645strict actions(19.7% fewer), scoped noncommitting references |
| Resolved native saddle diagnostic | Energy departure8.815e−9J/m; finalKKT1.202e−12≤3.481e−11; matches stable prototype branch, known mode positive;26min4s/1807MiB |
| Fresh economical onset |58steps to .53125s; core .60000194→.60457720;85min9s/1684.9MiB |
| Fresh field errors |All7unaligned targets pass; phase9.00e−9,q9.60e−6Pa,slip6.26e−10m,whole/local stress .0005605/.00026995Pa |
| Histories/admissibility |Zero healing; published KKT passes; FE recurrence1.735e−18m; I≥108.64890m; exact complete-step failure/retry,520surface components |
| Complete-prefix residual works |Mechanical +1.407e−4,friction−3.034e−6,phase−6.564e−5J/m; absolute powers separate; no exact-closure claim |
| Operators/compatibility |Retained unloaded/growing/asymmetric regressions;21,105assertions/54ASPECTcases pass; full angular spectrum unqualified |

Adaptive intermediate accuracy and one balanced tangential translation inverse
retain actual Hessian/full native modes. Extended sums of the same energy resolve
line-search arithmetic loss without changing integrands, allowances or physics.
The stationary-start forcing reference restarts after escape; physical targets
and immutable incoming histories never restart. The diagnostic saddle uses an
explicit1/256step, not fresh first release at1/1024. Fresh growth costs15.7–16.8min
per step;611total actions/1517inner calls include controlled failure overhead.

A154-step fresh .625s input is prepared, **27–35h estimated**(proposed36h bound,
2.5GiB reservation), and **not launched**. It needs fresh first-release and full
native branch verification; one onset schedule is not a native temporal order.
Native angular stability, positional/spatial selection, coupled restart,
postpublication rejection, transport/birth and P3 remain separate requirements.
Historical coarse-prefix15.3446J/m failure, stopped controls,91%-growth-cap
fixture failure and interrupted matched prefix retain their original status.
No histories are converted to resume a run. All471historical benchmark/script
files in the preservation manifest remain unchanged and present.

[Model, algorithm, source purposes, costs and commands](P2_native_solver_report.md).
[Accepted field charts](../../../benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/onset-economical-resumed-comparison.png).
[Separate energy charts](../../../benchmarks/reconstructed_fault/propagation/native_solver_evidence/onset-economical-resumed-energy.png).
Baseline de7beaa65846c7372711f7922abf1c5d9a745ee6, fault-propagation, existing
worktree; GCC12.4/dealII9.6.2/Trilinos14.2/p4est2.8.7;one rank/thread.
Exact inputs/executable/plugin/seed hashes and commands are retained under
native_solver_evidence/native_endpoint_evidence. The authorized task is complete;
next proposed scope is the one costed full native P2 gate, awaiting selection.
No full run, material optimization or P3 starts automatically.

---
# Completed P2 native tangent / onset recovery (2026-10-11)

The authorized bounded task completes. Governing equations, physical settings,
true outer KKT, bounds and accepted-history rules are unchanged; no P3.

Fresh native `onset-linearized-obstacle-recovery` reaches .53125 s in **3257.95 s
(54 min 18 s), 1719.9 MiB peak RSS**, within its one-hour budget. Three accepted
core increments raise peak phase .60000194→.60426168. All seven unaligned
matching-prototype targets pass; healing is zero, physical published KKT≤4.85e−13
J/m at original targets 2.44e−11–2.68e−11, and FE recurrence≤1.74e−18 m.
Complete outer-step failure before publication restores histories/bookkeeping
exactly and retries successfully (520 unchanged surface components).

| Check | Result / scope |
|---|---|
| Retained unloaded/growing/asymmetric operators | Pass; analytic normalization adjoint, full u/p/V, tangential checks and independent inner repeats |
| Native maximum differences | Phase2.17e−10; q9.96e−5Pa; slip1.09e−8m; whole/local stress .01231/.006797Pa |
| Native growing-step cost | 15–16min each; 656 total actions /1660 inner calls; no dense native Hessian |
| Economical fresh prototype | 154steps to .625s; all field and separate residual-work gates pass;149.03s /155.6MiB |
| Native fine-window energies | Separate channels match prototype≤5.4e−10J/m; residual works retained |
| Coarse loading prefix | Historical15.3446J/m contact-path error remains; no cumulative energy qualification |

Linear residual refinement fixes the retained inner-accuracy failure. Existing
exact sparse B/G and homogeneous velocity preconditioning reduce action cost.
Fixed-entry shadow identifies slow layerwise admission (19 updates); all-iterate
obstacle prediction with true active-column Newton forcing reaches the same
root in four updates. Energy, physical Hessian, tolerances and histories remain
fixed. Compatibility:21,105assertions/54cases pass. Historical failed diagnostics
and cost controls retain their original status.

Baseline: de7beaa65846c7372711f7922abf1c5d9a745ee6, `fault-propagation`, existing
worktree. GCC12.4/dealII9.6.2/Trilinos14.2/p4est2.8.7; one rank/thread.
Executable/plugin/input hashes, exact commands, changed source functions and
operator/field/energy tables: [completed report](P2_native_tangent_report.md).
Evidence: `benchmarks/reconstructed_fault/propagation/native_tangent_evidence/`
and uniquely tagged native inputs/logs/accepted CSVs. No VTUs generated.

The option stays experimental/default off. Complete native .625s, saddle escape,
energetically resolved native prefix, full tangential stability, restart and
transport/birth/postpublication rejection remain unqualified. Next proposed task:
a separately bounded native check of the economical loading prefix, conditioned
on measured cost; await selection. No full640-step campaign, tuning or P3.
Historical completed reports below retain their original qualification status.

---

# P2 native consistent-endpoint bounded result (2026-10-11)

The user explicitly authorizes the default-off native consistent-endpoint
solution algorithm, unchanged governing energy/constitutive laws, shared
continuum initialization,16×512 Q2/Q1 mesh and bounded equivalence/cost assessment.
The complete matching FE prototype dt1/1024→1/2048 and inexpensive onset-schedule
accuracy gates pass. Native fixed-state operators and fresh flank growth pass
through .5283203125 s, but the3600-s cost gate interrupts the first core-changing
entry. The planned .53125-s schedule is incomplete; no640-step native launch.
Improve/test reduced-tangent conditioning and mechanical-action cost before
a new fresh core-growth comparison. Keep the default-off option experimental.
The coarse prefix's field accuracy does not qualify its reconstructed contact
path energy: the initial interval contributes15.3446J/m residual work.
Additional tangential modes stay active. Translation escape is a
sampled native stability check, not a global critical-cone proof. Preserve all
old failures/defaults and immutable accepted history; no new phase law/P3.

[Algorithm, settings, target tables and current results](P2_native_endpoint_report.md).
This latest user selection supersedes the prior prototype-only stop; historical
results and their qualifications below remain unchanged.

---

# P2 matched-initialization decision (2026-10-10)

The selected common continuum initialization, fresh511/512/1024 exact-bulk
controls atdt=1/1024, shape/position separation, complete512 FE spatial assessment
and actual compact ASPECT postprocessing are complete. All practical spatial
phase/traction/slip/stress/storage targets pass for the retained512 FE case.
This changes the next proposed gate from basic spatial adequacy to optional
native consistent-endpoint operator/accepted-state integration.

Band position remains unqualified: peak excursion decreases .739→.262mm under
512→1024 refinement, but first saddle/recentering depend on placement. Aligned
shape converges much better than ordinary fields; conditional continuum
translation does not authorize a centering constraint or physical branch rule.
Stop microscopic near-release gain searches. Preserve old failures and incoming
work. Residual work remains separately finite at this timestep; no full energetic
or temporal qualification follows from spatial passes. No model/default/production
change, material tuning, general transport/MPI campaign or P3.

[Decision, cost caveat and proposed native scope](P2_matched_initialization_report.md).
Native integration is recommended, not executed by this task.

---

# P2 asymmetric diagnostic continuation result (2026-10-10)

The selected translation/admissibility derivation, deterministic two-direction
minimization,256/512/shifted 511/exact-bulk controls, physical mass-normalized
curvatures and continued diagnostic branches are complete. All 14 paths reach
.625s. The historical saddle stop is preserved rather than reclassified as a
physical pass. Separate residual works decrease under halving, but centroid
motion/release timing remain mesh dependent; the 512 exact control also loses
symmetric stability. Very close release probes retain arithmetic/weak-contact
uncertainty, so neither a finite jump nor a unique physical asymmetry is selected.

This changes the plan's blocker from existence of stable continuations to
spatial qualification and justified physical branch selection. The bounded next
question is phase-front/obstacle discretization versus initialization dependence,
not a new kinetic law or centering constraint. It is proposed, not authorized.
Production, defaults, original failures and worktree are preserved. No native
adoption, material sweep, transport/MPI expansion or P3. Only512-cell results
receive new visualizations in this task.
[Report](P2_asymmetric_branch_report.md); [equations](P2_asymmetric_branch_equations.md).

---

# P2 full-interval continuation gate (2026-10-10)

The selected extension to .625s was attempted at all three retained dt without
changing equations, loading, spatial operators or tolerances. Every path stops
at the same .59765625-s admissible antisymmetric saddle; no unstable trial is
committed. The common accepted interval to .59375s has converging fields and
separate residual works, but that does not qualify the requested endpoint.
Independent derivatives and positive incoming/fixed-history checks isolate a
mechanically reduced stability issue when two outer phase nodes release.
The next bounded question is stable asymmetric continuation and possible
discrete pinning, with a justified branch-selection rule before further history.
This is a proposed question, not authorization to implement a new rule. Preserve
all failed trials and old evidence; native adoption, tuning and P3 stay deferred.
[Report](P2_coupled_branch_extension_report.md).

---

# P2 consistent-endpoint short-branch result (2026-10-10)

The selected precise replay classification, continuous spatial energy identity
and three-step-size consistent-start trajectory are complete. The replay was
already phase-inconsistent, so positive incoming profile cost/negative tangent
at a nonstationary point is not evidence of unstable coupled equilibrium.
The LAST pre-damage checkpoint fails phase KKT, so fresh unloaded trajectories
complete0–.5625s with a locally stable growing onset branch and unchanged
physics/Q2/normalization/bounds. Same-path residual works and smooth energy
remainder decrease under refinement without a compensating loss. This resolves
the short-onset question and supports further prototype continuation rather
than damage kinetics. It does not select a native algorithm or qualify the
later fast-tail interval/.625s/P3. Stop after this bounded assessment.
[Report](P2_coupled_branch_report.md); [equations](P2_coupled_branch_equations.md).
Production ownership/defaults and all failed evidence remain unchanged.

---

# Completed P2 evolving-profile consistency diagnosis (2026-10-10)

**Native gate remains blocked; no timestep or new native run is selected.** Two
permitted1/2048,1/4096 endpoint refinements preserve mesh, initialization, material,
loading, bounds and tolerances. Fixed-history phase/mechanical checks pass, but
completed-state phase KKT and profile redistribution do not show ordinary temporal
consistency. This is coupling lag, not failure of the incoming phase solver.
A common C2 prescribed-profile control gives first-order force/work reduction.
Boundary reactions identify external work; lifting extensions change work away
from equilibrium. Both original work measures and all failed evidence remain.

A derived consistent-endpoint saved-entry prototype removes KKT lag at the original
target but yields a .0529 phase increment and +159.949J/m incoming profile cost.
It is not energetically admissible under the current instantaneous fixed-history
phase→Maxwell path. No physical history is published by this replay. A mechanical
redistribution/finite-transition energy-path decision is required before native
adoption; no compensating loss, kinetics, constitutive change or P3 is introduced.

[Report, settings, scaling, external-work audit, candidate equations and evidence](P2_profile_temporal_report.md).
Baseline de7beaa65846c7372711f7922abf1c5d9a745ee6/fault-propagation;
production/core/plugin unchanged. New files are benchmark diagnosis/replay scripts
and documents, with commands/source inventory in the report.

| Outcome | Measurement | Gate |
|---|---:|---|
| Solved versus endpoint phase KKT | ≤5.22e-12 versus5.79e-3 | Incoming solve passes; coupling remains unresolved |
| Last halving phase / traction | .00564886 /3.40121Pa | Original accuracy targets fail |
| Consistent replay KKT / profile cost | 8.91e-14 /+159.949J/m | Numerical pass; energy path fails |

Read-only command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_profile_temporal.py`.
check_profile_temporal.py verifies unchanged production/old trajectories, numerical
checks and the failed native gate. Only diagnostic/prototype/documentation additions.
The next selected scope should derive the work-consistent redistribution/transition
path, rather than extending the native family or changing physical parameters.

---

# Completed bounded P2 timestep assessment (2026-10-09)

**No timestep is qualified; the native selection gate is not met.** The matching
16×256 Q2/Q1, six-point endpoint prototype completes40/80/160/320/640 steps to
.625s, with identical initialized phase/particle H and zero accepted FE/Z history.
The continuous .256 ramp is applied as interval means; displacement is .05m in
every run. Last1/512→1/1024 full phase/traction differences .00241105/2.82566Pa
exceed .001/1Pa targets. Slip/common-rate differences .005799%/.09062%; the finest
reference remains unresolved, and its self-comparison is not a pass. No new native
run is launched; the previous34.4min baseline is retained without conversion.

Largest phase increments nearly halve (.0170487→.00109694). D_relax decreases
.179911→.0229697J/m, with unresolved limit. Boundary-lifting absolute work error
falls .204356→.0232568J/m; bulk-rate absolute error fails to decrease during damage
and is .111051J/m at1/1024. Frozen controls and an exact general-beta identity
isolate the intermediate profile-induced force/work difference (identity error
≤1.79e-12J/m per step). No history, operator, bound or tolerance is changed.
Selected proper-incoming derivative/curvature/quadrature and replay checks pass;
these numerical checks do not qualify temporal/energetic accuracy. Spatial choice
remains16×256; production sources, binary, specifications and old failed artifacts
are preserved. Only benchmark diagnostics and documentation change.

[Decision report, settings, plots, costs and commands](P2_time_resolution_report.md).
Evidence: benchmarks/reconstructed_fault/propagation/time_resolution_evidence/;
check_time_resolution.py verifies numerical/preservation checks and records the
failed accuracy gate. Further temporal refinement, evolving-profile interpretation
or a scheme revision requires a separately selected scope; no RSF/P3 is authorized.

---

# Completed selected P2 Q2 normal-resolution study

The requested bounded study is complete. Q2/Q1 with six coupled bulk points
remains fixed. At dt=1/64,T=.625, reused16/32/64 and added128/256;256 is the
least tested resolution passing full phase, traction, slip, whole/localized
stress RMS and bulk-storage targets. One selected16×256 native confirmation
finishes, with proper incoming curvature/derivatives and measured cost.
A performance-only phase row-iterator fix and corrected observer operand-scale
estimate resolve two retained initial-only attempts; no scientific algorithm,
bound, projector or solver-tolerance change is made.
[Decision and evidence](P2_Q2_normal_resolution_report.md).

The spatial gate now supports a separately selected time-accuracy assessment.
Finite phase release and mechanical work discrepancies remain explicit;
full stability and P3 are not qualified. No additional stage is authorized here.

---

# Current coupled bulk quadrature and cost outcome

Updated 2026-10-09. [Completed rule/accuracy/cost assessment](P2_bulk_quadrature_report.md).
The user selected compatible height-one controls to .625s, one consistent bulk
rule across all coupled operators, inexpensive rule and Q2 mesh studies, matched
native Q2/Q3 plus one warranted refined-Q2 case, new curvature probes and costs.
Six normal points pass independent24/48-point tests; four is insufficient at the
chosen derivative/force thresholds. Existing consistent normalization/CPDI and
physical settings are retained. New explicit setting defaults to3 normal points;
tangential panels use at least velocity degree+1. Three-point outputs are archived.

Q3 buys substantially smaller phase/q/stress errors at2.11x measured runtime.
At similar runtime,16x32 Q2 alone meets the declared phase/q/slip targets but has
worse stress error than Q3. Neither is full spatial/energetic/time qualification.
New free-mode checks use proper incoming anchors; they do not establish full
stability or remove historical affine negative modes. No P3. The next concrete
accuracy question is stress/storage spatial resolution, separately from time
integration and constrained stability. Further native refinement, branch rules,
material/transport/MPI/restart work are not automatically selected.

---

# Current P2 plane-strain follow-up outcome

Updated 2026-10-09. [Completed audit and bounded spatial/stability evidence](P2_plane_strain_report.md).
The user selected 3D embedding, helper/consumer and assembly/pressure audits,
algebra/manufactured checks, a separate demonstrated FE-only correction, then
bounded spatial/stability follow-up. All are completed. Existing Newton and
Picard trace-term assemblers are reused with kappa; no pressure shift or friction
model change. Compatible shear and mature defaults remain invariant. Q2/Q3 short
native controls pass numerical/history checks, while Q3's3-point stress aliasing
precludes spatial qualification. Frozen12/24-point checks resolve that uncertainty;
normalization error is negligible here. Sampled positive free curvature does not
replace full constrained-stability analysis or remove prior affine negative modes.

Do not begin P3. The next concrete numerical scope is a matched sufficiently
integrated normal mechanical/energy comparison; a further native time or velocity
refinement alone would not resolve the exposed quadrature error. Full stability,
rollback/restart/MPI/transport and energetic accuracy remain open. No new phase
branch rule, kinetics, material search or general-projector change was selected.

---

# Current bounded P2 native localization outcome

Updated 2026-10-09. [Completed numerical/localization/transfer assessment](P2_native_localization_report.md).
The user selected stall diagnosis, a rich ux(y) total-energy reference, native
side/normal-domain controls and a separate transfer audit. Those tasks are
complete. The numerical root is repaired at its unchanged target; original
arithmetic differences do not justify relaxation. All three native fixed-dt
controls finish to1 s without healing. Compatible sides reproduce the matching
coarse one-dimensional discretization and retain a band; strong coarse native
stress oscillations remain a principal spatial error. The original affine trial
has feasible negative-curvature directions, so KKT convergence is not minimum
selection. The current specification permits the stationary solver; no stability
rule or physical model change is inferred from this evidence.

Two controlled normal heights retain ell, dx/dy and the same local shear-rate
loading. Their different total boundary displacement/bulk compliance is reported.
Consistent normalization stays fixed and its representation error is negligible
beside native velocity-space error. Only three redundant projected-stress reads
are bypassed; authoritative FE state and legacy defaults are preserved.

No additional coupled timestep refinement, phase-branch algorithm, material sweep,
transport/MPI campaign or P3 is selected. Next scientific/numerical scope must be
explicitly chosen in light of spatial accuracy and negative-curvature evidence.
All failed entries and old qualification status remain preserved below.

---

# Current restricted native P2 selection

Updated 2026-10-08. The user explicitly selected the exact accepted FE
internal-strain route proposed by the endpoint comparison, resolving the
ownership decision below. [Implementation checkpoint](P2_native_FE_strain_report.md).
Optional production integration, nonuniform FE Maxwell/derivative/ownership
probes, zero hold, restriction guards and mature-default compatibility are complete.
The loaded full-Stokes case reaches t=.296875 and stops at its t=.3125 phase
entry; its KKT remains above the unchanged target despite bounded numerical
repeats. It is not a completed or refined coupled baseline. Preserve the failed
paths and investigate that entry before expanding native runs. Full Stokes and
side-boundary restrictions produce substantial differences from the one-mode
prototype; comparisons are reported as different spatial models.

Keep defaults, constitutive parameters and explicit restrictions. No changes to
the generic projector, material optimization, history transport, new kinetics,
tolerance relaxation or P3 follow from this checkpoint. Remaining P2 includes
native temporal/spatial and energy accuracy, rejected-step rollback, coupled
restart, transport/birth and distributed evolution. Older recommendations below
are preserved history and do not override this selected scope.

---

# Latest selected P2 evolving endpoint follow-up

The missing endpoint family and one concrete uncertainty refinement are complete
(2026-10-08). [Comparison and native-entry derivation](P2_total_energy_endpoint_comparison.md).
Endpoint is now selected; mean remains a diagnostic comparator. Both phase release
and independently accumulated mechanical discrepancy decrease. Full-field event
errors remain appreciable and the one-mode widened band is not propagation.

The user's conditional native integration is authorized without another planning
approval. Entry has been advanced through actual source/specification/state audit,
which exposes a governing discrete-state choice: parent-published stress projected
to FE does not commute with the candidate's QP profile redistribution. Installing
the candidate literally would change native history ownership; retaining particles
requires the transfer adjoint and matching projected mechanical operators, beyond
adding normalization's adjoint to phase. The report presents concrete alternatives
before implementation. Resolve that scientific/ownership choice; do not silently
reset accepted internal strain, add transfer energy to H or change the projector.
The projected-stress energy's matching mechanical Hessian has a proven rank
obstruction in the native full velocity space; this concerns that pullback only.
For identical uniform bulk/fault Maxwell coefficients and unloaded R0=0, retain
R=symgrad d_R with d_R_new=beta d_R_old+(kappa/G)u_accepted. This exact FE route is
recommended and its algebra passes on all seven saved paths; it does not require
a new physical driver or transfer correction. Its accepted-state ownership is
distinct from the currently specified particle/mapped-FE history and must be
explicit. The existing benchmark stress hook is plugin-owned and unarchived;
it alone is not an integrated native model. Native implementation is not complete.

No P3, material optimization, hidden dissipation or broad MPI/transport campaign.
Native qualification, full Stokes/localization, restart/rollback and transported
histories remain separate. The earlier mean selection below is historical and is
superseded by this evolving comparison; its numerical evidence is preserved.

# Preserved preceding P2 prototype follow-up

The loaded common-path comparison (2026-10-08) is complete.
[Report](P2_total_energy_time_comparison.md). It retains energy/internal variables
and production. A diagnosed surface-basis undersampling defect required
intersection quadrature and affected repeats. The corrected unloaded-seed
family shows decreasing release/increments and common-time errors rather than
a demonstrated persistent jump. Mean-integrated mechanics is the implementation
candidate; endpoint mechanics has better frozen-path accuracy but a work defect.

Proceed toward an optional native P2 implementation by first specifying ownership
of accepted total-minus-inelastic strain and surface Z, then implementing the
actual energy/projection adjoint and both mean mechanical blocks on compatible
measures. Retain endpoint/contact and onset accuracy tests beside the integrated
ledger. This is an implementation direction, not authorization to change
production in the current prototype-only task. The finest prototype pair still
has .0787 full-field phase error around onset and appreciable endpoint imbalance;
no qualified native baseline, kinetics/transition change, tuning or P3 follows.
Earlier failed native/preload evidence and direct irreversibility qualification
remain preserved. The question is now native full-space/accuracy qualification;
a finite-jump review becomes necessary only if that behavior persists under
properly initialized/resolved loading.

# Evolving phase field and reconstructed-fault propagation

Complete isolated total-energy candidate (2026-10-08):
[algorithm](P2_total_energy_algorithm.md), [verification](P2_total_energy_verification.md).
The user's explicit selection authorizes derivation/prototype, superseding the
previous review-only stop for this bounded task. No production adoption follows.
Projection-adjoint phase residual, accepted lower bound, general Maxwell kernels
and immutable publication pass. Exact integrated mechanical exchange requires
the candidate's averaged balance; endpoint controls retain a defect. Finite phase
relaxation produces substantial energy-inequality slack in unstable preload
controls. This changes the next scientific question to its physical interpretation
and mean-balance temporal accuracy, rather than declaring energy consistency from
endpoint equivalence. Native integration, coupled time refinement, tuning and P3
remain unselected. Stop after this prototype report.

## Preserved preceding review roadmap

Bounded P2 review checkpoint (2026-10-08) is complete. The inexpensive
normalization extension failed .5% at .0125 m and passed all six retained frozen
fields at .00625 m, without advancing histories or changing the retained .1 m
coupled configuration. [Checkpoint](P2_implementation_checkpoint.md),
[formulation comparison for review](P2_formulation_comparison.md).
This resolves the frozen representation target only; it does not resolve
profile energetics or qualify new coupled fields. Direct irreversibility remains
verified. Preserve the current baseline and stop for user review. Proposed total
phase energy, further coupled time refinement, tuning, scaling and P3 are deferred;
no roadmap entry authorizes them. The scientific choice concerns the total-energy
phase potential, retained internal variables and the role of peak H.

## Preserved accuracy/energy roadmap update

Focused P2 accuracy/energy assessment (2026-10-07) is complete. Direct
irreversibility remains successfully verified for the fixed-mesh fixture.
[Report](P2_accuracy_energy_report.md), [derived governing question](P2_accuracy_energy_derivation.md).
The .025 m consistent representation still misses .5% maximum source accuracy;
native authoritative stress partitioning passes but FE representation error is
separate. Large profile work is mechanically accounted, while the reduced phase
driver lacks the bulk-profile variation required by a total-energy potential.
This evidence changes the next action to a scientific energy-definition review
and targeted spatial resolution qualification, rather than further temporal
runs alone. No governing correction is selected here. Conditional finer coupled
runs and P3 are not authorized by a failed accuracy/energy gate. No material sweep.

## Preserved preceding P2 roadmap

Latest selected P2 irreversibility task (2026-10-07) is complete: optional accepted
phase lower bounds, same-H failed-entry replays and three common-end-time runs.
[Report](P2_irreversibility_report.md). The explicit admissible-set revision
supersedes the preceding no-lower-bound restriction for this comparison. Phase
healing is eliminated and feedback differences improve, with meaningful temporal
sensitivity still present. Normalize-source spatial error and profile work remain
separate; no qualified spatial/energetic baseline, material search or P3 follows.
Next P2 work needs a user-selected scope addressing those outstanding questions
or deferred restart/rollback/transport checks; it is not an instruction to run them.

## Preserved preceding roadmap updates


Current P2 coupling repair (2026-10-07): optional surface-owned Z history and
Stokes-QP conjugate shear/ds resistance are implemented and pass fixed-phase,
local-history, tangent, default-compatibility and small distributed checks.
The common stronger ramp produces resolved endpoint damage; lag-aligned feedback
agreement improves. All three paths nevertheless stop on resolved tail healing,
confirmed by native corrections and identical-H repeats. This replaces the
publication/work-operator blocker with an explicit admissibility decision, not
permission to change irreversibility. Normalization spatial error and changing-
profile energetics also remain separate. See [P2_coupling_report.md](P2_coupling_report.md).
No more loading/material search, scaling claim or P3 follows this gate.

Latest selected P2 recoverable-energy revision (2026-10-06): optional state-based
H and unloaded initialization are implemented, with local and zero-load checks.
Three unchanged a=0.016 trajectories pass numerical gates but expose native
traction-reprojection smoothing and a bulk/fault work-measure mismatch. The reduced
energy derivative is verified; full coupled energetic consistency and feedback
refinement are not. See [P2_recoverable_report.md](P2_recoverable_report.md).
Next is a focused coupling-consistency scope, not a material/loading search.
Stress scaling remains conditional; no P3. Historical diagnoses below are retained.


Latest selected P2 history diagnosis (2026-10-06): pause material tuning. Native
fixed-phi tests and unchanged reference reruns confirm that the specified maximum
of single-step work derivatives loses feedback under timestep refinement. No unique
physical replacement follows from the current derivation; review recoverable-energy
versus accumulated signed-work driving and projected initialization before a law
change. Stress scaling remains conditional, unexecuted. See
[P2_history_report.md](P2_history_report.md). No P3. This updates the next action
because the issue is now isolated from endpoints and phase discretization.


Latest bounded P2 material study (2026-10-06): fresh core 0.6 with fixed
G/c/eta/ell/p and consistent normalization gives a small late reference response
at a=0.016; fourfold Gc stays inactive. At half dt the reference phase is unchanged
and cumulative H growth is almost eliminated. Bounds/healing/equilibrium and
finite normalization pass, with <=0.15625% measured source-representation error.
The O(dt) partially damaged candidate compared with a stored maximum is now a
concrete feedback-loss diagnosis. No optional stronger Gc, additional amplitude
or algorithm change was used. See [P2_material_report.md](P2_material_report.md).
Next requires scientific review of H semantics, not a larger parameter sweep.
No P3; prior high-load/normalization failed evidence below remains valid.


Current outcome (2026-10-06 normalization follow-up): I_h-only lumping is
implemented behind an optional setting and passes the restricted P1 lifecycle
checks. Frozen-field tests show inadequate coarse representation at 0.1/0.05 m
and an exactly saturated fine field with divergent normalization. This replaces
the earlier projection-only blocker with two separate issues. Further coupled
runs require a reviewed fully damaged-state formulation and adequate surface
accuracy; neither a phi cap nor a new H law is authorized. See
[normalization_report.md](normalization_report.md). Stop before P3.


Prepared 2026-10-05 for a new dedicated Codex session.

## Purpose

Establish that the current ASPECT implementation correctly evolves its phase field before asking it to extend reconstructed fault geometry. Use a sequence of small experiments, changing one major capability at a time. The initial assignment is P0 only; later stages are a roadmap to refine after reading the code.

This handoff was prepared from the research discussion, not an inspection of the current repository. Function names, input keys, build commands, checkpoint compatibility, and current branch status must be established in P0. Do not treat historical details as verified current code.

## Essential context

- The project couples phase-field shear damage, rate-and-state friction (RSF), viscoelastic Stokes flow, particle-carried history, and reconstructed fault/surface integration.
- The familiar discretization uses Q1 phase field, an AT1 crack-density functional, Lorentz degradation, and an irreversible phase-field treatment. Confirm the current definitions of phi, its peak/history field, H, and the implemented constraints.
- Fault kinematics use a normalized diffuse localization, described in discussion as chi = h(phi)/I_h. Here h(phi) is a localization function, not mesh spacing. Use h_cell for mesh spacing in this task's reports.
- BP3 concerns slip on established geometry. Its production run and refactor work remain separate. The user reports that the current code has not been challenged with non-frozen phase field.
- The refactored implementation was developed on pf-rsf-refactor in YiminJin/aspect. This is a discovery hint, not permission to use an old revision or a claim about the current branch tip.
- Existing particle-density controls and history initialization matter: earlier velocity failures were associated with particle imbalance. Preserve the established settings and inspect their actual behavior where needed.
- The later conceptual specimen contains a horizontal finite seed entering from the left of a sheared, confined box. Propagation is preferred over independent nucleation because it introduces fewer new state transitions.

## Stage sequence

| Stage | New capability under examination | Geometry | Phase field | Main restriction |
| --- | --- | --- | --- | --- |
| P0 | Audit and select a minimal fixture | Unchanged | Unchanged | No physics implementation |
| P1 | Existing phase-field operator and history constraints | Fixed | Evolving under controlled input | No Stokes/RSF feedback |
| P2 | Coupled evolution on an existing fault | Fixed | Evolving | Through-going straight fault; no tip growth |
| P3 | One controlled geometry extension and state transfer | One synthetic extension | Prescribed diagnostic field | No physical propagation claim |
| P4 | Physical straight-tip propagation | Extends | Coupled | Fixed candidate direction; local activation region |
| P5 | Directional propagation | Extends and may curve | Coupled | Local activation region retained |
| P6 | Remote nucleation check | Evolves | Coupled | Remove the activation restriction |

Velocity-weakening propagation, independent nucleation, multiple faults, branching, coalescence, and a large server run are later tasks, not automatic extensions of this plan.

## P0 — Audit and define the first experiment

Read instructions and source before changes. Create/update only the task documentation during this stage. Do not modify production C++, parameter defaults, or the governing equations. An existing short frozen-field smoke test may be run if a matching executable is available and the command/cost are known. Do not start a new build or a production run merely to finish an audit.

1. Identify the actual baseline and instructions. Record commit, branch, dirty state, existing worktrees, current executable/build provenance, and relevant fixture paths. If the intended baseline is ambiguous, finish the read-only audit and state precisely which revision choice remains unresolved.
2. Trace the execution path for the phase-field solve, including initialization, coupling order, trial versus committed state, particle/FE transfers, H accumulation, irreversibility, solver convergence, restart, and failed-step restoration.
3. Find every place where frozen phi or fixed reconstructed geometry is assumed: material evaluation, fault association, quadrature/normalization, caches, Jacobians/preconditioners, output, and checkpoint state. Distinguish a genuine requirement from an optimization valid only while phi is frozen.
4. Inspect the existing phase-field-only and reconstructed-fault fixtures. Identify the smallest reusable setup for P1 and a through-going-fault candidate for P2. Do not shrink a production BP3 model blindly.
5. Explain how P1 can exercise the production phase-field operator with externally controlled driving/history data, without duplicating its equations in a toy implementation. If the infrastructure is missing, propose the smallest test-only entry point for P1; do not implement it during P0.
6. State the actual residual/active-set or KKT diagnostics available, how they are scaled, how irreversibility is represented, and what values should be measured. Propose tolerances based on discretization and solver scales rather than arbitrary universal constants.

Deliver a concise P0_report.md in the task directory: code-path map with real symbols/paths, implemented/untested/missing capability table, selected P1 fixture, proposed file changes and commands, and a small verification budget. Mark source inferences separately from executed evidence. Update STATUS.md and stop at the P0 boundary.

P0 passes when there is a concrete, bounded P1 experiment using the real operator and a defensible baseline. It does not require resolving all later propagation design questions.

## P1 — Phase-field evolution without coupled mechanical feedback

Latest fixed-support result: the user selects full normal cross-section
association from initialization for the single straight through-going fault.
The new explicit default-off setting derives width from actual domain/fault
coordinates; it preserves H, sharp endpoints and existing checkpoint metadata.
Affected serial/two-rank/restart controls pass with zero association omission.
The padded finite seed is unchanged. P1 now passes for these restricted controls;
the earlier failed equality and narrow-strip evidence below remain historical.


Production execution (2026-10-06): optional bound solve, explicit accepted-state
handling, relaxed initialization, changed-history restart and small MPI checks
are implemented. Algebraic phase checks pass. **Full P1 gate remains blocked**:
the straight +5% ramp has active transverse tails outside retained prescribed-
profile association widths. No support admission/truncation policy was silently
introduced. See [P1_integration_report.md](P1_integration_report.md) and
[P2_report.md](P2_report.md). P2 is authorized conditionally but has not run.
A scientific support-policy decision, followed by the affected P1 checks, is
needed before mechanical feedback testing. Later historical entries below
retain the scope and conclusions of the earlier experiments.

Production-integration planning update (2026-10-06): the user selected the
tested box solver, relaxed initialization with consistent dependent data,
unchanged particle H, and separately measured healing as the direction for
production integration. [P1_integration_plan.md](P1_integration_plan.md) records
the source-audited lifecycle, state ownership and small serial/restart/MPI gate.
Native phase-only restart is now included in this P1 extension; physical-step
rollback and mechanical restart remain P2. The latest user instruction authorizes
implementation and immediate P2 entry only after the complete P1 gate passes.
Implement and pass its phase-only gate before the requested P2 mechanical
feedback investigation. The earlier test-only restriction below describes
the completed experiment, not the newly selected integration direction.

P1 bounded result (2026-10-06): implementation and the one permitted refinement
repeat are complete; both meshes converge and evolve, but persistent negative
phi and load-induced healing prevent physical qualification. See
[P1_report.md](P1_report.md). The separately authorized bounded initialization/
tail diagnosis is also complete: padding does not cure finite-cap failure,
and a straight band retains a smaller tail undershoot on both tested meshes.
See [P1_tail_report.md](P1_tail_report.md). The subsequent user-selected
reference and test-only box-constrained follow-up is complete; see
[P1_bounds_report.md](P1_bounds_report.md). A well-resolved reference isolates
sampled H and nonlinear integration contributions. The bounded prototype
passes restricted straight/padded equilibrium and ramp checks without a seed
constraint. The production equality solver remains unchanged and unqualified.
Remain before P2. The subsequent integration plan above addresses production
adoption and initialization; see the current execution result below.

P0 source correction (2026-10-05): the current phase solver has affine FE
constraints and a degradation-domain guard, but no phase bound/irreversibility
active set or KKT algorithm. Irreversibility is carried by particle H. Use the
controlled-H production-operator fixture in [P0_report.md](P0_report.md); measure
raw bounds and phase decrease without claiming an implemented inequality solve.
Do not add an active-set method merely to match the provisional wording below.
The later explicit user request authorizes box constraints only in the P1 test
harness; production constraints and history semantics remain unchanged.

Run only when requested. Prefer an existing small phase-field harness. Hold geometry, orientation, and the mechanical inputs fixed; prescribe the appropriate driving/history input through the established interface. Start without advection to isolate the solve. Use the real initialization, constraints, and phase-field solver rather than a separate reimplementation.

Use three short parts in one fixture where possible:

1. Initialize a compatible pre-existing band and hold a subcritical driving input. Separate initial profile adjustment from subsequent drift.
2. Raise the driving input enough to cause a resolved, nontrivial phi change in a partially damaged region. A fully saturated band or a run with unchanged phi cannot establish evolution correctness.
3. Reduce the applied driving input while retaining the required committed history; verify the intended irreversible response. If the stored H itself is irreversible, do not lower it artificially and then label that a physical unloading test.

Measure phi bounds, the actual equality weak residual after affine constraint elimination, H monotonicity and phase decrease, change in phi, iteration counts, and the crack-density integral. Mark phase active-set/KKT diagnostics unavailable; if a diagnostic projected residual is added, distinguish it from the production stopping criterion. Verify checkpoint/restart equivalence once at an intermediate state if this harness supports the actual restart path; otherwise carry the check explicitly into P2. The P0-selected scratch-state probe carries native restart and full timestep rollback into P2.

Repeat one evolving segment with smaller load increments if the response jumps or convergence depends on increment size. A prescribed-H experiment does not have the coupled mechanical energy balance of P2.

Exit evidence: reproducible hold and evolution, satisfied bounds/constraints, no history advance on rejected trials, and a concise explanation of any initialization transient. Fix only blockers demonstrated by this fixture. Keep a small regression for each actual defect.

## P2 — Coupled evolution with fixed fault geometry

Initial execution completed as a **failed admissibility test**, not a full P2
qualification. At dt=0.25 and 0.125, native mechanical H increases and phase
responds with the expected one-step lag. Both paths stop when consistent
normalization projection produces negative nodal I_h from positive transverse
integrals; the fine failed-entry field also has small resolved healing. The
common accepted interval ends at 0.25 s, short of the selected 1 s schedule.
The cumulative feedback refinement conclusion remains inconclusive. See
[P2_report.md](P2_report.md). Resolve normalization representation/resolution
before continuing; do not hide the failure with a mask, floor or H-law change.
Physical rollback, coupled restart and transported/birth histories remain P2.


Use a small straight through-going fault to avoid physical tips. Keep its topology and length fixed while allowing phi and the coupled constitutive fields to evolve. Reuse the verified P1 initialization. Start with controlled compressive normal traction and velocity-strengthening RSF if supported; document whether normal stress is prescribed or solved. Do not alter normal-stress treatment silently for convenience.

Choose a partly damaged, mechanically consistent state and loading that produces measurable phi evolution without deliberately driving an earthquake. Hold the fault path fixed and restrict activation to a sufficiently wide neighborhood of that path. This tests band evolution, not propagation or nucleation.

Trace the actual approved sequential cycle, verified in P0: incoming committed H, phase solve, temperature/composition and particle transport, coefficient/cache preparation, trial Stokes/RSF solve, then terminal history commit. Phi and geometry remain fixed during mechanics. Show the phase residual against its input H and the mechanical residual against its frozen phi; measure the phase defect against newly committed H separately as a splitting diagnostic. Do not claim simultaneous equilibrium at the new H or introduce an outer coupling iteration without a demonstrated need and a separately resolved scientific decision. Nonlinear iteration counts alone are insufficient.

Measure phi, H, the irreversible field, I_h, integrated slip/localization, V, theta, normal/cohesive traction, and a compact boundary-response history. Include transport/particle transfers in at least one short check once the isolated case works. Evaluate irreversibility relative to transported material history, not only at unchanged Eulerian nodes.

Verify a short restart and, through an existing retry mechanism or a narrowly scoped diagnostic, failed-step rollback. State, slip, H, time, particle properties, and phi must not be committed twice. Check a relevant frozen-field regression after production-code changes. Perform one smaller-step comparison on the evolving interval; do not launch a convergence matrix.

Exit evidence: nontrivial coupled evolution without geometry changes, consistent normalization and history updates, stable/reproducible behavior at a smaller increment, and preserved frozen-field behavior. If this fails, remain here rather than introducing a tip.

## P3 — Geometry extension as an isolated state-transition test

Use a synthetic phase-field ridge or a controlled extension input to exercise one append operation independently of fracture physics. Do not build a general branching/network framework. Test one short straight extension first.

Check endpoint identification against the initialized rounded cap; a threshold crossing already present at time zero is not a new segment. Preserve established vertex/material identities and histories. Follow existing material-transport rules for positions rather than accidentally freezing motion. Distinguish old material newly represented by quadrature from genuinely newly activated material.

Specify and test which associations, quadrature data, I_h/localization terms, constraints, matrix structures, and solver caches become invalid. Define theta/slip/cohesive history initialization for new material and keep old values unchanged by reindexing. Ensure the extended state is serialized and that rejection restores the pre-extension state. Exercise rank ownership/ghost synchronization with one small multi-rank case if these structures are distributed.

Exit evidence: one deterministic, repeatable extension with correct state preservation and restart/rollback behavior. The imposed field is a software verification fixture, not evidence that the physical propagation law works.

## P4 — First physical straight-tip propagation

Combine the verified capabilities. Use a finite seed with a short diffuse cap, a fixed candidate direction, velocity-strengthening RSF, and a two-dimensional activation region around the tip plus the existing band. Make the region large enough to contain the diffuse and cohesive zones. Document the restriction and check one enlarged region.

Resolve the activation dependency: evaluate the approved intact-material driving rule ahead of the endpoint before an RSF fault association exists. Do not insert geometry first merely to make damage growth possible. The accepted phi, geometry, mechanical state, and histories must satisfy the chosen coupled acceptance conditions; record the actual update/commit order.

Keep computational-boundary completion separate from physical-tip treatment. Audit tiny I_h and localization near the cap. Do not add hidden floors, long prescribed tails, path forcing through initialization, or history resets to make the example propagate.

Candidate values from the discussion, all provisional until checked against source conventions and cost:

| Quantity | Candidate |
| --- | --- |
| Domain and seed | 4 km by 2 km; 1 km horizontal edge seed at mid-height |
| G and bulk viscosity | 30 GPa; 1e22 Pa s |
| Initial confinement | 50 MPa |
| Intact friction and cohesion | mu_pre = 0.6; c = 10 MPa |
| G_c and ell | 2e5 J/m^2; 10 m |
| Fine mesh | h_cell about 2 m, with broad local refinement |
| RSF | a = 0.015, b = 0.010, D_c = 0.008 m, mu_0 = 0.6, V_0 = 1e-6 m/s |
| Fresh-surface state convention | theta_birth = D_c/V_0 = 8000 s; verify strength continuity |
| Initial observation window | About 0.2–0.5 km of advance, reduced if sufficient |

The tentative loading is top/bottom opposite horizontal velocities with zero vertical velocity, and confining normal traction on the side boundaries. Check compatibility, pressure reference, particle inflow/outflow, and boundary artifacts before accepting it. Prior BP3 boundary findings do not automatically transfer to this specimen.

Check the actual degradation-function shape parameters and strength/length calibration. Do not automatically reuse a frozen-profile choice of Lorentz p. Do not generate a purported executable prm from this table without resolving actual input names and initialization.

Track tip advance relative to the equilibrated initial cap, boundary force/displacement, phi/H/I_h, fault histories, and the energy partition. Use a slip/state increment control plus an assessment of fracture advance per increment. A large quasi-static jump may be physical; a smaller-step comparison must distinguish it from numerical skipping. Start with one mesh and refine once only where needed to establish a conclusion.

Exit evidence: measurable propagation under the stated restrictions, no spurious advance under a subcritical hold, consistent history/normalization, and reasonable increment/mesh sensitivity. Force balance and plausible morphology alone do not establish an independent quantitative benchmark; identify the needed matching cohesive-interface or fracture-energy reference.

## P5 — Release the growth direction

Keep the same specimen, strengthening RSF, and local activation restriction. Introduce the principal-stress/intact-friction direction rule and the chosen rotation-based branch selector only now. Specify eigenvector sign handling and the fallback at negligible rotation. Established fault directions/history must remain consistent with geometry and material transport.

Compare with P4. Do not label curvature a failure merely because the seed was horizontal. Monitor whether a single ridge remains representable. If branching appears, report it and its resolved structure instead of silently dropping a branch or automatically expanding the project to fault networks.

## P6 — Remove the remote-activation restriction

Permit intact activation throughout the specimen while keeping the remaining controls unchanged. Record the intact failure margin on admissible planes away from the tip, including boundary corners. Compare the unrestricted result with the restricted case. Raising G_c alone is not a substitute for checking peak-strength calibration.

If remote damage occurs, determine whether it reflects the loading/strength choice, boundary conditions, unresolved numerics, or missing topology support. Do not suppress it invisibly. This is the readiness decision for a subsequent unrestricted propagation/nucleation task.

## Reporting and cost

For each requested stage, save a short Pn_report.md with: exact source/build/input identity; question tested; temporary controls; changed files; commands and runtime; small measured result table; supported conclusion; unresolved issue; next action. Link only the necessary plots/log excerpts. Track active work in STATUS.md.

Prefer the existing local test harness and a few MPI ranks where sufficient. Inspect fixture cost before execution. Use separate outputs per case, a bounded stopping condition, and sparse diagnostics. Do not start a production server job as a side effect of local readiness checks.

Scientific references for later model choices, not substitutes for inspecting this implementation:

- Fei and Choo (2020), frictional shear fracture: https://doi.org/10.1016/j.cma.2020.113265
- Fei et al. (2026), in-plane quasi-dynamic fault growth: https://doi.org/10.1002/nag.70416

## Decision record

- 2026-10-05: Use a dedicated session and preferably a separate worktree based on the verified refactored baseline.
- 2026-10-05: Begin with P0. Verify evolving phi in P1/P2 before physical tip propagation.
- 2026-10-05: Keep later specimen parameters provisional. Independent nucleation and velocity weakening are deferred.
- 2026-10-05, P0: User selects the existing worktree/branch and installed doc directory; no new workspace or directory is needed for the audit. Baseline is de7beaa65846c7372711f7922abf1c5d9a745ee6.
- 2026-10-05, P0: Source establishes H-based irreversibility and sequential phase-before-mechanics coupling, superseding the handoff's assumed phase KKT and joint-acceptance wording. Historical evolving K3 evidence is useful context, not current qualification; its missing safeguards.h dependency favors the self-contained native test access for P1. See P0_report.md. P0 complete; P1 remains proposed.

- 2026-10-06, P1: Original AGENTS guidance restored and propagation instructions
  appended. Two bounded runs establish convergence and prescribed-H evolution,
  but not admissibility. Refinement does not remove the violations. No production
  changes; next work remains a selected P1 diagnosis, not P2. The historical
  small-increment mechanical-feedback loss remains an explicit P2 question that
  prescribed-H P1 cannot resolve.

- 2026-10-06, P1 continuation: User selected bounded initialization/tail
  diagnosis and allowed tests under benchmarks/reconstructed_fault/propagation.
  Four controls separate boundary necessity and cap relaxation from a residual
  straight-band tail failure. The conditional finer repeat did not restore
  admissibility. Further causal separation of H-transition representation and
  particle integration remains P1 work requiring selection; P2 stays deferred.

- 2026-10-06, P1 reference/bounds: Latest user selection explicitly authorizes
  a box-constrained diagnostic using the production operator. Independent 1D
  integration shows a negative equality tail for particle-domain H even at
  fine resolution; native pointwise H shows no observed undershoot. Straight
  and padded bounded trajectories pass restricted KKT/bounds/repeat/evolution
  checks with no resolved healing. No seed constraint was needed. Production
  adoption and initialization integration require review; P2 stays deferred.

- 2026-10-06, P1 production-integration plan: User chooses box-solver integration,
  relaxed baseline/dependent-data initialization, unchanged H representation
  and law, and healing diagnostics without a previous-phase bound. Prepare
  the small integration plan first; require actual production holds, native
  restart and two-rank consistency before P2. Preserve the sampled-H accuracy
  question and make small-increment mechanical-feedback loss central to P2.

- 2026-10-06, production execution: user supersedes the test-only/stop-after-P1
  restrictions and selects immediate initial P2 after a passing P1 gate. The
  optional production solver reproduces equilibria/ramps and passes phase-state,
  changed-history restart and two-rank algebraic checks. Straight support
  coverage fails after +5%; the missing normalized source fraction is about
  7.256e-7, with active phi outside the retained strip. Stop for support-policy
  resolution, not another planning approval. P2/P3 have not run; H law unchanged.

- 2026-10-06, fixed support / initial P2: explicit user policy resolves P1's
  straight support failure. Native-H invariance, zero omission, phase equality,
  changed-history restart and two ranks pass. The initial coupled pair then
  exposes negative current I_h from consistent Q1 projection, independently
  reproduced from positive normal integrals. Corrected a separate zero-base
  pressure-roundoff bookkeeping defect using the existing bound and cap. Stop
  at the normalization/healing decision; P2 refinement remains inconclusive,
  no P3 or propagation. No further timestep refinement was spent on the
  already inadmissible normalization trajectory.
