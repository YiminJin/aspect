# P2 checkpoint / closeout (2026-10-13)

[Current closeout](P2_CLOSEOUT.md) distinguishes native operators, retained
growth/saddle and MPI/recovery checks from complete prototype trajectories.
The fresh native endpoint trajectory ends at 0.53125 s; the full 0.625-s native
run, physical qualification, coupled restart, transport and propagation remain
deferred. Reduced Newton with FGMRES is the reference. Direct mechanics is a
default-off serial experiment; unsuccessful coupling alternatives stay inactive.
This scope is preservation, documentation and curated commits only. No further
scientific or performance experiment is active. Historical results follow.

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

Authorized: audit/verify the complete endpoint path at 1/2/4 ranks; screen
16/8/4 tangential cells at fixed 512 normal cells and 64 surface segments;
make one measured optimization if needed; conditionally launch one fresh
154-step four-rank trajectory to .625s with a first-attempt 12-hour cap.
Physics, discretization orders, integration, immutable histories, bounds,
true energy and final KKT criteria remain fixed. No P3.

Baseline `33c0adba59132890b386f3e984b3e1650feffe44`, `fault-propagation`, current
worktree. Incoming tracked state is clean; retained untracked evidence is
preserved. Serial executable/observer copies are in
`build-propagation-integration/serial-reference/`; source/input hashes are in
`benchmarks/reconstructed_fault/propagation/native_mpi_evidence/baseline.json`.
Serial-only assumptions were identified in FE ray construction, global phase
index access, sparse-search row elimination and benchmark output. Legacy CPDI,
normalization and mechanical surface operators already provide owned assembly
and collectives. MPI qualification and the full run are pending.

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

# Bounded P2 native consistent-endpoint assessment (2026-10-11)

The full matching **512-cell FE**, shared-initialization prototype completes
0–.625 s at dt=1/1024 and1/2048. All seven unaligned temporal targets pass:
phase5.36e−5, traction2.57e−6Pa, slip2.95e−8%, common-rate7.71e−6%,
whole/local stress .000345/.000970% and bulk storage .00000149% of fixed scales.
Mechanical/friction/phase residual works are separate and decrease about4×;
finite normalization-profile work is retained. Centroid remains selection-sensitive.

A default-off restricted serial native endpoint algorithm is implemented, using
immutable accepted FE d_R, surface Z and last physical phase, actual consistent
normalization adjoint, projected matrix-free Newton and translation saddle escape.
No dense native Hessian, predictor history or per-trial physical publication.
The shared-profile file is a fresh prescribed pre-existing-damage initialization;
H is not regenerated. Native operator controls pass unloaded/growing/asymmetric
states, full2D phase gradient, reduced actions and tangential adjoint directions.
Fresh native evolution accepts through .5283203125 s, with resolved flank
growth 5.62566e−6 and unchanged core. All seven matching-schedule field targets
pass: phase3.28e−13, q1.50e−8 Pa, slip1.63e−12 m, common rate6.10e−11 m/s,
whole/local stress6.83e−7/8.37e−7 Pa and bulk storage3.01e−10 J/m.
Healing is zero; recurrence<=1.74e−18 m; accepted publication/KKT checks pass.
Tangential phase variation6.50e−13 and normal stress<=4.37e−6 Pa are measured,
not suppressed. The one-hour cost gate interrupts the uncommitted core-growth
entry at .529296875 s. The planned .53125-s comparison is incomplete.
Keep the option experimental/default off; no native core/saddle/full-run claim.

Native AT1 quadratic reuse, sparse positive search factorization and first-entry
obstacle prediction change search cost only. Original CPDI residuals are checked
before publication; true energy, bounds, anchors and tolerances remain unchanged.
Fine growing-step mechanical/friction/phase works match the prototype around
1e−8 J/m. The coarse prefix has a separate 15.3446-J/m contact-path error;
its field/storage accuracy gate is not a cumulative energetic qualification.

HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6, fault-propagation, existing worktree.
Build GCC12.4/dealII9.6.2/Trilinos14.2/p4est2.8.7, one rank/thread.
Operator control267.98s/1296MiB; final native3600.69s/1314MiB,
3534.31s user CPU. Compatibility21,105assertions/54cases passes.
Detailed source functions/settings/evidence/cost and commands:
[P2_native_endpoint_report.md](P2_native_endpoint_report.md).
Evidence: benchmarks/reconstructed_fault/propagation/native_endpoint_evidence/.
No VTUs, material tuning, history transport, native restart conversion or P3.
All historical evidence preserved (1306 unchanged files, zero missing).
Next proposed numerical task: reduced-tangent conditioning/cost using existing
linearized mechanical operators, then fresh native core growth. No640-step launch.

---

# Completed P2 matched initialization / positional assessment (2026-10-10)

**The retained 512-cell FE prototype passes practical spatial targets over
0–.625 s; position remains selection-sensitive.** One independently resolved
continuum seed/H reference replaces mesh-dependent particle sampling only in
fresh diagnostic controls. Shared 511/512/1024 exact-bulk paths at dt=1/1024
all complete 640 stable steps, with unchanged equations, bounds and tolerances.

| Shared initialization control | Peak centroid excursion (mm) | Final shift (mm) |
|---|---:|---:|
| 511 cells | .7113 | ~0 |
| 512 cells | .7386 | .05785 |
| 1024 cells | .2615 | ~0 |

512→1024 full/aligned phase errors .002079/6.85e-5; q error .01781 Pa.
Retained 512 FE versus shared 1024: phase .002424, q .02717 Pa, slip .0000603%,
stress RMS .7863% whole /1.8888% localized, storage .01012% of fixed scales.
All retained practical spatial targets pass. No unique positional continuum
limit, physical asymmetric branch, full Stokes qualification or P3 is claimed.
Signed mechanical/friction/phase works remain separate and individually finite;
no compensating loss. Near-release microscopic searches are stopped.

The common core is .600001937763; maximum relaxation correction from physical
seed 6.774e-5, no box correction. Reference field uncertainty <=1.14e-11;
unloaded protected-seed KKT=0. All 1315 prior source/evidence hashes are intact.
Production/specification equations and defaults are unchanged. Worktree/branch
unchanged; HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6, incoming work preserved.

[Report/settings/source table/native recommendation](P2_matched_initialization_report.md)
and [reference/reduced equations](P2_matched_initialization_equations.md).
New diagnostic files: matched_initialization.py, assess_matched_initialization.py,
check_matched_modes.py, render_matched_snapshot.py, check_matched_visualization.py,
snapshot_visualization/{CMakeLists.txt,snapshot_visualization.cc}.
Evidence: benchmarks/reconstructed_fault/propagation/matched_initialization_evidence.

Checks: `matched_initialization.py check-reference`, `... equivalence`,
`assess_matched_initialization.py`, `check_matched_modes.py`,
`check_matched_visualization.py`, with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
and MPLCONFIGDIR=/tmp/p2-matched-mpl for assessment. Three fresh run commands
use `matched_initialization.py run --cells 512|511|1024` (refuse overwrites).

Actual ASPECT output-only postprocessing renders five accepted 512 exact-bulk
snapshots on16×512 cells: accepted.pvd and five compressed VTUs,780605 bytes total,
152–157kB each. Rendering6.19s/323MiB; decoded payloads checked. This is not a
native coupled run or checkpoint conversion. Next recommendation: separately
select optional native consistent-endpoint operator/state integration, followed
by a cost-bounded matched comparison. No native adoption or P3 started.

---

# Completed P2 translation-mode / branch diagnosis (2026-10-10)

**Stable reflected diagnostic trajectories reach 0.625 s; physical band position
remains mesh dependent.** The mode matches −phi_y with L2 cosine .99987–.99997.
Continuum translation is neutral only where boundaries and the immutable
previous-phase obstacle permit it. All 42 original starts find stable minima;
all 14 continuations finish with unchanged numerical/history/bound gates.

| Control at dt=1/1024 s | First saddle (s) | Final centroid shift (mm) |
|---|---:|---:|
| 256 normal cells | .59765625 | ±.762917 |
| 512 normal cells | .5595703125 | ±.113303 |
| 511, half-cell-centered seed | .556640625 | effectively zero |
| 512 exact bulk | .5595703125 | ±.114761 |

The exact control excludes Q2 mechanics as the sole cause. At 0.625 s the
512-cell phase maximum is .750194, q=188.834462 Pa, slip=.048804730 m.
Full-field 256→512 difference is .002818; placement difference .000472;
exact-bulk difference 1.60e-5. All separate late residual works decrease about
fourfold per halving; the 512 temporal full-field difference is 9.92e-5.
KKT≤9.63e-12, force≤2.35e-8 Pa, contact≤1.04e-10 Pa; no measured healing.

Release is bracketed within 8.35e-7 s at unchanged incoming history. Near-release
probes resolve small branch corrections but retain six weak-contact/unresolved
trials without publication. The zero-amplitude limit remains inconclusive at
the arithmetic floor; no physical jump or branch-selection rule is adopted.

[Report, settings, source table and 512-cell-only figures](P2_asymmetric_branch_report.md),
[translation/obstacle derivation](P2_asymmetric_branch_equations.md).
Evidence: benchmarks/reconstructed_fault/propagation/asymmetric_branch_evidence.
New test-only scripts: asymmetric_branch.py, assess_asymmetric_branch.py,
probe_asymmetric_release.py. Production/specification equations unchanged;
1105 original source/evidence files preserved. Existing worktree/branch/base
de7beaa65846c7372711f7922abf1c5d9a745ee6 retained. Read-only check:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/assess_asymmetric_branch.py --check`.

Stop after this selected P2 diagnostic. Next proposed question: separate phase
front/obstacle discretization from initialization effects before selecting a
physical branch or adopting native coupling. No native run, material tuning,
transport/birth/MPI campaign or P3 is selected.

---

# Completed P2 coupled continuation attempt (2026-10-10)

**The .625-s endpoint remains blocked by admissible reduced instability.**
All three unchanged coupled prototype partitions resume their accepted .5625-s
states and reach a stationary saddle at .59765625s. Trials are saved and never
committed. Last accepted times .59375/.595703125/.5966796875; comparisons use
the common .59375 endpoint. Core .70295163, q227.667285Pa, slip .043891263m.
Successive full-field differences 1.54e-8/3.84e-9; late separate mechanical,
friction and phase residual works each fall ~4 per halving. No finite jump is
shown on the accepted interval. Energy cancellation is exposed, not a pass.

Incoming states satisfy all coupled conditions. Failed trials pass KKT/bounds/
mechanics/contact but have one antisymmetric admissible reduced eigenmode,
lambda_min=-6.31119e-5, independently resolved to <=5e-11. Fixed-history
curvature is positive .0403675. Two outer nodes release at y=.19140625/.80859375.
No unstable publication, new branch rule, kinetics, parameter/tolerance change
or production edit. Nearby stable asymmetric equilibria remain unresolved.

[Report, settings, error/stability tables and plots](P2_coupled_branch_extension_report.md).
New benchmark: extend_coupled_branch.py; evidence: coupled_branch_extension_evidence.
Native equations/defaults and 1060 production/older-evidence files preserved;
original fresh prefixes are exact. Added costs28.00/43.52/59.06s, peak105/107/115MiB.
Checks pass for accepted states and correctly retain the failed stability gate:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/extend_coupled_branch.py check`.
Baseline de7beaa65846c7372711f7922abf1c5d9a745ee6/fault-propagation unchanged.
Next proposed task: diagnose stable asymmetric continuation/discrete front
pinning and justify branch selection before resuming physical history. No native,
material tuning or P3. Stop after reporting this bounded attempt.

---

# Completed P2 consistent-endpoint onset assessment (2026-10-10)

**Short coupled branch passes; full-interval/native/P3 gates remain deferred.**
The LAST unchanged seed at .5263671875s fails phase KKT (7.08e-8). Primary
runs therefore follow the required fresh unloaded fallback, R=Z=slip=0. Three
identical-start Q2/256, six-point, unchanged-material endpoint coupled runs
at dt=1/256,1/512,1/1024 complete from0 to .5625s. Earlier .5234375 checkpoint
paths are preserved only as controls. Incoming R,Z
and phase bound stay immutable through trials; complete state publishes once.
Core grows .600057→.654367; increments .0062815/.0031462/.0015735.
Successive common-time phase differences 1.54e-8/3.84e-9; traction 1.36e-5/
3.41e-6Pa. Admissible reduced curvature is positive at sampled accepted states.

The historical replay's incoming phase defect is .00518968: it is not a
consistent coupled equilibrium. Its +159.949J/m frozen profile cost differs
from instability; the corrected endpoint is locally stable. On a continuous
phase/exact-Maxwell diagnostic path its remainder is +.125178J/m, explained by
separate mechanical/friction/phase residual work. No compensating dissipation.
Fresh smooth-branch residual powers and external remainder decrease by ~4 per
halving; onset controls whole-interval errors. Normalization, derivatives,
energy identity, recurrence, publication and 1020-file preservation checks pass.
Production/defaults/specification equations unchanged; no native build/run.

[Short report and settings](P2_coupled_branch_report.md),
[continuous identity and algorithm](P2_coupled_branch_equations.md).
New files: coupled_branch.py, assess_coupled_branch.py, check_coupled_branch.py
under benchmarks/reconstructed_fault/propagation plus reports/handoff updates.
Primary evidence: coupled_branch_fresh_evidence; earlier checkpoint controls
retained in coupled_branch_evidence. Commands and costs in report. Read-only check:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/propagation/check_coupled_branch.py`.
Baseline remains de7beaa65846c7372711f7922abf1c5d9a745ee6/fault-propagation.
Next proposed bounded question is consistent continuation through the later
fast-tail interval to .625s before native adoption; no further stage authorized.

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

# Completed P2 Q2 normal-resolution decision (2026-10-09)

**Select 16×256 Q2/Q1, six normal Gauss points**, with unchanged corrected
formulation, consistent normalization/CPDI, accepted FE history, physics,
loading, bounds and tolerances; dt=1/64 s, T=.625 s. Reused16/32/64; added128
then256 because128 fails localized stress (8.41% versus5%). Native40-cycle
confirmation passes: whole/localized RMS1.09%/2.10%, storage .0126403% of
fixed reference peaks; matching discrete1D full phase1.293e-8, q1.690e-5Pa,
slip1.693e-11m. No healing. Proper incoming31/39 derivatives and selected
free curvature pass; .5s has no free nodes, .625s minimum eigenvalue .139611.
Measured cost2063.04s,2041916KiB peak RSS (one rank/thread, diagnostics included).
Baseline remains de7beaa65846c7372711f7922abf1c5d9a745ee6/fault-propagation;
all incoming Q3/three-point numerical artifacts and failed evidence are retained.
Production change: cache row-end iterators in constrain_phase_newton_system,
identical algebra. Observer change: use the existing coefficient/basis arithmetic
estimate for the manufactured Maxwell probe, retaining256eps. Both earlier
initial-only attempts and before-fix sources are retained; no physical failed
state is overwritten. The rebuilt core/plugin and all input hashes are recorded.

[Decision report: settings, convergence, cost, source changes and commands](P2_Q2_normal_resolution_report.md).
Evidence: benchmarks/reconstructed_fault/propagation/normal_resolution_evidence/;
check_normal_resolution.py verifies preservation and numerical gates.
Fixed-profile control separates velocity-space error from different obstacles.
Spatial storage error1.122e-4J/m, D_relax .179911J/m, absolute internal/boundary
work errors .175633/.204356J/m remain separate. Practical spatial targets pass;
time accuracy, full constrained/coupled stability and energy closure remain
unqualified. Next proposed work is a separately selected native time assessment
at this spatial configuration. No P3, material tuning, transport or broad MPI run.

---

# Current P2 bulk-quadrature accuracy/cost result

Updated 2026-10-09. **Consistent six-point coupled bulk rule verified; all three
native controls finish to .625s. Restricted phase/traction/slip target met by
16x32 Q2; stress/energy/time/full stability and P3 remain unqualified.**
[Readable answer, settings, curves, cost, source audit and diffs](P2_bulk_quadrature_report.md).
The corrected3D plane-strain law and physical settings are fixed. One explicit
normal rule feeds Stokes residual/Jacobian/preconditioning, bulk-slip map and
transpose, bulk phase energy/gradient/Hessian and diagnostics. Consistent
normalization and CPDI fracture remain unchanged. Signed II and positive J2
are distinguished in documentation; tensor helper code is unchanged.

| Evidence | Measurement | Interpretation |
|---|---|---|
| Rule selection4/6/12 | Six/twelve phi<=4.01e-8; independent force<=5.38e-4Pa; four fails selected thresholds | Six resolves quadrature below spatial error |
| Matched native/prototype | phi<=1.32e-8; q<=1.56e-5Pa; recurrence0 | Shared coupled operator/state reproduced |
| Q2/Q3 physical errors | Full phi .03823/.01374; q50.17/5.79Pa; stress RMS2919.6/404.4Pa | Q3 improves accuracy at2.11x cost |
| Cost | Q2/Q3/refined75.9/159.9/156.7s;265.1/276.7/339.4MiB | Sequential single-process measurements |
| Normal-refined Q2 | phi .00648; q5.05Pa; slip .0452% | Only tested native candidate meets all declared phase/q/slip targets |
| Remaining stress/storage | Q3 RMS404Pa/storage error11.8%; refined908Pa/55.9% | Restricted recommendation is not bulk-energy qualification |
| New free curvature | Positive sampled modes at .5/.625; proper incoming histories31/39 | Not full critical-cone/path/global stability |
| Bounds/KKT/publication | No measured healing; all unchanged targets pass; exact FE recurrence | Numerical checks separate from accuracy |
| Normalization/default | Source error<=1.67e-15, endpoints; two mature CSVs identical | Unchanged projection/default compatibility |

Final executable SHA0963b3aee8fac8fb35d5f15926549b2237d426544f03a359687a64c394d65a18;
GCC12.4 Release Unity/deal.II9.6.2/Trilinos14.2, same branch/HEAD. Nineteen
incoming snapshots, manual notation edits and three-point/failed evidence are
preserved. Four production files: one numeric setting/shared factory plus timing
scopes; no physical parameter, generic projector, helper, bound or history change.
Commands and hashes: benchmarks/reconstructed_fault/propagation/quadrature_evidence/;
checker: python3 benchmarks/reconstructed_fault/propagation/check_quadrature.py.

D_relax, profile storage, fracture, internal and boundary-lifting work discrepancies
are kept distinct. Prototype internal-work labels are reconciled with the native
boundary-lifting measure on the same frozen histories. No cancellation certifies
accuracy. Only one native normal-refinement comparator followed the cheap study.
No additional native timestep family, material/transport/MPI/restart campaign or
P3. Next uncertainty is stress/storage spatial resolution, with full stability
and accepted-step rollback/restart still open. The64-cell cheap result is context,
not permission or native qualification.

---

# Current P2 plane-strain and spatial/stability audit

Updated 2026-10-09. **3D plane-strain correction verified for the optional FE
owner; two short compatible controls finish. Spatial accuracy/full stability
remain unqualified; no P3.** [Report, equations, profiles and task-only diffs](P2_plane_strain_report.md).
The tensor helpers are correct and unchanged. The FE path now deviates raw
plane-strain gradients once, includes reconstructed zz in energy/loss, and
registers existing kappa trace subtraction at both Picard and Newton entry
points. Pressure and prescribed-adiabatic friction are not reinterpreted.
Legacy defaults, consistent normalization, accepted-state recurrence, bounds,
H+ diagnostics, histories and incoming manual notation edits are preserved.

| Evidence | Result | Qualification |
|---|---|---|
| Helpers / explicit3D | Block0, invariant<=5.55e-17 | Correct semantics; repeated deviation invalid |
| General beta / work | Algebra<=2.37e-16 / <=1.99e-16; native nonuniform<=3.24e-15 | Exact FE recurrence at ratios .00125,1,4 |
| Nondivergence operator | Missing7370.493519; final error1.27e-9 at unchanged allowance1.89e-9 | Newton residual/tangent corrected |
| Compatible/default controls | Small phase/surface byte-identical; loaded phase1.92e-15; mature two CSVs identical | Shear invariance/default compatibility |
| Native Q2 / Q3 | 40 steps to .625s; phi maxima .65744/.68338; healing0, recurrence0 | Numerical/history checks pass |
| Band widths phi/chi | Q2 .27885/.11915m; Q3 .27795/.10865m | Center and endpoints agree |
| Rich1D common-time phase | .03643 -> .01312 with enrichment | Improved but not spatial convergence |
| Frozen stress/integration | Q3 native RMS1e-9 vs accurate409Pa; bulk underestimation9.52% | Native3-point aliasing remains limiting |
| Sampled free curvature | Q2 minima9.136/2.450; Q3 final2.243; no negative free modes | Not full critical-cone/path/global stability |
| Normalization | Direct-source error<=1.23e-15 | Unchanged representation adequate for these fields |

Final executable SHA6fbedb652a2b0af8eb0d8b202b14d12edd0c93b26b8d8ee60c8155f2a57e0c50;
same GCC12.4 Release Unity/deal.II9.6.2/Trilinos14.2. Baseline HEAD/branch
remain de7beaa65846c7372711f7922abf1c5d9a745ee6 / fault-propagation. Eighteen
incoming snapshots and historical failed trajectories are preserved. The initial
Picard-only attempt is retained; the actual Newton route caught and resolved it.
Evidence: benchmarks/reconstructed_fault/propagation/plane_strain_evidence/;
read-only verification: python3 benchmarks/reconstructed_fault/propagation/check_plane_strain.py.

Five production files/six functions are listed in the report. Numerical correction,
benchmark diagnostics and documentation diffs are separate. No generic helper,
projector, new setting/default or physical parameter changed. No coupled quadrature
correction, time-refinement, restart/MPI/transport/rollback or P3 qualification.
The concrete next accuracy issue is matched normal mechanical/energy quadrature
before further velocity/time refinement. Historical affine instability persists
as a separate issue; fixed-anchor trace storage is phase independent.

---

# Current P2 native localization assessment

Updated 2026-10-09. **Bounded diagnosis complete; three native controls finish;
localization/stress accuracy and stability remain unqualified. No P3.**
[Report, equations, profiles, production inventory and specification diff](P2_native_localization_report.md).
The unchanged total-energy/accepted-FE-state model and direct irreversibility
are retained. G=1e6 Pa is a dimensional synthetic verification value.

| Outcome | Measurement | Supported conclusion |
|---|---|---|
| Original stall | KKT1.20765e-12; target1.12564e-12; independent arithmetic differences O(1e-14) | No tolerance increase justified |
| Exact tiny polish | KKT5.77e-15; original target retained | Numerical stationary residual repaired |
| Free curvature | Two negative modes; feasible +/-1e-4 decreases energy3.788e-10 J/m | Original affine trial is not a local minimum |
| Native controls | Affine/compatible/height2: 64 steps to1 s, no healing; FE recurrence0 | Fixed-dt numerical/history checks pass |
| Boundary effect | Center phi/chi FWHM: affine1/1 m; compatible .38795/.13305 m | Compatible sides retain a band; affine sides spread it across the height |
| Normal size | Height2 .44950/.13670 m; same ell/dx/dy/local shear rate | Added loading distance/compliance is explicitly part of this comparison |
| Native coarse 1D agreement | Common-time phase8.72e-9; traction1.30e-5 Pa | Native path agrees with matching Q1/Q2/Q3 reference |
| Rich 1D spatial repeat | 128/256 full phase .00036218; native/fine .04642 | Coarse native velocity/phase space remains limiting |
| Consistent normalization | Final max source error .0002798%; endpoints included | Representation is accurate for these fields; projection unchanged |
| Transfers/defaults | Three unused old-stress reads bypassed; 195 affine/16 compatible state files and two mature CSVs byte-identical | FE owner/default compatibility retained; generic projector unchanged |

Final executable SHA664c05fe66c9c1a7cb1c820ac906094af92b4d3f4b2462605128330d9dae3338;
GCC12.4 Release Unity, deal.II9.6.2, Trilinos14.2. Same worktree/branch/HEAD
(fault-propagation, de7beaa65846c7372711f7922abf1c5d9a745ee6). Incoming snapshot
and all original failed trajectories are preserved. Evidence/commands/source
hashes: `benchmarks/reconstructed_fault/propagation/localization_evidence/`;
read-only checker: `python3 benchmarks/reconstructed_fault/propagation/check_localization.py`.

Production changes are the near-root exact polish and exceptional dump, actual
Box-derived quadrature/coverage, pre-cancellation publication arithmetic scale,
and three FE-owner read bypasses. No physical parameter, normalization, H law,
phase bound, default, accepted-state rotation or generic projector changed.
Independent analytic stationary controls, energy/Hessian checks and quadrature
comparison pass; separate phase release/work errors remain in the report.
The fine height2 reference failed its unchanged KKT at1 s and is retained.

Stop at this assessment. Before further native qualification, select an explicit
spatial/stability investigation: richer bulk velocity resolution and the affine
negative-curvature branch issue must not be hidden by KKT counts. Native time/
spatial convergence, energetic accuracy, rollback, restart, MPI evolution and
transport/birth remain open. No new branch rule, kinetics, tuning or P3 selected.

---

# Current restricted native FE-strain P2 checkpoint

Updated 2026-10-08. **Optional production FE-state integration implemented;
representation verified; loaded trajectory incomplete. No P3.**
[Report, equations, actual specification diff and comparison](P2_native_FE_strain_report.md).
The user explicitly selected accepted FE strain, superseding the ownership-review
stop in the preserved sections below. Authoritative R=symgrad d_R=E_tot−E_in
updates exactly as d_R_new=beta d_R_old+(kappa/G)u_accepted for identical uniform
bulk/fault coefficients. Defaults remain particle-authoritative. Full total-energy
phase trials hold this accepted anchor fixed; H+ is diagnostic. Fresh serial fixed
geometry only; reject restart/conversion, transport, replenishment and variation.

| Result | Measurement | Qualification |
|---|---|---|
| Nonuniform native Maxwell | Relative <=2.43e-14, dt/lambda=.00125,1,4 | Exact FE representation; no stress projection |
| Accepted FE recurrence | Saved coefficient error 0 | One terminally published owner |
| Derivative/Hessian/poisoning | <=7.08e-11/6.29e-11; bitwise scratch independence | Full normalization adjoint; normal/exception restoration |
| Zero hold / mature default | No growth/release; pinned CSVs byte-identical | Startup and legacy compatibility pass |
| Loaded native | 19 accepted steps, t=.296875; no healing; tail increment .08217 | Incomplete at requested T=1 |
| Failed phase entry | t=.3125, KKT1.20765e-12 > target1.12564e-12 | Near-roundoff stall; tolerances not relaxed |
| Separate energy errors | D_relax .563841; absolute mechanical .886128 J/m | Not conflated with each other or solver convergence |
| Frozen accepted normalization | .02763% worst source error, endpoint | Passes .5% for these states only |

Fresh same-seed endpoint prototype completes; native/prototype fields differ
substantially (common-time phi .29613, local slip .01031 m). Full native Stokes
and lateral Dirichlet restrictions differ from its one-mode space. No exact
curve-reproduction, native temporal/spatial convergence or propagation claim.
Extended-precision assembly repeat also fails and is removed; all failed runs
are retained. Final core SHA a202b05044e9f33eba2324223b32b1383dbcced2b8301f4570d11c7f3e18beaa;
GCC12.4/Release/deal.II9.6.2/Trilinos14.2; same worktree/branch and HEAD
de7beaa65846c7372711f7922abf1c5d9a745ee6. Exact commands/source inventory/hashes
are in the linked report and native_fe_evidence. The 411 incoming snapshots are
verified; incoming tracked diff reconstructs omitted tracked files. See report
for the accidental old mature-output reuse and restoration of pinned files.

Next bounded issue is the stalled native nonlinear entry at fixed accepted
state, before any further coupled refinement. Native time accuracy, energetic
accuracy, localization, physical rollback, coupled restart, evolving MPI and
transport/birth remain unqualified. No material optimization, scaling or P3.

---

# Current P2 evolving endpoint comparison

Updated 2026-10-08. **Endpoint family complete; endpoint selected. Native entry
audit stops at a discrete energetic-state/ownership decision.**
[Report, equations, plots and source audit](P2_total_energy_endpoint_comparison.md).
Three dt=.015625/.0078125/.00390625 endpoint paths reuse identical retained mean
controls; one .001953125 endpoint repeat resolves the concrete work/tail question.
Same material, unloaded seed, total-energy residual and intersection quadrature.

| Outcome | Measurement | Qualification |
|---|---|---|
| Endpoint refinement | Max increment .08379 -> .01988; release 5.68834 -> .738156 J/m | Decreasing finite-drop slack; no persistent jump established |
| Independent mechanical error | Sum absolute error 6.62273 -> .755608 J/m | Decreasing; finest .467% boundary work / 3.14% fracture; not conflated with phase release |
| Actual field sensitivity | Endpoint256/512 phi .05533, stress1319.73 Pa, slip .00057125 m at common times | Fast growth still underresolved; mean256 can be closer to reference through leading-error cancellation |
| Selected endpoint balance | Weak bulk <=9.94e-11; slip <=8.53e-14 Pa m | Passes existing tolerance; mean remains diagnostic, not an endpoint-equilibrium pass |
| Pattern / scope | Reference core .70774; normal boundaries .26950 | Widening prescribed band with one free bulk velocity mode; no propagation or full Stokes qualification |

Seven saved-field checks pass bounds, irreversibility, phase/mechanical KKT,
actual slip deposits, Maxwell endpoints and separate work identities at unchanged
tolerances. 378 incoming artifacts, all old failures, mean trajectories and
production source are preserved. New files are two read-only assessment/check
scripts, plots/records and documentation; no native build/run or production change.

Native entry identifies Pi[Z_p Delta chi_p] != Z_Q Delta chi_Q: particle-owned
publication and the prototype's authoritative QP energy are not one state.
The report derives the additional transfer adjoint and presents particle-owned
projection-aware mechanics, new QP history, and an exact FE internal-strain
encoding for the uniform unloaded case. The latter is recommended; its accepted
state recurrence passes a saved-field check to 1.39e-16. It changes authoritative
bulk-history ownership, which is not yet assigned in the native specification.
The conditional native implementation is not complete. This governing
discretization choice must be resolved before installing the optional model;
the straightforward projected-energy mechanical pullback would have at least
3777 uncontrolled velocity directions in the actual native Q2/Q1 space (a rank
bound for that proposed operator, not a defect of current native Stokes);
the rank obstruction does not rule out the FE encoding or all native routes.
The source and existing specification agree today. No generic-projector repair,
material tuning, new loss/driver, history reset or P3 is selected. Earlier P1/
direct-irreversibility qualifications and native lifecycle/spatial limitations stand.

# Preserved loaded P2 prototype time comparison

Updated 2026-10-08. **Loaded comparison complete; production unchanged.**
[Report/settings/decision](P2_total_energy_time_comparison.md),
[algorithm](P2_total_energy_algorithm.md). Mean balance is the energy-consistent
implementation candidate; endpoint is more accurate in the frozen monotone
control. Onset remains underresolved; no qualified native timestep or P3.

| Result | Measurement | Qualification |
|---|---|---|
| Surface sampling repair | Constant-load error 77.78% -> 7.78e-15 using intersection Q3 | Prototype quadrature correction; no projector/physics change |
| Frozen reference | Endpoint256/512 stress difference .002864 Pa | Time reference for the declared restricted mechanical space |
| Corrected evolving refinement | Max increment .23865 -> .039716; total release 72.68 -> 1.52157 J/m | Decreasing release/increments; persistent jump not established |
| Finest response | Core .70664887; q135.53781 Pa; slip .12684844 m; F23.97106 J/m | Symmetric through-fault profile; substantial tail/boundary growth retained |
| Remaining temporal error | Phi128/256 .078704 common-time / .027650 final; endpoint bulk326.28 Pa m | Accuracy uncertainty remains; mean KKT is not endpoint equilibrium |

32 completed common-path trajectories include the retained undersampled family
and corrected repeats; prior unstable preload controls/failed entries preserved.
Scaled phase KKT <=8.319e-12, no healing; original tolerances retained. General
Maxwell kernels, energy, inelastic strain and material unchanged. 240 incoming
artifacts preserved under `total-energy-time-baseline`; same HEAD/branch.
Source functions, commands, raw fields, plots, force/work gates and preservation
are in the report and `total_energy_time_evidence/manifest.json`.

Next implementation direction: optional native mean-balance total-energy model,
with authoritative accepted-state ownership, projection adjoint and common
integration measures. This task implements no production adoption. Full-space
mechanics/spatial accuracy, stress transfer, rollback/restart, transport/birth,
distributed evolution and general friction remain unqualified. No kinetics,
material optimization, stress-scaling qualification or P3 is inferred.

# Propagation task status

Updated 2026-10-08. **Complete total-energy candidate and isolated verification;
production remains unchanged.** [Algorithm](P2_total_energy_algorithm.md),
[compact verification report](P2_total_energy_verification.md).
Instantaneous recoverable bulk/fault phase energy uses the exact normalization
projection adjoint and immutable accepted-phase bound. General exponential
Maxwell evolution follows an instantaneous elastic redistribution. The chosen
candidate uses time-averaged mechanical equilibrium, not production endpoint
balance. Peak H is diagnostic only in the prototype.

| New result | Measured evidence | Qualification |
|---|---|---|
| Unloaded prescribed band | No phase/startup growth, all work/losses zero | Constrained equilibrium of named isolated seed |
| Discrete derivative / phase KKT | Refined relative error <=2.713e-8; KKT <=3.766e-13 | Correct supplied-trial energy/adjoint and irreversibility |
| Maxwell + mean-balance ledger | <=9.837e-14 J/m defect at dt/lambda=.00125,1,4 | Exact accounting including explicit phase release |
| Finite phase relaxation | 5.24085/5.85812 J/m release beyond fracture | Nonnegligible algorithmic slack; physical jump equality unqualified |
| Endpoint comparator at ratio one | .118317/.473700 J/m exchange defects | Endpoint agreement does not qualify finite work |

27 accepted isolated steps, three elastic and three free-bulk-mode probes,
nine rejected-trial publication
checks; no ASPECT trajectory or core rebuild. Same HEAD/branch and 211 incoming
artifacts preserved under `total-energy-prototype-baseline`. Direct irreversibility
and frozen .00625 normalization qualification remain preserved. No native phase
law adoption, further coupled refinement, tuning or P3. Next review concerns
phase-release interpretation and averaged-equilibrium temporal accuracy before
any production integration; stress transfer and earlier lifecycle/spatial limits
remain unqualified.

## Preserved bounded review checkpoint

Updated 2026-10-08. **Bounded review checkpoint complete; direct irreversibility
remains successfully verified for the fixed-mesh P2 fixture. Stop for review.**
[Model/settings/source inventory](P2_implementation_checkpoint.md),
[implemented versus proposed equations](P2_formulation_comparison.md).
No production code, phase law, model option, physical history or coupled run changed.

| New frozen assessment | Maximum normalized-source error | Outcome |
|---|---:|---|
| Consistent I, .0125 m, strong fields | .6266–.7034% at accurate integration | Fails provisional .5%; permitted .00625 repeat needed |
| Consistent I, .00625 m, all six accepted fields | .0250722% worst; endpoint .00722251% | Passes **these frozen fields only** |
| Split Q32/Q48 uncertainty | <=3.45e-15 relative | Separate from representation accuracy |

.00625 m aligns bulk phase knots. Retained coupled runs still use .1 m;
no spatial/energetic qualification of a new coupled configuration follows.
The proposed total-energy phase variation and the role of peak H need review;
the implemented reduced-fault peak law is unchanged. G=1e6 Pa is a dimensional
synthetic verification value, not a calibrated rock property. Energetics and FE
stress representation remain separate limitations. Further coupled time refinement,
material tuning, stress scaling and P3 are deferred, with physical rollback,
coupled restart and transport/birth requirements still unqualified.

Baseline HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6, branch fault-propagation.
183 current artifacts preserved under `review-checkpoint-baseline`; existing
production/binary/physical evidence unchanged. Incoming notation edits in
pf_rsf.tex and P2_accuracy_energy_derivation.md are preserved; the latter's prior
manifest mismatch is documented, not erased. Commands, exact cumulative production
diff, numerical records and preservation checks are linked from the checkpoint.

## Preserved accuracy/energy assessment

Updated 2026-10-07. **Direct irreversibility successfully verified for the fixed-mesh
P2 fixture; focused accuracy/energy assessment complete; P3 not ready.**
[Report](P2_accuracy_energy_report.md), [derivation](P2_accuracy_energy_derivation.md).
No production source or selected model option changed. Prior failures are preserved.

| Assessment | Measured result | Qualification |
|---|---|---|
| Frozen strong-field consistent I, .025 m | .8569–.9698% max source error at fixed accurate integration | Fails provisional .5% target; native Q3 gives .9013–1.0193% |
| Native zero-input stress, 18 paths / 336 updates | Particle error <=1.79e-12 Pa; additional FE drift <=1.59e-12 Pa | Maxwell partitioning passes; FE representation error up to 5.24% remains |
| Profile work, retained dt=.03125 | -151.785 J/m, bulk identity <=3.00e-11 J/m | Mechanically accounted; closed phase-energy budget unresolved |
| Frozen nonuniform stress energy variation | Bulk +.65894 vs fault -.75516 J/m | Missing bulk-profile term for a total-energy potential; no correction adopted |

The conditional .03125/.015625 fresh coupled runs were not launched: none of the
selected spacings passes the accuracy target and a governing energy decision is
needed. Surface accuracy and stress representation are separate spatial errors;
ordinary integrated exchange defects improve under refinement but do not remove
profile energy. Remaining P2 also includes physical rollback, coupled restart,
transport/birth histories, evolving MPI, general support admission and scaling.
No material search or P3. Next requires review of the derived total-profile
phase variation before expanding implementation.

Baseline HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6, branch fault-propagation;
3,360 incoming artifacts copied/hash-verified under accuracy-energy-baseline.
Same pinned core SHA 44fd12df9313033820771066d1c754f95ab8fa1c9ad8739b17f56c6f86e810dd.
New diagnostic plugin and commands are in the report and
`accuracy_energy_evidence/manifest.json`; final local native probe exit 0 (11.29 s).
Existing irreversibility bounds/KKT/healing/history checks are reused unchanged.

## Preserved direct irreversibility result

Updated 2026-10-07. **Direct phase irreversibility verified; three fresh P2 paths
complete to the same t=1 s and improve under refinement.**
[Report](P2_irreversibility_report.md),
[actual specification diff](../../../benchmarks/reconstructed_fault/propagation/irreversibility_evidence/specification.diff).

Optional `Enforce phase irreversibility` (default false) snapshots accepted
old_solution phase once and holds it as the physical lower bound. Initial
relaxation stays [0,1]; recoverable H, unloaded initialization, surface Z history
and bulk-conjugate work stay fixed. Failed phase solves publish nothing; the
option rejects continuation after nonconvergence. Original failed box-only
entries remain failed and preserved. Same-H replays reproduce 556 prior CSVs;
KKT/complementarity and restoration pass, with <=1.52e-10 core corrections.

| dt (s) | Final time (s) | Phase maximum | Largest step increment | Healing | H integral gain (J/m) |
|---|---:|---:|---:|---:|---:|
| .125 | 1 | .7316116394 | .08774616 | 0 | 4621.811968 |
| .0625 | 1 | .7409356550 | .04471502 | 0 | 3960.501865 |
| .03125 | 1 | .7462178833 | .02379854 | 0 | 3738.376738 |

At common t=1, full-field phase differences shrink .02480→.01721, sampled
traction differences 52.86→25.63 Pa. Common-slip differences remain appreciable.
The constraint affects tails; old accepted traction/slip intervals change only
at numerical precision. Normalization is positive, with 6.2–6.4% source spatial
error. Cumulative profile work is -94.51/-127.09/-151.79 J/m and remains separate
from cohesive storage and fracture. Numerical completion and direct admissibility
do not qualify spatial accuracy, timestep independence or full energy balance.

Preserved baseline: HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6, branch
`fault-propagation`, incoming snapshot 3,154 artifacts. Production edits only
phase_field.h/phase_field.cc; exact functions and commands in report. Final
GCC12.4 executable SHA256 44fd12df9313033820771066d1c754f95ab8fa1c9ad8739b17f56c6f86e810dd.
Three native failure probes, zero-load control, configuration rejection and
622 native cohesive assertions pass. MPI/restart for the new option was not run.

**Selected task complete. Stop before material optimization/P3.** Remaining P2:
normalization spatial representation, evolving-profile energy/work and stress
transfer, temporal error, physical rejected-step rollback, coupled restart,
transport/birth histories. Stress scaling remains unclaimed.

## Preserved preceding coupling-repair result


Updated 2026-10-07. **Two coupling repairs verified; evolving P2 stops on
resolved tail healing. No qualified damage-feedback baseline or P3.**
[Report](P2_coupling_report.md), [derivation](P2_coupling_derivation.md),
[actual specification diff](../../../benchmarks/reconstructed_fault/propagation/coupling_evidence/specification.diff).

Optional surface elastic-slip history eliminates q reprojection; optional
bulk-conjugate work uses the native Stokes-QP adjoint and ds resistance with
matching R/K/G. Legacy defaults and the mature-only guard remain unchanged.
Nonconstant-history tests (1,456 steps), native cohesive units (622 assertions),
three fixed-phase trajectories and a four-step serial/two-rank check pass.
All 99 historical default CSVs reproduce. Work adjoint error <=6.284e-15;
matched-Q3 endpoint exchange defect <=3.91e-13 W/m. Stress-transfer errors
remain separately measured. Fixed-phase q refinement differences shrink 2.53→1.13 Pa.

One fixed-phase pilot selects a=0.256 for every evolving path. Same material,
recoverable H and unloaded core0.6; frozen particles and fixed full-cross-section
association. No per-dt tuning. Lag-aligned phase differences shrink 0.01588→0.00749,
but each trajectory hits the unchanged 4.922e-12 healing gate:

| dt (s) | Last accepted t | Failed entry t | Phase decrease |
|---:|---:|---:|---:|
| 0.125 | 0.875 | 1.0 | 1.848e-9 |
| 0.0625 | 0.8125 | 0.875 | 1.577e-7 |
| 0.03125 | 0.84375 | 0.875 | 7.357e-7 |

Identical-H scratch repeats leave all failed fields unchanged; Jacobian
corrections <=1.06e-16, KKT<=7.92e-16. Numerical uncertainty cannot explain
healing. Positive Jacobian off-diagonals preclude assuming a discrete comparison
principle. Accepted I is finite/positive but h/I spatial error reaches 2.11%;
profile work and discrete phase-energy derivative remain separate limitations.

Existing branch fault-propagation / HEAD de7beaa65846c7372711f7922abf1c5d9a745ee6;
981 prior pinned artifacts preserved under coupling-baseline. Source functions,
commands, binaries and plots are in the report and coupling_evidence/manifest.json.
Next decision: spatial/comparison-principle diagnosis versus intended material
irreversibility, without silently adding a lower bound. Stress scaling, coupled
restart, physical rollback, evolving MPI, transport/birth and full profile-energy
closure remain unqualified. Historical entries below are retained.

Updated 2026-10-06. **Recoverable-energy revision implemented and locally verified;
three unloaded P2 trajectories complete, but coupled qualification fails.**
[Report and plots](P2_recoverable_report.md), [derivation](P2_recoverable_derivation.md).

Explicit optional `Phase driving history = recoverable energy` now publishes
max(H_old,q²/(2Gg²)); `Initial cohesive state = unloaded` retains seeded H and
relaxed core0.6 while initializing q=Z=0. Defaults/preloaded evidence remain intact.
560 native path checks give H/H0=1.7777778 for loading and 1 for relaxation/cycles/
hold, at all three dt and both viscosities. 42 evolving-phase energy checks pass;
zero-load phase/H growth is exactly zero. All 99 legacy trajectory CSVs reproduce.

| dt (s), a=0.016 | Phase-max gain | Maximum nodal gain | Integral H gain (Pa m²) |
|---|---:|---:|---:|
| 0.125 | 0 | 0 | 2.51635 |
| 0.0625 | 2.86463e-8 | 4.21140e-6 | 2.50612e-4 |
| 0.03125 | 6.34079e-8 | 9.32168e-6 | 2.92778e-5 |

58 physical steps including zero-load controls pass bounds, healing, mechanical
and phase residual/publication checks. Normalized-source error <=1.277e-7.
Damage is tiny and late at side-boundary transverse flanks. One-step lag is
reported; feedback convergence is not established. Native Q1→particle→surface
traction transfer is not identity (max entry defect .003345); repeated projection
adds a dt-dependent smoothing rate. The existing surface particle-volume measure
also differs from localized bulk work. Internal-work budget defects stay large;
this is not an independent boundary-work balance. Generalizing the mature-only
work-consistent assembly or changing q publication is a further coupling scope.

No loading pilot was needed: seeded thresholds are crossed. No stronger ramp,
material search, factor-30000 run or P3 followed the failed coupling gate.
Current executable e17833bed2efebd3277e7368240a5cb759fc8033be1d0a5d0677921c0a55ec67;
419 prior pinned artifacts preserved in `build-propagation-integration/recoverable-baseline`.
Exact functions, commands and limits are in the report and
`recoverable_evidence/manifest.json`. Next: consistent cohesive-traction publication
and bulk/fault work exchange, then repeat the feedback gate and conditional scaling.

The entries below retain preceding historical statuses.


Updated 2026-10-06. **Material search paused; native H semantics diagnosis and
unchanged reference reruns complete. A scientific definition is required before
changing the production law.** See [P2_history_report.md](P2_history_report.md).

The governing derivation and current source both take a maximum of a one-step
work-potential derivative. At fixed partial damage C=O(dt); in the elastic limit
C is an energy increment, not an energy state. 560 native path evaluations
(dt/dt/2/dt/4, loading/relaxation/cycles/hold, reference and elastic viscosities)
and 622 existing cohesive assertions pass and confirm this mismatch. Accumulating
positive increments would cause artificial elastic-cycle damage. Peak recoverable
energy versus signed work including viscous dissipation remains a model decision;
projected initial q also does not invert retained particle H pointwise.

Fresh unchanged reference reruns at dt=0.125/0.0625 reproduce all 99/187 diagnostic
CSVs byte-for-byte: phase gains 0.002326043753/0; cumulative H gains
36.20138531/9.942159e-7 Pa m². Bounds, equilibrium, healing, normalization and
publication checks pass; feedback refinement fails. Production code and executable
are unchanged. Factor-30000 scaling is conditional on resolved history and was not
run. No P3, new sweep or qualified gradual-damage claim. Prior 755 pinned artifacts
are preserved under `build-propagation-integration/history-baseline`; commands,
functions, physical alternatives and evidence are in the report and
`history_evidence/manifest.json`. Next: select driving-state and initialization
semantics; then implement/verify and run the conditional scaling check.

The following entries retain the preceding material-study status.


Updated 2026-10-06. **Bounded P2 material study complete. Numerical checks pass,
but no gradual, timestep-robust damage model is qualified. Stop before P3.**

[Material report and plots](P2_material_report.md),
[model/settings](../../../benchmarks/reconstructed_fault/propagation/MATERIAL_STUDY.md),
[source-change overview](SOURCE_CHANGES.md). Previous high-load and
[normalization follow-up](normalization_report.md) failures remain preserved;
this fresh lower-core study does not change their status.

Fresh core=0.6 cases retain G=1e6 Pa, c=1000 Pa, eta=1e8 Pa s, ell=0.15625 m,
p=1, consistent I_h, zero radiation damping and the fixed-geometry/fixed-parent
control. The three permitted reference amplitudes were exhausted: a=0.004 and
0.008 have no H/phase feedback; a=0.016 gives two late phase-changing steps.
The common selected ramp is a=0.016 over 0–1 s.

| Case | Final phase-max increase | Integral H−H0 (Pa m²) |
|---|---:|---:|
| Gc=26.6667, dt=0.125 | 0.00232604 | 36.2013853 |
| Gc=106.6667, dt=0.125 | 0 | 0 |
| Gc=26.6667, dt=0.0625 | 0 | 9.94216e-7 |

The primary case is inactive, so slower post-onset damage is unestablished and
the optional stronger Gc was not run. Refinement is unacceptable for damage
feedback despite similar mechanical paths. Native candidate/max-H diagnosis
shows the partially damaged fixed-phi candidate scales as O(dt), and crosses
the nonzero stored-history threshold on the coarse path but not the fine core
probes. One-step phase lag is accounted for; final fine H remains unconsumed.
CPDI/pointwise phase differences are <=3.33e-16 in these fixed regular domains.

All 48 physical steps are accepted; zero healing, bounded phase, KKT<=2.713e-15,
mechanical residual<=6.040e-9, and history-publication checks pass without changing
tolerances. Normalization is finite and positive; worst measured normalized-source
representation error is 0.15625% at the reference endpoints. This is not 1e-8
spatial accuracy. Native calibration prints m≈128/512 and E≈64/256 Pa, with Hc=0.5.

No production C++ or core executable changed. Added benchmark observer, fixtures,
runner/checker/plots and documentation; removed two old core=0.8 assumptions only
from shared observer diagnostics. Preserved all 282 prior pinned artifacts.
Commands, identities and changed-file purposes are in the material report and
`material_evidence/manifest.json`. Recommend retaining the reference/core0.6/a0.016
fixture for scientific review of the H candidate/max semantics before more tuning.
No H law, cap, damping, support or geometry change was made. Coupled restart,
physical rejected-step rollback and transported/birth histories remain P2 work.
The following sections retain the preceding package's historical evidence.

## Current evidence and baseline

[P1 integration report](P1_integration_report.md), [P2 report](P2_report.md),
[approved integration plan](P1_integration_plan.md). Historical [prototype](P1_bounds_report.md),
[tail diagnosis](P1_tail_report.md), [failed equality P1](P1_report.md) and
[P0](P0_report.md) remain preserved. The user's fixed-support selection supersedes
the prior support blocker; it does not select a normalization or H-law redesign.

| Item | Value |
|---|---|
| Worktree / branch | `/home/ein/repository/aspect-fault-propagation` / `fault-propagation` |
| HEAD | `de7beaa65846c7372711f7922abf1c5d9a745ee6` plus preserved incoming work and uncommitted integration |
| Build | `build-propagation-integration`; GCC 12.4.0 core/final observers, OpenMPI 5.0.6, deal.II 9.6.2, Trilinos 14.2/Epetra, Voro, Release |
| Final P2 executable SHA256 | `ab3e920506cbc26025ed9547a38d5cf184ce79330ebd24572c60f2dfb3868e5c` |
| Other worktrees / publication | Untouched / no commit or push |

## P1 support resolution

Added default-off `Fault reconstruction / Association support = full normal
cross-section`, restricted to one straight fixed prescribed fault spanning a
2D Box's tangential extent. Width comes from actual domain/fault coordinates:
0.50000000000001588 here. Sharp geometry/endpoints, phase boundary data and native
H initialization are unchanged. The manager builds parent and actual Stokes-QP
associations before native cohesive initialization. All cross-section particles
remain eligible for the existing constitutive update; no new phase mask exists.
Existing manager state preserves widths through restart and rebuilds caches.

| Affected straight P1 check | Result |
|---|---|
| Initial H before/after relaxation and association | Exact; 9,216 parents; identical across ranks |
| Phase history versus prior complete narrow-strip trajectory | Exact in serial |
| Serial/two-rank phase-history difference | <=1.11e-16 |
| Unassociated particles / mechanical QPs / omitted source | 0 / 0 / 0 |
| Box bounds, KKT, repeats, holds, publication, scratch/failure handling | Pass |
| Native restart after first +5%, distinct old/older phases | Exact through remaining schedule, one and two ranks |

Mechanical quadrature error remains about 6.7e-6 independently of zero association
omission. Padded finite-seed/cap behavior and prior compatibility evidence are
reused where unaffected; no broad matrix was rerun. Original equality failures
remain. Sampling accuracy is not established by box admissibility.

## Initial P2 findings

Fixed mesh/fault, fixed particles using the existing documented frozen-RK2
control, native mechanical H, stateless non-weakening friction. Initial phase,
H and cohesive state agree across dt=0.25 and 0.125 s. Loading is the identical
quasistatic shear-rate ramp ux=0.4*t*(y−0.5), uy=0. No transport/birth, geometry
update or nucleation qualification. The initial subcritical damped pilot is
retained; physical inputs were not retuned after the failure appeared.

| Measurement | dt=0.25 | dt=0.125 |
|---|---:|---:|
| Last accepted time | 0.25 | 0.25 |
| Cumulative integral of H increase there | 25158.86946 | 18636.97874 |
| Maximum current phi there | 0.80847755 | 0.89503304 |
| Next consumed-phase time, before failed mechanics | 0.5 | 0.375 |
| Consumed-phase KKT | 8.526e-16 | 2.409e-15 |
| Maximum consumed-phase increment | 0.22031456 | 0.11731760 |
| Incremental healing at failed entry | 0 | 4.2683e-11 |
| Minimum current projected I_h | −7474.77608 | −10087.93124 |

The native consistent Q1 projection maps positive transverse integrals to negative
nodal I_h. Independent Q16/Q32 integration reproduces the negative projection;
this is not a support omission or uncertainty-obscured sign. Identical-H repeats
and Jacobian correction estimates are ~1e-16; the fine failed-entry healing
exceeds the retained straight allowance 4.922e-12. No previous-phase bound, floor,
mask, H-law change or compensating factor was added.

Off-original-band H does not increase through the common accepted time. A diffuse
phase tail does: maximum particle phi is 0.00152347 on the half-step path. Failed
phase entries reach off-band nodal maxima 0.01308695 / 0.004064795; these are not
accepted complete mechanical steps or new reconstructed geometry.

One-step lag matters: coarse phase at 0.25 still consumes initial H; half-step
phase consumes H at 0.125. Cumulative H at common time is 25.9% smaller with half
dt, but both paths stop before the requested 1 s interval. This does not separate
splitting error from historical candidate/max-history feedback loss. No third
timestep was spent on an already inadmissible normalization trajectory.

## Changes, commands and verification

This continuation changes manager support configuration/construction and its
header, extends benchmark observers/runners/checkers and test access, and updates
design/specification/AGENTS/task documentation. It also corrects a demonstrated
mechanical arithmetic-check defect: fresh continuity residual verification now
includes the nonzero Newton correction's arithmetic scale when base velocity
and continuity RHS are zero. The existing gamma formula, nonlinear cap, RHS
compatibility guard, convergence targets and physical laws remain unchanged.
A targeted regression and all Stage-I units pass: 20,141 assertions in 15 cases.

[Support commands](../../../benchmarks/reconstructed_fault/propagation/INTEGRATION.md),
[P2 commands](../../../benchmarks/reconstructed_fault/propagation/COUPLED.md),
[P1 numerical gate](../../../benchmarks/reconstructed_fault/propagation/integration_evidence/full_support_checks.json),
[P2 numerical gate](../../../benchmarks/reconstructed_fault/propagation/coupled_evidence/summary.json),
[manifest/preservation](../../../benchmarks/reconstructed_fault/propagation/coupled_evidence/manifest.json).
The P1 checker exits 0; P2 checker exits 2. Final local P2 attempts take roughly
17–25 s and <270 MiB reported child RSS, with ~2.2 MB outputs. All failed attempts,
source/build identities and historical evidence are retained.

## Decision and remaining scope

Resolve how positive normalization is represented/resolved on the fixed fault
under strong along-fault phase variation before repeating the complete coupled
loading interval. Investigate the separate small phase decrease as an admissibility
outcome; do not silently impose a previous-phase constraint. No scientific fix
for either issue has been selected or implemented.

Still unqualified: complete dt refinement of native mechanical feedback, physical
rejected-step rollback, coupled restart, transported/birth histories, and broader
MPI coupling. The predefined single-fault region is not a general support-admission
rule for multiple faults, bulk nucleation or propagating endpoints. No P3 or
propagation. On resume read AGENTS, PLAN and both current reports, then recheck
HEAD/local changes and the user's next scientific selection.
