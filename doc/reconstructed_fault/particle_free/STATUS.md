# BP3 DG server startup review

17 October 2026. The current task selects a stopping/output repair and a
**bounded server startup package**, not a server launch or further science run.
[Review report](BP3_DG_STARTUP_REVIEW.md),
[direct server procedure](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/README.md),
[resource record](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/RESOURCES.md).

The complete stop request, heavy/profile output and selected diagnostic-field
branches now use collective decisions. The two-rank rank-1-only wall-stop control
writes terminal diagnostics/checkpoint without hanging; its restart matches the
uninterrupted step-five history, stress, loads and particle audits exactly.
The existing scheduled-output/restart regression is retained. All four inputs
validate; rebuilt native unit tests pass 124 cases/118034 assertions.

`startup.prm` stops at accepted step 5; `startup-resume.prm` continues to 8.
Both checkpoint every accepted step and use a 7200 s graceful wall budget.
They preserve event geometry/mesh/physics and native adaptive 4e6/4e7 s caps,
log-state .2, growth/cutback controls, initial interval 4e6 s and physical end cap.
`parameters.prm` and `resume.prm` retain the separate first-event experiment.
The source package includes explicit SUNDIALS requirements and checkpoint-read,
write and dense-output timing/accounting guidance. Full-grid replicated-history
checkpoint memory and cost still require actual server measurement.

Local fixture qualification is completed; full-grid startup is prepared; the
first-event trajectory is unqualified and unlaunched. Spatial/endpoint temporal
failures and `strict_reference_match_passed=false` remain. Next action: the user
builds the matching stack, runs/inspects the bounded startup and resume, and
collects resource evidence before deciding on first-event continuation. No
limiter/filter/constitutive changes or further stages are authorized here.

---

# DG BP3 first-event source handoff

17 October 2026. The user now selects preparation of a **first-event server
experiment** and explicitly selects native adaptive BP3 time controls. The
bounded pilot and all its evidence below remain intact.
[Package report](BP3_DG_FIRST_EVENT_PACKAGE.md),
[manual build/run recipe](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/README.md).

The source distribution targets ASPECT at `$WORK/build/aspect/pf-rsf-dg/build`,
plugins at `$SCRATCH/bp3_dg/plugin/build`, and direct
`ibrun ./plugin/aspect-release parameters.prm` from `$SCRATCH/bp3_dg`.
Unlimited domain-free DGQ2, Gauss6/source3, Helmholtz20 and all native laws remain.
Native first/max timestep caps are 4e6/4e7 s with log-state bound .2; first-event
completion, 1500-year end cap and 82800 s accepted-state wall stop replace only
the separate pilot's bounds. Native checkpoints occur every ten accepted steps
and on graceful termination. No server job/full-geometry trajectory is launched.

A default-off observer output schedule reduces dense exports while preserving
per-step checks/profiles/weak loads. Two-rank regression and checkpoint scheduling
controls pass, with all final compared histories/fields bit identical. Core
source/defaults and the independent publication guard are unchanged. Surface
representation, spatial and endpoint temporal accuracy limits remain; strict
reference matching remains **false**. Source packaging/configuration and plugin
build checks do not qualify a full first event. Next action belongs to the user:
build on the matching server stack and launch/inspect the selected experiment.

---

# Bounded mature DG pilot qualification complete

17 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved
incoming work. **Unlimited domain-free DGQ2 is selected for the next server
pilot.** Particle controls remain comparison evidence; neither backend is an
accuracy reference. [Report](PF6_MATURE_PILOT_REPORT.md),
[scope](PF6_MATURE_PILOT_FORMULATION.md), [source audit](SOURCE_AUDIT.md),
[evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_pilot/README.md),
[bounded server preparation](../../../benchmarks/reconstructed_fault/particle_free/bp3_dg_pilot/README.md).

- Fixed-FF mechanical refinement: 35342 cells, finest 1.953125 m, exactly
  unchanged Q1 phase on all 36575 overlay vertices; native I difference
  <=8.52e-16 relative. Local tensor change is 6.59765 kPa, increasing the stress
  signal. This measures sensitivity; spatial accuracy remains unresolved.
- FF dt/dt2/dt4 and paired finer-mechanics dt2 controls reach T=600000 s.
  The signed mechanical contrast changes by .1935 Pa localized tensor and
  .247/.380/.130 Pa shear/raw/filter projection after halving dt. Absolute
  endpoint temporal loads still fail the 1% shear/filter screen; estimated
  temporal order is about .925. No WENO/threshold or filter settings changed.
- Mature native serial/2/4-rank ownership, serial/cross-rank restart, actual
  outer rejection, native nonlinear cutback and one-owner negative controls
  pass. Serial split restart and completed-trial rejection are bit identical;
  other compared tensor differences are <7.2e-9 Pa RMS. Every accepted step
  publishes once; rejection changes no accepted history/property bytes.
- Intended full-geometry surface initialization: 5776 nodes at requested 10 m,
  preserved 15/18 km material anchors, independent RSF defect <=2.23e-16.
  Q1 state representation retains a 49.704 Pa interior prestress imbalance;
  no compensating stress change. This qualifies initialization, not mechanics.
- Fixed diagnostic IDs/coordinates show zero motion, births or deletions.
  Particle domains/CPDI/face data remain disabled; remaining point dependencies
  are initialization/bookkeeping, stationary registry and the retained independent
  publication validator. Accepted mechanics/loads use DG history and mesh data.
- 124 native unit cases / 118034 assertions pass; standalone reference 55/2625.
  All 12597 entry files remain, all 34 material defaults match entry and three
  retained evidence archives are unchanged. `strict_reference_match_passed=false`
  and all prior errors remain. Functional checks do not override accuracy failures.

The separate full-geometry stationary pilot is prepared and parser-validated,
not launched: unlimited DGQ2, Gauss6/source3, Helmholtz20; dt<=50000 s,
T<=400000 s, <=8 positive steps, accepted-step checkpoints, 7200/7500 s wall
bounds. Authoritative full tensors and perturbation weak loads are enabled.
Next task: explicitly launch that bounded pilot on a compatible server stack,
measure conditioning/cost and inspect weakening behavior under these accuracy
limits. No full event, phase evolution, transport, AMR or moving geometry follows
automatically. All prior status/evidence below is retained.

---

# Mature mechanical/profile-resolution separation complete

17 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved work.
[Measured report](PF6_MATURE_RESOLUTION_REPORT.md),
[declared scope](PF6_MATURE_RESOLUTION_FORMULATION.md), [source audit](SOURCE_AUDIT.md),
[evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_resolution/README.md).

- Six fresh serial Gauss6 trajectories reach T=600000 s: coarse/coarse (CC),
  fine mechanics/exact coarse phase (FC), and fine/fine (FF), for DG and matched
  stationary particle controls. Unlimited DGQ2, source Gauss3, native laws,
  loading, 59-node surface and all limiter/production defaults are retained.
- FC exactly prolongs only the coarse **step-zero** Q1 phase, including hanging
  values and the coarse exterior Q1 lattice. All physical vertices match exactly;
  common-point phi defect <=2.22e-16, chi <=1.11e-16 per m, nodal I <=9.09e-12 m.
  No history/normalization values are imported and no domain is needed by DG.
- Final DG local tensor differences: mechanics 24.417 kPa, profile 35.536 kPa,
  combined 29.163 kPa; effect cosine -.5811. Whole-domain differences are
  3.125/3.944/3.540 kPa. Local Cauchy differences include pressure. Mechanical
  refinement at fixed coarse profile increases the physical tau signal
  30.205→36.375 kPa; FF is 8.174 kPa. These are differences, not accuracy errors.
- Native bulk-work mass changes with quadrature partition/profile despite the
  fixed surface. Compare background-subtracted **own-mass** weak loads and a
  common geometric diagnostic functional, recording background-measure effects.
  Mechanics-only raw-normal projection changes are 425.9 Pa interior / 401.3 Pa
  endpoints; profile-only 554.0 / 6052.3 Pa. The combined interior change is only
  144.9 Pa. Small combined changes can conceal cancellation.
- DG/particle matched initial mesh/phase/quadrature/physical identities are byte
  identical; first positive-step tensor differences <=1.20e-9 Pa. Final local
  differences CC/FC/FF are 18.593/8.689/5.252 kPa. Neither backend is an accuracy
  truth. All 66176/184448 particle IDs/coordinates remain fixed in each run.
- Independent saved-coefficient reconstruction retains all nine modes and gives
  <=7.14e-15 scaled T defect. Native Maxwell/state/work/single-publication checks
  pass; CC/FF reproduce 32 retained field CSVs byte for byte; default particle
  Gauss3 surface fields reproduce exactly. All 34 material defaults match entry;
  55 unit cases/2625 assertions pass. Eight native attempts total 743.68/3600 s.
- Every old failure remains, including `strict_reference_match_passed=false` and
  the 2.48% temporal perturbation-load screen failure. Retained final timestep
  shear sensitivity 141.3 Pa is comparable to the mechanics-only 175.9 Pa.
  Fine/FC temporal uncertainty and continuum spatial order are unqualified.

Next proposed task: one finer mechanical control with the **FF Q1 phase and
exterior lattice exactly retained**, plus timestep halving at the selected
fixed mesh/profile before ranking weak loads. Phase refinement is a separate
selection. No next run is authorized automatically. Limiter changes, phase
stepping, transport, automatic AMR, mature restart/MPI, moving geometry,
propagation and the full event remain deferred; no full PF5 claim. Stop here.

---

# Bounded mature DG compatibility complete; spatial accuracy remains open

16 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved work.
[Measured report](PF6_MATURE_DG_REPORT.md), [pre-run scope](PF6_MATURE_DG_FORMULATION.md),
[source audit](SOURCE_AUDIT.md),
[evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_dg/README.md).
Hanging-node phase evolution is explicitly deferred.

- Extend default-off homogeneous mesh-data DG to fresh serial stationary mature
  faults: fully prescribed frozen phase, fixed graded Cartesian mesh, fixed
  straight oblique geometry, configured homogeneous materials and no domains/CPDI.
  Mature cohesive resistance/storage stay zero; nonzero Z is auxiliary slip
  memory, independent of cohesive elastic storage. General oblique normalization,
  paired completion and native rate/constitutive conventions remain.
- Non-proportional prescribed histories and actual weak coupling match native FE
  to 3.80e-15/5.67e-16 scaled. Independent Q2 quadrature/all-nine-mode algebra and
  fixed Gauss16/32 oblique normalization checks pass. The independent publication
  guard and operand-derived allowance are retained without tolerance changes.
- Same-core DG/FE/particle ten-step strengthening controls pass at physical Vp,
  fixed phase/geometry and the same selected 3D deviator. DG/FE final tensor
  relative L2 differences are 9.07e-14 domain / 2.21e-14 localized. Particle/FE
  localized difference is 83.34%, despite only .1933% Vp difference.
- Gauss6/8 localized tensor sensitivity is .00508%; timestep halving gives .1119%.
  Perturbation-load timestep differences reach 2.48%, failing the declared 1%
  screen. No temporal-order qualification follows from two timestep levels.
- One finer fixed-surface control (4136->11528 cells, finest h halved) passes,
  reduces prescribed-phase sampled error 2.51%->.65% and physical-profile I
  deviation 6.63%->1.80%. Spatial stress/load sensitivity remains large: 22.59 kPa
  localized RMS on common normal cuts and 6.20 kPa raw-normal projection change
  at the short endpoint. This is a spatial diagnostic, not a converged continuum
  reference or separate mechanical/phase-space error estimate.
- Authoritative reconstructed stress, all DGQ2 coefficients, unfiltered point
  tractions, raw/filtered weak-load projections and both interface traces are
  exported. All 66176 (fine 184448) diagnostic IDs/coordinates/H stay fixed within
  each trajectory. Points remain independent validation/diagnostic dependencies;
  accepted DG/FE mechanics read no particle stress/material projection.
- Eighteen native attempts cost 1278.61/3600 s, including the preserved initial
  diagnostic-allowance and frozen-phase callback failures. Their repairs do not
  change physical laws. The original legacy allowance failures, prior native
  quadrature/reference errors and `strict_reference_match_passed=false` remain.

Unlimited DGQ2 remains the FE-compatible experimental mature-history baseline.
All declared defaults and prior PF0--PF6/WENO evidence are preserved. Next select
fixed-phase oblique mechanics versus profile/I resolution separately, then one
further timestep level for perturbation loads at the chosen resolution. Keep the
surface fixed and retain the independent publication check. Mature rollback/
restart/MPI, diagnostic-point replacement, phase evolution, automatic AMR,
transport, moving geometry, fault birth, propagation and full first-event runs
remain subsequent work; full PF5 completion is not claimed.

---

# Homogeneous DG setup without particle domains — complete

16 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming work.
[Measured report](PF6_DOMAIN_FREE_REPORT.md), [declared checks](PF6_DOMAIN_FREE_FORMULATION.md),
[portable evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_domain_free/README.md).

- The default-off `Use homogeneous mesh data` configuration uses the prescribed
  physical Q1 phase seed, configured homogeneous materials, bulk mesh and fixed
  fault geometry. It constructs no domains/CPDI and performs no particle material
  projection. Fixed particles serve independent matching-cell publication checks
  and diagnostics; particle projection rejects their unweighted associations.
- Restored 64→256→64 remapping passes on 1/2/4 ranks with the same 144 fixed
  points, no domain object/read or added particles. Max remap stress error is
  1.38e-15 scaled L2. All nine tensor modes and representable-phase/lower-obstacle
  coarsening remain verified; nonrepresentable phase still rejects before mutation.
- Initialization matches the same-build domain-backed physical state exactly.
  Subsequent max stress/phase differences are 9.73e-15 scaled L2 / 5.63e-16 Linf.
  The coarse seed's 8.02% representation error is retained separately. Independent
  I/DI/D²I errors are below 1e-15 scaled; the actual normalization operators and
  publication validator/error budget are unchanged.
- Full cases pass 100 native gates each. Rejection/retry and single publication
  pass. Same-count restart and continued versus uninterrupted advancement are
  exact for all compared fields/histories/loads. Serial-to-four preserves histories
  exactly; weak-load differences are <=1.71e-13 Pa m. Changed-count RNG replay
  remains unavailable with population changes disabled. All 21 legacy FE/CPDI
  CSVs reproduce byte-for-byte.
- The fresh domain-backed deep test still fails its original coverage guard.
  That backend remains separate; its domains and coverage checks are unchanged.
  Prior Gauss6 failure, strict-reference false flag, errors and archives remain.

Unlimited DGQ2, equations, normalization/derivatives, limiter settings and production
parameter defaults remain unchanged. New-mode checkpoint conversion is rejected.
Next selectable work: constrained independent-master phase/obstacle KKT and native
lifecycle checks on fixed-surface hanging-node meshes. Its evolution guard remains;
no automatic AMR, lossy obstacle transfer, adaptive quadrature, transport, moving
geometry, propagation or full PF5 completion is claimed.

---

# Gauss12 and bounded local-transfer qualification complete

16 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming work.
[Measured report](PF6_LOCAL_REPORT.md), [declared rules/amendments](PF6_LOCAL_FORMULATION.md),
[portable evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_local/README.md).

- Gauss12 passes fixed-field and native two-step trajectory comparisons against
  Gauss24/32. Worst-state scaled bulk energy/load errors are 6.66e-16/2.12e-15;
  normalization differs by 6.19e-16 scaled. The original Gauss6 screen remains
  failed, with every original error and `strict_reference_match_passed=false` retained.
- The manual 64→67-cell fixed-surface remap has four phase hanging nodes. Complete
  Q2 tensors, physical phase/next obstacle, all native time levels, stress, energy,
  weak loads and particle geometry pass independent checks at 1/2/4 ranks.
  Each full case passes 107 native gates; max remap stress scaled L2 <=6.94e-16.
- Same-count and serial-to-four local restart reproduce all compared physical
  histories/fields exactly. Restore-only runs take no physical step or publication;
  continued uniform restart still advances once and matches uninterrupted results.
  Same-count RNG replay is exact. Changed-count RNG replay is unavailable under
  the retained native policy; population changes are disabled.
- All 144 IDs/coordinates stay fixed. Repairs address stale post-repartition
  domains, dropped Voro upper-face candidates, canceled-tag preparation and
  completed-restart termination. Original volume/coverage/publication checks remain.
  An independent Voronoi oracle verifies every exported supported domain.
- A genuine one-layer particle-neighborhood failure remains on the 256-total-cell
  diagnostic; its guard still fails with all intended candidates inserted. The
  short physical trajectories retain the same 64-cell steps and explicitly omit
  that identity-only diagnostic. Failed/capped/intermediate evidence is preserved.

Unlimited DGQ2, laws, rates, limiter settings and production parameter defaults
remain unchanged. Gauss12 is recommended only for this experimental fixture.
No coupled phase solve on hanging nodes is admitted: the independent-master
variation/obstacle KKT design remains required. General/local automatic AMR,
postmutation AMR recovery, lossy obstacle transfer and full PF5 remain unqualified.

Next separately selectable work: replace the remaining particle-domain/material
projection dependency or qualify neighborhood completion/preflight; then verify
C^T*g and C^T*H*C for constrained phase variations and its native acceptance lifecycle.
Adaptive quadrature, transport, moving geometry, birth and propagation remain deferred.

---

# Bounded fixed-reference transfer qualification complete

16 October 2026; base `a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved
incoming work. [Measured report](PF6_TRANSFER_REPORT.md),
[pre-run rules](PF6_TRANSFER_FORMULATION.md), [particle-read audit](PF6_PARTICLE_READ_AUDIT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_transfer/README.md).

- Native FE gradients and DG traces now use the same particle CellId; the
  averaged-trace negative control fails the retained consistency guard. The
  prior operand-derived arithmetic allowance is unchanged.
- All nine modes of all three DGQ2 components transfer through native distributed
  SolutionTransfer. Independent mass/projection, exact global-polynomial and
  nonzero-Z accepted-state roundtrips pass; stress scaled L2 <=1.40e-15.
  Lossy tensor coarsening is independently correct but loses .701247 scaled L2
  in the deliberately unresolved control; loss is reported, not physical evolution.
- Physical phase/next lower obstacle transfer as the same Q1 function. A +.01
  nonrepresentable fine peak rejects coarsening before mesh mutation. Surface
  geometry/Z/I stay fixed; physical clocks and publication counts do not advance.
- Full 1/2/4-rank runs pass 70 native gates each, including actual outer and
  terminal rejection, immutable accepted state/RNG, retry and single publication.
  Same-rank restart is identical; serial-to-four restart stress scaled L2
  <=1.16e-14. All 144 particle IDs/coordinates stay identical; RK2 advection dt=0,
  constitutive dt unchanged. Remaining setup/material/validation/persistence
  particle reads are explicitly inventoried, with replacement proposals.
- **`native_rule_vs_high_rule_screen_passed=false`:** exact-field transfer passes
  common high-order evaluation, but unchanged native Gauss6 physical quadrature
  misses the declared 1e-9 functional screen on the coarse evolving fixture
  (bulk energy error 1.52e-8 J/m; weak load 5.90e-7 Pa m). This is not erased by
  a new tolerance or quadrature setting.

The publication repair remains accepted only within its documented scope.
**`strict_reference_match_passed=false` and every original error remain preserved.**
Unlimited DGQ2, laws, limiter settings and production defaults remain unchanged.
This is uniform Cartesian, fixed-surface transfer evidence, not general PF5 or
automatic AMR qualification. No transport, moving geometry or propagation runs.

Next separately selectable task: fixed-geometry local AMR with a declared
physical stress/load/energy error budget, admissible lossy obstacle rule,
quadrature below that budget, and hanging-node ownership/restart checks. Surface
remeshing requires separate Z/slip/material transfer if later selected. Stop here.

---

# PF6a publication repaired; bounded native run complete; strict branch match open

Updated 16 October 2026 at `a8707307008401ac9cff702a369e2e2a611fa0f7` plus
preserved incoming work. [Publication report/formulas](PF6A_PUBLICATION_REPORT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a_publication/README.md).

- The first 512-normal-cell failure is reproduced with the original guard intact:
  error 8.29015e-13 Pa, allowance 6.63367e-13, ratio 1.24971. An 80-digit replay
  identifies cancellation in native point-gradient evaluation, with smaller DG
  projection/evaluation error. Same velocity, dt, phase, I/Z and recurrence.
- The repair derives an additional source error budget from actual velocity
  operands, widths, Q2 basis operations and computed inverse/analytic mass defect.
  It keeps the comparison, equations, rates, histories and defaults. Inconsistent
  incoming xx/yy/xy and staged histories still fail. Native rejection/retry,
  persistent/RNG rollback, actual timestep-manager cutback and single publication
  pass. Bounds are calculated only for publication candidates; all tensor
  coefficients remain identical with/without the scratch calculation.
- Two pointwise-diagnostic attempts hit the 900 s cap in step 72; all their
  evidence is retained. Default-off batched **postpublication diagnostics** are
  independently checked against the original point evaluator at 0/1/68/72.
  They change no production guard, physical input, quadrature, solver or history.
- The fresh unchanged 2x512, dt=1/128, q12 experiment completes t=.5625 in
  **879.915 s**, 72 physical publications, no healing, 372 native gates. All
  663552 particle comparisons pass; max error/bound .000465792. Independent
  source, physical stress, normalization, energy, weak loads, quadrature and
  work-accounting checks pass. Focused units: 628 assertions/eight cases; PF1:
  94 checks; frozen FE baseline exact and same-build FE/DG compatibility passes;
  all 23 prior PF4 operator/control CSVs reproduce byte-for-byte.
- **Strict selected-reference comparison is not fully passed.** Agreement holds
  through step 71, but the final state takes the opposite mirror phase branch.
  Unregistered max differences: whole/localized stress 1.50e-6/3.66e-6 scaled,
  phase relative L2 1.153e-4, Linf 5.909e-5, driving 2.384e-4 scaled; these miss
  the retained 1e-6 screen. Traction/slip/energy pass. Both branches have positive
  tested translation curvature and energy difference below the native numerical
  allowance. A numerical between-candidate tie comparison differs in native and
  independent reference. No reflection, alignment, sign lock, reference
  replacement, tolerance change or constitutive change is used to erase errors.

Unlimited DGQ2 remains the experimental baseline; both WENO paths remain
experimental. Remaining particle dependencies include startup H/CPDI/domains,
material projection, association/fixed support, derived samples/publication
validation and persistence/RNG, with no particle-stress feedback into mechanics.
The result is serial, fixed geometry/grid and no transport. No general PF5,
AMR/MPI, moving geometry, birth or propagation qualification follows.

Next: specify the unresolved branch/tie reference criterion and verify both
endpoint branches without aligning away primary errors before using a strict
trajectory oracle. Then select fixed-geometry AMR/MPI T/phase-obstacle/Z transfer
and restart checks. In particular, AMR can put particles on child faces: native
averaged FE gradients and an explicit DG one-sided CellId trace require matching
ownership/evaluation; their difference must not be absorbed into roundoff.
Do not automatically continue to those tasks.

---

# Particle-free status — matched spatial reference measured; native confirmation failed

Updated 16 October 2026. The selected fixed-geometry study is complete as a
bounded experiment. **The 1D model selects 512 normal cells; native growth at
that resolution is not confirmed.** Unlimited DGQ2, laws, quadrature, native
acceptance rules, limiters and production defaults remain unchanged.

- Main trajectories start from the documented continuum seed and unloaded
  histories. Existing native table values match that definition exactly at all
  selected native nodes. The finer phase reference samples the same retained
  continuum definition. Interpolation errors are measured separately; no loaded
  coarse PF6a state supplies the main reference. A literal-table-refinement
  control and its spurious early growth are preserved as an initialization
  diagnostic, not silently used as the main reference.
- Independent coupled 1D mechanics/phase equations, derivatives and reduced
  tangent pass. Projected Q2 versus continuum at matching phase resolution
  separates mechanical effects; continuum phase 64/128/256/512/1024 separates
  phase-space/obstacle effects. All main onset times are .53125 s. Translation
  departures use the retained actual energy and obstacle; no symmetry lock.
- Frozen 256 errors .6893%/1.2651% are verified, but growing errors reach
  1.176%/2.439% and fail fixed whole/localized 1%/2% targets. One bounded 512
  candidate gives .2944%/.6107% and meets the practical 1D targets. Time and
  quadrature controls remain smaller; soft branch-position uncertainty still
  limits phase convergence/order claims. Full histories and driving are retained.
- The **single fresh native 2×512 confirmation exits 1 at t=.0078125 s**, after
  15.152 s. Startup matches phase/history/I/energy. The existing Maxwell/DG
  publication consistency assertion rejects the first endpoint before terminal
  publication; zero positive accepted exports exist. Its exception omits the
  error/bound/position values, so the cause remains unresolved. No tolerance was
  relaxed, no second native trajectory launched, and no native growth accuracy
  or positive-step rollback-byte check is claimed.

[Measured report](PF6A_REFERENCE_REPORT.md),
[formulation and source-origin audit](PF6A_REFERENCE_FORMULATION.md),
[source audit](SOURCE_AUDIT.md),
[evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a_reference/README.md).

**Next concrete task:** instrument and independently diagnose the same first-step
native publication mismatch, preserving its guard. Capture actual components,
error/bound, native gradient, DG source, CellId/position, T/Z and localization
before selecting a correction or another native onset/growth confirmation.
512 remains a mathematical candidate, not an adopted native resolution. General
PF5 AMR/MPI/transfer, transport, moving geometry, birth and propagation remain
subsequent. No automatic continuation or production/default change.

---

The preceding spatial diagnostic and earlier evidence remain historical records.

# Particle-free status — bounded PF6a spatial diagnostic complete

Updated 16 October 2026. **Retain unlimited DGQ2, the current constitutive
laws and all limiter/production defaults.** The selected serial fixed-geometry
spatial diagnostic is complete; no production source changed in this task.

- Identical physical phase/T/Z evaluated on common points agree across nested
  meshes and compatible DG/accepted-FE history. This is distinct from the
  frozen physical phase/incoming-history experiment that actually re-solves
  native mechanics on 32/64/128 normal cells.
- The existing inexpensive 1D reference reproduces the native mechanical
  cancellation to <=7.22e-7 Pa. Its localization projection remainder explains
  whole-domain tensor errors 783.653 → 200.261 → 50.353 Pa and localized errors
  1419.286 → 366.311 → 92.357 Pa. Common phase-gradient bias decreases
  1.42288 → .095064 → .006045. All component, moment and interface exports remain.
- Matched DG/FE resolved stress differs by <=7.05e-8 Pa. Actual FE-versus-CPDI
  fracture is assessed separately and agrees to roundoff in this aligned phi(y)
  fixture; that is not general CPDI equivalence or particle-history validation.
- The justified short common-entry native event reaches phase peaks
  .620381/.646898/.651129. The imported coarse state is not equilibrated in
  enriched fine-grid spaces; the event includes that relaxation. The 64/128
  stress difference remains .162708 relative L2: 128 is not a qualified evolving
  reference. Halving its timestep changes stress by 4.27e-6 relative L2, far
  below spatial differences. Native and independent quadrature controls also
  remain well below those differences.
- All frozen nonpublication/history-immutability and short-event single
  publication/irreversibility checks pass. Accepted phase KKT <=3.41e-11;
  independent full-tensor recurrence at all nine Q2 nodes agrees to 1.57e-10 Pa.
  Prior source/evidence/manual work is preserved; see the preservation manifest.

[Measured spatial report](PF6A_SPATIAL_REPORT.md),
[pre-run formulation](PF6A_SPATIAL_FORMULATION.md), [source audit](SOURCE_AUDIT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a_spatial/README.md).

**Next proposed task:** a separately selected matched fixed-geometry evolving
spatial reference using the documented physical initialization/obstacle and the
1D mechanical model. The diagnosed error concerns mechanical resolution of
localization, not missing compatible DG history modes. Further order/mesh
studies require selection. This diagnostic stops here; general PF5 AMR/MPI and
surface/phase transfers, moving geometry, birth, RSF evolution, transport and
propagation remain subsequent work. Particle dependencies remain those listed
in the report; particle stress remains absent from mechanical/load feedback.

---

The previous PF6a closeout and earlier stages remain historical evidence.

# Particle-free status — bounded PF6a complete; spatial accuracy unresolved

Updated 16 October 2026. The user selected serial fixed-mesh, fixed-reference-
surface cohesive phase/mechanics, initially without bulk/surface advection.
The retained consistent-endpoint/reduced-Newton design now uses explicit DGQ2
endpoint history and the PF3 FE fracture functional. **Unlimited DGQ2 remains
the experimental baseline; no limiter setting or production default changes.**
No one-pass phase-first update was adopted. PF7 has not begun.

- Independent energy/gradient/Hessian actions, consistent mechanical-response
  tangent and symmetry pass the frozen screens (best reduced FD 1.69e-9,
  symmetry 3.59e-14). Frozen same-build FE/DG compatibility passes; the retained
  disabled FE baseline matches exactly. Native source recurrence, reconstructed
  full-3D stress and fault weak loads agree with independent integration.
- A short native coupled event develops nonzero T and Z and physical phase
  growth: max phase .60000194 to .62038134 on 2×32 cells, Z=39123.10 Pa m.
  No healing; native phase KKT <=5.05e-11, free fault residual <=8.30e-12 Pa.
  Admissible sticking residuals are reported separately.
- Stored-field dt differences decrease by about four; physical-stress relative
  L2 difference dt/2 versus dt/4 is 4.33e-7. Quadrature 6/12 evolution difference
  is 2.67e-10; the higher-rule KKT difference is separately retained. **The
  bounded 32/64-normal-cell comparison changes physical stress by .42237 and
  phase by .05979 relative L2.** Initial seed representation errors are .00703
  and .00175. The finer mesh is not an exact reference or spatial qualification.
- Independent evolving-profile work accounting closes its signed identity to
  5.08e-14 J/m; the external-balance remainder decreases .05711 → .01880 →
  .005226 J/m with timestep refinement. Individual absolute residual/phase work
  terms remain measurable and are not absorbed into invented dissipation.
- Actual prepublication exception/retry and native timestep-manager cutback
  pass. DG publication is deferred until the outer repeat decision; retry
  preserves the same anchor. All 22 accepted cutback/control states match byte
  for byte. Actual automatic checkpoint/restart matches physical fields, all
  persistent values/RNG and native vectors exactly; complete native-loaded
  manager archives match bitwise (known raw +1 shear-sense encoding difference).
  PF6a payload v2 separates accepted step and next-step native clock; old PF5
  v1 restart and defaults remain intact; mode conversion is rejected.
- All 23 PF4 operator-control CSVs reproduce byte for byte, 628 native assertions
  in eight cases and 94 PF1 checks pass. Incoming WENO implementation/evidence,
  original phase reports and manual work are preserved. Failed preliminary runs
  and the first spatial timeout remain evidence.

[PF6a measured report](PF6A_REPORT.md), [frozen formulation](PF6A_FORMULATION.md),
[source audit](SOURCE_AUDIT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a/README.md).

This completes only PF6a's selected bounded experiment. General PF5 AMR/MPI and
surface/phase transfers remain incomplete. Particles still supply startup
H/CPDI/domains, surface material projection, association/support, fixed-particle
checks, derived samples and native persistence; particle stress does not feed
independent history, mechanics or fault loads. Moving geometry, birth, coupled
transport, RSF evolution and propagation remain subsequent work.

**Next proposed task:** isolate and resolve fixed-geometry normal spatial
representation on frozen common physical phase/T/Z snapshots, separating seed
interpolation, velocity/tensor representation and reconstruction, before one
bounded finer short event. Keep constitutive/limiter conventions fixed. General
PF5 transfer/MPI remains a separate prerequisite; no automatic continuation.

---

The preceding-stage status entries below remain historical records.

# Particle-free status — pre-PF7 WENO candidate comparison complete

Updated 16 October 2026. The selected audit and controlled candidate-construction
comparison are implemented and measured. **Retain unlimited upwind DGQ2 as the
experimental baseline; do not adopt constrained candidates on this evidence.**
Both WENO detectors remain experimental. PF7 has not begun.

- The pinned incoming build reproduces 63 retained non-cost scalar/smooth CSVs
  byte for byte. The freshly built instrumented old path agrees within 4.23e-15
  relative L2. Physical means are conserved at limiter calls (maximum 1.31e-12
  Pa drift); interface changes independently account for subsequent native flux
  and SSPRK3 mean changes. Traces identify nearly constant neighbor candidates
  dominating old weights and flattening cells despite mean conservation.
- A private default-off constrained least-squares option retains all nine Q2
  modes, common tensor useful sets/weights and the full embedded 3D metric.
  Independent Legendre/full-tensor KKT checks, mean preservation, complete global
  Q2/x²y² reproduction, basis covariance, amplitude scaling and order independence
  pass. Detectors, thresholds, epsilon, smoothness, quadrature, upwind operator
  and SSPRK3 stages are unchanged. This is a controlled adaptation motivated by
  Zhu–Zhong–Shu–Qiu (2016), not the complete published method.
- All 30 bounded 32×64 sharp/finite-width jobs at dt, dt/2, dt/4 pass execution
  and nonpublication checks. At dt/4, constrained KX/Fu–Shu reduce sharp relative
  L2 from .436260/.432500 to .224786/.222842 and narrow fronts, but unlimited
  remains .177547. Finite-width errors are .000771065/.000433813 versus
  unlimited .000427621. Every WENO path fails the frozen temporal screen; both
  unlimited controls pass. Held-field limiting is not idempotent. Constrained
  candidates also create transverse variation in x-only scalar fixtures.
- All 23 original operator CSVs reproduce byte for byte; 69 independent limiter
  checks, 628 native assertions/eight cases and 94 PF1 checks pass. A strict
  cached/uncached final-field equivalence target failed (1.85e-10 relative L2
  versus 1e-12 target); local KKT checks pass. This failure and preliminary runs
  are retained, with no relaxed tolerance.

Production defaults, physical laws, history backend/publication and PF0–PF5
evidence remain unchanged. Particle dependencies remain startup H/CPDI/domains,
surface material projection, association/support, fixed-particle checks, derived
samples and particle/RNG persistence. No accepted evolution, transfer,
AMR/MPI/periodic/moving geometry, birth or coupled propagation was added.

Any separately selected stabilization work should address limiter-call/time
dependence and x-only preservation before adoption or tensor qualification.
This task stops after the comparison; no automatic PF7 or further tuning.

[Measured report and recommendation](WENO_CLS_REPORT.md),
[frozen equations](WENO_CLS_FORMULATION.md), [source audit](SOURCE_AUDIT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/weno_cls/README.md).

---

The PF5 closeout and earlier evidence below remain historical records.

# Particle-free status — selected PF5 persistence subset complete

Updated 16 October 2026. User accepted the PF4 method recommendation and selected
fixed-mesh/fixed-geometry native history advancement, rollback and restart.
Implemented default-off `Enable prescribed DG accepted history`; retain
**unlimited upwind DGQ2** as the experimental baseline. Production defaults and
limiter settings are unchanged; both WENO paths remain experimental.

- Native exact Maxwell/source half steps surround pure unlimited SSPRK3
  transport; surface Z advances once with the existing cohesive law. Independent
  exact semidiscrete reference establishes **second-order combined accuracy**:
  T relative L2 falls from 3.75525e-6 to 5.86693e-8 over four step counts
  on one 4×8 grid. Finest physical stress/weak-load errors are 5.87360e-8/8.92354e-8.
- Non-proportional tensor and nonzero physical elastic slip are included.
  The fixed-profile cohesive commutator is retained; final Z=299.550449662703 Pa m,
  maximum analytic Z error 9.10e-13. No mature Qgeo=0 substitution or spin change.
- 280 fresh-run and 18 restart gates pass: repeated trials, rejection/retry,
  nonfinite/changed-context rejection, single publication and checkpoint creation
  without accepted mutation. A real native filesystem restart matches all checked
  persistent histories, physical stress and both fault weak loads exactly.
  Persistent/RNG and canonically loaded complete manager archives match bitwise;
  raw manager archives retain the existing implicit-to-explicit +1 shear-sense
  representation difference. Failed preliminary evidence is preserved.
- Existing checkpoint/solution/manager/particle infrastructure is reused.
  Four measured guards reject initial AMR, coefficient/quadrature mismatch and
  mode conversion. Native 628 assertions/8 cases and 94 PF1 checks pass;
  default-off FE matches the retained incoming-core result exactly, same-build
  FE/DG operator/publication checks pass, and all 23 PF4 operator-control CSVs
  reproduce byte for byte. No further WENO sweeps or threshold tuning.

This completes the **selected bounded subset**, not general PF5 transfers or
coupled stepping. The observer prescribes accepted clocks; surface V/Z are
uniform, phase/geometry/particles fixed and transport boundaries closed.
Remaining particle dependencies include startup H/CPDI/domains, surface material
projection, support/association and fixed-particle checks, derived samples and
particle/RNG persistence. RSF Theta, phase-bound transport, nonuniform surface
transport, AMR/MPI/periodic/open-boundary splitting, moving geometry and birth
remain unsupported or unqualified.

**Proposed next task:** fixed-geometry PF5 bulk AMR/MPI transfer and restart
qualification, reusing native ownership/SolutionTransfer, Q2 prolongation and
conservative restriction. First keep the surface grid fixed, then qualify any
material-coordinate Z/Theta transfer and native phase/physical-lower-bound
transfer actually required by a separately selected small fixed-geometry PF6
phase/mechanics fixture. Measure remapping work/energy defects separately from
physical evolution. No PF6, moving geometry, fault birth, coupled propagation or
BP3 is automatically selected. Stop here.

[PF5 progress, measurements and concrete next task](PF5_REPORT.md),
[source audit](SOURCE_AUDIT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf5/README.md).

---

The prior PF4 method comparison and earlier evidence below are historical and
retained unchanged.

# Particle-free status — bounded transport-method comparison complete

Updated 16 October 2026 UTC. The selected comparison is implemented and measured.
**Retain unlimited DGQ2 as the resolved smooth/prescribed-history baseline.**
Both WENO paths remain experimental; no threshold tuning or further WENO
sweeps were performed. Scaling remains a scalar benchmark reference.

|32×64, common base step|Unlimited DGQ2|Scaling|Native scalar EV|Native scalar SUPG|
|---|---:|---:|---:|---:|
|Sharp relative tensor L2|.177548|.226747|.225369|.211070|
|Finite-width relative tensor L2|.000427638|.000427638|.001275466|.000545821|
|Scalar DoFs|18,432|18,432|8,385|8,385|

Independent exact split-cell projection gives **.149862170276557** for the sharp
DGQ2 representation floor, confirming the predicted .1498621703. The report
separates representation, numerical evolution and native interpolation errors;
CG sharp initialization alone is .223607. Fixed-mesh time differences, extrema,
all three front widths, signed tensor-integral errors, norm changes and measured
costs are retained. Exact projection itself can overshoot substantially.

A native FE-strain-generated independent tensor is transported without
publication, with nonzero shear and normal fault weak loads. At half the base
step, weak-load relative errors are DG **2.272e-5**, EV **1.140e-4**, SUPG
**3.648e-5**. DG and SUPG tensor errors are close (.004595/.004629), but DG has
lower measured fixed-coefficient cost and better resolved-pulse accuracy. SUPG
weak-load temporal uncertainty is comparable to its gap from DG; no converged
native-load ranking is claimed.
Native EV/SUPG are actual continuous-Q2 scalar solves with native BE/BDF2 and
unchanged stabilization defaults, not reimplementations labeled native.

Production sources/defaults/laws/backend and all PF0–PF4 evidence are preserved.
All fourteen comparison records match the same core and 879 current production
source hashes; ten new successful controls are bounded to 180 s, one rank/thread,
at most two simultaneous. DG/scaling fields reuse unchanged-source evidence.
Projection/error/load checks, 628 native assertions and 94 PF1 checks pass.

Remaining particle dependencies: H/CPDI/domain startup, fixed-particle checks,
support/association, surface material projection and derived samples. Transport
reads no particle properties. No production adoption, coupled evolution, restart,
AMR/MPI/periodic/moving geometry, BP3, propagation or PF5 is selected. Stop here;
any next experiment needs a separate user selection.

[Measured comparison and recommendation](PF4_METHOD_COMPARISON_REPORT.md),
[source audit](SOURCE_AUDIT.md),
[portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_methods/README.md).

---

The prior PF4 reports and evidence below are retained unchanged.

# Particle-free status — selected PF4 follow-up complete

Updated16 October2026 UTC. All four user-selected tasks are implemented and
measured: fixed-mesh timestep sensitivity, direct norm changes at limiter calls,
common scaling on the signed pulse, and a physically scaled finite-width control.
The user confirmed the benchmark envelope, largest common factor and
`delta=ell/ln9`, with errors against the actual overlapping analytic profile.
**Keep unlimited as the smooth baseline; retain both WENO paths and scaling as
experiments. No more elaborate stabilization is justified by this resolved
finite-width control. Stop after bounded PF4; no production adoption or PF5.**

|32x64, original dt|Unlimited|KXRCF/WENO|Fu–Shu/WENO|Scaling|
|---|---:|---:|---:|---:|
|Sharp relative tensor L2|.17755|.37629|.37658|.22675|
|Sharp Q loss|3.574%|28.948%|28.903%|12.549%|
|First sharp front width, m|.02353|.09967|.09686|.02655|
|Finite-width relative tensor L2|.00042764|.00055178|.00044103|.00042764|

Fixed32x64 dt/dt2/dt4: sharp WENO dt2-to-dt4 field changes12.98–13.16%; scaling
.02812% (largest common-time change.03615%). Unlimited behaves at third order.
KXRCF finite-width L2 grows to.051565 at dt4. The WENO family is not temporally
resolved; no threshold retuning or unbounded refinement was attempted.

Direct-call Q measurements distinguish raw sums, RK-weighted contributions
(1/6,2/3,1) and remaining transport/SSP changes. Sharp RK-weighted scaling loss
9.595%, versus WENO6.30%, despite smaller total scaling loss. Finite-width KXRCF
increases Q in257/309 calls: WENO is not a norm-contraction guarantee. Held-field
reconstruction changes sharp WENO fields by5.06–5.51% after12 passes without
transport, versus exactly0 for scaling. These are not mechanical-energy balances.

The fixed analytic ell=.15625m pulse spans5 tangential cells on32x64. Unlimited
error.04276%, positive peak attenuation.003715%, first-front broadening.02889%
at dt4; scaling is idle and byte-identical to unlimited. Actual analytic peak
918.505516Pa and reference widths include transition overlap. All actual scaling
means stay inside the envelope (zero excess); deliberate mean1.1 is rejected.
GL9 envelope enforcement is distinguished from small between-point polynomial
overshoot; no production component/stress bounds or chi filtering were added.

149 transport/ledger,45 local/algebra and77 retained WENO gates pass;628 native
assertions/8 cases,94 PF1 checks and standalone production DG compilation pass.
Original operator/publication controls, prior base-step fields and the final
instrumentation-nonintrusiveness comparisons are byte-identical. Accepted
DG/phase/Z/I and incoming states are unchanged. All four final runs are within
180s (12.087/39.292/118.492/164.762s), at most two simultaneous, one rank/thread;
core SHA prefix`3ee53a8645b2`. Raw preliminaries/binaries remain preserved.

Remaining particle dependencies: fixed-particle startup checks, H/CPDI/domain
initialization, support/association, surface material projection and derived
sampling. Limiting/advancement read no particle properties. Native accepted
publication, surface-history/phase-bound transport, restart, MPI/AMR/periodic/
moving/coupled paths remain guarded or unimplemented. Defaults/physical laws,
manual instructions/derivation and all PF0–PF4 evidence are preserved. No commit,
push, BP3, PF5 or propagation work.

[Full report](PF4_STABILIZATION_REPORT.md), [source audit](SOURCE_AUDIT.md),
[portable evidence and replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_stabilization/README.md).

---

The prior limiter comparison and PF4 closeouts below are retained unchanged.

# Particle-free status — bounded PF4 limiter comparison complete

Updated16 October2026 UTC. Latest user selection is fully implemented and run:
**serial fixed-grid prescribed-transport tensor limiter comparison only**.
Evidence favors **neither limiter for adoption**; production/default remains
unlimited. Stop after this report; no PF5, coupled BP3 or propagation campaign.

- Same worktree/branch/HEAD as the preserved status below. Incoming manual
  instructions/derivation and PF0–PF4 source/evidence preserved; no commit/push.
- New private `dg_limiter.cc`: complete DGQ2 immediate-neighbor compact WENO,
  target-cell means, shared full-plane-strain tensor weights, stable dimensionless
  smoothness/zero treatment and immutable stage reads. Optional SSPRK3 hook only;
  no physical-law/default/accepted ownership change. No initial limiting.
- Safeguarded Fu–Shu experimental RMS/mean-plus-polynomial decision usesC=.1,
  fixed before validation from saved .05/.1/.2 screening. KXRCF-style threshold1
  uses pointwise inflow and the same WENO. Source quadrature audited; Gauss3 is
  full rank for Q2. This is a port of `limiters.cc`, not `elasticity.cc`.

| Measured gate /32×64 tradeoff | Unlimited | KXRCF+WENO | Fu–Shu RMS+safeguard+WENO |
|---|---:|---:|---:|
| Signed-jump over/undershoot |16.50/15.82%|.016/.024%|.049/.022%|
| Relative full-tensor L2 error |.17755|.37629|.37658|
| Quadratic-norm loss |3.574%|28.948%|28.903%|
| First10–90% front width, m |.02353|.09967|.09686|
| Finest smooth L2 error |.00026275|.00026515|.00029650|
| Finest smooth convergence rate |2.997|3.342|4.658|
| Jump advance time, s |.405|2.513|5.089|

77 local/snapshot and66 transport checks pass. Zero/constants, forced piecewise
constants, global affine/full Q2 includingx²y², means, finite weights, traversal,
in-plane tensor covariance, amplitude rescaling, zero crossings, non-proportional
components, direction changes and mixed boundary inflow pass. The locally
centered mixed-xy mean-only blind spot is independently demonstrated; Ipoly=.288
flags it, while actual forced reconstruction changes it only.0375%. Constant
background sensitivity is measured and remains a limitation. Maximum stage-local
mean error1.07e-15 scaled; incoming and accepted bulk/phase/Z/I remain unchanged.

Fresh operator/source/publication fields/metrics are byte-identical to PF4;
628 native assertions/8 cases and94 PF1 checks pass. New limiter/transport
translation units compile independently without unity/PCH. Validation4/8/16/32
runs take4.024/4.522/10.983/42.143s, all within180s. Same core SHA prefix
`a54c7f7b2e47`; plugin/input/source hashes and pinned raw binaries remain under
`build-particle-free/runs/pf4-limiter-*`. Supplementary local covariance checks
use a separately pinned observer; governing kernel is identical.

No particle reads enter limiting/advancement. Startup fixed-particle checks,
H/CPDI/domain initialization, support/association, surface material projection
and derived sampling still need particles. Native surface-history/phase-bound
transport, accepted publication/restart, MPI/AMR/periodic/moving-geometry/coupled
paths remain guarded or unimplemented. No chi limiting, stress clipping, adaptive
quadrature, new spin or constitutive law.

Details, source corrections, exact formulas and recommendation:
[PF4_LIMITER_REPORT.md](PF4_LIMITER_REPORT.md), [SOURCE_AUDIT.md](SOURCE_AUDIT.md).
[Portable evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/README.md).
A further selected task could examine reconstruction-induced mode damping and
spatial accuracy; this comparison does not authorize it or coupled deployment.

---

The earlier PF4 closeout below is retained as historical evidence.

# Particle-free work status

Updated 15 October 2026. **PF4 prescribed transport scope complete; no limiter
enabled. Stop before further integration or PF5.** Latest authorization:
“Proceed with PF4 as proposed.” A noncommitting DGQ2 material transport operator
uses native Q2 velocity, upwind faces, explicit inflow/source and the material
rather than density divergence correction. All284 final transport gates pass.
Defaults, native Maxwell/cohesive/friction/H laws and physical outer-step guards
remain unchanged. No coupled trajectory or nonsmooth-history qualification.

## Baseline and preservation

- Worktree `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
  `pf-rsf-particle-free`, HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus
  uncommitted PF2–PF4 source/tests/docs. Nothing staged, committed or pushed.
- Root `AGENTS.md` applies; no override. Incoming instructions, derivation,
  identical installed plans, PF0–PF3 evidence and other checkouts/builds preserved.
  Derivation: [../particle_free_history.tex](../particle_free_history.tex), SHA256
  `bb05702d5c712bc51a87f0b2a75d881d56601d86106bd600d305815b00927539`.
- Native/plugin builds: `build-particle-free/native` and `/plugin`, GCC12.4,
  deal.II9.6.2, Trilinos14.2, OpenMPI5.0.10, Release, serial/one thread. Each
  run bounded180s; at most two native simulations overlap.
- Final core SHA256 `964340e5af68fcff87ef065a151f5247fd343e4b2c1e88acb3376aae3ad0e30d`.
  Pinned binaries, raw/preliminary states/logs and incoming manifest remain under
  `build-particle-free/`. No task process remains active after verification.

## Measured results

| Gate | Result | Record |
|---|---|---|
| PF0–PF3 retained | Audited laws;94 independent checks; native history and phase operator/bounds verified; historical failures retained | [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [PF3_REPORT.md](PF3_REPORT.md) |
| PF4 independent matrix | Matrix/divergence/inflow errors≤1.554e-15 scaled; polynomial derivative4.330e-15 | [PF4_REPORT.md](PF4_REPORT.md), [assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4/assessment.json) |
| Material/density distinction | Constant tensor error≤8.28e-16; removing divergence correction gives3.921% finite error | Same report/assessment |
| Smooth spatial refinement |4x8→32x64; finest rates2.972–3.017. Translation/divergence pulse errors .02628%/.02212% on32x64 | Same report/assessment |
| Fixed-space time refinement | SSPRK3 orders3.0013–3.0030;512/1024 uncertainty1.37% of128-step error. Finest pulse dt/2 effect .00322% of spatial error | Same report/assessment |
| Commutators | Cohesive/mature/residual converge; omitted normal/tangential source errors40.78%/5.45%. Rotating-profile objective/non-objective distinction verified separately | Same report/assessment |
| Source quadrature |8x16 Gauss4→8 difference .004770Pa m;8→12 difference6.397e-12Pa m | Same report/assessment |
| Signed nonsmooth data | Integrated errors decrease, but32x64 retains16.50%/15.82% sampled over/undershoot and3.574% quadratic-norm loss | Same report/assessment |
| Trial state | Repeated trials identical; invalid source/unsafe dt rejected; accepted DG/phase/Z/I unchanged | Same report/assessment |
| Compatibility | PF3 assessment byte-identical,67 gates; frozen FE baseline fields byte-identical; same-core FE/DG max field/load differences3.195e-14/7.319e-13 | [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4/README.md) |
| Builds/tests | Full native/plugin, both DG translation units without unity/PCH;628 native assertions/8 cases and94 PF1 checks pass | Same evidence |

Final transport runs: `pf4-final-4/8/16/32`, wall times4.122/5.273/11.687/40.393s,
peak child RSS214360/241216/308028/651508KiB. Native phase and frozen controls
use the same final core. Compact provenance and replayable fixed-entry archive
are under `benchmarks/reconstructed_fault/particle_free/evidence/pf4/`.

## Boundaries and remaining dependencies

- `dg_transport.cc` owns the new private operator. Existing DG definitions move
  to shared `dg_history_internal.h`; accepted ownership/layout and updates stay
  in the material. No new public selector, manager or physical parameter.
- The observer projects analytic scratch tensors in the actual native space.
  Transport time T=.2s is controlled; the outer simulator clock, particles,
  native surface histories and accepted field are not advanced. Published-state
  transport, full rollback/restart and coupled physical stepping remain guarded.
- Surface histories/localization are prescribed characteristic controls.
  Co-moving/rotating fixed-amplitude templates do not qualify native moving
  normalization or geometry. No production spin or mature residual state added.
- Direct/residual DG closure is convergent, not roundoff exact: continuum source
  projection and DG trace differentiation do not commute algebraically.
- No limiter is needed for the selected resolved smooth checks. The jump case
  establishes a significant oscillation issue for future nonsmooth use, whose
  acceptable mechanical effect and limiter benefit remain unqualified. No fixed
  stress-component bounds, chi filtering or historical WENO port.
- Advancement consumes no particle properties/domains. Construction still
  checks fixed particles; startup H/CPDI/domains, surface material projection,
  support/association and derived sampling require managers. No end-to-end
  particle-free claim.
- Native Z/theta and material phase-bound transport, geometry/birth/AMR, MPI,
  persistence, coupled time/energy/work and propagation remain subsequent stages.
  No PF5 or broader campaign has started.

Proposed next selection: a **bounded PF4 tensor-limiter comparison** using the
saved jump and smooth controls, after reviewing a mean-preserving common-tensor
reconstruction. Verify constants/zero, affine/smooth accuracy, means, tensor-basis
covariance and degenerate weights; measure actual overshoot/norm-loss benefit.
Keep coupled adoption and PF5 persistence/transfer separate. Full equations,
commands, costs and limitations are in [PF4_REPORT.md](PF4_REPORT.md).
