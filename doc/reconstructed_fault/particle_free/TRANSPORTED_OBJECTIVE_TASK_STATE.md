> **Current selection complete, 4 November 2026:** the ASPECT architecture refactor removes the rejected dedicated mature driver. Ordinary simulator/solver schemes own advancement and acceptance; tensor discretization, material laws and geometry have explicit boundaries. All 183170 unit assertions and seven moving controls pass; stationary/non-fault outputs and retained serial numerical results match their baselines. See [current guide](MATURE_MOVING_FAULT.md) and [report](ASPECT_ARCHITECTURE_REPORT.md). Defaults, checkpoint meanings and 51 protected artifacts are preserved. Scientific campaigns remain paused; this is not new physical or production qualification.

## Historical record: Stress-history consolidation complete — 4 November 2026

The tested direct-stress/publication baseline is preserved as `e4b74c556`,
separately from the subsequent behavior-preserving restructuring. The local
origin-tracking reference remains `69e8a10a93a08`; no fetch/push was performed.
See [report](STRESS_HISTORY_CONSOLIDATION_REPORT.md),
[baseline](STRESS_HISTORY_CONSOLIDATION_BASELINE.md) and
[plan](STRESS_HISTORY_CONSOLIDATION_PLAN.md).

- Serial limiter/statistics and standalone spatial relaxation now live in
  benchmark support; unused production wrappers and limiter hooks are removed.
- Physical stress access, material projection and affine residual reporting have
  identifiable modules. Direct/separated share mechanics; mature/cohesive state
  remains distinct. Transaction ownership and checkpoint meanings are unchanged.
- All 183170 assertions/152 unit cases pass. Four direct/separated constant/
  layered pairs agree within 1.58e−15 bulk and 1.07e−14 surface; default,
  forced rejection and 2→4-rank restart pass. The layered operator CSV is
  byte-identical. Migrated serial fixtures pass; measured roundoff is reported.
- All 39 protected files match their saved hashes. Defaults, incoming-gradient
  spin lag, quadrature and criteria are unchanged; checker DSOs were rebuilt.
  No production run, checkpoint conversion or new physical qualification.

Scientific work is paused. Finite-end approximation, coupled energy and the
independent cohesive failures remain open. A frozen endpoint-aware tensor-space
comparison is the next justified scientific experiment **only if selected**.
Earlier recommendations below are historical. Cleanup losses remain unchanged.

## Historical record: Frozen finite-end comparison complete — 4 November 2026

The selected common-frozen spatial/quadrature study is complete. **Uncut DGQ2
approximation at finite mature curve ends dominates; increasing quadrature alone
is not the remedy.** [Report](FROZEN_FINITE_END_REPORT.md) ·
[Protocol](FROZEN_FINITE_END_PROTOCOL.md) ·
[Evidence/figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/frozen_finite_end/README.md).

- One retained64² Q1 profile, 18 labels/I and prescribed source are fixed across
  16²–128² tensor grids, q8–24 and constant/layered eta. Cut integration resolves
  all source breaks; Gauss6→10 reference changes are below2.7e−14 Pa m.
- At64² constant eta, best tensor error is2.888e−4 Pa m; q8 quadrature error is
  3.211e−5. More than99.99% of squared best error is in fixed end windows.
  Fine-grid best errors scale near sqrt(h). Best point/work traction errors
  improve .936%/.317%→.579%/.159% from64² to128². Layered results agree.
- q8→24 reduces quadrature error but reduces total64² tensor error only.59%.
  Projection loss is nonnegative in its own rule; its independently integrated
  energy difference can be negative. Both measures are retained.
- Native source/binaries/checkpoints and defaults are unchanged. All numerical
  reference/identity checks pass. A measure-zero endpoint ownership error in
  the diagnostic was corrected to a one-sided limit; original evidence remains.
- Keep direct history as an opt-in reference and separated history as default.
  Next justified experiment: a frozen endpoint-aware tensor-space prototype
  with matching work integration, before any coupled implementation. It is not
  enabled or automatically selected. Broader Stage C, cohesive gates and
  production remain open; prior campaigns stay paused and cleanup losses remain.

## Historical record: Direct physical-stress comparison complete — 4 November 2026

The selected opt-in direct physical-history implementation and bounded native comparison are complete. **Retain it as an independent reference; keep separated history as default.** See [DIRECT_TRUE_STRESS_REPORT.md](DIRECT_TRUE_STRESS_REPORT.md) and its evidence index.

- Direct owns 27 DGQ2 coefficients/cell, versus 54 for separated T/R; auxiliary Z is mechanically inactive. Both constant and spatial eta use a converged projected reaction in direct mode. A final-gate slip-guess defect was fixed without changing tolerances.
- Constant ± and layered + coupled pairs pass. Positive direct/separated stress difference is 2.24% at 32² and 1.59% at 64², with unchanged .1 m labels. More than 99.5% of squared difference is near finite curve ends; central-band difference improves .173%→.0414%.
- Endpoint energy remains about17.20%→8.741% under joint refinement, essentially unchanged by representation. Native source-projection losses are separately measured; no new dissipation or energy gate was introduced.
- Rejection/cutback is bitwise identical to the matched accepted schedule. Layered restart2→4 continues two cell-crossing steps within3.2e-15 bulk/7.2e-15 surface differences. Both representation mismatches reject. Default regression and183170 assertions in152 unit cases pass. The changed old/new layered-control arithmetic comparison fails its original bound and remains recorded; A/B uses a fresh corrected control.
- Production, moving AMR, birth, cohesive energy/tip/shape work remain unqualified. The prior broader accuracy campaign stays paused. Next justified experiment: frozen finite-end stress/projection/quadrature comparison, only on a new user selection. Protected binaries/checkpoints, BP3 evidence and inherited39/45-frame/temporary-log losses are preserved.

## Historical record: Positive-translation refinement complete — 3 November 2026

The two selected 64² constant-viscosity runs at .05 m initial label spacing
and dt=.025/.0125 s passed, using four MPI ranks sequentially and unchanged
native source/binaries. [Report](POSITIVE_TRANSLATION_REFINEMENT_REPORT.md) ·
[Evidence/figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/positive_refinement/README.md).

- Endpoint active-energy remainder: 8.7411% → 4.4151%; one-halving order .9859.
  Cost: 106.10 → 180.08 s. No endpoint or constitutive change was made.
- Surface refinement at fixed dt changes stress .520%, work traction .903%,
  and slip rate .670%. The slip-rate difference is already .669% at initialization.
- Timestep halving at fixed grids changes stress .861%, point/work traction
  .0442%/.0258%, and slip rate .0563%. These are pairwise sensitivities,
  not absolute continuum-error guarantees.
- Raw boundary face-power magnitude worsens from 3.115e-6 to 4.192e-6 W/m;
  weak reaction power remains near1e-20 W/m. Boundary traction is unqualified.
- All 8/16 physical steps publish exactly once, with zero retries. Independent
  profile, donor-bound and surface-source checks pass. Earlier lifecycle/unit
  evidence is reused for unchanged code; production readiness is not claimed.

Next: diagnose independent weak boundary residual/reaction components and raw
face traction resultants from these saved states before further trajectories
or solver changes. Broader Stage C and the unrelated cohesive energy/tip gates
remain open. Protected data and inherited cleanup losses are unchanged; no
server/production restart or deferred physical campaign ran. The two-run
recommendation in the historical entries below is now completed.

---

# Bounded profile implementation and coupled comparison — 3 November 2026

**Implemented and locally verified:** native donor-bounded MacCormack on the
existing Q1 field, selected accepted-trace tensor inflow, full-normal coverage
admission/guard and checkpoint identity, all within the existing mature
material acceptance transaction. Legacy Q1/zero defaults, physical bounds,
transported obstacle, total Jaumann spin, variable-viscosity reaction and
endpoint convention remain. See [report](BOUNDED_PROFILE_TRANSPORT_REPORT.md),
[protocol](BOUNDED_PROFILE_PROTOCOL.md) and
[evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/bounded_profile/README.md).

**Coupled accuracy comparison:** 28 corrected full-support runs cover constant
and layered viscosity, both directions, separate space/time refinement and
finer numerical references. Eight lifecycle runs and two surface-resolution
runs complete the selected checks. Rejection/restart agree within 8.89e-15;
all 183,170 unit assertions pass. The original truncated-support matrix is
retained separately after correcting its support/I mismatch. This is numerical
comparison of physical fields, not an exact continuum or production benchmark.

**Adoption:** retain the bounded method as an opt-in improvement for this
restricted smooth mature-fault problem. It reduces profile diffusion; coarse
point traction is not uniformly improved. Nonconservative mass drift, the
approximately first-order endpoint energy remainder and surface-grid error
remain measured limitations. Raw DG face-traction work is not equilibrated
weak reaction work. Neither a smaller trapezoidal diagnostic nor the large
cancelling friction/background scale qualifies the endpoint scheme.

**Remaining Stage C:** surface/bulk/time convergence with coupled work/flux,
physical exterior-boundary qualification, birth/resampling, moving AMR and
deformed cohesive shape/pressure work. Cohesive energy and tip failures remain
as diagnosed; production BP3 is distinct and its first event is incomplete.
The old missing mature-transaction/heterogeneous-reaction statements below are
historical, superseded by the coupled continuation and this implementation.

**Next justified experiment:** two bounded positive-translation constant-eta
runs at 64² cells and initial label spacing .05 m, dt=.025 and .0125 s, with
identical endpoint physics and finite-domain data. Compare with the existing
64²/.1 m/.025 reference to separate surface resolution from temporal energy
error. This is a recommendation, not authorization for an automatic campaign.
No server or production restart, long replay, crust or propagation campaign
was launched. The three-example roadmap remains; no subduction example is
planned. The inherited 39/45-frame and temporary-log losses remain recorded.

---

# Boundary/profile isolation complete — 3 November 2026

[Measured report](BOUNDARY_PROFILE_ISOLATION_REPORT.md) ·
[Evidence and figures](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/boundary_profile/README.md).
The user-selected follow-up ran five bounded two-rank cases plus independent
profile/tensor reductions. Native source, core and existing checker binaries
are unchanged; a separately named checker exercises native DG scratch operators.

- Exact polynomial inflow translates to <=6.69e-15 relative error; compact tensor
  errors converge at spatial orders2.84/3.03 in both directions. Independent
  volume/face material-energy identity closes to4.68e-15 W/m. Boundary flux,
  numerical upwind loss and time-quadrature remainder are reported separately.
- Repeated Q1 interpolation has fixed-h diffusion proportional to1-CFL; smaller
  dt alone can increase accumulated profile error. The independent variance
  identity agrees to3.47e-17m². Native profile replay agrees to4.45e-16 after
  explicitly retaining deal.II's point-lookup tolerance/averaging convention.
- About99% of the retained squared tensor defect is in the inflow strip. At
  halfdt, about97% of the nominal interior defect is in its upstream edge.
  Compacting the coupled profile leaves that bulk-history boundary/front effect.
  Localization projection is separately quantified and is not its dominant cause.
- Next: a coupled test with physically specified compatible tensor inflow or a
  buffered domain, joint h/dt refinement at fixed CFL, deeper-interior errors and
  full coupled energy/flux accounting. No longer trajectory or dt-only remedy.

These experiments qualify the prescribed transport components, not production
BP3 or the full moving-fault energy balance. Cohesive energy/tip failures,
birth/resampling/moving-AMR/shape-work limits, incomplete BP3 event and recorded
39/45-frame and temporary-log losses remain. All protected data verify unchanged.
The sections below retain historical task boundaries and evidence.

---

# Current selected Stage C continuation — 3 November 2026

Stages A/B remain complete; the broader Stage C is in progress. The four-item
narrowed task has implemented the mature coupled material state and variable
viscosity reaction, locally verified translation/rejection/restart, and completed
the independent energy/tip diagnoses. See the
[coupled report](TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md) and
[diagnosis](STAGE_C_COHESIVE_DIAGNOSIS.md) for exact evidence and limitations.

- Implemented: opt-in `Jaumann DG material Lie`, complete same-parent T/R,
  curve/profile/obstacle/age/Z transaction, rebuilt work maps and schema7.
- Locally verified: matched translation in both directions, complete rollback,
  matched-schedule byte identity, two-to-four-rank cell-crossing restart;
  heterogeneous coupled bulk/surface reaction and converged outer correction.
- Physically compared: independent characteristic tensor/profile/source
  reference, one spatial halving and bounded timestep checks. Finite boundary
  and profile errors remain measured, not hidden by a solver-pass label.
- Regression: final full unit suite183170assertions/152cases passes; the new
  legacy accepted-step CSV is byte-identical to the pre-C control. Heterogeneous
  two-to-four-rank restart agrees within8.89e-15 across exported fields.
- Production-ready: no. Growth/birth, resampling, boundary completion, moving
  AMR and deformed cohesive work remain outside this enabled scope. No production
  checkpoint conversion or campaign is authorized by these results.
- Independent failures: cohesive first-order numerical energy remainder still
  misses the 1% target; tip force-merit failure precedes admission under both
  objective and legacy. Endpoint scheme and physical acceptance are unchanged.
- Next justified experiments: compact interior moving-state/inflow reference;
  higher-precision replay of the stalled tip force residual. Preserve BP3's
  incomplete first event, all controls/failed evidence and recorded data losses.

---

# Historical pre-coupling state

# Transported objective task state — 3 November 2026

Current selection is **Stage C, in progress**; Stages A/B are complete.
The earlier entries below are historical. Current evidence and exact remaining
work are in [the Stage C report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md) and
[assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c/assessment.json).
No complete native transported configuration or production qualification exists.

- Same root/branch/HEAD as entry, with preserved incoming source/manual work.
  Protected binaries, old RSF marker2/step748 and original BP3 exports remain.
- Mature matched stationary T/R reaction: full residual/load/tangent/observer,
  once-only publication/schema6, rejection/retry,2->4restart and restricted
  physical-field AMR checks pass. Legacy accepted-step CSV is byte-identical
  to the pre-C control. Final explicit transfer-cache invalidation is verified.
- Native distributed DG advection and layered relaxation operator checks have
  approximately third-order errors at512/2048/8192cells; these do not move or
  publish native curve/profile/surface state.
- Cohesive uncut DGQ2 fails the1e-3 increment target. Implemented formed-piece/
  side Q2 history retains complete native remainder/lifting. Independent cut
  errors1.93e-9/4.04e-10/1.26e-10 pass; BE forcing, sided observers, rejection,
  restart/repartition pass. Both old uncut checkpoint and old prototype-trajectory
  import are rejected. No protected continuation was converted.
- Material geometry now stages a same-label private candidate with unchanged
  histories/rates and cache invalidation. Translation/shear, association,
  invalid-trial, archive and swap rollback tests pass in serial and four ranks.
  Normal error4.49e-17. This container alone is not the complete material step.
- Full unit suite:183170assertions/152cases pass. Native simulations use a
  sequential lock and ranks1/2/4; builds respect the four-worker cap after the
  earlier scheduling mistake documented in the report. Earlier `/tmp` logs were
  lost across interruption; durable retained evidence identifies actual builds.
- Physical limits stay visible: phase-work errors149.6/74.8/37.4% of fracture
  work fail1%; tip line search fails under legacy/objective/half-dt controls.
  The missing current-measure cohesive shape/pressure work is a separate
  governing-energy gate. No damping/tolerance/load tuning substitutes for it.
- Remaining engineering: connect bulk transport, curve/phase/profile/obstacle,
  surface Z/age/slip and maps to a single accepted/rejected material-step
  transaction; implement heterogeneous coupled reaction/tangent/outer solve,
  resampling/boundary and moving AMR semantics. Those requirements are unfinished,
  not falsely called fundamental scientific conflicts.
- Next concrete experiment: after integrating that matched moving transaction,
  translate a nonuniform oblique profile/history across several cells in both
  directions with explicit inflow, then force completed-trial rejection and
  compare uninterrupted vs repartitioned restart. No server campaign follows
  from the current reaction/operator passes. BP3 first event remains incomplete,
  and cleanup loss of39/45historical animation frames remains.

---

# Historical Stage B completion — 2 November 2026

- Current user selection: **rest of Stage B, complete**; A and B1 remain complete.
  Stage C–E and production trajectories are not selected by this record.
- Root/branch: `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`,
  HEAD `27178813cebd471b93a7a9308f41215b71b4a20b` plus retained dirty spin work.
  No native source change/build/timestep in Stage B.
- Implemented bounded fixture: Eulerian Cartesian DGQ2 upwind transport and
  local heterogeneous reaction, independently evolved direct tau and T/R,
  prescribed material shear/translation, carried deforming normal profile and
  nonuniform surface memory with projected auxiliary rate. Reference uses exact
  characteristics/local attenuation and separate physical/residual integrals.
- New results:20 numerical criteria pass. N8/16/32 direct tensor errors/G are
  .00437/.000699/.0000937; separated .000568/.0000921/.0000125. Spatial orders
  approach3; characteristic Lie80/160/320 errors are first order. Reverse motion,
  q8-versus-q12, half-time-step and variable-eta energy checks pass. Omitting
  geometry, extension-gradient or rate mismatch leaves independent defects.
- **Representation selected for Stage C mature extension:** existing T DGQ2 +
  complete independent R DGQ2 + surface scalar memory. Generic full tensors
  remain in the current3-component/full3D convention. At the declared1e−3 target,
  this passes on8² cells with3456 tensor coefficients; direct first passes on16²
  with6912. At the same externally fixed mesh separated still uses twice the
  tensor storage; no mechanics-mesh coarsening/performance claim follows.
- Smaller-space screen: P1 residual at N8 best tensor projection .0013455 fails
  the unchanged target; Q2 .0004436 passes. P1 was not numerically transported.
  Q1 surface-memory16/32/64 capacity converges at second order;65 scalars give
  B-extension error9.89e−5G. Native scalar/ALE evolution is still unqualified.
  Eight consolidation/profile/selection checks pass, including material-profile
  pullback9.44e−16 and a detected .8439 Eulerian-max trail.
- Native limits: s/curve/profile are prescribed references; native ALE surface
  history, solved kinematics, phase/admission, pressure/work tangent, accepted
  transaction, restart, MPI/AMR and cohesive shape-work still require integration.
  RK4 only isolates spatial error; native lagged finite-rotation Lie is retained.
- Preserved failed evidence: initial B1 q6 error2.7011e−8 above1e−8 remains
  rejected under its original source/protocol/JSON/log; B1 selected q8 still
  passes14 criteria. Old RSF fracture/spatial screens and cleanup losses remain.
- Preserved state:921 existing source/header/unit files; protected executables,
  all RSF marker2/step748 restart generations at374.58003334330067s;220 server
  exports and all prior B1 evidence. Legacy BP3 accepted4237/saved4182 remains
  incomplete and accelerating. No server or long scientific trajectory ran.
- Evidence: [completion report](TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md),
  [index](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_b_moving/README.md),
  [spec](TRANSPORTED_OBJECTIVE_SPEC.md), [maps](IMPLEMENTATION_MAPS.md).
- Next concrete action: Stage C native R owner, complete material source/
  tangent/observer and once-only history/geometry acceptance/restart transaction,
  initially on the declared affine fixed mesh, before native moving-state and
  coupled lifecycle qualification. Cohesive native remainder/lifting/shape
  coupling remains a separate affected branch, not a mature-state equivalence.


## Stage C active milestone

EntryHEAD/branch unchanged:27178813cebd471b93a7a9308f41215b71b4a20b /
pf-rsf-particle-free. Pre-C native source delta,921file hashes, protected data
and live binary controls recorded under evidence/transported_objective/stage_c/.
Native source now implements fixed-profile matched T/R anchors, full3D spin,
shared physical reconstruction, once-only publication and checkpointv6.
The initial2rank3step case `fixed-native-moments` passes independent monomial
moments and publication: maxerror1.13e−10Pa; Rmax13.893Pa; maxspin5.92e−6rad.
Focused objective suite35assertions/6cases passed. MPI needs socket escalation;
one sandbox rejection and input/audit failures are retained, not overwritten.
The same near-equilibrium affine audit fails in `pre-c-tangent-control`, proving
it predates C; the base A*x operand precision is missing from its bound.
Distributed/hanging-face material transport and complete T/R AMR transfer are
in source, building/unverified. Moving accepted geometry, heterogeneous rates,
cohesive remainder/lifting/physical shape coupling and complete lifecycle are
still active work; do not claim Stage C complete from the first fixed fixture.
Next: finish native rebuild, qualify operator/tangent/lifecycle/AMR, then close
remaining physical moving-state integration or isolate an exact model conflict.


## Interrupted Stage C resumed (native cut-space integration active)

Root/branch/HEAD unchanged; no running simulations/builds survived the
interruption. Sixteen entry-protected files still hash-identical. The current
uncut cohesive prototype core was preserved as controls/cohesive-dg-prototype
(hash in stage_c/resume.json). Earlier /tmp build/unit logs were lost across
interruption; retained per-run logs/JSON/inputs and all original evidence remain.
Native mature matched fixed-profile checks pass: T/R Galerkin moments; exact
completed-trial rejection vs equivalent actual-dt control;2->4rank restart
T error1.33e-8Pa; T/R mesh transfer error3.84e-9Pa with3 unchanged publications
and a following solve. Mesh smoothing gives650->2600->674cells; the original
exact-cell-ID roundtrip expectation was defective and is retained as a failure.
Default legacy accepted-step CSV matches the pre-C control exactly.
Native distributed transport has third-order full-tensor error at512/2048/8192
cells:3.90e-4/4.88e-5/6.11e-6 after400m in both directions.1/2/4rank checks exist.
The layered-rate operator is in source but not yet rebuilt/qualified.
Cohesive uncut DGQ2 fails a declared1e-3 increment-relative target: errors
.176/.114/.0809 at16/64/256cells, although moments are~1e-13Pa. Native remainder
checks on warped meshes pass. Piece/side cut-Q2 capacity gives1.93e-9/4.04e-10/
1.26e-10 with1242/2916/9018 independent tensor coefficients. This is genuine
extra history state, not a quadrature remedy.
Now implementing cut physical tensor plus retained native remainder, complete
physical(A+H) finite rotation before BE, formed-geometry partition ownership,
versioned checkpoint and source restriction. The failed uncut runtime selector
is explicitly rejected; its core, input/results remain preserved.
Next: finish cut native tests/phase/restart and qualify the layered transport
operator, then continue missing native material geometry/profile/surface
transaction. Stage C is still active; general transported/cohesive-deformed
configurations and production readiness remain unqualified.
