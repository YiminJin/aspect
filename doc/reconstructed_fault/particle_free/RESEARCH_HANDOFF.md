> **FaultFriction public-interface review complete.**
> [Interface, callers and equivalence](FAULT_FRICTION_INTERFACE_REVIEW.md):
> one operation per coefficient quantity, explicit optional age, internal law
> dispatch, and zero stateless age derivative. The mature mechanics block is
> 10 lines instead of 21, with one history test instead of three law tests.
> Core/plugins build and all seven short controls pass; compared physical fields,
> histories and energy ledgers are unchanged. Both cleanup passes are included
> in the accompanying ASPECT-baseline source commit; generated evidence stays local.
> Scientific limits and deferred campaigns are unchanged.

> **Fault-friction API cleanup complete.**
> [API, caller map and verification](FAULT_FRICTION_API_CLEANUP_REPORT.md):
> one explicit-method `update_state` computes candidate age and speed derivative;
> resistance evaluation, section root solve and search potential have distinct
> names and derivative fields. Mature rate guards and all time locations remain.
> Units, focused laws, mature transport, coupled recovery and the two-rank state
> limiter pass; compared physical endpoints are identical. A historical unused
> aggregate checker dependency remains documented. Growth qualification and
> the deferred fine continuation retain their previous scope.

> **Material cleanup and bounded midpoint assessment closed — 10 November 2026.**
> [Results, implementation and decision](MATERIAL_MODEL_CLEANUP_AND_OVERNIGHT_RESULTS.md):
> shared age/RSF calculations and isolated edge fixes pass retained controls.
> Three coarse midpoint histories reach .06 s; the final M2/M3 screen passes,
> but substantial spatial sensitivity prevents resolved-growth qualification.
> S1 stops at .045625 s; the verified scalar stopping-floor correction advances
> its exact parent through .04625 s with unchanged tolerances. The fine .06
> endpoint remains missing. BE stays default and midpoint selectable. No job
> remains; longer growth, .08 continuation and instability calibration are
> deferred. The next selected experiment should complete the fine comparison
> from the retained repaired endpoint, after checking final-kernel restart.
> Historical entries below retain their original scope.

> **Free-phase synthetic control complete — 9 November 2026.**
> [Response, timestep comparison and decision](SYNTHETIC_FREE_EVOLUTION.md):
> both fresh BE histories (24/48 intervals, dt=.0025/.00125, h=.125) reach .06 s
> from the identical equilibrated t=0 seed. Central-band evolution is detected
> at .03 s; weak axial extension at .055 s. The phi=.01 tip shift is about .61h,
> while phi=.5 has no axial advance; most added fracture energy stays in the
> original band. Halving dt changes added fracture by .265%, but fine physical
> work error still exceeds it 10.42x overall and 3.50x in .04–.06 s.
> The refinement sequence is closed. Next issue: time-integration accuracy and
> cost; mesh refinement, longer growth and extra cases remain deferred.
> Solver/source/binary identities and tolerances are unchanged. Two sequential
> two-rank runs total 31.47 min; no jobs remain. Older entries retain their
> historical scope and do not authorize further runs.

> **Storage availability — 9 November 2026.**
> [Cleanup report and restoration commands](DISK_CLEANUP_20261109.md) supersede
> older storage-size estimates. Completed dense CSVs were losslessly archived;
> consult each affected run’s `ARCHIVED_FIELDS_20261109.md` before raw-field
> analysis. All checkpoint sets, current synthetic preflight and runtime
> identities remain unchanged. No scientific run was launched or resumed.

> **Synthetic specimen selected and preflight complete — 9 November 2026.**
> [Selected input and report](SYNTHETIC_PROPAGATION_BENCHMARK.md): Gc=2/3 J/m²,
> c=1000 Pa, ell=.125 m, m=4; 8×4 m box, 2 m seed half-length; prescribed
> confinement 200 Pa, Dc=.0005 m, initial age .5 s. This is a synthetic
> verification specimen, not natural-fault calibration. Uniform shear has
> compatible side velocities; no compact perturbation.
> One fixed-phase physical preflight completes eight BE intervals to .08 s in
> 80 s on two ranks. Tip/boundary Y/K=2.751/.934; ahead-of-seed native phase
> rows become favorable between .05–.06 s while boundary rows remain unfavorable.
> Cancellation is 1.06%, with small signed local reversals explicitly retained.
> The selected specimen is retained for a bounded growth verification focused
> on .04–.06 s. No growth run or refinement was launched; coarse BE work error
> remains large, so time/spatial accuracy are not qualified. Two startup failures
> and checker corrections are preserved. The old large-toughness specimen remains
> rejected. Solver equations, gates, checkpoints and protected binaries are unchanged.
>
> Older entries below retain their historical scope and do not select new runs.

> **Specimen selection closed — 9 November 2026.**
> The present 4×2 m, a=.8 m, Gc=26.6667 specimen is **rejected for the next
> growth test**. The actual AT1/cohesive calibration gives G Gc/c²=26.67 m,
> versus a=.8 m; positive sliding resistance increases that screening length.
> Retained fields confirm remote/boundary driving dominates the tip and compact
> loading hides substantial opposite-sign fault deformation through cancellation.
> The conditional baseline/perturbation pair was not launched: the scaling gate
> fails. No new coupled runs, builds, parameter changes or solver edits.
> See [assessment and definite decision](PROPAGATION_SPECIMEN_ASSESSMENT.md).
> Next: select a calibrated toughness-to-strength ratio and compatible seed/body
> size before another fixed-phase loading pair. BE accuracy and midpoint
> convergence limits remain separate; long growth campaigns remain deferred.
>
> Older entries below are historical evidence and do not select further work.

> **Consolidation complete — 9 November 2026.**
> Native DG Bernstein is now the supported fault phase representation. BE remains
> the reference/default; the shared midpoint option remains experimental. The
> single bounded repair audit reproduces the .0575→.06 failure and identifies
> near-intact phase search stagnation; no justified local correction was found.
> The compact-loading experiment is closed. Q1/CPDI fault paths and replicated
> endpoint solvers are retired; tensor transport uses owned rows on every rank
> count. Ordinary ASPECT FE_Q/FE_DGQ fields are unchanged.
> The full unit suite passes (189,151 assertions/169 cases). BE, successful
> midpoint and layered mature controls reproduce retained fields exactly;
> tip rollback and rank-changing tip/mature restarts pass at roundoff. Nine
> bounded wrappers total 166.95 s, including failed evidence. No jobs remain.
> See the [current reading path](ACTIVE_RESEARCH_MODEL.md),
> [consolidation report](TIP_CONSOLIDATION_REPORT.md), and updated
> [tip guide](TIP_TREATMENT.md). Preserved identities and data losses are unchanged.
> Next proposed experiment: fixed-phase mechanical screening with uniform shear
> plus a compact perturbation, before selecting another growth run. Broader
> midpoint solver redesign and long campaigns remain deferred.
>
> Older entries below retain their historical claims and next-action proposals;
> they do not supersede this closeout or authorize a campaign.

> **Completed bounded midpoint and loading study — 9 November 2026.**
> Entry `e8b6dd091` plus eleven preserved manual documentation changes. The
> selectable native-Bernstein midpoint candidate is implemented; **BE stays
> default**. Point equations/convergence, coupled derivatives, active MPI
> restart/rejection and regressions pass. All three difficult common-parent
> time grids fail to complete the requested window, including the final-source
> recheck. The old-loading midpoint control reduces work remainder 4.075× on
> halving dt, without establishing full coupled second order or adoption.
> The compact-loading screen and single BE pilot show boundary damage before
> resolved axial advance and strong opposite-sign section-rate cancellation.
> See [report and figures](TIP_TIME_UPDATE_AND_GROWTH_REPORT.md),
> [time equations](tip_treatment/time_integration.md) and
> [deferred cleanup](TIP_DEFERRED_CLEANUP.md). All jobs finished; 24 wrappers
> total 14.03 min, with ≤4 workers and heavy jobs sequential. Protected artifacts
> and inherited data losses are preserved. Next proposed work is the targeted
> first-failure active-row/search diagnosis at the retained .0575 parent;
> longer growth and a revised loading specimen remain deferred.
>
> Entries below describe prior tasks and retain their original evidence.

> **Current native basis closeout, 9 November 2026.**
> The bounded DG Bernstein task is complete on `d481a09c` plus preserved local
> efficiency work. See [report](NATIVE_DG_BERNSTEIN_REPORT.md) and
> [current guide](NATIVE_DG_BERNSTEIN.md). Native coefficients remove repeated
> phase transforms without changing BE, SSPRK3, bounds, quadrature or gates.
> Fixed-state/endpoint, MPI rollback/restart, moving profile, layered coupling
> and ordinary controls pass; a mature checkpoint assumption was fixed.
> Costs are mixed. All jobs finished; longer campaigns, loading changes and time
> integration remain separate subsequent tasks. Historical entries below retain
> their original evidence and do not authorize another campaign.

> **Completed bounded task — tip efficiency and localized growth, 7 November 2026.**
> Entry `d481a09c1415578ec6fb148e6b9e13c80184b6c4`; final core `8a652f3e…89115`.
> Progress-based search ordering retains the exact operator, original inverse,
> physical inputs, BE convention and final gates. Difficult-step accounted time
> falls 200.51→101.36 s (1.98×); continuation through .065 and two half steps pass.
> Bulk/phase/slip agree closely, but near-zero-rate resistance differs up to
> 35.35 Pa: full reaction equivalence is not claimed. Halving dt halves work
> error, still 5.17× added fracture. Localized growth stops at .09 on remote
> Y/Kf=.98149, with no resolved advance of fixed phase/H contours and no remote
> damage. See [final report and figures](TIP_EFFICIENCY_AND_GROWTH_REPORT.md).
> 3319 assertions, oracle/derivatives, MPI restart/rejection, ordinary Stokes
> and one layered mature check pass. All jobs finished: 24.28 min wrapped checks,
> 13.28 min separate builds; protected evidence and inherited losses preserved.
> Next proposed physical check: intact response to compact tangential loading;
> derive time accuracy/cost separately before expensive growth. Neither is
> launched. No general propagation/production qualification or mature campaign.

> **Previous completed task — bulk-tip solver integration, 7 November 2026.**
> Entry `663937bed9305df2823ac77a3fc2c3925bb94caa`; retained final core
> `1cc27f9c…7cac6`. Both-bound release, distributed exact coupling and real
> ASPECT Stokes reuse are implemented under ordinary simulator control. All
> 3319 point assertions pass. The original refined failure is reproduced and
> its unchanged checkpoint continued to .08; localized mesh/time, active MPI
> restart/rollback, ordinary Stokes and native DGQ2 mature checks pass. The
> alternative Krylov scaling is rejected as a cost improvement and inactive.
> Localized damage remains weak; mesh refinement changes slip about 11%, and
> the original fine active-window work remainder is 77.1% of added fracture.
> This is local numerical qualification, not general propagation or production
> readiness. See [report](TIP_SOLVER_INTEGRATION_REPORT.md),
> [equation-to-code guide](TIP_TREATMENT.md) and linked figures/evidence.
> About 35 min wrapped simulation/check time and 33.6 min separate builds;
> all jobs finished, protected identities and inherited data losses preserved.
> Its proposed fixed-parent comparison is now completed in the newer report
> above. No mature or server campaign is resumed.

> **Previous implementation — bulk energetic tip control, 7 November 2026.**
> The user-approved cross-section RSF closure is implemented with physical true
> stress/cohesive tensor history and authoritative DGQ2 bulk damage. This is an
> additional constitutive assumption for a fixed-reference, fixed-direction,
> prescribed-compression specimen; it is separate from mature production and
> the historical surface-damage episode. Coarse and half-dt controls reach .08 s;
> 632 point assertions, SIPG/coupled derivatives, rollback and active-slip 2→1-rank
> restart pass. The refined run fails its line search at .05 s; temporal energy
> error remains large relative to fracture and no developed tip advance is shown.
> [Current guide](TIP_TREATMENT.md) · [Implementation/evidence report](TIP_TREATMENT_IMPLEMENTATION_REPORT.md).
> The next proposed experiment is the refined .0475→.05 constrained active-set/
> stability diagnostic, with unchanged physics; not an automatic continuation.
> Candidate was tested as local changes on `36de6bcf3`; seven protected mature binaries
> remain unchanged. New simulations total about 175 s; separate builds about
> 23.6 min. No active run, production conversion or server job. Historical
> cleanup losses and mature deferred refinements remain unchanged. The previous
> prepared interface continuation is superseded as the next propagation choice.

> **Previous selection — mature closure and propagation recovery, 7 November 2026.**
> The four completed A/B/D/C runs are preserved at numerical source `84f57503c`
> and core `6ea374c5…a67d9`; no further mature trajectory is selected.
> [Temporary mature closure](MATURE_FAULT_TEMPORARY_CLOSEOUT.md) records accuracy
> limits, checkpoints and parked h=1/16/surface/quadrature/architecture work.
> The original cohesive–RSF episode checkpoint restores bitwise with its compatible
> binary; four new steps reach step 752, t=374.5801868754 s. Current-source
> fresh/rejection/restart checks pass; the additional diffuse DGQ2 growth replay
> timed out before completing its first physical step. Total launches: 8.08 min,
> plus a small benchmark-plugin build; no core changes or active jobs remain.
> [Formulation/restart report](PROPAGATION_FORMULATION_RESTART.md) distinguishes
> surface damage/fracture from mature true stress and DGQ2 phase, provides the
> proposed next episode experiment and targeted mature retest table. The event
> remains incomplete; new-window work error is 21.72% of fracture work. The
> prepared continuation is **not launched**; it requires the next selection.
> No server job, new law or checkpoint conversion. Historical failures,
> shared-assembler review and cleanup losses remain preserved.

> **Current scope, 6 November 2026 — native DGQ2 phase:** native operators,
> short growth/lifecycle checks and both positive coupled timesteps pass. The
> user deferred the reverse-direction, finer-mesh and remaining tests after the
> current run. The report and figures are complete using available results;
> no task runs remain active. Next proposed check is the deferred positive
> finer mesh, followed by reverse translation. See
> [report](DGQ2_PHASE_FIELD_REPORT.md), [design](DGQ2_PHASE_FIELD_DESIGN.md) and
> [original task](CODEX_DGQ2_PHASE_FIELD_ASSEMBLER_SOLVER.txt). Prior campaign
> results below remain evidence; no production or checkpoint conversion.

> **Completed, 6 November 2026 — corrected-spin layered campaign:** the direction/time/law/mesh, adaptive, 4→2-rank restart, provider rejection and wider-domain comparisons are complete. The finite-slip law effect persists; profile mass error remains 7.26% on the adaptive coarse mesh and 2.39% on the finer uniform mesh. Widening the box changes mechanics but leaves the remapping mass defect essentially unchanged. See [report](LAYERED_BULK_CAMPAIGN_REPORT.md), [closeout handoff](LAYERED_BULK_CAMPAIGN_RESUME.md) and [figures/evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/layered_bulk_campaign/README.md). Native/default laws and protected binaries are preserved; no production/server run or checkpoint conversion. Next proposed work is material-equation-consistent bounded profile transport, not another timestep-only sweep. No campaign run remains active.

## Historical evidence — prior selections, not current execution instructions

> **Current selection complete, 5 November 2026 — mature bulk-spin correction:** `Constitutive spin = bulk slip subtracted` now uses the coherent accepted localized slip-rate dyad in moving direct/separated histories and the stationary DG anchor. Total spin remains the default/control. All 183390 unit assertions pass; bounded constant/layered comparisons, native anisotropic-stress checks, MPI, rejection/retry and same-/cross-law restart checks pass. Pure-slip artificial rotation is removed; coupled timestep and finite-end errors remain measurable. See [BULK_SPIN_CORRECTION_REPORT.md](BULK_SPIN_CORRECTION_REPORT.md) and its source/binary/input evidence. This is local implementation/verification and a bounded physical comparison, **not production readiness**. The stopped layered campaign stays stopped; no server job or checkpoint conversion occurred. Next proposed experiment: a fresh compatible-boundary layered case with a timestep pair, explicitly selecting the corrected law.

> **Stopped at the user’s request, 5 November 2026:** the layered campaign is paused for immediate scientific revision. The active L0 test was terminated; partial outputs, completed runs, source changes and failed evidence are preserved. No further tests are authorized by this campaign continuation until the issue is addressed. The main ±3.125, refinement and adaptive/restart matrix has not run.

> **Superseded, stopped selection, 5 November 2026:** integrated layered moving-fault campaign selected by `CODEX_LAYERED_MOVING_FAULT_CAMPAIGN.txt`. Homogeneous .00625 closure, an accepted-time equilibrated layered exterior, substantial ± translation, refinement, adaptive/scheduled timesteps and matching restart are now in scope. Prior completed evidence below is preserved; its proposed stopping point is superseded. See [protocol](LAYERED_MOVING_FAULT_PROTOCOL.md). Native physics/solver and retained builds remain unchanged.

> **Previous selection complete, 5 November 2026:** prescribed exterior stress is implemented for direct mature material transport through the existing boundary callback, using the transported parent before native rotation/reaction. All five selected coupled runs pass (104 physical steps, zero retries). At 32²/.025, prescribed inflow lowers the original zero-top peak 58.8→7.93Pa; compatible loading gives 2.34Pa. Compatible temporal energy remainder decreases 6.22%→3.16%→1.59%, and 64² gives 1.591%. Latest cumulative-slip time sensitivity remains 7.05%/8.62% central/ends versus .204%/.0545% under mesh refinement. All 183170 unit assertions, split/initialization controls and provider/checkpoint guards pass. The new core/checker are under `stage-c/exterior-stress/`; retained binaries and prior evidence remain intact. See [report](EXTERIOR_STRESS_REPORT.md), [protocol](EXTERIOR_STRESS_PROTOCOL.md) and [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/exterior_stress/README.md). The frozen audit stays deferred. Next proposed test is one 32²/.00625 cumulative-slip time check with the same boundary model and labels; it was not run. Production/cohesive limits and inherited cleanup losses remain.

> **Previous selection complete, 5 November 2026:** frozen original non-affine corner velocity replay, nine manufactured transport controls on 32² with identical RK substeps and four MPI ranks. Exact exterior reproduces stress within 9.16e-15; exact-parent temporal lag converges first order, while numerical-parent inflow retains 1.37% box/16.5% corner error. Severe coupled stress amplification is not reproduced by this smooth frozen-velocity control. Native source/defaults/binaries and 994 prior evidence files remain unchanged. See [report](CORNER_REPLAY_REPORT.md), [protocol](CORNER_REPLAY_PROTOCOL.md) and [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_replay/README.md). Next proposed work is a frozen-parent mechanical corner residual/reaction audit, without material advancement. Production, broader Stage C and cohesive limits remain; inherited cleanup losses are unchanged.

> **Previous selection complete, 5 November 2026:** prescribed-flow corner isolation reproduces exact-inflow DGQ2/SSPRK3 references within 8.1e-15, while held trace retains a boundary-model bias. Two compatible-top-traction coupled controls reduce whole-domain timestep stress difference from 22.36% to .608% and endpoint energy remainder from 6.22% to 3.16% under halving. This changes physical loading and slip; it does not validate the original free-top problem or generic held inflow. Native source/defaults/binaries, 63 protected artifacts and all original shear outputs remain unchanged. See [report](CORNER_ISOLATION_REPORT.md), [protocol](CORNER_ISOLATION_PROTOCOL.md) and [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/corner_isolation/README.md). Next proposed work is prescribed replay of the original non-affine corner velocity to separate inflow feedback from mixed-boundary mechanics.

> **Previous selection complete, 4 November 2026:** four bounded sheared moving-fault runs on `cf0652547`, ordinary DGQ2/SSPRK3, 18 fixed material labels and four MPI ranks sequentially. All 44 steps publish once without retries; measured label spin is 5.25–5.58 degrees clockwise. Near-fault stress/shear/slip show useful temporal refinement, but whole-domain stress and energy fail to converge: 99.86% of the smallest-dt pair's squared stress difference is in the upper-left boundary corner. Pressure-inclusive normal traction and cumulative-slip accuracy remain open. Native source, binaries and 63 protected artifacts are unchanged. See [report](SHEARED_MOVING_FAULT_REPORT.md), [protocol](SHEARED_MOVING_FAULT_PROTOCOL.md) and [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/sheared_moving_fault/README.md). Next proposed work is a focused boundary/corner control, not another sweep or production run.

> **Previous scientific selection complete, 4 November 2026:** frozen locally split Q2 at the finite mature localization ends, on `cf06525471ce89789f1ac10905dc79f3aac7562a`. The unchanged common fixture gives 78.36×/77.93× lower 64² tensor error for 4.05% extra physical-stress coefficients; work traction and energy comparisons improve with all reference screens passing. This supports a bounded coupled direct-stress prototype, which is not implemented here. Native source/defaults/binaries/checkpoints and all 63 protected artifacts are unchanged. See [report](FROZEN_SPLIT_Q2_REPORT.md), [protocol](FROZEN_SPLIT_Q2_PROTOCOL.md) and [evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/frozen_split_q2/README.md). No rotation rechecks, native evolution or server campaign ran; broader Stage C and cohesive limits remain.

> **Previous selection complete, 4 November 2026:** stress consolidation and linked developer guide on `d7e4adbe99126f8c9faf9f35c657e287f2c29cea`. Shared `localize_surface_values`, one single/multiple-field mass projector, and explicit solver-owned outer/inner mechanics replace misleading names, duplicated algebra and hidden recursion. Full units and bounded moving, stationary, non-fault, analytical and rejection/restart controls pass; seven matched moving controls are bitwise identical. See [report](STRESS_CONSOLIDATION_DOCUMENTATION_REPORT.md) and [guide](MATURE_MOVING_FAULT.md). All 63 protected artifacts remain unchanged. Defaults, physics and checkpoint meanings are retained; scientific campaigns remain paused.

> **Previous selection complete, 4 November 2026:** the ASPECT architecture refactor removes the rejected dedicated mature driver. Ordinary simulator/solver schemes own advancement and acceptance; tensor discretization, material laws and geometry have explicit boundaries. All 183170 unit assertions and seven moving controls pass; stationary/non-fault outputs and retained serial numerical results match their baselines. See [current guide](MATURE_MOVING_FAULT.md) and [report](ASPECT_ARCHITECTURE_REPORT.md). Defaults, checkpoint meanings and 51 protected artifacts are preserved. Scientific campaigns remain paused; this is not new physical or production qualification.

> **Previous selection complete, 2026-11-04:** behavior-preserving stress-history consolidation. Tested direct/publication baseline committed as `e4b74c556`; limiter/experimental relaxation and checker-only helpers moved out of core, stress access/projection/audit boundaries clarified. Full units and bounded constant/layered, default, rejection/restart comparisons pass. See [STRESS_HISTORY_CONSOLIDATION_REPORT.md](STRESS_HISTORY_CONSOLIDATION_REPORT.md). Scientific experiments remain paused; no new physical or production qualification. Cleanup losses remain.

> **Historical selection complete, 2026-11-04:** the frozen finite-end spatial/quadrature comparison identifies uncut DGQ2 approximation as the dominant local error. Higher quadrature alone is insufficient; native source and defaults are unchanged. See [FROZEN_FINITE_END_REPORT.md](FROZEN_FINITE_END_REPORT.md). Next proposed experiment is a frozen endpoint-aware tensor-space prototype with matching work integration, only on a new selection. No coupled/production campaign was resumed. Cleanup losses remain.

> **Previous direct-stress selection complete, 2026-11-04:** direct physical-history implementation, paired native comparison and lifecycle checks are complete. Retain the opt-in path as an independent reference pending finite-curve-end stress/traction accuracy; separated remains default. See [DIRECT_TRUE_STRESS_REPORT.md](DIRECT_TRUE_STRESS_REPORT.md). The broader previous accuracy/energy campaign remains paused; plans below do not authorize continuation. All production/cohesive limits and cleanup losses remain.

# Historical record: Positive-translation refinement complete — 3 November 2026

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

# Current Stage C handoff — 3 November 2026

[Coupled material report](TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md) ·
[Cohesive energy/tip diagnosis](STAGE_C_COHESIVE_DIAGNOSIS.md) ·
[Task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md).

The selected narrower deliverable now has a complete experimental mature
material transaction: DGQ2 T/R, labelled curve, physical phase/obstacle, age,
auxiliary slip memory and rebuilt normalization/work maps use one accepted
state and publish or roll back together. Both translation directions, completed
trial rejection and two-to-four-rank restart through cell crossing pass.
Spatial heterogeneous viscosity is included in bulk and surface reaction with
a converged projection-correction outer iteration; both directions pass.
Legacy/default, stationary controls and production inputs remain preserved.
The final full suite passes183170assertions/152cases and the legacy accepted
steps are byte-identical to the preserved pre-C control. Heterogeneous
two-to-four-rank restart passes to roundoff.

Independent characteristic comparison exposes finite discretization error:
matched whole-domain error is 2.76% at 256 cells and 1.62% at 1024; the interior
error decreases from 0.0186% to 0.00842%. This is a synthetic physical comparison,
not production qualification. The general Stage C task remains **incomplete**:
birth/resampling, boundary completion, moving AMR and deformed cohesive
shape/pressure work are outside the admitted material configuration.

Existing cohesive outputs identify first-order bulk BE and surface endpoint
remainders; projection/quadrature defects are around roundoff. Signed and
absolute cumulative errors coincide, and the 1% fracture-relative target still
fails at 149.6/74.8/37.4%. The original endpoint scheme is retained. Tip traces
identify force-residual line-search merit failure before admission, with passing
contact/complementarity and linear solves. Legacy and objective traces coincide;
no load, physical tolerance or solver behavior was changed to conceal failure.

Next: isolate inflow/profile projection in a compact interior manufactured
moving-state reference; separately replay the stalled tip residual in higher
precision. No production campaign follows from these checks. Original BP3
accepted4237/latest spatial4182, protected RSF marker2/step748, old failures,
39/45-frame cleanup loss and earlier lost temporary logs remain explicit.
All work stays in the verified current worktree; no server, protected checkpoint
conversion, long replay, crust or propagation campaign was launched.

---

# Historical pre-coupling handoff

# Stage C active handoff — 3 November 2026

Continue from [the Stage C report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md),
[task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md) and retained assessment.
The native reaction/history and transport-operator work is implemented and
locally verified, including full unit suite, mature T/R AMR/restart and cohesive
cut tensor/native/lifting restart and rejection. Cut history was necessary:
uncut increment errors remain8.1% at the finest screen; native piece/side Q2
passes at1.26e-10. The uncut selector is disabled and its evidence is retained.
Same-label material geometry staging now preserves histories/rates and rebuilds
associations through native archive/swap rollback; independent translation and
shear checks pass in serial and four ranks. Full suite183170assertions/152cases
passes. This owner is still separate from the missing complete material step.

Do not treat Stage C as complete or enable native transported production inputs.
Native curve/profile/surface scalar motion and complete geometry transaction,
projected heterogeneous reaction/outer convergence, resampling/boundary/moving
AMR and deformed cohesive shape work are still missing or unqualified.
Cumulative phase physical error is149.6/74.8/37.4% of new fracture work at
8/16/32steps; all fail1%. The localized Airy tip fixture fails the same signed
line search under legacy, objective and half-dt controls. No new extension is
qualified. Next useful work is the complete matched moving-mature transaction,
followed by independent translation/rollback/restart checks; retain the
independent cohesive numerical/shape gates.

The current worktree/branch/HEAD is unchanged, with a large retained dirty delta.
Protected binaries, RSF step748 slots and original server data are preserved.
Earlier temporary /tmp build logs did not survive interruption; durable final
unit/build/run evidence is now in the worktree. Preserve the cleanup record's
39/45 historical-frame loss. No server submission, old-trajectory conversion,
long replay, crust or new propagation campaign was launched.

---

# Research handoff — BP3 review, controlled crust, propagation

**2 November Stage B completion:** the [moving/layered report](TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md)
and [task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md) supersede the next-B2 action
below. Independent Eulerian direct/separated histories now cover prescribed
material shear/translation, changing profile and layered local relaxation with
all extension/coefficient terms. Twenty numerical +8 consolidation/profile/
selection checks pass. Select complete R DGQ2 beside existing T DGQ2 and surface
memory for the first native mature extension; retain direct stress for regression.
A P1 residual misses the coarse target even in its best projection. Native
integration, solved geometry/RSF/phase, lifecycle/MPI/AMR and cohesive shape work
remain unqualified; no source build, native timestep or production conversion.
Next is Stage C, not a server event. All prior failed screens, q6 rejection,
production evidence/checkpoints and cleanup losses are preserved.

**2 November Stage B first-part handoff:** [fixed-profile report](TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md)
and [task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md) supersede Stage A's “next
comparison” entry below. Independent direct and complete separated Cartesian
DGQ2 histories now converge under time/space/quadrature refinement. Q8 meets
14 selected criteria; the initial q6 coarse-grid quadrature failure is retained.
Both representations remain candidates; the complete separated form uses twice
the tensor coefficients. Next is the remaining manufactured moving-profile/
heterogeneous-eta comparison, before native integration or representation
adoption. No native build/timestep, production checkpoint conversion, server
job or new physical campaign; all previous failed screens/losses remain.

**2 November Stage A handoff:** the current selection is Stage A of
[CODEX_TRANSPORTED_OBJECTIVE_TASK.md](CODEX_TRANSPORTED_OBJECTIVE_TASK.md),
completed as a design/algebra audit. Read [task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md),
[spec](TRANSPORTED_OBJECTIVE_SPEC.md), [acceptance report](TRANSPORTED_OBJECTIVE_REPORT.md)
and the consolidated maps before later implementation. Existing dirty native
source is retained; no binary rebuild, native timestep or checkpoint conversion.
The full residual adds profile/transport/local-versus-surface-rate terms.
Single-state interface labels and ALE node motion are explicit. Deforming
cohesive geometry additionally requires shape/pressure/measure work; that
missing coupling blocks its admission, not the mature Stage B comparison.
No transported DG/cohesive path is enabled. Next is the bounded oblique-profile
representation experiment when selected. Old RSF step748 and BP3 evidence,
packages and cleanup availability losses below remain unchanged.

**30 October review update:** the user subsequently supplied `output_start/` and
`output_resumed/`. [BP3_REVIEW.md](BP3_REVIEW.md) supersedes the older local-data
availability statement below: accepted4237 is observed, with a wall-time stop
during4238 and continued acceleration in the latest saved4182 profile. A complete
first event, any later remote job and the terminal restart label remain unknown.
The selected spin work and its explicit formulation restrictions are tracked in
[REVIEW_PLAN.md](REVIEW_PLAN.md) and [OBJECTIVE_STRESS_REPORT.md](OBJECTIVE_STRESS_REPORT.md).
The original handoff and cleanup losses below remain historical evidence.

**Binary preservation update:** both configured core paths below were rebuilt
during the selected local spin task. Their old qualified contents are now
protected under `build-particle-free/objective-stress/baseline/` as
`0-aspect-release` (old BP3), `1-aspect-release` (old cohesive) and
`2-libcohesive_rsf_native_checks.release.so`. The original hashes below describe
those copies, not the rebuilt live executables. The matching old BP3 runtime
plugin remains at its original optimization-build path; it was used successfully
with copy0. All copied binaries' runtime libraries resolve. See
[verification identity](../../../benchmarks/reconstructed_fault/particle_free/evidence/objective_stress/verification.json)
for the final BP3 build and
[entry/preservation evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/objective_stress/README.md)
for exact old identities. Neither live build nor old copy is assumed to match
HEAD solely from an embedded Git label.

Prepared 30 October 2026 after [cleanup](CLEANUP_REPORT.md). This is a research-review handoff, not authorization for another run. No server job or physical timestep was launched during cleanup.

## Source and build identity

Worktree: `/home/ein/repository/aspect-pf-rsf-particle-free`; branch `pf-rsf-particle-free`. Entry HEAD is `f2f730529289eff2c5e8fbba09a224c9bd9a7247`. The useful pending portable-pilot, geometric-clearance and activation-estimator implementation is recorded in source commit **af89a5bccf9c6771ff51b28f54c59b344cdf7cd0**; the following commit contains this cleanup/handoff. Use `git log -2` for both exact identities. The older reviewed BP3 baseline is `b2dfdebe43437d6028b06d4f9f862cc80a00cc71`; it is not a reset target.

The entry had22 tracked modifications and many untracked inputs, sources, results and packages. No incoming unique source was removed. Selected source/tests/fixtures/docs are committed; large raw data, builds, checkpoints, packages and other incoming manual/untracked files remain local and are not implied Git payload. The complete entry status/patches, exact deletion manifests and preservation checks are in the [maintenance audit](../../../benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/README.md). Recoverable source manifests identify dirty historical runs more precisely than HEAD alone.

Working binaries are intentionally distinct:

| Role | Retained path | SHA256 prefix |
|---|---|---|
| BP3/native mature qualification | `build-particle-free/native/aspect-release` | `d62541e3fc2a` |
| New cohesive propagation / RSF | `build-particle-free/cohesive-rsf-native/native/aspect-release` | `7277eebc6636` |
| Cohesive boundary/output plugin | `build-particle-free/cohesive-rsf-native/checks-native/libcohesive_rsf_native_checks.release.so` | `7bfbeaa35e7a` |

These existing binaries were not rebuilt for cleanup; embedded Git labels therefore predate the new commits. Their exact hashes and runtime libraries remain unchanged. Objects/PCH were removed; the next source build regenerates those. Historical standalone executables/plugins require indexed re-extraction before use. Shared installations and other worktrees were untouched.

Local qualified stack: GCC12.4.0, OpenMPI5.0.6, deal.II9.6.2, Trilinos14.2, Boost1.84 and p4est2.8.7; see the RSF package `environment.json` and preserved CMake caches/ldd records. SUNDIALS support is required in deal.II. NetCDF/WorldBuilder are enabled in these builds. Keep compiler/MPI/deal.II compatible between ASPECT and plugins; archive portability is not assumed. Analysis needs Python3, NumPy, Matplotlib and Pillow; archive recovery uses zstd.

Future rebuild commands, **not executed by cleanup**, using the existing configuration:

```sh
export DEAL_II_DIR=/opt/dealii/9.6-local
export CXX=/opt/gcc/12.4.0/bin/g++
cmake --build build-particle-free/cohesive-rsf-native/native -j 4
cmake -S benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native \
  -B build-particle-free/cohesive-rsf-native/checks-native \
  -DAspect_DIR="$PWD/build-particle-free/cohesive-rsf-native/native" \
  -DCMAKE_CXX_COMPILER="$CXX"
cmake --build build-particle-free/cohesive-rsf-native/checks-native -j 4
```

For a fresh portable native smoke case use [ACTIVE_WORKFLOW.md](../../../benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/ACTIVE_WORKFLOW.md) and standalone `active.prm`. It prescribes seed, all initial histories and loading; no prototype JSON/cache/checkpoint is a fresh-run prerequisite. Use a new output directory and the current plugin. Do not replace a production parameter file with this different physical experiment.

## BP3: inspect supplied results before further experiments

The selected BP3 route is unlimited domain-free DGQ2, frozen mature fault/profile/mesh, stationary diagnostic particles, Gauss6/source3 and Helmholtz20. It retains the mature RSF/contact interpretation and its own acceptance/history lifecycle. Local optimization, MPI, rejection/restart, collective wall-stop and source/publication checks are qualified only in their stated small-fixture scope. FE/particle controls are comparisons, not physical accuracy references.

Original unreviewed first-event material remains under:

- `benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/`: both original/replay stdout/stderr, `log.txt`, four server inputs and environment/resource instructions.
- `.../evidence/first_event/`, `.../evidence/server_3610007/`, `.../evidence/server_3613912/` and supplied packets/archives: original fields, accepted clocks, diagnostics, resolved inputs and independent reductions.
- `build-particle-free/bp3-optimization/`, `bp3-work-diagnostic/`, `bp3-work-repair/`: local reference/correction controls and preserved restart evidence. Some full local arrays are lossless `.csv.zst`; they are not original server trajectory files.

Verified observations: the supplied parent label is809 at5172072493.8510532s; replay reaches815 at5172072493.8982706s. The observed failure at816 is an observer work-consistency check. The local-frame correction is independently diagnosed/qualified in [BP3_DG_WORK_OBSERVER_FIX.md](BP3_DG_WORK_OBSERVER_FIX.md). The earlier homogeneous-Maxwell coefficient issue and repair are separate ([report](BP3_DG_HOMOGENEOUS_COEFFICIENT_FIX.md)). Neither repair demonstrates first-event physical accuracy or successful remote completion.

**Current remote completion is unknown.** The local packets have checkpoint labels, not the complete160-rank server restart payload. Preserve the actual server restart directories. The prepared815→818 verification and continuation commands are in [the replay README](../../../benchmarks/reconstructed_fault/particle_free/evidence/server_3613912/README.md); their remote completion has not been verified here. No cleanup removed or reconstructed any remote data. Retain `strict_reference_match_passed=false`, spatial/endpoint temporal findings and raw sided traction errors.

BP3's historical complete source package remains at `build-particle-free/bp3-optimization/package/bp3-dg-stationary-optimized-20261017-r3.tar.gz`. Its later coefficient/observer corrections must follow the documented scope. Build/run locations requested for BP3 remain `$WORK/build/aspect/pf-rsf-dg/build` and `$SCRATCH/bp3_dg/plugin/build`; native launch is `ibrun ./plugin/aspect-release parameters.prm` or the verified branch's native resume input. **Do not launch simply because these commands are recorded.**

## New cohesive-interface propagation and RSF status

This newer default-off formulation is distinct from BP3 and from the earlier evolving bulk-phase laws. Native bulk state retains27 DGQ2 coefficients/cell plus the complete supported native-FE gradient remainder; pressure and constraints are actual mixed ASPECT mechanics. Surface Gauss records own signed jump, irreversible surface damage a, normalized age, cohesive reference, relaxed jump and accumulated absolute slip. The system owns geometry, Q1 jump/memory/rate, identities and admission clocks. Temporary zero-history geometry is replaceable; formed material/history is persistent. One fracture inventory is `Gc integral(a ds)`; geometric phi is not a second fracture charge. The accepted law, pressure coupling, radiation and physical persistence R are unchanged.

[Native pilot](COHESIVE_RSF_NATIVE_REPORT.md): ASPECT advances179 positive steps over .440–.462375s, admits curved material, fully decoheres two newly admitted cells and continues RSF sliding during hold. General nonuniform tensor memory, actual rejection/cutback, single publication and same-rank restart are checked. The original preparation-to-native mapping has explicit stress/storage checks; the active fresh workflow no longer depends on that mapping. [Portable/domain report](COHESIVE_RSF_PORTABLE_REPORT.md) qualifies configuration/restart contracts and full sampling/lifting clearance, not domain independence.

The first instability experiment is a separate fresh16×12m finite-wall case,128×96 bulk cells and96 seed intervals,4MPI ranks. The matched b=a control and acceleration before damage establish RSF feedback. The near-intact raw-log-rate floor failure is preserved, diagnosed and repaired with residual/tangent-informed continuous-history defects ([equations](RSF_EPISODE_ESTIMATOR.md)); no damage exemption, smaller floor or tolerance change.1343 assertions and native lifecycle/default controls pass.

Current accepted continuation: **step748, t=374.58003334330067s, four ranks**, under `build-particle-free/cohesive-rsf-instability/runs/episode-local/output/restart/`. Marker2 selects slot02; complete slot01/03 fallbacks remain. Maximum observed rate .0450089m/s, final .0444673m/s: only shallow decline, **not a completed acceleration episode**. Added damage is cap/cohesive-zone amplification; fully decohered length remains3m and no new persistent geometry forms. No physical/numerical guard failed in the last segment; its wall stop is operational. [Current report/figures](RSF_EPISODE_REPORT.md).

Warm throughput8.11s/step implies roughly5.9local hours per another .1physical seconds. The [complete server handoff](../../../benchmarks/reconstructed_fault/particle_free/rsf_episode_server/README.md), sibling47MB `rsf_episode_server.tar.gz` and checksum retain this exact checkpoint, source, environment, ten inputs and validation. No server job was submitted. Read-only restart/startup/control are required on the actual server before continuation. Native direct launch after those checks is `ibrun ./plugin/aspect-release parameters.prm`, then `resume.prm` in that **separate RSF package**, not in BP3. The1.5h accepted-state wall budget needs scheduler margin for another solve, output and checkpoint;376s and20000steps are bounds, not proof of decay. Fresh doubled bulk/surface spatial and focused temporal inputs are prepared, not executed.

## Limits that remain visible

- Cleanup pruned some raw bulk frames needed to re-render older animations, including39 of45 original native-pilot frames after exact-alias recovery. The renderings and quantitative/surface histories remain; see the explicit [availability loss](CLEANUP_REPORT.md#legacy-animation-data-loss). Do not promise complete raw-animation reproducibility or replay the pilot without authorization. Current RSF surface/event data, accepted histories and original BP3 server results are unaffected.

- Quasi-static bulk plus radiation damping does not resolve seismic waves or validate physical rupture speed.
- Finite walls affect pressure and cohesive response. R is an uncalibrated physical path-persistence hypothesis; numerical lifting width and temporary reserve are separate. Free topology, general bends, AMR/transport and production adoption are not implied.
- The resolved-lifting correction improves the demonstrated response, but point/sided tractions and ahead/bend normal reactions remain unqualified. Contact multiplier accuracy alone does not establish point-stress accuracy.
- Original early spatial, branch and temporal failures remain. The RSF repair is a local continuous-error estimator, not a1% global certificate. Prior mature-age differences reach7–9%; spatial weak-load spread25.2% remains. No finer server event comparison was run.
- In the last RSF segment the absolute physical-work remainder is .0573% of sliding loss but20.8% of new fracture work. The fracture-relative screen still fails. Exact backward-Euler ledger closure is separate from physical energy accuracy; retain cancellation-aware, per-step and accumulated remainders.
- The old symmetric incremental-potential stability result applies only to its qualified homogeneous matched-coefficient bulk law. Do not transfer that certificate to nonsymmetric frictional phase/contact laws.

## Research-review sequence and revised paper plan

1. **BP3.** Inspect the supplied first-event outputs first: actual completion, slip/state/stress evolution, timestep/rejections, pressure convention, matching-frame work and publication. Do not infer success from a package or local control.
2. **Controlled viscoelastic crust experiment.** Use a simple, explicitly bounded geometry/loading to show Maxwell history, friction/contact, stress redistribution and energy accounting with interpretable physical scales. Parameters/boundaries require a declared experiment; this handoff does not select another run.
3. **Fault propagation.** Present the energy-accounted cohesive-interface/native ownership, curved admission and mature sliding milestones with their path hypothesis, boundary sensitivity and numerical limits. Complete the first RSF episode only under a selected continuation task, then separate event timing from progress-aligned growth. No realistic subduction application is planned.

Immediate next work is review and mathematical maps, not a new simulation campaign. Prepare:

- A variable/ownership map distinguishing physical stress from T_DG/native remainder; production mature memory from cohesive reference/relaxed jump; bulk phi from surface a; actual jump/slip rate/age; solved p' from p0 and variational/raw sided pn. Include units and exact source functions.
- A timestep map from accepted n through immutable trial n+1: prescribed displacement increment, Maxwell beta, current-pressure return/admission, native residual/tangent, error estimation, rejection/cutback and once-only commit. Distinguish numerical metadata from physical history and restart serialization.
- An energy map separating bulk/cohesive recoverable storage, single fracture inventory, friction, radiation, Maxwell dissipation, boundary work, BE numerical terms and trapezoidal physical remainder. Plot per-step as well as accumulated error against both sliding loss and newly created fracture work.

Start from [current_design.md](../current_design.md), [specification.tex](../specification.tex), [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [STATUS.md](STATUS.md), [native formulation](COHESIVE_RSF_NATIVE_FORMULATION.md), [candidate admission](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_FORMULATION.md), [front weak admission](COHESIVE_INTERFACE_FRONT_FORMULATION.md), and the indexed [native](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/INDEX.md), [BP3](../../../benchmarks/reconstructed_fault/particle_free/evidence/bp3_optimization/README.md) and [RSF](../../../benchmarks/reconstructed_fault/particle_free/evidence/rsf_episode/INDEX.md) evidence. Historical P2/P4/bulk and CPDI reports retain their scope; they are not the newer interface equations. Source governs APIs. Resolve a scientific conflict explicitly rather than silently merging these formulations.
