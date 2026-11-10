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

> **Storage maintenance — 9 November 2026.**
> [Cleanup and recovery](DISK_CLEANUP_20261109.md): 19.37 GiB recovered,
> 21.27 GiB available at verification. Large completed CSV fields now require
> selective restoration; no scientific data discarded. Current preflight,
> checkpoints, executables and the scientific selection below are unchanged.

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

> **Tip documentation synchronized to `e8b6dd091`, 9 November 2026.**
> [TIP_TREATMENT.md](TIP_TREATMENT.md) and its linked pages describe the committed
> native Bernstein tip route, coefficient/physical-field distinction, diagonal
> search metric, exact acceptance ledger, section-rate publication and basis-safe
> restart. The prescribed-displacement search functional is documented without
> subtracting diagnostic reaction work. Historical reports remain unchanged.
> This is a documentation-only audit; no solver, input, binary or numerical
> qualification changed, and no simulation was launched.

> **Completed bounded task — native DG Bernstein phase, 9 November 2026.**
> Entry remains `d481a09c` plus the preserved efficiency implementation. Native
> phase coefficients, initialization, fracture/transport, bounds and restart are
> integrated; defaults remain nodal. BE, SSPRK3, quadrature, closure and gates
> are unchanged. Final core `76e9c5c1…5641a`; [report](NATIVE_DG_BERNSTEIN_REPORT.md)
> and [current basis guide](NATIVE_DG_BERNSTEIN.md) contain evidence and limits.
> 5692 assertions, operator/limiter equivalence, fixed-parent difficult step,
> localized growth, active MPI restart/rejection, layered coupling and ordinary
> Stokes pass. A mature serializer's nodal-point assumption was found and fixed;
> its failed run is preserved. Difficult phase/stress differences are 4.68e-16 /
> 1.59e-12 Pa. Cost is mixed: difficult step 101.36→109.94 s; localized total
> 12.49→10.61 s. No general speedup or propagation qualification is claimed.
> New numerical wrappers total 4.05 min; builds about 11.7 min of log intervals.
> All jobs finished; protected identities and inherited losses preserved.
> Next selected scientific milestone remains endpoint time accuracy and an
> interpretable localized-growth specimen. Neither was started in this task.

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

> **Completed — restored DGQ2 coupled validation, 7 November 2026.**
> All selected A/B/D/C cases reached T=.8 on verified native integration
> `84f57503c` behavior: 384 accepted steps, zero retries, matching exterior
> references, four ranks sequentially. Numerical source and protected binaries
> are unchanged; the shared-assembler review remains inactive. Phase balance
> defects stay below 3.22e-15, but B/D central/end surface stress differs by
> 5.48%/3.71%, endpoint age by 4.13%, and localized-work quadrature remains
> sensitive. The coupled solution is not yet spatially qualified for a paper
> server pilot. The state transient also remains timestep-sensitive.
> [Report, data and four figures](DGQ2_COUPLED_VALIDATION_REPORT.md).
> Measured run/reduction overhead was about 4 h 20 min; no compilation, replay,
> source change or production/server run. All jobs are finished. Architecture
> remains frozen. Next proposed single comparison: +U, h=1/16, dt=.00625,
> T=.8, unchanged 20 labels; **not launched or authorized by this closeout**.
> Other deferred experiments and inherited cleanup losses remain unchanged.

> **Previous selection — phase architecture restored and frozen, 6 November 2026.**
> Normal revert `319b10b94` restores the native-integration implementation from
> `84f57503c`; the preceding DGQ2 and native-integration commits remain intact.
> Native phase retains its registered phase cell assembler, mapped private
> transport/SIPG face assembly, explicit SSPRK3/BP and ordinary composition state.
> Full shared face assembly is historical review work at
> `review/shared-scalar-dg-3a21d59cf`, not current implementation.
> [Restoration and short checks](PHASE_ARCHITECTURE_RESTORATION.md) ·
> [Historical shared-assembly report](SHARED_SCALAR_DG_REPORT.md).
> Freeze phase architecture, quadrature, time integration and acceptance rules
> for the next numerical-validation stage. No further architecture work,
> quadrature reduction, cost profiling or experiment is selected by old plans.
> Deferred reverse/fine/full-trajectory and server experiments remain deferred
> until explicitly selected. Historical findings and cleanup losses remain.

> **Previous selection complete — native phase ASPECT integration, 6 November 2026:**
> native phase is one semantic composition, with ordinary field indexing/data,
> simulator-owned one-attempt transport and registered spatial assembly. Mapped
> scalar transport/SIPG, physical-mean bounds and stage-time inflow are implemented;
> mass/fracture caches avoid rebuilding stable operators. Zero-particle mature
> stepping, loaded fixed-fault cohesive growth, rejection and 4→2-rank restart pass.
> All 155 unit cases / 185,551 assertions pass. Bounded test runtime is 451.00 s;
> protected artifacts and retained binaries are unchanged. Legacy P1 physical
> inadmissibility is reproduced; the old I_h smoke initialization failure remains
> explicit. Native Q1 and legacy CPDI frozen-input checks pass. Coupled tensor
> history is still Cartesian; no new physical or production qualification.
> [Report](NATIVE_PHASE_INTEGRATION_REPORT.md) · [Current contract](DGQ2_PHASE_FIELD_DESIGN.md).
> Reverse/fine/full-trajectory and server runs remain **deferred**; no runs remain
> active. A matched-subcycle frozen velocity replay is the next proposed accuracy
> check, not an automatically selected campaign.

> **Previous selection, 6 November 2026 — native DGQ2 phase:** native transport,
> SIPG constrained evolution, short cohesive growth, units and lifecycle checks
> pass. Both positive-translation coupled cases (h=.125; dt=.0125 and .00625)
> completed 64/128 steps without retries. At the user's request, reverse
> translation, the finer mesh and further tests are **deferred**. The automatic
> queue was closed before either remaining full run started. Report, compact
> comparisons and five coupled PNG figures are complete. DG phase balance
> closes at about 6e-15; profile shape improves substantially, while coupled
> time/space accuracy remains limited. This task has no active runs. See
> [report](DGQ2_PHASE_FIELD_REPORT.md)
> and [design](DGQ2_PHASE_FIELD_DESIGN.md). No production/server run or checkpoint
> conversion is selected.

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

> **Previous selection complete, 4 November 2026:** fault-code readability and ownership refactoring on `47c2aae3be15d3ff60aab2a818c2f6e179077702`. Numerical `TensorField` is separate from material representation/residual and fault observations; material projection receives explicit inputs. All 183170 unit assertions, seven matched moving controls, stationary/serial/non-fault controls, analytic source checks and old/new checkpoint controls pass. Matched moving results are bitwise identical; new 2→4-rank restart agrees within 8.9e−16 bulk / 1.1e−14 surface. AStyle 2.04 is idempotent over all 44 primary files; all 57 protected hashes are unchanged. See [report](CODE_READABILITY_REPORT.md), [current guide](MATURE_MOVING_FAULT.md) and [provenance](../../../benchmarks/reconstructed_fault/particle_free/evidence/code_readability/final-provenance.json). Scientific campaigns remain paused; no new physical or production qualification.

> **Previous selection complete, 4 November 2026:** the ASPECT architecture refactor removes the rejected dedicated mature driver. Ordinary simulator/solver schemes own advancement and acceptance; tensor discretization, material laws and geometry have explicit boundaries. All 183170 unit assertions and seven moving controls pass; stationary/non-fault outputs and retained serial numerical results match their baselines. See [current guide](MATURE_MOVING_FAULT.md) and [report](ASPECT_ARCHITECTURE_REPORT.md). Defaults, checkpoint meanings and 51 protected artifacts are preserved. Scientific campaigns remain paused; this is not new physical or production qualification.

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

# Stage C coupled continuation — 3 November 2026

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
The rebuilt full unit suite passes183170assertions/152cases; the legacy
accepted-step control is byte-identical to the preserved pre-C output.
Heterogeneous two-to-four-rank restart also agrees to roundoff.

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

# Historical earlier Stage C boundary

# Stage C native integration active — 3 November 2026

[Measured Stage C report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md) ·
[Assessment and figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/stage_c/assessment.json) ·
[Task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md).

Implemented and locally verified: matched mature T/R reaction anchors and
schema6; native residual/load/tangent/observer consistency; distributed/hanging
DG transport and spatially varying relaxation; complete T/R AMR transfer;
cohesive piece/side Q2 physical history with native remainder/lifting, sided
observers, BE forcing, rejection and2->4rank restart. Legacy remains default
and its matched accepted-step control is byte-identical to the preserved core.
Material geometry staging preserves same-label histories and supports archive/
rollback with rebuilt associations; its translation/shear checks pass in serial
and four ranks. The full unit suite passes183170assertions in152cases.

Stage C is **not complete**. Native material geometry/profile/surface transport
and its full transaction, coupled heterogeneous reaction, moving resampling/
boundary/AMR, and deformed-cohesive shape/pressure work remain unqualified or
unimplemented. The cohesive uncut DGQ2 representation fails and is disabled;
the implemented cut space passes. The phase physical-energy screen still fails
at8/16/32steps; localized tip admission also fails with legacy and half dt.
Numerical convergence is not a physical or production-readiness claim.

Next: integrate and qualify the complete matched moving-mature transaction,
using the already checked operators. Preserve the independent tip/shape-work
gates. Original BP3 accepted4237/latest4182, protected RSF marker2/step748,
old failure evidence and39/45-frame cleanup loss remain. No server job,
production conversion or long event replay was launched. Entries below are
historical and retain their original scope.

---

# Transported objective Stage B complete — 2 November 2026

[Moving/layered report and figure](TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md) ·
[Stage record](TRANSPORTED_OBJECTIVE_REPORT.md) · [Task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md).
Twenty prescribed Eulerian-transport/profile/heterogeneous-rate criteria and
eight consolidation/profile/selection checks pass. At32² cells, scaled tensor
errors are9.37e−5 direct /1.25e−5 separated; independent references verify the
geometry, extension-gradient and local/projected-rate sources. Their omission
leaves nonvanishing defects. Native adoption is separate from these fixtures.
Selected for the first native mature implementation: T DGQ2 + complete R DGQ2
+ surface memory, with direct stress retained as an independent control. It
meets the benchmark target with fewer isolated-history coefficients than the
other tested options; the same-mesh storage penalty remains explicit.
Next is Stage C native source/tangent/observer and transactional state/restart
integration. Current native guards/defaults and protected source/binaries/
checkpoints remain. Prior q6 rejection, physical screens and cleanup losses
are retained. No native build/timestep, server or long trajectory ran.

---

# Transported objective Stage B first part — 2 November 2026

[Fixed-profile report/figure](TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md) ·
[Acceptance record](TRANSPORTED_OBJECTIVE_REPORT.md) ·
[Task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md).
Implemented an independent direct-stress and complete T−B−R comparison on
Cartesian DGQ2 cells cut by a37° fixed fault, with nonzero slip/zz and varying
spin. Spatial errors converge at approximately third order; Lie time errors
at first order. Finest scaled tensor errors are1.16e−5 direct /9.05e−6 separated.
Omitting R leaves .106 error. The selected q8 rule passes14 criteria; the
original q6 coarse-mesh quadrature candidate fails its unchanged1e−8 bound and
is retained/rejected. Complete Q2 residual is viable for this fixed-profile
control, at twice the tensor storage of direct DG. No final native choice yet.
Next: remaining Stage B moving-profile/heterogeneous-eta comparison. Native
transport/growth/AMR guards, source and protected checkpoints remain unchanged;
no server/native/long physical trajectory ran. Existing cleanup losses and
failed physical screens are retained.

---

# Transported objective Stage A — 2 November 2026

[Task](CODEX_TRANSPORTED_OBJECTIVE_TASK.md) · [Spec](TRANSPORTED_OBJECTIVE_SPEC.md) ·
[Report/acceptance matrix](TRANSPORTED_OBJECTIVE_REPORT.md) ·
[Task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md).
The current user selected Stage A only. Physical transport/total-spin equations,
interface versus node velocities, geometry/profile/measure derivatives,
material irreversibility, complete residual/coefficient terms, loading and
mature/cohesive compatibility are derived. The existing maps now carry the
consolidated proposed owner/transfer/checkpoint contract. 512 mature and128
native-lifting exact-rational identities pass; independent geometry differences
are below1.3e−10; scalar variable-rate errors show first order.
No native configuration is newly enabled. Solution-driven deformed cohesive
admission requires the missing shape/pressure/measure-work coupling; pure
translation and the mature representation comparison are independent.
All921 pre-existing source/header/unit files, protected binaries/checkpoints
and server inputs are preserved. Existing failed scientific screens and cleanup
losses remain. Next: Stage B oblique-profile direct versus complete separated
tensor comparison, not a production replay. No server or native timestep ran.

---

# Server computing-statistics figures — 31 October 2026

Visualization revised to one four-panel figure with unified histories and
combined wall-time-weighted timer bars (61.1% condensed solves,13.6% checkpoint
writing). Per-log accounting remains in the report and data sidecars only.

[Report and figures](BP3_COMPUTING_STATISTICS_REPORT.md) now show timing,
solver effort, residual reduction and timestep histories from the two server
logs. Condensed solves consume approximately68%/59% of the last reported
invocation wall time; checkpoint writes11%/14%. All4238 explicit accepted
records finish below the logged nonlinear tolerance; unaccepted attempts are
excluded. Timer coverage ends at800/4200, earlier than the accepted tails.
Three parser checks pass. No solver/source/binary/checkpoint or input change;
physical accuracy, incomplete-event and cleanup limits remain unchanged.
Next performance experiment, only if selected: bounded condensed-solve profiling.

---

# Server cumulative-slip figure — 31 October 2026

[Plot/report](BP3_CUMULATIVE_SLIP_REPORT.md) ·
[Python](../../../benchmarks/reconstructed_fault/particle_free/plot_bp3_cumulative_slip.py).
The two supplied server segments now produce a Figure8-style cumulative-slip
PNG/PDF using5-year blue and1-second red targets, nearest recorded states only.
All33 regular blue targets are supported. Ten early red target times fall in
a10.69 s saved-data gap and are omitted; the main figure draws34 blue/9 red
contours including dashed regime/end states. Actual times and hashes are exported
with the plot, plus a sampling diagnostic. Five focused data tests pass.
All220 server-export hashes remain unchanged. The final profile is4182, still
coseismic; no completed event or unsaved4237 contour is inferred. No solver,
binary, checkpoint or physical input was changed. Existing spin work and all
accuracy/cleanup limitations below remain. Next plotting action is the same
read-only reduction when additional profiles arrive; no simulation is selected.

---

# BP3 review, implementation maps and bounded objective stress — 30 October 2026

[BP3 review](BP3_REVIEW.md) · [Formulation/variable/timestep/energy maps](IMPLEMENTATION_MAPS.md) ·
[Spin report](OBJECTIVE_STRESS_REPORT.md) · [Active plan](REVIEW_PLAN.md).

Entry `27178813cebd471b93a7a9308f41215b71b4a20b`, required branch/worktree,
initially no tracked changes. Setup supplement was already present exactly
once; plan/spec matched the supplied package and were reconciled in place.
Both qualified binaries and the cohesive plugin were preserved by hash before
rebuilds; shared runtimes remain available. No reset, checkout or cleanup.

- **BP3 evidence reviewed:** new exports show accepted4237 at
  5172072500.6796865 s; job3616642 hit its wall limit during4238. Latest saved4182
  has Vmax1.37553 m/s, still the largest saved rate. First-event peak/decay and
  completion remain unobserved. Exact missing records are listed in the review.
  Label this incompressible diffuse-fault calculation BP3-derived: official
  ν=.25 differs, and prior spatial/temporal/sided-traction failures remain.
- **Source implemented:** default-legacy `Stress rotation` option, with
  `Jaumann lagged Lie` for matched-constant-material stationary mature physical
  particle stress only. Total accepted-gradient spin; finite rotation followed
  by existing exponential Maxwell/source; first-order overall. Bulk/surface
  residuals, observers, diagnostics, outer acceptance and restart provenance
  use the same convention. No spin/forcing dissipation or tolerance tuning.
- **Locally verified:** final Voro++ build183073 assertions/148 cases; independent
  rotation/relaxation/forcing/covariance/shear/energy references; pressure-coupled
  residual audit, exact rejection/retry and same-rank restart; 2→4 tensor
  relative L2 difference9.93e−15. Legacy control agrees with the protected older
  binary to roundoff-scale measured precision, not bitwise or revision identity.
- **Physically compared:** only three-step stable BP3-like off/on sensitivity:
  tensor RMS .124 Pa, max normal-traction change .064 Pa. This does not qualify
  the remote event, full physical energy balance or finite-strain elasticity.
- **Production readiness:** objective DG/FE and cohesive paths explicitly
  disabled. Mature DG needs spin-induced localized residual history to retain
  Qgeo=0; the new cohesive pilot needs the complete native-remainder/lifting
  and structural-frame energy derivation. Transport/AMR remain disabled.

Next: one resolved fixed-profile, spatially varying-spin comparison of direct
physical stress and T−B−tau_res, including normal/shear/work/storage/projection
errors under dt and normal-resolution refinement. Decide its independent history
space before an objective DG implementation or production spin branch. The
controlled-crust and propagation paper examples remain on the roadmap, deferred;
no realistic subduction example is planned. Preserve the separate RSF step748
checkpoint, incomplete episode and failed fracture/spatial screens. The cleanup
record's historical data losses remain unchanged. No server or long campaign ran.

---

# Cleanup and research handoff — 30 October 2026

[Cleanup report](CLEANUP_REPORT.md) · [Research handoff](RESEARCH_HANDOFF.md).
Generated artifacts were reduced; unique source, original BP3 data, native runtimes,
current RSF step748 and complete server package are preserved. No scientific
evolution or conclusion changed. The next research-review work is BP3 inspection
and variable/timestep/energy maps, then the controlled-crust/propagation paper
sequence. No realistic subduction application or new run is selected.
The RSF episode remains incomplete and all recorded accuracy failures remain.
Some historical bulk-animation frames were pruned in error; the cleanup report
records the missing frames, exact recoveries and limits on re-rendering. Current
RSF surface/event data and original BP3 server results are unaffected.

---

# First RSF episode: repair verified; server continuation handoff

[Report](RSF_EPISODE_REPORT.md) · [Estimator](RSF_EPISODE_ESTIMATOR.md) ·
[Event PNG/GIF/data](../../../benchmarks/reconstructed_fault/particle_free/evidence/rsf_episode/INDEX.md) ·
[Server instructions](../../../benchmarks/reconstructed_fault/particle_free/rsf_episode_server/README.md).

The selected default-off estimator now controls continuous jump/age/damage
defects and algebraic resolution from the actual residual/tangent. No low-damage
exemption, state reset, tolerance change or smaller dt floor.1343 independent
assertions pass. The original failed step661 advances with identical j/a/V;
rejection/retry/restart and retained defaults are verified, including2→4 ownership.

A healthy4-rank continuation reaches accepted748, t=374.58003334330067s.
Largest observed rate .0450089m/s, final .0444673: only shallow decline, not a
completed episode. Added damage2.76832e−4m, max Δa=.00433, fully decohered length
still3m, no new persistent geometry. Native guards hold. The current window's
physical remainder is .0573% of sliding but20.8% of fracture: fracture screen
still fails; previous global state/spatial errors remain visible.

The native wall-limited segment writes a valid checkpoint; it is not physical
arrest. Warm steps8.11s imply roughly5.9h locally for another .1 physical seconds.
Per latest steering, server materials include this complete accepted4-rank state,
source/plugins/environment, strict zero-advance reference, startup/control/resume
inputs and direct launches. Local parameter/source-config/restart checks pass;
server build/startup and completion of the physical episode remain unqualified.
No server job or finer trajectory was launched. Fresh matched coarse/fine inputs
are prepared only. Next: continue this checkpoint through real peak/decay, then
separate event timing from progress-aligned growth comparisons. Preserve BP3,
all earlier failures and physical parameters; no new propagation law or calibration.

---

# Bounded native RSF comparison closed; numerical continuation limit retained

[Concise report](COHESIVE_RSF_INSTABILITY_REPORT.md) · [GIF/PNG/CSV index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_instability/INDEX.md).

Native RSF feedback is demonstrated: maximum observed rate .0446 m/s versus
1.63e−7 in the initially matched b=a control, with acceleration before any
cohesive growth. Small adjacent activation occurs, max Δa=.00291; fully decohered
length remains3 m. The600-step primary ends during acceleration, not at arrest.

The half-limit replay stops before publication at step661 when the rate-change
indicator at a near-zero-damage point requests dt<1e−10 s. The failure reproduces
exactly; accepted checkpoint files and all incoming physical exports remain
bitwise unchanged. Long replay38.37 min; control11.26 min; two MPI ranks.

Common-time growth/slip differences .58%/.039%; mean mature state7.47% (up to9.04%
in the window), mature-state L2 14%. The1% history target fails. Physical work
remainder .157% of sliding dissipation but14.9 times new fracture work: fracture
screen fails. Frozen bulk weak-load spread25.2%, prior point/ahead-normal failures
and absolute-onset uncertainty remain. No developed coupled propagation claim.

Next numerical action: qualify an admission/event-aware rate error estimate with
algebraic uncertainty, retaining continuous-history checks, physical equations,
strict guards and accepted state. No floor reduction or newborn exemption was
adopted. Production/BP3/defaults and prior evidence remain protected; no further
trajectory or parameter campaign is launched automatically.

---

# Portable native workflow and geometry-aware domain protection

[Report](COHESIVE_RSF_PORTABLE_REPORT.md) · [active inputs/direct commands](../../../benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/ACTIVE_WORKFLOW.md) · [compact verification](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_portable/README.md).
Baseline f2f730529 plus bounded local cleanup. Original qualified data and failures
are preserved below; no longer propagation experiment was run.

`active.prm` starts from explicit seed/Gauss/tensor/native-memory functions without
untracked prototypes, kernels or checkpoints. Loading, geometric/computational
limits and build/MPI paths are configurable. A direct MPI launch and the bounded
runner reproduce identical physical exports; relocated active analysis needs no
historical reference. Changed loading/domain and inconsistent Q1 memory are
rejected. Custom initialization/restart preserves all physical records exactly.

The complete ahead-R sampling disk and all formed/temporary lifting support now
require positive certified clearance inside the actual rectangular bulk mesh.
Approach is logged with limiting wall; missing support rejects before publication.
The formed-length default remains 7.5 m. Negative sampling/strip/reserve controls
fail for their intended reasons; recovery preserves the accepted checkpoint.
The retained final pilot disk has only **.0277772 m right-wall clearance** and is
reported explicitly. Its actual next-step and final checkpoint fields reproduce
bitwise; this does not establish wall independence or point-stress accuracy.

Three geometric unit controls, twelve fixed native controls, two optional retained
pilot checks and ten parameter validations pass. The fresh four-step smoke case
has scaled source <=4.321e−16 and weak <=2.360e−12 Pa; no measurable growth or
new physical-energy/propagation qualification is claimed. Original normal/point,
spatial, branch and path/R findings remain. Production/BP3/defaults and all
acceptance tolerances stay unchanged. No new scientific campaign is launched.

---

# Default-off native ASPECT cohesive–RSF pilot complete

[Report](COHESIVE_RSF_NATIVE_REPORT.md) · [equations/ownership](COHESIVE_RSF_NATIVE_FORMULATION.md) · [direct run inputs](../../../benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/README.md) · [PNG/GIF/indexed evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.

The actual native mixed solver completes 179 positive steps on two ranks,
.440–.462375 s, from the explicitly checked loaded curved state. General DGQ2
plus native FE remainder, current-pressure contact, temporary/formed ownership,
all six Gauss histories and one fracture inventory are native state. Two
native-admitted cells become fully decohered and slide further during the hold;
first post-full slip .047493 Dc, normalized age change −.023575, q=0.
Connected a=1 front advances 1.442594 m; added damage integral 2.100124 m.

Same-rank rejection/retry, first-admission rollback, exactly one publication,
formed-history restart and the final four-step maturation restart pass. Final
bulk physical records, geometry, all surface columns and reconstructed fields
are bitwise identical. Distorted-cell components and a general incompatible
Airy-history native case cover 1/2/4-rank ownership; cross-rank physical-record
differences <=1.64e−11. A supplied native restart works with prototype input absent.
All nine parameter files validate; 18 legacy unit cases / 1042 assertions pass.

Half-step final-window growth/slip differences are −.127322% / −.043615%, with
identical developed front brackets at 20 common times. Maximum scaled source
1.923e−13; mechanical weak 1.129e−10 Pa; surface weak 9.977e−9 Pa; BE ledger
1.574e−11 J/m. Separate absolute trapezoidal-work remainders are <=.237% of
sliding / .572% of fracture in each loading/hold window. Same-field quadrature
repartition defects are separately recorded. No tolerances changed.

Wall time 700.50 s; observed high-water RSS <=385.1 MB/rank. No MPI speedup or
full-event cost guarantee. The 7.5 m formed-length guard stopped a separate
longer attempt before publication; all original failures/checkpoints remain.
Old point/ahead normal-reaction, spatial, branch and boundary findings remain
unresolved. This finite-wall, fixed Cartesian 2D pilot does not qualify general
paths, topology, opening, transport, AMR, inertia or changed-rank restart.
The path hypothesis and R remain uncalibrated. Production/BP3/defaults are
protected. Next work is review/scalable integration and the stated mechanical/
scientific limitations, not an automatically launched campaign.

---

# Integrated adaptive propagation and mature sliding complete

[Report](COHESIVE_INTERFACE_PROPAGATION_REPORT.md) · [selection](COHESIVE_INTERFACE_PROPAGATION_SELECTION.md) · [PNG/GIF/indexed evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_propagation/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved work.

Qualified the opt-in parallel native bulk implementation on 1/2/4 ranks before
continuing the verified .440 s checkpoint on **two ranks** through original
loading to .455 and hold to .460 s. All 160 positive steps and 27 formed-registry
commits completed. The connected fully decohered front advances .497439→1.593763 m
beyond the seed; five whole cells (.46875 m) selected during the loading turn
become fully decohered. The first .1875 m band subsequently slips .141799 Dc,
changes age by 13.1714%, and has exactly zero cohesive resistance. This is actual
RSF sliding on newly selected curved material, not a tied-tail extension or
steady-state RSF. Initial histories and formed geometry are preserved.

Half-step controls at first maturation/hold and the final mature hold change
added damage -.13335/-.12773%, slip -.05489/-.04538%, with unchanged developed
front brackets. The mature slip/age metrics change +.01078/+.04578%. Continuing
the own finer .440 s branch through .457375 changes growth +.17947%, slip
+.07463%, and developed/process fronts -14.383/+14.360 mm. Later spatial error
and full bulk convergence are not established. Main absolute physical-work
remainder is .1931% of sliding/.5126% of fracture; loading and hold screens both
pass independently. BE ledger maximum 1.27e-12 J/m; weak residual <=9.98e-9 Pa.

**Normal/point limitations remain:** event fitted normal differences are
2.545% bend / 3.255% late-turn perturbations; raw sided traction remains
inaccurate (bend normal-jump RMS 422.8 versus 18.0 Pa). The same-field pressure
perturbation changes late velocity -.42164% and growth -.08074%, supporting
this bounded response without qualifying point stresses. Selection tensor and
angle differ .0004609%/.0001256 degrees. Old ahead-region, point-traction,
early spatial and branch failures are retained. No pressure lag, tolerance
change, new damping, law change or R calibration.

Native operators/bases/short histories agree bitwise across 1/2/4 ranks;
selector/diagnostic sums differ only measured roundoff from legacy serial.
Nonroot failure, unchanged accepted input, deterministic retry, once-only
publication and loaded JSON restart pass. The single maintained driver is
`cohesive_interface_propagation.py`. Main wall time 1759 s; final 1/2/4-rank
export times 8.93/10.11/12.01 s: **no robust local MPI speedup**. Mesh/matrix
remain replicated, expensive columns/cell quadrature are parallel, reduced
history ownership coordinated. This is not production Simulator/MPI integration.

Next work is the concrete default-off ASPECT integration described in the
report: native jump/contact/pressure blocks, actual general Maxwell tensor
history, temporary/formed registry, single fracture inventory and native
outer-loop/restart/MPI ownership. Current homogeneous response-basis history
must not silently replace arbitrary DG/native history. The finite-wall result
does not qualify free topology, opening, inertia or physical rupture speed.
No further campaign is launched. Production/BP3/defaults and prior evidence are
preserved; the complete previous status follows unchanged.

---

# Short full loading turn with future-material refinement complete

[Report](COHESIVE_INTERFACE_FORMED_TURN_REPORT.md) · [declared controls](COHESIVE_INTERFACE_FORMED_TURN_SELECTION.md) · [PNG/GIF/CSV evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_formed_turn/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.

Both native trajectories resume the same verified .425 s half-step checkpoint,
preserve its ten formed pieces and 132 Gauss identities, and complete the
physical loading turn at .440 s with dt=.000125 s. Only future commit spacing
changes .09375→.046875 m; R=.75 m, resolved support .75 m, bulk/constitutive
settings and all tolerances stay fixed. Existing formed history is never remapped.

At .440 s, total incremental damage changes -.0586%; damage and slip confined
to new material change -.9317%. Maximum common-time increment differences are
.8533% damage / .2835% slip. Final process-front estimate differs 3.82 mm with
overlapping Gauss brackets; developed fronts share the same bracket at about
.497 m beyond the seed. Six/twelve new cells form the same .5625 m length.
They remain a partial cohesive zone (peak a≈.25); the developed front is still
on older geometry, so adaptive fully decohered deflection is not established.

Both energy screens pass: physical absolute remainder .0962% of sliding /
.409% of fracture. Late .4375–.440 s timestep halving changes growth -.2128%
and slip -.0342%, keeping both front brackets. Small residual spacing effects
are comparable to temporal uncertainty; two levels do not establish an order.
Actual contact rejection, immutable histories, deterministic retry, once-only
publication and loaded rebuilt-context JSON restart pass for both spacings.
Coupled pressure FD errors <=5.40e-10; transfer stress/storage defects are zero.

Common-field Cartesian64/fitted64/fitted96 checks support the R-disk tensor
(.000710%, .000203°) and regional weak shear (<=1.03%). **Bend normal-work target
remains reference-sensitive:** 1.937% versus fitted96, 2.006% versus fitted64;
the retained 2% criterion is not robustly established. Raw normal-jump RMS ~494
versus 15.2 Pa fitted remains unqualified. All historical point-traction and
ahead/old-region failures remain open. Production/BP3/defaults and old evidence
are unchanged. Serial experimental native/JSON scope, not Simulator/MPI or
quantitative propagation qualification.

Next: one frozen bend normal-work uncertainty control before a longer
quantitative directional claim. A later bounded catch-up/hold experiment could
then test developed advance through newly selected material. Neither is started.

---

# Temporary candidates and energy-accounted formation qualified in the bounded native experiment

[Report](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_REPORT.md) · [equations/ownership](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_FORMULATION.md) · [PNG/GIF/indexed evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_candidate_admission/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.

Unused zero-history candidate geometry is now distinct from the persistent
formed registry. A new Q1 cell commits only with physical admission; all old
formed geometry, evolved Maxwell fields and six Gauss histories remain intact.
Candidates with any physical history or nodal memory, even 1e-30, cannot retire.
An optional exact Q1 master space prevents new numerical SVD shadows, without
zeroing incoming history or changing default behavior.

The same accepted-stress R=.75 m selection/persistence rule is anchored at the
formed endpoint, explicitly removing unused reserve from the sample location.
New direction commits only after a joint native solve from that actual incoming
accepted state; stale-direction predictors are rejected before publication.
Metadata adds no fracture/storage/work. Intact-overloaded candidates cannot
publish as tied births: the old guard still rejects. A positive-time joint
admission from the original late failed state pays .000317655 J/m new fracture
and .000316220 J/m cohesive storage, with 5.99e-9 Pa weak defect.

In .410–.425 s, 2 versus 8 temporary cells give identical persistent geometry
and IDs, with history differences <=6.67e-16 and stress/pressure component
difference 1.36e-12 Pa. Both admit eight cells in 60 physical publications.
Timestep halving changes measured growth -.5445%; refresh-cadence halving gives
no response change in both onset (.410–.41675) and .420–.425 s controls. The coarse physical fracture-work
screen FAILS at 1.4757% (suffix 1.6894%); half stepping PASSES at .7371%. The
1% criterion and all native tolerances are unchanged. This is partial cohesive
formation; no new fully decohered advance is qualified in this window.

Actual native contact rejection, residual-valid stale-geometry rejection,
unchanged accepted input, deterministic retry, one atomic publication, false
zero-cell commit rejection, active-history JSON restart and candidate completion
pass. Exact endpoint/persistent fracture and cohesive inventories agree; no
Gc times newly registered length is added. Coupled FD 6.27e-10, independent Q1
projector 4.54e-16; active-admission FD 5.52e-10; eight default native tables are bitwise identical.

Production/BP3, legacy checkpoints and prior evidence remain protected. Old
point-traction and ahead/old-region normal-reaction failures remain open. This
is serial fixed-grid homogeneous native/JSON ownership, not production MPI,
AMR, transport, general topology, a global RSF stability theorem or quantitative
adaptive propagation. The adopted one-cell formation extent remains numerical.

Next: one new-material cell-spacing control and a short continuation through
more of the loading turn at the qualified half step, retaining old material,
physical R/support and fitted weak-load checks. Do not start it automatically.

---

# Boundary-driven causal extension: bounded experiment complete

[Report](COHESIVE_INTERFACE_LOAD_TURN_REPORT.md) · [inputs/gates](COHESIVE_INTERFACE_LOAD_TURN_SELECTION.md) · [PNG/GIF/indexed data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_load_turn/INDEX.md).
Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.

A gradual physical boundary-strain turn from 0 to −20° over .410–.440 s changes
the accepted maximum-shear preference from ~20° to ~.5°. Native mixed mechanics
and both fixed-laboratory Maxwell components advance normally; no stress,
damaged geometry or history is rotated/reset. New material selected during
the turn reaches actual damage .783/.325 on successive appends, with integrated
damage .2083/.0714 m. The half-quantum control includes .0676 m weighted saturated
records on an append selected during the turn. This establishes a bounded
stress-responsive formation response, not reserve-independent adaptive propagation.

Physical R=.75 m and resolved numerical support w=.75 m stay fixed. At common
.445 s, S=.1875 versus .75 m changes new damage by 10.94% and process-front
position by .2244 m. Fresh bulk refinement changes damage 1.22%; timestep halving
.456%. Halving future append quanta changes damage 5.77% and process front
.1586 m near its endpoint: scheduling is not negligible. Virgin committed
directions remain permanently frozen. No change to that assumption is adopted.

Short-reserve runs stop before inadmissible tied birth (3065.4 Pa original,
3083.9 Pa fine, 3066.8 Pa half-step, 3081.5 Pa half-quantum; strength 3000 Pa).
Accepted states and rejected candidates are retained. Turned large-reserve run
completes .460 s, including hold; developed advance 1.0020 m. Unchanged large
control reaches .455 s and the declared geometry capacity, not physical arrest.
Primary developed fronts still occupy earlier reserved geometry; the small
half-quantum developed response remains scheduling-sensitive.

Matched-field fitted controls support the selection-disk tensor and weak shear:
64² tensor discrepancy .000370%, angle .000104° versus fitted96; 96² regional
shear differences ≤.75%. **Point traction remains unqualified:** current raw
normal-jump RMS ~520 Pa Cartesian versus 16.6 Pa fitted. Historical ahead-region
normal failure 2.85%/2.29 Pa remains. New old-region relative weak normal
perturbation error is 3.56% (.118 Pa); not hidden by confinement normalization.

Default identity, full pressure tangent FD (7.60e−10), actual native contact-loss
rejection, deterministic retry, once-only publication and rebuilt-context JSON
restart pass within the isolated serial experiment. Old physical fields/history
and storage are unchanged at append. Physical work remainder is .149% of
sliding/.670% of added fracture for turned-large. Half-quantum suffix narrowly
**fails** the retained fracture screen at 1.000894%; no tolerance change.
Production/BP3 and all earlier evidence remain protected.

Next decision: distinguish temporary zero-damage/zero-slip/zero-memory candidate
geometry from permanently committed formed geometry, with explicit energy,
admission and basis/history preservation. This would revise permanent virgin
locking, not tune S or rewrite damaged material. It is proposed only; no longer
run, R calibration, constitutive change or production integration is started.

---

# Fixed-path bulk/contact correction and verification completed

[Report](COHESIVE_INTERFACE_BULK_CONTACT_REPORT.md) · [controls/gates](COHESIVE_INTERFACE_BULK_CONTACT_SELECTION.md) · [six PNGs/indexed data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_bulk_contact/INDEX.md).
Unresolved gradient kinks at the arbitrary lifting support add artificial
stiffness. A pressure-free 1D FE/projection calculation predicts the actual
old-region load change within .047 Pa (.037%). Accurate quadrature and a normal
work multiplier alone cannot fix this missing bulk approximation freedom.

A fixed, resolved numerical lifting/normal-test half-width .75 m is selected
explicitly for fresh experimental runs. It is independent of physical R=.75 m,
which remains uncalibrated and unchanged. Same-field frozen Q2/Q3, independent
fitted meshes, an off-grid support control and full coupled return checks
support this numerical correction. No physical parameter/tolerance changes.
The tied-ahead 2% weak normal-perturbation gate still fails (2.85%, 2.29 Pa).
Raw cut-cell normal jumps remain ~410 Pa in the developed-region test, versus
17 Pa on the fitted reference; no point-traction qualification is inferred.

Imported common-history replays are explicitly diagnostics: independent bulk
reconstruction leaves a free surface-force defect and a measured storage change.
Four fresh equilibrated common-path controls remove that ambiguity: wide Q2
mesh growth spread is 1.01%, wide96/fitted96 difference is 1.54%, fitted mesh spread is .104%.
Late time uncertainty is ~.4%; smaller fitted differences lie within it.

Fresh causal runs retain all four insertions/1780 physical publications, old
physical stress/pressure columns and six histories bitwise, with zero measured
transfer stress/storage defect. Corrected Q2 growth spread is 1.01% (old 9.75%);
fine developed advance .43508m along Γ, tangent 8.84 degrees, full-front mesh difference
.01213 m. Physical work error .2495% of sliding/.6047% of added fracture.
Actual native rejection/retry, same-build JSON restart and duplicate-publication
checks pass in the short controls. This remains isolated serial native code,
not production owner/checkpoint integration or a global RSF stability theorem.

Cleanup gross recovery of 2.49 GiB is logged under maintenance/disk_cleanup_2026_10_27:
125 raw tables losslessly compressed, two rebuildable PCH caches and three fully
redundant ZIPs removed. No accepted histories, unique historical evidence or BP3
files were discarded. New task raw output ~.58 GiB; compact evidence has no ZIP.
Production/BP3/default paths remain unchanged. Historical failed spatial/onset,
branch, energy and Q2-admission evidence follows intact. Retain fitted reference
and the explicit fixed-support route; defer R calibration and another long run.

Full earlier status follows verbatim.

---

# Causal selected extensions completed within bounded scope

[Report](COHESIVE_INTERFACE_PATH_REPORT.md) · [equations and retained corrections](COHESIVE_INTERFACE_PATH_SELECTION.md) · [PNG/GIF/indexed evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_path/INDEX.md).
Accepted-state incipient-shear work replaces future-endpoint damage scoring.
The homogeneous angles are compression −25°, maximum shear +20°/−70°;
μ=.1 Coulomb angles are diagnostic only. Physical R=.75 m, directional
persistence and tied reserve/freezing are explicit uncalibrated hypotheses.
Conjugate ties use the existing structural preference or remain unresolved.
The intact law and single fracture inventory are unchanged.

Four causal insertions preserve the old geometry and entire native Maxwell
stress span, including evolved nonuniform memory. Old stress/storage changes
at zero-time birth are zero. Actual rejection, deterministic retry, once-only
publication and rebuilt-context JSON restart pass in the isolated native
benchmark owner; production Simulator/Manager integration is not claimed.
A derived closed-normal-bubble mixed reaction supplies current friction
pressure. Raw sided Cauchy tractions remain visible and mismatched (~226 Pa RMS).
Derivative-consistent geometry interpolation fixes the demonstrated extension-
step artifact; all original source/data/errors are retained.

The primary new fully decohered front advances .24147 m along Γ and turns
5.54°; the a=.05 process front turns 18.01°, distinct from the prospective
19.82° terminal tangent. Half extension/surface/late timestep changes in new
damage are .317%/.097%/.323%. Bulk refinement and alternative Q3 change it
9.75%/17.35%: the predeclared quantitative spatial target FAILS. Physical work
error is .1725% of sliding/.4564% of added fracture, using absolute per-step
remainders. All six final controls complete with existing tolerances.

This demonstrates a short history-preserving selected extension under the
adopted rule, not a calibrated or quantitatively qualified free-path law.
Next useful work is a fixed-path mixed bulk/contact approximation diagnosis
of the saved Q2/Q3 spread; R calibration and tied-reserve assumptions remain
physical decisions. No further campaign starts automatically. BP3, defaults,
legacy point-pressure mode and all previous evidence are unchanged.
Full prior status follows verbatim.


---

# Resolved turning and conditional direction selection completed

[Report](COHESIVE_INTERFACE_TURN_REPORT.md) · [formulation/selection](COHESIVE_INTERFACE_TURN_SELECTION.md) · [PNG/GIF/indexed data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_turn/INDEX.md).
Native vector Q2/P1disc curved mechanics retains pointwise closed normal
contact, full tangent-gradient terms, current sided pressure and the existing
cohesive–RSF/radiation law. A mature smooth Γ sustains 3.84–5.11 mm/s through
two resolved turn widths. Spatial/temporal integrated-slip changes are
.85%/.0101%. Small-strain curvature/slip ratio is .00221; finite sliding and
the zero-radius sharp-corner limit remain unqualified.

The bounded finite-preview rule selects a +30° terminal candidate from its
1.5 m-width family: fine half-step score gap .03956 exceeds the declared
.02360 uncertainty test. New cohesive damage spans 18 resolved cells; four
whole cells fully decohere, advancing .37533 m along Γ. Damage-weighted
direction is 12.27°, while the full front has turned only about 5°.
At 3 m width the best sampled terminal direction is the +60° sector boundary.
Thus a conditional computed deflection is demonstrated; a geometry-independent
physical angle is not. Turn geometry and preview horizon are explicit extra
path assumptions, not consequences of current-pressure friction.

The old loaded straight geometry and all six histories are preserved at tied
birth, including nonzero Maxwell stress, permanent slip and nonuniform age.
Selected native recomputation and rebuilt-context JSON restart are bitwise;
one geometry publication and exactly 1800 physical publications occur.
Physical work remainder is .114% of sliding/.293% of added fracture, using
absolute per-step errors; algebraic ledger closure is reported separately.
These are isolated benchmark/native operators, not production-owner integration.

Qualified evolution uses the retained Q1 incremental jump trace with its
full-Q2 permanent offset. The new full-Q2 first-admission line-search failure
and undersized pilot/work failures remain unresolved and preserved. Historical
65.4% early spatial sensitivity, onset and boundary findings remain visible.
Production/BP3/default history settings and previous evidence are unchanged.
The next decision is the physical path-admission/turn-length and preview rule;
no further run, inertia qualification, or model adoption starts automatically.
Full earlier evidence follows verbatim.

---

# Focused surface/jump comparison completed: developed response passes, onset remains sensitive

[Report](COHESIVE_INTERFACE_SURFACE_REPORT.md) · [selection](COHESIVE_INTERFACE_SURFACE_SELECTION.md) · [PNG/GIF/CSV evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_surface/INDEX.md).
Fresh 256/512 Q1 surfaces use the same exact mixed-strip bulk, finite L16/H1
specimen, seed, current-pressure cohesive–RSF and native half-step clock sequence.
No nonlinear histories are transferred. Initial jump/inventory differences are zero.
A focused 512-cell time halving starts from its own 2.42375 s checkpoint.

At 2.58, surface refinement changes growth .0284%, slip .0217% and full left
advance .836 mm; corresponding fine temporal changes are .0322%, .00500% and
.874 mm. Right full advance changes .507 mm (.518%), versus .00363 mm temporally.
Maximum developed-time integral-growth change .1003%, full-left-front change
1.163 mm. Declared developed spatial/time/work targets pass. This establishes
no convergence order or continuum error estimate. Initial new-damage brackets
shift [1.4800,1.4825]→[1.4075,1.4100] s; precise initiation remains unqualified.
Preserve the historical 65.4% early spatial failure and all boundary/speed limits.

Frozen native steps match histories <3.75e−14, shear <4.24e−10 Pa; complete
pressure-coupled derivatives ≤5.14e−9 at the retained 1e−6 gate. The preserved
fine first-hold predictor stop recovers with only a zero-increment guess;
complete signed residual/tangent checks still gate acceptance and hold damage
is nonzero. All physical histories remain unchanged before acceptance.
Independent NPZ round-trip/retry is bitwise on both meshes. This is reference
lifecycle evidence, not additional native/production checkpoint qualification.

Fine work errors halve to .163% of new fracture/.126% of sliding; hold-only
.191%/.137%, with zero hold work and no cancellation. One fracture inventory,
positive contact and retained admissible tangent; no law/tolerance changes.
Evolution cost 116.2 s total, ~137 MiB peak, sequential one-thread references.
Production/BP3 and old archives retain entry hashes. Bulk inertia and precise
initial-tip admission remain the next scientific questions; no further run starts.
Full previous evidence follows verbatim.

---

<!-- CI_BOUNDARY_CURRENT_BEGIN -->
# Boundary mechanism and longer finite-strip native experiment completed

[Report](COHESIVE_INTERFACE_BOUNDARY_REPORT.md) · [selection](COHESIVE_INTERFACE_BOUNDARY_SELECTION.md) · [PNG/GIF/CSV evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_boundary/INDEX.md).
The mixed-strip reference explains normal-pressure period sensitivity by finite-wall
incompressibility: N(k)≈−i6G(t−b)/(H³k), with a distinct zero/mean mode. The same-memory
8→16 m normal change (635 Pa RMS) is predicted to .020 Pa RMS. Physical p0 remains
10000 Pa; no pressure shift/lag is introduced. Native proper rotation passes
(damage 2.14e−14, bulk stress 2.49e−10 Pa), swapping the pressure/developed flank.

The original off-centre **finite-height periodic** L=16/H=1 specimen reached
2.58 s after continued loading to 2.55 and .03 s hold. Refined full left advance
is **3.2135 m**, with **51 complete new fine cells** and ~16-cell cohesive zone;
right full advance .09737 m. No detached a≥.01 growth on Γ, positive contact.
Halving the new-window timestep changes growth **.0666%**, integrated slip
**.0107%**, and full front **1.71 mm**; bulk tensor .0342%, normal pressure
.736 Pa RMS. Frozen-memory continuum mechanics gives ~1.10 Pa shear-load/.00210 Pa
projected-normal RMS tip coupling errors. This is distinct from surface-history spatial accuracy.

All selected temporal/work targets pass. Raw new-window errors halve to .325%
of new fracture/.251% of sliding; hold-only .382%/.274%. Hold growth/sliding
use stored mechanical energy, with zero imposed work. Single fracture inventory
and all constitutive/history/pressure conventions remain. The 1500 s wall stop,
input-string rejection and valid completion are preserved. Replayed checkpoint
overlap and the pre-extension past are bitwise; a post-hold loading-plan change
is rejected before publication. Native publication counters advance once per accepted state.

This is a boundary-defined finite specimen, **not distant-boundary or physical
rupture-speed qualification**. Retain the 65.4% early spatial failure and old
period/branch/energy findings. The quasi-static model's hold-front secant is
28.95 m/s versus nominal 19.35 m/s used in radiation; this validates no wave
causality/supershear physics. Next: focused surface/jump spatial reference for
the new front, and an explicit bulk-dynamics decision before physical-speed claims.

Native commands cost ~78.6 min total, ~2120 MiB observed peak; repeated setup and
complete derivative checks dominate. Production/BP3, 1,051 protected files and
11 prior archives retain entry hashes. No further loading/free path/bend/MPI/AMR
or transport was started. Full earlier evidence follows verbatim.
<!-- CI_BOUNDARY_CURRENT_END -->

---

# Enlarged-period control completed: domain sensitivity remains

[Report](COHESIVE_INTERFACE_DOMAIN_REPORT.md) ·
[selection](COHESIVE_INTERFACE_DOMAIN_SELECTION.md) ·
[PNG/GIF/CSV evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_domain/INDEX.md).
Fresh 8×1 and 16×1 m native exact-jump strips retain the physical seed,
Δx=.0625 m, Δy=.0125 m, law, current-pressure coupling and loading/hold to
2.47 s. All 1,057 accepted times match bitwise. Added Γ records remain exactly
tied and contribute no new fracture/sliding work. Initialization is matched
fresh physical data, without projection or transferred histories.

At 2.47 s, enlargement reduces new fracture **11.92%** and integrated slip
**6.11%**. Left full advance changes .752326→.730207 m; the right remains a
cohesive zone. Near-tip full-tensor difference is **2.53%**, normal-pressure
RMS **461.5 Pa** (common Γ: 583.4 Pa). Growth/slip, stress and pressure targets
fail; the .05 m full-front target passes. Frozen identical-history maps also
show a much larger normal-response than shear-response change. This is not
distant-boundary qualification; height and wall distance remain fixed.

Both raw physical-work screens pass, including hold alone. Whole-run errors
are .453/.486% of new fracture and .154/.155% of sliding. The original enlarged
attempt's derivative stop at 1.9425 s is preserved: sub-1e−14 m perturbations
made double residual subtraction roundoff dominate. An opt-in, independent
extended-precision reference passes the **same 1e−6 gate and perturbations**;
50-digit checks and biased-tangent rejection pass. No physical solve/tolerance
changes. Restarted overlap is bitwise; rejection/retry and once-only publication
pass. Original failed records/checkpoints are retained.

Fresh 8 m cost: 270 s/920 MiB observed peak. Enlarged failed attempt plus valid
restart: 1439+860 s, ~2121 MiB. Production/BP3, 1,051 protected source files and
previous archives retain their hashes. Preserve the **65.4% early spatial
failure**, onset/branch limits and earlier coarse-timestep hold-energy failure.
Before a longer quantitative run, diagnose period-dependent normal loads with
the inexpensive mixed reference and select a focused boundary/specimen control.
No longer event, free path, MPI, transport or further campaign starts here.

---

# Native coupled evolution and developed straight propagation completed

[Report](COHESIVE_INTERFACE_EVOLUTION_REPORT.md) ·
[selection](COHESIVE_INTERFACE_EVOLUTION_SELECTION.md) ·
[figures/CSV/GIF](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_evolution/INDEX.md).
The native C++ component computes its own mixed mechanical/admission/RSF states,
using full current pressure and the existing point law. The oblique reference
matches all six Gauss histories to 1.11e−15; native insertion, rejection/retry,
restart and once-only publication pass. Actual loaded cutback matches its
uninterrupted two-half-step control bitwise. The production/BP3 paths are unchanged.

The straight 8×1 m strip grows a new fully decohered left region about .75 m
beyond its mature seed by 2.47 s; twelve complete fine cells reach a=1. The left
cohesive zone spans about 14 fine cells. Loading ends at 2.45 s; .02 s hold is
verified. Candidate-buffer guards stop longer exploratory loading/hold attempts
before publication; those trials/checkpoints are retained. This is fixed Γ,
not selected free-path propagation; phi remains provisional geometry.

At equal dt, late spatial integrated-growth change is .136%; focused time
halving changes growth .330%, full front 3.08 mm and window sliding loss .645%.
Whole-run fine absolute physical error is .866% of fracture/.295% of sliding.
**Baseline hold-only screens fail** (1.260%/1.064%); half-step hold passes
(.633%/.537%). Preserve the **65.4% early spatial failure**, first-onset timing
uncertainty and finite-boundary influence. No quantitative general speed claim.

Fine prefix/restart cost 210/115 s; half-step window 146 s. Native evolution is
an isolated FE/Schur component in the existing scheduler/checkpoint framework;
production nonlinear integration, MPI/AMR/transport, bends and automatic path
creation remain later work. Recommend one enlarged-domain/tied-buffer control
at unchanged near-tip resolution/loading before a longer trajectory. No next
scientific task starts automatically. Previous evidence follows verbatim.

---

# Activated-append temporal work qualification completed

[Report](COHESIVE_INTERFACE_ENERGY_REPORT.md) ·
[selected scope](COHESIVE_INTERFACE_ENERGY_SELECTION.md).
The unchanged fixed-spatial component problem passes both 1% raw physical-work
screens at Δt=1.953125e−5 s: absolute remainder 3.13557e−7 J/m, .5070% of
sliding loss and .005578% of fracture. A .58–.66 s half-step replay changes
sliding loss .01024%, integrated damage/slip .001528%, and halves window energy
error. Full/replay costs: 106.51/46.39 s. Existing checks and reference
rejection/retry/restart/once-only publication pass without tolerance changes.

Independent decomposition identifies Maxwell time error as the dominant prior
failure. An exact tied control checks individual work, dissipation and response;
no numerical term is subtracted to pass. Preserve the original .005/.0025 s
failures below. Production/BP3 and the admission law/solver remain unchanged.
This is a native-matrix independent evolution reference, not a new full native
outer solve. The 65.4% spatial failure, closed corner, provisional phi and lack
of free-path/fully-decohered propagation remain. Next recommendation: integrate
signed admission in the native outer time solve on this same small fixture,
using these qualified references and existing lifecycle infrastructure. No
longer run or new task starts automatically.

---

# Prescribed oblique admission completed; sliding-scale energy screen remains failed

[Report](COHESIVE_INTERFACE_ADMISSION_REPORT.md) ·
[scope/equations](COHESIVE_INTERFACE_ADMISSION_SELECTION.md).
The fitted candidate is inserted tied at .25 s (2478.78 Pa required, below
3000 Pa) and evolves with the unchanged signed cohesive–RSF return after discrete
tied loss near .58144 s. The complete physical trajectory matches preinstallation
bitwise. General Gauss tie reduction, full pressure derivatives, weak equilibrium,
reference rejection/retry and native history publication/checkpoint continuation
pass. Default-off replay leaves the previous native owner bitwise unchanged.

At .66 s the fine case has peak a=.0269453 and added fracture .00560302 J/m;
no a=1 region develops. This is broad activation of a prescribed append, not a
selected free path or tip-localized propagation. Halving dt changes final
integrated damage/slip by .249%; first-activation timing is not generally qualified.
The cumulative absolute physical-work remainder halves to 4.00062e−5 J/m:
.714% of fracture (pass), but 62.97% of sliding loss (**fail**). An analytic
homogeneous Maxwell control accounts for about 95% of its magnitude; this is
not subtracted from the failed acceptance error. BE closure is separate.

Phi remains provisional geometry; fracture is counted once on Gamma. The closed
corner, fixed mesh/pressure space and prescribed-path restrictions remain. No
production/BP3 change, broad convergence campaign or free-path progression was
made. The earlier 65.4% spatial failure remains. Next: a bounded temporal
work/Maxwell-dissipation consistency check on this same activated append before
quantitative sliding-energy or longer propagation claims. No next task starts
automatically. Previous status follows unchanged.

---

# Non-collinear tied admission and native-owner transfer completed

[Report](COHESIVE_INTERFACE_KINK_REPORT.md) ·
[design/transfer contract](COHESIVE_INTERFACE_KINK_FORMULATION.md).
A prescribed 26.565° append is represented by actual vector jump modes and
independent pressure traces on a fitted native Q2/P1disc mesh. Loaded/unloaded
tied insertion preserves full physical stress, old Gauss histories and all
storage exactly. The separate native manager/checkpoint adapter passes atomic
rollback/retry, single geometry publication and bitwise ASPECT restart. A fresh
axial stress state leaves the old horizontal plane admissible but requires
4000 Pa on the append; tied insertion rejects against the unchanged 3000 Pa
strength, with no new damage or energy.

A focused 16×8/32×16 common-radius tensor diagnostic differs by .9345% and
.09622° in principal axis. The local tied multiplier still changes 27.3%; precise
activation and pointwise tip stress remain unqualified. The imposed kink is not
a selected maximum-shear path. The revised straight-growth comparison has an
early inset, absolute/relative differences and common 2.42 s endpoint; the
**65.4% / .0055521 m early spatial failure remains**.

This is an isolated fitted-mechanics/native-owner control, not a production outer
solver transaction, pressure-space remap, arbitrary DG transfer, MPI or free-path
propagation qualification. Continuous traces and zero opening on both branches
force zero vector jump at the corner; unrestricted sliding through a kink remains
outside scope. Phi stays provisional geometry; surface a counts Gc once.
Production/BP3 and previous evidence are preserved. Next recommended: event-resolved
admission on a prespecified oblique candidate inserted while still tied. Any
freely sliding-corner extension first needs an explicit normal-contact/kinematic
choice. No next task begins automatically. Prior status follows unchanged.

---

# Finite exact-jump insertion and localized straight growth completed

[Report](COHESIVE_INTERFACE_INSERTION_REPORT.md) ·
[transfer formulation](COHESIVE_INTERFACE_INSERTION_FORMULATION.md).
Finite tied-record birth in a loaded state preserves physical stress, old Gauss
histories and bulk/fracture inventory exactly. Different insertion times/append
lengths and tied buffers match the preinstalled control bitwise. Native geometry
rollback/retry, old-geometry rejection, once-only publication and restart pass.
The fixed bulk pressure approximation is retained; its remapping is not qualified.

The fresh 8 m × 1 m component specimen develops connected cohesive growth and
leftward full decohesion. At 2.42 s the finest continuous jump-threshold front
advances .24292 m (at least three complete fine cells); rightward growth remains
cohesive. Late integrated-growth differences are .399% spatial and .722% temporal;
front differences are 3.81/5.20 mm. Physical work error is .559% of new fracture
and .129% of sliding loss after halving. The full-interval spatial target fails: early integrated growth differs by
65.4% at 1.8 s, and first admission remains spatially sensitive. The coarse run stops at the declared candidate-end buffer; no endpoint
is moved to evade it. Negative unrestricted release comes from cohesive softening
in tied directions; admitted curvatures remain positive. Remote physical-stress
checks at saved states remain below intact strength.

Phi remains geometric; a owns the single fracture inventory. This is serial
native-component evidence, not free-path, production, MPI, arbitrary-history or
changing-pressure-space qualification. Production/BP3, calibration and earlier
failures remain unchanged. Next: bounded path-admission/topology design and a
non-collinear transfer control before free-path evolution. No next test starts
automatically. Previous status follows unchanged.

---

# Weak-front admission and bounded native seeded growth completed

[Report](COHESIVE_INTERFACE_FRONT_REPORT.md),
[weak formulation](COHESIVE_INTERFACE_FRONT_FORMULATION.md),
[architecture proposal](COHESIVE_INTERFACE_ARCHITECTURE_PROPOSAL.md).
The exact-jump fixture now certifies intact weak equilibrium with bounded Gauss
multipliers and exactly tied jumps, rather than a nodal stress cutoff. Independent
LP/history checks, full-pressure tangent, activating-step owner rejection/retry,
once-only publication and real ASPECT restart pass; restart is bitwise identical.

Fresh 32/64-segment native cases with a timestep halving complete loading through
.84 s and a hold to .86 s. The fine admission bracket is [.205,.206] s, but new
full decohesion starts only in [.823,.824] s. The candidate line is fully admitted
by .368 s and fully decohered by .826 s. This is broad seed-connected cohesive-zone
expansion followed by near-uniform completion, not isolated free-path rupture.
No negative incremental curvature or disconnected a>=.01 growth occurred.

Early spatial onset and peak growth remain underresolved (18 ms onset shift;
45.5% peak-growth difference at .24 s). Later common-box growth differences are
2.04% at .36 s and .687% at .40 s; integrated differences are smaller. Physical
work error halves to .123% of sliding loss/.0929% of added fracture energy.
The proposed roles of phi/a/Gamma count fracture energy once; maximum-shear
virgin-plane admission and progressive contact activation remain explicit.
That architecture is a proposal, not an enabled bulk-phi/free-path solver.

Production/BP3, calibration and earlier failures/evidence are unchanged. Owner
rejection is not production outer-cutback qualification. Next recommended: a
bounded tied-record insertion/extension with explicit state/energy transfer before
free-path growth; an isolated moving process zone also needs a different physical
specimen. No further experiment is started automatically. Prior status follows.

---

# Exact-jump fixed-surface native test completed

[Report](COHESIVE_INTERFACE_EXACT_REPORT.md),
[equations](COHESIVE_INTERFACE_EXACT_FORMULATION.md),
[selection](COHESIVE_INTERFACE_EXACT_SELECTION.md). The optional native fixture
now retains the full displacement-jump enrichment and divergence/pressure work,
with separate pressure traces. The prior 7.42% width bias is removed: the resolved
1 m mode has 0.001535% shear and 0.00002027% normal-coupling error on 32×160.
Changing lift width changes K/N by 8.9e−16/1.1e−13 relative. Actual two-side trace
mismatch remains reported as FE error, separate from exact generalized-work checks.

Uniform intact admission, mature sliding/hold/reversal through 7.260 Dc,
nonuniform current-pressure derivatives, rejection/retry/once-only commit and
real ASPECT restart pass. Restart histories/rates are bitwise identical; changed
representation mode is rejected before publication. Constitutive law, production
sources/defaults and BP3 remain unchanged. Earlier failures/evidence are retained.

Scope remains isolated native components, not production Stokes/phase integration.
The .125 m mode still has 4.81% shear error, and initiation's small sliding-energy
component retains 2.28% temporal error under the tested halving. Nonuniform partially
tied fronts, physical tips, MPI, arbitrary history conversion and propagation
are unqualified. Next: one bounded intact-to-damaged front on the same fixed
surface, checking weak admission/KKT and history ownership before tip growth.
No automatic follow-on test is started. Previous status follows unchanged.

---

# Native fixed-Γ component qualification completed

[Report](COHESIVE_INTERFACE_NATIVE_REPORT.md),
[formulation](COHESIVE_INTERFACE_NATIVE_FORMULATION.md),
[selection](COHESIVE_INTERFACE_NATIVE_SELECTION.md). The new opt-in C++ return
and native Q2/Q1/Γ fixture verify intact admission, current-pressure derivatives,
mature velocity-step/hold/reversal through 7.260 Dc, and real simulator restart.
Histories, signed velocities and publication sequences restart bitwise; invalid
trials and changed-width restart are rejected before publication. Native force
residuals remain below 1e−8 Pa. Production/BP3 and prior evidence are unchanged.

An initial nodal constitutive-integration checkerboard was diagnosed and retained;
consistent Gauss3 surface histories remove it in the tested cases. Prepublication
incremental-monotonicity checks use the actual condensed nonsymmetric tangent.
This is not a global stability certificate or the old phase-potential Hessian.

**Scope:** isolated native component assembly and native manager/checkpoint owners,
not production Stokes/phase integration or arbitrary DG-history conversion.
Nonuniform finite-width mechanics remains 7.42% below the sharp-interface shear
stiffness; width halving reduces this to 3.99%. Uniform sliding success does not
qualify the sharp nonuniform limit. Formation work error halves, but remains
2.28% of the small sliding-loss component at the finer timestep.

Next recommended task: one bounded exact-jump fixed-surface realization with
consistent pressure/divergence work and signed-rate contract, checked against
the same sharp mixed-layer reference. No propagation, MPI or geometry extension
is authorized automatically. Earlier limitations and evidence follow unchanged.

---

# Cohesive compliance / geometric-interface RSF reference completed

[Report](COHESIVE_INTERFACE_REPORT.md), [formulation](COHESIVE_INTERFACE_FORMULATION.md),
[selection](COHESIVE_INTERFACE_SELECTION.md). The user-selected separation is
implemented in independent references: exact intact strength admission with
recoverable fracture compliance; one geometric interface velocity/age and zero
residual cohesive resistance at full decohesion. Surface damage is a new variable,
not the old bulk phi energy. Gc is counted once. Loaded tensor history mapping
preserves stress and exposes any required model-switch work instead of resetting
history. Arbitrary heterogeneous old ages/native checkpoint conversion remain open.

Intact onset converges to .375704888 s. Mature sliding reaches 7.261 Dc with
physical-work error .008553% of sliding loss. The seeded mixed-layer test produces
connected low-damage extension, not fully developed crack advance. Focused
256→512 refinement gives 1.03% integrated-growth difference; the original 3.94%
coarse failure is retained. First-cell onset and peak remain spatially sensitive.
Current-pressure tangents, mixed symbols, energy separation and independent-owner
rejection/retry/restart/publication are verified within their stated scope.

Next: bounded native geometric-interface realization and Gamma work/trace
qualification; no automatic large propagation. Linear contact activation, the
material age clock, fixed-reference strength and path-restricted formation are
explicit assumptions. Production/BP3 and earlier failed accuracy evidence are
unchanged. Previous status follows.

---

# Substantial structural-plane sliding: fixed-phase evidence; birth remains open

[Report](STRUCTURAL_SLIDING_REPORT.md), [derivation](STRUCTURAL_SLIDING_FORMULATION.md),
[criteria](STRUCTURAL_SLIDING_SELECTION.md). Independent experiments reach
5.09/5.72 Dc signed slip by the loading endpoint with developed/mature-center
profiles, velocity change, hold and reversal. Focused 512-cell and timestep
checks resolve the declared criteria; physical energy errors are also small
relative to sliding dissipation. The distributed response is not identical
to one interface RSF age/speed; local concentration and cohesive flanks explain
measured differences. Full mixed QZ/FD checks and nonlinear oblique pressure
experiments converge while retaining bounded physical growth. That is not an
unbounded short-wave failure or a solver rejection criterion.

Phase progression stops at a demonstrated admission gap: on an existing-support
ray, the naive intact return gives Y proportional to dt² near dt=0, changing
the growth decision at identical trial stress. New zero-support rays have a
different one-sided limit and undefined exact normalization. A front-only
interpretation is possible but changes homogeneous nucleation semantics.
Do not add a phase floor or reset histories to produce tip growth. Next physical
choice: intact fracture/compliance admission versus frictional sliding, and
whether mature contact should use a distributed or registered-interface rate.
No phase-active/native/BP3 change; 104 protected files and earlier evidence
remain unchanged. Previous status follows.

---

# Structural-plane RSF continuation: point-qualified, fixed-band evidence

[Report](DIRECTIONAL_CONTINUATION_REPORT.md), [equations](DIRECTIONAL_CONTINUATION_FORMULATION.md),
[selection](DIRECTIONAL_CONTINUATION_SELECTION.md). Independent reference only:
existing faults slide on recorded material planes; new-record formation selects
maximum instantaneous kinetic speed with declared preference for genuine ties.
This explicitly changes every-increment global maximization. No phase threshold,
tensor reset, native/BP3 edit or propagation run. Finite/mature, reversal,
nonproportional, pressure-tangent and point restart/retry checks pass.

Three equilibrated band meshes give 128/256-cell slip and width differences
.000302%/.02614%. Separate final timestep pair gives .1583% slip change.
The original coarser temporal pair fails, and the literal monotone work-trend
gate remains false below the retained algebraic-check scale. Keep those flags.
This is a short .298 micrometre slip test, not sustained RSF or a general
oblique mechanical/stability qualification. No propagation readiness claim.
Next: a bounded fixed-phase mixed/oblique and meaningful state-slip control;
formation/secondary-plane phase energy and state admission remain unresolved.
Prior failures, constitutive controls and BP3 evidence are preserved below.

---

# Endpoint/RSF reconciliation: conditional kinetic regularity, continuation open

[Report](CONSTITUTIVE_RECONCILIATION_REPORT.md),
[equations](CONSTITUTIVE_RECONCILIATION_FORMULATION.md) and
[selection](CONSTITUTIVE_RECONCILIATION_SELECTION.md) complete the bounded
independent task. Exact Feng angle/update formulas reproduce the nonglobal
finite-h endpoints. Complete trigonometric enumeration repairs a shortcut
branch/sign issue, but does not supply an admissible first-activation tangent
under symmetric loading. Refined diagnostic paths remain nonglobal; no such
path is published as accepted. Phase/history reachability is distinguished
from synthetic obstacle-compatible reversed-memory inputs.

A new single-direction asinh-RSF/radiation candidate uses a normalized AT1
profile for the coherent slip/work mapping, without arbitrary reference width
or K storage. On each tested smooth selected branch the full mixed mechanical,
pressure, history and state spectrum is finite; no spatial cutoff or global
switching uniqueness is established. Positive state-dependent growth remains.
Fei's IN-PLANE previous-normal-pressure update is compared with current
pressure, retaining previous-state perturbations: both approach the same
continuous generator. Extending that temporal rule to freely changing global
maximizers produces rejected mature endpoints, confirmed at 80 digits.

The coupled mature point completes .3 s loading/reversal at three timesteps.
Full mixed QZ/independent elimination error is 2.67e-13; full coupled FD error
2.65e-12. A separate fixed-plane interface control qualifies only the coherent
mapping/state timing, not free-direction continuation. Three PNGs, compact
CSV/JSON and runnable references are linked from the report. Native/BP3/defaults
and previous failures remain untouched.

Next scientific decision: globally admissible evolution at conjugate ties and
between distinct return states, with consistent direction/pressure time
levels. Intact state birth and noncoherent bulk–surface mapping remain open.
No propagation/native integration or additional remedy is selected automatically.
Earlier entries follow unchanged.

# Single-direction reconciliation completed; mechanical gate still fails

[Report](FENG_DIRECTION_REPORT.md), [formulation](FENG_DIRECTION_FORMULATION.md)
and [selection](FENG_DIRECTION_SELECTION.md) close the bounded independent
Feng direction/dissipation task. Actual increment directions must globally
maximize F; continuity resolves genuine ties only. No conjugate averaging,
permanent locking or spin requirement is adopted. The recovered old spin
selector is observer-dependent, with a separate literal 2-D NaN-guard defect.

Single-direction flow removes the mature two-system redistribution mode, but
the complete pressure-coupled mechanical coefficient remains negative in
developed damage: phi=.8 gives -0.00195453 G while fixed pressure gives
+0.00467879 G. The rate-limit threshold is phi=.76761128 for m=4, mu=.1.
Seventeen archived states were independently re-solved; full coupled FD error
is 4.87e-11 relative. Finite-step zero-backstress controls fail final global
maximization, and a reversed-memory control has two admissible endpoints.
These failures are reported, not bypassed. No propagation test is launched.

Next action requires a constitutive decision addressing mechanical regularity
and endpoint/branch selection consistently with mature sliding. No remedy is
chosen automatically. Native/BP3/defaults and all prior accuracy/energy/branch
failures remain preserved. Two PNGs, compact CSVs and the reproducible reference
are linked from the report. Earlier entries follow unchanged.

# Explicit scalar spatial rate/state candidate fails the mixed mechanical gate

[Report](PRANGER_FEASIBILITY_REPORT.md), [complete formulation](PRANGER_FEASIBILITY_FORMULATION.md)
and [predeclared selection](PRANGER_FEASIBILITY_SELECTION.md) close the bounded
feasibility task. Constant energetic total-K storage is retired as the
mature-RSF candidate; the accepted gradient-hardening experiment remains intact.

One explicitly selected Pranger-inspired adaptation uses a common equivalent
rate/state and M_ex(q)=q+ell_r² Delta q in STATE evolution only. The paper's
implicit Helmholtz inverse is distinguished, not substituted or calibrated.
The full mixed incompressible displacement/pressure, Maxwell histories,
changing Coulomb planes and both conjugate rates are retained. At Q=0, opposite
conjugate rate perturbations leave scalar rate/state unchanged. A 45-degree
divergence-free mechanical mode then has no physical stress response at any
wavelength: the linearized evolution is singular even with positive direct
rate dependence. A finite-h reversed-memory counterexample has oblique poles
at 30.04149248/59.95850752 degrees, also with state feedback removed. These
are structural counterexamples, not physical earthquake growth predictions.

Independent coupled directional FD, mixed QZ/elimination and analytic
constraint checks pass at 1.37e-9 relative, 3.35e-9 relative and 6.00e-16
absolute. The constant-friction endpoint is singular. Kernel audit costs
6.48 s; no native/MPI/specimen run or conditional cross-fault prototype is
begun. The explicit mature control is not a phi=1 native API call; finite-h
synthetic states are labeled separately from archived physical data. The old
native three-root reversal guard and its original error remain retained.

Recommendation: STOP this common-scalar-state candidate, without choosing a
replacement automatically. Scientific review must first specify how independent
conjugate/directional flow is controlled. Phase/state birth, intact strength
calibration and nonsymmetric branch selection remain unresolved. The old
potential Hessian is not a certificate for this law. BP3, production defaults,
all 71 protected entry artifacts/sources and previous failures are unchanged.
Evidence: evidence/pranger_feasibility.zip, with PNG/CSV/JSON, code and commands.
Earlier entries below preserve history; their proposed rate/state task is now
superseded by this failed prerequisite. Propagation remains blocked.

# Mature sliding compatibility fails for the selected K-gradient

[Report](MATURE_GRADIENT_REPORT.md), [straight-band reduction and literature](MATURE_GRADIENT_FORMULATION.md),
and [predeclared controls](MATURE_GRADIENT_SELECTION.md) close the bounded task.
For k=2K:S, delta=integral k dr [m], V=integral 2W:S dr [m/s], a fixed normalized
band has Egrad=G ell_s² delta² J/2 and Rgrad=G ell_s² delta J. Constant,
non-relaxing gradient storage therefore survives the Q→0 mature limit as
accumulated-slip hardening. This is different from intended steady RSF sliding.

The mechanically equilibrated independent strip retains physical pressure,
both Coulomb systems, Maxwell/tensor histories and microfree ends. A prescribed
near-mature .999 core and stronger .02 surroundings remain fixed. At matched
1 mm plastic slip, gradient resistance is 2.74/17.97/58.30% of initial friction
for uncalibrated 1/3/6 mm lengths. The selected 6 mm and 3 mm fail the declared
5% finite-slip screen. The 1 mm case does approximate the narrow window; this
is not an asymptotic mature limit or selection of a new length. Flow widths
settle early, so continued hardening is not continuing phase/band widening.
At final 2 mm imposed motion, 6 mm traction=1998.76 Pa partitions into
865.78 gradient +153.16 cohesive +979.82 friction Pa; core cohesive force=.0021 Pa.
Most extra cohesive resistance is in the shoulders, separately measured.

N=257→513 changes final traction .01494% and rate-profile L2 .01884%; one dt
halving gives .000515%/.000211%. Initial representation errors are separate.
1600 intervals retain force/source/contact/energy checks; the complete mixed
strip action and independent local reference pass. Cost 73.25 s wall. This is
an independent equilibrated reference, not a new ASPECT native/MPI qualification.
No production source/defaults/BP3 or accepted evidence changes. All commands ended.

Stage 4's selected-candidate compatibility condition fails; coupled phase is
NOT begun. Recommend scientific review/derivation of a Pranger-type normalized
spatial rate/state interaction in place of energetic total-K hardening, with
steady/velocity-step interface tests, thermodynamics and a new mixed localization
audit first. No coefficient degradation, K reset, checkpoint conversion or
replacement law is implemented. Exact intact behavior, general branch selection,
MPI, propagation and earlier P4/P5c energy/spatial/boundary limitations remain.
Compact PNG/CSV/JSON/code/commands: evidence/mature_gradient.zip. Earlier entries
below retain their documented scope; the previous phase-task recommendation
is superseded by this constitutive prerequisite.

# Experimental K-gradient fixed-phase qualification complete

[Report](COULOMB_GRADIENT_REPORT.md), [published mapping and assumptions](COULOMB_GRADIENT_LITERATURE.md),
and [weak form/tangent](COULOMB_GRADIENT_FORMULATION.md) complete the selected bounded task.
Gurtin (2003) is the closest microforce/strain-gradient antecedent; the Maxwell,
cohesive and non-associated Coulomb specialization is explicitly ours. New
G ell_s²|grad K|² is physical recoverable storage, not a filter or added viscosity.
The demonstrated nonunique prescribed-X inverse remains rejected. A joint
u/p/K normal-map solve avoids that elimination, retaining physical pn, both
conjugates, all tensor memories and zero dilation.

The branch-symbol/forward-strip checks pass, including developed damage,
changing directions, boundary work, hold/reversal and complete derivatives.
Fresh serial 4/8/16 native blocks at phi=.8, ell_s=6 mm reach .125 s; mixed
condition numbers are 7.02/8.37/8.89. Final 8→16 peak/participation-width changes
are .10%/.18%. Native timestep halving changes deviator stress/slip rate by
1.24%/4.03%, so temporal uncertainty exceeds the last spatial difference.
The lengths remain uncalibrated. Native hold retains gradient storage while
Maxwell stress relaxes. Rejection/retry and once-only publication pass;
12 post-restart files are bitwise identical, as are 84 retained default-path
files. Changed-length restart is rejected. Two original fine failures and
checkpoints remain; first-guess lifting and premature sticking cleanup are
causally corrected, followed by a fresh fine confirmation. Existing source,
mechanical, contact and energy allowances remain intact.

No phase-active/large specimen is enabled. Next is a separately selected small
nonsymmetric coupled phase–microforce formulation and stability/selection check,
including the exact intact constraint; the old symmetric Hessian certificate
does not apply. Microboundary/length calibration, long-wave/global uniqueness,
MPI and non-Cartesian spaces remain open. The local Coulomb defect, original
P4/P5c accuracy/energy/branch/boundary findings, BP3 evidence and
strict_reference_match_passed=false remain. All commands ended. Compact data,
PNGs, source delta and commands are in evidence/coulomb_gradient.zip; full
matrices/checkpoints/failures remain separate. Earlier entries follow unchanged.

# Coulomb mechanical prerequisite fails — phase-active progression stopped

[Mixed localization report](COULOMB_LOCALIZATION_REPORT.md) and
[derivation/remedy comparison](COULOMB_LOCALIZATION_FORMULATION.md) complete
the selected audit. The actual incompressible displacement–pressure symbol,
including friction-pressure and changing-direction derivatives, loses rank at
developed damage. Single-system continuous-rate loss begins at phi=.7676113
for the retained m=4, mu=.1. At phi=.8 the fixed-pressure stiffness is positive
while the full mixed coefficient is negative; deeper double slip also fails.
Phase-gradient storage and local Maxwell relaxation do not control the
admissible fixed-phase short-wave mode. The reduced phase operator cannot
serve as a stability certificate for this mechanics.

Four-rank 4/8/16 blocks at fixed ell and dt=.00625 s give scaled condition
numbers 13.77/31.04/73.17. Slip remains broad in the short window: no collapsing
band is claimed. Full/half-step differences 9.60%/11.92% are not convergence.
Two fine full-step failures are separately diagnosed as initial boundary-lift
trial-contact violations, including the low-phase control. Both explicit
same-parent restart/cutback controls pass without relaxed guards. Phase stays
fixed; high-phase obstacle forces remain positive. 114 accepted intervals retain
source, mechanical, energy, ownership and single-publication checks. 33 old
exports are bitwise unchanged; all 60 protected hashes and stationary particles
are verified. Original failures/checkpoints remain.

Only opt-in read-only tangent exports, benchmark controls/analysis and docs
change. No new law, tolerance, checkpoint layout, default or BP3 change.
Recommend scientific review of a physical slip-gradient/microstress length
and boundary work; compare against a consistently dilatant model with changed
continuity, volume history and calibration. Neither remedy is adopted.
No phase-active Coulomb or larger run is begun. Earlier P4/P5c temporal,
spatial, boundary, branch and physical-energy limitations, including
strict_reference_match_passed=false, remain. All native commands have ended;
compact evidence is in evidence/coulomb_localization.zip, full runs separate.
Earlier entries below retain their documented historical scope.

# Confined Coulomb native block complete — fixed phase only

[Report](COULOMB_NATIVE_REPORT.md) and [selected scope](COULOMB_NATIVE_SELECTION.md) close the bounded fresh 4×4, phi=.2 fixed block with explicit p0=10000 Pa and physical mean p'=0. The qualified two-system return is promoted unchanged into the native material module. The mixed tangent includes constitutive pressure/direction derivatives and remains nonsymmetric. Existing constant-resistance modes/defaults and BP3 are preserved.

Homogeneous means match the point control at 3.59e-15; the separate maximum pointwise defect is 1.04e-9 (the strict 1e-12 pointwise identity check remains false). Native mechanics/source gates pass unchanged, FD relative error is 2.44e-9, and all actual Gauss-point contact pressures are positive (minimum 3155.97 Pa over in-plane normals). Four-rank ownership, repeated rejection/retry, once-only publication and same-rank restart pass; restart has 48/48 bitwise files per rank count and 1/4-rank field defect 1.08e-13. The retained native control has 27/27 bitwise files. Changed background and incompatible checkpoint mode are rejected before publication. All native commands ended; full checkpoints and original failed diagnostics remain separate from compact evidence.

No phase variation or Coulomb phase-stability certificate is claimed. Enabling evolution is refused. The subsequent proposal is a nonsymmetric mechanically relaxed phase-force derivative and conservative critical-cone strong-monotonicity screen, with independent coupled actions and perturbation checks first. A physical stability/selection rule outside that sufficient scope remains for review. No phase-active block or specimen is begun. The BE numerical work remainder remains separate from physical dissipation; no temporal/mesh/propagation accuracy claim follows. Preserve all P4/P5c failed criteria, boundary/branch findings and strict_reference_match_passed=false below.

# Coulomb direction-selection point qualification complete

[Report and diagnostic figure](COULOMB_DIRECTION_REPORT.md), with [pre-implementation derivation](COULOMB_DIRECTION_FORMULATION.md), complete the selected bounded task. The native radial law has free in-plane tensor increments, constant R=mu p_N, no fixed-plane projection, mature normal lock or phase-growth mask. Stress principal axes, net tensor-flow labels and phase contours are distinct.

Four archived same-rank restores export mechanical pressure without advancement/publication; source/copied checkpoint hashes remain identical. p_N=10000 Pa is not added to the retained mechanical weak loads. None of 2,166 near-tip cell-mean states per snapshot is fully compressive in-plane. Added confinement is plotted only as a labeled hypothesis, with no reequilibration or native input change.

The benchmark-only closed-contact return retains T/A/K/Q/L, Maxwell/rate conventions and the physical phase obstacle, uses orientation-dependent physical normal traction, zero dilation and both conjugate families. It passes 24 analytical-angle cases, 84 active-set cases, 15 independent 70-digit references, complete strain/phase/pressure/direction derivatives, dissipation/work and a 60-step rejection/retry/file-restart fixture. Existing constant-R controls pass unchanged. No native Coulomb mode/checkpoint conversion is introduced.

The tangent is non-associated (major asymmetry .357573 G); the old incremental potential/relaxed-Hessian acceptance does not apply. Keeping m=4 also changes pressure-dependent peak/residual calibration. Native adoption is not ready: decide physical confinement/contact and calibration semantics, then qualify a small fixed-phase nonsymmetric mixed operator and a separate reviewed stability criterion before any phase-active integration. Do not bypass the existing guard. No new specimen, mesh, RSF, AMR, transport or geometry campaign is begun.

All 52 entry source/artifact hashes, production defaults and BP3 remain unchanged. P5c remains accepted within its documented constant-resistance scope; its unresolved spatial/temporal/boundary findings and strict_reference_match_passed=false remain below.

# P5c continued-loading campaign complete within scope

[Measured report, figures and archive](P5C_CONTINUATION_REPORT.md) complete the selected192×128/four-rank continuation, both admissible holds and four focused half-step replays. Radial φ=.05 reaches x=1.143368 m at.170 s, beyond the initial represented support by1.0585ell. Initially-zero cells contribute39.83% of its growth. Growth remains seed-connected; no remote/boundary stop occurs. The φ=.6 front stays at x=1 m: this is weak process-zone extension and cap amplification, not developed-crack-speed qualification.

Fixed flow reaches its own target earlier; actual loading ends.169375 s fixed/.170 s radial, followed by.005 s holds. At the common loading time.169375 s, radial integrated/peak growth reductions are6.20%/14.49%. Unequal endpoints are explicit. Holds have exactly zero phase growth and slip, with nonzero Maxwell-relaxing stress.

Principal replay growth-relative L² differences are.000146% fixed/.03686% radial; practical local energy B_E is.001207%/.003379%. All native gates and once-only publication pass across163 new continued/fork intervals. Minimum relaxed curvature2.83213; at most621 critical masters versus unchanged1024. Particles remain fixed. No new forced rejection is claimed.

The interrupted interactive observer allowed extra native loading to.175 s; these states/checkpoint and original errors remain separately labeled. Recovery reproduces96 exports bitwise and restores the intended.170 s checkpoint. The later persistent controller completes the plan. A replay safety-priority helper correction has independent simultaneous-event checks; actual replays have no remote/boundary flags, and its supplemental observer started after completion. Native source, binary, physics, tolerances, defaults and41 protected/BP3 files remain unchanged. All native jobs ended; full checkpoints/outputs remain separate from the compact review archive.

Retain P4's failed temporal/spatial and boundary findings, P5c's28.68%/37.97% axial growth differences and **strict_reference_match_passed=false**. Local two-step replay is not full temporal/spatial convergence. Recommend only a targeted spatial-check design for this identified interval, including the unchanged critical-space capacity limit; no next campaign begun. Earlier entries below retain their original scope.

# P5c targeted axial refinement complete — 22 October 2026

[Measured report, figures and recommendation](P5C_AXIAL_REPORT.md) completes two
fresh 192×128 four-rank trajectories with the exact matched .00125/.000625 s
schedule through .145 s. Continuum seed, zero histories, law, boundaries,
Gauss8/source3, unlimited DGQ2 and all acceptance/stability settings are unchanged.

Radial reduction survives: integrated growth31.41% versus35.68% coarse; peak
33.25% versus40.71%. Common-kernel pre-growth driving reduction is11.36% versus
11.31%. Onsets move earlier on the refined mesh. Growth-field spatial changes
remain28.68%/37.97%; the result does not qualify quantitative convergence.
All increments use each mesh's own initialization. No initially intact material
activates and developed-damage fronts remain unchanged. Weak additional radial
flank growth is retained and contributes.0976% of its integrated growth.

Both native trajectories pass all checks and publish137 times. Checkpoints,
41 protected files, original evidence and executable/plugin hashes are verified;
particles remain fixed. Native commands are complete. Evidence is in
benchmarks/reconstructed_fault/particle_free/evidence/p5c_axial and its ZIP;
full native runs/checkpoints remain separate. No production-source change.

Recommend only: continue each mode from the existing pre-hold step127/.13875s
checkpoint, loading at.018/s through.155s then holding to.160s, aimed at initially
intact activation. Do not begin automatically. Preserve P4 temporal/spatial
failures, boundary sensitivity and strict_reference_match_passed=false. Earlier
entries below are historical evidence.

# P5c complete within bounded scope — 22 October 2026

[Report, figures and recommendation](P5C_REPORT.md) completes the freshly initialized finite-seed fixed/radial comparison. Corrected P4 baseline is cohesion2000 Pa/m4, proven from resolved inputs and actual checkpoint; original archives remain unchanged.

Exact-intact activation and nonuniform tensor 1/4-rank rejection/restart checks pass. Four-rank specimen runs and one active-window halving per mode pass unchanged native checks. Radial flow delays onset and reduces integrated cap growth by36%; the endpoint temporal difference is0.231% (maximum3.19%). No initially intact material activates, no branch forms and developed-damage fronts do not advance. This is growth inside the prescribed diffuse cap, not propagation qualification.

Current laws, unlimited DGQ2, Gauss8/source3, physical obstacle, stability cap1024 and scalar defaults remain. Endpoint histories/checkpoints are retained; all native commands finished. All41 protected files/BP3 artifacts are unchanged. Full report/evidence distinguish actual physical stress from T, tensor flow direction from ridge geometry, and physical work from BE ledger closure.

Next recommendation only: one matched fresh192×128 axial cap refinement at Δt=.000625s through .145s before longer directional loading. No subsequent run selected or begun. P4 temporal/spatial failures, boundary sensitivity and strict_reference_match_passed=false remain. Earlier stage entries below are historical evidence.

# P5b native tensor integration qualified — 22 October 2026

[Report, figures and compact evidence](P5B_NATIVE_REPORT.md) complete the bounded
serial fixed-mesh/Gauss8 bulk integration. Explicit fixed/radial tensor modes
reuse DGQ2 plus the native FE source complement, physical FE phase/obstacle,
CellId point histories and once-only publication. The scalar default/v2-v3
archives remain; tensor modes have distinct tags and reject mode/layout changes.

Fixed/scalar identity is <=1.10e-13; full coupled action FD is 1.88e-10 scaled
(5.73e-8 relative). Independent C^T g/C^T H C defects are 5.42e-21/1.18e-16.
The actual mixed-mechanics relaxed minimum is 5.94086818667 on 89 critical
masters, agreeing independently to 1.50e-15. Uniform C identity is exact;
physical mean agreement with the fixed hanging patch is 6.33e-14. Source,
mechanical/KKT and BE checks retain all existing bounds. Rejected trials and
same-mode restart reproduce 24 radial states/120 files bitwise, plus a separate
fixed-mode replay. Original scalar and P5a point regressions pass again.

The control is a uniform pre-damaged material block, not a propagation seed.
Correction at P5c entry: native cohesion=2000 Pa reproduces P5a and actual P4 fracture.prm m=4; the former P4 cohesion=1000 Pa/m=16 attribution
was incorrect; original archived evidence is unchanged. The dt=.1 radial globalization failure and the artificial-row FD
failure are preserved; the latter is corrected only in the diagnostic, and the
qualified sequence uses dt=.0125. The native end-cap pending interval is retained.
No tolerance, constitutive law, BP3 input/artifact, MPI path or production
checkpoint conversion is changed. No directional specimen was run.

Bounded lossless cleanup recovered 1.90 GiB and verified 108619 retained files,
173 links and actual restoration. Full P5b runs/checkpoints remain separate.
Next proposed task: review the finite specimen's explicit material tuple,
boundaries/seed/loading, then select a short directional physical-response and
timestep control. Do not begin automatically. P4's quantitative failures,
boundary sensitivity and strict_reference_match_passed=false remain.

# Evolving-phase continuation

## P5a tensor material-point return qualified — 22 October 2026

The user selected the tensor-memory/changing-flow option. The
[P5a point report and three PNG/CSV/JSON figures](P5A_POINT_REPORT.md) qualify the
new stateless API and small point-history fixture. Fixed-direction identity
(stress/driving/storage/dissipation/path) has maximum scaled defect 4.87e-14;
complete return FD is 7.17e-9 <=1e-6, passive basis covariance 2.00e-14,
independent 60-digit minimization (including path samples) <=8.14e-16, and local
BE ledger <=7.99e-16.
The pure-shear intact threshold remains 3000 Pa. A frozen-direction derivative
fails at .0671, confirming the need for the retained direction derivative.

The non-proportional point path preserves and relaxes old tensor components,
then accumulates new flow in another direction. Two rejected trials/path,
deterministic retry, duplicate/stale publication rejection and exactly 800
accepted publications/path pass. File restart at step 480 matches every actual
history field and all 320 following states exactly; a separately populated
record includes nonzero Tzz/A/K/Q. These are point-fixture records, not native
checkpoint changes. Coordinate covariance is not finite-rotation/objective-rate
qualification. Phase is prescribed in the demonstration; no phase equilibrium
or propagation is claimed.

The original scalar source/tests, native solver/history/checkpoint sources,
P4 archives and BP3 hashes remain unchanged. The first harness normalization
failure and independent-reference probe/metric failures are preserved. The
one-sided energy probe passes the unchanged 1e-7 bound after one halving,
with the expected factor-four truncation reduction; no law/tolerance tuning. Final tiny-h controls additionally exposed and repaired
zero-times-overflow and squared-norm underflow using reordered arithmetic,
hypot and exact component-zero checks, with no phase floor or clipping.

Next proposed task: bounded P5b native owner/checkpoint integration, fixed-S
identity, complete coupled tensor tangent and mechanically relaxed stability,
then native lifecycle checks on one small fixed mesh. Do not start the full
directional specimen automatically. That later test is a finite physical body
with explicit top/bottom grips, restrained vertical/lateral-natural horizontal
conditions and natural phase flux. P4 quantitative temporal/spatial failures,
57.26% boundary sensitivity and strict_reference_match_passed=false remain.
No native directional, MPI, RSF, transport, AMR or geometry evolution is qualified.

## Hold-instability remedy qualified — 21 October 2026

[Report and compact native evidence](BULK_STABILITY_REPORT.md): supported
loading-aware accepted-state continuation plus a mechanically relaxed curvature
gate now precede bulk history publication. The forced predictor reproduces
-.208558482525 curvature and .01583691082 hold growth, is rejected with publication
still 140, then recovers to the stable hold and publishes once. The loaded control
retains .08560809023 growth and +.139907314547 curvature. Complete persistent
histories, physical stress means, phase/weak forces and reactions are bitwise
identical after recovery, two outer rejections/retry and same-rank four-rank
restart. All original mechanics/KKT/source/BE bounds pass; BP3 and the original
P4 archive are unchanged. The initial entrywise-symmetry diagnostic rejection
and its safe recovery remain recorded; the corrected norm check matches the
existing independent analysis, with no physical tolerance change.

**The hold-instability investigation is resolved within the tested homogeneous,
matched-coefficient fixed-direction bulk setting.** The full critical-space
check costs 85–88 s when growing, versus <1 ms for strict-obstacle holds; it is
bounded to 1024 masters and fails closed, not a global stability/minimum theorem
or scalable solver. No new hanging-node or cross-rank restart qualification.

P4 still fails the 5% temporal (8.302% L2) and spatial (7.442% interval L2)
criteria, with 57.26% growth sensitivity to the right boundary. Original energy
screen failures, branch errors and strict_reference_match_passed=false remain.
Quantitative propagation and all P4 criteria are not passed.

The [P5 scientific recommendation](P5_DIRECTION_SPECIFICATION.md) now selects
one proposal for review: tensor cohesive/accumulated memory with changing **flow**
direction even in loaded material. It preserves old memory, recovers the scalar
law when direction is fixed, and does not lock the pre-existing unloaded seed.
Virgin-only locking is a distinct unselected law. Next, after scientific selection,
qualify a small tensor material-point return/tangent/lifecycle experiment with
fixed-direction identity. Any subsequent specimen is finite, not claimed distant-
boundary. No free direction, RSF, AMR, transport or geometry work has begun.

## Bounded P4 continuation check complete — 21 October 2026

[Report and PNG/GIF/CSV/JSON evidence](BULK_CONTINUATION_REPORT.md) complete the
shared .15875 s checkpoint controls through .17875 s at Δt=.00125/.000625/
.0003125 s, plus one fresh 192x128 tangential refinement and one 1.75 m domain
control. The first-hold predictor roots have mechanically relaxed negative
curvature (-.208558/-.168661/-.143308). Native accepted-state continuation has
strict obstacle gaps and zero hold growth/slip, with phase/mechanics available.
This is a diagnostic continuation choice; no default, physical law or native
acceptance tolerance changes.

Common-time state differences are small, but half/quarter time sampling still
fails the declared 5% growth-relative criterion. The refined final growth L2
difference is 1.328%, rising to 7.442% over the selected interval; first growth
is (.12750,.12875] versus coarse (.13375,.13500]. This full-field bracket
corrects the preliminary log estimate. The enlarged domain reduces integrated
growth 57.26% and shifts the final .1 front by 111.10 mm. All selected holds are
locally stable. The old specimen is not domain independent, and neither
continuous front speed nor quantitative propagation is qualified.

The work Cauchy/absolute-remainder criterion is small (B_E=.00505735%), separate
from exact BE closure and physical response errors. A benchmark hard-coded
x=1.5 side diagnostic was corrected using native boundary-face DoFs; original
failure/DSO are preserved. The retained coarse step is bitwise unchanged, and
124 enlarged-prefix physical exports match. Source, phase/mechanical, ownership,
once-only publication, frozen-particle and same-rank restart checks pass.

The historical record describes a single evidence archive,
`benchmarks/reconstructed_fault/particle_free/evidence/bulk_continuation.zip`,
containing PNG, one actual-frame GIF, compact data, reports, inputs, source overlay,
logs and checksums. That archive is absent from this worktree at the 7 November
2026 tip-solver closeout; its availability elsewhere is unverified. The historical
record also states that full native checkpoints/matrices were separately archived.
The [P5 specification](P5_DIRECTION_SPECIFICATION.md) proposes tensor memory and
a consistent free-flow return, distinguishing it from virgin-only locking and
scalar-history rotation. Scientific selection and branch rules precede its
implementation. No P5 run, transport, AMR, geometry evolution or broader
propagation qualification begins. Earlier failures, strict_reference_match_passed=
false, existing local work and protected BP3 remain.

## Fracture scales / one propagation-focused specimen complete — 21 October 2026

[Report, PNG/GIF and compact CSV](BULK_FRACTURE_REPORT.md) complete the selected
audit, independent native-checked strip and one fresh released-side MPI test.
The original Lc=26.67 m / slip scale=.02667 m exceed the specimen/loading scales.
Gc=1 J/m², strength drop=2000 Pa, ell=.046875 m give Lc=.25 m, slip scale=.0005 m
and m=4, retaining the existing convexity guard. The m=3 startup rejection is
preserved. Fresh .6 continuum seed has a 1 m straight interval, .185293 m width;
initial L2 errors are .592% whole / 2.058% cap with zero unloaded phase drift.
No spatial activation gate or altered constitutive/source/acceptance rule is used.

The four-rank 96x128 trajectory completes 148 steps to .185 s in 1906.10 s.
Growth starts on the centerline in (.13375,.13500], reaches an initially-zero
node in (.14375,.14500], and develops a connected softening zone without detached
components at the reported contours/states. Phi=.1/.3 first exits advance
231.14/49.79 mm, while phi=.5 moves only 1.005 mm: process-zone development,
not mature core propagation. Far-ahead activation is (.17375,.17500], connected
at saved thresholds. The largest step increment .085608 in that same interval
is preserved for temporal verification. Physical work C_E=.004534% and all
native guards/once-only publication checks pass, but new-regime C_T is unmeasured.

The actual hold grows, yet an independent no-growth Maxwell-scaled candidate
at the same incoming state has positive phase gaps and satisfies KKT/mechanical
bounds. This exposes nonunique continuation; no native stability or branch
selection rule is added. The strip's extended-load nonconvex continuation stop
also remains evidence; stationary-profile excess work recovers the Gc AT1
normalization, not a completed irreversible post-softening fracture test.
Core sources/binary, baseline DSO and 34 protected BP3 identities are unchanged.
Earlier spatial/energy failures, branch findings and strict_reference_match_passed=
false remain. Full solver archives/checkpoints stay under bulk-fracture/archives;
compact deliverables are PNG, one GIF and CSV. Next: bounded timestep/branch
qualification of .1725–.17875 s before longer loading or propagation claims.

## Fixed-mesh bulk MPI and one boundary control complete — 21 October 2026

[Report, matched figures and animations](BULK_PARALLEL_REPORT.md) complete the
selected task. The fresh 24x32 1/2/4-rank growth controls take 118.34 / 67.24 /
48.57 s: measured 1.76x / 2.44x wall improvement. Physical-stress differences
are <=4.07e-13 relative with unchanged native tolerances. Each rank count passes
two outer rejections, deterministic retry and one accepted publication; all 74
common history/field files and 17 resumed ledger rows match its uninterrupted
control bitwise. Bulk assembly/history are distributed; the coupled search and
rank-zero direct factorization remain a bounded scaling limitation. v3 native
cell checkpoints preserve all DG/native-FE/local/phase owners; v2 serial restore
remains separate. MPI hanging phase evolution stays guarded.

One fresh 96x128 horizontal-side-release case reaches .185 s on four ranks in
2244.41 s, with identical physical initialization, mesh, law, Gauss8/source3,
unlimited DG/native-FE complement, loading and acceptance criteria. Lateral uy
is fixed while ux is free; mechanical pressure has no artificial gauge. Friction
p_N stays 10 kPa. The unloaded obstacle is unchanged exactly; endpoint reaction
work agrees with native work to 6.94e-17 J/m. All native/source/KKT checks pass.

Loading is substantially redirected toward the centerline. Cap growth starts
one interval earlier; remote growth starts four intervals earlier, in
(.145,.14625]. Integrated growth outside continuum seed support is 92.29%, and
80.22% lies at x>.9. Weak .001/.01 contours connect through to the right boundary
at .155/.1675 s by merging with remote damage. This is not a 0.70 m advancing
tip: .05 centerline motion is 3.56 mm and .3/.5 contours remain fixed. The
released case reduces 45-degree flank widening but does not isolate propagation
from bulk nucleation. Common-scale physical stress/Y/weak-force profiles and
whole-domain stress/phase/rate movies make that distinction explicit.

C_E=.00263215%, with separate BE numerical terms and residual work; changed-BC
extended-interval C_T is unmeasured. The original endpoint-screen failures,
35.86% failed spatial target, branch findings and strict_reference_match_passed=
false remain. No continuum growth speed, physical stability, general MPI/AMR,
transport or propagation qualification is claimed. Full four-rank endpoint
restore is also checked without advancement. Five native checkpoints and the
(.15,.15125] largest-increment interval are preserved for later verification.
All 34 protected BP3 identities/defaults and the point constitutive source are
unchanged. Next: verify the flagged new-growth interval and unresolved spatial
response before interpreting weak connectivity as physical tip advance. No
further trajectory or roadmap stage starts automatically.

## Exploratory continued loading complete — 21 October 2026

[Report and animation](BULK_CONTINUED_REPORT.md): one fresh 96x128 aligned-wide
trajectory completed .185 s in 148 accepted publications. Delta t=.00125 s,
physical law/seed/mesh, unlimited DG/native-FE complement, Gauss8/source3,
histories and tolerances are unchanged. Loading remains identical through .15,
continues at .018/s through .175, then holds displacement. No compatible old
pre-hold checkpoint existed. Fresh initialization and the entire .15 prefix
match the retained fine case exactly, including all six dense/history exports.

At .175, centerline advances are 9.50/2.12 mm at phi=.001/.01; higher centerline
contours remain fixed. Both 45-degree flanks advance 41.92/13.10 mm at those
levels, while phi>=.1 contours do not move. Interior nucleation near x=1.03
begins in (.15,.15125], followed by right-boundary activation. By .17375 the
lowest superlevel component joins through to the right boundary: this is not
a tip-propagation distance. Final integrated growth is 1.89394e-3 m2; 87.30%
lies outside continuum seed support. Peak cap/whole increments are
.006728/.009932. No developed high-phase tip extension is established.

The hold has exactly zero phase/slip-memory change and w=0; physical stress
matches homogeneous Maxwell scaling to 1.50e-15 relative L2. Native guards,
independent storage/rate/history checks and once-only publication pass. Endpoint
and trapezoidal work are 3.41077/3.38225 J/m; C_E=.00232113%, but new-regime
temporal uncertainty C_T is unmeasured. Prior energy-screen failures, the failed
5% spatial target (35.86% growth discrepancy), branch findings and
strict_reference_match_passed=false remain. BP3/defaults/core are unchanged.

Fifteen native checkpoints and full histories around new activation are saved.
(.15,.155] and (.17125,.17375] are retained for subsequent timestep verification;
none is run automatically. This closes only the selected exploratory loading
experiment, not spatial accuracy, growth-speed or propagation qualification.

## One aligned-wide spatial comparison — 21 October 2026

[Report, compact table and equal-scale figures](BULK_SPATIAL_REPORT.md) complete
the selected comparison: reuse 48x64, add one fresh 96x128 trajectory, both at
Delta t=.00125 s to .16 s with identical actual timestep sequences. Physical
inputs, continuum seed, Gauss8/source3, unlimited DG/native-FE complement,
constitutive law, integrator, obstacle and native tolerances are unchanged.
No history transfer or independent initialization relaxation occurs.

Initial seed L2 errors decrease from .33972%/.74985% (whole/cap) to
.085286%/.188486%; shared vertices agree exactly and unloaded drift is zero.
Nevertheless both refined cap flanks activate in (.1275,.12875] s, seven
timesteps earlier than (.13625,.1375] s. Final growth L2 differences are
**35.86% on each flank**, with peak differences 15.58%, integrated-growth
differences 34.18%, and cap-fracture-increment differences 30.12%, all scaled
by fine growth. The predeclared quantitative 5% target fails. Broad driving
lobes retain +/-43-degree outer peaks; angular Y differences at loading stop
are 2.29–2.87%, reaching 5.92% at an earlier common time. This supports a
qualitative loading-oriented response, not resolved onset/growth amplitudes.
Two meshes provide no spatial order or propagation qualification.

All source, mechanical/KKT, history, once-only publication and exact BE checks
pass. Both holds have no phase growth. The original endpoint screens remain
failed (1.12958% / 1.12982% here); prior temporal/energy findings are preserved.
The only C++ correction is extended accumulation in a read-only quadrature-area
observer after an initialization-only sum-roundoff failure; the 1e-12 coverage
check remains unchanged, and failed output plus old DSOs are retained. No ASPECT
core or checkpoint change. BP3/defaults and all original branch/orientation
findings remain protected; strict_reference_match_passed=false. Stop here.
Any further resolution/phase-force diagnosis requires a new selected task.

## Aligned-wide work/temporal qualification — 20 October 2026

[Report, formulas and figures](BULK_ENERGY_REPORT.md) complete the selected
bounded task. Reuse h=.0025 s to .16 s; run only .00125 and .000625 s from the
identical seed/zero histories. Mesh, loading switches, Gauss8/source3, bulk law,
unlimited DG, BE, solver limits, physical obstacle and BP3 remain unchanged.
A test-only DSO exports accepted boundary reactions and compact native QP data;
the native core is unchanged. No third native trajectory was run.

Endpoint / trapezoidal work at h is 2.328670618 / 2.276564165 J/m. The leading
endpoint excess obeys dgamma[Q(.15)-Q(.05)]/2 exactly. Historical endpoint-screen
results remain **2.23279% fail, 1.12958% fail, .568557% pass** at h,h/2,h/4.
No old failure or acceptance tolerance is changed. Trapezoidal balance total
variations are .00492688%, .00187098%, .000373072% of event work, with negligible
between-step cancellation. The hold has zero work but retained reaction/stress,
positive viscous dissipation and decreasing storage. All native numerical gates
pass; the quarter run's additional 6.66e-16 s terminal step is retained in all
budgets/publications.

Peak growth is .001671205 / .001671325 / .001671386. Relative to **growth**, the
maximum selected h/2–h/4 L2 difference is .05662%; final difference is .005071%.
First cap brackets narrow from (.135,.1375] to (.13625,.136875] s; h/4 first
activates ±42.879°, resolving the coarse six-node batch. Common angular driving
is stable; no mirror alignment is used. State/storage/work differences show
first-order convergence, not second order from trapezoidal postprocessing.

Recommend a work-based **reporting** criterion: sum of absolute per-step physical
balance defects plus a common-time sum-of-component temporal Cauchy uncertainty,
normalized by actual absolute trapezoidal event work, <=1%, with <=5% growth-
relative response differences and compatible shrinking onset brackets. The
measured fine budget is .0027709%, so this fixed-mesh work/temporal question passes
that practical criterion. Exact BE accounting remains a separate mandatory
implementation check; the original endpoint result stays historical evidence.

This does not clear spatial/orientation sensitivity or qualify propagation,
material physics, general MPI/AMR or transfer. Keep strict_reference_match_passed=
false and all protected BP3 artifacts/defaults. No next native stage starts
automatically. Dated entries below describe the preceding narrower scopes.

## Non-affine source history and repeated orientation — 20 October 2026

**Completed within the selected serial, fixed-mesh constant-coefficient bulk
scope:** [report, formulas and figures](BULK_SOURCE_HISTORY_REPORT.md). Independent
DGQ2 retains all nine modes and a native FE memory retains its unrepresentable
gradient complement. Full reconstructed history is point-evaluable on matching
cell traces; mechanics, tangent and energy use the same full stress. The source3
rule, physical law/seed/loading, unlimited DG, original 1e-12 consistency check,
acceptance rules and BP3 production path remain unchanged.

The final small affine/distorted pair passes independent point/source/cache,
storage and tangent checks. The distorted remainder reaches .03947 Pa, with
non-proportional normal stresses and a nonzero mixed Q2 mode. Native consistency
<=3.67e-15; deliberate inconsistency still fails by ratio 1.4142e7. Actual repeated
rejection/retry preserves all accepted owners; publications advance once.
Prefix/restart versus uninterrupted advancement matches 404 field/history files
bitwise. Bulk v2 checkpoints persist FE memory and rebuild its derived projection;
v1 bulk checkpoints explicitly reject conversion. BP3 serialization is unchanged.
The first complement's step-47 roundoff failure and 70-digit causal diagnosis
are retained: constant-translation subtraction/long-double summation corrects
native-memory gradient cancellation, without any tolerance change.

Both fresh orientation runs complete 58 steps to .145 s. Full source discrepancies
<=4.40e-14; mechanical/KKT/tangent gates pass. Broad physical Y lobes remain near
±43° on both meshes at outer cap radii. Discrete onset is sensitive: aligned
first growth .1375 s at ±40.6–49.4°; rotated lower growth .1300 s at -45°, upper
.1425 s over 29.5–57.5°. Rotated maximum cap increment is 31.68% larger. Initial
weak resistance differs before loading; the changed far-field mesh is also a
confounder. This supports loading-oriented broad lobes, not mesh-independent
first nodes, symmetry or convergence. Do not reshape the cap from this result.

**Physical-energy qualification still fails:** new interval remainders 2.35%,
with separate BE losses/residual work recorded; original full-interval/halving
failures and `strict_reference_match_passed=false` remain. All 34 protected BP3
file identities pass. Old native builds, DSOs, checkpoints and evidence remain.
No server/package, AMR/MPI expansion, transport, physical law/default, birth or
geometry change is made. Stop here. The next scientific prerequisite is the
separate physical energy/work qualification, not automatic propagation.

## Retained first mesh-orientation attempt — 20 October 2026

The user-selected [bounded control](BULK_ORIENTATION_REPORT.md) is attempted and
**blocked before flank onset**, with the original source check intact. A valid
45° core/annulus mesh has 800 rigid core and 2880 transition cells; all corner
Jacobians are positive and physical area remains 1.5 m². The physical seed,
loading, shear tensor, law, local core cell sizes, Gauss8/source3 and tolerances
are fixed. No phase/history transfer or runtime mesh change occurs.

The identity mapping control passes. The rotated mesh accepts 42 publications
through .105 s, with bitwise unchanged phase and q=Y=0. At .1075 s mechanics
converges but independent DG source publication fails <=1e-12 before commit.
The all-nine-mode projector passes on affine rotated cells and fails on
non-affine transition cells: independent/native worst synthetic gradient error
88.771337582 m^-1, unchanged by source8. This is a representation limitation;
it is not that large a measured physical stress error. Valid prior checkpoints
and all failed attempts are retained.

Common physical-kernel arcs, equal-scale mesh/phase/Y/increment overlays and
independent force checks are in the report. Both broad-lobe orientation and
first-node grid-following remain undecided because activation is not reached.
Do not change the seed on this evidence. The next prerequisite is bounded
qualification of an explicit native-gradient projection remainder/history on
non-affine cells, with unchanged equations and publication/rollback/restart
semantics. That extension is proposed, not implemented or automatically
selected. Core source/binary, original fixture DSO, BP3 and prior angular
evidence are unchanged. The physical-energy failure and strict reference
mismatch remain unresolved.

## Angular flank diagnosis — 20 October 2026

The [saved-state report and three figures](BULK_ANGULAR_REPORT.md) complete the
user-selected angular diagnosis without a new native run or physical/numerical
change. The exact seed is the existing 5000-point compact cohesive profile,
sampled into Q1. Its ideal smooth cap has angle-independent point fracture force
at fixed radius, with a positive curvature term and an inside/outside force jump
at R=.30882159390707 m. Native weak averages and represented curvature have
angular variation already at initialization; those are not pointwise material
weak spots or an unloaded phase relaxation.

First wide-case cap growth at .1375 s occurs at ±40.6013°, ±42.8789° and
±49.3987°. Immediately before, .135 s, the smallest combined margin near
r=.288111 m is 1.64670 Pa at 49.3987°; the driving maximum is near 42.75° and
the fracture minimum is at the band/cap join. Both generated driving and the
initial weak resistance pattern select the growing nodes. Saved QP histories
are decoded/verified directly and averaged compatibly; none are interpolated
into another constitutive state. Phase contours overlay all driving maps.

Existing aligned square/fine comparisons do not establish a physical,
mesh-independent onset angle. Recommend one 45° near-tip mesh-orientation
control with the physical problem fixed, after qualifying the rotated-mesh
source/diagnostic assumptions with existing guards intact. Do not reshape the
cap, increase drive or begin this experiment automatically. The energy-screen
failure, spatial/branch limitations, BP3 and strict_reference_match_passed=false
remain preserved.

## Selected bounded cap-flank / boundary diagnosis — 20 October 2026

The user selected three fresh controls of the completed bulk-model experiment:
32×64 Gauss8, 32×32 Gauss6, then [0,1.5]×[0,1] with 48×64 Gauss8. See
[fixed comparison contract](BULK_SENSITIVITY_SELECTION.md). The same continuum
seed, unloaded histories, physical loading interval, constitutive parameters,
source rule and acceptance tolerances are retained. No quadrature-history
transfer or changed physical propagation specimen is selected. Numerical
controls precede the one-boundary-distance check; stop after the diagnosis
and recommendation. The earlier failures below remain evidence.

**Completed:** [report and common-scale maps/cuts](BULK_SENSITIVITY_REPORT.md).
Fine 32×64 G6/G8 differ by at most 8.41e-11 in phase; the flank increment
.00129893 is not a quadrature artifact. Square 32×32 gives .00101077, versus
.0000267181 on 16×32; both mesh directions matter. Initial L2 errors are
1.33455%, .779183%, .339721%, separately measured from the same continuum.
Independent weak assembly agrees within 1.8e-14 after normalization by
Kf times the nodal mass.

Moving only the right extent to 1.5 m at matched fine spacing/G8 eliminates
right-face damage while increasing the flank peak to .00167120 (28.66%).
Near-field initial phase is identical. Low flank-cut contours advance a few
millimeters; all tracked connected centerline fronts and core-tip phase remain
unchanged. Boundary distance affects loading, but the flank response persists
without the boundary island. Left-face growth remains; domain independence is
not established. Recommend the tested 1.5 m/G8 specimen as the smallest physical
revision, without increasing loading or treating it as a propagating fault.

All three 64-step runs pass unchanged numerical/publication gates. New physical
endpoint energy screens still fail (2.1341–2.2328%); retain the earlier failed
dt-halving result and separate time-discretization explanation. No C++ or
production-default changes, further runs or physical-specimen revision are
performed automatically. BP3 and original evidence remain preserved.

## Selected bulk-model stage — 20 October 2026

The user-selected `codex_bulk_phase_implementation_brief.md` and
`bulk_surface_cohesive_derivation.tex` now supply the previously missing
intact-material activation choice. This is a **distinct default-off bulk
constitutive model**, not a repair declared equivalent to the P2 surface law.
The prior P4a zero-driving result, P2 strict branch failure and all BP3/P3
evidence remain unchanged. Historical “await activation selection” notes below
describe their earlier scope and no longer block this newly authorized stage.

[Discrete formulation](BULK_COHESIVE_FORMULATION.md) and
[pre-run native selection](BULK_COHESIVE_NATIVE_SELECTION.md) fix backward Euler,
the history-dependent return/mixed intact limit, physical obstacle, coupled
residual/tangent, separate energy accounting and the bounded ordered tests.
The [completed bounded report and three actual-data figures](BULK_COHESIVE_REPORT.md)
cover material points, raw-weight surface residual identities, a native loading/
hold trajectory and one timestep, mesh and quadrature control. The serial native
bulk ownership/solver/checkpoint path passes its numerical checks and loaded
rejection/retry, single publication and bitwise restart comparisons. Original
failed hold globalization is diagnosed and retained; feasible master-space
projection repairs the search without changing the coupled equations or bounds.

**The physical propagation/energy screens do not pass.** Primary tip-window
phase does not change. The finer mesh thickens an already damaged off-axis cap
flank; all five tracked seed-connected centerline fronts remain fixed. Most new
damage is boundary nucleation. Spatial stress moments differ .9334%, but growth
area differs 28.48% (declared limit 10%). The endpoint-work remainder is 2.1321%
and 1.0786% at dt/2 (declared limit 1%), despite discrete ledger errors below
1.73e-14 J/m. Initial representation error is separately 1.33455%/.339721%;
unloaded drift is exactly zero. Gauss6/8 and fixed-mesh temporal field checks pass.
Do not substitute post hoc trapezoidal work for the failed declared screen.

The observer is a pure straight-column moment diagnostic, not automatic M3
ridge reconstruction/birth. Local a/k/L/q/rate histories remain tied to the fixed
quadrature layout. No new-mode hanging-node, automatic cutback or MPI claim is made.
The existing `Fault constitutive mode` selector gains `bulk cohesive`; its
default remains `cohesive`. No RSF, AMR, advection, spin, moving mesh, branching,
MPI or server event is selected. **Stop after this bounded result.** A possible
next task is a boundary-loading/cap-flank sensitivity diagnosis under the same
law, selected by the user; do not raise loading or expand a mesh campaign.
All 34 protected BP3 identities and incoming manual work remain unchanged;
the old P3 control reproduces 16 physical/history/load/constraint files bitwise.
P2 `strict_reference_match_passed=false` remains.

19 October 2026. Worktree `aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, entry revision `612b75d3570d81763407875f3a12c12999b9fa8e`.
The consolidation requested in `../introspection.md` and the subsequently
approved **bounded serial fixed-mesh hanging-node phase task are complete**.
Continue from **cohesive, domain-free PF6a: independent DGQ2 history, ordinary
FE fracture, consistent-endpoint reduced Newton, fixed reference fault**.
Phase growth on that fixed fault is distinct from geometric extension or
physical tip propagation. The subsequently selected bounded P2 response and
original P3 synthetic geometry-transfer diagnostic are recorded below.
P3 is closed at `f1f9f29cf40c32d1ddcea000b724375b64853532`. The selected next
task is bounded **P4a finite-tip mechanics/phase driving**; its frozen
diagnostics are complete, while physical advancement awaits the specific
scientific activation dependency below. General propagation is not selected.

## P4a: integration passes; intact-material driving is unresolved

[Report, actual native tip close-up and field figure](P4A_TIP_REPORT.md) and
[pre-run equations/screens](P4A_TIP_SELECTION.md) cover the aligned control
and one endpoint inside a bulk cell at (.515625,.5) m. Actual native bulk
JxW, associations, analytic-normalization rays, phase/obstacle and complete
mechanical/fracture gradients are exported. Independent Gauss24/32 integration
resolves surface knots, the strip endpoint and represented phase support.
Maximum normalized native/reference error is 1.13e-13; full mechanical-gradient
errors are below 7.46e-16 J/m. Energy finite differences pass at two perturbation
sizes. The unloaded obstacle KKT has zero negative nodes. Initial phase L2
representation errors are .339721%/.321643%, separately measured.

The current cohesive energy has **zero local mechanical phase derivative
ahead of the actual finite surface**, even within the rounded damaged cap.
A strictly-ahead perturbation gives exactly zero mechanical derivative in
both computation paths; cap fracture derivatives remain positive. A basis
function centered ahead but straddling the tip gives a small nonzero assembled
derivative, separately identified. The cap is fracture storage/obstacle, not
an implicit added surface or a supplied intact-material activation law.

Required choice before physical response: define mechanical energy/traction
and activation ahead of Gamma, including its I=0 birth limit. The historical
intact-point q formula does not define q outside an associated profile.
No floor, segment insertion or legacy-H/endpoint combination was used to fill
that gap. Existing finite-tip mechanical/phase guards remain. Thus **the
equilibrated mechanics and phase-active tests are not yet run or qualified**;
the frozen snapshots do not establish a physical energy/work balance.
P3's zero-tip/zero-birth, positive-support transfer remains its limited policy.

All observations preserve accepted owners exactly. A fresh P3 default control
matches 16 original physical/history/constraint exports byte for byte.
Production source/headers/CMake, all 34 BP3 identities and manual AGENTS work
are unchanged. P2 strict_reference_match_passed=false and original failures
remain. No RSF, transport, AMR, MPI expansion or general propagation was begun.

## Closed bounded P3 native completion

User acceptance closes P3 within its reported scope.
[Report and two figures](P3_NATIVE_COMPLETION_REPORT.md) close the selected
serial, fixed-mesh, zero-time, zero-tip-history native integration task. Actual
material DG T/Z/I, manager q=Z/I/slip/rates, physical phase obstacle and Simulator
checkpoint owners are used. No physical tip/birth law was added. The finite-tip
mechanical/phase evolution guard remains, including after restart.

The original loaded result is reproduced: full-tensor stress error 2.11e-7
relative L2, versus 3.73e-7 with unchanged T. One stronger plateau/taper control
gives 6.43e-6 versus 1.00e-5. Physical cell integral defect is 1.68e-16 Pa m²
versus 1.63e-5 unchanged-T, but sampled pointwise maxima worsen; the report
retains that limitation. Added shear/friction loads (+203.732/+250 Pa m) are
separated from projection error. Phase/obstacle and fracture storage are exact.
Native normalization, full stress and weak loads pass independent references.

An unsupported append exercises preparation rollback; a fully prepared invalid
commit rejects. Retry is deterministic and one geometry publication occurs,
with zero physical publications. Native separate-process checkpoints before
and after append match persistent histories/fields/particles byte for byte in
both cases. The zero-time resume path rebuilds native boundary constraints;
every row matches fresh state, alongside exact stresses, loads and ledgers. Search/QP/ray/basis and native surface assembly rebuilds are used;
full Stokes objects are invalidated, not solved under this frozen guard.

The generic bootstrap I/analytic FE mismatch was diagnosed and fresh geometry
now constructs the existing canonical FE I; strict restart checks are unchanged.
Constant-friction zero-time observation leaves birth rates zero and Vmin intact.
The existing native suite passes; 34 BP3 identities and manual work are unchanged.
The older component-only evidence below keeps its original scope. P2's strict
branch failure remains false. The selected P4a task above audits finite-tip
mechanics/birth and physical advancement; no automatic MPI expansion, AMR,
RSF, transport or general propagation follows from P3 closure.

## Bounded P3 synthetic straight append

[P3 report](P3_SYNTHETIC_EXTENSION_REPORT.md) defines the transfer contract and
qualification boundary. The imposed finite seed has an interior tip at x=.5;
one collinear append reaches x=.75 on a fixed 32×64 mesh. The existing compact
stationary cap is sampled directly; phase and its physical obstacle stay fixed.
Old material/vertex identities, Z=G s_elas, accumulated slip and rates survive
exactly; prescribed new surface rows are zero and the old tip histories are
zero. No bulk material is born. This is new surface representation within
already initialized cap damage, not demonstrated virgin-material activation.

The joint stress-coordinate map is `T+=T-+Pi_Q2[2(Z+chi+-Z-chi-)S]`. Measured
full-tensor stress transfer error is 2.11e-7 relative L2 (localized 3.10e-7),
maximum .01544 Pa; retaining T unchanged gives 3.73e-7. Bulk energy transfer
defect is 3.12e-11 J/m; the separately predicted cohesive storage change is
4.96e-14 J/m. Weak-load transfer defect is 3.11e-8 relative. The imposed new
surface contributes 203.732 Pa m shear and 250 Pa m friction load, distinguished
from overlap-normalization changes and transfer error. Native Gauss8/12
coefficients agree to 5.74e-15 relative; independent Legendre projection,
normalization and weak loads pass their predeclared 1e-11 screens. All nine
DGQ2 modes, the full plane-strain metric and unlimited history are retained.

Native-component tests cover identity, rejected candidates, deterministic
retry, single geometry publication and archives on both sides of the append.
Serial/two-rank owned and ghost DG data are checked; a separate two-rank process
continues the serial pre-append archive to the uninterrupted accepted record.
The final native suite passes 131 cases /165436 assertions; portable replay
verifies all 31 archived files and the independent mathematical checks.
The neutral manager append stages properties/widths/rates together and
invalidates projection/search caches. Configured boundary-completion paths
remain guarded; this horizontal boundary contact needs no completion wedge.

**Historical scope limit (native frozen integration is now qualified above):**
this original result is a synthetic component transaction using native
manager/archive, FEValues and MPI vectors. It is not a simulator outer-loop
finite-tip geometry transaction or a native material checkpoint conversion.
Existing finite-tip mechanical/evolution guards remain; no live Stokes cache
is exercised by this fixture. Coupled owner/cache integration, a general
nonzero-tip birth rule and phase activation remain unqualified. The next
native extension task must close those integration gaps before physical
propagation. P2's strict branch failure and all BP3 evidence remain preserved.

## Completed bounded P2 physical response

[P2_PHYSICAL_RESPONSE_REPORT.md](P2_PHYSICAL_RESPONSE_REPORT.md) and
[replayable evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/p2_response/README.md)
record the interval/criteria frozen before trajectories and the **passing bounded
physical-response qualification**. The same continuum pre-existing-damage seed
is sampled directly at 2×512 native cell vertices, with unloaded T/Z/slip and
the physical seed as obstacle. Initial relative phase/gradient L2 errors are
0.002769%/0.67223%; independent unloaded KKT and native unloaded phase drift
are zero. No independently relaxed seed, coarse-state continuation or physical
history reset supplies the growth reference.

The native ramp/hold and its single dt halving complete 76/152 accepted steps.
Measurable onset is .53125 s, peak .654399 at .5625 s and irreversible growth
.0543973, followed by no further phase/slip growth through .59375 s. Nonzero
stress and cohesive history decay consistently with the retained Maxwell law.
Maximum common-time temporal differences are phase Linf 6.31e-8 and full-tensor
stress 0.001187 Pa. The targeted 512-projected/1024-continuum spatial reference
gives whole/localized stress differences 0.294%/0.609%, phase Linf 2.52e-4 and
common phase-driving difference 0.0876%. Physical quadrature differences are
below these differences. All predeclared practical screens pass.

Absolute loading external-work remainders fall 0.006033→0.001461 J/m
(0.01287%→0.003116% of boundary work); separate mechanical/contact/phase path
works also decrease. Hold remainders are at roundoff. Closure alone is not used
to qualify work. Native gates 14/318/622, exactly 2/76/152 publications, 2666 raw
CSV identities and all 34 protected BP3 identities pass. All 858 production
source/header identities match entry; core, laws, normalization, acceptance,
limiter/default settings and prior evidence are unchanged. Same-core
rejection/retry/restart and native-unit evidence is reused within its prior scope.

**strict_reference_match_passed=false remains.** Native and matched reference
mirror centroids differ by about 1.7e-5 m; maximum unregistered phase difference
is 5.91e-5. These branch differences remain in every comparison and are not
counted as convergence. Original Gauss6, spatial and endpoint temporal/BP3
findings remain. This passes the selected early physical-response interval,
not general propagation or full PF5/temporal/spatial qualification.

This P2 result motivated the original **bounded P3 synthetic straight geometry
extension**, subsequently selected and tested above. Unchanged T does not in
general preserve tau when localization changes. Native finite-tip integration
and physical activation are not inferred from the synthetic transfer result;
no next implementation is automatically selected.

`../current_design.md` and `../specification.tex` retain scientific authority.
The derivation is at `../particle_free_history.tex`; local PLAN/PROMPTS files
are proposals, not required tracked instructions. Detailed changes, commands,
cache audit and the implemented master-space formulation are in
[SOURCE_AUDIT](SOURCE_AUDIT.md) and [PF6_HANGING_REPORT.md](PF6_HANGING_REPORT.md).
The complete preceding status is preserved byte-for-byte in
[STATUS_THROUGH_20261019.md](STATUS_THROUGH_20261019.md); its dated next actions
are historical, not concurrent development instructions.

## Development line and ownership

| Route | Purpose and accepted owner | Maintained entry/control | Evidence and limit |
|---|---|---|---|
| Older P2 native FE / CPDI | Recoverable total-energy cohesive endpoint experiment. Material owns accepted displacement history d_R; surface Z and physical phase are fixed incoming anchors during trials. Fracture uses CPDI. | `fe_strain.cc`, `fe_phase_energy.cc`, `reconstructed_fault_endpoint_phase.cc`; retained `propagation/run_native_endpoint.py` and native operator/growth controls. | [P2 closeout](../propagation/P2_CLOSEOUT.md): native onset/growth/saddle and bounded MPI checks; full trajectories chiefly prototype evidence. Not the starting backend for new phase development. |
| **PF6a domain-free DG / cell FE** | Independent material-owned DG T (27 coefficients/cell), cohesive Z=G s_elas, I and accepted physical phase. Native FE velocity supplies strain. Mesh/fault geometry and configured homogeneous materials replace particle domains. | Same endpoint solver; `dg_history.cc`, `fe_normalization.cc`, `fe_phase_energy.cc`, `simulator/phase_field.cc`, `dg_accepted.cc`; maintained `run_pf6_domain_free.py` / `pf6_transfer_checks.cc`. | [PF6a](PF6A_REPORT.md), [publication](PF6A_PUBLICATION_REPORT.md), [domain-free](PF6_DOMAIN_FREE_REPORT.md): operators, short evolution, rollback, transfer and same-mode restart. The bounded serial homogeneous domain-free hanging-node phase extension is verified below; unsupported routes stay guarded. Spatial/temporal accuracy and mirror-branch reference matching remain open. |
| Stationary mature DG / BP3 | Frozen prescribed phase and geometry; T and auxiliary Z=G s_mem advance. Cohesive resistance/storage stay zero. This Z is not cohesive elastic slip. | Mature branch in `reconstructed_fault_stokes.cc`, shared material/coupling, `core.cc` outer publication; maintained BP3 `coupled_checks.cc` and server `work_audit.cc`. | [Mature pilot](PF6_MATURE_PILOT_REPORT.md), [optimization](BP3_DG_OPTIMIZATION_REPORT.md), [observer repair](BP3_DG_WORK_OBSERVER_FIX.md). Stationary evidence does not qualify phase evolution or full-event accuracy. Running/server artifacts are protected. |

The shared type name `FEStrainState` does not make FE displacement authoritative
in DG mode: its velocity-history block stays zero, its phase block holds the
accepted anchor, and `dg_history` owns T. Material owns constitutive computation;
the manager stores surface properties/rates; the simulator owns acceptance.

For every cohesive trial, retain the same accepted T_n, Z_n and phase obstacle
phi_n. Form T_e=beta*T_n+2*kappa*D3(symgrad u), Z_e=beta*Z_n+kappa*V;
reconstruct tau=T_e-2*Z_e*h(phi)/I(phi)*S. The phase row varies the selected
bulk/cohesive energy and FE fracture at fixed endpoint T_e,Z_e. Its reduced
tangent includes mechanical response and both normalization derivatives/adjoints.
Keep the full embedded 3D plane-strain norm, all nine Q2 modes, physical
irreversibility and the existing energy/KKT/stability acceptance. A predictor
or converged inner trial is not history. DG publication stays pending through
the actual outer repeat decision; acceptance publishes once, rejection restores
the incoming anchor. No one-pass phase-first combination is adopted.

## Fresh bounded verification

Separate build: `build-particle-free/evolving-phase/native` and matching
`plugin`; no live BP3 build/package was rebuilt. Candidate executable SHA256
`4b36efad2177deea24bd1aeae81b113644a2e730cca6bd623b04e8600cec6e5b`.
[Replayable compact checks and identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/evolving_phase_consolidation/verification.json)
use the existing physical seed and fixture, Gauss12 mechanics / Gauss3 source,
two accepted dt=.125 s steps through .25 s, unlimited DG and frozen diagnostic
particles. Physical steps use 64 cells; the retained manufactured/identity
remaps do not advance time or histories. Surface geometry stays fixed.

- Entry and consolidated runs pass **100 native gates each**. All **162**
  compared CSVs excluding the timing CSV are byte-identical, including every
  exported phase/history/time level, stress, normalization, energy and weak load.
  Mechanical residual and phase-search traces agree exactly. Runtime is
  14.49 / 14.24 s; this small comparison is not a speedup claim.
- Phase grows by **.0189269852885 Linf**, with **zero healing**. Its peak remains
  .600001937763: this short control measures flank growth, not the later core
  onset or mirror departure. Final Z_max=6870.63404561 Pa m. The final native
  scaled phase KKT is 9.28e-17 against its unchanged 6.04e-14 target;
  integrated fault residual Linf is 4.12e-13 Pa m.
- Independent Gauss24/32 physical checks give maximum scaled energy/load errors
  **6.66e-16 / 3.15e-15**. Independent I/DI/D²I scaled errors are at most
  **9.62e-16**. Initial nodal seed error is zero, but the original coarse seed
  representation error remains **8.01966% relative L2**, separately reported.
- Existing actual rejection/retry and terminal inconsistent-source controls
  retain accepted histories/RNG/bookkeeping and once-only publication.
  A native step-1 checkpoint continued to step 2 matches **24** common
  pre-step/final exports, including RNG, byte-for-byte; final publications=2.
- Fresh native suite: **127 cases / 165,094 assertions pass**. Protection hashes
  verify **34** existing BP3/input/executable/plugin files unchanged. No server
  job was launched, submitted, stopped or otherwise controlled here.

Consolidation shares the existing fracture-Hessian addition and fixed-layout
profile derivative, retaining separate stability/search masks. It corrects
FE/CPDI-only comments/log labels and adds explicit binary/plugin paths to the
existing fixture runner. It adds no history owner, numerical operator, physical
parameter, limiter, acceptance rule or production default.

## Remaining limits and next task

**`strict_reference_match_passed=false` remains.** The opposite mirror branches
in [the 512-normal-cell confirmation](PF6A_PUBLICATION_REPORT.md#comparison-throughout-onsetgrowth-strict-target-miss-retained)
still miss the original strict unregistered phase/stress/driving targets. No
alignment, branch preference, tolerance change or replacement reference is used.
The original spatial/temporal findings and Gauss6 quadrature failure remain;
the new checks establish bounded functionality and entry equivalence, not
scientific convergence. Previous MPI evidence is reused within its scope;
this pass adds serial evidence only. Diagnostic particles still own fixed
points, matching-cell publication checks and native persistence/RNG; they do
not supply selected mechanics, material coefficients or fracture integration.

## Completed fixed local phase qualification

[PF6_HANGING_REPORT.md](PF6_HANGING_REPORT.md) and
[replayable verification](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_hanging/verification.json)
record the approved fixed 67-cell serial extension. Native closed homogeneous
convex Q1 rows define physical phi=C*psi on 86 masters (90 physical nodes).
Use native `AffineConstraints` expansion and complete raw C^T g / C^T H C
reductions once, including fracture, normalization and mechanical response.
Canonical physical obstacles, projected KKT/linear accuracy, positive roundoff
majorants, search metric and stability variations all use admissible masters;
artificial physical constrained-row diagonals are excluded. Physical bounds
and once-only acceptance/rollback/checkpoint ownership are unchanged.

The single refined patch now intersects the growing flank at
`0.5<x<0.75, 0.1875<y<0.25`. Hanging-node growth is 0.01917 in step one and
0.000638 in step two, while one hanging edge stays obstacle-active; no healing.
Cumulative nodal growth is 0.0198283. Independent gradient/fixed-action errors
are <=1.27e-14, re-equilibrated FD action errors <=2.33e-9. The full ten-free-master
Hessian has minimum eigenvalue 7.3572 J/m and symmetry defect 9.29e-17;
76 masters are strictly active. Native KKT <=2.405e-13 J/m. Gauss12 physical
energy/load controls pass against Gauss24/32 without changing normalization.

The final uniform control preserves all 162 non-timing CSVs exactly. Native
failure/retry, pending discard, terminal sampling rejection and single accepted
publication pass. Actual step-two controller cutback keeps accepted histories
unchanged and matches all three accepted states of the no-rejection sequence
exactly. Same-count serial restart and its parent prefix match 20 complete
flat/reference exports exactly. The final full native unit suite passes
129 cases / 165144 assertions. Core SHA256:
`686288cbbda7cb9793249a5d9dc31cd6f5ea8414737896c376bf71aefa090f80`.
All 34 protected BP3/package/build identities still pass; no server action or
package regeneration occurred.

A single dt halving gives final full-tensor stress relative L2 difference
4.13e-5 and physical phase Linf difference 1.43e-7. This is sensitivity evidence,
not temporal qualification. The seeded coarse field initially has a nonzero
projected phase gradient; early growth includes relaxation. Independent work
accounting closes algebraically, but external interval remainders 0.4563 and
4.05e-5 J/m remain and do not establish converged physical work. Keep
**strict_reference_match_passed=false**, all original spatial/endpoint temporal,
Gauss6 and mirror-branch findings. The report retains two failed diagnostic
attempts and their guard-preserving fixes.

The hanging-node and subsequent P2 physical-response tasks are closed. The
P3 recommendation above awaits selection. Retain representable-phase coarsening only.
MPI local-phase evolution, runtime AMR, lossy obstacle transfer, transport,
surface birth/remeshing, moving geometry, physical tips and coupled propagation
need later authorization and qualification. Unlimited DG, all limiter/filter
settings, constitutive laws, physical inputs and production defaults remain
unchanged. Stop after this bounded task.
