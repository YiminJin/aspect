> **Current selection complete, 4 November 2026:** fault-code readability and ownership refactoring on `47c2aae3be15d3ff60aab2a818c2f6e179077702`. Numerical `TensorField` is separate from material representation/residual and fault observations; material projection receives explicit inputs. All 183170 unit assertions, seven matched moving controls, stationary/serial/non-fault controls, analytic source checks and old/new checkpoint controls pass. Matched moving results are bitwise identical; new 2→4-rank restart agrees within 8.9e−16 bulk / 1.1e−14 surface. AStyle 2.04 is idempotent over all 44 primary files; all 57 protected hashes are unchanged. See [report](CODE_READABILITY_REPORT.md), [current guide](MATURE_MOVING_FAULT.md) and [provenance](../../../benchmarks/reconstructed_fault/particle_free/evidence/code_readability/final-provenance.json). Scientific campaigns remain paused; no new physical or production qualification.

> **Previous selection complete, 4 November 2026:** the ASPECT architecture refactor removes the rejected dedicated mature driver. Ordinary simulator/solver schemes own advancement and acceptance; tensor discretization, material laws and geometry have explicit boundaries. All 183170 unit assertions and seven moving controls pass; stationary/non-fault outputs and retained serial numerical results match their baselines. See [current guide](MATURE_MOVING_FAULT.md) and [report](ASPECT_ARCHITECTURE_REPORT.md). Defaults, checkpoint meanings and 51 protected artifacts are preserved. Scientific campaigns remain paused; this is not new physical or production qualification.

> **Previous selection complete, 2026-11-04:** behavior-preserving stress-history consolidation. Tested direct/publication baseline committed as `e4b74c556`; limiter/experimental relaxation and checker-only helpers moved out of core, stress access/projection/audit boundaries clarified. Full units and bounded constant/layered, default, rejection/restart comparisons pass. See [STRESS_HISTORY_CONSOLIDATION_REPORT.md](STRESS_HISTORY_CONSOLIDATION_REPORT.md). Scientific experiments remain paused; no new physical or production qualification. Cleanup losses remain.

> **Historical selection complete, 2026-11-04:** the frozen finite-end spatial/quadrature comparison identifies uncut DGQ2 approximation as the dominant local error. Higher quadrature alone is insufficient; native source and defaults are unchanged. See [FROZEN_FINITE_END_REPORT.md](FROZEN_FINITE_END_REPORT.md). Next proposed experiment is a frozen endpoint-aware tensor-space prototype with matching work integration, only on a new selection. No coupled/production campaign was resumed. Cleanup losses remain.

> **Previous direct-stress selection complete, 2026-11-04:** direct physical-history implementation, paired native comparison and lifecycle checks are complete. Retain the opt-in path as an independent reference pending finite-curve-end stress/traction accuracy; separated remains default. See [DIRECT_TRUE_STRESS_REPORT.md](DIRECT_TRUE_STRESS_REPORT.md). The broader previous accuracy/energy campaign remains paused; plans below do not authorize continuation. All production/cohesive limits and cleanup losses remain.

# BP3 review and objective stress implementation

Status: the selected fault-code readability refactor is complete and locally verified; scientific work is paused and broader Stage C remains open.
Prepared 2026-10-30. Update this document from actual repository evidence.

## Current execution record

**4 November frozen finite-end comparison completed:** one common64² Q1 profile,
labels/I and prescribed source on16²–128² tensor grids, ordinary q8–24 versus
cut integration, both constant and layered eta. The finite-end DGQ2
approximation dominates; more quadrature alone is insufficient. See
[report](FROZEN_FINITE_END_REPORT.md) and [protocol](FROZEN_FINITE_END_PROTOCOL.md).
Native code and defaults are unchanged. Next proposed work is a frozen
endpoint-aware tensor-space experiment with matching work integration, before
any coupled change; it requires a new user selection. Earlier next-step
recommendations below are historical, not automatic authorization.

**3 November positive-translation refinement completed:** two fresh 64² cases,
.05 m initial label spacing and dt=.025/.0125. Endpoint remainder decreases
8.7411% → 4.4151%, while raw boundary face work worsens. See
[report](POSITIVE_TRANSLATION_REFINEMENT_REPORT.md). Next is an independent
weak boundary-reaction/trace diagnosis using these saved states, not a further
trajectory. Native source, defaults, endpoint physics and protected data remain.

**3 November implementation follow-up completed:** implemented donor-bounded Q1
profile transport in the complete material transaction, specify accepted tensor
inflow and matching pure-translation mechanical/reference data, and compared
constant/layered cases in both directions with refinement and lifecycle checks.
See [protocol](BOUNDED_PROFILE_PROTOCOL.md) and
[measured report](BOUNDED_PROFILE_TRANSPORT_REPORT.md). This supersedes the
isolation entry's recommendation to implement the method; its evidence is reused.
The full-normal support/I defect is fixed and incompatible checkpoints are
rejected. Retain the bounded method as opt-in; profile improvement does not
close endpoint energy, coarse point-traction or surface-resolution limits.
Next is the two-run 64²/.05 m surface-spacing temporal control described in
STATUS.md, not an automatic continuation. The three-paper roadmap and deferred
production campaigns remain unchanged.

**3 November boundary/profile isolation completed:** the
[report](BOUNDARY_PROFILE_ISOLATION_REPORT.md) separates native inflow/front
resolution, Q1 repeated interpolation and localization projection. Five bounded
two-rank cases pass their operator/energy/replay screens with unchanged core.
The prior halfdt interior tensor error is localized near inflow; shrinking dt
alone is not the next accuracy strategy. Next is compatible coupled inflow or
buffering, joint h/dt refinement at fixed CFL and full coupled work/flux checks.
This selects no server, long trajectory or physical campaign.

**3 November narrowed Stage C continuation:** the
[coupled material report](TRANSPORTED_OBJECTIVE_STAGE_C_COUPLED_REPORT.md) now
supersedes the missing complete-transaction/heterogeneous-reaction statements
in the earlier entry below. The experimental mature moving state and coefficient
outer solve are implemented and locally tested, including rejection and MPI
restart. [Energy/tip diagnosis](STAGE_C_COHESIVE_DIAGNOSIS.md) separates numerical
remainders and identifies the first failed tip condition. General Stage C and
production qualification remain open; no physical campaign was launched.

**Historical first Stage C checkpoint, 3 November:** native mature T/R and cohesive cut reaction
owners, numerical transport operators, same-label material-geometry staging,
and bounded lifecycle/AMR checks are implemented and locally verified. See
[the measured report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md). Full suite
183170assertions/152cases passes. Stage C is incomplete: the complete native
material-step transaction, heterogeneous coupled reaction and moving transfer
remain engineering work; deformed cohesive shape work and the failed physical
phase/tip screens remain independent gates. No production trajectory is selected.

**2 November remaining Stage B selection and completion:** the
[moving/layered comparison](TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md)
passes its20 numerical and8 consolidation/profile/selection checks. Independent
Eulerian Q2 histories exercise all Stage A transport/profile/coefficient terms.
Complete residual DGQ2 is selected beside existing T/s for first native mature
integration; direct stress stays an independent reference. This completes the
bounded Stage B representation decision, not native production readiness.
Next is Stage C integration; no native guard, checkpoint or campaign changed.

**2 November Stage B first-part selection:** the [fixed-profile representation
comparison](TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md) is complete: independent
direct/separated Cartesian Q2, nonzero slip/zz and varying total spin, separated
time/space/quadrature studies. Fourteen selected-q8 gates pass; the failed q6
candidate is retained. No native selector/physics/checkpoint changes. Next is
the remaining Stage B moving-profile/heterogeneous-eta comparison, not Stage C
integration or a new campaign. The three-example roadmap remains unchanged.

**2 November Stage A selection:** the new transported-objective task supersedes
review-only future scope for its explicitly selected Stage A. Equations,
kinematic/state-owner contract and algebra audit are complete; see
[specification](TRANSPORTED_OBJECTIVE_SPEC.md), [acceptance report](TRANSPORTED_OBJECTIVE_REPORT.md)
and [task state](TRANSPORTED_OBJECTIVE_TASK_STATE.md). Current guards remain;
Stage B native representation qualification has not run. Preserve the same
three-example roadmap and incomplete-event/failed-screen/cleanup limitations.

**31 October computing-statistics follow-up:** [server-log timing and convergence figures](BP3_COMPUTING_STATISTICS_REPORT.md) are complete, with explicit cumulative-timer coverage and acceptance filtering. This is analysis only; no new campaign or performance experiment was executed.

**31 October plotting follow-up:** the separately requested
[server cumulative-slip figure](BP3_CUMULATIVE_SLIP_REPORT.md) is generated with
5-year/1-second targets and explicit coarse-data omissions. Original server
data and the prior source work remain preserved. This is read-only analysis,
not selection of the proposed next spin experiment or any production replay.

Entry is `27178813cebd471b93a7a9308f41215b71b4a20b` on the required branch,
without tracked edits. The exact supplement already occurs once in current
AGENTS; both setup documents matched the supplied package byte for byte. No
checkout/reset was created. Both qualified executables and the cohesive plugin
were copied and hash-verified before builds; shared runtimes remain available.

- Milestone1: [BP3 review](BP3_REVIEW.md) now includes the newly supplied
  `output_start/` and `output_resumed/`: accepted4237, scheduler stop during4238,
  latest saved4182 still accelerating. Exact missing records are listed.
- Milestone2: [variable/timestep/energy maps](IMPLEMENTATION_MAPS.md) trace mature
  production DG, the physical particle control, the new cohesive interface and
  the relevant older P2/PF6a distinction in one formulation table.
- Milestone3: [spin report](OBJECTIVE_STRESS_REPORT.md) derives total-material
  Jaumann spin and implements an opt-in first-order finite-rotation Lie update
  for matched-constant-material stationary mature physical particle stress.
  Analytical tests and the final183073-assertion unit suite pass. Bounded native
  pressure/residual, off/on, energy, rejection/retry, restart and 2→4 transfer
  checks pass; final guard-only changes reproduce the qualified fields exactly.
  DG/FE and cohesive selections are rejected, with the exact
  missing residual/native-remainder/structural-frame derivation stated.
- Milestone4: measured evidence and limits are reconciled in
  STATUS/SOURCE_AUDIT/design/specification. Protected binary copies and328
  cleanup-critical files verify unchanged. No server, long event
  replay, crust or propagation campaign is selected by this progress record.

The next justified experiment is a bounded fixed-profile comparison of direct
physical stress and the complete mature separated residual representation under
spatially varying spin. Establish the required history resolution before an
objective DG implementation or production spin branch. The three-example
roadmap below remains in force; no realistic subduction case is introduced.

## Scientific scope

Maintain one roadmap for the paper, with the three examples below. This thread
does not run all three examples. Its shared audit must be broad enough to avoid
fixing BP3 in a way that contradicts the intended viscoelastic/propagation model.

| Example | Scientific role | Current thread |
| --- | --- | --- |
| BP3 | External comparison of established-fault RSF behavior | Inspect current evidence, define actual benchmark fidelity, map implementation, and use bounded paired fixtures to assess the selected spin update. |
| Controlled viscoelastic crust | Stress redistribution, relaxation, and energy partition | Record requirements and dependencies only. Keep the proposed uniform `a=0.010, b=0.015`, 15–18 km viscosity transition and 18–40 km low-viscosity region as a design to finalize, not executed or calibrated settings. |
| Fault propagation | Coupled slip, damage, admission and persistent fault extension | Preserve current evidence/checkpoint and audit shared conventions. No new event continuation or propagation campaign in this task. |

The future crust comparison should distinguish original BP3, uniform-friction
effectively elastic control, and uniform-friction layered-viscosity case. It
initially uses fixed geometry/phase. The material and loading state below 40 km,
initialization, observation duration relative to Maxwell time, and boundary
conditions remain explicit design choices. No realistic subduction case is required.

## Entry: identify and protect the baseline

Read live `AGENTS.md` instructions, `RESEARCH_HANDOFF.md`, `STATUS.md`,
`SOURCE_AUDIT.md`, relevant `current_design.md`/`specification.tex` sections, and
the relevant formulation/verification documents. Use the handoff's links.
Inspect source before naming APIs; do not infer code from document prose.

Record root, branch, HEAD, dirty/untracked work, binary hashes, runtime
dependencies, configured builds, and active processes. The reported source
commit `af89a5bccf9c6771ff51b28f54c59b344cdf7cd0` is an identification reference,
not a required checkout. Some retained binaries predate it. Archive only the
small baseline executable/plugin/dependency set needed for comparison before
rebuilding in place; do not duplicate the whole worktree or all build products.

Protect the original unreviewed BP3 outputs and remote restart payload. The
handoff's local BP3 packets are not the full 160-rank restart. Protect the
separate RSF continuation at step 748, t=374.58003334330067 s, four ranks, unless
a documented newer accepted state is now present. Preserve its complete
restart generations and the authoritative server package.

## Milestone 1: determine what BP3 actually shows

Produce `BP3_REVIEW.md` under this directory, based on actual available data:

- Distinguish server jobs, source/build/input identities, and accepted clocks.
  Inspect the previous observer-frame and homogeneous-Maxwell coefficient
  repairs separately. Neither repair establishes event completion or accuracy.
- Report event onset, peak, subsequent decay, final accepted time, and any
  physical or operational stopping condition. State the diagnostic used.
  Do not import the small RSF-fixture decay threshold without justification.
- Inspect slip-rate histories and space-time fields, accumulated slip, state,
  shear/normal tractions, nonlinear iterations, timestep limits/rejections,
  matching-frame work, and publication/restart consistency as data permit.
- Audit the external BP3 comparison: geometry/dip, compressibility or Poisson
  ratio, initialization, deep creep/loading, evolving normal traction, domain,
  regularization, and resolution. Label material deviations and use
  “BP3-derived” when the mathematical problem differs. Do not tune parameters
  to conceal a mismatch.
- Separate event-time differences from event-progress differences, and a
  first-event finding from a statement about repeatable earthquake cycles.

If newer server results are unavailable, identify the exact missing exports,
logs, resolved input and source identities. Provide a short retrieval checklist
and state what remains unknown. Do not wait for remote data to start Milestones
2–3. No access to a remote machine is implied by a path in the handoff.

Completion: a bounded claim about observed BP3 behavior and an explicit list of
remaining evidence gaps, not a requirement to manufacture a successful event.

## Milestone 2: connect equations to implementation

Produce one readable `IMPLEMENTATION_MAPS.md`, with source file/function links.
Start with a small scope table covering the production mature fault, the newer
cohesive interface, and any still-relevant older evolving-phase path. Do not
rewrite every historical experiment or merge their equations.

1. Variable map: physical meaning, units, physical/auxiliary status, owner,
   coordinate frame, independent versus reconstructed status, update equation,
   source location, and checkpoint representation. Distinguish stress from
   `T_DG`/native remainder; mature slip memory from cohesive reference/relaxed
   jump; bulk phi from surface damage a; scalar slip from vector/tensor fields;
   solved p' from p0 and variational versus sided normal traction.
2. Timestep map: accepted state, immutable trial state, native FE gradients,
   Maxwell coefficients, transport, rotation, surface return/contact,
   mechanics/nonlinear iteration, damage/admission where enabled, estimator,
   rejection, commit, publication, and checkpoint. Mark disabled operations
   explicitly. A fixed-geometry diagnostic is not a transport qualification.
3. Energy map: branch recoverable storage, fracture inventory, friction,
   radiation damping, Maxwell dissipation, imposed work, boundary/history flux
   when present, backward-Euler numerical terms, and physical work remainder.
   Specify sign, measure, quadrature, time integration, and source function.
   Use energy defined by the actual independent state, not an assumed square
   of the blended stress. Count fracture and radiation contributions once.

Include one concrete timestep walkthrough with actual variable/function names,
written for the researcher who knows the equations but not the implementation.
Identify proven reductions between formulations and assumptions still needing
proof. A successful BP3 run does not validate surface-damage propagation.

Completion: the researcher can trace an accepted step and each plotted energy
term to its equations and source; unsupported equivalences are visible.

## Milestone 3: add and locally qualify the selected stress spin

Use `OBJECTIVE_STRESS_SPEC.md`. Perform its source/kinematic audit first, then
derive and implement the coherent opt-in pathway. This is an active feature
task, not merely a request to recommend future work.

Start with the shared constitutive/history machinery and production BP3 route
where the derivation supports it. Inspect the cohesive route for shared code
effects and frame compatibility. Enable it only if its full stress/history and
energy update can be derived and verified within this scope. Otherwise keep it
explicitly legacy/unsupported for the new option and state the precise missing
derivation. Do not hide partial rotation behind a global “objective” label.

Keep legacy/default inputs and binary comparisons intact. An option name is to
be selected after checking existing parameters; do not create duplicate
switches. Record scheme/spin/integrator identity in output and restart metadata.

Use bounded analytic/manufactured tests before paired BP3-like fixtures. Use
normally at most four local workers. Reuse configured builds, allowing for the
objects removed by cleanup. This task does not include production trajectories,
server launches, an AMR redesign, a new transport limiter, or a finite-strain
constitutive replacement. Routine derived implementation choices do not require
new permission. Fundamental scientific conflicts must be stated and isolated.

Produce `OBJECTIVE_STRESS_REPORT.md` with equations actually implemented,
changed functions, independent references, exact inputs, errors, convergence,
lifecycle behavior, cost, unsupported routes, and residual scientific concerns.
Explain effects on the available BP3 evidence without declaring old results
invalid solely because they used the older approximation.

## Milestone 4: close the review and select the next experiment

Update `STATUS.md`, `SOURCE_AUDIT.md`, governing design/specification sections
affected by the selected feature, and this plan. Keep historical reports intact.

Report separately:
- What the existing BP3 result establishes.
- What the new objective-stress option implements and what tests establish.
- Whether a new BP3 production comparison is justified, and its concrete scope.
- Readiness and remaining design choices for the controlled crust experiment.
- Preserved propagation status and its independent next task.

No whole-worktree cleanup is repeated. Keep essential test data and compact
evidence; do not regenerate removed historical animations. Follow the existing
commit/push policy; this package does not authorize a push or remote submission.


## Current Stage C execution (3 November 2026)

The user selected Stage C of CODEX_TRANSPORTED_OBJECTIVE_TASK.md. Follow
[the active execution record](TRANSPORTED_OBJECTIVE_STAGE_C_PLAN.md) and
[measured report](TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md), preserving the
three-example roadmap above. Native T/R and cohesive cut reaction owners,
load/tangent/observer paths, operator transport, rejection, restart and restricted
AMR now have local evidence. Stage C remains incomplete: material curve/profile/
surface transport and its complete accepted transaction, native heterogeneous
reaction, moving transfer and deformed cohesive work are still pending.
Do not promote a tensor-only or stationary reaction option to transported
production readiness. Retain failed uncut capacity, phase physical-energy and
tip line-search evidence. The next experiment is the bounded matched native
moving-mature transaction, not a server event or production continuation.
