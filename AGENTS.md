# Codex instructions for reconstructed-fault development

The authoritative design specification for the reconstructed-fault
framework is:
    doc/reconstructed_fault/current_design.md
    doc/reconstructed_fault/specification.tex

For every task related to reconstructed-fault development:

1. Read the relevant sections of the specification before editing code.
2. Treat the specification as authoritative for scientific algorithms,
   architecture, ownership, and MPI design.
3. Treat the current source tree as authoritative for existing APIs,
   actual class/function names, and existing reusable infrastructure.
4. If the specification conflicts with the current implementation,
   stop and report the conflict instead of silently choosing a new design.
5. Implement only the stage explicitly requested by the user.
6. Do not introduce duplicate physical/numerical parameters when an
   existing ASPECT/phase-field interface already provides the quantity.
7. Reuse existing deal.II and ASPECT infrastructure where the
   specification requires it.
8. Do not use the core-phase-field algorithm for the reconstructed fault.
9. Do not use the old PhaseFieldRSF architecture as the design basis.
   Isolated numerical routines may be reused only where permitted by
   the specification.
10. Do not redesign scientific algorithms without explicit approval.
11. Inspect callers, tests, and relevant headers before changing an
    existing interface.
12. Run relevant non-destructive tests after implementation and report
    what was and was not tested.
13. Follow: `doc/reconstructed_fault/refactoring.md`
    for code-quality, assertion/validation, readability, and
    responsibility-boundary guidelines. Prefer straightforward scientific
    C++ whose structure follows the mathematical algorithm. Do not
    introduce additional abstraction merely to reduce line count or
    eliminate small amounts of duplication.

## Reconstructed-fault session recovery

- Read doc/reconstructed_fault/CURRENT_STATUS.md for implementation
  status, evidence, and unresolved questions. Treat suggested next
  tasks there as context. The user's current instruction determines
  the active task; do not automatically resume a scientific experiment.

## Refactoring workflow

For tasks explicitly designated as refactoring:

- Module boundaries are defined in doc/reconstructed_fault/refactoring.md.
  M1 particle domains/CPDI and M2 phase-field implementation are frozen;
  M3 reconstructed-fault infrastructure, M4 fault material physics, and
  M5 coupled mechanics are the active refactoring scope. Only the stage
  explicitly selected by the user is authorized; boundary-interface changes
  and prerequisite cleanup require separately selected tasks.

- Follow doc/reconstructed_fault/refactoring/refactoring_plan.md.
  The plan describes the roadmap; it does not authorize executing
  every stage. Perform only the stage/pass selected by the user.

- Confirm the working directory, branch, and local changes before
  editing. Keep source edits, builds, and outputs within this
  refactoring worktree or its designated directories. Do not modify
  the separate scientific-test worktree.

- Preserve numerical algorithms, parameter defaults, MPI ownership,
  state-publication semantics, and checkpoint compatibility.
  Report suspected numerical bugs separately from structural changes.

- Keep code movement, structural changes, formatting, and numerical
  fixes distinguishable in commits or diffs.

- When the user reports manual edits, reread the affected files and
  inspect staged and unstaged changes before continuing. Preserve
  unrelated edits.

- Use the agreed stage-specific verification. Record pre-existing
  failures separately; do not loosen tolerances or change expected
  numerical results to make a refactor pass.

- Finish the selected pass with a concise review report: revisions,
  changed responsibilities/interfaces, verification results, remaining
  limitations, and one proposed next task. Await the user's selection
  before beginning another pass.

---

# Fault propagation: task instructions

Installation: this is an additive task block for the dedicated propagation worktree's root AGENTS.md. Preserve existing repository instructions; do not replace them with this file. Read any applicable AGENTS.override.md and nested instructions as well. If the repository has a conflict, identify the conflicting text rather than silently overriding it.

## Objective and scope

Develop and verify evolving phase field in the existing ASPECT phase-field/RSF implementation, then add controlled propagation of one reconstructed fault tip. The current refactored code has not yet been established to work with non-frozen phase field. A successful frozen-phase-field BP3 run is not evidence of evolving-phase-field correctness.

Read doc/reconstructed_fault/propagation/PLAN.md and STATUS.md at the start and after context compaction. If P0 chooses a different directory to follow repository conventions, update these references together.

Execute only the stage or range requested by the user. The initial task is P0 only. Complete routine investigation, fixes, and relevant checks autonomously within an authorized stage; do not ask about each command. At the end of that scope, report evidence and the next proposed stage. A roadmap entry is not an instruction to run all later stages. Later user instructions may authorize a larger range.

## Baseline and ownership

- Establish the actual repository, worktree, base commit, branch, dirty state, instructions, build provenance, and relevant tests before editing code. pf-rsf-refactor is a historical branch hint, not a verified current base.
- Prefer a separate worktree and task branch based on the verified current refactored baseline. Preserve the existing BP3/refactor checkout, running jobs, outputs, and checkpoints. Do not reset, clean, rebase, or switch another session's checkout.
- Use a separate build/output directory for this worktree. Installed dependencies may be reused; do not reconfigure another worktree's build. Record compiler, dependencies, executable, commit, and input identity for reported results.
- Keep this a propagation task. Do not bundle architecture cleanup, header relocation, solver redesign, CPDI redesign, melt support, or broad changes to default parameters with it. A specific defect blocking the current stage may be fixed and documented.

## Scientific rules

- First verify the phase-field solve alone, then coupled evolution with fixed reconstructed geometry. No tip extension before these stages pass.
- Keep established AT1/Lorentz conventions, particle history, RSF, normalization, and coupling semantics unless a demonstrated defect requires a change. Confirm the definitions in source; do not implement a remembered equation without checking its convention.
- Separate phi, phase-field history, irreversible peak variables, reconstructed geometry, and constitutive history. Record which quantities are trial, committed, transported, or rebuilt. Do not accumulate physical history once per nonlinear iteration.
- Apply irreversibility consistently with the code's material/Eulerian history treatment. A fixed-node comparison across an advection step is not by itself an irreversibility test.
- Use the intact-material strength/driving rule for particles without a fault association. Do not invent mature-fault V or theta merely to activate intact material. Resolve the dependency between damage growth and new geometry explicitly.
- Treat the rounded initial diffuse cap as part of initialization. A phi threshold ahead of the seed is not sufficient proof of new propagation. Do not derive the mature fault normal from the cap's radial gradient.
- Preserve old histories on geometry changes; initialize genuinely new material once under an explicit birth rule. Rebuild affected transfer, quadrature, normalization, and solver data. Rejecting a step must also reject its trial geometry and histories.
- Keep physical crack-tip truncation distinct from computational-boundary completion. Do not fix small I_h by introducing an undocumented floor or fictitious fault continuation.
- Temporary activation masks, fixed directions, frozen inputs, and synthetic fields are permitted only as named diagnostic controls. Their results establish only the restricted behavior they test.

## Implementation and verification

- Reuse existing ASPECT code and test conventions. Prefer a small parameter fixture or existing test harness to new managers, frameworks, environment switches, or duplicated constitutive implementations.
- Keep production defaults and frozen-phase-field behavior unchanged unless the user explicitly requests a change. Add focused regression coverage when a scientific behavior or state transition is changed.
- Start with one small local case and bounded runtime/output. Use the repository's established build/run commands; do not guess parameter names. Increase testing only for a concrete unresolved risk. No production cluster run is part of P0–P3.
- Report measured numerical behavior, not just solver convergence. Separate supported conclusions, remaining uncertainty, and assumptions. Do not call a plausible image an independent physical validation.
- Preserve the established particle-density controls. Check insertion/history initialization if it affects the active stage; do not redesign particle handling as incidental cleanup.
- Keep diagnostics compact and targeted. Avoid large whole-domain particle dumps or dense cumulative-slip output unless needed to diagnose a specific defect.

## Handoff

Update STATUS.md with the active/authorized stage, exact baseline, changed files, commands, small result table, unresolved issues, and next action. Keep it concise and link detailed stage reports rather than copying logs. Update PLAN.md only when evidence changes the plan; record the reason. Follow existing repository/user policy for commits and pushes; this file does not grant additional publication authority.

## Selected propagation stage (user clarification)

The user authorized P1 after completion of P0. Keep this worktree and the
installed `doc/reconstructed_fault/propagation/` paths. Reuse the audited
sequential coupling, equality phase solver and H law unchanged. Only bounded
controlled-H tests and their documentation are selected; stop before P2.
The original repository guidance above and the additive task instructions are
both retained. Refactoring-only module freezes do not prohibit this selected
test-only investigation; no architecture or scientific-algorithm change is
selected. Solver convergence, resolved evolution and physical admissibility
must be reported separately. Prescribed monotone test H does not qualify the
mechanical H update. Significant bounds/healing violations preclude an
unconditional P1 qualification. Native restart/physical-step rollback and the
historical small-increment mechanical-feedback loss remain explicit P2 questions.

## P1 diagnostic extension (latest user selection)

The user explicitly selected a one-dimensional straight-band reference, a
bound-constrained test-harness prototype using the unchanged discrete operator,
and padded-seed relaxation followed by diagnostic commit/hold/ramps. This
authorizes box constraints in the test harness, superseding the earlier
no-algorithm-change restriction only for that diagnostic extension. Keep the
production equality solver, sequential coupling and H law unchanged. Isolate
0 <= phi <= 1 initially; report incremental healing separately. No mechanics,
RSF evolution, propagation or P2 work is selected. Any seed constraint must be
explicit and its continued model status reported.

## Production P1 and first P2 (current user selection)

The user now authorizes optional production box-solver integration and the
small phase-only serial/restart/MPI gate in P1_integration_plan.md, followed
immediately by the first bounded P2 mechanical-feedback comparison if P1
passes. This supersedes earlier test-only and stop-after-P1 restrictions.
Keep H and history semantics fixed, use accepted phase rather than the generic
predictor, checkpoint after the first phase-changing increment, and audit
retained prescribed-profile support. Stop for a scientific-design blocker or
after reporting the first P2 package; no P3 or propagation.

## Fixed cross-section support and initial P2 (latest selection)

The user resolves the P1 support blocker by selecting fixed full-normal-domain
association for the single straight through-going verification fault, from fresh
initialization and through native restart. Determine the width from actual
domain/fault geometry; preserve sharp endpoints and native H initialization.
Build associations before cohesive data. Keep the finite padded seed unchanged.
After affected P1 checks pass, execute the already-authorized initial P2 dt/dt/2
comparison with actual mechanical H, recording off-band H/damage. No mask, H-law
change, dynamic support admission, P3 or propagation is authorized.

## Normalization projection follow-up (latest selection)

The user selects frozen-field integrability and accuracy diagnosis, an I_h-only
row-sum lumped projection with consistent comparator, and consistent lifecycle/
restart handling. Retain all physical inputs and H law; no floor or phi cap.
Exact saturation that makes normalization undefined stops that trajectory.
Fresh P2 paths are conditional on finite, adequately represented normalization.
Update model/source-change reporting; no P3 or propagation is authorized.

## Bounded material study (latest selection)

The user selects fresh P2 runs with nominal core 0.6, fixed G/c/eta/ell/p and
Gc reference/primary comparison, at the three explicitly allowed modest ramp
amplitudes only. Use consistent I_h initially, retain numerical gates and
failed high-load evidence. Optional third Gc requires abrupt primary evolution;
refine dt once for a preferred evolving case. No physical-law changes or P3.

## P2 history semantics diagnosis (latest selection)

The user pauses material search and selects a derivation/source audit, native
fixed-phase traction-path tests, unchanged reference dt/dt/2 reruns and conditional
stress scaling. Derive any correction first; implement only if the existing model
unambiguously determines it. Otherwise report competing physical definitions for
review. No arbitrary rate conversion, accumulation into max H, new material sweep
or P3. The completed diagnosis identifies an unresolved physical definition;
production H is unchanged and scaling remains conditional on consistency.

## Recoverable-energy P2 revision (latest selection)

The user explicitly selects peak recoverable-energy driving, unloaded seeded
initialization, native local energy/path tests and three fresh coupled timestep
runs. Preserve legacy failed evidence and defaults. The local optional law and
initialization are implemented; the runs expose additional traction-transfer and
bulk/fault work-consistency limitations. No material optimization, P3 or scaling
claim is permitted before feedback convergence. The report distinguishes the
reduced constitutive energy derivative from full discrete coupling qualification.

## P2 coupling repair (current selection)

Retain recoverable-energy history and unloaded initialization. Repair surface
traction publication first, verify arbitrary nonconstant histories and fixed-phase
loading, then derive and verify the conjugate bulk/fault work operators. Preserve
legacy defaults, failed evidence, and the generic particle projector. Restore
evolving phase only after fixed-phase checks pass, using one common early-damage
loading history at three dt. Report normalization/profile work, the separate
phase-energy projection discrepancy, actual specification diff and remaining
approximations. No material tuning, P3 or qualified feedback claim without
corrected-coupling refinement.

## Direct P2 irreversibility (latest selection)

The user explicitly selects an immutable last-accepted-phase lower bound for
physical evolution, leaving initial relaxation on [0,1]. This changes the phase
admissible set and supersedes prior no-previous-phase-bound restrictions for this
optional comparison. Retain recoverable H, unloaded initialization, surface
elastic-slip history and bulk-conjugate work. Replay three failed entries at
unchanged H, then fresh a=0.256 trajectories at dt=.125/.0625/.03125 to t=1 if
replays pass. Keep normalization/profile work as separate questions; no P3 or
material optimization. Preserve old failures and numerical tolerances.

## P2 accuracy and energy assessment (latest selection)

Keep direct irreversibility and all selected constitutive/coupling options; mark
it successfully verified for the fixed-mesh fixture. Frozen accepted fields at
surface spacings .1/.05/.025 m isolate consistent-normalization accuracy, with a
provisional .5% maximum source-error target. Native zero-strain fixed-phase stress
paths isolate representation from repeated-transfer drift. Audit ordinary/profile
work, storage, losses and transfer energy on matched measures/time paths. Further
.03125/.015625 coupled paths are conditional on adequate accuracy and energetics;
a required governing change must be derived and reported before implementation.
No generic-projector change, material sweep, broad MPI/transport campaign or P3.

## Evolving endpoint comparison (latest user selection)

The user selects the missing evolving endpoint family in the existing total-energy
prototype at dt=.015625/.0078125/.00390625, using corrected intersection quadrature
and unchanged material/seed/loading. Reuse matching mean evidence. Compare actual
full fields, rates and slips at common physical times, with phase release and
mechanical work discrepancies and their cumulative magnitudes kept separate.
A focused additional endpoint refinement may resolve a concrete fast-tail/work
uncertainty. Optional native P2 integration using endpoint mechanics is conditional
on the comparison supporting converging physical histories and adequately small
mechanical error; there is no additional planning approval gate. Numerical KKT
passes alone do not establish temporal accuracy. Retain mean as a comparator and
preserve earlier failed evidence. Report widening of a prescribed band in one
free bulk mode; no fault propagation, material tuning or P3 is selected.


## Bounded P2 formulation/checkpoint review (latest selection)

The user selects an implemented-versus-proposed equation comparison using
pf_rsf.tex notation, an inexpensive frozen normalization extension at .0125 m
then .00625 m only if needed, and a readable settings/results/production inventory.
The .00625 frozen-field target passes; it does not qualify the full coupled model.
Keep direct irreversibility, constitutive options and the current baseline.
Stop for review. Defer a new phase law, coupled time refinement, material tuning,
stress scaling and P3. Preserve incoming manual notation edits and failed evidence.


## Isolated total-energy algorithm/prototype (latest selection)

The user explicitly authorizes one complete total-energy step derivation and
small isolated prototype: instantaneous recoverable energy, immutable accepted
phase lower bound, and general exponential Maxwell evolution after instantaneous
elastic profile redistribution. Derive the actual discrete normalization adjoint,
finite work/losses and publication; H+ is diagnostic, seed is prescribed damage.
The prototype selects averaged mechanical balance and records finite phase-release
slack explicitly. Production source/options remain unchanged. Complete the report
and stop; no native integration, material tuning or P3 is selected.

## Loaded total-energy prototype follow-up (latest user selection)

The user selects common-path endpoint/mean mechanical time comparisons with
phase frozen, then an unloaded constrained-stable evolving seed with the
prototype's free bulk mode and full surface slip system. Preserve energy and
inelastic-strain physics, material, failed preload controls and production.
Track instantaneous/cumulative relaxation release, actual forces/contact signs,
common-time/full-field/slip errors and reference sensitivity. An identified
surface-basis undersampling defect is repaired only in the prototype by common
intersection quadrature; affected comparisons are repeated without tolerance,
projection or material changes. Choose the implementation direction from the
measured trend. No production adoption, added damage kinetics/limiter, tuning or
P3 is selected. See doc/reconstructed_fault/propagation/P2_total_energy_time_comparison.md for the completed result.

## Restricted native FE strain (current user selection)

The subsequent user selection authorizes the restricted native accepted FE
internal-strain option. Define the authoritative variable and exact Maxwell
update; use it consistently in total-energy phase trials, mechanics and terminal
publication. Preserve particle ownership/defaults in the legacy path. Enforce
homogeneous matching coefficients, fixed mesh/geometry and no transport/birth;
reject checkpoint conversion. Verify nonuniform FE velocities, then compare
native endpoint results with the retained prototype and separate energy errors.
No P3 or material optimization is authorized.


## Bounded native P2 localization (latest user selection)

The user selects diagnosis of the native stalled KKT entry, a rich one-dimensional
ux(y) total-energy reference, native compatible-side/normal-domain controls and a
separate transfer audit. Retain physical parameters, accepted FE state, bounds,
normalization and defaults. Arithmetic evidence does not authorize tolerance
relaxation; the tiny exact polish resolves the root at its original target.
Negative-curvature evidence is separate from stationary convergence; no new
minimum-selection rule or damage kinetics is selected. Full-domain geometry
coverage is derived from the actual centered Box for these fixed controls.
Three unused projected-stress reads are bypassed only in the FE option; generic
mapping remains. The bounded assessment is complete; preserve failed states and
stop before new refinement, tuning, transport/MPI campaigns or P3. See
`doc/reconstructed_fault/propagation/P2_native_localization_report.md`.

## Plane-strain and bounded spatial/stability follow-up (latest selection)

The user selects the updated native-FE derivation audit, consistent3D embedding,
all tensor-helper consumers, existing trace-subtraction/pressure/friction checks,
algebra/manufactured general-beta tests, then a separate demonstrated FE-only
correction and small compatible/nondivergence controls. Spatial enrichment and
sampled free-curvature checks follow with normalization unchanged. This task is
complete; Q3's native3-point quadrature aliases stress, so no spatial/full-stability
qualification follows. Preserve earlier failures and defaults; no P3, kinetics,
material search, projector redesign or pressure reinterpretation is selected.
See P2_plane_strain_report.md and current STATUS/PLAN.

## Coupled quadrature and measured-cost P2 task (latest selection)

The user selects the fixed compatible height-one fixture to .625s, consistent
bulk quadrature in all coupled operators, cheap rule/Q2 mesh studies, matched
native Q2/Q3 and one warranted refined-Q2 comparator, new incoming-anchor
curvature checks, costs and notation clarification. The task is complete with
six normal points and unchanged physics/normalization/CPDI. Restricted phase/q/
slip targets are met by16x32 Q2; stress/storage/time/full stability/P3 remain
unqualified. Preserve three-point/failed evidence and default particle ownership;
no new phase law, material tuning, general projection change or P3 is selected.
See P2_bulk_quadrature_report.md and current STATUS/PLAN.

## Q2 normal-resolution verification (latest user selection)

The user selects Q2/Q1 with the corrected formulation and six-point shared bulk
rule, unchanged physics/history/normalization/CPDI/bounds/tolerances, dt=1/64 and
T=.625. Reuse16/32/64 and add128 normal cells, extending only for unmet stress/
storage criteria. Separate mean/oscillatory and localized stress, extrema/jumps,
storage decomposition and initialization/obstacle effects; use a same-physical-
profile frozen control. Select one native normal-refined mesh at fixed16
tangential cells and64 surface segments. Practical5% stress/storage plus prior
phase/q/slip targets are not temporal/full energetic/P3 qualification. Preserve
Q3 evidence; no production law changes, material sweep or P3 is selected.


## Fixed-spatial P2 timestep assessment (latest user selection)

The user selects the existing endpoint step on16×256 Q2/Q1, six normal points,
64 surface segments, fixed physics/normalization/history/irreversibility at
T=.625. Reuse1/64; fresh matching discrete1D1/128,1/256,1/512 and optional1/1024
for reference uncertainty, with identical initialization and interval-mean .256
loading. Compare common-time fields, common-interval rates, onsets, fixed stress/
storage scales, separate release/loss/work discrepancies and frozen controls.
A selected new native run is conditional on passing the accuracy/reference gate.
The bounded family completes but phase/traction gate fails; native is deferred.
No production algorithm, mathematical model, RSF, material tuning or P3 change
is selected. Preserve historical evidence and all incoming work. See
P2_time_resolution_report.md and current STATUS/PLAN.


## Evolving-profile temporal diagnosis (latest user selection)

The user selects fixed-state incoming/endpoint phase KKT diagnostics, profile-rate/
force/velocity work scaling, at most1/2048 and1/4096 prototype extensions, one smooth
prescribed-profile control and a boundary-reaction/lifting audit. Preserve current
physics, mesh, history, bounds and tolerances. A consistent-time-level candidate
must be derived with energy accounting before prototype implementation; its trials
restart from the one incoming accepted state and retain the previous physical phase
bound. The saved-entry endpoint candidate passes KKT but incurs a large positive
incoming profile-energy cost, so no physical history/native adoption is selected.
Stop for the redistribution/transition formulation decision; no compensating loss,
new kinetics, production algorithm or P3. Preserve current defaults and all earlier
accepted/failed evidence. See P2_profile_temporal_report.md and current STATUS/PLAN.

## Consistent-endpoint P2 onset assessment (latest user selection)

The user selects precise replay residual/profile-cost/external-work/stability
classification, a continuous energy identity with normalization adjoints and
separate imbalance/friction/phase powers, then a short coupled trajectory from
a verified consistent state at three nested dt. Immutable incoming FE R,Z and
last physical phase bound apply throughout trials; complete step commits once.
The LAST unchanged-seed checkpoint fails phase KKT, so three fresh unloaded
trajectories complete0–.5625s; earlier .5234375-checkpoint runs remain controls.
The fresh family shows smooth growth, positive sampled admissible reduced curvature and
decreasing residual work. This is isolated prototype evidence; native equations,
constitutive law, Q2 discretization, loading, bounds and tolerances are unchanged.
The old replay's incoming state is phase-inconsistent, not proved unstable.
No compensating dissipation, kinetics, material tuning, native adoption or P3.
Full-interval/fast-tail continuation requires a subsequent selected scope. See
P2_coupled_branch_report.md and current STATUS/PLAN. Preserve all earlier evidence.

## Coupled P2 continuation to .625s (latest user selection)

The user selects extension of the same prototype through the later interval,
retaining separate energy residual contributions and mechanically reduced
stability. Same-partition accepted FE states continue without history conversion
or changes to equations, spatial operators, loading, bounds or tolerances.
All three paths encounter an independently resolved admissible antisymmetric
saddle at .59765625s; no unstable trial is published. Common accepted fields and
residual works converge to .59375s, but the requested endpoint is not reached.
Preserve failed entries and all older evidence. Stable asymmetric continuation,
discrete pinning and a justified branch-selection rule are the next questions,
not authorized implementation here. No native adoption, tuning, kinetics or P3.
See P2_coupled_branch_extension_report.md and current STATUS/PLAN.


## P2 translation-mode / diagnostic branch task (latest user selection)

The user selects a continuum translation/obstacle/boundary derivation, deterministic
both-sign constrained minimization at unchanged incoming histories,256/512 normal
meshes and a comparable shifted placement with fresh consistent histories, exact-
bulk controls, physical mass-normalized curvatures and stable diagnostic
continuation through .625s. Keep equilibrium/history/irreversibility/work ledgers
unchanged and preserve saddles. A lower energy does not prescribe an admissible
jump or physical branch; closest-release uncertainty remains explicit. Only the
highest512-cell spatial controls receive new visualizations in this task.
This supersedes the earlier prototype stop only for this bounded diagnosis;
no production adoption, kinetics, material tuning, transport/MPI campaign or P3
is authorized. See P2_asymmetric_branch_report.md and current STATUS/PLAN.


## P2 common initialization / positional accuracy (latest user selection)

The user selects one accurately evaluated continuum physical seed/H reference,
matched512/511/1024 exact-bulk controls atdt=1/1024, one positive diagnostic
perturbation direction, full centroid histories and analysis-only alignment,
mass-normalized curvature, and full-interval512 FE spatial/error assessment.
No accepted history or obstacle is translated. Stop microscopic near-release
searches below the arithmetic allowance. Actual ASPECT compact VTU postprocessing
is authorized for one accepted test; output-only snapshot rendering must be
identified separately from native coupled qualification. Existing equations,
material, loading, bounds, normalization, failure evidence and defaults remain.
No production adoption, centering/kinetic rule, material optimization or P3 is
selected. See P2_matched_initialization_report.md and current STATUS/PLAN.


## Optional native consistent-endpoint P2 (latest user selection)

The user selects a complete matching512-cell FE reference dt1/1024 versus1/2048,
default-off restricted native endpoint algorithm, fixed-state operator checks,
and a bounded fresh native growth/cost comparison. Retain governing energy and
constitutive laws, shared seed,16×512 Q2/Q1, six normal points,64 surface segments,
fixed friction/pressure, bounds and tolerances. Every trial anchors the same
accepted FE d_R, surface Z and previous physical phase; commit once. Use sparse
operators/matrix-free tangent actions, admit saddle escape without modifying
energy/Hessian, preserve extra tangential modes and consistently followed
reflected histories. Full native640 steps is conditional on measured cost/gate;
no VTUs requested. Preserve legacy defaults and historical failures. No new
phase law, material tuning, checkpoint conversion, transport campaign or P3.
See P2_native_endpoint_report.md and current STATUS/PLAN.

## Native reduced-tangent / onset recovery (latest user selection)

The user selects analytic native reduced-tangent actions with actual normalization
derivatives, retained velocity/pressure/slip/tangential modes, fixed-state FD and
prototype comparisons, cost/conditioning and inner-action uncertainty, a fresh
onset trajectory through .53125 s within the initial one-hour budget, controlled
complete-step failure/retry, and an energy-qualified economical prototype schedule.
Keep governing equations, physical parameters, true outer KKT, bounds and accepted
history rules unchanged. Refresh phase-dependent operators; reuse valid linear
infrastructure. Preserve failed diagnostics and the coarse-prefix cumulative
energy limitation. No physical tolerance relaxation, new phase kinetics, full
640-step native campaign, material tuning or P3 is selected. See
P2_native_tangent_report.md and current STATUS/PLAN for measured progress.


## Native intermediate accuracy / targeted inverse (latest selection)

The user selects adaptive intermediate phase-Newton accuracy, one measured
preconditioner correction, fresh economical native onset, known-saddle escape
as a fixed-entry diagnostic, and a costed plan for one full .625s run. Governing
energy, actual Hessian/full native degrees, accepted-history ownership, physical
obstacle and final KKT remain fixed. The bounded task completes; full-run input
is prepared but not launched. No physical tolerance relaxation, compensating
loss, material optimization or P3 is selected. Future work follows the user's
selection, not automatic execution of the cost plan. See P2_native_solver_report.md.


## Bounded native MPI correctness and rank comparison (current selection)

The latest user instruction selects tasks 1–3 of the local MPI/cost request:
preserve the committed serial reference, establish endpoint MPI correctness,
and measure matched 1/2/4-rank fixed-state and complete-step controls, including
saddle escape and coordinated four-rank prepublication failure/retry. Keep the
16×512 mesh, physics, true energy/KKT acceptance, immutable FE/surface anchors,
irreversibility and defaults fixed. Correctness precedes MPI qualification.
Stop and report after task 3. Tangential-mesh screening, additional performance
optimization and the fresh full .625-s trajectory are deferred. No P3,
transport or checkpoint conversion is selected.

Tasks 1–3 are complete: original-mesh operator/growth/saddle controls pass on
1/2/4 ranks and coordinated four-rank prepublication failure/retry passes.
Qualification is restricted to these controls; substantial growth-step speedup
is not established. See P2_native_MPI_report.md. Stop here; the deferred mesh,
optimization and full-run tasks require a subsequent user selection.

## P2 single-pass comparison (latest selection)

The user selects a prototype-only comparison of the original fixed-incoming-
history phase-first update with the current consistent-endpoint reference,
using identical shared initialization, 512-cell Q2 mechanics, physical loading
and accepted histories at two successive dt. Retain separate incoming-phase KKT
and endpoint splitting defects, actual phase-release/fixed-profile work and
common reconstructed-path residual powers. No compensating dissipation, physical
law or acceptance change is authorized. The bounded comparison completes at
1/1024 and1/2048s: single-pass is cheaper but fails required field accuracy.
Native coupled/MPI work is preserved unchanged. See P2_single_pass_report.md.
Stop here; no native replacement, material tuning, extra MPI work or P3.


## Bounded cheaper endpoint search (latest selection)

The user selects retained early/saddle/late prototype screening with quasi-Newton
preferred and a fixed small staggered baseline; only a promising candidate proceeds
to fresh .625-s prototype paths at the two retained timesteps, then native growth
and saddle controls at1/4ranks. Preserve energy, physical equations, initialization,
mesh, actual coupled modes/Hessian, original KKT/energy acceptance and immutable
incoming d_R/Z/previous physical obstacle. Approximate inverse secants change
search only. Keep authoritative saddle checks and publish once. Preserve endpoint
references and incoming MPI/manual work. No full native launch, physical change,
material search or P3 is selected.


## Cheaper endpoint task — user stop (latest instruction)

The user terminates the expensive native screen and requests a report from data
at hand. Prototype fixed entries and full1024/2048families pass; native1rank core6
is interrupted before KKT acceptance/publication. The new search remains default
false and unqualified natively. Preserve all evidence and the Newton reference.
Do not launch the remaining4rank/growth/saddle checks or a full trajectory on
this selection. Stop after updating the report/status; no P3.


## Bounded reusable direct mechanics (latest selection)

The user selects a serial retained growing-state native surface Schur inverse,
using actual independent B/G, consistent pressure gauge and active constraints,
setup-inclusive batch cost/residual/action comparisons against FGMRES, and
operator-valid factor reuse. Keep the original outer Newton and all physics,
histories/bounds/acceptance unchanged. Complete growth then saddle only if the
measured cost is clearly favorable. No trajectory, QN experiment or P3.

The direct-mechanics task is complete: the cost gate, original-Newton growth
and retained saddle pass, with the default-off serial choice retained. The
latest clarification requests separate setup/repeat/reuse/memory/size metrics
and no reruns for missing data. Pure factorization and standalone entry peak
RSS were not recorded; report them as unavailable. Stop after this bounded
report; no trajectory, further MPI optimization or P3 is selected.

## P2 checkpoint and bounded closeout (latest user selection)

The user selects inspection/preservation, a concise evidence-based P2 closeout,
and commits of all ASPECT source/CMake changes plus necessary reusable
benchmarks/tests/docs. Preserve manual work and full local reproducibility.
Retain reduced Newton as the restricted P2 reference and FGMRES as default;
direct mechanics remains experimental, serial and default off. Unsuccessful
coupling experiments remain documented and inactive. Distinguish native
operators/retained entries/MPI/prepublication recovery from full prototype paths
and the shorter fresh native onset. Full native evolution, temporal/spatial
qualification, coupled restart, transport and propagation remain deferred.
No new quasi-Newton, arc-length, full-trajectory or scalability work is selected.
See doc/reconstructed_fault/propagation/P2_CLOSEOUT.md. Historical selections
above do not authorize automatic continuation after this checkpoint.
