# P1 production integration plan

Latest explicit user amendment: for the single straight through-going verification
fault only, use a fixed full-normal-cross-section association region derived from
the actual Box/fault geometry from initialization. Build associations before
cohesive data, preserve native H and accepted phase history, and serialize the
widths through existing manager state. The finite padded seed retains profile
widths and its separately reported diffuse cap. The affected P1 gate now passes;
[P1_integration_report.md](P1_integration_report.md) records execution, and the
subsequent [P2_report.md](P2_report.md) records the normalization failure. Profile-
width descriptions below refer to the original/default contract, superseded for
this selected straight case by the explicit amendment.

Prepared 2026-10-06 against `de7beaa65846c7372711f7922abf1c5d9a745ee6`,
branch `fault-propagation`, in the existing worktree. This is the approved source-audited
plan; execution and its support-coverage blocker are recorded in
[P1_integration_report.md](P1_integration_report.md). Preserve the incoming P1 work.
Implement and check this phase-only integration before starting P2.

## Decisions

| Item | Selected direction |
|---|---|
| Solver | Integrate the tested projected-Newton box solve into `PhaseFieldHandler`, using its current CPDI assembly and CG/AMG solve. Keep equality available for comparison. Initially select bounds explicitly in these evolving reconstructed-fault fixtures; retain existing defaults and the mature frozen-field bypass. |
| Initialization | Rounded profile is an uncommitted starting guess. Relax at the existing prescribed particle H, accept the converged field, then construct its dependent data. No extra seed constraint. |
| H | Retain `initialize_crack_driving_force()`, the activation/support transition, material mixing, particle property `crack_driving_force`, and the production history law. Never regenerate H from the relaxed phase. |
| Irreversibility | Bounds are exactly the existing material interface's `[0,1]`. No previous-phase lower bound. Measure incremental healing independently of KKT convergence. |
| Scope | Fixed geometry, no mechanical/RSF evolution, transport, or propagation during integration checks. Native phase-only restart moves into this P1 extension; physical-step rollback and mechanical restart qualification remain P2. |

The user-selected extension supersedes the earlier test-only limit for the
proposed integration; it does not discard other AGENTS instructions. Keep the
current paths. The design's geometry → H → phase → reconstruction ordering and
timestep-zero history rule remain authoritative. Record the new optional box
solver in the design/specification when implementing it; the older equality
description remains applicable to the equality option.

## Implementation boundary

Work primarily in `include/aspect/phase_field.h` and
`source/simulator/phase_field.cc`. Dispatch inside `evolve_phase_field()` so all
three callers in `source/simulator/solver_schemes.cc` use the same algorithm.
Carry over the prototype's active-set Newton step and KKT/energy line search;
evaluate energy through the same native coefficients and particle measure
(prefer an optional assembly output), without another constitutive formula.
All residual, Jacobian, roundoff and energy evaluations must use their supplied
scratch vector. Publish the phase block only after successful acceptance;
exceptions must not publish an intermediate Newton iterate or advance H.

Retain affine constraints and essential-value lifting. Apply box constraints
to independent phase unknowns and verify the distributed, constrained field.
Reject prescribed values incompatible with the physical bounds. Preserve
matrix symmetry when removing active rows/columns, including off-rank columns;
reuse the existing linear solver. Exchange active flags/ghost values and use
global reductions for stopping, line search, bounds and diagnostics. Do not
port the prototype's serial indexing assumptions. Uniform Q1 fixtures qualify
this bounded task; arbitrary AMR/3-D operation is not established by them.

Use existing linear/nonlinear tolerances and iteration controls; report the
actual absolute KKT target and roundoff scale. The initial production checks
must meet the prototype's `1e-11` KKT gate with CG relative tolerance `2e-7`,
at most 100 Newton iterations and 35 backtracks, Armijo coefficient `1e-4`.
Map the existing relative nonlinear control to the entry KKT scale explicitly;
do not reinterpret the equality residual at constrained contacts as a failure
or silently loosen the numerical gate. Reassemble the original operator for
free residuals, dual signs and complementarity after the modified linear solve.
With assembled RHS `r = -gradient`, valid lower contacts have `r <= 0`, upper
contacts `r >= 0`, and free entries `r = 0`. Include a small algebraic check of
both contact signs, release, and coupled row/column elimination: the prior
physical fixtures did not activate the upper bound.

## Initialization and state ownership

The exact startup sequence to preserve/complete is:

1. On the final initial mesh, initialize bulk fields, pressure, particles and
   domains. `Simulator::run()` calls `post_set_initial_state` after these bulk
   initializers; the manager's slot then initializes particle H. Initial
   adaptive cycles can repeat this fresh setup, so an earlier mesh's result is
   not the final accepted baseline. Use no initial adaptive cycles in the small
   qualification fixtures.
2. At the normal initial phase-solve location, form the rounded scratch guess
   from the existing prescribed profile utility. Relax using the already
   initialized particle H and unchanged phase boundary conditions. This guess
   belongs only to fresh initialization, never to an unchanged hold or resume.
3. Accept the relaxed phase into `solution` and synchronize its ghosts before
   reconstruction or material evaluation. Refresh the phase block of
   `current_linearization_point` for later consumers; keep other blocks intact.
   `initialize_current_linearization_point()` extrapolates from old vectors at
   timestep entry, so it cannot serve as the post-solve phase publication.
4. Call the existing `reconstruct_initial_faults()` once for this accepted
   field. Keep `Fit prescribed geometry to phase field = false` in these cases;
   retain prescribed geometry and prescribed-profile half-widths. The relaxed
   field does not recompute those widths in the current source. Audit actual
   particle and Stokes-quadrature coverage, distinguishing open-tip exclusion. Do not reconstruct again during the holds/ramps.
5. After initial temperature/composition setup is complete, initialize current
   normalization and cohesive history from the accepted live field. Reuse
   `initialize_cohesive_state_from_initial_fields()` and its validation/commit
   path. Currently it is reached through mechanical preparation; expose a
   narrow initialization entry usable by the no-Stokes path, sharing this code
   with mechanical preparation. Do not invoke a mechanical history commit to
   manufacture initialization. Compute every candidate before publishing the
   paired initial cohesive traction and previous-I_h properties.
6. Preserve the documented timestep-zero localization rule: current and
   previous *evaluation* both use converged phi. **Do not overwrite
   `old_solution` to implement this rule.** `advance_time()` already assigns
   both old vectors from the completed timestep-zero solution before native
   checkpoint writing. Later steps rotate the actual histories normally.

| State / owner | Required handling |
|---|---|
| `solution` phase block | Publish accepted relaxed/updated field; no subsequent initial-profile interpolation or whole-vector initial-field copy. |
| `current_linearization_point` phase block | Refresh after phase acceptance; prevent stale/extrapolated phi entering later assembly. |
| `old_solution`, `old_old_solution` | Keep simulator rotation/checkpoint semantics. Initial localization special case is in `constitutive.cc`; later old phase is genuine history. |
| Scratch trial/update, `system_rhs`, phase matrix/AMG | Transient numerical data; rebuild for the supplied iterate/active set. No checkpoint or physical-history meaning. |
| Particle H, composition, optional Maxwell stress and mapped state | Keep initialized/loaded values and representation; no phi-to-H back-projection, stress update, or invented RSF state. Current fixtures have H and initial composition, not a separate particle phase baseline. |
| Particle positions/domains, CPDI weights | Unchanged on this fixed mesh; phase-only changes do not rebuild particle geometry. Check IDs, properties and domain version. |
| Manager faults, support widths, projection metadata | Construct geometry after relaxation; widths are derived from prescribed stationary profiles, not the relaxed field. Preserve thereafter and serialize via the existing manager. Its reconstruction-complete guard prevents a second destructive initialization. |
| Fault chemical composition properties | `compute_normalization_integrals()` projects these first from unchanged particles; use them for the surface mixture and initial cohesive projection. |
| `current_normalization_integrals` | Recompute from accepted phi; validate finite positive values, including padded endpoints. Recompute after a ramp and after restart. |
| `phase field fault cohesive traction`, `phase field fault previous I h` | Initialize together once from relaxed phi/H and current I_h. Initial q uses native degradation and FE samples at particle positions. Preserve these committed properties on holds, diagnostic ramps and resume; subsequent physical updates belong to P2's terminal history commit. |
| `normalization_value_cache` | Publish only after successful integration. Its key compares owned phase entries collectively, geometry, surface composition and degradation revision; changed phi must miss. Check warm hold and cold recomputation. |
| Normalization point/cell lookups; manager particle/Stokes-QP projection caches | Geometry-only: reuse while mesh/domains/fault geometry are unchanged. Existing geometry-version and restart invalidation rules remain; do not confuse these with phase-value caches. |
| `current_fault_surface_temperatures`, optional Theta/V | Temperature is refreshed from finalized bulk data by its existing preparation. Phase-only cases use no RSF evolution. Preserve any initialized histories; fresh Theta/V initialization and solved V0 remain mechanical setup, never consequences of phase relaxation. |

The existing cohesive initialization guard returns when both committed
properties are already initialized. Therefore it must never run on the guessed
profile first: a later call would retain incompatible initial history.
An initialization failure must leave no accepted baseline/checkpoint with
partially initialized dependent properties. Retain normal and injected-failure
scratch-restoration checks; this is distinct from a physical timestep retry.

On resume, skip rounded-guess construction, H initialization and initial
reconstruction. Restore all three FE vectors, particles and manager properties
through the native snapshot path. The material's resume callback already
invalidates normalization caches; recompute evolving-field I_h rather than using
the mature frozen-field restart special case. Verify stored q/previous I_h
before the first resumed solve. No new persistent phase or H copy is proposed.

## Small verification gate

Rebuild the changed core in this worktree using the audited GCC/deal.II/Trilinos
stack; do not use the old sibling executable to claim production integration.
Reuse the existing benchmark runner/logging conventions and test signals.
The harness may prescribe H and observe state, but must call the actual
simulator timestep/phase path, not replace it with its own Newton loop.

| Check | Bounded execution / evidence |
|---|---|
| Straight + padded | One serial trajectory each on existing 32×32 / 64×32 meshes: fresh relaxation, identical-H hold, +5% and +10% controlled history stages with holds. Match retained nodal fields and phase changes, not only printed extrema. Record bounds, free/dual/complementarity defects, healing, particle H, initial q/previous I_h and current I_h. |
| Restart | Reuse the padded serial trajectory as uninterrupted reference. Write an ordinary checkpoint after the first +5% phase-changing increment, before a hold erases the old/older distinction; require distinct stored phase histories and resume the remaining hold/+10% schedule. Check restored state before a solve and continuation against the reference, including all FE history vectors and dependent data. Use native checkpoint timing after vector rotation, not a postprocessor-only norm as the whole reference. |
| Distributed | Repeat the padded trajectory on two ranks, including its checkpoint/resume, with phase-active support and particle domains crossing the partition. Compare by physical node/particle identity and replicated fault properties, not rank-local vector order; require collective KKT decisions and no hang. Same-rank-count restart only. |
| Compatibility / failure | Retain scratch-state poisoning and restoration checks; run existing equality and frozen I_h regressions (`phase_field_controlled_history`, `phase_field_fault_ih`) against the rebuilt core. Preserve their expected failing physical-qualification distinctions. Add the small contact/elimination check described above. |

This is two serial trajectories, one serial resume, one two-rank trajectory and
one two-rank resume, plus focused existing regressions; no new reference/mesh
matrix. Budget each new simulation at 120 seconds and 2 GiB per rank, compact
diagnostics under 10 MiB, checkpoint storage separately measured. Expand only
to diagnose a concrete failure. Record exact commands, source/build/input
hashes, runtime, memory and checkpoint size in the integration report.

Use the previous algebraic gates as fixed comparators, and report actual solver
targets. Raw nodal and CPDI bounds are checked before any observation-time
clipping. Support phase-error estimates with identical-H perturbed repeats;
the earlier allowances were 4.922e-12 (straight) and 1.298e-11 (padded), not
universal new tolerances. Compare serial/MPI/restart differences against
measured algebraic uncertainty; normalization uses its separately reported
quadrature/tail tolerances. Failure or unresolved uncertainty blocks a pass,
not a reason to increase tolerances. Report solver convergence, resolved
evolution, bounds and healing separately. Initial guess relaxation is not
incremental healing; significant post-acceptance healing blocks unconditional
qualification even if box KKT converges.

## Accuracy question and P2 handoff

Keep physical parameters, sampling and nonlinear integration unchanged here.
The reference measured `L_tail = 0.0139754`, `h/L_tail = 2.236, 1.118`, and a
native H switch at `y=0.188664692` versus sampled `y=0.1875`. Bounds establish
admissibility of that sampled problem; they do not establish fidelity of H to
the intended field. Retain this accuracy question with [the reference evidence](P1_bounds_report.md).

After this integration gate passes, proceed to the requested P2: fixed
through-going geometry, evolving phase, and actual mechanical H. Preserve the
audited cycle `H_(k-1) → phi_k → mechanics_k → committed H_k`; no outer coupling
iteration is introduced. Compare the same short loading interval with smaller
increments and instrument particle driving increments, committed H and the
next phase response. The central unresolved question is whether mechanical H
supplies the correct feedback, including the historical loss at small
increments. P1's prescribed monotone H cannot answer it. P2 also retains
physical rollback, coupled restart and transported-history checks from PLAN;
no tip extension is part of either stage.

Planning validation: reviewed the specification, phase solver/callers,
initialization/vector rotation, material/manager caches and serialization,
and existing restart fixtures. No new numerical tests or production edits.
Before changing rolling docs, saved their prior bytes at
`build-propagation-p1-bounds/PLAN.md.before-integration-plan` and
`STATUS.md.before-integration-plan`; both match the old bounds manifest.
