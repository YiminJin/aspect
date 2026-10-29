# Particle-free history verification and implementation plan

## Objective and current boundary

Establish a particle-free implementation with traceable equations, accepted
history semantics, and local verification. Start from the propagation branch
in `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`.

The initial task in `PROMPTS.md` requests **PF0 and PF1**. Complete those phases
and report a concrete PF2 recommendation. This boundary permits scientific
review while the user studies the derivation; it is not a request to ask
permission for every edit or test. A later task may authorize multiple phases,
which can proceed through their technical gates without repeated confirmation.
Record that instruction in `STATUS.md`.

## Equations to keep distinct

Use the definitions in `particle_free_history.tex`. For constant matched
`G, eta`, let `lambda = eta/G`. The circle denotes the selected corotational
rate; substitute the actual non-objective derivative if preserving that law.

$$
\mathscr L_t[s]=\chi s^{\rm ext}\boldsymbol{\mathcal S},\qquad
\boldsymbol{\mathcal C}[s]=(\mathscr L_t[s])^\circ-
\mathscr L_t[D_{\mathcal F}s].
$$

The cohesive candidate is

$$
(\boldsymbol\tau^{\rm tr})^\circ+\lambda^{-1}\boldsymbol\tau^{\rm tr}
=2G\dot{\boldsymbol\varepsilon},\qquad
D_{\mathcal F}s^{\rm elas}+\lambda^{-1}s^{\rm elas}=V,
$$
$$
\boldsymbol\tau=\boldsymbol\tau^{\rm tr}-2G\mathscr L_t[s^{\rm elas}],
\quad \tau_s^{\rm coh}=Gs^{\rm elas}/I_h,
\quad \boldsymbol Q^{\rm geo}=-2G\boldsymbol{\mathcal C}[s^{\rm elas}].
$$

For mature evolution with `Q_geo = 0`, use direct stress transport or retain
the exact separated residual:

$$
\boldsymbol\tau=\boldsymbol\tau^{\rm tr}
-2G\mathscr L_t[s^{\rm mem}]-\boldsymbol\tau^{\rm res},\qquad
(\boldsymbol\tau^{\rm res})^\circ+\lambda^{-1}\boldsymbol\tau^{\rm res}
=-2G\boldsymbol{\mathcal C}[s^{\rm mem}].
$$

The auxiliary scalar obeys
`D_F s_mem + s_mem/lambda = V`; the trial stress obeys the equation above.
Do not infer the physical law from whichever representation looks smoother.
Nonuniform parameters, geometry birth, and evolving surface measures need
explicit treatment. Record unsupported cases as such.

## Phase map

| Phase | Deliverable | Evidence before advancing |
|---|---|---|
| PF0 | Source and dependency audit | Equations, state ownership, baseline, build/test entry points |
| PF1 | Independent mathematical checks | Correct signs/product rules; counterexample to unsupported closure |
| PF2 | Selectable fixed-profile history path | Native-source, mechanics, adjoint, and baseline-control agreement |
| PF3 | Particle-free phase integration on fixed geometry | Energy derivatives, normalization, bounds, quadrature convergence |
| PF4 | DG transport and limiting if needed | Transport convergence and correct commutator behavior |
| PF5 | Required geometry/mesh transfers | Remaps, phase transfers, restart/MPI, and stated geometric model |
| PF6 | Bounded coupled propagation comparison | Resolved-reference comparison and remaining particle dependencies |

PF0/PF1 can falsify a proposed simplification. That is useful progress. Later
phases should follow the evidence rather than force this initial architecture.

## PF0: source audit and baseline

Read applicable instructions and record propagation base commit, current
revision, dirty state, and actual worktree. Do not assume a remote branch tip
represents the user's local propagation work.

Create `SOURCE_AUDIT.md` with a compact table:

| State/operator | Owner and exact source location | Accepted update and initialization | Mechanics/phase use | Particle dependency |
|---|---|---|---|---|
| Bulk stress or native strain/displacement history | Discover | Derive from source | Trace | Classify |
| Cohesive moment/slip and RSF state | Discover | Derive from source | Trace | Classify |
| Phase/lower bound and any `H` | Discover | Separate physical history from diagnostics | Trace | Classify |
| Surface reconstruction, labels, extension, normalization | Discover | Include transfer/boundary completion | Trace | Classify |
| CPDI phase operators and geometry services | Discover | Identify shared infrastructure | Trace | Classify |

Identify trial/accepted publication, rejected-step rollback, restart/AMR transfer,
normal stress/background pressure, spin convention, relaxation coefficients,
and zero-cohesion behavior. Trace dependencies through shared managers;
counting visible particle loops is insufficient.

Write actual continuum and discrete laws next to the relevant derivation labels
(`eq:C`, `eq:masterT`, `eq:p2stress`, `eq:residual`, `eq:phasevar`,
`eq:variableG`, `eq:DG`). Classify matches as exact under stated assumptions,
discrete approximation, constitutive difference, or unresolved.

Discover an existing minimal propagation/native-history fixture and build
commands. Run one bounded baseline if the environment is ready. Record an
environmental blockage explicitly if not; source audit and lightweight checks
can still proceed. Select test/run directories and a concrete runtime budget.
Prefer a short serial fixture; do not discover performance by launching an
unbounded simulation.

Exit: source audit, dependency inventory, baseline or precise blocker, and a
minimal candidate design. No production behavior changes.

## PF1: independent reference calculations

Implement small executable checks in the existing test structure or an isolated
reference harness. Prefer available dependencies. Use independent analytic
solutions or characteristic integrals, not the same routine under two names.
Requirements are V1--V4 in `VERIFICATION.md`.

Check the constant-parameter Maxwell step, evolving-profile product rule,
rigid co-motion/rotation, material passage through a fixed profile, and the
two-profile mature-memory counterexample. Check finite stress-preserving remaps
and cohesive physical phase increments separately. Include a heterogeneous
coefficient example to expose extra terms, even if the first implementation
supports only constant parameters.

Inspect phase-energy derivatives at fixed incoming histories. Do not declare
the discrete projection or CPDI replacement verified before PF3. Historical
checks in the derivation are context; the new branch's results start as not run.

Report in `PF1_REPORT.md`: references, commands, measured defects, falsified
hypotheses, and a concrete PF2 change list with actual source paths. State
whether source already contains redistribution or the candidate introduces
a physical change.

Exit: reproduce identities and demonstrate the necessary nonzero-history case.
If the physical law remains unresolved, report alternatives and consequences;
do not begin production implementation by guessing.

## PF2: fixed-profile history and mechanics

After the implementation task is requested, add the smallest selectable path
for the audited model. Hold phase/geometry fixed, omit transport, and use constant
matched parameters. Reuse surface state and coupling where semantics agree.

Choose independent tensor storage retaining native FE strain. DGQ2 is sufficient
for affine Cartesian Q2 gradients, not automatically exact on mapped cells.
Document mapping, projection, quadrature, and the full plane-strain basis.
Preserve symmetry and trace.

Use the same constitutive update in Stokes, fault loads, and accepted history.
Compute `1-exp(-dt/lambda)` stably, for example with `-expm1(-dt/lambda)`.
Do not evolve accepted state repeatedly during Newton iterations.

Verify V5--V6, the fixed-profile portion of V9, and a disabled-path regression.
Measure weak loads and stresses, not only slip plots. If phase/geometry still
uses particles, call this a particle-free history prototype.

Exit: agreement with the appropriate independent reference and retained
baseline, including acceptance/rollback semantics.

## PF3: phase energy and integration without particle domains

Keep the reference surface fixed while varying phase. Replace CPDI integration
only after explicitly defining the target FE phase functional. Retain the
chosen projected normalization and differentiate through it, including the
projection/mass solve where applicable.

Implement energy, residual, tangent, bounds, and incoming-history semantics
together. For this no-advection stage, the phase lower bound is the accepted
phase in the same material configuration. Maintain admissible denominators and
document inactive/near-singular profiles. An arbitrary phase cap or filter is
not a correction to a failed derivative.

Use V7--V8. Refine quadrature independently of mesh resolution, then refine
phase/history spaces. Hold an adequate quadrature partition fixed within a
Newton differentiation test. Compare CPDI and FE as two explicit discrete
functionals; bitwise equality is not an expected gate.

Exit: consistent discrete energy derivatives and a controlled cohesive phase
change with the correct stress redistribution.

## PF4: DG transport, then limiting if justified

Use prescribed smooth velocity/localization before coupled feedback. Specify
normal flux, inflow state, source integration, and material-transport divergence
correction. Distinguish bulk from surface material/ALE transport. Separate
temporal, spatial, and quadrature errors.

Verify V3 in the actual discretization and V10 without a limiter. A stationary
fault does not imply a zero commutator. If the selected law needs residual
history, retain it or transport stress directly.

Add a limiter only after a transported-history case demonstrates the need.
A known narrow reconstructed `chi` alone is insufficient evidence. If reusing
the historical WENO code, first reproduce its constant-candidate rank issue and
DGQ2 mass underintegration, then inspect detector activation and degenerate or
negative weights. Use V11.

Exit: convergent smooth transport, correct noncommuting behavior, and documented
limiter benefit if enabled. This does not establish universal history smoothness.

## PF5: geometry, mesh changes, and persistence

Implement only geometry operations needed by the next bounded experiment.
Distinguish pure reparameterization/remeshing from physical phase/fault evolution.
Verify finite remaps and transported phase lower bounds. Audit surface-state
transfer, slip-direction orientation, boundary completion, and partition ownership.

New segments, junctions, overlapping tubes, and changing surface measures need
stated initialization and constitutive/geometric rules. If open, restrict the
test domain and report that limitation; do not invent nearest-node inheritance
as physics. Resolve required energetic terms before claiming general
moving-geometry consistency.

Use V9 and V12 for restart, AMR, and a small MPI comparison. Check all new
histories, not just final stress.

Exit: selected geometry operations preserve the intended state, and accepted
histories transfer and resume reliably.

## PF6: bounded coupled comparison and next decision

Choose one existing small propagation case with matched physical inputs.
Compare retained baseline, candidate, and an adequately resolved reference
where feasible. Record differences in FE functional, transport, source
projection, and history law instead of conflating them.

Measure traction, normal stress, slip/state, phase increment, work/energy defects,
nonlinear iterations, runtime, memory, and refinement sensitivity. Use V13.
Keep filters and parameters fixed in the initial comparison.

End with a dependency inventory: can a run initialize, solve, advance phase
and geometry, restart, and adapt with no particle manager? A successful history
replacement alone does not establish that condition.
Recommend the next scientific test. A full BP3 event, large cluster submission,
default switch, and removal of the old implementation are subsequent tasks,
not gates for this local plan.

## Evidence and maintenance

Update one report per phase and `STATUS.md`; do not duplicate the audit.
Record revision/dirty state, commands, inputs, reference, norms, tolerances,
observations, limitations, and next action. Keep small summaries versioned and
raw output in the declared ignored directory. Follow repository commit conventions.

Replace provisional paths/commands with PF0 discoveries. When source or the
mathematical reference changes, record which evidence is invalidated.
Do not rerun unrelated tests without a concrete reason.
