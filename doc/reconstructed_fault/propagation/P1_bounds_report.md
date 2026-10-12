# P1 reference and test-only box constraints — 2026-10-06

The requested bounded follow-up is complete. The straight-band reference
identifies a sampling contribution to the negative equality tail, independently
of CPDI nonlinear integration. The test-only box-constrained solver produces
admissible equilibria for the straight band and padded finite seed. Both pass
free-residual, complementarity, bound, repeat and resolved-evolution checks;
no incremental healing is resolved above numerical uncertainty in these runs.
The production equality solver remains unchanged and unqualified for physical
bounds. This is not a production algorithm adoption or a P2 qualification.

## Scope and provenance

The latest user request explicitly authorizes a bound-constrained diagnostic,
superseding the earlier prohibition only inside the test harness. Root AGENTS
retains its committed prefix and earlier task instructions, with this selected
extension appended. Relevant current-design §§2,13, specification phase-range
and API rules, refactoring guidance and current status were reread. There is
no selected change to the production scientific design.

Worktree `/home/ein/repository/aspect-fault-propagation`, branch
`fault-propagation`, HEAD `de7beaa65846c7372711f7922abf1c5d9a745ee6` are unchanged.
All incoming P0/P1 code and evidence were retained. The read-only sibling R7b
`aspect-r7b-qualified` has SHA256
`ed74d4d4ad1cfa5d1bd4fbade496d9748ab298c24a259a4612a94d4df375b60d`, source
`6a3781277f2f594d52cd248d84f5b6fdc98fdeb1`; its production source/headers match
HEAD. GCC 12.4.0 C++, OpenMPI 5.0.6, deal.II 9.6.2, Trilinos 14.2/Epetra,
Voro, Release were reused. New build: `build-propagation-p1-bounds`, test plugin
only, `-fno-finite-math-only -ffp-contract=off`. No core rebuild or other-worktree
write. Exact source variants, inputs, binaries, commands and artifact hashes
are in the [manifest](../../../benchmarks/reconstructed_fault/propagation/bounds_evidence/manifest.json).

All controls retain AT1/Lorentz p=1, m=128, ell=0.15625, prescribed core 0.8,
two identical materials and the existing 3×3 reference-cell particles/CPDI.
Mechanics, RSF evolution, advection and propagation are disabled. No physical
parameter, production default or particle-density control was changed.

## 1. Straight-band reference resolved first

The independent 1D Q1 weak-integration reference runs on y in [0,1] with the
same natural zero-flux boundaries. It uses the native profile and calls the
native degradation/geometric functions and material coefficients through the
existing test friend. No constitutive formula is copied. A direct tridiagonal
Newton solve is independent of production CG/AMG. There is no center-value
constraint; the native profile is only the initial guess.

Two different H representations are kept explicit:

- **Particle-domain H:** the actual initialized particle values, held constant
  over each reference-cell particle strip. This is the same sampled data used
  by production CPDI. For this unadvected aligned fixture, the 1D three-midpoint
  rule reproduces its assembly. Agreement with the actual 2D solve checks that
  reduction independently.
- **Native pointwise H:** the native initialization rule evaluated at each
  quadrature point, including its activation-threshold switch back to baseline
  H. This is the underlying unsampled input, a separate reference; it is not
  silently substituted for particle H in the production comparison.

The particle values are checked against native initialization at their actual
locations. Composition is read from the particles and verified uniform.
Integration intervals are split at every relevant H discontinuity. The fine
reference meshes also include those discontinuities as nodes. No smoothing,
H floor, fictitious continuation or altered boundary condition is used.

For the intact tail, linearization of the native operator gives
`L_tail = sqrt(F / K_phi)`, where `F=2 E ell²` and
`K_phi=H g''(0)+E alpha''(0)`. Measured native coefficients are F=3.125,
K_phi=16000, hence **L_tail=0.0139754248593737**. With h=1/32 and 1/64,
h/L_tail is 2.2361 and 1.1180; the particle-strip widths divided by L_tail are
0.74536 and 0.37268. Neither production mesh finely resolves this tail scale.
The 8192-cell reference has h/L_tail=0.008735, about 114 cells per tail scale.

The geometric profile support ends at y=0.188351790134067. The actual native
H switch occurs at phi=1e-6, **y=0.188664692008851** (and its symmetric upper
edge). H outside is 0.5 and just inside is about 0.011114. In both sampled
fixtures the stripwise switch is at **y=0.1875**, a shift of 0.001164692
(0.08334 L_tail). The first interior sample is H=0.01165854 at h=1/32 and
H=0.01119879 at h=1/64. Thus refining these two particle layouts does not move
that strip boundary; it also changes the sampled interior H values.

The tabulated starting profile is not assumed to be the exact equilibrium:
the fine native-H solution differs from it by up to 7.906e-5 near the
activation edge. The actual H cutoff occurs before that profile vanishes.
Using the compact profile alone as the reference would miss this transition;
the reference includes it explicitly.

| Equality comparison | h=1/32 particle layout | h=1/64 particle layout |
|---|---:|---:|
| Actual CPDI minimum phi | -5.25097e-5 | -1.71362e-4 |
| Max 1D CPDI-reduction difference at actual center-line nodes | 5.11e-15 | 1.18e-16 |
| Same FE mesh, sampled H, split Gauss-8 minimum | -1.40368e-5 | -1.60156e-4 |
| Same FE mesh, native H, split Gauss-8 minimum | 3.39581e-13 | 1.25119e-10 |
| Fine sampled-H reference minimum, 8192 base cells | -1.52131e-4 | -2.02060e-4 |
| Fine native-H reference minimum, 8192 base cells | 2.20659e-10 | 2.20659e-10 |

Replacing the three-midpoint nonlinear reaction evaluation with split Gauss-8
on the **same mesh and same sampled H** changes nodal phi by up to 5.65863e-4
and 1.40372e-4. Diffusion integration is unchanged in this 1D Q1 reduction.
This isolates a nonlinear-integration contribution. The sampled-H fine
reference remains negative, so CPDI integration is not necessary for the
undershoot. Sampling the sharp H transition is independently consequential.
The larger negative tail on the second production mesh is therefore not
evidence by itself of a defective Newton solver or an asymptotic mesh trend.

The combined reference refinement/quadrature check uses 4096 base cells with
Gauss-4, then 8192 with Gauss-8. The maximum differences over the sampled
comparison line are 9.05e-7/8.90e-7 for sampled H, and 8.74e-7 for native H.
Outside the native activation edge, these reduce to 1.56e-7 for sampled H
and 7.75e-9 for native H. The negative sampled-H minima exceed the tail
refinement estimate by roughly three orders of magnitude. Native H gives no
observed undershoot on either fine mesh; this is not a rigorous pointwise
error bound or a proof of a general positivity theorem. Mesh and quadrature
were refined together, so their fine-reference errors are not separately
estimated.

All 14 reference solves take three Newton iterations. Acceptance requires
correction infinity norm <2e-12 and weak residual L2 <2e-9; actual maximum
correction is 5.77e-15 and residual 9.07e-11. The two production comparisons
also pass their fresh equality targets (5.59e-13 and 9.23e-13; residuals
4.36e-15 and 8.96e-15). The sign and scale conclusions are resolved relative
to these algebraic and refinement estimates. See the
[reference/constraint plot](../../../benchmarks/reconstructed_fault/propagation/bounds_evidence/reference-and-bounds.svg)
and [numerical summary](../../../benchmarks/reconstructed_fault/propagation/bounds_evidence/summary.json).

## 2. Bound-constrained diagnostic on the unchanged discrete operator

`bound_diagnostic.cc` uses production `assemble_phase_field_system()` and
`solve_phase_field_system()` through the existing friend. Only scratch
matrices/vectors are modified. Projected Newton identifies contact nodes with
admissible multiplier signs, solves the symmetric free-subspace system, and
projects line-search trials into [0,1]. Line search accepts decrease of the KKT
maximum defect or an Armijo energy decrease (constant 1e-4). The energy
observation uses the same native material functions and CPDI values/gradients.
There is no phi_previous lower bound and no change to the H law.

With production RHS r=-gradient, the independently checked conditions are:
free r=0; at phi=0, r<=0; at phi=1, r>=0. Multipliers are reconstructed from
the gradient signs; both phase-distance × multiplier products are checked.
Every final state is reassembled before checking free residuals, dual-sign
violations, complementarity, nodal bounds and raw CPDI bounds. The KKT target
is max(1e-11, production roundoff allowance), actually **1e-11** throughout.
Limits are 100 projected Newton iterations and 35 backtracks. Inner CG retains
the production relative tolerance 2e-7 and limit 1000. Matched equality solves
retain nonlinear tolerance 1e-8 and their own fresh residual targets.

The production residual, Jacobian and roundoff scale agree exactly before and
after poisoning live phi to 1.25 in the bounded runs. All subsequent assembly
paths use supplied scratch states while live phi remains poisoned. All runs
check bitwise restoration of live/current/old vectors, RHS, particle properties
and positions, particle-domain version and fault geometry/properties, normally
and after a real invalid-state evolve exception. These are diagnostic
restoration checks, not physical timestep rollback or restart tests.

An initial straight trial exposed a limitation of a free-only uncertainty
estimate near releasing contact nodes. The final correction estimate also
allows nodes with a wrong multiplier sign to move off their bound. A same-H
repeat after the +5% ramp starts from a clipped 1e-6 sinusoidal perturbation
of the accepted field, re-solves, and compares with that accepted field. The
perturbation changes only the solver starting guess, not H or physical history.

| Final diagnostic | Straight, 32×32 | Padded finite, 64×32 |
|---|---:|---:|
| Relaxation Newton iterations from native rounded profile | 3 | 8 |
| Committed relaxed nodal range | [0, 0.80847755] | [0, 0.80734068] |
| Maximum free residual across seven states | 2.11e-14 | 8.56e-12 |
| Maximum dual-sign defect | 7.03e-12 | 1.31e-18 |
| Maximum complementarity defect | 7.03e-12 | 8.54e-12 |
| Maximum nodal/CPDI bound violation | 0 / 0 | 0 / 0 |
| +5% max phase increment | 0.00215398 | 0.00578610 |
| +10% max phase increment | 0.00212318 | 0.00556093 |
| Max incremental healing on the two ramps | 2.233e-14 | 3.451e-19 |
| Same-H perturbed-repeat difference | 4.922e-13 | 1.298e-12 |
| Largest directional KKT correction estimate | 4.294e-13 | 1.298e-12 |
| Actual phase diagnostic allowance | 4.922e-12 | 1.298e-11 |
| Restricted prescribed-H result | Pass | Pass |

The phase allowance is ten times max(correction estimate, repeat difference,
machine epsilon). Repeat consistency requires repeat <= ten times the estimate
(with epsilon floor); both pass. Both ramps exceed 100 times that allowance.
The correction is an uncertainty estimate, not an inverse-Jacobian bound.
The tiny reported healing cannot be resolved above numerical uncertainty;
there is **no resolved incremental healing**, rather than a claim of exact
irreversibility. Significant healing or an inconsistent repeat would prevent
an unconditional restricted pass in the independent checker.

Lower contacts are exercised; upper contacts are not active in these states.
Contact counts fluctuate in weakly active intact regions under the perturbed
repeat; KKT defects satisfy the stated tolerance. This is not a general
active-set, upper-contact, MPI or production qualification.

The first padded attempt hit an overly exact diagnostic assumption `w>=0`
after reaching the KKT stopping condition. Measured minimum CPDI weight is
-1.02716e-33, and maximum partition-of-unity defect is 4.44e-16. The final
check allows only machine-roundoff-sized convexity defects, bounded by
`16 epsilon n_weights max(1,sum(abs(w)))`, and propagates negative-weight and
partition defects into a CPDI evaluation allowance (2.84217e-14 here).
Weights and phi are never clipped in this observation. The actual CPDI minimum
is zero. Appreciable negative weights or phase bounds still fail. This check
correction does not alter the operator, solver or physical tolerance policy.

## 3. Padded finite seed: relax, accept, hold and ramp

The seed remains x=0.2…0.8 at y=0.5 in [-0.5,1.5]×[0,1]. Its native rounded
profile is the starting guess. With unchanged native particle H, the box
solve reaches an admissible equilibrium without any pinned node, ridge/core
value, phase seed constraint or temporary constraint release. Maximum initial
relaxation decrease is **0.1769967**. This is adjustment of an uncommitted
starting guess, reported separately from incremental healing.

The relaxed vector is accepted as the harness's committed phase baseline.
The hold and identical hold preserve it exactly. The fixed loading mask
(0.1<committed CPDI phi<0.7) is constructed from that baseline. Particle H is
then updated once per prescribed load stage using the existing max-history
prescription; Newton and line-search iterations never advance it. Both the
bounded and equality branches receive **identical H**, and each continues from
its own accepted state. The +5% perturbed repeat leaves H unchanged; unloading
retains the +10% H. All recorded minimum H increments are zero (none negative).

At +5% and +10%, the equality branch still has minima -0.00639627 and
-0.00639780 and healing 4.10921e-6 and 3.97281e-6. Its residuals converge.
The bounded branch stays admissible and has no resolved incremental healing.
Consequently convergence, box admissibility and incremental history behavior
are demonstrably separate assessments. Monotone prescribed H does not qualify
the production mechanical H update.

“Commit” here is acceptance and reuse in the phase-only diagnostic trajectory,
including the next-step mask and increment comparisons. The harness restores
the simulator's original state on exit. No native checkpoint, mechanical
history publication, geometry/cache rebuild or restart claim is made. Production
initialization would need an explicitly reviewed integration of the relaxed
field with subsequent reconstructed-fault/constitutive setup before P2.

## Verification, cost and handoff

Four physical controls were selected. Two focused implementation repeats were
needed: the straight uncertainty/contact-release check and the padded CPDI
roundoff check. Six simulation executions in total (five completed, one padded
diagnostic assertion), plus one parameter-parsing failure before simulation.
Every attempt stayed below 120 s, 2 GiB RSS and 10 MiB output. The four final
controls took 5.13, 6.13, 5.28 and 5.98 s, with peak RSS 206, 246, 205 and
219 MiB. No cluster run or broad convergence campaign.

The independent evidence checker passes all 14 reference solves and the 14
final box-KKT states, plus their equality comparisons and restoration markers.
Four checker unit tests pass, including nine injected failure conditions;
significant healing and unresolved repeats are explicitly disqualifying.
Expected-output text is not used as a substitute for numerical checks.
Inherited unused-variable warnings in the included registration fixture remain.
The original equality-only tests and their failed qualification evidence are
preserved; they were not relabeled or relaxed.

Changed: the test-only coefficient accessor in `tests/phase_field_test_access.h`;
new reference/prototype C++, PRMs, CMake target, runners/checkers/plot and evidence
under the existing propagation benchmark; additive AGENTS clarification,
benchmark/documentation links, PLAN and STATUS. Production `source/`, `include/`,
root CMake and defaults are unchanged. No commit or push. Reproduction commands
are in [the benchmark entry](../../../benchmarks/reconstructed_fault/propagation/BOUNDS.md).

Stop before P2. The next proposed task is review of the bound-solve and
initialization integration contract, including whether and how the sampled-H
transition should be represented in production. These diagnostics do not
implicitly authorize production replacement, a new H law, phase-irreversibility
constraints, or more runs. Physical restart/step rollback, transport/birth
history, and the historical small-increment mechanical-feedback loss remain
specific P2 questions. P1 cannot answer that feedback-loss question with
externally prescribed H.
