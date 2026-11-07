# Native DGQ2 phase: implementation contract

Selected 6 November 2026 by `CODEX_DGQ2_PHASE_FIELD_ASSEMBLER_SOLVER.txt`.
This note records the implemented design within the admission limits below.
The report separates verified operators and positive coupled comparisons from
the reverse-direction/spatial checks deferred by the user.
The retained Q1 campaign remains evidence. Its MacCormack improvement proposal
is superseded for this task; physical stress and endpoint conventions are not.

## Space and ownership

The simulator's `phase_field` component is either historical continuous Q1 or
ordinary discontinuous Q2, selected through Discretization family/degree inputs.
There is one persistent phase vector. The phase handler owns its transport,
fracture operator and admissible set. Material history supplies accepted velocity,
constitutive driving and the joint geometry/history transaction. The accepted
parent remains immutable. Transport produces the candidate and, when growth is
supported, its lower obstacle. Mechanics and normalization read this same field.

## Material advection

On each affine Cartesian cell the weak rate is

    (w, phi_dot) = (grad(w).v, phi) + (w div(v), phi)
                  - <w, (v.n) phi_upwind>.

The full accepted native velocity is frozen through SSPRK3 subcycles. Five-point
Gauss–Lobatto tensor quadrature integrates the Q2 velocity/phase/test products exactly
on these cells (up to degree six in a transverse coordinate). Face traces use the same rule.
The four-point rule suffices for the exact Q2 mass matrix. Setting w=1 gives the discrete
material integral balance, including the divergence source and prescribed inflow.
A constant field satisfies the same discrete divergence theorem.

For the cell mean, the forward-Euler coefficient at a volume Lobatto point is
its positive volume weight times (1 + ds div(v)), minus its outgoing face weights
times ds (v.n)+. Choose ds so every such coefficient is nonnegative. With the
constant-state identity and bounded incoming traces this is a convex mean update;
apply this condition at all cells before the three SSP stages. The implemented
step uses a 0.8 safety fraction of this mean bound and rejects requests exceeding
65,536 subcycles; neither value alters the physical accepted timestep. For nonmatching
faces, also impose the positive-polynomial trace bound: outward face integral
<= (6/h_normal) times cell integral, using exact Lobatto3 integration of a
nonnegative Q2 polynomial in the normal direction. The sum of face maximum
outflow times 6/h plus max(0,-minimum volume divergence) bounds the mean loss;
the complementary upper-gap equation gives the upper bound. Split face rules
integrate the Q2 trace exactly. The limiter scales
about the quadrature mean, preserving it, and never repairs an invalid mean.
Bounds are certified over the whole cell using the tensor Bernstein enclosure;
subdivision tightens the enclosure without changing the represented polynomial.
The admissible range comes from material physics, including rejection of singular
localization at phi=1. No artificial epsilon cap or global renormalization.

## Fracture energy and constrained growth

Let A=Gc/(c0 ell), k=2 A ell^2. The DG gradient energy is

    1/2 sum_K (k grad(phi), grad(phi))
    - sum_F <{k grad(phi).n}, [phi]> + 1/2 sum_F <sigma [phi], [phi]>.

Add the existing integral A alpha(phi) and recoverable mechanical energy. The
residual and tangent are derivatives of this same functional, including both
consistency terms. Natural homogeneous Neumann boundaries add no terms; prescribed
Dirichlet data use symmetric Nitsche terms. Penalty depends on degree, normal
cell size and coefficient. No transport time mass enters this equilibrium law.

The constrained solver uses positive-basis coordinates: Bernstein coefficients
of phi-phi_adv are nonnegative, and coefficients of the upper gap are nonnegative.
This is a sufficient whole-cell obstacle condition, stronger than a sampled
pointwise constraint. Expansion and dual pullback must include every local Q2
mode. No post-equilibrium transport limiter is permitted. Growth may change
means. Nonlocal normalization derivatives and their mechanical work adjoint remain
part of the coupled functional. Unsupported constitutive combinations stay
explicitly rejected; an operator test is not moving-cohesive qualification.

## Evidence plan

First native operator and constrained manufactured tests, including nonzero
divergence, open flux, SIPG faces, nonconstant/moving obstacles and MPI. Then the
four selected fixed-schedule layered comparisons, fresh DG initialization,
rejection and rank-changing restart. No expensive run while cheap checks fail.
Retain source/build identities and the campaign's finite-domain, cohesive and
adaptive-controller limitations. No production run or old checkpoint conversion.

References: [ASPECT discretization](https://aspect-documentation.readthedocs.io/en/latest/parameters/Discretization.html),
[deal.II SIPG example](https://dealii.org/9.6.0/doxygen/deal.II/step_74.html),
[Zhang–Shu mean-preserving bounds](https://www.math.purdue.edu/~zhan1966/research/paper/maximum_principle.pdf).
The material source and whole-cell constraints above require their own checks;
the incompressible conservation-law theorem is not applied without adaptation.

## Source map and admission boundary

| Responsibility | Owner |
|---|---|
| Family/degree/limiter selection | `parameters.h`, `simulator/parameters.cc`, `PhaseFieldHandler::declare_variables` |
| Spatial mass, material weak form, flux, SSPRK3 | `phase_field/dg_transport.cc` |
| Certified enclosure and cell-mean scaling | `phase_field/dg_bounds.h` |
| SIPG sparse cell/face terms and Nitsche control | `phase_field/dg_sipg.cc` |
| Native fracture energy and all-mode derivatives | `phase_field/fracture.cc` |
| Native phase transport and one-sided point trace | `phase_field/native_phase.cc` |
| Bernstein coordinates and dual pullback | `reconstructed_fault/endpoint_phase_constraints.h` |
| Distributed constrained quadratic search | `phase_field/obstacle.cc` |
| Full nonlinear endpoint/KKT/energy acceptance | `solver/reconstructed_fault_endpoint_phase.cc` |
| Normalization and work adjoints | `phase_field_fault/normalization.cc`, `fe_normalization.cc`, `fe_phase_energy.cc` |
| Joint candidate/accepted lifetime | existing material history, geometry and simulator timestep owners |
| Phase-space checkpoint identity/all-mode packet | `checkpoint_restart.cc`, `phase_field_fault/dg_accepted.cc` |

The native selection is restricted to 2-D affine Cartesian mesh data and either
the mature transported path or the existing consistent-endpoint cohesive path.
CPDI and the separate bulk-cohesive prototype are rejected. The latter restriction
does not prevent the fixed-fault native cohesive check. Sparse phase rows and
vectors are distributed; the existing endpoint coordinate/gradient and normal-ray
geometry reductions remain replicated O(n) or O(n times surface labels). No global
dense DG phase Hessian is introduced into the production endpoint solve.

The growing path uses fixed 12-point normal-ray integration of nonlinear h(phi),
with identical nodes for values and derivatives, retaining all nine phase modes.
The native checker compares these values against independent 32-point integration.
The mature path retains its existing adaptive cell-clipped normalization.
The mean limiter permits only its explicit floating-point bound allowance; the
constitutive singularity at phi=1 is still rejected. Bernstein growth is stronger
than pointwise bounds and requires a feasible accepted lower polynomial.
