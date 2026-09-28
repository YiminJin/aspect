# Re-Re-Design: Reconstructed-Fault Phase-Field Fault Model

**Status:** authoritative architectural specification for the restarted reconstructed-fault implementation  
**Supersedes:** earlier reconstructed-fault RSF redesign notes wherever they conflict with this document  
**Current implementation scope:** 2-D bulk + 1-D ordered reconstructed fault polyline  
**Primary design principle:** separate **fault geometry/kinematics**, **common phase-field fault mechanics**, and **replaceable fault friction laws**.

---

# 0. Authority and document hierarchy

Use the following authority order:

1. **`pf_rsf.tex`**  
   Authoritative for the continuum theory, notation, time-discrete Maxwell law, cohesive law, phase-field equations, degradation functions, and constitutive sign conventions after the note is revised.

2. **This file**  
   Authoritative for software architecture, ownership, discretization choices, lifecycle, MPI strategy, caches, numerical integration of \(I_h\), solver coupling, and staged implementation.

3. **Older redesign/plan files**  
   Informational only. If they conflict with this document, this document wins.

If the theory note and this file appear mathematically inconsistent, **stop and report the conflict before coding**. Do not silently choose one.

---

# 1. High-level architecture

The current architecture should be:

```text
MaterialModel::Interface
        |
        v
MaterialModel::PhaseFieldModel
        |
        v
MaterialModel::PhaseFieldFault
        |
        +-- common Maxwell bulk rheology
        +-- common cohesive law
        +-- degradation g(phi), h(phi)
        +-- I_h evaluation
        +-- V <-> upsilon transformation
        +-- common reconstructed-fault constitutive residual
        +-- B, G, K_V coefficients / coupling support
        |
        +-- Rheology::FaultFriction
                |
                +-- rate-and-state law
                +-- rate-dependent law
```

Meanwhile:

```text
ReconstructedFaultManager
        |
        +-- reconstructed geometry
        +-- ordered vertices / implicit Q1 segments
        +-- tangents / normals
        +-- projection half-width metadata
        +-- particle/QP association geometry
        +-- generic vertex-property storage
        +-- distinguished fault slip-rate field V
        +-- Q1 interpolation of V
        +-- V lifecycle: timestep-committed / Newton-current / line-search-trial
```

The Stokes assembler/solver remains responsible only for the unavoidable bulk-fault algebraic coupling.

---

# 2. Class responsibilities

## 2.1 `MaterialModel::PhaseFieldModel`

Keep this class **generic to phase-field material models**.

It may provide functionality such as:

- \(G_c\);
- critical crack-driving force / \(H_c\);
- valid phase-field range;
- generic phase-field material information required by the phase-field solver.

It must **not** assume:

- reconstructed faults;
- slip rate \(V\);
- Maxwell bulk rheology;
- cohesive surface traction;
- \(I_h\);
- a particular friction law.

Do not move reconstructed-fault mechanics into this base class.

---

## 2.2 `MaterialModel::PhaseFieldFault`

Rename the current `MaterialModel::PhaseFieldRSF` to a generic phase-field fault material model, preferably:

```cpp
MaterialModel::PhaseFieldFault
```

This class derives from `PhaseFieldModel<dim>` and owns the mechanics common to reconstructed phase-field faults, independent of the specific friction law.

It owns the **meaning and lifecycle** of:

- Maxwell time-discrete bulk rheology;
- cohesive traction law;
- current \(I_h\);
- previous committed \(I_h\) when required by the discrete cohesive law;
- \(h(\phi)=1/g(\phi)-1\);
- \(\chi=h/I_h\);
- \(V \leftrightarrow \upsilon\);
- surface traction residual;
- common parts of \(K_V\);
- coefficients entering \(B\) and \(G\);
- radiation damping if retained as a common fault-mechanics term;
- phase-field crack-driving feedback.

It uses a `Rheology::FaultFriction<dim>` member for the actual friction law.

### Important ownership rule

`PhaseFieldFault` may register material-specific persistent quantities in the reconstructed fault's **generic property storage**, but the manager remains constitutively neutral.

So:

```text
physical storage container:
    ReconstructedFaultManager generic property pool

semantic / constitutive ownership:
    PhaseFieldFault
```

Examples of material-owned fault properties:

- committed \(\Theta\) for rate-and-state friction;
- committed \(T^{\rm coh}\);
- previous committed \(I_h\) if persistence is required.

Temporary Newton quantities remain private working data in `PhaseFieldFault`.

---

## 2.3 `ReconstructedFaultManager`

The manager owns **geometry and fault kinematics**, not constitutive physics.

It owns:

- fault vertices and segments;
- adjacency / ordered-polyline structure;
- tangents / normals / segment geometry;
- influence strips and association geometry;
- generic fault vertex-property storage;
- distinguished nodal slip rate \(V\);
- Q1 interpolation of \(V\);
- \(V\) trial/current/committed lifecycle;
- geometry/cache invalidation;
- generic particle-to-fault projection utilities.

It must not know about:

- RSF parameters;
- rate-dependent friction parameters;
- \(\Theta\);
- cohesive law;
- \(g(\phi)\);
- \(h(\phi)\);
- \(I_h\);
- Maxwell rheology;
- radiation damping.

### Why \(V\) is special

\(V\) is a **fault kinematic unknown**, analogous to velocity in the bulk.  
\(\Theta\), \(T^{\rm coh}\), \(I_h\), \(\mu\), etc. are constitutive/material quantities.

---

# 3. Fault-friction abstraction

## 3.1 Do not reuse ASPECT's existing `Rheology::FrictionModels`

ASPECT's existing friction model is a bulk viscoplastic model:

- it works with strain-rate invariants;
- it returns friction angles;
- its dynamic law uses bulk strain rate as a proxy for slip rate;
- it does not represent a reconstructed fault with true \(V\);
- it does not perform the surface constitutive return relation required here.

Therefore do not force the reconstructed-fault formulation through that interface.

---

## 3.2 Generalize the existing `RateStateFriction`

Rename:

```cpp
Rheology::RateStateFriction
```

to:

```cpp
Rheology::FaultFriction
```

and generalize it to support more than one **fault slip-rate-based** friction law.

For now, use a simple internal selector rather than creating a large plugin hierarchy.

Suggested enum:

```cpp
enum class FrictionLaw
{
  rate_state,
  rate_dependent
};
```

The class should support at least:

### Rate-and-state friction

\[
\mu=\mu(V,\Theta)
\]

with one state variable \(\Theta\), a state evolution law, and the fixed-state derivative

\[
\left.\frac{\partial\mu}{\partial V}\right|_\Theta.
\]

### Rate-dependent weakening friction

For example,

\[
\mu(V)
=
\mu_s(1-\gamma)
+
\mu_s\frac{\gamma}{1+V/V_c},
\]

with no internal state and

\[
\frac{d\mu}{dV}
=
-\frac{\mu_s\gamma/V_c}
{(1+V/V_c)^2}.
\]

The generic solver must **not assume**

\[
\frac{d\mu}{dV}>0.
\]

A stateless weakening law may have

\[
\frac{d\mu}{dV}<0.
\]

---

## 3.3 Friction-law API

Keep the API small and centered on what the surface residual actually needs.

At minimum, `FaultFriction` should provide operations conceptually equivalent to:

```cpp
double friction_coefficient(...);

double friction_coefficient_derivative_wrt_slip_rate(...);

bool has_state_variable() const;

double update_state(...);   // valid only for a state-dependent law

double compute_time_step(...); // if the selected law supplies a stability/timestep restriction
```

Do not introduce a fully generic arbitrary-state plugin framework unless a concrete future friction law requires it.

For the current two-law design:

- `rate_state`: one state variable;
- `rate_dependent`: no state variable.

Avoid dummy \(\Theta\) arguments for stateless laws.

---

# 4. Maxwell bulk rheology

Do **not** use `MaterialModel::Rheology::Elasticity`.

The intended constitutive model uses the non-rotational time-discrete Maxwell law

\[
\beta=\exp\left(-\frac{\Delta t G}{\eta}\right),
\qquad
\kappa=\eta(1-\beta),
\]

\[
\boxed{
\tau^{\rm trial}
=
2\kappa\dot\epsilon+\beta\tau_{k-1}.
}
\]

Rotation/objective-stress terms are intentionally omitted because they are not part of the current theory.

### Ownership

- particle property plugin stores committed \(\tau_{k-1}\);
- `PhaseFieldFault` owns the Maxwell formulas;
- \(\tau_{k-1}\) remains frozen during all Newton and line-search iterations;
- temporary trial/current stresses are not written into particles;
- only after successful timestep completion is \(\tau_k\) committed to the particle property.

Recommended particle-property name:

```cpp
Particle::Property::MaxwellStress
```

Plugin name:

```text
maxwell stress
```

Do not create a transaction abstraction unless there is a concrete failure mode that requires one. Prefer one final post-convergence particle update pass.

### Coding location

Maxwell helper functionality specific to `PhaseFieldFault` should normally be private members/nested types of that class, not a broad `MaterialModel::internal` namespace.

---

# 5. Cohesive law

Assume the cohesive law is common to all friction laws supported by `PhaseFieldFault`.

Let

\[
\kappa_k=(1-\beta_k)\eta.
\]

The surface cohesive update is

\[
\boxed{
I_{h,k}T^{\rm coh}_k
=
\kappa_kV_k
+
\beta_k I_{h,k-1}T^{\rm coh}_{k-1}.
}
\]

Equivalently,

\[
\boxed{
T^{\rm coh}_k
=
\frac{\kappa_k}{I_{h,k}}V_k
+
\beta_k
\frac{I_{h,k-1}}{I_{h,k}}
T^{\rm coh}_{k-1}.
}
\]

The exact diffuse crack-strain-rate magnitude is

\[
\boxed{
\upsilon_k(\zeta)
=
\frac{
h_k(\zeta)T^{\rm coh}_k
-
\beta_k h_{k-1}(\zeta)T^{\rm coh}_{k-1}
}{
\kappa_k
}.
}
\]

This can be written as

\[
\upsilon_k
=
\frac{h_k}{I_{h,k}}V_k
+
\upsilon_k^{\rm hist},
\]

with a history correction when the phase-field profile evolves.

When phase field, \(I_h\), and history are frozen within a Newton linearization,

\[
\boxed{
\frac{\partial\upsilon_k}{\partial V_k}
=
\chi_k
=
\frac{h_k}{I_{h,k}}.
}
\]

---

# 6. \(I_h\) ownership and numerical meaning

## 6.1 Ownership

\(I_h\) belongs conceptually to `PhaseFieldFault`, not `ReconstructedFaultManager`.

Reason:

\[
I_h=\int h(\phi)\,d\zeta,
\qquad
h(\phi)=1/g(\phi)-1,
\]

and \(g\) is a phase-field material/degradation law.

The reconstructed-fault manager provides the **geometry of the profile**; the material model determines **what is integrated**.

### Separation

```text
ReconstructedFaultManager:
    where is the normal profile?

PhaseFieldHandler:
    what is phi(x)?

PhaseFieldFault:
    what is g(phi, composition)?
    what is h(phi)?
    what does the integral mean?
```

---

## 6.2 Current versus previous \(I_h\)

Current

\[
I_{h,k}=I_h[\phi_k]
\]

is a recomputable derived quantity and may be stored in a private transient cache.

Previous committed

\[
I_{h,k-1}
\]

is required by the discrete cohesive law and therefore needs persistent timestep-history semantics.

Do not treat current \(I_h\) as an independently evolved constitutive state.

---

# 7. Adaptive numerical integration of \(I_h\)

For each reconstructed-fault segment quadrature point \(s_q\),

\[
I_h(s_q)
=
\int
h\!\left(
\phi_h(x_\Gamma(s_q)+\zeta n_q)
\right)
d\zeta.
\]

Use the actual distributed Q1 FE phase field.

## 7.1 Profile quadrature points

Use fault-segment quadrature points rather than vertices so the segment normal is unambiguous.

A reasonable first choice is:

```cpp
QGauss<1>(3)
```

per reconstructed segment.

After computing \(I_h\) at fault quadrature points, project it to fault vertices with the 1-D Q1 fault mass matrix.

---

## 7.2 Initial panel width

Use a mesh-aware initial normal-panel width:

\[
\boxed{
\Delta\zeta_0
=
\frac12\min(\ell,h_{\rm local}).
}
\]

Do not use \(\ell/2\) alone.

---

## 7.3 Panel quadrature

For each proposed panel:

- evaluate 4-point and 8-point Gauss rules;
- accept when the quadrature difference satisfies the configured tolerance;
- otherwise bisect.

After a subdivided panel is accepted, propose

\[
\boxed{
\Delta\zeta_{\rm next}
=
\min
\left(
2\Delta\zeta_{\rm accepted},
\frac{\ell}{2},
\frac{h_{\rm local}}{2}
\right).
}
\]

The growth factor is an internal implementation choice, not a user parameter.

---

## 7.4 Tail termination

Do not use the ordinary phase-field activation threshold as the integration boundary.

Integrate outward independently in \(+n\) and \(-n\) until the **integral contribution** of outer windows becomes negligible.

Use an integral-based tail criterion, not a fixed \(\phi\) cutoff.

Do not require strict monotonic decrease of tail contributions.

A small numerical-zero phase-field threshold may be used only as an optional optimization after the integral-tail criterion is already satisfied; it must not define the constitutive support.

---

## 7.5 Multiple faults

The integral represents the connected local diffuse profile associated with the current reconstructed fault.

Do not integrate a second physically distinct reconstructed fault encountered farther along the same normal line.

For the current implementation:

- overlapping influence regions are unsupported;
- if a normal profile reaches another reconstructed-fault influence region before the local tail terminates, raise an explicit unsupported-overlap diagnostic.

Synthetic tests should distinguish:

1. a non-monotonic shoulder/bump belonging to one profile;
2. a separate second fault that must not be integrated.

---

## 7.6 Domain boundary

If the normal profile reaches the physical model boundary:

- bracket the first out-of-domain point;
- use bisection until geometric/floating-point convergence;
- integrate the remaining connected in-domain interval;
- terminate that side;
- do not continue into any disconnected re-entry region.

A maximum of 64 bisections is an internal emergency safeguard, not a physical parameter.

---

## 7.7 Failure guards

Failure guards are internal implementation safeguards and should not be user-facing parameters.

Use conservative fixed limits with detailed diagnostics.

Do not use a 10,000-round adaptive guard.

Prefer separate counters for:

- panel-refinement depth;
- outward-support extension count;
- boundary bisection count.

Suggested conservative emergency bounds:

- panel refinement depth: 64;
- boundary bisections: 64;
- outward accepted extensions: 256.

Actual convergence is controlled by numerical tolerances, not these guards.

---

## 7.8 Validation strategy

At each FE phase-field sample, perform one meaningful release-mode validation against the authoritative phase-field range:

\[
\phi_{\min}\le\phi\le\phi_{\max},
\]

including finiteness.

Derived values \(g(\phi)\) and \(h(\phi)\) are internal constitutive results. Use debug assertions for their expected properties where appropriate.

Do not repeatedly use `AssertThrow()` for every derived quantity inside hot loops.

At the end of a profile evaluation, verify that the final \(I_h\) is finite and usable.

If the constitutive model permits \(\phi\to1\) such that \(g\to0\), handle that as an explicit constitutive singularity; do not silently clamp.

---

# 8. MPI strategy for \(I_h\)

Profiles are independent and should be distributed across MPI ranks.

Use deterministic profile IDs:

```text
fault-major -> segment-major -> fault quadrature point
```

Partition profile IDs over ranks.

Each rank owns the adaptive state of its assigned profiles.

For each adaptive round:

1. each owner rank generates all currently required sample points;
2. all point requests are evaluated in one batched distributed phase-field point-evaluation operation;
3. owner ranks update quadrature/refinement/tail state locally;
4. repeat only for profiles that still require work.

Do not centralize all adaptive profile management on rank zero.

After profile integrals are complete:

- owners assemble unique fault mass-matrix/RHS contributions;
- MPI-sum the small fault-sized data;
- solve the replicated Q1 fault mass system.

---

# 9. Testing the adaptive \(I_h\) kernel

Keep the adaptive integration kernel private to `PhaseFieldFault`.

Do not promote it to a generic public utility unless a second genuine production caller appears.

Use a narrowly scoped friend/accessor for unit tests.

The private integrator may optionally emit a **test-only diagnostic trace** containing:

- panel bounds;
- refinement depth;
- low/high quadrature estimates;
- accepted/rejected status;
- contribution;
- accumulated integral;
- tail-window state.

Visualization/output code belongs entirely in unit-test code.

A test-only CSV writer is encouraged for manual diagnosis.

Production code should not gain a visualization parameter or output path.

### Automated tests should include

- smooth analytical profile;
- compact support;
- weak integrable tail;
- non-monotonic single-profile tail;
- separate second-fault encounter;
- near-singular profile;
- panel bisection and width regrowth;
- domain-boundary clipping;
- one-rank/two-rank equivalence;
- analytical stationary-profile \(I_h\);
- bulk-mesh refinement convergence;
- fault-surface projection convergence.

---

# 10. Surface residual with generic friction

The common surface constitutive residual is

\[
\boxed{
F
=
t
-
T^{\rm coh}
-
\mu(V,\text{state})\sigma_n
-
\eta^dV
=
0.
}
\]

The exact friction law is supplied by `FaultFriction`.

Current shear stress is

\[
\tau
=
\tau^*
-
2\kappa\chi V_\Gamma S,
\]

with

\[
\tau^{\rm trial}
=
2\kappa\dot\epsilon
+
\beta\tau_{k-1},
\]

and

\[
\tau^*
=
\tau^{\rm trial}
-
2\kappa\upsilon^{\rm hist}S.
\]

Normal stress is

\[
\sigma_n
=
p-\tau:N.
\]

Because

\[
S:N=0,
\]

the direct current slip correction does not change normal traction.

---

# 11. Friction derivative and nonlinear linearization

The fault Jacobian must use the actual derivative supplied by the selected friction law.

For rate-and-state friction, during the mechanical Newton solve use the **fixed-state / lagged-state** partial derivative:

\[
\left.\frac{\partial\mu}{\partial V}\right|_\Theta.
\]

Do not use the full derivative through \(\Theta(V)\) inside the Newton Jacobian.

Avoid the term `IMPES` for this algorithm. If desired, use:

```text
fixed-state Newton linearization
```

or

```text
lagged-state Newton linearization
```

Optionally `IMEX` may be used in comments if explicitly defined, but descriptive terminology is preferred.

For rate-dependent friction, use the exact signed derivative \(d\mu/dV\).

---

# 12. \(V\) variable and lower bound

Continue solving directly for \(V\), not \(\ln V\), for the first implementation.

Reasons:

- this matches successful/local-return-map experience;
- direct \(V\) makes Jacobian finite-difference verification easier to interpret;
- transformed variables become degenerate near a strict \(V_{\min}\) bound.

Ownership of bounds:

- `ReconstructedFaultManager`: kinematic invariant \(V\ge0\);
- `PhaseFieldFault` / nonlinear solver: physical/numerical bound \(V\ge V_{\min}\).

If a Newton direction attempts to leave the lower bound, use a small bound-active-set / projected-step treatment rather than switching variables.

---

# 13. \(V\) lifecycle

Distinguish:

\[
V^n
\quad\text{timestep-committed},
\]

\[
V^{(k)}
\quad\text{current accepted Newton iterate},
\]

\[
V^{\rm trial}
=
V^{(k)}+\alpha\delta V
\quad\text{line-search candidate}.
\]

Required semantics:

```text
start timestep:
    current = committed

Newton direction:
    trial = current + alpha * delta_V

line-search accepted:
    current = trial

line-search rejected:
    discard trial

nonlinear solve converged:
    committed = current

failed/rejected timestep:
    committed remains unchanged
```

Checkpoint/output use timestep-committed values.

---

# 14. Fully coupled bulk-surface Newton system

The exact coupled linearization remains

\[
\boxed{
\begin{bmatrix}
A & -B\\
G & -K_V
\end{bmatrix}
\begin{bmatrix}
\delta x\\
\delta V
\end{bmatrix}
=
-
\begin{bmatrix}
R_{\rm bulk}\\
R_\Gamma
\end{bmatrix},
}
\]

with \(x=(u,p)\).

Definitions:

- \(A\): ordinary viscoelastic Stokes Jacobian with current fault slip frozen;
- \(B\): fault-slip increment \(\to\) bulk residual;
- \(G\): bulk perturbation \(\to\) surface traction residual;
- \(K_V=-\partial R_\Gamma/\partial V\): fault constitutive Jacobian.

The condensed bulk operator is

\[
\boxed{
A_{\rm eff}
=
A-BK_V^{-1}G.
}
\]

Do not attempt to represent this exact coupling by a local `Tensor<4>` tangent modulus.

---

# 15. \(K_V\) with generic friction

Schematically,

\[
K_V
=
K_{\rm mechanical}
+
K_{\rm cohesive}
+
K_{\rm radiation}
+
K_{\rm friction}.
\]

The friction contribution is based on

\[
\sigma_n\frac{\partial\mu}{\partial V},
\]

and may have either sign.

Do not assume positive definiteness solely from the friction law.

Poor conditioning or singularity of the total \(K_V\) must be diagnosed explicitly.

---

# 16. Bulk-to-fault and fault-to-bulk operators

For fixed \(V\), the bulk perturbation entering the surface residual is

\[
\delta(t-\mu\sigma_n)
=
2\kappa(S+\mu N):\delta\dot\epsilon
-
\mu\,\delta p,
\]

for the current sign convention.

This defines \(G\).

A fault slip increment changes bulk stress by

\[
\delta\tau
=
-2\kappa\chi S\,\delta V_\Gamma,
\]

which defines \(B\).

The material model provides coefficients; the Stokes assembler performs the weak-form insertion.

---

# 17. Nonlinear convergence and line search

Use **separate normalized residual blocks**.

Convergence requires both:

\[
r_b<\epsilon_b,
\qquad
r_\Gamma<\epsilon_\Gamma.
\]

Do not use a raw combined norm and do not let the bulk residual alone control convergence.

For the fault block, prefer a mass-matrix-consistent norm such as

\[
\|R_\Gamma\|_\Gamma^2
=
R_\Gamma^T M_\Gamma^{-1}R_\Gamma
\]

or an algebraically equivalent strong-residual RMS.

For line search, use a scalar merit function based on fixed-scale normalized blocks:

\[
\boxed{
\Phi
=
\frac12
\left[
\left(\frac{\|R_b\|}{S_b}\right)^2
+
\left(\frac{\|R_\Gamma\|_\Gamma}{S_\Gamma}\right)^2
\right].
}
\]

Use fixed normalization scales during one nonlinear solve, with absolute floors to avoid division by an initially tiny residual.

Do not require each block to decrease separately at every trial step.

---

# 18. Coding-style rules

## 18.1 Public interfaces

- no speculative getters/setters;
- add a public API only for a concrete current caller;
- prefer narrow semantic operations over exposing whole containers;
- keep explicit `(fault_index, segment_index, xi)` interpolation until a real reusable `FaultLocation` type emerges.

## 18.2 Helpers

Create a helper when it:

- represents a meaningful operation;
- is reused;
- removes substantial duplicated or nested logic.

Do not create trivial one-use helper functions merely for mechanical decomposition.

## 18.3 `Assert` vs `AssertThrow`

Use `Assert()` / `AssertIndexRange()` for programmer errors and internal invariants.

Use `AssertThrow()` for conditions that must remain protected in optimized builds, e.g.:

- invalid user/runtime configuration;
- unsupported fault overlap;
- malformed persistent data;
- constitutive singularity;
- physically inadmissible runtime state;
- unsupported solver mode.

Do not repeatedly perform release-mode checks inside hot particle/QP loops when the condition was already validated when building the cache/data structure.

Validate once at the boundary; use debug assertions downstream.

---

# 19. Generic fault-property persistence

The reconstructed-fault generic property pool stores **committed physical/material state only**.

Temporary Newton quantities stay inside `PhaseFieldFault`.

Do not add trial/current/rollback semantics to every generic property.

Examples:

```text
persistent generic fault properties:
    committed Theta        (only for rate-state law)
    committed T_coh
    previous Ih if required

transient PhaseFieldFault working data:
    current recomputed Ih
    temporary T_coh(V)
    mu
    dmu/dV
    residual coefficients
    K_V factors
```

---

# 20. Geometry mutation

Current fixed-geometry stages may retain existing mutable fault access if necessary for compatibility.

Before propagation is implemented:

- topology-changing operations must be mediated by `ReconstructedFaultManager`;
- direct external `append_*()` paths must not be allowed to invalidate the invariant between geometry, \(V\), and generic property layouts.

---

# 21. Revised staged implementation sequence

The earlier stage numbering may already have partially completed work. Re-map existing commits carefully rather than blindly renumbering.

The remaining conceptual sequence should be:

## Stage A — finalize ownership / naming

- rename `PhaseFieldRSF` -> `PhaseFieldFault`;
- rename/generalize `RateStateFriction` -> `FaultFriction`;
- retain current rate-state behavior unchanged first;
- establish friction-law selector without changing the mechanical solver;
- keep all existing tests passing.

## Stage B — Maxwell stress history

- add `Particle::Property::MaxwellStress`;
- implement private Maxwell formulas in `PhaseFieldFault`;
- freeze \(\tau_{k-1}\) throughout nonlinear iterations;
- commit only after successful timestep completion;
- do not use ASPECT `Rheology::Elasticity`.

## Stage C — adaptive distributed \(I_h\)

- implement the distributed adaptive integration algorithm in this document;
- current \(I_h\) remains transient/recomputable;
- test synthetic profiles and MPI behavior;
- stop for review before constitutive coupling.

## Stage D — common cohesive state

- register committed cohesive traction;
- persist required previous \(I_h\);
- implement exact \(\upsilon\) / history correction;
- verify \(\int\upsilon\,d\zeta=V\).

## Stage E — generic fault friction

- preserve existing rate-state law;
- add rate-dependent law;
- verify both \(\mu(V,\text{state})\) and \(d\mu/dV\);
- verify stateless and stateful paths.

## Stage F — surface residual and \(K_V\)

- assemble generic residual with selected friction law;
- fixed-state derivative for RSF;
- exact derivative for rate-dependent law;
- diagnose singular/ill-conditioned \(K_V\).

## Stage G — \(B\) and \(G\)

- implement each action independently;
- verify signs and local-limit behavior.

## Stage H — exact condensed solver

- implement \(A-BK_V^{-1}G\);
- recover \(\delta V\);
- keep unsupported solver modes guarded.

## Stage I — coupled nonlinear lifecycle

- bound handling for \(V\);
- line search;
- separate normalized residuals;
- rollback / timestep commit.

## Stage J — history feedback / phase field

- commit \(\Theta\), \(T^{\rm coh}\), Maxwell stress;
- compute \(\upsilon\);
- compute/update \(H\);
- evolve \(\phi\).

## Stage K — fixed-fault benchmarks

Progression:

1. fixed phase field + prescribed normal stress;
2. fixed phase field + real normal stress;
3. evolving phase field without propagation;
4. only then propagation.

---

# 22. Mandatory Jacobian verification

Before BP3-style coupled benchmarks, expose a non-committing residual evaluator and verify:

\[
J\delta y
\approx
\frac{
R(y+\varepsilon\delta y)-R(y-\varepsilon\delta y)
}{
2\varepsilon
}.
\]

Test:

- velocity-only perturbations;
- pressure-only perturbations;
- fault-slip-only perturbations;
- mixed perturbations;
- nonzero cohesive history;
- rate-state friction;
- rate-dependent friction;
- one-rank and two-rank execution.

Also verify the condensed action directly against

\[
A-BK_V^{-1}G.
\]

Do not trust the coupled Newton system until these tests show the expected finite-difference convergence.

---

# 23. Deferred features

Out of scope for the current implementation:

- 3-D reconstructed faults;
- true branching;
- tip-to-side merging;
- general coalescence;
- overlapping influence bands;
- multi-fault weighting;
- closed fault loops;
- full RSF derivative through \(\Theta(V)\) inside Newton;
- a fully generic arbitrary-state friction plugin framework;
- generalized cohesive laws;
- optimization of the Schur complement before correctness is established.

Tip-to-tip merging may be designed later as a dedicated topology stage.

---

# 24. Short conceptual summary

The final architecture should obey:

\[
\boxed{
\text{fault manager}
=
\text{geometry + kinematics}
}
\]

\[
\boxed{
\text{PhaseFieldFault}
=
\text{common phase-field fault mechanics}
}
\]

\[
\boxed{
\text{FaultFriction}
=
\text{replaceable } \mu(V,\text{state}) \text{ law}
}
\]

The common transformation is

\[
\boxed{
V_\Gamma
\rightarrow
\upsilon
\quad\text{through}\quad
I_h=\int h(\phi)\,d\zeta.
}
\]

The friction law changes only

\[
\mu
\quad\text{and}\quad
\frac{\partial\mu}{\partial V}
\]

(and optional friction state evolution), while the Maxwell, cohesive, \(I_h\), \(B\), \(G\), and reconstructed-fault machinery remain common.

When in doubt, prefer this responsibility rule:

> **Geometry answers where the fault is.  
> Kinematics answers how it slips.  
> The phase-field fault model answers how diffuse deformation is related to that slip.  
> The friction law answers what frictional traction that slip produces.**
