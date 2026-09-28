# Reconstructed-Fault Surface RSF: Authoritative Design for the Restarted Implementation

**Status:** design specification for the restarted Codex workstream  
**Implementation scope:** 2-D bulk model with a 1-D reconstructed fault (ordered Q1 polyline)  
**Primary goal:** replace profile-wise duplicated RSF state by one surface state per reconstructed-fault location, while retaining the diffuse phase-field representation for crack deformation and propagation.

---

# 1. How to use this document

This document is the **authoritative implementation specification** for the restarted reconstructed-fault work.

The existing code is potentially reusable, but it is **not an architectural constraint**. Before changing code, classify existing pieces as:

- keep unchanged;
- reuse with modification;
- replace;
- remove.

Do not preserve an old interface merely because it already exists if it conflicts with the mathematics below.

The present design deliberately changes the old local-return-mapping architecture:

1. RSF state is stored on the reconstructed fault, not duplicated across particles in a diffuse profile.
2. Cohesive traction is a surface history variable.
3. The current bulk stress depends on fault slip through a fault-to-bulk operator.
4. The exact Newton coupling is a bulk-surface block system (or its Schur complement), not a purely local quadrature-point tangent modulus.
5. The state-law derivative remains IMPES during the Newton solve; "fully consistent" means consistent with that IMPES residual, **not** restoring the unstable full `dTheta/dV` feedback.

---

# 2. Geometry and kinematics

Let the reconstructed fault be

\[
\Gamma_h = \bigcup_e \Gamma_e
\]

with ordered Q1 segments.

For a point \(x\) in the fault influence region, let

\[
\pi_\Gamma(x)
\]

denote its associated point on the reconstructed fault.

For a particle or bulk quadrature point \(p\) associated with segment \(e\), define:

- fault segment index \(e_p\);
- Q1 coordinate \(\xi_p\in[0,1]\);
- fault shape functions
  \[
  N_0(\xi_p)=1-\xi_p,\qquad N_1(\xi_p)=\xi_p;
  \]
- unit tangent \(s_p\);
- unit normal \(n_p\);
- tensors
  \[
  \boldsymbol S_p=\frac{\hat{\boldsymbol n}_p\otimes \hat{\boldsymbol s}_p+\hat{\boldsymbol s}_p\otimes\hat{\boldsymbol n}_p}{2},
  \qquad
  \boldsymbol N_p=\hat{\boldsymbol n}_p\otimes\hat{\boldsymbol n}_p.
  \]

For a fault vertex vector \(V\),

\[
V_\Gamma(p)
=
\sum_i N_i(\xi_p)V_i.
\]

The regularized crack strain rate is

\[
\dot{\boldsymbol\epsilon}^c_p
=
\upsilon_p\boldsymbol S_p.
\]

The crucial reconstructed-fault interpretation is:

> One point on the reconstructed fault carries one RSF state. Bulk points in the same normal profile do **not** carry independent copies of \(V\), \(\Theta\), or \(T^{\rm coh}\).

---

# 3. Influence-region association

## 3.1 Baseline rule

Use a **normal strip**, not a Euclidean endpoint cap.

For a segment:

1. compute the unconstrained orthogonal projection coordinate \(\xi\);
2. require
   \[
   0\le\xi\le 1;
   \]
3. interpolate the segment half-width
   \[
   R(\xi)=(1-\xi)R_0+\xi R_1;
   \]
4. admit the particle/point if
   \[
   |r|\le R(\xi),
   \]
   where \(r\) is signed normal distance.


## 3.2 Open tips

Particles whose closest point is only on a tangent extension beyond an open tip must not be included.

## 3.3 Multiple faults

Current restriction:

- one particle/point may contribute to at most one fault;
- if a particle is admitted by multiple faults, throw an explicit unsupported-overlap error;
- intersections, branching, coalescence, and multi-fault weighting are deferred.

Do not implement "all nearby faults".

---

# 4. Fault-side and bulk-side state

## 4.1 Volume FE fields

The following quantities are stored/evolved in the bulk FE system:

- velocity \(\boldsymbol u\);
- pressure \(p\);
- phase field \(\phi\).

## 4.2 Fault-surface fields

For RSF model, the reconstructed fault carries:

- slip rate \(V\);
- slip state \(\Theta\);
- cohesive traction \(T^{\rm coh}\);
- normalization factor \(I_h\);
- previous-step values required by the time discretization, especially
  \(T^{\rm coh}_{k-1}\) and \(I_{h,k-1}\).

The normal direction \(\hat{\boldsymbol n}\) and slip direction \(\hat{\boldsymbol s}\) are supplied by the fault geometry. They should not be duplicated as independent history fields unless a future model explicitly requires that.

Material/RSF parameters may be stored or projected to the fault as generic properties when they vary spatially.

## 4.3 Particle fields

Particles retain fields that are genuinely advected bulk/history quantities, for example:

- previous viscoelastic deviatoric stress \(\boldsymbol \tau_{k-1}\);
- crack driving/history field \(H\), if the phase-field algorithm stores it on particles;
- compositional/material fields as required.

Do not store independent particle copies of:

- \(V\);
- \(\Theta\);
- \(T^{\rm coh}\).

---

# 5. Evaluation of \(I_h\)

Define

\[
h(\phi)=\frac{1}{g(\phi)}-1.
\]

For a reconstructed-fault coordinate \(s\),

\[
\boxed{
I_h(s)
=
\int_{T_s} h(\phi(s,\zeta))\,d\zeta.
}
\]

This is a profile integral of the **actual FE phase field**.

## 5.1 Baseline numerical strategy

1. choose fault evaluation points;
2. construct the corresponding normal line/profile;
3. evaluate the Q1 FE phase field directly at sufficiently accurate 1-D quadrature points along that profile;
4. integrate \(h(\phi)\);
5. store/interpolate the resulting \(I_h\) on the reconstructed fault.

The existing distributed point-evaluation infrastructure may be reused.

`I_h` changes after the phase-field solution changes; it does not need to be recomputed inside every Stokes Newton iteration if \(\phi\) is frozen during that iteration.

## 5.2 Accuracy requirements

Because \(h(\phi)\) is strongly nonlinear near \(\phi\to1\):

- use a dedicated quadrature/convergence test;
- do not assume the Stokes quadrature order is sufficient;
- do not use sparse particle sampling as the primary integrator for `I_h`.

Add a diagnostic for the discrete normalization of crack strain:

\[
\frac{
\left|
\int_{T_s}\upsilon\,d\zeta - V(s)
\right|
}{
\max(|V(s)|,V_{\rm scale})
}.
\]

---

# 6. Surface cohesive law

The pointwise cohesive law in the original derivation is

\[
T^{\rm coh}_k
=
\frac{\kappa_k}{h_k}\upsilon_k
+
\beta_k
\frac{h_{k-1}}{h_k}
T^{\rm coh}_{k-1},
\]

where

\[
\kappa_k=(1-\beta_k)\eta.
\]

For the reconstructed formulation, \(T^{\rm coh}_k\) is profile-wise uniform and belongs to the fault.

Assume for the current implementation that the constitutive parameters entering the cohesive spring are profile-wise constants for a given fault location.

Multiply by \(h_k\) and integrate through the profile:

\[
I_{h,k}T^{\rm coh}_k
=
\kappa_k V_k
+
\beta_k I_{h,k-1}T^{\rm coh}_{k-1}.
\]

Therefore the **surface cohesive update** is

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

## 6.1 Exact bulk crack-strain distribution

From the pointwise cohesive law,

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

Substituting the surface cohesive update gives

\[
\upsilon_k(\zeta)
=
\frac{h_k(\zeta)}{I_{h,k}}V_k
+
\upsilon^{\rm hist}_k(\zeta),
\]

with

\[
\boxed{
\upsilon^{\rm hist}_k(\zeta)
=
\frac{
\beta_k T^{\rm coh}_{k-1}
}{
\kappa_k
}
\left[
h_k(\zeta)\frac{I_{h,k-1}}{I_{h,k}}
-
h_{k-1}(\zeta)
\right].
}
\]

Thus,

\[
\boxed{
\frac{\partial \upsilon_k}{\partial V_k}
=
\chi_k(\zeta)
=
\frac{h_k(\zeta)}{I_{h,k}}
}
\]

when \(\phi\), \(I_h\), and history terms are frozen during the current Newton iteration.

## 6.2 Fixed-profile special case

If

\[
h_k=h_{k-1},
\qquad
I_{h,k}=I_{h,k-1},
\]

then

\[
\upsilon^{\rm hist}=0
\]

and

\[
\boxed{
\upsilon=\frac{h}{I_h}V.
}
\]

---

# 7. Trial stress and current bulk stress

Define

\[
\boxed{
\boldsymbol \tau^{\rm trial}_k
=
2\kappa_k\dot{\boldsymbol\epsilon}_k
+
\beta_k\boldsymbol\tau_{k-1}.
}
\]

This is the stress obtained by suppressing the **current** crack-strain contribution.

Using the exact \(\upsilon\),

\[
\boxed{
\tau_k
=
\tau^{\rm trial}_k
-
2\kappa_k\upsilon_k\boldsymbol S.
}
\]

Absorb the history part of \(\upsilon\) into an effective trial stress:

\[
\boxed{
\boldsymbol\tau^{*}_k
=
\boldsymbol\tau^{\rm trial}_k
-
2\kappa_k\upsilon^{\rm hist}_k\boldsymbol S.
}
\]

Then

\[
\boxed{
\boldsymbol\tau_k
=
\boldsymbol\tau^{*}_k
-
2\kappa_k
\frac{h_k}{I_{h,k}}
V_\Gamma
\boldsymbol S.
}
\]

For shear traction,

\[
t=\boldsymbol\tau:\boldsymbol S,
\]

and \(S:S=1/2\), so

\[
\boxed{
t_k
=
t_k^{*}
-
\kappa_k
\frac{h_k}{I_{h,k}}
V_\Gamma.
}
\]

For normal stress,

\[
\sigma_n
=
p-\boldsymbol\tau:\boldsymbol N.
\]

Because

\[
\boldsymbol S:\boldsymbol N=0,
\]

the direct crack-slip correction does not contribute to normal traction:

\[
\boxed{
\sigma_n
=
p-\tau^{\rm trial}:\boldsymbol N.
}
\]

The global Stokes solution still couples \(V\) and \(\sigma_n\) indirectly.

---

# 8. Generic particle-to-fault projection

For a scalar particle field \(z_p\),

\[
M_{ij}
=
\sum_p m_p N_i(\xi_p)N_j(\xi_p),
\]

\[
b_i
=
\sum_p m_p N_i(\xi_p) z_p,
\]

and

\[
Mz_\Gamma=b.
\]

Here \(m_p\) is the particle-domain volume.

Only locally owned particles contribute before MPI reduction.

This generic utility remains useful for material/composition fields and diagnostics.

## 8.1 Important distinction for the coupled constitutive solve

Do **not** treat projected current trial stress as a frozen preprocessing result that loses its dependence on \(u,p\).

The surface residual and its derivative with respect to bulk unknowns must retain the current dependence of:

- trial shear traction on \(\dot\epsilon(u)\);
- normal stress on both \(\dot\epsilon(u)\) and \(p\).

---

# 9. Surface RSF law

Use the regularized RSF law already adopted in the note; the existing rheology helper should remain the single source of truth for the exact regularized formula.

The fault residual uses

\[
\boxed{
F
=
t
-
T^{\rm coh}
-
\mu(V,\Theta)\sigma_n
-
\eta^d V
=
0.
}
\]

The slip state \(\Theta\) lives on the fault.

## 9.1 IMPES derivative

During one Newton solve, freeze the state feedback exactly as in the existing IMPES strategy.

Use

\[
\boxed{
\left.
\frac{\partial\mu}{\partial V}
\right|_{\Theta\ {\rm frozen}}
}
\]

in the surface Jacobian.

For the logarithmic form this reduces to \(a/V\). For the regularized `asinh` law, use the exact partial derivative of the implemented regularized formula at fixed \(\Theta\).

Do **not** use the full derivative through \(\Theta(V)\) inside the Newton matrix.

---

# 10. Fault residual in a projection-consistent weak form

For each locally owned particle \(p\), let \(N_i^p=N_i(\xi_p)\).

Evaluate:

- \(t_p^{*}\);
- \(\sigma_{n,p}\);
- \(V_\Gamma(p)\);
- \(\Theta_\Gamma(p)\);
- \(T^{\rm coh}_\Gamma(p)\);
- current \(\chi_p=h_p/I_h(\pi_\Gamma(p))\).

Use the residual

\[
\boxed{
R_i^\Gamma
=
\sum_p
m_p N_i^p
\left[
t_p
-
T^{\rm coh}_\Gamma(p)
-
\mu(V_\Gamma(p),\Theta_\Gamma(p))
\sigma_{n,p}
-
\eta^d V_\Gamma(p)
\right].
}
\]

with

\[
t_p
=
t_p^{*}
-
\kappa_p\chi_p V_\Gamma(p).
\]

This avoids explicitly applying \(M^{-1}\) inside the nonlinear residual.

The exact implementation may assemble algebraically equivalent matrices/vectors, but it must be equivalent to this residual.

---

# 11. Surface Jacobian \(K_V\)

Define

\[
K_V := -\frac{\partial R^\Gamma}{\partial V}.
\]

With IMPES state derivative and frozen geometry/phase field, the main contributions are:

1. bulk mechanical slip stiffness;
2. cohesive stiffness;
3. friction direct-effect stiffness;
4. radiation damping.

Schematically,

\[
\boxed{
(K_V)_{ij}
=
\sum_p
m_p N_i^p N_j^p
\left[
\kappa_p\chi_p
+
c^{\rm coh}_p
+
\sigma_{n,p}
\left.
\frac{\partial\mu}{\partial V}
\right|_\Theta
+
\eta^d
\right],
}
\]

where

\[
c^{\rm coh}
=
\frac{\kappa_\Gamma}{I_h}.
\]

For the current 2-D Q1 ordered polyline, this matrix is banded/tridiagonal under the single-segment association rule.

---

# 12. Bulk-to-fault coupling \(G\)

Let \(x=(u,p)\).

Define

\[
G
=
\frac{\partial R^\Gamma}{\partial x}.
\]

For a particle contribution,

\[
\delta t^{\rm trial}
=
2\kappa\,\boldsymbol S:\delta\dot{\boldsymbol\epsilon},
\]

and

\[
\delta\sigma_n
=
\delta p
-
2\kappa\,\boldsymbol N:\delta\dot{\boldsymbol\epsilon}.
\]

Because the surface residual contains \(t-\mu\sigma_n\),

\[
\boxed{
\delta(t-\mu\sigma_n)
=
2\kappa(\boldsymbol S+\mu\boldsymbol N):\delta\dot{\boldsymbol\epsilon}
-
\mu\,\delta p
}
\]

when \(V,\Theta\) are held fixed for this partial derivative.

This is the reconstructed-fault analogue of the \(S+\mu N\) structure in the old local tangent.

Do not approximate \(G\) by interpolating a local tangent modulus to bulk quadrature points.

---

# 13. Fault-to-bulk coupling \(B\)

The current bulk stress contains

\[
-2\kappa\chi V_\Gamma\boldsymbol S.
\]

Therefore

\[
\delta\tau
=
-2\kappa\chi\boldsymbol S\,\delta V_\Gamma.
\]

Define

\[
B
=
-\frac{\partial R_{\rm bulk}}{\partial V}.
\]

Conceptually,

\[
B:
\delta V
\longmapsto
\text{bulk residual change caused by fault-slip-induced stress}.
\]

---

# 14. Fully coupled Newton system

The preferred mathematical system is

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
\end{bmatrix}.
}
\]

Here:

- \(A\): ordinary bulk viscoelastic Stokes Jacobian with current crack-slip field frozen;
- \(B\): fault slip \(\rightarrow\) bulk residual;
- \(G\): bulk \(u,p\) \(\rightarrow\) fault traction residual;
- \(K_V\): surface RSF/cohesive/mechanical/radiation Jacobian.

Eliminating \(\delta V\) gives

\[
\boxed{
\left(
A-BK_V^{-1}G
\right)\delta x
=
-R_{\rm bulk}
+
BK_V^{-1}R_\Gamma.
}
\]

Thus

\[
\boxed{
A_{\rm eff}
=
A-BK_V^{-1}G.
}
\]

The old local tangent modulus is the local limiting case of this operator.

## 14.1 Implementation decision to make in Plan mode

Codex must compare:

### Option A: explicit fault block

Add \(V\) as a true coupled unknown and solve the block system directly.

Advantages:

- transparent mathematical debugging;
- surface residual is explicit;
- finite-difference Jacobian verification is straightforward.

Disadvantages:

- requires globally unique fault-DOF ownership and integration into distributed block vectors/matrices.

### Option B: condensed replicated fault solve

Keep reconstructed-fault geometry/state replicated and apply the exact Schur complement using the small fault system.

Advantages:

- avoids adding a new distributed global unknown block;
- fits the current replicated fault geometry.

Disadvantages:

- more involved assembly/application of \(BK_V^{-1}G\);
- harder to debug initially.

Codex must recommend one after inspecting the actual solver architecture.

**Do not implement a fake "fully consistent" local `Tensor<4>` tangent in place of this coupling.**

---

# 15. Outer timestep algorithm

## 15.1 Before the mechanical nonlinear solve

1. have current reconstructed-fault geometry;
2. associate particles/bulk points with the fault influence strip;
3. update fault normals/tangents;
4. evaluate current \(I_{h,k}\) from the FE phase field;
5. have previous fault history \(\Theta_{k-1}\), \(T^{\rm coh}_{k-1}\), \(I_{h,k-1}\).

## 15.2 Mechanical Newton/IMPES loop

For each nonlinear iteration:

1. evaluate current \(u,p\) at particle/bulk constitutive points;
2. compute \(\tau^{\rm trial}\);
3. evaluate current normal traction and effective trial shear traction;
4. evaluate the surface residual \(R_\Gamma\);
5. evaluate the IMPES partial derivative \((\partial\mu/\partial V)_\Theta\);
6. assemble \(A,B,G,K_V\);
7. solve the coupled block system or exact Schur-complement equivalent;
8. update \(u,p,V\);
9. check both bulk and fault residual convergence.

## 15.3 After mechanical convergence

1. update \(\Theta_k\) with the chosen state-law time discretization;
2. update \(T^{\rm coh}_k\) from the surface cohesive law;
3. evaluate \(\upsilon_k\) in the diffuse zone using the exact current/history formula;
4. update current bulk stress/history as required;
5. evaluate the crack driving force \(H_k\);
6. solve/update the phase field;
7. update/reconstruct fault geometry if the crack has propagated;
8. rebuild geometry-dependent caches as required for the next step.

The exact staggered ordering between phase-field update and history commit must remain consistent with the existing time-discrete equations.

---

# 16. Crack driving force

With cohesive traction stored on the fault:

1. interpolate \(T^{\rm coh}_{k-1}\) from the fault to the associated bulk point;
2. evaluate local \(h_k,h_{k-1}\) from the FE phase fields;
3. evaluate \(\upsilon_k\) using the exact formula in Section 6;
4. compute \(H_k\) with the existing mathematically derived expression.

Do not reintroduce particle copies of cohesive traction merely to evaluate \(H\).

---

# 17. MPI rules

For particle-based profile-collapse assembly:

- only locally owned particles contribute;
- ghost particles must not contribute;
- reduce fault-sized vectors/matrices with deterministic MPI sums;
- every rank must obtain identical replicated fault residual/matrix data if using the replicated-fault design.

If the explicit distributed fault-DOF option is selected, Codex must design a unique ownership scheme.

Do not gather all particles to one rank.

---

# 18. Cache rules

Geometric particle-to-fault association may be cached.

Cache geometry only:

- particle ID;
- particle position if needed for validity checking;
- particle-domain volume if it enters the projection matrix;
- fault/segment association;
- \(\xi\);
- geometry version / cache generation.

Do not cache current stress, \(V\), \(\Theta\), or other nonlinear values as part of the geometric cache.

Expected cache lifetime:

- reusable across nonlinear iterations within one timestep while particle positions and fault geometry remain fixed;
- normally invalidated by particle advection/migration, AMR, repartitioning, or fault-geometry changes.

---

# 19. Mandatory verification tests before BP3

## 19.1 Geometry / projection tests

- straight fault, constant field reproduction;
- exact Q1 linear reproduction;
- unequal particle-domain volumes;
- \(K=1\): invariance to normal distance inside the strip;
- exclusion beyond open tips;
- internal-bend behavior;
- no ghost-particle double counting;
- one-rank vs two-rank equivalence.

## 19.2 \(I_h\) tests

For analytical stationary profiles:

- compare numerically measured \(I_h\) with the analytical integral;
- perform quadrature-order convergence;
- perform mesh-resolution convergence;
- verify discrete normalization.

## 19.3 Cohesive-law tests

### Fixed profile

Verify that

\[
h_k=h_{k-1},
\qquad
I_{h,k}=I_{h,k-1}
\]

gives

\[
\upsilon=\frac{h}{I_h}V.
\]

### Evolving profile

Verify numerically that

\[
\int\upsilon\,d\zeta=V.
\]

## 19.4 Surface return-map tests

Verify:

- residual sign convention;
- cohesive contribution;
- radiation damping contribution;
- regularized RSF contribution;
- IMPES derivative;
- local-limit agreement with the old scalar return map.

## 19.5 Fully coupled Jacobian test

This is mandatory.

For a small model, compare the implemented Jacobian action with a finite-difference directional derivative:

\[
J\delta y
\approx
\frac{
R(y+\varepsilon\delta y)-R(y)
}{\varepsilon}.
\]

Test:

- bulk-only perturbation;
- fault-slip-only perturbation;
- mixed perturbation.

Do not trust the fully coupled Newton system until this test passes.

## 19.6 Benchmark progression

Only after the synthetic tests pass:

1. fixed phase field + prescribed normal stress BP3-style test;
2. fixed phase field + real normal stress;
3. evolving phase field without propagation;
4. propagation test;
5. only later consider multiple faults / merging.

---

# 20. Deferred problems

Out of scope for the first restarted implementation:

- 3-D reconstructed faults;
- true branching;
- fault intersections;
- overlapping influence bands;
- tip-to-side merging;
- general fault coalescence;
- closed-loop faults;
- multi-fault particle weighting;
- generalized distance kernels;
- full state-law derivative through \(\Theta(V)\);
- optimization of the Schur complement before correctness is demonstrated.

---

# 21. Existing code: reuse policy

Potentially reusable components include:

- reconstructed geometry;
- ordered vertex/segment adjacency;
- vertex-major generic fault property storage;
- phase-field sampling at arbitrary points;
- reconstructed-fault VTU output;
- particle-domain volumes;
- geometric particle-to-fault association;
- MPI reduction helpers.

Existing code that assumes

> project current stress -> independently solve local V -> interpolate local tangent to quadrature points

must be reconsidered.

The new mathematics is authoritative.

---

# 22. Required Plan-mode output from Codex

Before writing code, Codex must provide:

1. a concise restatement of the mathematics in its own words;
2. a keep/refactor/replace/remove table for all relevant existing files/classes;
3. a recommendation: explicit fault block vs condensed replicated fault system;
4. exact proposed interfaces/data flow for \(I_h\), \(V\), \(\Theta\), \(T^{\rm coh}\), trial stress, residual, \(A,B,G,K_V\);
5. MPI ownership/reduction semantics;
6. storage lifecycle for all fault variables;
7. how \(I_h\) will be numerically evaluated;
8. cache invalidation rules;
9. a staged implementation sequence with independently testable commits;
10. finite-difference verification of the coupled Jacobian;
11. any mathematical conflict between this specification and the current code.

Codex must not modify code during this first planning pass.

---

# 23. Short conceptual summary

The restarted architecture is

\[
\boxed{
\text{bulk trial state}
\rightarrow
\text{profile-collapse / fault residual}
\rightarrow
\text{surface RSF solve}
\rightarrow
\text{fault-slip-induced bulk stress}.
}
\]

The Newton coupling is

\[
\boxed{
\begin{bmatrix}
A & -B\\
G & -K_V
\end{bmatrix}
}
\]

or its exact Schur complement.

The cohesive law is

\[
\boxed{
I_{h,k}T^{\rm coh}_k
=
\kappa_k V_k
+
\beta_k I_{h,k-1}T^{\rm coh}_{k-1}.
}
\]

The diffuse crack-strain rate is

\[
\boxed{
\upsilon_k
=
\frac{
h_kT^{\rm coh}_k
-
\beta_kh_{k-1}T^{\rm coh}_{k-1}
}{
\kappa_k
}.
}
\]

The IMEX direct-effect derivative remains the Newton derivative used in \(K_V\).

Update this document only when the mathematics changes deliberately.
