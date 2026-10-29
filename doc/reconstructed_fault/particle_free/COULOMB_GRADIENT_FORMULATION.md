# K-gradient candidate: thermodynamics, weak equations and tangent

This is the experimental fixed-phase derivation. Bounded native evidence and
its limitations are recorded in [the report](COULOMB_GRADIENT_REPORT.md).
[The published antecedent, variable mapping and additional assumptions](COULOMB_GRADIENT_LITERATURE.md)
were recorded before implementation. The isolated prescribed-X point inverse
has subsequently exhibited three admissible roots at a saved developed-damage
state. The derivation below does **not** assume that inverse is unique. The joint solve below avoids that inverse. Phase-active progression remains
blocked; no associated-potential stability certificate is inherited.

## State, energy and power

Let S be the in-plane symmetric trace-free space, including both xy entries;
K,A,Q,W,X belong to S. Auxiliary T and physical deviator tau retain full 3D
plane strain. Write P_S for orthogonal projection into S. Fixed-phase equations
are

```
tau=T-2G A,   Q=2G A/h(phi),  sigma=tau-p Id,  p=p0+p',
Tdot+(G/eta)T=2G D3(v),  Adot+(G/eta)A=W,  Kdot=W,  div(v)=0.
```

Use the retained total storage plus
`psi_grad=G ell_s² grad K:grad K`. Its microstress and bulk force are
`M=2G ell_s² grad K`, `X=-div M`. X is a tensor in S; it does not alter the
definition of physical normal pressure. Integration by parts gives

```
d/dt integral psi_grad = integral X:W + integral_boundary (Mn):W.
```

Define the dissipative shear-space force `F=P_S(tau)-Q-X`. On both conjugate
systems, `m_i=k_i sym(t_i*nu_i)`, k=+1,-1, use

```
pn_i=p-nu_i.tau.nu_i > 0,
f_i=F:m_i-c_r-mu pn_i <=0,
W=sum_i w_i m_i,  w_i>=0,  w_i f_i=0.
```

Planes maximize this complete orientation-dependent f; taking only the
principal axes of F would omit physical normal-traction sensitivity. At a
strict active maximum f_theta=0. Non-associated pressure dependence and zero
dilation are retained. The total fixed-phase identity is

```
d/dt integral [tau:tau/(4G)+h Q:Q/(4G)+psi_fracture+psi_grad]
 + integral [tau:tau/(2eta)+h Q:Q/(2eta)+sum_i(c_r+mu pn_i)w_i]
 = P_external_mechanical + integral_boundary t_K:W,  t_K=Mn.
```

Microfree means t_K=0, not K=0. Prescribing K or microtraction is additional
boundary data; both cannot be freely imposed at the same point. Gradient
energy is recoverable and its force can drive redistribution/reversal. It
does not relax automatically with A. Direct phase driving stays the retained
`Y=h' Q:Q/(4G)` for this phase-independent coefficient, but phase is held fixed
and the coupled phase law is not qualified here.

Backward Euler, using the same incoming histories in every trial, gives

```
beta=(1+dt G/eta)^-1,  Delta K=K-K_n,  W=Delta K/dt,
T=beta[T_n+2G D3 symgrad(du)],
A=beta(A_n+Delta K),  Q=2G A/h,  tau=T-2G A.
```

For quadratic gradient energy,
`dt integral X:W=Delta E_grad+E_grad(Delta K)-dt integral_boundary t_K:W`.
The positive `E_grad(Delta K)` is the BE remainder, not physical dissipation.
It must be added to the retained stress/traction BE losses. This accounting
does not by itself establish physical energy convergence or stability.

## Joint weak problem

For displacement test z, incompressibility test q and microtest V in S,

```
R_u(z)=integral [tau:D3 symgrad(z)-p div z]-external_load(z)=0,
R_p(q)=integral q div(du)=0,
R_K(V)=integral [(F+Q-P_S tau):V+2G ell_s² grad K:grad V]
        -integral_boundary t_K:V=0.
```

These are coupled to the local normal-cone relations above. All means,
constraints and quadrature measures must be consistent. A tensor H1 field
or a consistently derived DG gradient energy is required; differencing
unrelated quadrature records is not this weak form.

The key distinction is that **F/K and mechanics are joint unknowns**. Writing
an independent material return `(du,p,X) -> (A,Q,W)` adds a local uniqueness
assumption that is false in the saved counterexample. The original guard
rejects it correctly. The nonlocal weak problem must be studied jointly;
local multivaluedness alone does not prove global nonuniqueness or failure
of spatial regularization. Nor may one choose the lowest-energy local root
using an incremental potential that this non-associated law does not have.

## Equivalent local normal map and complete fixed-phase tangent

For a two-vector representing an S tensor by (xx,xy), let
`d=((tau_xx-tau_yy)/2,tau_xy)`, `pbar=p-(tau_xx+tau_yy)/2`, and J rotate
two-vectors by 90 degrees. The admissible set for F at fixed physical tau,p
is the convex lens

```
C(tau,p)={F: |F-c_+|<=R, |F-c_-|<=R},
c_+=mu Jd, c_-=-mu Jd,  R=c_r+mu pbar.
```

Closed contact makes R>|c_+|. For any fixed positive a with units Pa s,
the exact flow constraint is `F=Proj_C(F+aW)`. This projection is unique
even when the *complete material return* is not. Parameter a scales an
equivalent residual; it is not a new slip viscosity or physical dissipation.
This normal map is a possible joint solver formulation, not yet a qualified
native algorithm or a global branch-selection/stability rule.

For Z=F+aW, the projection and its derivatives are piecewise explicit:

- Inside the lens, Pi=Z, `delta Pi=delta Z`.
- On one circle centered at c, put v=(Z-c)/|Z-c|. Then
  `Pi=c+R v` and
  `delta Pi=delta c+v delta R+(R/|Z-c|)(Id-v*v)(delta Z-delta c)`.
- At a two-circle intersection, put c=c_+, u=Jc/|c| and
  `b=sqrt(R²-|c|²)`. The relevant corner is `Pi=s b u`, s=+1 or -1.
  `delta b=(R delta R-c.delta c)/b`,
  `delta u=J delta c/|c|-u(c.delta c)/|c|²`,
  `delta Pi=s(u delta b+b delta u)`.

Here `delta c_+=mu J delta d`, `delta c_-=-delta c_+`, and
`delta R=mu[delta p-(delta tau_xx+delta tau_yy)/2]`.
For coincident centers the set is one disk. At active-set boundaries only
directional/generalized derivatives apply. These derivatives include changes
of the physical Coulomb planes and both active families; freezing centers or
R would discard the pressure/friction tangent.

With h, dt and accepted histories fixed,

```
delta T=2G beta D3 symgrad(delta du),  delta A=beta delta K,
delta Q=(2G beta/h)delta K,
delta tau=2G beta[D3 symgrad(delta du)-delta K],
delta W=delta K/dt, delta M=2G ell_s² grad(delta K).
```

The complete joint actions are

```
delta R_u(z)=integral [delta tau:D3 symgrad(z)-delta p div z],
delta R_p(q)=integral q div(delta du),
delta R_K(V)=integral [(delta F+delta Q-P_S delta tau):V
                         +2G ell_s² grad(delta K):grad V],
delta R_normal=delta F-Pi_Z[delta F+a delta K/dt]
                        -Pi_tau[delta tau]-Pi_p delta p.
```

Prescribed microtraction has zero trial variation. These actions are
nonsymmetric because Pi_tau and Pi_p contain physical normal-pressure
dependence while total flow remains isochoric. No symmetric phase Hessian or
associated-flow minimum principle is imported. Phase derivatives, including
any future phase-dependent gradient coefficient, need separate qualification.

## Short-wavelength statement and its limits

At a continuous strict active state with m_i present and
`N_i=m_i+mu nu_i*nu_i`, a Fourier rate perturbation has

```
H_ji(k)=N_j:m_i + [1/h+(k ell_s)²]m_j:m_i.
```

The usual mixed momentum/incompressibility symbol must use this constitutive
action including its pressure derivative. For one active system the previous
formula has `1+1/h` replaced by `1+1/h+(k ell_s)²`. Therefore the single-system
cutoff obeys `k_c ell_s=sqrt(1/hcrit-1/h)` when h>hcrit. Wavelengths below
`lambda_c=2pi ell_s/sqrt(1/hcrit-1/h)` avoid that rate-level defect. The
formula is not a universal two-system threshold. For two independent active
systems the positive k² Gram term dominates; a dependent combination with
zero net W also has no mechanical plastic perturbation. High-k mechanics
approaches elastic response. This does not eliminate every finite-wavelength
bifurcation or provide global branch uniqueness.

Finite-step changing-direction derivatives must also be checked. If a unique
local point branch exists, let U=dt W and use its complete derivatives with
respect to strain, p/G and X/G. Eliminating X/G=2(k ell_s)² delta K gives

```
delta U=[Id-2(k ell_s)² U_X]^-1[U_e:delta epsilon+U_p delta(p/G)],
delta tau/G=C_e:delta epsilon+C_p delta(p/G)
              +C_X 2(k ell_s)² delta U.
```

This is only a local branch diagnostic. The inverse and its poles must be
checked; it must not be presumed valid across the saved multiple-return
state. A joint native weak solve, admissible branch/stability assessment and
resolved refinement remain prerequisites for propagation. Exploratory ell_s
values are uncalibrated physical inputs, never mesh-dependent choices.

## Bounded native discretization

The serial Cartesian prototype makes the existing Gauss6 history records the
nodes of a continuous bilinear auxiliary K space. Sorted physical quadrature
axes define that space; linear extrapolation through the first/last two nodes
extends it to the physical boundary. This preserves global affine and mixed-xy
fields and avoids an artificial zero-slip boundary skin. The constitutive mass
is the original native JxW diagonal; the gradient stiffness integrates the
auxiliary interpolant exactly (Gauss2 on each piece, including edge strips).
Mechanical Gauss6 and DG source Gauss3 remain unchanged. Thus this is a specified
mass-lumped discretization of the new weak equation, not an assertion that
quadrature history records already possessed a spatial gradient.

For symmetric stiffness L, implement `(LK)_i=sum_{j!=i}L_ij(K_j-K_i)` and
`Egrad=-G ell_s² sum_{i<j}L_ij (K_j-K_i):(K_j-K_i)`. These equal the variational
operator/energy and preserve constant fields exactly. Set
`X_i=2G ell_s² (LK)_i/JxW_i`. The native solve carries both K components together
with displacement and perturbation pressure; it never calls the nonunique
prescribed-X inverse. All nine DGQ2 T modes and the independent native-gradient
publication check remain in place. The Gauss6 K layout is fixed throughout.

The explicit benchmark environment `ASPECT_BULK_SLIP_LENGTH` is captured once in
the accepted owner; its nonzero value extends the checkpoint material tuple.
Default zero follows the retained local law and old tuple exactly. No loaded
local-law checkpoint is converted. Derived geometry/stiffness caches rebuild
after restart; accepted K/A/Q/W/L records remain the existing persistent owners.
This option is restricted to serial, fixed-phase, fixed Cartesian geometry.

At fresh initialization a zero-interior displacement lifting can generate a
mesh-dependent boundary layer before mechanics equilibrates. One fixed-K elastic
Newton guess is permitted on the first iteration only. Every K component is
released thereafter; final acceptance requires the complete coupled residual,
contact, source, dissipation and work checks. This is a solver initial guess,
not an imposed constitutive sticking interval or a hold-growth rule.

Native joint diagnostic `local.gamma` entries are physical per-system rates
(1/s), as explicitly documented in joint_point(), not the stress multipliers
of the eliminated local CR::update(). They are not persistent history and
CR::derivative() is never applied to this joint record. Native CSV system rates
are independently decomposed from accepted W and the actual Q+X planes.
The complete joint derivative uses joint_action()/projection_action() only.
