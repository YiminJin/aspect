# K5 endpoint weak equation: formulation decision before implementation

## Decision and scope

The existing volume-weighted surface equation has a well-defined algebraic
Jacobian, but it is **not the virtual-work pullback of the implemented crack
strain**. This distinction is present in the interior, not just at an endpoint.
Consequently an endpoint-only addition cannot give the complete endpoint row
the requested work-conjugate meaning while preserving its old interior part.

There are two distinct notions of consistency here:

1. **Residual averaging:** integrate the same local traction law against a
   selected positive measure. The present formulation does this. Extending
   its admission to a separately specified finite wedge is possible, but the
   measure and its intact-material treatment are additional choices.
2. **Kinematic virtual work:** use the test induced by the actual continued
   crack-strain map. This fixes the driving-traction weight. Distributing the
   existing pointwise resistance law with that same work measure gives the
   equation below, but changes the existing interior mechanical weighting.

The user's explicit stop condition therefore applies: **report the necessary
interior change before making it**. No production code, numerical parameter,
history, background traction, or surface measure was changed in this task.
Both successful boundary source/normalization corrections are retained. The
top's existing free-RSF guard remains in place. No new ASPECT solve was run;
in particular, a prescribed-uniform replay is not offered as a substitute for
the requested free-endpoint test.

This note is a proposal/review result, not an amendment to the authoritative
specification. It does not declare the old source inconsistent with that
specification or invalidate its demonstrated benchmark results.

## 1. Kinematics and explicit endpoint equation

Consider the currently tested straight, frozen, mature fault. Let
`s_tilde(x)` be its existing bulk-source coordinate, clamped to the relevant
endpoint in each enabled in-box tangent wedge. Denote the extended Q1 basis
by

\[
 b_i(x)=N_i(\widetilde s(x)),\qquad V_\Gamma(x)=\sum_i b_i(x)V_i.
\]

Set \(\chi(x)=h(\phi_h(x))/\widehat I_h(\widetilde s(x))\) on the *actual
assembled source support*, zero outside it. In a wedge this uses constant
endpoint Ih and physical FE phase; no outside-box material is introduced.
The already approved completed Ih is unchanged. The mature kinematics are

\[
 \dot\epsilon^{cr}=\chi V_\Gamma S,
 \qquad \delta\dot\epsilon^{cr}=\chi\sum_i b_i\delta V_i S.
\]

For a top endpoint e, \(b_e=1\) in its continued wedge. It remains the usual
Q1 endpoint hat on the last segment. It is not one over the whole last
segment. At a bulk QP, the exact shear power conjugate to \(\delta V_e\) is

\[
 \delta V_e\int_{\Omega} \chi b_e(\Delta\tau:S)\,d\Omega.
\]

With the existing background convention, define

\[
 q=\tau_{bg}+\Delta\tau:S,\quad
 \sigma_n=\sigma_{bg}+\Delta p-\Delta\tau:N,\quad
 F=q-C-\mu(V_\Gamma,\Theta_{k-1})\sigma_n-\eta^dV_\Gamma.
\]

The direct work-conjugate extension of **this pointwise traction law**, using
the same distributed work measure for its resistance, is

\[
 \boxed{R_e^{work}=\int_{\Omega}\chi b_e
 [q-C-\mu(V_\Gamma,\Theta_{k-1})\sigma_n-\eta^d V_\Gamma],d\Omega=0.}
\]

For the mature test C=0. If a cohesive version were later considered, its
term carries the same measure; its history/update is not changed by this
derivation. In particular, this is not a proposal to restore the rejected
growing BP3 cohesive branch.

All terms have **one common quadrature multiplier**
\(J_q\chi_q b_e(x_q)\): driving shear (including background), cohesion,
friction and damping. Friction's actual coefficient is therefore
\(J_q\chi_q b_e\sigma_{n,q}\); damping's is
\(J_q\chi_q b_e\eta_q^d\). They do not receive extra independent width or
area multipliers. There is no division by the truncated in-box integral of
chi. Completing the denominator does not create forces outside the box.

This resistance weighting is a physical closure choice: it distributes the
existing local resistance through the same crack-strain work map and preserves
every pointwise equilibrium F=0. An alternative that retains a separate sharp
surface resistance integral would need its own boundary measure and balance
derivation; it must not be mixed silently with the distributed driving term.

At a free node the equation is zero. If the lower rate bound is active, the
existing sign convention instead uses R_e<=0 and reaction -R_e>=0. No active-set
tolerance or lower bound is altered here.

### No double counting or intact friction

This integral is over a union, not a sum of an old parent-domain integral and
an independently filled geometric wedge. Some old admitted domains already
cross the endpoint plane. In an implementation on the actual bulk quadrature,
every locally owned cell/QP is visited exactly once with its existing source
map; the wedge is a coordinate case, not another overlapping integration pass.
Each full source basis is evaluated once, including the endpoint value.

At physical phi=0, chi=0 and **every** mechanical surface contribution is zero,
even when background normal stress, mu or damping are nonzero. No newly chosen
phase threshold or clipping rule is required. This statement requires chi at
the actual integration coordinate. Multiplying a full parent domain by its
center's chi does not give the same pointwise guarantee for a domain that
crosses the diffuse edge.

These are proposed mechanical integration rules, not permission to alter
generic property projections, live CPDI geometry, particle ownership or open
surface connectivity.

## 2. Frozen-state derivatives and the B/G relationship

During mechanics, geometry, phi, Ih, material fields and committed histories
are frozen. With bulk Maxwell coefficient kappa_b,

\[
 \Delta\tau=2\kappa_b(\epsilon(u)-\chi V_\Gamma S)
                +\beta_b\tau_{old}
\]

in the mature specialization. A frozen history-localization term would not
change the V derivative. Since S:N=0,
\(\partial\sigma_n/\partial V_j=0\) **at fixed bulk unknowns**. True
normal-stress feedback through u,p is retained.

For the proposed work measure, with \(K=-\partial R/\partial V\),

\[
 K_{ej}=\int_\Omega\chi b_e b_j
 \left[2\kappa_b\chi S:S+C_V+\sigma_n\mu_V+\eta^d\right],d\Omega,
\]

\[
 (G\delta x)_e=\int_\Omega\chi b_e
 \left[2\kappa_b(S+\mu N):\epsilon(\delta u)-\mu\delta p\right],d\Omega.
\]

Here \(C_V=0\) for mature friction. In the ordinary cohesive law it is
\(\kappa_\Gamma/I_h\), with the **surface** Maxwell coefficient, not kappa_b.
mu_V is evaluated at committed Theta, with no state-update derivative inside
Newton. Fixed background tractions have zero bulk derivative, but total sigma_n
still multiplies mu_V. No SPD assumption is introduced.

The bulk action is unchanged:

\[
 \langle w,B\delta V\rangle
 =\int_\Omega2\kappa_b\chi\left(\sum_j b_j\delta V_j\right)
                   S:\epsilon(w)\,d\Omega.
\]

Only the **shear part** of G satisfies the work reciprocity

\[
 \delta V^T G_{shear}w=\langle w,B\delta V\rangle
\]

when evaluated using identical quadrature, coefficients, basis and homogeneous
bulk constraints. The full G has additional friction/normal-strain and pressure
terms; it is not B transpose. The generally nonsymmetric condensed solver and
its FGMRES requirement remain unchanged. Bulk perturbation pressure here is
physical pressure; existing solver scaling must still be applied at its current
boundary.

## 3. Why appending a wedge term does not solve the formulation question

The authoritative current equation and actual implementation are

\[
 R_i^{vol}=\sum_{p\in\mathcal A}\sum_q
           w_{pq}N_i(\xi_{pq})F_{pq},\qquad\sum_q w_{pq}=|D_p|.
\]

There is **no chi multiplier in the integration weight**. chi occurs inside
the constitutive stress/slip tangent, not as the test measure. Bulk/history
inputs are parent-P0; surface fields vary across each admitted domain.

Three possibilities must not be conflated:

- Extending unit-volume admission to the entire source wedge introduces
  nonzero friction where physical phi=0. In the saved audit the geometric
  source-only parent measure is about 4.04e7 m², whereas its positive-phase
  subset is about 2.29e5 m². Zero bulk source does not imply zero F.
- Restricting new parents by positive center phi, or weighting only the new
  wedge with h/chi, creates a residual-averaging extension. It can have a
  correct algebraic Jacobian, but neither its interface weight nor its complete
  endpoint virtual work follows from the current formulation. Full parent
  domains can also extend into intact material. A convergence pass would not
  establish the missing work identity.
- Weighting **all** mechanical surface terms by chi supplies the work test,
  but also changes the last element and the interior. Combining an old
  unit-volume interior row with a chi-weighted wedge is not that equation.
  The raw residuals even have different length dimensions unless an additional
  scaling is specified; such scaling does not remove the different transverse
  weighting.

A counterexample needs no endpoint or numerical roundoff. At the same
tangential coordinate, take two unit-volume transverse samples with N_i=1/2,
chi=(1,3), and F=(1,-1). Then

\[
 R_i^{vol}=0,\qquad R_i^{work}=-1.
\]

No nonzero nodal row scaling, mass division or endpoint-only term can turn
the former into the latter for arbitrary transverse tractions. In particular,
choose a perturbation supported in an interior segment: an endpoint correction
has no opportunity to fix its work discrepancy. Uniform-through-profile
pointwise equilibrium is a special case in which both measures give zero;
that explains why uniform sliding cannot decide this question.

The existing method is a permissible residual-projection discretization, not
a promised variational method. A Petrov interpretation on chi>0 could use
test rates N_i/chi to recover unit-volume weighting, but these are not the
represented Q1 virtual slip rates and become singular at chi=0. It does not
establish the requested conjugacy to the actual endpoint DOF.

### A second, separate discrete issue

Changing weights alone in the present particle loop would make a quadrature
approximation of the proposed continuum equation, not an exact discrete
work identity with B. B uses bulk Stokes QPs, local FE materials and FE old
stress in the bulk residual; G uses parent FE samples and retained particle
history. In particular:

- exact shear B/G reciprocity requires the same samples and kappa*chi;
- consistent affine shear work against the *actual bulk stress* also requires
  the same frozen-history representation;
- pressure/friction remains non-associated and need not be reciprocal.

Moving the mechanical surface equation to the existing bulk-QP action measure
would address these discrete requirements but is a broader interior integration
and stress-history sampling change. It must not be introduced as merely adding
missing endpoint parents. Generic particle property projection may remain
volume-weighted; it is not the same operation as mechanical virtual work.

## 4. Checks and recommended decision

Source confirmation:

- `source/reconstructed_fault/surface_system.cc`, `assemble_surface_system`:
  inactive parents are skipped; `weight=q.weight` multiplies all traction terms
  and K_V; CouplingPoint records carry the same weight.
- `apply_G_reference`: that weight multiplies
  2*kappa*(S+mu*N):epsilon(delta_u)-mu*delta_p, without chi in the measure.
- `source/simulator/assemblers/reconstructed_fault_stokes.cc`:
  bulk-QP coefficients include 2*kappa*chi*S and current extended basis values.
- `source/material_model/phase_field_fault.cc`, point response:
  direct slip-induced normal derivative vanishes; mature C and C_V are zero.

The previous production evidence is reused, not rerun: top endpoint B finite
difference relative error 1.70589e-15, sparse/reference action error 2.50868e-16,
and a 57.0073% added-wedge fraction of the documented B-work probe. The last
number is not a measured G error or proof that G must equal B transpose.

A new cheap manufactured algebra check, not a production coupled test, is:

```sh
python3 benchmarks/reconstructed_fault/bp3/check_endpoint_weak_form.py
```

Saved output: `benchmarks/reconstructed_fault/bp3/endpoint_weak_form_algebra.json`.
It checks inclined S:N, the proposed frozen V and bulk derivatives, shear
virtual work, zero intact contribution and the interior counterexample.
Results: V-derivative absolute discrepancy 2.744e-10; bulk-derivative discrepancy
5.795e-10; discrete shear work discrepancy 0 in the manufactured common-point
rule. These validate the written algebra only; they do not qualify an ASPECT
implementation or a free-endpoint solve.

**Recommendation:** approve a narrowly scoped redesign of the *mechanical*
surface residual over the whole represented fault to the work-conjugate
measure, with an explicit decision on common bulk-QP/frozen-history sampling.
Leave generic property projection and the successful source maps unchanged.
This is a numerical-formulation revision, not an endpoint bug fix. It also
requires reviewing residual norms/scales and initial background projection
under the new measure, without retuning numerical acceptance thresholds or
silently recalibrating an existing run's background.

After approval, the first qualification should be the requested frozen-history,
nontrivial free-top coupled case: nonzero endpoint deltaV, a nonuniform bulk
perturbation, existing prescribed deep slip, consistent residual/K/G finite
differences, shear virtual work on the actual discrete measure, fresh linear
checks, and no history commit. That case is intentionally **not run now**:
it would otherwise test a weighting choice that has not been authorized.

Only this review, the manufactured check and its output were added by this task.
No solver, boundary source, Ih, quadrature, parameter or history code changed.
