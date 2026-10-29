# Published antecedent and explicit adaptation — before implementation

This note precedes implementation of the selected experimental K-gradient
option. It identifies a mathematical antecedent, not a calibrated rock-friction
law or a transfer of its well-posedness results. Phase-active and large-specimen
progression remain blocked by the accepted localization audit.

## Closest formulation

The closest published tensor formulation is **M. E. Gurtin (2003), “On a
framework for small-deformation viscoplasticity: free energy, microforces,
strain gradients,” International Journal of Plasticity 19, 47–90**,
[doi:10.1016/S0749-6419(01)00018-3](https://doi.org/10.1016/S0749-6419(01)00018-3),
particularly its simple separable theory with quadratic defect energy in
plastic-strain gradients. It supplies a tensor plastic-strain field, energetic
higher-order stress, a microforce balance, slip-related boundary work and a
weak formulation. Our proposed isotropic full-gradient energy is a restricted
choice within that type of construction. Its viscoplastic flow closure is not
our rate-independent, non-associated Coulomb closure.

[Gurtin (2000)](https://doi.org/10.1016/S0022-5096(99)00059-9) provides the related
slip-system formulation, but its crystallographic systems are not our freely
changing Coulomb planes. [Gudmundson (2004)](https://doi.org/10.1016/j.jmps.2003.11.002)
is a related tensor higher-order framework. The Burgers-vector/plastic-spin
formulation in [Gurtin (2004)](https://doi.org/10.1016/j.jmps.2004.04.010)
must not be identified with a full `grad K` norm. The plastically irrotational
tensor theory of [Gurtin–Anand (2005)](https://doi.org/10.1016/j.jmps.2004.12.008)
is another related microforce formulation. These distinctions matter: no dislocation-based
calibration or associated-flow theorem follows from the proposed adaptation.

## Mapping to our actual variables

The right column below is our proposed specialization and retained native
model, not a claim that the paper contains these Maxwell/cohesive equations.

| Published role | Our counterpart and qualification |
|---|---|
| Symmetric plastic strain E^p | Signed accumulated in-plane trace-free tensor K, dimensionless; `Kdot=W`. It is not monotone scalar L. |
| Plastic strain rate D | `W=sum_i w_i m_i`, including both conjugate systems and changing directions. |
| Elastic stress T in the paper | Physical stress `sigma=tau-(p0+p')Id`, with deviator `tau=T_native-2G A`. Paper T must not be identified with our auxiliary native T. |
| Elastic storage | Retained `tau:tau/(4G)`. Because of Maxwell relaxation and separate A, we do not assert the paper's simple `epsilon_e=epsilon-K` kinematics. |
| Local energetic flow force | Q acts in this role, but `Q=2G A/h` derives from `psi_coh=G A:A/h`, not a new `psi(K)`. A obeys `Adot+(G/eta)A=W` and differs from K. |
| Quadratic defect energy in grad E^p | New `psi_grad=G ell_s^2 grad(K):grad(K)` using full plane-strain contractions. |
| Energetic polar/third-order microstress | `M_K=partial psi_grad/partial grad K=2G ell_s^2 grad K`, units Pa m. |
| Energetic gradient force | `X=-div M_K`, units Pa, entering effective flow stress `tau-Q-X`. |
| Microtraction | `t_K=M_K n`, power `t_K:Kdot`, units W/m². |
| Accumulation variable | L has only a loose role-level analogy; our resistance has no new L hardening, and L does not replace K in gradient storage. |
| Constitutive dissipative force | `D_flow=tau-Q-X`. Its projections obey the retained pressure-dependent Coulomb resistance, not an associated J2 flow potential. |
| Nonlocal flow boundary data | Explicit microfree `M_K n=0` in the first block. Mechanical displacement data alone do not supply this condition. |
| Fracture/phase energy | No direct antecedent in this paper: retain `Kf(phi+ell^2|grad phi|^2)`, h(phi), physical obstacle and current Gc/m conventions. Phase is fixed here. |

Thus the proposed total storage at fixed phase is

```
psi = tau:tau/(4G) + G A:A/h(phi)
      + Kf[phi+ell^2 grad(phi).grad(phi)]
      + G ell_s^2 grad(K):grad(K).
```

The h=0 cohesive term uses the established constrained intact limit; it is
not evaluated as zero divided by zero. Full stress has zz; K/A and their
variations belong to the retained in-plane trace-free space. Factors of two
include both symmetric xy entries. The coefficient convention is equivalent
to writing `(c/2)|grad K|^2` with `c=2G ell_s^2`.

## Additional assumptions, explicitly

1. **K becomes energetic.** Before this option, K is an accumulated physical
   tensor without this storage. Signed reversal can reduce its gradients and
   release energy. Neither monotone L nor sum of absolute per-system slips
   is substituted. Existing loaded checkpoints are not converted.
2. **Full-gradient phenomenology.** Penalize all spatial derivatives of K,
   including compatible ones. This is not a Curl-only dislocation-density
   model, not a derivation from grain size, and not a crystalline slip system
   model. No independent plastic spin or director field is added.
3. **One isotropic coefficient.** Use constant positive `G ell_s^2`, with no
   anisotropy, phase degradation, density dependence or accumulated-slip
   dependence. Those alternatives would add terms to forces/tangents and
   potentially to phase driving. With this choice the direct phase force
   remains `Y=h' Q:Q/(4G)`; coupled responses can still change it.
4. **Independent length.** ell_s is a new physical constitutive input. It is
   neither the phase length ell nor the cell size. Prototype values will be
   labeled uncalibrated, held fixed under refinement, and translated to
   wavelengths using the actual mixed symbol before native runs.
5. **Recoverable, non-relaxing gradient memory.** No gradient dissipation,
   new slip viscosity or autonomous K relaxation is added. T/A retain their
   existing Maxwell equations. Gradient backstress can cause redistribution
   or reverse slip during a hold; stopping motion is not deleting stress.
6. **Retained physical normal pressure.** Use
   `pn_i=p0+p'-nu_i.tau.nu_i`, not normal traction from `tau-Q-X`.
   `f_i=(tau-Q-X):m_i-c_r-mu pn_i<=0`. Both conjugates remain admissible.
   Flow directions maximize this complete orientation-dependent expression;
   they cannot simply follow the principal axes of tau-Q-X alone.
7. **Zero dilation and non-association.** `tr W=0`, `div v=0`; no opening,
   normal plastic flow or volume history is introduced. The coupled tangent
   remains nonsymmetric. An energetic gradient term does not restore an
   associated incremental potential or validate the old phase Hessian.
8. **Explicit microboundaries.** First use zero microtraction on the exterior,
   with no imposed K=0 skin, passivation, internal interface energy or grain
   boundaries. This admits homogeneous K and isolates bulk regularization.
   A microhard boundary is different physics, not a default numerical fix.
9. **Small strain and fixed material coordinates.** Keep full 3D plane strain,
   the existing rate convention, fixed mesh/geometry and no transport. Passive
   basis covariance is required; finite-rotation/objective-rate physics is
   not qualified by it.
10. **Fresh state and new spatial ownership.** Start K=A=T=0. A gradient energy
    requires an H1 tensor field (or an explicitly consistent discontinuous
    formulation), not a Laplacian of unrelated quadrature records. Native
    integration must identify its authoritative field, trial/commit and restart
    semantics. It must not silently smooth accepted K or reset histories.

## Thermodynamic consequence to verify

At fixed phase, define `D_flow=tau-Q-X`. The proposed microforce weak form is

```
integral [(D_flow+Q-tau):V + M_K:grad V]
  = integral_boundary t_K:V.
```

Combining it with retained Maxwell equations and mechanical equilibrium gives

```
d/dt integral psi
 + integral [tau:tau/(2 eta) + h Q:Q/(2 eta)
             + sum_i(c_r+mu pn_i)w_i]
 = P_mechanical_external + integral_boundary t_K:W.
```

The boundary term is essential; gradient storage is not frictional heat.
This identity is our adaptation's necessary thermodynamic check, not evidence
of well-posedness. Pressure derivatives, changes of active systems/directions,
weak microforce actions and the mixed incompressible symbol require direct
qualification. Literature results for an associated or viscoplastic closure
are not imported. Non-proportional/reversal tests must expose new stored-energy
backstress rather than count its resistance as a harmless numerical filter.

The distinction between recoverable tensor history and monotone accumulation,
and the need to inspect non-proportional response, are also emphasized in
[Fleck, Hutchinson & Willis, *Guidelines for Constructing Strain Gradient
Plasticity Theories*](https://www.repository.cam.ac.uk/bitstreams/c97d4709-5de9-48d5-b368-4a8213df3442/download).
This supports the choice of diagnostics, not a quantitative prediction for
our Coulomb/Maxwell material. No implementation or new qualification result
is claimed by this pre-implementation note.

## Subsequent prototype finding: an assumption that must not be added

After recording the mapping, a small prescribed-X point prototype found three
distinct admissible returns at phi=.99, Xxy=-.1 Pa. Independent 70-digit
trigonometric equations confirm them. Thus **single-valued local inversion at
given X is not an available additional assumption**. It is not supplied by the
published framework or by thermodynamic accounting. See
[the joint weak formulation](COULOMB_GRADIENT_FORMULATION.md) and
[measured scope and limitations](COULOMB_GRADIENT_REPORT.md).
This is not proof that the joint gradient problem is nonunique. It prevents
implementing it by blindly reusing the existing independent unique-point
inverse. No local root-selection rule or native option has been adopted.
