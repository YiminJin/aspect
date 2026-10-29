# P4a finite-tip diagnostic: equations and screens before runs

Entry: `f1f9f29cf40c32d1ddcea000b724375b64853532` (closed P3). Preserve
the manual AGENTS edit, P2 branch failure, BP3 artifacts and P3 evidence.
This first diagnostic uses the existing native frozen-geometry observation
route. It does not authorize removing its physical advancement guards.

Retained cohesive energy, at fixed geometry and immutable incoming histories:

\[
 h=g^{-1}-1,\quad I_h=\Pi_{\Gamma}\int h(\phi_h)\,dn,
 \quad\chi_\Gamma={\bf1}_{\text{normal strip of }\Gamma}h/I_h,
 \quad\tau=T-2Z\chi_\Gamma S,
\]
\[
 E=\int_\Omega\|\tau\|_{3D}^2/(4G)\,dA
   +\int_\Gamma Z^2/(2GI_h)\,ds+E_f(\phi_h).
\]

I is the existing consistent surface FE projection, with its full derivative
and adjoint. Its finite positive admission rule is retained, without a floor.
The full plane-strain metric includes zz=-(xx+yy) and both xy contributions.
Outside the actual finite surface's normal strip, tau=T and the local
mechanical derivative with respect to phi is zero. The prescribed rounded cap
there still contributes fracture energy and a physical irreversibility obstacle.
A nodal basis function straddling the endpoint can have nonzero assembled
mechanical derivative; that is different from local driving ahead of it.

The documented intact-profile formula q^2/(2Gg^2) requires a defined q. It
does not supply q outside Gamma or at I=0. At finite Z, Z^2/(2GI) is singular
as I tends to zero; an activation/birth limit is a scientific dependency,
not a quadrature choice. P3's zero-tip/zero-new-history append within positive
cap support supplies neither an intact-material activation law nor a general
nonzero-tip joining energy rule. No segment is inserted to supply driving.

Two frozen native cases only: retain P3's aligned x_tip=.5 m control; move
the physical seed endpoint to x_tip=33/64=.515625 m, inside bulk cell
[.5,.53125]. Sample the same radial continuum cap prescription at Q1 nodes
around each endpoint, with a loaded complete DGQ2 tensor and
Z=1000 sin^2(pi*x/x_tip) Pa m (zero at both ends). This compares integration
of each prescribed state, not spatial convergence or equilibrated mechanics.
Uniform surface spacing remains 1/64 m. The cut endpoint is an actual native
tangential quadrature panel boundary: no sampling-based approximate clipping.

Before physical advancement, independently check native energy, weak loads,
normalization and complete mechanical/fracture phase derivatives, resolving
bulk cells, surface knots, strip endpoint and represented phase support.
Use Gauss24 and Gauss32 per smooth panel as an integration cross-check.
Screens: normalized energy/load/I errors <=1e-10; directional derivatives
<=1e-8 in max(1, |derivative|) units; finite-difference confirmation <=1e-7;
strictly-ahead mechanical derivatives exactly zero. These are integration
screens, not physical-response acceptance criteria. Export actual native
mechanical quadrature locations and JxW, analytic-normalization ray intervals
and surface weights, phase/obstacle, stress and separate energy derivatives.
The figure must distinguish counts from integration weights.

Mechanical equilibration and a phase-active test remain conditional on a
consistent finite-tip driving/activation policy. If the existing formulation
does not determine that policy, report the precise dependency and complete
the independent diagnostic without inventing a new constitutive rule.
