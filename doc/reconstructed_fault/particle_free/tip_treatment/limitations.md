# Limits of this control

[Guide](../TIP_TREATMENT.md)

The physical closure is explicit for a fixed-direction single-fault specimen;
it is not a completed general tip model. All bands at a given section coordinate
share one age and resistance. Opposing rates can reduce net slip and section
friction loss while leaving transverse deformation and cohesive variance.
General turning/branching requires a rule for orientations, intersecting bands,
material age ownership and a work-consistent nonlocal mapping. Reconstruction
alone cannot supply those laws.

Current-pressure friction, opening/contact, variable coefficients, nonzero spin,
transport, AMR and arbitrary off-quadrature stress sampling are unsupported here.
The mature bulk-slip-subtracted spin selector remains separately qualified; it
cannot be extended by removing a guard. A growing model must first define the
full fault velocity-gradient contribution J_f, not just its symmetric part.

General motion also adds fracture-gradient deformation work. For constant K,
\[
D_t\nabla\phi=\nabla D_t\phi-(\nabla v)^T\nabla\phi,
\]
\[
D_t\psi_f=K\alpha' D_t\phi+2K\ell^2\nabla\phi\cdot\nabla D_t\phi
-2K\ell^2(\nabla\phi\otimes\nabla\phi):D.
\]
A material-volume balance adds ψ_f div(v); an Eulerian balance needs boundary
flux. A configurational/microstress closure or explicitly prescribed geometry
work is required before moving growth is enabled. An observed residual cannot
be renamed a shape-work correction.

Dissipation positivity and a converged KKT system do not prove uniqueness,
ellipticity, global stability or physical validity. The earlier Coulomb
localization/gradient findings remain evidence for those different laws. No
uncalibrated slip-gradient energy, damping, phase cap, or tolerance relaxation
has been imported to bypass them. Velocity-weakening RSF may itself produce
multiple roots or instability; failed evidence must remain visible.

The bounded root-direct mixed solve is not a scalable production algorithm.
Point-history quadrature and phase-space/obstacle refinement are distinct
accuracy questions. The [mature guide](../MATURE_MOVING_FAULT.md) and its four-run
closeout remain unchanged qualifications of a different model and integrator.

## Measured qualification boundary

The completed coarse and half-dt controls demonstrate the closed law, not a
spatially converged growing fault. The refined control fails at .05 s in the
linearized-bound/line-search procedure; its coupled derivative passes. See the
[report](../TIP_TREATMENT_IMPLEMENTATION_REPORT.md) for exact diagnostics. Initial
fracture-profile adjustment depends strongly on the mesh. The loading reaches
the nominal homogeneous intact threshold near .05 s and activates a broad weak
halo; peak rates at outer boundaries are not localized tip propagation. No
fully developed phi≈1 advance is demonstrated. BE physical-work remainder
halves under timestep halving but remains large relative to added fracture.
The complete law is specified for this control; stable branch selection,
spatial accuracy and production readiness remain open.
