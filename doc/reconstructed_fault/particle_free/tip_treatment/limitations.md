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
has been imported to bypass them. The admitted endpoint section law is monotone and has a unique scalar root;
this does not exclude global phase instability or steady velocity weakening.
Failed evidence must remain visible.

The active coupling uses distributed sparse FE ownership and ASPECT Stokes
inverses. The root-direct mixed solve is now an explicit benchmark-only route.
Distributed storage is not a performance qualification: difficult phase searches
can still consume many Krylov and Stokes iterations.
Point-history quadrature and phase-space/obstacle refinement are distinct
accuracy questions. The [mature guide](../MATURE_MOVING_FAULT.md) and its four-run
closeout remain unchanged qualifications of a different model and integrator.

## Measured qualification boundary

The [original report](../TIP_TREATMENT_IMPLEMENTATION_REPORT.md) retains the
coarse/half-dt passes and refined .05 failure of the old add-only procedure.
The [integration report](../TIP_SOLVER_INTEGRATION_REPORT.md) records its corrected
failure-window replay and current continuation limits. They are different solver
generations, not contradictory descriptions of one executable.

The original loading reaches the homogeneous intact threshold near .05 s and
activates broad weak damage. Its unloaded profile relaxation and outer-boundary
rates prevent interpreting that experiment as localized propagation. The new
fixture equilibrates the declared seed before its physical clock and checks
remote driving, but its load-driven diffuse increment and BE work remainder must
still be judged against their measured sizes. No phi≈1 advancing core, stable
branch selection, mesh/quadrature qualification or production readiness follows
from successful residual/KKT/ledger tests.

The [bounded efficiency/growth task](../TIP_EFFICIENCY_AND_GROWTH_REPORT.md)
retains progress-based search ordering on measured cost and local checks. The
same gates permit tens of Pa of resistance variation at rates around 1e-13 m/s;
independent RSF evaluation explains that sensitivity, but it is not uniform
reaction accuracy. A unique physical stationary state is not established.
The localized extension reaches remote Y/Kf=.98149 at .09 before any resolved
advance of the measured phase or H contours. It shows halo adjustment, not an
advancing weakened front. Compact loading and a separately derived time-accuracy
strategy are next proposals; neither a new load nor integrator is enabled.
