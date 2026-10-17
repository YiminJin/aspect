# Bounded stationary mature DG qualification

Base a8707307008401ac9cff702a369e2e2a611fa0f7 plus preserved incoming work.
Hanging-node phase evolution is deferred. This task is fresh, serial, fixed
graded Cartesian mesh and fixed straight 60-degree fault, prescribed frozen
phase, homogeneous coefficients and stationary histories. No automatic AMR,
transport, moving geometry, cohesive phase solve or first-event run is selected.

For the mature law retain C=0 and Q_geo=0. With fixed chi=h(phi)/I and geometry,
store independent T and auxiliary Z: T_new=beta*T_old+2*kappa*D3(symgrad u),
Z_new=beta*Z_old+kappa*V, tau=T-2*chi*Z*S. Z is relaxation/slip memory, not cohesive
traction or cohesive stored energy. The native FE reference uses
T=2G*D3(symgrad d_R), d_R_new=beta*d_R_old+(kappa/G)*u. Background traction is
surface data. Both start with zero perturbation history. Use the actual segment
tangent/normal/shear sense and native bulk-source endpoint continuation.

Extend default-off homogeneous mesh data to the restricted mature independent
histories. The maintained BP3 callback prescribes every physical phase DoF and
the same positive nodal Theta function. In domain-free mode require these data
before preparation; never fall back to a particle state/phase/material projection.
Retain general oblique cell-interval I, consistent surface projection and paired
automatic completion, including exterior Q1 contributions. Do not use the P2
horizontal analytic profile or add integration floors/adaptive anisotropic rules.

First verify complete-Q2 native source projection, non-proportional tensor/slip
history, reconstructed stress and actual weak coupling against native FE at
matching physical fields. Use full embedded 3D tensor norms and preserve every
DGQ2 mode. Check bulk/endpoint continuation and zero cohesive traction/storage;
keep the independent native publication guard and its operand-derived allowance.
Require 1e-10 scaled matched-field/operator agreement; record absolute errors.
Independently check I with fixed high-order normal integration and bulk physical
quadrature at 3/6/8 points, without adapting it during a solve.

Reuse section 7B's translated 2000x1000 m, ell=20 m/core=.6, 4136-cell graded
mesh, 59-node surface, uniform strengthening, physical Vp=1e-9, initial smooth
Theta disturbance .02, full bottom/side loading, natural top and filter20.
Report raw unfiltered diagnostic tractions separately from the filtered physical
RSF input. Native FE and particle controls use matching initialization, schedule,
material/rate convention and quadrature. Preserve a legacy full-gradient particle
control separately if needed to identify its equation difference.

Gate one positive step before the existing ten-step dt=200000 s/T=2000000 s
short fixture. Bound quadrature and dt/2 checks to T=600000 s with the same
step-zero constitutive interval of 200000 s. Use one retained-style finer bulk
mesh at that short horizon if quadrature or traction results require spatial
separation. Keep surface resolution fixed for this bulk comparison. Evaluate
stress, V/Theta/slip, raw/filtered loads, residuals, history recurrence and costs;
separate absolute/background-scaled errors from perturbation-scaled errors.
Predeclared practical sensitivity screens: 1% of Vp for V; 1% of Dc/Vp for
Theta; 1% of measured perturbation scales for physical stress/tractions. Passing
identity/lifecycle alone does not establish those accuracy screens.

Use fresh source/build/plugin inputs, sequential jobs, one thread/rank, 400 s
gates and 700 s trajectory caps, total native budget 3600 s including failed
attempts (compilation excluded). Preserve defaults and every failed observation.
Export authoritative reconstructed stress and complete history data at selected
accepted states, plus raw surface traction profiles. Finish one report with the
measured qualification/limits and concrete next task; do not extend the event.
