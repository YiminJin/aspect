# Codex task: rotated bottom velocity constraint and small local A/B test

## Objective and scope

Implement an optional benchmark-plugin boundary treatment that keeps the prescribed fault-parallel bottom velocity and releases the fault-normal component. Compare it with the existing full Dirichlet treatment on a small, clean-start, coupled fault fixture that can run locally.

The production BP3 data show interior fault-normal accommodation of approximately 1–2% of plate speed that is excluded at the bottom. They do not yet establish that this restriction dominates the accumulated endpoint stress concentration. This task tests that hypothesis. It does not authorize changing the production run or declaring its 140-year stress history corrected.

Work in an isolated test directory/branch or worktree consistent with the repository instructions. Preserve unrelated changes. Use the existing signal and plugin mechanisms; avoid a general boundary-condition framework or solver refactor. Record the code revision and configuration used.

## 1. Boundary condition

Use the same straight-fault orientation as the restored BP3 chart:

\[
\mathbf t=(1/2,-\sqrt{3}/2),\qquad
\mathbf n_f=(\sqrt{3}/2,1/2).
\]

Here t points down dip and n_f points right/up. Obtain the orientation from the fixture geometry and check it against these values; do not infer it from the box boundary normal.

Let the existing loading function be u_D(x), and define g_t(x)=t dot u_D(x). Reuse that function and its stationary-profile table, including its current sign convention.

- **Case A:** retain the existing full bottom velocity u=u_D.
- **Case B:** impose only t dot u=g_t on the bottom. Leave n_f dot u free.

Use the natural complementary traction condition consistent with the stress convention. If mechanics solves for stress perturbations, the proposed condition is

\[
\mathbf n_f\cdot(\Delta\boldsymbol\sigma\,\mathbf n_b)=0,
\]

where n_b is the outward bottom-boundary normal. This is a component of the bottom traction, not zero fault-normal stress and not removal of the 50 MPa frictional background compression.

Verify the actual weak form and background/history stress terms. With a correctly formulated perturbation-stress weak form, no additional bottom traction load is needed for this zero natural component. Do not zero a constitutive stress field or omit history contributions from the volume assembly to implement it. If the code assembles total stress, use the matching background traction instead.

Keep a common zero-perturbation-traction top in both cases. Do not make every boundary Dirichlet in A while letting B fix pressure through a traction condition; that would introduce a pressure-reference difference into the comparison. Keep the same background stress and pressure convention in A and B.

## 2. Implement through the constraints signal

The pf-rsf header exposes:

```cpp
boost::signals2::signal<void (const SimulatorAccess<dim> &,
                              AffineConstraints<double> &)>
  post_constraints_creation;
```

The existing BP3 plugin already uses this signal. Inspect the current checkout's `compute_current_constraints()` call order and the coupled reconstructed-fault Newton path before adding the callback. Use `ASPECT_REGISTER_SIGNALS_CONNECTOR`, or the existing plugin connector if appropriate. Make the new behavior opt-in, defaulting to the current full Dirichlet behavior, using one clearly named plugin parameter rather than a new environment variable.

For B, remove bottom from the ordinary full-velocity, zero-velocity, and free-slip boundary lists. Keep left/right loading unchanged. An explicit zero-perturbation-traction bottom entry is acceptable if required by the boundary manager. The custom constraint supplies the prescribed component; reactions in that constrained direction remain possible.

At each independent bottom velocity support point impose exactly one equation:

\[
t_x u_x+t_y u_y=g_t.
\]

For this orientation, a well-conditioned choice is

\[
u_y=\frac{g_t}{t_y}-\frac{t_x}{t_y}u_x
    =\frac{g_t}{t_y}+\frac{1}{\sqrt 3}u_x.
\]

Conceptually, after resolving existing constraints:

```cpp
constraints.add_line(uy_dof);
constraints.add_entry(uy_dof, ux_dof, -tx / ty);
constraints.set_inhomogeneity(uy_dof, gt / ty);
```

This is illustrative, not a complete callback. Required details:

1. Pair x/y velocity DoFs by their shared FE support point. Do not assume adjacent global indices. Restrict to velocity DoFs on actual bottom faces, including Q2 edge-interior support points.
2. Respect locally relevant DoFs and distributed ownership. Add compatible, deterministic rows on ranks that need them. Avoid duplicate additions while traversing faces.
3. Preserve hanging-node constraints. Impose the relation on independent trace DoFs and resolve dependencies through the existing constraints machinery. Never silently skip a conflict and leave an unconstrained component or overwrite a hanging-node row.
4. At bottom corners, retain existing full side-boundary constraints and verify that their values satisfy t dot u=g_t. Do not add incompatible duplicate conditions there.
5. Confirm that constraints are inserted before the relevant close/assembly stages and that sparsity and preconditioner assembly support cross-component constraint entries. Use AMG for this task; GMG qualification is outside scope.
6. The physical solution must satisfy the inhomogeneous condition, while Newton corrections satisfy t dot delta_u=0. Inspect the existing physical-lift/homogenization path and reuse it. Do not reapply g_t to every Newton increment or let a callback rebuild nonzero increment data after homogenization.
7. Preserve the same behavior when constraints are rebuilt. Reuse existing restart infrastructure; no production restart experiment is required for this local task.

Emit a small constraint audit: support points covered, independent rows added, corners retained, resolved hanging-node cases, maximum physical tangential error, and maximum homogeneous-correction tangential error. Check that a fault-normal variation is not inadvertently eliminated. Evaluate strong trace accuracy against the FE representation of g_t; report interpolation error against the continuous profile separately.

Update the existing benchmark boundary-verification routine for B so it checks the prescribed projection rather than demanding both old Cartesian values. Keep A's original full-velocity checks intact.

## 3. Small local fixture

Reuse the smallest maintained reconstructed-fault fixture that can exercise the production mechanics. The following is the preferred target, with minor meshing changes permitted to reuse existing infrastructure:

| Item | Setting |
| --- | --- |
| Box | x in [-1000,1000] m, y in [0,1000] m |
| Fault | straight 60-degree line, bottom intersection (0,0), top intersection (-1000/sqrt(3),1000) m |
| Phase field | frozen mature fault, production stationary profile, ell=20 m |
| Mesh | fixed; approximately 5–6.25 m cells in a band covering the profile support, with graded coarsening away from it |
| Size target | roughly 5,000–15,000 active cells; report actual total and Stokes DoFs |
| Elements/history | production Q2/Q1 Stokes and current LLS/Q2 particle-history treatment; retain normal production particle density |
| Plate speed | Vp=1e-9 m/s |
| Material | G=3.203812032e10 Pa, reference viscosity 1e26 Pa s |
| Friction | uniform deep velocity-strengthening law: a=0.025, b=0.015, Dc=0.008 m, f0=0.6, Vref=1e-6 m/s; retain production regularization/damping |
| Normal stress | 50 MPa background plus actual mechanical perturbation; production 20 m normal filter |
| Left/right | same rigid block loading in A and B |
| Top | same zero perturbation traction in A and B |
| Bottom | A full velocity; B rotated mixed condition |
| Solver | AMG, same meaningful convergence tolerances in both cases |
| Run | clean start, fixed dt=2e5 s, initially 10 accepted steps per case |

Preserve the production endpoint source continuation and paired I_h completion treatment in both cases. Regenerate geometry-dependent fixtures and normalization data for this box. Do not reuse 50 km geometry constants, a 15–18 km friction transition, or completion files generated for the production mesh. Reuse the same material/profile law rather than substituting a tanh boundary profile.

Use 1–2 MPI ranks, one thread per rank, with a practical local budget of about 30 minutes for the initial pair. Measure the first step. If the target cannot fit that budget, reduce far-field mesh cost or use an existing smaller fixture while retaining near-fault resolution. Report the tradeoff; do not launch an unbounded sweep.

## 4. Give the fixture a controlled reason to accommodate

A perfectly uniform straight fault at steady creep may satisfy both boundary conditions almost identically. That is useful for checking the implementation, but is not evidence that the production accommodation mechanism is absent.

First perform a one-step uniform-creep check with Theta=Dc/Vp and the consistent steady frictional prestress. This should reveal constraint or normalization errors without an imposed along-fault disturbance. Reuse an existing suitable check if available.

For the primary A/B pair, initialize the same small smooth state perturbation in both cases:

\[
\Theta(y)=\frac{D_c}{V_p}
\exp\left[0.02\sin^2(\pi y/H)\right],\qquad H=1000\ \mathrm m.
\]

Use the fault-point height, with the normal extension consistent with the production property machinery. Keep the initial reference shear prestress at the uniform steady-creep value; do not locally re-equilibrate it to cancel this intentional disturbance. Document the disturbance as a local test input, not a BP3 parameter change.

Allow V, Theta, mechanical stress, and frictional normal-stress feedback to evolve through the real coupled solver. Do not prescribe all fault slip rates or replace the friction input with a fixed 50 MPa value in the main comparison. Keep the phase field frozen and omit nucleation physics from this uniformly strengthening fixture.

Ten steps span only about 23 days. The purpose is to compare the mechanical response and stress increments, not to recreate 140 years of accumulation or MPa amplitudes. Do not force an apparent effect by making the state perturbation extreme. If the baseline has no resolvable endpoint response, report that the fixture is inconclusive for stress mitigation.

## 5. Minimal outputs and interpretation

Write compact double-precision summaries and selected profiles; avoid full particle/quadrature CSV dumps and per-step volume VTU output. Initial/final volume output plus per-step scalar metrics are sufficient.

For each case report:

- Actual bottom tangential constraint error; bottom normal velocity; the comparable free-component weak traction residual in B, with corner constraints excluded.
- Flux through each boundary and net flux. Do not impose zero bottom vertical flux as an extra condition: fault-normal motion is oblique to the bottom, and the complete incompressible domain must balance.
- Raw and friction-used normal traction, shear traction, and reconstructed slip rate on the fault, especially the last 200 m down dip and an interior control segment.
- Pressure and deviatoric contributions separately, keeping the same background/gauge convention.
- Endpoint stress increments divided by dt, a weighted norm over a fixed physical endpoint neighborhood, and pointwise maxima with locations. Keep the upper endpoint and corners separate so a shifted concentration is visible.
- Deep/interior V/Vp, actual Delta ln Theta, nonlinear/linear iteration counts, rejected line-search trials, and wall time.

Make one concise A/B figure of final endpoint tractions and velocity components and one time-history figure of endpoint stress metrics. Compare matching physical times.

Interpret the outcome narrowly:

- Lower endpoint stress growth with preserved plate-parallel driving, acceptable flux balance, and coupled frictional response supports the mixed boundary treatment as a candidate.
- A reduction accompanied by loss of deep creep/loading or a relocated concentration is not a successful correction.
- No improvement does not eliminate all boundary mismatch hypotheses: this variant releases only fault-normal motion and keeps the same parallel profile.
- Do not rank solutions merely by how small u_n becomes. Its freedom is intentional.
- A local success validates implementation and provides mechanistic evidence. A short production A/B restart remains necessary before adopting the treatment for the full BP3 run.

If the initial pair gives a clear material difference and meets the runtime budget, do only the smallest follow-up needed to establish that it exceeds temporal or mesh error. Choose timestep halving or one local refinement based on the observed limitation; do not automatically run both or start a parameter sweep.

## 6. Deliverables

Provide the opt-in plugin change, common local configuration, A/B prm files, exact local build/run commands, compact results, and a short report with a specific next recommendation. Explain how default production behavior was preserved and whether any core change was actually necessary. Leave the production prm and existing restart files unchanged.

## Reference

ASPECT's signal documentation explicitly describes adding constraints through `post_constraints_creation`:
https://aspect-documentation.readthedocs.io/en/latest/user/extending/signals.html

The signal signature above was also checked in `pf-rsf/include/aspect/simulator_signals.h`. The coding agent should verify the active local revision and the current coupled Newton implementation before editing.
