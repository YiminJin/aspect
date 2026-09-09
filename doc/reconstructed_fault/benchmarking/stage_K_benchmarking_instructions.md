# Stage K — Step-by-Step Benchmarking Instructions

## Purpose and immediate assignment

Build a small, quantitative verification ladder for the reconstructed-fault phase-field model in ASPECT, then proceed to BP3-QD on the server. Preserve the completed Stage-J implementation. This is a benchmarking task, not another architecture redesign.

**Immediate assignment:** complete Step K0 and propose the concrete K1 fixture and reference calculation. Present the parameter table, boundary conditions, initialization contract, reference equations, error measures, and estimated cost before implementing K1. Do not implement the entire roadmap in one pass. After approval, implement and run K1, report the evidence, and stop at its review gate.

The sequence is:

| Step | Case | Primary question | Reference |
|---|---|---|---|
| K0 | Inventory and baseline | What equations, lifecycle, and tests are actually implemented? | Corrected repository specification and focused existing tests |
| K1 | Homogeneous in-plane shear, fixed phase field, prescribed normal stress | Does the production coupled solve reproduce the finite-width mechanical/history equations? | Independent scalar finite-step reference |
| K2 | Spatially varying slip, fixed phase field | Does non-local bulk/fault coupling converge, first with prescribed and then with true normal stress? | Fixed-length-scale spatial and temporal convergence |
| K3 | Homogeneous shear with evolving normal phase-field profile | Does the complete history-to-phase-field feedback reproduce a reduced reference? | Independent one-dimensional finite-length-scale calculation |
| K4 | Controlled regularization study | Which differences are discretization errors and which are finite-width/model effects? | Separately resolved runs at different regularization lengths |
| K5 | BP3-QD or an explicitly named modified BP3 | Does the model reproduce a matched earthquake-scale mechanical benchmark? | Versioned external reference with documented equation/parameter agreement |

K4 is a bounded study, not an invitation to exhaustively sweep parameters. A small pilot may establish what should be investigated on the server rather than locally.

---

## 1. Authority, scope, and working rules

### 1.1 Read the corrected checkout, not an obsolete planning snapshot

Inspect `doc/reconstructed_fault/current_design.md`, `doc/reconstructed_fault/specification.tex`, the current theory note corresponding to `pf_rsf.tex`, and the final Stage-J review/decision record. Identify the actual current paths if they differ.

Older Stage-F–J plans are historical context. They must not undo the subsequent initialization, temperature, convergence-reporting, or history-commit corrections.

This document defines the approved benchmarking progression. Its scalar and one-dimensional reductions are benchmark constructions derived from the stated equations, not permission to change production constitutive laws. Confirm their assumptions against the corrected checkout before implementing a reference. Report a concrete contradiction with the equations or lifecycle rather than silently reconciling it.

### 1.2 Preserve the established scope

- Work in the existing 2-D bulk / 1-D ordered reconstructed-fault implementation.
- Keep fault geometry fixed after initialization for K1–K3. Evolving a normal phase-field profile is allowed in K3; fault-tip extension, branching, merging, and reconstruction after propagation are not.
- Do not add 3-D support, a new nonlinear solver, a generic benchmark framework, speculative public interfaces, or new constitutive parameters.
- Reuse production geometry, associations, material evaluation, assembly, coupled solve, and history lifecycle. Independent references and diagnostic helpers belong in benchmark/test code.
- Preserve the approved distinction between local bulk Maxwell coefficients and profile-uniform surface cohesive coefficients, including the approved surface-temperature evaluation.
- Build with `-j4`. Run focused tests and benchmarks only, not the entire ASPECT test suite. Use one MPI rank initially and selected two-rank checks afterward.
- Do not submit a server job before presenting its resource request and receiving approval.

### 1.3 Do not repeat the Stage-J debugging failure pattern

Do not loosen solver tolerances, bypass an equation, change parameter meaning, or rewrite expected output solely to obtain a pass. Iteration exhaustion is a failure, not convergence. A larger iteration budget is acceptable when the recorded convergence history justifies it and the accuracy requirement is unchanged.

After a failed run, classify the failure and state the next hypothesis before editing. Separate fixture defects, reference defects, discretization error, nonlinear/linear solver failure, and model disagreement. Re-run the smallest case that distinguishes the hypotheses.

Preserve unrelated working-tree changes. Record the revision and diff used for each reported result. Follow the repository's commit policy; do not make broad resets.

---

## 2. Keep four different errors separate

### 2.1 Notation

Use:

- \(\ell\): phase-field regularization length;
- \(h(\phi)=1/g(\phi)-1\): constitutive function, **not mesh spacing**;
- \(\Delta x_n\), \(\Delta x_s\): bulk resolution normal and tangent to the fault;
- \(\Delta s_\Gamma\): fault-node spacing;
- \(w_{H_0}\): a documented measure of the narrow feature in the prescribed initial driving-force profile;
- \(L_{\rm var}\): a relevant mechanical variation scale, such as a perturbation width or curvature radius;
- \(\lambda_b=\eta_b/G_b\), \(\lambda_\Gamma=\eta_\Gamma/G_\Gamma\): bulk and surface Maxwell times.

Distinguish:

1. **Algebraic error:** incomplete linear/nonlinear solves and integration tolerances.
2. **Spatial error:** FE, particles/CPDI, reconstruction, surface Q1 representation, projection, and profile quadrature.
3. **Temporal error:** mechanical/history time discretization and operator splitting.
4. **Model/regularization error:** finite diffuse width, constitutive differences, and differences from the external reference.

### 2.2 What a relatively large \(\ell\) can verify

K1 and K3 compare the implemented finite-width equations with reductions that retain the finite profile. They are not miniature sharp-fault earthquake simulations. They do not require the physical interpretation of \(\ell\to0\).

Resolve the actual profiles, not just \(\ell\). Record \(\Delta x_n/\ell\), particle spacing, and \(\Delta x_n/w_{H_0}\) or an equivalent quantitative profile-resolution measure. If the profile has no convenient single width, use independently sampled profile/integral errors instead of inventing a width.

A sufficiently contained profile is still required: account for the contribution outside the domain or association strip. Do not normalize with an effectively infinite profile integral while silently dropping a significant part of its deformation in bulk assembly.

For sharp-fault comparisons, also assess ratios such as \(\ell/L_{\rm var}\). Smooth numerical data alone do not establish a valid thin-fault approximation.

### 2.3 Order of refinement

First hold \(\ell\), physical parameters, domain, and physical initial data fixed. Refine spatial resolution and timestep separately. Only afterward vary \(\ell\), keeping each profile resolved.

Changing \(\ell\) may change the calibrated degradation parameters and the stationary initial profile. Recompute them according to the approved model. Do not merely stretch an old \(H_0\) field or hold a derived degradation parameter fixed accidentally.

Changing meters to kilometers without changing dimensionless resolution requirements does not reduce the number of unknowns. Obtain cheap tests by simplifying spatial variation and duration.

---

## Step K0 — Inventory, baseline, and benchmark contract

### K0.1 Inspect and record the implemented mathematical contract

Create a short source map using repository-relative paths and fully qualified symbols. Record:

- the selected regularized or logarithmic friction law and its exact parameter meanings;
- the selected state-update formula and its treatment of a fixed accepted slip rate;
- timestep-zero semantics for each of \(V,\Theta,T^{\rm coh},\tau,H,I_h\);
- normal-stress modes and pressure scaling/gauge conventions;
- bulk/surface coefficient evaluation and surface-temperature sampling;
- fixed-phase-field semantics, including what happens to \(H\);
- the current finite-step \(H\) equation, endpoint branches, and irreversible maximum;
- raw accepted cohesive samples versus projected committed cohesive history, and the traction/strain-rate convention used to commit Maxwell stress;
- particle advection, previous-field sampling, cache invalidation, checkpointing, and supported timestep restrictions.

Do not infer these details from function names. In particular, `Evolve phase field=false` must not be presumed to freeze exactly the desired fields without checking its documented and implemented behavior.

The approved baseline retains supplied \(\Theta_0\) and \(H_0\) at timestep zero rather than physically evolving them over the numerical Maxwell initialization interval. Confirm the final implementation and document its separate initial cohesive/stress semantics. Do not replace it with a convenient reference initialization silently.

### K0.2 Establish a focused baseline

Run the affected Stage-J smoke/lifecycle test and the essential existing constitutive/coupling regression tests. Reuse recent trustworthy results where appropriate, recording the revision. Do not rerun every earlier test indiscriminately.

Verify that the initialization actually reaches its requested phase-field tolerance. Record solver exit status and final residual, not just process exit code.

### K0.3 Design the K1 parameter set before coding

Provide a dimensional parameter table covering geometry, \(\ell\), initialization, \(G_b,\eta_b,G_\Gamma,\eta_\Gamma\), normal pressure, friction parameters, radiation damping, loading, particle count, FE/fault resolution, and timestep strategy.

Use homogeneous temperature and composition first. Choose accessible, resolvable parameters; they need not represent an earthquake. Do not introduce independently tunable surface material constants unless the production API already supports them. Homogeneity will normally make the effective bulk and surface coefficient values coincide even though their roles remain distinct.

Estimate a small local pilot cost. Prefer a smoke run lasting minutes rather than an unbounded campaign. Measure actual runtime before proposing the convergence matrix. Do not assume anisotropic cells or extreme aspect ratios are harmless merely because the reference is one-dimensional.

### Gate K0

Present the source map, unresolved concrete conflicts, K1 fixture, independent-reference design, numerical error targets, and pilot command. Stop for approval before implementing K1. Do not reopen already documented decisions without identifying a specific contradiction.

---

## Step K1 — Homogeneous in-plane shear with fixed phase field

### K1.1 Construct a symmetry-compatible production fixture

Use

\[
\Omega=(0,L_s)\times(-W/2,W/2),
\qquad
\Gamma=\{(x,0):0<x<L_s\}.
\]

Take

\[
\mathbf s=\mathbf e_x,\qquad \mathbf n=\mathbf e_y,
\qquad \mathbf u=(u_x(y),0).
\]

This is in-plane shear. `u` denotes velocity here; do not confuse it with displacement notation in another document.

Prescribe the top/bottom tangential velocity difference

\[
U_k=u_x(W/2)-u_x(-W/2),
\]

with zero normal velocity. A convenient symmetric choice is \(u_x=\pm U_k/2\) at the two boundaries.

Use lateral boundary conditions compatible with an x-independent solution. Prefer existing periodic bulk support when compatible with the current fault representation. Verify endpoint/association behavior. Do not invent a periodic fault topology, or assume that an arbitrary zero-traction lateral boundary preserves uniform shear. If periodic bulk support is unsuitable, present a compatible alternative and its reference boundary data before implementation.

Use prescribed, constant, positive normal stress \(\sigma_*\) in friction through the supported adiabatic-pressure mode. Use homogeneous material and temperature, no gravity-driven complications, and a normal-only initial profile. Keep the phase field fixed through the documented supported path.

Use the production coupled solver and history updates for all fields that the fixed-profile mode is intended to evolve. Record any deliberately frozen histories. A benchmark-only fixture is acceptable when needed, but do not silently change a production parameter's meaning to create it.

### K1.2 Verify geometry and initial data

After reconstruction, measure fault position, segment orientation, and variation along the fault. The prescribed line being horizontal is not proof that the reconstructed line is horizontal.

Require geometric error to be below the stated benchmark error budget or demonstrate its convergence. If it dominates, repair/refine the symmetry-compatible initializer or reconstruction fixture. Do not inflate comparison tolerances to hide it.

Ensure \(I_h>0\), the permitted phase-field range is respected, and profile tails are adequately contained. Do not set \(\phi=1\) merely to imitate a fully broken fault when the implemented degradation evaluation excludes that endpoint.

Initialize through the approved production path. Check whether its accepted initial shear stress is uniform enough for the scalar reduction. If it is not, the scalar reference cannot simply average it and claim exact equivalence; investigate the fixture or explicitly re-derive the needed reference.

### K1.3 Implement an independent scalar finite-step reference

Keep the reference small and test-only, preferably a readable Python script using an existing tested scalar root solver. Do not call production constitutive routines to generate expected values.

For one real timestep, let

\[
\beta_b=e^{-\Delta t_kG_b/\eta_b},\qquad
\kappa_b=\eta_b(1-\beta_b),
\]

and define \(\beta_\Gamma,\kappa_\Gamma\) similarly using the surface cohesive coefficients. Suppress the timestep subscript on these coefficients below.

Let \(q_k=\tau_{xy,k}\) be the uniform shear stress, and use \(C_k=T_k^{\rm coh}\) to avoid confusing cohesive traction with temperature.

For \(\mathcal S=(\mathbf s\otimes\mathbf n+\mathbf n\otimes\mathbf s)/2\), simple shear reduces the bulk law to

\[
q_k=\kappa_b\left(\partial_yu_x-\upsilon_k\right)
       +\beta_bq_{k-1}.
\]

Using \(\int_{-W/2}^{W/2}\upsilon_k\,dy=V_k\) gives

\[
\boxed{
q_k=\beta_bq_{k-1}+\frac{\kappa_b}{W}(U_k-V_k).
}
\tag{K1-a}
\]

For a fixed profile,

\[
\boxed{
C_k=\frac{\kappa_\Gamma}{I_h}V_k+\beta_\Gamma C_{k-1}.
}
\tag{K1-b}
\]

Solve the scalar equation

\[
\boxed{
\begin{aligned}
F_k(V)={}&\beta_bq_{k-1}+\frac{\kappa_b}{W}(U_k-V)
-\frac{\kappa_\Gamma}{I_h}V-\beta_\Gamma C_{k-1}\\
&-\sigma_*\mu(V,\Theta_{k-1})-\eta^dV=0.
\end{aligned}
}
\tag{K1-c}
\]

Then independently apply the approved state update

\[
\Theta_k=\operatorname{StateUpdate}(V_k,\Theta_{k-1},\Delta t_k).
\]

Write the actual formula in the reference documentation. Do not substitute backward Euler for an exponential update, or an exponential update for backward Euler. Do not solve through \(\Theta_k(V)\) inside (K1-c) when production mechanics freezes \(\Theta_{k-1}\).

Use the exact selected friction law, including regularization. Start with `rate_state` and an interior solution safely above \(V_{\min}\), so this benchmark does not depend on a lower-bound active-set interpretation. Check the reference root's admissibility throughout the run.

For positive fixed-state friction slope, the derivative is

\[
F_k'(V)=-\frac{\kappa_b}{W}-\frac{\kappa_\Gamma}{I_h}
-\sigma_*\left(\frac{\partial\mu}{\partial V}\right)_\Theta-\eta^d<0.
\]

This is a useful check on the scalar reference, not a claim about global coupled Newton convergence. Use a dynamically expanded upper bracket if needed; it is reference-solver machinery, not a constitutive `Vmax`. If no admissible interior root exists, report a fixture issue rather than clipping the root.

The boundary difference is **not** the fault slip rate: do not set \(U_k=V_k\). The finite bulk deformation in (K1-a) is part of the reference.

### K1.4 Make the reference independent without hiding initialization errors

Use two clearly labeled checks:

**Conditional mechanical/time-update check.** Independently integrate \(h(\phi_h)\) from the exported initial FE profile and approved degradation formula. Use the once-recorded accepted initial histories as starting data if needed to isolate subsequent mechanical evolution. Record that this is conditional on production initialization. Do not use the production-reported \(I_h\) as the sole expected value, and do not reset reference histories from ASPECT after every step.

**Initialization/profile check.** Independently validate the prescribed \(H_0\), phase-field residual, reconstruction error, and profile integral. The fully independent one-dimensional initialization/reference is completed in K3. Do not call the conditional comparison a verification of the initial-condition algorithm.

For direct discrete-time comparison, feed the reference the actual accepted timestep sequence and loading convention used by ASPECT. This avoids confusing differences in adaptive timestep selection with a defect in the update equations. A separate refinement study still has to test temporal accuracy.

### K1.5 Recover a velocity-profile reference

For fixed profile, \(\upsilon_k(y)=h(\phi_0(y))V_k/I_h\). Thus

\[
u_{x,k}(y)=u_{x,k}(-W/2)
+\frac{q_k-\beta_bq_{k-1}}{\kappa_b}(y+W/2)
+V_k\int_{-W/2}^{y}\frac{h(\phi_0(z))}{I_h}\,dz.
\tag{K1-d}
\]

Compare the bulk velocity profile as well as the scalar fault histories. Verify \(u_y\), divergence, and along-fault variation are consistent with the uniform solution within discretization error.

### K1.6 Use informative loading and observables

Start with modest, stable sliding and a loading change that produces a measurable response. Align a discontinuous loading change with timestep boundaries, or specify a smooth ramp and its discrete sampling rule.

Choose a non-steady initial state so the expected \(\Theta\) increment is resolvable. A no-op state update must fail the comparison. Do not use a tiny timestep that hides a missing update inside the tolerance.

Output accepted-time histories of \(V,\Theta,C,q,I_h,U\), cumulative slip, actual timestep, residuals, and iteration counts. Record min/mean/max along the fault, not only one convenient vertex. Independently integrate slip using the declared time rule; label higher-order postprocessed quadrature separately if used.

Check

\[
E_{\rm norm}=\frac{|\int\upsilon_kdy-V_k|}
{\max(|V_k|,V_{\rm scale})},
\]

with a physically meaningful fixed diagnostic scale. Also report raw dimensional errors.

### K1.7 Run a small convergence matrix

1. One-rank smoke run with strict existing solver tolerances.
2. Three spatial levels, normally with approximately twofold refinement, at fixed \(\ell\) and sufficiently small timestep. Ensure particle/profile integration error does not dominate unnoticed.
3. Three timestep levels on a resolved spatial configuration. Compare at identical physical times and load histories. Record actual accepted steps if other limits shorten them.
4. One selected two-rank run and one short restart-versus-uninterrupted comparison, reusing existing infrastructure.

Do not run the full Cartesian product of every mesh, particle, fault-grid, and timestep choice. If a plateau occurs, vary the suspected limiting resolution separately.

After the rate-state baseline passes, a small `rate_dependent` variant may reuse the fixture. It has no \(\Theta\) update and may have a different root structure. Confirm a unique admissible branch for the chosen fixture rather than carrying over the monotonicity claim above.

### Gate K1

Provide reference code, fixture, exact run commands, error/convergence tables, profile/history plots, and a short interpretation. Pass requires actual convergence to the independent finite-width reference and successful lifecycle checks, not merely a small internal residual. Stop before K2.

---

## Step K2 — Nonuniform slip and non-local coupling, fixed phase field

### K2.1 Add one smooth along-fault perturbation

Reuse K1 as much as possible. Keep temperature, composition, and the normal phase-field profile simple. Introduce one smooth variation, preferably in the initial rate-state variable, for example

\[
\Theta_0(s)=\bar\Theta_0[1+\varepsilon f(s)],
\]

where `f` is specified analytically, compatible with the boundary conditions, and \(\varepsilon\) preserves positivity. Select a modest amplitude and a resolved tangential width. Report the realized nodal field after initialization/projection.

Use prescribed normal stress first. This is a nonuniform production-path test, not an extension of the scalar K1 reference. Do not compare it with (K1-c) independently at each vertex: that would discard the non-local coupling being tested.

Measure how traction, bulk deformation, and slip outside the initially perturbed region respond. A plausible picture or nonzero distant response is not sufficient; quantify its convergence.

### K2.2 Establish a finite-\(\ell\) numerical reference

At fixed physical data and fixed \(\ell\), refine bulk resolution, fault spacing, and timestep in a controlled sequence. Use common physical sampling locations/arclength for comparisons, not corresponding node indices on different meshes.

Report spatial profiles of \(V,\Theta,q,\sigma_n\), cumulative slip, and relevant bulk velocity/pressure/stress sections. Use mass/arclength-weighted or common-grid norms rather than raw nodal Euclidean norms that change with node count.

Verify any symmetry expected from the actual geometry and loading. Do not require a symmetry broken by the chosen loading direction or boundary conditions.

### K2.3 Repeat with true normal stress

Switch only the frictional normal-stress choice to the supported true-traction mode, retaining the matched pressure convention. Inspect short-wavelength variations and mesh dependence, particularly the behavior that motivated prescribed pressure in the earlier implementation.

Do not add smoothing or revert silently to prescribed pressure if this case fails. Separate an incorrect residual/Jacobian, underresolution, and a difficulty inherent to the formulation.

A symmetric homogeneous planar configuration may legitimately give no net normal-stress change on the fault. Do not call it a nonzero normal-feedback test merely because the switch is enabled. If that happens, retain it as a symmetry check and propose one minimal supported asymmetry or geometry/loading variant to exercise nonzero feedback before claiming coverage.

### K2.4 One alignment check, not an orientation campaign

After convergence is established, perform one controlled grid-offset or orientation variation if affordable. A translated line avoids changing the constitutive problem when boundaries permit it. A rotated fault requires correspondingly transformed loading and boundary conditions; rotating the line alone is not an equivalent benchmark.

### Gate K2

Report prescribed- and true-normal-stress results separately. Include resolution evidence for the non-local response, any oscillation diagnostics, and what remains unverified. Do not advance by deleting the troublesome branch. Stop before K3.

---

## Step K3 — Evolving phase-field profile with a 1-D reference

### K3.1 Return to the homogeneous-along-fault configuration

Use the K1 box and loading symmetries, now with phase-field evolution enabled through the corrected production path. Keep fault topology/location fixed. Choose a moderate interval and loading that produce an observable change in \(H\), \(\phi\), and \(I_h\), without immediately reaching an excluded degradation endpoint.

No along-fault propagation is requested. Homogeneity is deliberate: all particles at the same normal coordinate should see the same intended problem, and the independent reference remains inexpensive.

### K3.2 Write the independent normal-profile problem explicitly

For homogeneous \(\mathcal G_c\), the history-field phase equation in the unconstrained interior has the reduced form

\[
g'(\phi_k)H_{k-1}
+\frac{\mathcal G_c}{c_0}
\left[\frac{\alpha'(\phi_k)}{\ell}
-2\ell\,\partial_{yy}\phi_k\right]=0.
\tag{K3-a}
\]

Confirm this against the current specification. Supply the same normal boundary conditions, admissible phase range, and the actual history-field/irreversibility constraint treatment. Equation (K3-a) alone is not permission to replace bound or active constraints by an unconstrained solve.

Implement a small independently assembled one-dimensional reference. Use accurate normal integration and show that its own mesh and solver tolerances are below the production comparison error. Do not build a second general-purpose phase-field framework.

### K3.3 Follow the indexed cycle without a hidden extra iteration

For each real step \(k>0\):

1. Solve the normal phase-field problem using \(H_{k-1}\) to obtain \(\phi_k\).
2. Compute \(h_k(y)\) and \(I_{h,k}=\int h_kdy\).
3. Solve the homogeneous mechanical reference with frozen \(\Theta_{k-1}\) and old histories.
4. Compute the new cohesive response and the approved state update.
5. Compute the local crack strain, bulk stress, and finite-step driving candidate; apply the irreversible maximum.
6. Save the resulting histories for the next phase-field solve.

The evolving-profile cohesive relation is

\[
C_k=\frac{\kappa_\Gamma V_k+
\beta_\Gamma I_{h,k-1}C_{k-1}}{I_{h,k}}.
\tag{K3-b}
\]

Use this in place of (K1-b). Relation (K1-a) remains applicable under the stated homogeneous-stress/coefficient and profile-normalization assumptions. Verify those assumptions throughout the reference run.

The normal crack strain is

\[
\upsilon_k(y)=\frac{h_k(y)}{I_{h,k}}V_k
+\frac{\beta_\Gamma C_{k-1}}{\kappa_\Gamma}
\left[h_k(y)\frac{I_{h,k-1}}{I_{h,k}}-h_{k-1}(y)\right].
\tag{K3-c}
\]

For the retained homogeneous reference, check both

\[
\int\upsilon_kdy=V_k,
\qquad
\int\upsilon_k^{\rm hist}dy=0.
\]

Use the same consistent profile integration bounds in these identities and in \(I_h\).

The approved finite-step driving candidate is

\[
\mathcal H_k(y)=\frac{\Delta t_k}{2\kappa_\Gamma}
\frac{[h_k(y)C_k]^2-
[\beta_\Gamma h_{k-1}(y)C_{k-1}]^2}{[1-g_k(y)]^2},
\qquad
H_k(y)=\max(H_{k-1}(y),\mathcal H_k(y)).
\tag{K3-d}
\]

Use the final documented endpoint policies from the corrected Stage-J implementation/specification; do not infer a branch from a floating-point `0/0`. Preserve the distinction between the finite-step expression and the additional approximation \(\Delta t/\kappa_\Gamma\approx1/G_\Gamma\). Do not replace (K3-d) by current elastic stress energy.

Despite earlier informal uses of the word “increment,” \(\mathcal H_k\) is the candidate in a maximum rule here, **not** an instruction to compute \(H_{k-1}+\mathcal H_k\).

Where the same pointwise cohesive relation holds, independently compare (K3-d) with

\[
\frac{\Delta t_k}{(1-g_k)^2}
\left[\frac12\kappa_\Gamma\upsilon_k^2
+\beta_\Gamma h_{k-1}C_{k-1}\upsilon_k\right].
\]

Account explicitly for any Q1 projection difference before expecting algebraic equality: a raw constitutive sample and its projected/interpolated field are not generally identical. In the homogeneous-along-fault fixture, constant surface fields should remove that particular ambiguity to numerical accuracy.

### K3.4 Make feedback measurable and test timestep sensitivity

Require a real change above the comparison tolerance, not only monotonicity of \(H\). Verify that \(H_k\) actually enters the following phase solve. Do not accept a test that passes with the history update or following phase solve removed.

Compare \(\phi(y),H(y),I_h,V,\Theta,C,q\), cumulative slip, and profile-integral diagnostics at common physical times. Use an independently resolved reference initial profile as well as the later updates.

Perform a fixed-\(\ell\) normal-resolution study and a separate timestep study. Temporal agreement of two implementations at one timestep verifies that discrete update, not the continuum limit. If reducing the timestep substantially changes feedback or fails to approach a stable result, report it as a numerical/model finding; do not change the energy formula or maximum rule to force convergence.

### Gate K3

Require initialization plus at least two real steps demonstrating the full feedback cycle, and a longer short transient sufficient for meaningful convergence comparisons. Provide the independent 1-D reference and its own accuracy evidence. Stop before a regularization/server campaign.

---

## Step K4 — Separate finite-width effects from discretization

### K4.1 Choose one established case

Use K1 for a cheap profile/normalization check and, if affordable, one K2 configuration for mechanically meaningful length-scale sensitivity. Do not expect K1 to establish general thin-fault accuracy: its symmetry deliberately removes many width-dependent effects.

### K4.2 Vary \(\ell\) only after resolution is established

For a small number of \(\ell\) values:

1. Keep the intended physical geometry, loading, and material problem fixed.
2. Recompute length-dependent degradation calibration and initial profile consistently.
3. Resolve the actual profile and record all bulk/particle/fault resolution ratios.
4. Keep temporal and algebraic errors below the targeted comparison error.
5. Compare physically meaningful observables and document the remaining uncertainty.

An \(\ell\) sequence with fixed cells across the band is useful for exploration, but is not by itself a clean isolation of regularization error. Include at least one independent refinement check at a smaller \(\ell\) or label the study accordingly.

### K4.3 Decide readiness, not universal validity

Report which \(\ell\) is affordable and resolved locally, which scales would require the server, and whether finite-width effects are below the requested benchmark accuracy. Do not claim a universal minimum length or universal sharp-fault convergence rate from this small study.

If a comparison develops new short mechanical scales, revise the scale-separation assessment. Smooth \(H_0\) does not guarantee that later \(H\) and slip localization are resolved.

---

## Step K5 — Prepare and run BP3 carefully

### K5.1 Make a model-match table before a server run

Retrieve the official versioned BP3-QD specification and relevant reference outputs, using the repository's existing benchmark resources where available. Record source, version/date, file hashes, units, and output conventions. Do not recreate the parameter table from memory or an old approximate setup.

Compare the reference with the actual current implementation:

| Item | Required check |
|---|---|
| Kinematics | 2-D in-plane formulation and the specified elastic plane-strain response |
| Compressibility | Actual bulk equations and Poisson-ratio equivalence; do not equate an incompressible run with a different elastic reference |
| Bulk rheology | Elastic reference versus Maxwell response over the entire comparison interval, not just one timestep |
| Normal stress | Initial stress versus its subsequent evolution; prescribed-pressure mode is not automatically standard BP3 |
| Friction | Exact regularization, state law, parameters, and initialization |
| Cohesion | Whether residual surface cohesive traction/stiffness changes the prescribed reference problem |
| Geometry and loading | Dip, free surface, fault extent, far-field/basal conditions, and loading velocity |
| Domain | Truncation/boundary sensitivity, not just local mesh resolution |
| Regularization | \(\ell\), actual profile width, and residual finite-width effects |
| Numerics | Spatial resolution, timestep controls, output stations, and event definitions |

Classify every mismatch as matched, controlled approximation, or substantive model difference. If the current equations do not match standard BP3, call the run **modified BP3** and say exactly how it differs. A standard BP3 mismatch alone is then not a diagnosis of a coding defect.

Do not silently add compressibility, change normal-stress feedback, or suppress cohesion to obtain a better visual match. Such changes require a separate explicit decision.

BP3 is principally an external test of the prescribed fault's mechanical evolution. It does not replace the finite-\(\ell\) \(H\)/phase-feedback verification in K3 or verify fault propagation.

### K5.2 Prepare the server pilot

Use the existing server environment and supported ASPECT build. Record compiler/library/build provenance and ensure the executable is compatible with the selected node architecture.

Present one bounded job request with nodes, MPI ranks, walltime, expected memory, output size, and restart strategy. Do not submit a large mesh/time sweep as the first job.

Run in this order:

1. Initialization and a few early loading steps.
2. Loading through nucleation and the first event, with adequate event-time output.
3. A short post-event interval.
4. Only after satisfactory comparison, extend to additional cycles.

A coarse pilot verifies execution and estimates cost; it is not the quantitative reference comparison. Save useful restart points and verify restart reproducibility before relying on them for a long campaign.

### K5.3 Compare prescribed observables

Use the external benchmark's station locations and definitions. Compare slip rate, cumulative slip, state, shear stress, effective normal stress, event onset/arrival measures, and slip distributions as applicable.

Compare at the same physical times first. Event-aligned plots may be included as secondary shape comparisons, but must not hide onset-time or recurrence-time errors by shifting curves silently.

Report residual cohesive resistance, relevant Maxwell relaxation over the simulated duration, boundary sensitivity, and spatial/temporal/regularization uncertainty alongside the reference discrepancy.

If the lagged next-step RSF controller permits a poorly resolved acceleration, record the evidence and propose a separate cutback/retry decision. Do not silently introduce a new controller in a benchmark script or cap timestep decreases in violation of an admissibility restriction.

### Gate K5

Present the model-match table, measured resource costs, first-event comparisons, convergence evidence, and remaining discrepancies before requesting a long-cycle run.

---

## 3. Common error measures and acceptance rules

Choose numerical targets before examining a passing/failing result. Targets must have an interpretation in the benchmark; do not use machine-minimum denominators or adjust tolerances after the fact without recording the reason.

For scalar histories, report raw errors and a combined absolute/relative criterion such as

\[
|X_{\rm ASPECT}-X_{\rm ref}|
\le \epsilon_{{\rm abs},X}+\epsilon_{{\rm rel},X}|X_{\rm ref}|.
\]

For slip rates spanning orders of magnitude, also report

\[
E_{V,\log}=\max_{s,t}\left|\log_{10}
\frac{V_{\rm ASPECT}(s,t)}{V_{\rm ref}(s,t)}\right|,
\]

with positive admissible states and separately reported bound-active locations. Do not use this logarithmic diagnostic to conceal absolute slip errors.

For profiles, use a common physical grid or properly weighted norms. For uniform benchmarks, report the along-fault range as well as the mean error. For state changes, compare the increment with an error allowance small enough that an omitted update fails.

A passing verification stage requires: successful solver convergence; agreement with the stated reference or a demonstrated resolution trend; no unexplained conservation/lifecycle defect; and a reproducible record. A plausible plot and successful process exit are insufficient.

Do not demand an unqualified textbook convergence rate for nonsmooth profiles or constrained problems. State the regularity assumptions, observed rate, and error plateau mechanism. Two fine runs agreeing can be useful evidence, but is not a substitute for an independent reference where one is available.

---

## 4. Deliverables, code readability, and reporting

Follow the existing benchmark layout. Suggested paths, to adapt rather than duplicate, are:

```text
doc/reconstructed_fault/
    stage_K_benchmarking_instructions.md
    stage_K_progress.md
    benchmark_reference_equations.md

benchmarks/reconstructed_fault/
    README.md
    uniform_shear/
    nonuniform_slip/
    evolving_profile/
    bp3/
    reference/
    analysis/
```

Keep small reusable smoke assertions in `tests/` according to repository conventions; keep expensive convergence/server campaigns outside the default full test run. Reuse existing benchmark directories when present. Do not commit large raw result sets or duplicate an established postprocessor.

Each case needs a documented input, reference/analysis command, machine-readable table, units, sample locations, and a short result report. Scripts should validate input columns, identify nonconverged runs, fail clearly on missing data, and avoid hard-coded user paths.

For long reference or diagnostic functions, add short phase comments exposing the mathematical sequence. Explain non-obvious signs, pressure conversion, integration weights, and accepted/committed state use. Do not narrate trivial assignments or hide a reference behind many tiny wrappers.

After each stage, update a compact progress record containing:

1. Revision/diff, relevant build details, and exact commands.
2. Mathematical question and assumptions actually tested.
3. Parameters, resolution, timestep sequence, and initial-state provenance.
4. Quantitative results, runtime/memory, and reference accuracy.
5. Failures or limitations, the next concrete action, and the review gate status.

Keep an explicit distinction between a measured result, a theoretical derivation, an implementation assumption, and a proposed next experiment. Re-read this record after context compaction or model handoff rather than reopening settled choices.

---

## 5. Source and derivation map

This guide operationalizes the benchmarking sequence approved in the discussion. It does not supply a new constitutive model.

- **Corrected current repository specification and final Stage-J record:** authoritative equations, coefficient ownership, initialization, projection, history commit, and timestep semantics. Read the latest checkout before coding; earlier uploaded Stage-J plans predate the reported cleanup.
- **`reconstructed_fault_phase_field_fault_reredesign.md`, Stage K and Jacobian-verification sections:** staged progression from fixed phase/prescribed normal stress to true normal stress and evolving phase field; small coupled verification before BP3.
- **`reconstructed_fault_surface_rsf_design.md`, sections on `I_h` and the surface cohesive law:** profile normalization, surface cohesive update, and evolving-profile history correction. Resolve any difference against later approved corrections, not by reverting the implementation.
- **Theory note corresponding to uploaded `pf_rsf(1).tex`:** Maxwell law; `eq:H`, `eq:H small-step`, `eq:H+`; degradation calibration; and the narrow initial driving-force profile motivating CPDI.
- **Equations K1-a–K1-d:** benchmark-specific reduction of homogeneous in-plane shear, with finite bulk deformation and the normalized finite-width crack strain retained.
- **Equations K3-a–K3-d:** one-dimensional reduction for homogeneous materials and an x-independent evolving profile; constraints, endpoint branches, and time indexing must match the corrected specification.
- **Official SEAS BP3-QD materials:** retrieve and version them at K5. This guide deliberately does not substitute a recalled parameter table for those materials.

**Start now with K0 and the concrete K1 proposal only.** Do not begin all benchmark families or a server campaign before the intervening review gates.
