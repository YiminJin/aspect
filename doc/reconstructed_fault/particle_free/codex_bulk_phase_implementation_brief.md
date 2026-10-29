# Codex implementation brief: bulk cohesive phase evolution and bounded P4 testing

20 October 2026

Use this brief together with **bulk_surface_cohesive_derivation.tex**. The user authorizes implementing this experimental propagation model and proceeding through the tests below. The mathematical note is a constitutive proposal, not an already validated algorithm.

## Objective and scope

Demonstrate that a finite phase-field fault can evolve under mechanical loading using a constitutive law defined in the bulk, including ahead of the reconstructed surface. The reconstructed surface initially observes the solution; it does not decide where mechanical damage is allowed.

Use the current propagation worktree and inspect its actual source/status first. Read AGENTS.md, the propagation PLAN/STATUS, the supplied derivation, and the P2/P3/P4a reports. Preserve unrelated work. Record the new task as a distinct bulk-model stage; retain the scope and limitations of earlier evidence.

Implement a default-off propagation option with a clear constitutive module and small integration changes. Keep BP3 and the existing surface model available and unchanged by default. Do not begin a broad cleanup, solver rewrite, or performance campaign. Update the active plan to reflect this authorized scope rather than stopping at historical “await selection” notes.

The selected model has small strain, fixed shear orientation throughout the specimen, constant prescribed normal compression and frictional resistance, homogeneous G and eta, and equal bulk/cohesive Maxwell times. Initially use a fixed mesh and serial execution. RSF, advection, objective rotation, moving meshes, adaptive refinement, branching, and production-server runs are outside this task.

## 1. Implement one coherent bulk constitutive law

Use the derivation as the mathematical reference. With S:S = 1/2, h = g^(-1)-1, lambda = eta/G, and bulk recoverable cohesive engineering shear a, the central equations are

\[
\tau=T-2GaS,\qquad q=Ga/h,
\]
\[
\dot T+T/\lambda=2GD,\qquad \dot a+a/\lambda=w,
\]
\[
\tau:S-q\in\tau_r\,\partial|w|.
\]

Here w is a bulk shear-deformation rate [1/s], not a surface slip rate [m/s]. T is auxiliary history; tau is the physical deviatoric stress.

The mechanical storage, phase driving, and physical dissipation are

\[
\psi=\|\tau\|_{3D}^2/(4G)+Ga^2/(2h),\qquad
Y_{\rm mech}=h' q^2/(2G),
\]
\[
\mathcal D=\|\tau\|_{3D}^2/(2\eta)+h q^2/\eta+\tau_r|w|.
\]

Add the specified AT1 fracture functional once. Apply physical phase irreversibility through the accepted obstacle phi_(n+1) >= phi_n; do not restore a maximum-history H law. Reuse the verified independent-master constraint treatment, including C^T g, C^T H C and consistent obstacles when relevant. Distinguish a partial phase derivative at fixed physical histories from a derivative after local variables have been eliminated.

Use independent bulk T and a, or an equivalent mixed representation that preserves the same equations. Define the accepted/trial ownership and persistence of every new field, including signed and absolute slip histories where retained. Evaluate stress, energy, driving and tangents from the same constitutive state.

**Intact material is essential.** At h=0, use the closed/mixed limit from the derivation: a=0 can coexist with finite q. The virgin limit selects
\[
q=\operatorname{sgn}(t_0)(|t_0|-\tau_r)_+,
\]
where t_0 is the intact elastic trial shear stress. Verify the homogeneous onset
\[
h'(0)(|t_0|-\tau_r)_+^2/(2G)=K_f.
\]
This virgin rule is not a replacement for the history-dependent return map during unloading or reloading.

Do not evaluate 0/0, silently set q=0 at every intact point, or freeze a=0 during every trial phase change. Do not require a surface association, insert geometry to create driving, add an empirical driving source, or seed positive damage ahead merely to bypass the intact limit.

Keep this first experiment within the derived range 0 <= phi < 1. Do not turn an initialization value such as phi_max = 0.6 into an undocumented constitutive cap. If the run requires the fully failed endpoint, report that limitation instead of introducing a hidden g/h floor.

Before the first coupled run, derive and record the actual time-discrete residuals, constitutive update, phase conditions, tangent and energy ledger. The continuous derivation does not authorize copying the old endpoint potential unchanged. In particular, differentiating a phase-dependent viscous dissipation metric as stored energy changes the model. A staggered implementation is acceptable only with convergence to the chosen coupled conditions; one unconverged phase-first pass is insufficient. Explain how the intact active set is entered.

Choose tolerances and stopping limits before running. Keep physical dissipation, time-discretization loss and nonlinear residual work separate. Resolve ordinary implementation choices autonomously. If a genuine mathematical inconsistency prevents a faithful discretization, present a minimal counterexample and a specific proposed correction rather than silently changing the model.

## 2. Connect to the surface without using it to control growth

Retain bulk fields as the authoritative physical histories. Extract surface diagnostics only where their normalization is meaningful:

\[
I=\int h\,dn,\quad \chi=h/I,\quad
s_e=\int a\,dn,\quad Z=Gs_e,\quad \rho=a-\chi s_e .
\]

Verify, for the flat-profile assumptions,
\[
\int Ga^2/(2h)\,dn
=Z^2/(2GI)+(G/2)\int \rho^2/h\,dn.
\]

The existing surface energy is recovered when rho=0. In general, neither that assumption nor constant local q follows from defining Z. Retain the bulk residual information; do not discard it to force a match with the old surface model. Compare cohesive viscous and frictional dissipation using the corresponding identities in the derivation.

Use the actual discrete weights/operators for these checks. A projected nodal I is not automatically the raw normal integral. Distinguish continuum identities from the existing projected surface discretization.

Do not add the old surface cohesive energy or friction a second time to the bulk model. Updating the observer geometry must not modify physical T, a, phi, the obstacle, or the mechanical solution. The old P3 stress-correction map is not needed merely to redraw a bulk-authoritative solution. New observer records inherit diagnostic moments from bulk state; they do not reset physical histories.

## 3. Run these bounded tests in order

1. **Material-point verification.** Check the elastic/virgin onset below and above threshold, sticking/sliding, unloading and reloading without healing, and finite-viscosity relaxation. Use the analytical elastic limit and independent integration where available. Check power balance and tangents away from active-set switches; use appropriate one-sided checks at switches. These tests must expose the old “intact always has zero driving” failure.

2. **Straight-band bulk/surface comparison.** Recover the surface storage and dissipation under their stated profile assumptions, then deliberately use a nonuniform profile history and check the residual identities. Do not require arbitrary new-model trajectories to reproduce old P2 trajectories: the constitutive laws differ when the reduction assumptions fail.

3. **One native finite-tip experiment.** Start from the existing small specimen, reinitialize/equilibrate it consistently with the new law, and retain an unloaded or subcritical hold. Then apply a documented loading ramp selected from the new activation criterion. The P4a polynomial snapshot was not equilibrated and its plotted trial stress was below the example residual strength; it is not sufficient evidence that the repaired model should grow under that same load. Use a fixed candidate shear orientation without a hidden spatial gate on damage. Track remote damage as well as tip evolution.

   Show mechanically driven phase change ahead of the original surface endpoint. Distinguish extension of the diffuse fault from seed/cap relaxation and from an observer threshold moving inside pre-existing damage. Track several phase contours and the actual phase increment. Stop after a short resolved advance sufficient to assess the mechanism, or at the declared physical/numerical limit.

4. **Minimal qualification of the native result.** Check one smaller time increment, one spatial refinement at fixed physical ell, and adequate tip quadrature. Inspect actual integration weights and coverage; do not use point counts as an accuracy measure. Verify rejection/rollback, exactly-once physical publication, and restart of the new bulk histories on a small case. Include a zero-time observer rebuild that leaves the physical state unchanged. Avoid an open-ended mesh, parameter or MPI campaign.

Preserve the existing P2 qualification limitations. Passing these tests establishes an initial fixed-direction bulk propagation result, not general RSF rupture or a quantitatively validated crack-growth benchmark.

## 4. Make the outcome understandable to a human

Produce a concise report and no more than three principal figures from actual computed data:

- **What the specimen looks like:** dimensions, loading arrows, seed, initial surface endpoint, mesh close-up, and before/after phase contours. Mark what was prescribed and what evolved.
- **Why material weakens:** common-scale tip views/line cuts of physical shear stress, cohesive traction q, mechanical driving, fracture contribution and the applicable phase inequality/active set. Mark the old endpoint and the region without an original surface association.
- **What the response costs:** loading response and tip/phase evolution beside external work, bulk storage, cohesive storage, fracture energy, each physical dissipation and the remaining balance error.

State plainly: what was solved, what changed physically, why growth occurred or did not occur, which checks passed, and what remains unqualified. Include a one-paragraph interpretation without equations and a small table of thresholds, measured outcomes and errors. Give exact source/input identities and replay commands, but keep them out of the main visual explanation.

Proceed through the passing stages without requesting approval after each routine step. Stop and report a concrete failure if the constitutive law, intact activation, admissibility or energy accounting cannot be made consistent. Do not tune the law, relax acceptance criteria, or expand scope merely to produce a propagation picture.
