# Fixed-Γ native component realization

This implements the accepted [bond/contact law](COHESIVE_INTERFACE_FORMULATION.md)
in a new native C++ component and an isolated ASPECT qualification fixture.
It does **not** select it in the production material model, replace the Stokes
nonlinear driver, or reinterpret a bulk phase/checkpoint. The fixture uses the
actual ASPECT Q2/Q1 FE mesh, mappings and AffineConstraints, native reconstructed
fault container/property/projection/serialization classes, and real ASPECT
postprocessor checkpoint/restart. It assembles its experimental mixed mechanics
rather than calling the unchanged production fault Stokes assemblers.

## Discrete mechanics and history

The domain is periodic in x∈[0,1] m, with clamped fluctuations at y=0,.125 m.
The fault is y=.05 m. The imposed relative horizontal wall displacement is U.
The normal mean stress is p0=10000 Pa; pressure perturbation has zero bulk mean.
The full 3-D plane-strain deviator is used, including zz. Native pressure is
scaled by G in the mixed solve and converted back to Pa before friction.

Let N_j be the native geometric surface Q1 basis and

    χ(y)=max(1−|y−.05|/w,0)/w,       ∫χ dy=1.

This is a prescribed geometric representation, independent of surface damage a.
It does not call the old φ-dependent h/I or normalize zero phase support. Normal
profile support, segment selection and association come from the native manager.
Support knots lie on bulk cell lines in every selected test. w is not the old
physical phase length ell and is not a new material calibration: it controls
this numerical representation of a sharp slip surface.

Store relaxed wall displacement u_R and relaxed surface jump z_j:

    u_R,new=β(u_R,old+ΔU), z_new=β(z_old+Δδ), β=1/(1+G dt/η).

For the fixed load family and geometry, the equilibrated FE displacement memory
is d_R=u_R y/H e_x+Wz. Reconstruct **physical** elastic strain and stress as

    E=dev3(symgrad d_R−χ z_h S),  τ=2G E,
    σ=τ−(p0+p′)Id,             S_xy=S_yx=1/2.

Thus bulk Maxwell strain excludes represented interface slip. Bond storage
ψ_b(δ−b,a) is separate; no second bulk fracture or cohesive memory energy is
added. The bulk tensor is evaluable at every FE/diagnostic point from d_R and
χz, with no DG projection loss. This reduced history representation is exact
for this fixed linear bulk/load family, **not** a general replacement for native
DG T or arbitrary inherited tensor fields. No old history is converted.

With velocity/displacement basis v_i, scaled pressure basis p_i, and homogeneous
native master constraints, assemble

    A_ij=∫[2 dev3 ε_i:dev3 ε_j − div(v_i)p_j − p_i div(v_j)],
    B_ij=∫2 dev3 ε_i:χ N_j S,  D_jk=∫χ² N_j N_k,
    W=A^−1 B,                  K^w=G(D−BᵀW).

The pressure gauge is eliminated, then pressure is shifted to zero bulk mean.
Both velocity components and all unconstrained pressures respond in W; this is
not a fixed-pressure or frozen-mechanics approximation. For robust arithmetic,
K^w is computed equivalently by integrating 2G E_j:E_k for the reconstructed
response strains. An exact represented uniform solution is separated before
forming these products: W1=(∫_0^y χ−y/H)e_x, p′=0. Its measured correction to
the sparse solution is roundoff-sized. Fixed-factor iterative refinement uses
extended-precision residual sums. All original force/energy/derivative checks
are retained and physical stress-work moments independently verify the result.

The native test evaluates each quadrature point's p′ and τ_nn separately.
Normal input is the consistent surface projection of the current **physical**
normal pressure, not previous pressure or p0 alone. Define

    l_s,j=∫χ N_j τ:S,  l_n,j=∫χ N_j(p0+p′−τ_nn),
    M_jk=∫Γ N_j N_k,  s_q=N_q M^−1 l_s,  pn_q=N_q M^−1 l_n.

The evaluated pn_q must be positive. A separate frozen-field probe also checks
pn>0 at every actual bulk integration point. No pressure clamp/filter or opening
rule is added. At finite w these are projected normal-band moments, **not sharp
one-sided interface traces**. That distinction is measured below.

## Consistent surface integration and complete tangent

The final implementation uses **Gauss3 per surface segment** for bond energy,
fracture inventory, RSF/contact work, damage and state history. δ_h is Q1;
each physical surface Gauss record owns a, Θ, jump/reference, relaxed jump and
absolute slip. These are along-fault states, not independent cross-band ages.
No constitutive history is interpolated between quadrature layouts.

For Q mapping surface nodes to quadrature points, W_q the diagonal Gauss weights,
and nodal weights D_w=diag(∫N_j), the residual is

    F=D_w^−1 [l_s−QᵀW_q (q_b+a R)].

Both terms are integrated consistently; D_w merely scales rows into Pa. It does
not replace surface constitutive integration by nodal lumping. In the stored
matrix notation K=D_w^−1K^w and N=D_w^−1 times the native normal-response load,
P=Q M^−1D_w, the complete increment derivative is

    J=−βK − D_w^−1 QᵀW_q diag(q_δ+a_δR+a R_V/dt)Q
           −β D_w^−1 QᵀW_q diag(a sign(V) μ) P N.

R_V includes the implicit aging derivative. This derivative includes damage,
current pressure, both mechanical components and state. Tested intact formation
is uniform: exactly tied nodes are admitted at the calibrated multiplier, with
no initial compliance. Nonuniform partial intact/active fronts and new geometric
records are **not qualified** by this fixture.

The earlier nodal integration gave an alternating formation mode with negative
relaxed curvature despite force/energy closure. That result is preserved as
failed evidence. Consistent Gauss integration removes that tested mode. Before
publication the final fixture requires positive eigenvalues of sym(−J) on the
free surface variables; tied rows are excluded. This is a sufficient *local
incremental monotonicity* check for this tested branch, not the old potential
Hessian or a general non-associated stability certificate.

## Native ownership, rejection and restart

A native ReconstructedFaultManager instance is owned by the selected fixture.
It is deliberately separate from the simulator's legacy phase-initialized
manager: enabling that manager invokes a different material/particle contract.
Those guards are preserved rather than bypassed. The fixture's manager itself
is saved with its real native Boost schema through the normal ASPECT postprocessor
checkpoint mechanism. No mock JSON restart substitutes for this test.

The accepted manager contains nodal jump/relaxed-jump/activity records, signed
geometric velocity and segment-owned Gauss history blocks. The nodal activity
slot is a constraint flag, **not physical damage**; physical a and Θ are exclusively
Gauss records. u_R, time and publication generation are saved atomically with
manager data. At fixed geometry these coefficients recover the complete tested
bulk tensor history. Rejected candidates read accepted records without modifying
them. Publication validates a staged manager copy, then swaps it nonthrowingly
and updates the scalar owner clock/generation. Stale generations and duplicate
candidates are rejected. Restart rebuilds geometry/FE/operator caches and rejects
changed surface count, width or mode. Full ASPECT solution/mesh checkpoints are
used; the host's unchanged global Stokes solution is not relabeled as this new
experimental solution.

The old manager slip-rate solver admits only **nonnegative** V and uses its own
orientation semantics. It is intentionally not fed this law's signed reversal
rate. A separate registered signed-rate property preserves the new geometric
rate through native checkpointing. Reusing the production surface Newton driver
will require an explicit signed-rate API/contract, not taking |V| or changing a
material normal during reversal.

An invalid Gauss candidate, deterministic retry and a duplicate publication are
exercised. The real checkpoint comparison branches from the saved owner time,
not an uncheckpointed terminal output row. Native snapshot timing saves at step
74 in the selected split. Resume outputs are separate files; the uncheckpointed
step-75 branch is preserved rather than appended as if it were new publication.
The selected split end time is the actual uninterrupted floating-point step time,
so the accepted timestep sequence is identical.

## Scope of the representation result

At finite width the native continuous FE displacement has no literal jump at Γ.
δ is a represented geometric-slip source amplitude, with exact uniform jump/work
normalization. Nonuniform finite-width mechanics need not equal sharp-interface
mechanics even with exact FE resolution. The operator comparison uses the
independent mixed-layer sharp solution projected into the same periodic Q1
surface space (including Fourier aliases), so surface projection differences
are not confused with a width effect. Gc, strength, state and material constants
are identical. Bulk Gauss3/Gauss6 agreement isolates integration error.

An exact fixed-surface jump realization is a justified next bounded correction:
introduce a two-trace or equivalent enrichment. For the aligned case an
energy-consistent coordinate form is u=u_bar+[H(y−yΓ)−Fχ(y)]δ_h t, with Fχ'=χ.
Away from the jump it adds (H−Fχ) sym(t⊗gradΓ δ_h) to the current strain, **and**
its divergence and pressure-work terms to the mixed operator. Omitting those
terms would repeat the present nonuniform width error. This proposal is not
implemented here; it requires sharp normal-trace and work-adjoint checks before
using the unchanged bond/contact law for native tip growth. No extra damping,
phase floor, fracture charge or constitutive recalibration is justified by the
measured finite-width discrepancy.
