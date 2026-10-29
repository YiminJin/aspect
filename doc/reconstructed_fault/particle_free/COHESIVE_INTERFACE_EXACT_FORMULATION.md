# Exact-jump fixed-Γ realization

This is an opt-in kinematic/discrete correction to the isolated native-component
fixture. The selected [bond/contact law](COHESIVE_INTERFACE_FORMULATION.md),
parameters, surface Gauss3 histories and full mixed pressure coupling are
unchanged. It is not a production Stokes/phase selector or a new fracture law.

## Mixed kinematics and work

Write J(y)=H(y−yΓ)−Fχ(y), Fχ'=χ, and

    u = u_bar + J δ_h e_x,       δ_h = Σ N_j δ_j.

The regular strain and divergence of the unit surface basis lift are

    L_j = dev3[−χ N_j S + J N'_j e_x⊗e_x],
    d_j = J N'_j,               S_xy=S_yx=1/2.

The distributional jump is handled at Γ, not included a second time in the bulk
strain energy: [ux]=δ_h, [uy]=0. The bulk field is piecewise smooth. Unlike the
prior source-only representation, the along-fault derivative and its pressure
work are retained. Both vanish only for uniform slip. The enrichment vanishes
at the outer walls. Changing the hat width changes Fχ by a continuous exactly
representable Q2 field on the selected aligned mesh, and can therefore be
absorbed into u_bar. It must not change the exact discrete mechanics.

Keep the native Q2 displacement space and complete native AffineConstraints.
Duplicate only Q1 pressure supports on Γ for the upper layer, including their
periodic-master expansions. This gives two pressure traces; displacements remain
continuous before enrichment. The native pressure gauge is removed and the
computed perturbation is shifted to zero bulk mean. Both velocity components,
all unconstrained pressures and the interface normal displacement respond.
No pressure penalty or fixed-pressure elimination is introduced.

In dimensionless pressure p/G, assemble

    A_ij = ∫[2 dev ε_i:dev ε_j − div(v_i)p_j − p_i div(v_j)],
    C_ij = ∫[2 dev ε_i:L_j − p_i d_j],    A W = −C.

Store the same relaxed memories z_new=β(z_old+Δδ), U_R,new=β(U_R,old+ΔU),
β=1/(1+GΔt/η). Then

    E = U_R S/H + Σ z_j (dev ε(W_j)+L_j),
    τ=2G E,   σ=τ−(p0+p′)Id.

This includes zz and reconstructs physical history at arbitrary FE points within
the fixed homogeneous load family. It is not arbitrary old DG-history conversion.
The relaxed energy matrix is

    K^w_ij=∫2G E_i:E_j.

An independent generalized-force check evaluates

    f_j = −∫[τ:L_j − p′ d_j].

The p′d_j term is essential. Equilibrium and the pressure constraints make this
consistent with the derivative of bulk stored energy. Direct volume energy
assembly avoids subtracting large Schur terms. Background p0 does no work for
this zero-opening, volume-preserving periodic test. All boundary reaction work
and Maxwell/numerical remainders retain the prior ledger conventions.

## Sharp traces and friction

At each aligned surface face evaluate separate lower and upper physical traces
from native shape gradients and pressure. Form

    n_l/u = p0 + p′_l/u − τyy_l/u,
    l_n = ∫Γ N (n_l+n_u)/2,
    pn_q = N_q MΓ^−1 l_n.

This is the consistent Q1 projection of the average physical normal traction,
not an average over the diffuse band. The exact continuum normal traction is
continuous; its discrete two-sided mismatch is exported, not set to zero.
The physical shear traces are also exported and compared with the independent
weak energy force. Their FE error is separate from algebraic work closure.
The current projected pn enters the unchanged friction law and its full
nonsymmetric tangent. Closed contact is checked on both side projections.
Surface Gauss3 and Gauss6 trace integrals are independently compared.

The mixed pressure jump is allowed because incompressibility with a varying
prescribed tangential jump generally requires different τyy and p′ on the two
sides, even though their normal-traction combination is continuous. Sharing a
single continuous pressure trace would artificially restrict this response.
There is no additional constitutive pressure-jump condition.

## Signed history and lifecycle

Signed rate is the oriented geometric displacement increment Δδ/Δt in m/s.
The surface tangent is fixed +x, normal +y; reversal changes the rate sign,
not the normal, age or tensor memory. Absolute accumulated slip is separate.
The legacy nonnegative-rate manager API is unchanged. The experimental signed
property, Gauss physical records, bulk-memory coefficients and publication clock
are committed together via the existing native-manager clone/validate/swap path.
No |V| substitution is used for signed traction or geometric displacement.

The exact-jump mode is part of the experimental checkpoint configuration. A
restart into the source-only mode is rejected before publication. Older fixture
checkpoints without the new flag retain source-only meaning; no production
checkpoint schema is changed. Each rejected trial reads the same accepted state;
retry must reproduce it and duplicate publication must fail. The host's actual
ASPECT postprocessor checkpoint mechanism remains the owner.

## Boundaries of this qualification

Serial, fixed, aligned, through-going, periodic interface and homogeneous bulk.
Pressure duplication is specific to this aligned fixture and is not a general
cut-cell/XFEM implementation. Native cells must resolve Γ and the lift knots.
No hanging-node/MPI/AMR/moving-geometry coverage is inferred. Tested intact
admission is uniform; a nonuniform partially tied front is not yet qualified.
Positive sym(−J) on admitted variables checks local incremental monotonicity,
not a general stability theorem for the nonsymmetric law. Production/BP3,
old damage/history models and their numerical settings remain unchanged.
