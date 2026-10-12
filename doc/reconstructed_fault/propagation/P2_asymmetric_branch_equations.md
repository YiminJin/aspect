# Translation, obstacles and deterministic endpoint minimization

This is a selected isolated P2 diagnostic. Production equations and defaults
remain unchanged. Use the existing consistent-endpoint energy, normalization
adjoint and immutable incoming histories from P2_coupled_branch_equations.md.

In the homogeneous continuum normal reduction, let L be domain height,
I=integral h(phi)dy, chi=h(phi)/I, C=integral chi dy=1. Exact bulk equilibrium
gives constant shear stress T. With incoming R_a(y), Z_a, general beta and
kappa, endpoint mean strain is e_b=beta mean(R_a)+kappa Delta u/(G L).
Thus T=G e_b-Z_b/L, Z_b=beta Z_a+kappa V, and

    0 <= V perpendicular mu P+Z_b/I-T >= 0.
    E_reduced = L T^2/(2G)+Z_b^2/(2G I)+F(phi),
    F=A integral [phi+ell^2 (phi_y)^2]dy.

The algorithmic potential additionally contains kappa mu P V/G. Dependence on
the incoming nonuniform R_a reduces to its mean after mechanical elimination;
the compensating velocity is allowed to vary over the whole cross-section.
This is not the energy variation with mechanical history held fixed.

On the slipping set put Q=G e_b-mu P. Mechanical elimination gives
Z_b=Q L I/(L+I), q=Q L/(L+I), and

    J_reduced = L(G e_b)^2/(2G) - Q^2 L I/[2G(L+I)]
                - mu P beta Z_a/G + F(phi).
    g_continuum = A(1-2 ell^2 phi_yy) - q^2 h'(phi)/(2G).

Normalization variations remain essential even though the constant-stress bulk
profile term cancels. On the sticking set Z_b=beta Z_a is fixed; invariance of
I and F gives the same translation argument.

For a translated profile phi_s(y)=phi(y-s), I_s' is the difference of h(phi)
at the two boundaries, and F_s' is the corresponding difference of
A[phi+ell^2 phi_y^2]. A localized profile flat outside its support has invariant
I,F and reduced potential under small translations that remain inside the
domain. Compatible velocity conditions fix Delta u, not band position. Natural
phase boundaries introduce no essential phase-value constraint; nevertheless
tails meeting a boundary or nonmatching boundary traces destroy exact translation
invariance. For equal boundary traces and zero boundary slopes the first/second
translation derivatives vanish, but that alone is not global invariance.

Translations of a Q1 profile move its derivative kinks off the fixed mesh and
are generally not members of the same Q1 space. Its piecewise derivative is
compared in L2, not assumed to be an exact conforming phase tangent. Continuum
neutrality applies to the smooth stationary reduction; discrete curvatures need
not be zero.

The obstacle phi>=phi_a is fixed in physical coordinates. A translated family
is an admissible neutral family only while phi(y-s)>=phi_a(y) everywhere.
At strictly active obstacle contacts the critical direction must vanish;
at weak contacts it must be nonnegative. Therefore -phi_y is generally not an
admissible two-sided direction. Where phi equals a flat zero obstacle outside
the band, translation may be tangent; where it touches a sloping old profile,
one or both signs can be forbidden. Finite admissible amplitudes are limited by
the actual incoming obstacle. Neutral continuum translation does not imply a
negative admissible discrete mode is physical: Q1/CPDI sampling, finite Q2
mechanics, and discrete front release can lift or destabilize that direction.

Compare modes using the consistent Q1 phase mass matrix M_phi, not the Euclidean
nodal norm. Solve H_physical v=lambda M_phi v on the enlarged critical space;
H_physical=A H_scaled. Normalize v^T M_phi v=1, orient toward increasing phase
centroid, and compare with the piecewise-constant Q1 spatial derivative using
the same exact L2 measure. Weak-contact admissibility is tested separately.

At each failed entry use both signs and several strictly feasible perturbations
of the symmetric state. They change solver guesses only. Incoming R_a,Z_a,
slip, time interval and phi_a remain unchanged. A projected Newton minimizer
uses the exact gradient/reduced Hessian; negative eigenvalues are replaced by
positive magnitudes only in its search metric. Projected line searches evaluate
the actual potential and KKT conditions. A converged saddle is explicitly
escaped along a feasible negative-curvature direction; no Newton polish to an
indefinite root is accepted as a minimum. The original 1e-11 KKT target and
bound/history rules are retained. This changes a diagnostic minimization
procedure, not the energy or a physical damage law.

Endpoint energy reductions do not qualify a finite jump. Candidate and continued
trajectories use the same linear-phase/exact-Maxwell diagnostic path as before;
mechanical, friction and phase residual work, signed profile cost, storage and
losses remain separate. Histories publish once only after the complete diagnostic
step passes equilibrium, bounds and admissible stability. Reflection-related
solutions are reported separately; no physical branch preference is introduced.
