# Consistent endpoint prototype and continuous spatial energy identity

This is a restricted diagnostic prototype, not a native algorithm change. The
native option retains its incoming-history phase solve. The user selects a
short consistent-endpoint trajectory and a continuous-path audit, superseding
the previous replay's provisional stop. No compensating dissipation is added.

Use the existing Q2 derivative matrix D, positive six-point weights W, Q1
interpolation B, CPDI fracture functional F, and consistent surface projection
I=P b(phi). In the uniform tangential reduction I is a scalar and the retained
analytic transverse antiderivative supplies I and its derivatives. In general
P=M_surface^{-1} is fixed, and I_phi=P b_phi. For surface evaluation T and bulk
surface extension L, chi=h(B phi)/(L T I); differentiation includes

    chi_phi = diag(h'(B phi)/(L T I)) B
              - diag(h(B phi)/(L T I)^2) L T P b_phi.

The reverse action uses b_phi^T P^T T^T L^T, not another forward projection.
Each transverse-integral datum differentiates as integral h'(B_n phi) B_n dn
with the same support, boundary transition and surface measure as b. Since
geometry, coefficients, weights and M_surface are fixed, P itself has no phase
derivative. The cohesive variation is
-sum_s ds_s Z_s^2 (T I)_s^{-2}(T I_dot)_s/(2G). Thus both the bulk profile
derivative and cohesive normalization derivative use this same adjoint chain.
The same derivative acts in bulk storage and cohesive storage. In the scalar
uniform reduction C=diag(h'/I)B-h I_phi^T/I^2 and

    E = tau^T W tau/(2G) + Z^2/(2G I) + F(phi),
    tau = G D R - Z chi,
    g = -(Z/G) C^T W tau - Z^2 I_phi/(2G I^2) + F_phi.

R is the authoritative accepted FE elastic-strain coefficient vector, Z the
surface elastic cohesive-slip stress variable; neither is a particle projection.
For multiple surface unknowns the cohesive term and its derivative carry the
same surface weights and interpolation as the existing spatial energy.

## Before and diagnostic candidate

| Equation/state | Native and original trajectory | Coupled prototype |
|---|---|---|
| Phase energy | E(phi,R_a,Z_a) | E(phi,R_b,Z_b), derivative at fixed R_b,Z_b |
| Mechanical condition | Endpoint at newly solved phase | Same endpoint, solved together with phase |
| Irreversibility | phi_a <= phi <= 1 | Same immutable phi_a throughout all iterations |
| Publication | Phase then endpoint mechanics | Validate complete scratch step, publish once |

Keep beta=exp(-dt G/eta), kappa=eta(1-beta), including general beta. At every
trial reset to the same incoming R_a,Z_a and evaluate

    R_b=beta R_a+(kappa/G) u,  Z_b=beta Z_a+kappa V,
    D_I^T W tau_b=0,
    0 <= V perpendicular r_f=mu P+Z_b/I-chi^T W tau_b >= 0,
    phi_a <= phi <= 1,  g in the corresponding bound KKT conditions.

Boundary u is the interval mean of the unchanged .256 t(y-.5) loading. For fixed
mu P these are stationary conditions of
J=E(phi,R_b(u),Z_b(V))+(kappa/G) mu P V. Eliminate mechanics on its contact set;
the phase tangent is the Schur complement J_pp-J_pm J_mm^{-1}J_mp. Test it on
the admissible critical cone: strictly active lower bounds with positive g
cannot move in a zero-first-variation direction; weak contacts can move only
upward. A negative tangent at a nonstationary point does not establish an
unstable equilibrium. Positive free-set curvature alone does not test weak
contacts. The original local bound optimizer and exact Newton polish are used
at the unchanged 1e-11 phase target; no globally minimizing branch is assumed.

## Continuous identity, including profile redistribution

With the existing discrete spatial operators, continuous Maxwell evolution is

    Rdot=u-R/lambda, Zdot=G V-Z/lambda, lambda=eta/G,
    taudot=G D u-G chi V-tau/lambda-Z chidot,
    Idot=I_phi^T phidot, chidot=C phidot.

Consequently

    Eb_dot=tau^T W D u-(chi^T W tau)V-tau^T W tau/eta
           -(Z/G) tau^T W chidot,
    Ec_dot=q V-Z^2/(eta I)-Z^2 Idot/(2G I^2),
    F_dot=F_phi^T phidot.

The last terms in Eb_dot and Ec_dot are retained in g^T phidot; do not count
them again as a loss. Let f=D^T W tau, Wext_dot=u_B^T f_B,
Db_dot=tau^T W tau/eta, Dc_dot=Z^2/(eta I), Df_dot=mu P V. Then

    E_dot = Wext_dot - Db_dot-Dc_dot-Df_dot
            + u_I^T f_I + r_f V + g^T phidot.

Thus the external-work remainder equals minus the sum of the integrated
mechanical-imbalance, friction-residual and phase-residual powers. Each term's
signed integral and integral of absolute power must be reported separately.
At a smooth stable rate-independent solution g^T phidot=0; monotone history
alone does not imply this relation. Active lower-bound forces can be positive
when phidot=0. If a reconstructed path has Maxwell rate residuals e_R,e_Z,
add E_R^T e_R+E_Z e_Z explicitly. They vanish on the path selected below.
An arbitrary lifting l has Wlift_dot=Wext_dot+l_I^T f_I; it is not generally
physical external work away from equilibrium.

The embedding is the corrected plane-strain one, without a pressure change.
For the general FE tensor block use the three-dimensional metric
<tau,tau>_3=tau:tau+(tr_2 tau)^2, Eb=sum W<tau,tau>_3/(4G),
Db_dot=sum W<tau,tau>_3/(2eta), and
taudot=2G D_3(symgrad u)-2G chi V S-tau/lambda-2Z chidot S.
The profile power is -sum W (Z/G)<tau,S>_3 chidot. Momentum uses
tau:symgrad u-p div u; the selected divergence-free compatible shear has zero
pressure work and <tau,S>_3=tau_xy, yielding the scalar identity above.
In a general incompressible approximate path the pressure/constraint work
must also be retained, along with full constrained-DOF reactions. This test
does not redefine prescribed frictional normal pressure through tau or p.

## Diagnostic path and trajectory protocol

For each step reconstruct phi(s)=phi_a+(s/dt)(phi_b-phi_a), recomputing I and
chi. Simultaneously use the exact constant-rate Maxwell paths
R(s)=exp(-s/lambda)R_a+eta(1-exp(-s/lambda))u/G and
Z(s)=exp(-s/lambda)Z_a+eta(1-exp(-s/lambda))V. These match both endpoints and
have zero Maxwell rate residuals. This is a diagnostic interpolation, not an
assumption that intermediate equilibrium holds. Integrate all work, losses and
the three residual powers on this same path; verify the chain-rule identity
independently and refine time quadrature if needed. No instantaneous jump or
added D_relax is used in this ledger. Also retain the signed incoming-state
profile cost as a separate frozen-state comparison, never as dissipation.

First verify the LAST retained pre-damage state's bulk equilibrium, contact and
phase KKT with its proper incoming bound. The last exactly unchanged seed at
.5263671875 s fails phase KKT (7.08e-8 versus 1e-11), despite passing mechanics.
According to the user's fallback rule, the primary trajectories therefore start
fresh at t=0 with the unloaded seed, zero R,Z and slip. Earlier .5234375-s
consistent-checkpoint paths are retained only as controls, not substituted for
that rule. Use identical fresh fields in all three partitions. Check accepted states,
common physical times/loads and reduced stability. Large corrections from an
already inconsistent replay must be separated from any loss of stability on
this fresh consistent branch. Production, physical constants, spatial operators,
history definitions, tolerances and prior evidence remain unchanged.
