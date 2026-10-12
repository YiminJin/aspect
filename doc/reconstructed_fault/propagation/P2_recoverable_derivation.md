# Selected recoverable-energy revision: derivation before implementation

The user explicitly selects peak recoverable-energy driving, excludes accumulated
viscous dissipation as an extra phase driver, and selects unloaded initialization
for the new P2 baseline. Historical incremental-work defaults/evidence are retained.

For the single fault with uniform material through the transverse profile, let
h=1/g−1, I=integral h dz, q the cohesive traction and Z=I q. The implemented
surface update is Z_new=beta Z_old+kappa V, with beta=exp(−G dt/eta) and
kappa=eta(1−beta). Its continuous reduced constitutive equation is
Z_dot=G V−(G/eta)Z. The elastic cohesive slip is delta_e=Z/G, with spring
stiffness G/I. Thus Psi_Gamma=G delta_e²/(2I)=Z²/(2GI), per fault area.

Vary phi while holding elastic internal slip delta_e (hence Z) and material G
fixed, rather than holding q fixed:

    delta Psi_Gamma = −q²/(2G) delta I
                   = integral g'(phi) q²/(2G g(phi)²) delta phi dz.

This includes the phase dependence of normalization, including evolving phase;
it is not the fixed-phase work-increment argument. The local reduced-model
coefficient is Y=q²/(2Gg²). Its intact limit g→1 at finite q is q²/(2G), without
the old-history subtraction. Exact g=0 is outside the finite-normalization model.

At fixed material and evolving phase, the reduced cohesive energy identity is

    Psi_dot = q V − I q²/eta + integral g' Y phi_dot dz.

The first term is cohesive power; I q²/eta is nonnegative viscous loss. Friction
mu sigma_n V (and radiation damping eta_d V² if nonzero) is separate dissipated
power, not cohesive storage or phase driving. In bulk, Psi_b=tau:tau/(4G),
D_b=tau:tau/(2eta), and Psi_b_dot+D_b=tau:epsilon_b_dot for the native Maxwell
law. Elastic bulk strain and cohesive elastic slip are the independent stored
variables. Summing their power identities requires the work-consistent exchange
between resolved bulk motion and reduced fault slip. Native transfer/projection
and the imposed diffuse-profile closure need a separate energy-balance audit;
this derivation is not proof that those discrete exchanges cancel exactly.
In particular evolving-profile redistribution can perform additional bulk work
where resolved shear traction varies across the normal profile.

Selected history: H_new=max(H_old,Y_accepted), sampled using the accepted projected
q, accepted phase and the same surface G as cohesive mechanics. Histories are
published only after successful terminal validation. H remains a peak surrogate;
monotonic H alone does not prove phi irreversibility. Stored energy and fracture
expenditure are measured separately from H and from viscous/frictional losses.

Projection caveat: native consistent normalization is I_h=P I(phi). Differentiating
the discrete surface energy integral Z²/(2G I_h) at fixed Z gives
−<q²/(2G),P delta I>=−<P* [q²/(2G)],delta I>. For a consistent L2 projection P*=P
under its integration measure. Pointwise q²/(2G) is the continuum coefficient,
not an exact discrete adjoint coefficient when q is nonconstant Q1. CPDI phase
sampling is another discrete approximation. Retain the existing phase operator
and normalization representation; quantify the discrepancy rather than silently
claiming discrete variational identity or redesigning projection operators.

Initialization of the new fixture retains prescribed H and relaxed phi but sets
q=0, hence Z=delta_e=0, together with zero bulk Maxwell stress. Current/previous I
are initialized from accepted relaxed phase. This makes Y0=0<=H0 pointwise and
avoids reinterpreting historically projected preload as the initial energy.
The artificial 1 s initial mechanical interval must not advance physical H.

Local tests must check native energy history on dt/dt/2/dt/4 paths, independent
phase variations at fixed Z, and the cohesive energy identity with viscous loss
kept separate. Coupled qualification additionally requires bounded phase, no
resolved healing, positive/accurate normalization, correct publication and
feedback refinement. No P3 or material optimization is selected.
