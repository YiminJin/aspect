# Substantial structural-plane sliding and phase-admission audit

Independent reference, baseline `f1f9f29cf40c32d1ddcea000b724375b64853532`
plus preserved work. The accepted [structural-plane law](DIRECTIONAL_CONTINUATION_FORMULATION.md)
is unchanged. [Selection](STRUCTURAL_SLIDING_SELECTION.md),
[results](STRUCTURAL_SLIDING_REPORT.md). No production/native path is modified.

## Sliding references and work

Reuse the exact mixed cross-band equilibrium and mass-lumped P1 measure.
The fresh profiles are phi=(sqrt(phi_peak)-|y|/(2ell))_+² with peak .8 or 1,
on -.125<=y<=.125 m. All comparisons keep G, eta, Gc, ell, m, pressure and
RSF/radiation parameters fixed. Exact phi=1 is an analytic reference endpoint,
not a change to the native API's phi<1 guard. A mature **center** does not make
the rest of the diffuse profile fully mature: its flanks still have finite h.

For coherent v(y)=V and relaxing interface memory e, A=chi e S and
edot+(G/eta)e=V. The integrated energy and conjugate traction are

    psi_coh/area = C e²/2,    C=G integral chi²/h dy,
    q_interface = C e.

The coherent-profile reference therefore solves, on the same Maxwell block,

    sdot+r s=(G/H)(Udot-V),    edot+r e=V,   r=G/eta,
    s-C e=sign(V)[pn mu(|V|,Theta)+eta_rad |V|],
    Thetadot=(V0/Dc)(1-|V| Theta/V0).

Setting C=0 gives the intended purely frictional mature interface control.
Neither reference inserts an arbitrary slip width. The distributed strip does
not constrain v or Theta to be coherent. Its physical interface rate is
integral chi v dy; local v is a rate coordinate, not the entire interface
velocity. Local state weakening and radiation losses can therefore differ
substantially even when integrated slip nearly agrees. In particular,

    s V_interface = integral Qxy chi v dy
                  + integral pn mu chi |v| dy
                  + integral eta_rad chi v² dy.

There is no accumulated-K energy; cohesive A memory and viscous Maxwell
relaxation remain. Reversal is signed, and physical stress is T-2G A.

Compute endpoint work s_new Delta U and trapezoidal reaction work
(s_new+s_old)Delta U/2 independently. Preserve the exact BE ledger and its
nonnegative storage increment remainder. The physical trapezoidal remainder
is R_BE-(s_new-s_old)Delta U/2. Record both its signed sum and the sum of its
absolute step values, including loading changes and hold/reversal. Normalize
by sliding friction+radiation and by total external work. The declared 1%
energy criterion applies to the whole experiment, not ratios to vanishing
creep dissipation at the first elastic loading steps; the latter remain visible.

## Full mixed mechanical symbol

Let the recorded plane be horizontal: n=e_y, t=e_x, S=sym(t tensor n).
Take a wavevector k_hat=(cos alpha,sin alpha), transverse q_hat, and
E=sym(q_hat tensor k_hat). For a nonzero Fourier wavenumber, incompressibility
and equilibrium require E:delta tau=0 and delta p=k_hat.delta tau.k_hat.
The pressure on the actual plane is delta pn=delta p-n.delta tau.n.

The full reference retains the three plane-strain stress components, three
A components, age, two velocity components, pressure and signed flow rate.
The equations before elimination are

    delta taudot+r delta tau = 2G[dev sym(i k tensor delta u)-chi delta v S],
    delta Adot+r delta A = chi delta v S,
    delta Thetadot = -nu delta Theta-sign(v) Theta delta v/Dc,
    k.(delta tau-delta p Id)=0,    k.delta u=0,
    (S+sign(v)mu nn):delta tau - sign(v)mu delta p
      -(2G/h)S:delta A - Z delta v-sign(v)pn mu_Theta delta Theta=0,
    Z=eta_rad+pn mu_V >0,    nu=|v|/Dc.

K is retained in nonlinear histories; its passive integral adds zero-history
modes but no stiffness or energy. Full plane-strain zz is retained in the
mechanical pencil. No symmetric-Hessian certificate is substituted.

Independent elimination projects S to P_S=S-2(S:E)E. The two active roots
satisfy the quadratic

    Z(s+r)(s+nu)+K_bulk(s+nu)-F_state(s+r)=0,
    K_bulk=G chi [D(alpha)+1/h],
    D=sin²(2alpha)-sign(v)mu sin(2alpha)[1+cos(2alpha)],
    F_state=pn mu_Theta Theta/Dc.

Other relaxing modes are -r. All coefficients have consistent velocity-based
units: Z in Pa s/m, K_bulk and F_state in Pa/m. D can be negative for oblique
disturbances even when the state perturbation is suppressed. This is actual
normal-pressure feedback, not a frozen-pressure stiffness test. The growth
rate is independent of |k|, finite at fixed coefficients, and is not an
unbounded high-wavenumber Hadamard instability. Radiation and the direct effect
bound the temporal response but supply no spatial cutoff to these internal
bulk modes. Phase gradients cannot suppress a fixed-phase mode.

Verify the quadratic against the six finite roots of the full 11-variable
mixed generalized pencil and finite differences of the pressure/state row.
Use wavenumbers kell=.01,1,100,10000 and oblique angles, finite/developed/mature
phase, speeds 5e-6 through .02 m/s, fresh/steady age and both flow signs in the
pressure derivative check. State-removed and pressure-suppressed cases are
causal diagnostics, not mandatory successful limits or proposed replacements.

A nonlinear periodic oblique laminate is a second, independent check. Its
material plane stays horizontal; its fluctuation coordinate is inclined by
.04 rad. Start at uniform mature sliding V=.0005 m/s, Theta=V0/V, impose a 1%
cosine perturbation in age, and integrate 1 s. For trial stress b, c=2G beta
dt chi, and R=2(S:E)E, equilibrium gives

    tau_new=b-c[P_S v+R mean(v)],
    p_new=p0+k_hat.(tau_new-mean(tau_new)).k_hat.

All rates and states are solved together with this pressure using the exact
nonsymmetric diagonal-plus-rank-one Jacobian. Mean strain follows the
unperturbed sliding solution; local compatible shear fluctuations are solved,
not imposed. The normal traction on perturbation layers and their shear
traction must be constant; pn on the physical material plane varies.
Check stress reconstruction, equilibrium, contact and the full work ledger.
32/64/128-point comparisons verify this exact laminate reduction; a separate
central-difference wave-symbol control measures nonzero spatial operator error.
Neither is presented as a general native two-dimensional qualification.

## Comparison with a true closed interface

A closed straight interface between identical incompressible elastic
halfspaces has delta s=-G|k| delta, where delta is the tangential displacement
jump. One derivation uses the halfspace streamfunction B y exp(-|k|y): the
surface shear traction is -2G|k|B, while each side displaces by delta/2.
The normal traction perturbation vanishes for this symmetric interface mode.
For Maxwell halfspaces replace G by G s/(s+r); expressed in slip velocity,
use the same active quadratic with **K_interface=G|k|**. This is our Maxwell
reference derivation, not a claim of identical geometry to a bulk plane wave.

The known importance of the positive direct effect and short-wavelength
stability in continuum-interface RSF is discussed by
[Rice, Lapusta & Ranjith (2001)](https://authors.library.caltech.edu/records/5efnc-nm026).
Do not import that interface result into a distributed tensor-flow model:
its additional internal modes have K_bulk independent of wavenumber. Their
bounded growth is a physical prediction of the present constitutive extension,
not failed numerical convergence. A decision is still needed whether those
bulk modes are intended fault physics or whether a geometric slip/contact
operator should constrain the mature limit.

## Conditional phase coupling, and the actual intact obstruction

For phi>0, at independently owned T,A,K, the thermodynamic phase force is

    Y=G h'(phi) A:A/h² = h'(phi) Q:Q/(4G),
    g_phi=Gc delta Gamma/delta phi-Y,
    phidot>=0, g_phi>=0, phidot*g_phi=0

in the unconstrained local notation; spatial FE inequalities belong to the
admissible master space. Normalization chi=Gamma/I gives

    delta chi=delta Gamma/I-Gamma delta I/I².

Its derivative enters the flow/state residual and complete coupled tangent.
There is no explicit chi-dot term in the storage at independent fixed A;
if e=A/chi is used instead, its quotient derivative and history must be retained.
For the independent-A convention, adding fracture energy gives

    external power = d/dt integral[psi+Gc Gamma] + D_Maxwell+D_RSF+D_radiation
                     + integral(Y-Gc delta Gamma/delta phi) phidot,

including the phase boundary microtraction work where nonzero. Complementarity
removes the last term on an equilibrated smooth active branch. This identity
does not supply a symmetric potential, a stability certificate or a phase-birth
rule. Direction assignment alone changes no energy/history; creating phase or
new normalized support does. Never reset T,A,K or reinitialize existing Theta.

At exactly phi=0, A=0 does **not** determine Q=2G A/h. The earlier scalar
constant-resistance model supplied an intact multiplier from its rate-independent
return. That rule cannot simply be transplanted to the new kinetics. We test
the naive one-sided extension from the same A=K=0, Theta=1 and physical trial
shear f=4000 Pa. For a small uniform positive trial phase epsilon on a ray
with existing support I0>0 (including delta I=H delta Gamma),

    chi ~ 3 epsilon/(8 ell I0),  h ~4 epsilon,
    L=lim chi/h=3/(32 ell I0),
    [eta_rad+G beta dt L] V+pn mu(V,Theta_new)=f,
    Qxy=G beta dt L V,   Y=2 Qxy²/G.

Thus Qxy=O(dt) and Y=O(dt²) as dt tends to zero. Rejected intact trials retain
A=0, so a latent Q cannot accumulate in the accepted tensors. Coarse trial
steps can initiate homogeneous phase while sufficiently fine steps cannot: this is an
initial-state/admission gap, not an acceptable timestep choice for propagation.

An entirely new, uniformly incipient ray has I=H Gamma, chi=1/H and
chi/h tending to infinity. Taking epsilon to zero at fixed dt gives V→0,
Qxy→f and Y→2f²/G=32 Pa. At the exactly zero ray Gamma/I itself is undefined.
These are explicitly different support topologies, not contradictory evaluations
of one positive field. They demonstrate why birth cannot be implemented by
an epsilon floor or a universal call to the fixed-phase return. The audit
resolves epsilon down to 1e-24 rather than hiding the asymptotic limit.

The working hypothesis specifies first-record direction selection and existing
sliding, but does not provide pre-fracture compliance/traction history or an
intact formation energy law. Two substantive physical choices remain:

1. Define an intact fracture-admission law (and its energy/strength calibration)
   distinct from sliding, or add an explicitly derived latent traction/compliance
   state before birth. Such a state is not reconstructible from A=0 and must not
   be invented by resetting T/A/K at activation.
2. Decide whether mature RSF uses the current distributed local rates/ages or
   an actual geometric interface rate/state/contact operator. The latter
   restores the correct interface stiffness but is a constitutive/discrete
   coupling change, not pressure lag or damping tuning.

Recommendation: retain this tested sliding law on existing records; derive a
separate fracture-energy-controlled intact admission and a work-conjugate
registered-interface mature limit before phase activation. Existing material
age must transfer unchanged; genuinely new frictional age needs an explicit
physical initialization rule. Do not introduce a latent variable, extra birth
energy, phase threshold, normalization floor or artificial viscosity merely to
produce a tip-growth picture. No propagation test is run with the failed naive
intact extension.

This does not prove that all seeded fronts are unable to move: the fracture
gradient can seed a neighboring intact point. A front-only interpretation with
absorbing homogeneous intact material is therefore possible, but it changes the
previous model's intact nucleation semantics. It must be chosen explicitly.
Likewise, retaining a cached Q as a new latent state requires its own evolution
and birth work; old derived Q outputs do not supply that constitutive rule.
Separating damage-created recoverable compliance from rate-dependent frictional
slip is another physically motivated route. It changes the present common
source Adot+rA=Kdot, and needs a derived energy/dissipation split. None of these
different physics is silently adopted to pass a propagation test.
