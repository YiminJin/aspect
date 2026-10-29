# Structural-plane RSF continuation — experimental constitutive hypothesis

Independent fixed-phase reference at `f1f9f29cf40c32d1ddcea000b724375b64853532`
plus preserved local work. [Selection](DIRECTIONAL_CONTINUATION_SELECTION.md),
[measured report](DIRECTIONAL_CONTINUATION_REPORT.md). No production/native
constitutive path, checkpoints, or BP3 parameters are changed.

## What changes, and what does not

The preceding [reconciliation](CONSTITUTIVE_RECONCILIATION_FORMULATION.md)
required **every flow increment** to maximize the instantaneous force over all
planes, including other conjugate planes, at its final stress. That requirement
can leave finite-damage states without a single-direction continuation. Merely
picking one stationary root, resolving a numerical tie, or refining time does
not repair that incompatibility.

The new hypothesis distinguishes **sliding on a material plane that already
exists** from **forming another physical plane**. A structural record carries
an unoriented normal n. Existing-plane sliding has a signed speed on that
plane; it does not obey angular stationarity or yield inequalities on other
planes. A larger competing-plane force is a new-fracture diagnostic, not an
instruction to rotate the old fault. Thus this is a constitutive change to the
global-maximizing rule, not an improved solver for the same equations.

The normal is supplied by the pre-existing geometry throughout its entire
diffuse support, or admitted at an explicitly prescribed formation event.
There is no phase locking threshold. In the present fixed small-strain material
chart its transport is the identity. Passive basis rotation rotates n and all
tensors together. Total-spin sign is not a selection requirement. Finite
rotations, normal evolution from a moving surface, and secondary fracture
through already assigned material are not implemented. This is not a claim
that a material point can never develop another plane: doing so needs another
structural record and an energy/birth rule. A single record cannot silently
stand for both the old surface and a new one.

## Existing-plane equations and energy

Let t=(n_y,-n_x,0), S=sym(t tensor n), S:S=1/2, v be signed slip speed,
V=|v|, W=chi v S. All tensors are embedded in three dimensions, including zz
and both xy terms. The full accepted tensors remain physical history:

    Tdot + (G/eta) T = 2G dev D,
    Adot + (G/eta) A = W,       Kdot = W,
    tau = T - 2G A,            Q = 2G A/h,
    h = 4 phi(1+phi)/(1-phi)^2,
    sigma = tau - (p0+p') Id,
    pn = p0+p' - n.tau.n > 0.

The pressure is on the actual recorded plane, **not** the smaller pressure of
its orthogonal pair. The relative shear r=(tau-Q):S satisfies

    r = sign(v) [pn mu(V,Theta) + eta_rad V],
    mu = a asinh{ V/(2V0) exp[(mu0+b log Theta)/a] },
    Thetadot = (V0/Dc)(1 - V Theta/V0),
    eta_rad = G/(2 sqrt(G/rho)).

Theta is frictional age normalized by Dc/V0, not phase. The zero-speed limit
is continuous and has r=0; this regularized kinetics has creep rather than an
exact positive stick threshold. The constant-Coulomb and pressure-lagged
limits are diagnostic controls, not requirements of this candidate.

For fixed phase, psi=tau:tau/(4G)+G A:A/h. There is **no K storage**. The exact
power balance, with zero dilation, is

    tau:D = psidot + [tau:tau + h Q:Q]/(2 eta)
                       + chi [pn mu V + eta_rad V^2].

Normal pressure does no volumetric plastic work; it changes frictional
resistance. Radiation work is accounted as a separate nonnegative outgoing
loss, not recoverable storage. Transverse components of old A relax by the
same Maxwell equation; K retains all old components. No tensor is projected
onto the new structural plane or reset. At analytic exact maturity, h=infinity
and Q=0, while A and K still evolve. At intact h=0, finite-energy consistent
history has A=0 and flow is disabled; the band applies this exact intact rule.
General intact-to-damaged history/birth coupling is not qualified here.

The fixed phase fracture energy Gc integral Gamma dy is constant in time and
reported separately, including its initial representation error. Were phase
allowed to evolve, the existing stored-energy derivative contains
G A:A h'(phi)/h^2; this task neither drops it nor solves its obstacle equation.

## Finite-step update and complete point derivatives

Use beta=1/(1+G dt/eta), T+=beta(Told+2G de),
trial=T+-2G beta Aold, Qref=2G beta Aold/h. Write

    f=(trial-Qref):S,   pn=p-n.trial.n,
    c=eta_rad+G beta dt chi (1+1/h),
    Theta+=(Thetaold+V0 dt/Dc)/(1+V dt/Dc),
    c V + pn mu(V,Theta+) = |f|,
    Delta K=dt chi sign(f) V S,  A+=beta(Aold+Delta K).

The normal pressure contains the full new trial stress; the correction along
S has zero n-normal component. Thus no pressure lag is hidden in this scalar
elimination. With z=V exp[(mu0+b log Theta+)/a]/(2V0),

    mu'_total = z/sqrt(1+z²) [a/V - b dt/Dc/(1+V dt/Dc)],
    L=c+pn mu'_total,
    dv=[S:dtrial - sign(v) mu (dp - nn:dtrial)]/L,
    dtau=dtrial-2G beta dt chi S dv.

Use the analytic V=0 limit. This includes implicit state dependence and both
mechanical normal-stress and pressure derivatives. The tangent is generally
nonsymmetric; it is not assigned a symmetric potential Hessian certificate.
The sufficient global monotonicity bound `c-pn b dt/Dc>0` is checked, not
assumed. Failure of that sufficient bound would demand branch analysis; it
would not itself prove no physical solution. No such failure occurred here.
Closed-contact loss or missing/inconsistent structural history is a genuinely
unsupported state of this closure and is rejected without publication.

Backward Euler adds the explicit nonnegative remainder

    R_BE=Delta tau:Delta tau/(4G)+G Delta A:Delta A/h.

Endpoint work equals storage change plus physical losses plus R_BE. Ledger
closure is checked but is not a physical energy-accuracy or stability proof.

## Conditional formation, symmetry and histories

For a previously unassigned material record, a prescribed formation event
selects the direction of maximum **instantaneous kinetic speed**, at unchanged
incoming tau, A, K, Theta, phi and physical clock. There is no arbitrary
reference velocity in the selection. For each oriented physical plane use its
own pn; find V* and candidate normals from

    max_n [ |(tau-Q):S(n)| - mu(V*,Theta) pn(n) ] - eta_rad V* = 0.

An independent continuous-angle optimization checks the inherited analytic
angular enumeration. Taking a minimum over the orthogonal pair is equivalent
only while enumerating *all* new planes; it is not used for recorded-plane
sliding. Genuine tied maxima are resolved by closest declared geometry normal,
using (n.n_preference)^2. With no preference, retain the two alternatives as
unresolved symmetry. Under symmetric loading their mirrored normal stresses
are legitimate distinct responses, not failed roots or discretization errors.

This fixture admits a genuinely new record only with A=K=0. T and Theta can
be nonzero and are unchanged. An unassigned point with nonzero fault tensor
history requires an explicit structural transfer rule and is rejected; the
code does not erase that history. Existing records can carry nonproportional
A and K. Assigning a normal alone has no energy/work jump because no stress,
phase, or energetic history changes. If a formation event also creates phase
or changes normalization, its fracture and elastic energy changes must be
accounted by a future phase/birth formulation; no such event is simulated.
Virgin frictional-state initialization must also be specified in that future
coupling. Here fresh records have Theta=1 and thereafter retain their age.

## Cross-fault equilibrium and normalization

The independent band reuses the mass-lumped P1 layout and exact 1-D mixed
mechanical reduction of the earlier strip reference, without a K gradient.
The fixed continuum seed is

    phi(y)=(sqrt(.8)-|y|/(2 ell))_+²,  -.125 <= y <= .125 m,
    Gamma=3/(8 ell)[phi+ell² |phi_y|²],
    I=integral Gamma dy,  chi=Gamma/I.

No arbitrary width is inserted: chi has units 1/m, W units 1/s,
interface slip delta=integral 2Kxy dy has units m, and
V_interface=integral chi v dy has units m/s. Only for a coherent profile is
every local v identical to the interface speed. Here the local v(y), state,
and histories are solved independently; the measured integral is used.
The local rate law away from that coherent limit is an explicit distributed
constitutive assumption, not an exact reduction to a single interface RSF age.

P1 nodal phase is freshly sampled on each mesh. The lumped Gamma mass uses
trapezoidal phi and exact element integrals of the P1 squared gradient, split
equally between element endpoints. I and chi use that same measure. At zero
nodal phase flow is zero even if a neighboring element contributes gradient
energy; no normalization floor or coverage mask is introduced.

Mechanical equilibrium requires tau_xy=s (constant), and
p(y)=p0+tau_yy(y)-mean(tau_yy), with mean(p)=p0. Thus pn on the horizontal
plane is p0-mean(tau_yy). Fresh pure-shear histories have tau_yy=0, so p'=0 is
an exact solution symmetry, not a pressure-lag approximation. The point tests
separately exercise nonzero normal history and the pressure tangent. This
band does not test arbitrary oblique mixed mechanical perturbations.

For a trial s, each active node solves

    [eta_rad+G beta dt chi/h] V + pn mu(V,Theta+) = |s-Qref_xy|,
    Axy+=beta(Axy_old + dt chi v/2), Txy+=s+2G Axy+,
    Delta gamma=(Txy+/beta-Txy_old)/G.

The single mechanical compatibility row is integral Delta gamma dy=Delta U.
Its complete derivative is integral[1/(beta G)+dt chi/L_band]dy >0.
This is an equilibrated joint solve, not independently prescribed point
strains. Its external endpoint work is s Delta U. Normal displacement and
normal work vanish in this fixed-volume reduction. Rate participation width
is (integral |chi v|dy)^2/integral(chi v)^2dy, and is distinct from cell size.

## Scope of conclusions

This closure removes the tested *directional continuation obstruction* by
changing the physical admissibility rule. It does not vindicate old nonglobal
Feng endpoints, remove the demonstrated local constant-Coulomb defect, or
establish a new free-direction continuum localization theorem. Rate/state
softening, oblique coupled disturbances, phase birth and branching still need
qualification. No old symmetric stability certificate is reused. All trials
in this task are isolated independent references; their rollback/restart tests
are not native simulator or MPI qualification.
