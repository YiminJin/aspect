# Single-direction reconciliation: equations and limits

This independent reference implements the direction and dissipation definitions
in Feng, Freddi, Li and Ma, *Phase-field model for 2D cohesive-frictional shear
fracture: An energetic formulation*, JMPS 189 (2024), 105687,
[DOI](https://doi.org/10.1016/j.jmps.2024.105687),
[accepted manuscript](https://air.unipr.it/bitstream/11381/2991093/5/PFM_CM-3.pdf).
The manuscript was read, including equations 17–24, 43–66 and the finite-step
corrector. Its phase potential and elastic constitutive law are **not** silently
substituted for the retained Maxwell/phase model. This is a bounded reference,
not a production material option or a replication of the paper's simulations.

## Historical spin selector

Recovered `pf-cpdi` revision `e3ef3ae89ade2a16db04fcabecccc5db3a7f0ced`,
`source/material_model/phase_field_rsf.cc`, approximately lines 638–742;
introduced by `61d6c24ed07ffd274e9a41368d547529477b61f3`.
The saved excerpt and hashes are in the evidence directory. Let
`Lij=partial_j v_i`, `omega=(L21-L12)/2` (positive counterclockwise), and `a1`
be the eigenvector of the smallest tensile-positive stress eigenvalue.
The caller supplies `alpha_c=pi/4-atan(mu)/2`, despite the helper's argument
being named `friction_angle`. The intended normal is

    n = a1 sin(alpha_c) + sign(omega) (ez cross a1) cos(alpha_c),

with sign +1 when `abs(omega)<=machine epsilon`; axes are unoriented modulo pi.
The branch's slip direction follows tangential traction. The old surrounding
code blended normal projectors below its locking threshold and subsequently
locked them. Neither blending nor locking is imported here. Bare stress axes
also omit the new model's noncoaxial cohesive backstress.

There is a separate literal-source issue: the 2-D `rotation_axis` initializes
all three components to signaling NaN and assigns only component 2. The caller
tests `is_finite(w[0])`, so it takes the +1 fallback even at nonzero spin under
ordinary IEEE execution (invalid-operation trapping may fail earlier).
deal.II's tensor NaN initializer confirms that all components are filled.
The independent test distinguishes this literal behavior from the intended
rule; it does not claim a compiled legacy simulation or repair that backend.

For `Lxy=.02/s`, total spin is `-.01/s`. Adding rigid observer spin `.02/s`
at the instantaneous identity rotation leaves stress and symmetric strain rate
unchanged but flips the intended normal by 84.28940686 degrees at mu=.1.
At zero spin the fallback is arbitrary. Velocity reversal with retained stress
is distinct from reversing stress too; both are recorded. Total-spin agreement
cannot be a frame-independent constitutive selection requirement.

## Actual crack-strain direction and power

Write physical stress `sigma=tau-p Id`, with `p=p0+p'`, and cohesive backstress
`Q`. The full embedded 3-D contraction is used; in-plane crack flow is trace
free with zero zz, while mechanical tangents retain plane-strain zz stress.
For orthogonal unit axes alpha and beta=J alpha,

    m = k sym(alpha tensor beta), k = +/-1,
    N = sqrt(2) m, N:N=1, tr N=0,
    pn(N) = min(positive(-alpha.sigma.alpha),
                positive(-beta.sigma.beta)),
    F(N) = (sigma-Q):N/sqrt(2) - mu pn(N),
    W = dot(eps_c) = dot(lambda) N, dot(lambda)>=0,
    max_N F <= 0, dot(lambda) max_N F = 0,
    active N belongs to argmax_N F,
    Ddot_f = sqrt(2) mu pn(N) norm(W).

Thus if `W=dot(gamma) m`, friction power is `mu pn |dot(gamma)|`.
The direction is that of the **actual increment**, not accumulated crack
strain. Alpha is canonically the lower-pressure member of the perpendicular
pair. The sign k follows relative shear `(sigma-Q):sym(alpha tensor beta)`.
This is zero-dilation, non-associated flow; differentiating friction with
respect to rate as though it were a convex dissipation potential changes it.
Pressure is the solved mechanical pressure plus prescribed background, not an
independently fitted confinement.

The current two-system model has the same global closed-contact yield envelope
but permits `W=sum_i dot(gamma_i) m_i` with summed system dissipation. Feng's
actual W must itself have a maximizing direction. These are different flow
and dissipation rules even when the critical planes coincide. Averaging the
conjugate flows does not implement the published rule.

The reference partitions angle space at relative-shear sign changes, pressure
ordering switches and zero-pressure boundaries; F is sinusoidal on each piece.
It checks extrema and boundaries globally. Pressure-kink enumeration is present,
but the return/tangent qualification below is restricted to strictly closed
contact and smooth active branches, not opening or switching derivatives.

At genuinely tied maxima, supplied orientation/history chooses the largest
`(n.n_history)^2`. The force tie allowance is `64 eps` times
`max(1 Pa,norm(tau),norm(Q),abs(mu p))`. A larger objective always wins.
No preference, continuum symmetry or equal-distance ties remain unresolved.
History transport is identity in the retained fixed small-strain material
chart. Under passive coordinate changes the director and every tensor rotate
together. No finite-rotation/objective-rate law is introduced. Tests rotate
the stress maxima through 40 degrees and the selected normal follows; history
does not permanently lock the plane. This test is not a phase-front trajectory.

## Explicit mapping to the retained histories

The comparison retains G=1e6 Pa, eta=1e8 Pa s, mu=.1, p0=10000 Pa, residual
cohesion zero, m=4, Kf=8 Pa, Gc=1 J/m², ell=.046875 m, and

    h(phi)=4 phi(1+phi)/(1-phi)^2,
    tau=T-2G A, Q=2G A/h,
    dot(T)+(G/eta)T=2G dev(D),
    dot(A)+(G/eta)A=W, dot(K)=W.

Feng's elastic crack strain corresponds to A=K=eps_c only when relaxation is
removed; its backstress `2G eps_c/(c2 omega(phi))` corresponds to h=c2 omega.
Its distinct phase energy/calibration is not adopted. Phase is fixed here;
the exact mature h=infinity comparison is an analytic limit, not a phi=1 call
through the native API. The intact tangent is elastic; no new intact phase
activation qualification is claimed.

With beta=`1/(1+G dt/eta)`, incoming A_old, and current trial T,

    tau_trial=T-2G beta A_old, Qref=2G beta A_old/h,
    tau=tau_trial-2G xi m(theta),
    Q=Qref+2G xi m(theta)/h, Delta K=xi m/beta.

For a fixed angle the corrector is
`xi=[(tau_trial-Qref):m-mu(p-tau_trial:P)]/[G(1+1/h)]`, P=n tensor n.
The angular equation is `F_theta=(tau-Q):m_theta+mu tau:P_theta=0`.
All positive regular roots are tested against the **final global maximum**;
zero or multiple admissible endpoints are reported, never forced accepted.
Independent algebraic branch solutions check angular enumeration for the
symmetric and reversed-memory controls. Direction continuity can resolve a
tie at one state; it is not a stability rule for two different return states.

For the complete finite-step tangent, set f=m+mu P and
`f_theta=m_theta+mu P_theta`. Implicitly differentiate the two equations:

    R1=(tau-Q):m-mu p+mu tau:P=0,
    R2=(tau-Q):m_theta+mu tau:P_theta=0.

The xi column is `-2G [f:m+(m:m)/h,
f_theta:m+(m_theta:m)/h]`. The theta column is
`[-2G xi(f:m_theta+(m:m_theta)/h),
F_theta_theta-2G xi(f_theta:m_theta+(m_theta:m_theta)/h)]`.
The input columns are `[f:d(tau_trial)-mu dp, f_theta:d(tau_trial)]`.
Recover `d(tau)=d(tau_trial)-2G(m dxi+xi m_theta dtheta)`.
This includes pressure dependence and changing direction; it is nonsymmetric.

Fixed-phase storage is `psi=tau:tau/(4G)+G A:A/h`. Friction power above and
Maxwell loss `(tau:tau+h Q:Q)/(2 eta)` enter the BE identity; the separate
positive BE remainder is `Delta tau:Delta tau/(4G)+G Delta A:Delta A/h`.
The ledger check does not establish physical temporal energy accuracy.

## Full incompressible mechanical test

Let `C=d(tau)/d(eps)` and `Pp=d(tau)/dp`. For wave normal n, form

    Qac v = (C:sym(v tensor n)) n, b=(Pp-Id)n,
    M(n) = [ Qac  b ; n^T  0 ].

For transverse a=J n the mechanically eliminated coefficient is
`Lambda=a.Qac a-(a.b)(n.Qac a)/(n.b)`; the eliminated pressure and full M null
are independently checked. Holding pressure fixed omits the second term.
With alpha measured from the active plane tangent, k the slip sign, the
continuous-rate coefficient is

    Lambda/G = 1 - cos(2alpha)[cos(2alpha)-k mu sin(2alpha)]
                    / [1+1/h+k mu sin(2alpha)].

Its numerator changes sign when `1/h` falls below
`max_alpha {-sin²(2alpha)-k mu sin(2alpha)[1+cos(2alpha)]}`.
At mu=.1 this gives hcrit=100.49813740695, phi_crit=.76761127929 for m=4.
Both conjugate branches have the same threshold. A negative coefficient and
explicit angular mixed nulls persist in strict single-direction states;
removing the two-system redistribution freedom therefore does not cure the
mechanical problem. The Maxwell time term supplies no spatial coercivity.
The phase gradient contributes nothing to admissible fixed-phase perturbations;
on a free phase branch its high-wavenumber mechanical feedback decays as k^-2
and cannot repair this fixed-phase principal-symbol failure. The positive
obstacle residual in the archived phi=.8 control makes delta phi=0 admissible.
No mesh experiment, physical growth rate or general well-posedness certificate
is inferred from these point/symbol checks.
