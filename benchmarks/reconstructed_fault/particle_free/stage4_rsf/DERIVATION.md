# Fixed-phase restoring response and RSF stability diagnostic

This is a benchmark diagnostic of the current bulk-tip material-strip law. It
changes no physical solve, acceptance criterion or stored checkpoint. Phase,
formed planes, physical JxW and ds stay fixed. The fixed boundary perturbation
is zero displacement (not a change in imposed loading). All stresses and work
use the full plane-strain tensor convention.

Let h(phi)=m phi(1+phi)/(1-phi)^2, H=h/(1+h), S:S=1/2, A=eS.
For a rapid perturbation relative to the common Maxwell time, minimize

    delta E = integral [delta tau:delta tau/(4G) + G delta e^2/(2h)] dOmega,
    delta tau = 2G(E3(delta u)-delta e S), div(delta u)=0,
    delta u=0 on the complete mechanical boundary.

The imposed generalized slip is delta z_s=sum dn delta e, dn=JxW/ds_s.
Lagrange multiplier minus f_s is its driving traction. At fixed resistance f,

    delta e = H(2 E3(delta u):S - delta f/G),
    delta tau = 2G[E3(delta u)-2H(E3(delta u):S)S] + 2H delta f S.

The native point derivative and local mixed matrix already assemble this law.
The diagnostic sets only its **condensed section friction slopes** to zero and
holds phase coordinates fixed. Its six smooth homogeneous mechanical solves plus one cell-scale control retain
all velocity/pressure coupling and use the existing Stokes inverse. It is not
an eigenanalysis of the algorithmic Newton Jacobian.

In the native stage coordinates beta=(1+G dt_stage/eta)^-1, the mixed matrix
contains beta. Its response to delta f is delta u_stage; the elastic response
is beta delta u_stage. The exact conversion from its free-rate derivative is

    delta z = beta dt_stage delta V,
    delta z = -C delta f,
    C = diag(sum dn H/G) + the equilibrated finite-body response.

The diagnostic exports C applied to six sine force patterns supported on the
established central 3 m patch [2.5,5.5], normalized with the actual ds. Every
section responds; outside-patch resistance perturbations are zero, not blocked
slip. Thus this is a deliberately compliant, low-dimensional patch diagnostic,
not the complete 768-section spectrum. The response outside the loading patch
is retained and measured. Matrix symmetry uses the ds inner product.

For orthonormal force basis P, Cp=P^T W C P, Kp=Cp^-1, and the relaxed slip
basis Psi=C P Kp. Then P^T W Psi=I, and 1/2 q^T Kp q is the minimum elastic
energy at imposed generalized section slip q=P^T W z. Friction and state must
be projected with **Psi**, not P. The diagnostic verifies its actual force-slip
work against independent point-energy integration and symmetric point updates.

A useful full-space bound requires no patch truncation. Weighted Cauchy gives

    E_coh >= sum_s ds_s G z_s^2 / (2 sum_(q in s) dn_q h_q).

The bulk energy is nonnegative, so K is bounded below by this diagonal cohesive
stiffness. This uses the complete profile, not its center peak. Exactly intact
sections have zero compliance and no slip; there is no phase/rate cutoff.

For the current matched Maxwell coefficients r=G/eta, both tau and A obey the
same relaxation. With frozen phase/planes the continuous mechanical response
for a mode exp(lambda t) is K(lambda)=lambda/(lambda+r) K_elastic. In internal
elastic-slip coordinates z, zdot=delta V-r z. This factor is derived from the
continuous physical law, not from beta or a timestep-dependent Newton matrix.

For positive operating rate V, theta_dot=1-V theta/Dc. With

    x=V/(2 Vref) exp[(mu0+b log(theta Vref/Dc))/a], rho=x/sqrt(1+x^2),
    fV=p a rho/V+eta_rad, ftheta=p b rho/theta,

an actual transient operating point has instantaneous age slope 1-V theta/Dc.
Freezing those coefficients gives a local variational diagnostic, not a steady
state or a proof of stability along its entire evolving history. At theta=Dc/V
one obtains the steady-age diagnostic. The scalar elastic-mode roots satisfy

    fV lambda^2 + [fV(r+V/Dc)+K-ftheta theta/Dc] lambda
      + (V/Dc)(fV r+K)-r ftheta theta/Dc = 0.

For several slip modes use the actual native measures and section states:

    M = Psi^T W diag(fV) Psi,
    N = Psi^T W diag(ftheta),
    delta v = -M^-1(Kp z + N delta theta),
    zdot = delta v-r z,
    delta theta_dot = -diag(theta/Dc) Psi delta v - diag(V/Dc) delta theta.

Zero-rate intact sections have no mechanical response and are not invented
sliding degrees of freedom. Their physical aging remains unchanged. No minimum
velocity or clipping is introduced. For signed rates, the derivative of aging
uses sign(V) and the signed resistance state derivative changes consistently.

The regularized law, radiation and state derivatives are required even when
rho is practically one. The nominal kc=p(b-a)/Dc is only the r=0, logarithmic,
steady-age reference. A negative steady friction slope alone is insufficient.
Midpoint's native scalar-root admission is eta_rad >= p b dt_stage/Dc, hence
physical dt <= 2 eta_rad Dc/(p b); the state/unstable-mode accuracy requirement
is separate and can be stricter.

## Conditional reduced pilot and its qualification boundary

The economical pair uses the measured six-mode compliance only. It is **not**
a native timestep trajectory and cannot publish phase, change a material plane,
or qualify a completed event. It retains all responsive section ages and the
nonlinear signed RSF law. Define V=Psi v, generalized driving force F, and a
constant drive l:

    F = Psi^T W f(V,theta),
    Fdot = l - Kp v - r F,
    theta_dot = 1 - |V| theta/Dc.

Set Vbar=Psi vbar, uniform theta0=Dc/Vpeak, F0=Psi^T W f(Vbar,theta0) and
l=Kp vbar+r F0. The initial mechanical derivative vanishes, but the uniform
age is steady only at the peak speed. **Both backgrounds are transient.** Each
law therefore has its own unperturbed trajectory, and the same small force kick
is measured against that trajectory. A differentiated midpoint/aging step
provides an independent variational comparison along the evolving background;
an eigenvalue of the initial frozen coefficients is not called its long-time
physical growth rate.

The tentative low-speed peak 2e-4 m/s balances half the native monotone-root
maximum step and 20 steps per initial fastest state time. Dc is chosen so the
logarithmic critical stiffness there is twice the measured lowest resolved
restoring stiffness. Only Dc changes in the weakening specimen. This is a
synthetic scale selection, not a natural-fault calibration. The inexpensive
reduced integration includes a half-step comparison of state/rate perturbations;
actual maximum dt*|V|/Dc is retained. Exact constant-stage aging does not by
itself establish accuracy when rates vary; measured errors decide it.

The b=a control uses the single scalar
mu0_control=mu0_weak+(b_weak-a)log(theta0 Vref/Dc). It matches the complete
initial instantaneous rate law pointwise: both cases start at identical
traction, rate and age, receive the same force kick and the same constant drive.
No section-dependent material coefficient is introduced.

The constant generalized drive and preload are **not yet realized as a native
uniform-shear boundary/history state**. The reduced stored energy
E=F^T Kp^-1 F/2 and work balance belong to this mechanical reservoir; they do
not replace the complete native bulk/fracture ledger. Native qualification
requires a mechanically and energetically admissible preparation and the same
background-versus-perturbation comparison before a free-phase run.

Midpoint advances F with its actual linear equation; the native constant-stage
exact aging supplies stage/end ages. No section-root admission condition is
weakened. Half-step errors are measured on rate and age **perturbations**, not
only the large background or the small quadratic work remainder. Signed section
profiles, slip/Dc and background evolution are retained without a cold-section
or phase cutoff.

Confinement in these calculations is the model's prescribed 200 Pa. Stokes
pressure remains the incompressibility multiplier with its existing gauge;
its mechanical response is included, but there is no invented derivative of
friction with respect to current normal traction or solved pressure. This is
one reason the older sharp-interface instability matrix cannot be reused.

Units of the normalized modal coordinates: P and Psi have m^(-1/2),
q has m^(3/2), v has m^(3/2)/s and F has Pa sqrt(m). Cp has m/Pa and
Kp has Pa/m; physical section slip/rate are Psi q and Psi v. The diagnostic
force coefficient is 100 Pa sqrt(m), giving an approximately 81.65 Pa peak
for a normalized sine on a 3 m interval. `slip_response_per_force_coefficient`
has sqrt(m)/Pa. All physical work uses ds in m and is J/m of out-of-plane
thickness; this normalization does not change the section integration measure.

The additional sine-number48 action has a .0625 m half wavelength (one x cell).
It is a discretization-scale control, not a resolved physical mode or a mesh
convergence claim. Its restoring response and the change in the minimum when
it augments the six-mode space are reported. The nonlinear reduced pair retains
the six smooth modes; this single probe does not certify the entire omitted
spectrum.

The benchmark calls the Stokes inverse after physical postprocessing begins. It
therefore saves/restores the simulator's constraints, Newton flags and linear
tolerance, and installs the same homogeneous-increment context used inside the
coupled solve. Calling that inverse with restored physical Dirichlet values is
invalid; the retained initial diagnostic failures demonstrate why this context
is necessary. The explicitly recomputed mixed residual, not only Arnoldi's
estimate, must pass. This setup changes no physical boundary value or accepted
state and applies only while evaluating the read-only restoring actions.
