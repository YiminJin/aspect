# Fixed-phase short-wave mechanics and RSF spectrum

This derivation concerns the current synthetic material-strip specimen, with
its retained .075 phase, G=1e6 Pa, eta=1e8 Pa s, actual section planes and
ds=dx/cos(a). It changes no native constitutive or endpoint rule. The associated
independent reference is `stage4_wavelength/shortwave.py`.

## Mechanical compatibility

Freeze the slowly varying section coordinate x and retain the complete native
transverse phase polynomial phi(x,y). Put h=4 phi(1+phi)/(1-phi)^2 and H=h/(1+h).
For a Fourier perturbation exp(ikx), introduce the streamfunction

    ux = psi'(y), uy = -ik psi(y),
    psi=psi'=0 at y=0,4.

Thus incompressibility and zero displacement at both transverse boundaries
hold exactly. In full plane strain, Ezz=0, Exx=ik psi', Eyy=-ik psi',
2Exy=B=psi''+k²psi. For the retained direction S,

    D = 2 E:S = cos(2a) B - 2ik sin(2a) psi',
    e = H (D-f/G).

Eliminating the scalar cohesive strain e from elastic plus cohesive energy
and prescribed section resistance gives the Hermitian weak problem

    integral [4k² psi' conj(chi') + B conj(Bchi) - H D conj(Dchi)] dy
      = -(f/G) integral H conj(Dchi) dy.

The compliance is -cos(a)*integral(e)dy/f. The reference uses C1 cubic Hermite
streamfunctions, clamped at the physical top and bottom. It refines the normal
mesh independently of the frozen profile and checks quadrature, mixed-field
compatibility, and physical energy against half force times slip. Scaling
w=k²psi avoids poor conditioning in the short-wave mass limit.

For k tending to infinity, the diagonal strains become O(1/k); the off-diagonal
strain remains finite. A tilted cohesive plane therefore retains a diagonal
elastic penalty, while a horizontal plane can approach pure cohesive shear
through a compatible vertical displacement of amplitude O(1/k). Minimizing
the limiting energy gives

    C_inf(x) = cos(a)/G * integral h/(1+h sin²(2a)) dy.

This result follows from compatible displacement fields, not from dropping
bulk energy by assumption. Boundary layers and smoothed DG profile jumps can
be made to have vanishing energy as k grows. Compact oscillatory envelopes
inside the loading patch also satisfy the distant left/right boundaries; a
finite-k calculation based on local cross-sections remains a slow-x (WKB)
approximation and is not an exact finite-body native eigenanalysis.

The fixed-normal-Q2 velocity restriction has a separately computable limit:

    C_inf,Q2 = C0 + cos(a)/G * l^T M^-1 l,
    C0 = cos(a)/G * integral H dy,
    Mij = integral (1-H cos²(2a)) Ni Nj dy,
    li = integral H cos(2a) Ni dy.

Here Ni are the continuous Q2 transverse velocity functions with zero endpoint
values. The remaining velocity component enforces the weak pressure equations
at lower asymptotic order. Comparing this mass projection with C_inf separates
normal-space restriction from tangential wavelength resolution.

## Which limiting growth rate?

Let r=G/eta=.01/s, q(x)=abs(V)/Dc and c(x)=ftheta*theta*sign(V)/Dc, evaluated at
the same declared, generally transient, rate/age operating point. The local
Maxwell/RSF branches satisfy

    fV lambda² + [fV(r+q)+K-c] lambda + q(fV r+K)-r c = 0,
    K=1/C_inf(x).

Their maximal local real part describes the short-wave principal symbol and
packets whose envelopes can concentrate near the fastest section. It is not
the growth rate of a force carrier spread over the complete 3 m patch.

For that fixed patch, normalize the rapidly oscillating sine/cosine force and
eliminate **all** responsive ages. Fast-phase averaging gives the dispersion
relation

    integral(C_inf ds)/(lambda+r)
      + integral[C_inf² (fV-c/(lambda+q)) ds] = 0.

The actual transverse profile, V(x), theta(x) and ds all enter this expression.
The two carrier phases become degenerate as their wavelengths become resolved
and short. Six fixed smooth modes decouple from this rapidly oscillating sector
at leading order. A full family of nearby carriers could form localized
envelopes and approach the local-symbol edge instead.

Initial reference results approach about **41.13/s for the fixed-patch carrier**.
The limiting local packet edge near the center is about **61.67/s**, now subject
to the compatibility derivation above rather than an uncoupled assumption.
Neither value is the earlier **155.2/s general upper bound**. Native spatial
comparisons and reference verification are recorded in the continuation report;
these statements do not establish a native nonlinear event or selected patch
spacing.

## Two cells per wavelength

A native Q2 velocity has a piecewise-linear x derivative. At two cells per
wavelength, the squared projection of a sine half-wave onto that linear space
is rho_s=8/pi²; for the cosine it is rho_c=96/pi^4. Thus a fixed cells/wavelength
sequence need not have the same limit as resolved x mechanics. Writing Cm for
the transverse mechanical mass-projection compliance, the proposed fast-phase
moments are

    Cwork = C0 + rho Cm,
    Csquare = C0² + rho(2 C0 Cm + Cm²).

Replace the first and second compliance factors in the carrier dispersion by
these respective moments. This explains why refinement of a fixed disturbance
and admission of a shorter disturbance must be compared separately. These are
asymptotic estimates; a two-cell native result at finite wavelength is not a
measurement of the limiting growth rate.
