# Mature sliding compatibility — bounded selection before execution

Use the retained constant-coefficient tensor Coulomb/Maxwell law and joint
microforce normal map, with no production-source changes. The experiment is
an equilibrated one-dimensional cross-band reduction, not the preceding
prescribed-strain strip and not a native propagation qualification.

G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², ell=.046875 m, Kf=8 Pa, m=4,
mu=.1, residual cohesion=0 and mean total pressure p0=10000 Pa are unchanged.
At fixed phase, stronger surroundings are represented by h(phi), not changed
G or friction. Domain y∈[-.125,.125] m. Phase=.999 on |y|<=ell/2, .02 on
|y|>=ell, and the quintic smoothstep between them. This is a deliberately
prescribed mature-like plateau with smooth shoulders, not a phase equilibrium
or a new damage-to-maturity rule. Core h≈7.988e6, surroundings h≈.08497.
The exact phi=1 endpoint is NOT inserted into a law that admits phi<1 only.
The .02 surroundings are stronger nearly intact material, not exactly intact.

Fresh T=A=K=Q=L=0 for every case. Impose relative tangential motion at
1e-3 m/s to total 2e-3 m (2 s), no normal motion, zero body force, and
microfree outer boundaries M n=0. No internal K=0 skin, slip mask or direction
lock is added. Both Coulomb systems and tensor components remain in the joint
solve. Enforce constant physical shear traction and constant normal traction;
mean p'=0 recovers p(y)=p0+tau_yy(y)-mean(tau_yy). This includes the same
physical pressure dependence as the qualified bulk block. On the symmetric
solution p'=0, but its tangent is not frozen.

Main N=257 nodal P1 microforce discretization, native-positive trapezoidal
constitutive weights; exact P1 gradient stiffness. Lengths 0/1/3/6 mm are the
existing exploratory, uncalibrated values, never tied to mesh spacing. Limited
resolution checks: 6 mm on N=129/513; one dt halving at N=257,6 mm. Baseline
dt=.01 s. Initial target plastic slip >=.5 mm; remain within 2 mm imposed motion
or stop at an original contact/force/solver limit. No enlarged campaign if the
failure of mature compatibility is already established. Estimated total small
strip cost: minutes, not a specimen run. Full tensor contractions and BE
storage/dissipation accounting remain; original 1e-12 relative force and 1e-10
energy bounds are retained. Each step solves from immutable accepted histories.

Before inspection, a mature-friction compatibility screen is <=5% extra
accumulated-gradient resistance relative to the initial friction traction over
.25–1 mm accumulated plastic slip, and <=5% change of the active slip width.
These are interpretation criteria, not solver tolerances or phase masks.
Report magnitudes and the window actually reached. Resolution checks seek
<=1% traction and <=2% common-profile differences; failure does not permit
retuning lengths/profile. A dt halving separates temporal uncertainty.

For a fixed normalized engineering-slip profile rho, K=delta rho S,
S:S=1/2 and integral rho dy=1. Therefore Egrad=G ell_s² delta²/2 integral rho'^2,
and its conjugate traction Rgrad=G ell_s² delta integral rho'^2 grows without
bound during repeated sliding. That is a constitutive hardening term, separate
from relaxing cohesive Q and friction. Unrestricted computed profiles may
change; measure their widening and the actual gradient power as well.

If measured incompatibility appears, stop coupled-phase progression and
recommend one explicit alternative for scientific review. Do not change the
coefficient, reset K or implement an unreviewed replacement. If compatibility
is supported, proceed to the bounded coupled formulation/qualification in the
user instruction, without using the old symmetric potential Hessian.
