# Common initialization and exact-bulk comparison

This bounded P2 diagnostic changes the definition of the test initialization,
not the total-energy evolution law, normalization, obstacle or production code.
The previous particle-sampled initialization and all its trajectories are retained.

Let a=.6, A=Gc/(c0 ell), h=(1-g)/g, and r=|y-.5|. The physical seed is the
native profile defined by

    r(phi)=ell integral_phi^a [s-a h(s)/h(a)]^(-1/2) ds.

The endpoint singularities are removed by s=a cos²(theta). The native seeded
history is H_seed=A a[1+h(phi_seed)]²/h(a) when phi_seed>10^-6, and Hc=.5 Pa
outside. This is the existing initialization branch, not a damage activation
mask. The small actual transition at phi_seed=10^-6 is retained.

One continuous stationary initialization is solved at this prescribed H:

    A(1-2 ell² phi_yy) - H_seed h'(phi)/(1+h(phi))² = 0,
    phi_y(0)=phi_y(1)=0,

with continuity of phi and phi_y at the H transition, reflection at y=.5, and
box admissibility. A split-domain BVP resolves that transition. A second tighter
solve and doubled seed integration check its numerical uncertainty. This field
is then sampled at every mesh's Q1 nodes; no separate particle sample determines
its reference. Q1 interpolation error remains and is explicitly reported.
Any correction for box admissibility is measured, never hidden. The unloaded
state R=Z=slip=0 is protected by this interpolated phase obstacle; it is not
claimed to be an unconstrained equilibrium of the unloaded total energy.

For the unchanged exact-bulk mechanical elimination retain the sampled
C=W h(B phi)/I rather than silently asserting C=1. Let e=beta G W R_a+kappa
bar-u, and Z=beta Z_a+kappa V. Endpoint stress is T=e-Z C. The reduced potential
on the slipping branch is

    J = e²/(2G) - (e C-mu P)²/[2G(C²+1/I)]
                  - mu P beta Z_a/G + F(phi).

On the sticking branch Z=beta Z_a is fixed and
J=(e-Z C)²/(2G)+Z²/(2G I)+F. The envelope derivative with respect to phi is
exactly the previous discrete energy gradient and normalization adjoint.
The Hessian can be evaluated efficiently in the two variables C,I:

    H_phi = F'' + J_C C'' + J_I I''
            + J_CC C' C'^T + J_CI(C' I'^T+I' C'^T)+J_II I' I'^T.

At fixed Z, J_C=-ZT/G, J_I=-Z²/(2GI²), J_CC=Z²/G,
J_CI=0 and J_II=Z²/(GI³). On slip subtract the Schur term
(j j^T)/[G(C²+1/I)], j=(e-2ZC,Z/I²). The C derivatives include the actual
six-point bulk quadrature and analytic piecewise-Q1 I derivatives. This is an
algebraic simplification of the retained operator, checked against its original
matrix construction and independent finite differences, not a new discretization.

Accepted R_a,Z_a, slip and the previous physical phase obstacle remain immutable
through each trial. Only the positive translation direction is searched when a
negative admissible mode is found. Complete equilibrium/stability/bounds checks
precede a single publication. General beta=exp(-dt/lambda) is retained.

Centroid alignment is a postprocessing comparison of functions only. No accepted
state, obstacle, internal strain, slip or normalization is translated. Spatial
curvature uses the same consistent Q1 mass norm. Continuum translation can be
neutral only for a localized field whose shifted support stays inside the normal
domain and whose entire shift respects the fixed obstacle. Boundaries and active
sloping contacts can exclude this direction. Therefore a nearly translating
negative discrete mode does not establish a physically unstable continuum band.
