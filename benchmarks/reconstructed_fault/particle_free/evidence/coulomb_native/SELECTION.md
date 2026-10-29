# Selected bounded native Coulomb block

Fresh fixed 0.25 m square, 4×4 affine cells, Q2 velocity/Q1 perturbation
pressure, independent DGQ2/native source remainder and Gauss6 mechanics/source3.
Uniform physical phi=.2 remains fixed exactly, including its obstacle. Retain
G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², ell=.046875 m, m=4 and the distinct
strength-drop input 2000 Pa. New residual cohesion is c_r=0, mu=.1.

Total stress is sigma=tau-(p0+p')Id, p0=10000 Pa. Mean p'=0 is the prescribed
physical hydrostatic reference for this fully velocity-constrained block; it
is not an arbitrary gauge of a pressure-independent law. All prescribed
boundary reactions include -p0 Id, including initialization. No additional
Neumann confinement is applied or counted twice. Its work is
-p0 integral div(du), zero for the prescribed volume-preserving loading to
the measured mixed residual. The solved perturbation enters BOTH normal
friction and the complete mixed tangent, with the existing p'/G scaling.
Contact admission uses every actual mechanical quadrature point, not cell
means. Trial and returned in-plane stresses must both be fully compressive.

At p'=0 the unchanged material tuple has residual pure-shear strength
995.037190 Pa and intact activation threshold 2955.332456 Pa. These are not
recalibrated to the constant-resistance 1000/3000 Pa pair. The fixed phi=.2
block does not test intact phase activation or phase stability.

Use the qualified zero-dilation, two-conjugate-system point equations and
their complete nonsymmetric derivatives. Preserve T/A/K/Q/L, native memory,
once-only publication, immutable accepted trials and mode-tagged checkpoint
ownership. Existing checkpoint modes remain byte-compatible; conversion from
unconfined P5c is rejected. Promote the verified implementation into its native
material module without copying a second constitutive algorithm.

Run 24 steps at dt=.0125 s through .3 s: shear rate .02/s through .1 s,
in-plane normal deviatoric loading .02/s through .25 s, then displacement hold.
The nonuniform control adds a divergence-free sinusoidal boundary velocity of
amplitude .006/s times block length during loading. This is a material block,
not a crack or specimen campaign. Expected cost: minutes for the small runs,
plus bounded local compilation; record actual times. Verify homogeneous point
identity, nonuniform strain/pressure finite-difference actions at the existing
1e-6 bound, physical work/dissipation at the existing 1e-10 ledger bound,
source agreement at 1e-12, same-rank restart/rejection and 1/4-rank ownership.
Use common physical CellIds/points rather than partition-dependent DoF IDs.

The fixed-phase constraint removes phase variations from this experiment;
there is no Coulomb phase Hessian certificate. A phase-active selection is
explicitly prohibited. The follow-up proposal must treat the nonsymmetric
relaxed operator and dissipation/irreversibility separately; the old symmetric
potential certificate is neither reused nor disabled for its original modes.
