# Structural-plane sliding / mixed response: declared tests

Independent experimental reference at f1f9f29cf40c32d1ddcea000b724375b64853532
plus preserved work. The structural-plane hypothesis is now accepted for this
investigation; production/BP3 remain protected. Preserve all original evidence.

1. Fresh equilibrated strips use the same .25 m domain, ell=.046875 m,
G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², m=4, p0=10000 Pa and the previous
mu0=.1,a=.005,b=.01,V0=5e-6 m/s,Dc=.001 m,rho=2670 kg/m³. Compare fixed
profiles with peak phi=.8 (developed) and phi=1 (mature center), using the same
parabolic continuum prescription. No fixed phase or tensor history is reset.
Boundary speed .001 m/s until 2 s, .002 m/s until 4 s, hold until 4.5 s;
then -.001 m/s until 6 s. Target absolute accumulated slip >=Dc and meaningful
frictional-state change. These are finite small-strain block tests.

Start with 64/128/256 cells and dt=.002 s, then .001 s on 256 cells. Use
further bounded refinement only to resolve an observed issue. Compare
traction/slip/work/width and rate/state profiles at common times and mechanical
stage transitions. Meaningful tolerances: temporal <=.5% and spatial <=1%
relative to history maxima, with absolute floors 1e-5 Pa, 1e-10 m and
1e-10 J/m². Do not demand decreasing sub-floor differences. Energy accuracy:
absolute cumulative trapezoidal-work physical balance error and sum of
absolute step remainders each <=1% of sliding friction+radiation loss; also
report normalization by total work. The exact BE ledger is a separate check.
Retain 1e-8 Pa local rate and existing work/compatibility solver allowances.

Use two interface references without arbitrary width: mature RSF on the same
Maxwell block; and a coherent-profile reduction with cohesive coefficient
G integral chi²/h dy. These distinguish distributed local speeds/state from
additional stored cohesive energy. A mismatch with a pure mature interface is
physical model evidence, not numerical error to tune away.

2. Derive the full displacement-pressure-flow-Maxwell-state Fourier pencil for
a recorded plane, with actual normal pressure and incompressibility. Cross-check
an independent constrained reduction, full mixed generalized eigenvalues and
finite differences. Cover finite/developed/mature phase and rates around the
measured sliding range, both signs and oblique wavevectors. Test wavenumber
scaling and finite-step amplification, retaining all pressure derivatives.
Numerical checks <=1e-7 relative with 1e-9 absolute eigenvalue floor; no ban on
physical positive eigenvalues. A small perturbation integration must distinguish
bounded constitutive growth from timestep/solver error. A rotated band alone
is insufficient.

3. Only satisfactory sliding/mixed evidence admits phase coupling. Explicitly
resolve intact driving, direction/state admission, normalization derivatives,
fracture energy and history persistence before any tip-growth demonstration.
A substantive physical obstruction stops propagation progression; do not add a
constitutive remedy silently. The user authorizes necessary focused follow-up
and solver improvements, not changes to physics within a convergence comparison.
