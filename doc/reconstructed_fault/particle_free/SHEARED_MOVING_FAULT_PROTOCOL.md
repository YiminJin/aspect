# Bounded sheared moving-fault comparison — 4 November 2026

Selected task: four native runs, ordinary uncut DGQ2 and SSPRK3, on source
`cf06525471ce89789f1ac10905dc79f3aac7562a`. Preserve the prior frozen split-Q2
experiment; it is not used in these runs. Entry source/binary/protected hashes
are in `evidence/transported_objective/sheared_moving_fault/entry.json` under
the particle-free benchmarks directory. No rebuild or native change is planned.

Use the existing `run_material_native.py` fixture: direct physical stress,
constant G=10 Pa and eta=10 Pa s, total-material Jaumann Lie update with the
accepted-gradient lag, bounded MacCormack Q1 profile, accepted-trace tensor
inflow, full-normal support, no growth. The compact initial profile is
`.35*max(0,1-(x/1.4)^2)^4*max(0,1-((y-.3*x)/.6)^2)^4`.
Keep the fixture's boundary types, traction/background load, age initialization,
endpoint constitutive update, q8 integration and convergence criteria.

Boundary velocity on left/right/bottom is `(0.4+0.5*y,0)` m/s; top is natural.
Run to 0.4 s. Nominal affine accumulated spin is -0.1 rad, but measure the
actual solved gradient along material labels. The initial ends are
(-.8,-.24),(.8,.24); nominal affine final ends (-.688,-.24),(1.008,.24)
are safely inside [-2,2]^2. Verify actual clearance at every accepted step.
The top condition means this nominal affine solution is not an exact reference.

| Case | Mesh | dt (s) | Physical steps |
|---|---|---|---|
| shear32-dt100 | 32² | .1 | 4 |
| shear32-dt050 | 32² | .05 | 8 |
| shear32-dt025 | 32² | .025 | 16 |
| shear64-dt025 | 64² | .025 | 16 |

All use initial constitutive interval .1 s, initial label spacing .1 m,
identical physical initialization and four MPI ranks, sequentially. Compare
initial states explicitly: fixed-mesh temporal controls should match; spatial
initialization differences must be reported. No surface refinement or resampling.
Use the pressure-export checker and native stage-energy ledger; do not repeat
the completed isolated rotation/source tests. At most 300 s per 32² run and
900 s for the 64² run. The finer-run wall limit was increased before launch
after measuring 227 s for the 16-step 32² case; its physical duration and
number of steps are unchanged.

Compare final physical fields and matched material labels. Three dt levels give
successive differences and an observed self-convergence order, not a continuum
error certificate. One finer mesh gives spatial sensitivity, not a spatial order.
Use full 3D plane-strain tensor norms, own-curve shear and total normal traction
(report deviatoric normal separately), endpoint-integrated cumulative slip
`sum(dt*V)` from all accepted steps, and geometry displacement. Normal traction
is tensile-positive `n.tau.n-p`; compression for friction is `5+p-n.tau.n`.
Also retain total driving shear, adding the initial material-label background
`2.7+.15*cos(3*x_initial)` to bulk shear traction. The native CFL rule chooses
SSPRK3 substeps separately on each mesh; physical dt is the controlled time
parameter. The full lagged Lie/endpoint scheme is not third-order in physical dt.

Central region: material coordinate 0.2 <= a <= 0.8. End regions: the outer
20% at each end, also reported separately. For bulk stress use common Eulerian
points in a 0.2 m tube around the reference final curve, assigned by closest
material segment; retain the whole-domain norm. Surface metrics use identical
material coordinates on each run's own curve. Use the dt/4 curve for the common
temporal norm regions/measure and the 64² curve for the spatial comparison.
State these distinctions: geometry
error is included, not silently removed. Report absolute and relative norms,
normalization, actual accumulated spin and curve orientation (different under
shear), accepted-step counts, cost, signed/absolute energy remainders, slip-work
adjoint, physical boundary flux, and raw-face versus weak mechanical power.
Existing pure-translation energy assumptions must not be used to set mechanical
power to zero. Refine diagnostic quadrature on saved final states as needed.

Failures remain evidence. Do not tune loading, solver tolerance or physical law
to manufacture convergence. This is a bounded mature-fault sensitivity study,
not cohesive qualification, a coupled split-space implementation or production.
