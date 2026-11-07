# Frozen finite-end comparison protocol — 4 November 2026

Selected scope: offline spatial/quadrature comparison after the direct-stress
report. No native evolution, source change, rebuild, production checkpoint or
new tip model. Preserve the incoming dirty worktree and all earlier evidence.

Freeze the initial **64² Q1 profile** and 18 curve labels, normalization I and
support from `true-c-p-ref`. Do not resample/reinitialize them on candidate
tensor grids. Prescribe the same native scratch source as the previous check:
zero incoming physical tensor, D=(.03,-.03,.01), V_j=.002(1+.2 cos X_j.x),
G=10, dt=.1; eta=10 or 10(1+.5 tanh(4x)). Evaluate complete local products
f=2 kappa(D-chi V S), kappa=-eta expm1(-G dt/eta), with full plane-strain
tensor contraction. This is a frozen reaction test, not an evolved solution or
an exact separated-history solution for spatially varying eta.

Compare uncut tensor-product DGQ2 on 16²,32²,64²,128² grids with ordinary
Gauss orders 8,12,16,24. Build an independent projection P* using polygon
partitions at Q1/tensor cells, every surface-label normal line (including the
two finite ends), and diagnostic window boundaries. Integrate triangles by a
Duffy-mapped tensor Gauss rule, orders6 and10. The fixed native support must
cover the domain; reject this specialized straight-curve evaluator otherwise.
No smoothing or extension of the physical source is allowed.

Separate best-space error ||P*f-f|| from quadrature error ||Pqf-P*f|| and
verify their squared-norm sum against ||Pqf-f||² in the converged integration
measure. Report full tensor and component norms, fixed central/end windows,
point shear traction, chi-weighted work-adjoint traction with a common surface
mass, signed source energy change, localization normalization, and offline cost.
Central: distance along curve >.3 from either end, |normal|<.4. End windows:
within .3 tangential distance of either end (inside and outside), |normal|<.6.
Split quadrature on these boundaries; do not estimate windows by cell centers.

Predeclared numerical screens (not physical acceptance tolerances): partition
area/mass identities <2e-11 absolute; reference6→10 tensor coefficient L2
change <1e-9 times source L2 +1e-12; reference work-traction change <1e-9 Pa;
orthogonal error decomposition <1e-10 relative +1e-16 absolute. Check the fast
source against the retained independent native-source reconstruction and reuse
its existing native q8 bridge. Retain any failures and refine only the numerical
reference if necessary; measured approximation errors have no invented pass
threshold. Output new compact evidence under `evidence/transported_objective/
frozen_finite_end/`; do not overwrite the completed direct comparison.
