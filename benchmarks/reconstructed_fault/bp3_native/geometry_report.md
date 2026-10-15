# Moving-particle domain conditioning and operator verification

2026-10-14, `fault-propagation`, `c33edaaeb` plus preserved incoming work.
This follows the [retained transport failure](transport_report.md). The dense
shear now completes all 32 positive steps to T=1 s. The original 1e-10 coverage
guard remains unchanged. Two distinct numerical defects were resolved: absolute
Voro++ tolerance sensitivity and orientation-dependent 2-D CPDI gradients.
No constitutive, transport, loading or physical assumption was changed.

## Coordinate experiment and choice of correction

The failing step's actual 2,304-site host-cell patch was reconstructed with the
installed Voro++ 0.4.6, including its original box, artificial slab and block
counts. Its untransformed relative area error, 4.0743642e-8, reproduces the
saved native polygon. Reconstructing from the saved nearby sites gives the same
independent exact cell; differences in raw Voro++ approximations with a smaller
container are separately retained.

Each dataset uses eight length scales (.001 through 1000) and three consistent
translations: zero, parent-centered, and (1000,-2000) times the scale. Sites,
container and slab transform together; the result is transformed back before
evaluating moments and CPDI. Block counts stay fixed to isolate geometric
conditioning. Each chart choice has 24 configurations per dataset. The physical
reference is independent half-plane clipping, not a second Voro++ evaluation.

Representative full-patch results without translation:

| Chart / length scale | Relative area error | Symmetric difference / area | CPDI weight L1 error | Dimensionless CPDI gradient L1 error |
|---|---:|---:|---:|---:|
| Raw / .001 | -1.09448e-3 | 1.38716e-3 | 6.92808e-5 | 1.41018e-4 |
| Raw / 1 | 4.07436e-8 | 4.07436e-8 | 2.54647e-9 | 5.09383e-9 |
| Raw / 10 | 0 | 3.77e-15 | 2.71e-16 | 1.39e-15 |
| Centered particle units / .001 | -1.78e-15 | 4.88e-15 | 3.23e-16 | 2.33e-15 |
| Centered particle units / 1 | 0 | 1.78e-15 | 1.94e-16 | 1.04e-15 |

The scale effect, exact reproduction with the complete native patch, and
translation insensitivity at ordinary offsets identify Voro++'s absolute
tolerances as the source of this polygon error. They do not support changing
neighbor admission or periodic-image selection. Across all translated/scaled
full-patch tests, conditioning limits area error to 2.92e-11, symmetric difference
to 4.85e-11, weight L1 error to 8.53e-13 and gradient L1 error to 2.21e-12.
The unnormalized gradient weak contribution differs by at most 9.15e-11.
These worst cases include rounding the far-translated input coordinates;
conditioning cannot recover input bits lost at approximately epsilon*offset/h.

The minimal production correction is confined to 2-D domain construction:

    origin = center of the existing neighbor-patch box
    ell = sqrt(patch_box_area / number_of_inserted_sites)
    x_voro = (x_physical - origin) / ell

Real sites, existing periodic images, box and slab use this same chart; returned
vertices are mapped back before native cleaning, physical boundary checks and
CPDI. Existing neighbor sets, image IDs, block counts, physical domains and
particle properties retain their meanings. No geometric floor, new parameter,
library tolerance or production acceptance threshold is introduced. The 3-D
path is unchanged and is not qualified here.

## Local operators exposed a separate orientation defect

The old code's value integral uses absolute simplex area, but its gradient
integral keeps the signed orientation of its edge normals. Consequently 8,011
of 16,384 domains at the failed step reconstruct d(y)/dy=-1; the others give +1.
Partition of unity and the derivative of a constant both pass in either sign.
An area check therefore cannot detect this defect.

For the simplex's existing signed measure A_s and barycentric basis L_i,
the physical integral is |A_s| grad(L_i). Multiplying the existing oriented
gradient integral by sign(A_s) supplies exactly that expression. The correction
is 2-D only and does not change the scalar basis weights. This follows the
existing gradient definition; there is no new numerical or physical law.

Native operators were compared against independently clipped polygons and
independent barycentric integration of the **same CPDI centroid fan**. A second
comparison uses each native polygon unchanged, separating geometry error from
integration error. This does not claim that the generalized CPDI fan equals
exact integration of a cellwise Q1 basis over the polygon.

Here h_p=1/128 m and h_b=1/8 m. Centroid errors use h_p; second moments use
h_p squared. Weight errors sum absolute differences over each parent's nodes;
gradient errors sum vector norms multiplied by h_b. H-weighted weak differences
use the common sampled mass sum(H_p*A_p) as their scale; gradient contributions
also include h_b. These fixed scales avoid division by cancellation-prone nodal
entries or vanishing local gradients.

| Measured quantity | Original step 2 | Corrected maximum over steps 0–32 |
|---|---:|---:|
| Sum of absolute local area errors / box area | 7.31746e-9 | 2.35e-15 |
| Sum of polygon symmetric differences / box area | 7.70502e-9 | 5.09e-15 |
| Maximum local relative area error | 4.07436e-8 | 2.18e-14 |
| Maximum centroid error / h_p | 2.03722e-8 | 9.34e-15 |
| Maximum normalized second-moment error | 6.79057e-9 | 2.48e-15 |
| Maximum local symmetric difference / area | 4.07436e-8 | 2.85e-14 |
| Maximum CPDI weight L1 error | 2.54647e-9 | 2.69e-15 |
| Maximum dimensionless CPDI gradient L1 error | 6.68492 | 8.42e-14 |
| H weak-vector relative L1 difference | 7.87421e-9 | 4.48e-16 |
| Dimensionless H gradient-vector difference | 1.78990 | 3.15e-15 |
| H weak absolute local-error bound | 8.37998e-9 | 3.91e-15 |
| H gradient absolute local-error bound | 3.11984 | 1.32e-14 |

The absolute local-error bounds sum before assembling shared nodes, so neither
overlap/gap cancellation nor weak-vector cancellation can hide local defects.
Independent neighborhoods include all actual sites in nearby spatial bins and
the correct periodic images. The retained polygon radius certifies that every
omitted site is too far away to cut it. Physical top/bottom walls are clipped
explicitly. Corrected seam and wall weak errors stay below 1.74e-14 and 1.46e-14
locally. Linear-gradient error stays below 7.55e-15, with zero reversed domains.

Source inspection finds the current production CPDI-gradient consumer in
`source/simulator/phase_field.cc`: its quadratic gradient products are invariant
under a uniform sign flip on one domain. That cancellation explains why prior
phase checks need not expose the defect; it does not make the gradient correct.
All earlier P2/stationary evidence is preserved, without a new qualification claim.

## Completed trajectory and verification

Both corrected shear runs retain the original mesh, RK2, Maxwell data, passive
pulse, dt=.03125 s and T=1 s. All 32 physical steps plus step zero complete;
each particle crosses at least two cells. No normalization, solver or coverage
tolerance was relaxed. Every rebuild passes the independent local checks.

| Particles per cell per direction | Final pulse weak error | Maximum over time |
|---|---:|---:|
| 8 | .607266% | 2.180573% |
| 16 | .160968% | .543435% |

The dense particle stress error is .00693751 Pa, with once-only Maxwell
decomposition defect at most 2.33e-16 Pa. Particle pulse values remain exactly
unchanged; sampled mass relative error stays below 2.22e-15. Domain and operator
errors remain at roundoff through all subsequent rebuilds, rather than growing
after the formerly failing step. The two-density comparison establishes decreasing
error, not a general convergence order or full coupled moving-history accuracy.

The linear-gradient regression fails with the original native domain object and
passes with the corrected object. Additional native coordinate-unit checks cover
periodic domains, vertex geometry and CPDI under small/large scales and offsets.
Detailed actual test counts, commands, hashes and preservation checks accompany
this report in `build-bp3-native/transport-geometry/`. The original failed run and
strict guard are retained. Analysis with the explicit fresh-run overrides passes;
its default still reports the preserved historical failure.

Actual checks: corrected native tests pass **841 assertions in 12 cases**.
Five native coordinate configurations give maximum relative area discrepancy
1.18e-11, symmetric vertex distance/h_p 1.48e-11 and CPDI discrepancy 5.37e-12.
The original linear-gradient regression fails two assertions at error 2.8284;
the corrected regression passes. Isolated compilation/relink of the affected
native unity objects and the diagnostic plugin passes; no full core rebuild is
claimed. Style, Python syntax, whitespace and final analysis checks pass.
Three simulation launches and three bounded unit-test launches total **46.32 s**
excluding compilation; coordinate probes take under one second.

Changed production source: `source/particle/particle_domain.cc`. Regression:
`unit_tests/particles.cc`. Existing prescribed observer/analysis plus the focused
`geometry_reference.h`, `geometry_probe.cc` and three inheriting PRMs provide the
bounded diagnostics. No broad refactoring or full history campaign was performed.
The corrected executable is separate from `build-bp3-native/aspect-release`;
the original executable, plugins, archive and incoming work are preserved.

The remaining next scope is the previously identified common D3/G convention and
actual coupled moving-history publication check. Surface RSF, MPI transport,
birth/deletion, restart, automatic cutback, objective rotation and a moving full
BP3 first-event trajectory remain unqualified; none was launched here.
