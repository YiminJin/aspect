# Cap flanks and boundary loading — bounded diagnosis, 20 October 2026

**Fine-grid quadrature is not the cause of the flank response. Both mesh
directions and right-boundary distance affect it. Moving the right side to
1.5 m removes its damage island but increases flank growth; no tracked
centerline tip advances.** Recommend that wider specimen as the smallest
tested physical revision, while retaining the present law, seed and loading.
This does not establish domain independence or propagation readiness.

The three new serial runs complete the [predeclared selection](BULK_SENSITIVITY_SELECTION.md).
The existing core/plugin, unlimited DGQ2, Gauss3 source projection, material
parameters, continuum seed, physical obstacle, .018 s⁻¹ shear ramp (.05–.15 s),
holds, dt=.0025 s, end=.16 s and acceptance tolerances are unchanged. Every new
layout starts unloaded from its own samples of the same continuum seed; no
quadrature history is interpolated. Only mesh dimensions, the selected rule
and output paths differ in the expanded native parameters. No C++ source or
production default changes were needed.

## Numerical and boundary controls

| Mesh / bulk rule / right side | Initial phase relative L2 | Peak increment, each flank | First flank growth [s] | Peak right-face increment |
|---|---:|---:|---:|---:|
| 16×32 / G6 / 1 m (retained) | 1.33455% | .0000267181 | .1500 | .00340927 |
| 32×32 / G6 / 1 m | .779183% | .00101077 | .1425 | .00343350 |
| 32×64 / G6 / 1 m (retained) | .339721% | .00129893 | .1400 | .00346037 |
| 32×64 / G8 / 1 m | .339721% | .00129893 | .1400 | .00346037 |
| 48×64 / G8 / 1.5 m | .339721% | .00167120 | .1375 | **0** |

Flanks are the fixed lower/upper regions x=[.625,.8125] m,
y=[.25,.4375]/[.5625,.75] m. The two peaks agree to roundoff. The broader
regions deliberately include the cap edge; the prior narrower tip-window
results remain unchanged. Core-tip phase change, x=[.45,.85], |y−.5|<=.05 m,
is exactly zero in every case. Unloaded phase drift is exactly zero.

**Direct fine G6/G8:** maximum phase difference across common times is
8.4043e-11; maximum storage/work difference is 8.054e-11 on the retained
max(1 J/m, reference) scale. Both original quadrature screens pass.
At ramp end the largest flank differences in N-weighted driving, fracture
density and total weak density are respectively 1.905e-6, 8.596e-7 and
1.905e-6 Pa. On currently free flank nodes the residual densities are below
3.1e-14 Pa for both rules. This qualifies this particular fine-grid response
against Gauss8; it is not inferred from the earlier coarse-rule comparison.
Actual Gauss-point maxima sample different locations and are not used as
pointwise quadrature-error estimates.

**Directional refinement:** 16×32→32×32 refines only along the fault;
32×32→32×64 refines only across it. Their initial represented-phase Linf
differences are .0232633 and .00760268, respectively. Subsequent increment
Linf differences are .00101077 and .00126590 over the whole common domain;
the latter includes changed boundary damage shape. Within the flank the
normal-refinement difference is .000756974. These are differences between
trajectories from the same physical prescription, not errors against an
accuracy reference. Mechanics and phase/obstacle spaces change together.

At shared node (.71875,.3125) m, sampled initial phi is exactly .00395136858
on square/fine/wide meshes. Nevertheless initial weak fracture densities are
16.1418 Pa on 32×32 and 14.7486 Pa on 32×64. At .135 s their mechanical
driving averages are 10.8712 and 11.4768 Pa. The resulting weak residuals
5.27063 and 3.27179 Pa hold the obstacle; they reach zero at different later
steps. This exposes discrete profile-curvature and mechanical/averaging effects,
not an unloaded relaxation or failed KKT solve.

**Right distance:** 48×64 preserves dx=1/32, dy=1/64 m and G8. The near-field
initial phase is identical to fine G8 (common-domain Linf difference zero).
The right side moves; top/bottom stay at y=1/0, with the same affine velocity
also prescribed on their added portions. The peak flank increment increases
**28.66%**, with maximum common-field flank increment difference .000449041.
The right strip's actual Gauss driving maximum drops from 80.1231 to
58.8591 Pa. At its new, still-intact boundary, fracture density is 64 Pa and
the minimum total weak density is +6.03185 Pa: it does not activate. Flank
activation survives and starts earlier despite removal of that island.
Thus the island is boundary-sensitive; flank growth is not caused solely by
the island, but its amplitude remains sensitive to the imposed boundary.
Left-face growth persists (.00317155); no full boundary independence is claimed.

![Common-scale physical maps](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_sensitivity/maps.png)

*Same spatial axes/color scale in each row, t=.15 s. The cross marks the
original endpoint; the white line is the original diagnostic surface. Blank
space is outside the smaller domains. Driving is actual Gauss Y; fracture
and total rows are native weak nodal densities, not pointwise phase equations.*

![Centerline, both flanks and actual right-face cuts](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_sensitivity/linecuts.png)

*Cuts at y=.5, .3125 and .6875 m, plus each actual x=Lx boundary. Mechanical
line cuts show integral(NY)/integral(N), evaluated at the native nodes. Raw
weak contributions are retained separately. Changing basis/support changes
these diagnostic densities; the old x=1 boundary is an interior location in
the wide specimen. No local constitutive history is interpolated for these plots.*

## What grows, and what does not

All five tracked seed-connected centerline contours (.001, .01, .1, .3, .5)
remain exactly at their own initial positions. For the square/fine/wide cases
these are x=(.808036, .777466, .705079, .623786, .558443) m. Coarse offsets
are initialization differences; the largest is 15.36 mm for the .01 contour.
The seed-connected nodal component's rightmost extent also stays fixed.

Low contours on the **flank cuts** do move. At y=.3125 m the .001 front starts
at .742091 m for all three x-refined layouts and reaches .743702 (square),
.743944 (fine G6/G8) and .745381 m (wide): advances of 1.611, 1.852 and
3.290 mm. The .01 front advances 1.394, 1.693 and 2.524 mm. Upper-flank
results are symmetric; .1 does not advance. These are small lateral cap-edge
changes, distinct from centerline extension or a disconnected boundary island.
Initially intact flank-node increments reach .00007315 (square), .00051286
(fine) and .00075174 (wide), so the broader flank response includes both
pre-existing-cap thickening and activation just outside its represented support.

![Separate evolution and local weak balance](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_sensitivity/diagnosis.png)

*Growth histories separate flanks from the right face. At a shared physical
node, fracture and mechanical contributions balance on free steps; positive
residual during a hold can correctly retain the obstacle. Different final
centerline positions across coarse/fine meshes were already present initially.*

## Verification, energy and recommendation

Independent assembly of g_f uses exact Q1 Gauss2 integration; g_m uses each
run's actual positive Gauss/JxW data. Native g_f, g_m and g_f+g_m agree to
1.8e-14 after division by Kf times the nodal mass. Coverage is 1 m² or 1.5 m²,
explicitly checked. The wide rule's actual JxW range is
1.25088e-6–1.60571e-5 m², distinct from its 196608 point count. All 64 accepted
steps in each run pass the unchanged mechanical/KKT/source/publication gates;
no histories are imported or remapped between layouts.

The original physical energy failure is retained: endpoint-work remainders
were 2.1321% and 1.0786% at the earlier dt and dt/2. New square/fine-G8/wide
remainders are 2.1341%, 2.1354% and **2.2328%**, all failing the unchanged 1%
screen. Wide endpoint work is 2.32867062 J/m and remainder .05199430 J/m;
the separate post hoc trapezoidal-work diagnostic gives remainder −.000112154
J/m. This is consistent with the prior work-time-discretization explanation,
not a replacement acceptance test. Discretization and residual work remain
separate; exact algebraic ledger error stays below 6.41e-14 J/m. Whole-domain
work/storage change with added physical volume; those differences are not
accuracy errors.

**Smallest justified physical revision:** adopt the tested right extent
1.5 m at the same near-field spacing, law, seed and imposed strain history.
It removes a demonstrated parasitic right-boundary damage source without
inventing a traction, increasing the load or changing the cap. Keep Gauss8 on
that revised specimen; the unit-domain G6/G8 check alone does not qualify G6
on it. Do not declare the remaining flank
amplitude spatially converged or the specimen domain-independent. This revision
still does not produce centerline propagation; selecting a drive that does so
would require a separate physical experiment, not numerical tuning here.

Entry is f1f9f29cf plus preserved local bulk/P4a/manual work. Exact inputs,
commands, binary/source identities, comparisons, selected actual exports and
checkpoints are under `benchmarks/reconstructed_fault/particle_free/evidence/bulk_sensitivity/`;
see the [replay/measurement guide](../../../benchmarks/reconstructed_fault/particle_free/bulk_cohesive/SENSITIVITY_README.md).
New runs took 279.8, 89.2 and 429.6 s locally, with bounded timeouts and no
overlapping native runs. No C++ changes, new physics, additional sweep or server
job was introduced. BP3 and previous evidence, including
`strict_reference_match_passed=false`, remain preserved. Stop after this diagnosis.
