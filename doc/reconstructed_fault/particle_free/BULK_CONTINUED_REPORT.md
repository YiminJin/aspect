# Exploratory continued loading on the fixed 96×128 specimen

21 October 2026; f1f9f29cf40c plus preserved local work.
The one selected trajectory **completed at 0.185 s**, with 148 accepted
publications and no solver, admissibility or source-check failure. Extra loading
produces low-phase cap widening and some forward contour displacement, together
with substantial interior and boundary nucleation. It does **not** demonstrate
developed tip propagation: centerline contours at phi≥.05 do not advance, and
the eventual domain-spanning .001 component forms by joining damage regions.

[Full animation](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued/continued-loading.mp4)
([GIF preview](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued/continued-loading-preview.gif)),
[growth histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued/continued-growth.png),
[reaction/work histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued/continued-energy.png).

![Loading-stop whole-domain and tip views with reaction and energy histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued/continued-loading-stop.png)

**Caption.** The frame shows the loading stop at .175 s, before the hold;
the [final hold frame](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued/continued-final.png)
shows zero local rate. The imposed shear rate remains .018/s through .175 s, followed by
fixed displacement to .185 s. The native coupled solve calculates phase and
local slip strain rate w [1/s], rather than a surface slip velocity. White
dashed contours are the initial diffuse seed; cyan contours are the current
field; red marks the original seed axis/end. No surface segment is inserted.
Every frame uses identical axes and colour limits, includes the whole domain,
and shows damage relative to initialization. Rates are direct native-Gauss8
exports from .12 s; earlier frames use the equivalent accepted-a recurrence,
whose maximum discrepancy on direct-rate frames is 4.17e-15/s. Display-bin
areas are not quadrature weights. Spatial accuracy and growth speed remain
unqualified; playback is not a physical speed measurement.

## Same initial problem and matched prefix

The [pre-run selection](BULK_CONTINUED_SELECTION.md) retains the [0,1.5]×[0,1] m
domain, Delta x=.015625 m, Delta y=.0078125 m, **Delta t=.00125 s**, seed,
length scale, slip orientation, homogeneous law, unlimited DGQ2/native-FE
complement, Gauss8/source3, physical obstacle and original tolerances.
Only the selected loading stop changes from .15 to .175 s. The original
rolling checkpoints contain accepted states .15375/.15875/.16 s, already in
the old hold, so the new trajectory starts freshly from the same continuum
prescription, without transferring or resetting loaded histories.
The 110592 stationary diagnostic particles remain; domains/CPDI stay disabled.

Initial seed errors are .085286% whole-domain and .188486% in the original
cap (relative L2); unloaded drift and KKT are zero. The complete prefix through
.15 s agrees exactly with the retained fine case. At .15, DG/local/phase
history, native FE memory, solution, phase CSV, compact q and dense physical
fields are all **byte-identical**. There is no initialization-relaxation change
to explain the later response.

## What grew

Distances below are from each **initial contour**, not from the original
(.5,.5) endpoint. Centerline means the first outward crossing along y=.5;
flanks use exact Q1 root evaluation on the ±45-degree rays. Values at .175 s
remain unchanged through .185 s. Both flanks agree to the displayed precision;
no branch alignment or averaging is applied.

| Phase level | Centerline advance at .175 s [mm] | Upper/lower radial advance [mm] |
|---|---:|---:|
| .001 | 9.50 | 41.92 / 41.92 |
| .01 | 2.12 | 13.10 / 13.10 |
| .05 | 0 | .205 / .205 |
| .1, .3, .5 | 0 | 0 / 0 |

At .001 the first centerline front moves from .800619 to .810115 m: only
1.29 mm beyond the original continuum compact-support end .808822 m. This
subcell weak-phase displacement is not resolved evidence of a developed crack.
The much larger flank displacement describes cap widening. Meanwhile a new
interior patch activates in (.15,.15125] s near (1.03125,.5) m; right-boundary
activation follows in (.15125,.1525] s. Initially zero nodes first activated
earlier, at .1375 s near the left band, as contiguous band thickening.

At .155 s there are two disconnected .001 superlevel patches. Later merging
and additional weak patches culminate in a seed-connected .001 component
reaching x=1.5 m at .17375 s. Its previous rightmost extent was .82446 m at
.1725 s. **That jump is a connectivity change, not a tip-speed measurement.**
Eight versus sixteen diagnostic samples per cell give identical component
counts and rightmost extents at the checked appearances/joins and final state.
This checks field evaluation, not physical mesh convergence. A separate
1e-12 native-node activation graph also records weaker, sub-.001 activation;
its counts are explicitly distinct from superlevel-set counts.

| Accepted time [s] | Peak cap increment | Whole integral of damage [m²] | Increment outside continuum seed support [m²] | Reaction Q [J/m] |
|---|---:|---:|---:|---:|
| .150 | .001980 | 3.89538e-5 | 3.06380e-6 | 2316.920 |
| .175 | .006728 | 1.89394e-3 | 1.65341e-3 | 2535.564 |
| .185 | .006728 | 1.89394e-3 | 1.65341e-3 | 2535.310 |

Here cap means x>.5, radius<.35 m about the original endpoint. It contains
9.96% of final integrated growth; 87.30% lies outside the continuum support.
Using the stricter initially-zero **represented** field instead gives 86.17%.
The initial Q1 tail outside continuum support, 4.85e-7 m² in integrated phi,
is recorded separately. Final whole-domain peak damage is .009932 at the left
band boundary (0,.7890625) m; the right-boundary peak is .008779. Peak local
|w| is .16242/s at .16125 s.

During the hold, phi, signed/absolute shear memories and w are exactly fixed
(w=0). With beta=1/(1+Delta t G/eta), physical stress relaxes by beta-product
.9999000056247656; its measured
full-tensor L2 discrepancy from this scaling is 1.50e-15 (maximum absolute
5.12e-11 Pa). The reaction follows the same factor. Stopping motion leaves
substantial stress; it does not unload the specimen.

## Work, histories and limits

All energies are J/m of out-of-plane thickness. Initial fracture storage is
9.44746118; final changes are bulk 2.29084489, cohesive .140724768 and fracture
.110195199. Physical dissipation is .840562500, separated into bulk viscosity
.00295577549, cohesive viscosity .00006536212 and friction .837541363.

Endpoint work is 3.410773947; trapezoidal force–displacement work is
3.382248855. The endpoint remainder is .0284465919; the trapezoidal signed
remainder is -7.85004e-5 and its sum of absolute step remainders is 7.85065e-5,
or **C_E=.00232113%** of absolute trapezoidal work. Added loading alone supplies
1.10503630 trapezoidal work, .648225195 storage change and .456847097 physical
dissipation. The hold supplies zero work, with storage change -.000486359526,
dissipation .000486356486 and remainder 3.04e-9. Small final error does not
hide alternating step errors. Per-step values are retained in
[energy.csv](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued/energy.csv).

Keep the exact BE ledger separate: stress/traction/fracture losses sum to
.0277991502/.0005554890/.0001744205, and the signed phase remainder is
-.00008246774. Phase residual work is -2.23e-12, mechanical residual work
1.48e-11, and pressure work -1.64e-21. The maximum native ledger defect is
2.31e-13. The present endpoint fraction is .8340%; this different event does
not erase earlier failed 1% screens. **C_T and hence the previously recommended
B_E criterion are unmeasured for this new regime.**

Maximum native mechanics/KKT/source measures are 8.08e-10/7.91e-8/7.32e-15,
below unchanged 1e-9/1e-7/1e-12 limits. Independent cumulative storage checks
agree within 2.79e-12 J/m; dense physical bulk storage within 5.29e-14 J/m.
Slip-memory recurrence errors are ≤8.41e-19, native FE recurrence error
≤2.17e-19. Every accepted endpoint publishes once.

All accepted phase/q and rates from .12 s, full histories from .15 s, and
15 file-verified native checkpoints from .11875 through .185 s remain saved.
The largest single-step damage increment is .00069989 in (.15375,.155] s;
none reaches the predeclared .001 flag. Conservative acceleration flags occur
near first growth and remote nucleation. Preserve **(.15,.155]** for activation
and **(.17125,.17375]** for contour joining, using the saved pre-window anchors.
No growth speed is validated and no timestep rerun is started.
No resumed trajectory is run; the existing native persistence infrastructure
is retained and archived, rather than requalified for this entire regime.

The earlier 35.86% growth discrepancy and failed 5% spatial target remain;
`strict_reference_match_passed=false`, branch findings, energy history and BP3
are preserved. This exploration supports cap widening plus distributed
activation under the chosen load, not clean mature-tip propagation. The next
bounded numerical question is timestep sensitivity in the saved intervals.
[Reproduction, file identities and checkpoint guide](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_continued/README.md).
