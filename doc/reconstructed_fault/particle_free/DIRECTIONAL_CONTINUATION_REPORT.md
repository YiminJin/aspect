# Structural-direction RSF continuation: bounded result

**Point continuation passed under an explicit new structural-plane law. The
fixed-phase band shows resolved spatial behavior and controlled timestep
sensitivity, but the literal all-metrics monotone-difference gate did not
pass. Propagation has not been enabled.** This is an independent Python
reference; production, native ownership/checkpoints and BP3 are unchanged.
Baseline: `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local work.

[Equations and history contract](DIRECTIONAL_CONTINUATION_FORMULATION.md),
[predeclared tests and measured follow-up](DIRECTIONAL_CONTINUATION_SELECTION.md),
[evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/directional_continuation/),
[archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/directional_continuation.zip).

## Physical choice and directional outcome

An existing fault slides in either sense on its recorded material plane. A
prescribed **new-record** event selects a maximum instantaneous kinetic speed;
a declared geometry preference resolves genuine ties. Existing-plane sliding
no longer maximizes over every other plane at every increment. There is no
phase locking threshold, spin-sign requirement, tensor projection/reset, or
K-gradient energy. Formation of a second plane through an existing record
remains a separate fracture/history problem; it is not hidden in a direction
switch. Fixed geometry transports its normal identically in this small-strain
reference. This is a changed constitutive admissibility rule, not a numerical
repair of Feng's original global endpoint condition.

At virgin fault history, tau_xy=1100 Pa and p=10000 Pa, two tied plane normals
are 3.1546° and 86.8454° from +x. The speed is 4.13523e-5 m/s. Preferences near
vertical/horizontal select different legitimate structural records. Without
preference the result remains explicitly unresolved. Both continued paths
have identical shear response and opposite normal-stress responses. Finite
phase .2/.8/.99 and analytic maturity all admit loading and reversal. Other
planes exceed the sliding-plane kinetic force by as much as 598.08 Pa in the
combined controls: those are competing-fracture diagnostics under the new law,
not covertly accepted roots of the old law.

Actual recorded-plane contact loss and unassigned nonzero fault histories are
rejected. No tensors are erased to create a direction. The scalar update has
a checked sufficient monotonicity bound; a failure of that bound would require
branch analysis, not a claim of nonexistence. No such failure occurred.

## Point evidence

Three steps (.0125/.00625/.003125 s) cover .30 s with shear reversal at .15 s,
both symmetry preferences, finite/mature damage, varying pressure, and a
nonproportional drive. A separate loaded nonproportional tensor tests retained
memory and the full mechanical/pressure tangent. An actual loaded-record
admission is also continued (without resetting its T=1100 Pa shear history);
its worst finer-pair stress difference is .03309%. All contractions include zz
and both xy entries.

| Check | Measured result |
|---|---:|
| Finer-pair full-tensor stress relative L2, worst case | 0.8633% |
| Finer-pair signed-slip relative L2, worst case | 0.6614% |
| Complete stress/pressure tangent FD relative discrepancy | 6.79e-11 |
| Nonzero pressure tangent norm | 0.10315 |
| Passive basis covariance discrepancy | 1.02e-15 |
| Independent actual-plane formation maximum residual | 1.90e-13 Pa |
| Largest point rate-equation residual | 9.99e-10 Pa (bound 1e-8 Pa) |
| Largest BE energy-ledger defect | 1.27e-13 J/m³ |
| Perpendicular old-memory Maxwell-update error | 3.88e-20 |
| Forced rejection/retry and serialized restart | bitwise equal |
| Duplicate publication / opening / unassigned memory | rejected |

These lifecycle checks belong to the independent point owner, **not** the
native simulator. No finite-rotation/objective-rate claim follows from passive
basis covariance. The nonsymmetric pressure tangent is retained; the old
symmetric potential-Hessian certificate is not reused.

![Point choices and continuation](../../../benchmarks/reconstructed_fault/particle_free/evidence/directional_continuation/point_directions.png)

*Lines in the first panel are candidate physical planes; arrows are normals,
not crack fronts. The preferences choose mirrored responses. Loading reverses
at the dotted time. A competing-plane excess does not rotate the old record.*

## Equilibrated fixed-phase band

| Quantity | Fixed value |
|---|---|
| G; eta; Gc; ell; m | 1e6 Pa; 1e8 Pa s; 1 J/m²; .046875 m; 4 |
| p0; residual cohesion | 10000 Pa; 0 Pa |
| mu0; a; b; V0; Dc | .1; .005; .01; 5e-6 m/s; .001 m |
| rho; radiation coefficient | 2670 kg/m³; 25836.02 Pa s/m |
| Cross-band size and profile | .25 m; phi=(sqrt(.8)-abs(y)/(2ell))_+² |
| Fresh history | T=A=K=0, Theta=1, horizontal material plane |
| Relative boundary velocity | .001 m/s to .25 s; displacement hold to .30 s |
| Cells / final common timestep | 64, 128, 256 / .000390625 s |

The exact 1-D mixed reduction solves uniform shear traction and displacement
compatibility jointly with local RSF/state updates. Constant normal pressure
is the equilibrium solution of this symmetric fresh-history case, not a
pressure lag. A mass-lumped P1 profile/history layout is used; no native
2-D mechanical solver or oblique perturbation is represented by this test.
Initial phase relative L2 errors are .08808%, .02216%, .005539%; normalization
errors are .02734%, .006792%, .001698%, respectively. Every run initializes
freshly from the same analytic profile.

The original .00078125/.000390625 s pair missed the declared .25% slip-time
criterion: **.3185%**. The permitted further halving gave **.1583%**, so the
two coarser meshes were matched to .000390625 s. Original results remain in
the CSVs. The following uses common times .15–.30 s, with maximum absolute
differences normalized by the reference maximum over that interval:

| Measure | 64→128 cells | 128→256 cells | Finest .000390625→.0001953125 s |
|---|---:|---:|---:|
| Traction | 6.44e-7% | 3.61e-7% | .0002865% |
| Integrated slip | .0005391% | .0003019% | .1583% |
| Endpoint work | 7.27e-10% | 4.47e-9% | .07790% |
| Rate participation width | .10373% | .026137% | .0009591% |

Spatial differences meet the 1% magnitude criterion. Slip and width differences
decrease; temporal errors meet the separately stated criteria at the selected
step. **The strict decreasing-work-difference criterion remains false:** its
absolute differences are only about 9.1e-13 and 5.6e-12 J/m², below the retained
1e-10 J/m² per-step balance-check scale. No tolerance is changed, and no work
convergence order is claimed below that scale. The original JSON all-checks
flag remains false. This is unresolved numerical trend evidence, not a
physical bifurcation or a demonstrated constitutive failure.

At .30 s on 256 cells, traction=997.05653 Pa, slip=.298427 µm,
interface speed=4.65406 µm/s, and rate width=92.6153 mm (not 0.14 m as estimated
in an intermediate update). Widths 92.7326/92.6390/92.6153 mm do not collapse
with cell size. Loading ends at .25 s with slip=.0616295 µm; continued sliding
in the hold consumes stored stress, rather than implying zero stress when
boundary motion stops.

Final energy per fault area (J/m²): work .1250875789; bulk storage .1242652155;
cohesive storage 5.98362e-8; Maxwell loss .0003328582; friction .0002946818;
radiation 3.28813e-8; BE remainder .0001947307. Initial/final fracture storage
.7155539021 is unchanged. The largest band ledger defect is 6.44e-14 J/m²;
compatibility defect <1e-16 m; rate residual <1e-9 Pa. All three meshes use
at most three scalar mechanical Newton iterations per step. The seven
sequential band runs take 2.13 s of measured solve/recording time in total;
point checks take about 1 s. These are inexpensive independent references,
not native/server performance estimates.

![Band response](../../../benchmarks/reconstructed_fault/particle_free/evidence/directional_continuation/band_response.png)

*The boundary motion is imposed; traction, local rates, state and histories are
solved in mechanical equilibrium. Hold sliding is resolved. The short response
is dominated by elastic storage and samples only .00030 Dc of accumulated slip;
it is not a sustained state-weakening or mature-sliding qualification.*

![Spatial and temporal controls](../../../benchmarks/reconstructed_fault/particle_free/evidence/directional_continuation/band_resolution.png)

## Decision and remaining work

The tested new structural-plane closure has finite-damage continuation and
separates legitimate symmetry breaking from unsupported history/contact states.
The fixed-profile band has useful three-mesh accuracy evidence with independently
controlled temporal uncertainty. Do not label every predeclared gate passed,
or claim that this one-dimensional, weak-sliding result regularizes arbitrary
oblique disturbances or establishes propagation readiness.

The smallest next qualification is still fixed phase: retain the declared
structural-plane hypothesis, test the full mixed oblique response with its
pressure tangent, and extend a small equilibrated band far enough to exercise
state weakening (a stated fraction of Dc), with the same separate temporal
control. Formation/secondary-plane energy and state admission must be specified
before a phase-active propagation test. No such extension was started here.

Changed files: four new independent Python files `structural_rsf.py`,
`check_structural_rsf.py`, `structural_rsf_band.py`, `plot_structural_rsf.py`;
this report, formulation and selection; compact evidence/figures/archive; and
prepended status/source-audit/current-status entries. All prior source,
production defaults, BP3 and original failed evidence are preserved.
