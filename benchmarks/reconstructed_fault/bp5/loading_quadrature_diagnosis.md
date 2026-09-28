# Saved-initialization integration audit

## Decision

**Increasing the bulk coupling quadrature does not remove the observed
moment variation.** Accurate integration preserves the row-mass tooth RMS
and slightly increases the second-moment tooth RMS. Consequently no production
quadrature correction, bulk refinement, or new mechanical solve was performed.

The additional profile comparison identifies a more specific issue: the saved
Q1 phase profile has short-scale along-fault column variations, and the
three-point-per-element **surface projection of I_h** transfers those variations
into the much coarser fault representation. This denominator pattern dominates
the native localization-moment teeth. It is distinct from insufficient bulk-QP
integration of an already fixed chi field.

The next justified correction is to resolve the tangential integration in the
consistent Q1 I_h projection, not smooth its coefficients or replace the
normalization by a constant. That is a different correction from the proposed
bulk coupling overintegration and is presented for review. The requested
one-initialization validation remains pending that choice. The conditional
24.4-to-12.2-m bulk-accommodation test is not yet justified: the localization
moments themselves are demonstrably nonuniform on a mechanically relevant scale.

## What was reintegrated

Reference: accepted initialization of
`weakening30-dc010-ell100/loading-startup/startup/`. No checkpoint was resumed.
The 27–29.8 km window, actual 100-m fault basis, ell=100 m, initial-state ratio,
true-normal-stress solution and all physical inputs remain unchanged.

For each of 2607 relevant cells, the script reads **saved nodal Q1 phase
coefficients** from `phase_cells_rank*.csv`. It does not interpolate an
analytical profile onto another mesh. Those polynomials reproduce the saved
bulk Gauss samples to 2.41e-14. The saved Q1 I_h comes from the initial fault
VTU. Applying the configured degradation law and saved straight-fault mapping
reproduces the production chi samples to 5.21e-18 /m. Native row masses are
independently reproduced within 1e-12 relative tolerance.

The integration domain retains the existing normal support half-width
197.645820100527 m. There are no endpoints in this window. Cells are split at
the normal support boundary and every crossed fault knot; the resulting 3456
cell/basis polygons are triangulated and integrated with positive Duffy-mapped
tensor Gauss rules. Thus neither a bulk-cell boundary nor a fault-basis kink
is hidden inside an integration panel. MPI-owned cell IDs are unique and all
positive-phase cells touching the selected row supports are accounted for.

The quantities are

\[
 m_i=\int\chi N_i\,dA,\qquad
 d_i=\int\chi^2 N_i\,dA,\qquad c_i=d_i/m_i.
\]

Units are m, dimensionless, and 1/m respectively. The chord diagnostic is the
previous report's `z_i-(z_(i-1)+z_(i+1))/2`, with its original native row weights
for RMS calculations. It is not a smoothing operation or a modified field.

## Quadrature convergence: the variation survives

Maximum relative error across both moments and all 29 diagnostic nodes,
relative to the cell/basis-split order-16 result:

| Rule | Maximum relative moment error |
|---|---:|
| Production tensor Gauss-3 | 9.91446e-5 |
| Gauss-3, 2 subdivisions per coordinate | 1.70352e-5 |
| Gauss-3, 4 subdivisions per coordinate | 1.47038e-6 |
| Gauss-3, 8 subdivisions per coordinate | 1.21238e-7 |
| Cell/basis-split order 3 | 1.75426e-6 |
| Cell/basis-split order 6 | 1.25822e-12 |
| Cell/basis-split order 10 | 9.55e-15 |

There is measurable production quadrature error, but it is not the dominant
cause of the variation:

| Chord RMS | Production | Accurate split integration | Change |
|---|---:|---:|---:|
| m_i | 0.0128400 m | 0.0128874 m | +0.369% |
| d_i | 1.77824e-4 | 1.83954e-4 | +3.447% |
| d_i/m_i | 9.02923e-7 /m | 9.57236e-7 /m | +6.015% |

Accurate peak-to-peak variation relative to each mean is 0.02000% for m_i,
0.04146% for d_i, and 0.02150% for d_i/m_i. A smaller maximum excursion under
some rules must not be confused with removal of the alternating structure:
the RMS does not contract.

For scale, at uniform target Vp=1e-9 m/s the localized shear contribution alone
has chord RMS `kappa*Vp*chord_RMS(d_i/m_i)` of approximately **30.7 Pa**.
The saved net mechanical shear tooth RMS is 9.05 Pa. This is not a prediction
of a corrected coupled solution, but shows why a 0.02% localization variation
cannot simply be declared mechanically negligible.

## Saved FE profile versus the intended uniform profile

The intended distance profile is taken from the saved production
`stationary_profile.csv`, independently checked against its generating law.
It is used **only as a comparison**, never substituted in the main
reintegration. Its ridge cusp is split explicitly. Order-16/32 checks leave
approximately 1e-9 relative spatial variation in the nominally uniform
comparison, far below the measured FE/localization variations.

Cell-split normal-ray integration of the actual saved field, at 2-m along-fault
spacing, gives:

- ridge phase: **0.587912–0.599711**, versus intended 0.6;
- full column integral J(s): **12688.462–12697.737 m**;
- J peak-to-peak / mean: **0.07307%**;
- saved nodal I_h in the window: **12689.215–12695.983 m**;
- maximum sampled `abs(J/I_hat-1)`: **6.24650e-4**.

Normal-ray order 8 versus 12 differs by at most 4.78e-14 relative. Full ray
endpoints lie in zero phase. This column ratio is an offline **full-profile**
diagnostic; it is not the row-mass error, omitted-support fraction or a new
acceptance threshold.

At the same constant denominator, the FE profile's mean supported first and
second moments are respectively **2.77085% and 7.65502% lower** than the intended
uniform profile's moments. Thus the represented profile has both a mean
transverse discretization error and an along-fault modulation.

## Separate numerator and denominator effects

These are offline attribution calculations, not proposed replacements for
the production normalization:

| Profile / denominator | m_i chord RMS (m) | d_i chord RMS | c_i chord RMS (1/m) |
|---|---:|---:|---:|
| Saved FE / saved Q1 I_h | 1.28874e-2 | 1.83954e-4 | 9.57236e-7 |
| Saved FE / constant mean I_h | 3.39952e-4 | 1.21497e-5 | 9.82279e-8 |
| Intended uniform / saved Q1 I_h | 1.29053e-2 | 1.86057e-4 | 9.04530e-7 |
| Intended uniform / constant mean I_h | 1.06e-8 | 2.17e-10 | 1.38e-12 |

Holding only the denominator constant reduces the moment chord RMS by
**97.36%, 93.40%, and 89.74%** respectively. Keeping the saved denominator
while replacing only the numerator by the intended uniform profile retains
nearly all the tooth pattern. Chord correlations with the negative weakly
averaged I_h are 0.9999996, 0.9999960 and 0.9999891.

This isolates the dominant *moment* mechanism. It does not prove that a
constant denominator would preserve pointwise normalization or produce an
acceptable mechanical solution. No such change was applied.

## Why the I_h projection is the next target

Current `build_owned_normalization_profiles()` uses three surface Gauss points
per fault element. Re-evaluating the full saved FE columns at those same
coordinates approximately reproduces the saved projection RHS `M I_h`:
maximum relative discrepancy **1.60e-6**. That small remaining normal-profile
evaluation discrepancy is retained, not claimed to be roundoff reproduction.

An independent, full-profile cell/basis integral gives the more accurate
`b_i = integral N_i J(s) ds` without the three-point tangential sampling.
Normalized RHS peak-to-peak variation is:

- saved `M I_h` / geometric row mass: **195.46 ppm**;
- accurately integrated full-profile RHS / row mass: **5.13 ppm**.

The re-evaluated three-point RHS differs from the accurate RHS by up to
**108.71 ppm**. This difference is much larger than the 1.60-ppm mismatch with
the saved RHS, and the plotted coarse pattern is reproduced. The full-profile
cell integration is also checked with orders 10 and 16. Consequently the
evidence supports **tangential underintegration/aliasing in the I_h projection**
as the dominant source of these localization-moment teeth. It is not evidence
that the normal rays themselves need tighter quadrature.

## Scope decision and proposed correction

Do not increase only the surface diagonal, change plotted traction, smooth
I_h, or run a bulk refinement as if the source were already uniform.

The targeted next correction should retain the same consistent-Q1 projection
and full normal integrals, but converge its tangential RHS quadrature against
the FE/grid-scale column variation. Reuse the existing ray/cell geometry and
completed-value caching. Endpoint-completion inputs must correspond to the
chosen profile sampling; silently reusing three-point endpoint data with a
different rule would be inconsistent.

Only after that correction is selected should the requested single
zero-history, true-normal-stress initialization be run. The newly computed
I_h must reach the bulk source, residual, B/G/K and native work operators
through their common constitutive localization path. Compare shear/velocity
teeth, broad response, independent weak friction and runtime. A constant-I_h
initialization would not be an acceptable substitute for this test.

This task has changed **no production source, equation, support, normalization
value, solver setting or initial state**. There is therefore no new mechanical
equivalence/runtime result to report. The prior 9.05-Pa/1.31195e-13-m/s
initialization teeth remain the mechanical baseline, not a claimed corrected
result. All evidence here is offline; no architecture redesign or integration
campaign was started.

## Reproduction and files

```sh
python3 benchmarks/reconstructed_fault/bp5/reintegrate_initial_profile.py
```

The script's final integration/diagnostic pass takes about **11 seconds**
before plotting. It reuses cell/basis polygons between quadrature levels.
It verifies source/sample reproduction, native row masses, complete positive
cell coverage, unique MPI cell ownership, full-ray endpoint coverage, and the
independent order comparisons. It is deliberately restricted to the saved
2-D axis-aligned straight-fault fixture, not a new production integration API.

Outputs under `weakening30-dc010-ell100/coupling-quadrature/`:

- `moments.json`, per-rule moment CSVs;
- `moments.png`, `profile-and-projection.png`;
- `columns.csv`, `transverse_profiles.csv`, `projection_rhs.csv`;
- `inputs.json`: saved-input SHA256 provenance and zero mechanical-solve count.

Only this report and `reintegrate_initial_profile.py` were added for this task,
apart from generated artifacts. Existing unrelated working-tree changes were
preserved. Python syntax checks and `git diff --check` pass; no C++ tests or
mechanical runs are claimed.
