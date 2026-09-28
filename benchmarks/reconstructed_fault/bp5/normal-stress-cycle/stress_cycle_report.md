# Frozen BP5 stress-history cycle: server-results analysis

## Decision

The large stress bands are already present in the incoming history, not created
by this one current-step update. The stored particle tensor is itself strongly
nonconstant within cells. Distance-weighted transfer and continuous Q2 evaluation
substantially filter and reorganize that structure. There is no demonstrated
component/index, wrong-time-level, or wrong-constitutive-timestep defect in the
captured cycle.

Reducing the timestep by four does **not** reduce the approximately 1 kPa normal
stress increment by four: the solved strain rate increases approximately as
1/dt. This supports discrete re-equilibration of the fixed retained history as
the dominant response in these windows. It does not establish the origin of
the stored bands or justify changing the interpolator.

Next recommended experiment: one small clean-start transfer/update regression,
with zero or prescribed smooth incoming stress, tracking the first generation
of within-cell tensor structure and its transferred weak load. Do not launch
another full trajectory or replace the production transfer on this evidence.

## Inputs and qualification

These are three independent one-step trials from accepted step 5612 at
5310111071.5634108 s, not three sequential steps or three trajectories over a
common elapsed interval. Each disposable branch commits once after acceptance;
that new history is never an input to another branch. Particle advection is
frozen. The copied server runs used 32 ranks.

| Branch | Constitutive and particle-update dt (s) | Final relative bulk residual | Final relative surface residual |
|---|---:|---:|---:|
| full | 0.0028219241006433027 | 1.451509e-11 | 8.427786e-11 |
| half | 0.0014109620503216513 | 1.481271e-12 | 5.017483e-12 |
| quarter | 0.00070548102516082567 | 9.869709e-13 | 4.398315e-13 |

All printed fresh linear checks pass their requested targets. Each log contains
six such records, including repeated checks, not six distinct Newton directions.
All accepted line-search alphas are one. Diagnostic wall times are respectively
44.70, 44.20, and 44.35 s; these are not a new local performance measurement.

Resolved parameters differ only in output directory and timestep fraction.
The incoming global inventory (1,034,856 particles, identical hash), restored
fault data, native-QP geometry, phase, I_h, localization, and old FE stress agree
across branches. Geometry and I_h change diagnostics are zero. The 387 traced
parents in 43 cells retain their IDs and positions.

Use `actual_dt`/`stress_dt`, **not subtraction of the approximately 5.31e9-s
absolute timestamps**, to recover these millisecond increments. Timestamp
subtraction loses significant digits but does not change the timestep used in
the constitutive law.

## Transfer and update audit

The implemented split, without rotation, is

\[
\tau_{\rm mech}=\beta\tau^{old}_{\rm working\ FE}
 +2\kappa\operatorname{sym}\nabla u-2\kappa\chi V S,
\qquad d=-n^T\tau n.
\]

Particle publication uses the accepted gradient at the parent position and
that parent's retained old tensor, not the old FE tensor. This distinction is
the existing architecture; the two representations are not interchangeable.

All tensors were traced as xx, yy, xy (no engineering-shear factor). Captured
property indices are 3,4,5 and FE components 4,5,6 respectively. The actual
support-point proposals were independently recomputed from the saved neighboring
particle stencil, deduplicating ghost IDs. Published Q2 fields were recovered
from their nine native Gauss values to evaluate the **same represented FE
polynomial** at parent locations; this is not smoothing or a fabricated
centerline sample.

| Check | Maximum discrepancy |
|---|---:|
| Actual DWA support proposal versus independently recomputed proposal | 1.572e-9 Pa |
| Assembled Q2 nodal value versus complete incident-cell proposal mean (282 checks) | 1.863e-9 Pa |
| Recovered Q2 polynomial at native Gauss points | 9.313e-10 Pa |
| Published FE history versus constrained working history in traced cells | 0 Pa |
| Actual particle-update old tensor versus stored incoming tensor | 0 Pa |
| Recomputed full particle Maxwell update versus recorded candidate | 0 Pa |
| Recorded candidate versus committed particle tensor | 0 Pa |
| Mechanics tensor versus inherited + strain + slip parts | 8.467e-10 Pa |

Incomplete incident-cell sets at the extract boundary were excluded from the
nodal-averaging check. The localized-slip tensor has normalized normal-normal
contraction at most 6.34e-17; its direct normal contribution is roundoff, not a
source of the bands. Recomputing d_update from the actual velocity gradient and
kappa agrees within 7.28e-12 Pa.

Existing actual-transfer manufactured tests are reused, not rerun: constants
are preserved to 5.33e-15 with DWA; unlimited least squares reproduces the
affine tensor to 5.33e-15. Published/constrained fields and one-/two-rank results
agree. DWA's affine error is 0.427254 in arbitrary test units, including boundary
stencils; it is not an estimate of this BP5 interior error. See
`../stress_transfer_verification.json` and `../stress_cycle_README.md`.

## Where the structure resides

Native quadrature statistics below use the exported production work weights.
They do not use the 50 MPa background as a normalization scale.

| Window | Incoming d RMS (MPa) | Current-update d RMS (Pa), full | half | quarter |
|---|---:|---:|---:|---:|
| 70–71 km | 1.298081 | 951.030194 | 951.030472 | 951.030612 |
| 79–80 km | 1.371136 | 1012.443892 | 1012.443855 | 1012.443864 |

The small parent patches show much larger variation **before transfer**:

| Window | Stored-parent d RMS (MPa) | FE old d at identical parent positions (MPa RMS) | FE minus parent (MPa RMS) |
|---|---:|---:|---:|
| 70–71 km | 8.007992 | 1.242357 | 7.751637 |
| 79–80 km | 8.044300 | 1.269665 | 7.783750 |

This table uses equal parent weights, not work weights or domain volumes; it
must not be subtracted from the preceding native-QP table. The actual tensor
components have large cell-scale curvature: particle-row chord-defect RMS for
xx/yy/xy is 7.72/13.86/5.48 MPa in the first patch and 8.12/13.67/5.83 MPa in
the second. Thus a smooth stored tensor becoming rough only upon FE transfer
is not a description of these saved states.

The two-band signature already exists in the incoming FE field. Among actual
native points with |r| <= 2 m, group means of inherited d are:

| Window | Reference x = 0.112702 | Reference x = 0.5 | Reference x = 0.887298 |
|---|---:|---:|---:|
| 70–71 km | -0.539 MPa | -2.795 MPa | -0.460 MPa |
| 79–80 km | -0.420 MPa | -2.905 MPa | -0.414 MPa |

Mean offsets of these groups range from -0.201 to +0.095 m; all individual
samples remain in the same +/-2 m strip. Their mean updates are only tens to
hundreds of Pa. This controls the large transverse-offset confound, but is not
an exactly r=0 line or a pointwise pairing of identical down-dip positions.

Stored and transferred tensors use different spatial representations. The same
linear transfer weights apply to every component, so normal contraction with
this constant normal commutes with transfer. This is not evidence of an
algebraic tensor-cancellation or component-ordering bug. Their spatial filtering
and evaluation locations nevertheless materially change the represented normal
stress pattern.

## Timestep discrimination

The printed beta is one in all branches; kappa is respectively
9.040914387e7, 4.520457194e7, and 2.260228597e7 Pa s. The following quantities
are independently calculated from the exported gradients, not inferred by
dividing the stress increment:

| Window | RMS epsilon_nn, full / half / quarter (1/s) | RMS dt epsilon_nn |
|---|---|---:|
| 70–71 km | 5.25959e-6 / 1.05192e-5 / 2.10384e-5 | approximately 1.48422e-8 in all three |
| 79–80 km | 5.59923e-6 / 1.11985e-5 / 2.23969e-5 | approximately 1.58006e-8 in all three |

Pointwise differences, evaluated at identical native QPs with common work weights:

| Window | d_update difference full→half / half→quarter (Pa RMS) | Pressure difference (Pa RMS) | Total normal-traction difference (Pa RMS) |
|---|---|---|---|
| 70–71 km | 0.048915 / 0.024515 | 0.727513 / 0.364549 | 0.683969 / 0.342725 |
| 79–80 km | 0.276276 / 0.138434 | 0.520006 / 0.260569 | 0.795963 / 0.398843 |

The small differences contract by approximately two, while the leading ~1 kPa
update remains. This is consistent with a displacement-like equilibrium
correction of fixed history: kappa is proportional to dt while the solved
strain rate is approximately inversely proportional to dt. It is not evidence
that a larger-than-requested dt entered the update. These different-duration
single-step solves do not establish trajectory convergence or predict how that
correction accumulates over many accepted steps.

The unresolved particle-minus-FE normal-history component is positively
correlated with the particle increment (0.382 and 0.273 in the two patches).
That is a reason to test generation/retention of subcell stress in a clean-start
cycle, not proof that this mechanism originally generated all of the bands.

## Reproduction and artifacts

No ASPECT source or plugin code was changed and no new simulation was run for
this analysis. Original server outputs are unchanged. Run from the repository:

```sh
python3 benchmarks/reconstructed_fault/bp5/analyze_stress_cycle.py benchmarks/reconstructed_fault/bp5/normal-stress-cycle
python3 benchmarks/reconstructed_fault/bp5/analyze_stress_cycle_cells.py benchmarks/reconstructed_fault/bp5/normal-stress-cycle
python3 benchmarks/reconstructed_fault/bp5/test_stress_cycle_analysis.py
```

The two offline analysis unit tests pass (Q2 recovery with shuffled support
ordering and distinct statistical measures/nonfinite rejection). The test file
is invoked directly; package-style unittest invocation from the repository
root does not resolve its sibling imports.

Artifacts in this directory:

- `stress_cycle_summary.json`: original lifecycle/convergence checks, actual
  clocks, per-branch native statistics and update closure.
- `cell_cycle_analysis.json`: transfer checks, tensor/normal statistics,
  reference-coordinate groups and matched timestep differences.
- `cell_parent_trace.csv`: stable-ID parent tensors before/after, old FE tensor
  at the same parent point, coordinates and normal contractions.
- `cell_support_trace.csv`: actual and recomputed support proposals and
  assembled FE support values, including component/property indices.
- `cell_QP_trace.csv`: selected actual native QPs with the full recorded tensor
  split, gradients, coordinates, weights and timestep.

These extracts are sufficient to trace this cycle, but cannot identify which
earlier accepted step first generated the stored subcell tensor pattern. No
production interpolation change or new history evolution rule is qualified.
