# K5 bottom normalization: bounded completion experiment

## Decision

Follow-up: [the all-QP source-support audit](stage_K5_bottom_source_support_report.md)
finds an in-box missing-source wedge beyond the finite bottom tip. This is
tracked separately from denominator completion; the hotspot remains unresolved.
The subsequent [paired continuation test](stage_K5_bottom_source_continuation_report.md)
restores that source and reduces the bottom error to the same-resolution
interior scale, without claiming an exactly stress-free discrete solution.

Boundary truncation of I_h contributes to the bottom hotspot, but completing
normalization **does not remove it or restore the interior accuracy level**.
The one authorized four-rank run completed initialization and exactly two real
steps in **112.52 s**, with unchanged physical inputs and solver safeguards.
At the final state the bottom-layer strain mismatch fell from **14.11% to
10.03%** of crack-strain RMS, versus **1.37%** at the interior control.
Absolute strain mismatch decreased 41.3%, and the largest raw bottom normal
traction excursion decreased about 49%. Reference-velocity error, however,
increased slightly in the first 200 m and substantially in the adjacent layer.

Recommendation: retain completion as an **opt-in diagnostic**, not the default
normalization policy. This is evidence for a truncation contribution, not a
complete boundary treatment. The remaining open-tip/bulk compatibility error
has not been uniquely attributed. No retry, refinement, additional timestep,
support change, or further production correction was made.

## Definition in the actual weak projection

The production three-point Gauss rule provides profile origins s_q, weights
`w_q = element_length * Gauss_weight`, and Q1 shapes N_i(s_q). Keep

\[
 M_{ij}=\sum_q w_q N_i(s_q)N_j(s_q),\qquad
 b_i=\sum_q w_q N_i(s_q)
       [I_q^{\rm box}+I_q^{\rm outside}],\qquad M\widehat I_h=b.
\]

The new opt-in hook adds the outside contribution to the **unprojected profile
integral**, before the existing MPI reduction/Q1 projection. It changes neither
M nor profile weights. No endpoint coefficient is doubled, no nodal correction
is prescribed, and no physical cell, mechanical load, or outside stress history
is added. All constitutive uses of chi employ the resulting completed I_h.

With the actual straight-fault normal n=(-sqrt(3)/2,1/2), an origin at height y
loses the interval r<-y/n_y=-2y below the box. Only that interval is completed.
The top receives no addition. A complete physical column receives none, even
though inversion of the consistent mass matrix can propagate small nodal
changes beyond the directly affected profiles.

### Same-resolution continuation, not an analytic normalization replacement

The saved bottom strip has uniform cell spacing **97.65625 m**. Extend that
Cartesian Q1 lattice below y=0. At each virtual lattice vertex evaluate the
same prescribed stationary distance-profile table that initialized the physical
Q1 field, then bilinearly interpolate phi and evaluate the unchanged degradation.
Both material phases share this degradation in the fixture. This constructs
the bottom-resolution FE continuation; it does not evaluate an analytic h
directly at the integration points or substitute the analytic 13054-m integral.

The implementation splits rays at virtual grid lines, adaptively compares
Gauss subpanels, and independently checks order 8 against order 16. The search
enclosure is the actual nodal profile radius plus the cell's projected normal
diameter. Its integrand is zero beyond the extended Q1 support; this bound is
not a tuned completion width. Only **14** actual surface profiles have a nonzero
outside integral; their maximum height is **389.556857 m**. The nodal stationary
radius is 790.583280 m, so the geometric half-height is 395.291640 m. Q1 supports
and the projection may extend the footprint without a hardcoded cutoff.

Focused checks:

- Virtual Q1 versus every saved active bulk-QP phase sample below y=2000 m:
  maximum absolute difference **1.67e-15**.
- Outside-integral order comparison: maximum difference **8.19e-12 m**.
- Complementary inside/outside integration of the virtual field closes within
  **3.64e-12 m**.
- A fixed complete FE reference column has I=**12691.748895 m** independent of
  the imposed split height; maximum split error is **3.64e-12 m**. The actual
  nonuniform-grid Q1 mass projection reproduces a constant full integral within
  **1.10e-11 m**.
- Actual complete 2-D Q1 columns at different grid phases vary approximately
  **12689.6–12697.7 m** in the sampled checks. This small representation variation
  is retained, not flattened into an analytic constant.
- Reassembling the actual weighted **inside-only** projection reproduces the
  saved baseline nodal I_h within **1.64e-11 m**. Reassembling the augmented RHS
  reproduces the candidate equally closely. Thus the unchanged physical
  integrals and the proposed additive weak correction are separately verified.

There is one **failed precision check**, not hidden by a new tolerance: the
unchanged production remote integral plus completion differs from independently
cell-split integration of the virtual full column by up to **0.106017 m**
(about 8.4e-6 relative). The diagnostic's 1e-5-m absolute comparison fails.
This is consistent in scale with the previously documented remote-versus-cell
discrepancy in `stage_K5_first_cycle_preparation.md`; it is not a fresh claim of
quadrature-level equivalence. The in-box production integration and its
configured tolerances were not altered. No additional backend investigation
was undertaken. The much larger measured mechanical changes remain informative,
but this check must not be described as passing.

## One matched run and lifecycle checks

The candidate reuses `uniform-sliding-50-local4`: 42,880 cells, 1,236 vertices,
mature mode, all V=Vp=1e-9 m/s, zero initial retained perturbation stress, the
same fixed background, pressure and boundary treatment, phase profile, physical
support, Theta and H initialization. It uses the same artificial dt0=4e6 s and
the same accepted real steps:

| Step | Physical time (s) | dt (s) | Final relative bulk residual |
|---|---:|---:|---:|
| 0 | 0 | artificial initialization only | 5.34127e-10 |
| 1 | 2487214.2056652424 | 2487214.2056652424 | 6.81097e-10 |
| 2 | 4970143.2669182401 | 2482929.0612529977 | 8.83372e-10 |

All six fresh linear checks pass, worst fresh/target=0.883897. Every V remains
exactly prescribed; no free/lower-active nodes exist. A zero constrained surface
residual is therefore not a claim of zero unreplaced frictional reaction.
Independent Theta/slip checks pass; C stays zero; all **385,920** stable particle
IDs retain initialized H exactly. Initial particle stress and working FE stress
history are zero. Geometry and completed I_h remain fixed across accepted
states. Later preparations hit the completed-value cache. The top-half nodal
I_h is bitwise unchanged from baseline. Physics and history rules are unchanged;
subsequent committed Maxwell stress can naturally differ with the changed solve.

## Comparison at the common final state

Strain mismatch is the JxW-weighted RMS of the full tensor
`sym grad u - chi Vp S` at actual associated Stokes QPs. The denominator below
is each run's own crack-strain RMS, so the absolute change is also reported.

| Region | Baseline mismatch/crack | Completed mismatch/crack | Absolute mismatch change | Velocity-reference RMS/Vp, baseline -> completed |
|---|---:|---:|---:|---:|
| Bottom 0–200 m | 14.1086% | 10.0313% | -41.34% | 1.6738% -> 1.7744% |
| Bottom 200–1000 m | 2.8056% | 2.7048% | -3.82% | 0.4595% -> 0.9309% |
| Bottom 1000–2000 m | 1.3840% | 1.5263% | +10.28% | 0.3316% -> 0.5358% |
| Interior xd=59–61 km | 1.36668% | 1.36675% | +0.00535% | 0.33233% -> 0.33259% |

In the first 200 m, absolute mismatch is **1.50020e-13 -> 8.79976e-14 1/s**.
Maximum bulk-QP chi falls **0.004578617 -> 0.002742291 1/m**. The bottom vertex
I_h increases **6647.462047 -> 12691.226484 m**. The change is not an artificial
flattening: completed columns and their actual Q1 projection are retained.

Projection spillover is small and measured: maximum nodal I_h changes above
y=400,500,1000,2000 m are respectively **0.298948, 0.080103, 2.96e-5,
3.64e-12 m**. No geometrically complete profile gets a virtual integral.

Physical integration is still clipped. At the first three surface Gauss
origins, y=9.756,43.284,76.812 m, the physical integrals of chi are respectively
**0.55125, 0.71393, 0.82981** with the actual completed Q1 denominator.
They are intentionally not renormalized back to one. Small full-column ratios
departing from one are retained Q1 projection/representation effects.

### Raw stress and weak traction are different measurements

| Final-state quantity | Baseline | Completed |
|---|---:|---:|
| Bulk-QP delta p range, y<200 m | -93.9193 to +20.8797 kPa | -56.7971 to +47.0071 kPa |
| Actual raw surface sigma_n, bottom 2 km | 49.831855–50.098761 MPa | 49.922826–50.085605 MPa |
| Positive-test-weight surface sigma_n means | 49.956224–50.002017 MPa | 49.977194–49.999836 MPa |
| Consistent Q1 surface sigma_n coefficients | 49.934109–50.002317 MPa | 49.971916–49.999835 MPa |

Raw surface pressure at the completed minimum is -38.796755 kPa, while
`-delta tau:N=-38.377433 kPa`; their sum gives **49.922825812 MPa** including
the unchanged 50-MPa background. This sample maps to the bottom endpoint and
has a parent at y=113.933948 m. At the positive extreme, surface y=7.641244 m,
the two contributions are +34.798525 and +50.806780 kPa, giving
**50.085605305 MPa**. Pressure alone does not determine normal traction.

The raw bottom normal range narrows by approximately **39.0%**, and the largest
absolute departure from 50 MPa decreases by **49.1%**. Initialization already
shows a partial reduction: raw range **49.853258–50.075405 ->
49.927241–50.066860 MPa**. The unchanged top still dominates some global
extrema, so global min/max alone would conceal the bottom improvement.

Raw surface samples use production particle/domain quadrature and the history
retained during mechanics. Bulk stress diagnostics use the accepted constrained
FE working history, not newly committed particles. The candidate observer uses
the corrected raw symmetric-strain formula; the baseline's documented
diagnostic-only trace correction remains offline. Raw pressure ranges above
are explicitly bulk-QP ranges, not claimed extrema over every surface sample;
surface exports retain selected normal extrema and their pressure decompositions.

## Files, checks and limits

Implementation: `source/material_model/phase_field_fault.cc` contains the opt-in
preprojection addition. Ordinary behavior is unchanged when
`ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC` is unset. The diagnostic is restricted
to fresh, frozen, mature uniform sliding; restart is rejected. Its supplied
profile data are fixed during the process. The two authoritative documents
record this bounded exception, not a policy promotion.

Benchmark tools: `bottom_completion.py`, `run_bottom_completion.py`,
`analyze_bottom_completion.py`; the existing uniform analyzer is reused and now
labels corrected-versus-original observers accurately. No baseline simulation
artifacts or unrelated working-tree edits were replaced.

```sh
python3 benchmarks/reconstructed_fault/bp3/bottom_completion.py
cmake --build build-pf-cpdi --target aspect.exe.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_bottom_completion.py
MPLCONFIGDIR=/tmp/bottom-completion-mpl python3 benchmarks/reconstructed_fault/bp3/analyze_bottom_completion.py
```

The prepare/run commands refuse existing evidence; do not rerun them into this
directory. The single MPI run has a 600-s cap and no retry. Release build,
Python compilation, the retained legacy boundary regression, and the analytic
adaptive-kernel test pass. Completion geometry/quadrature/projection and lifecycle
checks pass **except** the explicitly recorded remote/full-column precision
comparison. There was no new one-/two-rank or restart campaign.

Evidence: `benchmarks/reconstructed_fault/bp3/bottom-completion-50-local4/` contains
the preflight, completion table, command/source hashes, resolved input, log and
ordinary outputs. Under `analysis/`, see `comparison.png`, `comparison.csv`,
`completion_verification.json`, `profile_completion_check.csv`,
`physical_profile_fraction.csv`, `uniform_normal_summary.csv` and the original
analyzer's bulk/traction diagnostics. This is the review point; ordinary
normalization remains unchanged.
