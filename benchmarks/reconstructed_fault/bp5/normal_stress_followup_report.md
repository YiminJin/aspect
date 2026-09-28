# BP5 pressure/deviatoric follow-up: saved steps 5613–5617

## Finding and limit

The deep fluctuations are present before the consistent mass inverse, and
pressure/deviatoric roughness partly cancels. The mass inverse makes the nodal
roughness more pronounced, but is not its sole source. Both deep zooms use a
uniform bulk mesh; a single-rank control is also oscillatory. An MPI boundary
or local refinement transition is therefore not necessary for this pattern.

**The incoming-history/current-step attribution remains unmeasured.** The
original CSVs exported total constitutive stress, not its terms. A small
first-to-final stress difference is not proof that the fluctuation was already
in incoming history. An output-only constitutive split has now been implemented
and tested, ready for the same bounded restart; no new BP5 solve was launched.
Do not interpret the newly added exporter as an already measured decomposition.

## Data, plots and matching

Input: `output-normal-diagnostic`, supplied server restart from accepted step
5612 at 5310111071.5634108 s. First/final diagnostic states are 5613 and 5617,
at 5310111071.5662327 and 5310111071.577446 s. Their separation is approximately
0.0112133 s; the entire five-step continuation covers approximately 0.0140352 s.
The unchanged state predictor controls all five selected timesteps, approximately
0.00282–0.00279 s. This cannot measure a long-term deep growth rate.

Run:

```bash
python3 benchmarks/reconstructed_fault/bp5/plot_normal_stress_diagnostic.py \
  benchmarks/reconstructed_fault/bp5/output-normal-diagnostic
```

New files go to `output-normal-diagnostic/normal_plots_revised/`, preserving the
old plots and original data:

- `normal_22_35.png`, `normal_60_90.png`, `normal_70_71.png`, `normal_79_80.png`:
  upper panels show p^w, d^w, their sum, and production sigma minus its exported
  reference in **kPa**. No mean or broad trend is removed. First/final upper
  panels use identical limits. Consistent coefficients, f/m, chord departure,
  increments and I_h are separate panels.
- `roughness_scatter_*.png`: pressure versus deviatoric chord departure, colored
  by down-dip position, with the exact-cancellation line d = -p.
- `offset_zoom_STEP_70_71.png` and `offset_zoom_STEP_79_80.png`: actual QPs in
  strips r = -50, 0, +50 m, each within **±1 m**. No normal interpolation or
  curve connecting different offsets. Full cell IDs, rank, QP index, exact r,
  fault segment/xi and stress are in the companion CSVs. C-labels identify cells;
  they do not infer physical cell boundaries. Dotted lines mark actual surface
  element boundaries. Native QP maps and cell sizes are also shown.
- `summary.json`: reproducible closure, roughness, increment and matching values.

All **256,655** exported QPs have unique cell/QP/fault ownership at each state.
The first/final samples match exactly in cell, QP, fault, segment, xi, position,
normal offset and MPI rank. Their work weights differ only by at most
4.44e-16 absolute / 3.51e-14 relative; the script reports this rather than
claiming bitwise identity. Geometry and nodal I_h are unchanged in the saved
accepted-state audits. Total split closure is at most 5.67e-7 Pa, raw split
closure is zero, and global load-sum discrepancies are at most 8.40e-5 Pa m.
These discrepancies are negligible compared with the measured signal.

## Pressure, deviatoric and sum

Final-state chord-departure RMS uses the native row-mass weights. Correlations
below are ordinary correlations of the same chord departures. Chord departure
is a roughness proxy, not a proven numerical error at a physical front.

| Window (km) | p RMS (kPa) | d RMS (kPa) | (p+d) RMS (kPa) | corr(p,d) | p f/m RMS (kPa) | d f/m RMS (kPa) |
|---|---:|---:|---:|---:|---:|---:|
| 22–35 | 5.574 | 13.686 | 14.776 | -0.00019 | 3.341 | 6.390 |
| 60–90 | 95.301 | 150.290 | 95.608 | -0.78646 | 44.293 | 59.683 |
| 70–71 | 95.717 | 167.939 | 108.589 | -0.79529 | 45.219 | 67.239 |
| 79–80 | 85.094 | 116.147 | 90.685 | -0.63545 | 42.923 | 49.624 |

Thus deep pressure and deviatoric fluctuations substantially cancel, but the
sum still oscillates. Across 60–90 km the consistent inverse increases these
roughness measures relative to f/m by factors **2.15 (p)** and **2.52 (d)**.
This is a comparison of representations, not a proposal to replace the mass
inverse with row lumping. The integrated inputs are already nonuniform.

At 79–80 km the final weak pressure ranges 103.651–331.897 kPa, d ranges
216.884–490.951 kPa, and the reference-subtracted normal traction ranges
354.874–621.574 kPa. The latter's minimum is at **79.882976 km**, its maximum
at **79.283194 km**. The pressure maximum/d minimum coincide at 79.383158 km,
illustrating local cancellation rather than a single unexplained total spike.

## Raw deep samples and geometry

Both 70–71 and 79–80 km have square-cell edge **24.4140625 m** and fault
spacing **99.963702 m**. There is no cell-size transition in either zoom.
The 70–71 km window belongs entirely to rank 12. Near 79–80 km the center strip
changes from rank 10 to rank 9 between sampled positions 79.371522 and
79.389204 km; this brackets the observed ownership change, not an exact
geometric MPI-boundary location.

The center strip r=0±1 m has 27 QPs in the control and 34 near the feature.
At the final state:

| Strip | d peak-to-peak (MPa) | (p+d) peak-to-peak (MPa) |
|---|---:|---:|
| 70–71 km, rank 12 | 3.0652 | 3.5901 |
| 79–80 km, rank 10 portion | 3.1600 | 3.5961 |
| 79–80 km, rank 9 portion | 3.1865 | 3.6192 |

These are raw, narrowly matched-offset samples, **not** native weak averages.
The ±1 m offset spread is retained explicitly. Large fluctuations exist on
both sides of the ownership change and in the single-rank control. They cannot
be explained solely by plotting node order or applying the surface mass inverse.
No MPI-partition causal test was done, so MPI effects are not ruled out globally.

I_h varies spatially by about 0.2064 m over a mean of roughly 12692.4 m in
60–90 km (relative peak-to-peak 1.63e-5), and does not change between these states.
Its chord correlations with p and d over that broad window are only 0.022 and
-0.011. The short zooms have stronger correlations but only a few nodal samples;
neither establishes an I_h cause. No normalization change is proposed.

## What changes during this short interval?

Maximum first-to-final changes of weak coefficients (Pa):

| Window | p | d | sigma minus reference |
|---|---:|---:|---:|
| 22–35 km | 120.58 | 82.17 | 197.15 |
| 60–90 km | 466.01 | 371.52 | 261.49 |
| 70–71 km | 220.09 | 205.81 | 129.71 |
| 79–80 km | 220.45 | 243.24 | 136.20 |

Matching actual QPs before projection gives work-weighted increment RMS
223.95 / 227.87 / 110.26 Pa for p/d/sum over 60–90 km, and
213.29 / 220.48 / 113.90 Pa over 79–80 km. Raw changes are not zero, but are
small compared with the accumulated fluctuations. This is persistence over
milliseconds, not a demonstrated history-transfer defect or long-term stability.

## Exact constitutive decomposition now instrumented

The existing material implementation uses

\[
 \tau_k=\underbrace{\beta\tau_{k-1}^{\mathrm{working\ FE}}}_{\tau^{hist}}
 +\underbrace{2\kappa\dot\epsilon(u_k)}_{\tau^{strain}}
 +\underbrace{-2\kappa(\upsilon^{hist}+\chi V_k)S}_{\tau^{slip}}.
\]

The new observer exports these tensors directly from that evaluation, without
changing its original arithmetic, and projects each `-tau_component:N` with the
same native work measure and mass matrix. It records the incoming unscaled FE
tensor separately from beta-weighted history. Direct component evaluation avoids
inferring a small current increment by subtracting two large total stresses.
Nothing is reconstructed from the newly committed particles.

For the current mature frozen straight fault, upsilon^hist=0 and S:N=0. Thus
the **direct** localized-slip normal term vanishes, while slip still changes
normal stress **indirectly through the solved bulk strain and pressure**.
An observed current-step normal oscillation would point to that mechanical
response, not to a nonzero direct normal projection of chi*V*S.

The supplied CSVs do not contain incoming FE history or velocity gradients at
these QPs. Particle sums/squares are not a local tensor field. The available
binary checkpoints bracket this continuation but do not by themselves provide
the precommit evaluation of each accepted solve. Reading the final committed
stress as incoming history would use the wrong lifecycle stage. Therefore no
history/current percentages are invented from the saved total stress.

**Next action:** repeat the same bounded step-5612 restart with the extended
observer (no physics or time-selection change). If beta-weighted incoming FE
stress carries the roughness, compare retained particles, published transfer,
and the constrained working FE field at matching points next. That would test
transfer, not assume it caused all inherited roughness. If the current strain
term creates the roughness, investigate the strain/localized-source spatial
response instead. These branches remain conditional on that measurement.

## Changes and verification

Changed this follow-up: the plotting script; material response header/source;
surface-system diagnostic header/source; diagnostic plugin; README; Python
tests; and `tests/phase_field_fault_surface_system.cc`. All added stress terms
are opt-in output. No history, pressure, I_h, mesh, solver or integration rule
was altered. Retain the server's separately approved restart-only I_h allowance
when merging these changes; do not replace it by looser integration tolerances.
Rebuild ASPECT and all loaded plugins for the response-layout extension.

Verification:

- Release ASPECT and diagnostic plugin build with `-j4`.
- `python3 .../test_normal_stress_diagnostic.py`: **3 tests passed**.
- `timeout 120 ctest --test-dir build-pf-cpdi/tests --output-on-failure
  -R '^phase_field_fault_surface_dynamic_pressure$' -j1`: **1/1 passed,
  42.30 s**. The added inclined/nonzero-history check requires unchanged stress,
  residual and tangent with capture on/off, component closure, and vanishing
  direct crack-strain normal projection. Existing action checks also run.
- Plot generation from all five saved states completes; matched ownership,
  geometry and closure checks pass. Original plots/data are preserved.
- No new BP5 trajectory or production algorithm change. Full late-state
  component-output/MPI qualification awaits the repeated diagnostic; the
  current point/action test does not substitute for that measurement.
