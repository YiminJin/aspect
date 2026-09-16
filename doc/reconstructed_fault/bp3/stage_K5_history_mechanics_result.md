# K5: does particle/FE history mismatch materially affect mechanics?

Consolidated synthesis of these four follow-up tasks:
[history, spatial-resolution and timestep report](stage_K5_history_resolution_consolidated_report.md).
This individual report is retained as the chronological evidence record.

Latest evidence: [completed matched-time localized resolution pair](stage_K5_matched_junction_resolution.md).
The weak-history discrepancy contracts under refinement, but lower contact
persists and raw tensile extrema do not converge. The original frozen and
alternate-history results below remain separate evidence.

Follow-up: [targeted spatial-resolution preflight](stage_K5_junction_resolution.md)
stopped after converged initialization because of matched-timestep and mesh
locality confounds. It is not a completed step-12 resolution comparison.

## Decision

**Yes, at the level of the bound margin and neighbouring rates; no, it does
not explain away the 39.9-km lower contact.** At the exact baseline step-12
solution, replacing only the complete surface retained-stress tensor by its
constrained FE evaluation increases the driving deficit from **127.699 to
153.310 kPa**. This is a **20.06%** change in the contact margin, although
only 0.0900% of the shear-driving term.

That material margin change justified the one authorized disposable solve.
After re-equilibration, the node remains at **Vmin=1e-20 m/s**, its resisting
reaction is **183.696 kPa**, and the neighbouring 39.8-km rate changes by
**+4.154%**. The pressure dipole persists. FE-based diagnostics remove the
reported tensile samples, but evaluating the original particle history at
the new solution still gives the same total tensile support weight.

**Recommended next action:** retain the current particle-history formulation
and make any proposed history-representation change contingent on a targeted
local spatial-resolution comparison. Do not adopt FE sampling as a locking
or tensile-stress “repair”: this test does not establish that it is the more
accurate representation. No further run was launched.

## 1. Scope and exact frozen state

This follows [the consolidated report](stage_K5_normal_stress_consolidated_report.md).
The original particle-based accepted solution, timestep, geometry, phase,
full I_h, support, pressure treatment, transfer, friction parameters, 40-km
cutoff and solver criteria are unchanged. No Theta retesting, smoothing,
clipping, junction relocation or long trajectory was performed.

The exact accepted Stokes FE polynomials were extracted from a disposable
copy of local step-12 checkpoint `restart/01`. They were identified by
**CellId and local FE index**, not interpolated from VTU and not assumed to
have the same global DoF numbering as another checkpoint. The final export
contains all 36,106 cells and their 22 Stokes coefficients. Shared imported
coefficients agree exactly; geometry and saved V/time checks pass.

A separate copy of local step-11 `restart/03` performs normal step-12
advection, transfer and constitutive preparation. At the pre-Newton callback,
the saved step-12 u,p and V are supplied privately, with the same physical
constraints on working fields. Theta, C and particle history remain the
preceding committed values. The original weak R is reproduced with maximum
absolute row error **0.00048828125 Pa m²**, only **9.7761e-17** of the maximum
weak shear-term scale. This verifies the frozen state using the actual
unreplaced residual, including its nonzero bound row.

At every production domain quadrature point, the point constitutive routine
is evaluated twice with identical inputs except for the **full symmetric old
stress tensor**, sampled at the same parent coordinate:

- original: retained parent-particle tau11;
- alternative: constrained FE tau11,h.

Both shear and normal contractions change. Cohesion, friction state and
localization do not change. The production Q1 test/trial weights assemble
both physical residuals and K_V=-dR/dV before prescribed/active replacement.
No old history is updated or republished during this comparison.

## 2. Physical bound balance—not the restricted convergence residual

Let m_i=integral N_i dnu>0. The following are weak terms divided by m_i,
not M-inverse-projected nodal values:

\[
 \bar R_i=(q_i-C_i-f_i-d_i)/m_i,\qquad
 \lambda_i=-\bar R_i\quad\text{at lower contact when }\bar R_i\leq0.
\]

At node **756**, xd=39.9 km, m_i=158130.718252235 m²:

| Term | Original particle history | FE history at identical u,p,V,Theta | Difference |
|---|---:|---:|---:|
| Shear driving (MPa) | 28.460551128 | 28.420536212 | −0.040014916 |
| Cohesion (MPa) | 2.596145523 | 2.596145523 | 0 |
| Friction (MPa) | 25.992104551 | 25.977700925 | −0.014403625 |
| Damping (Pa) | 0.001115658 | 0.001115658 | 0 |
| Physical R / m (kPa) | **−127.698947** | **−153.310237** | **−25.611290** |
| Normal traction / test weight (MPa) | 50.220498312 | 50.190932204 | −0.029566109 |

The weak resisting reaction changes from **2.01931261334e10** to
**2.42430578669e10 Pa m²**. Both frozen representations support contact:
V−Vmin=0, lambda>0, and lambda*(V−Vmin)=0. These are physical complementarity
quantities. The implemented active set is still selected from the coupled
projected Newton direction, not from a new scalar sign criterion.

Do not substitute the plotted Q1 nodal `F`: at this node it is +1.611 MPa,
whereas the physical weak row divided by positive test weight is negative.
Mass-inverse projection mixes neighbours; it is not the bound reaction.

**Tension does not cause locking here.** The original tensile samples give
a signed friction contribution of −1451.912 Pa after test-weight division,
which *reduces* resistance, helping motion. This is 1.14% of the 127.699-kPa
contact margin. Removing a tensile sign is not proof that locking is repaired.
The full FE-history change reduces shear driving more than it reduces
friction, so the net frozen reaction becomes stronger.

### Neighbours and interior control

Changes in physical test-weight mean terms, in kPa, at fixed baseline state:

| xd (km) | Status | Delta shear | Delta friction | Delta R |
|---|---|---:|---:|---:|
| 40.09996 | prescribed | +13.744 | +4.603 | +9.141 |
| 40.0 | prescribed | +84.923 | +8.982 | +75.941 |
| 39.9 | lower-active | −40.015 | −14.404 | **−25.611** |
| 39.8 | free | +57.994 | +0.182 | **+57.812** |
| 39.7 | free | +6.104 | +2.527 | +3.577 |
| 25.0 | free interior control | +14.418 | −0.00297 | +14.421 |

The 39.8-km and control changes are respectively about 0.197% and 0.0512%
of their shear terms. Ratios to their near-zero converged residuals are not
used to claim materiality. The 20% contact-margin change was the reason to
proceed to the single conditional solve.

### V tangent and coupling consistency

At the bound node, K_ii/m_i changes from **2.634162813e16** to
**2.630540993e16 Pa/(m/s)**, a **−0.13749%** change. The summed off-diagonal
row coefficients change from 4.882529494e15 to 4.881321716e15 Pa/(m/s).
The stored CSV contains these coefficients for all rows.

Forward perturbations of V_756 by 1e-17, 1e-18 and 1e-19 m/s remain feasible.
For h=1e-17, the three affected weak rows' finite differences agree with
−K*e_756 within **1.29e-6 relative**, for both representations. Smaller h
eventually increases subtraction noise; no tangent/solver tolerance is changed.

The alternative calls the same point routine consistently in residual and
Jacobian assembly. In particular the total normal stress multiplies dmu/dV.
The FE stress-composition fields remain frozen auxiliary data, not Newton
unknowns. There is therefore no new history derivative in G; its existing
bulk strain/pressure terms retain their meaning. Fresh condensed linear
checks during the conditional solve also pass.

## 3. Common velocity-test weak loads

These are actual common test functions, not a comparison of unrelated cell
load norms. For down-dip centers 25, 35, 39.9 and 115.470054 km, use

\[
 w_a(x)=a\,b((s-s_c)/1000)\,b(z/800),\quad
 b(t)=(1-t^2)^3\ (|t|<1),\quad b=0\text{ otherwise},
\]

where a is the fault normal or tangent, s is down-dip position and z is
signed normal distance, all lengths in metres. The same analytic symmetric
test gradient is integrated against beta*tau_old with the negative bulk-RHS
sign. These are smooth virtual-velocity tests, **not** replacements for the
production Stokes basis.

Particle history is P0 over each full actual Voronoi polygon, integrated once
on its real owner's rank using a triangle fan and deal.II Gauss quadrature.
FE history is integrated on actual bulk cells using FEValues with the same
physical test gradient. No support/volume clipping or point-volume
substitution is used. The compact test is simply zero outside its support;
the bottom test integrates its intersection with the physical box. This
does not apply a force or alter production particle domains.

Order 4 and 6 are compared separately for both representations. Order-6
loads (Pa m for the chosen test normalization) are:

| Center / test direction | Particle load | FE load | FE minus particle | Difference / absolute particle load |
|---|---:|---:|---:|---:|
| 25 km / normal | −1.237825700e9 | −1.234163107e9 | +3.662593e6 | +0.2959% |
| 25 km / tangent | −1.036133828e7 | −1.035750542e7 | +3.832857e3 | unresolved tiny difference |
| 35 km / normal | −2.309513025e9 | −2.302848277e9 | +6.664749e6 | +0.2886% |
| 35 km / tangent | −4.689605490e6 | −4.511750950e6 | +1.778545e5 | +3.7925% |
| 39.9 km / normal | **−2.029858923e10** | **−2.018691831e10** | **+1.116709269e8** | **+0.5501%** |
| 39.9 km / tangent | **−3.354768317e6** | **+5.581376421e5** | **+3.912905959e6** | sign change of a much smaller component |
| Bottom / normal | +3.701275208e8 | +3.154810767e8 | −5.464644e7 | −14.764% |
| Bottom / tangent | −1.124935386e8 | −1.687336222e8 | −5.624008e7 | −49.994% |

At 39.9 km the sum of the order changes in the two loads is **556.3** for
the normal test and **720.7** for the tangent test—respectively 4.98e-6 and
1.84e-4 of the measured representation differences. These effects are resolved
by this quadrature check. The 25-km tangent difference is smaller than its
4-to-6 quadrature change and is deliberately left unresolved. Large relative
changes of cancellation-small tangent loads are not described as equally
large changes in total mechanics. Bottom effects remain a separate concern.

## 4. The one conditional solve

Only after the material frozen comparison, a separate process starts from a
new copy of step-11 `restart/03` and solves step 12 with FE retained history
in **both shear and normal surface terms**. The bulk transfer/history load is
unchanged. Time is **2232176379.2516127 s**, dt **376359254.49064183 s**;
the 40-km prescribed mask, parameters and numerical settings are unchanged.

| Check | Result |
|---|---:|
| Runtime, four ranks | **97.115 s** |
| Newton updates | 5 |
| Returned linear directions / total Krylov iterations | 11 / 267 |
| Worst fresh linear residual / target | **0.857958475** |
| Minimum accepted alpha | 0.50979931959948 |
| Rejected line-search candidates | 0 |
| Final absolute bulk residual | 0.000636537142 |
| Fixed bulk scale | 3.803481499864e9 |
| Final normalized bulk residual | **1.67302282e-13** |
| Final surface RMS / fixed scale (Pa) | 6.70440259e-8 / 3.143481324790e6 |
| Final normalized surface residual | **2.13279543e-14** |
| Free / lower-active RSF nodes | **399 / 1** |

All fresh checks and unchanged convergence criteria pass. The solver stops
before publication, and the repaired in-memory rollback verifier **passes**:
complete bulk vector, committed/current V, all particle and surface properties,
particle IDs/positions and fault vertices match the pre-solve snapshot.
Source and copied checkpoints are unchanged. Exit 1 is the intentional
noncommitting stop, not a failed mechanical solve. No history is committed.
This verifies the repaired checker on this run; it does not retroactively
change the earlier 35-km run's recorded verification status.

### Bound and rates after re-equilibration

| xd (km) | Baseline V (m/s) | FE-history solve V (m/s) | Change |
|---|---:|---:|---:|
| 40.0 and deeper | 1e-9 | 1e-9 | exact prescribed rate |
| **39.9** | **1e-20** | **1e-20** | **same lower contact** |
| 39.8 | 4.492974245e-10 | 4.679627839e-10 | **+4.1543%** |
| 39.7 | 5.831054519e-10 | 5.764738654e-10 | −1.1373% |
| 25.0 control | 2.300380034e-10 | 2.321582981e-10 | +0.9217% |

The new bound reaction is **183.696345 kPa**, **43.85% larger** than baseline.
At this same new u,p,V, reevaluating the original particle history gives
R/m=−158.100617 kPa at the bound, and −57.812087 kPa at the neighbouring
39.8-km free row. Thus the alternative is a genuinely different discrete
equilibrium, not merely a renamed stress output. The largest absolute nodal
rate change is 1.866536e-11 m/s (1.8665% of Vp).

### Pressure and dual-history traction profiles

Consistent Q1 peak-to-peak variation over 37–43 km, in MPa:

| Field | Baseline particle solution | FE-history solution, FE diagnostics | Same new solution, original particle diagnostics |
|---|---:|---:|---:|
| p | 0.319999 | **0.321120** | 0.321120 |
| −tau:N | 0.125083 | **0.115763** | 0.125364 |
| sigma_n | 0.444821 | **0.350676** | 0.446483 |
| q | 10.012363 | **9.966066** | 9.995119 |

The pressure variation changes by only **+0.35%**. Most of the apparent
21.2% decrease in projected normal-stress variation comes from the diagnostic
history representation: using particle history at the new solution retains
the original variation almost unchanged (+0.37%).

The actual FE-mode raw junction sigma range is **3.059064 to 97.683530 MPa**;
the global FE-mode minimum is about **0.039509 MPa** at the bottom. However,
the original particle evaluation at the same new solution still has tensile
test weight **1059.643059 m² at the junction** and **5297.872946 m² at the
bottom**, identical to baseline (total 6357.516005 m²). FE-mode tensile weight
is zero. All-sample dual-response moments, not a selected extrema subset,
establish this comparison.

`dual_history_selected_samples.csv` additionally reports both representations
at the retained FE-selected extrema, using the exact linear beta*Delta-tensor
correction. Its particle extrema are **not** the global particle extrema:
selection was by FE stress. Do not interpret absence of a negative value in
that selected subset as absence of the measured particle tensile pocket.

## 5. Interpretation and stop point

The retained-history mismatch changes the physical bound margin and nearby
rates materially, despite being small relative to individual traction terms.
It has not been shown to cause locking: removing it in the specified
alternative strengthens contact. Nor has it removed the underlying junction
pressure concentration. A positive reported tensile minimum is not a repair
when the original evaluation at the same state still has tensile support.

Remaining uncertainty is **accuracy**, not whether a difference exists. Neither
representation is an independent reference; particle history and FE transfer
can both carry spatial/history error. Common weak tests show especially
large bottom discrepancies, but do not select a correct boundary/history
formulation. No additional solve, convergence sequence or correction was run.

The one recommended next action is the local resolution comparison stated
above, before any change to the production history evaluation. The current
40-km cutoff, pressure, physics, history semantics and tolerances remain the
accepted baseline.

## 6. Reproducibility and implementation status

Performed successful commands, from the repository root:

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_history_mechanics.py export --tag synchronized-local4
python3 benchmarks/reconstructed_fault/bp3/run_history_mechanics.py frozen --tag verified-local4 --bulk-dir history-mechanics-export-synchronized-local4
python3 benchmarks/reconstructed_fault/bp3/run_history_mechanics.py solve
```

The final frozen comparison took **26.082 s**, the complete read-only export
2.078 s. Runs are not automatically repeatable into existing directories;
the runner refuses overwrite. Exact hashes/commands/environment are in each
directory's `provenance.json` and `execution.json`.

All paths below are under `benchmarks/reconstructed_fault/bp3/`:

- `history-mechanics-frozen-verified-local4/`: raw per-rank dual weak rows,
  `surface_comparison.csv`, `comparison.json`, `common_history_tests.csv`,
  `history_tangent_fd.csv`, log and checkpoint hashes;
- `history-mechanics-solve-local4/`: log, `noncommitting_surface.csv`, verified
  `noncommitting_history.csv`, dual weak rows, bounds/line-search records,
  `solve_comparison.json`, `re_equilibrated_profiles.csv/png`,
  `dual_history_selected_samples.csv`;
- `history-mechanics-export-synchronized-local4/`: complete exact Stokes
  cell coefficients and input V profile.

Diagnostic development failures are preserved separately: initial exports
could terminate a rank before all files closed, and an early common-test
wrapper incorrectly used a stable particle ID where the existing domain API
requires a local index. The first apparent DoF incompatibility was therefore
not proof of a production repartition/transfer defect. All-rank close
synchronization, complete cell-coverage checks and the correct local index
fixed the diagnostic. A trial deal.II barrier spelling failed compilation
and was replaced by MPI_Barrier. None of these attempts entered a mechanical
solve; the conditional FE solve was performed **once**, without retry.

Code changes:

- `source/reconstructed_fault/surface_system.cc`: opt-in dual point-response
  audit and alternative FE-history response, with consistent R/K_V and
  ordinary G meaning. FE mode requires both audit and the noncommitting flag;
  default physics are unchanged. This is **not promoted as a production
  history method**.
- `bp3.cc`, new `history_mechanics_diagnostic.h` and `history_common_tests.h`:
  benchmark-local exact-state extraction, frozen comparisons, common virtual
  tests and FD checks; no generic integration framework.
- `junction_diagnostic.h`: reuse the repaired complete rollback check for
  the FE-history diagnostic without enabling the 35-km mask.
- New runner and offline analysis scripts preserve separate physical weak
  rows, projected fields and selected raw samples.

Release builds, final frozen state/quadrature/FD checks, the single solve's
fresh-linear/nonlinear checks, checkpoint hashes and complete rollback pass.
No broad suite, Theta regression rerun, production convergence campaign or
one-/two-rank invariance claim is added. The earlier incomplete diagnostic
directories are not counted as passing tests.
