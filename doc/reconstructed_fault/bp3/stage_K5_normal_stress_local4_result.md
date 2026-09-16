# K5: completed fresh four-rank normal-stress diagnosis

## Decision summary

The fresh local run completed initialization and steps 1–12 in **907.34 s**.
The corrected Theta checker passed, as did all 105 fresh-linear checks and
the unchanged bulk/surface convergence criteria. No physical model or solver
criterion changed; the only new core change invokes the opt-in, noncommitting
bulk-transfer diagnostic before accepted histories are published.

There are **two different concentrations**. The global tensile minimum is at
the bottom fault tip, entirely on prescribed-slip support: −12.235 MPa at
step 11 and −20.406 MPa at step 12. The 40-km junction has a separate,
pressure-dominated dipole. Its mixed-support minimum changes from +8.927 to
−0.660 MPa, while its maximum reaches 100.177 MPa. At step 12 its tensile
contribution reaches the unprescribed node at 39.9 km, which is lower-active.
Zero tensile load on the remaining free rows is therefore not a reason to
dismiss this pocket.

Prescribed deep slip is transferred into bulk deformation, including the
bottom segment. The evidence does not justify extending RSF to the bottom.
A frozen-history junction-location diagnostic is the next discriminating
test; no changed-cutoff solve or first-event continuation was run here.

## 1. Execution and verification

All paths below are relative to `benchmarks/reconstructed_fault/bp3/` unless
otherwise stated. The full evidence is in `normal-stress-complete-local4/`.

Command from the repository root:

```sh
python3 benchmarks/reconstructed_fault/bp3/run_normal_stress_local4.py
```

The runner records exact source/binary hashes and environment in
`provenance.json`. It runs Release ASPECT using `mpirun -np 4`, one thread per
rank, sparse B/G, and the qualified tridiagonal surface inverse. It starts
from the actual server `first_cycle_coarse/original.prm`, not a checkpoint,
and overrides only local paths, fresh-start selection, and termination after
step 12. The 120-s cap was removed by explicit user authorization. The
previous capped attempt and the original server checkpoint were preserved.
The latter's four checksums still pass.

The realized mesh is unchanged: 36,106 cells and 1,282,726 DoFs. There was no
parameter, pressure, support, I_h, initialization, history, or tolerance change.
There was no automatic retry and no solve beyond step 12. Exit status is 0;
completion is additionally established by accepted-state messages and the
numerical checks below, not exit status alone.

| Quantity | Step 11 | Step 12 |
|---|---:|---:|
| Physical time (s) | 1,855,817,124.760971 | 2,232,176,379.251613 |
| Accepted dt (s) | 884,306,322.054474 | 376,359,254.490642 |
| Newton updates | 6 | 5 |
| Krylov iterations, recorded by solver | 198 | 268 |
| Minimum accepted alpha | 1 | 0.544299719945 |
| Free / lower-active RSF nodes | 400 / 0 | 399 / 1 |
| Absolute bulk residual, algebraic norm | 0.0015190941 | 0.0006363143 |
| Fixed bulk normalization scale | 8.44784146e9 | 3.80348150e9 |
| Surface residual RMS (Pa) | 9.64672e-8 | 2.61255e-8 |
| Fixed surface normalization scale (Pa) | 6.57931995e6 | 3.14348262e6 |

All 13 accepted states pass the unchanged bulk target and 1e-8 relative
surface criterion. All **105** returned linear directions pass their fresh
residual target; the largest fresh/target ratio is **0.997666455**. Total
recorded Krylov iterations are 2,140. `analysis/convergence.json` and
`analysis/linear_checks.csv` contain the individual checks.

The step-12 benchmark Theta relative error is **2.22044605e-16**, versus the
unchanged 1e-12 assertion. An additional offline, 50-digit Decimal calculation
uses the accepted nodal rates and actual accepted dt, with the preceding
committed nodal Theta as input. Its worst step-12 relative error is
**2.09843747e-16**. This is an independent arithmetic check of the history
update, not a reset of any trajectory. See `analysis/theta_independent.csv`.
The captured near-bound regression cases, deliberately incorrect histories,
and zero-interval case also pass (`theta-unit.log`).

The new diagnostic hook leaves every accepted record at steps 0 and 1
**bitwise identical** to the previous four-rank attempt. Against the original
64-rank server records through step 11, maximum relative differences are
2.11e-13 in physical time, 5.06e-12 in maximum free V, and 3.92e-12 in maximum
normal stress. The step-11 tensile minimum differs by only 3.17e-6 Pa.

## 2. Actual constitutive extrema: two locations, not one

Compression is positive:

\[
\sigma_n=50\ {\rm MPa}+\Delta p-\Delta\tau:N.
\]

These are **pre-history-publication constitutive samples**, evaluated from
the parent bulk FE state and retained particle stress. The surface coordinate
labels the domain quadrature's Q1 evaluation point; it is not necessarily
the physical parent point at which the bulk fields were sampled.

| Sample | Surface xd (km) | Delta p (MPa) | −Delta tau:N (MPa) | Total sigma_n (MPa) | Support |
|---|---:|---:|---:|---:|---|
| Step 11 global minimum | 115.470054 | −21.379684 | −40.855155 | **−12.234839** | prescribed only |
| Step 12 global minimum | 115.470054 | −21.065329 | −49.341024 | **−20.406352** | prescribed only |
| Step 11 junction minimum | 39.993648 | −39.670972 | −1.401657 | **+8.927371** | mixed |
| Step 12 junction minimum | 39.961334 | −43.096021 | −7.563831 | **−0.659852** | mixed |
| Step 11 global maximum | 39.925530 | +37.231415 | +3.612528 | **90.843944** | mixed |
| Step 12 global maximum | 39.983708 | +43.971442 | +6.205388 | **100.176830** | mixed |

Several domain quadrature points can share the same extremal stress because
bulk/history inputs are parent-P0. The CSVs retain their distinct quadrature
identities; the selected minimum is not asserted to be a unique point.

### Bottom minimum

Both steps identify parent **11936**, rank 0, cell
`0_10:0011010111`, fault 0, segment 0, xi=0, domain-q 0. The surface point is
**(21132.486540519, 0) m**, the bottom tip. Parent positions are:

- step 11: (21012.773830887, 81.771694394) m; normal offset +144.560094894 m;
- step 12: (21012.855768411, 81.850114902) m; normal offset +144.528345171 m.

The parent touches no unprescribed surface support. V is exactly Vp=1e-9 m/s.
Its domain volume at step 12 is 1059.831018 m²; the particular selected small
quadrature weight is 4.88431356e-5 m². The latter must **not** be used as the
total weight of the tensile anomaly. Step-12 phi=0.504199995, chi=0.00278809649
1/m, and total shear q=−54.134343 MPa. This is not a solved RSF traction
equation: V is prescribed there and the corresponding surface equation is
replaced by its kinematic condition.

### Junction minimum

At step 12 the mixed-support tensile parent is **217673**, rank 2, cell
`0_10:3021033300`, segment **755**, xi=0.386655821, domain-q 0. Parent position
is **(59065.290703567, 65315.107048699) m**, normal offset **−193.218518 m**;
the surface quadrature point is **(58886.846250536, 65392.469224981) m**.
Its V is 6.133441789e-10 m/s, phi=0.449358031, chi=0.00101660863 1/m,
mu=0.546189115, and total q=27.997901 MPa. Its whole domain measure is
1059.643059 m². The much stronger positive counterpart is on the opposite
side of the band, at normal offset +198.270684 m.

`analysis/step11/` and `analysis/step12/` contain `raw_extrema.csv` (the
strongest 20 on each side), `raw_selected.csv`, and
`raw_selected_with_theta.csv` (preceding committed Theta used by mechanics
and newly committed Q1 Theta interpolated at the sample). Parent coordinates,
surface coordinates, rank/cell/particle/QP identity, xi, weights, V, phi, chi,
pressure, normal traction, friction and shear are separate columns.

## 3. All tensile weights and their equation-level effect

The following sums use **all** domain quadrature samples and positive Q1
test weights on all four ranks, not just the retained extrema. Units are the
actual admitted 2-D domain measure (m²), with no renormalization.

| Region / test support | Step 11 tensile weight | Step 12 tensile weight |
|---|---:|---:|
| Bottom-tip basis functions, all prescribed | 3178.982847 | 5297.872946 |
| Junction prescribed node at 40.0 km | 0 | 414.439966 |
| Junction unprescribed node at 39.9 km | 0 | 645.203094 |
| Other unprescribed nodes | 0 | **0** |
| Total | **3178.982847** | **6357.516005** |

Total admitted weights are 182,258,727.405 and 182,260,741.822 m². Thus
tensile fractions are 1.74421e-5 and 3.48814e-5 of total weight. These small
global fractions do not themselves establish harmlessness.

At step 11, every tensile sample has prescribed-only test support at the
bottom. At step 12, additional tensile weight occurs on the 39.9–40.0 km
mixed element. Node **756**, at **39.9 km**, is now lower-active at Vmin=1e-20;
node 755 at 40 km is prescribed. Tensile weight on node 756 is **0.4080%**
of its total test weight. Its signed tensile friction weak load is
**−2.29591925e8 Pa m²**, or **5.58597e-5** of that node's net friction-load
magnitude. Its ratio to the L2 norm over *all unprescribed* friction rows is
2.66910e-6. The remaining 399 currently free nodes receive zero tensile
weight. No negative sample extends onto their positive test support.

This distinction matters: lower-active is not the same as prescribed deep
slip. The tensile contribution remains part of the physical residual used
to determine active-set status; its removal from the current restricted free
solve does not remove the model issue or justify clipping it.

The source evaluates **mu(V,Theta) sigma_n before integration**, without
clipping sigma_n. This term enters R=q−C−mu sigma_n−damping*V, and sigma_n
also enters the sigma_n*dmu/dV coefficient of K_V. It is not equivalent to
evaluating friction from a smoothed or projected normal stress. The full Q1
normal-stress ranges remain positive: **[25.219,52.582] MPa** at step 11 and
**[20.853,53.081] MPa** at step 12. Neither those ranges nor positive station
values contradict the raw tensile samples.

## 4. Profiles, widths and growth at Wf

Full Q1 profiles, 10–45 km CSVs, raw samples and separate plots are in the
two step folders. `junction_profiles.png` includes 13–20 km, 37–43 km and
the bottom tip. `projected_stress_detail.png` resolves the much smaller
projected stress perturbations; `raw_locations.png` shows band offsets.
15, 18 and 40 km are explicitly marked. Raw selection is bounded extrema
sampling, not a complete pressure/normal-column profile.

Peak-to-peak **consistent Q1** variation, MPa:

| Step / region | Delta p | −Delta tau:N | sigma_n | total shear q |
|---|---:|---:|---:|---:|
| 11 / 13–20 km | 0.223826 | 0.121747 | 0.335798 | 3.770770 |
| 11 / 37–43 km | 0.272830 | 0.097460 | 0.348600 | 8.172248 |
| 12 / 13–20 km | 0.268602 | 0.149577 | 0.407423 | 4.131953 |
| 12 / 37–43 km | 0.319999 | 0.125083 | 0.444821 | 10.012363 |

The projected sigma feature near the constitutive transition is broad. A
diagnostic half-amplitude width after subtracting the straight line joining
the region endpoints is approximately **4.35–4.38 km**, versus **0.23–0.26 km**
for the junction feature. The junction pressure feature is about 0.23–0.24 km
wide by the same definition (`analysis/feature_widths.csv`). These are
descriptions of one discrete solution, not convergence-qualified widths.
The junction spans only a few 100-m fault elements. The bottom tensile
test support is on its final approximately 100-m surface element.

The 40-km shear variation is about 2.17 times the transition-region variation
at step 11 and 2.42 times at step 12. Its raw normal-stress range is much
larger than the projected range because the pressure dipole lies across the
band. This is not a purely isolated zero-weight point: it has a finite domain
measure, affects an unprescribed basis function, and has a several-element
projected signature on the RSF side. This single grid does not establish a
continuum point singularity or a resolved finite-width limit.

Growth, from `analysis/junction_history.csv`:

| Step | Last free-region nodal V at 39.9 km (m/s) | V−Vp (m/s) | Junction projected sigma peak-to-peak (MPa) |
|---|---:|---:|---:|
| 5 | 8.98689143e-10 | −1.01310857e-10 | 0.001296 |
| 8 | 4.83348590e-10 | −5.16651410e-10 | 0.021896 |
| 10 | 1.66805341e-10 | −8.33194659e-10 | 0.143615 |
| 11 | 6.85828010e-11 | −9.31417199e-10 | 0.348600 |
| 12 | 1e-20 (lower-active) | −9.9999999999e-10 | 0.444821 |

**V(40−)=Vp exactly in the continuous Q1 representation.** The table is the
last unprescribed node, 100 m above the junction, not a discontinuous limit.
The concentration grows with the increasingly steep represented transition
from that node to prescribed Vp. Correlation plus localization supports the
hard-junction hypothesis, but is not the requested shifted-junction causal
test.

## 5. Deep-slip transfer is present

The accepted-state diagnostic evaluates the same bulk-QP constitutive helper,
exact Stokes quadrature, cached associations and physical Q1 V as conventional
assembly, before particle history publication. It invokes the existing
standalone residual operation; its vector is discarded. This avoids both
reevaluating Maxwell from newly committed history and claiming that an old
history visualization is accepted current stress.

For both steps, localization-weighted deep V is **1.000000000e-9 m/s**.
The physical total crack rate is chi*V+history, not chi*(V−Vp). The frozen
profile history-localization contribution is exactly zero. The largest
integrated identity discrepancy at step 12 is 2.65e-23. This check is not a
new claim about full-profile normalization accuracy.

| Bulk-QP interval (km) | Step 11 chi-weighted V (m/s) | Step 12 chi-weighted V (m/s) |
|---|---:|---:|
| 39.8–39.9 | 2.65539801e-10 | 2.15321061e-10 |
| 39.9–40.0, mixed | 5.57051082e-10 | 5.24435540e-10 |
| 40.0–40.09996, prescribed | 1e-9 | 1e-9 |

The bottom segment has **85 associated bulk QPs**, sum chi*JxW=100.976813 m,
and integral crack rate **1.00976813e-7 m²/s**, not zero. Prescribed entries
are not masked out. Source trace also confirms the same physical V enters
the particle Maxwell update; only Newton increments/equations are restricted.
See `bulk_transfer.csv` and `support_and_transfer.json` in each step folder.

## 6. Remaining uncertainty and one recommended next action

The faulty Theta checker is closed. No missing-deep-slip implementation
defect was found. The observed negative minimum cannot be attributed to the
40-km junction alone: it is a separate bottom-tip concentration. Conversely,
the step-12 mixed tensile pocket means the junction cannot be dismissed
using the bottom location or positive projected stations.

**Recommended next action:** the previously proposed single, noncommitting
frozen-history junction-location experiment (40 to 35 km), with all other
data fixed and no history publication. Test whether the pressure dipole
moves with the imposed-slip junction while the bottom-tip feature remains
separate. This would discriminate a junction-driven concentration from a
fixed material/transfer hotspot. It was **not executed here**: this run kept
the current cutoff and completed the requested baseline through step 12.

The data do not yet distinguish a continuum endpoint/junction singularity
from discretization or parent-sampling amplification. They do not justify
extending RSF to the bottom, smoothing, altering Vp, changing pressure,
clamping normal stress, or resuming the long event run. Any such correction
requires a separate review.

## 7. Changes and reproducibility

This turn changes core source only in `source/simulator/solver.cc`: an opt-in
diagnostic evaluation after convergence and before any accepted history
write. All preceding working-tree changes were preserved. Normal graphical
postprocessor output is untouched.

Added benchmark files:

- `normal_stress_complete_local4.prm`: fresh, unchanged physical input through step 12;
- `run_normal_stress_local4.py`: four-rank execution, provenance, no timeout/retry/overwrite;
- `summarize_normal_stress.py`: offline weak/profile, independent Theta, convergence and plotting evidence.

Existing `analyze_normal_stress.py` reduced all accepted steps successfully;
its projection consistency checks were not weakened. The Release build used
`cmake --build build-pf-cpdi --target aspect.exe.release -j4` and passed.
`git diff --check` passed. The BP3 run itself supplies the four-rank accepted
lifecycle test for the new hook. No separate broad integration suite or new
one-/two-rank campaign was run; prior focused action evidence remains valid.

Main source references: `PhaseFieldFault::evaluate_reconstructed_fault_point`
for total traction and signed friction/Jacobian; `ReconstructedFaultSurfaceSystem::
assemble_surface_system` for parent-P0/domain-Q1 sampling and weak moments;
`ReconstructedFaultStokes::{execute,evaluate_slip_dependent_bulk_residual}`
for the bulk load; `ReconstructedFaultManager::interpolate_slip_rate` for the
unmasked physical field. Exact compiled files are hashed in the run manifest.
