# Peak-.60 matched-parent half-step comparison

The selected task is four midpoint intervals of .0025 s from the saved accepted
step 12 / .060 s parent through .070 s, on the same 128×128 mesh. Compare
.065 and .070 with the retained dt=.005 free response. This is local timestep
sensitivity from a common evolved parent, not full-from-zero temporal convergence
or mesh qualification. The original t=0 equilibrated seed remains phi0.
This is the synthetic fixed-direction bulk cohesive-tip/section-RSF control;
it does not qualify mature-fault production or general turning/branching.

## Configuration, execution and preservation

Entry is `pf-rsf-particle-free`, revision `b32a2d4f6e21`, with incoming dirty work
preserved. No core, checker, CMake, quadrature, physical parameter, search-policy
or acceptance-tolerance changes and no rebuild. The protected step-89 executable
SHA256 is `034d2a848a3a10c496fa097475f31497c00bc68c3456fb6d87c1ee700f453c27`;
checker SHA256 is `b800ef431a2469760efd16befcba94e8cd743156629b55e4c921f769255d9d84`.
The cached executable banner is not the source revision.

The retained [free-response configuration](RESOLUTION60_FREE_GROWTH_REPORT.md)
has hx=.0625 m, hy=.03125 m, fixed-direction section RSF, native DG Bernstein,
true stress, midpoint and the opt-in positive-metric-first constrained search
with exact polishing. Material, strength, toughness, Dc, pressure reference,
uniform shear and all boundary velocities are unchanged. Only the physical
timestep changes after restoration. The first banner is .0625 / .0025; step 16
is the explicit end-step guard. Two MPI ranks, one thread each, with at most one
compression and one analysis worker. The process cap is one hour; the graceful
wall limit is .95 hour. Actual remote/boundary phase integrity checks remain.

The source checkpoint is the independently saved
`build-particle-free/resolution60-free-growth/parent-step12/output/restart/01`.
The runner copies it into a fresh run directory. Exact hashes, incoming dirty
file hashes, input and command are in
[`entry.json`](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_halfstep_window/entry.json)
and the run's `command.json`. The restore observer reproduces all exported
stress/cohesive/phase samples, native phase coefficients, section age/slip/rate
and force exactly. Its pending .065/.005 clock precedes the existing restart-dt
signal override; accepted parent time is correctly .060 and all actual new
physical intervals must be .0025. This distinction is recorded in
`restore-verification.json`.

Reproduction (from repository root; two ranks):

```sh
python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/run_resolution_matched.py coarse \
  --growth --label resolution60-halfstep-window \
  --evidence-directory benchmarks/reconstructed_fault/particle_free/evidence/resolution60_halfstep_window \
  --resume-from build-particle-free/resolution60-free-growth/parent-step12/output \
  --snapshot-id 1 --dt .0025 --end .07 --end-step 16 --ranks 2 \
  --checkpoint-steps 1 --seconds 3600 --wall-hours .95
python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/analyze_resolution60_window.py --restore
```

The existing label is preserved; reproduction requires a fresh label/output
and corresponding analysis path. Original outputs and checkpoints are retained;
new closed diagnostic exports are losslessly compressed with verified hashes.
No historical cleanup losses are recovered or reclassified.

## Observation definitions and figures

The region masks, native Q6 point reductions and telescoping SIPG row-energy
allocation are those of the preceding free-response report. Each accepted step
records increments from the original seed, the .060 parent and the preceding
accepted step. The disjoint central/taper/ahead/remote/other row energies sum to
the solver's fracture increment. Signed allocations of face contributions are
not interpreted as independently positive local fracture densities.

Physical-work error is Wext−deltaE−Db−Dc−Df. Compare both its signed cumulative
sum and accumulated absolute sum over .060–.070. Inherited errors before .060
are common and are not claimed to have been reduced. Field differences use
actual Q6 weights; symmetric tensor norms include tauzz and twice tauxy².
Endpoint slip and age are directly compared; a coarse midpoint V is compared
with the duration-weighted mean of the two half-step V values over the same
interval, not a rate at a different stage time.

The requested profiles show phi0, phi and delta phi together on the native
upper trace at y=2 across the original right endpoint x=6. CSVs also retain the
lower trace and cuts y=2±hy/2. Repeated x coordinates retain DG jumps. The phase
maps show the same three fields side by side and superpose the **absolute
phi=.05 contour** (cyan) and its original-seed counterpart (white dashed).
This is distinct from the earlier delta-phi=1e-4 connected-damage threshold.
No clipping, smoothing, normalization floor or new physical admission rule is
introduced.

Absolute-contour tracking includes exact piecewise-Q2 crossings on four cuts
and a cellwise native 2D contour. Each cell is sampled separately for contour
rendering, so interpolation never bridges a DG jump. Subdivision 16 versus 32
measures contour extraction sensitivity, not mesh convergence. The central
excess-compliance width W uses the retained native polynomial peak and exact
half-height crossings. W/hy remains a screening measure, not a resolution proof.

## Results

**All four intervals accepted through .070 s; exit 0.** Process time is 1674.186 s (27.90 minutes). The wrapper including final export compression took 1679.168 s. The simulator timer labelled “including restarts” also contains inherited runtime and is not this run’s elapsed time. No simulation ranks remain.

| Time (s) | Max delta phi, central | Old taper | Initially intact ahead | Window added fracture (J/m) | Interval work error (J/m) | W/hy |
|---|---:|---:|---:|---:|---:|---:|
| 0.0625 | 0.0140024 | 0.003639901 | 2.086528e-05 | 0.0015336126 | 1.053766e-06 | 1.599277 |
| 0.0650 | 0.01756354 | 0.005438452 | 0.002912008 | 0.0034425084 | 1.192352e-06 | 1.648650 |
| 0.0675 | 0.02095858 | 0.0138048 | 0.00749615 | 0.0062384933 | 1.509357e-06 | 1.698191 |
| 0.0700 | 0.02555985 | 0.02116965 | 0.01419069 | 0.010004641 | 2.083668e-06 | 1.764599 |

Remote, boundary and `other` phase increments and added fracture remain zero. All four integrated negative-rate contributions are zero. The maximum native phase remains .60; its central compliance width broadens, rather than narrowing. Bounds are preserved to roundoff (minimum coefficient −5.42e-19), and the minimum accepted coefficient increment is zero. W/hy still remains below two: broadening is not evidence of spatial convergence.

### Matched endpoints

All relative differences below use the half-step result as denominator. The phase denominator is its **change since .060**, not the much larger total phase field.

| Endpoint (s) | Phase-increment L2 difference | Maximum phase difference | Stress L2 difference | Slip L2 difference | Age L2 difference | Interval-mean V L2 difference |
|---|---:|---:|---:|---:|---:|---:|
| 0.065 | 15.508% | 0.001963584 | 0.01963% | 0.06928% | 0.02834% | 0.6564% |
| 0.070 | 7.601% | 0.003393034 | 0.03139% | 0.12146% | 0.06132% | 0.7202% |

At .070, cohesive A differs by 0.1422% and exported qcoh by 1.9309% in weighted L2. The phase difference is measurable despite close bulk stress and slip. At .065, axial added fracture changes from 2.60924e-6 to 1.32111e-5 J/m; early axial evolution is particularly timestep-sensitive.

| Region at .070 | Max delta phi, .005 | Max delta phi, .0025 | Window added fracture, .005 (J/m) | Window added fracture, .0025 (J/m) |
|---|---:|---:|---:|---:|
| central_band | 0.02523777 | 0.02555985 | 0.0082535775 | 0.0084059006 |
| old_tip_taper | 0.02093594 | 0.02116965 | 0.0012630009 | 0.001464188 |
| axial_ahead | 0.01463231 | 0.01419069 | 0.00012125494 | 0.00013455286 |
| remote | 0 | 0 | 0 | 0 |

### Absolute contour versus increment extension

| Half-step endpoint (s) | Absolute .05 right crossing, y=2 (m) | Advance from initial .05 contour (m) | Delta phi=1e-4 extension beyond x=6 (m) |
|---|---:|---:|---:|
| 0.0625 | 5.92185466 | 0.00000000 | 0.00000000 |
| 0.0650 | 5.92185466 | 0.00000000 | 0.05131977 |
| 0.0675 | 5.92903357 | 0.00717891 | 0.05552788 |
| 0.0700 | 5.93732497 | 0.01547031 | 0.06122409 |

The initial absolute crossing is x=5.92185466 m. At .070 the coarse crossing is x=5.93944124 m, versus x=5.93732497 m for the half-steps: 2.116 mm apart (0.0339 hx). Both remain inside the original endpoint x=6. The small-amplitude connected increment extends 0.05751/0.06122 m beyond x=6, respectively (0.920/0.980 hx). These are different diagnostics, not inconsistent definitions of a single tip. The absolute .05 level has moved within the old taper; it has not reached initially intact axial material.

Both y=2 native traces and y=2±hy/2 cuts are retained in CSV. Global cellwise 2D extents also remain inside x=6. Across the observations, changing contour subdivisions from 16 to 32 changes the rightmost extent by at most 2.094e-5 m. An independent exact quadratic circle gives a 2.287e-6 m extremum error at subdivision 32, within its interpolation bound; exact line roots agree to roundoff. This quantifies rendering accuracy only.

### Physical work and cost

| Quantity | dt=.005 | dt=.0025 |
|---|---:|---:|
| Added fracture, .060–.070 (J/m) | 0.00963783332 | 0.0100046415 |
| Signed work error, .060–.070 (J/m) | 2.18708361e-05 | 5.83914217e-06 |
| Accumulated absolute error, .060–.070 (J/m) | 2.18708361e-05 | 5.83914217e-06 |
| Added fracture, t=0–.070 (J/m) | 0.0125862096 | 0.0129530177 |
| Signed work error, t=0–.070 (J/m) | 3.36049261e-05 | 1.75732322e-05 |
| Accumulated absolute error, t=0–.070 (J/m) | 3.36049263e-05 | 1.75732324e-05 |
| Accepted-step solver time for window (s) | 646.118 | 1613.921 |
| Line trials for window | 52 | 156 |

The window remainder falls by 3.746×: from 0.2269% to 0.0584% of added fracture. Window added fracture differs by 3.666%. Signed and absolute window sums coincide; cancellation is not hiding a large balance defect. The full-history rows include the common inherited error through .060 (1.173409e-5 J/m); they are not a full-history refinement comparison.

The new solver times are 425.343, 286.859, 218.784 and 682.935 s. Together they cost 2.50× the two retained coarse intervals, with 40,779 versus 16,727 inner Stokes iterations. Step 16 required 104 line trials and three exact-search attempts; its phase-bound reclassification and half-length polishing remain in the log. There were no exhausted search failures or repeated active-set classifications reported by the retained counters. Accepted phase KKT ranges from 2.60e-12 to 7.00e-8; the final value is 5.03e-8. No tolerance was weakened.

The scheme ledger closes within 1.44e-14 J/m, but it includes the numerical phase remainder. The physical-work rows above deliberately exclude that remainder. At .0625 the axial added fracture is only 3.43e-8 J/m, below the interval physical-work error 1.05e-6 J/m. A small global balance error cannot certify the exact first axial onset or provide a local energy-error bound.

### Verification, artifacts and next decision

The restore nonmutation check and four single-publication checks pass. Independent analysis verifies native reconstruction, coefficient bounds/irreversibility, fracture totals and the discrete section-rate/power identity. Protected binaries, saved parent, older step-89 checkpoint and unrelated incoming source hashes are unchanged. `completion-verification.json` records the final checkpoint in `resolution60-halfstep-window/output/restart/02`, its hashes and accepted residuals. This task verifies restoration of the selected parent; it does not claim a separate final-checkpoint recovery run or new whole-step rejection test.

Figures (PNG, with matching PDF files and CSV data):

- [Four half-step profiles: phi0, phi and delta phi together](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_halfstep_window/halfstep-profiles.png).
- [Matched endpoint profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_halfstep_window/endpoint-profiles.png).
- [Phase maps and absolute contour at .065](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_halfstep_window/phase-fields-0014.png) and [.070](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_halfstep_window/phase-fields-0016.png).
- [Absolute-contour tracking, width and work](../../../benchmarks/reconstructed_fault/particle_free/evidence/resolution60_halfstep_window/contour-energy-history.png).

**Retain the specimen and the qualitative sequence, but do not assign a converged onset or growth rate.** The smaller timestep preserves connected axial damage and quiet remote material, improves physical work and leaves significant phase sensitivity. The next justified experiment is eight dt=.00125 intervals from the same accepted .060 parent through .070 on the same mesh, testing whether the phase and axial-energy differences decrease. It would still be local time qualification; spatial qualification and full-from-zero timing would remain separate. No further run is selected or launched here. Old peak-.95 continuation, longer growth and other campaigns remain deferred.
