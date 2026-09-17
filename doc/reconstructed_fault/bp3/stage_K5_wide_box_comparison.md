# Modified BP3: 100- versus 200-km box width

## Scope and comparison definition

One fresh four-rank Release replay of `bp3_modified_wide.prm` uses the saved
seven-step clock through 132230424.76671731 s (4.190129 years). The control is
the existing `fully-frictional-cleanup-local4` trajectory, not a rerun. Both
use the fully frictional continuous-Q1 fault, mature C=0, split aging,
corrected work measure, paired endpoint treatment, frozen ell=400-m phase
profile, fixed initial background, and unchanged plate speeds/tolerances.

The wide box is x=[-50,150] km, y=[0,100] km. Its center and the fault
midpoint remain (50,50) km. All 42,880 original bulk cells and all 1236
surface vertices are unchanged; 88 graded lateral cells extend the original
box. The lateral loading boundaries move outward; this intentionally changes
the finite-domain problem. It is not a mesh-convergence study.

The maintained inputs were relocated after the control run. The comparison
checks their contents by SHA-256, not filenames. Termination guards also
became explicit, but the actual accepted physical clock must match exactly.

## Quantities and lifecycle

Use `work_weak_k.csv` from each run. For traction T the reported nodal weak
average is

\[
 \overline T_i=\frac{\sum_q J_q\chi_qN_i(q)T_q}
                       {\sum_q J_q\chi_qN_i(q)}.
\]

These are the native mechanical-work weights, not particle-volume averages,
point samples or a reinterpreted mass-inverted Q1 field. T is current total
shear or normal traction from the accepted velocity/pressure and the frozen
FE history actually used by mechanics. Newly committed particle stress is
not reevaluated as if it were the preceding history. The production observer
checks the weak loads against mechanical assembly at every accepted state.

For each observable Q report all three quantities:

- initialization offset: D0 = Qwide(0)-Qnarrow(0);
- total difference: Dk = Qwide(k)-Qnarrow(k);
- difference of evolution: Ek = [Qwide(k)-Qwide(0)]-[Qnarrow(k)-Qnarrow(0)].

Subtracting D0 is diagnostic accounting, not removal of its physical effect.
Stress differences are compared to each run's stress changes since its own
initialization, never to the 50 MPa normal background. Pressure and minus
deviatoric normal traction are retained separately. Spatial RMS/mean values
integrate the linearly interpolated reporting profiles over physical
down-dip length, so dense node spacing does not receive extra weight.

## Reproduction and evidence

From the repository root:

```sh
python3 benchmarks/reconstructed_fault/bp3/run_research.py --configuration wide --output benchmarks/reconstructed_fault/bp3/wide-seven-local4
python3 benchmarks/reconstructed_fault/bp3/compare_box_width.py
```

The launcher uses four ranks, the existing 2400-s wall cap, and no retries.
It refuses to overwrite an existing run directory. Source diff, binary/plugin
hashes, effective parameters, input hashes, launch command and environment
are recorded in the run directory. The analysis produces
`width-comparison/{metrics.csv,profiles_0_40km.csv,stations.csv,summary.json,comparison.png}`.

## Initialization offsets over 0–40 km

| Observable, wide minus narrow | RMS | Maximum absolute |
|---|---:|---:|
| V/Vp | 5.97762e-8 | 1.07353e-7 |
| Weak shear traction | 0.0931921 Pa | 0.103376 Pa |
| Weak normal traction | 0.0920991 Pa | 0.156038 Pa |
| Accumulated slip | 0 | 0 |

The largest initial rate offset occurs at 13.60 km, the largest weak shear
offset at 8.60 km and the largest weak normal offset at the top endpoint.
Supplied initial Theta and the fixed background input file are identical.
The observer's mass-inverted weak-background output differs at about 1e-7 Pa
because of changed summation/projection order; this is not a changed input.
The completed Ih differs by at most 3.18e-13 relatively and the native weak
weights by at most 1.92e-13: physical mesh/fault preservation does not require
bitwise equality after changed MPI partitioning and reduction order. These
roundoff-level differences are reported, not repaired or renormalized.

## Completed short trajectory

The wide replay passed initialization and all seven real steps, ending at
4.190129311694087 years, with exactly the control's accepted time/dt sequence.
All 1236 nodes stayed free; there were no prescribed or lower-active nodes.
The wide run took 721.060 s on four ranks versus 704.962 s for the saved
control (these are separate executions, not a controlled timing benchmark).
Peak reported child RSS was 1,574,720 KiB, not aggregate MPI memory.

All 71 fresh linear checks passed. Newton update counts at steps 0–7 were
1, 1, 13, 11, 12, 11, 7, 7 in both cases; wide Krylov total was 1301 versus
1360. Final relative residuals were 1.898e-13 bulk and 4.817e-13 surface.
Every accepted state's final residuals met the unchanged 1e-8 criterion.
Independent split-aging verification had maximum relative error 2.22e-16;
accumulated slip matched the once-per-accepted-step update. Initial particle
stress remained zero. Inert H, frozen Ih and geometry checks passed. The
largest wide-run discrepancy between exported weak stress and mechanics was
6.583e-7 Pa, below the existing 1e-5-Pa observer check.

### Final comparison over 0–40 km

| Quantity | RMS wide-minus-narrow | Maximum absolute difference | Location of maximum |
|---|---:|---:|---:|
| V/Vp | 0.00485600 | 0.00776261 | 24.20 km |
| Accumulated slip | 0.421539 mm | 0.680959 mm | 23.50 km |
| Weak shear traction | 17.5349 kPa | 27.3340 kPa | 0 km |
| Weak normal traction | 19.2053 kPa | 47.1574 kPa | 0 km |

The wide box produces lower rates, slip and tractions over this interval.
Mean V/Vp is 0.485521 versus 0.489144; mean accumulated slip is 68.7219 versus
69.0357 mm. The rate maximum difference is 0.776% of Vp, **not** a claim of
0.776% relative accuracy at nearly locked shallow nodes. In the 0–15 km
region mean V/Vp is 1.75044e-6 versus 1.78619e-6; the physical shallow slip
deficit remains. At 40 km V/Vp is 0.965179 versus 0.969254, and accumulated
slip is 129.348 versus 129.685 mm.

### Stress differences measured against evolution

The following RMS scales use changes from **each run's own initialization**.
The initial sub-Pa offsets above are retained separately.

| 0–40 km quantity | Narrow evolution RMS | Wide evolution RMS | Difference of evolution RMS | Difference / narrow evolution |
|---|---:|---:|---:|---:|
| Weak shear | 250.998 kPa | 243.992 kPa | 17.5348 kPa | 6.99% |
| Weak normal | 62.3989 kPa | 46.4169 kPa | 19.2052 kPa | 30.78% |
| Pressure contribution | 51.1273 kPa | 35.8806 kPa | 17.6831 kPa | 34.59% |
| Minus deviatoric normal contribution | 24.0682 kPa | 18.3366 kPa | 6.99305 kPa | 29.06% |

The signed *mean* normal-evolution difference is -13.7961 kPa, composed of
-16.2292 kPa from pressure and +2.43312 kPa from minus tau:N. The normal
difference is not just a uniform shift: its mean-removed RMS is 13.3607 kPa.
At the top, the total normal difference -47.1574 kPa is approximately
-31.5001 kPa pressure plus -15.6573 kPa minus tau:N. No pressure shift was
introduced in analysis or mechanics.

The narrow/ wide normal-evolution ranges are respectively
[-9.326,231.916] / [-14.388,184.759] kPa. Shear-evolution ranges are
[-540.520,851.829] / [-554.507,829.497] kPa. Thus the same qualitative
shear-loading pattern persists, but normal feedback has substantial width
sensitivity relative to its evolving signal.

Regional detail prevents the large shallow normal response from hiding the
smaller interior signal:

| Region | Shear-evolution difference RMS | Fraction of narrow shear change | Normal-evolution difference RMS | Narrow normal-change RMS | Fraction |
|---|---:|---:|---:|---:|---:|
| 0–15 km | 24.7854 kPa | 11.59% | 30.4133 kPa | 101.679 kPa | 29.91% |
| 15–18 km | 20.5870 kPa | 3.25% | 12.0195 kPa | 1.97863 kPa | 607.47% |
| 18–40 km | 9.07672 kPa | 5.36% | 4.50106 kPa | 5.44579 kPa | 82.65% |

The 607% figure is not a huge absolute stress: it reflects the small normal
signal left by cancellation in the narrow-box transition region. There,
pressure and minus tau:N evolution RMS values are 24.287 and 25.073 kPa,
while their sum is only 1.979 kPa. The wide sum is 12.837 kPa. Reporting only
normal stress divided by 50 MPa would conceal this change.

### Development at common physical times

These are total RMS width differences; the CSVs also retain D0 and Ek.

| Step | Time (yr) | V/Vp difference | Slip difference (mm) | Shear difference (kPa) | Normal difference (kPa) |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 5.978e-8 | 0 | 9.319e-5 | 9.210e-5 |
| 1 | 0.0788151 | 2.836e-8 | 7.053e-8 | 4.475e-5 | 4.414e-5 |
| 2 | 0.157494 | 8.409e-5 | 0.000208831 | 0.306474 | 0.336351 |
| 3 | 0.307772 | 0.000253276 | 0.00140965 | 0.895747 | 0.985017 |
| 4 | 0.594802 | 0.000600683 | 0.00684849 | 2.03835 | 2.24285 |
| 5 | 1.14303 | 0.00129930 | 0.0293142 | 4.27036 | 4.69802 |
| 6 | 2.19014 | 0.00261020 | 0.115485 | 8.68250 | 9.54014 |
| 7 | 4.19013 | 0.00485600 | 0.421539 | 17.5349 | 19.2053 |

After the early retained-initial-state interval, the normal-evolution width
difference grows from 27.71% of the narrow change at step 2 to 30.78% at
step 7. The stress sensitivity is persistent, not a timestep-zero offset.

## Conclusion and limits

Widening preserves the short trajectory's overall rate/slip pattern but is
**not negligible for evolving stress**, especially normal traction. The final
rate difference is below 0.008 Vp and slip difference below 0.681 mm, while
normal-stress evolution differs by about 31% RMS overall and much more
relatively where the narrow normal signal cancels. These discrepancies are
orders of magnitude above initialization and observer roundoff.

The result supports treating the 100-km lateral boundaries as a material
finite-domain sensitivity for stress interpretation. It does not establish
that the 200-km box is domain-converged: only two widths were compared, and
the newly added far field uses graded coarse elements. No pure continuum
width limit or earthquake-cycle accuracy is claimed. No additional runs,
physics/solver changes, restart modifications or criterion changes were made.

The main visual summary is
`benchmarks/reconstructed_fault/bp3/wide-seven-local4/width-comparison/comparison.png`.
The raw log is `wide-seven-local4/run.log`; `execution.json` and the analysis
`summary.json` retain the complete convergence and history checks. Analytic
constant/linear checks of the reporting norm, Python syntax checking and
`git diff --check` passed; no broad test suite was run.
