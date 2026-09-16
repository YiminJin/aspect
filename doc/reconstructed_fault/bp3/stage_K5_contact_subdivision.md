# K5 fixed-mesh first-contact timestep subdivision

Consolidated synthesis of these four follow-up tasks:
[history, spatial-resolution and timestep report](stage_K5_history_resolution_consolidated_report.md).
This individual report is retained as the chronological evidence record.

## Result and decision

The four-rank replay passed all four subdivided steps and stopped at the
original final time. Lower contact at 39.9 km persists. The final reaction
changes by **-2.21%**, but the first common endpoint reaction changes by
**+129.35%**, and the final 39.8-km rate is **8.69% lower**. Thus the contact
decision is robust in this comparison; the neighbouring trajectory and
contact transient are not established as timestep-converged.

The junction raw tensile minimum changes only from **-5.402 to -5.381 MPa**.
Timestep subdivision does not remove the stress concentration or justify
replacing particle history by FE history. All initial fields and histories
are identical at the branch point, so this comparison has no spatial
initial-projection confound.

**Recommended next action:** retain particle-history mechanics; if quantitative
junction rates/slip are required before resuming the first-event run, obtain
one further subdivision of this same fixed-mesh window to test contraction.
Do not call the present two levels temporal convergence or introduce a
history-transfer/solver correction from these results. No additional level,
production correction, or long continuation was run here.

## 1. Actual accepted times and verification

The common branch point is accepted step 11, `1762776093.4253557 s`.
The checkpoint contains the complete accepted state, including accumulated
slip and background traction. Retiming affects only its pending clock, as
specified below. The loaded mesh has **42880 cells / 1518040 DoFs**.

| Physical time (s) | Original step / dt (s) | Subdivided step / dt (s) |
|---:|---:|---:|
| 1964123725.5765171 | — | 12 / 201347632.15116143 |
| 2165471357.7276783 | 12 / 402695264.30232239 | 13 / 201347632.15116119 |
| 2198823868.4896455 | — | 14 / 33352510.761967182 |
| 2232176379.2516127 | 13 / 66705021.523934364 | 15 / 33352510.761967182 |

The first half-step is below the accepted original step-11 controller cap.
Subsequent requested steps equal the production manager's selected steps;
the replay guard retained the convection, RSF and generic restrictions.
All four states have **399 free / 1 lower-active** free-region nodes;
deep prescribed nodes retain Vp. The first accepted contact is already at
the new midpoint. This brackets accepted-state contact more finely; it is
not an exact physical onset time. Lines connecting plot markers are guides,
not reconstructed continuous-time trajectories.

| Subdivided step | Newton updates | Krylov iterations | Final relative bulk / surface | Surface RMS (Pa) | Min accepted alpha |
|---:|---:|---:|---:|---:|---:|
| 12 | 5 | 239 | 1.068e-13 / 1.107e-9 | 0.00236722 | 0.3671724 |
| 13 | 4 | 220 | 1.857e-13 / 1.106e-13 | 2.30205e-7 | 1 |
| 14 | 4 | 173 | 2.892e-14 / 1.749e-12 | 2.06382e-6 | 1 |
| 15 | 3 | 140 | 1.015e-13 / 3.099e-10 | 0.000365366 | 1 |

All **39 fresh-linear checks** pass; worst fresh/target is **0.954679**.
Total Krylov count is **772** (original two intervals: 459). The final absolute
bulk residual is `6.36370e-5`, below its unchanged target `6.26982`; both
nonlinear relative residuals independently satisfy `1e-8` as well. No failure
exit or lifecycle-only message was counted as convergence.

Independent double-precision exact-aging and slip-accumulation checks pass
at every new accepted state and every fault node, alongside the benchmark's
extended-precision Theta assertion. Maximum independent relative Theta
error in the new branch is `2.210e-16`, below the unchanged `1e-12` criterion.
Fault coordinates and background tractions are bit-identical to step 11;
I_h agrees within `1e-12` relative after restart/cache reconstruction. The
production executable hash is identical to the baseline. Source checkpoint
hashes remain unchanged. No new rollback or MPI-rank-count campaign was run;
existing rollback coverage is retained, not re-claimed as newly tested.

## 2. Bound reaction versus split history

All reactions and terms below use the **unreplaced physical weak row divided
by its positive test weight**, not an M-inverse projected nodal residual.
`R = q - C - friction - damping`; lower reaction is `-R`.

| Quantity at 39.9 km | First common time: original / half | Final time: original / half |
|---|---:|---:|
| V (m/s) | 1e-20 / 1e-20 | 1e-20 / 1e-20 |
| Reaction (MPa) | 0.312463 / 0.716640 | 0.957235 / 0.936078 |
| Theta used by mechanics (s) | 170875264.514 / 372222896.597 | 573570528.629 / 606923039.366 |
| Committed Theta (s) | 573570528.629 / 573570528.629 | 640275550.102 / 640275550.102 |
| Accumulated slip (m) | 0.342264124670 / 0.342264124670 | 0.342264124671 / 0.342264124671 |
| Committed C (MPa) | 1.371222 / 1.384318 | 1.386585 / 1.414096 |

The different mechanics-input Theta is the intended split ordering: each
mechanical solve consumes the preceding committed state, then updates it.
At the lower bound, `x=V*dt/Dc` is about `5e-10` for the original first
interval, so the exact aging law is essentially Theta_old + dt. Subdivision
therefore leaves the node's **committed** Theta equal at common times, but
exposes an intermediate aged state to the next mechanical solve. It does
not represent a second computational history lag or a cancellation failure.
The original first interval has `dt/Theta_old=2.35666`; a tiny V-based x does
not make the relative aging increment small.

At the first common endpoint, half-minus-original weak terms (kPa) are:
shear **-0.140**, cohesion **-3.164**, friction **+407.200**. They account for
the reaction increase **+404.177 kPa**. This is a friction-dominated split
response, not a large normal-stress change. We did not separately freeze
Theta or neighbouring V to attribute all of it to one coefficient.

At the final time the same differences are shear **+13.357**, cohesion
**+3.122**, friction **-10.922 kPa**, giving reaction **-21.157 kPa**.
Friction need not increase merely because this node's input Theta increases:
the nonlinear domain/Q1 response also contains changed neighbouring rates
and states. Nonzero weak damping at the bound node similarly includes its
nonzero-rate neighbours.

Final half-step weak driving/cohesion/friction are respectively
**28.339904 / 2.478174 / 26.797808 MPa**. Its test-weight mean normal stress
is **+50.132912 MPa**, compressive. Do not interpret raw tensile samples as
the cause of lower contact.

Frozen constrained-FE substitution would change the final physical residual
by **+12.876 kPa**, only **1.376%** of the reaction. It cannot release the
node; the substituted reaction remains about **0.923 MPa**. This is diagnostic
evaluation at the same solution, not a second FE-history mechanical solve.

## 3. Neighbouring solution and stress concentration

| Final quantity | Original | Half steps | Change |
|---|---:|---:|---:|
| V at 39.8 km (m/s) | 2.849348e-10 | 2.601753e-10 | -8.690% |
| Theta at 39.8 km (s) | 27213612.46 | 29739191.95 | +9.281% |
| Slip at 39.8 km (m) | 1.17472277 | 1.15074863 | -0.02397414 m |
| V at 39.7 km (m/s) | 6.543704e-10 | 6.517528e-10 | -0.400% |
| V at 25-km control (m/s) | 2.206020e-10 | 2.156919e-10 | -2.226% |
| Slip at 25-km control (m) | 0.71175503 | 0.70856280 | -0.00319223 m |

At the first common time the 39.8-km rate differs by **-22.359%**; its final
smaller discrepancy is time evolution, **not** a timestep contraction factor.
The half-step branch's final node C differs by +1.984%; the comparison is not
an aging-only scalar test because all mechanics and histories are re-evolved.

Consistent Q1 fields over 37--43 km:

| Final field | Original peak-to-peak (kPa) | Half peak-to-peak (kPa) | Difference RMS (kPa) |
|---|---:|---:|---:|
| delta p | 101.796 | 91.610 | 1.797 |
| -delta tau:N | 62.933 | 62.837 | 1.044 |
| sigma_n | 132.882 | 132.265 | 1.071 |

At the first common time, sigma_n difference RMS is **1.393 kPa**. The final
1.071-kPa change is much smaller than the previous matched spatial change
of 34.735 kPa; it does not account for that spatial uncertainty.

Raw constitutive junction minimum at the final time:

| | Original | Half steps |
|---|---:|---:|
| Down-dip location (m) | 39983.561696 | 39983.552141 |
| delta p (MPa) | -51.185246 | -51.303968 |
| -delta tau:N (MPa) | -4.216621 | -4.076823 |
| Total sigma_n (MPa) | -5.401867 | -5.380791 |
| Junction particle-evaluation tensile weight (m²) | 6622.701 | 6357.792 |
| Junction FE-evaluation tensile weight (m²) | 5827.979 | 6092.888 |

Both minima involve parent particle 250740 on the mixed free/prescribed Q1
junction segment; their projected coordinates are on the free side. Opposing
p and tau:N changes largely cancel. Raw extrema are selected from production
rank/support extrema, whereas tensile weights use the full production weak
accumulation (not just selected samples). The global minimum, at the bottom
rather than this junction, changes from **-18.783 to -20.485 MPa**. That
separate unresolved endpoint sensitivity is not repaired by this comparison.

## 4. Execution and recoverable artifacts

Successful replay: `benchmarks/reconstructed_fault/bp3/junction-contact-half-qualified-local4/`.
Wall time **375.920 s**; ASPECT's current-run timer is 374 s. Its larger
"including restarts" time includes the original checkpoint's accumulated
wall time and is not this replay's cost. Peak memory was not measured.

The first launch in `junction-contact-half-local4/` stopped after **1.505 s**
before checkpoint loading: the runner omitted the saved-mesh target filename,
which that plugin requires even on restart. Its log is preserved. The runner
was corrected to pass the unchanged baseline target list; the successful
launch used a fresh output directory. No mechanical failure was retried.

Commands:

```sh
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_contact_subdivision.py prepare --tag qualified
python3 benchmarks/reconstructed_fault/bp3/run_contact_subdivision.py run --tag qualified
python3 benchmarks/reconstructed_fault/bp3/analyze_contact_subdivision.py
```

The driver refuses existing output/log targets. Replay uses four ranks with
explicit core binding, sparse B/G, pivoted tridiagonal inversion and diagnostic
CSV flags, with no FE-history override. Relevant artifacts are `run.log`,
`checkpoint_retime.json`, `provenance.json`, `execution.json`, `sequence.csv`,
`subdivision_comparison.json`, `node_history.csv`, `contact_history.png`,
`final_junction_profiles.png`, and `analysis/step*/` plus the raw source CSVs.
All source checkpoint files are protected by before/after hashes. Rebuilding
the retimed copy from the script is possible even after its rolling checkpoints
have advanced. Archive-layout checks deliberately restrict that operation to
the verified local BP3 checkpoint; it is not a portable archive-editing API.

Changed files: two benchmark-only output conditions in `bp3.cc`, the new
`run_contact_subdivision.py` and `analyze_contact_subdivision.py`, this report,
and a follow-up link in the matched spatial report. Python compilation and
`git diff --check` pass. No production source or authoritative equation was
changed, and no full test suite was run. Changes remain uncommitted.

## Predeclared comparison

Start from the accepted refined step-11 history in
`junction-matched-qualified-local4/refined/restart/03`. Hold the 42880-cell
mesh, fault grid, physics, initialization, support, full I_h, pressure,
particle-history mechanics and all solver tolerances fixed. Compare against
the completed refined trajectory, not against the coarse spatial level.

Halve each of the two remaining accepted intervals. This gives four solves
instead of two and retains both original interval endpoints. The final time
remains 2232176379.2516127 s. The production convection and reconstructed-fault
controllers remain active alongside the benchmark replay cap; stop if either
requires a smaller step. No additional timestep level or continuation is
included in this comparison. Estimated cost: 6--12 minutes, roughly 6 GiB
summed rank memory, four MPI ranks, one job with explicit core binding.

ASPECT saves checkpoints **after** `advance_time`: the pending clock says
step 12 even though the solution and histories are accepted step 11. The
disposable archive changes only the serialized pending `time` and `time_step`
to the first half-step. `old_time_step`, step number, all solution vectors,
particle fields, fault properties, accumulated slip and postprocessor state
remain identical. The script checks the unique known binary clock layout,
verifies a byte-identical inverse replacement, hashes every source checkpoint
file, and leaves the source untouched. This is a benchmark-only clock edit,
not a production restart or history change.

At common times compare the actual weak-row lower-bound reaction at 39.9 km,
the free/active transition, neighbouring rates, Theta used and committed,
cohesion, slip, and raw/consistent-Q1 pressure and traction. Preserve
particle/constrained-FE diagnostic differences. Require genuine nonlinear
and fresh-linear convergence and the existing 1e-12 Theta audit. Initial
projection differences are absent between these two branches because their
complete accepted state is identical at the branch point.

The question is whether subdividing the first-contact/split-aging interval
materially changes contact and the final reaction. A change demonstrates
timestep sensitivity, not a defect in the exact aging formula or justification
for a history-transfer change. Two timestep levels cannot establish a limit.

Only benchmark output selection is extended: a replay sequence requests
per-accepted-step profile and dual-history CSVs just as the matched clock
already does. Ordinary graphical output and production mechanics are unchanged.
