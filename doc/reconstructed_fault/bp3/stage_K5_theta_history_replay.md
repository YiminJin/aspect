# K5: consistently evolved fixed-coordinate Theta, 50-m replay

## Completed result

**Consistent fixed-coordinate aging delays contact by one saved step, but
does not eliminate it or the stress concentration.** Original nodal aging A
first places node796 (39.95 km) at its lower bound in step12, t=68.619647 yr.
The independent functional state B leaves it free there at 0.020031 Vp;
it reaches the bound in step13, t=70.733401 yr. The difference between these
first observed accepted contacts is 2.113755 yr. These are discrete accepted
states, not a temporally resolved estimate of a continuous crossing time.

At the common final state, the bound reaction is 1.098618 MPa in A and
**0.069573 MPa in B**, a 93.67% reduction. The 39.9-km neighboring rate is
68.43% larger. Nevertheless, a pronounced last-node depression persists,
and the last-element accumulated-slip gradient is only 2.85% smaller.
The junction tensile minimum remains **−9.658362 MPa**, versus −10.711552 MPa
in A. The more negative global minimum is at the deep tip, not the junction.

This establishes a material state-discretization sensitivity in the contact
margin and rate profile. It does not establish that nodal aging alone causes
the accumulated-slip concentration, or that the functional alternative repairs
the hard free/prescribed junction. No new model, tolerance, mesh, support or
pressure change is made on the strength of this result. Stop here for review;
the behavior beyond the common final time is untested.

## Matched trajectory and onset

Both runs use the same 42,880-cell bulk mesh, 1,236 fault vertices, all 14
accepted states (initialization plus steps1–13), and the exact recorded clock.
Final physical time is 2,232,176,379.2516127 s; dt13 is 66,705,021.523934364 s.
Years below use 365.25 days. Original and alternative initial Theta, C, I_h
and background traction agree exactly. Maximum initial V difference is
6.20385e-25 m/s. No later baseline state is copied into B.

| Step | Time (yr) | A: last V/Vp | B: last V/Vp | A: last slip gradient | B: last slip gradient |
|---|---:|---:|---:|---:|---:|
| 6 | 2.190142 | 0.810109 | 0.810576 | 0.000167326 | 0.000166999 |
| 7 | 4.190129 | 0.667353 | 0.669693 | 0.000587224 | 0.000583945 |
| 8 | 8.010105 | 0.489379 | 0.496547 | 0.001818323 | 0.001797762 |
| 9 | 15.306257 | 0.306602 | 0.322081 | 0.005011409 | 0.004919566 |
| 10 | 29.241909 | 0.152487 | 0.178448 | 0.012465718 | 0.012145538 |
| 11 | 55.859004 | 0.044870 | 0.080395 | 0.028511364 | 0.027594374 |
| 12 | 68.619647 | **1e-11, bound** | **0.020031, free** | 0.036565269 | 0.035486952 |
| 13 | 70.733401 | **1e-11, bound** | **1e-11, bound** | 0.037899370 | 0.036821053 |

Here Vp=1e-9 m/s, Vmin=1e-20 m/s. The last gradient is
`(slip(40 km)-slip(39.95 km))/50 m`, an exact derivative of the accepted Q1
accumulated-slip field, in m/m. Diagnostic slowdown levels 0.9, 0.5 and
0.1 Vp are first crossed at the same accepted steps6,8,11 in both runs.
Thus the early onset is not shifted at this clock's resolution; the terminal
approach to the bound changes. The 0.01 Vp level is first crossed at step12
in A and step13 in B. These levels describe the history, not new acceptance
thresholds.

Final accumulated slip at 40 km is exactly 2.232176379 m in both runs.
At 39.95 km it is 0.337207903 m in A versus 0.391123752 m in B:
an additional 0.053915850 m, not a removal of the physical slip gap.
The penultimate-element gradient changes from −0.011029385 to −0.010412559.
The maximum absolute gradient in the measured 39–40.5-km window remains on
the last element. All accepted slip increments pass `slip_k=slip_{k-1}+dt_k V_k`.

| Final coordinate | A rate (m/s) | B rate (m/s) |
|---|---:|---:|
| 40.00 km, prescribed | 1e-9 | 1e-9 |
| 39.95 km, lower-active | 1e-20 | 1e-20 |
| 39.90 km | 1.80187504e-10 | 3.03485314e-10 |
| 39.85 km | 3.93101036e-10 | 3.50217600e-10 |
| 39.80 km | 4.54780187e-10 | 4.57769631e-10 |

## Raw stresses: junction versus deep tip

Normal-stress extrema below are actual production constitutive samples,
`sigma_n=50 MPa+delta_p-delta_tau:N`. The source retains each rank's extrema
separately for free/free, prescribed/prescribed, and mixed-basis elements;
collecting these gives the extrema of each class. They are not projected
nodal tractions or bulk-column means.

| State | A junction minimum (MPa) | B junction minimum (MPa) |
|---|---:|---:|
| 11 | +3.612587 | +4.328824 |
| 12 | −8.807228 | −7.829090 |
| 13 | −10.711552 | −9.658362 |

These minima are on segment795, which has one prescribed and one free basis
function. At the final state, the same real parent250724 gives the minimum:

- A: xd=39997.673692 m, xi=.046526158, parent
  (59040.862847,65274.418576) m;
  **50 − 52.697742 − 8.013809 = −10.711552 MPa**.
- B: xd=39997.671065 m, xi=.046578707, parent
  (59040.863719,65274.417999) m;
  **50 − 52.167200 − 7.491163 = −9.658362 MPa**.

The constitutive pressure and tau:N contributions change by approximately
+0.530542 and +0.522647 MPa respectively in this signed decomposition.
The sample remains on the free side of 40 km, with mixed free/prescribed Q1
support. The small change of particle positions and quadrature coordinates
is retained; this is an evolved comparison, not a fixed-particle substitution.

Final mixed-element maximum sigma_n changes from 109.378260 to 108.317826 MPa.
The adjacent free/free class minimum is also still tensile:
−1.209408 → −0.666052 MPa, at xd≈39949.688 m on segment796.
Thus tension is not confined to the prescribed side in either trajectory.

The **global** minimum is instead on the deep prescribed tip at
xd≈115.470 km: −18.782967 → −18.782936 MPa (only about 30.75 Pa difference).
It must not be used to obscure the approximately 1.05-MPa junction change,
nor mistaken for the 40-km minimum. Its B decomposition is
50 − 24.700237 − 44.082699 = −18.782936 MPa.

Independent raw bulk pressure samples are extracted at the same Q1 VTU
vertices in 39–40.5 km, |normal distance|<=1500 m. They are a common discrete
sample set, **not** pressure extrema inferred from sigma-selected samples:

| Step | A pressure range (MPa) | B pressure range (MPa) |
|---|---:|---:|
| 11 | [−43.822756, 41.726708] | [−43.364688, 41.208812] |
| 12 | [−54.732620, 51.895440] | [−54.159940, 51.238524] |
| 13 | [−56.109112, 53.144524] | [−55.498028, 52.627320] |

The final pressure peak-to-peak amplitude decreases by about 1.03%, while
the extrema remain at the same bulk vertices: minimum
(59033.203125,65283.203125) m; maximum (58691.40625,65478.515625) m.
These values have the existing VTU output precision. No pressure or stress
field was smoothed, relabeled, or substituted in ordinary output.

## Verification and artifacts

The completed four-rank Release replay took **1551.31 s (25.86 min)**.
Every accepted state satisfies both unchanged 1e-8 nonlinear criteria.
All **112 reported fresh-linear checks** pass, with worst fresh/target=0.98527.
Final relative bulk/surface residuals are **4.13658e-14 / 1.35803e-14**.
At the final state there is one lower-active node and 439 free RSF nodes;
the prescribed region remains exactly Vp.

Independent checks of **1,120 saved B constitutive samples** in the pure-VS
RSF region reproduce mu using the correct preceding functional history;
the maximum error across A/B checks is **2.11e-15 absolute**. The initialized
function, all 13 accepted V/dt records, nodal endpoint states, and accepted
slip updates are checked against the actual run files. Fault coordinates,
background tractions, I_h and mesh guards pass throughout. Standard nodal
Theta aging checks also pass; this is a shadow diagnostic, not the between-node
mechanical state in B. The functions are not advanced during Newton trials.

The focused C++ helper test passes initialization/no artificial aging, two
recursive updates, new-coordinate queries, exact nodal endpoints, memoization,
a decisive noncommutation example, and wrong-clock rejection. Both Release
targets build with -j4; Python compilation and `git diff --check` pass.
No extra MPI-size, solver-tolerance, spatial, or temporal campaign is run.
The frozen one-step comparison from the preceding task is not reused as the
new trajectory's history.

Execution and analysis:

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
g++ -std=c++17 -O2 benchmarks/reconstructed_fault/bp3/test_theta_history.cc -o /tmp/test-theta-history
/tmp/test-theta-history /tmp/test-theta-history.txt
python3 benchmarks/reconstructed_fault/bp3/run_theta_history.py
python3 benchmarks/reconstructed_fault/bp3/analyze_theta_history.py
```

Data root: `benchmarks/reconstructed_fault/bp3/theta-history-50-local4/`.

- `run.log`, `execution.json`, `provenance.json`: exact command/environment,
  numerical convergence and source/binary/plugin/input hashes.
- `theta_function_history.txt`: immutable initial geometry/Theta followed by
  the alternative trajectory's own accepted V/dt records.
- `comparison/series.csv`, `nodes.csv`, `slip_gradients.csv`: all common times,
  nearby rates/reactions, and signed accumulated-slip gradients.
- `comparison/functional_state_profiles.csv`: separate mechanically used
  preceding state, newly committed functional state and Q1 nodal shadow.
- `comparison/raw_sigma_extrema.csv`, `raw_support_extrema.csv`, `summary.json`:
  raw extrema, identities, pressure decompositions, verification summaries.
- `comparison/comparison.png`: rate/onset, gradient, junction/global stress
  and final rate profiles. Lines connect accepted samples only.
- Unmodified ordinary `fault_k.csv`, station files, bulk VTUs, stress samples
  and weak histories remain available alongside the separate analysis.

The initial MPI sandbox launch failed before ASPECT started; it is preserved
in `theta-history-50-sandbox-launch`. The subsequent run was interrupted by
the reported power loss during step5, after accepted states0–4; its outputs
are preserved in `theta-history-50-power-interrupted`. All source/binary/input
fingerprints still matched after recovery. Because the experimental functional
history is deliberately not checkpoint-integrated, the completed replay
restarted **from the original initial state**, not from an inconsistent
history/checkpoint pair. No failed physical solve was retried or retuned.

Implementation is opt-in: a file-local functional-history evaluator and the
point-state override in `source/material_model/`, BP3-local initialization/
accepted-publication helpers, one runner, one analysis script, and a focused
test. Ordinary production nodal aging is unchanged when the diagnostic flag
is unset. This mode remains **fresh-start only**, with an explicit restart
guard; it is not a general state-storage or restart implementation. Existing
unrelated working-tree changes are preserved. No commit or longer continuation
was requested or performed.

## Bounded experiment definition

User-authorized alternative to the specification's nodal aging rule. Run
initialization and steps1–13 on the existing 50-m case, four ranks, using its
recorded accepted timestep sequence. Do not continue past the saved comparison
time. All bulk, fault geometry, support, pressure, friction-law parameters,
history-transfer choices and solver tolerances remain unchanged.

The alternative is a function, not another nodal field:

\[
T_0(s)=I_{Q1}\Theta_0(s),\qquad
T_k(s)=U(T_{k-1}(s),I_{Q1}V_k(s),\Delta t_k),
\]

where U is the existing production exact aging update. Mechanics k consumes
T_{k-1}; initialization consumes T0 without aging through the artificial
initial interval. The update is applied throughout the RSF region (and
uniformly extended through the prescribed region, where it leaves the steady
state unchanged). No last-two-element substitution is made in this replay.

The finite accepted history of Q1 rate functions and timesteps is an exact
functional representation of T. Evaluating their composed aging updates at
any fixed coordinate gives that coordinate's independently evolved history,
including new quadrature coordinates after particle motion. Only the original
Theta0 is read. Later production nodal Theta is **never** used to reconstruct
this function. Memoization within a mechanical step changes no values; the
cache is discarded when the accepted-history index changes. Nothing is
projected back to Q1. Thus there is no added state-grid interpolation error.

The ordinary nodal state continues to evolve as a shadow diagnostic. At
vertices it agrees with the functional state because the rate interpolation
is exact there. It is not used by pointwise residual/K/G evaluation in this
opt-in run. Existing `fault_k.csv` Theta values remain valid vertex states;
interpolated station Theta outputs remain the Q1 shadow, not the between-node
functional state. Functional profiles must be identified separately.

An immutable initial-state file plus append-only accepted V/dt records supplies
each rank's evaluator. A record is appended only at the end of accepted-state
postprocessing, after the production convergence/history checks. Mechanics k
requires exactly k−1 update records. Rejected trials never append records.
The prior records define the retained function even if quadrature moves.
Geometry is checked against stored coordinates on each load. Failed writes
are diagnosed collectively. This experimental file is **not** checkpointed;
restart is explicitly rejected for this mode. Original baseline checkpoints
are untouched.

Tests before replay cover initialization (no artificial aging), two recursive
updates, newly queried coordinates, exact nodal endpoints, cache reuse, a
decisive difference from updated nodal interpolation, and wrong-clock rejection.
The helper calls the production aging routine in mechanics; no second Dc or
constitutive formula is introduced there. The earlier frozen-state K/G checks
remain applicable because T is independent of current mechanical unknowns.

Expected cost: approximately the saved 23-minute four-rank trajectory, plus
coordinate-history evaluation. One fresh replay, no automatic retry, no
parameter changes or further convergence campaign. The existing replay cap
also retains the other timestep-controller guards; a shorter required step
will stop the matched comparison instead of silently changing the clock.

Comparison: contact-node V/reaction versus time, onset of slowdown and first
bound state; accepted-slip element gradients; complete weak loads and selected
raw constitutive extrema near the junction; initial-condition agreement;
fresh-linear/nonlinear convergence and split-history checks. No claim of
trajectory equivalence follows merely from matching a late stress sign.
