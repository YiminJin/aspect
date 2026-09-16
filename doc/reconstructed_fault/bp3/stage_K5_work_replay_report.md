# K5: committing mature-fault work-measure replay

## Decision summary

The committing replay reached the requested 29.24190894-year state on four
ranks, with unchanged physical settings and solver criteria. The first finite
history update passed before continuation. At common physical mesh nodes,
pressure variation decreased 24.7% near 18 km and 44.2% near 40 km, but increased
3.2% near 15 km. The bottom improvement survived evolving histories. The
39.95-km rate depression and final-element slip gradient are smaller, not gone.
Common-rule weak normal-traction variations near 15/18/40 km barely changed;
raw pressure reduction alone does not demonstrate a normal-traction repair.

Thus the revised formulation/boundary package addresses an important contributor,
but does not explain away the RSF-transition and imposed-slip-junction features.
The next focus should be their remaining slip gradients and state/friction
transition response, not another boundary or weighting redesign.

One launcher limitation is explicit: roundoff in accumulated time generated a
final extra 2.384e-7-s step before the process could be stopped. It also converged,
but is preserved separately and **excluded from the trajectory comparison**.
All reported final comparisons below use saved step 10, not step 11. No rerun
was launched.

## Scope and implementation

This is the authorized fresh four-rank replay through 922804465.5975173 s
(29.24190894 yr), with a 2400 s process cap. It is not a restart conversion or
a continuation of the noncommitting qualification case. The benchmark parameter
`Postprocess/BP3/Committing work-measure replay = true` selects the already
qualified mechanical work measure, both completed boundary denominators and
in-box source continuations, mature friction, and ordinary history publication.
The default formulation and old noncommitting qualification switch are unchanged.

For each owned physical bulk Stokes quadrature point the mechanical equation is

\[
 R_i=\sum_q J_q\chi_q b_i
 [\tau_{\rm bg}+\Delta\tau:S-\mu(V,\Theta_{k-1})\sigma_n-\eta^d V],
 \qquad \sigma_n=50\,\mathrm{MPa}+p-\Delta\tau:N,
\]
\[
 \Delta\tau=2\kappa_b(\dot\epsilon-\chi V S)
                 +\beta_b\tau_{\mathrm{old,FE}}.
\]

All terms, derivatives and residual mass retain the qualified work measure.
The observer evaluates this equation with the frozen **working FE old history**
and accepted velocity/pressure. It does not feed newly committed particle stress
back through another Maxwell update. Nodal split aging, zero mature cohesion,
inert H, pressure treatment, solver criteria and timestep controllers are unchanged.
The supplied clock specifies comparison times; the controller remains allowed to
insert smaller steps. No additional trajectory or baseline rerun was launched.

## Comparison measures

The previous mature run used particle-domain-volume mechanical weights and
parent particle stress. Its native weak averages are **not** the same observable
as the new work-weighted, bulk-QP FE-history averages. The comparison therefore
contains four separately named observations:

1. `work_weak_k.csv`: new native J*chi mechanical weak integrals and mass row sums.
2. `common_fe_weak_k.csv`, compared with the saved baseline's `fe_q`, `fe_sigma`
   and `p` columns: the same parent-position FE sampling and domain-Q1 volume
   test weights, observational only. Each trajectory has its own advected
   particles/domains; this is a common rule, not frozen identical sample positions.
3. `work_qp_k_rank*.csv`: raw accepted constitutive stresses at physical bulk QPs,
   including positive-phase points outside the original finite-segment projection.
   These are not interchangeable with the baseline's raw particle-history extrema.
4. Existing `bulk_k_rank.vtu`: physical pressure at identical exported mesh nodes,
   compared directly (Float32 output). Published stress-history fields are not
   misidentified as current constitutive stress.

The CSV weak columns are integrals; divide by their `weight` column for a weak
mean. `comparison/profiles_k.csv` already performs that division. Element slip
gradients are exact differences of adjacent Q1 nodal accumulated slips divided
by element length; no smoothing or initial-error subtraction is used.

Windows in down-dip distance are 13–16.5 km (15-km transition), 16.5–20 km
(18-km transition), 37–43 km (junction), 59–61 km (interior control), and the
first/last 2.5 km (boundaries). Mesh-node pressure comparisons additionally
restrict normal distance to |r|<1200 m. Raw-QP exports use physical top/bottom
2-km windows and all positive phase; small differences in sampling windows are
not evidence of a change of physics.

## Verification and lifecycle

The requested sequence comprises initialization and ten real steps. All eleven
comparison times match the saved clock to at most 2.384e-7 s (floating-point
accumulation at 9.228e8 s); no controller-required subdivision occurred.
The first accepted update gate ran before the next step:

| Check | Result |
|---|---:|
| First update time and dt | 2487214.2056652424 s |
| Independent Theta relative error | 2.22045e-16, below unchanged 1e-12 |
| Frozen working old FE stress | exactly zero |
| Newly committed Maxwell stress versus independent first-step formula | max 5.37134e-8 Pa |
| New stress scale in that check | 5280.04087 Pa |
| Stable particle IDs with unchanged initialized H | 385920 |
| Mature C | exactly zero |
| Fault coordinates and completed I_h | exactly unchanged |

Across steps 0–10, 85 fresh linear-residual checks pass, totaling 1913 Krylov
iterations. The largest fresh/requested ratio is 0.99643949. Every accepted state
meets the separate 1e-8 relative bulk/surface criteria; worst final residuals
over these states are 5.46817e-10 and 9.834e-9. At step 10 they are
2.26754e-13 and 1.87773e-12. The independent Theta error never exceeds
2.22045e-16. Deep Vp is exact, all 440 RSF nodes remain free, and there are no
lower-active nodes. Accumulated slip satisfies its accepted-step recurrence.
The observer reproduces the actual frozen accepted weak traction to at most
4.52264e-6 Pa, despite summing it independently in postprocessing.

Initial Theta, C and zero slip are identical to the baseline. Completed endpoint
I_h changes its range from [6647.46205,12966.98696] m to
[12673.10116,12966.98696] m, with max nodal change 6043.76444 m. Initial V spans
[9.99274612e-10,1.00045943e-9] m/s instead of
[9.87250603e-10,1.01854248e-9]. The immutable prestress file is reused and hashed,
not recalibrated. These initial differences remain physically present throughout
the comparison; this is a package comparison, not isolation of work weighting
from boundary completion.

## Final mechanical response (step 10)

Pressure below is measured at **identical physical exported mesh nodes**, not
different native surface averages. Values are MPa.

| Window | Baseline p min/max | New p min/max | Peak-to-peak change |
|---|---:|---:|---:|
| top | 0.771837 / 1.065670 | 0.769558 / 1.033407 | -10.2% |
| 15 km | -3.221787 / 3.588877 | -3.282054 / 3.743959 | +3.2% |
| 18 km | -5.180225 / 5.596458 | -3.852813 / 4.262438 | -24.7% |
| 40 km | -4.883373 / 4.803925 | -2.736481 / 2.670427 | -44.2% |
| interior control | -0.070563 / 0.251707 | -0.069632 / 0.252606 | -0.01% |
| bottom | -20.669736 / 23.820610 | -0.151429 / 0.162532 | -99.3% |

The **common parent/domain FE-history observation rule** gives the following
peak-to-peak variations in kPa. Both trajectories use this same rule, distinct
from the new mechanical work measure.

| Window | sigma_n old → new | q old → new |
|---|---:|---:|
| top | 407.119 → 412.230 | 70.434 → 69.368 |
| 15 km | 85.549 → 87.010 | 1354.776 → 1123.354 |
| 18 km | 47.631 → 48.774 | 3364.447 → 2567.584 |
| 40 km | 16.430 → 16.237 | 720.406 → 647.457 |
| interior | 12.288 → 12.290 | 9.328 → 9.439 |
| bottom | 7959.598 → 8.389 | 2388.588 → 9.150 |

In particular, the 40-km common weak sigma_n still lies close to 50 MPa:
[49.993109,50.009538] → [49.993813,50.010051] MPa. This nearly unchanged weak
quantity coexists with a material reduction in raw pressure; the dipolar raw
field is strongly averaged. Signed p and -tau:N profiles are exported separately.
Native mechanical weak sigma_n variations near 15/18/40 km are respectively
84.444/47.212/16.668 kPa in the new run; their old native counterparts are
85.516/47.647/16.476 kPa, but those pairs use different measures and histories.

New **raw accepted constitutive bulk-QP** sigma_n ranges (MPa):

| Window | Minimum | Maximum |
|---|---:|---:|
| top | 51.083289 | 51.543466 |
| 15 km | 48.898130 | 51.241768 |
| 18 km | 48.629647 | 51.478873 |
| 40 km | 47.208131 | 52.673123 |
| interior | 49.807961 | 50.261617 |
| bottom | 49.536437 | 50.226106 |

These raw ranges are not the baseline's particle-history extrema. The bottom
remaining variation is modest but not zero, and the top retains a roughly
1–1.5 MPa positive normal-stress increment with the nearly arrested shallow
fault. Uniform-sliding boundary tests do not imply zero stress under nonuniform
evolving RSF slip. Positive-phase inactive-QP samples are retained in the raw
exports, rather than hidden; their |r| starts at about 790.6 m in the existing
normal support-tail region. No support expansion or tail renormalization was
made here.

Selected nodal changes:

| xd (km) | V old → new (m/s) | Theta old → new (s) | Slip old → new (m) |
|---|---:|---:|---:|
| 15 | 2.04208e-14 → 3.24664e-14 | 9.21585e8 → 9.21108e8 | 0.00250155 → 0.00250673 |
| 18 | 3.81265e-10 → 2.92663e-10 | 2.09828e7 → 2.73352e7 | 0.448794 → 0.331841 |
| 25 | 6.12812e-10 → 5.95563e-10 | 1.30546e7 → 1.34327e7 | 0.647736 → 0.633106 |
| 39.90 | 8.09950e-10 → 9.57303e-10 | 9.87715e6 → 8.35681e6 | 0.804488 → 0.890417 |
| 39.95 | 7.29546e-10 → 8.39369e-10 | 1.09657e7 → 9.53097e6 | 0.754548 → 0.816117 |
| 40 | 1e-9 → 1e-9 | 8e6 → 8e6 | 0.922804 → 0.922804 |

The last RSF element's slip gradient decreases from 0.00336514 to 0.00213375
(36.6%). Maximum |slip gradient| decreases from 6.69657e-4 to 2.20960e-4 near
18 km (67.0%), but increases from 1.19661e-4 to 1.65614e-4 near 15 km (38.4%).
Thus the improvement is spatially selective and accompanied by changed
state/rate evolution; it is not evidence that all concentrations were a sampling
artifact. The nonuniform slip field still carries the 40-km imposed-slip junction.

## Cost and end-time limitation

One four-rank launch finished in **967.306 s (16.12 min)**, below the 40-minute
cap, including the extra remainder step and diagnostic output. Recorded child
peak RSS is **1536008 KiB (1.465 GiB)**; aggregate four-rank peak was not measured.
The existing timer reports 547 s in condensed linear solves, 25.2 s total fault
property preparation, and 17.9 s postprocessing. These scopes overlap with their
subscopes and are not an additive full timing decomposition. No performance
campaign was undertaken.

Accumulated time at step 10 was 922804465.59751701 s versus the requested
922804465.59751725 s. The ordinary end-time controller subsequently generated
dt=2.384185791015625e-7 s. Step 11 completed before the attempted stop took effect.
Its different Maxwell interval changes V despite negligible elapsed time
(e.g. min free V 7.76891e-17 → 2.93919e-17 m/s), so it is **not** an equivalent
replacement for step 10. It passes its own nonlinear/history checks, but is
preserved as a launcher/end-time artifact, not interpreted as another physical
comparison step or used to claim stronger verification. Total including it:
12 states and 93 passing fresh linear checks. No automatic retry occurred.

For a future continuation, select a known accepted physical state explicitly;
do not blindly use the final step-11 checkpoint. A benchmark-level accepted-clock
termination guard should be added before reusing this runner. No production
stopping rule was changed in this task.

## Evidence and recommended next action

All files below are under `benchmarks/reconstructed_fault/bp3/work-replay-50-local4/`:

- `run.log`, `execution.json`, `provenance.json`, `accepted_steps.csv`;
- `first_update_maxwell.csv`, `history_k.csv`, `fault_k.csv`;
- `work_qp_k_rank*.csv`, `work_weak_k.csv`, `common_fe_weak_k.csv`;
- `comparison/summary.json`, `regions.csv`, `mechanics.csv`, `selected_nodes.csv`,
  `pressure_common_vertices.csv`, `raw_qp.csv`;
- `comparison/profiles_10.csv`, `slip_gradient_10.csv`, and
  `final_transitions.png`, `final_junction.png`, `final_top.png`, `final_bottom.png`.

Build with -j4, parameter validation, Python syntax checks, and the offline full
comparison all pass. Existing derivative/work qualification is reused, not rerun;
no new generic solver, MPI comparison, restart test, or long continuation is
claimed. The new explicitly selected replay mode is fresh-start-only.

**Recommendation:** retain the qualified work measure and paired boundary
treatments. Focus the next bounded scientific diagnosis on RSF transition/state
evolution and remaining slip gradients, particularly the sharp prescribed-Vp
junction. The present evidence supports a partial mechanical improvement, not
complete removal of interior concentrations or a unique attribution between
weighting, FE stress sampling and boundary completion. No further solve is
needed to make that decision.

## Reproducibility and changed files

New benchmark files are `work_replay.h`, `run_work_replay.py` and
`analyze_work_replay.py`. `bp3.cc` adds the explicit selector and routes only this
mode through its observational exports and first-update gate. The current design
and specification record this separately authorized committing use. No core
production implementation was changed for this replay.

Commands (preparation refuses to overwrite an existing attempt):

```sh
cmake --build benchmarks/reconstructed_fault/bp3/build -j4
python3 benchmarks/reconstructed_fault/bp3/run_work_replay.py prepare
python3 benchmarks/reconstructed_fault/bp3/run_work_replay.py run
MPLCONFIGDIR=/tmp/aspect-work-replay-mpl python3 benchmarks/reconstructed_fault/bp3/analyze_work_replay.py
```

The runner records binary/plugin/source/input hashes and exact MPI command in
`benchmarks/reconstructed_fault/bp3/work-replay-50-local4/provenance.json`.
The archive copy of `mature-fault-50-local4` was restored for offline comparison;
it was not rerun. The first launcher preflight discovered that `/usr/bin/time`
is absent before starting MPI; its artifacts remain in `launch-preflight-no-time`.
The launcher now records Python child peak RSS, explicitly not aggregate RAM.
The actual four-rank simulation was launched once. Analysis-only repairs replaced
an old junction-only pressure reader with a full physical-node reader; they did
not alter or repeat the simulation.
