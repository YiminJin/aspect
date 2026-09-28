# Fresh small-step startup and bound-contact diagnosis

This follow-up preserves the failed 4e6-s startup and all its physical inputs.
It separates a fresh-start accuracy check from an observational replay of the
failure. No bound, active-set formula, nonlinear/linear tolerance, Armijo
budget, constitutive equation or history update is changed.

## Definition of the two checks

1. Start fresh with the identical weak initial state and a 4e6-s **artificial**
   Maxwell initialization interval. Take three physical 300-s steps to 900 s;
   compare a separately initialized run with six 150-s steps to the same time.
   The ordinary convection/RSF restrictions remain selected. Add the opt-in
   benchmark model `BP5 state startup` with
   `Maximum logarithmic state change = 0.1`.
2. Replay the failed case from the same fresh inputs with its **saved plugin
   binary**, the original 4e6-s physical steps, `V_min=1e-20`, and all 30 allowed
   nonlinear iterations. Enable the existing observational
   `ASPECT_FAULT_NONLINEAR_DIAGNOSTIC`. Add only an ordinary accepted-step-1
   checkpoint for future reproduction; no failed candidate is checkpointed.

The startup caps are 1200 s each; the failed-solve diagnostic cap is 1800 s.
Each uses four MPI ranks and the same Release executable. Existing output is
never overwritten and a failed/expired simulation is not automatically retried.
The initial MPI sandbox launch never entered ASPECT; its log is preserved as
`300/launch-sandbox.log`. The permitted launch outside the sandbox is the one
counted as the 300-s simulation.

## Predictor and implementation

`startup_time_step.cc` calls the configured `FaultFriction::update_state` with
committed nodal V/Theta and the same projected chemical mixture as friction.
It imposes

\[
 \max_i (b_i/a_i)|\log(\Theta_i^{pred}(\Delta t)/\Theta_i)|\le 0.1.
\]

The existing public derivatives provide the ratio without new material
parameters: `Theta*mu_Theta/(V*mu_V)=b/a`; their regularization multiplier
cancels. If the configured physical timestep ceiling violates the bound, a
monotone bisection keeps the safe bracket endpoint. The artificial interval
is not read or modified by this plugin. The time-stepping manager still takes
the minimum over the selected models and existing restrictions.

This is a frozen-rate **accuracy heuristic**, not a nonlinear convergence
criterion or a proof of temporal accuracy. The comparison uses fixed 300/150-s
ceilings; it checks the realized accepted state change as well as the predictor.
No core ASPECT source changed in this follow-up. Only the separate BP5 plugin's
CMake target gains this opt-in timestep model.

Independent scalar preflight on the saved, identical initial state:

| Physical dt | Maximum weighted logarithmic state change |
|---:|---:|
| 150 s | 0.04472614984 |
| 300 s | 0.08918708981 |
| 4e6 s | 37.93879074 |

The limiting dt is **336.6146901444 s**, agreeing with an independent analytic
inversion of the exact aging map. The limiting node is at 29.9 km. This explains
why the old `V*dt/Dc` check alone did not flag the problem: initially
`V*Theta/Dc` is much smaller than one, so state grows almost at unit rate even
when `V*dt/Dc` is small. The new criterion measures the consequent frictional
state change directly.

## Fresh-start results

Both fresh initializations reproduce the failed case's initial nodal geometry,
V, incoming/outgoing Theta, slip and weak shear/normal loads **bit for bit**.
Neither run inherits the first large physical aging update.

| Check through 900 s | 300-s ceiling | 150-s ceiling |
|---|---:|---:|
| Real accepted steps | 3 | 6 |
| Final minimum V/Vp | 0.8371191454 | 0.8012456795 |
| Lower-active nodes | 0 at every accepted state | 0 at every accepted state |
| Accepted Newton alpha | 1 throughout | 1 throughout |
| Maximum initial predictor measure | 0.0891870898 | 0.0447261498 |
| Exact-aging relative error | 2.22e-16 | 2.22e-16 |
| Wall time, four ranks | 471.30 s | 817.03 s |
| Maximum child RSS (not sum over ranks) | 2,752,592 KiB | 2,763,992 KiB |

All bulk/surface convergence and fresh linear checks pass. Native raw-QP loads
reproduce the weak loads, with the same endpoint source/normalization treatment;
phase, geometry and inert H checks pass. Initialization keeps its artificial
4e6-s interval and retains supplied histories. The first real 300-s step changes
Theta by at most a factor of **1.0119626**, not 157. The largest realized weighted
logarithmic state change is also below 0.1; the predictor is not being used as
a substitute for measuring accepted history updates.

At 900 s, over the 0–30 km weakening interval, 300 versus 150 s gives:

| Quantity | Maximum absolute difference | Difference / fine maximum |
|---|---:|---:|
| V | 3.58735e-11 m/s | 4.3453% |
| Committed Theta | 0.00457972 s | 1.6545e-7 |
| Accumulated slip | 1.79059e-8 m | 2.1857% |
| Native weak shear **change** | 0.592947 Pa | 19.1747% |
| Native weak normal-traction **change** | 0.0159785 Pa | 18.8371% |

Ratios use the maximum magnitude of the corresponding fine field on the stated
interval, not a pointwise relative error. Stress differences are normalized by
their small evolving changes, **not** by the 50-MPa background. All common times
and transition/deep controls are retained in `temporal.json`.

The velocity discrepancy is already visible at t=300 s: the one-step solve uses
Theta_0, whereas the second half-step uses Theta_150. Their committed Theta_300
is nearly identical, but their incoming mechanical states are not. Thus the
successful small-step startup does not make the split mechanical response
temporally converged. Two timestep levels establish sensitivity, not an order
or a limiting solution. No further timestep sequence or long run is performed.

## Contact mechanism in the failed large-step solve

The capped replay exhausts the unchanged 30-iteration budget in **1534.92 s**,
before the 1800-s cap. It reproduces every printed bulk/fault residual exactly
(maximum difference zero across 30 linearizations). The accepted incoming
bulk and particle CSVs are byte-identical on all four ranks; nodal histories
also match. Only states 0 and 1 are published. The expected-failure regression
passes; the trajectory itself fails, as before. Maximum child RSS is
2,750,956 KiB (not the sum over ranks).

Across the captured Newton bases, **27 nodes contact the bound, five leave it,
and two return**. The final captured base has 24 active nodes. All 30 line
searches accept the fraction-to-boundary limit with **zero rejected candidates**;
the smallest alpha is **8.5130230468e-8**. The final normalized bulk/fault
residuals are **0.7215125 / 0.3328721**, not close to convergence. Their
dimensional norms are 4,739,202.1303 and 1,300,115.9193 respectively.
All recorded free-node contact fractions agree with the existing exact formula,
and every available next-base V agrees with the accepted absolute trial,
including exact contact at `V_min`.

There are two observed regimes, not just one:

- Early contact/release: the 28.6, 29.9, 30.0, 29.7 and 30.1-km nodes
  touch the lower bound and subsequently move back into the interior. The
  29.9 and 29.7-km nodes return to contact at iterations 7 and 8 respectively.
  The top node remains active after its first contact.
- Subsequently, successive near-bound nodes limit the global step. At
  iteration 12 the 7.3-km node has `V=1.000002428171616e-20` m/s and
  `dV=-1.168838594859696e-19` m/s. Its distance to the bound is only
  `2.428171616e-26` m/s, giving `alpha_max=2.077422517283872e-7`.
  This distance is still much larger than the unchanged local contact
  tolerance (approximately `2.22e-34` m/s), so it is not already active.

For an explicit release example, the 29.9-km node is exactly at `V_min` at
iteration 3 but has a positive direction `4.626022845959919e-14` m/s and is
free. The accepted update takes it to `1.407924748209394e-15` m/s. It returns
to the bound by iteration 7. This is actual motion, not an artifact of resetting
the active mask before each Newton solve.

The observed tiny steps are imposed by the exact fraction-to-boundary rule;
they are accepted without Armijo backtracking. The residual remains large,
so this is not residual roundoff or a false-convergence issue. The large first
physical aging update supplies the difficult incoming friction state. The fresh
small-step runs avoid that jump and never encounter contact over their tested
interval. This identifies the failure mechanism without demonstrating that
the existing active-set mathematics should be changed.

`Fmin_weak_density` is an all-unprescribed-nodes-at-minimum probe at fixed bulk
state, **not** the individual node's unreplaced residual at the current iterate.
Its sign cannot establish that an observed release violates complementarity.
No new KKT interpretation or solver correction is inferred from that column.

## Evidence locations and reproduction

All new runs are under `weakening30-dc010-ell100/small-startup/`:

- `300/` and `150/`: independent fresh initialization and short trajectories;
  `checks.json`, `state_startup_predictor.csv`, accepted histories and native
  work/QP exports. Checks require true nonlinear/fresh-linear convergence,
  exact initial-state equality with the original case, correct incoming/outgoing
  state, exact aging/slip publication and native weak-load reconstruction.
- `predictor_preflight.json`: independent scalar/analytic predictor check.
- `temporal.json`: matched-time comparison, including stress **changes** rather
  than errors normalized by the 50-MPa background.
- `bounds/nonlinear_bounds_2.csv`: all production node values/directions and
  final active masks. `contact_nodes.csv` adds physical down-dip coordinates,
  local contact fractions and limiter flags. `contact_summary.json` distinguishes
  actual arrival/departure from the bound from merely rebuilding an active set.

The diagnostic fractions are exactly `(V-V_min)/(-dV)` for free downward
directions, with global limit `min(1, fractions)`. This is not a fitted step
rule. Source inspection confirms that active sets are rebuilt from empty at
each Newton iteration and can only grow within that iteration. Re-adding a
still-contacting node is therefore **not** evidence that it physically left
and re-entered contact. Actual V and the local bound test are tracked separately.

```sh
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_initialization -j4
python3 benchmarks/reconstructed_fault/bp5/startup_predictor_preflight.py
# For new output labels only; do not rerun completed labels:
python3 benchmarks/reconstructed_fault/bp5/run_small_startup.py prepare 300
python3 benchmarks/reconstructed_fault/bp5/run_small_startup.py run 300
python3 benchmarks/reconstructed_fault/bp5/run_small_startup.py prepare 150
python3 benchmarks/reconstructed_fault/bp5/run_small_startup.py run 150
python3 benchmarks/reconstructed_fault/bp5/analyze_small_startup.py temporal
python3 benchmarks/reconstructed_fault/bp5/run_small_startup.py prepare bounds
python3 benchmarks/reconstructed_fault/bp5/run_small_startup.py run bounds
python3 benchmarks/reconstructed_fault/bp5/analyze_small_startup.py bounds
```

The failed case is an **expected-failure regression**, not a passing trajectory:
its launcher requires nonlinear exhaustion and only accepted steps 0–1.
The analysis compares its incoming state and residual sequence with the saved
failure, then records the contact evolution. Increasing iteration count or
lowering `V_min` is not a regression fix.

The recoverable `small-startup/tested-source.tar.gz` contains the tested plugin,
new timestep model/launchers/analysis, and the preceding source overlay/patch
based on HEAD `359ea223cd0fa088ba7c8b03338cc7fe19d4acdc`.
SHA256: `782dc9fb9ccd6b16dcb172e5b8bae5f63cac68c44d24a12115246c4f409c0e3c`.
Individual executable, plugin, input and launcher hashes are in each
`launch.json`. The existing blocked `server-30km` package was not replaced.

The compiled predictor's direct (already safe ceiling) branch is exercised in
both simulations. The limiting-root calculation is independently verified by
bisection versus analytic inversion in the scalar preflight; the compiled
bisection branch was not separately exercised by a simulation. Restart and
later bound-heavy evolution are not qualified by these short runs.

## Changed files and remaining decision

- `startup_time_step.cc`, `CMakeLists.txt`: benchmark-only, explicitly selected
  frozen-rate state-change restriction; production evolution is unchanged.
- `run_small_startup.py`: separate fresh 300/150-s runs and immutable-plugin
  expected-failure replay, with caps, hashes and no automatic retries.
- `startup_predictor_preflight.py`, `analyze_small_startup.py`: independent
  predictor check, lifecycle/native-load and common-time comparison, exact
  contact/update accounting.
- `package_30km.py`: include the new translation unit in future source overlays;
  no existing server package is regenerated or enabled.
- `README.md`, this report: distinguish passed short startup, measured timestep
  sensitivity, preserved failure, and still-unqualified long/restart execution.

Build with `-j4`, Python syntax checks, independent predictor preflight and both
complete fresh-start lifecycle analyses pass. No complete ASPECT suite, new
MPI-rank comparison, restart simulation or long trajectory was run. The earlier
velocity-interpolation fix and its one-/two-rank tests are retained unchanged.

The recommended next decision is startup **accuracy**, not a larger nonlinear
iteration allowance or smaller slip-rate bound. The opt-in state-change bound
prevents the demonstrated oversized first aging update, but its coefficient
0.1 is only a heuristic: the measured 300/150-s velocity and slip differences
must remain visible before selecting a continuation accuracy budget. This
follow-up does not redesign the active-set method to accommodate an inaccurate
4e6-s first physical step.
