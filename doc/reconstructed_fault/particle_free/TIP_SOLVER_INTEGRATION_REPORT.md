# Bulk-tip constrained solve and ASPECT integration

The constrained solve and distributed ASPECT integration are implemented and
locally verified. The original refined failure is reproduced, rescued from its
unchanged accepted checkpoint, and continued to **.08 s**. The original coarse
control, equilibrated localized mesh/time comparisons, MPI/restart/rollback,
ordinary Stokes and short native DGQ2 mature regressions pass. This qualifies
bounded solver behavior, not general propagation, stability or production use.

## Identity and preserved baseline

Worktree `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, entry
revision `663937bed9305df2823ac77a3fc2c3925bb94caa`, no incoming tracked changes.
The user-supplied instructions and pre-existing untracked evidence remain intact.
No checkout, reset, production conversion, server submission or commit was made.
[Entry identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_solver/entry.json)
record the protected original core `340b6c37516a…ebf2`, checker
`5c5ba77cddf0…446`, and mature core `6ea374c5b9b5…67d9`.
The original tip pair remains in `build-particle-free/bulk-tip/qualified-340b6c37/`.

The numerical candidate used for the completed comparisons is core
`ec45e1be5a8a2e852d474791c8fc6f570e6d3d3f0f546fe23dd9a78c4588cfa3`.
Each run's `command.json` records its exact core/checker/input and changed-source
SHA256 hashes, revision, command, numerical-thread settings and runtime.
After token-identical formatting, the retained final core is
`1cc27f9ced605c5e03886ea810e792959e1ae2783c0e6e27d84a7915c087cac6`.
It is byte-identical to the preserved unweighted control executable. Final
source/checker identities and execution records are in
[closeout.json](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_solver/closeout.json).

## Implemented numerical correction and architecture

The ordinary `Simulator::solve_single_advection_iterated_newton_stokes` owns
initialization, nonlinear iteration, globalization, convergence signals and
pending publication. There is no new timestep executor. The integrated route
is opt-in through the existing bulk-tip mode, with `Bulk tip solver = ASPECT`.
Ordinary fields/solvers, the retained interface route, mature histories and
advection face assemblers retain their prior responsibilities.

`internal::BulkTipCoupling` assembles the exact mechanics/phase residual and
condensed section tangent. Cell/face FE algebra uses distributed owned rows;
section integration and its actual work adjoint use sparse rank-local support
and small section reductions. The active path has no global phase-pair graph,
replicated global FE vector or full-system UMFPACK solve. The retained
`benchmark direct` implementation is explicitly historical; a second small
benchmark oracle compares the current coupled directions at a fixed state.

The flexible coupled Krylov solve actually calls ASPECT's ordinary Stokes
assembly, `build_stokes_preconditioner()` and `solve_stokes()` on velocity and
pressure. The reference inverse uses block AMG, its pressure quotient and
homogeneous corrections; the adapter reconstructs the omitted continuity row
and converts displacement/pressure scalings. It is an approximate preconditioner,
not the physical cohesive tangent. Native `PhaseFieldHandler::assemble_dg_fracture`
provides the unchanged SIPG functional and distributed derivative. The retained
observer wrapper still gathers for its older consumers. The new coupled operator,
section condensation and phase VI remain specialized mathematical components.

The former add-only bound search is replaced on the active path by primal/dual
classification with both lower and upper release. Imposed phase displacements
enter all coupled equations. Repeated classifications or failed exact Newton
searches fall back to the existing SPD obstacle solver with a positive phase
search metric, followed by the exact mechanical/section correction. This search
subproblem does not replace the mixed indefinite physical tangent. The ordinary
scheme checks Armijo decrease of the restricted incremental functional on
admissible incompressible corrections. Roundoff-sized functional changes also
require scaled residual decrease. Every trial recomputes the physical response.

Three numerical details were necessary for strict feasibility: exact imposition
of prescribed active rows before residual rechecking, stable interpolation to a
tiny positive lower target, and positive Bernstein-basis evaluation of the same
Q2 polynomial. None changes the physical obstacle or introduces phase clipping.
The final mechanical 1e-9 and phase 1e-7 gates, dissipation checks and independent
scaled 1e-10 endpoint ledger remain unchanged. Mechanical Gauss6, fracture
Gauss5, BE age/endpoint convention, full 3-D plane-strain metric and physical
material/RSF inputs are preserved.

The physical owner prepares candidate tau/A/section histories from an immutable
parent. The simulator installs them once after acceptance. Native accepted phase
now has distributed ownership; the serialized per-cell values and signature
`bulk tip physical tau A DGQ2 sections BE v1` are unchanged. The original fine
checkpoint and one-to-two-rank active restart test this serialization boundary.
`native tip stress` is the clearer representation spelling; the original input's
`DG trial stress` spelling remains accepted without reinterpreting its history.

The [equation-to-code guide](TIP_TREATMENT.md) includes the weak cell/face forms,
condensation, scalings, state dictionary, endpoint-law monotonicity, search versus
physical energy, source/test links and an [actual accepted point/section trace](tip_treatment/state_and_trace.md#an-actual-accepted-step).

## Verification and retained failures

- Point/bounds suite: **3319 assertions in 15 cases**. Includes independent
  coupled quadratic lower/upper release, point derivatives/power, Q2 basis
  equivalence and tiny-obstacle arithmetic.
- Current coupled Krylov versus small direct oracle: relative direction error
  **9.0051e-10**; independently manufactured direction **9.0053e-10**. At the active .08 state, the final retained solver gives
  **3.2858e-8** for both comparisons (318 free unknowns), with functional-weighted
  tangent asymmetry **5.1006e-17**. The benchmark tolerance is 2e-5 relative.
- Localized coarse directional derivative at .04: **2.2431e-9** relative;
  phase functional derivative **6.6697e-8**; scaled saddle Lagrangian derivative
  **1.8616e-7**. The pressure-inclusive check is not a stability test.
- Active two-rank derivative at .08: **1.1340e-9**, functional **4.3965e-8**,
  Lagrangian **2.5439e-8**. Signed section-law/primitive derivatives, native SIPG
  derivative and exact discrete power identity also pass.
- Localized coarse and active two-rank trial/reject/retry checks preserve accepted
  histories and restore fields bitwise. Restart from the one-rank .055 snapshot
  to two ranks reaches .08; stress differs by **2.3561e-16** relatively and
  section slip by **7.2727e-16**. Age agrees to printed precision.

The unchanged original executable again fails at .05 after accepting .0475
(`integration-original-fine`, 22.36 s). Its checkpoint slot 2 is preserved.
The corrected run starts from exactly that history, not a new initialization.
`integration-fine02` accepts .05, .0525, .055 and .0575, then reaches its
300 s cap during .06. The first rescued step has seven nonlinear iterations;
.0575 takes 23 iterations and 12,665 Krylov iterations. This is a measured search
cost limitation. No stability Hessian or competing physical branch has been
established. The unweighted continuation `integration-fine03` accepts through .0725, then
reaches its 600 s cap during .075; its latest regular checkpoint is .0675.
The tested unweighted core/checker and exact linear source remain in
`build-particle-free/bulk-tip/controls/integration-unweighted/`.

An exact coordinate change y=S d, S_phase=sqrt(Kf/(beta G)), was checked using
S J S^-1 and S R. It passes the direct and weighted-symmetry checks but increases
fine Krylov counts from 1728→2351 at .07 and 993→1072 at .0725. It is **rejected
as a conditioning improvement** and removed from active source; its core/checker
and linear source remain in `controls/integration-balanced/`. Overlap at .0725
agrees within 2.36e-12 in phase and 1.06e-7 Pa in shear stress. This comparison
changes linear coordinates only, not the law, obstacle or acceptance gates.

`integration-fine04` reaches .08 with those equivalent coordinates in 70.37 s;
`integration-fine05` then repeats the final step with the retained scaling and
passes in 11.35 s. The final-step phase difference is 5.20e-16, shear stress
4.97e-11 Pa and section slip 1.62e-17 m. The exact accepted ancestry is
`integration-original-fine:1–19 → integration-fine02:20–23 →
integration-fine03:24–27 → integration-fine04:28–31 → integration-fine05:32`.
The [chain reduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_solver/original-fine-chain.json)
checks contiguous steps/clocks and includes every numerical loss without
inventing a fresh trajectory. Its maximum mechanics/KKT/ledger residuals are
8.67e-10 / 6.45e-8 / 4.14e-15.

| Original fixture, active window (.04,.08] | Physical remainder J/m | Added fracture J/m | Ratio |
|---|---:|---:|---:|
| Retained coarse, dt=.005 | .0528775 | .0308367 | 171.5% |
| Retained coarse, dt=.0025 | .0266422 | .0298617 | 89.2% |
| Completed fine, dt=.0025 | .0256767 | .0332946 | 77.1% |

For the fine window, stress BE contributes .0232415 J/m (90.5% of the remainder),
cohesive-conjugate BE .00203063, compliance endpoint −.0000692796 and fracture
quadratic .000473805. Signed/absolute totals are both retained. The initialization
and remote-nucleation limitations of this original fixture are unchanged.
Convergence of its revised solve establishes neither a stable branch nor spatial
or physical-work accuracy.

`integration-control05` completes the original coarse dt=.005 trajectory to .08
in 103.99 s including wrapper overhead. Its most expensive .06 step takes
45 nonlinear iterations. Against the retained `control04` final native samples,
maximum phase difference is 1.497e-12, shear-stress difference 6.747e-8 Pa,
section slip difference 3.663e-14 m, and stored energy differences ≤2.452e-12 J/m.
[Exact sample comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_solver/original-coarse-final.json)
uses actual retained final samples; no unavailable initial frame is invented. Earlier probe crashes exposed incorrect assembler
registration and pressure-gauge treatment in the adapter; those were corrected.
Earlier .035 search failures, the .065 strict-bound failure and timed-out trials
remain in `integration-probe*`, `integration-control01…04`, `integration-fine01`.
They are not relabeled as passes. Physical inputs were not adjusted to repair them.

## One equilibrated localized fixture

[`localized.prm`](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/localized.prm)
uses a 4×2 m closed specimen, fixed ell=.25 m, a declared finite seed centered
at (2,1), half-length .8 m, half-width .2 m and peak .8. Before the physical clock,
the unchanged DG fracture functional is minimized under this seed and bounds.
The resulting mesh-dependent profile becomes the accepted initial obstacle.
Age is then 10 s everywhere, including intact sections. No loaded state is
smoothed and no age is reset by an observed tip.

Boundary velocity is (.01(y−1),0) m/s. Nominal intact Maxwell shear is below
800 Pa at .08, compared with the 1000 Pa calibration. The measured remote
region |x−2|>1.3 m remains intact: maximum endpoint Y/40 is .7321/.7318/.7252
for coarse/half/fine, and its damage increment is zero. This checks the actual
coupled solution, not only the nominal loading estimate.

| Quantity at .08 s | h=.25, dt=.005 | h=.25, dt=.0025 | h=.125, dt=.0025 |
|---|---:|---:|---:|
| Initial fracture, J/m | 20.987724 | 20.987724 | 21.671730 |
| Added damage integral, m² | 8.42235e-5 | 8.37014e-5 | 7.66100e-5 |
| Tip-zone added damage, m² | 4.58072e-5 | 4.55680e-5 | 3.62680e-5 |
| Max section rate, m/s | .0113033 | .0115720 | .0109241 |
| Max cumulative slip, m | 1.77558e-4 | 1.76683e-4 | 1.69276e-4 |
| Added fracture, J/m | 1.27122e-5 | 1.26132e-5 | 8.99281e-6 |
| Physical work remainder, J/m | .1512660 | .0757599 | .0761393 |
| Runner wall time, s | 19.27 | 27.79 | 191.77 |

Tip zones are ||x−2|−.8|<.25 m. Native quadrature samples define extent reports
at phi=.001,.01,.1,.5,.99; a weaker plotted contour is not counted as propagation.
The coarse unloaded profile remains stationary through .08 to 1.11e-16 maximum
point difference, zero stress/slip/work and KKT ≤3.79e-8. The fine unloaded
check reaches .01 s in 9.04 s; its integrated phase drift is
-2.73e-18 m², with KKT ≤9.03e-08, zero stress and zero slip.

At fixed mesh, halving dt changes stress by .03114%, section rate by 2.1301%
and cumulative slip by .36383% in common .25 m cell/section means. The matched-dt
mesh comparison changes stress by 1.0764%, rate by 10.6632% and slip by 10.9986%.
Initial equilibrated phase differs by .0117039 in the coarse-cell L2 measure;
the mesh comparison therefore includes discretization of the same physical seed,
not identical discrete initial fields. Point-history quadrature is not refined
independently here. These results do not establish spatial convergence.

The load-driven response is a weak diffuse adjustment around the seed, not a
fully damaged advancing core. The phi>.5 sampled extent is unchanged at
[1.54235,2.45765] m on coarse and [1.52117,2.47883] m on fine; no phi>.99
region exists. Endpoint remainder halves with dt, but is thousands
of times the tiny added fracture. Stress BE loss dominates. Algebraic ledger
closure (≤2.16e-14 on coarse) and power adjoint closure (≤5.56e-17) do not make
that physical-work error negligible. Signed and accumulated absolute remainders
are both retained; the positive total remainder makes them equal in these runs.
The component tables retain stress BE, cohesive-conjugate BE, compliance endpoint,
fracture quadratic, physical viscosities, signed friction/radiation and residual
work. Nothing is subtracted and relabeled physically exact.

Opposite-sign rates and multiple sampled bands are retained without clipping.
Their measured signed work and cancellation statistics are in the summary;
coarse positive work is .1836202 J/m and negative local work −9.09e-12 J/m.
Two active sampled bands occur in four section-steps under the explicit diagnostic
threshold max(1e-10/s, 1e-6 max|w|). Such detection is a diagnostic, not a new
constitutive localization rule. Half-dt has −9.60e-10 J/m negative local work
and four two-band section-steps; fine has −5.73e-11 J/m and no two-band detections
at that threshold. Full transverse tau/A history remains stored.

## Cost, reproducibility and remaining scope

For the localized coarse/half/fine runs, nonlinear counts are 13/23/24;
Krylov counts 707/933/3675; actual Stokes solves 723/965/3707. Sparse nonzero
counts are 146689/146689/591865 for 1152/1152/4608 phase rows. This verifies
cell/face rather than all-pairs storage. It does not establish scalable performance.
On two ranks, each rank owns 576 of the 1152 phase rows. Small differences in
stored constrained entries do not create a dense phase graph.
The existing no-fault `tests/depth_average_06.prm` control, with explicit block
AMG and separate output paths, produces byte-identical `depth_average.txt` and
`statistics` with protected and candidate cores. A current native DGQ2 mature
layered fixture completes one .0125 step: one history/geometry publication, no
retry, phase balance −9.51e-18, and independent exterior-reference discrepancies
≤8.89e-16. Its older Q1 harness attempt correctly failed the pre-existing native
phase registration guard; that setup failure remains recorded. All seven mature
protected artifacts and both original tip binaries retain their recorded hashes.
The 31 wrapped execution/check records total **2093.13 s (34.89 min)**,
including failed attempts and capped runs. Point and reduction checks add well
under two minutes; aggregate new simulation/check time remains below two hours.
Separate configure/build logs account for about **2016 s (33.6 min)**, measured
from log creation to last write. Jobs ran sequentially, at most four build workers
or two MPI ranks, with one numerical thread per simulation rank. All task jobs
have finished; no next experiment is launched.

Every run is launched sequentially through
`benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py`; full commands
are in the paired JSON and the fresh run directory. Typical reproduction:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py LABEL 195 \
  python3 benchmarks/reconstructed_fault/particle_free/bulk_tip/run.py LABEL \
  --fixture benchmarks/reconstructed_fault/particle_free/bulk_tip/localized.prm \
  --dt .0025 --seconds 180
```

Use `--refinement 1` for the finer mesh, `--unloaded` for stationarity, and
`--resume-from PATH/output --snapshot-id 3 --ranks 2 --rejection --rejection-step 13
--tangent --tangent-step 16` for the selected active restart. Labels must be fresh.
The original fixture has not been edited. Use the protected original pair for
its historical failure, not a candidate binary inferred from HEAD.

[Compact reduction](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_solver/localized.json),
[damage figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_solver/localized-damage.png),
[slip/energy figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/tip_solver/localized-slip-energy.png),
and [reduction script](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/analyze_integration.py)
provide the numerical comparisons. The damage figure uses flat nearest-sample
tiles of native quadrature data, with common color scales and no smoothing; all
reported integrals use the original quadrature weights. No old campaign was recreated. The historical
cleanup losses of 39/45 pilot frames and early logs remain losses; prior reports
remain evidence. The mature four-run closure and deferred server/long campaigns
remain deferred.

## Scientific conclusion and next experiment

The old refined failure is not evidence that the bulk law cannot continue: a
proper constrained solve reaches the requested endpoint with unchanged physical
inputs. The fixed-section closure remains an explicit constitutive assumption,
not a derivation of general turning/branching physics. Full transverse histories,
intact reaction, signed local work and cancellation remain visible.

The equilibrated specimen removes the large unloaded adjustment and avoids remote
intact activation over the tested interval. It produces localized slip and weak
diffuse damage, with no stronger-core advance. Mesh-sensitive initial equilibrium,
roughly 11% slip sensitivity and dominant BE work error prevent a physical
propagation or production-readiness claim. Sparse distributed ownership and
actual Stokes reuse are implemented; expensive constrained searches still limit
practical performance. No unresolved stable-branch selection is diagnosed from
the raw saddle Jacobian or from a timeout.

The next justified numerical experiment is one **fixed-parent .0575→.06
preconditioner/search comparison**, using the preserved fine checkpoint, exact
coupled operator, both-bound release and the same physical gates. Target the
observed 19,964 Krylov iterations with coupled mechanical/phase information;
retain the current solve as the controlled reference. Do not begin a longer
physical trajectory before reducing that cost and deciding the required BE-work
accuracy. Any later localized load extension must keep an explicitly checked
remote-intact margin and report strong-core growth separately from its diffuse
halo. Neither that extension nor another mature/production campaign is launched.
