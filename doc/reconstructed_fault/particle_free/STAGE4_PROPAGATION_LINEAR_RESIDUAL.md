# Explicit linear residuals in the step-68 propagation replay

The [phase-row audit](STAGE4_PROPAGATION_PHASE_ROW_AUDIT.md) traced the
5.44–5.48 ms interval on the 256 × 128 mesh. Its exact active-set solver
returned 14 corrections with explicit residuals above their requested
tolerance. Summing the off-diagonal entries of the *next* classification
matrix gives 276,521 row-status changes after those corrections. This counts
transitions across passes. A coefficient can contribute several times, and a
correct direction could independently cause some of the same transitions.

## Linear and nonlinear certificates

For each exact bound classification, the simulator builds the restricted
operator B and right-hand side b. After FGMRES returns z, it now evaluates
‖Bz−b‖₂ with the same B, mask and b. A failed check returns control to the
existing search over positive phase metrics before applying z or making another
classification. Following a successful check, imposing the exact prescribed
values only zeros identity rows whose right-hand side is zero. The solver
then checks feasibility and the complete linearized VI.

For a positive metric the phase obstacle solve measures each CG correction
against its own restricted metric matrix and correction load. Its final
projected-gradient and feasibility test still uses the full metric. The
subsequent FGMRES mechanics correction holds the phase increment prescribed
and measures its residual against that restricted coupled action. Only a
certified pair enters the nonlinear search. The estimated and explicit norms
are logged with their respective stopping tolerances. Ordinary obstacle
callers retain the original entry point.

Each nonlinear evaluation uses the original coupled mechanics, section
elimination and phase equations. Acceptance requires the physical mechanics
and phase KKT tolerances of 1e-9 and 1e-7, plus the usual bound and virgin-plane
checks. The midpoint law, history, physical inputs and quadrature come from the
same retained half-step fixture. The accepted parent is step 68 at 5.44 ms.
An initial restore-only launch ended before solving because the stored pending
clock exceeded the one-interval end time. The replay retained the original
5.60 ms end time and used the end-step criterion to select the first interval.

## Residual check alone

The first rejected correction reproduced the original FGMRES estimate
9.11e-16 and explicit norm 0.00206574 on the identical starting mask. The
shifted phase VI residual was 3.10e-16 against 1.09e-14. Three exact
corrections failed at Newton iterations 0, 4 and 7. In every case the next
classification using that correction was skipped. Across the accepted
interval, 22 exact restricted solves returned, six shifted mechanics solves
passed and all 20 restricted phase CG solves passed. The native counter
recorded 1,985 active-bound releases (including changes from one bound to
the other). Comparing that counter to the audit's 276,521 off-diagonal
transitions would mix different definitions. Summing lower-to-free/upper and upper-to-free/lower entries in the old audit
gives 306,612 releases through Newton 11, compared with 1,985 through acceptance
in the guarded replay. For all changes associated specifically with failed solves, the
comparable number is 276,521 transitions following failed explicit tests in the baseline and zero
following failed tests in the guarded run.

The guard-only executable (SHA-256
`95fc13f1d25959f8aea77732724e569558e3a43f3d58045df8c23c83e515a7a4`)
accepted at step 69, time 5.48 ms. The simulator reported 1198.52 s for the
step. Process wall time including restore, the tangent observations and final
checkpoint was 1220.80 s. The original attempt spent 3600.37 s without
accepting and logged 36 completed exact solves. Thus this run took at least
66% less time to reach acceptance, and used 14 fewer completed exact solves
than the incomplete original. A baseline time to acceptance is unavailable.
The final mechanics residual was 4.72e-16 and phase KKT 6.55e-8. After checking
the proposed bound polish and updating virgin planes, the solver re-evaluated
the physical equations. The history publication counter advanced once.

## Coupled tangent and contact trial

At Newton 8 the full exact direction again drove row 258249 to its obstacle.
For five feasible fractions the diagnostic changed *all* components along the
same Newton displacement, assembled the nonlinear residual at the displaced
state (including section mobility, exact stage aging, resistance, driving and
bulk stress), and compared its change against the assembled coupled Jacobian
applied to the realized displacement. The tangent stayed frozen at the
incoming state. Values below are signed changes in the **unprojected** row
gradient divided by its mass.

| Fraction | Reassembled | Jacobian action | Relative discrepancy |
|---:|---:|---:|---:|
| 1e-2 | −5.75915e-5 | −5.73037e-5 | 5.02e-3 |
| 1e-3 | −5.73323e-6 | −5.73037e-6 | 4.99e-4 |
| 1e-4 | −5.73065e-7 | −5.73037e-7 | 4.99e-5 |
| 1e-5 | −5.73039e-8 | −5.73037e-8 | 4.99e-6 |
| 1e-6 | −5.73037e-9 | −5.73037e-9 | 4.78e-7 |

The full phase-block Taylor remainder decreased from 7.34e-10 to 7.29e-12
to 7.29e-14 over the first three fractions. Mechanics and continuity
differences reached floating-point noise (absolute discrepancies up to
1.8e-18 and 1.4e-22). This directional test supports the implemented derivative
for the displacement implicated in the stalled search.

For a material strip, if a full exact trial fails the nonlinear search within
the energy roundoff band, and a coefficient arriving at a bound attains the
maximum physical phase residual, the solver tries 99% of the coupled direction.
The 1% interior gap is a numerical line-search choice. The usual energy and
merit checks judge the trial, and ordinary halving is available if it fails.
No coefficient is independently clipped. The nonlinear iteration reassembles
the original equations before declaring convergence.

## First physical interval and cost

All three runs started from the same four-rank checkpoint. Linear-solve
counts below distinguish completed restricted exact solves from separately
checked shifted mechanics and obstacle systems. The original trace does
not give a comparable count of completed obstacle CG solves.

| 5.44–5.48 ms replay | Completed exact solves | Shifted mechanics / phase CG | Active-bound releases | Tangent assemblies / line trials | Wall time |
|---|---:|---:|---:|---:|---:|
| Original | 36, including 14 above explicit tolerance | — | 306,612 through Newton 11 | — | 3,600 s process, interval incomplete |
| Residual checks and diagnostic tangent probes | 22, including 3 discarded | 6 / 20 | 1,985 | 18 / 37 | 1,199 s step |
| Residual checks and interior contact trial | 16, including 3 discarded | 6 / 20 | 1,985 | 12 / 25 | 976 s step |

The near-full step was accepted at Newton 8 and 9. Across those two updates
the physical phase KKT went from 1.67e-5 to 1.86e-7 to 1.87e-9. After the
plane check, the step completed at Newton 11 with mechanics residual 1.39e-17.
Both guarded runs removed 99.35% of the baseline count of releases. Changing
the trial length reduced the number of tangent assemblies and line trials by
one third. Five additional coupled response evaluations were used for the
guard-only derivative probe, so the measured 223 s difference includes that
instrumentation.

The worktree was on `pf-rsf-particle-free` at `b32a2d4f6` with existing local
changes. The build reused the native release configuration and rebuilt the
affected unity translation units 37, 57 and 59. The final executable SHA-256
is `911f39bdf5b0425b9df9481dd91035a9760a9ae9ca92fa0919865b924fdd3420`.
Its inherited ASPECT version banner reads `5f1fc6f3c`. Executable identity,
exact commands, input files and accepted parent member hashes are recorded
separately for each launch. The half-step uses the original pilot checker.

## Four accepted physical intervals

The final executable published steps 69–72 once each and exited normally in
4,149.57 s on four cores, including restore, native output and checkpoints.
The table gives the simulator's per-step clock. Each phase CG solve and
shifted mechanics solve that supplied a correction passed its explicit test.

| End time (ms) | Step | Exact solves / rejected | Shifted mechanics / phase CG | Final Newton iteration | Physical phase KKT | Mechanics norm | Step wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.48 | 69 | 16 / 3 | 6 / 20 | 11 | 1.87e-9 | 1.39e-17 | 16 m 16 s |
| 5.52 | 70 | 9 / 0 | 0 / 0 | 5 | 2.01e-12 | 1.87e-18 | 8 m 58 s |
| 5.56 | 71 | 40 / 13 | 16 / 29 | 23 | 5.41e-8 | 5.09e-16 | 34 m 52 s |
| 5.60 | 72 | 7 / 0 | 0 / 0 | 5 | 1.96e-12 | 1.89e-18 | 7 m 51 s |

Step 71 accounts for half of the four-step wall time. Thirteen exact
corrections failed certification. After several small shifted updates near the
physical phase gate, an admissible one-eighth shifted trial brought KKT below
the gate. Step 72 accepted four full exact Newton directions. This variation
limits extrapolation of the average cost to a longer propagation experiment.

At 5.48 ms the guard-only and revised-search trajectories agreed to 6.09e-11
in every phase coefficient and 2.45e-15 m in section slip. The right contour
positions differed by 2.85e-14 m, the left positions agreed to output precision,
and integrated slip differed by 5.08e-17 m². The change in search cost can thus
be distinguished from the much larger differences between timesteps.

## Matched physical endpoints

The 80 μs reference uses the retained trajectory and its original solver. The
40 μs run starts with the identical native parent, uses the same physical
acceptance criteria, and includes the solver changes described above. Each
entry below measures change from the accepted 5.44 ms state. The phase
percentage is the spatial L² difference divided by the L² norm of the coarse
phase increment. The contour is the connected absolute φ = 0.05 level.
Slip is integrated over actual section lengths. Columns with two entries give
80 μs followed by 40 μs.

| Time (ms) | Integrated slip increment (10⁻⁶ m²) | Fine minus coarse slip | Phase difference relative to increment | Connected right advance (mm) | Added fracture energy (10⁻⁶ J/m) |
|---:|---:|---:|---:|---:|---:|
| 5.52 | 2.96218 / 2.95639 | −0.195% | 12.82% | 0.14242 / 0.10675 | 6.20434 / 6.25141 |
| 5.60 | 5.99968 / 5.99019 | −0.158% | 2.78% | 0.17779 / 0.19562 | 12.98063 / 13.06537 |

At 5.52 and 5.60 ms, the phase L² differences were 2.20e-6 and 9.76e-7 m.
The largest Bernstein coefficient differences were 8.86e-5 and 4.88e-5.
Every accepted coefficient increment on both branches was nonnegative to
output precision. The minimum coefficient (−2.71e-19) was inherited and the
maximum remained 0.60. The right-position difference changed sign, from
−35.7 μm to +17.8 μm. The largest matched left-position difference was
0.0201 μm. Both right advances through 5.60 ms were less than 0.0063 of the
31.25 mm cell size. Neither trajectory admitted a new section or changed an
existing formed-plane angle.

Across all material stations, the slip-increment RMS differences were 0.822%
and 0.585% of coarse evolution. For the right taper they were 0.0319% and
0.0242%. Comparing mean rates over the same last 80 μs gives relative RMS
differences of 0.822% and 0.633%. Reintegrating these means reproduced the
endpoint slip increments to 2.29e-15 m/s in rate units. Mean section resistance
differed by 0.231% and 0.412% in RMS over those same intervals. Endpoint age
differed by 0.0228% and 0.0299% in RMS. The retained outputs contain bulk
energy and section resistance for both branches. Full-field stress accuracy
remains open.

For the physical work check we summed external work less stored energy change
and the three physical dissipations over each interval. At 5.52 ms the signed
sums were +7.98e-12 and +2.48e-12 J/m for 80 and 40 μs. At 5.60 ms they
were +1.96e-11 and +5.30e-12 J/m. Summing interval absolute errors gave the
same four values. All were small relative to the corresponding fracture-energy
increments. The good work balance coexists with appreciable timing sensitivity
in the local phase increment.

The comparison supports the bulk slip and added fracture work on this short
window. The connected contour moved a small fraction of one cell and the
changes in that displacement differed by 25% at the first matched time and
10% at the second. These data do not resolve a propagation event or establish
a temporal order. If contour timing is needed, the next discriminating
experiment is four 20 μs intervals from the same step-68 parent through
5.52 ms, where the relative phase discrepancy was largest. The 40 μs
trajectory already provides both comparison times. Spatial qualification
would still be needed for a propagation claim.

## Evidence and reproduction

The build directory `build-particle-free/guard68` contains both executables,
the local build and launch scripts and separate verified native-output archives.
Each archive manifest records SHA-256 for each file. The guard-only archive
refers to the byte-identical retained parent for its duplicate slot 03.
The final archive includes native generation 01 for step 72. The reducers
`reduce_guarded_solves.py` and `compare_half68.py` are in the benchmark's
`stage4_propagation_pilot` directory. Use the latter with
`--fine-run build-particle-free/guard68/window --destination <new-directory>`
from the repository root. The per-step exports and latest accepted restart are available in the durable
worktree output directories. The full archives preserve intermediate native generations.

[Identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard68/build-identities.json) ·
[Replay command and parent hashes](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard68/window-command.json) ·
[Guard-only tangent trace](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard68/step69-run.log) ·
[Full-window residual trace](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard68/window-run.log) ·
[Certificates by step](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard68/window-cost.json) ·
[Physical comparison figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard68/half68-comparison.png) ·
[Convergence figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard68/half68-solver.png) ·
[Matched values](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard68/half68-comparison.json).
