# Recovering the 6.32 ms interval and testing a Coulomb formation direction

The first four-rank replay was interrupted by the host reboot. Its replacement reached the 90-minute limit with step 78 still the accepted parent. The overnight assessment covers subsequent solver work. [Recovery outcome](STAGE4_OVERNIGHT_PROPAGATION.md) · [Historical sleep handoff](STEP79_WAKE_HANDOFF.md).

## Solver comparison

The large pilot starts from the six verified members of the step-78 checkpoint
at 6.24 ms. The selected interval is 80 µs on the 256 × 128 mesh. The retained
executable stopped at Newton 27, after the shift-1e6 auxiliary phase obstacle
exhausted 100 active iterations. Its mechanical residual was 1.88e-13 and
phase KKT was 1.03e-6. The physical gates are 1e-9 and 1e-7 respectively.
[Original log, checkpoint and failure analysis](STAGE4_PROPAGATION_REPAIRED_80US.md#step-79-failure-and-restart-point).

Numerical active-set exhaustion now returns the iteration count and explicit
projected defect through `SolverControl::NoConvergence`. The existing shifted
search discards the failed trial and advances to the next available direction.
Inconsistent box data and nonpositive diagonals remain hard errors. Two
consecutive backtracked, shifted trials with unresolved energy decrease and
less than 1% improvement in the scaled physical residual return the adaptive
search to exact-first order. Every direction still needs its own explicit
linear certificate. The existing feasible line search and the fully
reassembled mechanics and phase residuals decide acceptance. The search-order
counter is local to an attempt.

On two MPI ranks the restricted three-row oracle forced a zero-iteration
budget failure above tolerance, caught the typed exception, and solved again
from the same initial vector. The corrected solution needed one active pass.
Primal, dual, projected and analytic errors were zero. The historical oracle
exhausted 100 passes with an explicit residual of 1e-15.
[Oracle and retry](../../../benchmarks/reconstructed_fault/particle_free/bulk_tip/phase_obstacle_replay.cc).

## Saved-stress comparison

The retained 32 × 16 positive-load fixture has two groups of twelve new
sections at 75 and 80 ms. A separate Python reader collected both ranks'
accepted Gauss-point deviators, verified each reconstructed strip area against
four metres times its reference width, and scanned angles every 0.01°.
The scalar confinement was 200 Pa and the configured reference coefficient
was 0.6. The native predictor in the original rule uses a later time within
the interval than the accepted stress in the proposed rule.

| Stations sampled | Accepted stress time | Forward Coulomb angle | Original angle at registration |
|---|---:|---:|---:|
| Both groups before the loading tilted | 35 ms | −15.48° | — |
| First twelve, immediately before formation | 70 ms | −5.26 to −5.08° | +10.69 to +10.86° |
| Next twelve, immediately before formation | 75 ms | −4.59 to −4.42° | +11.30 to +11.48° |

This is an orientation experiment. On the first twelve Coulomb planes,
the strip-average deviatoric normal stress less scalar confinement was
+165 to +171 Pa just before formation, with tension positive. For the
second group it was +201 to +207 Pa. At 35 ms it was compressive on
both groups. The pressure-independent part of the directional ranking
is well defined in each case. Interpretation as closed-face Coulomb failure
would additionally require a physically referenced compressive normal
traction and a contact model. The mechanical pressure export has a stage time
and a multiplier gauge. [Before tilt](../../../benchmarks/reconstructed_fault/particle_free/evidence/turn_formation/saved-before-tilt.csv)
· [Before birth](../../../benchmarks/reconstructed_fault/particle_free/evidence/turn_formation/saved-before-birth.csv)
· [Stress measure, conjugates and lifecycle](tip_treatment/friction_and_direction.md#optional-coulomb-plane-selection-on-material-strips).

## Coupled turning fixture

Both simulations start from the same equilibrated peak-0.60 horizontal seed.
The 8 × 4 m box has 32 × 16 cells, six-point native quadrature per coordinate,
and fixed reference membership. Both use midpoint at 5 ms, two bound ranks,
and the positive affine loading with the normal component applied after 40 ms.
Only the formation selector differs. All existing remote-phase, boundary and
physical convergence checks are active. I selected 100 ms as the upper time limit.
The work error below is external work less endpoint stored-energy change,
bulk and cohesive viscous dissipation and section dissipation. Signed and
accumulated absolute errors use every accepted interval in the common window.
The ledger closure also includes the discrete numerical work terms.

| Through the common 85 ms endpoint | Current intact-driving rule | Accepted-stress Coulomb |
|---|---:|---:|
| First registration, number at that time | 75 ms, 12 | 80 ms, 12 |
| Total newly formed sections | 60 | 12 |
| New recorded plane angles | +10.69 to +12.01° | −4.41 to −4.20° |
| Maximum native phase on the new sections at 85 ms | 0.15033 | 0.02018 |
| Rightmost absolute 0.05 contour (x, y), m | (6.20962, 2.06250) | (5.95935, 2.00000) |
| Phase increment integrated over 6 ≤ x < 6.5 m, m² | 0.0060324 | 0.00031835 |
| Centroid y of that increment, m | 2.04072 | 1.99937 |
| Added fracture energy, J/m | 0.182284 | 0.141623 |
| Signed physical-work error, J/m | −7.247e-5 | +1.359e-4 |
| Accumulated absolute physical-work error, J/m | 2.606e-4 | 1.359e-4 |
| Maximum absolute discrete ledger error, J/m | 6.82e-15 | 4.45e-15 |
| Maximum work-adjoint discrepancy | 1.73e-17 | 2.78e-17 |
| Maximum accepted mechanics / phase KKT | 8.57e-10 / 5.01e-8 | 8.57e-10 / 8.25e-8 |
| Solver time on the same 17 accepted steps | 66.39 s | 37.61 s |

The original endpoint was (6, 2) m. Two independent sampling densities on
the native polynomials found one connected 0.05 component ahead for the
current rule and none for Coulomb at 85 ms. The difference of the rightmost
contour positions is 0.25027 m on a 0.25 m mesh. The phase-field difference
has a spatial L² norm of 0.03158 m and native-quadrature maximum 0.13055.
The group of 36 current-rule registrations at the final common step reached
only 1.67e-11 phase. The older groups carry essentially all appreciable new
damage. The Coulomb sections reached a peak of 0.02018 after first registering
at 1.92e-10 or less. A negative plane angle thus coexists with modest damage
near the centreline. A material plane angle alone cannot give a crack path.

The independent 0.01° grid scan reproduced all twelve Coulomb births in the
common window within 0.00316°. Across both histories, the checker found
identical formed angles and section measures on consecutive endpoints, exact
inheritance of accepted age and slip as the next parent, and preservation of
the reference width. The normalized accumulated absolute physical-work errors
are 0.143% and 0.0960% of the respective added fracture energies.
[Per-step comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/turn_formation/history.csv)
· [Births, work, cost and connectedness](../../../benchmarks/reconstructed_fault/particle_free/evidence/turn_formation/summary.json)
· [Independent angle scan](../../../benchmarks/reconstructed_fault/particle_free/evidence/turn_formation/native-coulomb-births.csv)
· [Turning figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/turn_formation/turn-comparison.png).

The current-rule process exited 1 after 216.72 s. At 90 ms its 80th Newton
iterate had mechanics 3.95e-13 and phase KKT 5.94e-7, above the 1e-7 gate.
Coulomb accepted 90 ms with a connected front at x ≈ 6.136 m and 95 ms at
x ≈ 6.516 m. Those two later endpoints have no current-rule counterpart.
At the 100 ms endpoint the integrity monitor detected remote phase 0.119 and
boundary phase 0.0453. Coulomb exited 0 in 151.50 s. The late remote damage
limits interpretation of that endpoint as a single turning fault.


## Reproduction and remaining validation

The worktree was `pf-rsf-particle-free` at `c07eea8e7`. The original
large-run executable, checkpoint and checker were retained. The recovery
executable rebuilt the three affected solver/obstacle unity objects using
the retained CMake commands. A separate executable and matching checker
were built for both formation runs. The executable identities are independent
of the version banner. Both small runs used the same binary. The two inputs
derive from the tracked positive first-free-turn input with the box
repetitions changed to 32 × 16. [Binary, input, checkpoint and source identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/turn_formation/identities.json)
· [Commands and analysis](../../../benchmarks/reconstructed_fault/particle_free/turn_formation/README.md).

Keep the Coulomb option experimental. The comparison establishes sensitivity
to the extra formation assumption on this mesh. The advancing contour differs
by one cell and strict-positive first registration can freeze an angle at
phase below 1e-10. The next physical comparison would halve the timestep
across first registration and the 80–90 ms advance, starting each law from
its own **pre-registration** checkpoint. Compare the connected contour,
ahead damage and strip directions at common times before interpreting the
one-interval delay. Closed-contact interpretation also requires a physically
qualified normal-stress reference. The prepared lifecycle check restores
the earlier Coulomb parent and separately attempts a cross-law restore;
it is deferred until work resumes.
