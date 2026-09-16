# K5: coupled-state two/four-substep check

## Decision

**All three changes decrease under refinement, but the notch deepens and the
remaining timestep sensitivity is material.** Reuse the earlier coupled
one-step result and compare two half-steps and four quarter-steps, each starting
from the identical accepted step-9 checkpoint. All six requested accepted
substeps finish, with genuine nonlinear/fresh-linear convergence and verified
single candidate-state publication. No further steps or timestep levels run.

Successive-change ratios (two-to-four divided by one-to-two) are **0.655** for
the velocity deficit, **0.766** for neighboring contrast, and **0.865** for the
accumulated slip-gradient increment. These are decreasing differences, not
evidence of a timestep-independent limit. In particular, the last refinement
still changes the contrast 16.78% and the gradient increment 5.39%.

## Identical incoming state and numerical problem

- Source: `work-replay-50-local4/restart/01`, accepted step 9 at
  **483028744.22168553 s = 15.30625726 yr**. Its checkpoint files retain their
  hashes after both runs.
- Common final time: **922804465.597517 s = 29.24190894 yr**; the quarter-step
  clock differs by only one floating-point ULP (1.19e-7 s).
- Real interval: 439775721.37583148 s. Half dt = **219887860.68791574 s**;
  quarter dt = **109943930.34395787 s**. Artificial initialization is not rerun.
- Only the pending time/dt doubles in disposable checkpoint copies are retimed,
  using the previously tested archive procedure. Every other uncompressed byte,
  preceding dt, step index, vector, mesh/history file and benchmark slip/state
  remains identical at preparation. Ordinary later checkpoints may overwrite
  the disposable copies; the original input is untouched.
- Spatial formulation, mesh, fault, fixed phase, full I_h, both boundary
  corrections, background, friction, pressure, support and solver settings are
  unchanged. Existing controllers accept the requested subsequent steps. The
  accepted-clock guard stops at global steps 11/13, without a remainder step.

The [addendum](stage_K5_coupled_substeps_addendum.md) defines the opt-in time
discretization. Mechanics recomputes nodal T(V;Theta_old,dt), then interpolates
to QPs using immutable old Theta throughout Newton. Its verified nonsymmetric
Jacobian is retained. The existing terminal history routine commits that same
candidate **once** after convergence. Maxwell stress, transfer, advection and
accumulated slip advance normally between substeps; mature H stays inert and
C stays zero. This does not change the default production split algorithm.

## Final-time comparison

V_p=1e-9 m/s. Define

    deficit = 1 - V(39.95 km)/Vp
    contrast = [V(39.90 km) - V(39.95 km)]/Vp
    gradient increment = sum_k dt_k [Vp - V_k(39.95 km)]/(50 m).

The common incoming last-element slip gradient is **0.000720916008090**.
One-step slip is the previously saved noncommitting result's hypothetical
accepted increment; it was not rerun. The two/four-step values are actual
accumulations from their accepted substeps.

| Quantity | One coupled step | Two half-steps | Four quarter-steps |
|---|---:|---:|---:|
| V(39.90 km)/Vp | 0.954896631 | 0.975591930 | 0.993604246 |
| V(39.95 km)/Vp | 0.811847356 | 0.792396049 | 0.779660686 |
| Velocity deficit | 0.188152644 | 0.207603951 | 0.220339314 |
| Neighboring contrast | 0.143049275 | 0.183195882 | 0.213943560 |
| Accumulated gradient increment | 0.001654899299 | 0.001764990709 | 0.001860168785 |
| Final total slip gradient | 0.002375815307 | 0.002485906717 | 0.002581084793 |
| Theta at 39.95 km (s) | 9854069.17 | 10095961.50 | 10260872.51 |
| Slip at 39.95 km (m) | 0.804013700 | 0.798509130 | 0.793750226 |
| Lower-active nodes | 0 | 0 | 0 |

| Metric | Change 1 -> 2 | Change 2 -> 4 | Absolute difference ratio | Last relative change |
|---|---:|---:|---:|---:|
| Deficit | +0.019451307 | +0.012735363 | 0.65473 | +6.13% |
| Contrast | +0.040146607 | +0.030747678 | 0.76588 | +16.78% |
| Gradient increment | +1.1009141e-4 | +9.5178076e-5 | 0.86454 | +5.39% |

All metrics increase monotonically. The adjacent upstream velocity approaches
Vp while the final free node slows further; the effect is not merely a uniform
rate shift. No bound contact occurs. Contraction is modest, especially for
accumulated slip. No extrapolated limit or convergence order is claimed from
these three levels. This interval inherits the common earlier split-history
trajectory; it is not a full coupled-state history from initialization.

## Acceptance and publication checks

| Run/global step | Final relative bulk residual | Final relative surface residual |
|---|---:|---:|
| Half 10 | 2.42652e-13 | 2.90607e-9 |
| Half 11 | 2.45885e-13 | 3.38784e-10 |
| Quarter 10 | 2.47222e-13 | 3.78880e-14 |
| Quarter 11 | 2.43554e-13 | 1.50316e-9 |
| Quarter 12 | 2.56385e-13 | 4.87661e-9 |
| Quarter 13 | 2.77131e-13 | 6.78710e-9 |

- All are below the unchanged 1e-8 nonlinear target. **16/26** fresh-linear
  checks pass, with **600/828** total Krylov iterations. Actual surface inverse
  comparisons with UMFPACK remain enabled and pass.
- At every substep, all committed nodal states equal the candidates computed
  from the captured incoming Theta. At the 600 exported production QPs on the
  last two elements, the committed interpolation equals the state actually
  evaluated by final mechanics. Maximum discrepancy: **exactly zero**.
- Independent long-double exp/expm1 aging checks carried from the original
  step-9 Theta through each accepted rate: maximum relative errors
  **2.36e-16 / 2.81e-16**. Independent slip accumulation error <= **2.22e-16 m**.
- All 385920 real particle IDs are captured before first advection. Inert H,
  surface geometry, I_h and zero C checks pass throughout. Prescribed Vp is exact.
- First-substep state/Jacobian finite differences pass for both timestep sizes.
  The rebuilt surface-direct focused tests pass: **545 assertions, 2 cases**.
  No broad suite or extra MPI campaign was run.

## Audit repair and preserved limitation

One initial half-step attempt converged mechanically but stopped in the new
benchmark invariant checker before accepted output. The checker's snapshot was
registered in a postprocessor resume callback, which runs **before particle
deserialization**, and captured an empty particle-ID baseline. Move the snapshot
to first timestep entry, after loading and before advection, and require it to
be nonempty. The repeat's converged raw QP file is **byte-identical** to the
discarded attempt's file; no mechanical correction was involved.

The discarded attempt remains in `substeps2-failed-audit-order/` (175.461 s).
No physical state from it was used for the successful runs. Six successfully
audited substeps, plus this one discarded mechanical attempt, were executed.

As in the preceding A/B experiment, restore the exact saved frozen I_h only
after checking the cold restart recomputation differs by roundoff (4.44e-16
relative here). This narrow benchmark support is not a general committing
restart qualification. Histories are not reset between substeps.

## Cost, implementation and artifacts

Four ranks, sequential runs: **273.623 s** for two halves and **390.745 s** for
four quarters (664.368 s successful execution total). Largest child peak RSS
reported by the runner is **1560016/1562384 KiB** (about 1.49 GiB), not an
aggregate MPI memory measurement. The earlier one-step result is reused.

Changes: guarded permission for candidate-state evaluation in the bounded
committing benchmark; raw state export during its final assembly; benchmark
resume/clock/invariant wiring; candidate-publication audit; independent runner
and analysis. No new state update, solver, or physical parameter is introduced.

Evidence root: `benchmarks/reconstructed_fault/bp3/coupled-substeps-50-local4/`.
`comparison.csv`, `comparison.json` and `analysis.log` contain the final values
and checks. Each `substeps2/` and `substeps4/` contains input/provenance hashes,
clock-retiming verification, `run.log`, accepted steps, fault/history exports,
`candidate_commit_*.csv`, raw final-iteration `state_qp_step*_rank*.csv`, work
stress diagnostics and execution metadata. Build/unit logs are at the root.

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
build-pf-cpdi/aspect-release --test '[fault_surface_direct]'
python3 benchmarks/reconstructed_fault/bp3/run_coupled_substeps.py prepare 2
python3 benchmarks/reconstructed_fault/bp3/run_coupled_substeps.py run 2
python3 benchmarks/reconstructed_fault/bp3/run_coupled_substeps.py prepare 4
python3 benchmarks/reconstructed_fault/bp3/run_coupled_substeps.py run 4
python3 benchmarks/reconstructed_fault/bp3/analyze_coupled_substeps.py
```

Runners refuse overwrites. No further refinement or production-integrator
adoption follows automatically from this bounded check.
