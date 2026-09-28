# BP5 normal-stress diagnostic: implementation and qualification status

## Decision

The diagnostic and restart staging are implemented and compile. The requested
late restart has **not** been executed: its checkpoint is on the server, not in
the local copied profile outputs. The exact target is `restart/01`, accepted
step **5612**, time **5310111071.5634108 s**. The prepared branch permits five
new accepted states (5613–5617), or an earlier accepted-state wall stop.

Local four-rank evidence verifies the stress split, projection and ownership.
End-to-end enabled/disabled restart equivalence remains **pending**. One local
mechanical solve converged and produced valid captured stress files, but a
subsequent diagnostic-only phase-norm audit failed. That audit was corrected;
the corrected replay reached its 200-second cap before convergence. Neither
run is represented as a completed diagnostic lifecycle or a valid new checkpoint.
No automatic retry, control simulation, or long server run followed.
Small copies of both logs and the partial split-analysis JSON are retained in
`normal-stress-diagnostic-verification/`; raw CSVs remain in the scratch paths
below. The partial JSON predates the final weighted-roughness/controller reporting
enhancements; its closure and ownership results remain valid.

## What is evaluated

The opt-in surface-system observer captures the same physical pressure and
current constitutive stress as the work-measure surface equation, with its
incoming FE Maxwell history and current velocity/crack-strain contribution.
It does not reevaluate a Maxwell update after history publication.

At each owned production Stokes quadrature point it records
`p`, `-stress:N`, and the separately supplied background normal traction.
The off-diagonal tensor contraction includes the factor two. The background
is not assigned to pressure; BP5's nominal 50 MPa remains a separate term.
There is no inferred residual correction. Friction uses the raw QP total;
the nodal weak coefficients are a diagnostic representation of that total.

The pressure/deviatoric/background loads use the existing `JxW*chi*N_i`;
the unchanged physical mass matrix uses `JxW*chi*N_i*N_j`. In 2-D, loads have
units Pa m, mass/row sums m, and projected coefficients and `f/m` Pa.
Bound-active mechanical rows do not remove physical diagnostic projection rows.
Raw samples include the exact cell/QP/fault identity, geometry, phase, I_h,
localization and work weight. The source association includes the existing
boundary continuations without duplicating them.

The last successfully built linearization supplies the capture; residual-only
line-search trials cannot replace it. Publication checks nonlinear success,
absolute step/time and equality with the accepted absolute V vector. Restored
history inventories are explicitly not step-5612 mechanical traction. The
first newly accepted state supplies the mechanical baseline.

No equations, tolerances, quadrature, eight-panel I_h, histories, mesh, support,
pressure convention or timestep criteria were changed. The added response
fields change an in-memory C++ type, so **ASPECT and all loaded plugins must
be rebuilt together**. Checkpoint serialization is unchanged.

## Local evidence

The available earlier checkpoint came from
`weakening30-dc010-ell100/loading-surface8/startup/restart/01`: accepted step 1,
time 1,000,000 s. It is useful for implementation checks, not for explaining
the late server noise. The staged copies preserve every checkpoint file's
SHA-256 and are independent files, not writable hard links.

### Completed mechanical solve, followed by diagnostic audit failure

Artifacts: `/tmp/bp5-normal-diagnostic-enabled/run.log` and its
`output-normal-diagnostic/normal_profile_2.csv`, `normal_qp_2_rank*.csv`.
The accepted mechanical evaluation was step 2 at **2086454.6055890049 s**.

| Check | Result |
|---|---:|
| Final relative bulk residual | 3.320421e-12 |
| Final relative surface residual | 7.958557e-10 |
| Fresh linear residual / requested target, direction 1 | 1.104956e-3 / 2.087087e-3 |
| Direction 2 | 4.671337e-6 / 1.038096e-5 |
| Direction 3 | 7.974509e-10 / 8.917648e-10 |
| Maximum projected split closure | 4.842877388e-7 Pa |
| Row-mass-weighted RMS closure | 1.463547901e-7 Pa |
| Pressure/deviatoric projection residual divided by mass | 1.851050939e-14 Pa each |
| Background projection residual divided by mass | 9.540802593e-9 Pa |
| Raw owned window samples | 256,655 |
| Duplicate cell/QP/fault owners | 0 |
| Independent tensor contraction discrepancy | 4.547473509e-13 Pa |
| Raw normal-split closure | 0 Pa |

All accepted line-search steps had alpha 1. These closure errors are roundoff
scale and far below the 0.5 Pa dimensional comparison scale obtained from the
configured 1e-8 relative accuracy and 50 MPa traction. This comparison is not a
new physical acceptance threshold or a claim that all physical errors are small.

Independent reaccumulation of the raw four-rank exports reproduces the native
assembled loads on **426 rows whose complete support is inside the two export
windows**. Maximum discrepancies divided by row mass are 4.913215e-13 Pa
(pressure), 3.821434e-13 Pa (deviatoric) and 2.384992e-7 Pa (background).
Reaccumulated row mass differs relatively by at most 3.554039e-15. Boundary
rows of the export windows are deliberately excluded because their missing
outside-window QPs are not zero. This is not a full-fault global-total test;
the final plugin writes independent global totals for that remaining check.

After writing these captures, the old audit called `l2_norm()` on a ghosted
Trilinos vector, whose overlap map cannot provide that norm. The fix sums
squares on owned phase DoFs and performs an MPI sum. It changes only an output
fingerprint, not the phase equation or stress evaluation.

### Corrected replay and pending comparison

`/tmp/bp5-normal-diagnostic-verified/run.log` preserves the corrected replay.
It timed out at 200 s during the third linear solve. The last reported surface
residual was 9.278392e-6, so **it did not satisfy final convergence**. The
prepared disabled-capture branch is
`/tmp/bp5-normal-diagnostic-control-verified`; it was not launched.
Consequently, post-fix complete output, terminal checkpoint, full global-load
closure and enabled/disabled state equivalence remain unverified in execution.

Cold restart reconstruction reported an I_h relative difference of
3.33067e-16 before restoring the validated frozen values. This is the only
locally measured restart reconstruction difference; it is not evidence about
the inaccessible step-5612 checkpoint.

## Commands and scope of tests

```bash
cmake --build build-pf-cpdi --target aspect -j4
cmake --build benchmarks/reconstructed_fault/bp5/build \
  --target bp5_normal_stress_diagnostic bp5_steady_initialization -j4
python3 benchmarks/reconstructed_fault/bp5/test_normal_stress_diagnostic.py
```

The two Python tests pass: inclined tensor/consistent mass closure and staging
immutability/unchanged settings. Parameter validation passed on the staged local
restart with the registered plugins. The library containing the additional local
test plugins was also rebuilt for the changed response type. No complete ASPECT
suite was run.

The local enabled replay used four ranks, one thread per rank, explicit B/G and
the pivoted tridiagonal inverse, with `timeout 240 mpirun -np 4 ...`; it failed
only in the postprocessing norm audit described above. The corrected replay used
`timeout 200 mpirun -np 4 ...` and exited 124. An earlier wrapper attempt could
not launch because `/usr/bin/time` is absent; that failure was preserved as
`launch-missing-time.log`. No simulation result is attributed to that wrapper.

## Timestep interpretation and questions still open

The unchanged predictor bounds **max (b/a) |log(Theta_pred/Theta)|**, not raw
logarithmic state change. It uses the exact configured aging map at the previous
accepted rate. The diagnostic also records the realized change using the newly
committed state. The controller record after accepted step k-1 selects step k;
the plotting script respects that offset.

For the local implementation check, the state-predictor proposal controlled
the selected 1086454.6055890049 s step. Other proposals were approximately
1.220714e10 s (convection), 6.667934e6 s (fault), 1e7 s (ceiling), and
1.91e6 s (growth cap). These are **not** the late restart's limiter values.

The requested late-state conclusions remain open: pressure versus deviatoric
noise, cancellation/reinforcement, amplification by consistent projection,
association with refinement or I_h, and increments over five states. Earlier
local implementation data cannot answer them. The scripts now provide the
native loads, consistent coefficients, row-scaled loads and unsmoothed QP maps
needed for that discrimination. No smoothing or causal explanation was imposed.

Next action: rebuild on the server, stage the exact checkpoint and actual resolved
production input using `normal_stress_diagnostic_README.md`, and execute the
bounded diagnostic and one-step disabled-capture control. Keep the original
checkpoint untouched. Interpret the stress decomposition only after closure,
restart invariants and unchanged-mechanics comparison are checked.
