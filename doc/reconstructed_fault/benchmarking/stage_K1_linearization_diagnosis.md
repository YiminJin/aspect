# K1 dt=.5, t=4.5: bounded failing-linearization diagnosis

## Finding

The immediate failure is **false convergence of the condensed linear solve**.
At Newton iteration 2, FGMRES reports a residual of 1.55177e-20, but a fresh
application of the same condensed operator to its returned direction gives
2.11212e-11. This is 1.282 times the right-hand-side norm and 1.282e9 times
the requested linear tolerance. The resulting direction increases the bulk
residual, and all six Armijo candidates are correctly rejected.

The large pressure-gauge component in this direction strongly implicates
pressure-nullspace contamination. The replay establishes the false convergence
and the resulting bad direction; it does **not** fully separate Arnoldi/projected-
solve residual-gap mechanisms from finite-precision action on the near-null
mode. It also does not establish that the nonlinear stopping target will be
attainable after the linear solve is repaired. Do not change that target on
the strength of the present evidence.

This is separate from both the accepted physical pressure-normalization
correction and the previous surface-scale correction. Neither a new scale nor
a production solver correction was implemented in this task. Gate K1 remains
unmet, and K2 has not started.

## Replay and controls

One Debug, one-rank replay used the unchanged 16x64, dt=.5 s fixture, with
an output-directory override only. It was bounded to 1800 wall seconds and
finished with the expected failure at 4.5 s in **622.68 s**, peak RSS
**462684 KiB**, exit status **1**. All nine accepted states, t=0 through 4 s,
match the saved pressure-corrected run exactly in the exported velocity,
phase, V, surface histories and particle H/stress comparisons.

Temporary, opt-in source probes activated only at timestep 9. They recomputed
linear residuals, applied the full block Jacobian, and made non-committing
residual evaluations on private copies. Probe-only field changes were never
used as the mechanical base or committed as histories. Assembly controls,
RHS and current linearization point were restored before the ordinary line
search. The matrix-building and residual-only evaluations at each base agreed
exactly. The failing base residual and candidate sequence reproduce the
previous run; instrumentation did not turn the failure into a pass.

The temporary changes to `source/simulator/solver.cc` were then removed. Its
SHA256 before and after is
`fca059d967049353d9913477ddccf4a5385ee68c60f163b300f1655e5e7b895f`.
The restored Debug executable was rebuilt with `-j4`. Existing working-tree
changes, including the accepted pressure correction, were preserved.

All artifacts below are in
`benchmarks/reconstructed_fault/uniform_shear/linearization-diagnosis/`:

- `replay.prm`, `parameters.diff`: resolved parameters differ only in output path.
- `instrumentation.patch`, `probe.inc`, `solver-before.cc`: recoverable probes
  and the exact pre-probe source checkpoint; **not included by production now**.
- `replay.log`, `diagnostics.json`: raw and parsed observations.
- `replay.resources.json`: command, elapsed time, memory, executable/plugin hashes.
- `accepted-comparison.json`: unchanged accepted-state comparison.
- `build.log`, `restored-build.log`, `before.sha256`, `restored.sha256`:
  build and restoration evidence. The rebuilt binary has its own recorded hash.
- `restored-stage-i-unit.log`: 31 assertions in 7 Stage-I cases passed after
  restoration. The complete ASPECT suite was not run.

Commands (the opt-in instrumentation must first be applied to reproduce probes):

```sh
cmake --build build-pf-cpdi --target aspect.exe.debug -j4
ASPECT_K1_LINEARIZATION_AUDIT=1 python3 \
  benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py \
  benchmarks/reconstructed_fault/uniform_shear/linearization-diagnosis/replay.prm \
  --timeout 1800
python3 benchmarks/reconstructed_fault/uniform_shear/linearization-diagnosis/summarize.py \
  benchmarks/reconstructed_fault/uniform_shear/linearization-diagnosis/replay.log
build-pf-cpdi/aspect --test 'Stage-I*'
```

## Absolute residual and fixed scales

The bulk quantity is the Euclidean norm of the assembled velocity and scaled
continuity residual blocks. Its units in this 2-D fixture are Pa m (force per
unit out-of-plane thickness), **not** a pointwise stress norm. Continuity is
scaled by `pressure_scaling` = **1.9950083229264674e8 Pa s/m**; solver pressure
is physical pressure divided by this factor.

| Quantity | Measured value |
|---|---:|
| Initial bulk residual | 5.8727992293055973e-5 |
| Zero-velocity bulk reference norm | 6.606833065309758 |
| Floor factor, max(1e-9, sqrt(machine epsilon)) | 1.4901161193847656e-8 |
| Floor contribution to scale | 9.844948448702332e-8 |
| Fixed bulk scale | 5.8727992293055973e-5 |
| Absolute nonlinear bulk target, 1e-8 times scale | **5.872799229305598e-13** |
| Fixed surface RMS scale | 128.00324286710813 Pa |
| Absolute surface RMS target | 1.2800324286710813e-6 Pa |

The floor is **not active**: the initial bulk residual sets the scale.

| Newton iteration | Velocity residual | Scaled continuity residual | Total bulk residual | Surface RMS, Pa |
|---:|---:|---:|---:|---:|
| 0 | 5.872799229305597e-5 | 4.649350751776555e-13 | 5.872799229305597e-5 | 5.707917593296965e-1 |
| 1 | 5.553574343352778e-12 | 2.101158852914984e-11 | 2.173313232153726e-11 | 4.704277031508383e-4 |
| 2 | 1.084817703556094e-12 | 4.379575424468040e-13 | **1.169887284713687e-12** | **3.181412404176262e-10** |

At iteration 2 the bulk is 1.992 times its target. The surface is already well
within its target; its contribution does not explain rejection.

Independent non-committing field subtraction isolates these velocity-row weak
contribution norms at iteration 2:

| Contribution | Norm, Pa m |
|---|---:|
| Current velocity/viscous contribution | 6.647476281164068 |
| Bulk pressure contribution | 1.484187975983414e-6 |
| Frozen beta*tau_old contribution | 3.842519496509210e-3 |
| B V action | 6.646826217945598 |

These are vector norms, not scalar terms that may be subtracted to reconstruct
the residual. They demonstrate substantial cancellation. The frozen slip-history
term was not separately isolated. Fresh residual assembly at the identical base
differs from matrix-building assembly by **zero in both bulk blocks**.

## Achieved linear residual, not just the reported estimate

For C=A-B K_V^-1 G and condensed RHS b, the check is a fresh evaluation of
`||C delta_x-b||`. The uncondensed check independently applies
`[A,-B; G,-K_V]` after slip-increment recovery.

| Newton iteration | ||b|| | Requested linear tolerance | FGMRES iterations | Reported residual | Fresh true residual |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.946803378646943e-2 | 2.946803378646943e-11 | 12 | 2.171135542436826e-11 | 2.171144473352548e-11 |
| 1 | 2.430754452960792e-5 | 2.430754452960792e-14 | 12 | 2.189056404255706e-14 | 2.189053776754881e-14 |
| 2 | 1.647300249127905e-11 | 1.647300249127905e-20 | 37 | **1.551772095697243e-20** | **2.112123372599594e-11** |

The iteration-2 full-block bulk equation errors are 1.371582598892489e-11
(velocity) and 1.606183773886665e-11 (continuity), agreeing with the condensed
check. Relevant action norms are:

| Action | Norm |
|---|---:|
| A_uu delta_u | 2.767043536232826e-11 |
| A_up delta_p_solver | 7.044576478603776e-12 |
| A_pu delta_u | 1.607651482809141e-11 |
| B delta_V | 2.217537936781575e-11 |

The physical velocity increment norm is 5.178648370816739e-15 m/s, whereas
the physical pressure increment norm is **4894.084729245132 Pa**. The full
trial removes **147.2279770562702 Pa** of mean pressure. With 1105 pressure
DoFs, that mean times sqrt(1105) matches the pressure norm to 4.44e-15 relative.
Together with the tiny pressure-gradient action, this is strong evidence of
an almost pure pressure-nullmode increment. It is evidence about the Krylov
direction, not a failure of the already-corrected physical trial normalization.

The installed deal.II implementation
`/opt/dealii/9.6-local/include/deal.II/lac/solver_gmres.h:2227` checks the Arnoldi
residual estimate and then forms the final vector from the projected solution;
it does not perform another true-residual check on that final vector before
returning success. The coupled wrapper likewise had no final check. This
explains how the demonstrated residual gap reached nonlinear line search;
it is not, by itself, proof of a defect in the library's mathematical algorithm.

## Directional consistency and Armijo

Use the actual recovered Newton direction d=(delta_x,delta_V), with homogeneous
constraints and the existing physical pressure conversion. Evaluate centered
differences `[R(x+h d)-R(x-h d)]/(2h)` non-committingly. Here h multiplies the
already tiny Newton direction; it is not a unit-sized physical perturbation.
Every tested trial remains admissible. Bulk residual is the negative of ASPECT's
assembled Newton RHS, so its finite-difference sign is reversed accordingly.

| Multiplier h | Bulk FD norm | Velocity action error | Continuity action error | Surface RMS action error, Pa |
|---:|---:|---:|---:|---:|
| 1 | 2.182345115869302e-11 | 6.385680479508646e-13 | 2.384397161958355e-13 | 8.963814908578375e-15 |
| 100 | 2.181541876167827e-11 | 1.035919340488607e-14 | 2.450792153655399e-15 | 1.191688152903700e-16 |
| 10000 | 2.181539471962363e-11 | 8.097343588423332e-15 | 2.484971624044077e-17 | 1.254900058568720e-18 |
| 1000000 | 2.181539900604002e-11 | 8.369664800308675e-15 | 2.492035527559528e-19 | 1.476833228873043e-20 |

The assembled full bulk Jacobian action norm is 2.181535872767747e-11.
At large h the discrepancy is about 3.8e-4 relative, overwhelmingly smaller
than the order-one linear equation error. This does not certify the operator
to the impossible-to-resolve 1e-20 level with these finite differences, but
does not support a wrong block sign/derivative as the dominant failure.
Subtraction/representation noise is visible at h=1, so a residual-floor question
remains after the linear-solve problem is repaired.

| Armijo alpha | Trial absolute bulk residual | Trial merit / allowed merit |
|---:|---:|---:|
| 1 | 2.142841286676483e-11 | 335.53 |
| 2/3 | 1.419729275318313e-11 | 147.28 |
| 4/9 | 9.387176701557935e-12 | 64.39 |
| 8/27 | 6.152366231175279e-12 | 27.66 |
| 16/81 | 4.074744359309014e-12 | 12.13 |
| 32/243 | 2.721439519774270e-12 | 5.41 |

The base merit is 1.984119130079724e-16. All candidates are admissible but
fail sufficient decrease; exhaustion correctly preserves committed state.
The six evaluations are the configured first candidate plus five reductions,
not an enlarged line-search budget.

## Minimal correction proposal — awaiting review

1. **Do not return a falsely converged linear direction.** Recompute the true
   condensed residual on the returned iterate. Use the existing linear tolerance
   and total iteration budget. If residual replacement/restart is used, keep it
   bounded by that same budget; otherwise use the existing MPI-safe linear-failure
   path. An estimated residual alone must not authorize nonlinear line search.
   This safety correction is directly justified by the replay, but a guard
   alone would report linear failure rather than complete K1.
2. **Handle the pressure nullspace inside this prescribed-pressure linear solve.**
   For the closed/periodic K1 configuration, establish the discrete constant-
   pressure null vector with homogeneous constraint semantics, then solve on
   its complement, including compatible operator/preconditioner actions. Reuse
   deal.II/ASPECT constraint and vector infrastructure; do not add a new physical
   parameter, modify B/G/K_V physics, or normalize a Newton direction to the
   nonzero physical surface-pressure target. Check both the projected true
   linear residual and the removed compatibility component. Do not silently
   discard a significant incompatible RHS or transfer this operation to
   true-normal-stress or absolute-pressure boundary configurations.
3. **Leave the nonlinear scale and tolerances unchanged for the first corrected
   replay.** The current nonlinear target is close to observed assembly noise,
   but this run cannot tell whether a properly solved, gauge-controlled direction
   can reach it. If it cannot, return that separate measured floor for review
   rather than folding a stopping-rule change into the linear fix.

Focused regression proposal:

- A small prescribed-pressure coupled block solve with a nearly balanced bulk
  RHS and a pressure null mode: compare against an explicit full two-block
  solve on the pressure complement, and assert the **fresh** residual meets the
  unchanged target. Exercise a gauge-contaminated direction/optimistic residual
  estimate so omitting the final check fails the test. Verify homogeneous
  constraints, gauge-invariant physical increments, and detection of significant
  RHS incompatibility. Do not use independent per-vertex K1 scalar solves.
- Run the existing failing dt=.5 fixture through 4.5 s on one and two ranks,
  with true-residual assertions on every returned linear solve. Preserve the
  existing forced exhaustion/rollback tests. Continue through 6 s only after
  the failing step genuinely converges; no initialization or settings bypass.

The unresolved distinction before asserting a complete repair is whether a
pressure-complement solve alone removes the residual gap and then achieves
the unchanged nonlinear target, or exposes a separate bulk assembly floor.
No second replay or speculative production correction was made in this task.
After correction review and completion of the remaining K1 convergence checks,
the existing K2.1 design remains the next stage, not work begun here.
