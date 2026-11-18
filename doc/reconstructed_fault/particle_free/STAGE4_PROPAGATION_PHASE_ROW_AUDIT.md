# Phase-row audit from step 68

The replay starts at 5.44 ms on the 256 × 128 mesh. It inspects only the trial
toward 5.48 ms. The accepted 80 μs trajectory and the original four-rank 40 μs
timeout are recorded in the [timestep check](STAGE4_PROPAGATION_TIMESTEP_CHECK.md).

## What is measured

For the midpoint stage let x be a Bernstein coefficient, l its accepted parent
value, u = (1 + l)/2 and m the assembled row mass. With scaled phase gradient g,
the signed natural residual is x − clamp(x − g/m, l, u). The prediction x − g/m
determines whether a row is lower-active, free or upper-active. A row exactly at
its obstacle can release if the gradient there is negative. The stage bounds
enforce the endpoint bounds through the relation φ_end = 2x − l.

Each exact constrained pass fixes the bound displacements and solves for the
remaining correction z. The restricted matrix B masks active phase columns,
sets active rows to identity and includes the assembled mechanical and section
couplings. The diagnostic reapplies that same B to the returned z and measures
‖Bz − b‖₂ **before** the solver enforces active values and reclassifies rows.
The next observation measures the complete linearized VI with the displaced
stage coefficients and counts every transition. These two diagnostics have
different units and need separate thresholds. The immutable tangent and
active mask are shared by the Krylov solve and the explicit residual check.

The nonlinear trial diagnostics compare the assembled g(x + δ) with
g(x) + Jδ at the same trial, using the actual displacement after affine
interpolation to a prescribed bound. Native quadrature tables identify the
section force and phase driving on each sampled cell. Bernstein locations
denote control coefficients. The corresponding physical values at quadrature
come from evaluating the polynomial.

## Reproduction

The worktree was on pf-rsf-particle-free at b32a2d4f. A separate executable
relinked two diagnostic translation units with the retained material-overnight
build objects. Its SHA-256 is
`207b563efc32a0a17bb6c266c08e93d18b3c8c2ca421180c613ee1d5e67d9ecf`.
The original half-step executable has SHA-256
`089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`.
Both use the same checker, step-68 slot 03 payload, 40 μs timestep, adaptive
exact-first search and four bound MPI ranks. The replay writes into a private
directory and requests a diagnostic stop before Newton 12. The replay
launcher, build script, parameter file and raw log are in
`build-particle-free/phase-row-audit/`. The
[instrumentation diff](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/row-audit-instrumentation.patch)
shows the added measurements relative to the incoming source.

## First constrained solves

At Newton 0 the largest initial residual (0.0194300) is on row 208869,
coefficient 6 in the cell centered at (6.109375, 1.921875) m. Of the 294,912
stage coefficients, 292,011 classify lower-active and 2,901 free. Reapplying
the restricted matrix exposes a discrepancy on the very first pass.

| Pass solved | FGMRES estimate | Explicit norm on same mask | Leading row at next classification | Observed change |
|---:|---:|---:|---:|---|
| 0 | 9.11e-16 | 2.066e-3 | 258131 | Lower to free at its lower value 0.01776045. The scaled gradient is −0.0280920. |
| 1 | 9.51e-16 | 9.51e-16 | 258180 | Lower to free at 0.00116491 with scaled gradient −0.00467041. |
| 2 | 9.35e-16 | 1.802e-2 | 258252 | Free to upper in the classification. Its trial value is −4.889e-8 and its gradient per mass is −95.867. |
| 3 | 7.53e-15 | 7.53e-15 | 184007 | Lower to upper at value 0 with gradient per mass −0.506816 and upper bound 0.5. |

After pass 0, 257 lower rows release and 206 free rows become lower-active.
After pass 2, 1,149 lower rows classify upper-active. The initial discrepancy is
already present inside the fixed linear problem. Conversely, pass 1 satisfies
its actual linear tolerance and still causes reclassification, as permitted by
the VI. The leading rows after passes 0, 1 and 2 lie in cells centered at
(6.046875, 2.078125), (6.078125, 2.109375) and (6.203125, 2.046875) m.

The installed deal.II 9.6.2 FGMRES implementation checks an Arnoldi residual
estimate before applying the final projected correction. Its successful inner
exit can return without a fresh matrix application. In this configuration
`control.last_value()` reports that estimate. The measured gap establishes
a failure of the inner stopping certificate on some passes. It does not
identify which floating-point operation first lost accuracy.

At pass 4 the requested linear tolerance is 1.0851e-14. After 365 iterations
the reported estimate is zero. The explicit phase-block norm is 1.52391e18 and
the mechanics norm is 0.795055. Both observations precede the next change of
mask. The pass-5 VI leader is row **258727**, local coefficient 4 in the cell
centered at **(6.296875, 2.109375) m**. Its trial value is −35913.1651 and
g/m is −3.28998e15. Its old status was free. Its projected target is the upper
bound 0.5 and its signed residual is −35913.6651. In that reclassification,
8,528 previously lower or free rows select the upper bound and 2,419 upper rows
switch to lower. These counts describe the trial search. Its primal
infeasibility prevents admission. The next solve (pass 5) meets its explicit
tolerance again. Passes 6–8 repeat the false-convergence pattern, culminating
in an explicit norm of 8.10e25 at pass 8. The original logs and the independent
replay agree on the resulting phase VI residual of 1.68267e9.

## Nonlinear response at the lower obstacle

The first appearance of the later residual scale is already visible at Newton 3.
Row **258249** is coefficient 3 of cell 28694, at Bernstein location
**(6.1875, 2.046875) m**, the midpoint of the cell's left edge. Its lower
obstacle is 1.1932604311532853e-13 and its row mass is 1.0850694444444445e-4.
The full displacement takes it from 6.38226e-8 to its obstacle exactly. The
linearized model predicts a zero projected residual there. Nonlinear
reassembly gives g/m = −9.51885e-4, releases the row, and makes it the worst
phase coefficient. This particular trial passes the existing search test
because its functional decrease is resolved. The following Newton update
therefore starts from the released row.

The quadrature points in that cell sample sections 1188–1193. All six already
have their formed flag set. At the full trial the sampled phases return to
9.11e-14–2.65e-13 and section V drops from 1.53e-10–1.26e-7 to
6.90e-16–2.44e-15 m/s. Section resistance falls from 85.3–105.4 to
48.3–52.1 Pa. Integrating the increase in local driving with the actual
Bernstein weight and dividing by the fracture scale and row mass gives
+0.104422. It enters the gradient with negative sign. The observed change in
g/m is −0.104660, of which the tangent predicts −0.005909. At fixed planes
the fracture gradient is affine. The finite-displacement nonlinear remainder
comes from the mechanical driving, including the eliminated section response.

At Newton 8–11, each full trial lands row 258249 exactly on the same lower
obstacle. Each time the predicted projected residual is zero and the measured
residual is −9.52038e-4. At iteration 11 the linear mask is stable across the
last pass (291,844 lower, 3,068 free). The full linear VI is feasible to a
projected residual of 1.90e-13. At nonlinear contact the coefficient goes
from 9.98763e-10 to 1.19326e-13. The predicted gradient per mass is +0.0740181
and the assembled gradient is −0.000952038. This sign change requires release
from contact. The six sections retain their formed status, their resistances
fall from 72.7–93.0 to 48.3–52.1 Pa, and the trial mechanics norm is 1.12e-16.

The full-trial functional decreases at iterations 8–11 are smaller than the
2.90e-13 energy roundoff allowance. Armijo alone admits them. The additional
merit condition rejects them because the phase residual increases. Each half
trial reduces the merit. A different coefficient, **36702 at (1.8125,
1.96875) m**, remains free and controls every half-trial maximum. At iteration
11 its predicted and measured residuals are −1.04764e-6 and −1.04771e-6.
Thus continued halving reduces the residual visible on the left while repeatedly
avoiding the strongly nonlinear return to the obstacle on the right.

An independent scalar check integrates the section mobility M over each full
parent-profile strip using the 6 × 6 Gauss weights and actual section measure.
At the full contact trial it evaluates the exact stage aging and regularized
resistance F(V) with the accepted age. A centered difference using
V ± 1e-5 |V| measures dF/d(V + M F). Across six sections and six full trials
(36 comparisons), resistance agrees within 3.6e-14 Pa and the derivative agrees
within 2.9e-11 relative. Near contact the native derivative ranges from 1.75e14
to 6.29e14 Pa s/m. This checks the condensed scalar derivative at those states.
The finite-step row observation also includes changes in mobility and mechanics.
A full coupled infinitesimal-direction test would require perturbing those
fields together.

## Decision and retained evidence

The four-rank diagnostic took **56 min 36 s** including the launcher. Its
selected stop before Newton 12 produced native exit 1. All 134 comparable
Newton, constrained-pass, linear-solve and search observations match the
original log to its printed precision. All six checkpoint payload hashes agree
between the source, the earlier half-step copy and the replay copy. The
protected executable, checker and incoming source edits were verified. Physical
time on the diagnostic branch remains 5.44 ms.

**First repair experiment:** verify the actual restricted residual on return
from FGMRES, using the existing mask, right-hand side and linear tolerance. If
the returned direction fails that check, discard it and exercise the existing
shifted fallback before passing its coefficients into another classification.
Fourteen of the 36 completed linear solves failed this explicit check in the
audit. Compare cost, row 258249 and the nonlinear acceptance gates on this
same 5.44–5.48 ms interval. The later contact difficulty persists even on
accurately solved linear directions, so a reduction in inner work alone would
leave physical timestep accuracy open.

[Diagnostic figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/row-audit.png)
· [Passes, transitions and trial leaders](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/row-audit.json)
· [Section derivative comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/row-audit-section-derivatives.json)
· [Raw log, sparse rows, quadrature points and row maps](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/row-audit-traces.tar.gz).

The subsequent [guarded replay](STAGE4_PROPAGATION_LINEAR_RESIDUAL.md) implements
the explicit residual test, checks the full coupled tangent, and records the
accepted-state comparisons using the original physical tolerances.
