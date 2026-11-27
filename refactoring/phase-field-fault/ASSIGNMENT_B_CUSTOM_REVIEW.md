# Assignment B — custom review instructions

Perform an independent, findings-first maintainer review of Assignment B on `pf-rsf`. Review the committed range `269cfd8b0..623bb1c57`. The intermediate commit `7dea28403` adds the experiment against the old implementation. A clean working tree still contains the entire review range. Read the diff and its production callers. Use the recap and working record as claims to check. This is a review-only task. Temporary probes in an isolated build are welcome. Leave the branch, reference data and tolerances untouched.

I need two decisions. Does the boundary-spanning fault now transfer slip-rate work consistently between the fault and the Stokes equations? Does the evidence explain the change in the 38-label free-fault solution well enough for me to authorize a new reference? Give a separate verdict on each.

## 1. Verify the two production actions

Trace one trial-rate perturbation through `project_boundary_slip_load`, material evaluation and the ordinary `FaultStressLoad::execute` local RHS before boundary constraints. Trace the reverse action independently from a virtual FE velocity through `sample_bulk` and `BulkPoint::add_balance`. Write the short identity those paths should satisfy. Check sign, shear factor of two, units, label interpolation, phase weight, spatially varying Maxwell coefficient and cell weights. Identify which terms cancel when the reverse rows are differenced. Limit the claim to the shear transfer that was measured.

Check the argument for **every local Q2 velocity test** on the supported cells. The transfer experiment uses affine virtual shear and one middle-label perturbation. The separate ridge probe uses a spatially varying velocity with a constant imposed rate. A discriminating spot check combines a nonconstant rate and a nonconstant test strain on the same inclined cut cell. Compare the actual assembled vector, component by component, to direct Q8 integration of the unprojected signed load against the local velocity basis. A scratch probe is sufficient. Verify the FE degrees, mapping, reference-cell projection and Stokes quadrature used in the executable. If the mathematical argument suffices for a case you did not execute, say so.

Reproduce or inspect the one-cell negative control. The old Stokes and reverse actions should be about 0.5696685171 and 0.4942359683 m. After repair both should be about 0.4942359683 m. The split integral is exactly 0.5 m. Explain the residual integration error and check what the inclined 8², 16² and 32² results establish with the discrete normalizer held fixed. Verify what was compared by CellId across MPI ranks and which comparisons were global sums.

## 2. Explain the first change in the free-fault solve

At initialization, geometry and accepted DG stress agree and all 38 rate differences exceed the historical tolerances. Label 4 changes from 0.135738293 to 0.138155429 m/s. Use the free 32² case and look at the **first differing assembly or Picard iteration**. Hold the fitted geometry, phase, parent stress and incoming trial rates identical. Compare the old and new local RHS with an independent evaluation of the slip term on a cell that contributes to the difference. Follow the change through the Stokes velocity/pressure and back into the returned section rates. A focused replay of the old load on identical inputs is particularly informative. Check which terms in the friction balance transmit the change.

Distinguish an expected consequence of changing the discrete operator, a defect in the implementation, and insufficient evidence to choose between them. Agreement on a prescribed three-label fault or a passing tangent around the new trajectory cannot settle the free-solution question alone. If one further comparison is needed, specify the smallest case, observable and result that would discriminate. Reference replacement is a separate author decision.

## 3. Check state ownership, controls and cost

- Follow the projected load from attempt entry through Picard updates, RHS-only assembly, rejection and retry. Check that a cell becoming inactive cannot reuse stale coefficients. Trace accepted DG stress and physical output back to the cut-cell constitutive update. Check that the interior-fault route still selects its own reaction. Flag a new API or duplication only if it introduces a concrete maintenance or correctness issue.
- Verify that the old executable passes the strict free-32 reference and the new executable still fails it at the original tolerance. Inspect changes to the prescribed-ridge oracle, test material, rate-tangent recorder and both control-driver modes. In particular, the strict driver exits at free-32 before the **narrow-versus-wide shear concentration** check, and the independent driver does not call that comparison. Determine whether it ran elsewhere. If outputs are available, execute that check on both grids and report the result. Distinguish a skipped comparison from a failed one.
- Inspect the new all-cell Q8 sweep at attempt entry and each Picard preparation. The working record gives 21.697 versus 28.389 s for 143 assemblies in each run. Check for redundant work and whether avoiding it would preserve the prescribed-rate and free-rate paths. Treat the single timing pair as an observation. Request optimization only for a demonstrated cost with a practical remedy.

## Review output

List actionable findings first, ordered by severity. For each, give a copyable repository-relative path and line, the trigger, consequence, evidence and smallest remedy. Report which checks you ran yourself, which came from archived outputs, and anything that blocked verification. Close by stating whether the weak-transfer repair is sound, whether the changed frozen trajectory is explained, and whether Assignment C can proceed or requires a specific prior action. Keep Assignment C's growing-fault direction experiment outside this review.

