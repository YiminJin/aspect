---
name: verify-fe-operator
description: Verify or review a finite-element residual, tangent, transfer, projection or assembled load against its intended weak form. Use for ASPECT/deal.II and similar FE changes where signs, measures, quadrature, state timing or MPI ownership can alter the result.
---

# Verify a finite-element operator

Follow one operator from an equation to a result the solver consumes. In review mode, use read-only evidence or isolated scratch calculations. In implementation mode, add the smallest discriminating regression available through the project's normal entry points.

1. **Write the contract.** State the weak form or discrete identity, trial and test spaces, sign, physical units, integration measure, and the state held fixed. For a nonlinear update, identify the accepted parent, trial iterate, and publication point.
2. **Trace the real call path.** Locate the caller, FE inputs, quadrature and mapped gradients, accumulation into a local vector or matrix, and the consumer. Follow face terms, constraints, reduction or projection when they affect this operator. Use repository-relative paths and symbols.
3. **Choose an independent reference.** Use an analytic integral, a manufactured field, or separate quadrature whose error you can estimate. Pick nonconstant data that expose a plausible wrong implementation. Compare the actual operator action against that reference. If the check only integrates exported weights or exercises an observer, specify which production step remains untested.
4. **Match the check to the claim.** For a residual tangent compare its action to centered residual differences at fixed history over a useful range of perturbations. For transfer compare paired virtual work on the same discrete measure, then quantify differences from production quadrature separately. For a projected field compare independent cell moments on a cell crossed by a knot or support edge when relevant. For distributed assembly compare by global identities and verify active contributions on both ranks. For an acceptance hook compare a rejected/retried step with clean steps from the same parent. Select only the checks the change warrants.
5. **Show that the check discriminates.** When practical, reverse a sign, drop a weight or use the wrong time level in an isolated variant and show what fails. Include the reference's error, the absolute defect and a meaningful scale for the tolerance. Use an absolute scale for a zero reference.
6. **Report the evidence.** Give the production path, command or input, expected and observed values, tolerance rationale and the first unverified link. Distinguish results you executed from source inspection and previously recorded runs. Preserve accepted baselines and consult the project's existing verification guide for canonical commands and resource limits.

