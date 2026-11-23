# Increment 3B · Crack geometry and a phase-only surface-energy diagnostic

**Assignment for Codex.** Start in the ASPECT 3.1.0 `pf-rsf` worktree at `5bed31e635a69acc4d7ce6a8994a0f0916d11271` or a descendant containing the completed 3A repair. Implement three selectable crack-geometric laws and use them in an ordinary ASPECT postprocessor that measures crack-surface energy from a phase field. Demonstrate all three in a calculation containing an ordinary material, one phase composition, and zero reconstructed faults. Finish with the corrected mature AT1 regression and stop for review.

## Starting point and ownership

Read the root `AGENTS.md`, the 3B row in `PLAN.md`, and the **Increment 3A repair** section at the top of `STATUS.md`. Put this assignment at `refactoring/phase-field-fault/INCREMENT_3B_TASK.md`. Update the active pointers in the plan and working record. Preserve the historical records and link the 3B evidence once. The eleven 3A repair checks met their recorded criteria. The remaining independent free-RSF donor gap and the pre-existing original-elasticity comparison are recorded limitations of the baseline.

Work in two reviewable changes if convenient: first the mathematical laws and their unit tests, then their production consumer and integrated examples. Before choosing a public interface, walk through one future use: a phase-evolution assembler should be able to evaluate the same law and derivatives without including the postprocessor, constructing a fault, or duplicating the model selection. Exercise design judgment about names, class layout and ASPECT conventions. The scientific definitions and acceptance checks below are fixed.

## A. Implement the three geometric laws

Add a small, fault-independent **base interface** in the phase-field module for the dimensionless crack-geometric function α. Provide named AT1, AT2 and CZM selections and a single selection mechanism. Share the polynomial implementation where appropriate. The interface supplies α, its first and second derivatives with respect to φ, and its surface normalization c₀. Keep it independent of spatial dimension, FE vectors, mesh, MPI, fault labels, the material model and the postprocessor. Place declarations and definitions in matching phase-field paths. Preserve the existing phase-side mature degradation function and its unit test.

| Selection | α(φ) | α′(φ) | α″(φ) | c₀ = 4 ∫₀¹ √α(u) du |
| --- | --- | --- | --- | --- |
| AT1 | φ | 1 | 0 | 8/3 |
| AT2 | φ² | 2φ | 2 | 2 |
| CZM | 2φ − φ² | 2 − 2φ | −2 | π |

Test public selection, values and derivatives at both endpoints and at interior points, and the three normalization constants using an independent reference. The phase-only laws are finite at φ = 1. The fault-side inverse degradation has its own stricter domain. Inspect the donor's `GeometricFunction` and parameter definitions as mathematical references. The donor's handler also derives m = Gc α′(0)/(c₀ ℓ Hc). At finite positive Hc this sets m = 0 for AT2 and erases slip localization in the mature rational degradation law. Leave this calibration outside 3B.

## B. Use the laws in an ordinary ASPECT calculation

Implement one opt-in statistics postprocessor for the phase-field crack-surface energy. Let the user select AT1, AT2 or CZM, the phase composition (default `phase_field`), a constant fracture energy Gc > 0 in J/m² and length scale ℓ > 0 in metres. Default geometry is AT1. Use conventional plugin parameter declarations and explain explicitly that the selection is for the measured functional. Initialize the selected law once. Keep constitutive definitions out of the postprocessor. Accept existing field storage through ASPECT introspection; actual field and parameter validation applies when the diagnostic is selected. Existing inputs with the diagnostic disabled must continue to work.

For physical cell measure dx and mapped gradient ∇φ, evaluate the two terms and their sum:

    E_local = (Gc / (c₀ ℓ)) ∑_K ∫_K α(φ_h) dx
    E_grad  = (Gc ℓ / c₀)   ∑_K ∫_K |∇φ_h|² dx
    E       = E_local + E_grad

Expose the local, gradient and total contributions in the normal statistics output. In a 2-D Cartesian calculation the result is J/m of out-of-plane thickness. In a 3-D volume it is J. Read ASPECT's current solution at the ordinary postprocessing point, integrate locally owned cells using physical JxW and gradients, and reduce once across ranks. Choose quadrature appropriate to the phase FE degree. The installed postprocessor must run from a normal parameter file with no donor or test plugin.

**DG interpretation.** This is the cell-integrated surface functional with the elementwise (broken) gradient. It coincides with the continuum expression on the continuous fixtures below. On a discontinuous DG field it omits interelement jump and flux terms. Document that fact in the parameter help or user-facing description. A discrete energy consistent with a future SIPG evolution operator must be specified with that operator. Keep 3B read-only and keep ASPECT's original elasticity disabled in the fixtures.

## C. Make the numerical result independently checkable

1. **All three selections on identical data.** Use a 2 m × 1 m rectangle, 0 ≤ x ≤ 2, 0 ≤ y ≤ 1, and φ = x/(4 m) + y/(2 m), represented exactly in a static DGQ2 composition. Use an ordinary material, zero fault objects, Gc = 2 J/m² and ℓ = 0.25 m. Exercise the installed postprocessor three times, changing only the geometry selection and output directory. Keep the underlying field and FE discretization identical. The area integrals of α are respectively 1, 7/12 and 17/12 m². For all three, the integral of |∇φ|² is 5/8. Check both contributions separately against these analytic integrals. The total outputs must match:

   | AT1 (J/m) | AT2 (J/m) | CZM (J/m) |
   | --- | --- | --- |
   | 399/128 = 3.1171875 | 239/96 ≈ 2.48958333333333 | 559/(48π) ≈ 3.70698388284873 |

   Run at least one selection on one and two MPI ranks with nonzero contributions on both. Compare to the same analytic answer. Set the comparison tolerance from floating-point summation and actual statistics precision. Check the feature-disabled path in an ordinary model with no phase composition and ensure invalid selection, missing selected field or invalid physical constants are diagnosed when applicable.

2. **Surface energy of one straight crack.** Use a centered, mesh-aligned AT1 profile with φ = [max(1 − |x|/(2ℓ), 0)]², independent of y over length L_y. Include both flanks in the domain and align x = 0 and ±2ℓ with cell faces. DGQ2 can represent this profile exactly. The local and gradient parts should each give Gc L_y/2 and their sum Gc L_y per unit out-of-plane thickness. This fixture verifies the factor of two across the crack and the meaning of c₀. It is a phase-only surface-energy control; the partially damaged mature-RSF distance profile has its own physics.

3. **Preserve 3A.** Run the existing phase-law unit tests, the scoped stationary regressions and the corrected free 60° AT1 case against `results/3a-repair-free-labels.csv` at initialization, 12.5 ms and 25 ms using the existing tolerances and all 38 IDs. Reuse the established check for accepted bulk stress and the associated controls if a shared path changes. Make no new fault model selection implicit in an existing input. Measuring an alternative surface functional, if explicitly requested, makes no statement about the calibration used in the RSF solve.

At handoff record the active parameters, commands, expected and measured values by term, actual MPI/build coverage, and one source walkthrough from parameter selection through the phase-side law to the statistics table. Report failed checks accurately. Point to the exact places a later phase assembler would reuse. Use four or fewer total local workers and fresh output directories.

**Review boundary.** Completion means independently usable AT1, AT2 and CZM *crack geometry* and a measured phase-only surface functional. Energetic degradation g, fracture driving force, irreversibility, DG transport/evolution, coupled AT2/CZM fault calibration, and changes to global field layout are separate decisions. Preserve the donor and the frozen midpoint update. Stop for review before opening the next increment.

## Source landmarks relative to the target worktree root

```text
include/aspect/phase_field/mature_degradation.h
source/phase_field/mature_degradation.cc
unit_tests/phase_field.cc
source/postprocess/composition_statistics.cc
refactoring/phase-field-fault/STATUS.md
refactoring/phase-field-fault/results/3a-repair-free-labels.csv
../aspect-pf-rsf-particle-free/include/aspect/phase_field.h
../aspect-pf-rsf-particle-free/source/simulator/phase_field.cc
```
