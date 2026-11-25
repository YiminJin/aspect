# Work on the inclined mature ridge

The horizontal axis gives the number of bulk Q2 cells per side of a unit square. The vertical axis gives the absolute error in either action after dividing the shear power per unit out-of-plane length by the Maxwell coefficient, applied rate perturbation and virtual shear rate. The resulting units are metres. The section row uses the driving shear of the virtual velocity. The Stokes curve measures the RHS supplied to the local Q2 velocity equations before constraint elimination.

After repair, the plotted Stokes RHS and section row coincide to the resolution of the figure. Each retains a nonzero quadrature error against the split integral. The Stokes assembler consumes the Q8 slip-stress moments after a local projection into DGQ2.

The central material vertex is at (0.485, 0.47) m, strictly inside a bulk cell on each mesh. The bottom and top contacts are (0.25, 0) and (0.75, 1) m. With signed physical normal distance \(r\) in metres, the prescribed ridge \(\phi=0.8-1.7\,r^2\) is represented exactly in DGQ2. The excess-compliance ridge has a full width at half maximum of 0.394 m, spanning 3.15 cells on the coarsest grid and 12.6 on the finest. The unit virtual shear and the hat perturbation isolate shear transfer.

The reference reconstructs the three coefficients of the prescribed Q1 normalizer using its three station samples per segment and analytic trace mass. It then integrates the remaining bulk action independently in physical tangent and normal coordinates, splitting at the interior vertex, at both contacts and where the box face bounding the ray changes. Increasing both integration orders from 32 to 48 moves the result by less than \(4\times10^{-16}\) m. This tests the transfer with its discrete normalizer held fixed.

From the worktree root, regenerate the figure from the measured tables with

```sh
python3 benchmarks/fixed_fault/transfer_work/refinement.py plot \
    refactoring/phase-field-fault/results/transfer-work/original.csv \
    refactoring/phase-field-fault/results/transfer-work/repaired.csv \
    refactoring/phase-field-fault/results/transfer-work/work-error.png
```
