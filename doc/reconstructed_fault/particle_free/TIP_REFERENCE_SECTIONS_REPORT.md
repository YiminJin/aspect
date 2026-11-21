# Fixed material sections on aligned and oblique meshes

**The native single-fault route now accepts an arbitrary fixed incoming direction.** The horizontal control reproduced the protected executable at two accepted endpoints. A 90° rotation and a 30° rotation of the same discrete specimen preserved section membership, slip, tensor histories and the selected relative planes to the precision below. On an unchanged Cartesian mesh the 30° case passed coupled acceptance, the work and tangent checks, forced rollback, and a native checkpoint replay. [Formulation and code guide](tip_treatment/reference_sections.md) · [Inputs, commands and reducers](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/README.md).

## Baseline and changes

The entry worktree was `pf-rsf-particle-free` at `c407e612bf7c9aa1a9c0fc60fd623208dfd53b39` with a clean tracked tree and existing untracked work. Before editing, 28 path identities covering the original two inputs, executable, checker, parent and terminal checkpoint members were verified against the 80/40 μs run records. The tested solver source SHA-256 was `9c1f726d994595a436711ee46f04faa764ab53e6c6b9569327465b0ed56b1497`. The tested executable was `5e6bdb65646c94610e2605675773d7701d87ea9819758e355be8ae8d95fa4014`. The retained difference from `c07eea8e7` and the verification manifest are in `build-particle-free/reference-sections/`. The new full Release build used a separate build directory. [Original source and runtime comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/comparison.json) · [Retained long-run comparison](STAGE4_FINAL_LOCAL_40US.md).

| Dependency traced | Current implementation |
|---|---|
| Cell x-Gauss index, x weight and common specimen height | Project vertices into the fixed incoming frame, intersect each station with cell edges, and integrate actual chords. The original horizontal mesh retains its ordinary Gauss order and weights. |
| Global xy shear, two direction-search moments, angular sector | Construct the physical shear tensor from the fixed tangent and relative turn. Rotate the stress moments into that basis for both virgin-plane criteria. |
| Bulk, phase and history samples | The material owner supplies the same cell rule to point updates, section reductions, tangent, search diagnostics and observers. The separate DG fracture operator retains its tested quadrature. |
| Stored direction, work and restart | Record relative plane angle and arclength measure. Cell-local direct stress and recoverable strain remain global tensors. The cut archive identifies the frame and chord layout. |
| Benchmark outputs | Cut-section tables use `xi`, `reference_dxi` and `beta`. Native point positions and stress components are global. The dated horizontal tables keep their column names. Prepared-slip inputs, forward probes and remote masks defined for the horizontal specimen retain that restriction. |

## Bounded tests

All coupled tests used 32 affine cells (8 × 4), 6-point line Gauss, two MPI ranks and two 5 ms midpoint intervals. Both material laws for prospective direction retain their original admission conditions. The covariance fixture adds a trace-free normal strain rate of 0.003/s to specimen-frame shear of 0.02/s, so the direction comparison exercises a nonzero turn (down to −0.1508 rad). Its zero-stress initial state develops nonzero tensor and sliding histories. Maximum absolute discrepancies at 10 ms follow. The reference tensor was compared to the rotated tensor brought back into its basis.

| Quantity | 90° rotation | 30° rotation |
|---|---:|---:|
| Physical point correspondence (m) | 1.57e−16 | 4.58e−16 |
| Deviatoric stress, in-plane components (Pa) | 1.04e−12 | 2.14e−10 |
| Recoverable cohesive strain | 2.98e−19 | 2.11e−16 |
| Phase | 6.43e−17 | 6.35e−17 |
| Section slip rate (m/s) | 1.56e−17 | 1.27e−16 |
| Accumulated section slip (m) | 8.47e−20 | 6.27e−19 |
| Resistance (Pa) | 4.26e−14 | 2.56e−13 |
| Relative plane angle (rad) | 9.80e−9 | 7.70e−9 |
| Formed-state mismatches | 0 | 0 |

A preliminary analytic re-evaluation of the compact seed at rotated interpolation points produced tiny positive values at nominal support zeros. At 30° twelve additional stations registered and the local reaction differed by 20.9 Pa, although the maximum phase discrepancy was only 2.68e−16. The matched test uses the same cellwise initial scalar coefficients under rigid rotation. This isolates covariance and preserves the model's exact-positive registration rule. Analytically resampling a seed on another mesh remains sensitive at that threshold.

**Unchanged Cartesian mesh.** With the sections inclined at 30°, 264 stations and 8,640 native points covered the 2 m² box. Direct intersection of each complete line with the four specimen half spaces gave a maximum chord-length error of 8.88e−16 m. The maximum error for the integral of \(1+\xi/5+\zeta^2\) was 1.33e−15. A transverse Gaussian of width 0.09 m gave a maximum integral error of 1.37e−5 m and a relative weighted L² error of 3.38e−5. These comparisons use an independent specimen-wide antiderivative. No physical field interpolation between meshes enters them.

At 10 ms the cut intact-driving case had mechanical residual 4.35e−11 and phase KKT 3.15e−8 against gates of 1e−9 and 1e−7. The unshifted tangent action differed from a centered finite difference by 5.93e−9 relative at the first step. The reassembled station rate matched the stored rate within 2.61e−18 m/s. The maximum stage-integrated friction-work adjoint defect was 3.91e−18 J/m. The full two-step physical-work remainder was +2.37e−6 J/m, which includes the existing midpoint compliance remainder. These two work measures answer different questions. The second endpoint had integrated slip 2.638e−5 m². There were 216 stations with positive seed support, 240 registered by 5 ms and 24 still prospective at 10 ms. The latter had aged from 10 to 10.01 s.

**State lifecycle.** The checker discarded two converged trials before the second physical step and verified unchanged accepted histories, field restoration and a deterministic retry. Restarting from 5 ms reproduced the uninterrupted endpoint's exported phase, stresses, strain, section rates, ages, slip, resistance, formed flags and angles exactly at printed precision. The cut Coulomb selector also completed both steps and its restart, with relative virgin angles down to −0.2702 rad. A fresh horizontal run and a new-executable restart from the protected executable's small 5 ms checkpoint reproduced the horizontal endpoint with zero measured field differences. Changing the declared angle from +30° to −30° rejected the cut checkpoint at the material identity check.

## Cold-creep direction score

The first nonaligned intact-driving attempt correctly stopped publication at the sector guard. The search had computed a spurious boundary maximum at station 20. Subtracting integrated terms of order stress squared had lost the much smaller squared stress difference. A pointwise scan reconstructed from the first fixed-plane solution gives a maximum of 5.55e−20 at 0°, versus 6.30e−26 at either sector edge. The solver now retains its expanded score when well conditioned. Under cancellation it evaluates the same sum as weighted traction variance plus the square of mean traction minus resistance. Rank-local centered moments are combined about the common mean. The search interval, tie rule, physical criterion and tolerances are unchanged. The original rejected attempt and the successful replay are retained separately. [Independent score scan](../../../benchmarks/reconstructed_fault/particle_free/reference_sections/cold_section_score.py).

## Cost and scope for server use

The small cut test used 7.5 times as many material points as the aligned control. Its native steps took 1.44 and 3.67 s including the enabled diagnostics and recovery exercise. Exact splitting at every distinct projected vertex can be expensive for incommensurate directions. Counting projected intervals for an 8 × 4 m box at 256 × 128 cells and the same quadrature gives 1,179,648 points at 0° or 90°, 2,359,296 at 45°, and **217,314,432 at 30°** on that unchanged Cartesian mesh. These are geometry counts, not measured runtimes. For a large single-fault experiment, aligning the mesh with the fixed incoming frame keeps the transverse sections exact at tensor-product cost in point count. A large generic oblique Cartesian run needs a separate storage and checkpoint budget.

The verification supports the fixed-frame geometry and short coupled lifecycle. A propagation trajectory, long-time spatial accuracy, material transport, evolving section orientation and branching remain outside it. The preserved native executable for the final physical checks is `build-particle-free/reference-sections/aspect-centered` (SHA-256 `1057c6092dc774f8cbc286e600f62e94921d856191c834433c5b6fefffe116ea`). The matched-seed checker is preserved as `checks-matched.so` alongside it. The final rebuilt checker also reproduced the cut restart. Each `build-particle-free/bulk-tip/runs/reference-frame-*/command.json` records its exact input, executable, checker and invocation. The compact comparisons are in `build-particle-free/reference-sections/verified.json`.
