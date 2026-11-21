# True-residual restart in the fourfold-rate test

**Both fourfold-rate intervals passed.** The restart triggered once and certified the first shifted correction in one additional Krylov iteration. The solver published steps 79 and 80 at 6.32 and 6.40 ms in 86 min 43 s elapsed on four ranks. The original executable spent 115.7 minutes on step 79 and evaluated zero nonlinear trials.

The test restarts the 256 × 128 weakening-law pilot from accepted step 78 at 6.24 ms. Its target is two 80 μs midpoint intervals at four times the steady boundary velocity. The previous attempt exhausted its search at Newton iteration zero. Six shifted mechanics solves passed their Arnoldi estimates but failed the explicit residual test. The steady-load run already reached 6.40 ms and provides the matched physical reference. [Original failed diagnostic and baseline](STAGE4_OVERNIGHT_PROPAGATION.md).

## Solver and comparison

The shifted search first solves the phase obstacle, then solves the restricted mechanics system for the additional correction. Let that second system be \(A c=b\). If FGMRES reports convergence but \(\lVert b-Ac\rVert_2\) exceeds the original tolerance, the solver retains \(c\) as a starting vector, discards the Krylov basis, and starts FGMRES once more. For a nonzero initial vector deal.II forms \(b-Ac\) on entry. Both solves use the same matrix action, preconditioner, load and tolerance. Their combined limit is 600 iterations. A genuinely exhausted first solve returns to fallback immediately. Each returned correction is checked against the original shifted system before it can enter the physical line search. Physical acceptance still requires mechanics ≤ 1e-9 and phase KKT ≤ 1e-7, followed by the existing direction and publication checks. [Shifted mechanics solve](../../../source/simulator/solver/bulk_tip_linear.cc).

All six parent checkpoint hashes, the prepared material inputs, the checker and the protected reference executable were verified before the build. The new executable was linked separately from the same native objects with only unity unit 57 rebuilt. The source baseline is the locally modified worktree at `c07eea8e77e02dd464f5e050e773083963c5ccb2`. The run uses four bound ranks, one thread each, and the original 9,000 s wall limit. The old and new parameter files differ only in output directory. [Original run](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-probe/summary.json).

## First recovered direction

All 12 recorded solver observations through the first failed certificate matched the old run after excluding timing and the new restart index. At shift 1, both binaries returned the same explicit norm of 3.9089201204e-14 after 31 iterations. The Arnoldi estimate of 2.3582316674e-14 was inside the 2.3759657237e-14 tolerance. One additional iteration from the true residual gave an estimate of 2.0395238761e-14 and an explicit norm of 2.0395238619e-14. The full line trial passed. The reassembled mechanics and phase residuals fell from 2.38046e-4 and 0.233743 to 4.52905e-7 and 0.00186370. This recovers the first nonlinear search direction. [Matched source, input and executable identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/comparison.json) · [Shifted solve certificates](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/shifted-certificates.csv).

## Physical acceptance and cost

| Step | Native step time | Newton iterations | Line trials | Final mechanics | Final phase KKT |
|---:|---:|---:|---:|---:|---:|
| 79 | 30.5 min | 14 | 30 | 1.96e-18 | 1.71e-10 |
| 80 | 55.6 min | 17 | 95 | 1.55e-16 | 7.30e-8 |

All 61 shifted phase CG returns passed their checks. The two steps used 16 shifted mechanics directions with 17 linear returns including the single restart. Step 80 needed no further restart. Contact-restricted exact searches took 24.9 and 45.5 minutes respectively. Virgin-plane selection took 0.30 and 0.37 s. Both native checkpoints were written and the launcher exited successfully. [Terminal record, hashes and per-step counts](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/summary.json) · [Native log](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/native.log.gz).

## Matched loading response

Rates and feasible driving below are evaluated at the last midpoint (6.36 ms). Ages and contour positions are evaluated at the accepted endpoint (6.40 ms). Added work and slip use the common 6.24 ms checkpoint.

| Measurement | Steady rate | Fourfold rate |
|---|---:|---:|
| Slip rate at original right tip, mm/s | 1.448 | 3.896 |
| Peak slip rate at x = 3.001 m, mm/s | 19.87 | 25.78 |
| Age at original right tip, ms | 0.2340 | 0.1294 |
| Near-tip / farther-ahead feasible net driving, J/m³ | −0.04044 / −0.18470 | −0.03750 / −0.19419 |
| Added integrated slip, m² | 7.177e-6 | 9.518e-6 |
| Added fracture energy, J/m | 3.606e-5 | 7.682e-5 |
| Signed / accumulated absolute physical-work error, J/m | +4.69e-11 / 4.69e-11 | +4.14e-10 / 4.14e-10 |
| Rightmost seed-connected φ = 0.05, m | 6.004871 | 6.005075 |
| Maximum coefficient in initially intact cells | 0.002379 | 0.002660 |
| New sections | 0 | 12 |

The extra near-tip mechanical driving was 0.01113 J/m³, about one quarter of the 0.0418 displacement-based estimate. The fracture derivative increased by 0.00819 J/m³, leaving a net gain of only 0.00294 J/m³. Both prescribed positive test increments remained energetically unfavorable. Fourfold loading roughly doubled the added fracture energy, but the connected 0.05 contour advanced only 0.395 mm (0.013 cell) from the common parent. At both sampling densities the connected 0.001 contour reached two initially intact cells. The coefficient bound excludes a 0.01 crossing in any initially intact cell. Remote phase increment and worst coefficient decrease were zero. [Matched measurements](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/matched-physical.json) · [Growth, slip, age and driving figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/accepted-response.png).

The 12 new sections registered on step 80, six on each side at x = 1.657–1.686 and 6.314–6.343 m. Their maximum endpoint phase at native quadrature points was 6.11e-12 and their largest midpoint rate was 1.27e-11 m/s. Recorded angles were 13.88–13.91°. Registration requires positive phase, so the count alone overstates physical advance. [Per-section phase, rates and angles](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/new-sections.csv).

At 6.36 ms the +5 mm/s region containing the main slip peak ended at the station x = 5.918 m. A second group of stations at x = 6.043–6.130 m exceeded the threshold, peaking at 7.34 mm/s. The original tip between them slipped at 3.90 mm/s. The rightmost crossing therefore measures a separate region of fast slip. The external rate change, two sampled midpoints and subcell contour displacement leave spontaneous RSF propagation unestablished. **Next experiment.** From the saved fourfold step-79 endpoint at 6.32 ms, compare the last 80 μs interval with two 40 μs intervals at the same rate. Measure the spatial extent and integrated slip of the second group, the phase increments in fresh cells and the fracture-work change. [Front measurements](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/new-fronts.csv) · [Step-79 checkpoint hashes](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/overnight-recovery/rate-true-residual/parent79.json).

The proposed last-interval comparison has since completed. [Final 40 μs result and figure](STAGE4_FINAL_LOCAL_40US.md).

## Reproduction

From the worktree root, choose an unused label and output directory for each invocation.

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py shifted-true-residual-repeat 9090 \
  python3 build-particle-free/overnight-recovery/launch-verified.py rate-true-residual-repeat \
  build-particle-free/guard80/through80/output/restart/01 78 80 .00008 9000 4
```
