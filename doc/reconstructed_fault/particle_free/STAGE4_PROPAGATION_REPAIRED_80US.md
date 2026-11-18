# Repaired 80 μs propagation pilot

The two-interval replay starts at accepted step 68 (5.44 ms) on the 256 × 128
mesh. Its native parent is slot 03 of `stage4-propagation-pilot-resume51`.
All six checkpoint member hashes match the common parent of the previous
40 μs check. The retained original 80 μs reference consists of steps 69 and
70 at 5.52 and 5.60 ms. Each comparison uses the accepted endpoint for phase,
age and cumulative slip, and the identical midpoint time for slip rate and
resistance. The prepared seed, loading, boundary velocity, dissipation laws,
mesh, quadrature, time integrator and convergence criteria are common to both
branches.

The repaired executable is `guard68/aspect-contact`, SHA-256
`911f39bdf5b0425b9df9481dd91035a9760a9ae9ca92fa0919865b924fdd3420`.
The native checker is `8b582573e7b199d4f0fd31ab0ec2eb91e418a3eed52fd3d5a013a42cffb46796`.
The worktree was on `pf-rsf-particle-free`, HEAD `b32a2d4f6` with local
source changes. The original control and repaired executable identities are
recorded independently of the source revision. Each launch uses four MPI
ranks, one thread per rank, a separate output tree and a checkpoint after
every accepted interval.

The replay comparison evaluates the complete DG Bernstein phase coefficient
field, its spatial L² norm, stationwise age, slip, rate, resistance, formed flags
and orientations, boundary reaction forces and energy components. Its first
screen asks whether solver-induced differences in phase L², right contour
position, integrated slip and fracture energy are all less than one tenth
of the measured 80 versus 40 μs differences at that physical time. This
threshold selects whether a trajectory discrepancy deserves investigation.
The simulator's own physical acceptance gates remain mechanics ≤ 1e-9
and projected phase KKT ≤ 1e-7. The preceding [40 μs check](STAGE4_PROPAGATION_LINEAR_RESIDUAL.md)
sets the scale of the temporal uncertainty. A 20 μs replay is deferred.

Connected advance uses the absolute φ = 0.05 contour attached to the seed.
Both DG traces participate in identifying the component. Within each half of
the specimen, the rightmost intersection is recovered from the cell
polynomial. The upper and lower searches are independent, so the upper flank
can move into a new column while the lower flank stays behind. Initially
intact cells have all nine seed coefficients exactly zero. Their phase
integrals use exact Bernstein cell means. Slip rates are plotted at material
stations and midpoint times. Accepted contour and fracture work are plotted
at endpoints. The upper-flank map shows the phase increment since step 68
and the locations of initially intact cells that developed damage.

## Same-timestep replay

| Endpoint (ms) | Max phase coefficient difference | Right contour difference (m) | Integrated slip difference (m²) | Fracture energy difference (J/m) | Original → repaired time (min) | Original → repaired Krylov iterations |
|---|---:|---:|---:|---:|---:|---:|
| 5.52 | 2.74e-11 | 2.31e-14 | 2.42e-16 | 1.69e-13 | 110.09 → 32.08 | 7,476 → 2,194 |
| 5.60 | 1.87e-11 | 7.73e-14 | 1.90e-16 | 1.22e-14 | 10.41 → 10.50 | 668 → 668 |

The repair cut the hard first interval's cost by 71%. The active-set release
counter fell from 336,675 to 2,393 and tangent assemblies from 23 to 17.
The second interval had 272 releases with each solver. The replay took
2,595.6 s including restore and checkpoints. At both endpoints the formed
station sets matched and the largest station slip difference was below
3.4e-15 m. The evidence includes native boundary reactions, phase and
energies. Full pointwise stress was unavailable for this historical comparison.
[Full same-timestep measurements](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard80/same-dt.json).

**One local quantity is sensitive even at the same timestep.** Section
resistance differed by more than 1 Pa at five stations at 5.52 ms and six
at 5.60 ms, all in the column from x = 6.1886 to 6.2177 m. The largest
discrepancies were 12.357 and 6.589 Pa. The largest speed on any flagged
row was 2.06e-12 m/s. At station 1190, the two first-interval midpoint
speeds were 8.17e-13 and 1.33e-14 m/s. Inverting the regularized RSF law
using each saved parent age, the actual 40 μs half-interval and each saved
midpoint speed reproduces both forces to 2e-13 Pa. This logarithmic
sensitivity at very small absolute speeds limits local force reproducibility.
The affected column's maximum phase difference was 3.88e-12 and its flagged
stations contributed at most 1.29e-16 J/m against 3.85e-4 J/m total section
work in the first interval. Above 1 mm/s the maximum discrepancy was
1.45e-7 Pa. On this evidence the accepted propagation observables pass the
replay check. [Station diagnostic](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard80/resistance.json).
