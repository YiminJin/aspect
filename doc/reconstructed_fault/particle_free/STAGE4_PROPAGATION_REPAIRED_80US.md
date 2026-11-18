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

## Continuation and selection of the refinement interval

The repaired 80 μs trajectory advanced from 5.60 to **6.08 ms**, with a
native checkpoint at every accepted step. The six physical intervals cost
21.34, 9.45, 15.33, 10.02, 23.95 and 10.12 minutes, respectively. The
continuation exited normally after 5,523.0 s. The odd steps needed more
search work, but each converged within 24 minutes. The final checkpoint is
step 76. The common parent for the local timestep comparison is separately
preserved at step 73.

The first change in the footprint of initially intact cells occurred between
5.84 and 5.92 ms. At 5.92 ms, a cell centered at (6.078125, 1.890625) m
below the fault reached a peak Bernstein coefficient of 2.47e-5. The cell
at (6.109375, 2.140625) m on the upper flank reached 7.89e-7. Both had
zero seed coefficients. They continued evolving through 6.00 ms. At 6.08 ms
another lower-flank cell joined the footprint. These small increments motivated
a comparison across their first appearance, even though the absolute 0.05
contour was still within the original nonzero seed support. The refined run
starts at 5.84 ms, before either new cell evolved, and uses two 40 μs
intervals to reach the same 5.92 ms endpoint.

The rightmost connected contour reached x = 6.004493 m on the upper flank
and 6.002632 m on the lower flank. The upper advance since 5.44 ms was
0.738 mm, or 0.024 of the 31.25 mm cell. At 6.08 ms the largest coefficient
in initially intact material was 0.001893. No new material section formed.
The maximum section slip rate stayed at x = 3.0011 m. It reached 19.63 mm/s,
while the station nearest the right tip reached 1.34 mm/s. There was some
spreading of sliding: the rightmost station above 1 mm/s reached x = 6.1444 m.
The 5 and 10 mm/s thresholds still ended well behind the tip at 5.8006 and
5.5885 m. The observations support sustained loading and a widening
low-rate footprint. A traveling acceleration arriving at the tip or a
separate rupture episode was not resolved.

## Matched timestep check at the new footprint

Both branches started from the same six-file step-73 checkpoint. On the
40 μs branch both newly active cells still had all-zero coefficients at
5.88 ms. At 5.92 ms, both branches had positive damage in the same four
initially intact cells. The new lower cell's peak coefficient was 2.47e-5
at 80 μs and 5.44e-5 at 40 μs. In the new upper cell the respective peaks
were 7.89e-7 and 4.21e-5. Halving the timestep confirmed which cells
activated in the interval while leaving the magnitude of the smallest
increments poorly determined.

| Change from 5.84 to 5.92 ms | One 80 μs interval | Two 40 μs intervals |
|---|---:|---:|
| Connected right φ = 0.05 advance (μm) | 42.30 | 78.74 |
| Spatially integrated slip increment (m²) | 3.24859e-6 | 3.24569e-6 |
| Added fracture work (J/m) | 9.04708e-6 | 9.22595e-6 |
| Physical-work error, signed and accumulated absolute (J/m) | +1.75e-11 | +5.43e-12 |
| Endpoint projected phase KKT | 5.63e-9 | 1.44e-8 |
| Newly formed sections | 0 | 0 |

The two trajectories differed by 0.089% in integrated slip increment, 0.51%
in section-weighted RMS mean rate, 1.98% in fracture work and 12.73% in the
phase increment's spatial L² norm, each relative to the coarse value. The
absolute contour difference was 36.44 μm, comparable with the earlier 35.7 μm
difference at 5.52 ms. For this particular interval it is 86% of the small
coarse advance. With either timestep the 0.001-level damage sampled in
initially intact material was connected to the seed. Neither run reached
0.01 there. All accepted coefficient increments were nonnegative.
[Matched comparison and cell values](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard80/focused-5.92ms.json).

## Physical work, cost and next experiment

Across the eight 80 μs intervals from 5.44 to 6.08 ms, fracture energy
increased by 6.733e-5 J/m. The signed physical-work error and sum of absolute
interval errors were both 1.059e-10 J/m. At the endpoint the upper-right
initially intact phase integral was 5.472e-7 m² out of 5.659e-7 m² in all
initially intact cells. The remote region had zero phase increment. Both
8- and 16-point-per-axis component searches gave the same reported contour
extents. The upper flank remained farther right in absolute position. The
lower flank advanced 0.848 mm relative to its own position at 5.44 ms.
[Per-step observables](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard80/window-history.csv) · [Initially intact cells](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard80/initially-intact-cells.csv).

The replay, continuation and local refinement took 43.26, 92.05 and
30.46 minutes, respectively, including restore and checkpoint I/O. All
three processes exited successfully. The total was **2 h 45 min 46 s**.
The next odd 80 μs interval would have had 14.23 minutes available. Each
of the three preceding odd continuation steps cost 15–24 minutes, so we
closed at step 76. The refined endpoint is step 75 on its own branch.
[Checkpoint paths and SHA-256 hashes](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard80/accepted-checkpoints.json).

**Next experiment.** Resume the 80 μs branch at step 76 for the four intervals
to 6.40 ms. Track whether the connected φ = 0.01 level first enters initially
intact material and whether the 5 mm/s slip-rate front approaches the tip.
The two matched timestep checks support using 80 μs for integrated slip
and exploratory work accounting. A claim about the timing of rapid tip motion
needs a local comparison around such a transition. No acceleration pulse
reached the tip, new section formed or resolved arrest occurred within this
window.

[Slip rate, connected contour and flank damage (PNG)](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard80/slip-and-upper-flank.png) · [PDF figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_propagation_pilot/guard80/slip-and-upper-flank.pdf).
