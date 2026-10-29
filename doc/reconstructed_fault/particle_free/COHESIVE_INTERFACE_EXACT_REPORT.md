# Exact-jump fixed-surface test: completed within its scope

**The native exact-jump realization removes the previous finite-width bias and
passes the bounded work, tangent, history and restart checks.** This qualifies
the isolated fixed-surface components, not a production Stokes/phase path or
propagation. Production/BP3, physical parameters and previous evidence are intact.

[Equations and ownership](COHESIVE_INTERFACE_EXACT_FORMULATION.md) ·
[predeclared scope/criteria](COHESIVE_INTERFACE_EXACT_SELECTION.md) ·
[figures/CSV index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_exact/INDEX.md).

## What changed

The existing opt-in native fixture now offers `CI_EXACT_JUMP=1`. It adds
`(H−Fχ) δ_h e_x` to continuous displacement, including the **along-fault strain,
divergence and pressure-work terms**. Interface pressure supports have separate
upper/lower traces with the native periodic constraints expanded on each side.
The contact pressure is the consistent Q1 projection of the average of physical
normal-traction traces. Each trace is exported and their mismatch is measured.
No band average, pressure lag, penalty, damping or strength recalibration is used.

This is a kinematic/discrete correction. The bond/contact law, implicit RSF age,
surface Gauss3 histories, Maxwell memory, pressure scaling and acceptance checks
remain fixed. Full plane-strain stress includes zz. Signed geometric rate remains
Δδ/Δt in the fixed surface orientation, separate from absolute accumulated slip
and the unchanged legacy nonnegative-rate API. Exact/source-only mode is now an
experimental checkpoint invariant. No old production checkpoint is converted.

## Sharp reference and width invariance

The independent reference solves both mixed incompressible layers, then projects
the response into the **same periodic Q1 surface space**, including Fourier aliases.
The table is for the 1 m along-fault mode, G=1e6 Pa, heights .075/.05 m, p0=10000 Pa.
Both displacement components and current solved pressure respond.

| Bulk mesh / surface segments | Shear stiffness error | Normal-coupling error | Two-side normal mismatch / sharp shear symbol |
|---|---:|---:|---:|
| 4×40 / 4 | 3.323% | 5.245% | 0.00408% |
| 8×80 / 8 | 0.1748% | 0.1322% | 0.000961% |
| 16×160 / 16 | 0.01346% | 0.002581% | 0.000235% |
| 32×160 / 32 | **0.001535%** | **0.00002027%** | **0.000233%** |

The coarse failure of the 2% reference criterion is retained. The final shear
stiffness is 10.7301718 MPa/m versus 10.7303365 MPa/m sharp. The source-only
control was 7.42% low. On the same 32×160 grid, changing lift half-width from
.025 to .0125 m changes the exact operator by **8.88e−16 relative in K** and
**1.09e−13 in N**. This is a change of coordinates within the FE space, as derived,
not a change of material length. Bulk Gauss3/Gauss6 differences are **1.70e−15 /
5.28e−14**. Width and quadrature are no longer the limiting effects.

The residual two-side mismatch plateaus when normal resolution is held fixed;
it is not erased or forced to roundoff. Shorter wavelengths remain less resolved:
on the finest selected mesh, shear/normal errors are 0.0172%/0.00520% at .5 m,
0.199%/0.198% at .25 m, and **4.81%/1.50% at .125 m**. These diagnostic modes do
not justify a convergence-order claim or quantitative sharp-front accuracy.
The full-matrix trace discrepancy includes these short modes and does not
monotonically decrease merely by increasing the number of surface unknowns.

## Work, derivatives and accepted histories

| Verification | Measured result |
|---|---|
| Two-trace displacement jump | error ≤4.45e−16 in the integrated unit-jump operator; zero normal jump |
| Mixed native residual | ≤1.97e−13 relative |
| Independent generalized stress work, including pressure term | discrepancy ≤5.05e−10 Pa |
| Independently integrated volume energy | discrepancy ≤1.57e−15 J/m |
| Full tensor trace | ≤9.44e−12 Pa |
| Surface Gauss3/Gauss6 trace integration | discrepancy ≤1.18e−7 Pa/m (operator scale ~1e7 Pa/m) |
| Native nonlinear force / exact BE ledger | <1e−8 Pa / ≤2.04e−13 J/m |
| Complete pressure/state/damage FD tangent | near-zero error 7.56e−8 → 1.89e−8 → 4.72e−9; loaded 1.23e−7 → 3.07e−8 → 7.68e−9 |
| Native uniform versus independent scalar | traction difference ≤4.55e−12 Pa |
| Independent return using native K/N | maximum traction difference 6.18e−11 Pa; damage difference ≤1.80e−15 |
| Minimum tested relaxed incremental curvature | +1.933e6 Pa/m |
| Rejection/retry/duplicate publication | accepted owner unchanged; retry bitwise; duplicate candidate rejected |
| Real ASPECT checkpoint/restart | all persistent Gauss/nodal histories, signed rates, tractions, energy rows and publication counts bitwise identical; timing column excluded |
| Restart into source-only mode | rejected before any accepted output changes |

Uniform intact onset remains bracketed by [.374,.376] and [.375,.376] s at
Δt=.002/.001 s, enclosing .375704888 s analytically. Mature sliding reaches
**7.25966 Dc** through a velocity change, hold and reversal, with exactly zero
remaining bond storage and zero new fracture. The nonuniform developed control
reaches a 176.99 Pa current normal-pressure range, versus the retained source-only
187.61 Pa. This is a measured representation difference, not new friction physics.
Both projected side pressures remain above **9911.5 Pa** during this control.
The frozen physical-field probe separately checks positive pressure at all
bulk Gauss6 points. Raw unprojected trace positivity over arbitrary load families
is not established by these runs.

Physical energy is separate from the BE ledger. The absolute sum of trapezoidal
work remainders is 0.08446% of sliding loss for mature sliding and 0.04657% for
the nonuniform control. For initiation it halves from 1.9239e−4 to 9.6006e−5 J/m:
0.199%/0.0992% of new fracture energy, but **4.48%/2.28% of the small sliding loss**.
That unresolved small-component accuracy is retained. The final roundoff-sized
mature step is retained, with identical reference timestep sequence.

The first implementation failed its invariant-mode guard because a negative
surface derivative was formed from an unsigned integer. This was corrected to
signed floating-point arithmetic before publication; failed source/log are saved.
No tolerance was relaxed. Previous nodal-integration failures and the 7.42%
source-only discrepancy remain in their original evidence directories.

## Scope and next task

Actual native mesh/FE/AffineConstraints, mixed assembly, manager/property owner
and ASPECT postprocessor restart are exercised. The fixture still runs separately
from the production Stokes/phase driver. Its reduced tensor-memory representation
is exact for this fixed homogeneous geometry/load family, not arbitrary old DG
history. No particles are read. The positive symmetric part of the actual
condensed tangent is a tested local incremental-monotonicity check, not a global
stability theorem for nonsymmetric evolution.

**Recommendation:** one bounded nonuniform intact-to-damaged front on this fixed
surface, with explicit weak multiplier/admission conditions, consistent Gauss
history ownership and loading/hold checks, before a native seeded tip-growth run.
Uniform admission and a developed nonuniform interface do not yet qualify a
partially tied front. Use the present wavelength diagnostics to resolve that
front; do not infer its accuracy from the 1 m mode alone. No new direction/birth
law, finite physical endpoint, moving geometry, MPI or propagation was started.

## Changed files and reproduction

Modified `benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.cc`
only in the existing optional fixture; the C++ bond/contact implementation and
all production source/header files are unchanged. Added `run_`, `analyze_`, and
`plot_cohesive_interface_exact.py`; this report, selection, formulation and
compact evidence. Updated STATUS, SOURCE_AUDIT and CURRENT_STATUS by prepending
new results. The existing CMake target is reused; no build selector is added.

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-exact/plugin -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/cohesive-interface-exact/plugin --target cohesive_interface_native -j2
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_exact.py operators
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_exact.py histories
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_exact.py baseline
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_exact.py reject-mode-change
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_exact.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_exact.py
```

All native runs are serial and sequential, limited to 180 s each (30 s negative
restart); measured wall times are 1.8–3.9 s per short trajectory/control, 4.7 s
for 16×160 and 14.7–16.8 s for 32×160. Largest setup: 46,015 mixed masters and
5120 cells, 12.8 s. No full event/server run. Exact input files, binaries/source
hashes and timings are indexed. The retained host is
`build-particle-free/coulomb-native/aspect-release`; only the new fixture DSO was
built. The default source-only operator agrees with its archived result within
1.35e−16 (K) / 8.79e−16 (N) relative. Minor arithmetic rearrangement is not
claimed bitwise identical. Prior evidence and production files pass hash checks.
