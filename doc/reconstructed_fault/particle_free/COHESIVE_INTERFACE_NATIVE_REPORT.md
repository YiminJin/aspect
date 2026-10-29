# Native fixed-Γ components: implemented and boundedly qualified

**The native FE/Γ component and checkpoint experiment passes after correcting
surface integration. The finite-width representation is not yet an accurate
realization of the selected sharp-interface model.** No propagation run or
production/BP3 change was made.

[Discrete equations and ownership](COHESIVE_INTERFACE_NATIVE_FORMULATION.md) ·
[selection and recorded amendments](COHESIVE_INTERFACE_NATIVE_SELECTION.md) ·
[figures/data index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_native/INDEX.md).

## What was implemented

A new native C++ bond/contact return implements the previously accepted law,
including exact intact admission, irreversible surface damage, implicit RSF
aging, current normal-pressure derivatives, signed slip and zero mature bond
storage. The opt-in fixture uses ASPECT's actual Q2/Q1 mesh, mappings and complete
AffineConstraints, both mixed mechanical components and solved pressure, native
fault associations/property containers, and a **real ASPECT checkpoint/restart**.
Physical stress is reconstructed from native FE displacement memory minus the
represented slip source, including the full plane-strain zz component.

This is **component integration in an isolated native fixture**, not a new
production material selector or a call through the production Stokes/phase
solver. The host runs `no Advection, no Stokes`; the fixture assembles and solves
the experimental mixed mechanical problem on its actual FE mesh. Its manager
uses the native container/serialization implementation through postprocessor
checkpointing. The simulator-owned manager's old bulk-phase/particle startup
guards remain intact. No existing native checkpoint is converted.

Bulk and bond compliance are separate. Bulk storage uses
`E=dev3(symgrad d_R−χ z S)`; bond storage uses the actual geometric-slip variable
and its damage-created compliance. Surface fracture is charged once as Gc a.
A fixed normalized hat kernel represents the geometric slip. At finite width
this gives a represented jump amplitude, not a literal discontinuity of the
continuous native displacement; nonuniform equivalence must be measured.

## Corrections established by the tests

The first native version used nodal constitutive quadrature. At t=.376 s its
mechanically relaxed tangent had an **alternating mode with curvature
−1.1745e5 Pa/m**, although force and energy ledgers passed. The failed source and
results are retained. The final implementation uses consistent **surface Gauss3**
for bond energy, fracture, friction, damage and age, with history owned at each
surface integration point. Minimum formation curvature is now **+1.6320e6 Pa/m**.
The actual condensed nonsymmetric tangent is checked before publication; tied
rows are excluded. This is local incremental monotonicity, not the old symmetric
phase-potential certificate or a global stability theorem.

Two arithmetic/testing corrections retain the original tolerances. Direct energy
integration replaces cancellation in `D−BᵀA⁻¹B`, with the exact represented uniform
mode separated and fixed-factor iterative refinement. The largest response
correction is about 4e−14 m/m. Also, finite differences at near-zero RSF speed
must resolve the asinh transition: the original fixed increment was orders of
magnitude too large. Refined scale-aware checks converge to the unchanged target.
Neither correction changes physics or adds damping.

## Measured verification

| Check | Result |
|---|---|
| Native intact onset | [0.374,0.376] / [0.375,0.376] s at Δt=.002/.001 s; analytic .375704888 s |
| Homogeneous native vs independent scalar reference | maximum physical traction difference 4.55e−12 Pa for formation, 2.28e−12 Pa for mature sliding |
| Independent complete Gauss return | maximum traction difference 6.80e−11 Pa; damage difference ≤1.84e−15 |
| Mature velocity step, hold, reversal | 7.25966 Dc absolute slip; exactly zero bond storage and new fracture |
| Nonuniform developed interface | solved pressure range reaches 187.61 Pa; complete loaded tangent FD error 7.85e−9 relative |
| Near-zero tangent | FD error 7.56e−8 → 1.89e−8 → 4.73e−9 |
| Mixed FE residual | <5e−13 relative across the selected mesh controls |
| Independent reconstructed-stress work | shear discrepancy <5.10e−10 Pa; normal-pressure discrepancy <6.01e−11 Pa |
| Independent volume energy | discrepancy <1.80e−15 J/m; full tensor trace error <8.68e−12 Pa |
| Bulk Gauss3 vs Gauss6 | relative K difference 1.85e−15, N difference 5.57e−14 |
| True force / BE ledger | force <1e−8 Pa; ledger error <2.04e−13 J/m |
| Rejection / retry / publication | bad Gauss state rejected before mutation; deterministic retry; stale/duplicate candidate rejected |
| Actual simulator restart | all Gauss histories, nodal histories, signed rates, reconstructed surface tractions and publication sequence **bitwise identical** to uninterrupted control |
| Changed-width restart | explicitly rejected before publication |

The mature run has 601 accepted rows because the native time controller takes
a final roundoff-sized interval at 6 s. The reference uses the exact same
accepted timestep sequence; no row or history is silently dropped.

Physical-energy errors are separate from BE closure. For formation through .42 s,
the absolute sum of stepwise trapezoidal-work remainders is 1.9239e−4 / 9.6006e−5
J/m under halving: **0.199% / 0.0992% of newly added fracture energy**. Relative
to the much smaller sliding loss these are **4.48% / 2.28%**; this short onset
interval does not establish sub-1% accuracy for that small energy component.
For mature sliding the corresponding error is **0.08446% of sliding loss**;
for the nonuniform developed control it is **0.04660%**. Maxwell, friction,
radiation, bulk BE and signed bond endpoint remainders remain separate in CSV.
No physical-energy claim is inferred solely from algebraic closure.

## FE error versus finite-width representation

The 1 m along-fault mode is compared with the independent exact mixed-layer
solution, **projected into the same periodic Q1 surface space**, including
Fourier aliases. Thus Q1 projection is not mislabeled as a constitutive effect.
The two layers have heights .075/.05 m, so normal pressure coupling is nonzero.
The fixed width w=.025 m and halved width .0125 m are numerical representation
controls; G, eta, Gc, strength, state, radiation and loading are unchanged.

| Surface segments / bulk cells | K mode (MPa/m), w=.025 m | Sharp, same Q1 (MPa/m) | Deficit |
|---|---:|---:|---:|
| 4 / 4×40 | 6.57154 | 7.34678 | 10.55% |
| 8 / 8×80 | 9.02297 | 9.76958 | 7.64% |
| 16 / 16×160 | 9.74466 | 10.52837 | 7.44% |
| 32 / 32×160 | 9.93394 | 10.73034 | **7.42%** |

The final 16→32 changes are **1.91% in shear stiffness and 1.91% in the normal
symbol**, meeting the selected 2% resolved-mode FE comparison. The initial
coarser comparisons fail that target and remain visible. At 32 segments, width
halving reduces the shear deficit to **3.99%** and the normal-symbol difference
from **0.498% to 0.124%**. Quadrature error is negligible at this scale. This is
therefore a finite-width representation effect; retaining the same local law
and getting uniform sliding right does not establish sharp nonuniform mechanics.
No convergence order or complete interface accuracy is claimed.

## Ownership boundaries and remaining integration work

The native manager's existing nonlinear slip-rate API requires V≥0. Feeding it
|V| would corrupt reversal. The fixture instead checkpoints a separately
registered **signed** geometric-rate property with the new histories; the old
API and BP3 convention are unchanged. A production driver needs an explicit
signed-rate contract before reusing that solver.

The native snapshot in the split occurs at accepted step 74, t=.074 s. The
subsequent uncheckpointed step-75 output is preserved as its own branch; resume
writes separate files and the comparison joins only the checkpoint prefix with
the resumed branch. This prevents apparent double publication from appended
replay output. Derived FE/geometry caches are rebuilt on restart; the accepted
manager, time and relaxed bulk-memory coefficients are restored unchanged.

The bulk-memory parameterization is exact for this fixed geometry, homogeneous
linear bulk and prescribed load family. General DG-history ownership, arbitrary
old cross-band ages, heterogeneous coefficients, moving geometry, MPI, AMR,
nonuniform intact-front admission and integration into the production outer
mechanical cutback loop are **not** qualified. There are no particle reads in
this fixture. Frozen field maps are independent nonuniform history probes,
clearly labeled rather than presented as a new evolving specimen.

## Recommendation

Keep this component as the verified native point/Γ/owner reference. Before native
tip growth, implement one small **exact-jump fixed-surface realization** (two
traces or equivalent energy-consistent enrichment), including its divergence,
pressure-work and actual normal-trace terms, and the signed-rate contract.
Compare it with the same sharp mixed-layer reference. The required change is
kinematic/discrete, not a new fracture law or friction calibration. Do not accept
the present finite-width error as an unmeasured physical interface property.
The [formulation](COHESIVE_INTERFACE_NATIVE_FORMULATION.md) gives a concrete
aligned enrichment and the terms that cannot be omitted.

## Files and reproduction

New core files: `include/aspect/reconstructed_fault/cohesive_interface.h` and
`source/reconstructed_fault/cohesive_interface.cc`. New fixture/input:
`benchmarks/reconstructed_fault/particle_free/cohesive_interface_native.cc/.prm`.
The existing benchmark CMake file has one appended opt-in target. New run,
analysis and plot scripts share the `cohesive_interface_native` name. Status,
source audit and this report/formulation/selection document the experimental
boundary. Existing defaults, native solver sources, BP3 and earlier evidence
are unchanged; exact protected/source/binary hashes are in provenance.

Build and run commands (from the worktree root):

```sh
cmake -S benchmarks/reconstructed_fault/particle_free -B build-particle-free/cohesive-interface-native/plugin -DAspect_DIR="$PWD/build-particle-free/native"
cmake --build build-particle-free/cohesive-interface-native/plugin --target cohesive_interface_native -j2
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/run_cohesive_interface_native.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/analyze_cohesive_interface_native.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/plot_cohesive_interface_native.py
```

The driver pins the retained current-revision host at
`build-particle-free/coulomb-native/aspect-release`; the older `native/aspect-release`
lacks the geometry transaction symbols and is not used for results. The new
constitutive component is compiled into the isolated plugin, so no protected
host executable is overwritten. Runs are sequential, one process/thread, with
a 180 s timeout each. Local wall times are approximately 2–4 s for each short
trajectory/restart, 4–5 s for 16×160 operator probes and 13–16 s for 32×160 probes.
The largest probe has 45,983 independent mixed masters; setup dominates. Exact
per-run/setup/step timings are in CSV/JSON. These are not production performance
predictions. Full checkpoints/build logs and failed development source remain
under `build-particle-free/cohesive-interface-native/`; compact evidence is indexed.
