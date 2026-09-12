# K3 frozen homogeneous phase residual: periodic-domain diagnosis

## Decision

The fault32 I_h representation/normalization result remains accepted. The
remaining seam failure has a phase/domain source: exactly homogeneous nodal
phi(y) and particle H(y) produce a strongly nonuniform **production phase
residual on the saved advected domains**, before invoking any surface equation.
The initial domains give a tangentially uniform residual. This is not another
I_h projection problem or a recurrence of the old constant-reproduction bug.

Stop for review of a periodic-image-aware particle-domain/CPDI correction.
None is implemented here. Open surface endpoints may still matter in a coupled
replay, but they are not needed to produce this measured residual defect.

## Frozen-data experiment and provenance

The source data are `evolving/spatial0375_n128_f32`: the accepted common
dt=0.375 s, 32x128 bulk / 32-element fault configuration. Use three saved
particle states:

| Saved state | Physical time | Geometry consumed at following phase entry |
| --- | ---: | --- |
| 0 | 0 s | step 1, t=0.375 s |
| 2 | 0.75 s | step 3, t=1.125 s |
| 3 | 1.125 s | step 4, t=1.5 s |

Stable-ID positions equal the corresponding saved phase-entry positions
exactly. Both advected snapshots still have exactly H0 by stable ID; thus
history evolution is not needed for the discrimination. Maximum x displacement
is 0.000310033692 / 0.000478328680 m. No particle has wrapped in these states.

The saved files contain positions and volumes, not polygon/stencil checkpoints.
A disposable one-rank Release diagnostic creates the same mesh/particle set,
then regenerates each snapshot with the unchanged production domain generator.
**All 36,864 regenerated per-particle volumes equal their saved values exactly**
in all three snapshots. The original positions/domains are regenerated before
the diagnostic's deliberate terminal exception. No checkpoint is overwritten.

The phase input is the x-average of saved `phase_0.csv` at each mesh ordinate;
the H input is the x-average of `particles_0.csv` at each initial particle
ordinate, linearly interpolated in y to the actual advected particle positions.
Grouping ordinates uses 1e-12 m rounding. This defines one common transverse
function, not a separate fit per column or a local particle-history reset in
production. Nodal phi is **bitwise identical across x at every y** after the
existing FE constraints are applied. Inputs are held identical across snapshots.
The initial residual need not vanish: the finite-tolerance saved solution and
the explicit averaged/interpolated input are not a newly solved exact state.
Its uniformity, not a zero residual, is the control.

`PhaseFieldTestAccess::assemble(..., false)` calls the existing production
`PhaseFieldHandler::assemble_phase_field_system` with private matrix/vector
buffers. H substitution has scoped restoration. Live solution/old solution,
linearization, RHS, matrix, particle properties/positions and fault state have
equal fingerprints before/after every normal probe and every forced exception
after assembly. Geometric setup is separate from these noncommitting probes.
There is no mechanics solve or physical time advancement in this diagnostic.

An independent accumulation splits the load into reaction and gradient terms,
using the production material coefficients and identical constraints:

    rhs_i = -sum_p V_p [ w_ip (H_p g'(phi_p) + E_c alpha'(phi_p))
                       + 2 E_c ell^2 grad(w_ip).grad(phi_p) ].

Its sum differs from complete production assembly by at most 6.505e-16 in
Euclidean norm. The sign in the CSV files is **minus** the weak residual.
The diagnostic adds only a single-material coefficient accessor to the test
friend; no production API or algorithm is changed.

## Residual localization

Periodic slave contributions are combined using the current production
constraints. The slave at x=L is not counted as a zero-residual independent
node. Define the diagnostic residual density by dividing each independent
nodal weak load by its constrained CPDI lumped test mass, in Pa. Subtract the
mean transverse density profile on the fixed interior
x in [0.046875, 0.203125] m. All y rows are retained. This normalization is
**only a diagnostic**, not a changed solver norm or acceptance criterion.

| Domains | Production weak RHS L2 (Pa m^2) | Seam-node nonuniform density RMS (Pa) | Fixed-interior nonuniform density RMS (Pa) |
| --- | ---: | ---: | ---: |
| Initial | 4.39318648e-6 | 1.09352e-12 | 1.10542e-12 |
| Saved step 2 | 1.26973567e-4 | 0.147951 | 2.63279e-7 |
| Saved step 3 | 2.00441030e-4 | 0.233182 | 2.02668e-7 |

The combined periodic seam node plus its two adjacent independent node
columns (distance <= hx=0.0078125 m from x=0,L) contain
0.9999999999417 / 0.9999999999852 of the squared nonuniform density norm.
This is not merely an endpoint extremum: it is a column-by-column weak-load
comparison over the whole transverse profile. Raw weak nonuniform L2 norms
are 1.26447471e-4 / 1.99839524e-4 Pa m^2, versus 4.55231e-15 initially.

The gradient contribution dominates after removing the local test mass:

| Domains | Reaction nonuniform density RMS, all x/y (Pa) | Gradient nonuniform density RMS (Pa) | Total (Pa) |
| --- | ---: | ---: | ---: |
| Initial | 1.93864e-14 | 1.16231e-12 | 1.16087e-12 |
| Saved step 2 | 1.47742e-4 | 3.20846e-2 | 3.21463e-2 |
| Saved step 3 | 1.82768e-4 | 5.06823e-2 | 5.07386e-2 |

The raw reaction/gradient weak-load variations also partly cancel; both are
saved, rather than replacing the production weak balance by a column average.
The interior mean residual changes with the domain geometry even though the
input is frozen. That uniform discretization response is distinct from the
seam-localized nonuniformity.

## CPDI consistency and geometry

| Maximum over all particles | Initial | Step 2 | Step 3 |
| --- | ---: | ---: | ---: |
| abs(sum w - 1) | 4.44e-16 | 5.55e-16 | 4.44e-16 |
| norm(sum grad w), 1/m | 1.14e-13 | 1.57e-13 | 1.50e-13 |
| norm(sum w*x_vertex - polygon centroid), m | 3.41e-13 | 4.06e-13 | 4.44e-13 |
| independent polygon area / stored volume - 1, absolute | 2.04e-12 | 2.36e-12 | 2.70e-12 |
| Total measure / 0.25 m^2 - 1 | 2.22e-15 | -7.71e-9 | -5.28e-9 |

First moments use the actual polygon centroid, **not** the particle position.
The coordinate functions are evaluated in physical, unwrapped coordinates;
x itself is not a periodic scalar. These constant/linear-value properties pass
at the seam as well as the interior. Total measure is nearly conserved, not
claimed exactly conserved on the advected snapshots. Summation-order differences
between the C++ and NumPy totals are about 1e-13 m^2.

The physical gradient-of-coordinate check is not a pass: the Frobenius error
relative to +I is either roundoff or 2*sqrt(2). Counts of the latter are
19,396 / 19,957 / 20,354. This confirms the **already documented and deferred**
polygon-orientation sign issue in `stage_K1_cpdi_cause_and_correction_plan.md`
and `stage_K1_ownership_correction_review.md`, not a new ownership gap.
`SimplexIntegrator` uses absolute signed area for values but signed face sums
for gradients. For each convex polygon the common sign multiplies both test
and trial gradients, and therefore cancels identically in the phase residual
and Jacobian. It cannot explain the residual's seam localization. Do not fold
an unrelated gradient-sign change into the periodic correction.

The wall geometry has a much larger, directly measured effect:

| Quantity | Step 2 | Step 3 |
| --- | ---: | ---: |
| Seam-cell particle volume / initial volume range | [0.880946, 1.119051] | [0.816320, 1.183676] |
| Fixed-interior volume ratio range | [0.999999713, 1.000000265] | [0.999999817, 1.000000205] |
| Seam centroid-minus-parent y RMS, m | 6.63831e-8 | 1.42241e-7 |
| Interior centroid-minus-parent y RMS, m | 6.76534e-11 | 8.89169e-11 |
| Seam RMS abs(CPDI grad_x phi), 1/m | 5.24657e-5 | 7.63790e-5 |
| Interior RMS abs(CPDI grad_x phi), 1/m | 5.93973e-14 | 5.31133e-13 |

For a displaced regular row, a nonperiodic left/right wall predicts
V/V0 approximately 1 +/- dx/d_particle, with d_particle=0.25/96 m.
For the actual outermost particle rows, the RMS errors of this elementary
wall-strip prediction are only 3.17e-6 / 6.58e-6 in relative volume, versus
the observed O(0.1) changes. This independently identifies wall clipping,
not lost constant weights, as the dominant geometric mechanism.

## Mechanism and smallest proposed next action — not implemented

The domain generator constructs each patch from ordinary vertex-adjacent cells
and creates its Voro++ container with all periodic flags false
(`ParticleDomainHandler::generate_particle_domains`). At x=0,L, it has neither
the translated opposite-edge particle neighborhood nor a periodic domain
closure. Particle wrapping and periodic FE constraints do not repair this:
the constraints combine loads **after** the wrong wall-bounded domains and
their CPDI weights/gradients have been constructed. No wrap is needed for the
wall effect to occur. Changed centroids and generalized shape integration
introduce a seam-specific phase gradient/load for a nonlinear transverse
profile even though affine-value and constant reproduction remain correct.

The smallest justified correction scope is **periodic-image-aware domains and
CPDI sampling on the existing axis-aligned, fixed-mesh 2-D box**, using the
existing declared x periodicity, not a new benchmark flag or surface topology:

1. Complete the local neighbor cloud using opposite-boundary particles shifted
   by +/-L. Construct each real parent's full nearest-neighbor domain in a
   consistent unwrapped chart; do not impose a physical wall at the seam.
2. Locate/evaluate CPDI corner/centroid samples through their periodic images
   in actual FE cells, preserving the existing unique-owner tolerance and
   applying each sample once. Do not just set Voro periodic=true on each
   **local patch box**, which would impose the wrong period.
3. Integrate translated domain fragments once per real parent and retain its
   full measure. MPI ownership follows the real parent; image neighbors are
   not new particles or extra history owners. Keep physical top/bottom walls.
4. Review the shared domain representation and its surface-quadrature consumers
   explicitly before editing: a seam-crossing domain is contiguous unwrapped
   but can have two fragments in the physical box. Returning a single clipped
   polygon or silently integrating an outside-box fragment against an open
   fault tip is not an adequate implementation. Retain parent history and
   admission, and do not identify the two independent surface endpoint DoFs.
   If this needs a domain-fragment API change, document that bounded change
   for review rather than hiding it in a flag adjustment.

The first regression should be a small periodic-box particle lattice with
prescribed phi(y), H(y), no fault residual, and controlled uniform/shear x
displacement (dx/d_particle about 0.12 and 0.18 from these snapshots).
Require constant/affine-value reproduction, full-domain measure, and periodic
column invariance of the **production** weak phase load for the nonconstant
profile; constant reproduction alone is not decisive. Cover pre-wrap and
wrapped states, unique shared-face/image ownership, seam and interior controls,
and one/two ranks. Keep the previously deferred orientation-sign observation
separate. Then repeat this saved-domain phase probe and affected full-domain
surface projection/action checks before another coupled K3 trajectory.

This evidence is sufficient to propose the correction; it is not proof that
all later coupled endpoint effects disappear after it. No buffered-fixture
substitution is proposed: the homogeneous phase residual is demonstrably
nonuniform, so the conditional premise for blaming only open fault endpoints
is false.

## Commands, artifacts, and unchanged scope

All paths below are relative to
`benchmarks/reconstructed_fault/uniform_shear/evolving/seam-audit/`.

- `diagnostic.cc`, `CMakeLists.txt`, `audit.prm`, `run.py`: isolated harness.
- `output/state{0,2,3}_{nodes,particles}.csv`: production residual/decomposition
  and actual regenerated-domain consistency data.
- `output/domain_recovery.csv`: stable-ID saved-volume agreement.
- `columns.csv`, `summary.json`, `seam-audit.png`: x-column decomposition,
  complete quantitative evidence, and profiles.
- `summary.json`: SHA256 of production domain/phase source, executable, harness,
  and loaded Release plugin. Repository HEAD at execution:
  `dc1a96d3f72dce123393c90ad025e729885e8ee9`; existing dirty changes preserved.
- `run.log`, `resources.json`: **14.701 s**, **597,852 KiB peak RSS** (~584 MiB).
  The runner succeeds only after the completion marker; ASPECT intentionally
  exits 1 with `K3_SEAM_AUDIT_COMPLETE` before mechanics. This is a successful
  frozen diagnostic, not a claimed nonlinear convergence result.

Executed:

```sh
cmake -S benchmarks/reconstructed_fault/uniform_shear/evolving/seam-audit \
  -B benchmarks/reconstructed_fault/uniform_shear/evolving/seam-audit/build \
  -DAspect_DIR=/home/ein/repository/aspect/build-pf-cpdi -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/uniform_shear/evolving/seam-audit/build -j4
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/seam-audit/run.py
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/seam-audit/analyze.py
```

One initial compile exposed a benchmark-only Trilinos vector-reference/tensor
type mismatch, fixed by reading the scalar into a double. The first launcher
could not find `/usr/bin/time` and did not start ASPECT; the saved `audit.log`
records that. The Python resource runner then made one actual invocation.
Normal and forced-failure state restoration, three production/split residual
comparisons, exact saved-volume matches, and analysis assertions pass. Signed
gradient reproduction remains explicitly nonpassing/deferred as above.

No production source file changed. No fault64, normal512, new normal-resolution
run, support/I_h modification, physical initialization change, pressure change,
MPI campaign, solver-tolerance change or acceptance-criterion change occurred.
K3 convergence remains unestablished; K2 references and Gate K2 remain as before.
