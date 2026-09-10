# Domain quadrature implementation: focused verification stop

2026-09-09. **Partial implementation, not a verified correction baseline.**
The approved discrete rule and its evaluation convention are recorded in
`stage_K2_domain_quadrature_addendum.md`, written before production edits.
The first implementation is restricted to straight 2-D faults. Focused tests
now show that this restriction is too narrow for existing reconstructed-fault
coupling fixtures. Do not mark this revision finalized or start the trajectory
comparison until the geometry issue is resolved and coupling tests pass.

## Implemented scope

- `ParticleDomainHandler` retains the polygon it already constructs and exposes
  it read-only. A geometry generation counter invalidates surface quadrature
  even when the domain volume happens not to change. No Voronoi construction,
  clipping, deduplication, periodic flag or CPDI weighting was changed.
- The fault utility integrates a convex domain's linear transverse width,
  splitting at projected polygon vertices and fault nodes. It retains full
  measure, supports inclination/nonuniform straight segments, and uses constant
  endpoint continuation. Positive three-point Gauss weights integrate geometric
  Q1 first/second moments exactly. Normal domains are not clipped.
- The manager uses these points in generic property/scalar loads, mass,
  support and projection-residual diagnostics. Parent-center reverse
  interpolation and history update locations remain unchanged.
- Surface assembly reevaluates the nonlinear response at every surface
  integration coordinate, holding the parent bulk/history samples P0 as
  defined in the addendum. R_Gamma, K_V, G and mass-based residual norms share
  the rule. G's bulk lookup is deduplicated by parent, not repeated remotely
  for every integration point.
- Point responses expose the existing traction decomposition for diagnostics.
  The surface helper retains its integrated weak terms and mass in the frozen
  linearization. The benchmark exports `surface_weak_*.csv` from that snapshot,
  before-history-publication semantics, instead of inferring the accepted
  mechanical equation from newly published particle stresses. The measurement
  script recognizes this export while preserving old saved-data analysis.
- `current_design.md` and `specification.tex` explicitly distinguish the
  approved discrete revision from the previous correctly implemented rule.
  The first implementation's straight-geometry limitation is documented,
  but has not passed the affected integration coverage.

No physics, full I_h, admission widths, solver tolerances, endpoint DoF
identification or history-commit timing was changed. Changed initial
projections have not yet been measured in a corrected trajectory and must
not be replaced with the old projections. Bulk cell-average/shared-node
history transfer remains the separate item documented in the endpoint-moment
report; no history interpolator was modified.

## Passed focused checks

Build command: `cmake --build build-pf-cpdi -j4`, exit 0, Debug and Release.
Build evidence is in
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/domain-quadrature/build.log`.

Commands from the repository root:

```sh
build-pf-cpdi/aspect --test '[fault_domain_quadrature]'
mpirun -np 2 build-pf-cpdi/aspect --test '[fault_domain_quadrature]'
```

**149 assertions / 3 cases pass on one rank and on each of two ranks.**
These include:

1. Analytic mass, constant loads, segment crossings and full tip-domain coverage
   on horizontal and inclined geometry, with independent open endpoint DoFs.
2. Nonlinear log-response quadrature comparisons at orders 3, 5 and 7, separately
   from exact geometric moments. Order-3/7 load differences are below 1e-8;
   order-5/7 differences below 1e-12 for the stated smooth manufactured field.
   This is not a replacement for production constitutive quadrature verification.
3. Nonconstant transverse stresses on **production ParticleDomainHandler
   domains**, with controlled shear and one-/two-rank ownership. Exact analytic
   strip mass entries agree within 1e-12 m^2. No periodic domain flags are used.

The production manufactured cross uses the same physical .25 m by 2w strip,
w=.3088215939070757 m, with a 1x2 coarse mesh refined to nx=8/16/32, three
particles per cell direction, and fault spacing six tangential particle
spacings. Thus its transverse spacing differs slightly from the earlier
offline near-square cloud; those numerical error values are not interchangeable.
The exact transverse mean of both prescribed centered stresses is zero.

| nx | displacement/spacing | Even error: point / domain (Pa) | Odd error: point / domain (Pa) |
|---:|---:|---:|---:|
| 8 | .1 | .00129896 / .000733678 | .0100904 / .0000269803 |
| 16 | .2 | .00240114 / .000191287 | .0207293 / .0000144904 |
| 32 | .4 | .00890617 / .0000556488 | .0414967 / .00000873458 |

The physical shear displacement is fixed across this cross. Integrated errors
decrease; the point-rule endpoint error grows. Remaining integrated error
includes the prescribed smooth stress represented by parent-constant samples.
One-/two-rank printed values agree to the displayed precision. Logs:
`domain-quadrature/unit-one.log` and `unit-two.log`.

An earlier executable invocation before rebuilding the newly added unit source
reported no matching tests; it is not counted as verification. The results
above come from the rebuilt executable. MPI tests required execution outside
the filesystem/network sandbox to initialize local MPI sockets.

## Failed integration gate and bounded diagnosis

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_(surface_(adiabatic_pressure|dynamic_pressure|rate_dependent)|condensed_adiabatic(_mpi)?)$' -j2
```

**0/5 passed, 222.05 s.** All five stop at the straight-fault geometry guard
before surface derivative/action verification. This is not a Jacobian,
linear-solver or nonlinear-tolerance failure. No expected outputs or fixture
parameters were changed to hide it.

A single follow-up replay of `phase_field_fault_condensed_adiabatic` with an
improved geometry error message failed in **55.78 s**. At reconstructed vertex
1, transverse deviation from the endpoint chord is **-1.4720652613542379e-7 m**;
the roundoff allowance is **1.7053025658242413e-14 m**. The deviation exceeds
roundoff by about 8.6 million, so relaxing a roundoff check is not justified.
The fixture has a prescribed straight reference but its actual reconstructed
polyline is not straight. Existing reconstruction permits this.

Evidence: `domain-quadrature/actions.log` and `geometry-diagnosis.log`.
The latter was obtained with the actual checked-in fixture, unchanged.

**The straight-only implementation scope is insufficient for existing
integration coverage.** Recommended next action is a reviewed geometric
addendum for full-domain partition onto the existing 2-D polylines, preserving
the parent admission/fault assignment and full measure. It must define corner
and tip coordinates, segment partition coverage and basis evaluation before
implementation. This is more than increasing a numerical tolerance.
Do not flatten the reconstruction, silently project everything onto its
endpoint chord, or replace the affected fixtures merely to obtain a pass.

## Not yet verified / held

Production K_V/G finite differences, full coupled actions and the affected
lifecycle/rollback/restart tests have not passed this geometry gate. The short
K1 reference replay and unchanged 64/128 K2 trajectories through 1 s were
**not started**. Consequently there are no corrected-trajectory initial
projection changes, endpoint traction, support/normalization measurements or
runtime/memory comparisons to report yet. Prior saved evidence is retained
but cannot certify this new rule. No larger temporal campaign, true-normal-
stress test or full ASPECT suite was run.

Existing unrelated working-tree changes were preserved. This partial revision
is uncommitted. `git diff --check` passes; the production change is not ready
to be treated as a verified correctness baseline.
