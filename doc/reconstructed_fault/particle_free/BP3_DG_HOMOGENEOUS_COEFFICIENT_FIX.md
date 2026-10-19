# BP3 homogeneous-coefficient startup repair

This repairs the server failure at `p.eta_ve==eta` in
`prepare_surface_linearization()`. The supplied `bp3_dg_first_event/log.txt`
reports 48 ranks, 552084 cells and 19498528 native DoFs, failing during the
initial mechanical solve at step zero. It contains no completed positive step.
The existing worktree remains based on
`b2dfdebe43437d6028b06d4f9f862cc80a00cc71` plus the preserved optimization work.

## Demonstrated cause

The configured homogeneous modulus is G=32038120320 Pa and viscosity is
eta=1e26 Pa s. Native bulk assembly and accepted independent history already
use these values directly. However, the bulk/fault point constitutive responses
still calculated G through compositional averaging, including when
`Use homogeneous mesh data=true`.

For equal moduli, harmonic averaging evaluates

```
G_mixed = 1 / ((1-f)/G + f/G)
kappa = -eta * expm1(-dt*G_mixed/eta)
```

The first expression is algebraically G, but its rounded value depends on f.
At the server's initial interval dt=4e6 s, this makes the surface derivative's
kappa vary by a few ULPs across mixed chemical labels. The exact guard is testing
the cache's actual coefficient dependency, so a coefficient inconsistency must
be corrected at its material source. Replacing equality by a tolerance would
leave native bulk and fault-point arithmetic inconsistent.

The pre-fix local executable reproduces the same assertion in a two-rank
4136-cell oblique fixture when its constant chemical label is replaced by a
continuous label in [.2,.8], keeping the same homogeneous bulk coefficients,
uniform strengthening friction and native 4e6 s initialization/adaptive controls.
The earlier small optimization fixture had a constant label and did not exercise
this variation. The first two reproduction-input attempts failed earlier on an
inadmissible additive label and an unsupported model operator; those logs are
retained separately from the successful assertion reproduction.

## Repair and preserved behavior

The material now has one `compute_elastic_shear_modulus(fractions)` accessor:
homogeneous mesh data returns the configured G directly; otherwise it calls
the original averaging operation unchanged. It is used in native bulk viscosity,
frozen stress, bulk/fault point responses, surface history candidates and the
history coefficient/initialization consumers. The test accessors follow the same
production material accessor. Creep viscosity already had the corresponding
homogeneous direct-value rule.

Chemical labels still affect friction and the existing localization/normalization
calculations. This change does not replace them by homogeneous friction, modify
normalization or its derivatives, or alter the Maxwell law. The exact coefficient
assertion, filtered G factorization/cache invalidation, B/C operators, quadrature,
limiter/filter settings, acceptance/publication allowance and checkpoint format
remain unchanged. Heterogeneous/default/legacy material averaging is retained.
All four server parameter files are unchanged.

## Verification and evidence

The rebuilt core and matching local/ordinary plugins pass these fresh checks:

- **127 native cases / 165094 assertions pass.** The new test uses 1001 mixture
  fractions, four averaging selections and dt=200000, 4e6 and 4e7 s. Configured
  homogeneous G and both Maxwell coefficients compare exactly with their direct
  reference. Independent heterogeneous-mixture controls retain all four legacy
  averaging rules. The old harmonic kappa range is
  1.2815248119788469e17–1.2815248119788472e17 Pa s: a 32 Pa s spread, two ULPs,
  approximately 2.50e-16 relative.
- Optimized and retained-reference two-rank mixed-label runs complete two
  positive adaptive steps with the identical sequence **4e6, 7.64e6 s**.
  The exact coefficient assertion stays active. Source coefficients and
  operand-derived allowance arrays match their uncached computation bitwise,
  including non-proportional manufactured histories/all nine Q2 modes.
- Frozen B/G/C relative discrepancies are **1.01e-15 / 4.17e-15 / 1.71e-15**,
  with 3060 stored Stokes hanging relations, three additionally restricted
  surface rows, endpoint completion, variable friction and actual pressure
  scaling. The filtered map rebuilds on the changed second-step coefficient.
  Independent reconstructed-QP stress differs by <=1.08e-10 Pa on an operand
  scale of 1.31e6 Pa.
- Optimized/reference whole-domain full-tensor RMS difference is **4.31e-9 Pa**,
  relative L2 6.11e-14, localized RMS 1.15e-8 Pa and maximum 5.34e-8 Pa.
  Phase, I and native surface mass match exactly. Largest native weak-load
  difference is 2.39e-7 Pa m; worst interior perturbation-relative difference
  across t0/steps 1/2 is 8.11e-12. These are algorithm-comparison differences,
  not a physical accuracy estimate.
- Serial step-one checkpoint → four-rank step-two continuation completes with
  the same timestep sequence. Against the uninterrupted two-rank run, tensor
  RMS is **1.10e-8 Pa** (relative 1.56e-13), maximum 5.18e-7 Pa, weak-load maximum
  2.75e-6 Pa m and interior perturbation-relative difference <=3.90e-11. These
  MPI solver/reduction differences are measured rather than called bitwise equal.
- A completed one-owner candidate rejection leaves accepted state unchanged;
  identical-dt retry matches the uninterrupted optimized run **exactly** in all
  compared persistent fields, reconstructed stress and loads. Accepted histories
  publish once on every rank, including restart. Existing true linear checks,
  nonlinear residuals and publication guards pass without changing allowances;
  maximum publication error/allowance across the matrix is .00334.
- A deliberately heterogeneous G input still fails the original homogeneous
  material admission check. One earlier launch of that negative test could not
  open MPI sockets in the sandbox; the bounded rerun outside the sandbox reaches
  and verifies the intended material rejection. All four unchanged server
  parameter files validate again with ordinary plugins and no test macros.

Fresh local results, native commands/inputs, source hashes and the original
failed reproducer are recorded in
`benchmarks/reconstructed_fault/particle_free/evidence/bp3_homogeneous_coefficients/`.
The raw native outputs remain under
`build-particle-free/bp3-optimization/runs/coefficients-*`; build and repair
artifacts are under `build-particle-free/bp3-homogeneous-coefficients/`.
The original server log and all earlier optimization evidence/packages remain.
The new source archive includes the failure log under qualification evidence,
not as a runtime input.

## Server update

The small `bp3-homogeneous-coefficients.patch` is against the **r3 source
snapshot** from `bp3-dg-stationary-optimized-20261017-r3.tar.gz`. Its six paths
include the core fix and regression tests; it is verified by applying it to
copies of those original files and comparing all resulting bytes. Apply it once
to that source, then rebuild the existing core/plugins with the same stack:

```sh
cd "$WORK/build/aspect/pf-rsf-dg/source"
patch --dry-run -p1 < /path/to/bp3-homogeneous-coefficients.patch
patch -p1 < /path/to/bp3-homogeneous-coefficients.patch
cmake --build "$WORK/build/aspect/pf-rsf-dg/build" -j 4
cmake --build "$SCRATCH/bp3_dg/plugin/build" -j 4
```

The regenerated r4 full source archive already contains this repair; do not
apply the small patch again to it. Preserve the failed output/logs separately.
For the supplied failure, initialization did not finish, so start the bounded
startup using a fresh `output/`, rather than trying to resume that failed state:

```sh
cd "$SCRATCH/bp3_dg"
source ./environment.sh
ibrun ./plugin/aspect-release startup.prm
ibrun ./plugin/aspect-release startup-resume.prm
```

Inspect the first launch before continuing the second. Follow the existing
checkpoint/evidence and scheduler-margin instructions. A genuinely valid earlier
checkpoint can still be used with the unchanged same-mode restart format;
a partial failed state is not a checkpoint. No server job is launched locally.

The inherited spatial/endpoint temporal findings and
**strict_reference_match_passed=false** remain. Bounded coefficient/operator/
lifecycle checks do not qualify a full first event, 48-rank resource costs or
the server compiler stack.
