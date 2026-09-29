# Rotated bottom: local coupled A/B experiment, September 28, 2026

The opt-in constraint works in the local AMG runs. Releasing bottom fault-normal
motion gives a modest reduction of the deep-end normal-stress concentration,
with a larger adjacent negative excursion. This is evidence of redistribution,
not a correction of the production BP3 stress history. Production PRMs and
checkpoints were not changed, and there is no core solver change.
The user subsequently confirmed that this treatment remains an option only:
the current production BP3 model retains full bottom velocity loading.

## Executed configuration and evidence

Base revision: `0fc1ce782` on `pf-rsf`, plus the plugin/fixture changes in this
directory and `../plugin/`. Exact commands are in [README.md](README.md).
The locally rebuilt Release executable/plugin use GNU 16.2.1, deal.II 9.6.2 and
OpenMPI 5.0.6. This does not qualify Intel builds or GMG.
`source_and_input_sha256.txt` records the delivered source/PRMs/fixture inputs;
`core_executable_sha256.txt` identifies the executable used. Build/run logs are
retained in `logs/`. The numerical runs preceded a formatting-only pass on the
three new C++ files; both final plugin targets were rebuilt successfully after
that pass. The hashes describe that final source, not a claimed byte-identical
copy of the pre-format shared library.
Generated fixtures, numerical outputs, analysis figures/CSVs, logs and build
products are preserved locally and excluded from the source/documentation
commit. Artifact links below therefore require this local evidence or a rerun
using the committed reproduction instructions.

- Box: 2 x 1 km; bottom fault intersection `(0,0)`; top
  `(-1000/sqrt(3),1000)` m. Tangent is derived from those endpoints and checked
  against `(1/2,-sqrt(3)/2)`.
- Fixed 6,275-cell mesh, 52,338 velocity + 6,579 pressure = **58,917 Stokes
  DoFs**; 222,510 DoFs including all auxiliary fields. Near-fault cell size
  6.25 m. Fault: 186 vertices, 185 segments, 4,440 normalization quadrature
  profiles. Mandatory leaf-tree/profile/completion checks passed.
- Production ell20 stationary profile, frozen mature fault, uniform deep
  strengthening friction: a=.025, b=.015, Dc=.008 m, f0=.6, Vref=1e-6 m/s,
  Vp=1e-9 m/s; G=32038120320 Pa, eta=1e26 Pa s, production damping.
- Q2/Q1 Stokes, LLS/Q2 history transfer, 3x3 particles per cell, no LLS limiter;
  actual mechanical normal traction + 50 MPa background, 20 m Helmholtz filter.
  No prescribed fault slip rates; all 186 nodes remain free in the accepted
  active-set records. Source continuation and regenerated Q1 I_h completion
  cover both endpoints. No nucleation or phase evolution.
- Common left/right loading and natural zero-perturbation-traction top;
  pressure normalization `no`. A prescribes both bottom velocity components.
  B prescribes the fault-parallel component and leaves the normal one free.
- Primary runs: two MPI ranks, one thread each, dt=200,000 s, 10 accepted
  positive-time steps to t=2,000,000 s (23.148 days), plus step-zero equilibrium.
  Initial state is `(Dc/Vp)*exp(.02*sin(pi*y_fault/1000)^2)` with normal
  extension; reference shear prestress remains the uniform steady value.
  Nonlinear tolerance 1e-8; linear Stokes tolerance 1e-9.

Resolved inputs, accepted step records and profiles are in `output-A/` and
`output-B/`. [Final profiles](analysis/final_endpoint.png) and
[stress histories](analysis/stress_history.png) have corresponding PDF files.
`analysis/` contains compact CSV copies and exact arc-length norms. The online
`local_metrics.csv` norms instead use lumped traction-mass weights; the table
below uses `analysis/*-norms.csv` throughout.

## Boundary implementation and weak form

`../plugin/bottom_constraint.cc` adds `uy = gt/ty - tx/ty*ux` through the existing
`post_constraints_creation` connector. Components are paired by FE support
point on actual bottom faces, including Q2 edge-interior points. It inserts
compatible locally relevant rows, counts by DoF ownership, preserves corners
and validates paired hanging interpolants. Conflicts throw; they are not skipped.
The benchmark verifier checks B's projection and retains A's Cartesian checks.

The callback runs before `close()` and the constraint comparison that controls
sparsity rebuilding (`source/simulator/core.cc`). Cross-component entries
therefore reach the assembled AMG matrices. ASPECT also creates homogeneous
velocity constraints during initial-field setup. The callback identifies the
physical versus homogeneous lift from the two retained side-boundary corner
rows, checks their consistency collectively, and mirrors that mode. This avoids
guessing from the nonlinear iteration counter. During the coupled solve,
`source/simulator/solver.cc` explicitly rebuilds/distributes a private physical
lift, then zeros Stokes inhomogeneities for residuals and Newton directions.
The callback is not rerun between that homogenization and the Newton solve.

The mechanics uses perturbation stress. The benchmark registers 50 MPa normal
and steady shear as background *fault tractions*; initial Maxwell perturbation
history is zero. `source/simulator/assemblers/reconstructed_fault_stokes.cc`
retains the frozen Maxwell stress volume term and the source/history correction.
Neither was changed. Natural zero complementary **perturbation traction** in B
needs no added bottom load. It neither zeros fault-normal compression nor
removes any constitutive history. Both cases retain the same pressure convention
and a natural top, so this is not a comparison of different pressure gauges.

## Qualification checks

Uniform-creep checks (`output-uniform-A/`, `output-uniform-B/`) each completed
step zero and one dt=200,000 s step in approximately 12 seconds on one rank.
Their fault rates differ slightly from exact creep due to spatial discretization:
maximum departure from Vp is 0.175% for A and 0.184% for B. The check does not
establish an exactly represented stationary solution. Step one accepts the
existing velocity without a Newton update and publishes the state/history;
this is the existing split initialization convention, not a frozen-V option.

The B one-step check also passed with two ranks in 10.4 seconds
(`output-uniform-B-mpi2/`), retaining the same mesh and 125 bottom support points:
123 independent rows, two corners, zero boundary hanging rows. The normal
variation test attains 1.284 Vp while its tangential part is zero to roundoff.
This fixture exercises MPI sharing and Q2 edge-interior points; it does **not**
exercise a nonzero count of boundary hanging rows, a restart, or GMG.

Across the primary B run:

- Maximum physical tangent error and homogeneous-variation tangent error:
  1.034e-25 m/s, measured against the FE lift. Continuous-profile interpolation
  error: 1.002e-12 m/s (0.1002% Vp), equal in A and B; do not interpret it as a
  failed strong constraint.
- Maximum free-component weak momentum residual: 1.44e-9 Pa m. This is the
  algebraic residual on independent noncorner bottom ux rows, multiplied by
  sqrt(3)/2 to represent unit normal variation; it is not a pointwise traction
  error. Constrained reactions are excluded.
- Maximum absolute net boundary flux: 1.38e-21 m²/s. At the final time,
  left/right fluxes are each -2.5e-7, bottom -3.02923e-10, top +5.00302923e-7
  m²/s. The oblique free motion redistributes bottom/top flux without adding an
  extra zero-bottom-flux condition. A's final bottom flux is roundoff zero.
- Final max |bottom u_n| = 0.003852 Vp, versus roundoff zero in A.
- The projected raw normal traction equals pressure + deviatoric contribution
  + 50 MPa to 2.24e-7 Pa across both primary profile series.
- Both runs pass the existing exponential state-update audit to 2.22e-16
  relative error and every fresh linear residual check. All line searches
  accept alpha=1 without rejected candidates. Including step zero, A/B use
  20/20 Newton updates and 479/389 Krylov iterations; elapsed times are
  71.2/66.8 seconds. No runtime reduction of mesh resolution was needed.

The separately rebuilt default production plugin compiles and its production
PRM passes parse-only validation. The existing restored-model C++ test passes
157 exact model/event/output comparisons. These checks preserve default APIs
and model values; they are not a repeat of a long production run.

## Primary comparison at t=2,000,000 s

Stress increments are relative to **each case's own step-zero equilibrium**.
RMS values integrate the exported Q1 increment squared exactly over the fixed
interval, then divide by its arc length and take a square root. The deep and
upper intervals are each 200 m; interior is the remaining 754.701 m.

| Quantity | A: full | B: tangent only |
|---|---:|---:|
| Deep raw-normal RMS increment | 5.15083 kPa | 4.97659 kPa |
| Deep max absolute increment | 48.8153 kPa | 42.8143 kPa |
| Location of deep max, down dip | 1154.701 m (bottom) | same |
| Deep RMS last increment / dt | .00267726 Pa/s | .00257662 Pa/s |
| Interior RMS increment | .814490 kPa | .809430 kPa |
| Upper 200 m RMS increment | .967618 kPa | .935698 kPa |
| Upper maximum increment | 2.21456 kPa at top | 2.21654 kPa at top |
| Deep V/Vp range | .997135–1.013566 | .996764–1.016074 |
| Interior V/Vp range | .990171–.998067 | .990095–.997698 |
| Final max absolute Delta ln Theta | .000295818 | .000352518 |

B reduces the deep RMS increment by **3.38%** and its maximum by **12.29%**.
However, the adjacent node, 6.242 m up dip, has a more negative raw-normal
perturbation: -25.517 kPa in B versus -17.428 kPa in A. The final endpoint raw
perturbation falls from +54.883 to +48.519 kPa; the friction-used endpoint
perturbation falls from +5.247 to +3.105 kPa. Pressure and deviatoric terms both
matter: at the endpoint they are +70.381/-15.498 kPa in A and
+66.101/-17.582 kPa in B. The spatial oscillation visible in the unfiltered
profile remains present. Deep creep/loading is retained, with a slightly higher
local peak slip rate in B.

The upper endpoint does not develop a comparable concentration, although its
pointwise increment increases very slightly. The four corner disks are tracked
separately in `corner_metrics.csv`. Final maximum corner pressure magnitudes
are 1.215 kPa (A) and 1.268 kPa (B); maximum committed deviatoric norms are
1.753 and 1.819 kPa. Bottom-left committed deviatoric norm increases from .177
to .464 kPa. Thus some stress redistribution reaches corners, but these sampled
corner signals remain much smaller than the deep-end peak. These are finite
200 m neighborhoods, not a claim of pointwise corner regularity.

## Accuracy follow-up and interpretation

The sole follow-up halves dt to 100,000 s for both cases, using 20 steps to the
same t=2,000,000 s. Mesh, material, disturbance and solver tolerances are fixed.
The material's initial pseudo-step is also halved to match the new fixed dt;
each increment series uses its own corresponding step-zero equilibrium.
Its metrics are recorded separately in `output-A-half/`, `output-B-half/` and
`analysis/`; no refinement sweep or production restart was launched.

| Final quantity with dt=100,000 s | A | B |
|---|---:|---:|
| Deep RMS increment | 5.39424 kPa | 5.28173 kPa |
| Deep maximum increment | 51.0894 kPa | 45.8982 kPa |
| Deep RMS last increment / dt | .00262327 Pa/s | .00255551 Pa/s |
| Interior RMS increment | .852836 kPa | .847251 kPa |
| Upper RMS increment | 1.01657 kPa | .983156 kPa |
| Newton updates / Krylov iterations, including step zero | 37 / 871 | 40 / 779 |
| Wall time | 131.0 s | 135.3 s |

The peak reduction persists at **10.16%**, but the RMS reduction shrinks to
**2.09%**. Halving dt changes A/B's individual RMS increments by +4.73%/+6.13%,
larger than the boundary-induced RMS reduction. Even the paired RMS difference
changes from 174.24 to 112.51 Pa. Therefore the small area-wide benefit is **not
established beyond temporal error** by this check. No formal convergence order
or extrapolated limit is inferred from two timesteps. The endpoint peak trend
is more persistent, but neither spatial convergence nor removal of the
neighboring negative excursion is established. Both half-step runs pass the
constraint and state audits, with no rejected line-search trials. The complete
primary-plus-follow-up simulation time is under seven minutes.

**Next bounded recommendation:** retain full bottom loading in production.
Use the existing uniform fixture to add a same-quadrature comparison of imposed
boundary-profile strain and implemented continued slip source in the bottom
200 m, separating pressure/deviatoric cancellation and the first few fault
nodes. This addresses the discrete-mismatch conjecture before requesting a
single targeted spatial-resolution test. Do not begin a production restart or
claim a stress-concentration fix from the present small RMS change.

The local comparison tests only release of the normal velocity component. It
does not test a different prescribed parallel profile and therefore cannot
eliminate the discrete source/profile mismatch hypothesis. The uniform check's
small departure from steady creep and the short-run nodal oscillations remain
relevant limitations. A controlled short production A/B restart would still
be required before adopting this condition in the full BP3 model; the existing
restart identity intentionally rejects changing boundary mode silently.

## Preserved setup failures

The first two-rank adaptive setup attempts failed the mandatory leaf-tree check
before mechanics (`output-uniform-A-mesh-check*`). A one-rank attempt reached
the initial mechanical solution but encountered production station coordinates
outside the local fault (`output-uniform-A-station-check`). The first B setup
exposed the homogeneous initial-field lift (`output-uniform-B-initial-lift-check`).
These checks were fixed, not bypassed; their artifacts and earlier generated
fixtures are preserved. The final global tagging recipe verifies the same mesh
on one and two ranks.

A's completed ten-step ASPECT run terminated normally with all outputs and
audits, but its outer shell returned 2 because the launcher was edited while
that shell was running. This is recorded rather than hiding/repeating the run.
The final launcher passes `bash -n`; subsequent B and half-step runs use the
unchanged script. Production outputs from earlier user sessions are untouched.
