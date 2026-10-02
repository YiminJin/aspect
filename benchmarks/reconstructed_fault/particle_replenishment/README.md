# Native particle replenishment after R6

**Outcome: history limiting works in these local cases, but the proposed 12/24
variant is not qualified for server continuation.** The rejected-attempt test
exposes missing restoration of the particle manager's random-number generator.
No production source or physical parameters were changed. R7 was not started.

Reference: R6 closure `9980ff30162ba4b6f058e313eb7a2be16b947cd4`, branch
`pf-rsf-refactor`. All 865 entries in the R6b source/header manifest match the
current tree. Executable `build-refactor-r6b/aspect-r6b-qualified`, SHA256
`cdf7e0c58009d11dfe8c216caeb31ba6d8edc08b71360fec9fb63fcd95287596`;
matched R6b BP3 plugin. Its embedded `1a3b57eda` banner predates qualification;
it is not the source baseline identifier. See [provenance](results/provenance.json).

## Experiment and results

Native `reference cell` 4×4 and `random uniform` (2048 requested, `Random cell
selection=true`, seeds 5432/5433/5434) run on 128 equal-volume cells. All realize
2048 initial particles. Both families use native point-density addition/removal,
12/24 bounds, bandwidth 0.3, `cutoff c1 dealii`, candidate granularity 6. Counts
below 12 are filled to 12; counts above 24 are reduced to 24. Neither bound
prescribes the mean. Placement uses sparse density candidates and randomized
tie breaking, not independent uniform sampling.

Prescribed U=(1,0.6), dt=0.2, t=4 carries histories four horizontal and 2.4
vertical cell widths. There is no imposed history reset at inflow: newly inserted
particles use the actual native interpolator on the surviving cloud. Analytic
references extend to the entire plane, q(x,t)=q0(x−Ut). Three trace-free tensors
are [q,−q,q/2], with q/1e8 respectively 1, 1+.02x+.03y, and
1+.1 sin(pi x/4) cos(pi y/4). The curved wavelength is eight cells. This is
transport/reconstruction, not Maxwell constitutive evolution.

The constant/affine consistency criterion is error/1e8 ≤1e−10 for unlimited
well-conditioned fits. Constant error is ≤4.45e−14; affine error ≤3.18e−14 over
all recorded support/quadrature/LLS/Q2 samples. No cell had fewer than three
particles or a singular-value ratio below 1e−8. Thus no rank-deficient fallback
or uncontrolled near-singular fit was observed. Bounds alone do not guarantee
these outcomes for other clouds.

| Initial generator | Startup births/removals | Final count | Transport births/losses¹ | Worst σmin/σmax | Final curved Q2 RMS /1e8, unlimited → limited |
|---|---:|---:|---:|---:|---:|
| Regular 4×4 | 0/0 | 1837 | 886/1097 | .19200 | .05731 → .03442 |
| Random 5432 | 49/10 | 1926 | 1018/1179 | .11053 | .05341 → .03399 |
| Random 5433 | 48/4 | 1937 | 1013/1168 | .09836 | .05122 → .03420 |
| Random 5434 | 36/7 | 1899 | 1023/1201 | .10538 | .05671 → .03401 |

¹Serial membership losses combine outflow and native removals. Startup has no
advection, so those losses establish removal directly. Limited and unlimited
runs have identical counts/geometry. Random initial cell counts span 7–28,
7–26 and 7–27 respectively; after management all lie within 12–24.

Unlimited curved error grows under repeated extrapolation through open inflow:
maximum recorded error/1e8 is .31673, .27273, .22950, .28321 respectively. This
is not a reproduced 112-GPa or singular-cloud failure. Native limiting with
boundary extrapolation disabled reduces these maxima to .12294–.12673 and
keeps sampled curved Q2 quadrature values within the analytic .9–1.1 range.
It costs affine accuracy: maximum error/1e8 becomes .15909–.16404. Constant
history remains accurate to 1.5e−16. This tradeoff must not be described as
exact affine transport. Component masks come from runtime property names;
integrator properties are excluded. Proportional tensor components receive
the same limiter policy, preserving the test tensors' trace constraint.

Metrics use identical physical support points and tensor-product Gauss-3
points with cell-volume weights, never unweighted particle means. Initial
random-cloud unlimited curved RMS errors are .00348–.00352. Initial replacement
changes the volume-weighted LLS mean/1e8 by +8.09e−7, +2.12e−7, −8.52e−7;
it does not conserve that mean exactly. Final Q2 mean errors/1e8 are −.00576,
−.00201, +.00068, −.00374 unlimited, versus +.00147 to +.00286 limited.
Transport mean errors include advection, extrapolation and transfer; they are
not an isolated per-step replacement conservation measurement. All-cell geometry/reconstruction statistics
are captured at generation and after initial management; during transport,
first-insertion diagnostics capture the old cell, and post-step diagnostics
capture the managed cloud. There is no whole-domain pre-removal transport dump.

[Comparison](results/comparison.csv), [time/stage fields](results/stage_fields.csv),
[count-conditioned metrics](results/count_conditioned.csv),
[insertion proposals](results/insertion_summary.csv), and tiny worst-cell CSVs
retain all seeds. Two plots summarize [conditioning/error](results/conditioning_and_error.png)
and the [limiter tradeoff](results/limiter_tradeoff.png). Three seeds do not
establish statistical robustness. No extra generator or count sweep was run.

## Coupled BP3 and the observed defects

The maintained coarse `auto-bp3-base.prm` fixture has 1875 uniform cells and
65,560 DoFs. Its existing broad profile (length scale 4000), direct effect .025,
full-bottom Dirichlet lift, filter20, free fault-slip unknowns, AMG, tolerances,
and 100-s timestep policy remain unchanged. These are inherited **functional
fixture settings**, not production-resolution BP3 qualification. Run steps 0–2.
Seed 5433 was selected by the poorest cheap-test singular-value ratio.

With Maxwell-only limiting, random startup inserts 529 and removes 113
particles (30000→30416). Eleven births acquire negative H from positive inputs;
the next history commit rejects stored H. The first proposal is −1146.42 from
[869.83,22925.53], with σmin/σmax=.27018; the worst stored H is −49805.17.
[The actual cloud](results/negative_H_cloud.csv) and
[unique negative proposals](results/negative_H_proposals.csv) reproduce this
positivity failure. It is not a conditioning failure. Interpolation may be
called once for each late-initialized property; proposal counts are deduplicated
by step/cell/location/component, not counted as multiple births.

Explicit native limiting of **H and Maxwell components**, identically for both
generators, keeps H positive and both coupled runs pass. Other initial
composition/integrator components are not accidentally limited. The runtime
BP3 mask is `true,false,false,true,true,true`. No conditioning fallback or
production-code correction was needed to address these invalid H proposals.
Regular retains 30000 particles; random retains 30416. At t=200, maximum
nodal V difference is 3.20e−18 m/s and Theta difference 3.20e−7 s (Theta≈8e6 s).
The largest accepted maximum-particle-stress difference is .0852 Pa, and the
largest extrema difference in friction-used normal stress is .00389 Pa against
50 MPa. Newton update/iteration counts agree; fresh residual checks pass. These
short, initially zero-stress trajectories do not establish long-loading safety.

For actual transport events, a separately named generator delegates to native
reference-cell generation, then moves three staggered rows in the bottom cells
just below the **internal** y=2000 face, *before* native history initialization.
No native comparison population is altered and Vp is unchanged. Step 1 creates
245/removes 121 particles; step 2 creates/removes three, exercising nonzero
stress inheritance. Near-external-boundary prototypes failed domain geometry
checks and are recorded as fixture failures, not production defects.

The unchanged BP3 audit assumes fixed IDs and rejects these births. The
`Track birth audit` fixture adapter records each genuinely new ID's H before
mechanics; surviving IDs retain exact H checks. Adapter entries are discarded
on the controlled rejection and retained on acceptance. It changes no physical
state. The original failing audit case remains reproducible.

- Serial and two-rank event runs complete. At 7701 common visualized Q2 support
  positions, 83 MPI piece-boundary occurrences have identical shared values.
  Serial/MPI maximum differences: mapped stress 2.49e−4 Pa, pressure 1.53e−4 Pa,
  velocity 1.01e−14 m/s; fault V ≤6.64e−22 m/s. Partition-dependent random choices
  are permitted; visualization is Float32 and is not a full-precision DoF audit.
- Checkpoints immediately before and after the first event reproduce the
  uninterrupted serial hashes of positions, physical properties, integrator
  history, next ID and counts through step 2 exactly. Fault/profile/accepted
  physical diagnostics also match. Derived domains/projection/normalization
  rebuild through normal restart paths; solver fresh checks pass.
- A controlled precommit rejection at t=100 includes 245 births/121 removals.
  Before its t=50 retry, all particle/ID hashes equal the step-zero backup.
  Both retry and direct t=50 accept with identical fault/solver diagnostics,
  30122 particles and next ID 30242, **but their position, property and integrator
  hashes differ**. The native manager RNG is advanced by the rejected attempt
  and omitted from `backup_particles`/`restore_particles`. This is a demonstrated
  rollback bookkeeping defect, not fixed by documentation or the H limiter.

The same RNG is absent from checkpoint serialization. The passing short
restart cases do not establish preservation of RNG state: their later choices
did not distinguish it. A correct extension must retain per-rank RNG streams;
only rank zero's serialized simulator stream is written to `resume.z`. Simply
appending rank zero's generator is insufficient for a general MPI restart.
That compatibility/MPI decision is left explicit rather than hidden in this
test-only commit. [Exact checks](results/lifecycle_checks.csv),
[states](results/lifecycle_states.csv), [field differences](results/coupled_differences.csv),
and [shared supports](results/mpi_supports.json) retain the failed retry checks.

## Current lifecycle source map

- `source/particle/manager.cc`: `setup_initial_state` generates and initializes;
  `advance_timestep` integrates, sorts/migrates, enforces bounds, updates and
  exchanges properties, rebuilds domains/CPDI. `apply_particle_per_cell_bounds`
  computes density candidates/removals and calls `initialize_late_particle`
  **before insertion**. An improved new position cannot rescue its first value
  from an unsafe old-cloud fit. Initial bounds normally run in timestep-zero
  advection; the cheap observer explicitly invokes that native dt=0 preparation.
- `source/particle/property/interface.cc`, `maxwell_stress.cc`,
  `crack_driving_force.cc`: property initialization/late interpolation; no oracle
  history overwrite. H and Maxwell stress are particle-owned. Fault V and Theta
  remain reconstructed-manager properties; legacy initial Theta composition is
  not a new copy of evolving fault state.
- `source/particle/interpolator/linear_least_squares.cc`: reference-cell affine
  QR, insufficient-particle/failed-column fallback to `cell_average`, native
  component limiter and boundary extrapolation. No near-singular SVD cutoff.
  The fixture SVD is independent observation only.
- `source/simulator/initial_conditions.cc`: `interpolate_particle_properties`
  averages cell proposals at continuous Q2 support DoFs using distributed sums/
  counts and constraints. Bounded proposals do not prove bounded Q2 interiors;
  both support and Gauss-3 values are measured here.
- `source/material_model/phase_field_fault/history.cc`: prepare/validate/publish
  accepted Maxwell and H histories; simulator/solver still owns acceptance.
  `source/particle/particle_domain.cc` builds native Voronoi/CPDI data;
  reconstructed manager projection caches consume those domains.
- `source/simulator/core.cc` and `solver_schemes.cc`: backup/restore around
  repeats, native particle advection and transfer. Particle handler backups
  contain membership/properties/integrator data, but omit manager RNG.
  `source/postprocess/particles.cc`, `include/aspect/particle/manager.h`,
  `source/simulator/checkpoint_restart.cc`: checkpoint, deserialization and
  transient domain/cache rebuilding; RNG omission remains.

## Reproduce and limitations

Build with the same Release toolchain as R6b (GCC12.4, OpenMPI5.0.6,
deal.II9.6.2/32-bit indices, Trilinos14.2). From repository root:

```sh
export ASPECT_SOURCE_DIR="$PWD"
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
cmake -S benchmarks/reconstructed_fault/particle_replenishment/plugin -B benchmarks/reconstructed_fault/particle_replenishment/build -DAspect_DIR="$PWD/build-refactor-r6b" -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/reconstructed_fault/particle_replenishment/build -j2
python3 benchmarks/reconstructed_fault/particle_replenishment/prepare_coupled.py
```

`run_cases.py CASE RANKS [LABEL]` records the exact command/environment/status,
uses one thread, caps each run at 180 s and the sum at 900 s. Run **sequentially**
in a fresh output area; logs refuse overwrite. CASEs: `regular`, `random-5432`,
`random-5433`, `random-5434`, each also with `-limited`; then
`coupled-regular`, `coupled-random-5433` (expected H failure), their `-Hlimited`
cases, `crossing-audit-original` (expected fixed-ID audit failure),
`crossing-serial`, `crossing-mpi` (2 ranks), `crossing-direct`, `crossing-retry`.
For each of `before`/`after`: run `crossing-create`/`crossing-after-create`,
`prepare_restart.py before|after`, then `crossing-resume`/`crossing-after-resume`.
`candidate-parse` invokes `--validate` for the native proposed include and passes.
Run `analyze.py`, `compare_lifecycle.py`, `plot.py` to regenerate summaries
(Python numpy, matplotlib, vtkmodules). `verify_evidence.py` checks the recorded
evidence and explicitly reports `server_continuation_qualified=false`; its
consistency checks do not turn the known retry defect into a qualification pass. See [exact run commands](results/runs.csv).

Total measured simulation/startup wall time, including failed diagnostic
attempts, was **486.9 s**, excluding builds. Some independent jobs overlapped;
these are summed run times, not benchmark speedups. Final cheap runs took
7.7–9.3 s under contention; earlier isolated runs took 2–4 s. Coupled runs were
roughly 10–47 s. Reported child maximum RSS was ~211 MiB cheap and at most
332 MiB coupled; `RUSAGE_CHILDREN.ru_maxrss` is not aggregate MPI memory.
Native logs retain advection, interpolation, domain and mechanical timings.
No touched production interfaces require a new R6 solver/GMG/melt campaign;
the existing BP3 fresh-residual/work/Theta checks ran in the selected cases.

The [proposed regular include](inputs/proposed-bp3-12-24.inc.prm) is parsed but
**awaiting qualification, not approval to launch**. Keep regular 4×4: random
seeding shows no compelling benefit, and its global-count semantics do not
carry over to adaptive BP3 cells. The old 142.93-year mechanism remains a
hypothesis: no old cloud was recovered, no collinear depletion control or
long trajectory was run. Q2 limiting, all-rank peak memory, adaptive-mesh
behavior, nonzero-stress rollback and RNG-sensitive MPI restarts remain gaps.

Recommended next bounded task: correct native particle RNG backup/restore and
specify compatible per-rank checkpoint restoration (including old snapshots
and changed rank counts), then rerun these exact event/retry/restart fixtures.
Keep any production correction in a separate commit. No numerical correction
is included here; native limiter configuration addresses H validity, while the
RNG defect prevents continuation qualification.
