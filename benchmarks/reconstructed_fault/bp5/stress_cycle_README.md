# Frozen stress-transfer / timestep investigation

This is a diagnostic, not a new BP5 trajectory or a new interpolator selection.
Use accepted step **5612**, physical time **5310111071.5634108 s**, from
`output-normal-diagnostic/restart/01`. Do not use the later checkpoint.

## Local outcome and execution budget

The initial estimate was 15–35 minutes on four local ranks (roughly 24 GiB
aggregate memory), based on the saved 106-second/32-rank four-step server run
and the earlier local startup timings. Trials are capped at 900 s each and
3000 s aggregate. These caps exclude compiling and copying input files.

The actual local four-rank restore failed **before a mechanical solve**, after
3.98 s, with `Cannot seem to deserialize ... class version
St6vectorIN6dealii5PointILi2EdEESaIS2_EE`. This is a checkpoint/software
compatibility blocker, not evidence about stress transfer or timestep response.
No archive conversion, loading prefix, relaxed restart check, or further large
local run was attempted. Use the server toolchain compatible with this archive.
Failure log: `stress-cycle-qualified-20260923/dt/run.log`; execution metadata:
`stress-cycle-qualified-20260923/dt/execution.json`. The half/quarter branches
were not launched. Exact source/library causes of this archive incompatibility
have not been diagnosed; do not assume changing the MPI rank count fixes it.

The small actual-transfer tests passed on one and two ranks. Maximum errors
in the first, initialization-only check (stress units are arbitrary):

| Interpolator | Constant tensor | Affine tensor |
|---|---:|---:|
| Distance weighted average | 5.3291e-15 | 0.4272543044 |
| Unlimited linear least squares | 1.7764e-15 | 5.3291e-15 |

Published and constrained results agree on this uniform test mesh; one- and
two-rank results agree. The DWA maximum includes physical boundary stencils:
it does **not** establish the cause or magnitude of the interior BP5 bands.
The final test additionally passes initialization plus a one-second step with
nonzero rigid velocity `(0.01,0)` and frozen particle advection, exercising both
RK2 stages. The actual timestep is 1 s; every parent position is unchanged.
The table is unchanged after that step. Comparing all published FE QP values,
not just the maxima, gives zero one-/two-rank difference for both interpolators.
Results: `stress_transfer_verification.json`; full local logs and cell-level
extracts: `/tmp/bp5-stress-transfer-20260923-final-qualified/`.

Earlier setup attempts are retained separately. The first manufactured input
omitted the `particles` postprocessor required to construct the manager; finite
and nonempty checks now prevent false zero-error passes. A custom integrator
name was rejected by ASPECT's property-layout whitelist; the final trials keep
the existing `rk2` name. Neither issue reached checkpoint mechanics. The final
tiny test uses the standard parallel CFL bound; its small rigid velocity lets
the one-second ceiling select the step without overriding that bound.

## What is traced, and what is held fixed

Three separate disposable processes restore independent, hashed copies of the
same checkpoint. Their first pending timestep is multiplied by 1, 1/2, or 1/4
through the existing `post_resume_time_step` hook. Expected constitutive steps:

* 0.0028219241006433027 s;
* 0.0014109620503216513 s;
* 0.00070548102516082567 s.

The runtime clock is authoritative; large absolute timestamps incur ordinary
floating-point rounding. The artificial initialization interval is never used
in these later-step stresses. The ordinary RK2 scheme and checkpoint property
layout remain selected. `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1` makes
only its displacement timestep zero; it does not change the constitutive or
aging timestep. The observer verifies every owned particle ID/position remains
unchanged. Do not set this variable for a physical trajectory.

Each trial solves mechanics once and then performs the ordinary single accepted
history commit, so the actual new particle values can be recorded. These writes
are confined to its disposable branch and never feed another trial. No newly
committed history is used to reevaluate the captured mechanics stress.
Fault state, phase, geometry, incoming stress, background and full normalization
are not changed between branches. The analysis checks the incoming particle
inventory, restored fault fields, and matched native QP phase/localization data.
All ordinary convergence and restart guards remain enabled.

The cell patches are centered at 70.5 and 79.5 km within the two existing
windows, with all vertex-neighbor cells retained. The existing native work/QP
observer still covers both complete 70–71 and 79–80 km windows.

### Tensor and lifecycle definitions

The 2-D symmetric component order is **xx, yy, xy**, with no engineering-shear
factor in the stored xy value. The raw gradient ordering is xx, xy, yx, yy.
The production law has no rotation term:

\[
\tau^\mathrm{mechanics}
 =\beta\,\tau_\mathrm{working\ FE}^{old}
 +2\kappa\,\mathrm{sym}\nabla u
 -2\kappa\,\upsilon S,
\quad\beta=e^{-\Delta tG/\eta},\quad
\kappa=-\eta\operatorname{expm1}(-\Delta tG/\eta).
\]

The inherited diagnostic is the first term; the current update is the sum of
the last two. In this frozen mature fixture, the cohesive history correction
vanishes and `upsilon=chi*V`. `d=-n.tau.n`. The localized-slip contribution to
`d` should vanish to roundoff because `S=sym(s outer n)` and `s.n=0`.

The **particle update** instead starts from that parent's retained stress,
using the accepted FE velocity gradient sampled at the parent position
(`RemotePointEvaluation`, averaged FE traces where applicable), local bulk
material coefficients, and the production localized source. It is not an
update starting from the FE-interpolated old stress. This distinction is
intentional in the existing implementation and is precisely what is audited.

Files, all separate from graphical output:

* `stress_particles_before_rank*.csv`: actual owner values before transfer;
  `stress_particles_after_rank*.csv`: actual values after the single commit.
* `stress_transfer_STEP_rank*.csv`: actual per-cell support proposals with
  component/property/global-DoF indices, then published FE values at Stokes QPs
  after MPI ADD/count. These rows precede the private mechanical constraints.
  Columns: `stage,step,time_s,dt,cell,index,field,component,property,dof,x,y,
  ref_x,ref_y,value`. Index is a support index or a QP index according to stage.
* `normal_qp_STEP_rank*.csv`: actual constrained working FE history, full
  accepted mechanics tensor, direct particle interpolation at the same QP
  (a comparison, not the mechanics input), stress split, gradient, dt/beta/kappa,
  and native physical/work weights. Mechanics rows exist on source-associated
  QPs; the complete neighboring-cell transfer and parent-update traces also
  include non-source points.
* `stress_update_STEP_rank*.csv`: actual pre-publication particle candidates,
  old tensor, sampled strain, localized crack strain, full velocity gradient,
  dt/beta/kappa, parent index/ID, cell, physical/reference coordinates, and the
  actual remote-sampling position. Candidate rows alone do not prove acceptance.
* `normal_summary.csv` is emitted only after the existing genuine convergence
  and lifecycle checks. `stress_cycle_clock.csv` records the pending dt and
  multiplier. The analysis requires an accepted branch before reading candidates.

## Build and prepare on the server

Rebuild ASPECT and **all** loaded BP5 libraries with the same compiler/deal.II/
Boost environment as the checkpoint. The optional response layout changed;
do not mix an old plugin binary with the new executable. The code adds no new
physical parameter and does not alter the restart normalization guard. Retain
the server's previously reviewed restart-only consistency policy; do not loosen
quadrature/tail tolerances or patch an archive to pass a new discrepancy.

Source changes for this task:

* `source/simulator/initial_conditions.cc`: opt-in actual transfer trace.
* `source/particle/integrator/rk_2.cc`: opt-in zero-displacement diagnostic.
* `source/material_model/phase_field_fault.cc` and its header: actual
  pre-commit parent trace and optional constitutive coefficient output.
* `source/reconstructed_fault/surface_system.cc` and its header: accepted
  point gradient/coefficient capture in the existing diagnostic.
* `benchmarks/reconstructed_fault/bp5/normal_stress_diagnostic.cc`: CSV columns.
* BP5 `CMakeLists.txt`, new `stress_cycle_diagnostic.cc`, test PRM and scripts.

Example commands from the source checkout; replace the job/checkpoint paths
with the existing server job's paths. `--input` must be a flat actual input or
resolved parameters, not an include-only overlay.

```sh
cmake --build build-pf-cpdi --target aspect -j4
cmake -S benchmarks/reconstructed_fault/bp5 -B benchmarks/reconstructed_fault/bp5/build -DAspect_DIR="$PWD/build-pf-cpdi"
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_steady_initialization bp5_normal_stress_diagnostic bp5_stress_cycle -j4

python3 benchmarks/reconstructed_fault/bp5/prepare_stress_cycle.py \
  --checkpoint /path/to/original-output/restart/01 \
  --input /path/to/actual-job/original.prm --job /path/to/actual-job \
  --build benchmarks/reconstructed_fault/bp5/build --destination /path/to/new/stress-cycle

python3 benchmarks/reconstructed_fault/bp5/run_stress_cycle.py /path/to/new/stress-cycle \
  --binary build-pf-cpdi/aspect-release --launcher 'ibrun' --cap 3000

python3 benchmarks/reconstructed_fault/bp5/analyze_stress_cycle.py /path/to/new/stress-cycle
```

For ordinary local MPI use `--ranks 4` instead of `--launcher 'ibrun'`.
The runner enables only the qualified explicit B/G, tridiagonal inverse,
single-thread settings and the two new diagnostic switches; it preserves the
caller environment. Source the **qualified run** environment, not old probe
settings. No unsuccessful branch is automatically retried; no subsequent
branch runs after failure. Use a new destination to preserve failed evidence.

Tiny transfer/RK2 qualification:

```sh
python3 benchmarks/reconstructed_fault/bp5/run_stress_transfer_tests.py \
  --binary build-pf-cpdi/aspect-release \
  --library benchmarks/reconstructed_fault/bp5/build/libbp5_stress_cycle.release.so \
  --output /path/to/new/transfer-test-results
```

No interpolation-method change in BP5 is recommended from these manufactured
results alone. Until the server trials finish, the relative roles of history
reconstruction and the timestep-dependent mechanical response remain unresolved.

Build verification: Release ASPECT and the steady-initialization, normal-stress,
and stress-cycle plugins built with `-j4`. The four existing
`test_normal_stress_diagnostic.py` tests passed. Python syntax checks and
`git diff --check` passed. The late-state particle-update capture and three-way
analysis are compiled/prepared, but not end-to-end qualified because restore
failed before those paths ran. No complete ASPECT suite was run.
