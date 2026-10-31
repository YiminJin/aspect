# Research handoff — BP3 review, controlled crust, propagation

Prepared 30 October 2026 after [cleanup](CLEANUP_REPORT.md). This is a research-review handoff, not authorization for another run. No server job or physical timestep was launched during cleanup.

## Source and build identity

Worktree: `/home/ein/repository/aspect-pf-rsf-particle-free`; branch `pf-rsf-particle-free`. Entry HEAD is `f2f730529289eff2c5e8fbba09a224c9bd9a7247`. The useful pending portable-pilot, geometric-clearance and activation-estimator implementation is recorded in source commit **af89a5bccf9c6771ff51b28f54c59b344cdf7cd0**; the following commit contains this cleanup/handoff. Use `git log -2` for both exact identities. The older reviewed BP3 baseline is `b2dfdebe43437d6028b06d4f9f862cc80a00cc71`; it is not a reset target.

The entry had22 tracked modifications and many untracked inputs, sources, results and packages. No incoming unique source was removed. Selected source/tests/fixtures/docs are committed; large raw data, builds, checkpoints, packages and other incoming manual/untracked files remain local and are not implied Git payload. The complete entry status/patches, exact deletion manifests and preservation checks are in the [maintenance audit](../../../benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_30/README.md). Recoverable source manifests identify dirty historical runs more precisely than HEAD alone.

Working binaries are intentionally distinct:

| Role | Retained path | SHA256 prefix |
|---|---|---|
| BP3/native mature qualification | `build-particle-free/native/aspect-release` | `d62541e3fc2a` |
| New cohesive propagation / RSF | `build-particle-free/cohesive-rsf-native/native/aspect-release` | `7277eebc6636` |
| Cohesive boundary/output plugin | `build-particle-free/cohesive-rsf-native/checks-native/libcohesive_rsf_native_checks.release.so` | `7bfbeaa35e7a` |

These existing binaries were not rebuilt for cleanup; embedded Git labels therefore predate the new commits. Their exact hashes and runtime libraries remain unchanged. Objects/PCH were removed; the next source build regenerates those. Historical standalone executables/plugins require indexed re-extraction before use. Shared installations and other worktrees were untouched.

Local qualified stack: GCC12.4.0, OpenMPI5.0.6, deal.II9.6.2, Trilinos14.2, Boost1.84 and p4est2.8.7; see the RSF package `environment.json` and preserved CMake caches/ldd records. SUNDIALS support is required in deal.II. NetCDF/WorldBuilder are enabled in these builds. Keep compiler/MPI/deal.II compatible between ASPECT and plugins; archive portability is not assumed. Analysis needs Python3, NumPy, Matplotlib and Pillow; archive recovery uses zstd.

Future rebuild commands, **not executed by cleanup**, using the existing configuration:

```sh
export DEAL_II_DIR=/opt/dealii/9.6-local
export CXX=/opt/gcc/12.4.0/bin/g++
cmake --build build-particle-free/cohesive-rsf-native/native -j 4
cmake -S benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native \
  -B build-particle-free/cohesive-rsf-native/checks-native \
  -DAspect_DIR="$PWD/build-particle-free/cohesive-rsf-native/native" \
  -DCMAKE_CXX_COMPILER="$CXX"
cmake --build build-particle-free/cohesive-rsf-native/checks-native -j 4
```

For a fresh portable native smoke case use [ACTIVE_WORKFLOW.md](../../../benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/ACTIVE_WORKFLOW.md) and standalone `active.prm`. It prescribes seed, all initial histories and loading; no prototype JSON/cache/checkpoint is a fresh-run prerequisite. Use a new output directory and the current plugin. Do not replace a production parameter file with this different physical experiment.

## BP3: inspect supplied results before further experiments

The selected BP3 route is unlimited domain-free DGQ2, frozen mature fault/profile/mesh, stationary diagnostic particles, Gauss6/source3 and Helmholtz20. It retains the mature RSF/contact interpretation and its own acceptance/history lifecycle. Local optimization, MPI, rejection/restart, collective wall-stop and source/publication checks are qualified only in their stated small-fixture scope. FE/particle controls are comparisons, not physical accuracy references.

Original unreviewed first-event material remains under:

- `benchmarks/reconstructed_fault/particle_free/bp3_dg_first_event/`: both original/replay stdout/stderr, `log.txt`, four server inputs and environment/resource instructions.
- `.../evidence/first_event/`, `.../evidence/server_3610007/`, `.../evidence/server_3613912/` and supplied packets/archives: original fields, accepted clocks, diagnostics, resolved inputs and independent reductions.
- `build-particle-free/bp3-optimization/`, `bp3-work-diagnostic/`, `bp3-work-repair/`: local reference/correction controls and preserved restart evidence. Some full local arrays are lossless `.csv.zst`; they are not original server trajectory files.

Verified observations: the supplied parent label is809 at5172072493.8510532s; replay reaches815 at5172072493.8982706s. The observed failure at816 is an observer work-consistency check. The local-frame correction is independently diagnosed/qualified in [BP3_DG_WORK_OBSERVER_FIX.md](BP3_DG_WORK_OBSERVER_FIX.md). The earlier homogeneous-Maxwell coefficient issue and repair are separate ([report](BP3_DG_HOMOGENEOUS_COEFFICIENT_FIX.md)). Neither repair demonstrates first-event physical accuracy or successful remote completion.

**Current remote completion is unknown.** The local packets have checkpoint labels, not the complete160-rank server restart payload. Preserve the actual server restart directories. The prepared815→818 verification and continuation commands are in [the replay README](../../../benchmarks/reconstructed_fault/particle_free/evidence/server_3613912/README.md); their remote completion has not been verified here. No cleanup removed or reconstructed any remote data. Retain `strict_reference_match_passed=false`, spatial/endpoint temporal findings and raw sided traction errors.

BP3's historical complete source package remains at `build-particle-free/bp3-optimization/package/bp3-dg-stationary-optimized-20261017-r3.tar.gz`. Its later coefficient/observer corrections must follow the documented scope. Build/run locations requested for BP3 remain `$WORK/build/aspect/pf-rsf-dg/build` and `$SCRATCH/bp3_dg/plugin/build`; native launch is `ibrun ./plugin/aspect-release parameters.prm` or the verified branch's native resume input. **Do not launch simply because these commands are recorded.**

## New cohesive-interface propagation and RSF status

This newer default-off formulation is distinct from BP3 and from the earlier evolving bulk-phase laws. Native bulk state retains27 DGQ2 coefficients/cell plus the complete supported native-FE gradient remainder; pressure and constraints are actual mixed ASPECT mechanics. Surface Gauss records own signed jump, irreversible surface damage a, normalized age, cohesive reference, relaxed jump and accumulated absolute slip. The system owns geometry, Q1 jump/memory/rate, identities and admission clocks. Temporary zero-history geometry is replaceable; formed material/history is persistent. One fracture inventory is `Gc integral(a ds)`; geometric phi is not a second fracture charge. The accepted law, pressure coupling, radiation and physical persistence R are unchanged.

[Native pilot](COHESIVE_RSF_NATIVE_REPORT.md): ASPECT advances179 positive steps over .440–.462375s, admits curved material, fully decoheres two newly admitted cells and continues RSF sliding during hold. General nonuniform tensor memory, actual rejection/cutback, single publication and same-rank restart are checked. The original preparation-to-native mapping has explicit stress/storage checks; the active fresh workflow no longer depends on that mapping. [Portable/domain report](COHESIVE_RSF_PORTABLE_REPORT.md) qualifies configuration/restart contracts and full sampling/lifting clearance, not domain independence.

The first instability experiment is a separate fresh16×12m finite-wall case,128×96 bulk cells and96 seed intervals,4MPI ranks. The matched b=a control and acceleration before damage establish RSF feedback. The near-intact raw-log-rate floor failure is preserved, diagnosed and repaired with residual/tangent-informed continuous-history defects ([equations](RSF_EPISODE_ESTIMATOR.md)); no damage exemption, smaller floor or tolerance change.1343 assertions and native lifecycle/default controls pass.

Current accepted continuation: **step748, t=374.58003334330067s, four ranks**, under `build-particle-free/cohesive-rsf-instability/runs/episode-local/output/restart/`. Marker2 selects slot02; complete slot01/03 fallbacks remain. Maximum observed rate .0450089m/s, final .0444673m/s: only shallow decline, **not a completed acceleration episode**. Added damage is cap/cohesive-zone amplification; fully decohered length remains3m and no new persistent geometry forms. No physical/numerical guard failed in the last segment; its wall stop is operational. [Current report/figures](RSF_EPISODE_REPORT.md).

Warm throughput8.11s/step implies roughly5.9local hours per another .1physical seconds. The [complete server handoff](../../../benchmarks/reconstructed_fault/particle_free/rsf_episode_server/README.md), sibling47MB `rsf_episode_server.tar.gz` and checksum retain this exact checkpoint, source, environment, ten inputs and validation. No server job was submitted. Read-only restart/startup/control are required on the actual server before continuation. Native direct launch after those checks is `ibrun ./plugin/aspect-release parameters.prm`, then `resume.prm` in that **separate RSF package**, not in BP3. The1.5h accepted-state wall budget needs scheduler margin for another solve, output and checkpoint;376s and20000steps are bounds, not proof of decay. Fresh doubled bulk/surface spatial and focused temporal inputs are prepared, not executed.

## Limits that remain visible

- Cleanup pruned some raw bulk frames needed to re-render older animations, including39 of45 original native-pilot frames after exact-alias recovery. The renderings and quantitative/surface histories remain; see the explicit [availability loss](CLEANUP_REPORT.md#legacy-animation-data-loss). Do not promise complete raw-animation reproducibility or replay the pilot without authorization. Current RSF surface/event data, accepted histories and original BP3 server results are unaffected.

- Quasi-static bulk plus radiation damping does not resolve seismic waves or validate physical rupture speed.
- Finite walls affect pressure and cohesive response. R is an uncalibrated physical path-persistence hypothesis; numerical lifting width and temporary reserve are separate. Free topology, general bends, AMR/transport and production adoption are not implied.
- The resolved-lifting correction improves the demonstrated response, but point/sided tractions and ahead/bend normal reactions remain unqualified. Contact multiplier accuracy alone does not establish point-stress accuracy.
- Original early spatial, branch and temporal failures remain. The RSF repair is a local continuous-error estimator, not a1% global certificate. Prior mature-age differences reach7–9%; spatial weak-load spread25.2% remains. No finer server event comparison was run.
- In the last RSF segment the absolute physical-work remainder is .0573% of sliding loss but20.8% of new fracture work. The fracture-relative screen still fails. Exact backward-Euler ledger closure is separate from physical energy accuracy; retain cancellation-aware, per-step and accumulated remainders.
- The old symmetric incremental-potential stability result applies only to its qualified homogeneous matched-coefficient bulk law. Do not transfer that certificate to nonsymmetric frictional phase/contact laws.

## Research-review sequence and revised paper plan

1. **BP3.** Inspect the supplied first-event outputs first: actual completion, slip/state/stress evolution, timestep/rejections, pressure convention, matching-frame work and publication. Do not infer success from a package or local control.
2. **Controlled viscoelastic crust experiment.** Use a simple, explicitly bounded geometry/loading to show Maxwell history, friction/contact, stress redistribution and energy accounting with interpretable physical scales. Parameters/boundaries require a declared experiment; this handoff does not select another run.
3. **Fault propagation.** Present the energy-accounted cohesive-interface/native ownership, curved admission and mature sliding milestones with their path hypothesis, boundary sensitivity and numerical limits. Complete the first RSF episode only under a selected continuation task, then separate event timing from progress-aligned growth. No realistic subduction application is planned.

Immediate next work is review and mathematical maps, not a new simulation campaign. Prepare:

- A variable/ownership map distinguishing physical stress from T_DG/native remainder; production mature memory from cohesive reference/relaxed jump; bulk phi from surface a; actual jump/slip rate/age; solved p' from p0 and variational/raw sided pn. Include units and exact source functions.
- A timestep map from accepted n through immutable trial n+1: prescribed displacement increment, Maxwell beta, current-pressure return/admission, native residual/tangent, error estimation, rejection/cutback and once-only commit. Distinguish numerical metadata from physical history and restart serialization.
- An energy map separating bulk/cohesive recoverable storage, single fracture inventory, friction, radiation, Maxwell dissipation, boundary work, BE numerical terms and trapezoidal physical remainder. Plot per-step as well as accumulated error against both sliding loss and newly created fracture work.

Start from [current_design.md](../current_design.md), [specification.tex](../specification.tex), [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [STATUS.md](STATUS.md), [native formulation](COHESIVE_RSF_NATIVE_FORMULATION.md), [candidate admission](COHESIVE_INTERFACE_CANDIDATE_ADMISSION_FORMULATION.md), [front weak admission](COHESIVE_INTERFACE_FRONT_FORMULATION.md), and the indexed [native](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_native/INDEX.md), [BP3](../../../benchmarks/reconstructed_fault/particle_free/evidence/bp3_optimization/README.md) and [RSF](../../../benchmarks/reconstructed_fault/particle_free/evidence/rsf_episode/INDEX.md) evidence. Historical P2/P4/bulk and CPDI reports retain their scope; they are not the newer interface equations. Source governs APIs. Resolve a scientific conflict explicitly rather than silently merging these formulations.
