# K5 normal-stress diagnosis: source trace and bounded replay instrumentation

## Latest: mechanical significance of the history representation

The [frozen comparison and single FE-history solve](stage_K5_history_mechanics_result.md)
are complete. History sampling changes the frozen 39.9-km bound margin by
20%, but the re-equilibrated node remains at Vmin with a stronger reaction.
The neighbouring rate changes by 4.15%; the junction pressure feature remains.
Both stress representations are retained in diagnostics, and the complete
rollback check passes for this new disposable solve. No history method has
been promoted and no additional trajectory is authorized by these results.

## Consolidated current report

The last three tasks and their current combined conclusion are now collected
in [the consolidated normal-stress report](stage_K5_normal_stress_consolidated_report.md).
Use that report for the current decision, numerical tables, verification
limitations and evidence index. The sections below preserve chronological
notes, including recommendations that the later tasks have already completed.

## Frozen history/load follow-up completed

The approved four-rank extraction stopped before Newton in 26.4 s. The
40-km concentration already exists in retained particle tau11; transfer
attenuates its normal-contraction RMS by 2.9%, and working constraints do
not modify the local samples. Independent integration reproduces the actual
frozen weak load to roundoff. Local particle/FE differences remain important
to the tensile sign, especially at the bottom; no representation correction
or new mechanical solve is justified automatically. See
[the frozen history/load report](stage_K5_frozen_history_load_result.md).

## Approved junction-location probe completed

The single 35-km noncommitting probe has now been run. Its mechanical solve
converged, but a new rollback-verifier vector-layout error prevented completion
of the in-memory checks. No histories were published. Converged exports show
that roughly 80% of the old 40-km dipole remains; the new 35-km feature is
smaller and the bottom-tip concentration is nearly unchanged. See
[the junction diagnostic report](stage_K5_junction_location_result.md) for
the numerical comparison, verification limitation and next frozen-data audit.

## Completed uncapped fresh run (supersedes pending status below)

After explicit authorization to remove the 120-s cap, the fresh four-rank
run completed through accepted step 12 in 907.34 s. The corrected Theta audit,
all 105 fresh-linear checks, and the unchanged convergence criteria passed.
The diagnostic export gap was closed with a noncommitting accepted-state
evaluation before history publication. Full results and the current decision
are in [the completed local report](stage_K5_normal_stress_local4_result.md).

The global tensile minimum is at the prescribed bottom tip, not 40 km.
However, step 12 also develops a −0.660 MPa mixed-support pocket at the 40-km
junction, contributing to its now lower-active 39.9-km RSF node. Deep slip
transfer is present. No changed-cutoff experiment, model correction, or event
continuation was run. The following sections retain the earlier attempt
history; their pending/replay recommendations are superseded by that report.

## Latest update: authorized fresh four-rank run

Following the user's instruction to start from the beginning, a new four-rank
Release run was launched with the actual server physical input, no restart,
the corrected Theta audit and opt-in diagnostics. Only local library/output
paths, fresh-start selection and termination at step 12 were overridden.
Fixture: `benchmarks/reconstructed_fault/bp3/normal_stress_fresh_local4.prm`.

The run reached the unchanged **120-s wall cap** (status 124, elapsed
120.307 s). Initialization and real step 1 completed with genuine convergence
and passed the corrected Theta assertion. Step 1 was at
2,618,120.216489729 s, with max V=1.0201553409641049e-9 m/s and raw normal-stress
range [49.9024494, 50.0493557] MPa. Timestep zero retained zero particle Maxwell
stress and reported zero Theta-reference error.

Step 2 was interrupted after nonlinear iteration 5, with relative
bulk/surface residuals 2.3970e-11 and 1.81644e-2. It was **not accepted**.
There was no logged numerical failure before timeout. Steps 11–12 and the
late-time tensile anomaly were not reached; no retry or continuation was made.

Evidence: `benchmarks/reconstructed_fault/bp3/normal-stress-fresh-local4/`
contains `run.log`, `execution.json`, source/binary `provenance.json`, accepted
records and locally generated checkpoints. Its last-good checkpoint is
`restart/02` (after step 1). The original server checkpoint remains untouched.

The step-1 offline analysis exposed a diagnostic coverage gap: raw surface
samples, weak moments and projected CSVs exist, but `bulk_slip_transfer_*`
does not. That exporter is currently attached to the standalone residual
action exercised by the focused tests, not the conventional assembly path
used in this production solve. The analyzer stopped explicitly with
FileNotFoundError rather than substituting data. Its log is preserved as
`analysis-step1.log`. This instrumentation gap must be addressed before a
complete transfer report; it is not evidence of missing physical deep slip.
No instrumentation was changed during this run.

## Update: user-authorized four-rank local attempt

The requested local attempt was executed once on 2026-09-14 with four MPI
processes, a copied `restart/03`, the corrected checker/diagnostic binary,
unchanged physical/solver parameters, end step 12, and a 120-s wall cap.
It exited with status **1 after 1.604 s**, during checkpoint deserialization:

```
Cannot seem to deserialize the data previously stored!
Some part of the machinery generated an exception that says:
class version N6aspect11Postprocess9ParticlesILi2EEE
```

The named type is `aspect::Postprocess::Particles<2>`. All four ranks reported
the failure. No timestep, mechanical solve, Theta check or stress diagnostic
evaluation was reached; no retry or archive conversion was attempted.

Evidence is in `benchmarks/reconstructed_fault/bp3/normal-stress-step12-local4/`:
`replay.log`, `local_execution.json`, the exact `diagnostic.prm`, copied
checkpoint, and source/binary `provenance.json`. Original checkpoint hashes
were reverified unchanged afterward. The preparation manifest's server
requirements are historical preparation metadata; `local_execution.json`
records the actual authorized four-rank attempt.

The observed failure is a **binary-archive restoration failure**, not evidence
of nonlinear nonconvergence or faulty Theta evolution. Local deal.II is 9.6.2
with 32-bit indices and bundled Boost 1.84; the server reports deal.II 9.6.0
with 64-bit indices and Boost 1.85. The exception alone does not isolate which
source/dependency/layout difference caused it. Next action: restore with a
matching build, rather than suppressing the version check or discarding
serialized postprocessor/history state. The scientific diagnosis remains pending.

## Decision

No source-level omission of prescribed deep slip was found. The same physical
Q1 V, including prescribed nodes, enters bulk crack deformation and particle
Maxwell history. This does **not** yet establish the location or physical
extent of the negative samples. Extending the RSF-solved section is **not
justified by the present evidence**.

The next action is the single instrumented step-12 replay on the compatible
server, from the untouched step-11 `restart/03`. No new BP3 solve was run
locally, no 35-km cutoff experiment was prepared, and no model change is
proposed. This follows the latest `stage_K5_normal_stress_codex_instructions.md`.

## Prescribed-slip transfer trace

| Responsibility | Relevant source/function | Finding |
|---|---|---|
| Set deep rates | `bp3/bp3.cc`, `BP3Benchmark::prepare` | Installs Vp on vertices at xd >= Wf−1e-7, including after restart. |
| Interpolate actual V | `source/reconstructed_fault/manager.cc`, `interpolate_slip_rate` | Uses both endpoint values; exact endpoint reads at tips. No free-row mask. |
| Bulk deformation/residual | `source/simulator/assemblers/reconstructed_fault_stokes.cc`, `evaluate_slip_dependent_bulk_residual`, `execute` | Every associated bulk QP uses chi*V + history correction. Prescribed rates are included in the physical residual/RHS. |
| Particle stress update | `source/material_model/phase_field_fault.cc`, `commit_reconstructed_fault_mechanical_history` | Every admitted parent uses interpolated V; subtracts crack strain from total FE strain before Maxwell update. No prescribed/free exclusion. |
| Newton restrictions | `source/simulator/solver.cc`, coupled active-set solve | Prescribed increments are zero; this does not remove the prescribed base rate or its bulk load. |

For the frozen BP3 profile, the implemented history localization correction
is zero when current/previous phi and Ih agree. The crack rate is then
chi*V, including chi*Vp in deep support—not chi*(V−Vp). Bulk Maxwell stress
and pressure are perturbations, while background traction is a separate
surface input. A small stress perturbation does not imply missing total
crack deformation.

At the saved step-11 last free node (39.9 km), V=6.85828e-11 m/s, versus
1e-9 m/s at the first prescribed node (40 km). Q1 remains continuous; the
100-m segment carries a steep gradient, not a discontinuous value jump.
Saved shear variation grows near this junction (see
`theta_junction_audit/saved_junction_history.csv`). This is correlation,
not a localized normal-stress or causal measurement. Bottom/profile-support
effects and transverse variation remain competing explanations.

## What stress enters friction

`PhaseFieldFault::evaluate_reconstructed_fault_point` computes the parent's
bulk pressure/gradient and retained particle-history response with the
domain quadrature point's surface Q1 data. Compression is positive:

\[
 \sigma_n=\sigma_{bg}+\Delta p-\Delta\tau:N,\qquad
 F=q-C-\mu(V,\Theta_{old})\sigma_n-\eta_d V.
\]

There is no normal-stress clipping or projection before multiplication by mu.
The weak residual integrates w*N_i*F; K_V includes the signed
sigma_n*dmu/dV term. Consequently tensile samples can change free equations
and their Jacobian even if projected station traction is positive. Neither
the sign nor magnitude of that effect can be inferred from station extrema.

## New opt-in diagnostics

Enable `ASPECT_FAULT_STRESS_SAMPLE_DIAGNOSTIC=1` in the rebuilt core and plugin.
With it unset, there is no diagnostic sampling or output. Normal solver and
constitutive arithmetic are unchanged.

- `stress_samples_STEP_rankR.csv`: 20 lowest/highest raw samples **per rank
  per unprescribed/prescribed/mixed Q1 support class**. Includes stable parent
  ID, parent cell, domain-Q index, segment/xi, parent and surface coordinates,
  signed transverse distance, integration weight/full parent domain volume,
  unprescribed shape weight, parent support spanning flags, V, delta_p,
  delta_tau:N, background/total normal stress, q, mu, mu*sigma, phi and chi.
  Raw extrema are bounded subsets; their weights must not be summed to infer
  total tensile support.
- `stress_weak_moments_STEP_rankR.csv`: moments over **all** constitutive
  samples, not just extrema. Includes total/tensile weights and signed/absolute
  friction contributions at every node. MPI ranks own disjoint parent
  contributions. Summing these tables quantifies tensile influence on free
  equations, including mixed domains.
- `bulk_slip_transfer_STEP_rankR.csv`: actual bulk-QP integrals per surface
  segment of chi, V, instantaneous/history/total crack rate and the stress
  coefficient entering bulk assembly. Includes endpoint V and physical
  coordinates, covering the junction and deep tip without exporting every QP.
- `stress_projected_STEP.csv`: full consistent Q1 normal traction, q, V,
  accumulated slip, prescribed/lower-active status, mass matrix, and explicitly
  **committed** Theta. Computed from the accepted pre-publication weak loads,
  not by advancing Maxwell again. It is written before the Theta audit.

Raw/moment files are replaced by each surface **linearization**, never by a
surface trial-only evaluation. They are accepted evidence only after genuine
convergence; the analyzer checks their V against accepted nodal V. It solves
the same consistent mass system for the pressure/normal decomposition and
separates unprescribed nodes from lower-active and genuinely free equations.
Ordinary ASPECT graphical postprocessor output is unmodified.

## Verification and execution status

- Reused the passing cancellation-safe Theta checker and its unchanged 1e-12
  assertion/regressions; no production history update changes.
- Core and BP3 Release plugin built with -j4.
- Existing condensed surface/B/G action and finite-difference fixture passed
  with diagnostics enabled on one and two ranks (about 6 and 5 s reported
  wall time). This is **not** a new BP3 trajectory test.
- Sample decomposition error was zero. Instantaneous+history versus total
  integral error was at most 5.90e-17. Summed surface measure was
  0.3776041666666689 versus 0.3776041666666688 on one/two ranks.
- These fixtures had no tensile samples: the BP3 tensile classification and
  influence numbers remain unmeasured. The offline reducer is syntax-checked
  but has not yet consumed a complete BP3 diagnostic replay.
- The initial two-rank launch was blocked by sandbox MPI sockets; the same
  focused test passed after permission to launch MPI outside the sandbox.
- `restart/03` hashes still pass. Build/test logs and source/binary hashes:
  `benchmarks/reconstructed_fault/bp3/normal_stress_diagnostic/`.

The newly supplied `BP3.e3498041` confirms the old assertion at bp3.cc:542,
with no failing values. It records Intel 24.0, Intel MPI 21.11, Boost 1.85,
Trilinos 15.0 and p4est 2.8.5; supplied CMakeCache confirms the server paths.
The original log records 64-bit indices, whereas installed local deal.II
builds use 32-bit indices, including a different particle-index type. The
new files do not provide a compatible executable or establish cross-width
checkpoint compatibility. No archive conversion or unchecked replay was tried.

## Exact bounded server preparation

Rebuild the modified core and plugin against the original 64-bit-index server
environment first. From the current source root, prepare a separate output:

```sh
python3 benchmarks/reconstructed_fault/bp3/prepare_normal_stress_replay.py \
  --output /scratch/11463/yiminjin/bp3/normal-stress-step12 \
  --binary /scratch/11463/yiminjin/bp3/plugin/aspect-release \
  --plugin /scratch/11463/yiminjin/bp3/plugin/libbp3.release.so
```

This **does not launch ASPECT**. It copies and hash-verifies only restart/03,
inherits the actual server `original.prm`, and overrides only output/plugin
paths, resume, and termination at **end step 12**. It prints the exact `ibrun`
command for the original 64-rank allocation with a 120-s cap and records
source/binary/checkpoint hashes. No automatic retry or first-event continuation.
Do not use the local 32-bit executable at those paths. Leave experimental
preconditioner, reference-comparison and profiling callbacks unset.

After a fully accepted replay:

```sh
python3 benchmarks/reconstructed_fault/bp3/analyze_normal_stress.py \
  /scratch/11463/yiminjin/bp3/normal-stress-step12 --step 12 \
  --output /scratch/11463/yiminjin/bp3/normal-stress-step12-analysis
```

**Missing:** exact original step-11 sample locations, current step-12 extrema,
free-equation tensile influence and anomaly width. Step-11 checkpoint history
is already committed; using it as the old input would evaluate a different
state. The single replay instead captures correct step-12 pre-publication
samples using retained step-11 history. Until those measurements arrive,
neither a 40-km origin nor a bottom-boundary origin is established.
