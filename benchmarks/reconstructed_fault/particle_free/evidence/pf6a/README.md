# PF6a evidence and replay

[Measured report](../../../../../doc/reconstructed_fault/particle_free/PF6A_REPORT.md),
[frozen formulation](../../../../../doc/reconstructed_fault/particle_free/PF6A_FORMULATION.md),
[status](../../../../../doc/reconstructed_fault/particle_free/STATUS.md).
Serial fixed bulk mesh/reference surface, independent unlimited DGQ2 history,
consistent-endpoint/reduced Newton, PF3 ordinary FE fracture, no advection.
Production/limiter defaults and prior evidence are preserved. **Operators and
serial lifecycle pass; spatial stress accuracy remains unresolved.** This is
not full PF5 completion, moving geometry, fault birth or propagation qualification.

- `assessment.json`: independent normal integration/projection, full-3D tensor
  source/stress, fixed-endpoint energy/gradient/Hessian, weak loads, phase KKT,
  native dt/dt2/dt4, bounded normal-mesh and quadrature comparisons, initial
  seed representation and evolving-profile work ledger. Best reduced FD 1.69e-9;
  normal spatial stress difference .42237, temporal dt2/dt4 difference 4.33e-7,
  full evolution quadrature 6/12 difference 2.67e-10 relative L2.
- `contact-verification.json`: independent consistent-Q1 free residual and
  sticking complementarity (max free RMS 8.30e-12 Pa). Raw near-992-Pa early
  sticking residuals are admissible inequality values, not free residual errors.
- `lifecycle.json`: actual native manager cutback versus identical accepted
  schedule (22 full states byte identical), final implementation versus all
  available primary physical exports (19/37 states), checkpoint-parent prefix
  and real automatic restart versus uninterrupted advancement. Persistent/RNG
  and physical/native vector exports match exactly. Native complete manager
  load/resave equality is recorded separately in the archive-control gates;
  raw restart archives retain the known 13-byte default shear-sense difference.
  Final deferred-validation controls add a staged stale-context failure test
  and unchanged accepted coarse/cutback/resumed fields; it fails on the prior core.
- `restart-assessment.json`, `cutback-assessment.json`: independent recurrence,
  phase/stress/load/energy/work checks for final implementation lifecycle runs.
- `fixed-profile-assessment.json`: final same-build frozen FE/DG compatibility,
  retained disabled FE equivalence, source projection, stress poisoning,
  work/adjoint, repeated-trial/publication and nonzero physical elastic slip.
- `regression.json`: old PF5 v1 checkpoint restart against canonical retained
  reference, plus byte identity of all 23 PF4 operator-control CSVs and independent
  operator matrix/load checks. `pf1.json`, `units.log`: 94 mathematical checks,
  628 native assertions/eight cases (embedded ASPECT suite). The separately
  invoked dependency suite passes 2625 assertions/55 cases; its accidental
  invocation is distinguished in `upstream-units.log`. No WENO sweep/tuning.
- `preservation.json`: 10371 incoming file hashes; exactly 18 selected existing
  paths changed, all other 10353 unchanged. WENO source/evidence, old reports,
  manual AGENTS and root CMake are preserved. New PF6a files are additional.
- `source-delta.patch`: exact selected PF6a changes against the reconstructed
  incoming files, whose hashes are verified, rather than assigning earlier
  manual/WENO changes to this task. `comparison.png`: exported measured figure.
- `reference.tar.xz`: native records, expanded inputs, logs and compact CSV/bin
  exports, actual automatic checkpoints, source snapshots, incoming manifest
  and patch, build logs, primary controls and failed preliminary attempts.
  Executables/DSOs remain pinned in `build-particle-free/runs/`, identified by
  hash; they are not duplicated in the archive.
- `provenance.json`: every payload SHA256, native command arrays/inputs/budgets,
  build groups, source/core/observer identities and costs. `verification.json`
  records final gates and replay. The initial 64-normal-cell timeout is excluded
  from the completed matrix and retained with its successful bounded retry.

The physical matrix was measured before checkpoint/outer-publication corrections.
Fresh final-implementation dt/dt2 runs reproduce every available primary physical
state CSV exactly. Later observer versions add persistence/vector exports and
interval-mean loading for variable dt; historical observer DSOs are pinned by
hash, while current source includes those additional diagnostics. Native primary
production `source.patch` snapshots and incoming changes are retained. Historical
records are not relabeled with current source identities. The final deferred-validation rollback regression uses core SHA256
`c994b6b809bde83f456ebdf2d0c573ea2ff8dbf69427add898982a3b573a9693`.
It also verifies unchanged accepted coarse/cutback/restart exports; the pinned
previous core fails the intended rollback gate with the new observer.

All native processes use one rank/thread, initially at most two concurrently,
180 s budgets; the timed-out spatial case was retried alone with a 240 s budget
and finished in 93.87 s. The other matrix runs take 33.03–126.38 s including
startup, solves and exports. Native outer cases explicitly use the existing
`ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1`; zero-end-time frozen observers
do not enter particle advection. The physical seed, loading and material inputs are identical across the
selected comparisons; phase and histories evolve in the coupled runs.

The archive retains missed-freeze startup rejection, initial checkpoint clock
failure, the spatial timeout, wrong observer-name failure and intermediate
controls. The prepublication-only test is not mislabeled as actual manager
cutback coverage. Final publication is staged until the real manager decision.
An early lifecycle comparison assumed old observers had persistence exports;
comparison now explicitly reports the five additional exports while comparing
every historical physical export. Failed analysis invocations are documented
in verification; no numerical tolerance was relaxed.

## Replay

Python, NumPy and SciPy are sufficient; ASPECT is not needed for mathematical
replay. Choose a new destination:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/pf6a/replay.py /tmp/pf6a-replay
```

This checks all payload hashes, independently recomputes the full five-run
assessment including work and refinement comparisons, requires byte-identical
assessment JSON, and repeats native-export lifecycle equality checks. The native
manager canonicalization itself remains a recorded ASPECT test; Python does not
pretend to interpret its archive. `--verify-only` checks extraction/hashes only.
For native reruns use new labels and `run_pf6a.py`; commands and explicit scopes
are in the benchmark README/report. Keep current unsupported-path guards.
