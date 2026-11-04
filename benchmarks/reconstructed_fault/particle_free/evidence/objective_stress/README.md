# BP3 review and objective-stress evidence

Entry HEAD `27178813cebd471b93a7a9308f41215b71b4a20b`, required current worktree
and branch, initially no tracked edits. This evidence is new bounded review and
local verification; it does not replace any propagation or production evidence.

Reports: [BP3 review](../../../../../doc/reconstructed_fault/particle_free/BP3_REVIEW.md),
[maps](../../../../../doc/reconstructed_fault/particle_free/IMPLEMENTATION_MAPS.md),
[spin report](../../../../../doc/reconstructed_fault/particle_free/OBJECTIVE_STRESS_REPORT.md).

| Evidence | Meaning |
|---|---|
| `entry.json`, `entry-status.txt` | Actual root/branch/revision, incoming work, original executable/plugin hashes and ldd dependencies. Copies under `build-particle-free/objective-stress/baseline/` preserve all three original binaries. They are not asserted to match HEAD. |
| `setup-integration.json` | Attached setup documents matched local versions; exact supplement was already present once. No duplicate append or blind overwrite. |
| `bp3-input-manifest.json`, `bp3-review.json`, `bp3-profile-history.csv`, `bp3-review.png` | Hashes and independent reductions of new `output_start/` and `output_resumed/`. Accepted4237 is observed; no completed event or final peak is inferred. |
| `native-results.json`, `native-output-manifest.json` | Full3D physical tensor and surface comparisons, native recurrence/energy/residual checks, exact retry/restart and measured 2→4 differences. |
| `verification.json`, `source-manifest.json`, `source-delta.patch` | Final binary identity, configured stacks/commands/environment and source hashes. The patch contains tracked deltas; new source files listed by the manifest remain at their repository paths. |
| `preservation.json` | All328 cleanup-critical files unchanged, no missing file; protected executable/plugin copies unchanged; separate RSF marker remains2. The cleanup report's prior data losses remain, without recreating deleted campaigns. |

Final scoped native build: GCC12.4.0, OpenMPI5.0.6, deal.II9.6.2,
Trilinos14.2, Voro++0.4.6, optimized ASPECT. Final unit suite passes183073
assertions in148 cases. Earlier no-Voro cohesive-stack tests pass182986 in143;
that build is not substituted for the Voro-dependent particle/CPDI control.
Native tests use one thread per rank and no more than four ranks. Builds use
at most four workers.

Reproduce the reductions without launching ASPECT:

```sh
python3 benchmarks/reconstructed_fault/particle_free/review_bp3_results.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/objective_stress/analyze_native.py
```

Exact native inputs, commands, core/plugin hashes and elapsed times are under
`build-particle-free/objective-stress/runs/<case>/command.json`; each physical
case has at most four positive steps. Cases:

- `legacy-retained`, `legacy-current`: protected older binary versus the current
  default control. Tensor relative L2 difference1.99e−14; not byte equality or
  proof of matching source revisions.
- `spin-on-qualified`: three-step pressure/residual audit; `spin-four` is the
  uninterrupted restart control. No current-spin Newton tangent is claimed;
  the new source uses frozen accepted spin.
- `spin-restart`, `spin-repartition`: copied local step3 checkpoint advances to4
  on2/4 ranks. Same-rank physical fields equal exactly; 2→4 tensor relative
  L2 difference9.93e−15. This does not qualify transport or AMR.
- `spin-rejection`, `spin-retry-control`: completed step2 trial rejected before
  publication; accepted schedule200000/100000/191000 s reproduces exactly.
- `spin-final-admission`: final constant-material guard-only source change;
  physical fields match `spin-on-qualified` exactly. Final full-unit log is
  `build-particle-free/objective-stress/final-units-all.log`.
- `spin-from-legacy-negative`, `unsupported-{dg,cohesive,transport,coefficients}`:
  expected failures with useful law/history/kinematic admission diagnostics.
  The BP3 configuration fingerprint rejects the law-changing restart before
  the core registry-marker check. No production checkpoint is used.

Retain failed attempts `spin-on`, `spin-on-local`, `spin-on-v2`, `spin-on-v3`:
MPI sandbox sockets, missing obsolete observer declaration, missing Voro++,
and prohibited domain-free particle initialization. The initial matrix assertion
expected a later restart guard and failed after the earlier intended fingerprint
rejection; original and continued logs remain. These are not scientific failures
hidden by changed tolerances. Restart summary files are append fragments without
a fresh header; the reader supplies their declared parent schema without editing
the bytes.

The opt-in option qualifies a stationary local spin/source reaction operator
with matched constant coefficients, retaining full3D tensor convention and
existing Maxwell coefficients. It is not the transported material PDE when
v.grad(tau) is nonzero, finite-strain hyperelasticity, or production objective DG.
The separate native reconstructed branch-energy check does not replace a full
fault/external-work energy balance. No server, long event, crust or propagation
campaign was launched, and no production input/checkpoint was converted.
