# PF5 accepted-history and native persistence evidence

See [PF5 progress report](../../../../../doc/reconstructed_fault/particle_free/PF5_REPORT.md)
and [status](../../../../../doc/reconstructed_fault/particle_free/STATUS.md).
This completes only the selected serial, fixed-mesh/fixed-geometry prescribed
subset. Unlimited upwind DGQ2 remains experimental. Limiter settings and
production defaults are unchanged. No AMR/MPI, surface/phase transfer, moving
geometry, birth, coupled propagation or PF6 is performed.

- `assessment.json`: independent Gauss5 upwind matrix, native-Q2 preload,
  exact semidiscrete affine ODE, independent Strang/SSPRK3 composition, full-3D
  tensor/physical-stress and weak-load errors, analytic Z/slip/q, observed
  second-order accuracy, counting-error negative controls, and restart equality.
- `reference.tar.xz`: four 8/16/32/64-step runs and the separate 16-step restart,
  complete diagnostic histories/fields and manager/RNG archives, the actual
  midpoint native filesystem checkpoint, expanded inputs/records/logs, source
  snapshots and preserved preliminary failures. Also includes three disabled
  FE/DG controls, the rebuilt PF4 transport/operator control, four negative
  scope/provenance controls and retained PF4 operator CSVs for bytewise replay.
  The archive is about 2.7 MB; pinned executables/DSOs and all raw runs stay in
  `build-particle-free/runs/` and are identified by hash.
- `provenance.json`: exact core/observer/source identities, structured native
  command arrays, process costs, build commands and hashes for all 468 payload
  files. All selected new controls share core SHA256
  `21cf5d8589be2194a7a02e40e518f3d2f3cabe3528b6a22490e53be7063adfdd`
  and the same 880 production source hashes. The five main PF5 runs share one
  observer. The retained incoming-core FE comparison is explicitly distinguished:
  its runner records launch-time source hashes, not that older binary's build
  sources. Its original build identity belongs to preserved PF4 evidence.
- `verification.json`: 280 fresh and 18 restart gates, four negative controls,
  628 native assertions/8 cases, 94 PF1 checks, exact mathematical replay and
  preservation checks. All 1299 incoming files outside the 13 selected edit
  paths retain their hashes, including manual AGENTS, original phase reports
  and every prior evidence file. The new source/benchmark/report files are
  additional, not replacements.
- `disabled-regression.json`: pre-PF5 incoming-core versus rebuilt disabled FE
  fields are exactly equal; same-build FE/DG values remain within roundoff.
  Physical inputs match, with the new selector explicitly default false.
- `fixed-profile-assessment.json`: same-build FE/DG source, stress-poison,
  adjoint/work, publication/repetition, physical elastic-slip and full-tensor
  field/load checks. Its baseline argument is the rebuilt FE control; the
  separate incoming-core comparison is in `disabled-regression.json`.
- `operator-verification.json`: independent weak material-operator checks and
  byte identity of all 23 PF4 operator-control CSVs.
- `pf1.json`, `incoming.json`: complete PF1 checks and pre-edit preservation
  manifest. The incoming tracked patch and all build/test logs are in the archive.

Native candidates compose exact Maxwell/source half steps around pure transport;
surface Z is updated once. The existing candidate validator/publisher owns the
terminal writes. Native checkpoint serialization is extended only for the
selected material-owned DG payload; mesh/solution, fault manager, particles and
RNG retain their existing infrastructure. The observer prescribes accepted
clocks and bypasses coupled physical stepping on restart. This is actual native
publication and filesystem persistence, not a production coupled integrator.

Unused provenance registry values intentionally contain NaNs. Comparison uses
exact bytes/matching NaN locations rather than changing those values. Native
manager load makes an omitted default shear sense explicitly +1, adding 13 raw
archive bytes. Both raw archives are retained; complete archives after the same
native load operation agree bitwise, along with all persistent value/RNG
archives, stress, loads, native vectors and clocks.

Every native run is capped at 180 seconds, one rank/thread, at most two
concurrent. Four main fresh runs take 4.777–4.878 s and restart 3.975 s; these
include startup, repeated/rejected trials and checkpoint work. They are not
isolated transport throughput timings. Failed initial NaN-equality, missing
surface-temperature-cache restart, formatter compile and observer-name attempts
are preserved and are not included as successful measurements.

## Mathematical replay without ASPECT

Requires Python, NumPy and SciPy. From the repository root, choose an empty
destination (the example assumes `/tmp/pf5-replay` is new):

```sh
mkdir /tmp/pf5-replay
tar -xJf benchmarks/reconstructed_fault/particle_free/evidence/pf5/reference.tar.xz -C /tmp/pf5-replay
OPENBLAS_NUM_THREADS=1 python3 /tmp/pf5-replay/source/assess_pf5.py /tmp/pf5-replay/pf5-verified-8 /tmp/pf5-replay/pf5-verified-16 /tmp/pf5-replay/pf5-verified-32 /tmp/pf5-replay/pf5-verified-64 --restart-parent /tmp/pf5-replay/pf5-verified-16 --restart /tmp/pf5-replay/pf5-verified-restart-16 --output /tmp/pf5-replay/assessment.json
cmp benchmarks/reconstructed_fault/particle_free/evidence/pf5/assessment.json /tmp/pf5-replay/assessment.json
```

This replay was executed from the final archive: all 468 payload hashes verify
and its assessment is byte-identical. The reference assembles its own operator
and source, uses the exact augmented matrix exponential, and checks each time
composition. The native matrix export is only a verification target. Physical
reconstruction and weak loads use actual native quadrature/association metadata,
not a different localization law. T uses exact polynomial mass integration;
nonpolynomial physical-stress/load comparisons are explicitly tied to the shared
native mechanical rule.

For native reruns, use `provenance.json` build/command arrays and unique new
labels with `run_fixed_profile.py`. A fresh run takes `pf5-N.prm` and the rebuilt
`libpf5_checks.release.so`, with postprocessor `PF5 accepted history`. Resume
takes `pf5-restart-N.prm` and `--restart-from <fresh>/run/restart`; the runner
copies that snapshot before launch and refuses to overwrite a run directory.
The native dependency stack and this working tree's preserved PF0–PF4 source
are required; the archive is not a standalone ASPECT distribution.
