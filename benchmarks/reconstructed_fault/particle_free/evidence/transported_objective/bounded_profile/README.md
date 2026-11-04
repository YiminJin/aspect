# Native bounded profile transport: evidence index

Primary corrected comparisons are `bpf-*`, full normal support, final time
0.2 s. Earlier `bp-*` runs end at 0.4 s with inherited truncated support and
remain diagnostic controls. Do not combine their accuracy tables. The report
is `doc/reconstructed_fault/particle_free/BOUNDED_PROFILE_TRANSPORT_REPORT.md`.

- `comparison.json` / `comparison.csv`: 28 primary runs, independent profile,
  tensor, normalization, traction, age/slip/memory, energy/flux and cost data.
- `MEASUREMENTS.md`: compact tables; `coupled-profile-comparison.png` and
  `coupled-energy-cost.png`: primary figures.
- `coupled-work.json`: independent pressure-aware surface/bulk work and raw
  versus weak mechanical boundary power, including quadrature sensitivity.
- `full-support-lifecycle.json`, `identity-checks.json`: completed-step
  rejection, repartitioned restart, publication counters and expected refusals.
- `surface-sensitivity.json`: halved initial material-label spacing; no remeshing.
- `prescribed-profile.json`: independent smooth translation mechanism check.
- `truncated-comparison.json`, `truncated-coupled-work.json`, `lifecycle.json`:
  retained earlier partial reductions; raw complete truncated runs remain.
- `entry.json`, `entry.patch`, `pre-support*`, `support-admission*`: preserved
  incoming work, intermediate source and binary identities.
- `resume-notes.md`: failed attempts, duplicate-launch protection and one
  retained roundoff-only terminal interval excluded from physical reductions.
- `final-provenance.json`, `preservation.json`, `source-inputs.tar.xz` and hash
  manifests: exact source/binary/input recovery and protected-data checks.

All native outputs, exact input files, copied runner versions, command hashes
and compact runtime logs remain under
`build-particle-free/transported-objective/stage-c/material-runs/<case>/`.
Core/checker builds and protected executable copies are in the parent Stage C
build directory. Unit evidence is
`benchmarks/reconstructed_fault/refactoring_boundary/evidence/bounded-profile-final-unit.log`.

From the repository root, completed evidence can be reduced without MPI:

```sh
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/reduce_full_support_profile.py
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/summarize_bounded_profile.py
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/record_bounded_profile.py
```

The primary native matrix driver is `run_full_support_profile.py` with groups
`base`, `refinement`, `reference`, `lifecycle`, `identity`. It retains completed
cases and preserves directories. Native invocations and reductions
share the local lock; reference/restart runs use four ranks, other cases two,
with one thread per rank. The two surface-spacing sensitivity runs are recorded
exactly in their command manifests. Existing completed cases need not be rerun.
The historical `run_bounded_profile_matrix.py` requires its saved truncated
source/executable pairing: the final core deliberately rejects those nondefault
options with clipped support. No server runs or production restarts occurred.

Cleanup losses remain: 39 of 45 historical animation frames and earlier
lost temporary build logs. They were not reconstructed or silently removed
from the preservation record.
