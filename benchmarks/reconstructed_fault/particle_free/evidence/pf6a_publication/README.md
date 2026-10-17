# PF6a publication diagnosis and confirmation evidence

[Scientific report and operand-bound derivation](../../../../../doc/reconstructed_fault/particle_free/PF6A_PUBLICATION_REPORT.md),
[source audit](../../../../../doc/reconstructed_fault/particle_free/SOURCE_AUDIT.md),
[status](../../../../../doc/reconstructed_fault/particle_free/STATUS.md).

The final fresh 2x512, dt=1/128, q12 run completes 72 physical steps in
**879.915 s**, with 72 publications and no healing. Operator/publication checks
pass; **strict unregistered reference comparison is not fully passed** because
the final phase takes the opposite mirror branch. The error flags remain false,
with no alignment or threshold change. See `phase-branch.json` and the report.

The original guard rejects a reproduced first-step sample with error/bound
1.249709. An 80-digit independent replay identifies native point-gradient
cancellation, with a smaller projection/evaluation contribution. The derived
source allowance uses actual velocity operands, cell widths, explicit Q2 basis
bounds and the computed inverse's defect against the analytic Q2 mass. It is
added to the retained history/localization allowance. The guard remains active;
no physical law, timestep, acceptance criterion, limiter or production default
changes.

- `replay.json`: exact offending cell/components/operands, separate evaluation
  paths and independently derived source budget. `reference.tar.xz` includes the
  complete failing-cell CSV and the original failed record/log/source patch.
- `native.json`: every common-time full-tensor/localized stress, phase,
  traction, slip, energy and driving comparison with the selected 1D trajectory;
  source recurrence, reconstructed stress energy and weak-load checks. All nine
  DGQ2 coefficients and native Q2 velocity coefficients remain in the archive.
- `verification.json`: measured gates, lifecycle, frozen FE/DG/default and PF4
  operator regressions, source preservation, hashes and qualification limits.
- The native lifecycle controls test inconsistent incoming xx/yy/xy and staged
  histories, actual outer rejection/retry, immutable accepted histories/RNG,
  discarded staging and exactly one accepted publication. The separate native
  timestep-manager cutback is retained too.
- Original failures, the two capped repaired trajectories (71 accepted steps each),
  and intermediate controls remain in their original directories. A plugin
  compilation overlapping an edit missed two new diagnostic checks; the final
  verified observer was explicitly rebuilt, checked, and source-pinned.
- Archive/manifests pin exact source patches, inputs, commands, executable/DSO
  hashes and compact native fields. Existing native executables and full dense
  raw outputs remain under `build-particle-free/runs/pf6a-publication-*`; binaries
  are not duplicated in the portable archive. This is an evidence replay bundle,
  not a standalone ASPECT build distribution.

Replay without launching a native trajectory (requires NumPy/SciPy/mpmath):

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/evidence/pf6a_publication/replay.py /tmp/pf6a-publication-replay
```

The output directory must not already exist. The replay verifies every archive
member and recomputes the first-step diagnosis and all native/common-time
measurements. Native rebuilding/running uses the repository, source patches and
recorded commands; keep its 900-second confirmation cap.

Fixed-grid, fixed-reference-surface, serial and no-advection restrictions remain.
Prior AMR/MPI, moving geometry, birth, coupled transport, surface transfer and
propagation limitations are unchanged. This does not declare general PF5 complete.
