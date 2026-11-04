# Open-boundary/profile isolation evidence

[Measured report](../../../../../../doc/reconstructed_fault/particle_free/BOUNDARY_PROFILE_ISOLATION_REPORT.md)
and [pre-run protocol](../../../transported_objective/BOUNDARY_PROFILE_PROTOCOL.md).
`entry.json` verifies the actual worktree/branch/HEAD,928 native source files and
core/four existing libraries against the previous closeout. `final-provenance.json`
verifies their preservation and identifies the new checker, five bounded
native runs, inputs, source recovery and all postprocessing data.

- `boundary-transport.json/png`:30 native scratch tests across three meshes,
  both directions and selected timesteps. Correct versus zero inflow, compact
  tensor, actual versus continued references, energy flux, upwind loss and signed
  integration remainders. Native core and accepted states are unchanged.
- `profile-projection.json/png`: compact Q1 representation/repeated-remapping
  decomposition, signed cross term, finite/padded-domain variance controls,
  and replay of actual native Q1 profile updates. Lookup-tolerance effects are
  separately reported. The figure compares repeated and one-shot errors against
  the same analytic endpoint in its top-right panel.
- `tensor-defect.json`: retained/new coupled errors partitioned by inflow,
  interior/upstream strip and deeper core, plus independent Gauss8 analytic
  localization projections and signed cross terms. The remaining transport/
  rotation/reference mismatch is not a new constitutive or energy term.
- `source-inputs.tar.xz` and its manifest: exact diagnostic sources/documents and
  all five new run inputs. The unchanged native source has its existing verified
  recovery archive in `../stage_c_coupled/`.
- `input-fields-sha256.json`, `preservation.json`: compact provenance for existing
  and new exports and unchanged original protected data. Earlier failed physical
  tests and cleanup losses are retained.

From the worktree root, reproduce the reductions (no native jobs launched) with:

```
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/assess_boundary_transport.py
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/isolate_profile_projection.py
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/isolate_tensor_defect.py
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/record_boundary_profile.py
```

Native run commands, rank counts, core/plugin/input hashes and times are in the
final provenance. Run logs, new local checkpoints and original exports remain
in `build-particle-free/transported-objective/stage-c/material-runs/`.
The separate new checker build is recorded in `boundary-profile-build-2.log`;
the first attempt's missing-target log is retained. None of these results is
a production BP3 or full coupled energy qualification.
