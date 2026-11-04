# Stage C native evidence

Worktree `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, HEAD
`27178813cebd471b93a7a9308f41215b71b4a20b` plus preserved incoming and Stage C
changes. **Stage C is incomplete and production_ready=false.**
Read [the report](../../../../../../doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_STAGE_C_REPORT.md)
and [assessment.json](assessment.json) for equations, declared scope and results.

- [entry.json](entry.json), [entry-source-sha256.json](entry-source-sha256.json),
  [entry-native.patch](entry-native.patch) and `entry-status.nul` preserve entry
  identities. Old live controls and original protected binaries remain separate.
- [resume.json](resume.json) records the rejected uncut prototype control and
  loss of earlier temporary `/tmp` logs. Those lost logs are not retained evidence.
- [native-protocol.txt](native-protocol.txt) contains the original criteria,
  including the defective exact-AMR-cell-count expectation. The report records
  its replacement by an independent common-point physical-field comparison:
  mesh smoothing retains eight families, giving674 rather than650 cells.
- [assessment.json](assessment.json) contains the complete retained-run inventory,
  selected measurements, per-run input/core/plugin hashes and exact commands.
  [native-qualification.png](native-qualification.png) shows numerical convergence,
  the failed uncut space and the failed physical-energy screen separately.
- [unit-all.log](unit-all.log) records the final repository unit suite.
  Final-source/protected-file identities and source recovery are in
  [final-provenance.json](final-provenance.json).

Run the reducer from the worktree root:

```sh
python3 benchmarks/reconstructed_fault/particle_free/transported_objective/assess_stage_c.py
```

It reads retained output and launches no simulation. The bounded native runner
is `benchmarks/reconstructed_fault/particle_free/transported_objective/run_stage_c_native.py`.
Each run keeps its exact runner copy, input, command/environment/hash record and
log under `build-particle-free/transported-objective/stage-c/runs/<label>/`.
Labels cannot overwrite existing results. A lock serializes simulations; ranks
are restricted to1/2/4, with OMP/deal.II/OpenBLAS threads1 and a process-group
timeout. Restart inputs must be new local Stage C checkpoints, never production.

Selected evidence includes:

| Question | Retained cases |
|---|---|
| Mature residual/history and affine tangent | `native-qualified-fixed` |
| Completed-trial rejection | `native-reject`, `native-cutback-control-time` |
| Restart2->4ranks | `native-uninterrupted-five`, `native-restart-four` |
| Physical T/R AMR and following solve | `native-transfer-field`, final cache fix `native-transfer-cache-final` |
| AMR then restart2->4 | `native-amr-uninterrupted5`, `native-amr-restart4` |
| Legacy comparison | `legacy-pre-c-control`, `legacy-stage-c-control` |
| Uniform and layered transport,3resolutions | `transport-uniform-r3/r4/r5`, `layered-native-r3/r4/r5` |
| Actual cut-history representation | `cut-native-history-0/1/2`, final check `cut-native-final` |
| Final core/DSO verification | `material-owner-native-final`, `material-owner-cut-final`; native T/R matches the prior transfer result exactly |
| Coupled cut-law restart and rejection | `cut-phase-uninterrupted10`, `cut-phase-restart4`, `cut-rejected-phase`, `cut-reject-control` |
| Failed physical phase work screen | `cut-native-phase8`, `cut-phase16`, `cut-phase32` |
| Failed tip admission | `cut-tip-airy`, `tip-airy-legacy`, `tip-airy-half-dt` |
| Conversion/import rejection | `reject-new-dg-to-legacy`, `reject-legacy-to-new-dg`, `cut-reject-old-objective-checkpoint`, `reject-old-cohesive-import` |

An exit code0 is process completion, not proof that the representation/physical
screen passed. In particular, the uncut increment-relative errors remain
17.6/11.4/8.1%, and all three physical phase-work screens fail the original1%
fracture-relative target. The early `cohesive-cut-capacity-0/1/2` attempts used
a stale checker after a failed build; they are retained and excluded from the
selected capacity evidence. Graded-mesh transport attempts are also retained
but do not establish the uniform spatial order. Negative conversion tests
are expected to exit1; the mature monitor identity guard fires before the
underlying DG schema guard, which these cases do not isolate.

No transport-operator, material-geometry-container or fixed-reaction pass
establishes complete native material motion. Bulk/profile/curve/surface
publication, heterogeneous coupled reaction, moving resampling/boundary/AMR,
deformed cohesive shape/pressure work and successful persistent extension
remain separate requirements. Original BP3 and RSF checkpoints were not
continued or converted. The cleanup loss of39/45 older animation frames remains.
