# Frozen mature BP3 bottom treatment

Use the paired setting, not a source-only or denominator-only production change:

```text
subsection Postprocess
  subsection BP3
    set Bottom normalization completion file = /absolute/path/to/completion.txt
  end
end
```

The file begins with the number of three-point surface profiles, then one row
`profile_id x y outside_integral` per profile in global profile order. Values
are nonnegative outside-bottom integrals, not desired nodal I_h values. The
material adds them before its unchanged consistent Q1 projection. Origins and
profile count are checked against the actual fault. Keep the file immutable.

This parameter also configures the manager's **bulk-only** straight bottom
continuation. Beyond the bottom tangent plane, inside the physical Box, bulk
assembly and particle Maxwell updates use actual local FE phase and constant
endpoint V/I_h/surface fields. They do not reapply the normal-width cutoff
inside this wedge. Zero phase gives zero source. Already admitted sources,
surface-particle admission, ownership and open connectivity are unchanged.

Qualified scope: the tested 2-D straight 60-degree, fixed-profile mature BP3
fault, prescribed bottom Vp, existing physical boundary conditions. No top
continuation, phase evolution, curved geometry, internal open tip, or ordinary
cohesive-mode qualification is implied. The setter is reattached on restart;
this task did not perform a new restart-equivalence test.

The completion data depend on the exact mesh, phase profile, materials and
fault discretization. `bottom_completion.py` documents their construction from
the saved physical Q1 profile and the same-resolution virtual outside grid.
The existing `bottom-completion-50-local4/completion.txt` is only for the
qualified 42,880-cell/1236-vertex fixture. Do not reuse it after changing the
profile or mesh, or substitute the analytic integral. The pre-existing small
remote/full-column discrepancy remains documented; it was not tuned away.

## Reproduction

`run_bottom_source.py` records exact command, environment and source hashes and
refuses to overwrite a run. Cases:

- `control`: old completed-denominator mechanics, with all-QP observations.
- `continued`: historical width-limited source experiment; retained evidence,
  not the final complete-wedge qualification. Its source hash identifies the
  earlier behavior; today's executable must not be mistaken for that revision.
- `complete-wedge`: the paired parameter above; initialization plus the saved
  two real steps, four ranks, 600-s cap. This is the final verified case.
- `supported --prepare-only`: writes the same supported setting and provenance
  without launching a simulation.

```sh
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_bottom_source.py complete-wedge
MPLCONFIGDIR=/tmp/bottom-source-mpl python3 benchmarks/reconstructed_fault/bp3/analyze_bottom_source.py
```

Those output directories already exist and are preserved. To intentionally
repeat later, choose a distinct output directory; do not delete accepted data.
The final run's `provenance.json` contains its exact environment and launch
command. `ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC` and
`ASPECT_BP3_BOTTOM_SOURCE_CONTINUATION` are **not needed** with the paired
parameter. They remain legacy diagnostic controls only.

`ASPECT_BP3_ALL_SOURCE_QPS` exports inactive positive-phase QPs as well as
active ones in the bottom/control windows. `ASPECT_FAULT_SOURCE_HISTORY_DIAGNOSTIC`
records continued particle Maxwell updates before terminal publication. Both
are observational and may be omitted outside this verification. Ordinary ASPECT
graphical fields retain their meanings.

See `doc/reconstructed_fault/bp3/stage_K5_bottom_source_continuation_report.md`
and `bottom-source-comparison/` for the matched all-QP and surface evidence.
The earlier `bottom-source-width-limited-comparison/` is retained separately.
