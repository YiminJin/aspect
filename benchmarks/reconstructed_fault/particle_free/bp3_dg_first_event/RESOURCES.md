# Resource record for the two bounded server launches

Full-grid resources are **unmeasured**. The inherited 35–100 GiB range is only
an estimate. The local regression resource table is in `VERIFICATION.md` and
`qualification/`; do not scale its wall time or memory as a server guarantee.

Keep the direct `ibrun ./plugin/aspect-release startup.prm` and
`ibrun ./plugin/aspect-release startup-resume.prm` launches. Retain scheduler
stdout/stderr and ASPECT's `output/log.txt` after **each** invocation (the next
invocation can replace parts of the log). Record allocation, nodes, ranks per
node, thread counts, compiler/MPI/dependencies and executable/plugin hashes.
Do not launch another trajectory merely to obtain these measurements.

| Quantity | Collection and interpretation |
|---|---|
| Cells and native DoFs | Actual `Number of active cells` and `Number of degrees of freedom` in the fresh native log; retain its block decomposition. Record surface nodes too. |
| Independent DG history DoFs | 27 per bulk cell (three components times nine DGQ2 modes), **additional** to native solution DoFs. State both counts. |
| Particles and ownership | `particle_summary.csv` is the global count. Rank-prefixed `coupled-checks.csv` records owned fixed particles; sum across ranks and check against the global count. No domains/CPDI or population changes are enabled. |
| Mechanical cost | Native timer sections `Assemble Stokes system*`, `Build Stokes preconditioner`, `Fault: condensed linear solve`, plus accepted-step iterations/cutbacks. Keep the full timer table; nested sections overlap, so do not blindly sum them. |
| Checkpoint write | Native `Create snapshot` elapsed time/call count, including history serialization and terminal metadata. Startup requests every accepted step; retain rotation IDs and accepted labels. |
| Checkpoint read/restore | Native `Resume snapshot` timer includes deserialization, mesh/solution restoration, independent history and post-resume callbacks. It is a read-and-restore cost, not filesystem bandwidth alone. |
| Dense output | New observer timers `BP3 dense stress output` and `BP3 dense coefficient output` cover the large tensor and DG/FE CSV exports (including associated evaluation). Normal cuts, scalar checks and weak loads also contribute to the enclosing `Postprocessing` timer. Record number of emitted snapshots from `field-outputs.csv`. |
| File sizes | Save logical byte counts and allocated disk usage for the complete `restart/` directory, each numbered checkpoint, and each dense snapshot across **all rank files**. Keep snapshot counts: total output size is not size per snapshot. |
| Per-rank peak physical memory | Use the site's rank/process accounting or sampling, covering the whole launch, particularly each checkpoint write and the resume read/restore window. Keep rank and hostname. ASPECT's optional memory-statistics VmPeak is virtual memory, **not** RSS. |
| Per-node peak physical memory | Collect scheduler/cgroup/node high-water memory for every node during both launches, with timestamps where available. Sum concurrent RSS samples on a node if needed; summing per-rank peaks is only an upper estimate because their times can differ. |

On a Slurm allocation, accounting examples (use the site's available equivalent
if its MPI step accounting differs):

```sh
sacct -j "$SLURM_JOB_ID" --units=M \
  --format=JobID,JobName,NNodes,NTasks,Elapsed,MaxRSS,MaxRSSNode,MaxRSSTask,State,ExitCode \
  > evidence/startup-accounting.txt
sstat --allsteps -j "$SLURM_JOB_ID" \
  --format=JobID,NTasks,MaxRSS,MaxRSSNode,MaxRSSTask,AveRSS
```

`MaxRSS` is normally the maximum task value for the step, not the node total or
the sum of ranks. Scheduler sampling may miss a short serialization peak and may
not expose all individual ranks. Request site process/node high-water accounting
or a short-interval monitor when those fields are absent; record sampling
interval and whether the read/write peak could be missed. Preserve accounting
for the actual MPI step, not just the batch shell. Archive the same record for
startup-resume, including memory during initial checkpoint reading. If sufficient
memory data cannot be collected, leave resource qualification incomplete rather
than infer it from rank count.

Examples after each graceful stop (change the evidence prefix for the second
launch; the directory must not replace previous evidence):

```sh
cp output/log.txt evidence/startup-native-log.txt
cp output/statistics evidence/startup-statistics.txt
cp output/accepted_steps.csv evidence/startup-accepted_steps.csv
cp output/particle_summary.csv evidence/startup-particle_summary.csv
cp output/field-outputs.csv evidence/startup-field-outputs.csv
cp output/restart/last_good_checkpoint.txt evidence/startup-last_good_checkpoint.txt
find output/restart -type f -printf '%p %s bytes\n' > evidence/startup-checkpoint-bytes.txt
find output -maxdepth 1 -type f \
  \( -name '*authoritative-stress-*.csv' -o -name '*dg-coefficients-*.csv' \
     -o -name '*physical-fe-*.csv' -o -name '*bulk-normal-cuts-*.csv' \) \
  -printf '%p %s bytes\n' > evidence/startup-dense-bytes.txt
du -sk output/restart output > evidence/startup-allocated-KiB.txt
```

Preserve a complete checkpoint copy before continuing; three rotating native
slots do not preserve all previous accepted states. Use measured worst accepted
solve + terminal dense output + checkpoint duration to choose scheduler margin
beyond the graceful budget. The budget is checked only at accepted states.

## Replicated DG checkpoint cost remains

`source/material_model/phase_field_fault/dg_accepted.cc` constructs an owned
`map<CellId, vector<double>>`, then `Utilities::MPI::all_gather` materializes all
rank maps on **every** rank and merges a global map before packing the archive.
In 2D each cell's numerical record contains 4*(2 coordinates + 1 phase value)
+ 3*9 history coefficients = **39 doubles**, or 312 bytes, before CellId strings,
map/vector allocations, serialized buffers and additional histories/fields.
The gathered maps, merged map and packed vectors have overlapping lifetimes.
Reading also restores globally serialized history before selecting owned cells.
At the estimated 552084 cells, **one numerical copy alone is 172250208 bytes
(164.27 MiB) per rank**. This is a lower payload count, not an RSS estimate; the
actual peak includes multiple copies and ASPECT/particle/solver allocations.
More MPI ranks do not divide this replicated term. Dense output is distributed
but full-grid CSV volume can be several GB per snapshot. No checkpoint layout
redesign or removal of diagnostic particles is part of this revision.
