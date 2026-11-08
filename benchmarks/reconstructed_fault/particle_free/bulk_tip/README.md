# Bounded bulk energetic tip control

This is the user-selected fixed-reference, fixed-direction, single-fault RSF
control. It uses direct physical stress and cohesive tensor histories, bulk
DGQ2 fracture and a complete-section fault-rate integral. It is not the old
surface-damage episode, mature production, or a general turning/branching model.

Read the [current guide](../../../../doc/reconstructed_fault/particle_free/TIP_TREATMENT.md),
[build/configuration](../../../../doc/reconstructed_fault/particle_free/tip_treatment/configuration.md),
and [measured report](../../../../doc/reconstructed_fault/particle_free/TIP_TREATMENT_IMPLEMENTATION_REPORT.md).

- `finite_tip.prm`: all physical parameters and the exact compact initial patch.
- `run.py`: one fresh bounded case, provenance, timeout, optional snapshot copy.
  `--snapshot-id` selects only within the copied checkpoint tree. Avoid extending
  a terminal roundoff timestep when an identical constant-dt comparison is intended.
- `checks.cc`: independent signed RSF/section and native fracture perturbations;
  joint accepted-state publication/rollback checks through the normal simulator.
- `CMakeLists.txt`: independent point tests and matching ASPECT checker plugin.
- `analyze.py`: NumPy reduction; first three supplied cases are plotted in at
  most three PNGs. `--time` makes common-time comparisons; spatial comparisons
  use coarse-cell averages, not invented off-quadrature tensor interpolation.
- `lifecycle.py`: MPI/restart physical-state differences and aggregate run cost.
- `diagnose_interface.py`: read-only decomposition of the existing four-step
  interface data. No new interface evolution.

Every run lives in `build-particle-free/bulk-tip/runs/LABEL/`; labels cannot be
overwritten. Compact reductions go to `particle_free/evidence/bulk_tip/`.
Source commits should include reusable inputs/code/docs/tests, not generated
run trees, binaries, checkpoints or bulk numerical evidence. No commit is made
by these scripts. The finer line-search failure and terminal-restart timestep
artifact are retained alongside passing results. At most four workers/ranks,
one numerical thread per rank, one heavy job at a time. No server submission.
