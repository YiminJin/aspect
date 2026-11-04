# Remaining Stage B evidence — 2 November 2026

Scope: particle-free Cartesian history-space comparison with prescribed material
shear/translation, transported/deforming profile and vertical layered viscosity.
Twenty numerical criteria pass. Eight consolidation/profile/selection checks
also pass. No native integration, production checkpoint conversion or server job.

- `results.json`, `run.log`: complete declared numerical checks, physical scales,
  independent references, omissions, costs and source hashes.
- `space.csv`: Eulerian Q2 mesh8/16/32, full3D/zz tensor, normal/shear and storage
  errors, independent best projection errors.
- `time.csv`: finite-rotation Lie80/160/320 on exact material characteristics.
- `quadrature.csv`: source/reaction4/6/8 versus12 at fixed mesh/time steps.
- `profiles.csv`, `comparison.png/.pdf`: one fixed spatial cut and convergence;
  profile panel shows localization, including the deliberately wrong Eulerian max.
- `space_screen.json/.csv`: best P1 versus Q2 residual capacity; no evolved P1
  scheme is claimed. Its failed coarse target is retained as a rejected candidate.
- `surface_screen.json`: Q1 scalar snapshot16/32/64; include surface-information
  cost. It does not qualify native scalar evolution, derivative or ALE transfer.
- `surface-protocol-original.txt`: preserved wording error (“ratios” rather than
  log2 convergence “orders”); initial checker used the mathematically correct
  order test. No failed run or numerical error bound was changed to obtain a pass.
- `qualification.json/.log`: selected complete DGQ2 R beside T/s, candidate
  coefficient costs, independent material-profile pullback and expected bad-max
  damage trail. Direct physical stress remains an independent control.
- `entry.json`, `entry-status.nul`, `verification.json`: exact worktree identity,
  retained source/binary/checkpoint/server/cleanup and prior B1 preservation,
  installed Python/NumPy identities and all new deliverable hashes.

Reproduce sequentially from repository root (Python3, NumPy, Matplotlib):

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/stage_b_moving.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/screen_moving_spaces.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/screen_surface_memory.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/qualify_stage_b.py
```

Main numerical study about33s, one process/BLAS thread; no field-frame sequence.
The benchmark source/protocol lives under
[transported_objective](../../../transported_objective/STAGE_B_MOVING_PROTOCOL.md).
[Completion report](../../../../../../doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_STAGE_B_MOVING_REPORT.md) ·
[Specification](../../../../../../doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_SPEC.md) ·
[Task state](../../../../../../doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_TASK_STATE.md).

Selection is an isolated affine mature-history choice, not production readiness.
At a common mesh separated tensors still cost more than direct storage; no
mechanics coarsening is authorized. Stage C native source/tangent/observer,
accepted geometry/history transaction, scalar transport and versioned restart
are next. The distinct cohesive native remainder/lifting/shape-work gate,
prior B1 q6 failure, physical screens and cleanup data losses remain visible.
