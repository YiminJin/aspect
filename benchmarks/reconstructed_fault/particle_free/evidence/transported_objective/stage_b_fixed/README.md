# Stage B first part — fixed-profile reaction/history space

2 November 2026. Only the first, fixed-geometry/no-advection/matched-coefficient
comparison is selected and complete. The final selected Q8 configuration passes
14 required checks. The original Q6 candidate fails its unchanged quadrature
bound; its complete initial source/protocol/results/log are retained.

- `results.json`: full parameters, criteria, errors, timing, storage/DOFs and
  rejected-Q6 record; status `passed_selected_q8`.
- `space.csv`: independent direct and complete separated DGQ2 spatial errors,
  full3D/zz, pointwise and profile-weighted tractions, stored energy, analytic
  strain-source projection and best residual projection.
- `time.csv`: selected finite-rotation Lie integrators evolved independently;
  explicit continuum and discrete-closure errors.
- `quadrature.csv`: Q3/Q4/Q6 at N8 compared with Q12. Q8 follow-up is in JSON.
- `cost.csv`: projection/eigen/history construction and coefficient counts;
  Python algebra, not native mechanics or MPI timing.
- `profiles.csv`, `comparison.png/.pdf`: one normal cut and convergence figure.
- `initial/`: first failed experiment and exact source hash, not overwritten.
- `entry.json`, `entry-status.nul`, `verification.json`: root/branch/source,
  protected binary/checkpoint/server/cleanup preservation and final file hashes.

Run from repository root, one Python process and one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 python3 benchmarks/reconstructed_fault/particle_free/transported_objective/stage_b_fixed.py
```

Requires Python3, NumPy and Matplotlib. Numeric study takes about5s locally;
outputs stay under1MB. No native executable/build/trajectory or server was used.
To reproduce the retained initial failure, run its exact `initial/stage_b_fixed.py`
with `--output /tmp/CHOOSE_A_NEW_DIRECTORY`; it exits1 for the recorded q6 gate.
Do not overwrite the retained initial record. The default in that relocated
source is not an authoritative output path; always pass the explicit new path.

[Protocol](../../../transported_objective/STAGE_B_FIXED_PROTOCOL.md) ·
[Source](../../../transported_objective/stage_b_fixed.py) ·
[Report](../../../../../../doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_STAGE_B_FIXED_REPORT.md) ·
[Task state](../../../../../../doc/reconstructed_fault/particle_free/TRANSPORTED_OBJECTIVE_TASK_STATE.md).

The reference is a closed direct physical-component solution, independently
checked with real-component RK4; the residual has its own source evolution.
Spatial studies use exact semidiscrete time so timestep error cannot hide
history-resolution defects. Fixed-profile DGQ2 success does not qualify
transport, layered eta, native pressure/work/lifecycle or cohesive growth.
Stage A shape-work and all prior physical/cleanup limitations remain.
