# K5 targeted bulk-resolution comparison

Consolidated synthesis of these four follow-up tasks:
[history, spatial-resolution and timestep report](stage_K5_history_resolution_consolidated_report.md).
This individual report is retained as the chronological evidence record.

Completed follow-up: [matched-time localized resolution pair](stage_K5_matched_junction_resolution.md).
The report below preserves the earlier stopped preflight; it is not the result
of that subsequently completed comparison.

## Outcome: preflight stopped, not a completed resolution comparison

The single fresh four-rank run took **66.234 s** and stopped at the declared
matched-timestep guard after converged timestep-zero mechanics, before step 1.
No step-12 spatial evidence was obtained. No trajectory was retried or continued,
and neither particle nor FE history has been selected as an accuracy reference.

Two fixture issues must be resolved before interpreting a spatial comparison:

1. The actual split-RSF controller selected **2618113.9222809309 s**, whereas
   the saved coarse first step is **2618120.216489729 s**. The difference is
   **6.2942087981 s (2.40409465 ppm)**. At shallow node 1154 (100 m down-dip),
   the refined initial rate is `1.0185449318964074e-9 m/s`; the implemented
   rate-weakening restriction `CFL*a*Dc/(b*V)` gives
   `2618113.922280931 s`, reproducing the selected value. This is an expected
   solution-dependent controller change, not a solver failure or evidence of
   a large temporal error. The cap deliberately did not override it.
2. Increasing initial adaptive refinement from 6 to 7 does **not** produce
   only the requested local patch. Comparing actual exported CellIds identifies
   3794 split coarse cells, of which 1544 have centers outside 36--44 km;
   24 coarse cells are also coarsened. The net change is
   `3*3794 - 3*24/4 = 11364` cells. The minimum-refinement function is evaluated
   at cell centers on every adaptive pass; an extra pass also completes
   off-patch minimum refinements and permits ordinary coarsening elsewhere.
   These are benchmark-mesh confounds, not a newly demonstrated defect in
   history transfer or mechanics.

### Measured initialization and mesh

| Quantity | Saved coarse | Refined preflight |
|---|---:|---:|
| Bulk cells | 36106 | 47470 |
| Total DoFs | 1282726 | 1674376 |
| 37--43 km centerline cell size | 97.65625 m | 48.828125 m |
| 24--26 km centerline cell size | 97.65625 m | 97.65625 m |
| 24--26 km, within 800 m normal distance | 97.65625--195.3125 m | 97.65625 m |

Here “centerline” selects cell centers within 50 m of the prescribed line;
the full per-cell CSVs are retained. The control is therefore unchanged only
on its centerline, **not over its entire stress-transfer neighbourhood**.
The exported fault target coordinates agree within `1e-8 m`. Official initial
Theta, a, and 50-MPa background normal traction are unchanged. Reinitialization
changes the evaluated initial cohesion and corresponding shear-background
target by up to **17558.954 Pa**. That initial discrete change would influence
subsequent history and cannot be subtracted away physically.

The initial target weak-balance construction has relative error
`2.29408984e-16`. Realized side-velocity constraint error is exactly zero.
The three returned linear directions total 53 Krylov iterations; every fresh
residual passes, with worst fresh/target ratio **0.9781842**. Final nonlinear
bulk/fault relative residuals are **5.49334e-15 / 4.85473e-11**, respectively;
absolute values are **7.84634e-6** in the bulk weak residual's units and
**5.31060e-5 Pa** for the surface RMS residual.

The stop occurs in timestep selection before standard accepted-state
postprocessing. There is no `accepted_steps.csv`, ordinary accepted fault
profile, or valid new restart checkpoint. The raw initialization exports are
converged-mechanics diagnostics, **not a completed accepted-state/lifecycle
verification**. At this zero-retained-stress initialization, the dual full-tensor
particle/FE audit agrees exactly, as expected; that does not address the
nonzero retained history or bound reaction at step 12.

### Recommended next action

Prepare a **paired common-timestep comparison**, using a mesh refinement tied
to the saved coarse topology (plus required mesh grading), not an extra
unrestricted adaptive pass. Choose a shared sequence with margin below both
controllers and rerun the coarse comparison too. Do not force the saved coarse
steps through a stricter refined controller, or mistake this 2.4-ppm difference
alone for a significant history error. A benchmark-local prescribed sequence
would be an explicit alternative requiring review, not a silent controller
override. Keep the particle-based surface history and all physical parameters.

This is a fixture-design review point. The previous step-12 bound/history
conclusions remain unchanged; no new claim about stress convergence is made.

### Commands, artifacts and verification limits

```
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_junction_refinement.py
python3 benchmarks/reconstructed_fault/bp3/compare_junction_resolution.py
python3 benchmarks/reconstructed_fault/bp3/analyze_history_mechanics.py \
  benchmarks/reconstructed_fault/bp3/normal-stress-junction-refined-local4
```

Artifacts: `benchmarks/reconstructed_fault/bp3/normal-stress-junction-refined-local4/`:
`run.log`, `provenance.json`, `execution.json`, `initial_mesh_*.csv`,
`initial_traction_target.csv`, `velocity_constraints.csv`, `history_surface_rank*.csv`,
`surface_comparison.csv`, and `resolution-analysis/comparison.json`.
The source/input/library/binary hashes used by the run are in provenance.
MPI exit 1 is the intentional matched-step guard, not a successful trajectory.

Only benchmark files were edited: the new replay-cap header, one include in
`bp3.cc`, the local-refinement prm, runner, analysis script, and this report.
No production file was modified by this task. The guard's failing branch and
four-rank reduction were exercised. Its successful multistep branch and the
analysis script's later-state plotting branch were **not** exercised. Python
syntax checks and `git diff --check` pass. The cap's unused repeat-step proposal
was subsequently made neutral (`max double`, matching the base plugin);
this changes nothing on the tested `advance` path and was rebuilt, not rerun.

The diagnostic analyzer initially assumed the new mesh only refined existing
leaves; the ancestor check exposed coarsening and was extended to count both
operations. This was an offline analysis correction, not a numerical change.

## Predeclared experiment

Retain particle history in surface mechanics. A fresh four-rank trajectory
refines bulk cells from level 10 to 11 (97.65625 to 48.828125 m) in the existing
1500-m half-width strip between down-dip 36 and 44 km. Quadtree refinement
halves both bulk directions locally; it is not a pure normal refinement.
Measure 37--43 km, the 39.9-km bound row and its neighbours; 25 km is the
unrefined control. The bottom tip is **not** refined by this experiment.

Retain the original 100-m fault grid, physical parameters, prescribed deep
rates, particle density per cell, full I_h, support, quadrature, transfer,
initialization laws and tolerances. Initialize normally from the original
physical data, rather than transferring a coarse history. Keep initial
projection/background differences visible. They influence subsequent history;
subtracting them diagnostically does not eliminate that influence.

Use the saved accepted step-0--12 sequence as an additional timestep **cap**,
with the existing CFL and reconstructed-fault timestep models still active.
If either requires a smaller step, stop rather than forcing its override or
interpreting unmatched trajectories as spatial error. No first-event
continuation or automatically repeated run is authorized by this fixture.

The question is whether the inherited junction anomaly, contact reaction and
particle/FE full-tensor discrepancy substantially change under a targeted bulk
refinement. Neither representation is an accuracy reference. A two-level
change cannot establish asymptotic convergence or isolate bulk interpolation
from increased particle density. Preserve the previous frozen weak-test
evidence; do not repeat the FE-history alternate solve.

Expected cost: about 15--25 minutes on four local ranks, versus 907 s for the
saved coarse replay. The patch adds approximately 7500 cells before grading,
about 20--25% over the 36106-cell baseline. Existing mesh CSVs verify realized
resolution; ordinary BP3 output is unchanged. Stop on a genuine numerical
failure, changed timestep sequence, or completion of step 12.
