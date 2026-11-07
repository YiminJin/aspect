> **Historical total-spin campaign.** The revised user selection resumes fresh corrected-law work under [LAYERED_BULK_CAMPAIGN_PROTOCOL.md](LAYERED_BULK_CAMPAIGN_PROTOCOL.md). Preserve the evidence and interrupted L0 below; do not resume its checkpoints under bulk spin. Its stop was explicit SIGTERM, not a demonstrated native failure.

> **Stopped at the user’s request, 5 November 2026:** the layered campaign is paused for immediate scientific revision. The active L0 test was terminated; partial outputs, completed runs, source changes and failed evidence are preserved. No further tests are authorized by this campaign continuation until the issue is addressed. The main ±3.125, refinement and adaptive/restart matrix has not run.

# Layered moving-fault campaign — in progress

Selected task: [campaign instructions](CODEX_LAYERED_MOVING_FAULT_CAMPAIGN.txt).
[Protocol](LAYERED_MOVING_FAULT_PROTOCOL.md). This report will be completed with
the coupled matrix; the homogeneous closure below is not the stopping point.

## Tested starting state and implementation

Root `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD cf06525471ce89789f1ac10905dc79f3aac7562a plus the
retained four-file prescribed-exterior implementation and documentation changes.
The tested core/checker under `stage-c/exterior-stress/` are reused. The deliberately
restored `native/aspect-release` is not this candidate. No core solver or physical
update has changed during this campaign. Benchmark source is in
`transported_objective/layered_background/`; runner/reducers have `layered` names.
All work is local, sequential, with at most four MPI ranks/build workers.

The scalar reference uses Gauss8 on 64 fixed depth panels. It implements the
selected equilibrium split recurrence with accepted spin and actual attempted
dt. An initial .1-s mechanics profile supplies initial accepted gamma but leaves
stored stress zero. The plugin attaches the existing pure-transport stress
callback. Ordinary start_timestep prepares boundary candidates; accepted-step
postprocessing publishes only after the controller/native publication. Existing
plugin save/load stores accepted stress/gamma/time/step and provider identity.
Mechanical boundary plugins read the candidate; actual interior velocity remains
part of the native coupled solve. There is no alternate timestep driver.

Reference-only checks pass: 32/64/128 depth-panel comparison differs by at most
5.1242e-12; homogeneous recurrence comparison by 1.8752e-13; integrated velocity
difference by 1.1324e-13; top velocity by 1.1369e-13. These are scalar reference
checks, not yet evidence for a coupled physical trajectory.

The first native screen found a benchmark parameter-lookup defect: the
nonlinear-postprocessing guard read its parameter outside the Postprocess
subsection. It failed before initialization. The guard is corrected; the two-step coupled screen passes. The rejected-step
run and its identical accepted-schedule control agree bitwise in every bulk,
surface and background field (five accepted physical steps); ASPECT's .91 relative
timestep-growth limit is retained. Failed
input/log and the pre-fix source/DSO remain preserved. No mechanics changed.

## Homogeneous temporal closure completed

The 32²/.00625 compatible/exterior run completed 64 physical steps to .4 s in
700.96 s, with zero retries. All histories and geometry publish exactly once.
Retained .05/.025/.0125 cases are reused. Common-time comparisons retain the
same original 18 material labels and central/endpoint masks, using the finest
trajectory as common region reference. Complete history comparisons, including
startup, are in `homogeneous-history-comparison.json` under campaign evidence.

For .0125 versus .00625 at the final time:

| Quantity | Central absolute L2 | Central relative | Ends absolute L2 | Ends relative |
|---|---:|---:|---:|---:|
| Cumulative slip |1.16541e-4|3.54596%|4.41635e-5|4.28285%|
| Slip rate |2.16103e-4|about .78%|1.41487e-4|1.51543%|
| Pressure-inclusive normal traction |2.09963e-3|.207308%|1.84791e-3|.223776%|
| Work-adjoint shear traction |6.96887e-4|.052510%|1.07638e-3|.097718%|

Surface L2 units include sqrt(m) from physical arc length. These are pairwise
sensitivities, not exact errors. Cumulative-slip sensitivity approximately halves
from 7.05%/8.62%; it remains larger than the preceding bulk mesh sensitivity.
The endpoint energy remainder decreases from 1.59157% to .798570% of active work.
This supports .0125 as a bounded comparison cap and .00625 as a temporal
reference; it does not establish a universal accuracy threshold.

## Preliminary coupled motion check

The U=2.5, h=.125, dt=.0125 run completed 32 steps in447.53 s without
retries. Its mean label displacement is1.15584 m, including about.156 m
shear/fault drift. To ensure the negative case also traverses roughly eight
coarse cells, the main matched pair uses |U|=3.125 (ten added cells). This
preliminary run remains evidence, separate from the main pair. Its profile
mass increases3.22% and peak falls to.3133, motivating explicit mass/flux
and spatial-refinement diagnostics rather than relying on global energy alone.

## Remaining selected work

Complete L0/L+/L- at .0125, L+ at .00625 and one finer bulk mesh at .00625;
then existing slip/age timestep control, nonuniform schedule if necessary,
matched checkpoint continuation with a rank change and rejection comparison.
Assess directional differences against discretization sensitivities and include
one wider box if finite-boundary influence dominates. Complete history/field/
energy/cost figures and cell-crossing animation, and the BP3-like pilot decision.

L0 is moving under shear and fault flow. Existing stationary DG controls do not
provide the same layered material transport law with just geometry frozen; no
such switch is introduced here. The layered pressure convention p=tau_yy differs
from the earlier homogeneous benchmark's p=0 even at constant viscosity. Stress
recurrence equivalence therefore does not imply identical normal fault loading.

No production, growth/AMR or cohesive qualification is implied. Independent
cohesive failures and inherited loss of 39/45 native-pilot frames and earlier
temporary logs remain unchanged.
