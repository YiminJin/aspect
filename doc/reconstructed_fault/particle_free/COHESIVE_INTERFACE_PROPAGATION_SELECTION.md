# Integrated adaptive formation, developed advance and sliding

User-selected demonstration; baseline f1f9f29cf40c32d1ddcea000b724375b64853532
plus preserved work. Prior load-turn/candidate/formed-turn milestones are
accepted. Their residual accuracy findings remain documented, not prerequisites
for another isolated campaign. No production/BP3, law, inertia or R change.

First qualify **parallel execution**, as subsequently required by the user.
The fixed bulk mesh/matrix remains replicated in this bounded experiment;
independent response columns and Schur contractions are owned round-robin;
whole bulk cells own the expensive quadrature/stress work. The matrix is
assembled once, then broadcast. Actual columns and cell basis values retain
serial arithmetic; broadcasts copy them without summing. R-disk quadrature
contributions are gathered by cell and folded in deterministic cell order,
independent of rank count. This groups sums differently from legacy serial
sampling, with a declared <=1e-12 roundoff comparison; quadrature points/weights
are unchanged. One coordinator runs the
small reduced surface solve and owns accepted history/publication. This is
parallel native bulk work, not production ASPECT distributed-mesh integration.
Keep the old serial path selectable. Diagnostic energy/normal-work checks use
MPI sums of owned-cell contributions; their roundoff is measured separately,
not inherited as an exact identity or a changed native allowance.

Before the longer run compare old serial and opt-in 1/2/4-rank output at the
actual .440 s geometry (and a cheap layout if needed): K, Q, current normal work,
sided traces, basis columns and pressures should be bitwise identical. Selector
and diagnostic energy differences must be <=1e-12 relative; all existing
native 1e-4 energy-check / 1e-6 derivative / 1e-8 Pa weak gates stay unchanged.
Test short accepted advancement/restart across rank counts and a deliberately
injected nonroot worker failure with bounded timeout, unchanged accepted input
and deterministic retry. Measure setup phases, memory and actual wall speedup.
Do not start the longer trajectory before these checks pass.

Use one maintained `cohesive_interface_propagation.py` entry point for native
export/caching, evolution, material ledger, embedded checks and PNG/GIF/data
reporting. Reuse the validated candidate owner, constitutive return and reduced
native solver. Historical scripts/evidence are retained as references.

Economical baseline: exact verified **coarse .440 s checkpoint**, all formed
geometry/identities/histories unchanged; future cell length .09375 m, temporary
extent .1875 m, dt=.000125 s, cadence .0025 s. Keep bulk Cartesian64 Q2/P1disc,
Gauss6/source rules, surface Gauss3, selector Gauss8, R=.75 m and numerical
lifting/normal half-width .75 m. Same confinement/current-pressure law, Maxwell
and RSF/radiation coefficients. The spacing study's newly formed damage changes
.93%; late timestep uncertainty and old weak-normal/point-traction failures stay.

Follow existing loading **unchanged** through .455 s, then hold displacement
through .460 s. If needed extend ONLY that same hold in .005 s observations,
at most .490 s. No increased rate/load or reloading to force an outcome.

Track every persistent Gauss identity, cell admission time/direction, point
damage/full-decohesion brackets, signed/absolute slip, age, Maxwell memory,
current pressure, cohesive/frictional/radiation resistance and work. Distinguish
early-turn (.410–.425), **late-turn (.425–.440)** and subsequent material. The
late-turn cohort was not fully decohered at the selected initial state: it is
the decisive entry target, not an already-mature initial point.

Aim for two entire late-turn cells (all three Gauss records) reached by the
connected a=1 front and continuing mature RSF sliding during a hold. Report
post-decohesion slip/Dc and state change. Meaningful sliding diagnostic: >=.1 Dc
and >=5% age change on a late-turn mature cell; seek Dc if the bounded history
permits it, without treating that value as an acceptance mask. q must vanish
where a=1. Report arrest/insufficient development as outcomes rather than
changing the law. Fully decohered and a=.05/.5 fronts remain separate.

Integrate focused verification into challenging parts of the trajectory:
one half-dt replay around first late-turn cell maturation/loading-to-hold,
one short continuation of the existing fine .440 s branch if material/flow
localization challenges the known spacing error, and one representative
common-field fitted comparison plus mechanically coupled pressure sensitivity.
No broad mesh/threshold sweep and no demand to move a borderline percentage.
The fine branch runs through .00125 s after the first late-turn cell matures,
at common physical times; later mature hold response remains exploratory beyond
that spatial-control window. Mature material averages use the first .1875 m
of completely decohered late-turn cells, not a mesh-dependent count of fine cells.
After the main trajectory reaches that material target, also replay the final
.45875–.460 s at half timestep: the first-maturation window alone does not
measure the final mature-slip/state response. Declare this extra twenty-step
control before execution, with the same 5% increment/slip and one-cell screens;
no new trajectory length, loading or parameter search.
Use 5% increment/slip targets, one coarse-cell front bracket and 1 degree
direction uncertainty; preserve every failure. Physical absolute work remainder
screen remains 1% of both sliding loss and added fracture, separate from exact
BE ledger, with hold windows reported even when added fracture is small/zero.

Stop for contact/admissibility/stability failure, noncontiguous formation without
a topology rule, unresolved material sensitivity or need for new physics.
Retain all failed trials and unchanged accepted checkpoints. Candidate completion
remains bounded by four retries/1.5 m. Numerical cutback may address convergence
from the same immutable state; it cannot waive a physical failure. Other caps:
formed length 7.5 m, at least 2R to the outer boundary for the whole sampling disk,
3 GiB new raw storage, 1 GiB disk reserve, 300 s/export (60 s injected failure),
90 min/main trajectory and 45 min focused checks. Save every accepted state
atomically and preserve event/hold checkpoints outside rolling files.

Report actual success/limits, measured numerical uncertainty, runtime, energy,
parallel scope and concrete remaining integration into a usable ASPECT model.
No automatic further calibration, inertia, free branching or long campaign.
