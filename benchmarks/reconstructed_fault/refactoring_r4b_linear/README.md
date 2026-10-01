# R4b first subpass: private condensed linear solve

Reference: accepted R4a commit `0c7ed1a0b`, executable
`build-refactor-r4a/aspect-r4a-qualified`, SHA256
`ed827270970996854ae2425e519255422f80a25b0602d1cbf863c88a8d4910be`.
Reference source/artifacts and the user's remaining `refactoring/tmp/` files are
captured before editing. Candidate builds and plugin/output directories are separate.

## Contract recorded before source edits

Extract one private Simulator member in `reconstructed_fault_stokes.cc`, moving
`solve_condensed_system` and its exclusive pressure-assembly-scale lambda.

| Aspect | Existing contract to retain |
|---|---|
| Inputs | Current immutable condensed Linearization; active/free mask; condensed RHS; accepted physical bulk iterate; initial/reference bulk residuals, original scale, precision and mixed convergence scale; already-converged flag; existing preconditioner setup duration |
| Outputs/mutations | Overwrite bulk direction, initially zero; accumulate the existing whole-solve Krylov count, including iterations before failure; existing diagnostics/timers/observer calls and temporary GMG initialization only |
| Simulator dependencies | Canonical surface system, assembled A/preconditioner matrices and AMG/pressure preconditioners; mesh/mapping/FE/introspection; current homogeneous constraints and pressure scaling; solver parameters, communicator, iteration/time identifiers, output/timing/signals |
| Collectives/order | RHS norm/zero return; verified nullspace; pressure FE assembly and MPI sum; compatibility projection/check before already-converged return; FGMRES/preconditioner actions; direction projection and fresh residual/null checks; same-budget restarts; synchronous observer before temporary objects die |
| Lifetimes | Caller keeps matrix, coupling generation, restricted inverse and constraints valid throughout the call. Local Schur/wrappers/Krylov storage die on return. GMG adapter borrows the velocity cycle only within its existing callback. No lambda/view escapes through the synchronous observer. |
| Not owned here | Nonlinear preparation, residual evaluation, bulk preconditioner matrix assembly, linearization, active-set rebuild, convergence decisions, V recovery/trials, history publication and whole-solve rollback |

Use one small private input record for the five related residual/precision
scalars, defined only in the implementation file and constructed at the call.
It copies no canonical manager/surface state and adds no persistent member or
serialized state. Other inputs and both outputs remain explicit arguments.
The private member signature uses the existing nested Linearization type, so
`simulator.h` includes its defining header; no public accessors or generic
operator interface are introduced. Existing bulk preconditioner assembly and
its timer stay in the driver before linearization, preserving their ordering.

This subpass does not extract trial residual evaluation, a step helper or shared
ordinary-solver setup. Those require subsequent selection. Keep zero-RHS and
already-converged exits exactly where they are relative to collective/checks;
no correctness repair is bundled with extraction.

## Selected verification

Build/link independently with 2D/3D instantiations and candidate-built plugins.
Reuse R4a one/two-rank condensation/Stage-I units, rollback and four short BP3
legacy/automatic trajectories, with exact fields/history/solver/cache comparisons.
Add matched existing residual-consistency, linear-budget-exhaustion and physical
pressure-gauge fixtures on one/two ranks, and the existing bounded Stage-I GMG-Q1
case on one rank. Compare against the actual R4a executable; retain existing
fixture assertions and all parameters/tolerances. No new numerical test is needed.
Ordinary Stokes and restart evidence are reused because those algorithms and
initialization/restoration are unchanged. Known Stage-J/cohesive failures remain
separate; no Debug/3D simulation or production/performance campaign is requested.

## Qualified result and post-extraction contract check

The pre-edit contract above still matches the implementation. Only
`include/aspect/simulator.h` and `source/simulator/solver/reconstructed_fault_stokes.cc`
change in production. The driver outside the replaced definition/call is exact;
all six source/protection checks pass. No state owner or public interface changes.
The explicit Linearization signature adds a header dependency, without adding
an abstract interface or new canonical state. The record is five scalar inputs
constructed for the call, not a persistent solver context.

All 29 selected build/runtime checks have their expected outcomes: seven
build/configure/compile checks and 22 runtime invocations. Four reference/candidate
exhaustion invocations intentionally exit 1 and verify the one-iteration budget,
no accepted trial, and restoration. There are no unexpected failures. Units
pass 20,149 assertions/16 cases per rank on one/two ranks. All **31 focused
comparisons** pass, including exact affine/fresh residual diagnostics, pressure
and history fingerprints, solver decisions, accepted-Newton rollback and the
existing one-rank GMG-Q1 lifecycle case. Four BP3 legacy/automatic trajectories
match in **372 field/history groups** (maximum absolute difference zero),
**24 cache/work checks** and **four detailed solver-decision comparisons**.
Only timing/path metadata are excluded. Candidate effective PRMs confirm that
candidate-built plugins are loaded.

Qualified candidate: `build-refactor-r4b-linear/aspect-r4b-linear-qualified`, SHA256
`9ff850bc7d4060ab75ee80dfdd44dc09b94c7862b8bfcf448a653ecdc17cd30d`.
`evidence/qualification.json` records results and reference/candidate hashes;
`candidate-source-hashes.json`, `executed-artifacts.json` and
`reference-plugin-hashes.json` preserve provenance. `linked-symbols.txt` confirms
one 2D/3D definition of both private methods. The initial include-order compile
failure is resolved and recorded separately; physical algorithms, tolerances,
assertions, checkpoint format and the accepted post-R3 implementation are retained.

No ordinary-solver algorithm or restart lifecycle changes, so those qualified
R4a/post-R3 checks are reused. No new Debug, 3D simulation, BFBT/melt/direct-solver
campaign, production earthquake or performance campaign was run. Historical
Stage-J/cohesive failures were not rerun or repaired. No new numerical defect
was found in the checked paths. The R4b changes remain uncommitted for review.
Proposed next selection: extract the existing non-committing trial residual
operation, documenting its input/output/mutation/MPI/lifetime contract first.

## Implementation and reproducibility notes

The extracted member is `Simulator::solve_reconstructed_fault_condensed_system`.
Its signature names the canonical Linearization and active set, RHS, accepted
bulk iterate, five-value scales record, convergence flag and setup duration,
with explicit direction and cumulative-iteration outputs. The pressure-scale
lambda stays local to that operation. The Schur/AMG/GMG implementation, budget,
true-residual restarts, compatibility checks and observer calls are retained.
`verify_source.py` checks the moved operation modulo indentation and the five
explicit scale accesses, the unchanged driver outside its replacement call,
and the header's private declarations/include only. It also checks reference
artifacts and user-local files. No scientific defect was identified during
source inspection; zero-RHS and already-converged behavior is deliberately
preserved, rather than redesigned in this extraction.

`evidence/independent.log` records an initial declaration-order compile failure:
the new condensed-system include needs the particle declarations already
provided by Simulator's existing particle-manager include. Placing it after
that include resolves the issue (`independent-final.json`). Neither particle
nor surface-system source changes. The complete independent command and 2D/3D
symbols are saved in `independent-command.json` and `independent-symbols.txt`.
The full build also compiles the driver outside unity/PCH.

`plugin/CMakeLists.txt` builds existing test sources. Reference extra fixtures
use `build-refactor-r4a` and a captured R4a Simulator header before the source
include directory (`reference-plugin-build/.../flags.make` records the order).
All other source dependencies are unchanged. Candidate tests load libraries
from `plugin-build` against `build-refactor-r4b-linear`.

`prepare_inputs.py` reuses original PRMs, changing only library/output paths.
The GMG case is the existing `server_gmg/gmg_q1.prm` fixture without parameter
changes. `run_checks.sh reference` runs the seven added checks against R4a;
`run_checks.sh candidate` adds the existing units, rollback and BP3 trajectories.
`compare_checks.py` compares against fresh R4a extended runs and saved R4a
unit/rollback evidence. Exhaustion's exit 1 is expected only with the retained
one-iteration-budget, no-accepted-trial and restoration checks. Other failures
are not accepted merely because they match. `compare_bp3.py` preserves exact
field/history, cache/work and detailed solver-decision comparisons, excluding
timing/path metadata. `qualify.py` records outcomes, executed plugin paths,
source/binary hashes and 2D/3D definitions and preserves the tested executable.

Run all tools from the repository root through `run_logged.py LABEL SECONDS
COMMAND...`; the runner records commands, stack-related environment, logs,
elapsed time and exit status, and refuses to overwrite an existing label.
Use the R4a configure command with build directory `build-refactor-r4b-linear`;
both builds use GCC 12.4/OpenMPI 5.0.6/deal.II 9.6.2-local, Voro++, Release,
`-fno-finite-math-only -ffp-contract=off`, and two build jobs. Runtime uses one
compute thread per rank, with explicit B/G switches only in the established
BP3 cases. MPI requires local-socket permission in this environment.

Subsequent trial-residual extraction, a possible iteration helper, and R4c's
shared setup remain unimplemented. Stop after this subpass for review.
