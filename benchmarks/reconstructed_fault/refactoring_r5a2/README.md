# R5a2: manager particle-projection organization

Reference: accepted R5a1 `c3ce532be86765c7c9edcacb2f44377ff62820e1`,
`build-refactor-r5a1/aspect-r5a1-qualified`, SHA256
`4159f38bb530c97bed3fddb12009cd892b3428fb28c1e28530a230f050146ec6`.
Candidate: uncommitted R5a2, `build-refactor-r5a2/aspect-r5a2-qualified`, SHA256
`8f696bedb006147c564f4146b96f765e88b9cd11f68f3fb120c5f0c222a1bf27`.
The accepted R4c/repaired-fixture artifacts remain preserved; their unrelated
runtime campaigns were not repeated.

## Distinguishable changes

1. Both specifications clarify the user-confirmed contract: reverse interpolation
   with a valid particle cache is local, without search or MPI. Public lazy
   preparation can rebuild geometry collectively. The local validity predicate
   does not establish all-rank agreement; callers must enter rebuild consistently.
2. Ten complete manager members and the two exclusive tridiagonal factor/solve
   helpers move byte-for-byte into `manager_particle_projection.cc`. No owner,
   header/API/layout, checkpoint, key, expression, validation, publication or
   MPI ordering change. The remaining manager is exactly the original minus
   those blocks; source continuation and Stokes-QP methods remain there.
3. The new TU joins the existing CMake independent-compilation list (no unity
   or PCH), preserving every baseline unity group. No M1/M2 source change.
4. A small two-rank test uses existing timer call counts to check one cold
   rebuild after all-rank invalidation, then zero warm rebuilds. It compares
   interpolated values and support exactly before/cold/warm and emits full local
   values for matched-executable comparisons. It uses no production test API.

The pre-move inventory/cache-lifetime table and completed review are in
[the rolling review](../../../doc/reconstructed_fault/refactor_review.md).
The table remains valid after movement: manager storage, reference expiry,
local associations versus replicated factors/diagnostics, fresh P0 value
sampling, non-atomic rebuild publication and distinct error paths are unchanged.

## Verification

- Fresh Release build/link, independent original/new TUs, and 20 unique moved
  member symbols (ten in each dimension) pass. Nine source/structure checks pass.
- On both executables, one/two-rank geometry/quadrature/projection/registry/restart
  units pass 834 assertions in 16 cases per rank.
- Matched I_h composition/no-composition, surface adiabatic-pressure and both
  remote/cell I_h cache fixtures pass on one/two ranks. Existing surface checks
  exercise the actual manager's full-domain measure/mass/constant projection.
- Cold/warm regression passes on two ranks for both executables. Each rank has
  admitted particles and records one cold rebuild and zero warm rebuilds. Both
  emitted local value/support files match byte-for-byte between executables.
  This tests lazy preparation/reuse, not arbitrary rank-asymmetric invalidation
  or direct interception of every MPI call on the warm path.
- Short coupled pressure/history and accepted-update rollback on one/two ranks
  match preserved R5a1 decisions, statistics, cache-build counts and history
  fingerprints. These small fixtures do not emit complete field dumps.
- The preserved cohesive checkpoint restores original histories/V/geometry/bulk
  checks. Its known step-two nonlinear nonconvergence remains; the candidate's
  complete phase-field/coupled decision trace matches R5a1. Checkpoint payloads
  remain byte-identical. No physical parameter or tolerance was retuned.

All 115 matched checks pass (including 18 exact environment checks). The qualification records 39 expected build/runtime
outcomes, including the one known restart exit 1, 858 source/header/unit files
and 61 executed-artifact entries. Timing/path metadata is excluded. Prior
protected/checkpoint/local manifests and accepted artifacts match (apart from
the explicitly changed CMakeLists.txt entry). No demonstrated MPI correctness
defect occurred; documentation does not certify global cache-miss agreement.

## Initial failures and investigation

The first build let CMake insert the new file into its automatic unity batches.
This shifted unchanged `source/simulator/phase_field.cc` ahead of its implicit
provider of `ExcNonlinearSolverNoConvergence`. Compiling the identical baseline
file with baseline flags independently reproduced the missing-declaration
problem (and other incomplete-type dependencies). Its successful baseline unity
batch already supplies those includes. The small CMake entry for the new TU
restores the exact old unity membership and separately compiles the moved code;
no unrelated include cleanup or build-system redesign was made. Initial failure,
baseline diagnostic and successful build logs are retained separately.

Open MPI could not open local sockets in the sandbox; those launch failures are
retained as `sandbox-reference-*` and the runs were executed with permission.
The first reference surface runs omitted the I_h plugin required while parsing
the inherited input; `missing-ih-registration-*` preserves those logs. The harness
now loads both required plugins. The original symbol-check regex also counted
an STL instantiation containing the local `ResolvedComponent` type as a second
manager method. Anchoring it to the actual manager symbol yields the expected
20 definitions; no C++ correction was needed. Failed checker output is retained.
The first five candidate coupled/restart runs enabled optional performance
diagnostics absent from the reused R5a1 runs. They already matched numerically,
but were repeated with the exact baseline environment to preserve optional MPI
call ordering too. Those initial outputs/logs remain under `extra-performance`.
The final comparator also requires exact recorded environments for all 18 cases.

## Reproduction and artifacts

`evidence/*.json` records exact commands, exit statuses and environment; matching
`.log` files retain full output. `run_logged.py` refuses to overwrite logs.
`configure.json` records the existing GCC/OpenMPI/deal.II/Voro Release stack,
including `-fno-finite-math-only -ffp-contract=off`. `build-independent-projection`
and `build-final-check` are the successful builds; `build` is the initial unity
failure. `independent-commands.json` records both independent compile commands.

Build the `plugin` CMake project separately against the R5a1 and R5a2 build
packages into `reference-plugin-build` and `plugin-build`. It reuses existing
observers plus the new cold/warm test. `prepare_inputs.py` generates the matched
inputs and private checkpoint copies and refuses to replace existing branches.
Initialization uses the existing cache-fixture budget of 50 iterations; surface,
coupled and restart inputs keep their existing settings.

`run_checks.py reference` and `run_checks.py candidate` run the bounded matrix;
optional case names restrict a run (for example `reference surface-1 surface-2`).
Run with local MPI socket access. The accepted R5a1 pressure/rollback/restart
outputs are reused directly for comparison. No separate scientific worktree is
modified. `compare.py` compares decisions/statistics/work counters, cache CSVs,
local interpolation/support values, coupled fingerprints and restart payloads.
`verify_move.py` uses the saved pre-move manager and block offsets to prove exact
movement and preserved unity membership. `verify_symbols.py` checks original/new
objects and the executable. `qualify.py` reruns those read-only checks, validates
all expected outcomes/protection manifests and freezes the candidate binary.

Broader equal-volume domain-regeneration, particle migration/reordering,
empty-owner and rank-asymmetric invalidation coverage remain separate gaps.
No Debug/3D runtime, new full BP3 campaign, performance claim, Stokes-QP move
or surface-system implementation is included. Stop for review. Recommended next
bounded task: Stokes-QP association inventory/cache-lifetime assessment only.

## Accepted R5a baseline

R5a2 is accepted. The commit containing this entry saves the source movement,
specification clarification, focused regression and verification harness after
R5a1 `c3ce532be`. The qualified executable/hash above are unchanged. Its 858
source/header/unit files, 61 executed artifacts, prior protection/checkpoint
records and local temporary-file hashes were verified again before acceptance.
The user selected R5b1 next; the earlier Stokes-QP inventory recommendation is
not the active task. No runtime campaign was repeated for this commit.
