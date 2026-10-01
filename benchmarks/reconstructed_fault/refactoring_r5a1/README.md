# R5a1: manager slip-rate lifecycle organization

Reference: accepted fixture repair `aea2a80b0` after production R4c `fa6013678`.
Use `build-refactor-r4c/aspect-r4c-verified`, SHA256
`c6811cbd877ff56af9113dc2f110cfd9a8b71998ea63f8e249ff94b292eeed85`.
Repaired frozen-fixture manifests and 226 checks are reused, not rerun.
Local R5 instructions and temporary review edits are preserved.

## Pre-edit responsibility inventory

| Group | Existing implementation | Disposition |
|---|---|---|
| Geometry/reconstruction | `manager.cc`: parameter handling, initial H support/reconstruction, add fault, shear sense, accessors | Remain; geometry creation sizes slip-rate vectors |
| Generic property registry/persistence | `manager.cc`: registration, lookup, `rebuild_after_deserialization`; `manager.h`: archive save/load | Remain; no property or archive changes |
| Slip-rate lifecycle | 16 methods in `manager.cc`, including trial rollback below the old restart section marker | Move complete definitions to `manager_slip_rate.cc`; no exclusive local helpers |
| Particle projection | `manager.cc`: domain associations, Q1 mass/projection, reduction and publication, invalidation | Remain; distinct measures/lifetime from QP associations |
| Stokes-QP associations | `manager.cc`: quadrature identity, normal/bulk-source lookup and cache validity/rebuild | Remain; no cache reorganization |
| Boundary contacts/source continuation | Already extracted `boundary_contact_manager.cc` and `boundary_contact.cc`; legacy continuation and lookup remain in `manager.cc` | Retain all existing boundaries |

M3 owns storage and generic admissibility; M4 computes initial physical rates and
histories, M5 decides nonlinear acceptance and terminal publication. The existing
Q1 interpolation utility is shared; do not move it or invent a new owner.

## Slip-rate state table (recorded before move; verified unchanged afterward)

All rows remain members of `ReconstructedFaultManager<dim>`; none is relocated
into a new struct or owner. Arrays are replicated and indexed by manager fault
and vertex; moved operations add no MPI calls.

| Member | Readers | Writers / initialization / reset | Checkpoint handling |
|---|---|---|---|
| `timestep_committed_slip_rates` | committed accessor, solve begin/rollback, material/output/timestep callers | geometry adds empty row; explicit initializer fills it; validated converged commit copies current values without allocation | Serialized; validated against geometry on load |
| `current_newton_slip_rates` | active accessor when no trial, interpolation, M4/M5 callers | initializer; solve begin copies committed then applies prescribed entries; trial acceptance copies trial; whole-solve rollback copies committed | Not serialized; load copies committed |
| `trial_slip_rates` | active accessor during trial and acceptance | begin copies current; affine setter recomputes from current; absolute setter validates then copies exact values; begin-solve/accept/rollback clear per-fault rows | Not serialized; load reconstructs empty per-fault rows |
| `slip_rate_initialized` | readiness/access/deserialize validation | geometry adds false; explicit initializer sets true; reconstruction clears | Serialized; restored/validated |
| `slip_rate_nonlinear_solve_active` | lifecycle guards and geometry/shear-sense guards | default false; begin sets true; commit/whole rollback/reconstruction/load set false | Transient; false after load |
| `slip_rate_trial_active` | accessor selection and lifecycle guards | default false; trial begin true; accept/trial rollback/whole rollback/reconstruction/load false | Transient; false after load |
| `prescribed_slip_rates` | solve begin, masks, trial setters | geometry adds empty map; caller setter validates outside solve; reconstruction clears; load sizes empty maps per fault (R3 fix) | Never serialized; caller reattaches conditions after restart |

`reconstruct_initial_faults()` clears all arrays/flags before adding geometry.
`add_reconstructed_fault()` appends their per-fault slots. These and
`rebuild_after_deserialization()` remain in manager.cc, with the latter's
validation/write/invalidation order unchanged. Archive save/load stay in the
unchanged header. Borrowed accessor references keep their existing lifetime:
working-vector assignment/reset can invalidate them. Commit validation remains
separate from the existing no-allocation `noexcept` copy operation. Constitutive
lower-bound policy remains with M4/M5; the manager accepts finite nonnegative V.
No stronger atomicity or topology-change guarantee is introduced.

## Selected verification

Byte-exact moved definitions and otherwise unchanged manager/header/archive;
separate Release build/link and independent original/new TUs without unity/PCH;
16 method symbols in each dimension, with no duplicate definitions. Existing
manager lifecycle/initialization/prescribed/absolute-bound/interpolation/commit
and archive/restart tests on one/two ranks. Reuse R4c pressure/history coupled
and accepted-update rollback cases on one/two ranks, comparing exact fields and
decisions to preserved R4c outputs. Reuse repaired frozen AMG/GMG and ordinary
solver evidence; do not repeat that unrelated runtime campaign.

## Completed result and reproduction

The 16 method definitions moved byte-for-byte, including arithmetic, assertions,
comments and `noexcept`. The misplaced old restart section heading now precedes
`rebuild_after_deserialization()`; no method body or validation moved within a
method. The remaining manager is otherwise byte-identical; no header/API/layout,
archive, caller, cache or ownership change. Standard source discovery finds the
new file; CMake/unity/PCH settings are unchanged. Only moved members receive new
explicit instantiations. Independent original/new object and linked symbol
checks find exactly one definition of each of the 16 methods for each dimension,
with none in the independent remaining-manager object. The pre-edit state table
and all cross-file initialization responsibilities remain valid after the move.

Candidate: `build-refactor-r5a1/aspect-r5a1-qualified`, SHA256
`4159f38bb530c97bed3fddb12009cd892b3428fb28c1e28530a230f050146ec6`.
Seven source checks and 31 matched verification checks pass. The full Release
build/link and two independent translation units pass. The existing seven
lifecycle tests pass 20,130 assertions per rank on one/two ranks on both accepted
reference and candidate. Coupled pressure/history and accepted-update rollback
match the preserved R4c outputs exactly on each rank count, including nodal V/surface-history fingerprints, bulk velocity norm and particle
stress mean. These fixtures do not emit full bulk/particle fields.
No new test assertion, numerical tolerance or physical parameter was introduced.

Two private copies of the preserved pre-fix one-rank cohesive checkpoint resume
under reference and candidate with all original restored-state assertions passing.
They retain the known step-two nonlinear failure, with exact matching full
phase-field/coupled traces and unchanged checkpoint payloads. This is a successful
restart-compatibility comparison, not a converged cohesive step-two trajectory.
The unit restart regression additionally checks an absolute trial **before** any
prescribed-rate setter and nonempty caller reattachment afterward. No new full
BP3 trajectory was run; prior BP3 cache/work and repaired frozen evidence are
reused because their implementations did not move.

There are 17 recorded configure/build/runtime outcomes: 15 successes and the two
explicitly verified pre-existing cohesive nonconvergences. No candidate-only
failure occurred. `evidence/reference.json` records the exact accepted source and
binary, `reference-source-hashes.json` / `candidate-source-hashes.json` cover the
857 resulting source/header/unit-test files, `protected-hashes.json` protects
prior R4c/fixture evidence and local edits, and `executed-artifacts.json` records
18 executable/plugin/test/input artifacts. The original checkpoint has its own
protection manifest. Full commands, environments, logs and JSON verdicts remain
under ignored `evidence/`; candidate and reference outputs are separate.

Reproduction from the worktree root:

1. Recreate the separate `build-refactor-r5a1` configuration using
   `evidence/configure.json` (same compiler/libraries/floating-point flags as R4c),
   then `cmake --build build-refactor-r5a1 -j4`. Use `run_logged.py` to preserve
   bounded command outcomes and refuse overwrites.
2. Run `compile_independent.py`; configure/build `plugin/` separately against the
   reference R4c and candidate R5a1 builds. Only the restart target is needed for
   the reference plugin; all three targets are needed for the candidate.
3. On fresh output paths run `prepare_inputs.py`, `run_checks.py reference`,
   `run_checks.py candidate`, and `run_restart.py` with each version. The latter
   expects exit 1 and must also pass `compare.py`'s restored-state/failure checks.
4. `qualify.py` runs exact source/symbol/result checks, validates protection
   manifests, and freezes the qualified candidate. Do not overwrite accepted
   reference outputs or invoke the old full R4 campaign.

The protected pre-edit manager snapshot and manifests are retained locally;
`git show aea2a80b0:source/reconstructed_fault/manager.cc` supplies the same source
reference. No Debug configuration, 3-D simulation, long campaign, ordinary solver
or frozen-probe rerun was needed. No projection/cache or surface-system code was
reorganized. R5a1 remains uncommitted for review. Proposed next bounded task:
R5a2 particle-projection inventory/cache-lifetime assessment and one coherent
move proposal, keeping Stokes-QP associations separate.

## Accepted R5a1 baseline

The user accepted R5a1. The source/documentation/harness commit containing this
record is the reference for the following R5a2 inventory/proposal. Its parent is
`aea2a80b0`. The qualified executable and SHA256 above remain unchanged. Before
commit, all 857 source entries, 18 executable/plugin/input entries and reference/
checkpoint/local-edit protection entries matched. Verification is reused without
another runtime campaign. Local temporary review files remain excluded.

- `candidate-source-hashes.json` SHA256: `d41c0ba2857db10949660a4ab886d6f06070e269e7d03a94d1ff505c136ebf72`
- `executed-artifacts.json` SHA256: `7149755453d7a3c9bc234e5f32ceec46b60f808bcedc4f95f1ffd70ab6e25260`
- `qualification.json` SHA256: `7895b80e4d78214591819bde2729a99b0f9357a5e9c3df5d299ab7b1d7111513`
