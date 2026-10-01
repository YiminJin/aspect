# R3b: expose the existing accepted-history lifecycle

Reference: corrected R3a commit `bef79b31a`, executable
`build-restart-fix/aspect-r3a-corrected-qualified`, SHA256
`2bd6dca29898f79eff45ca616e9db6f86e9d3c6611ee867b57d289c85798436b`.
The preceding restart correction is retained unchanged.

The user's suspected deletion was checked before edits. The complete R3a section
in `doc/reconstructed_fault/refactor_review.md` exactly matched the committed
section; no restoration was necessary. Its table/report are captured in
`evidence/R3a-review-verified.md` and remain unchanged. Local `refactoring/tmp/`
files and the deletion/move of `refactoring/R2b_review.md` are preserved.

## Extraction and ownership

Only `history.cc` and private declarations in `phase_field_fault.h` change.
The driver retains entry validation, the timestep-zero return, positive timestep
and particle-property/composition mapping checks, then calls:

| Private operation | Inputs and output | Explicit dependency/order |
|---|---|---|
| `sample_accepted_history` | Accepted bulk state → per-association points, velocity gradients, temperatures, current/previous phase values | Existing association order, FE mapping/evaluation and old solution; identical remote evaluation order |
| `compute_history_candidates` | Samples, dt, particle property offsets and chemical mapping → projected cohesive traction, Theta and particle stress/H candidates | Frozen surface data and retained histories; collective error propagation before cohesive projection; projected traction before H; second error propagation after candidates |
| `validate_history_candidates` | Candidates → existing checks and projection diagnostics | Completeness MPI min, cohesive validation, local particle-ID checks, then diagnostics |
| `publish_history_candidates` | Validated candidates and particle property offsets → existing persistent scalar writes | Cohesive/previous-I_h first, Theta second, particle stress/H last; no new acceptance decision |

Two private types defined only in `history.cc` retain scratch buffers for one
call. They add no persistent material members, checkpoint data, public interface
or committed-state owner. Sampling cache and diagnostic streams remain alive
through publication. The numerical bodies are unchanged; local values become
buffer members with references in their consuming operations. Manager/particle
aliases are acquired within the operations that use them. No new noexcept or
atomicity guarantee is added. Timestep-zero, mature/frozen and cohesive/evolving
branches remain distinct. Both R2 files, constitutive.cc, initial history setup,
solver acceptance/rollback and manager serialization are unchanged.

The ten-row R3a ownership/lifecycle table was checked before editing and again
after extraction. It still applies: material computes; manager/particle system
stores and serializes; solver/simulator accepts timesteps. The intermediate
collectives belong inside candidate construction, not all at the third stage.

## Verification

See `evidence/*.json` for individual outcomes, source proof and comparisons.
`verify_source.py` checks exact numerical blocks (allowing only buffer references
and assignment declarations), unchanged validation/publication order, unchanged
surrounding history methods, private-only header additions, protected source and
the intact R3a review. It verifies 5,822 other source/header/test files unchanged.
`instantiations.json` checks all five affected methods for both 2D and 3D.
The full Release build uses the same compiler/options as R3a; history also
compiles independently without unity or PCH.

A final timestamp check caught that the initial full build had compiled history
before the last comment/whitespace cleanup. That object was rebuilt and the entire
candidate suite rerun. The first executable and its evidence are preserved as
`build-refactor-r3b/aspect-r3b-first-qualified` and `evidence/first-candidate/`.
All individual final cases survived the session-server restart and completed
without another run. The interrupted outer wrapper left no aggregate suite JSON;
the verifier checks each completed case record independently. The recorded qualified hash below identifies the rebuilt executable.

All verification is complete:

- Fresh Release build and independent history compilation passed. All four
  helpers and the driver have definitions for dimensions 2 and 3.
- Reference/candidate on one/two ranks: 20,828 assertions/28 unit cases per rank
  pass, including the restart fix, prescribed rates, Maxwell/cohesive/state
  limiter and Stage-I coverage. Temperature, frozen stress, and original plus
  open-top accepted-Newton rollback fixtures pass. All 40 lifecycle comparisons
  match, including preserved original Stage-J pressure-compatibility failures.
- Cohesive fresh and old-checkpoint resumed runs pass all 15 comparison checks:
  accepted step-one mesh/particle payloads, history/V/geometry/bulk fingerprints
  and fresh statistics match. Original restored-state assertions pass. Step-two
  solver traces match exactly and retain the known nonconvergence (exit 1).
  There is no accepted step-two field state to compare; no assertion, tolerance
  or parameter was changed to obtain convergence.
- Four short legacy/automatic BP3 runs on one/two ranks and an automatic
  one-to-two-rank old-checkpoint continuation pass: 405 field groups exact,
  30 cache/work comparisons exact, including solver decisions. Timings are
  excluded. Cohesive/evolving and mature/frozen paths are covered.
- No 3D simulation, full 200-km BP3 run or long production trajectory was run.
  The prior cross-rank cohesive observer limitation remains separate and was
  not repaired. Nonempty prescribed-rate reapplication remains covered by the
  unchanged regression; the short BP3 fixture reapplies empty maps.

Qualified candidate: `build-refactor-r3b/aspect-r3b-qualified`, SHA256
`fbd3a74e0238c0fcf5026c9ec8b554f5296302fe14f3d0de4d986096d2c109db`.
`qualification.json` and `qualified-source-hashes.json` preserve its identity.
This pass is saved in its own local commit for future refactoring. A separately selected R4a solver-method
relocation is the proposed next task; no R4 implementation is included.

## Reproduction

With the qualified reference build/plugins/checkpoints and captured entry
snapshots available, run from the worktree root:

```sh
python3 benchmarks/reconstructed_fault/refactoring_r3b/build.py
python3 benchmarks/reconstructed_fault/refactoring_r3b/compile_separate.py
python3 benchmarks/reconstructed_fault/refactoring_r3b/stage_inputs.py
bash benchmarks/reconstructed_fault/refactoring_r3b/run_checks.sh reference
bash benchmarks/reconstructed_fault/refactoring_r3b/run_checks.sh candidate
python3 benchmarks/reconstructed_fault/refactoring_r3b/verify_source.py
python3 benchmarks/reconstructed_fault/refactoring_r3b/compare_lifecycle.py
python3 benchmarks/reconstructed_fault/refactoring_r3b/compare_cohesive.py
python3 benchmarks/reconstructed_fault/refactoring_r3b/compare_bp3.py
python3 benchmarks/reconstructed_fault/refactoring_r3b/qualify.py
```

Logs refuse overwrite; use fresh output/evidence names for repeated runs.
Generated snapshots, inputs, outputs and checkpoints are ignored and retained
locally. The suite deliberately continues past known Stage-J failures: individual
exit records and comparers, not the shell suite exit status, determine its verdict.
`extract.py` records the mechanical extraction from saved entry source/header;
it is provenance, not a general source transformation tool. The standalone
compile command is recorded in `evidence/separate-command.json`.
