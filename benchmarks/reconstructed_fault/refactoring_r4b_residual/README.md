# R4b second subpass: private coupled residual evaluation

## Accepted R4b baseline

Both R4b operations and the recommendation to retain the driver without a
whole-iteration helper are accepted. The commit containing this baseline record
saves their production source and verification scripts over R4a `0c7ed1a0b`.
Use `build-refactor-r4b-residual/aspect-r4b-residual-qualified`, SHA256
`6ccdcf81b65ad5cbe0c949cdcd45da6332c3949354e0a034dcc830fa889fe7a7`.
All local qualification records remain unchanged; their uncommitted revision
labels describe the time of execution, not an older source baseline.

Before commit, all 5,910 source entries and 21 executed executable/plugin/input
entries matched. Local evidence manifest SHA256 fingerprints:

- `evidence/candidate-source-hashes.json`: `32d05d9b9a1d74d1d4bb0904c2eb0f8e2051983e9908874466e4a9e915b8e8dc`
- `evidence/executed-artifacts.json`: `0350bdd7969f47f99cd7b15f8e65d7a39408397729e53f46b0471ebdd9b8b881`

The 20 expected build/runtime outcomes, 31 focused checks, 372 exact field groups
and 28 counter/decision checks below remain the qualification. No new runtime
campaign was run for acceptance/commit. The following sections retain the
original extraction provenance and verification procedure.

Reference: R4a commit `0c7ed1a0b` plus the accepted, still-uncommitted first R4b
condensed-solve extraction. The entry diff is saved in `evidence/accepted-entry.patch`.
All entries of its qualified source manifest and executed-artifact manifest
match before edits. Qualified executable:
`build-refactor-r4b-linear/aspect-r4b-linear-qualified`, SHA256
`9ff850bc7d4060ab75ee80dfdd44dc09b94c7862b8bfcf448a653ecdc17cd30d`.
The first-subpass artifacts and user-local `refactoring/tmp/` files are protected.
Candidate builds/plugins/outputs use separate directories. No new commit is
requested in this subpass; distinguish its diff from the accepted entry patch.

## Contract recorded before editing

Extract the existing `evaluate_coupled_residual` lambda into one private
Simulator member in `source/simulator/solver/reconstructed_fault_stokes.cc`.
Move its local result type to a private nested declaration/definition; retain
its two fields and order. No new state owner, framework or persistent scratch.

| Aspect | Existing contract |
|---|---|
| Explicit inputs | Physical bulk vector (already lifted/normalized by caller), absolute nodal slip-rate vector |
| Result | Existing square-root sum of squared owned velocity/pressure RHS norms, and the surface residual including its mass data |
| Mutations | Open manager trial, set absolute V; set current_linearization_point; disable Newton matrix and preconditioner rebuild flags; choose matrix rebuilding from prescribed-velocity presence; assemble system_rhs and any existing matrix/cache work; roll back temporary V |
| Non-committing scope | Current/committed V, production bulk solution and constitutive histories are not published. Assembly controls, current_linearization_point and RHS are NOT restored per call; the driver/audit callers retain their existing restoration responsibilities. |
| Collectives/order | Begin/set trial, bulk assembly, velocity norm, pressure norm, surface evaluation/reductions, trial rollback. On an exception after trial opening, perform the same guarded rollback and rethrow. No new catch, RAII cleanup, reduction or validation. |
| Dependencies/lifetimes | Existing manager/surface-system instances, homogeneous constraints, pressure scaling, prepared frozen histories and valid geometry/projection state; caller's input vectors live throughout the synchronous call. Result owns its returned surface data. |
| Callers | Initial residual, zero-velocity reference residual, audit-channel evaluation, line-search trial residual, nonzero-V audit probe; retain all five call sites and their surrounding order. |

The driver still prepares histories, applies physical constraints/pressure
normalization, builds linearizations/active sets, accepts trials, publishes
convergence and restores the whole solve on failure. Audit matrix/RHS restoration
remains in the existing callers. The accepted condensed-solve helper is unchanged.
No optional iteration helper or R4c work is selected.

## Selected verification

Fresh candidate Release/link, independent driver compilation and 2D/3D
instantiations, with candidate-built plugins. Reuse the first-subpass reference
outputs for one/two-rank residual consistency, pressure/history, accepted-Newton
rollback, expected linear exhaustion, and condensation/Stage-I unit cases.
Compare four short BP3 legacy/automatic trajectories at matching ranks exactly,
including histories, solver decisions and cache/work counters. The existing
small GMG case also checks the shared residual path on that backend. No new
numerical test or changed parameter/tolerance is required by this extraction.
Ordinary/restart evidence is reused; no Debug/3D simulation or production campaign.
Keep the existing Stage-J/cohesive limitations separate.

## Implementation and reproducibility

`Simulator::evaluate_reconstructed_fault_coupled_residual()` is private. Its
`ReconstructedFaultCoupledResidual` result has the original bulk norm and surface
residual fields, defined only in the implementation file. Existing canonical
manager/surface objects remain the storage owners. The declaration uses the
existing `ReconstructedFaultVector` alias; there is no new include or public API.
The pre-edit contract above still applies after extraction. In particular,
trial opening remains outside the existing try block, and the same guarded
rollback/rethrow remains inside its catch. There is no new exception guarantee.

`verify_source.py` compares the captured accepted source to the candidate:
unchanged body tokens/order except the result type spelling, identical result
fields, otherwise byte-identical driver with five retained calls, byte-identical
accepted condensed solve, private-only header additions, only two production
files changed, and protected reference/local hashes unchanged. The incremental
diff is `evidence/second-subpass-source.patch`; the ordinary Git diff also includes
the accepted, uncommitted first subpass. Historical review text is retained.

`evidence/configure.json` records the matching Release stack/flags. The driver
compiles independently without a forced include/PCH (`independent-final.json`),
then the separate CMake build links it. `plugin/CMakeLists.txt` builds the existing
fixtures and BP3/cache observers against that candidate. `prepare_inputs.py`
copies the 13 accepted inputs with only artifact/output path substitutions;
`record_artifacts.py` freezes their hashes with the executable and seven plugins
before execution. From the repository root, `run_checks.sh` runs the bounded
checks through `run_logged.py`; MPI needs permission to open local sockets.
Each record retains the exact command, environment, duration and exit status.

`compare_checks.py` and `compare_bp3.py` compare the accepted first-subpass outputs
at matching rank counts. `qualify.py` requires expected outcomes, exact numerical
comparisons, unchanged manifests, candidate plugin paths and exactly one 2D/3D
definition of each of the three affected Simulator operations. It freezes the
candidate as `build-refactor-r4b-residual/aspect-r4b-residual-qualified`.

## Result

All 20 selected build/runtime invocations have their expected outcomes, including
two deliberately failed exhaustion cases with original budget/rollback. Fresh
Release/link, independent compilation, seven candidate plugins and exactly one
2D/3D definition of each operation pass. All eight source/protection and 31
focused comparisons pass. Units report 20,149 assertions/16 cases per rank on
one/two ranks. All four short BP3 trajectories match in 372 field/history groups
(maximum difference zero), 24 cache/work checks and four solver-decision checks.
Recorded runtime environments also match. No tolerance or parameter changes.

See `evidence/qualification.json`, `source-verification.json`,
`focused-comparison.json`, `state-comparison.json` and `lifecycle-comparison.json`.
Candidate SHA256: `6ccdcf81b65ad5cbe0c949cdcd45da6332c3949354e0a034dcc830fa889fe7a7`.
No new scientific defect was found. Historical pressure/cohesive/observer
limitations remain separate. This subpass is finished and uncommitted for
review; further iteration-helper or R4c work requires selection.
