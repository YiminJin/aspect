# R4c shared Schur construction

Reference: accepted R4b `983d57e28`, qualified executable
`build-refactor-r4b-residual/aspect-r4b-residual-qualified`, SHA256
`6ccdcf81b65ad5cbe0c949cdcd45da6332c3949354e0a034dcc830fa889fe7a7`.
Source/artifact manifests verified before editing; reference/local files protected.

## Pre-edit contract

One source-private non-template constructor helper replaces two identical choice
branches. Inputs: BFBT flag, caller-selected pressure matrix block, existing
pressure preconditioner, S tolerance, whole inverse-lumped-mass vector, velocity
block index and full system matrix. Return the existing unique_ptr to a Schur
wrapper. The caller owns its counter; all referenced data retains its existing
owner and must outlive the wrapper. Select the velocity mass block only in the
BFBT branch, since this vector is initialized only in that mode. Constructors
retain references and zero the counter; no MPI, solve, assembly or state writes.
Keep both call sites at the same point and preserve ordinary melt block selection.
No simulator.h/condensed-interface change; no change to the fault/melt rejection,
outer solvers, pressure treatment, AMG/GMG wrappers, observers or history lifetime.

Planned checks: separate Release build/link, independent ordinary/fault TU and
private header, 2D/3D symbols; existing coupled units/residual/pressure/exhaustion/
rollback on one/two ranks and legacy/automatic BP3; ordinary AMG, BFBT, melt and
expected failures; frozen four-rank AMG/GMG observer and actual GMG-Q1 case.
Reference extra fixtures and plugins use the accepted executable; candidate
artifacts are separate. Investigate baseline failures without tuning parameters.

## Evidence and reproduction

`run_logged.py` records bounded commands, environments, exit codes and logs.
`evidence/configure.json` retains the same Release compiler/library stack and
floating-point flags as R4b. `compile_independent.py` compiles the ordinary solver,
coupled solver and private header without unity/PCH. `linked-symbols.txt` records
one helper definition and both dimensions of all affected Simulator methods.
The source manifest and executable/plugin/input manifests protect provenance;
`entry.patch` preserves the approved, previously uncommitted inventory documents.

The candidate `run_checks.sh` reuses R4b's one/two-rank coupled fixtures and four
short BP3 trajectories. Their reference outputs are the accepted R4b outputs.
`prepare_extra_inputs.py` creates matched ordinary/BFBT/melt/failure and frozen
inputs; ordinary extra cases explicitly select block AMG to exercise the affected
assembled branch, preserving physical parameters and tolerances. `run_extra.py`
runs the five ordinary cases with either reference or candidate. Reference plugins
are built against the accepted R4b build; candidate plugins against the R4c build.
Simulator's public header is unchanged. All outputs remain in this worktree.

`run_frozen.py` uses four ranks, the archived clock and exact-mesh fixture copied
from the scientific worktree, and its retained `bp3/reference_200km` replay
plugin. The scientific worktree is read-only. The initial baseline attempt used
the wrong historical BP3 plugin (missing `BP3 replay complete`) and is retained
as `reference-frozen.log`. Correcting the harness plugin selection requires no
production or parameter change; subsequent runs are named `*-frozen-replay`.
The first build attempt for the added plugin target preceded CMake regeneration;
its retained failure is superseded by the successful configure/final-build logs.

`verify_source.py` proves unchanged constructor arguments, BFBT-only block access,
otherwise identical caller bodies, unchanged Schur algorithms, Simulator header
and fault/melt restriction, and preserved reference/local files. Comparators
separately cover ordinary fields/failure histories, coupled fixtures, BP3 fields
and counters, and frozen-probe coverage. `qualify.py` records explicit limitations
alongside outcomes; equal failure is not treated as a completed frozen-GMG test.

## Baseline frozen-probe limitation

The corrected archived probe reaches timestep 2 / Newton 4 on accepted R4b.
Its frozen AMG solve converges, with zero difference from the accepted direction.
GMG initialization then fails at deal.II `dof_handler_policy.cc:3827`, called
from `StokesMatrixFreeHandlerLocalSmoothingImplementation::setup_dofs()` and
`with_velocity_preconditioner()`: `construct_multigrid_hierarchy` was not set.
The fixture selects block AMG. `source/simulator/core.cc` builds that hierarchy
only for block GMG/default with local smoothing, and the archived
`ASPECT_FAULT_GMG_HIERARCHY` switch no longer affects that choice. This prevents
the old observer from adding a GMG comparison to its AMG-selected mesh.

Do not change production backend selection, physical parameters, tolerances,
assertions or hierarchy interfaces to make this historical test pass. A separate
fixture update is needed to construct the hierarchy while retaining the intended
frozen comparison. The supported actual GMG path is covered separately by the
existing GMG-Q1 case. The frozen comparison record explicitly distinguishes the
AMG evidence from the missing GMG result and terminal probe-pass marker.

## Result

Candidate: `build-refactor-r4c/aspect-r4c-verified`, SHA256
`c6811cbd877ff56af9113dc2f110cfd9a8b71998ea63f8e249ff94b292eeed85`.
All nine source checks, independent/full builds, 31 coupled and 22 ordinary
comparisons pass. Units pass 20,149 assertions/16 cases per rank on one/two ranks.
The four legacy/automatic BP3 runs match in 372 exact field/history groups and
28 counter/decision checks. Ordinary melt, BFBT and actual coupled GMG pass.
Both ordinary failure histories and the coupled exhaustion/rollback outcomes
retain their original behavior. No candidate-only numerical difference occurred.

Both corrected frozen-probe runs fail at the same pre-existing hierarchy guard.
Seven matched checks confirm identical pre-observer decisions/work counters and
frozen AMG numerics: 17 iterations, fresh residual `0.00044068965535591382`, target
`0.0012258892187472356`, zero direction difference. Frozen GMG and the final probe
success assertion were not reached. This coverage gap remains explicit in
`evidence/qualification.json`; the supported GMG-Q1 check is a separate pass.

There are 38 selected build/runtime records, including six intentional ordinary/
exhaustion failures and the two pre-existing frozen failures. Initial harness
setup attempts remain separately documented above. No new ordinary GMG/direct,
restart, explicit fault/melt-rejection runtime or production campaign was run.
The existing fault/melt assertion is source-identical. The eight-cell BFBT case
is the sole 3D runtime check. All R4c changes remain uncommitted for review.

## Accepted R4c baseline

The user accepted this pass. The R4c source/documentation/harness commit containing
this record is the post-R4c baseline; its parent is accepted R4b `983d57e28`.
Use the immutable `build-refactor-r4c/aspect-r4c-verified` binary and hash above.
Before commit, all 5,910 candidate source entries, 42 candidate artifacts and
33 reference artifacts match. The only old protection-manifest difference is
the user's local `refactoring/tmp/R4c_review.md`; preserve it outside the commit.
The earlier qualification and hierarchy limitation remain unchanged.

Manifest SHA256 fingerprints:

- `candidate-source-hashes.json`: `534c34dd6aa5f37ca97078d74d85be74c5816deba9e0d63644d0f223563afe56`
- `candidate-executed-artifacts.json`: `d031d5c4e9f0b67dc43564dedd28a81b67bb2cedaa86935dbdcceedd13823acc`
- `qualification.json`: `9ee31dcd6c43943ea74108c49b237114965e7107c115a9165001e017ccc720e5`

The separately authorized frozen-fixture repair is not part of the R4c source commit.
