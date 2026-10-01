# Historical frozen AMG/GMG fixture repair

This is a separate test-only task after accepted R4c, before R5.
R4c is committed as `fa6013678b525b189a1d27ef08465d4a6ef263f6`.

| Role | Source commit | Immutable executable | SHA256 |
|---|---|---|---|
| Pre-R4c | `983d57e2863af798de29cb9601d33bfe3a53a6af` | `build-refactor-r4b-residual/aspect-r4b-residual-qualified` | `6ccdcf81b65ad5cbe0c949cdcd45da6332c3949354e0a034dcc830fa889fe7a7` |
| Post-R4c | `fa6013678b525b189a1d27ef08465d4a6ef263f6` | `build-refactor-r4c/aspect-r4c-verified` | `c6811cbd877ff56af9113dc2f110cfd9a8b71998ea63f8e249ff94b292eeed85` |

## Bounded correction

The old fixture selected block AMG, which omitted multigrid level ownership,
and set the obsolete `ASPECT_FAULT_GMG_HIERARCHY` flag. Select the existing
`default solver` value in the fixture instead: `core.cc` constructs the hierarchy
before `select_default_solver_and_averaging()` resolves reconstructed faults to
block AMG with the existing material averaging. No production code, interface,
obsolete switch, preconditioner algorithm or numerical parameter changes.

The observer asserts resolved `block_amg` and an existing hierarchy. Its AMG
row uses the borrowed production action; its GMG row uses an explicitly built
local-smoothing velocity cycle inside the existing assembled-A inverse and
Schur wrapper. A collective guard proves every rank actually applies that cycle.
Both solves use the same borrowed condensed operator, RHS, pressure inverse,
tolerance and iteration budget, and start from zero as before.

The historical launcher `../performance/gmg/run.py` now selects the same default,
uses the retained `libbp3_research` replay plugin, drops the dead hierarchy flag,
and writes to a new output directory. It checks backend/state/pass markers.
Its source and syntax were checked; the executed matched harness here uses the
same observer/replay source and archived physical configuration with explicit
immutable executable paths. The old deployment-layout launcher was not launched
separately, since its hard-coded build/output paths are not this worktree's
qualified artifacts.

## Preservation evidence

The observer compares exact rank-local snapshots before/after both probes:
owned bulk solution, current linearization, old/old-old solutions, serialized
manager geometry/registered surface histories/committed V, active V/prescribed
masks, particle IDs/positions/properties, RHS and original production direction.
It also compares operator applications to that direction and the RHS before
and after the probes. These are test evidence, not new checkpoint formats.
The existing state checks remain. The complete pass marker precedes the
intentional exception; no step-2 acceptance/history publication is permitted.

Rank-local snapshots, both returned directions and operator actions are saved
for exact matched-executable comparisons. CSV and bulk VTU outputs also compare
against the historical AMG-prefix runs, including mesh guards, accepted states,
Maxwell/Theta history and work diagnostics. Only VTU wall-clock comments and
probe timing/RSS columns are excluded. New snapshot/operator-audit work is
outside the measured solves; no performance claim is made.

## Reproduction

From the worktree root, run `python3 benchmarks/reconstructed_fault/frozen_gmg_repair/prepare.py`.
Configure separate plugin builds with `plugin/CMakeLists.txt` and `Aspect_DIR`
pointing to `build-refactor-r4b-residual` / `build-refactor-r4c`, then build both
with `cmake --build <version>-plugin-build -j2`. Exact configure/build commands
and environments are in `evidence/{reference,candidate}-{configure,build}.json`.
The observer compiles both dimension instantiations; runtime is the intended
2-D Q2 wide fixture on four ranks, step 2 / Newton 4.

Run `run.py reference`, then `run.py candidate`, then `compare.py` in this
directory (Python scripts accept execution from the worktree root). The runner
uses a 720-second cap and refuses to overwrite a prior log; outputs/plugins are
isolated from R4c and scientific-worktree evidence. Exit 1 is required for the
intentional stop, together with both solved rows, all preservation/backend
markers and the full pass marker **before** that exception. Exit 1 alone never
qualifies the test. The preserved `clock.csv` is byte-identical to the original
saved clock. No physical values or tolerances were changed.

`evidence/protected-hashes.json` protects production source/headers, local review
edits, all R4c evidence and both immutable binaries. `comparison.json` records
all exact comparisons and `executed-artifacts.json` records fixture/plugin/input
hashes. Large build/run artifacts remain local and ignored.

## Verified result

Both separate plugin builds and four-rank runs pass. The intentional exit is 1
in each run, strictly after the complete comparison-pass marker. All **226**
checks pass: 59 field files per matched comparison (pre/post and each repaired
run versus its historical prefix), 20 exact rank-local binary payloads, backend
identity, preservation, input changes, residuals, decisions and work counts.
The only effective PRM differences from the historical case are library/output
paths and `block AMG` to `default solver`; the resolved production backend is AMG.

| Probe | Iterations | Fresh residual | Unchanged target | Relative direction difference from production AMG |
|---|---:|---:|---:|---:|
| AMG | 17 | 0.00044068965535591382 | 0.0012258892187472356 | 0 |
| GMG | 17 | 0.00048708263397949716 | 0.0012258892187472356 | 7.5436570141764325e-10 |

Both rows, returned vectors and all non-time/non-RSS counters match exactly
between executables. All physical snapshots/RHS and both operator actions are
unchanged within each probe; published step records remain exactly `[0,1]`.
No numerical correction, new interface or production rebuild was needed.

The first comparison script incorrectly required the old failed probe's final
aggregate profile to match a completed probe. `comparison-initial.json` retains
that result. The final profile is emitted during exception cleanup **after**
the observer: the repaired GMG solve adds 17 operator applications plus one
fresh residual, and the preservation audits add four, totaling exactly 22.
The comparator now verifies that exact delta in A/B/G/inverse/other counters;
all other counters and all earlier profiles remain unchanged. This is additional
test work, not a changed production solve. Pre/post-R4c repaired profiles match
in full. An initial parameter-JSON reader needed to skip alias metadata; neither
harness correction changed or reran the successful simulations.

Limitations: this qualifies one four-rank 2-D Q2 frozen linearization and its
fresh prefix. It does not qualify long GMG trajectories, performance/memory
improvements, restart behavior or other meshes/dimensions. Existing historical
scientific limitations are unchanged. R5 has not begun; review this separate
fixture repair before selecting the next refactoring task.

## Accepted fixture baseline

The user accepted this repair. The fixture commit containing this record is
the post-R4 reference for R5; production remains `fa6013678` and the immutable
R4c executable above. All 2,315 protected entries and 101 executed artifacts
were rechecked before commit. The 226-check evidence is reused, not rerun.

- `executed-artifacts.json` SHA256: `61b6b47abcfbced8ad0154a4710b222089f8a6ec5ad1a45df0d9f67386ebd08d`
- `qualification.json` SHA256: `6a2cc36b56959a2f7988ff48f2081b70c74a96904d9d9439e0e173a45eac4c6c`
- `comparison.json` SHA256: `b557e926014518570d212d6935a89db06a59132c462bc620952315c927a27c03`
