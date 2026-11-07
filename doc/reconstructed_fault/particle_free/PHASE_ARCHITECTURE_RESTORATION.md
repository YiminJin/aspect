# Phase architecture restoration and freeze

## Selection and source identity

The user withdrew the shared-assembler architecture on 6 November 2026 and
selected restoration with only a short native control and cheap checks. Entry
root was `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, HEAD `3a21d59cf`. There were no subsequent commits,
tracked manual changes or staged changes. Untracked work was left in place.

Normal revert **`319b10b94`** reverses only `3a21d59cf`. Its complete tracked tree
is identical to **`84f57503c8fa63196e073eec330101b70ff2b966`**. The preceding
DGQ2 commit `e42919227` and native-integration commit `84f57503c` remain ancestors;
no reset, history rewrite, checkout or checkpoint conversion occurred. The
subsequent documentation reconciliation adds this report, the historical report
and freeze notices; source, CMake, tests and benchmark support still match
`84f57503c` exactly.

The restored implementation retains ordinary semantic composition registration,
initial/boundary data, simulator-owned once-per-attempt phase transport,
registered `PhaseTransport` cell assembly, phase-specific mapped transport/SIPG
face loops and the small shared scalar Nitsche kernel. Explicit SSPRK3/BP,
local mass inverses, cached operators, fracture obstacle/KKT solve, transaction
ownership and checkpoint meanings remain those of native integration. Ordinary
advection retains its original implicit assembly. Full shared scalar face
assembly and its spatial adapters are no longer active.

## Review preservation

Annotated tag **`review/shared-scalar-dg-3a21d59cf`** preserves the complete
shared-assembler source. Its annotation identifies the executable and local
archive manifest. No push was requested or performed.

Local review archive: `build-particle-free/review-shared-scalar-dg-3a21d59cf/`.
It contains copies of the five tested executable/operator/plugin artifacts,
shared-assembler evidence, logged-run records, original report and runtime
library listing. `preservation.json` hashes the archive and **1,632 original
files** retained in place. Manifest SHA256:
`b6e0e296ce6bcaa1121fddd88818099ed8e2fbdb17745187487c5e6b70afd70c`.
The original `build-particle-free/shared-scalar-dg/` build, its outputs,
`benchmarks/reconstructed_fault/particle_free/evidence/shared_scalar_dg/`,
`shared-dg-*` logged-run records and required runtime libraries remain available.
Original and archive hashes were rechecked after testing. These generated
artifacts remain local and outside the source commit; a Git tag alone does not
contain or remotely back them up.

Shared executable SHA256:
`8777e0917d794989990896c5a48c950797f6ffaddd32e1be33d2f644ad7fc9f5`.
The [shared-assembler report](SHARED_SCALAR_DG_REPORT.md) is retained with a
historical banner; its original findings, failures and performance limitations
remain evidence for that review revision, not claims about current architecture.

## Bounded restoration verification

The restored tree matches all **950 source/include/unit hashes** in the retained
native-integration identity ledger; all **seven binary/plugin hashes** match.
Thus the existing native-integration executable is a verified matching binary,
not an assumed HEAD build. No rebuild was needed or performed. Executable:
`build-particle-free/native-phase-integration/native/aspect-release`, SHA256
`6ea374c5b9b50a20af00fa6e2c365df61d7ac59ed33ad26c7b338d7bb7ba67d9`.

Only four existing checks ran, sequentially, with one numerical thread per rank
and at most four MPI ranks. Total launch-inclusive runtime was **32.29 s**.
Each used `benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py`,
with the label below and a 60 s timeout (120 s for the coupled control), followed
by `/opt/openmpi/5.0.6/bin/mpirun -np N EXECUTABLE ARGUMENTS`:

| Label suffix after `phase-restoration-` | Ranks | Executable beneath `build-particle-free/native-phase-integration/` | Arguments | Result |
|---|---:|---|---|---|
| `native-inflow` | 4 | `native/aspect` | `build-particle-free/phase-architecture-restoration/inflow/input.prm` | One step to .0125, one history/geometry publication, zero retries; 19.66 s |
| `units` | 1 | `native/aspect` | `--test` | 155 cases / 185,551 assertions pass; 7.08 s |
| `affine-constant` | 1 | `operators/dgq2_phase_checks` | `2 constant 1 .05 .1 0 affine` | Constant L2 3.21e-16; maximum step balance 8.05e-16; 1.67 s |
| `bilinear-sipg` | 2 | `operators/dgq2_phase_checks` | `2 elliptic 1 .025 .1 1 bilinear` | L2 .00372610; energy identity 2.89e-15; symmetry 1.17e-15; 3.88 s |

The coupled input and `fault.txt` are copied from the existing
`transported-objective/stage-c/material-runs/layered-phase-integration-inflow01/`
under `build-particle-free/`, changing only that run directory in paths. Physical
inputs, mesh, timestep, initialization, nonzero time-dependent phase inflow,
matched plugins and all solver criteria are unchanged. The fresh run's **15
physical/diagnostic CSVs are byte-identical** to the original; storage/RSS CSVs
are excluded from the physical comparison. Phase material balance is
-2.32e-17 and transport uses two substeps. Bilinear nonmatching SIPG independent
residual/tangent errors are 1.41e-11/6.63e-12. The numeric comparison screen was
2e-12 relative to max(1, magnitudes); actual physical differences are zero.

Local evidence is under
`benchmarks/reconstructed_fault/particle_free/evidence/phase_architecture_restoration/`:
`identity.json`, `runs.json`, `native-comparison.json`, `preservation-check.json`.
Logs use the `phase-restoration-*` prefix in the existing logged-run evidence
directory. All commands and exit codes are recorded there. No complete
validation matrix, new refinement study, loaded-growth replay or new
rejection/restart matrix ran. Existing native-integration lifecycle evidence
remains applicable to the identical restored source; these short checks do not
claim additional physical or production qualification.

## Frozen boundary and next selection

For the next numerical-validation stage, **freeze the phase architecture** at
the restored native-integration contract in [DGQ2_PHASE_FIELD_DESIGN.md](DGQ2_PHASE_FIELD_DESIGN.md).
Keep the present spatial paths, degrees, quadrature, SSPRK3/BP, obstacle and
acceptance rules, constitutive conventions and checkpoint meanings fixed.
Shared-assembler adoption, further architecture work, quadrature reduction and
its proposed cost profiling are not active tasks. Numerical validation should
measure accuracy of this fixed implementation before selecting further changes.

Reverse translation, finer coupled mesh, full trajectories, production/server
runs and other deferred long experiments remain deferred until explicitly
selected. A matched-subcycle frozen-velocity accuracy comparison remains a
possible small next experiment, not an authorized queue. No jobs remain active.
Mapped complete stress history, mature moving growth, cohesive-tip and endpoint
energy issues remain unresolved. Preserve failed evidence and inherited losses:
**39/45 native-pilot frames and earlier temporary logs** have not been recovered.
