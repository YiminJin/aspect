# Bounded native source-history qualification

Read [the report](../../../../../doc/reconstructed_fault/particle_free/BULK_SOURCE_HISTORY_REPORT.md)
for equations, failure operands, physical results and limitations. `assessment.json`
and the two figures are the main physical comparison. Small final-build checks
are `distorted-checks.json` and `affine-checks.json`; the former records the
44+4 restart and actual rejection/retry in the resumed tail. The uninterrupted
distorted control does not request rejection. Its accepted data still match
that restarted/rejected sequence bitwise. `roundoff-diagnosis.json` independently
reevaluates the failing transition cell at 70 decimal digits.

Entry: f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work, branch
pf-rsf-particle-free. The exact native/DSO identities are in `final-binaries.json`.
The old qualified executable and plugins were preserved. Native build used the
unchanged public-header/PCH/original objects, rebuilding both private-header
consumers (unity 19 and 53), and linking a separate executable. The later gradient
arithmetic correction rebuilt unity 19. Both matching diagnostic DSOs were built
separately. `build-records.json` records the final unity-19 compile/link; the
archive also retains earlier complete commands/source and failure provenance.
This is not a clean-build or production-BP3 qualification.

`logged/*.json` contains actual commands, environments, timeouts and exit codes.
The original pure-DG failure remains in `../bulk_orientation/`. The first native
complement also failed at step 47; its failure-only replay operands and the
pre-correction source are retained. V1 bulk restart attempts fail explicitly;
no conversion or history erasure is performed. A diagnostic postprocessor's
initial missing status-key failure is preserved alongside the corrected material
version rejection. Two initial analysis-reader errors (overbroad phase filename
pattern and the residual column name) were corrected before these results;
no native input or scientific tolerance changed for them.

All native runs were serial, one thread, fixed mesh and geometry, no transport,
frozen diagnostic particles. Gauss8/source3 and dt=.0025 s remain the selected
orientation settings. Example direct native invocation **in a fresh output
location**, with the appropriate output-directory override in the input:

```sh
env ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1 \
    ASPECT_BULK_VERIFY_SOURCE_HISTORY=1 ASPECT_BULK_RECOVERY=1 \
    ASPECT_BULK_CHECK_TANGENT=1 OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
    build-particle-free/bulk-source-history/native/aspect-release \
    benchmarks/reconstructed_fault/particle_free/bulk_cohesive/source_history.prm
```

Set `ASPECT_BULK_SOURCE_AFFINE=1` only for the small affine control. The two
orientation runs instead use their saved `orientation-*.prm` inputs with
`ASPECT_BULK_SOURCE_EXPORT=1` and tangent/freeze flags. They stop at exactly
.14500000000000007 s. Do not overwrite existing run directories. No server job,
full event, additional mesh or parameter search was launched.

`reference.tar.xz` is a **data/source-delta replay archive for this worktree's
current compatible bulk implementation**, not a complete ASPECT/server source
package. It contains complete small verification exports, the selected original
QP histories needed for common-kernel comparison, one accepted native checkpoint,
actual inputs, source/build provenance and the saved failure operands. It omits
redundant rolling checkpoints and large intervening dense exports. Original raw
runs/checkpoints remain under `build-particle-free/bulk-source-history/`.
`archive-manifest.json` hashes every member; `archive.json` hashes the archive
and records successful verification of every member. Extract only in an empty
replay tree (or verify matching files first); archive paths are relative to the
worktree root. Data analysis needs Python, NumPy, SciPy, Matplotlib and mpmath.

From a populated replay worktree root, independent checks are:

```sh
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_source_history.py \
  build-particle-free/bulk-source-history/runs/distorted-centered /tmp/source-distorted.json \
  --before build-particle-free/bulk-source-history/runs/before-centered \
  --after build-particle-free/bulk-source-history/runs/after-centered
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_source_history.py \
  build-particle-free/bulk-source-history/runs/affine-centered /tmp/source-affine.json
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_source_failure.py \
  build-particle-free/bulk-source-history/runs/roundoff-replay /tmp/source-failure.json
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/assess_source_orientation.py \
  /tmp/source-orientation
python3 benchmarks/reconstructed_fault/particle_free/bulk_cohesive/plot_source_orientation.py \
  /tmp/source-orientation
```

The current geometry is a matching-cell trace problem. T_full is independently
evaluable off the native QPs; local cohesive histories and energy are checked on
their original constitutive QPs. No off-QP cohesive state is manufactured. The
full tensor metric includes zz=-(xx+yy) and both xy contributions. Common kernels
use physical JxW, never point counts, and differ from native nodal test functions.
Phase-only Gauss24 integration is a diagnostic of those tests, not remapping of
local histories or general mechanical quadrature qualification.

This closes the representation/lifecycle task and completes the one orientation
comparison. Broad driving is robustly oriented; first nodes, symmetry and growth
amplitudes are not spatially qualified. Physical-energy screens still fail and
strict_reference_match_passed remains false. No seed/law/default/BP3 change,
propagation, AMR, MPI expansion, transport or moving geometry is authorized here.
