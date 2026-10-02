# R5b1: surface assembly implementation boundaries

Reference: accepted R5a2 `d29115ada` after R5a1 `c3ce532be`.
Qualified executable: `build-refactor-r5a2/aspect-r5a2-qualified`, SHA256
`8f696bedb006147c564f4146b96f765e88b9cd11f68f3fb120c5f0c222a1bf27`.
All accepted source/artifact/protection/checkpoint/local hashes checked before
committing R5a2. The separate scientific worktree remains untouched.

## Pre-edit inventory and selected movement

| Operation | Inputs / measure / outputs / retained responsibilities |
|---|---|
| Dispatcher in `assemble_surface_system` | Explicit bulk state, absolute nodal V and Jacobian flag; bulk-work selector chooses existing alternate path, otherwise particle path. Keep dispatch in surface_system.cc. |
| Particle/domain assembly | Ordered locally owned parent associations; full admitted-domain quadrature weights; bulk FE gradients/physical pressure/temperature/current and previous phase sampled at parent points with RPE averaging; particle Maxwell stress and chemistry remain P0. Surface Q1 inputs/material response vary across domain quadrature. Return reduced R/K/mass/weak terms and local coupling points. Move backend body unchanged to surface_system_particle.cc behind one private method. |
| Bulk-work assembly | Each owned physical Stokes QP once; cached source map including qualified endpoint continuation; FE phase, temperature, pressure, strain, composition and incoming stress (or existing retained-history callback). Weight JxW*chi, zero skipped, frozen previous phase=current. Return R/K/mass and QP coupling data; optional filter and diagnostic preparation remain in this backend. Move complete existing method unchanged to surface_system_bulk_work.cc. |
| Residual-only evaluation | Calls dispatcher with false and returns residual by value. Does not publish history or linearization; can prepare geometry and mutate the existing filter-factor cache. Keep in lifecycle file. |
| Linearization | Reset old inverse, increment generation, clear diagnostic; assemble, invoke diagnostic observer, then construct/factor candidate, build RPE/optional sparse G, publish diagnostic and candidate. Failure leaves old inverse invalid; observer runs before candidate factorization/publication. Keep entire method unchanged. |
| Full/restricted solves | Existing FaultSurfaceDirect factors; restricted solve owns principal-free-block factors and borrows surface owner/generation; active solution entries exactly zero, stale generations rejected. Keep original operations/helper. |
| G/reference G | Published coupling positions/coefficients and RPE match K sampling; explicit sparse path only when enabled and unfiltered; physical-pressure scaling remains caller responsibility. Shared-face multiplicity and missing-point behavior unchanged. Keep operations in lifecycle file. |
| Norm/filter/diagnostics | Consistent mass/free-set norm stays; filter exact operator reuse and current RHS stay in bulk backend, with existing FaultNormalFilter. Raw/projected/Helmholtz including zero length stay distinct. Optional particle audit and native-QP/line observers remain at original sites. No diagnostics cleanup. |

Both paths retain their distinct loop order, pressure conventions, measures and
reductions. They do not become a generic integration framework. Bulk mode retains
mature/2D and legacy straight-fault admission, plus the separately qualified
automatic-completion exception. Particle guards still reject filtering and the
retained-stress benchmark path before its original timer starts.

## Private records, owners and lifetime (pre-edit contract)

| Record / resource | Owner, preparation, publication and expiry |
|---|---|
| SurfaceAssembly | Private nested scratch record; temporary returned by each backend. Replicated arrays, local coupling samples, optional shared diagnostic/filter factors. Move its single complete definition to source-private surface_system_internal.h. No public record/accessors. |
| SurfaceLinearization | Private nested persistent computational record, unique_ptr owned by canonical simulator surface helper. Keep complete definition and out-of-line constructor/destructor in surface_system.cc; never checkpointed or merged with scratch. |
| Full inverse/RPE/sparse G | Candidate constructed after assembly/observer; all fault factors and G lookup/matrix completed before pointer publication. Borrowed residual views expire on reset/replacement/destruction. |
| Restricted inverse | Owns free factors but borrows surface owner; must not outlive owner. Generation check rejects subsequent linearization/filter reconfiguration. Preserve existing enable_bulk_work_measure reset behavior; no new invalidation policy. |
| Filter factors | Mutable surface-owned cache keyed by exact assembled operators/length. Residual trials may replace cache; published linearization retains shared immutable factors. No constitutive state or checkpoint. |
| Diagnostic snapshot | Candidate scratch captures local owned samples and reduced moments; old snapshot cleared at linearization start, observer before factorization, snapshot installed with successful candidate. Setting diagnostic configuration clears prior snapshot. |
| Physical state / acceptance | Manager/particles store V/properties/history, M4 computes constitutive values/history updates, solver/simulator M5 accepts/rejects steps. Assemblers perform no history commit. B and condensed operator stay with existing owners; no B=G-transpose assumption. |

Selected implementation: two backend files, one private assembly declaration,
and one narrow source-private record header. Move no other method or helper.
Use explicit 2D/3D member instantiations and the existing independent-build
mechanism for the two new TUs, preserving baseline unity grouping. Existing
surface_direct_internal.h, normal_filter_internal.h and sparse_coupling stay.
This is R5b1 boundary separation only; no R5b2 phase extraction.

## Focused verification planned before editing

Byte-exact backend/record movement and otherwise unchanged lifecycle/callers;
private-header/declaration audit, full Release build plus independent three TUs
and unique 2D/3D symbols. Existing surface dynamic/adiabatic/rate-dependent
residual/K/G/restricted/stale checks on one/two ranks; explicit/reference G case;
normal-filter free-equation fixture on one/two ranks (bulk work raw/projected/
Helmholtz, consistent derivatives and raw preservation); singular-factor failure
invalidation; short coupled pressure/history and accepted-update rollback.
Use small existing automatic-completion BP3 endpoint/coupling comparisons where
needed. Compare deterministic emitted fields/actions, decisions and counters
at matched ranks/stacks, excluding times/paths. Preserve all settings and known
failures. Repaired frozen AMG/GMG evidence is retained; no solver unification,
new physics, long BP3/BP5 campaign or 3D runtime qualification is intended.

## Build dependency discovered during separation

The initial independent backend build exposed an existing missing direct include
in the public surface header: NormalTractionDiagnostic::ParticleSample uses
deal.II's types::particle_index, defined in particles/property_pool.h. Compiling
the saved unmodified header independently with baseline flags reproduces that
error. Add that direct include beside the one private method declaration; no
public API/data-layout change. The initial failure and baseline probe are kept
in build.log and baseline-header-independent.log. The successful rebuild is
build-direct-type-include.log. All three independent TUs compile, and baseline
unity groups remain identical. Candidate plugins are rebuilt after this include
addition. No scientific or numerical correction accompanies it.

## Completed verification and limitations

Candidate: `build-refactor-r5b1/aspect-r5b1-qualified`, SHA256
`fd8c03ab1f1f4e363e2b69cb69d8517a35c648c5a682308004aee4453ed4910c`.
The two backend bodies and private record move byte-for-byte. Eight source
checks and independent original/new TUs pass; exactly four backend definitions
and two dispatcher definitions cover 2D/3D. The pre-edit table above remains
valid after movement. Public API/layout and ownership are unchanged.

All 148 matched checks pass. Both binaries pass 636 inverse/filter assertions
in three cases per rank on one/two ranks. Dynamic, adiabatic and rate-dependent
particle tests, explicit/reference B/G probes, bulk-work projected/Helmholtz
finite differences with raw preservation, restricted/stale views, and short
coupled history/rollback pass. Automatic-completion BP3 matches across seven
steps: 166 CSV files (bulk/particle audits, profiles, accepted states, weak work,
completion/cache diagnostics) plus all DataArrays in 16 VTU files across the
one/two-rank cases. Four filter action CSVs and four coupled history fingerprints
also match exactly. No physical settings/tolerances or expected numeric results
were changed; timing and output paths are excluded.

The original singular test fails on both binaries because it expects the obsolete
`Failed to factor reconstructed-fault K_V block` diagnostic. The current direct
solver reports `GTTRF failed, info=1, singular pivot vertex=0`. Keep those four
original failures and the original source unchanged. The separate
`singular_current_diagnostic` probe is generated from the same fixture with only
that diagnostic guard replaced by the exact current singular-pivot text, plus
printing the caught exception. It retains the generation and unavailable-old-
inverse checks and reaches the intended terminal marker on both binaries/rank
counts (four deliberate exit-1 runs). This closes the selected failure-lifecycle
verification without hiding the original stale fixture. Its permanent correction
is the recommended next separate test-only task; no production fix was made.

The first comparison report is retained. Its other flags were absolute output
paths in statistics and asynchronous ordering of rank-local timer summary blocks.
The final comparison excludes only output paths and compares the exact multiset
of timer names/counts, while preserving trace order for solver decisions and
exact labelled per-rank CSV values. No numerical tolerance was introduced.

Qualification records 52 final build/runtime outcomes, 861 source/header/unit
files and 69 artifacts. Original tests, accepted binaries/inputs/plugins,
checkpoint sources and local temporary files remain unchanged; the authorized
CMake change is the sole excluded entry in the prior executed-artifact check.
No MPI correctness defect was demonstrated. No Debug/3D runtime, new restart,
long BP3/BP5 run or dedicated detailed native-QP/line diagnostic observer run was
performed. Detailed diagnostic code and observer timing are byte-preserved.
Accepted repaired frozen AMG/GMG and restart evidence are retained, without
repeating unrelated solver campaigns or claiming historical issues resolved.

## Reproduction

`evidence/*.json` records exact commands/environments/exit codes, with full logs.
The stack is the qualified GCC/OpenMPI/deal.II/Voro Release configuration with
`-fno-finite-math-only -ffp-contract=off`; see configure.json. The original build
failure, baseline-header failure and successful build-direct-type-include are
separate records. The first attempt to build the new supplemental target before
CMake reconfiguration is also retained; configured supplemental builds pass.

Build `plugin` separately against the R5a2/R5b1 build packages. The current-
diagnostic probe is generated in each plugin build directory; its generator,
source and artifacts are hashed. `prepare_inputs.py` creates the unchanged
physical fixtures with local output/library paths. `run_checks.py reference`
and `run_checks.py candidate` run the matrix with MPI socket access; optional
case names restrict a run. Reference coupled pressure/rollback outputs are reused
from qualified R5a2, with exactly matched recorded environments. All other cases
use fresh matched reference/candidate runs. No external worktree is written.

`compile_independent.py`, `verify_move.py`, `verify_symbols.py` and `compare.py`
record the structural and numerical checks. Saved pre-move source/header and
move-ranges.json identify the exact reference blocks. `qualify.py` checks all
final outcomes/protection manifests and freezes the candidate. Python syntax
and `git diff --check` pass. The rolling review records the result and stop.
R5b1 remains uncommitted; no R5b2 operation extraction has begun.

## Accepted R5b1 baseline

The user accepted R5b1 and requested its commit before R5b2. The commit
containing this entry records the accepted source, guidance and harness.
The immutable executable and SHA256 above remain the qualified reference.
All source/artifact/protection/checkpoint/local hashes and 148 comparisons
were rechecked before committing, without repeating runtime campaigns.
Local `doc/reconstructed_fault/refactoring/tmp/` files remain excluded.
The stale singular-fixture expectation remains a separately recorded issue;
the user's next selected task is R5b2, not that fixture correction.
