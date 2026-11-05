# Fault tensor/history readability refactor

Starting state: current worktree `aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, revision `47c2aae3be15d3ff60aab2a818c2f6e179077702`.
Tracked source was clean. The six working binaries matched the preceding
architecture evidence; they were copied before rebuilding. The 51 previously
protected artifacts plus those six copies are recorded in
[`entry.json`](../../../benchmarks/reconstructed_fault/particle_free/evidence/code_readability/entry.json).
No checkout, reset, fetch, production job or checkpoint conversion was used.

## Implemented boundaries

- `HistoryDiscretization::TensorField` owns numerical Q2 space/coefficients only.
  Material `DGHistory` composes primary tau or T, the representation tag, and
  optional independent numerical R. It no longer recursively owns another
  constitutive history. The bulk-cohesive owner uses raw numerical fields and
  retains its separate physical interpretation.
- Tensor advection has its own header. Numerical mass projection receives an
  evaluated cell field; `project_material_tensor` explicitly samples the supplied
  profile and accepted velocity. The same quadrature, whole-product projection,
  component order and local accumulation remain.
- `FEStrainState` groups `bulk_history`, `objective` and `observations` by lifetime.
  The native mixed vector's velocity/profile meanings are documented, not hidden
  behind a phase-only name. `FEStrainUpdate::prescribed` distinguishes the retained
  control's interval start from `MaterialHistoryInputs::end_time`.
- Fault-dependent point observations moved from numerical `Space` to shared
  material `observations.tensor_points`. State copies retain sharing, geometry
  invalidation and the mechanical sampling mutex scope. Numerical inverse-mass
  and source-roundoff geometry stay in `Space`.
- Normal rays, projection matrices, geometry identity and integration cell maps
  remain together: they are built together and separating each into a new owner
  would add navigation without a lifetime boundary. No new timestep driver,
  solver, plugin hierarchy or callback pipeline was introduced.

## Reading the code

| Before | After / entry point |
|---|---|
| Numerical tensor recursively mixed representation, residual and fault cache | [`tensor_field.h`](../../../include/aspect/material_model/history/tensor_field.h), [`tensor_history_internal.h`](../../../include/aspect/material_model/phase_field_fault/tensor_history_internal.h), [`fe_strain_internal.h`](../../../include/aspect/material_model/phase_field_fault/fe_strain_internal.h) state those ownership boundaries directly. |
| Numerical projection discovered phase and sampled velocity implicitly | [`material_tensor_projection.cc`](../../../source/material_model/phase_field_fault/material_tensor_projection.cc) owns material sampling; [`tensor_projection.cc`](../../../source/material_model/history/tensor_projection.cc) owns mass/RHS integration. |
| Dense nested loops and unexplained T/R arithmetic | [`stress_reaction.cc`](../../../source/material_model/phase_field_fault/stress_reaction.cc) names coefficients/localized rate, separates loops, and explains whole-field projection and the correction. [`stress_law.h`](../../../include/aspect/material_model/phase_field_fault/stress_law.h) gives the direct physical law without auxiliary reconstruction. |
| Attempt times, rollback fields and cache sharing were implicit | [`material_history_internal.h`](../../../include/aspect/material_model/phase_field_fault/material_history_internal.h) documents each member; the [public/private declarations](../../../include/aspect/material_model/phase_field_fault.h) describe borrowed inputs, mutation and collective participation. |

Reader check: direct tau lives in `bulk_history.tensor->primary`. Its update
uses the accepted parent/velocity, staged profile/geometry, endpoint bulk/slip
iterate and sampled Maxwell coefficients. `stage_material_geometry` binds the
private candidate; the fixed transported/rotated anchor reaches mechanics through
physical-stress access and the projection correction. The ordinary simulator's
pending endpoint callback publishes once or calls `finish_dg_material_step(false)`
to restore the parent. Friction changes belong in `rheology/fault_friction.cc`,
numerical transport in `history/tensor_advection*.cc`, and fault observations in
`physical_stress.cc` and material geometry/normalization. The updated
[reading guide](MATURE_MOVING_FAULT.md) follows these same entry points.

## Style, review and verification

The scope is all 41 incoming primary files, now 44 after replacing one header
with numerical storage/advection and material interpretation headers plus the
material projection adapter. The explicit file list and token review are in
[`evidence/code_readability`](../../../benchmarks/reconstructed_fault/particle_free/evidence/code_readability).
AStyle **2.04** is used through the actual functions from `contrib/utilities/indent`
and `astyle.rc`, including nested-template and whitespace preprocessing, with
four workers. Compressed controls were expanded without changing tokens; an
additional operator-spacing pass used the same AStyle version. Formatting-only
changes are committed separately as `0eab303e9`. The local
`mechanical-format.patch` and `mechanical-readability.patch` snapshots are also
preserved, without duplicating them in Git.
Token review is an audit aid, not a substitute for compiling/running the code.

| Check | Result |
|---|---|
| Core and matching libraries | Core, four native checker DSOs, BP3 support and ten retained checker targets build. Eleven affected source/checker translation units also compile independently. |
| Existing unit suite | 183170 assertions / 152 cases pass (7.57 s). |
| Seven matched moving controls | Direct/separated constant/layered, default, forced rejection and existing-checkpoint restart: zero bulk/surface, tensor L2 and work-traction differences; clocks, publication counts and solver iterations unchanged. |
| New checkpoint lifecycle | Fresh two-rank prefix restarted on four ranks through the same two endpoints: bulk max 8.88e−16, surface max 1.07e−14 versus uninterrupted run. Prefix is bitwise identical. |
| Rejection versus matched accepted schedule | Bulk max 1.58e−15 and surface max 1.24e−14 against the retained `true-cutback-control`; original criteria unchanged. |
| Stationary, serial support, ordinary non-fault | Six stationary CSVs and non-fault statistics byte-identical. Limiter/stabilization numerical values identical, excluding timer columns. |
| Explicit source/projection inputs | Existing analytic source fixture passes: within-DG remainder <=1.03e−16; velocity/slip finite-difference errors <=3.65e−10 / 1.09e−11. Approximation/projection remainders remain, not relabeled as errors fixed by refactoring. |
| Style/documentation | AStyle 2.04 idempotent over 44 files; whitespace check clean. Doxygen 1.18.0 with existing `doc/options.dox` and `doc/filter`, scoped inputs/output, reports zero source warnings. Old configuration-option warnings remain; no full manual/network regeneration. |
| Preservation | All 57 protected hashes unchanged. No server/production runs. Native simulations were sequential, using at most four ranks; builds used at most four workers. |

Commands, hashes and detailed numbers are in `verification.json`,
`final-provenance.json`, `equivalence.json`, `auxiliary-equivalence.json`, and
`independent-compile-commands.json` in the evidence directory. Seven final
comment/whitespace-only file edits are explicitly identified in
`postbuild-comments.json`; their preceding contents are verified against the
core-build hashes. Final-header independent compilation also passes.

Compilation caught caller migrations that confused endpoint trials, raw residual
fields and unrelated coefficient containers; these were corrected before their
final builds/tests. The first restart selected the newest 0.2 checkpoint and
advanced to 0.25 instead of the reference 0.15–0.2 interval. That completed output
is retained, and clock comparison correctly rejected it. The corrected controls
use the existing midpoint checkpoint and a separately generated new 0.1 prefix.
An initial lifecycle reducer also paired constant-viscosity rejection with the
layered run; it was corrected to the existing matched constant schedule. These
were orchestration errors, not grounds to alter numerical tolerances.


## Scope retained

Defaults, spin lag, physical equations, endpoint convention, quadrature,
projection-correction criterion, final slip guess, acceptance and checkpoint
packet ordering remain the comparison contract. This work does not establish
new physical accuracy or production readiness. Finite-end tensor approximation,
coupled energy/flux accuracy, moving AMR/birth and independent cohesive tip/energy
failures remain open. Scientific campaigns stay paused. The next justified
scientific experiment remains the frozen endpoint-aware tensor-space comparison,
only after user selection. Inherited cleanup losses (including the recorded
39/45-frame and temporary-log losses) are unchanged.
