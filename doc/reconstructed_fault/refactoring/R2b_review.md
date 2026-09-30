## R2b proposal 2 — boundary completion (complete; ready for review)

Selected instruction: [prescribed boundary completion](refactoring/codex_boundary_completion_instructions.md).
Start state is the completed geometry extraction, including its local changes;
the recorded R2b artifact hashes matched before editing. The first step is
complete: `apply_boundary_normalization_completion()` privately owns the legacy
file loading, validation, addition and diagnostic output in `normalization.cc`.
Its body is byte-identical and runs at the same preprojection point. All 5,810
other source/header/test files were unchanged at this checkpoint. A fresh
Release build passed; short BP3 on one/two ranks matches the geometry-pass
reference exactly in all 186 field groups, including I_h, mechanical fields,
history and solver records. No tolerances or expected outputs changed.

The preserved move-only executable is `build-refactor-boundary/aspect-completion-move`.
Snapshots, separate diffs, hashes and comparison records are in
`benchmarks/reconstructed_fault/refactoring_boundary/evidence/`. Subsequent
automatic treatment is a behavior extension, recorded separately from this move.

The paired interior path remains `ReconstructedFaultManager::project_to_bulk_source`:
legacy bottom/top wedges use constant endpoint surface coordinates, retaining
physical phase, pressure, stress and history at each physical point. The Stokes
QP association cache supplies the common basis to bulk/coupling and the bulk-
work surface rule; particle Maxwell history uses that same source association.
No mechanical assembly is relocated into the material model. BP3 qualifies its
frozen field through full phase-DoF constraints to the handler's stationary
profile, with H initialized from that same profile. H-driven initialization
without compatible oblique boundary data remains unqualified for automatic
completion; the separate phase-boundary follow-up is recorded in the plan.

### Automatic extension and ownership

The second step adds the explicit selector `Fault reconstruction / Boundary
completion = automatic prescribed`; existing inputs default to `legacy`.
It classifies every prescribed fault without fault-zero/bottom assumptions.
`boundary_contact.h`, `boundary_contact.cc` and `boundary_contact_manager.cc`
(M3) own boundary-facet gathering, contact identity, support admission and
lifecycle. `PhaseFieldFault::prepare_automatic_boundary_completion()` and
`apply_automatic_boundary_completion()` in `boundary_completion.cc` (M4) own
profile compatibility, ghost-Q1 constitutive evaluation and integral addition.
`manager.cc` supplies endpoint associations; `surface_system.cc` admits the
existing per-fault bulk-work assembly after automatic qualification (M5).
`bp3/plugin/bp3.cc` selects the new path without attaching a legacy CSV.
The material header declares private operations/data; the manager header exposes
geometry records and geometric extents. No ownership or checkpoint schema changed.

Supported: 2-D fixed prescribed geometry on an axis-aligned affine Box, uniform
aligned boundary ghost lattice, mature/frozen fully constrained compatible Q1
field, constant prescribed core per fault, identical profile/degradation material
coefficients. Either/both transverse endpoints, different faces, independent
faults, and curves outside the straight terminal support are covered. The
selected automatic mode uses the existing bulk-work mechanical measure.

Unsupported cases fail explicitly: corner/tangential/unrepresented crossings,
short curved terminals, intersecting completion envelopes or nearby diffuse
branches, overlapping prescribed profiles, undefined exterior material/lattice
data, mesh deformation, 3D, and unqualified H-driven/evolving fields. Conservative
support rectangles can reject close geometry even when a more elaborate
analysis might establish disjoint physical wedges. That generalization requires
a separate design; no nearest-continuation rule was invented.

Geometric extents include the full discrete Q1 support. They are not an activation
threshold or a new truncation of nonzero mechanical phase support. The material
verifies the entire physical nodal profile and full constraints before use.
Legacy and automatic completion are mutually exclusive. Exterior integration
retains the BP3 ghost-grid/profile convention and ordinary localization law,
adds once before unchanged projection, and keeps all physical mechanics in the
existing assembly/history operations. Free endpoint shape functions and K/B/G
couplings remain active. A tolerance-limited endpoint guard retains an existing
association when raw/resampled frames disagree only at roundoff in s=0.

M1/CPDI, M2/evolution, the limiter edits, proposal 1's cell traversal, completed-
value reuse criteria and proposal 3 remain unchanged. Contact/continuation caches
are transient and follow geometry, mesh and restart invalidation. Runtime adaptive
refinement has not been qualified; uniform boundary spacing is checked after
rebuilding. MPI decisions are collective and corrections retain profile ownership.

### Focused verification

The final Release build, registered CTest, two contact unit cases (56 assertions each rank), ten small fixture
runs, both seven-observation BP3 runs and a one-to-two-rank restart are recorded
in the [isolated harness](../../benchmarks/reconstructed_fault/refactoring_boundary/README.md).
The focused plugin checks actual interior-wedge association and centered
free-endpoint K, pressure G and bulk B derivatives; it uses nonzero free first/last
endpoint directions. Geometry units additionally cover shared-facet deduplication,
periodic-face exclusion, 3D rejection and diffuse branch envelopes.

| Fixture | Contact identity `(fault, endpoint, boundary)` | Treatment |
|---|---|---|
| interior / diffuse support touching boundary | none | no continuation |
| perpendicular | (0,0,bottom), (0,1,top) | zero correction |
| oblique | (0,0,bottom), (0,1,top) | paired, both ends |
| reversed | (0,0,top), (0,1,bottom) | same physical corrections |
| left/top | (0,0,left), (0,1,top) | paired, different faces |
| multiple | (0,0,bottom), (1,0,right) | independent paired contacts |
| curved interior | (0,0,bottom) | paired straight terminal; interior curve retained |
| BP3 | (0,0,bottom), (0,1,top) | paired; 77 affected profiles per end |

Oblique and multiple-fault nodal I_h are exact across one/two ranks; reversal
differs by 8.58e-16 relative. Perpendicular/interior correction files contain no
rows. Free-endpoint derivative relative errors are at most 5.51e-13 (K),
7.78e-14 (G), and 6.01e-15 (B), against the existing 1e-8 criterion.
Corner, tangent, between-vertex crossing, H-driven/unconstrained, and undefined
exterior-material cases reject on two ranks with the intended messages.

The automatic BP3 correction covers 154 of 744 profiles without duplicate IDs.
In-domain integrals are exact versus legacy. Maximum outside-integral difference
is 3.75e-7, or 3.23e-13 of the largest correction; nodal I_h differs by at most
3.35e-7 absolute / 1.44e-13 relative. This is consistent with independently
computed C++ versus Python ghost-Q1 panel integration at 1e-11 panel accuracy.
No legacy tolerance was changed. Final V differs by at most 4.86e-23 m/s,
accumulated slip by 1.67e-20 m, and Theta/phase/H/geometry are unchanged.
Mechanical weak-table differences are below 2.01e-12 of their field scale.
The final executable also reproduces the default legacy one-rank trajectory
exactly. All discrete solver decisions match. Near-zero residual differences are retained
in absolute units in the detailed comparison; their large relative percentages
are not interpreted as trajectory changes. MPI/restart field comparisons pass
1e-10 scaled/absolute checks; cold restarted normalization differs by 4.45e-16
before restoring the validated frozen snapshot. No long cycle was run.

Exploratory fixture failures are preserved, not hidden: inherited unavailable
postprocessor names; an under-resolved new profile giving a negative consistent
projection; an existing stationary-H roundoff assertion for an exact 45-degree
line; and the existing normalization overlap rejection for insufficiently
separated faults. The fixture was resolved and made independent. Production
projection, scientific algorithms, M2 assertions and solver tolerances were not
relaxed. The phase-boundary prerequisite is established by full prescribed DoF
constraints and a nodal profile check, not by the initial H flag.

Stop for review after this task. The remaining separate decision is whether to
select the documented evolution-compatible phase-boundary design; no new
refactoring extraction has begun. Detailed commands, hashes, stage-separated
diffs, contact records and comparison metrics remain in the harness evidence.

## R2b — private cell-profile geometry preparation (complete; ready for review)

Historical proposal-1 review boundary: only geometry preparation had been
selected. Proposals 2 and 3 were unchanged, and boundary generalization was
deferred. The later proposal-2 selection is recorded above; proposal 3 remains
unselected and unchanged.

### Revisions and responsibility boundary

Worktree `/home/ein/repository/aspect-pf-rsf-refactor`, branch `pf-rsf-refactor`,
HEAD `3af7c2aa2590aa8e39eaf2a1e46df0bc6c9d9700`, with the existing uncommitted
R1/limiter/R2a changes preserved. No commit was made. The immediate reference
is the actual R2a entry source and preserved R2a executable/plugins/results,
whose 25 recorded hashes matched before editing. Matched results also compare
against qualified R1. Candidate artifacts and evidence are isolated in
`build-refactor-r2b/` and `benchmarks/reconstructed_fault/refactoring_r2b/`.

`PhaseFieldFault::prepare_cell_normalization_geometry()` now performs backend
admission, physical-box reductions, profile clipping, half-open face ownership,
cell searches, cached DoF-index construction and interval sorting. It remains
private, implemented in `source/material_model/phase_field_fault/normalization.cc`.
Its input is the existing profile vector; mesh/mapping, geometry model,
introspection/FE/DoFHandler, communicator and traversal cache remain dependencies
of the owning material object. It returns a private value record containing the
two endpoint distances per profile and four local geometry work counters.
Cached intervals remain in the existing material-owned cache.

The integration definition shrinks from 366 to 252 lines. It still samples
current phase values, invokes the existing material integrand, performs all
quadrature/tail/coverage operations and emits the existing work diagnostic.
Geometry preparation does not receive the material callback or sample the
solution. The header adds only the private result record and method declaration;
no public interface, persistent member, cache layout, checkpoint field or
ownership changes. Explicit member instantiations retain 2D/3D linking.

Movement verification confirms that geometry statements are exact apart from
qualifying the returned fields and sizing their vector. All earlier methods
(including boundary completion, value reuse and invalidation) are byte-identical.
The integration tail and later methods are identical except for diagnostic
field qualification and the new helper instantiation. MPI reduction ordering,
backend fallback, clipping, interval order and origin/normal reuse criteria
are unchanged. All 5,810 other source/header/test files match entry hashes,
including the user's limiter changes. No numerical fixes or formatting sweep
are included.

### Verification and differences

- Fresh full Release/unity/PCH build and matching lifecycle/BP3 plugins passed
  with the qualified GCC 12.4.0/OpenMPI 5.0.6/deal.II 9.6.2 stack and unchanged
  floating-point flags. The executable contains both 2D and 3D helper symbols.
- All 15 selected runtime invocations passed: accuracy/cache cases on 1/2
  ranks; both backends' lifecycle cases on 1/2 ranks; background-only material
  on one rank; both six-step BP3 trajectories; and paired R2a/R2b warm traversal
  runs on 1/2 ranks. Unit coverage includes four accuracy and two lookup-cache
  cases per rank (59,260 assertions on the one-rank run).
- Each reference comparison passed **186 field groups with zero differences**:
  all saved bulk components, particle properties, fault profiles, final native
  I_h/history and solver records at matching rank counts. This holds against
  both R2a and qualified R1; no cross-rank equality is asserted.
- All **22 lifecycle/work checks** passed in each comparison. Cold construction
  and warm traversal work counters match exactly. With completed-value caching
  disabled, each warm case changes from 21 rebuilt/0 reused profiles to
  0 rebuilt/21 reused and zero cell candidates. The one-rank case retains
  1,345 intervals and 23,688 FE samples; rank-zero diagnostics in the two-rank
  case retain 673 intervals and 12,096 samples. Existing coverage assertions
  run on every rank. Only elapsed geometry time is excluded from work-record
  equality; no performance improvement is claimed.
- The first sandbox MPI probe failed before tests because local sockets were
  denied. The permitted rerun passed. This infrastructure failure is retained
  separately from the passing suite; no test expectations/tolerances changed.
- All 23 untouched R2a artifact hashes and 45 qualified R1 artifact/input/
  checkpoint hashes still match. Source verification and `git diff --check`
  passed. The scientific-test worktree and reference outputs were not modified.

Reproduction and evidence: [R2b harness](../../benchmarks/reconstructed_fault/refactoring_r2b/README.md).
Ignored local evidence includes entry snapshots, the two source diffs, build/run
commands and logs, comparison JSON, source-preservation checks and artifact hashes.
No full test campaign, production BP3, restart/rollback rerun, dedicated cell-cache
AMR/moved-profile/fallback run or 3D simulation was performed. Existing R1 fixture
failures retain their prior dispositions; this extraction makes no additional
scientific qualification claim for the opt-in backend.

Stop for review. Proposed next task: a short design-only review of boundary-
completion generalization, if selected; do not implement it or proposals 2/3
without the user's next selection.
