# Filter-derivative correction — qualified

Reference: the corrected endpoint implementation, preserved as
`aspect-endpoint-plane-qualified`, and the bounded timestep-zero diagnosis.
The selected numerical correction is confined to the private filter-stiffness
operation and its call in `surface_system_bulk_work.cc`. No source association,
Q1 value, vertex ordering, material/solver setting, history, checkpoint, MPI
ownership or collective ordering changes. Section 3 has not started.

## Convention

At a shared vertex's normal plane use the arithmetic mean of the two one-sided
**gradient outer products**. At an endpoint plane average the interior trace
with the zero derivative of the constant exterior continuation. This symmetric
energy trace preserves the tridiagonal stencil, positive semidefiniteness and
constant null mode; it does not average gradients or introduce a new neighbor
coupling. Unequal segments retain their own physical inverse-length derivatives.

A local `32*epsilon` bound accounts for coordinate subtraction and tangent
conditioning from both adjacent segments. It resolves only floating-point plane
ambiguity. Away from the band, ordinary segment derivatives and zero exterior
continuation are unchanged. The existing valid source association is read,
never reassigned. M, loads, K_V/G definitions, factor reuse comparisons and
publication remain in their existing owners. Both filtering and its stiffness
diagnostic call the same small private operation.

The specification and current design now state this discrete trace convention.
The refactoring guidance records it as the separately selected numerical fix.
`results/correction.patch` separates it from the still-local section-2 geometry
work and earlier endpoint admission correction.

## Verification

- Final Release build passes, including 2D/3D instantiations. Focused filter,
  trace, contact, legacy-continuation and profile units pass **39,226 assertions
  in seven cases per rank on two ranks**. Trace tests cover reversal, either
  valid source side, both sides of the plane, unequal lengths, constant null
  mode and the originally reported endpoint points.
- **927/927 unchanged qualification checks pass.** All seven original failing
  comparisons now pass. Reversed 45-degree physical profiles pass through
  steps 0–2 on one/two ranks, with maximum V differences 5.191e-23 / 4.219e-23
  m/s respectively; the previous timestep-zero failure was 7.070e-18 m/s.
- The accepted 60-degree comparisons, birth-before-audit H, survivor H, Maxwell
  inheritance, RNG and rollback/retry checks pass. Required retry/direct and
  restarted/uninterrupted comparisons remain exact. Fresh residual checks pass.
- All **16,875** QP associations are unchanged from the endpoint-corrected
  reference in both input orders: no new association and every previously
  active entry preserved exactly. Point regressions pass in both orders on
  one/two ranks.
- At the common initial iterate, the matched filter K difference falls from
  5.325% to **4.424e-15 relative** (4.879e-18 absolute). M, K and M_mu agree at
  all three observed Newton bases; all nine matched physical G probes pass.
  These **18 matrix/action checks** are separate from the 927-check suite.

The qualification thresholds and production solver strategy/tolerances are
unchanged. Geometry-only guards, unchanged restart-identity rejection and
native evolving-H fallback retain their prior section-2 evidence; the relevant
field/lifecycle/point/unit/matrix checks above are fresh. No large production
run, GMG campaign, broader cache redesign or Section-3 implementation is claimed.

The final immutable executable is
`build-refactor-r6b/aspect-filter-derivative-qualified`, SHA256
`5ded9532b898a8b5f569ef9c568644c4e929c11db550515a74373344c0d2fa83`.
Earlier binaries/plugins/results and unrelated local files remain preserved.
An initial unit run preceded the final include-only rebuild; it is retained as
`filter-first-unit-np2`, and the full unit suite was rerun on the final executable.

## Reproduction

`run_cases.py --filter-unit` runs focused units. `run_cases.py --batch` accepts
`filter-CASE:RANKS` counterparts of the endpoint field/lifecycle cases. Clone
`output-filter-create` into absent `output-filter-{resume,retry2,direct2}` with
`prepare_restart.py` before those cases. Outputs/logs use separate filter paths.
The point, dump and timestep-zero matrix cases have `filter-points-*`,
`filter-dump*` and `filter-t0-*` names.

Run `endpoint/compare_associations.py --filter-derivative`,
`compare.py --filter-derivative`, then `filter_derivative/check_matrices.py`.
The inherited timestep-zero observer's legacy `extended` sample column is only
its historical strict-plane diagnostic; it is **not** the new trace convention.
The matrix checks use the actual production-assembled K, not that column.
Compact results, logs, matrix snapshots, run metadata and hashes are under
`results/`; raw builds and outputs remain ignored.

Next bounded task: review this correction, then select Section 3 if desired.
