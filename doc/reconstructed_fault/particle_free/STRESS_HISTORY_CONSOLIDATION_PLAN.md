# Selected stress-history consolidation — 4 November 2026

The source starting point is commit `e4b74c556`, recorded before restructuring.
Its parent and unchanged origin-tracking reference are `69e8a10a93a08`.
The scientific finite-end experiment is on hold. This task preserves numerical
behavior; it does not select endpoint-aware history, a limiter, or new physics.

Selected implementation boundaries:

1. Keep the tested direct physical history and final publication slip guess in
   a separate source baseline, with actual binary/source/control identities.
2. Move the serial limiter, statistics and limited-stage integration to benchmark
   support. Remove limiter arguments from the core transport operation. Retain
   serial smooth transport: the PF5 prescribed accepted-step control still uses
   it. Move checker-only snapshots/reference setup and standalone prescribed
   spatial relaxation out of material operations; keep all callers buildable.
3. Separate physical stress access from independent tensor initialization/source
   projection. Keep direct and separated history behind the same physical
   observer, and isolate material projection from the geometry/history
   transaction. Do not merge mature and cohesive histories or alter their laws.
4. Extract the affine residual/load audit from the Newton driver. Preserve its
   flag semantics, extra assemblies, precision assertions, output and scratch
   restoration. Keep actual trial construction, mechanics and acceptance in the
   solver. Do not duplicate a mechanical solver.

Bounded checks, declared before post-refactor comparisons:

- Full existing unit suite, unchanged numerical assertions.
- Fresh entry cases `consolidate-before-{dc,sc,dl,sl}`:16², two MPI ranks,
  dt=.05, end=.1, fixed initialization interval. Direct/separated and constant/
  layered eta, full-normal support, compact phase, bounded MacCormack, accepted
  tensor trace and zero imposed shear. All four passed using the preserved
  entry binaries before rebuilding.
- Matching post cases; direct constant also exercises retained affine and
  native source checks. Compare all common accepted bulk and surface records,
  mechanical iteration counts, publication/retry counts and reconstructed
  stress/traction/work. Target bitwise equality; absolute regression screens
  are5e-13 for bulk records and5e-12 for surface records. No relaxation in
  response to a failure. Run the independent established reducers as well.
- Layered direct continued restart from the **entry local test checkpoint**
  (.1 s, two ranks) on four ranks through .15/.2 s, compared with uninterrupted
  post-refactor evolution. This is an explicit same-law refactoring check, not
  a production checkpoint conversion. Rejection/cutback compares with retained
  `true-reject` and its matched accepted schedule.
- Rebuild the migrated serial checkers and run their bounded4² local invariant
  fixtures. Rebuild affected checker interfaces; no large limiter/time matrix.
- Reuse unchanged physical qualification limits and failed evidence. Check
   protected files and cleanup losses at closeout. At most four concurrent
   build/test workers; simulations remain sequential. No server job or push.

Completed: all selected boundaries and bounded checks above. See
[report](STRESS_HISTORY_CONSOLIDATION_REPORT.md) for measured errors and retained
limitations. The only post-test header edit adjusts whitespace. No additional
scientific task is authorized by this plan.
