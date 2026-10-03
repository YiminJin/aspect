# Endpoint-plane correction: review, qualification still blocked

**Qualified follow-up:** the selected [filter-derivative correction](../filter_derivative/README.md) resolves all remaining comparisons (927/927 pass). This report retains its earlier evidence.

**Subsequent diagnosis:** [the matched timestep-zero attribution](../diagnosis/README.md)
identifies order-dependent filter derivatives as the remaining cause. No further
numerical correction has been implemented. The original review below is retained.

Reference: `8df2bc5b5`, the uncommitted section-2 BP3 geometry implementation,
and immutable `build-refactor-r6b/aspect-profile-bounds-qualified`. This is a
separate numerical correction in `source/reconstructed_fault/manager.cc`, not
part of the geometry movement. Section 3 has not started.

The automatic bulk-source handoff now uses a local roundoff band. With terminal
segment endpoints a,b and offset x-e, its length bound is
`32*epsilon*(|x|+|e|+|x-e|*(1+(|a|+|b|)/|a-b|))`.
It accounts for coordinate subtraction/dot products and tangent conditioning
when resampled endpoints are subtracted. It is not an adjustable physical
length or a solver tolerance. Inside the band, an already admitted strip
association owns the point; only an unassigned point receives endpoint shape
functions. Outside it, the signed endpoint classification is unchanged.
Transverse support, the existing physical-half-plane tolerance, zero-influence
perpendicular behavior and overlap assertions are retained. Generic particle
admission, native geometry/resampling, cache ownership and MPI operations are
unchanged.

## Verification

- Release core and regression plugins build; both dimension instantiations compile.
- Two-rank contact/support rejection, legacy bottom-source and profile unit tests
  pass: **39,045 assertions in five cases per rank**.
- Reported four points, their immediate floating-point neighbors, and points
  1 micrometre on either side pass in both input orders on one/two ranks.
  Tests also preserve exact strip associations, exercise the documented
  cached-inactive shortcut, and reject points outside transverse/physical
  support or beyond the roundoff band on the positive-s side.
- All **16,875** velocity quadrature points have identical admission in the
  corrected forward/reversed runs. Against the immutable reference, every
  previously active association is preserved exactly. The correction adds
  nine forward and five reversed associations, all within **2.26e-12 m** of an
  endpoint plane. Reoriented native tangents still differ by up to 1.45e-15;
  physical segment coordinates differ by up to 1.43e-14. Native order and
  valid associations were deliberately preserved.
- The 60-degree reversed comparisons and accepted 60-degree birth, retry,
  restart, survivor-H, newborn-H-before-audit, nonzero-Maxwell and RNG checks
  pass with unchanged thresholds. Retry/direct and restart/uninterrupted
  comparisons remain exact where required. Fresh residual checks pass.
- **Six of the seven original failing comparisons now pass.** The timestep-zero
  45-degree slip-rate column still fails: max difference
  **7.070469181247823e-18 m/s**, relative **7.0664e-9**, versus the unchanged
  `5e-10*column_scale + 1e-22` bound. The one-rank comparison reproduces it
  (`7.070475385102417e-18 m/s`); one/two-rank comparisons themselves pass.
  Initial weak normal traction differs by 0.016796544 Pa and shear by
  8.32826e-5 Pa, both within their existing column thresholds. Step-one/two
  physical profile comparisons pass. Initial Theta is identical.

`compare.py --endpoint` records **925 passed / 2 failed** checks (the remaining
original failure and its new serial counterpart). It intentionally exits 1.
The geometry-only guards, unchanged identity-rejection and evolving-H checks
reuse the prior section-2 evidence explicitly; the endpoint/field/lifecycle
runs and focused units are fresh. No tolerance, physical parameter or expected
answer was changed. Full association agreement rules out another *admission*
gap in this velocity quadrature; it does not establish the cause of the
remaining solution sensitivity. The correction is **not fully qualified** under
the user's seven-comparison requirement.

Next bounded task: isolate the remaining timestep-zero input-order sensitivity
with matched assembled-system/normal-traction diagnostics before proposing any
further numerical change. Do not canonicalize input, alter valid associations,
change solvers, or proceed to section 3 to hide this failure.

## Reproduction and artifacts

Use `../run_cases.py --batch endpoint-CASE:RANKS ...` for the cases mirrored
from `qualified-CASE`; outputs are isolated as `output-endpoint-CASE`.
Before resume/retry2/direct2, use `../prepare_restart.py` to clone
`output-endpoint-create` into the absent corresponding endpoint output.
`../run_cases.py --endpoint-unit` runs the focused units. The `endpoint-points`
cases perform the final point checks (one step, four order/rank combinations).
The `endpoint-dump` and `qualified-endpoint-dump` cases dump every association
with the corrected and reference executables, respectively. Run
`compare_associations.py` then `../compare.py --endpoint` from any directory.
Existing runners refuse to overwrite logs; use new isolated paths to repeat.

`results/` retains compact checks, field/lifecycle CSVs, full association
comparisons, exact run metadata, logs for the final point checks and unit tests,
and artifact/preservation hashes. Large raw builds and outputs remain ignored.
The initial test-helper build was requested before CMake regeneration and
failed with an unknown target; regeneration/build succeeded. The initial point
helper's artificial cached-inactive probe was replaced by a genuinely inactive
point satisfying the API precondition, then all four final point cases reran.
Neither adjustment changed simulation parameters or core source.
