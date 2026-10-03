# Separate profile interpolation roundoff correction

The section-2 45-degree regular-particle run failed before mechanics on the
accepted executable `aspect-particle-lifecycle-qualified`: the stationary profile
returned `0.60000000000000009` for peak `0.59999999999999998`. The stack enters
`PhaseFieldProfile::value()` through the shared BP3 startup/birth initializer,
then fails the unchanged `stationary_crack_driving_force` input assertion.
The runtime-cleanup brief explicitly anticipates this failure and requires a
separate justified correction rather than lowering the peak or loosening gates.

The tabulated endpoints are valid. Floating-point evaluation of the existing
convex expression `(1-xi)*left + xi*right` overshoots by one ULP at very small
positive xi. `results/profile_roundoff_reproduction.json` reproduces four such
weights from the saved reference table. The source correction preserves this
expression and bounds its result by the two tabulated endpoints. It does not
change the quadrature, tabulation, profile parameters, stationary-H function or
its assertions. It does not infer H from an evolving phase field.

The focused `phase_field_profile_bounds` unit test covers near-peak rounded
weights and interior intervals. Together with the existing `bp3_restore_profile`
comparison it passes 38,967 assertions in two test cases on each of two ranks;
the table comparison's maximum phase error is `6.41154e-14`, within its unchanged
`2e-11` gate. The formerly failing 45-degree case reaches accepted steps 0–2
with the corrected executable, at unchanged physical and solver settings.

Build: `cmake --build build-refactor-r6b -j2`, using GCC 12.4/Open MPI 5.0.6 via
the suite's logged runner. The unchanged accepted executable is retained; the
new immutable copy is `build-refactor-r6b/aspect-profile-bounds-qualified`.
Unit command: `python3 benchmarks/reconstructed_fault/bp3_geometry/run_cases.py --unit 2`.
Failure and success logs: `evidence/dip-45-np2.log` and
`evidence/qualified-dip-45-np2.log`. This correction is committed separately from
the geometry implementation and its remaining verification.
