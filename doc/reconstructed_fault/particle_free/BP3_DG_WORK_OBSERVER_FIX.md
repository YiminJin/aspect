# BP3 accepted-work observer frame repair

19 October 2026. Reviewed base `b2dfdebe43437d6028b06d4f9f862cc80a00cc71`
plus preserved optimization, Maxwell-coefficient repair and diagnostic work.
This fixes the demonstrated observer defect in server jobs 3610007/3613912.
It changes one benchmark runtime source file, not constitutive evolution,
mechanics, accepted history, solver tolerances, mesh, quadrature or filtering.
No server job or full event was launched locally. Existing spatial/endpoint
accuracy findings and `strict_reference_match_passed=false` remain.

[Data, exact hashes, patch and server instructions](../../../benchmarks/reconstructed_fault/particle_free/evidence/server_3613912/README.md).
The supplied archive's initial hash and its five unchanged regular data files
are retained. Extraction preserved existing analysis. All original errors remain.

## Demonstrated cause

The fault is straight physically, but subtracting nearby resampled coordinates
produces slightly different floating-point segment frames. Mechanics projects
stress using the actual cached Stokes-QP association's tangent and normal. The
observer used the first segment's frame everywhere. At event stress amplitudes,
that difference exceeds the very small existing absolute load-consistency bound.
The independent accepted-history publication check passed; this failure is in
the subsequent observer's projection functional.

The actual signed step-816 data contain 5776 nodes. At offending node 4918,
8569.999999999998 m down dip, the native mass row is 10.000004160420962 m:

| Quantity | Pa m |
|---|---:|
| Frozen mechanical raw-normal load | 512360082.0618272 |
| Old first-segment observer load | 512360082.06172705 |
| Matching-frame observer load | 512360082.06182724 |
| Directly integrated frame-difference contribution | -9.93162750313624e-5 |

The normalized original error is **-1.001357615619641e-5 Pa**, failing the
unchanged strict **1e-5 Pa** bound. At the same node the matching-frame error
is **5.960461997735958e-9 Pa**. Across every node, matching-frame maxima are
**5.067868297252547e-7 Pa shear** and **2.0867824647698363e-7 Pa raw normal**.
Both pass the existing bound. Adding an inconsistent 2e-5 Pa load still fails it.

`verify_packet.py` independently recomputes every exported normalized error
bitwise from its load operands. A 100-digit Decimal calculation of the exact
binary operands verifies the half-ULP quotient-rounding bound; load subtraction
satisfies Sterbenz. Directly integrated frame differences and differences of
separately rounded load integrals do not cancel exactly: their maximum closure
defects are 1.5589e-7 Pa shear / 2.6726e-7 Pa normal. These are reported separately,
not used to enlarge any allowance.

All columns of the six replayed accepted records (810–815) have identical parsed
values to the original trajectory. The resolved replay parameters
match the prepared input after normal ASPECT source-directory expansion. The
supplied marker and accepted label verify replay checkpoint **815 / slot 03** at
**5172072493.8982706 s**. These checks validate this repair's provenance, not
first-event physical accuracy or full checkpoint contents on this workstation.

## Source correction

Only `benchmarks/reconstructed_fault/bp3_native/server/plugin/work_audit.cc`
changes runtime behavior:

- The evaluator receives explicit slip/normal tensors. Mechanical QPs use
  `sense*sym(t_a tensor n_a)` and `n_a tensor n_a` from the **actual native
  cached association**, preserving its segment and endpoint ownership.
- The corrected frames enter stress evaluation, shear/normal work loads,
  detailed QP exports and elastic-rate diagnostics. First-update particle
  checks and optional domain diagnostics use their selected segment's native
  frame, with the same construction as DG stress reconstruction.
- The original first-segment projection remains a separately accumulated
  diagnostic of the same physical stress. It does not control the guard.
  Existing trace field names are retained: `original_*` is that counterfactual;
  `matching_*` is now the actual checked load. The explicit marker
  `guard_uses_matching_frame=1` identifies the repair.
- The literal `consistency<1e-5` assertion, message, native work weights,
  pressure/background signs and raw-normal comparison remain. There is no
  comparison bypass, threshold change, filtering change or history feedback.
  Failure export still uses an MPI maximum before collective root I/O.

Independent accepted-history publication and all core source bytes stay
unchanged. Particle infrastructure, checkpoint format, full plane-strain
convention, unlimited DGQ2, Gauss6/source3 and Helmholtz20 are retained.

## Bounded verification and limits

The ordinary server plugin and the diagnostic fixture compile with the unchanged
core. In the two-rank, 4136-cell oblique mature fixture with mixed chemical labels
and native adaptive controls, the first accepted intervals remain 4e6 / 7.64e6 s.
The corrected observer's maximum checked error through initialization and two
positive steps is **1.7969e-8 Pa**. Nine physical output files are byte-identical
to the previous diagnostic build, including both ranks' DG coefficients,
authoritative reconstructed stress/checks, accepted records, profile and weak
loads. Work-observer exports change as intended to use local frames.

A completed one-owner candidate rejection leaves accepted state unchanged;
identical-dt retry gives eleven compared outputs byte-identical to the control,
including observer loads. Every accepted observation publishes once on each rank.
An isolated compile-only diagnostic-load perturbation gives a corrected error
**2.0001508222976958e-5 Pa**, and still fails the unchanged guard; its finite
117-node failure CSV is written before MPI abort. It exits 1 in 23.37 s under
a 180 s timeout. The negative macro is absent in ordinary builds.

A fresh same-mode two-rank checkpoint restart at accepted step 2 advances to
step 3 with the same adaptive timestep as an uninterrupted three-step control.
Fourteen final outputs are byte-identical, covering all persistent DG modes,
physical FE fields, authoritative stress, particles/cohesive-memory diagnostics,
surface state/profile, mechanical and observer weak loads. Accepted records
match and publication counts remain one on both ranks.

Both prepared server inputs validate with ordinary matching plugins. All local
native commands, parameters, hashes and results are retained in the linked
evidence. These runs are verification, not performance measurements. The full
server checkpoint is not copied locally; native advancement of the repaired
full-grid state beyond 815 remains a prepared **server verification**, not a
completed result. The frozen matching-frame evaluation of the actual step-816
state is measured evidence from the user's diagnostic replay.

The concrete next action is the supplied three-step 815→818 server verification
in a fresh branch, retaining all adaptive controls and the guard. After successful
checks, the separate manual continuation input permits resuming the selected
first-event experiment on that branch. It makes no new accuracy claim or
automatic job submission.
