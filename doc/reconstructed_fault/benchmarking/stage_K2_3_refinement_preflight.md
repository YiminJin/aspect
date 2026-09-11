# K2.3 single spatial check: execution budget approved

The authorized pair is now complete within budget. See
[the spatial review](stage_K2_3_spatial_review.md): verification passes, but
the 0.5-s spatial normal-feedback change is not small enough to recommend
K2.4. No 128x512 run was launched.

The user accepts the 32x128 true-pressure pair as a feasibility pass and
authorizes one 64x256 bumped/homogeneous pair through 1 s. No third level,
temporal refinement, forcing amplification or production change is included.

Prepared inputs are `nonuniform/true-pressure/pilot64.prm` and
`homogeneous64.prm`, relative to the uniform-shear benchmark directory. They
inherit the tested coarse pilot, changing global refinement 5 -> 6 and
structural-point spacing 0.015625 -> 0.0078125 m, preserving the existing
bulk/fault refinement relation. The control removes only the initial Theta
bump. Maximum dt stays 0.5 s and end time stays 1 s. Top total normal traction
-1000 Pa, tangential top velocity, full bottom velocity, x periodicity and
`Pressure normalization = no` are unchanged.

The initial phase equation/profile data and histories are unchanged; each
mesh's converged initial Q1 profile is then frozen by the existing fixture.
Resolution-dependent initialization must be reported, not overwritten using
the coarse FE samples. Keep support, full I_h, domain quadrature, Q2 ADD/count
history transfer, solver tolerances, iteration budgets and lifecycle unchanged.

## Saved-data cost estimate before launching

The coarse true-pressure pair took 39.184/36.589 s, using 615768/557076 KiB.
The earlier same-size 64x256 domain-rule prescribed-pressure replay through
1 s took 271.046 s and 1329992 KiB (`global-accumulator/k2_64.resources.json`).
That older binary predates the accepted lookup/rejection optimization, so its
runtime is context, not a prediction for the current executable. The saved
through-2-s 64 run took 407.600 s.

Quadrupling the current coarse cell/particle count suggests about 150 s even
with linear work scaling; initialization and true-pressure linear solves
add uncertainty. Budget estimate: **150--300 s per case, about 1.3--2.4 GiB
peak RSS**. No new performance run was used to obtain this estimate.

This exceeds the preceding task's 120-s per-run bound. The user explicitly
approved **up to 300 s per case with the 600-s aggregate execution limit**.
Run sequentially and reserve analysis time from the remaining aggregate budget;
stop without retry if a cap or numerical check fails. No executable rebuild
or production modification is needed. The existing verifier's 120-s resource
check will need a separately stated execution-budget option after approval;
that is not a change to any mathematical convergence criterion.

## Comparison and decision, once execution is approved

Reuse all coarse outputs. At 0, .5 and 1 s form bump-minus-homogeneous fields
within each resolution using actual pre-commit surface constitutive moments
and the production consistent mass matrix. Compare the two piecewise-Q1
fields on the union of their physical fault coordinates, not vertex indices.
Use exact piecewise-polynomial integration for means and RMS differences.

For sigma_n provide its signed mean, mean-removed RMS, extrema and profile at
both resolutions, plus the 64-minus-32 change relative to the fine signal.
Show delta p and -delta(tau:N) separately, so their signed cancellation to
delta sigma_n is explicit. Also compare V and accumulated slip. Initial
projection differences remain visible; no initial-error subtraction removes
their physical influence.

Require existing genuine nonlinear/fresh-linear checks, actual weak balance,
fixed-profile/history checks, and separate 1e-4 omitted-fraction and actual
normalization allowances at measured locations/times. No normal-column
stress average may substitute for surface constitutive evidence. No retuning
of pressure, quadrature, support, I_h or tolerances is permitted.

If the spatial change is clearly smaller than the sigma_n signal and the
profile is qualitatively stable, recommend proceeding to K2.4. Otherwise
report the evidence and whether a single 128x512 confirmation is justified;
do not start it. K2.2's reference remains provisional and Gate K2 unmet.
