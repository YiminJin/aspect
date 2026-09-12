# K3 I_h feedback: bounded impact check and disposition

The user has closed further spatial-convergence pursuit of this sub-metric
unless a later benchmark demonstrates consequential sensitivity. This is a
documented limitation, not a newly passed convergence criterion. Retain the
independent continuum reference, the nonmonotone 128/256/512 evidence, and the
separate support/normalization checks. Gate K3 has not been demonstrated in
full; this disposition does not authorize further runs or change any equation,
support width, I_h definition, tolerance, or acceptance threshold.

## One offline impact check

Command (no ASPECT execution):

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/evolving/check_ih_impact.py
```

The diagnostic reuses the independent scalar `Reference.mechanics` operation
and the eight accepted .375-s steps through 3 s. First it reproduces the saved
reference q, C, V, Theta and I_h trajectory (relative comparison tolerance
1e-12). For each saved normal level, prescribe

\[
 I^{\rm diagnostic}_k=I^{\rm reference}_k+
 \big[(\hat I^{\rm production}_k-\hat I^{\rm production}_0)
       -(I^{\rm reference}_k-I^{\rm reference}_0)\big].
\]

The initial offset is zero by construction. This isolates the feedback
discrepancy from the initial surface projection difference. The scalar
mechanical balance, cohesive recurrence and exact Theta update are unchanged.
Each replay initializes q/C/Theta/I_h once from the retained reference
histories; it does not use timestep-zero evaluated q/C as committed history,
and never resets from subsequent production states. q/C/Theta and slip then
evolve with the diagnostic I_h sequence. All roots remain admissible and meet
the existing scalar residual check. The zero-perturbation replay reproduces
the saved reference. Calculation time was 0.012 s (about 0.2 s process time).

This is **not** a self-consistent phase/history trajectory: no H or phase
equation is resolved and the perturbed I_h is not asserted to be the integral
of a new profile. It measures direct scalar mechanical sensitivity with
history propagation, not an upper bound on future fully coupled feedback.
It cannot certify normalization or replace the primary independent reference.

## Measured impact

The cumulative I_h-feedback errors at 3 s remain 70.05%, 31.85%, and 36.32%
of the small reference feedback signal at 128, 256, and 512. At 512, however,
the absolute feedback error is only **1.10982e-5 of total I_h** (I_h is about
108 m, whereas its reference cumulative change is .0033047 m).

For the 512 error sequence, maxima over the eight real steps are:

| Quantity | Maximum absolute change in diagnostic replay | Maximum relative change |
| --- | ---: | ---: |
| V | 5.17739e-9 m/s | 3.89595e-6 |
| q | .00386798 Pa | 3.04554e-6 |
| C | .00391100 Pa | 1.09975e-5 |
| Theta | .000236172 s | 3.16712e-6 |
| Accumulated slip | 3.94913e-9 m | 3.89595e-6 |

Absolute and relative maxima can occur at different times. Final signed
changes are +7.57520e-10 m/s in V, -.00386798 Pa in q, -.00391100 Pa in C,
-4.62079e-6 s in Theta, and +3.94913e-9 m in accumulated slip.
The largest relative change across all three resolution-error sequences and
all five quantities is 2.12153e-5 (C, the 128 sequence).

These are small direct mechanical effects in this fixture despite the large
relative error of the small I_h increment. In particular, the 512 direct q
shift is much smaller than the saved .47856 Pa raw-stress RMS error. This
comparison is a scale assessment, not a causal decomposition of that raw
stress error or proof that no other feedback mechanism matters.

## Recorded limitation and stopping decision

The final per-step I_h-increment error contracts weakly, but cumulative error
does not: 1.05255e-3 m at 256 becomes 1.20014e-3 m at 512. Total phi and
cumulative phi errors improve; raw stress and H errors retain the documented
nonmonotonicity and initialization influence. Do not describe K3 as a fully
spatially/temporally converged reference, or reinterpret initial-error
subtraction as removal of its physical influence.

The measured direct impact supports the requested decision to stop pursuing
this sub-metric now. Reopen only if a later benchmark demonstrates that this
I_h-feedback uncertainty materially affects an observable or acceptance
decision. No extra normal/fault refinement, support-tail diagnosis, smoothing,
normalization redesign, or production correction follows from this report.

Machine-readable results are in `evolving/ih-feedback-impact.json`, with the
script and log beside it. The unchanged spatial evidence and checks are in
`stage_K3_normal512.md` and `evolving/normal512-comparison.json`. Existing
periodic-domain, phase-precision, MPI/action and lifecycle verification is
retained rather than rerun. The selective commit includes implementation,
fixtures, diagnostic scripts, reports and key compact result summaries.
Generated logs, full profiles, raw particle/FE/VTK dumps and build products
remain locally preserved, not in Git. Saved-output analyses require those
local artifacts (or regenerated runs); their inclusion as scripts does not
imply that the complete raw-data archive is versioned.
