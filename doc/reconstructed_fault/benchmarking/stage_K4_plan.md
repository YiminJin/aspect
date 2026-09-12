  # Stage K4 — Separate finite-width effects from numerical error

  **Approved scope addendum:** K4.2 is now restricted to finite-width accuracy
  assessment on [4,6] s at dt=.125 s, using the saved .125/.0625 reference
  comparison. All production runs still start at t=0 with unchanged histories,
  and all physical/numerical guards apply over the complete 0--6 s trajectory.
  The unresolved early transient is retained; K4.1 is not a full-history pass.
  See `stage_K4_post_transient_recommendation.md` and `stage_K4_2_execution.md`.
  The original criteria below remain recorded; no coefficients are weakened.

  ## Approach and fixed choices

  Start with K1’s fixed-profile, homogeneous shear problem, using its independent scalar mechanics reference and K3’s independently calculated initialization. Do not
  start with evolving K3 profiles: that would mix width effects with the documented history/timestep sensitivity.

  Use two regularization lengths only: (\ell_0=0.15625) m and (\ell_0/2=0.078125) m.

  Keep physical geometry, boundary conditions, K1 loading, material constants—including (G_c), cohesion, (G), and (\eta)—friction, supplied initial stress/state, and
  phase-core value fixed. Recompute degradation calibration, stationary initialization, reconstructed properties and full (I_h) independently at each width. Freeze the
  converged phase profile afterward, using the established K1 mechanism.

  Preserve production algorithms, solver tolerances, open fault topology and support policy. A policy-derived support distance may change with (\ell); record that change
  rather than silently holding the old distance or widening it manually.

  Explicitly extend the approved (10^{-4}) omitted-profile allowance to K4, retaining the separate (10^{-4}) actual slip-normalization requirement. Neither certifies
  that truncation is negligible compared with a small measured width effect.

  No public API or production changes are planned.

  ## Stage 4.1 — Establish the width-dependent reference

  Question: What changes when (\ell) changes in the finite-width equations, before production discretization enters?

  Minimum analysis:

  - Compute independent initial profiles at both widths, then run the existing fixed-profile scalar mechanics/history reduction through 6 s with unchanged K1 loading.
  - Check initialization/reference resolution with two normal grids. Use cheap scalar timestep comparisons at (0.5) and (0.25) s; add (0.125) s only if needed to
    establish temporal uncertainty.

  - Compare (V), accumulated slip, (q), (C), (\Theta), velocity profiles and integrated crack strain. Record initial (H/\phi/C), total (I_h), actual profile width and
    localization moments.

  - Report profiles in both physical (y) and scaled (y/\ell) coordinates. Differences caused by the model’s required initialization/calibration belong to the width-
    dependent model family, not discretization.

  Decision criterion: Select the largest tested common timestep whose estimated temporal error is below one quarter of the existing resolved K1 observable-error
  allowance. Require independently resolved initialization and admissible roots at both widths.

  Stop: If either profile violates support/admissibility, or reference uncertainty prevents interpretation, report that limitation. Do not change loading, support or
  calibration to obtain a comparison.

  ## Stage 4.2 — Check that production reproduces the width effect

  Question: Is the observed change with (\ell) a finite-width/model effect or an under-resolved profile?

  Minimum experiment: A three-case, non-product design:

   Case         Width    Bulk mesh    Fault elements
  ━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━━
   A         (\ell_0)       32×256                32
  ──────  ────────────  ───────────  ────────────────
   B       (\ell_0/2)       32×256                32
  ──────  ────────────  ───────────  ────────────────
   C       (\ell_0/2)       32×512                32

  Use the common timestep selected in 4.1. Reuse an existing result only if its physical setup, accepted sequence and numerical implementation are compatible; evolving
  K3 trajectories are not substitutes for fixed-profile K1 runs.

  A–B changes width on the same mesh. B–C changes normal resolution at fixed width, without changing the tangential/fault discretization. Each case also has its own
  independent reference, so production-reference errors need not be mistaken for width effects.

  Decision criterion:

  - Retain existing K1 observable tolerances and all convergence/history/support checks.
  - For each observable, compare the width difference against a conservative empirical uncertainty estimate combining production-reference, reference-resolution and
    temporal errors.

  - Call a width effect distinguishable only when it exceeds that estimate by at least a factor of four. Otherwise report it as unresolved or below the established
    observable accuracy—not zero.

  - Keep initialization differences, raw stress and truncation diagnostics visible.

  Stop: If B–C exposes an unexplained plateau, or the uncertainty remains too large, identify the limiting component. Do not automatically add another normal/fault level
  or reopen K3’s (I_h)-feedback investigation.

  ## Stage 4.3 — Conditional nonuniform check

  Question: Does finite width affect nonlocal mechanics that homogeneous K1 suppresses?

  Use K2’s fixed-profile, prescribed-normal-stress (\Theta)-bump case, not the true-pressure branch. Keep the bump’s physical width and amplitude fixed, so changing
  (\ell/L_{\rm var}) has a clear meaning.

  First inspect the saved early-time K2 evidence through 1 s. Proceed only if its spatial/temporal uncertainty permits a meaningful comparison. Then use the two widths
  with matched homogeneous controls; reuse compatible controls/results from earlier work. Do not launch a K2 convergence campaign.

  Compare bump-minus-homogeneous (V), slip, (C), (\Theta), actual weak surface traction and raw bulk stress. Report total and mean-removed fields, with endpoints
  separate from the same fixed physical interior.

  Decision criterion: Apply the same uncertainty-versus-width-signal test as 4.2. A mesh-scaled endpoint layer is not evidence of a finite-width effect.

  Stop: If the existing K2 uncertainty prevents attribution, conclude that nonuniform width sensitivity remains unverified. This is a valid bounded outcome; do not
  repair K2 or amplify forcing automatically.

  ## Stage 4.4 — Close with a bounded conclusion

  Stage K4 is sufficiently answered when the report distinguishes:

  1. Reference-predicted width dependence.
  2. Production spatial and temporal errors.
  3. Initialization/calibration dependence.
  4. Support truncation and endpoint limitations.

  State which width is resolved and affordable, which observables show distinguishable width effects, and which effects are below the existing accuracy allowance or
  remain unresolved. Record runtime and memory during necessary runs.

  Do not infer universal sharp-fault convergence, a minimum acceptable (\ell), or a passed K2/K3 gate. Further (I_h)-feedback work remains deferred unless K4
  demonstrates a consequential effect on the chosen observables.
