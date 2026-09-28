## Stage K4.3 — lightweight nonuniform finite-width check

Keep this task deliberately bounded. K4.2 has already established that finite-width effects can be separated from normal discretization error in the homogeneous fixed-profile problem. K4.3 is only a sanity check for whether along-fault nonuniformity introduces a materially different width sensitivity.

Use the existing K2 fixed-profile, prescribed-normal-stress \(\Theta\)-bump case, not the true-pressure branch and not an evolving-phase case.

### Step 1 — preflight existing K2 evidence

Inspect the saved K2 results through the already validated early-time interval (nominally through 1 s).

Determine:

whether its existing spatial/temporal uncertainty is small enough to distinguish a finite-width difference;
whether compatible homogeneous controls already exist and can be reused;
whether the K2 timestep and numerical settings can be reused unchanged.

Do not launch new runs during this preflight.

If the existing K2 uncertainty is too large for attribution, stop and report:

“Nonuniform finite-width sensitivity remains unverified at the existing K2 accuracy.”

This is an acceptable K4.3 outcome. Do not start a K2 convergence campaign.

### Step 2 — minimum production comparison, only if preflight passes

Use only the two widths already studied:

$$ \ell_0=0.15625\ {\rm m},\qquad \ell_0/2=0.078125\ {\rm m}. $$

Use matched relative normal resolution, following the successful K4.2 A/C comparison:

wide case: resolution equivalent to K4.2 A;
narrow case: resolution equivalent to K4.2 C.

Do not include the deliberately under-resolved B-type case and do not add another mesh level.

Keep the \(\Theta\)-bump physical width, amplitude, loading, material parameters, support policy, fault discretization, and timestep fixed when \(\ell\) changes. Recompute only quantities that must physically/calibrationally change with \(\ell\).

Reuse compatible homogeneous controls from existing K2/K4 evidence whenever possible. If compatible controls are unavailable and obtaining them would require a substantial additional run set, stop and report that before expanding the task.

Primary diagnostic

For each width, form the nonuniform response relative to its matching homogeneous control:

$$ \delta Q_\ell = Q_{\rm bump,\ell} - Q_{\rm homogeneous,\ell}. $$

Then measure the width sensitivity of the nonuniform response:

$$ \Delta_\ell^{\rm nonuniform} Q = \delta Q_{\ell/2} - \delta Q_{\ell}. $$

This is the main K4.3 quantity. Do not confuse the ordinary homogeneous width effect already established in K4.2 with a new effect caused by along-fault heterogeneity.

Focus on:

slip rate \(V\);
accumulated slip;
cohesive traction \(C\);
actual weak surface traction \(q\);
\(\Theta\);
raw bulk stress only as a supporting diagnostic.

Compare total and mean-removed along-fault fields. Treat the endpoint region separately from the same fixed physical interior used in K2; a mesh-scaled endpoint layer is not evidence of finite-width physics.

### Decision

Use the existing K2 numerical uncertainty together with the established K4 factor-four signal-versus-uncertainty rule.

The useful conclusions are only:

1. __No distinguishable additional nonuniform width sensitivity__: the two widths give essentially the same bump-induced response within established accuracy.
2. __Distinguishable but modest sensitivity__: record its magnitude and which observables are affected.
3. __Strong or qualitative sensitivity__: identify it, but do not automatically launch further refinement.
4. __Existing K2 uncertainty is insufficient__: report K4.3 as unresolved and stop.

Do not use K4.3 to determine an optimal \(\ell\), a universal \(\ell/h\) requirement, or a sharp-fault convergence rate. Do not add timestep levels, normal-refinement levels, support studies, or true-pressure tests.

### Stopping condition

Stop as soon as the above two-width comparison provides one of the four conclusions. Do not proceed to additional runs without review.

Produce a short report emphasizing:

what existing evidence was reused;
how many new runs were actually required;
the magnitude of \(\Delta_\ell^{\rm nonuniform}Q\);
whether it exceeds the known numerical uncertainty;
whether K4 can now be closed.
