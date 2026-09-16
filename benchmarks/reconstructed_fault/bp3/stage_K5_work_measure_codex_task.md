# K5: implement and qualify the mechanical work measure

Read `stage_K5_endpoint_weak_form_review.md` and the current design/specification. Implement the proposed mechanical formulation as an explicit opt-in mode for the straight, frozen, mature BP3 configuration. This task includes changing the mechanical weighting and stress sampling throughout the represented fault. The existing volume-weighted mode remains available.

## 1. Formulation and common sampling

Use the existing bulk-source coordinate and continued basis
\(b_i=N_i(\widetilde s(x))\), with constant endpoint extension in the enabled wedges. Keep both successful boundary normalization/source corrections unchanged. At each locally owned physical Stokes quadrature point, use the exact source support, FE phase, completed Ih, material coefficients, and frozen working FE stress used by bulk assembly.

Assemble every mechanical surface row as

\[
R_i=\sum_q J_q\chi_q b_i(x_q)
\left[q_q-\mu(V_\Gamma,\Theta_{k-1})\sigma_{n,q}-\eta_q^d V_\Gamma\right],
\qquad V_\Gamma=\sum_j b_jV_j.
\]

Here C=0, and q and sigma_n retain the review's background and signed-stress conventions. Apply the common multiplier to all traction terms. Evaluate committed surface Theta using the existing interpolation and endpoint extension; freeze it during mechanics.

Derive K=-dR/dV and G=dR/d(u,p) from this same evaluation, using the reviewed formulas. Retain true normal-stress feedback and current pressure scaling. Only the shear part of G must satisfy the discrete work identity with B; keep the nonsymmetric solver.

Reuse bulk quadrature/cache data where practical. Visit each owned QP once, with no overlapping wedge pass. Do not approximate this by multiplying parent-center particle weights by chi. At chi=0, all mechanical surface contributions vanish. Keep generic property projection, particle ownership, surface topology, phase evolution policy and history-commit rules unchanged. Restrict the new mode to the stated scope.

## 2. Scales and initialization

Use the new positive row measure
\(m_i=\sum_qJ_q\chi_qb_i\) for traction-scaled residual/reaction diagnostics. Check all existing stopping and active-set scales dimensionally; preserve their physical accuracy and sign conventions. Document required unit conversions instead of relaxing acceptance thresholds.

Keep the existing fixed background fields for this qualification and report their initial residual under the new measure. Do not recalibrate them to force agreement. Use a fresh diagnostic fixture; do not reinterpret a production checkpoint as the revised model. Preserve the tested constant endpoint extension; reflected-field continuation is outside this task.

## 3. One bounded qualification case

Use the existing mesh and a frozen-history, noncommitting coupled fixture with the original RSF nodes free, including the top, and deep V prescribed as before. Include a nonzero top-velocity perturbation and a nonuniform, constraint-compatible bulk perturbation so the checks exercise more than uniform equilibrium.

On the actual production paths, verify:

- Directional finite differences of R against K and G, including pressure coupling.
- Shear virtual work against B on the common quadrature with homogeneous bulk constraints; also check the affine shear contribution uses the same old FE stress.
- Zero intact contribution, unchanged source support, and no duplicated wedge contributions.
- One coupled convergence result with the existing fresh-linear checks, reporting top V, traction-scaled residual and any active bound. Confirm no history was committed.

Target one four-rank qualification case with a 10-minute solve cap. Reuse valid prior checks. If it fails, report the specific blocker; do not launch a refinement, timestep, restart or long-trajectory campaign. A prescribed-uniform replay cannot qualify the free endpoint.

## Deliverable and stopping point

Update the formulation/mode documentation and provide a short report with commands, changed files, measured check errors and the coupled outcome. Stop when this free-endpoint formulation is qualified or a concrete blocker is identified. Successful qualification authorizes proposing the short mature-RSF replay next; that replay is outside this task. Keep unrelated work intact.
