# Codex task: bounded BP5 normal-stress filter tests

Implement and test an optional spatial filter for the **normal stress entering reconstructed-fault friction**. The goal is to determine whether this is a useful numerical regularization for the existing BP5 run. This task replaces the proposed ordinary-transfer/native-history comparison; retain the production stress-history treatment in every branch.

Complete the implementation, focused checks, offline checkpoint assessment, and short comparisons below. Keep the experimental option disabled by default. Return a short scientific decision and reproducible commands. A short successful restart is evidence of feasibility, not long-term qualification.

## 1. Starting evidence and scope

Read the current implementation and the supplied `report(4).md` traction-definition audit. Treat its routine names as navigation hints and verify them against the checkout:

- `PhaseFieldFault::evaluate_reconstructed_fault_point` in `source/material_model/phase_field_fault.cc` forms the pointwise friction input.
- `assemble_bulk_work_system` in `source/reconstructed_fault/surface_system.cc` assembles the native work integrals.
- The BP3/BP5 diagnostics use the consistent fault mass inverse for reported traction coefficients.

In true-normal-stress mode the audited formulation is

\[
\sigma^{\mathrm{raw}}_{f,q}=\sigma^{\mathrm{bg}}_{q}+p_q-\mathbf n_q^T\boldsymbol\tau_q\mathbf n_q,
\qquad
b^{\mathrm{fric}}_i=\sum_q w_q N_i(q)\mu_q\sigma^{\mathrm{raw}}_{f,q},
\qquad w_q=JxW_q\chi_q.
\]

The previous small inclined fixture prescribed every slip rate and used constant adiabatic friction pressure. It cannot provide the physical comparison required here. Use the actual BP5 checkpoint, its true-normal mode, and its existing free friction equations.

Keep BP5 geometry, phase field, material/friction parameters, incoming state, pressure convention, background traction, boundary conditions, particle advection, stress transfer/publication, and timestep criteria identical across branches. Do not import the native-history branch or the earlier frozen-particle diagnostic protocol into this experiment. Retain the checkpoint's fixed mesh and phase-field configuration.

Use isolated output directories and the same executable/plugin build. Preserve the original checkpoint and existing results. If the server checkpoint is unavailable locally, complete the implementation and local checks and deliver runnable server PRMs/scripts; distinguish prepared work from executed runs.

## 2. Filter definition

Use the existing reconstructed-fault Q1 basis and actual production work integration. For physical arc length \(s\), assemble

\[
M_{ij}=\sum_q w_qN_i(q)N_j(q),\qquad
K_{ij}=\sum_q w_q\frac{dN_i}{ds}(q)\frac{dN_j}{ds}(q),\qquad
b_i=\sum_qw_qN_i(q)\sigma^{\mathrm{raw}}_{f,q}.
\]

For a constant filter length \(L_s\) in metres, solve

\[
H\mathbf z=\mathbf b,\qquad H=M+L_s^2K,
\qquad \overline\sigma_{f,q}=\sum_jN_j(q)z_j.
\]

Use \(\mu_q\overline\sigma_{f,q}\) in the friction term, with the existing friction coefficient and incoming state. The raw bulk pressure, current constitutive tensor, and retained particle tensors remain their original mechanical quantities. Any changes to them should arise through the changed coupled solution, not through overwriting history with filtered values.

Implementation requirements:

- Filter the total compressive friction normal input, including background exactly once. Apply one operator to the combined pressure/deviatoric contribution; retain the existing pressure gauge. Report background and mechanical perturbation separately where useful.
- Use physical arc-length derivatives, not derivatives with respect to the reference coordinate without its length factor. For a straight segment of length \(h_\Gamma\), the ordinary interior Q1 derivatives are \((-1/h_\Gamma,1/h_\Gamma)\).
- Use the actual production map, weights, endpoint treatment, and owned integration points. Check any endpoint continuation explicitly. Partition of unity and the zero sum of basis derivatives must remain valid.
- Assemble over the full supported fault before selecting plotting windows. A diagnostic window must not create a new filter boundary. Use natural zero filter flux at physical endpoints. Slip-rate Dirichlet constraints must not replace filter rows.
- Respect disconnected-fault topology and existing supported-DoF handling. Do not join separate faults or conceal unsupported rows with an arbitrary diagonal shift.
- Reduce distributed contributions once using existing ownership rules. Cache/factorize \(H\) while geometry, weights, and \(L_s\) are fixed. Invalidate caches when their dependencies change. Use sparse/tridiagonal solves or operator actions rather than forming a dense inverse.
- Recompute \(\mathbf b\) from the current raw mechanical state. Do not repeatedly filter a previously filtered field: that would make the effective smoothing depend on iteration/step count.

Provide explicit, documented modes, for example `raw`, `projected`, and `helmholtz` (names are suggestions):

| Mode | Friction normal input |
| --- | --- |
| raw, default | Original pointwise production input |
| projected, diagnostic | Q1 reconstruction from \(M\mathbf z=\mathbf b\) |
| helmholtz | Q1 reconstruction from \((M+L_s^2K)\mathbf z=\mathbf b\) |

**Zero filter length is the projected mode, not the original raw formulation.** A disabled option must genuinely bypass the new projection/filter path.

With \(K\mathbf1=0\), the filter must satisfy

\[
\mathbf1^TM\mathbf z=\mathbf1^T\mathbf b.
\]

This preserves the whole-fault work-weighted mean. It does not preserve each row load, the mean in every plotting window, or the friction load when \(\mu\) varies. Do not artificially recenter individual windows or branches to make them agree.

## 3. Coupling, nonlinear solution, and lifecycle

The filtered normal stress is a function of the current mechanical unknowns. Apply it in the actual friction residual and all associated merit/acceptance evaluations. Merely changing output, or changing residual values while retaining an inconsistent exact Jacobian, is insufficient.

For fixed geometry/weights and incoming state, the required variation is

\[
H\,\delta\mathbf z=\delta\mathbf b,
\qquad
\delta\mathbf b=\sum_qw_q\mathbf N_q\,\delta\sigma^{\mathrm{raw}}_{f,q},
\]
\[
\delta b^{\mathrm{fric}}_i
=\sum_qw_qN_i(q)
\left[\overline\sigma_{f,q}\,\delta\mu_q
+\mu_q\,\mathbf N_q^T H^{-1}\delta\mathbf b\right].
\]

Use the actual implemented derivatives of raw stress and friction. This introduces coupling between fault locations; audit any condensation or assembly assumptions that previously relied on locality. An approximate preconditioner may remain approximate, but the evaluated residual and claimed exact Jacobian must represent the new equations.

Prefer a consistent Newton/operator-action implementation. A fully converged outer iteration with filtered stress held fixed in an inner solve is an acceptable alternative if simpler: recompute the filter from the updated mechanical state, and accept only when the recomputed coupled mechanical/friction residual and filter consistency both satisfy the existing accuracy requirements. Report this choice and its extra iterations. A value held fixed from the preceding accepted timestep is a different experiment and is not authorized as a silent substitute.

Evaluate every line-search trial consistently. Keep incoming \(\Theta\) fixed within the existing mechanical solve, and commit state and stress history only once after acceptance. Rejected trials must not publish history or leave a stale filtered field in a later evaluation. Restart initialization must reconstruct the filter from the restored state without changing the retained physical state.

## 4. Focused verification before the BP5 comparisons

Reuse existing projection checks where applicable. Add only checks needed for this new operator/coupling:

1. **Operator invariants:** symmetry/support, \(K\mathbf1=0\), constant reproduction (including a 50 MPa constant), whole-fault weighted-mean preservation, and agreement with an independent small dense solve. Report absolute and scale-aware errors; judge mechanical-perturbation errors against their own scale as well as the background.
2. **Spatial selectivity:** demonstrate stronger attenuation of a short-wavelength field than a smooth one on a small uniform fault. An independent reference is a generalized mode \(K\mathbf v=\lambda M\mathbf v\), whose nodal amplitude is reduced by \(1/(1+L_s^2\lambda)\). Do not require an arbitrary linear field to be unchanged at Neumann endpoints.
3. **Coupled derivative/consistency:** for a small true-normal test with free fault equations and spatially varying \(\mu\), check the new Jacobian action against directional finite differences over a useful perturbation range. If using an outer iteration, verify the converged recomputed coupled residual instead and clearly identify the algorithm.
4. **Disabled path and MPI:** show that `raw` reproduces the unchanged formulation within existing solver tolerances, and compare filter assembly/application on one and two ranks using a small fixture. Reuse existing bulk-solver qualifications; no unrelated full-suite expansion.

Check the minimum/maximum filtered friction input. Report unexpected tension, overshoot, or activation of an existing normal-stress safeguard. Do not add clipping or stress floors to hide a failure.

## 5. Offline assessment at the actual BP5 checkpoint

Use the verified available checkpoint, expected near accepted step 5612 and \(t=5310111071.5634108\) s; confirm its metadata rather than assuming the filename/time is exact. Follow the already audited current-stress/history time levels when capturing its mechanical field.

Without advancing history, compare four representations of the same captured state:

- Original raw normal stress at native work points.
- Consistent Q1 projection, \(L_s=0\).
- Helmholtz filter, \(L_s=100\) m.
- Helmholtz filter, \(L_s=200\) m.

The two nonzero lengths are trial numerical parameters, not established physical lengths and not automatic multiples of the phase-field length scale. Verify the actual fault spacing (previously approximately 100 m).

Plot the full fault and the existing 22–35 km, 70–71 km, and 79–80 km windows; inspect physical endpoints separately. Use identical axes and preserve absolute levels. Report the weighted mean, mean-removed variation, neighboring-chord metric, and the field removed by filtering. The chord metric includes smooth curvature and must not be called pure numerical noise.

At this same frozen state, compute the actual friction loads for each representation using the same \(\mu(V,\Theta_{\rm in})\). This isolates the immediate constitutive change caused by projection/filtering before slip and state respond. Include the projected-only result so suppression due to reconstruction can be distinguished from additional along-fault smoothing.

Proceed with 100 m as the primary dynamic trial if its offline behavior is acceptable. Use 200 m for the sensitivity trial if it also preserves the broad resolved features reasonably. If 200 m plainly over-smooths those features, assess 50 m offline and use 50/100 m instead. If none is acceptable, finish with the evidence rather than expanding into a filter-length search. No approval pause is needed for these bounded choices.

## 6. Short matched restart comparisons

Prepare three complete PRMs and a runner:

| Branch | Treatment |
| --- | --- |
| R | Original raw pointwise formulation |
| F1 | Primary accepted filter length, normally 100 m |
| F2 | Neighboring accepted length, normally 200 m or otherwise 50 m |

Use copies of the same checkpoint, all associated restart files, resolved production parameters, and physical boundary/loading data. Keep the original particle/advection/history scheme in every branch. Enable true-normal friction and retain the existing free/prescribed slip constraints; do not prescribe the free slip rates to make convergence easier.

Target **10 accepted steps per branch**. Record R's actual accepted constitutive timestep sequence and replay it in F1/F2, maintaining all existing accuracy, stability, and acceptance checks. Use the recorded `actual_dt`/`stress_dt` and cumulative elapsed time; subtraction of absolute timestamps near \(5.31\times10^9\) s is unsuitable for millisecond increments.

If a filtered branch cannot accept the common schedule, record the first failure. Permit at most one controlled retry of all compared branches from the original checkpoint with a common uniformly halved schedule, provided the normal timestep criteria allow it. Limit each branch to at most 20 accepted steps across the original attempt and retry. Do not force acceptance, loosen tolerances, enlarge the state-change limit, or conduct an open-ended timestep search. Retain and report any comparable accepted prefix.

**Unlike the earlier history-transfer comparison, filtered and raw first-step solutions need not match.** Filtering intentionally changes the friction input immediately at this already-noisy checkpoint. Report that initial adjustment separately from subsequent evolution; do not ramp the filter or reinitialize stress to force agreement.

## 7. Compact diagnostics and assessment

At the initial evaluation and each accepted step, save compact whole-fault profiles and global/window summary values. Export raw quadrature samples only for the existing diagnostic windows; avoid full-volume snapshots and repeated large CSV dumps.

Required quantities:

- Raw current mechanical normal stress, total raw friction input, and actual filtered friction input. Label consistent nodal coefficients and pointwise values distinctly.
- Pressure/deviatoric splits where existing diagnostics provide them, incoming working-history roughness, and retained-particle roughness using their own consistent sampling measures. Do not compare particle and work-point RMS as if they used identical weights.
- Actual assembled friction load, shear driving load, radiation damping, and full coupled residual.
- Slip-rate profiles, incoming/committed state labels, accepted slip increments, maximum slip rate and its location, and actual timestep.
- Mean normal-stress drift, raw/filtered roughness changes, nonlinear/linear/outer iteration counts, rejected steps, minimum compression, and time spent applying the filter versus the whole step.

For two states evaluated on the same production map/weights, separate the friction-load difference into a direct normal-input contribution and a coefficient contribution, for example

\[
\Delta b_i^{\mathrm{fric}}
=\sum_qw_qN_i\left[\mu_{R,q}(\sigma_{F,q}-\sigma_{R,q})
+\sigma_{F,q}(\mu_{F,q}-\mu_{R,q})\right].
\]

Here \(\sigma_F\) is the actual filtered friction input and \(\sigma_R\) the raw branch's actual input. Verify the common map/weights before using this identity. This is an algebraic attribution, not a separate causal simulation. Shear stress also changes when the coupled solution changes.

Compare branches at matched elapsed times. Report relative slip-rate changes only where the baseline is meaningfully above its numerical floor; elsewhere report absolute rates and slip increments. Judge broad physical profiles separately from small-scale roughness. A nearly identical mean pressure or a visually smooth graph is not by itself success.

Conclude with one of these bounded outcomes:

- **Promising regularization:** filtered-input roughness decreases; coupling converges; short-term slip behavior has modest sensitivity to the two accepted lengths; raw history shows no new rapid deterioration. Recommend a longer event-level comparison, not immediate production qualification.
- **Strong filter sensitivity or distortion:** key slip/stress features or solver behavior depend strongly on the chosen length. Report the measured differences; do not choose the strongest filter merely because its curves look smoother.
- **Underlying accumulation persists:** raw retained/current stress continues to deteriorate despite a smooth friction input. Filtering has not cured the history problem; quantify the observed growth over this short window.
- **No meaningful mechanical benefit:** filtering changes appearance but has little effect on the actual friction load or slip response. Avoid further filter tuning without a specific new reason.

Do not infer recurrence-time accuracy, unchanged nucleation onset, or long-term stability from this late-checkpoint test. These are provisional restart results.

## 8. Deliverables and stopping point

Provide the code diff and experimental option documentation; focused check results; three complete restart PRMs with required plugins/data paths; exact build/run/analysis commands; and compact provenance identifying the checkpoint, executable, plugin, parameters, and timestep schedule.

Return **one short report, one summary table/JSON, and at most two composite figures**: (1) the offline stress/filter comparison, and (2) the matched restart evolution of stress, friction load, and slip. Keep large diagnostic data on the server and identify its paths.

Stop after this bounded assessment. Do not launch a full earthquake cycle, change BP5 to BP3 parameters, alter bottom loading, redesign stress transfer, or add smoothing to fault properties within this task. State clearly what was actually run and which remaining questions the short trial cannot answer.
