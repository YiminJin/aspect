# Instructions for Codex: bounded 2D test with BP5 friction, Dc=0.1 m and ell=100 m

## Purpose and scope

Create a separately named research case using the existing 2D dipping-fault implementation, BP5 friction coefficients, Dc=0.1 m and ell=100 m. Conduct a short, bounded check of initialization, spatial response, coupled evolution, restart and timestep sensitivity. This task does not request a full earthquake cycle or reproduction of the 3D BP5 benchmark.

Read the applicable AGENTS.md and relevant reconstructed-fault design/specification sections before editing. Inspect the actual checkout and record its revision and uncommitted changes. Reuse the current maintained initialization, length-scale diagnostics and coupled-comparison infrastructure. Earlier reports describe `length_scale_study.py`, `length_coupled.py`, `analyze_length_coupled.py` and the saved-mesh generator; verify current names and interfaces. Preserve existing cases, fixtures, qualification results and unrelated edits.

Do not redesign the constitutive law, work measure, aging update, normal-stress coupling, boundary completion or solver. Retain the solver backend already working on the test machine. Keep the entire fault frictional; do not restore a prescribed deep segment. No new smoothing, state diffusion, normal-velocity constraint, friction clipping or adiabatic-pressure substitution.

## Configuration

| Quantity | New case |
|---|---:|
| Shallow direct effect a | 0.004 |
| Deep direct effect a | 0.04 |
| Evolution effect b | 0.03 everywhere |
| Characteristic slip distance Dc | 0.1 m everywhere |
| Phase-field length ell | 100 m |
| Background effective normal stress | 50 MPa |
| Reference friction coefficient | 0.6 |
| Reference slip rate | 1e-6 m/s |
| Loading and intended initial slip rate | Existing 1e-9 m/s |
| Uniform shallow weakening region | 0–15 km down dip |
| Existing linear material transition | 15–18 km down dip |
| Fault spacing | Existing approximately 100 m; measure actual spacing |
| Candidate finest bulk-cell side | 24.4140625 m |
| Local fine reference cell side | 12.20703125 m, at the same ell=100 m |

Retain the existing 300-by-100-km domain, 60-degree dip, loading boundaries, G=32.03812032 GPa, viscosity, radiation damping, AT1 geometry, degradation curvature p=1 and fixed core phase value 0.6. Keep mature friction with zero cohesive history and the fixed phase profile. Fifty MPa remains the background normal stress; retain the computed normal-stress perturbation in friction.

The relevant parameter entries, subject to verification against the current parser, are:

```text
subsection Material model
  subsection Phase field fault
    set Direct effect parameters = 0.004, 0.04
    set Evolution effect parameters = 0.03
    set Characteristic slip distance = 0.1
  end
end
subsection Phase field model
  set Length scale = 100
end
```

Do not enlarge either material region in this short test. With the existing linear strengthening fraction S, a=0.004+0.036*S, and a=b at S=13/18: the actual velocity-weakening region ends at approximately 17.1667 km, not 16 km. Verify the production composition/property mapping and record this crossing. Continue using the existing initial-composition property mechanism for strengthening. Widening this material transition is different from enlarging the rupture process zone.

## Initialization: use all the new parameters consistently

Audit benchmark helpers, fixture generators, material evaluation, diagnostics and independent history checks for old a, b and Dc constants. Use the material configuration as the authoritative source; do not create independent duplicate physical parameters or silently change defaults for the original cases. A PRM-only change is insufficient if benchmark initialization still uses old constants.

Generate a fresh initial state. Retain the current initialization strategy: compute nominal shear prestress from steady sliding in the deep strengthening material, then obtain the along-fault initial state by inversion of the same regularized friction law at the intended initial velocity. In symbols,

    q0 = sigma0 * mu(Vinit, Dc/Vinit; a_deep, b, Dc) + eta_d * Vinit,
    q0 - eta_d*Vinit = sigma0 * mu(Vinit, Theta0(s); a(s), b, Dc).

Use the existing stable inverse-friction implementation with the NEW parameters. Do not merely multiply the old Theta profile by Dc_new/Dc_old: a and b also change. Do not replace the current nonsteady shallow initialization with steady state everywhere.

Useful nominal interior sanity values, before spatial background corrections and the initial coupled solve, are:

- Deep state: Theta0 = Dc/Vinit = 1e8 s.
- Shallow state: approximately 2.512e4 s from the inverse law.
- Nominal shear prestress: approximately 26.5461 MPa, including the negligible radiation-damping contribution. The deep steady-state contrast a-b remains 0.01, so this nominal prestress is essentially unchanged.

Retain the current physical background spatial correction as the same physical function. Do not reinterpret a captured physical correction using a new I_h denominator, or fit it to a new accepted solution. Regenerate mesh-dependent projections with the existing procedure. Distinguish the nominal inverse-law check from actual t=0 V after the coupled solve; existing physical background structure can make those velocities differ.

Check configured a(s), b, Dc, initial state, friction residual and positive state at representative shallow, transition and deep locations. Verify the new values in actual production quadrature/material evaluation as well as benchmark helpers. Do not restart this altered physical model from an old-case checkpoint. Checkpointing below refers only to this freshly initialized case.

## Profile and spatial setup

Generate the stationary physical profile for ell=100 m through the existing implementation. Regenerate H/profile metadata, realized Q1 phase, I_h, endpoint completion and all dependent inputs/caches for each exact mesh. Retain the qualified source/work treatment at both physical endpoints. Do not reuse completion files from ell=50 or ell=400 m.

For the unchanged nominal profile, the full support is approximately 395.292 m and the chi RMS width approximately 44.224 m. Report intended and realized peak, support, RMS width and completed-column normalization. Check the existing endpoint mass/source/weak-row identities; an in-box truncated column need not itself integrate to one when the formulation includes completion.

Use the candidate resolution throughout the active band, with the existing coarser far field. No special refinement at 40 km. A locally finer reference should include the transition/probe region and both physical endpoints with adequate halos; reuse the previous bounded reference layout where appropriate. Keep the physical profile, fault grid and loading identical. Identify artificial patch edges and include them in diagnostics: a local fine mesh is not automatically a globally more accurate solution.

Preserve the original 1% profile-width target and earlier failed records. For THIS bounded diagnostic, an approximately 2% candidate RMS-width error is an explicitly permitted diagnostic exception if normalization/work checks and mechanical comparisons pass. Use the existing opt-in diagnostic path; do not weaken global production gates or relabel the old result as a pass. This permits testing the economical ell/h approximately 4.1 candidate without repeating the earlier refusal loop.

## Bounded test sequence

Aggregate simulation runtime cap: 60 minutes, including probes and all comparison branches. Record setup/build/completion-generation time separately. Use at most the previously successful rank count unless resources require fewer; record memory and actual wall time. Estimate the remaining budget after the first solve. Use graceful termination between accepted steps plus an external timeout. Do not automatically extend the budget, repeat failed simulations, launch a parameter sweep or continue until an event appears. Mark checks left unfinished by the cap as UNRUN.

### A. Two frozen mechanical probes on two meshes

Reuse the existing frozen bulk-response machinery for a nominal 3125-m disturbance and the 200-m alternating fault-grid input, on candidate and local-reference meshes: four linear responses total. Keep the same physical envelope, input amplitude, incoming histories, coefficient kappa and physical observation windows across meshes. Make the envelope and fine patch large enough for the long input; do not squeeze it into the old 3-km window merely to reuse a filename.

Compare native work-conjugate shear response, sign, work closure and stiffness. Report the actual Q1/tapered input spectrum and continuum prediction; do not compare its measured stiffness directly to a pure-sinusoid number. Convert rate stiffness to elastic stiffness with G/kappa. Target less than 5% change on bulk refinement for these probes. Do not treat agreement between two meshes alone as proof of sharp-fault accuracy.

For orientation, the pure-mode continuum ratio K_band/K_sharp at ell=100 m is about 0.8523 at 3750 m and 0.8261 at 3125 m. The first is the provisional target obtained by scaling the previous 600-m wavelength with Dc/(b*sigma); the second allows for a 60-MPa stress level. This target is an engineering choice, not an exact process-zone wavelength. Do not require 80% or 90% sharp-fault response at the 200-m grid wavelength.

Reuse the cheap analytical short-wave stability screen with the new parameters. At 50 MPa, Kc approximately equals sigma*(b-a)/Dc = 13 MPa/m in the shallow weakening region; at 60 MPa it is 15.6 MPa/m. The nominal consistent-Q1 alternating stiffness at ell=100 m and 100-m fault spacing is approximately 104.94 MPa/m. Check the intended short-wave interval, not only this endpoint. This is a uniform, steady-state screen, not a proof of transient stability; intended long-wave nucleation modes need not be stable.

### B. Short coupled candidate and spatial comparison

If initialization/work checks and the frozen comparison pass, run the fresh candidate for six accepted REAL timesteps, stopping earlier at the runtime cap. Do not count the artificial initialization interval as elapsed physical time. Use the production timestep controller, capped at 4e6 s per real interval for this test; let it shorten steps whenever required. Keep the existing artificial initialization coefficient unless current maintenance requires a documented correction.

Save compact diagnostics at t=0 and every accepted step. Create only the specific restart checkpoint needed below and a final checkpoint. Use initial/final heavy visualization and the existing slip-triggered output, not a full snapshot at every step.

Run the fine reference only through the first two candidate physical intervals. Use the candidate's accepted time boundaries as targets; subdivide further if the reference controller requires it. Compare at common physical times, never by step number. Do not force a timestep larger than the production safety limit.

### C. Short restart and half-step checks

If the remaining budget permits, reuse the candidate checkpoint after its second accepted real step:

- Restart and reproduce candidate intervals 3 and 4, using their recorded time targets. Compare against the uninterrupted candidate at the same final time and use the existing restart-equivalence tolerances.
- From that same checkpoint, subdivide each of intervals 3 and 4 into halves, or finer if required by the safety controller. Reach the same final time. Recompute all timestep-dependent Maxwell and history coefficients normally; do not change only the diagnostic clock.

These are two short branches, not additional six-step runs. A smaller step should not be imposed by bypassing the constitutive update. If rapid acceleration makes this bounded schedule impractical, stop with the accepted data and report that behavior rather than trying to finish a rupture in this task.

## Diagnostics, interpretation and stopping rules

Record resolved parameters/input and binary hashes, cells/DoFs/particles, peak memory, accepted/rejected steps, physical time, dt, nonlinear and linear convergence, maximum V*dt/Dc, state positivity, and any rate floors/clipping. Check the ordinary aging, Maxwell-history and fixed-profile invariants using the new parameters. Verify that no deep nodes have become prescribed inadvertently.

Export raw fault-coordinate profiles of V, incoming and outgoing Theta, cumulative slip, native work-averaged shear and normal traction, and the force residual. Current stress must use accepted u,p,V with the SAME incoming working history as the mechanical solve. Do not reconstruct it from newly committed stress or substitute outgoing Theta into a balance that used incoming Theta.

Compare t=0 and matched evolving profiles in the top 0–2 km, transition 13–20 km, a deep interior control, the last 2 km and all artificial patch-edge neighborhoods. Report weighted RMS and localized maximum differences. Show both absolute and relative errors, especially near rate floors and nearly steady nodes. Separate initial differences from changes accumulated during the short run.

Use approximately 1% agreement in resolved V, Theta and slip profiles as a practical diagnostic target, while explicitly showing localized exceptions. For tractions report errors in Pa and relative to their evolving increments. Do not reject solely because an almost-zero increment has a large percentage error, or conceal such errors by dividing only by 50 MPa. A useful additional scale is the resulting friction-load error relative to sigma*dmu/dln(V)+eta_d*V; label this as a fixed-state sensitivity estimate, not a bound on future nonlinear error.

Stop on failed mechanical/work closure, incorrect parameter/history plumbing, solver failure, unexpected loss of source support, or a substantial unexplained discrepancy in the resolved coupled fields. Do not alter physics or tolerances until plots look smooth. Report any mismatch with an authoritative specification instead of silently changing the formulation.

No oscillation during six steps is evidence of short-run feasibility only. It does not establish convergence of earthquake cycles, absence of a late notch, correct rupture-front resolution, or sustained deep sliding at Vp.

## Deliverables and next decision

Deliver a separately named PRM and reproducible input/launch commands, minimal parameter-consistency changes, resolved parameter/initialization checks, compact comparison plots/tables and a report listing PASS / DISCREPANCY / UNRUN for every requested comparison. State actual time and memory costs and any diagnostic exception used.

Do not execute an ell=50-m evolution or any full-cycle simulation in this task. Keep ell=50 m as a subsequent width-sensitivity reference, clearly distinct from the ell=100-m fine-BULK-mesh comparison. Prepare a bounded follow-up command only if useful; do not launch it automatically.

Keep the 0–15 / 15–18-km material layout for this test. The homogeneous low-a/b full nucleation-length estimate is approximately 11.77 km for the selected parameters; it suggests feasibility but is not a prediction for the dipping heterogeneous model. If later scientific goals call for more room for nucleation, consider a separately named case with uniform weakening extended to 20 km and the same 3-km transition moved to 20–23 km. That geometry change and transition-width experiments are outside this short task.
