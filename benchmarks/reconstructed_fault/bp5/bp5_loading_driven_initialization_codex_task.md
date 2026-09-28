# Codex task: loading-driven initialization with controlled early locking

## Decision and scope

This task supersedes the proposed localized nucleation seed. Do not implement a seed, hot patch, imposed fault velocity segment, local friction-parameter change, or time-dependent triggering load.

The objective is a dipping-fault run that develops slip deficit and mechanical stress concentration through its own evolution, then may nucleate an earthquake. Retain the current 2D geometry, BP5 friction parameters, resolved phase-field band, and fully frictional fault. This is a modified BP3-style loading experiment, not the standard BP3 benchmark.

The new initialization should balance friction at the initial plate rate while placing the shallow state moderately below its steady-sliding value. Use one scalar parameter to control that departure. The candidate selected here is \(R_{\rm VW}=0.8\). This is an analytically screened starting choice, not a full-model calibrated optimum or a guarantee of an earthquake.

Implement the benchmark-local change, perform the bounded checks below, and regenerate a separate server package. The user will launch the long server run. Do not start another local long-run or parameter-search campaign.

## 1. Why the original uniform-prestress recipe must be adjusted

Standard BP3 chooses a uniform initial shear traction based on the deep strengthening region and solves for a spatially variable initial state. Thus initial frictional force balance need not imply steady aging. See [BP3 specification, equations 25–26](https://strike.scec.org/cvws/seas/download/SEAS_BP3.pdf).

With our current \(a_{\rm VW}=0.004,\ a_{\rm VS}=0.04,\ b=0.03\), copying that recipe gives approximately

\[
R_{\rm VW}\equiv\frac{V_p\Theta_{\rm VW}}{D_c}
=\left(\frac{V_p}{V_{\rm ref}}\right)^{(a_{\rm VS}-a_{\rm VW})/b}
=10^{-3.6}\simeq2.5119\times10^{-4}.
\]

For \(D_c=0.1\) m and \(V_p=10^{-9}\) m/s, the shallow state is only 25119 s instead of its steady value \(10^8\) s. This recreates the rapid-aging initialization of the failed eight-day run.

Preserve BP3's qualitative loading-driven evolution, but explicitly change the prestress profile so that the initial state departure is controlled. Keeping a uniform 26.546-MPa shear background, initial velocity \(V_p\), the current friction parameters, and \(R_{\rm VW}=0.8\) simultaneously is impossible. State and prestress must be changed together.

## 2. Keep the current physical parameters

Retain:

- 300 by 100 km domain; 60-degree fault; fully frictional geometry;
- uniform weakening region 0–30 km; existing transition 30–33 km and deep strengthening region;
- \(a=0.004/0.04,\ b=0.03,\ D_c=0.1\) m, \(V_p=V_{\rm init}=10^{-9}\) m/s;
- \(V_{\rm ref}=10^{-6}\) m/s, reference friction 0.6, background effective normal traction 50 MPa, and the current evolving mechanical normal traction;
- radiation damping, elastic modulus, viscosity, frozen AT1 \(\ell=100\) m, zero cohesive history, mesh, particles, fault grid and endpoint corrections;
- AMG/condensed solver and existing nonlinear/linear tolerances.

Do not restore the old small \(D_c\), widen the transition, add an Airy load, change pressure treatment, impose deep slip, or modify the constitutive/time-integration equations in this task.

## 3. Initial state profile

Let \(T=D_c/V_p=10^8\) s. Introduce a benchmark parameter named, for example, "Weakening initial state ratio", with default 1.0 for the existing steady fixture and value 0.8 for this run. This is a new parameter to implement, not an already available input entry. Require \(0<R_{\rm VW}\le1\).

Use the existing projected strengthening mixture on the reconstructed fault. With the configured mixture's effective \(a_i\), define

\[
f_i=\frac{a_i-a_{\rm VW}}{a_{\rm VS}-a_{\rm VW}},
\qquad
R_i=R_{\rm VW}^{\,1-f_i},
\qquad
\Theta_{0,i}=T R_i.
\]

Use the production composition-fraction evaluation. Its valid fractions imply \(0\le f_i\le1\); handle only roundoff at those bounds, not an invalid projection by silently clipping it.

The resulting target is:

| Region | Initial state | Initial \(R=V_p\Theta_0/D_c\) |
|---|---:|---:|
| Uniform weakening, 0–30 km | \(8\times10^7\) s | 0.8 |
| Transition, 30–33 km | Smooth change through \(T\,0.8^{1-f}\) | 0.8 to 1 |
| Deep strengthening | \(10^8\) s | 1 |

Keep initial velocity \(V_p\) over the entire fault, initial accumulated slip zero, and committed Maxwell perturbation stress zero. Artificial initialization must not advance state, slip or history.

Reuse the current initialization hooks for nodal state and the initial composition inputs. The final reconstructed-fault nodal state is authoritative for the friction solve. Construct it after the ordinary material projection, then preserve it through initialization. Do not allow a legacy constant-state initializer, inverse-state solve, or subsequent particle projection to overwrite it. Align diagnostics with this explicitly chosen nodal state and the production Q1 interpolation.

This changes state and stress initialization only. All fault nodes remain free throughout the physical run.

## 4. Build the background from the actual discrete state

Use the native weak initializer that removed the earlier transition pulses. Evaluate the friction law using the ACTUAL Q1 interpolated initial state \(\Theta_{0,h}\), actual projected material mixture, and constant target velocity \(V_p\):

\[
M\tau_{\rm bg}
=\int N_i\left[
\sigma_{\rm bg}\,
\mu(V_p,\Theta_{0,h},\mathbf f_\Gamma)
+\eta_{\rm rad}V_p
\right]d\nu.
\]

Keep the same work measure, mass matrix, quadrature, source associations, MPI reductions and endpoint continuations as the qualified initializer. Retain the established normal background, correction conventions and empty captured-prestress filename.

Do not analytically evaluate \(\Theta(s)\) at quadrature points in this integral if production uses Q1 nodal state. Do not substitute pointwise/nodal friction inversion or independently interpolate an analytic shear-stress profile. Those inconsistencies caused earlier pulses.

The private target evaluation must retain the qualified initializer's exclusion of crack-induced mechanical shear. It supplies friction plus damping only. Solve the actual initial mechanical problem normally and report its departure from target \(V_p\); do not cancel that mechanical response by recalibrating the background.

For plateau checks, relative to the steady-state background:

\[
\Delta\tau_{\rm VW}
\simeq b\sigma_{\rm bg}\ln(0.8)
=-0.3347153\ {\rm MPa}.
\]

The expected plateau background tractions, including the negligible plate-rate damping, are:

- shallow weakening: approximately 38.64536654 MPa;
- deep strengthening: approximately 26.54612237 MPa.

The exact configured regularized law and native weak construction remain authoritative. Small projection overshoots are not to be clipped afterward. Distinguish the imposed background profile from the subsequent mechanical traction change in outputs.

## 5. Expected behavior and analytic screening

The aging law gives

\[
\dot\Theta_0=1-R_i.
\]

Thus the shallow state initially grows at 0.2 s/s while the deep strengthening state is stationary. As shallow state grows, resistance at fixed velocity rises. If the shear loading does not keep pace, shallow velocity decreases, producing slip deficit relative to \(V_p\). Elastic coupling can then concentrate additional traction near the transition/creep front. This is the intended pathway, not a requirement that the code force locking or nucleation at a predetermined location.

A local constant-traction calculation was used to screen the choice. For positive velocity in the logarithmic regime, with constant normal traction and negligible damping at these slow rates,

\[
\frac{V}{V_p}=x^{-m},\quad
x=\frac{\Theta}{R_{\rm VW}T},\quad m=b/a=7.5,
\qquad
\frac{dx}{dt}=\frac{1-R_{\rm VW}x^{1-m}}{R_{\rm VW}T}.
\]

Numerical quadrature of this scalar equation gives:

| Initial ratio | Shallow prestress offset from steady (MPa) | Initial predictor allowance (s) | Local time to \(V/V_p=0.1\) (yr) |
|---:|---:|---:|---:|
| 0.5 | -1.03972 | 267380 | 0.736 |
| **0.8** | **-0.33472** | **1073835** | **1.541** |
| 0.9 | -0.15804 | 2432551 | 2.044 |

These are inexpensive analytic-screen results, NOT predictions for the coupled model. They omit elastic loading, spatial interaction and normal-stress evolution. They support using 0.8 as a moderate departure: less abrupt than the old young-state initialization, but with a definite early aging tendency. Do not run three production cases merely because this table lists three choices.

Initialization does not fix the unresolved low-velocity active-set limitation. Under constant traction indefinitely, even this candidate could eventually approach the velocity floor. A later floor-related failure must be diagnosed as such rather than repeatedly avoided by retuning initial state.

## 6. Timestep settings

Use the following changes to the existing full configuration:

~~~text
set Use years instead of seconds = false
set Maximum first time step = 1e6
set Maximum time step = 1e7
set CFL number = 0.5

subsection Material model
  subsection Phase field fault
    set Initial time step = 1e6
  end
end

subsection Time stepping
  set List of model names = convection time step, reconstructed fault time step, BP5 state startup
  subsection BP5 state startup
    set Maximum logarithmic state change = 0.02
  end
end
~~~

Add the newly declared initial-state ratio parameter with value 0.8 in the appropriate existing benchmark subsection. Keep the current timestep growth restriction.

Artificial initialization and the first physical ceiling are both \(10^6\) s, about 11.57 days. The first physical step remains subject to all controllers. These values replace both the seeded millisecond proposal and the earlier \(4\times10^6\)-s first step.

For an initially constant \(V_p\),

\[
\Theta_{\rm pred}(\Delta t)
=T+(\Theta_0-T)e^{-\Delta t/T}.
\]

At shallow \(\Theta_0=0.8T\), a \(10^6\)-s interval gives a weighted log-state change of approximately 0.0186334. The exact frozen-rate 0.02 predictor allowance is approximately \(1.073835\times10^6\) s. The ordinary fault allowance is approximately \(6.67\times10^6\) s, so the first-step ceiling should control initially, subject to actual accepted velocity.

The predictor is an accuracy heuristic, not a guarantee about the realized change after the new mechanical solve. Record both measures. Do not relax 0.02 in this task or force a longer step when it becomes restrictive.

## 7. Minimal bounded qualification

Reuse the qualified mesh, initialization machinery and restart infrastructure. Run only the 0.8 candidate:

1. One fresh initialization plus at most three accepted adaptive real steps; save a checkpoint after the first real step.
2. Reuse its second real step as a full-step control. Restore the same checkpoint and compare two half-steps to the identical physical endpoint, using the existing checked branching mechanism. Respect live controller restrictions.

Use the existing four-rank release configuration and 1100/1200-s soft/hard wall limits per run. No long local run, parameter sweep, or repeated cap-halving campaign.

Verify native weak background construction with the existing allowance, nodal state retention, initial zero committed stress/slip, exact once-only state/slip/Maxwell updates, source/profile/\(I_h\) retention, fresh linear residuals and restart without reinitialization. Confirm all nodes remain free initially and that initial velocities stay close to \(V_p\). Investigate an initial maximum \(|V/V_p-1|>0.005\), rather than forcing it away.

Plot accepted \(V/V_p\), incoming/committed \(\Theta\), \(R=V\Theta/D_c\), slip deficit \(V_pt-\delta\), and native weak mechanical traction changes relative to initialization. Include both the whole fault and 27–36 km. Report endpoint behavior separately as well.

The early test should establish whether the intended aging/slowdown begins smoothly. It is too short to require earthquake nucleation or a mature stress concentration. Do not assert either outcome from three steps.

For the full/half comparison, report maxima and work-weighted RMS differences in log velocity, log state, slip and weak tractions. Also retain the evolving-change metric, but do not normalize only by background fields. Predeclare provisional startup screens:

- maximum \(|\log(V_{\rm full}/V_{\rm half})|\le0.02\) where either velocity is at least \(0.1V_p\), with full-domain absolute rate differences also reported;
- maximum \(|\log(\Theta_{\rm full}/\Theta_{\rm half})|\le10^{-3}\);
- maximum accumulated-slip difference divided by \(D_c\le10^{-4}\).

Passing supports an exploratory run, not full-cycle temporal convergence. If a screen fails, report which term dominates and a focused next action; do not automatically change physical initial state to make a numerical comparison pass.

## 8. Server package

After the bounded checks pass, prepare server-30km-loading/ with a distinct output directory. Preserve existing steady and historical packages. Reuse the initializer implementation with an explicit mode/ratio and distinct initial-condition metadata; avoid a duplicated solver or a new ongoing particle property.

Archive the ratio, profile definition, projected material inputs, initial nodal state, native weak background and exact configuration. A default ratio of 1 must preserve the steady variant's behavior. Restart restores authoritative histories/background and output/event state, never recomputes initialization, and rejects incompatible steady/inverse/seeded initial-condition identities.

For the server, remove the local three-step/time limits and restore ordinary first-event-through-decay stopping, the existing 1500-year safety end, scheduler wall margin, and hourly/on-termination checkpoints. Start fresh. Keep all adaptive restrictions. The \(10^7\)-s ceiling is exploratory, not a claim of prior long-step accuracy qualification.

Retain sparse coordinated ASPECT/particle/fault output, cheap accepted-step station/convergence histories, cumulative-slip and slip-deficit profiles, and the actual timestep limiter. Disable expensive full-state dumps and probe callbacks every step. Preserve restart points for later replay near acceleration. No automatic server submission.

The final report should include the realized initial \(R\), plateau prestress, transition weak-balance errors, actual timestep proposals, observed early slowdown, mechanical traction changes, full/half differences and execution cost. Clearly separate the scalar screening above, the bounded coupled test, and the future first-event result.
