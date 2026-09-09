• K0 is complete as an inventory and baseline audit. I recorded the source map, lifecycle contract, test evidence, and concrete K1 proposal in doc/reconstructed_fault/benchmarking/stage_K_progress.md. It uses revision f4032b1824ef4892020af8c58ef981aca03dc0ec, the corrected specification, and the final Stage-J recovery record.

The audit found one prerequisite for K1: the coupled solver does not appear to establish nonzero velocity boundary values in its initial bulk iterate. It copies solution into `working_x`, then updates that vector using homogeneous Newton directions. I found no corresponding application of the physical, inhomogeneous velocity constraints to the base iterate. That affects both initial imposed shear and subsequent loading changes. The zero-velocity Stage-J fixture does not cover it. See the bulk iterate construction (`source/simulator/solver.cc`:1020) and the homogeneous direction conversion (`source/simulator/solver/reconstructed_fault_condensed_system.cc`:218).

This is a source-level finding, not a new runtime reproducer. It needs verification and a focused correction before K1’s imposed-shear results can be trusted. I made no production changes.

The K0 baseline results are:

Check | Result
--- | ---
Build with -j4, Debug, Voro enabled | Passed
Stage-J lifecycle, two ranks | Passed in 320.99 s
Initial and subsequent phase residuals | 9.395e-6, 4.747e-6, 4.879e-7, all below the requested 1e-5
Constitutive/domain/nonlinear unit selection | 65 assertions, 15 cases passed               
Adaptive and distributed Q1 \(I_h\) accuracy | 59,117 assertions, 2 cases passed
Existing rate-state integration test | Passed in 27.46 s
Original condensed-system fixtures | Both failed before coupling verification: inherited ten-iteration phase budget exhausted
Same dynamic-pressure checks, temporary budget 50 | Passed, one rank, 125.4 s
Same adiabatic-pressure checks, temporary budget 50 | Passed, two ranks, 68.23 s

The temporary coupling inputs retained the original accuracy and assertions. Both converged in 20 phase iterations to 9.395e-6. The original checked-in fixtures therefore remain stale; I did not rewrite their expected outputs. Exact commands and preserved diagnostics are in the packet.

The proposed K1 case is a short periodic strip with square cells:

Quantity | Proposed value
--- | ---
Domain | \((0,0.25)\times(-0.5,0.5)~\mathrm m\)
Fault | Ordered horizontal line from \((0,0)\) to \((0.25,0)\)
Phase model | AT1; prescribed core \(\widehat\phi=0.6\)
Regularization length | \(\ell=0.15625~\mathrm m\)
Activation / upper admissibility | Existing \(0.1 / 0.99\)
Bulk and surface shear modulus | \(G_b=G_\Gamma=10^6~\mathrm{Pa}\)
Bulk and surface viscosity | \(\eta_b=\eta_\Gamma=10^8~\mathrm{Pa~s}\)
Relaxation times | \(\lambda_b=\lambda_\Gamma=100~\mathrm s\)
Viscosity cutoffs | \(10^6–10^{10}~\mathrm{Pa~s}\), explicitly avoiding the much larger default lower cutoff
Cohesion / fracture energy | \(c=1000~\mathrm{Pa}\), \(G_c=80/3~\mathrm{J~m^{-2}}\)
Degradation | Curvature (p=1); calibrated (m=128)
Temperature / composition | Uniform \(293~\mathrm K\); one background material, no chemical fields
Frictional normal pressure | Prescribed \(\sigma_*=1000~\mathrm{Pa}\), adiabatic mode, zero gravity
Friction | Regularized rate state; \(\mu_0=0.6,\ a=0.025,\ b=0.013\)
Reference/minimum slip rate | \(10^{-5}/10^{-12}\,\mathrm{m\,s^{-1}}\)
State parameters | (D_c=0.001,\mathrm m,\ \Theta_0=200,\mathrm s)
Radiation damping | \(10^5\,\mathrm{Pa\,s\,m^{-1}}\)
Initial particle stress | \(\tau_{xx}=\tau_{yy}=0,\ \tau_{xy}=1500\,\mathrm{Pa}\)
Numerical initialization interval | \(2\,\mathrm s\), held fixed during refinement
Loading | \(U(t)=10^{-4}[1+0.2\min(t/(4\,\mathrm s),1)]\,\mathrm{m\,s^{-1}}\)
Pilot duration | Initialization plus three nominal \(2,\mathrm s\) real steps
Pilot mesh | \(16\times64\) square cells; \(\ell/\Delta x_n=10\)
Particles | \(3\times3\) per cell: 9,216, with Voronoi/CPDI
Fault discretization | Eight segments, spacing \(1/32\,\mathrm m\)
FE | Q2 velocity, Q1 pressure and phase field

Independent screening of the ideal stationary profile gives a half-width of approximately \(0.30882~\mathrm m\) and an \(H_0\) full width at half maximum of \(0.07875~\mathrm m\): approximately five cells and fifteen particle spacings across the peak at pilot resolution. These are screening estimates; the production activation cutoff and FE initialization change the realized profile.

For boundary conditions, prescribe
\[
  \mathbf u_{\rm top}=(U_k/2,0),\qquad
  \mathbf u_{\rm bottom}=(-U_k/2,0),
\]
and use existing left/right bulk periodicity. The reconstructed fault retains its two separate endpoints. K1 must check endpoint support, reconstructed orientation, particle-domain volume, and along-fault uniformity. Existing FE periodicity and particle wrapping do not establish that the Voronoi construction is periodic. The short pilot is expected not to reach particle wrapping.

For the fixed phase field, Evolve phase field=false alone is insufficient: it freezes \(H\), while subsequent phase solves may further refine \(\phi\). I propose a benchmark-only constraint fixture using the existing constraint hook: after normal production  initial Q1 values. This makes the fixed-profile restriction explicit while retaining production mechanics, particle advection, and history updates.

Initialization follows the approved Stage-J semantics:

- Solve and commit \(V_0\) and the bulk kinematics.
- Retain supplied \(\Theta_0\), initialized \(H_0\), initial projected chesive traction \(C_0=T_0^{\rm coh}\), and supplied particle stress.
- Store \(I_{h,0}\) as the previous-normalization snapshot.
- Do not age \(\Theta\) or evolve \(H\) through the numerical initialization interval.

The initial evaluated stress and cohesive response must be recorded separately from the retained histories. The reference must use those retained histories for the first real step.

The independent calculation will integrate (h(\max(\phi_h,0))) from exported FE profile data, using independently written degradation formulas. It will also integrate\ outside the actual association strip, so omitted deformation cannot be hidden in the normalization.

For each actual accepted timestep, the scalar reference solves
\[
  F_k(V)=
  \beta_bq_{k-1}+\frac{\kappa_b}{W}(U_k-V)
  -\frac{\kappa_\Gamma}{I_h}V-\beta_\Gamma C_{k-1}
  -\sigma_*\mu(V,\Theta_{k-1})-\eta^dV=0,
\]
using a bracketed Brent solve. It then advances (q,C), and
\[
  \Theta_k=
  \Theta_{k-1}e^{-V_k\Delta t_k/D_c}
  -\frac{D_c}{V_k}\operatorname{expm1}
  \!\left(-V_k\Delta t_k/D_c\right).
\]
The mechanics uses frozen (\Theta_{k-1}). Reference histories are initialized once, never reset from later ASPECT output. The velocity-profile comparison retains both bulk deformation and localized slip:
\[
  u_x(y)=-\frac{U_k}{2}
  +\frac{q_k-\beta_bq_{k-1}}{\kappa_b}(y+W/2)
  +V_k\int_{-W/2}^{y}\frac{h(\phi_h(z))}{I_h},dz.
\]

The proposed targets are:

Measure | Target
--- | ---
Phase / coupled nonlinear residual | 1e-8, with actual convergence required
Phase / Stokes linear tolerance | 2e-7 / 1e-9
Same-FE \(I_h\) comparison | Relative error \(\le10^{-6}\)
Omitted association-strip integral | Fraction \(\le10^{-6}\)
Integrated-slip normalization | Relative error \(\le10^{-4}\)
esolved \(V,q,C,u_x\), cumulative slip | \(0.2\%\) relative plus declared dimensional absolute allowances
Along-fault variation | \(\le10^{-4}\) of fixed diagnostic scales
State update | Independent per-vertex comparison; expected increment exceeds 100 comparison tolerances
Geometry | Position error \(/\ell\le10^{-6}\), segment-angle error \(\le10^{-6}\,\mathrm{rad}\)

  Estimated pilot runtime is 2–6 minutes on one local rank, excluding first compilation, with a ten-minute pilot limit. Actual wall time and peak memory will determine whether the proposed spatial levels \(16\times64,\ 32\times128,\ 64\times256\) and timestep levels \(2,\ 1,\ 0.5\,\mathrm s\) are affordable. The packet includes the proposed commands and the distinction between measured costs and estimates.

  K1 implementation is paused for review, with the nonzero-boundary-condition prerequisite explicitly identified.
