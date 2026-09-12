# K3 preparation: homogeneous evolving-profile reference and bounded smoke

## Scope and sequencing

The user closes K2.4 as the completed bounded parity/alignment-sensitivity
check. At the real steps its fixed-interior matched true-normal-stress changes
are approximately 1% or less; it does not explain the larger K2.3 spatial
uncertainty. The initial-state comparison was about 1.3%, and is not erased.
K2 is bounded verification with unresolved convergence: K2.2 retains temporal
traction uncertainty and K2.3 retains interior true-normal-stress spatial
uncertainty. Their references remain provisional; **Gate K2 is not passed**.
No 128x512 K2 run is authorized.

K3 preparation is explicitly authorized as a sequencing exception because it
returns to the homogeneous-along-fault problem with an independent 1-D
reference. This is not evidence of K2 convergence. This document specifies
the reference and proposes one initialization-plus-two-real-step smoke test.
The independent reference and opt-in benchmark mode are implemented and build.
The first smoke stopped on a benchmark fingerprint initialization bug. The
explicitly approved corrected invocation now passes initialization and both
real steps; see `stage_K3_smoke_result.md`. No production algorithm was edited.
The original .009-m/s preflight was rejected for smoke. The authorized bounded
adjustment selects .00225 m/s after .0045 m/s also failed normalization.
Both independent and separately production-initialized references pass the
unchanged preflight checks; see `stage_K3_bounded_adjustment.md`. The successful
one-rank smoke also passes both support gates, narrowly for total normalization.
No further invocation or convergence campaign is authorized automatically.

## 1. Independent reference equations

Use y in [-W/2,W/2], W=1 m, horizontal fault y=0, tangent e_x and normal e_y.
All coefficients and surface fields are homogeneous in the tangential
direction. The reference is independent numerical quadrature, a scalar root
solve, and a 1-D phase solve, not calls to production constitutive functions.
Write C=T^coh; distinguish bulk b and surface Gamma coefficients even though
they coincide for this isothermal, single-background-material fixture.

### Phase equation and boundary/admissibility treatment

For AT1, alpha(phi)=phi, c0=8/3. With ell=.15625 m,
Gc=26.666666666666667 J/m^2, G=1e6 Pa, cohesion=1000 Pa,

\[
H_c=\frac{1000^2}{2G}=0.5\ {\rm Pa},\quad
E_c=\frac{G_c}{c_0\ell}=64\ {\rm Pa},\quad m=E_c/H_c=128,
\]
\[
g(\phi)=\frac{(1-\phi)^2}{(1-\phi)^2+m\phi(1+\phi)}.
\]

The physical phase problem at step k has weak residual

\[
\mathcal R_\phi(\phi_k;H_{k-1})[w]
=\int_{-W/2}^{W/2}
\left[(g'(\phi_k)H_{k-1}+E_c\alpha'(\phi_k))w
+2E_c\ell^2\phi_k' w'\right]dy=0.
\]

Thus in the unconstrained interior
g'(phi_k)H_{k-1}+Gc/(c0 ell)[alpha'(phi_k)-2 ell^2 phi_k'']=0.
Use natural zero normal derivative at y=+/-W/2, **not prescribed phi=0**.
Bulk x periodicity does not change those normal boundary conditions. A
half-domain reference may use symmetry phi'(0)=0; the full-domain reference
is the primary formulation and avoids adding a core-value Dirichlet condition.
The prescribed .6 core value initializes H; it does not pin evolved phi.

Physical bounds are [0,1]. Production uses an unconstrained Q1/CPDI phase
solve: there is no nodal phi_k>=phi_{k-1} active set and no nodal lower-bound
projection to reproduce in the reference. Irreversibility is through H below.
Do not add an obstacle problem to make the independent solve appear physical.
Newton trials use the nonsingular degradation branch containing [0,1]:
A=(1-phi)^2+m phi(1+phi)>0, phi<=1, and the connected branch on the negative
side. The phase residual evaluates raw phi and its analytic derivatives;
it does not replace phi by max(phi,0).

For localization and I_h use phi_eff=max(phi,0), h=1/g(phi_eff)-1, with the
existing empirical -1e-4 excessive-undershoot diagnostic. This is not an
accuracy parameter. Never clamp to the configured activation threshold (.1
in the validated K1 fixture). Do not clip
the upper end: g=0 is singular; phi>1 is invalid. The .99 fault-model
admissibility threshold is distinct from the physical range and is not a
license to operate near g=0. The proposed smoke should remain comfortably
inside the branch; record maxima and stop for review if phi reaches .8
(a proposed fixture-envelope guard, not a production model change).

### Independent initial histories, including truncation

Construct the prescribed stationary profile phi_star with core phi_c=.6 from

\[
\ell^2(\phi_\star')^2
=\alpha(\phi_\star)-\alpha(\phi_c)h(\phi_\star)/h(\phi_c),
\quad \phi_\star(0)=\phi_c,
\]

on its descending half, with compact zero continuation. Independent adaptive
quadrature/inversion can use phi=phi_c sin^2(t) to remove endpoint singularities.
Then reproduce **the actual initializer**, not an untruncated stationary ideal:

\[
H_0(y)=
\begin{cases}
E_c\alpha(\phi_c)/(h(\phi_c)g(\phi_\star(y))^2),&\phi_\star(y)>\phi_{\rm act},\\
H_c,&\phi_\star(y)\le \phi_{\rm act}.
\end{cases}
\]

Here phi_act is read from the resolved PhaseFieldFault parameter `Phase field
activation threshold`, **0.1 in validated K1**, not the generic PhaseFieldModel
default 0.01. The virtual PhaseFieldFault accessor returns its parsed member;
the manager obtains it through that accessor, not a hard-coded cutoff.
This corrects the earlier preparation's mistaken attribution of the generic
default to the benchmark. No K1/K3 initialization parameter has been changed.

This replaces the baseline H_c where the profile contributes; initialization
does not take max(H_c, prescribed H). Independently solve the phase weak
problem with this H_0 to obtain phi_0. Because of activation/truncation, do
not assume the analytic phi_star is already an exact solution of that problem.
The cutoff is applied to initialization only, not to I_h profile samples.

Retain supplied particle stress q_old,0=1500 Pa and Theta_0=200 s. Initialize
C_0 by the admitted-domain consistent projection of
g(phi_eff,0)*sqrt(2 G H_0). In the continuum 1-D limit with uniform admission
half-width a, this is its integral over [-a,a] divided by 2a. For a conditional
discrete check use the actual parent-P0 samples and full domain moments; keep
that separate from the independently initialized continuum reference. Initial
q sample variation and the difference between those two C_0 values are evidence,
not a reason to reset C_0 later. Store I_h,0 as the previous-integral snapshot.

### Mechanics, cohesive memory and normalization

For the actual accepted dt_k use

\[
\beta_{b,k}=e^{-dt_kG_b/\eta_b},\quad
\kappa_{b,k}=-\eta_b\operatorname{expm1}(-dt_kG_b/\eta_b),
\]

and the analogous surface coefficients beta_Gamma,k, kappa_Gamma,k. Keep
viscosity cutoffs on eta as in the validated fixture; do not floor kappa.

Define the full-profile I_k=int h_k dy with consistent normal bounds. Solve
at frozen Theta_{k-1}, q_{k-1}, C_{k-1}, I_{k-1}:

\[
q_k=\beta_b q_{k-1}+\frac{\kappa_b}{W}(U_k-V_k),\qquad
C_k=\frac{\kappa_\Gamma V_k+\beta_\Gamma I_{k-1}C_{k-1}}{I_k},
\]
\[
0=q_k-C_k-\sigma_*\mu(V_k,\Theta_{k-1})-\eta^d V_k,
\]
\[
\mu=a_f\operatorname{asinh}\!\left[
\frac{V}{2V_*}\exp\left(\frac{\mu_*+b_f\log(\Theta V_*/D_c)}{a_f}\right)\right].
\]

Use sigma_*=1000 Pa prescribed through the adiabatic-pressure friction mode,
mu_*=.6, a_f=.025, b_f=.013, D_c=.001 m, V_*=1e-5 m/s, eta^d=1e5 Pa s/m,
and V_min=1e-12 m/s. This is the K1 pressure convention, not K2.3's top-traction
boundary variant. Both top/bottom velocity components are prescribed, pressure
normalization is volume, and the homogeneous physical pressure is zero mean.
Friction's prescribed 1000 Pa is not a shift of that FE pressure.
The fixed-state scalar residual is strictly decreasing for this fixture;
require F(V_min)>0 and bracket an interior root, never clip a missing root.

The diffuse crack strain is

\[
\upsilon_k(y)=\frac{h_k(y)}{I_k}V_k
+\frac{\beta_\Gamma C_{k-1}}{\kappa_\Gamma}
\left[h_k(y)\frac{I_{k-1}}{I_k}-h_{k-1}(y)\right].
\]

Verify int upsilon_hist dy=0 and int upsilon dy=V using the same **full**
normal integration as I_k and I_{k-1}. Recover the velocity profile by
integrating u_x'=(q_k-beta_b q_{k-1})/kappa_b+upsilon_k from the prescribed
bottom velocity. V is not U. Bulk stress follows the nonrotational Maxwell
law tau_k=2 kappa_b(dev strain_rate_k-upsilon_k S)+beta_b tau_{k-1},
S=sym(e_x tensor e_y). No elasticity plugin or stress-energy replacement is used.

Finite production support is a distinct approximation: only admitted particle
domains contribute to surface work, localized bulk terms act on their supported
QP associations, and only associated particles receive the cohesive H update.
For the production-support reference apply the H maximum on that supported
set and retain H outside it. Also report the full-profile ideal and the omitted
localization/history contributions separately. The full-integral scalar
mechanical reduction is conditional on the measured normalization and
homogeneous-stress assumptions; do not declare exact equivalence if these fail.
Do not shrink I_h to force the supported integral identity to hold.

### Finite-step H and state update

With the accepted projected C_k (constant in the ideal homogeneous surface),

\[
A_k=C_k/g_k,\quad B_k=\beta_\Gamma h_{k-1}C_{k-1}/(1-g_k),\quad
\mathcal H_k=\frac{dt_k}{2\kappa_\Gamma}(A_k-B_k)(A_k+B_k),\qquad
H_k=\max(H_{k-1},\mathcal H_k).
\]

Use this factorized expression for g_k<1. Do not add the candidate to old H
or independently clip a negative candidate. At g_k=1 with h_k=h_{k-1}=0 use
dt_k C_k^2/(2 kappa_Gamma); g_k=1 with h_{k-1}>0 is inadmissible healing.
Independently check the equivalent cohesive-work expression
dt_k[.5 kappa_Gamma upsilon_k^2+beta_Gamma h_{k-1}C_{k-1}upsilon_k]/(1-g_k)^2
where defined. Do not use dt/kappa approximately equal to 1/G.

After mechanics, not inside the friction residual, update

\[
\Theta_k=\Theta_{k-1}e^{-V_kdt_k/D_c}
+\frac{D_c}{V_k}[-\operatorname{expm1}(-V_kdt_k/D_c)].
\]

## 2. Exact indexed cycle and source cross-check

1. Initialize particle H_0 from the rule above, supplied stress, and Theta_0;
   solve phi_0, reconstruct the line once, project initial C_0 and store I_0.
2. Solve timestep-zero mechanics with the artificial 2-s Maxwell interval,
   using phi_0 as both current/previous profile. Commit V_0 and the accepted
   bulk solution, but retain supplied stress, Theta_0, H_0 and initialized C_0;
   keep I_0 as previous snapshot. Do not count this interval as physical slip.
3. For real step k, preserve committed particle histories during the phase
   solve: H_{k-1} -> phi_k. Preserve the actual old FE phi_{k-1} separately.
   Reconstructed geometry/topology remains its initial snapshot.
4. Prepare current I_k and solve mechanics with old committed histories and
   frozen current phi_k. Residual/Jacobian/line-search evaluations are noncommitting.
5. Form accepted cohesive samples, project C_k consistently, form bulk stress,
   Theta_k and the H maximum candidates, and validate before terminal publication.
   Commit histories, I_k, V_k and bulk state through the existing lifecycle.
6. The next step consumes that H_k. In particular phi_1 uses retained H_0 and
   need not differ from phi_0; **phi_2 must respond to the measured H_1-H_0**.

Confirmed against `current_design.md` sections 13, 17, 19--22 and 25;
`specification.tex` common cohesive state, physical ranges, domain quadrature
revision and Stage-J history feedback; and K3-a--d in the benchmark instructions.
Source checks: `source/simulator/solver_schemes.cc` solves phase before
mechanics and reconstructs only at step zero; `source/simulator/phase_field.cc`
supplies the weak CPDI equation, degradation branch and iteration-exhaustion
failure; `source/simulator/core.cc` supplies periodic/hanging-node constraints
without a phase Dirichlet boundary or irreversibility obstacle;
`source/reconstructed_fault/manager.cc` applies the material-model activation
accessor (configured .1 for this fixture) to the initial prescribed profile;
`source/particle/property/crack_driving_force.cc` supplies H_c;
`source/material_model/phase_field_fault.cc` supplies the cohesive projection,
factorized maximum update and retained timestep-zero histories. No conflict
requiring a production/model change was found.

The independent 1-D solver will use its own accurate normal quadrature and
mesh, not claim identical discretization to CPDI. To distinguish discretization
from wrong history input, separately evaluate production's frozen CPDI weak
residual using its actual particle H/weights. Parent-P0 constitutive sampling
and domain-integrated surface test/trial weights remain the production rule.

## 3. Proposed smoke parameters and expected signal

Start from the validated homogeneous K1-style box: 32x128 bulk cells,
h=.0078125 m, ell/h=20, 17 open surface nodes with h_Gamma=.015625 m,
3x3 reference-cell particles, continuous Q2 stress ADD/count transfer,
eta=1e8 Pa s, G=1e6 Pa, T=293 K, and no chemical fields. Keep the prescribed
fault (0,0)--(.25,0), core=.6, physical box and K1 pressure/loading symmetries.

Proposed explicit differences from the frozen K1 fixture:

- `Evolve phase field = true` enables the production H update; its meaning is
  not changed. Remove the benchmark-only phi freezing and H-frozen assertion
  through a clearly selected evolving diagnostic mode, not a production bypass.
- Use boundary u_x(y,t)=y U(t)/W, u_y=0, with
  U(t)=1e-4+(2.25e-3-1e-4)*min(t/2,1) m/s. This is the selected bounded ramp, not an
  automatic forcing search. Retain the original initial loading U(0).
- Maximum timestep 2 s and existing convection plus reconstructed-fault
  timestep selection, CFL=.5. The rate-strengthening a_f>b_f branch imposes
  no RSF splitting restriction. The second step may be shortened by convection.
  Use existing `Termination criteria = end step`, `End step = 2`, as in the
  Stage-J two-step fixture, so no unwanted extra real steps occur. Expected
  interval is 0 to at most 4 s; record the actual accepted times.

Do not change the existing phase (1e-8 nonlinear, 2e-7 linear, 50 nonlinear
iterations) or coupled solver settings. First verify the selected first dt
is actually 2 s and the reference predicts observable feedback at that dt;
do not disable timestep safeguards to obtain the desired signal.

A superseded **ideal, untruncated-profile scalar estimate**, not a K3 reference
solution or production run, gives I_h=108.6525644 m and, for first dt=2 s,
U=.009 m/s, V_1=.0086750603 m/s, C_1=468.0642 Pa, q_1=2113.7217 Pa,
Theta_1=.115279 s. It predicts H_candidate/H_0=1.24239 in the ideal profile
interior (core 29568 -> 36735 Pa). This predicts a clear H_1 input change for
phi_2. The sampled C/I_h initialization and activation cutoff must be included
in the complete independent reference before trusting these values.

The selected ramp is 22.5 times the initial velocity, with first-step
macroscopic shear increment U_1 dt/W=.0045. The independent reference predicts
max|H1-H0|=3.93555 Pa in the transverse tail, while the core H maximum remains
unchanged. Max|phi2-phi1|=.000953568 and I_h increases by .151855 m (.140431%).
These signals exceed reference noise by 628/1483/3480 times, respectively.
Maximum phi2=.599784 passes the .8 envelope. This supersedes the ideal .009-m/s
estimate above without discarding its documented failed support diagnostic.

## 4. Reference accuracy, containment and diagnostics before execution

The implemented independent 1-D solver uses the weak residual above with
consistent Jacobian and natural boundaries, its own split normal mesh/Gauss
quadrature and scalar bracketing. Independently tighten
its normal grid/quadrature until changes are below 1e-6 absolute in phi and
1e-5 relative in H/I_h/V/C where nonzero. These are proposed reference error
targets, not changes to production tolerances or a Gate-K3 pass claim. Use the
actual accepted step sequence; initialize once and never reset from later
production output. A conditional reference initialized once from saved FE/H
data may accompany, but not replace, the fully independent initialization.

The K2 omitted-profile allowance is **not inherited**. Initial K1 evidence
already has roughly 6e-5 omission, so the original 1e-6 target cannot be claimed
even before evolution. For this bounded K3 smoke only, propose review of:

- Omitted fraction <=1e-4 for **each current and previous** full-profile h
  integral, independently remeasured at initialization and every accepted time.
- Actual |integral upsilon - V|/max(|V|,V_*) <=1e-4 is the **primary
  normalization check**, with signed history and instantaneous contributions
  reported separately. K3's rejected ramps demonstrate that a small omitted
  h fraction does not certify support adequacy: truncating the history
  correction can dominate. Retain both diagnostics without changing full I_h.
- Separate tail/support error estimates in q, C and H from the independent
  scalar/profile calculation; do not absorb them into the reference solver
  tolerance. Report numerical, initialization and support errors separately.

This budget requires explicit approval. Check predicted evolving-profile
containment with the independent solve before production execution. A failed
budget means stop; no widening support, renormalization, or fitted allowance.
The observed change in H, phi and I_h must each exceed its corresponding
reference/discretization noise estimate by at least an order of magnitude for
the smoke to demonstrate feedback. Do not use monotonicity alone as evidence.

Required exports/checks:

1. At phase assembly entry: timestep/physical time, stable particle IDs/current
   positions, consumed H, old FE phi and the noncommitting production residual.
   Match consumed H to the preceding commit, not a post-solve visualization.
2. Reuse production assembly to evaluate R(phi1;H0), R(phi1;H1) and R(phi2;H1)
   noncommittingly. Evaluate the frozen pre-phase residual on phi_1 with both H_0 and H_1:
   R(phi_1;H_1)-R(phi_1;H_0)=integral g'(phi_1)(H_1-H_0)w. Freeze the same
   domains/constraints during this diagnostic; no physical or history mutation.
   Show the subsequent phi_2 solve reduces R(phi_2;H_1) genuinely. Verify H1
   by stable ID against the preceding commit, including migrated particles.
   Full CPDI weight/gradient export is required only if this check fails.
3. Independently solve the counterfactual phase problem retaining H_0 in the
   1-D reference. It must lack the measured H_1-driven phi_2 response. This is
   reference-only, not a second production run or disabled-H workaround.
4. Export raw phi/H profiles, current/previous I_h, C, Theta, V, q, slip,
   candidate/max-active H mask, geometric snapshot, normal stress and actual
   weak balances. Record raw undershoots, min g, maximum phi, full/admitted
   normalization, and along-fault variation including endpoints.
5. Require fresh-linear and final separate bulk/surface convergence, real
   phase convergence (not budget exhaustion), t0 history retention, and fixed
   reconstructed geometry. Keep bulk old-history output and accepted particle
   stress output labeled distinctly. Reuse established rollback semantics.

## 5. Implementation and resource boundary

`uniform_shear.cc` now has a default-false `Postprocess/Uniform shear pilot/
Evolving profile` option. The opt-in mode omits benchmark phi constraints and
H-frozen assertions, reports imposed boundary loading, and records stable-ID
particle H at timestep entry. It saves old surface C/I_h before mechanics so
post-commit localization diagnostics do not mistakenly consume new history.
It does not refresh live fields or update Maxwell stress for visualization.
Default fixed-profile behavior is retained. The parameter overlay is
`benchmarks/reconstructed_fault/uniform_shear/evolving/smoke.prm`.

The noncommitting production phase-entry residual comparison and stable-ID
history handoff are now verified in the successful smoke. Both normal return
and a forced exception restore the live H/state/solver data. Full CPDI weight
export was not needed. This completes bounded smoke verification, not a
convergence campaign or a fully resolved numerical reference.

The selected reference predicts zero periodic-x crossing events on step 1 and
358 on step 2 for the 96x384 particle lattice. RK2 advects before mechanics;
use old and extrapolated predictor velocities, not the new accepted velocity.
These are estimates, not measured CPDI periodic-domain behavior. Any eventual
smoke must measure H, phi, I_h, C and V along the fault including both open
endpoints and the periodic seam. A seam artifact comparable to the intended
H/phi/I_h feedback prevents interpretation as a 1-D K3 result; wrapping alone
does not constitute failure.

Expected one-rank Release smoke cost: about 60--120 s and 0.7--1.2 GiB,
estimated from the accepted 32x128 fixed-profile 37--39-s cases with allowance
for two additional phase/preparation passes. Evolving domains/conditioning
remain unmeasured. The approved hard smoke cap is 180 s, no automatic retry, with review if
it times out. Keep independent exploratory calculations under 120 s each
and 600 s aggregate; no automatic mesh/timestep campaign. A plugin build,
if approved, uses -j4. A longer convergence sequence belongs to a later review.

The bounded adjustment tests exactly two lower peaks; independent and conditional
accuracy checks pass for the selected .00225-m/s case. Six cheap tests pass.
All reference execution remains well below 120 s per command/600 s aggregate.
Neither support nor the budget was changed to obtain a pass. See the
bounded-adjustment record for reference timings and initialization distinctions,
and `stage_K3_smoke.md` for the failed invocation and remaining verification.
