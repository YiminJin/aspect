# Remaining Stage B protocol — transported/deforming profile and layered eta

Recorded before implementation/runs, 2 November 2026. No native integration,
RSF/growth, AMR/restart qualification or server work is selected. Complete the
independent history-space comparison, then select a representation for Stage C.
One Python process, BLAS threads1, sequential bounded runs; compact outputs.

## Prescribed kinematics and independent histories

Square [-.4,.4]^2m, initial infinite through-going fault at37°, ell0=.06m,
G=3MPa, duration.8s. Affine divergence-free material velocity
v=(cx+k y,cy), k=.4/s, cy=.12m/s, cx=+.24m/s (main) or−.24m/s (reverse check).
F=[[1,k t],[0,1]], centre d=(cx t+.5 k cy t²,cy t).
Curve X(alpha,t)=d+F alpha t0, tangent t=F t0/g, g=|F t0|, n=rot90(t).
For z=(x−d)·n, alpha=(x−d)·t/g, material normal z0=g z,
chi=g exp(−(z0/ell0)^2)/(ell0 sqrt(pi)). Thus material profile is carried,
physical width ell0/g changes, and D_v chi=(t·L t)chi. No Eulerian max or
constant-width rebuilding. Normal/tangent follow geometric motion, not W.

Set s(alpha,t)=.025(1−exp(−.35t))/.35 [1+.4sin(6alpha)]m,
V=D_Gamma s+r_Gamma s, so s=0 initially and slip forcing is nonzero.
The surface-label derivative and bulk derivative of s_ext differ:
D_v alpha=z(t·L n+n·L t)/g. Keep this extension-gradient term.
Use physical local rate r_b=.5+.3tanh(y/.08m) /s, eta=G/r_b.
Auxiliary r_Gamma is the global material-P1 L2 projection of the surface trace
r_b(X(alpha,t)) on alpha∈[-1.2,1.2]m, integrated with Gauss96.
Its measure g d_alpha is common and cancels in this projection. Check positivity
and domain coverage, no clipping/floor. This deliberately declared **rate**
projection is a benchmark auxiliary owner, not an assertion about existing
native material-mixture averaging. Physical direct stress always uses local eta.

Evolve tau, T and R independently. For compact full3D plane strain,

    D_v tau=[W,tau]−r_b tau+2G(D−chi V S),
    D_v T=[W,T]−r_b T+2G D,
    D_v R=[W,R]−r_b R+Q_geometry+Q_extension+Q_rate,
    Q_geometry=−2G chi s[(t·L t)S+S_dot−[W,S]],
    Q_extension=−2G chi S (D_v alpha) s_alpha,
    Q_rate=−2G chi s S(r_b−r_Gamma).

Initial physical xx/G=.06+.02cos(4X)+.01sin(5Y),
yy/G=.02+.015sin(3X)cos(4Y), xy/G=.025+.01cos(4(X+Y));
zz=−xx−yy; T=tau initially, R=0.

The direct reference integrates its physical component equation along exact
characteristics. Since y=Y+cy t, accumulated r_b has an independent analytic
log(cosh) antiderivative. Constant W gives a complex propagator; source
integrals use converged temporal Gauss rules. R has a separate source integral;
never infer it from tau. Verify reference quadrature32/48/64 and projected
surface coefficients96/128. Also verify direct real-component RK4 on a small
characteristic set. The scalar memory is manufactured, not numerically evolved
RSF age or a claimed native surface-transfer implementation.

## Eulerian history-space test and criteria

Implement Cartesian modal DGQ2 material transport with volume/face upwind
advection, using the same weak-form structure as the existing PF4 operator;
include exact prescribed inflow from the independent reference. Variable-r_b
reaction is projected as a multiplication operator. Integrate this reference
fixture with RK4 to suppress time error in the spatial study; this does not
change the selected native lagged Lie integrator. Independently test the latter
at exact moving-characteristic samples with80/160/320 steps.

1. Operator: constant and linear advected fields match independent −v·grad f
   to scaled1e−11. Mass defect<1e−12. Div(v)=0 exactly.
2. Reference: quadrature and real-component RK4 differences<1e−10G, independent
   continuous direct/separated closure<1e−10G. No stage calls reconstruction twice.
3. Spatial: N8/16/32 cells per direction, source/reaction quadrature8, independent
   error quadrature12; RK4 dt chosen from CFL .25/[(2p+1)sum(max|v_i|)/h].
   Last two full3D L2 orders>2.0; finest tensor/normal/shear errors<1e−3G
   (scaled volume RMS), nonzero-reference stored-energy error<1e−3.
   Report sampled maxima, zz, physical projection errors, material width and
   distances in cells/ell. No pointwise bound inferred from weak traction error.
4. Isolate time: at N16 repeat with doubled RK4 steps; its solution change must
   be<10% of its spatial error. Separate characteristic finite-rotation Lie
  80/160/320 errors require ratios1.8–2.2 and finest<.01G.
5. Isolate quadrature: N8 q4/6/8 versus q12 at the **same step count**; errors
   decrease and q8 difference<1e−8G. Retain any failed candidate and original
   thresholds. Do not tune physical parameters or inflate the width.
6. Reversed horizontal motion: N16 versus independent reference must meet .01G
   tensor RMS; no full second refinement matrix unless a discrepancy requires it.
7. Negative controls: independently omit geometry/profile, extension-gradient,
   or coefficient-mismatch source. Each must leave a continuous tensor RMS
   defect>1e−5G. A wrong homogeneous eta update must differ from the changing-
   coefficient relaxation reference by>1e−4G. Keep full model unchanged.
8. Varying-eta relaxation: selected material paths crossing the layer, zero
   forcing, independent analytic attenuation and viscous-loss quadrature; energy
   closure<1e−10 times nonzero initial storage. This is a constitutive-path
   check, not coupled boundary/friction/fracture energy qualification.
9. Profile: actual normalized Gaussian, measured by normal quadrature at three
   times, integral error<1e−10, width ell0/g and no untransported damage trail.
   Surface frame/association derivatives must match independent coordinate
   finite differences to2e−9. All sampled auxiliary rates positive and labels
   within the projection domain. Full3D stress and actual deformation reported.

Budgets: main spatial three meshes, one half-step check, three quadrature
candidates plus reference, one reverse case, three characteristic Lie step
counts and reference/negative-control audits. No whole native test suite/build,
particles or diagnostic frame sequences. Select direct tensor versus complete
separated history using measured errors, coefficient/storage costs and known
native-source limits. Stage C residual/tangent/acceptance/restart integration
remains a separate selected task.

## Smallest-space screen after the declared comparison

The full Q2 separated representation meets the1e−3 target at N8, while direct
Q2 first meets it at N16. Before choosing Stage C storage, screen a P1 residual
against the independent exact R: compare best-projection P1 versus Q2 residual
error on N8/16/32, and reconstruct using independently projected exact T and R.
Use the same1e−3 tensor/traction/storage target and count T-Q2+R-Q1 versus
T-Q2+R-Q2 versus direct-Q2 coefficients. This is a **best-space capacity screen**,
not an implemented/evolved P1 transport scheme or a P1 convergence claim. If
P1 cannot meet the target on the least-cost Q2 mesh, it does not justify a new
native history-space abstraction. Keep its measured limitation visible.

## Surface-memory representability screen

The transported comparison prescribes analytic s, so its analytic B accuracy
must not be counted as free native information. Before freezing the decision,
measure Q1 interpolation of s on16/32/64 material intervals over[-1.2,1.2]m.
Report scalar error and the induced full3D B-extension RMS/G on the independent
volume rule. Require second-order convergence orders1.8–2.2 and finest extension error<1e−3,
then count the65 scalar coefficients separately. This is a snapshot capacity
check; native scalar evolution, actual projected source/derivative and ALE
transfer still need Stage C qualification. Do not infer their accuracy from it.

Protocol wording correction: the surface screen originally said “second-order
ratios1.8–2.2”. For h halved, a second-order error ratio is4, while log2 of that
ratio is2. The checker from its first run uses the correct convergence orders;
the original text is preserved as `surface-protocol-original.txt`. This fixes a
wording error, not a changed physical parameter, relaxed error bound or rescued
failed numerical run.
