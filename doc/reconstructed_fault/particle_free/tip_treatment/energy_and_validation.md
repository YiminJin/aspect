# Energy ledger and evidence

This page describes the retained **BE default**. The selectable midpoint stage,
exact aging, endpoint publication and work identity are in
[time integration](time_integration.md); do not apply BE loss formulas to that mode.

[Guide](../TIP_TREATMENT.md)

The instantaneous bulk identity is
\[
\dot\psi=\tau:D_3-\tau:\tau/(2\eta)
-hQ:Q/(2\eta)-(\tau-Q):D_f-Y\dot\phi.
\]
Section integration converts the last mechanical loss into signed RSF plus
radiation power. Bulk and cohesive viscosity remain separate. There is no
surface cohesive-energy surrogate or second fracture inventory.

For the endpoint step, the code records external work, bulk/cohesive storage,
diffuse fracture, both viscosities, section friction and radiation, and the
following **numerical** terms:

- Stress BE loss: \(|\tau^+-\tau^-|^2/(4G)\).
- Cohesive conjugate-stress BE loss: \(h^-|Q^+-Q^-|^2/(4G)\).
- Compliance endpoint work minus \(Y^+\Delta\phi\).
- Fracture quadratic increment: \(\tfrac12\Delta\phi^TH_f\Delta\phi\).
- Phase residual work and mechanical residual work.

Stable limit expressions handle zero/infinite compliance. Compliance work is
\(E_c^+(1-r^+/r^-)\) when the ratio is defined, with its intact value \(E_c^+\)
and fully damaged endpoint value zero. It is not generally YΔφ. The ledger
checks the independently evaluated stress work against storage, physical loss
and all these terms before publication. Signed work remainders and accumulated
absolute remainders must both be compared with added fracture and active work.
When added fracture is tiny, use absolute errors rather than a misleading ratio.

The fixed reference has no advective boundary flux or transfer operation.
Quadrature error is not declared zero merely because the algebraic energy ledger
closes; it requires an integration comparison. There is no history projection in
the new point representation, but its fixed-quadrature restriction must be kept.
No numerical remainder is charged to Gc.

The old interface four-step diagnostic is retained separately in
`evidence/bulk_tip/interface_decomposition.json`. Its physical remainder uses
trapezoidal work and totals 9.65253715005e-6 J/m: bulk BE 8.40356655008e-6,
surface endpoint 1.24893547552e-6, trapezoid-minus-endpoint 3.51255261278e-11,
and approximately −1.07e-15 algebraic ledger error. Recorded quadrature
repartition and transfer-stress defects are zero. The remainder is 21.72% of
4.44353502785e-5 J/m added fracture. All four timesteps are the same; these data
establish decomposition, not timestep scaling or convergence of the new model.

The [native Bernstein report](../NATIVE_DG_BERNSTEIN_REPORT.md) gives current
representation-equivalence checks. The [integration report](../TIP_SOLVER_INTEGRATION_REPORT.md)
records the corrected solve and equilibrated fixture, and the
[original implementation report](../TIP_TREATMENT_IMPLEMENTATION_REPORT.md) for
retained add-only results. Their physical windows and initialization differ.

## Restricted incremental stationarity functional

For this constant-β, prescribed-compression, fixed-direction closure, define
\(R_s(V)\) using the endpoint age and
\(\Phi_s(V)=\int_0^V R_s(v)\,dv\). At independent current cohesive strain e,
\(V_s=\sum dn(e/\beta-e^-)/\Delta t\), and
\(T_*=\beta(\tau^-+2GA^-+2G\Delta\epsilon)\). A restricted incremental
stationarity functional is
\[
\mathcal I=\sum_q JxW_q\left[
\frac{(T_*-2GA):(T_*-2GA)}{4G}+\frac{GA:A}{h}\right]
+E_f(\phi)+\beta\Delta t\sum_s ds_s\Phi_s(V_s).
\]
This is exactly the stored `Evaluation::functional`: bulk plus cohesive plus
fracture energy plus the section primitive. The supported boundary is wholly
prescribed velocity/displacement. There is no applied-traction potential to
subtract; computed reaction work `external_work` is an accounting diagnostic,
not an extra term in the search functional. Add the incompressibility multiplier
with the same β scaling when checking unconstrained variations. Its displacement
variation is β times mechanics, its e variation is −t+q+R(V), and its phase
variation is the fracture derivative minus Y (with the constrained one-sided
intact limit). This is an incremental **stationarity functional**, not a new
stored energy or a replacement for the physical work ledger. In particular,
Φ(V) is not f(V)V, and differentiating the latter would give the wrong law.

The admitted monotone section law makes Φ convex in V. It does not make the
whole φ problem convex or establish a unique stable growing solution. The old
bulk stability checker has not been extended to eliminate the section variables
and assess this functional's constrained second variation. The current solver
therefore reports residual/KKT acceptance, not a stability certificate. Earlier
refined-step failures and their corrected replay remain historical evidence;
this statement does not assert that the current difficult-step control fails.

## Current common-parent and localized evidence

The [efficiency/growth report](../TIP_EFFICIENCY_AND_GROWTH_REPORT.md) compares
one .0025 step with two normal .00125 steps from the identical .0575 parent.
Physical remainder halves (.00174784→.000874876 J/m), while added fracture
changes only .761%. Both signed and accumulated absolute contributions are
reported; the independent ledger closes without removing numerical losses from
the physical remainder. This is a local sensitivity test, not global convergence.

For spatial fracture accounting, point volume density omits SIPG faces. The
benchmark instead exports the actual Bernstein gradient g_f and uses
.5[g_f(B)+g_f(B_seed)]·(B−B_seed). Its sum is the exact quadratic fracture change
(up to roundoff); assigning rows to their cells defines a weak-work partition,
not a unique point density. The final localized partition closes to 4.17e-14 J/m.
At .09 s the front VI is converged, but the fixed phase/H extents have not
advanced and the observed remote driving margin has reached its diagnostic stop.
Residual convergence alone does not supply physical propagation evidence.


<a id="acceptance"></a>
## Actual convergence and publication gates

[`BulkTipCoupling::evaluate`](../../../../source/simulator/solver/bulk_tip_assembly.cc)
keeps the raw assembled residual and the residual with homogeneous constrained
rows zeroed. Its mechanical norm is the Euclidean norm of the displacement and
pressure residual blocks. With the positive phase weights
\(m_i=\int\mathcal B_i\,dx\), the KKT diagnostic is the natural map

\[
\kappa=\max_i\left|B_i-\operatorname{clamp}
 (B_i-R_{B,i}/m_i,\ B_i^-,\ 1)\right|.
\]

The fixture gates are mechanics ≤1e-9 and κ≤1e-7. The outer solver uses their
maximum after division by these respective tolerances. This is neither a
relative reaction-force tolerance nor a guarantee of accuracy at tiny rates.
Every section root must separately satisfy `section_endpoint`'s roundoff-scaled
scalar equation; the difference between its resolved rate and the subsequently
integrated point rate is also reported.

Let ΔE include bulk, cohesive and fracture storage; D_b, D_c and D_f denote
step-integrated bulk viscosity, cohesive viscosity and section friction plus
radiation. Define C_rem as compliance work minus YΔphi, N_tau and N_Q as the two
BE losses, N_f as the fracture quadratic increment, and W_phi as phase residual
work. The code's independent ledger is

\[
\mathcal L=W_{int}-\Delta E-D_b-D_c-D_f-N_\tau-N_Q-C_{rem}-N_f+W_\phi.
\]

The [simulator acceptance block](../../../../source/simulator/solver_schemes.cc)
requires |L|≤1e-10 max(1, |W_ext|, E_bulk+E_cohesive+E_fracture), and nonnegative
D_b, D_c, friction and radiation integrals. W_int is integrated stress work;
W_ext is reaction work computed from prescribed displacement and raw mechanical
rows. The separately reported `mechanical_work` is W_int−W_ext.
The phase work and numerical losses remain visible even when the ledger closes.
The discrete section adjoint defect is `Df − sum_s dt ds f V_integrated`;
checker power tests assess it independently.

Only then does the material prepare a candidate for the simulator's single
publication callback. A small ledger residual proves this discrete accounting
identity, not small physical work error, quadrature convergence or propagation.

## Native-basis evidence versus physical accuracy

The native Bernstein difficult-step comparison preserves the accepted parent,
physical tau/A, Y, section state and diagnosed active set to near roundoff.
The localized control shows actual phase growth and agrees with the nodal result;
MPI rejection/retry preserves the parent. These tests qualify representation and
integration. The short moving-profile and layered mature checks in the same
report concern the shared phase implementation and a different constitutive
route; they do not enable moving tip growth.

The earlier .09 remote-margin stop and large BE physical remainder remain the
scientific boundary. No endpoint integrator or loading change has been made by
the native-basis implementation.
