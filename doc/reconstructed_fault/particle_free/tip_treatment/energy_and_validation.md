# Energy ledger and evidence

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

See the [current integration report](../TIP_SOLVER_INTEGRATION_REPORT.md) for
the corrected solve and equilibrated fixture, and the
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
+E_f(\phi)+\beta\Delta t\sum_s ds_s\Phi_s(V_s)-\beta W_{ext}.
\]
Add the incompressibility multiplier with the same β scaling. Its displacement
variation is β times mechanics, its e variation is −t+q+R(V), and its phase
variation is the fracture derivative minus Y (with the constrained one-sided
intact limit). This is an incremental **stationarity functional**, not a new
stored energy or a replacement for the physical work ledger. In particular,
Φ(V) is not f(V)V, and differentiating the latter would give the wrong law.

The admitted monotone section law makes Φ convex in V. It does not make the
whole φ problem convex or establish a unique stable growing solution. The old
bulk stability checker has not been extended to eliminate the section variables
and assess this functional's constrained second variation. The current solver
therefore reports residual/KKT acceptance, not a stability certificate. This
functional provides a concrete diagnostic for a remaining refined-mesh
admission/globalization failure without changing loading or physical parameters.
