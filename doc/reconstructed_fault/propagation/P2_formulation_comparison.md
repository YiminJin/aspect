# P2 formulation comparison for review (2026-10-08)

The selected implementation uses a **peak recoverable fault-energy driver** and
direct phase irreversibility. The proposed total-energy phase condition also
varies bulk storage when the diffuse slip profile changes. That additional
variation is derived and checked on frozen controls; **it is not implemented**.
No equation, constitutive option or coupled trajectory changes in this checkpoint.

Notation follows [pf_rsf.tex](../pf_rsf.tex). That manuscript contains the historical
incremental-work derivation; the reviewed production revisions are recorded in
[specification.tex](../specification.tex) and [current_design.md](../current_design.md).
In particular, manuscript \(\bm u\) denotes **velocity**, and
\(\bm\varepsilon=\operatorname{dev}(\operatorname{sym}\nabla\bm u)\) is a strain
**rate**. \(\lambda=\eta/G\) is a relaxation **time**. Source variables `q`, `I_h`
and `Z` correspond to \(\tau^{\rm coh}\), \(I^h\), and \(G s^e=I^h\tau^{\rm coh}\).

## Common definitions and state ownership

\[
 g(\phi)=\frac{(1-\phi)^2}{(1-\phi)^2+m\phi(1+p\phi)},\qquad
 h=g^{-1}-1,\quad I^h=\int_{\mathcal T_{\bm x_{\mathcal F}}}h\,d\zeta,
 \quad\chi=h/I^h,\quad
 \bm{\mathcal S}=\operatorname{sym}(\bm e^n\otimes\bm e^s),
 \quad\bm{\mathcal S}:\bm{\mathcal S}=1/2.
\]
\[
 \gamma=\frac{\phi/\ell+\ell|\nabla\phi|^2}{c_0},\qquad c_0=8/3.
\]

Geometry, mesh, particles and material parameters are fixed in this fixture.
Independent mechanical unknowns are \(\{\bm u,p,V\}\); the diffuse slip strain rate
\(\upsilon\bm{\mathcal S}\) is reconstructed from surface slip rate \(V\) and
profile data. Phase \(\phi\) is solved separately. Bulk stress, surface elastic
slip \(s^e\), particle \(\mathcal H^+\), and previous normalization are committed
histories, not additional simultaneous Newton unknowns.

| Question | Selected implementation | Proposed total-energy interpretation |
|---|---|---|
| Phase trial | Vary supplied scratch \(\phi\); immutable accepted lower bound; particle \(\mathcal H^+\) and CPDI domains fixed | Vary \(\phi\) at fixed total displacement \(\bm d\), elastic slip \(s^e\), and committed inelastic/internal strains \(\bm E^{\rm in}\) |
| Other quantities fixed | Incoming mechanical/history data, geometry, material and loads; no mechanical solve inside phase trials | Same geometry/material/loads; stress and cohesive traction **respond** to trial \(\chi,I^h\). Pressure remains an incompressibility multiplier |
| Phase potential | CPDI-discretized \(\int_{\mathcal B}\mathcal H^+g(\phi)+\mathcal G_c\gamma\,dA\), up to constants | \(\Psi_b+\Psi_c+\Psi_f\) below, using consistent spatial measures and normalization adjoint |
| Free phase equation | \(\delta_\phi[\int\mathcal H^+g\,dA+\Psi_f]=0\) | \(\delta_\phi[\Psi_b+\Psi_c+\Psi_f]=0\); the additional bulk-profile term is explicit below |
| Admissible set | Initial relaxation \(0\le\phi\le1\); physical steps \(\phi_{\rm accepted}\le\phi\le1\) | Same direct constraint can be retained; it enforces irreversibility without a peak substitution in the energy derivative |
| Peak \(\mathcal H^+\) | Particle max of seeded history and terminal accepted recoverable candidate; fixed during phase solve | Proposed role: peak diagnostic/seed provenance, **not an extra energy or viscous-dissipation driver**. Changing its phase role requires approval |
| Verification level | Bounds, KKT, publication and fixed-phase histories verified for named controls; coupled temporal agreement improves but is incomplete | Frozen energy first variations verified; internal-variable ownership, discrete adjoints and coupled energy/time formulation remain proposed |

\(\bm d\) and \(\bm E^{\rm in}\) are explicitly introduced accumulated quantities,
not reinterpretations of the manuscript's velocity or strain rate. Production
stores bulk stress, rather than this complete proposed set of accumulated internal
variables. Their consistent retention/reconstruction must be specified before
implementing the proposal.
The implemented phase solver does not recompute \(Y\), cohesive traction or
normalization during phase trials. Normalization is refreshed for downstream
mechanics after publishing accepted phase; its previous value remains history.

## Equations beside their proposed counterparts

**Implemented phase/history equations.** With the incoming particle history held
fixed, the free phase equation is the native CPDI weak form of

\[
 g'(\phi)\mathcal H^+
 +\frac{\mathcal G_c}{c_0}\left(\ell^{-1}-2\ell\Delta\phi\right)=0,
 \qquad\mathcal H^+_{k+1}=\max\left(\mathcal H^+_k,
 Y_{k+1}\right),\quad
 Y=\frac{(\tau^{\rm coh})^2}{2Gg(\phi)^2}.
\]

The max is published only after accepted mechanics, using mutually consistent
accepted phase and traction. Native residual is minus the potential gradient:
at a lower bound it is nonpositive, at an upper bound nonnegative, and zero on
free DoFs. A previous-phase lower bound is enforced by projected Newton, not by
clipping an equality solution. The sequence remains
\(\mathcal H^+_k\to\phi_{k+1}\to\text{mechanics}_{k+1}\to\mathcal H^+_{k+1}\).

**Proposed total-energy phase equation.** At fixed \(\bm d,s^e,\bm E^{\rm in}\), set
\(\bm E=\operatorname{dev}(\operatorname{sym}\nabla\bm d)\) and

\[
 \bm\tau=2G[\bm E-\bm E^{\rm in}-s^e\chi\bm{\mathcal S}],\qquad
 \Psi_b=\int_{\mathcal B}\frac{\bm\tau:\bm\tau}{4G}\,dA,
 \quad\Psi_c=\int_{\mathcal F}\frac{G(s^e)^2}{2I^h}\,dL,
 \quad\Psi_f=\int_{\mathcal B}\mathcal G_c\gamma\,dA.
\]
\[
 \delta_\phi\Psi_f
 -\int_{\mathcal B}s^e(\bm\tau:\bm{\mathcal S})\delta\chi\,dA
 -\int_{\mathcal F}\frac{(\tau^{\rm coh})^2}{2G}\delta I^h\,dL
 \ \ge 0\quad\text{for every feasible phase direction},
\]
\[
 \delta\chi=\frac{h'\delta\phi}{I^h}
              -\frac{h\,\delta I^h}{(I^h)^2}.
\]

Equality holds on free DoFs. With direct normalization,
\(\delta I^h=\int_{\mathcal T}h'\delta\phi\,d\zeta\), the fault term alone yields
\(Yg'\delta\phi\) with the implemented expression for \(Y\). It does **not** supply
the bulk term. The latter cancels only for special uniformly weighted transverse
stress fields with exact normalization. For the actual consistent surface
representation, use \(\delta I_h^h=\mathsf P(\delta I^h)\) and its adjoint in
both terms; substituting pointwise \(Y\) is not the exact discrete derivative.
The CPDI phase measure versus pointwise mechanical/normalization measures remains
a separate approximation.

The continuum first variations and their frozen discrete energy counterparts
have centered-difference checks. A nonuniform frozen stress control gives
\(\delta\Psi_b=+0.65894\) versus \(\delta\Psi_c=-0.75516\) J/m: the opposing bulk
term is 87.26% of the reduced fault variation. These are controlled fields,
not a new coupled equilibrium. See the preserved
[energy derivation](P2_accuracy_energy_derivation.md).

For a total-energy proposal the instantaneous derivative uses current recoverable
state, while direct irreversibility supplies the inequality. Retaining peak
\(\mathcal H^+\) as the **phase** driver instead defines a history-based reduced
potential and requires an explicit source/dissipation interpretation for profile
work; it is not identical to \(\Psi_b+\Psi_c+\Psi_f\). No signed bulk term may
simply be accumulated into \(\mathcal H^+\). The unloaded seeded equilibrium was
prepared using prescribed native \(\mathcal H^+\); it cannot be assumed an
unconstrained equilibrium of the proposed unloaded total energy. A pre-existing
accepted phase can remain protected by the lower bound, but its preparation and
history role need a governing decision. Neither branch is adopted here.

## Mechanics, work and dissipation

The currently selected homogeneous Maxwell step is

\[
 \beta_k=e^{-\Delta t_k/\lambda},\quad\kappa_k=(1-\beta_k)\eta,
 \quad Gs^e_{k+1}=\beta_k Gs^e_k+\kappa_k V_{k+1},
 \quad\tau^{\rm coh}_{k+1}=Gs^e_{k+1}/I^h_{k+1},
\]
\[
 \upsilon_{k+1}=\chi_{k+1}V_{k+1}+C_k,\quad
 C_k=\frac{\beta_k Gs^e_k}{\kappa_k}(\chi_{k+1}-\chi_k),
 \quad\bm\tau_{k+1}=\beta_k\bm\tau_k
       +2\kappa_k(\bm\varepsilon_{k+1}-\upsilon_{k+1}\bm{\mathcal S}).
\]

Surface-owned \(I^h\tau^{\rm coh}\) is interpolated as a nodal product; it is not
reconstructed through particles. The selected mechanical balance is the discrete
adjoint of the bulk slip-strain map \(L\), with Stokes quadrature for \(L^*\bm\tau\)
and surface \(dL\) quadrature for cohesive/frictional resistance:
\(L^*\bm\tau-M_{\mathcal F}(\tau^{\rm coh}+\mu\sigma_n+\eta^{\rm dpr}V)=0\).
It replaces the legacy particle-volume work option when explicitly selected.
Fixed-phase work adjointness and its tangent are verified. Endpoint adjointness
does not by itself close integrated work throughout a changing-profile step.

Recoverable storage terms are those above. The corresponding instantaneous
nonnegative losses, for the positive-slip fixture, are

\[
 \mathcal D_b=\int_{\mathcal B}\frac{\bm\tau:\bm\tau}{2\eta}\,dA,
 \quad\mathcal D_c=\int_{\mathcal F}\frac{(Gs^e)^2}{\eta I^h}\,dL,
 \quad\mathcal D_f=\int_{\mathcal F}\mu\sigma_n V\,dL,
 \quad\mathcal D_r=\int_{\mathcal F}\eta^{\rm dpr}V^2\,dL=0.
\]

Frictional/viscous dissipation is **not** an additional phase driver in the
selected model. Fracture expenditure is \(\Delta\Psi_f\), not an additional
copy of viscous loss. Profile work
\(W_C=\int dt\int_{\mathcal B}(\bm\tau:\bm{\mathcal S})C\,dA\) is signed
bulk/cohesive redistribution work, not necessarily positive dissipation.
Matched-measure native bulk and fault identities account for it and the fault
normalization storage jump separately; the total phase-energy budget remains
open. Spreading \(C\) over a Maxwell step and instantaneous elastic profile
redistribution have the same zero-input endpoint but different finite-step work
and losses. A proposed total-energy **time-discrete** formulation must resolve
that choice and stress-transfer state changes; the equations above are not a
verified new incremental minimization algorithm.

The historical manuscript's fixed-\(\upsilon\) incremental-work driver, legacy
particle traction projection and default particle-volume work option remain
available historical/default behavior. They must not be mistaken for the
selected P2 equations or for the proposed total-energy condition. **Review only:
no new phase law, coupled refinement, tuning or P3 is executed.**
