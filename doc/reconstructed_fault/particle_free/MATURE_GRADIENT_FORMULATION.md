# Straight-band reduction and mature-friction compatibility

This note analyzes the retained bulk law; it does not change the native mature
fault specialization, its auxiliary slip memory, RSF law or BP3. A frozen
phase field is not by itself the existing explicit mature constitutive mode.
The numerical control below is a prescribed **near-mature bulk profile**;
the exact mature limit is derived separately. The bulk API admits phi<1 only.

## Slip, units and normalization

Let t and n be fixed orthonormal tangent/normal vectors to a straight fault,
S=sym(t⊗n), S:S=1/2, and r the cross-band coordinate [m]. With both tensor
components free, define

```
k(r,t)=2 K:S                 engineering accumulated shear [1]
w(r,t)=2 W:S=dot k           engineering slip-strain rate [1/s]
delta(t)=integral k dr        signed accumulated interface slip [m]
V(t)=integral w dr            signed interface slip velocity [m/s]
delta_abs=integral L dr       accumulated equivalent slip [m]
s_e=2 integral A:S dr         relaxing elastic/cohesive slip moment [m]
```

V is not the boundary relative velocity: elastic/Maxwell bulk deformation
also contributes to boundary motion. In general delta_abs>=|delta|, and K
need not remain parallel to S. Under the symmetric monotonic shear used here,
K=k S and W=w S; both conjugate systems remain available. This is a solution
symmetry, not a direction lock. Delta_abs=delta then holds.

For a scalar profile ansatz k=delta rho, w=V rho at fixed rho,
`integral rho dr=1`, rho has units 1/m. For the retained cohesive ansatz
`2 A:S=s_e chi`, use `I=integral h dr` [m], `chi=h/I` [1/m]. This yields
`q=G s_e/I` and `Ecoh=G s_e²/(2I)` [J/m²]. Chi does not define a general
K/rate profile once gradient forces are present: computed rho_K=k/delta and
rho_W=w/V are measured separately. No unnormalized phase mask substitutes
for interface slip. At V=0, work-averaged resistance ratios are undefined.

## Gradient energy and independent resistance

With K=k S and the full symmetric-tensor metric,

```
Egrad/area = (G ell_s²/2) integral (k')² dr                 [J/m²]
X = -2G ell_s² K'';  X:S = -G ell_s² k''                  [Pa]
microtraction = (2G ell_s² grad K) n_exterior              [Pa m]
```

For a fixed normalized shape, J_rho=integral (rho')² dr [1/m³], hence

```
Egrad = (G ell_s²/2) delta² J_rho
Rgrad = dEgrad/ddelta = G ell_s² delta J_rho                [Pa]
Hgrad = G ell_s² J_rho                                    [Pa/m].
```

This is recoverable accumulated-slip hardening even if cohesive Q vanishes.
It is not Coulomb/RSF friction or Maxwell dissipation. For rho=f(r/b)/b,
Hgrad scales as b^-3. A nonuniform fixed-width band has J_rho>0. The only
zero-gradient profile on the whole microfree transect is uniform slip across
the whole specimen, which loses fault localization. Allowing b to grow can
reduce the force but changes the mechanical slip band. For changing rho,

```
Rgrad = integral X:W dr / V
      = G ell_s² [delta J_rho + delta² integral rho' dot(rho)' dr / V].
```

Thus `2Egrad/delta` is the fixed-shape force; it need not equal the actual
power resistance during redistribution. Boundary microtraction work must be
included unless zero. No boundary skin or extra internal microboundary is
inserted here.

For constant physical shear traction tau_s and monotonic V>0, the exact power
partition at fixed phase is

```
tau_s = Rcoh + Rgrad + Rfr
Rcoh  = integral Q:W dr / V
Rfr   = integral sum_i (c_r+mu pn_i) w_i dr / V
pn_i  = p0+p' - nu_i.tau.nu_i.
```

The phase profile stays fixed. Cohesive resistance in shoulders/stronger
surroundings is recorded separately from that in the nearly mature core.
Recoverable bulk/cohesive/gradient storage, bulk/cohesive Maxwell loss,
friction, and the three BE quadratic losses are distinct. In particular
`dt integral X+:W dr = Delta Egrad+Egrad(Delta K)` for microfree boundaries.

The intended native mature specialization explicitly has C=Ecoh=0 while
retaining auxiliary Maxwell/slip memory. In the bulk formal limit h→infinity,
Q and Ecoh vanish for bounded A, but the new constant gradient coefficient
does not. At fixed V and steady RSF state, desired traction is N mu_ss(V)
(plus any explicitly selected radiation term), independent of indefinitely
accumulated delta. A nonzero Hgrad violates that requirement. A narrow finite
slip window can approximate friction within an error budget; that does not
establish an asymptotic mature limit. Current tests use constant mu, the
necessary a=b=0 friction limit, without introducing state evolution.

There is a separate orientation distinction: for symmetric pure shear with
both free Coulomb conjugates and Q=X=0, the existing model gives
`tau_s=mu p0/sqrt(1+mu²)=995.0371902 Pa`. Sliding on a prescribed aligned
interface would give mu p0=1000 Pa. This known ~0.5% difference is not removed
by recalibrating mu or locking directions in this test. Physical stress,
individual Coulomb planes, flow-tensor orientation and fault geometry remain
distinct.

## Mechanically equilibrated numerical reduction

Let u=(u(r),0), no normal motion/body force. Exact mixed equilibrium imposes
constant tau_xy=tau_s and constant sigma_yy. Txx=Tyy=Tzz=0 remains invariant
under these fresh shear histories; tau_xx=-2G Axx, tau_yy=2G Axx. The same
mean perturbation-pressure reference as the native confined block gives

```
p(r)=p0+tau_yy(r)-mean(tau_yy),  sigma_yy=-p0+mean(tau_yy).
gamma_increment(r)=[tau_s+2G Axy+]/(beta G)-Txy_n/G
integral gamma_increment dr = imposed relative displacement increment.
```

Solve both K components, tau_s/G and mean(tau_yy)/G jointly. The complete
normal-map Jacobian includes dependence on tau and the recovered pressure;
pressure is not held fixed in derivative tests. The mechanical constraints
above are an exact 1-D elimination, not prescribed point strains. Current
sliding solutions have Axx=0 by symmetry, but arbitrary tensor/pressure
variations are included in the independent action check.

Reuse the qualified C++ lens projection/action, Maxwell recurrences and
positive P1 mass/gradient structure. The new benchmark does not call the
multivalued prescribed-X inverse. Natural M n=0 is imposed at the external
ends, no internal K constraint. The algebraic counterpart of X is the weak
gradient stiffness divided by constitutive mass. All states start freshly
from the same continuum profile. This is an independent strip reference,
not a new ASPECT native mesh or a production history backend.

## Published comparisons

| Formulation | Relevant construction and distinction |
|---|---|
| [Gurtin (2003)](https://doi.org/10.1016/S0749-6419(01)00018-3) | The microforce balance, energetic polar stress and boundary work underlie our full-K-gradient choice. Gradient defect energy can produce size effects/backstress; that is not a theorem of frictional steady sliding. Our Maxwell A and non-associated pressure-dependent closure are additional ingredients. |
| [Miehe, Teichtmeister & Aldakheel (2016)](https://doi.org/10.1098/rsta.2015.0170) | This specific 2016 paper combines gradient plasticity and fracture with independent plastic/damage lengths and micromorphic regularization. Its accumulated equivalent plastic variable alpha differs from signed tensor K; its variational standard-dissipative setting does not supply our nonsymmetric Coulomb tangent or an RSF limit. Related finite-strain work is [Miehe, Aldakheel & Raina (2016)](https://doi.org/10.1016/j.ijplas.2016.04.011). |
| [Pranger et al. (2022)](https://doi.org/10.1029/2021JB023511) | Spatial interaction regularizes anelastic rate/state evolution. Transect-integral calibration connects continuum and interface friction/state; it is not quadratic recoverable storage in total accumulated K. Their normalized convolution/implicit spatial interaction preserves a relevant integral, but nonlinear transient agreement requires more than setting an integral of a Laplacian to zero. The paper reports regime/calibration limitations. |
| [Fei et al. (2023)](https://doi.org/10.1002/nag.3465), [in-plane extension](https://doi.org/10.1002/nag.70416) | Displacement jump/slip velocity is mapped through a diffuse crack-density kinematics. The 2023 paper is antiplane; the later paper retains tensorial in-plane contact. Mature fracture traction follows RSF/contact with state separate from phase. Phase-gradient fracture energy is not an added quadratic energy in accumulated cross-band slip. Neither formulation establishes our Maxwell/tensor-history/gradient model as equivalent. |

These are model comparisons, not reasons to inherit a published stability or
length-insensitivity claim. In particular, a plastic regularization length
and the phase length are not automatically interchangeable.
