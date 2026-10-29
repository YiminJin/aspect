# Native exact-jump pilot: discrete problem and ownership

Default off; fixed Cartesian bulk mesh, homogeneous small-strain Maxwell law,
closed contact, quasi-static mechanics and the previously accepted cohesive–RSF
law/path hypothesis. No bulk inertia, opening, transport, AMR, bulk-phi fracture
inventory, new direction law or R calibration. This is the declared finite-wall
specimen, not an approximation newly qualified for distant boundaries.

## Physical state

| State | Native owner / persistence | Meaning |
|---|---|---|
| Complete independent tensor | `BulkHistory`: CellId + 27 DGQ2 coefficients | xx, yy, xy; nine modes each; zz=−xx−yy, xz=yz=0 |
| Native FE displacement memory | `BulkHistory`: native distributed vector, physical per-cell checkpoint records | Retains the native gradient not representable by DGQ2 |
| Signed jump and relaxed jump trace | System: Q1 coefficients j,m and immutable initial Gauss offset | Their Gauss evaluations must match the owned point histories |
| Gauss jump, a, normalized age, reference, relaxed jump, absolute slip | Manager left-vertex rows, 18 components/cell | Exactly three Gauss3 physical records per formed cell |
| Exact curve and material identity/admission clock | System accepted state | Persistent formed prefix; never rotated or rewritten |
| Zero-history candidate view | Separate zero tail in the immutable accepted/trial geometry | No formed identity; can retire before admission; saved to reproduce scheduling |
| Velocity and current pressure perturbation | Actual ASPECT solution vectors | Recomputed mixed equilibrium, normal native solution checkpoint lifecycle |

The manager's old nonnegative nodal-rate channel is initialized to zero solely
for its append/container contract. It is unused by this signed formulation;
physical rates are the Q1 jump increment divided by the physical interval.
Derived q/resistance/normal diagnostics never feed accepted mechanics.

With native displacement memory d_R, Gauss3 projection Pi3, and fixed compact
signed tangential lifting L_j,

```
T_b = T_DG + 2G dev3(sym grad d_R) - Pi3[2G dev3(sym grad d_R)]
tau = T_b + 2G sum_j m_j dev3(sym grad L_j)
beta = 1/(1+G dt/eta)
T_DG+ = beta (T_DG + Pi3[2G dev3(sym grad du)])
d_R+ = beta(d_R+du);  m+ = beta(m+dj).
```

The native source and complete tensor are checked independently at Gauss6;
all nine DGQ2 modes and the full native remainder survive. Gauss3 is the source
projection rule, not an assertion that every native gradient is representable.
The separate distorted-cell fixture demonstrates a nonzero retained remainder.

The general-history integration control adds the divergence-free Airy tensor
`(20(x−6)^2/36,20y^2/36,−40(x−6)y/36)` Pa. Its corresponding plane-strain elastic
strain has incompatibility `80/(36G)` m^-2 away from the interface. It therefore
cannot be silently replaced by the prototype's compatible affine/lifting
response span. It remains in the DG history, decays by Maxwell relaxation and
participates in actual native residual, pressure, storage and restart checks.

## Native mixed/condensed solve

A is the actual native constrained Q2/P1-discontinuous reference mixed operator
with `2G dev3(eps_i):dev3(eps_j)` and both pressure/divergence blocks. The solver
adapter initializes the standard ASPECT pressure scaling, builds native AMG and
calls `solve_stokes` in process. The smooth unknown is an incremental displacement;
the accepted ASPECT velocity is du/dt. Physical G and eta belong to the pilot;
the simple material's numerical viscosity G supplies its reference preconditioner.

Solve reference unit-jump columns A w_j = −B_j. K_jj=D+B^T w and the normal-test
loads use the same bulk cut quadrature and constraints. The normal bubble measure
is **−integral(div L_n)** on the native bulk rule; surface point count and a
surface-only normalization are not substituted. Mean-zero p' is a pressure
reference; p0=10000 Pa is physical confinement and is never shifted to match data.

For the immutable incoming *general* tau, the native base solve supplies f0 and
normal offset n0. The full surface equation is

```
f0 + beta K dj + Q^T W r(old_Gauss, Q dj, p0+n0+beta N dj, dt) = 0.
J = beta K + Q^T W [r_j Q + r_p beta N].
```

The nonsymmetric current-pressure term is retained. General history enters f0,
n0 through native bulk integration/inversion, not fictitious response coefficients.
The exact intact ties use the admissible Q1 master space; the full signed weak
certificate remains <=1e-8 Pa. Accepted-active-set Newton is only a predictor.
The retained positive symmetric-tangent diagnostic is not a claim of a global
RSF incremental potential or general nonlinear stability/uniqueness.

## Single fracture inventory and work

C0=2Gc/tau_p^2, df=2Gc/tau_p. For e=j−reference,
`a+=max(a_old,min(|e|/df,1))`, `q=(1−a)e/(C0 a)` when 0<a<1, and q=0 at a=1.
At a=0 the weak intact constraint supplies |q|<=tau_p; no phase floor substitutes
for it. Cohesive storage is `(1−a)e^2/(2C0 a)` and fracture is `Gc integral(a ds)`.

The normalized implicit aging update is
`theta+=(theta_old+V0 dt/Dc)/(1+|dj|/Dc)`.
The retained regularized RSF coefficient is
`mu=a_RSF asinh(|V| exp((mu0+b_RSF log(theta+))/a_RSF)/(2V0))`.
Resistance is `q+a(sign(V) pn mu+eta_rad V)`, with current closed-contact pn.
Friction and radiation dissipations are separately integrated, nonnegative.
Metadata admission adds zero work and zero extra fracture charge.

Physical full-tensor bulk storage, Maxwell loss, BE bulk remainder, cohesive
storage, Gc da, friction/radiation and the surface endpoint remainder are computed
from native fields and actual Gauss weights. The exact BE ledger and trapezoidal
physical boundary work are separate. Boundary work uses the actual prescribed
FE particular displacement, including current total pressure; old pressure enters
trapezoidal work only, never current friction. A small signed final remainder
is not used to hide cancellation: report the sum of absolute step remainders
against both sliding dissipation and added fracture work.

## Geometry, acceptance and restart

R=.75 m is the retained physical persistence length. The ahead-disk accepted
stress average selects the nearest forward maximum-shear conjugate; true ties
remain an explicit unsupported ambiguity. Integrate dtheta/ds=(theta*−theta)/R.
The .75 m lifting half width is a separately selected numerical support. Candidate
spacing is .09375 m, initial trial reserve .1875 m, refresh .0025 s. Actual
admission always refreshes from the incoming accepted stress when necessary.
A formed cell (any nonzero physical damage/slip/memory) is persistent; its
geometry, material identities and all previous histories survive.

Every bulk/surface trial anchors the same accepted state. The real outer
TimeStepping repeat decision discards the candidate and restores the native
solution; exactly one accepted publication swaps geometry and histories.
Checkpointing a live unaccepted trial is rejected. Native checkpoint v2 stores
physical CellId tensors/native coefficients, manager Gauss histories, exact
curve, temporary view, identities and clocks. Geometry/operator caches rebuild
on resume. The payload is absent when the option is off; old checkpoints are
not converted. Checked prototype initialization is a separate explicit operation,
qualified for the retained loaded state by common physical stress, full storage
and native jump/normal-operator comparisons.

The endpoint guard closes only operand-derived floating-point clock gaps after
actual publication. Count n accepted `advance_time` additions and two operations
per actual outer retry r; three additional roundings cover the inputs/final
comparison. With u=epsilon/2, k=n+2r+3, gamma=k*u/(1-k*u), use the outward-rounded
bound gamma/(1-gamma)*max(|t|,|end|). The retry count persists in native checkpoint
diagnostics (the audited older unrejected fixture has r=0). This replaces a
fixed-eight-ulp allowance that failed after 160 additions and created a tiny
closure interval/pending restart step. The last meaningful interval is retained
at the accepted endpoint. No residual, source, energy or constitutive allowance
changes. Original extra-step/ramp outputs remain failure evidence.

Bulk integration/unknowns remain MPI-owned with native ghosts. Only small surface
operators/records are replicated. Checkpoint global CellId gathering reuses the
established convention. No Python trajectory or external mechanical kernel is
called by the native evolution solver.


## Initialization diagnostic correction

Full mixed-element interpolation of an affine displacement leaves the nonnodal
P1-discontinuous pressure entries nonfinite in this deal.II path. An independent
small fixture demonstrates `unmasked_pressure_finite=0`, while velocity-only
interpolation gives identical velocity coefficients and exactly zero untouched
pressure. The explicit snapshot initialization now interpolates **only velocity**
and obtains pressure from the actual mixed solve. It independently checks the
converted pressure against the saved physical reference, alongside tensor/storage.
Native normalized-pressure bookkeeping is initialized explicitly; nonfinite
initial/restarted fields or physical-work diagnostics reject before publication.
Original nonfinite initialization/first trapezoidal-work exports are preserved.
The corrected native prefix is compared against every persistent history in the
old prefix before reusing its later, already-finite continuation. No constitutive
history, law, pressure datum, or numerical allowance is changed.
