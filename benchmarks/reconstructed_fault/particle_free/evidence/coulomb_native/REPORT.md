# Confined Coulomb block: fixed-phase native qualification

Completed on `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local
work. The default-off **closed Coulomb fixed phase** mode is connected to the
actual mixed native solver, DG/native source history, outer publication and
checkpoint owners. This qualifies one small fixed-phase material block, not
phase evolution, a physical propagation specimen or mesh/time convergence.

## Physical problem and implementation

[Selected inputs](SELECTION.md) use a fresh 0.25 m square with
4×4 cells, Q2 velocity/Q1 pressure, Gauss6 mechanics/source3 and exactly fixed
phi=.2. There are 617 native DoFs (162 velocity, 25 pressure, 25 phase and
static fields), 432 independent DG coefficients and 576 mechanical Gauss
points. Zero T/A/K/Q/L histories are initialized directly. Diagnostic particles
remain stationary; no domain construction or particle history feeds mechanics.

The unchanged material tuple is G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m²,
ell=.046875 m, m=4, strength-drop input=2000 Pa. The new law has mu=.1 and
residual cohesion c_r=0, **distinct from that strength-drop input**. At the
reference pressure 10000 Pa the residual pure-shear strength is 995.037190 Pa
and intact threshold 2955.332456 Pa. Neither is recalibrated; intact activation
is not exercised by fixed phi=.2. [Parameter table](parameters.csv).
Resolved ParameterHandler inputs and the accepted checkpoint independently
confirm this tuple and the separate p0/mu/c_r metadata.

Use tensile-positive sigma=tau-(p0+p')Id, with actual prescribed p0=10000 Pa
and solved perturbation p'. This mode explicitly interprets `Surface pressure`
as p0, with prescribed **integral p' = 0**. Pressure-dependent friction removes
the freedom to treat this physical reference as an arbitrary gauge. All boundary
velocities are prescribed. Initial reactions are exactly -p0 n: the integrated
force is 2500 N/m on each face, with the appropriate opposite signs. No second
Neumann pressure is applied. Background pressure does no work for the imposed
volume-preserving motion; it adds no volumetric storage to this incompressible
model. Total reactions, including p0, are used in work calculations.

The promoted material-point body is identical to the qualified implementation.
Use pn=p0+p'-n.tau.n and f=|(tau-Q):S|-mu pn on both conjugate systems, with
zero dilation. Accepted tensors remain physical memory. Net accumulated L and
the sum of plane slips remain distinct; friction power is sum(mu pn_k w_k).
No opening, pressure clipping or imposed direction is added.

With native unknown pi=p'/G and fixed phase, the assembled weak rows are

    R_ui = integral [tau:D_i/G - (p0/G+pi) div(N_i)],
    R_pj = -integral P_j div(du).

The complete tangent includes tau_e[D_j]/G and
tau_P[G P_j]/G-div(N_i)P_j in the velocity rows, and -P_i div(N_j) in
continuity. tau_P includes changing plane directions, not only the explicit
pressure term. Native AffineConstraints apply C^T R and C^T J C for the physical
mean-pressure reference. UMFPACK solves the nonsymmetric reduced matrix;
nothing symmetrizes its pressure coupling. MPI assembly retains unique CellId
ownership and the existing rank-zero direct solve.

Phase is constrained to its accepted field throughout the new mode. Enabling
phase evolution is explicitly refused. The retained symmetric phase Hessian
check remains intact for the old laws and is **not invoked as a Coulomb
certificate**. New Coulomb point caches are allocated only in this mode.

## Measured verification

Both loads run 24 steps at dt=.0125 s: shear rate .02/s through .1 s, normal
deviatoric rate .02/s through .25 s, then a .05 s displacement hold. The second
case adds the declared divergence-free sinusoidal boundary motion. All starts
are fresh; no P5c checkpoint is converted.

| Check | Measured result |
|---|---:|
| Homogeneous physical means/storage/dissipation vs point schedule | 3.59e-15 scaled defect |
| Homogeneous maximum pointwise field defect | 1.04e-9 scaled; not 1e-12 identity |
| Complete mixed strain/pressure FD, including pressure-only probes | 2.44e-9 relative; bound 1e-6 |
| Maximum native mechanical residual | 9.962e-10; bound 1e-9 |
| Independent native/DG source discrepancy | 7.79e-15; bound 1e-12 |
| Native A/K/Q/L recurrence defect | 2.61e-15 scaled; bound 1e-12 |
| Minimum trial normal pressure, independently reconstructed from accepted W | 2550.31 Pa |
| Minimum normal pressure over every in-plane direction at Gauss points | 3155.97 Pa |
| Minimum out-of-plane compression | 9065.76 Pa |
| Nonuniform p' range over trajectory | -1111.33 to +1121.34 Pa; mean zero |
| 1/4-rank physical quadrature-field comparison | 1.08e-13 scaled; bound 1e-12 |
| Same-rank restart, each of 1 and 4 ranks | 48/48 exported files bitwise identical |
| Retained constant-R native control vs entry executable | 27/27 files bitwise identical |

The strict 1e-12 pointwise postprocessing assertion failed and is retained in
the original log/JSON. The local shear-stress spread is about 1.03 microPa,
whereas physical means meet the original 1e-12 mean-control criterion. These
are different norms: this is not pointwise roundoff identity or a relaxed source
check. Native solve/source tolerances are unchanged. The material-point checks
also pass again after promotion, with the original identity/FD bounds.

The nonuniform case exercises stick, each individual conjugate system and
both together. Every actual mechanical point is checked for compressive trial
and returned stress; the CSV also records the independent minimum over all
in-plane normals. Isotropic initialization has undefined candidate directions,
marked by false flags/NaN angles, and 10000 Pa normal pressure in every direction.
No constitutive return is evaluated on averaged histories for these diagnostics.

Outer native trials are rejected twice and retried deterministically before
the ordinary single accepted publication, on both rank counts and after
restart. Accepted tensors, native memory and solution remain unchanged during
rejection. All 120 positive intervals across the controls/forks pass the gates.
The distinct checkpoint tag preserves all existing tensor histories and rejects
conversion. Changed p0 is rejected by ASPECT's existing critical-parameter
check; copied checkpoint files remain unchanged. A fresh constant-R checkpoint
is also refused by the new mode before publication.

The first FD attempt used a nonzero-mean pressure probe; imposing the physical
mean constraint amplified its dependent coefficient and crossed active sets
during the hold. Its original error (5.16e-5 against 1e-6) and switches remain.
The corrected probes have compatible mean-zero layout and explicitly check
active-set consistency, with at most 12 branch-preserving halvings allowed.
Final probes require no halvings. No verification tolerance or physical timestep
changed. An initial CSV filename-glob error is preserved as a setup failure.

## Work, output and cost

Independent full-reaction work agrees with native work within 1e-10 J/m;
integrated plane-friction power agrees within 1e-12 J/m. Dissipation is
nonnegative, and the BE ledger error is at most 1.63e-16 J/m. Over the
nonuniform trajectory, in J/m:

| Quantity | Value |
|---|---:|
| Endpoint / trapezoidal reaction work | 1.012362 / .946449 |
| Recoverable storage change | .711793 |
| Bulk / cohesive viscous dissipation | .001057 / .000781 |
| Friction dissipation | .234189 |
| Stress / cohesive numerical remainder | .037570 / .026972 |
| Mechanical residual work, signed / sum of absolute values | 2.560e-8 / 2.579e-8 |
| Background-pressure work | 1.28e-16 |

Initial fracture storage is .1 J/m and remains fixed. The endpoint physical
remainder is .064542 J/m (6.38% of work), separate from the algebraic ledger.
The endpoint/trapezoidal difference is not a physical-energy accuracy pass;
no timestep convergence or 1% physical-work claim follows from this block.
The hold retains nonzero stress while Maxwell memory relaxes.

![Accepted response](response.png)
![Accepted quadrature fields](fields.png)

The imposed block motion, confinement and fixed phase are inputs; stress,
pressure perturbation, plane slips, reactions and histories are native results.
White plane marks show active flow systems, not crack geometry. Nothing here
tests phase-front selection, opening, finite rotations or propagation.

Final nonuniform full-run wall times are 3.62 s/4.03 s on 1/4 ranks. Accepted
solve totals are 1.018/.845 s; rank-zero assembly .678/.245 s, linear solves
.0345/.0450 s and history work .0466/.1949 s. Peak memory is about 216/214 MiB
per rank. These include diagnostic/checkpoint costs, lack repetition, and do
not imply a scaling or full-specimen speed guarantee. Native tests used bounded
timeouts, ran sequentially and all ended.

## Concrete proposal for the subsequent phase task

Do not seek a minimum of the old incremental potential. Retain physical
irreversibility/bounds, closed contact, nonnegative dissipation, true mechanical
residual, phase KKT and the separate work ledger. At immutable incoming history,
eliminate mechanically equilibrated variables m=(du,p') and differentiate the
physical phase force g. In admissible independent-master coordinates use

    L = g_phi - g_m (R_m,m)^(-1) R_m,phi.

L is nonsymmetric. A sufficient **local VI robustness/uniqueness** screen is
positive definiteness, beyond operator/linear-solve uncertainty, of the symmetric
part of M^(-1/2) L M^(-1/2) on a superset of the phase critical cone. M is the
physical phase test mass; no artificial constrained diagonal enters. This is a
strong-monotonicity screen, not an energy Hessian or a claim of general dynamic
stability. A negative symmetric-part direction is inconclusive by itself for a
non-associated dynamical law; it must not be relabeled as the old saddle test.
Invertibility and conditioning of the constrained mechanical/pressure block
also need verification; a failed mechanical elimination cannot certify L.

The next bounded task should qualify every coupled action and this reduced
operator against independent constrained differences with mechanics/pressure
allowed to respond, including yield switches and intact right derivatives.
Verify admissible perturbation/recovery, continuation, rejection and restart
on a small confined phase-active block before a specimen. Generalized derivatives
at switching sets need consistent treatment; one smooth-branch matrix alone is
insufficient. If the sufficient screen fails, stop without publication rather
than disable a guard, rank roots by the unavailable potential or add viscosity.
Selection outside a monotone neighborhood needs an explicit reviewed stability/
evolution rule. This is a proposal only; no phase-active run is begun.

All prior P4/P5c limitations, branch evidence and strict_reference_match_passed=false
remain. BP3 artifacts and production defaults are preserved. Source changes,
commands, useful PNG/CSV/JSON and checksums are in the compact review archive;
full native runs/checkpoints and original failed diagnostics stay separately
under build-particle-free/coulomb-native.
