# Non-affine native source-history qualification

20 October 2026. Entry `f1f9f29cf40c32d1ddcea000b724375b64853532` plus the
preserved local work. Scope and original tolerances were fixed in
[BULK_SOURCE_HISTORY_SELECTION.md](BULK_SOURCE_HISTORY_SELECTION.md).
This changes only the default-off, serial, fixed-mesh homogeneous bulk
experiment. The constitutive law, seed, loading, Gauss8/source3, unlimited DG,
phase obstacle, nonlinear/energy acceptance and BP3 path are unchanged.

## Representation and ownership

Let E(A)=dev3 symgrad(A), and P3 denote the existing nine-mode Gauss3 DGQ2
mass projection on the matching cell. In addition to independent DG T, keep a
native FE velocity-memory vector A:

    A+ = beta (A_n + du),
    T_DG+ = P3[beta (T_DG,n + 2G E(du))],
    R = 2G E(A) - P3[2G E(A)],
    T_full = T_DG + R,   beta = 1/(1+dt G/eta).

For fixed mesh and constant matched coefficients, linearity gives
R+=beta[R_n+2G(Id-P3)E(du)]. Thus this is a representation extension, not a
constitutive change. It retains all nine independent DG modes. The independent
history is now **DGQ2 plus a native FE complement**, not pure DGQ2.
T_full is auxiliary stress history; physical deviatoric stress is
tau=T_full-2GaS. It is that reconstructed stress which enters mechanics/storage.
E(A) can be evaluated at arbitrary matching-cell/reference points, including
faces. The local cohesive histories remain on the original mechanics QPs;
there is no interpolation of those histories into a different material state.

The existing native FE gradient/velocity-memory representation is reused,
without its separate surface/normalization/Z owner. This bulk law has different
history ownership and must not instantiate that other constitutive route.
Every trial anchors the same accepted T_DG, A, local a/k/L/q and physical phase
obstacle. Mechanics and its consistent tangent use full incoming T and the
actual native incremental gradient. Bulk storage is |T_full-2GaS|_3D²/(4G),
including zz and both xy contributions. There is no separate remainder energy;
its cross terms are included. The existing pending-publication callback commits
both histories once, or discards both. No particle stress enters this solve.

The bulk checkpoint signature is now `bulk cohesive BE native remainder v2`.
It stores A's velocity coefficients; the projection is derived and rebuilt on
restore. A v1 bulk checkpoint is explicitly rejected without conversion.
BP3 checkpoint format and implementation are unaffected. The test-only
postprocessor's new diagnostic status flag is optional on load, allowing an old
checkpoint to reach the actual material-version guard. The earlier map::at
failure and the corrected explicit version rejection are both retained.

## Guard-preserving arithmetic correction

The first complement implementation passes the small controls but fails the
larger transition mesh at step 47, t=.11750000000000006 s. The retained check is
|T_actual-T_expected|_3D / max(1000 Pa,|T_actual|_3D) <= 1e-12.
Actual error **1.068849159616764e-12**, ratio **1.068849159616764**;
CellId `2862_0:`, reference (.019855071751231884,.019855071751231884),
physical (.50022080645749312,.94410239048639699) m. No publication occurs.
Actual/expected components [xx,yy,zz,xy] Pa are respectively
[-11.430251648338645,11.425444739146183,.0048069091924620722,1177.9780355589794]
and [-11.4302516497824,11.425444739867947,.004806909914452354,1177.9780355588307].
The original earlier pure-DG failure at .1075 s remains separate evidence.

A bounded checkpoint replay reproduces the same failure. Failure-only exports
save both FE-memory paths, du, DG/projection coefficients, vertices and point
coordinates before publication. Independent 70-digit reconstruction finds
maximum projection-coefficient roundoff 6.86245e-10 Pa, FE recurrence gradient
roundoff up to 2.39066e-10 Pa, versus only 1.00056e-11 Pa local DG-update
roundoff. Direct evaluation contributes another 3.65431e-10 Pa in xx. The FE
coefficient recurrence itself differs by at most 7.74246e-20 m. The numerical
loss is cancellation when differentiating accumulated memory, not a missing
source term, changed physical state, incorrect Maxwell rate or quadrature
capacity problem.

The correction subtracts one constant cellwise translation per velocity
component before summing the native memory gradient, accumulating in long
double. Partition of unity makes that translation's exact gradient zero.
This leaves the discrete gradient/source/projection mathematically unchanged,
while reducing cancellation. Incremental mechanics/source gradients retain
their native calculation. No tolerance, allowance multiplier or check bypass
is introduced. The final-build tests below use this correction.

## Small independent controls and lifecycle

Fresh affine and smoothly distorted 24×32 meshes share the same 1.5×1 m physical
rectangle and continuum initialization. Boundary vertices are fixed. Distortion
is dx=.006 sin(2πx/1.5)sin(2πy), dy=.004 sin(2πx/1.5)sin(2πy) m.
Both run 48 accepted steps to .12 s. Independent reconstruction implements Q2
basis derivatives and the bilinear Jacobian directly, using exported cell
coefficients, including seven arbitrary reference/face points. All tensor norms
use the embedded plane-strain metric.

| Measured maximum | Distorted | Affine control |
|---|---:|---:|
| Native full-history consistency (limit 1e-12) | 3.6050e-15 | 3.6670e-15 |
| Independent point tensor discrepancy | 2.9382e-14 | 2.4057e-14 |
| Independent nine-mode source update discrepancy | 1.3230e-14 | 1.3066e-14 |
| Retained remainder norm, Pa | .03947109 | 3.2194e-12 |
| Maximum native tangent FD error (limit 1e-6) | 2.8003e-9 | 2.1444e-9 |
| Independent bulk-energy integral difference, J/m | 1.0658e-14 | 4.0967e-14 |

The distorted control has normal stresses up to 233.37 Pa and mixed u²v² DG
coefficient .62775 Pa, exercising a non-proportional tensor history and all nine
modes. FE memory advances once with maximum coefficient discrepancy 1.0842e-19 m
against beta(A_n+dt velocity); operand-scaled discrepancy <=1.18 eps (screen 8).
Independent cohesive/fracture energy discrepancies are <=1.78e-15 J/m. Initial
unloaded hold has exactly zero nodal phase change.

A deliberate +.01 Pa DG xx inconsistency at step one is rejected by the same
1e-12 check: error 1.4142135623730951e-5, ratio 1.4142135623730951e7,
CellId `0_0:`, reference/physical point (0,0). Accepted persistent data remain
bitwise unchanged. Two converged trials at step 48 are actually rejected via
the outer pending-publication callback; retry is deterministic and accepted
histories/obstacle/solution remain unchanged after each rejection. Every accepted
step increments publications once. A 44-step prefix plus four restarted steps
matches uninterrupted advancement: ledgers and **404 step-tagged binary/CSV
exports are byte-identical**, including FE memory, DG modes, local histories,
phase, reconstructed stress and derived projection coefficients. Final-build
affine recovery is exercised directly; distorted recovery is exercised in its
resumed tail. These are serial fixed-layout checks, not MPI/AMR qualification.

## Repeated orientation control

Both fresh runs complete 58 identical prescribed timesteps to
.14500000000000007 s: aligned 3072 cells, rotated 3680 cells. The existing
45° core/transition and physical near-tip cell sizes are unchanged. Mesh,
geometry and histories are fixed during each run. The original continuum seed
is sampled anew, not transferred. Initial relative L2 errors are .33972% whole /
.74985% cap for aligned and .56561% whole / .73610% cap for rotated. Comparable
cap L2 error does not imply equal angular weak-force error. The changed annulus
and far-field discretization remain a confounder, not an accuracy reference.

| Measured result | Aligned | Rotated |
|---|---:|---:|
| Full-history source check (limit 1e-12) | 6.4051e-15 | 4.3987e-14 |
| True mechanical residual (limit 1e-9) | 1.4583e-12 | 2.7012e-12 |
| Phase KKT (limit 1e-7) | 4.7894e-8 | 2.7216e-8 |
| Coupled tangent FD at .11 s (limit 1e-6) | 2.0811e-9 | 1.0761e-8 |
| Final full 3D remainder maximum, Pa | 9.03e-12 | 6.59988 |
| First lower / upper cap activation, s | .1375 / .1375 | .1300 / .1425 |
| Final maximum cap phase increment | .000820422 | .001080328 |
| Right-region phase increment (x>1 m) | 0 | 0 |

Every accepted state publishes once. Independent original-QP weak-force
reassembly agrees within 5.25e-13 after normalization; final stress exports
satisfy tau=T_full-2GaS exactly, and independently integrated bulk storage differs
by <=8.66e-15 J/m. The fresh affine route reproduces retained aligned phase to
4.56e-18 Linf and full-tensor stress at .14 s to 1.13e-15 relative L2, on the
same timestep sequence. No mirror branch is registered or discarded.

Compare physical radial kernels K=(1-|x-z|²/a²)^3_+, a=.0625 m, with actual
JxW and unit-mass normalization. q and Y are averaged **after** evaluation on
the original QP histories; no local histories are remapped. Fracture action is
Kf integral[K+2ell² grad(phi).grad(K)]/integral K. Native nodal KKT conditions
remain distinct from these common tests. Times are .1275 s (before either cap
activates) and .145 s. Original-QP q, phi, Y, fracture and total cuts are exported.

The broad lobes stay near the same physical directions: at r=.288111 and
.298515 m, both meshes' final Y peaks are **±43°** on the 1° angle grid. At
r=.225 m they are ±45° aligned and ±44° rotated. Relative L2 differences in
angular Y are 2.51%, 3.64%, 3.03% at those three radii. This supports loading-
oriented broad driving rather than a 45° rotation with mesh axes.

Discrete onset is less robust. Aligned first nodes are ±40.6013°, ±42.8789°,
±49.3987°. Rotated first growth is one lower node at -45°, r=.296875 m;
its pre-step native weak margin is 1.05155 Pa and accepted residual
-2.04e-10 Pa. Upper growth waits five more steps, then activates nodes spanning
29.4759–57.5288°, r=.28125–.29190 m. The rotated peak increment is **31.68%**
larger. Symmetric physical loading therefore does not guarantee symmetric or
mesh-independent discrete onset. These angles do not undergo a rigid 45° shift,
but exact first-node selection and ordering remain sensitive to the represented
profile, test functions and mechanical discretization.

Initial common-kernel fracture differences already reach 1.47934 Pa at
r=.288111 m, comparable with the last positive native onset margins. The sampled
final phase/fracture kernel Gauss8-versus-24 differences are .00772 / .01278 Pa,
far smaller than that initial representation difference. This is a phase-only
sampled diagnostic control; it neither re-evaluates constitutive histories on
new points nor qualifies all mechanical quadrature errors. The supported
mechanism remains mechanical driving lobes combined with discrete weak
resistance/sampling. The run does not identify a unique physical onset angle.

![Common-scale mesh, phase, driving and actual increments](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_source_history/physical-overlays.png)

The same seed and boundary shear were imposed on two fixed meshes, with fresh
unloaded histories. Colours show calculated initial phase, accepted driving and
actual nodal growth; white contours show phase, not the driving field. Red
circles mark the first-growing nodes and black crosses the original endpoint.
The broad lobes persist while the growing-node pattern changes. This is not
propagation or convergence qualification.

![Common physical angle cuts](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_source_history/common-kernel.png)

These compatible weighted cuts separate initial represented fracture resistance
from generated mechanical driving. Pale driving curves are at .1275 s; dark
curves are at .145 s. Vertical marks indicate first-node angles, not a new
constitutive threshold. Positive common-kernel totals are compatible with a
zero native nodal residual because those are different test functions.

The new interval's external energy remainders are .0499507 / .0499188 J/m,
**2.3536% / 2.3528%**, still failing the retained 1% physical screen. Stress
backward-Euler losses are .0492729 / .0492285 J/m; traction losses
.000665822 / .000681515 J/m, negative phase remainders -1.331e-7 / -7.794e-8 J/m,
and fracture losses 1.208e-5 / 8.915e-6 J/m. Phase/mechanical residual-work
magnitudes are <=7.68e-12 J/m. The ledger closes, but the physical time-
discretization remainder remains. The earlier full-interval/halving failures
are preserved without relabeling them as source error.

## Limits and reproducibility

The external physical-energy screen remains a separate failed qualification;
local ledger closure and a converged tangent do not repair it. Preserve
`strict_reference_match_passed=false`, all original errors/branch findings and
the protected BP3 evidence. No server run, mesh sweep, limiter experiment, AMR,
transport, birth or moving geometry is included.

The representation limitation is removed and this bounded serial lifecycle /
orientation-through-activation task is complete. Retain the seed and loading;
there is no basis here for reshaping the cap to suppress flank patches. Resolve
the separate physical energy/work qualification before claiming propagation.
MPI, AMR, varying coefficients, transport and moving geometry would require
additional FE-complement/local-history transfer rules and are not qualified.

Evidence: [assessment and figures](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_source_history/assessment.json),
[small distorted control](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_source_history/distorted-checks.json),
[affine control](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_source_history/affine-checks.json),
[70-digit failure diagnosis](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_source_history/roundoff-diagnosis.json),
and [replay/provenance](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_source_history/README.md).
The original build/output/checkpoints and failed attempts remain under
`build-particle-free/bulk-source-history/`; commands and bounded timeouts are
recorded in copied run JSONs. Validation of source, aligned and rotated inputs
passes. Tests use GCC 12.4 / deal.II 9.6.2 / Trilinos 14.2, serial Release.
Two private-header-dependent native unity units were rebuilt and linked with
unchanged original objects into an isolated executable, plus matching new DSOs;
a full clean build is not claimed. Native SHA256:
`31ad1e59820079e712bdae23732697394cbfe36e8377001fa50cdac2a05b8d5a`.
