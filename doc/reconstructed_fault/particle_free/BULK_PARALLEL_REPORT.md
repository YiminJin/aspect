# Fixed-mesh bulk MPI and horizontal-side-release control

21 October 2026; entry `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved
local work. [Predeclared scope and comparison criteria](BULK_PARALLEL_SELECTION.md).
Evidence is under
[`bulk_parallel`](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/).
The selected qualification and the one fresh boundary control are complete.
All numerical runs were local; no server job or full event was launched.

## Retained equations and physical fields

This is the selected default-off **bulk** cohesive model, distinct from mature
BP3 and the earlier surface-only model. The material-point implementation is
byte-identical. At fixed mesh and matched constants,

    beta = 1/(1 + Delta t G/eta),
    T_DG+ = P3[beta (T_DG,n + 2G dev3 symgrad(du))],
    A+ = beta (A_n + du),
    T_full = T_DG + 2G dev3 symgrad(A) - P3[2G dev3 symgrad(A)],
    tau = T_full - 2G a S,  S_xy=S_yx=1/2,
    h(phi)=m phi(1+phi)/(1-phi)^2,  q=G a/h,
    Y=h'(phi)q^2/(2G),
    g_i=Kf integral(N_i + 2 ell^2 grad(phi).grad(N_i)) - integral(N_i Y).

The original return map, virgin h=0 multiplier, rate conventions and BE update
of signed/absolute shear histories are unchanged. All tensor contractions retain
zz=-(xx+yy) and both symmetric xy entries. Source projection is the actual
Gauss3 nine-mode DGQ2 projection; mechanics/fracture use Gauss8. No transport,
limiter, new material assumption, geometry or phase cap is introduced.

Saved affine states at .15,.15125,.15375,.155,.16,.17375,.175,.185 s were
independently reconstructed from their DG coefficients, native FE memory and
actual local histories, using the exported native cell/shape associations on
the identical fixed mesh. The complement was computed, not dropped. Maximum
agreement with original dense physical-stress exports is **1.5462e-11 Pa**;
maximum Y difference is **8.53e-14 Pa**. Original histories are never moved into
an interpolated constitutive state.

[Whole-domain stress/Y/weak-force maps](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/remote-stress-balance.png)
and [centerline/upper/lower profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/remote-profiles.png)
separate physical tau_xy, local Y, weak fracture resistance and the weak
mechanical-minus-fracture balance. QP maps use positive JxW cell means; curves
use the same compact physical kernel, radius .0625 m. Nodal force comparisons
average integrated forces with their matching nodal mass. Spatial axes are
physical metres. Original endpoint (.5,.5), continuum compact support, initial
phase contours and newly active initially-zero regions are marked separately.

In a virgin, uniform intact patch, Y=Kf gives
|tau_xy|=R+sqrt(2 G Kf/m)=**2000 Pa**, with R=1000 Pa. This is a consequence of
the retained law, not an imposed envelope. The affine maximum shear stress
changes from 1973.88 Pa at .15 to 2000.80 Pa at .15125, alongside its first
remote activation. A pointwise maximum alone is not the nodal growth rule:
growing nodes balance their integrated driving and resistance within native
KKT accuracy, while obstacle-active nodes can retain negative driving-minus-
resistance. The smooth initial cap also has a strongly nonuniform weak fracture
force; a large local Y lobe must not be read as a phase contour.

## MPI implementation and qualification

Each owned cell contributes its raw residual/tangent once. Collective reduction
precedes native AffineConstraints condensation. Physical phase duals retain
C^T g and the tangent C^T H C; already-global fracture entries are added once.
The coupled search is replicated, with UMFPACK factorization/solve on rank zero
and collective correction/error publication. This is parallel cell assembly and
history handling, **not distributed direct factorization**. No linear/nonlinear
or source tolerance is relaxed. MPI hanging-node phase evolution remains guarded.

The old, unused complete phase sparsity extension for surface normalization
caused the first two-rank setup failure by inserting nonlocal rows. Bulk mode
now uses the ordinary cell graph there. A sparse raw-fracture output avoids an
O(n_phase^2) replicated Hessian; the old dense API and return type are retained.
The failed setup output, the stale-object retry and the sandbox PMIx failure are
preserved separately from the passing runs.

Accepted DG coefficients, local a/k/L/q and native FE memory have unique cell/
DoF owners. FE ghost values are refreshed before cell evaluation. v3 native bulk
checkpoints serialize unique physical CellId records, including corners/physical
phase, all 27 DG coefficients, local native FE coefficients and every QP history.
Restore verifies mesh, coefficients, quadrature and physical obstacle, then
rebuilds the native projection. v2 serial bulk restore remains readable;
conversion to MPI is not introduced. BP3 checkpoint code/format is unchanged.

Fresh 24x32 controls at Delta t=.0025 s through .16 s use identical accepted
steps and develop peak phase increment .00549053. Every physical field is
compared at common coordinates/CellIds, without mirror alignment. Relative
physical-stress differences are <=4.07e-13; full owner, growth, Y and storage
comparisons pass the predeclared 1e-10 reporting threshold. Near-zero unloaded
stress comparisons use the stated 1000 Pa scale; their absolute differences
are tiny roundoff, not finite physical disagreement. Native mechanics <=
3.15e-10, KKT <=8.18e-9 and independent source <=4.24e-15 pass their original
1e-9,1e-7,1e-12 checks. Maximum exact BE ledger defect is 2.11e-14 J/m.

For each of 1/2/4 ranks, the accepted step-47 checkpoint (.1175 s; pending clock
.12) resumes through .16, including two converged outer rejections, deterministic
retry and one accepted publication per physical step. All **74** common
phase/DG/native-FE/local-history/mean/dense/weak-force files and all **17** accepted
ledger rows match the uninterrupted same-rank control **bitwise**. Cross-rank
results agree within the reporting bound; they are not claimed bitwise.
Diagnostic particle IDs/positions remain fixed within each run and restart.
Fresh partition-dependent ID numbering is not used as a cross-rank accuracy test.

## Measured costs

Timing runs do not overlap. Whole-run wall time includes startup, checkpointing
and output; the loaded profile is the same 16 accepted steps (.12,.16], with
97 residual evaluations and 38 Newton linear solves in each case. Times are s.

| Ranks | Wall | Speedup | Loaded assembly | Reduction | Root linear/setup | History | Loaded solve total | Whole output |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 118.34 | 1.00 | 33.65 | .00038 | 2.79 | .77 | 43.34 | 15.37 |
| 2 | 67.24 | 1.76 | 19.25 | .36 | 3.19 | .56 | 24.37 | 12.82 |
| 4 | 48.57 | 2.44 | 10.80 | .58 | 3.59 | .41 | 16.19 | 11.34 |

Maximum per-rank resident peaks are 340/289/277 MiB; sums of per-rank peaks are
340/547/1018 MiB (not simultaneous node-memory measurements). The parallel
coupled sparsity graph omits unused-field couplings; the serial reference retains
its original graph. Both retain exactly the selected physical action. The table
reports actual implementation benefit, not pure factorization scaling. Rejection/
restart runs take 60.08/38.09/27.87 s, including their two extra solved trials.
The larger control already exposes root factorization as a principal bottleneck.
No small-case speedup is extrapolated into a full-event guarantee.

## One physical boundary control — completed

Fresh 96x128 mesh, Delta x=.015625 m, Delta y=.0078125 m; Delta t=.00125 s.
Same continuum seed, G=1e6 Pa, eta=1e8 Pa s, ell=.15625 m, Gc=26.6666666667,
Kf=64 Pa, m=128, R=1000 Pa; loading .018/s on (.05,.175], hold to .185.
Top/bottom retain relative horizontal rate .018 m/s and uy=0. Lateral uy=0,
ux free, with zero **horizontal** natural mechanical traction. The sides are
not free in their vertical component.

The normal-velocity constraints determine pressure nullspace treatment: retain
a gauge in the closed control, remove it when side ux is free. Mechanical p is
the Lagrange multiplier; fixed friction p_N=10000 Pa is not substituted for it
or imposed as a side traction. The initial adiabatic pressure is an algebraic
guess; the unloaded solve equilibrates it before loading. Free-side weak loads
are checked against the unchanged mechanical tolerance. Reaction work sums only
prescribed components; free-side displacement is excluded. Endpoint and
trapezoidal force-displacement work remain separate from the exact BE ledger.

The fresh 96x128 control reaches .185 s in **2244.41 s**, with 148 accepted
publications. The same sampled continuum initialization agrees with the affine
case exactly at every physical vertex; initial representation errors remain
.085286% whole-domain and .188486% in the cap. The 40 unloaded intervals have
exactly zero phase drift. Thus the response is mechanically driven under the
pre-existing-damage obstacle, not independent seed relaxation.

| Response | Retained affine | Horizontal sides released |
|---|---:|---:|
| First cap-growth bracket [s] | (.1275,.12875] | (.12625,.1275] |
| First remote-growth bracket, x>.9 [s] | (.15,.15125] | (.145,.14625] |
| First right-boundary bracket [s] | (.15125,.1525] | (.14875,.15] |
| Final total integrated increment [m2] | .00189394 | .00256628 |
| Increment within original rounded cap [m2] | .000136665 | .000197793 |
| Retained cap window, r<.35, increment [m2] | .000188675 | .000324660 |
| Cap-window peak increment | .00672790 | .0165773 |
| Remote x>.9 integrated increment [m2] | .00142516 | .00205869 |
| Increment outside continuum seed support | 87.30% | 92.29% |
| Increment at represented-zero initial phase | 86.17% | 91.33% |

The original rounded cap and the cap *diagnostic window* are distinguished:
the latter also contains an initially intact outer ring. Horizontal release
eliminates the old initial left-boundary/band widening, increases original-cap
growth, and markedly redirects the low-damage field toward the centerline.
It also triggers remote damage four timesteps earlier. These are differences
between boundary-value problems, not accuracy errors against the affine case.
Upper/lower flanks remain symmetric to roundoff; no mirror alignment is used.

**Connected weak damage is not a travelling high-phase tip.** In the released
case the .001 contour joins the remote/right-boundary patch at .155 s and the
.01 contour at .1675 s. Their centerline superlevel sets then span the domain;
reporting the resulting 0.70/0.72 m extent shifts as tip travel would be wrong.
Before joining, the last near-cap first-crossing advances are 20.54 mm (.001,
.15375 s) and 45.13 mm (.01,.16625 s). The .05/.1 centerline shifts at loading
stop are only 3.56/.023 mm; the .3/.5 contours do not move. The 45-degree flank
advances at .001/.01 are 10.81/3.39 mm, compared with 41.92/13.10 mm in the affine
run. Thus this control is more centerline-oriented and less dominated by radial
flank widening, but **80.22% of total new damage lies at x>.9**. A diffuse band
forms through bulk nucleation and merging, not isolated high-damage tip advance.

[Matched stress/damage/driving maps](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/boundary-comparison.png),
[common-kernel centerline/flank profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/boundary-profiles.png),
and [contour/reaction/energy histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/boundary-histories.png)
use identical scales and physical coordinates. The
[direct stress/phase/damage/rate animation](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/boundary-comparison.mp4)
starts at .15 s, the beginning of saved full affine histories. The
[entire released trajectory](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_parallel/released-full-trajectory.mp4)
covers initialization, loading and hold. The whole domain remains visible.
Rate is the native **bulk slip strain rate** in 1/s, displayed as positive-JxW
cell means, not a surface slip velocity. Original seed/endpoint and support are
marked; phi is solved throughout the domain. No segment is appended.

Plain-language reading: only the horizontal side prescription was released;
the mesh, seed, law and imposed relative grip motion are identical. Native
mechanics/history/phase equations calculate the fields. The new loading gives
a more fault-aligned weak-damage band, but much of it is born remotely and then
connects back. Images establish this discrete response, not continuum crack
speed, full domain independence or a qualified propagation event.

### Work, pressure and numerical checks

Only prescribed-component reactions contribute to external work. In the affine
case lateral actuators can do negative work, so its **total** reaction and its
grip reaction are shown separately. In the released case all external work is
through the grips. With height H=1 m, Q_gamma/H is the force per unit thickness
conjugate to relative grip displacement; numerical values equal Q_gamma, but
their units differ. Corners are assigned to grips in the nodal decomposition.
Independent endpoint reaction work agrees with the native ledger to
**6.94e-17 J/m**. The mechanical pressure guess starts near 10 kPa, equilibrates
to zero mean in the first unloaded solve, and later ranges approximately
+/-1322 Pa; this is distinct from fixed friction p_N=10 kPa. Every attempted
physical solve records open_normal=1, with no pressure gauge. Maximum scaled
free-horizontal-side weak residual is 4.03e-12, below the unchanged 1e-9 bound.

| Energy/work [J/m] | Affine | Released |
|---|---:|---:|
| Final bulk storage | 2.290845 | 1.793625 |
| Final cohesive storage | .140725 | .183174 |
| Added fracture storage | .110195 | .159873 |
| Total storage change | 2.541765 | 2.136672 |
| Physical dissipation | .840563 | 1.120372 |
| Endpoint work | 3.410774 | 3.282332 |
| Trapezoidal force-displacement work | 3.382249 | 3.256958 |
| Signed trapezoidal physical remainder | -7.85004e-5 | -8.57229e-5 |
| Sum absolute step remainders | 7.85065e-5 | 8.57279e-5 |
| C_E (sum absolute / absolute work) | .00232113% | .00263215% |

Released physical dissipation is Db=.00262907, Dc=.000091813 and friction=
1.117651 J/m. The exact BE numerical terms remain separate: stress loss
.02471708, traction loss .000518453, phase remainder -.000306413 and fracture
loss .000359819 J/m. Cumulative phase/mechanical/pressure residual works are
-5.59e-11 / 4.97e-11 / 1.31e-18 J/m. Maximum exact-ledger defect is 6.53e-14 J/m.
The small signed physical remainder is not hiding stepwise cancellation; its
absolute variation is almost identical. **C_T is unmeasured for the changed
boundary/extended-loading response**, so C_E alone is not physical-energy
qualification. Historical failed endpoint screens remain.

During the hold, phase and signed/absolute shear histories are exactly fixed,
and local slip rate is zero. Full physical stress matches homogeneous Maxwell
scaling beta_product=.9999000056247656 to 6.18e-13 relative L2; maximum difference
is 1.71e-7 Pa. Stopping grip motion does not remove stress. The relaxed virgin
q multiplier at h=0 is not assumed to obey the positive-h cohesive recurrence.

Native mechanics <=9.23e-10, KKT <=8.65e-8 and independent source <=6.74e-15
pass all unchanged tolerances. The open-boundary consistent tangent action has
FD discrepancy 2.70e-9 (existing 1e-6 screen). Independent dense checks reproduce
physical full-tensor storage, Ga=hq, the virgin multiplier, friction interval,
raw mechanical/fracture nodal forces, and cumulative friction from stored L.
A final four-rank checkpoint restores without another physical step. This is
not a new large-mesh evolving-restart comparison; that lifecycle is tested by
the 1/2/4-rank growing controls above.

The full control uses 510 residual evaluations and 176 root linear solves:
assembly 978.85 s, collective reduction 76.36 s, linear/setup 886.43 s,
history 41.15 s, coupled total 2074.36 s and observer output 135.14 s. Maximum
rank RSS is 1329 MiB; sum of rank peaks is 3849 MiB. Thus root factorization
and assembly are the remaining principal costs. Comparing this wall time to
the differently loaded affine case is not a clean speedup measurement.

The largest nodal step increment is .00109145 in (.15,.15125] s; it and the
merging intervals remain candidates for a later temporal check, not validated
growth speeds. Native checkpoints at .11875,.13875,.15875,.17875,.185 s are
preserved. Numerical runs use the pinned executable 38336e67b35b; the final
rebuild changes only parameter help text and reproduces a growing step bitwise.

**Conclusion:** bounded fixed-mesh bulk MPI and the selected lifecycle pass.
Horizontal-side loading is a substantial cause of the earlier angular response,
but releasing it does not isolate propagation from bulk nucleation. Retain this
control and its more centerline-oriented loading as evidence; verify the flagged
new-growth interval and the unresolved spatial response before interpreting
connected contours as physical tip motion. No further run starts automatically.

## Preserved limitations

The original energy-screen failures, the failed 5% spatial target (35.86%
growth-relative discrepancy between 48x64 and 96x128), branch/orientation
findings and strict_reference_match_passed=false remain. Extended-loading
C_T is not measured. Phase KKT and ledger closure alone do not establish
physical stability, spatial accuracy or propagation speed. This qualifies only
fixed-mesh homogeneous 2-D bulk MPI with stationary history and the tested
boundary masks; no MPI hanging evolution, AMR, transport, geometry change, RSF
or moving surface is qualified. Same-rank restart is tested; cross-rank restart
is not claimed.

Particle/H-property infrastructure remains in generic setup, fixed-point
advection/output and native particle checkpoints. Domain/CPDI construction is
disabled. No particle stress, domain or H value enters the bulk coupled residual,
material coefficients, history source or physical obstacle. Particles are not
removed. All 34 protected BP3 file identities pass; mature BP3 and defaults are
not changed or newly qualified by this bulk experiment.
