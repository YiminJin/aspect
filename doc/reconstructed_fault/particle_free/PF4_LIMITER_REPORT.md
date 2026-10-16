# Bounded PF4 tensor limiter comparison — 16 October 2026 UTC

**Comparison complete. Evidence favors neither limiter for adoption.** Both remove
most sampled jump overshoot but approximately double full-tensor L2 error,
quadruple front width, and increase quadratic-norm loss eightfold at32×64.
KXRCF is cheaper and disturbs the resolved smooth pulse less; the safeguarded
Fu–Shu adaptation detects an algebraic blind spot but provides no consistent
transport advantage here. Keep production/default behavior unlimited and both
private paths experimental. No coupled deployment, PF5 or propagation is selected.

Baseline: worktree `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`,
HEAD `3232056cde12dbae17ef4265ecb804f9098b88f1` plus incoming PF2–PF4 changes.
Root AGENTS.md applies; no nested override. The user's
[handoff](PF4_LIMITER_HANDOFF.md) specifies this experiment. Existing physical laws,
inputs, defaults and PF0–PF4 evidence are preserved. Native/compiler/dependencies
remain GCC12.4, deal.II9.6.2, Trilinos14.2, OpenMPI5.0.10, Release, one rank/thread.
All runs use the same new core, SHA256
`a54c7f7b2e47` (full identity and inputs in the [provenance](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/provenance.json)).

## Exact experiment

For stored `A=(xx,yy,xy)`, reconstruct `zz=-(xx+yy)`, with xz=yz=0, and use
`|A|₃²=xx²+yy²+(xx+yy)²+2xy²`. All differences, derivatives and weights use this
metric. `<·>K` denotes a physical target-cell average; `EJ AJ` is the **unmodified**
neighbor polynomial extrapolated over K, without clamping reference coordinates.

```
M = max_{J in K and genuine face neighbors} sqrt(<|AJ|₃²>J)
Imean = sum_neighbors |<AK>K - <EJ AJ>K|₃ / M
Ipoly = max_neighbors sqrt(<|AK - EJ AJ|₃²>K) / M
Fu–Shu experimental decision: max(Imean,Ipoly) > 0.1

KXRCF experimental decision:
| integral_{inflow(K)} (AK-Aup) ds |₃ /
[ (diam(K)/sqrt(2 m²))^(3/2) * |inflow(K)| * M ] > 1
```

The reference length is the actual unit-Box diagonal, sqrt(2)m, calculated from
mesh vertices. KXRCF takes the norm **after** signed tensor integration, with p=2.
Native pointwise normal FE velocity selects Gauss4 inflow samples; zero speeds
contribute nothing. Boundary upwind data are the same explicit prescribed PF4
inflow; outflow uses the interior. A zero interior patch with nonzero boundary
inflow has an unbounded KXRCF relative jump; it is flagged without a stress floor.
Zero patch and zero inflow give zero indicators and finite linear weights.

Fu–Shu's RMS denominator and the complete-polynomial safeguard are **new
adaptations for this experiment**, not the published scalar detector. Both
indicators are computed **before mean shifting**. Original own-cell RMS values
supply M. The mean-only indicator and safeguard-only counts are retained.
See [Fu–Shu, equations2.3–2.5 and Remarks2.2/3.1](https://archive.ymsc.tsinghua.edu.cn/pacm_download/113/9050-paper-revsub.pdf).

For either detector the reconstruction is identical:

```
P0 = AK; PJ = EJ AJ - <EJ AJ>K + <AK>K
B_j = < |P_xi|₃² + |P_eta|₃² + |P_xixi|₃²
         + 2|P_xieta|₃² + |P_etaeta|₃² >K / M²
alpha_j = d_j * ((min(B)+1e-6)/(B_j+1e-6))²
omega_j = alpha_j / sum(alpha)
A_limited = sum_j omega_j Pj
```

Here xi,eta are target reference coordinates on[0,1]². This first/second
reference-derivative sum is a **Q2 adaptation**, not the full total-degree-four
Jiang–Shu sum. Every nonconstant Q2 mode is represented. A common preliminary
nodal amplitude scaling avoids squaring huge/tiny physical data. If every B is
zero, normalized linear weights are returned directly; otherwise the minimum
ratio above keeps unnormalized weights bounded. Epsilon is dimensionless,
without a physical stress floor. Interior d0=.996,dJ=.001. Missing physical-boundary
neighbors return their .001 weight to the center (.997 edge,.998 corner).
No exterior reconstruction polynomial is invented.

Candidates use all9 independent DGQ2 coefficients, including xy,x²y,xy²,x²y².
Direct nodal polynomial combination replaces the reference routine's equivalent
mass projection. Gauss3 per coordinate integrates means, squared Q2, mass and
smoothness exactly on the present affine Cartesian cells. Actual Gauss3 mass
rank is9; Gauss2 rank is4. Gauss3/5 mass discrepancy is below1e-14.
Existing PF4 mass/transport/source-inflow rules remain Gauss3/Gauss4/analytic
projection Gauss8. No adaptive quadrature is added.

Each SSPRK3 candidate is limited at t+dt,t+dt/2,t+dt, before its next use.
All cells read one immutable stage and write separately. There is **no initial
limiting pass**. Null limiter remains the original behavior; no parameter/default
selector or accepted-state publication is introduced.

## Source reuse and bounded selection

Primary reference: `../aspect-weno/source/simulator/limiters.cc`, SHA256
`348ac0ad13904cf0c9bd6a25fd225b71bd90cef0be3839cc77d17f781c5a69fc`.
Its target-cell extrapolation, target-cell mean restoration, compact convex
combination and separate writes are retained, following
[Zhong–Shu's whole-polynomial reconstruction and rectangular extension](https://users.math.msu.edu/users/zhongxh/paper/wenolimiter.pdf).
The source's configured reconstruction rule is degree+1
(`source/simulator/introspection.cc`), hence Gauss3 for degree2. A deficient
Gauss2 reconstruction mass is **not** a finding about this implementation.
Corrections/adaptations: zero-smoothness weights; common full-tensor decision and
weights; target-reference Q2 derivatives; RMS normalization; pointwise inflow and
prescribed boundaries; complete Q2 coefficient retention. The reference's Gauss2
KXRCF sampling, whole-face velocity sign, scalar weights, skipped boundary faces,
coarse/fine and face-once MPI machinery are not imported. No `elasticity.cc` port.

Thresholds .05/.1/.2 were screened on saved **unlimited** initial/final smooth and
jump snapshots at4×8 and32×64. The initial73-check calibration passed before any
limited transport validation. The timestamped [selection](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/threshold-selection.json)
fixes C=.1. This retains the proposed middle value; .05 increases smooth selection,
while .1/.2 detect the same fine jump fraction. No per-mesh/case retuning.

| Saved32×64 snapshot | C=.05 flagged | C=.1 flagged | C=.2 flagged |
|---|---:|---:|---:|
| Smooth initial |54.49%|42.19%|34.67%|
| Smooth final |57.28%|45.26%|31.25%|
| Jump initial |18.75%|18.75%|18.75%|
| Jump final |65.63%|59.38%|59.38%|

The coarse4×8 fields flag100% at all three values. This amplitude-based detector
flags weak Gaussian tails and is not insensitive to a constant tensor background.
KXRCF uses1 throughout; no threshold search was undertaken.

## Local and compatibility results

**77/77 final local/snapshot gates and66/66 transport gates pass.** Independent
NumPy physical-coordinate Lagrange evaluation checks native indicators, weights,
all9 reconstruction coefficients and physical cell means. Checks cover zero,
nonzero/near constants, forced piecewise constants, affine and complete global
Q2, smooth zero crossing, non-proportional tensors, rotating tensor direction,
amplitudes1e-250/1e250, reversed cells/stencils, boundary/no-inflow and a face with
both inflow/outflow. A supplementary native local run adds explicit covariance
of both detector values/masks under an in-plane tensor-basis rotation; no physical
rotation/spin law is introduced. No arbitrary3D-rotation claim.

| Property | Measured result |
|---|---|
| Global affine / complete Q2 forced change |≤2.28e-16 relative; both indicators≤7.38e-16 |
| Independent full reconstruction / weights |≤1.65e-15 /1.34e-15 scaled |
| Independent physical mean change |≤2.55e-16 scaled across local fixtures |
| Cell/stencil order / in-plane reconstruction covariance |≤3.13e-16 /2.11e-15 relative |
| Finite nonnegative weights / sum defect |Pass /≤2.23e-16 |
| Embedded symmetry/trace and both xy terms |Pass |
| Mixed-xy mean-only / polynomial indicator |≤9.00e-16 /0.287678 |
| Mixed-xy forced reconstruction change |0.03749%; **detection is not removal** |
| Background diagnostic, max Ipoly |3.1616→4.67e-8 after adding basis(1e5); detection suppressed |
| Limited repeated trials / source failure after first stage |Exact repeat; rejection leaves incoming/accepted state unchanged |

The mixed-xy signed KXRCF integral also cancels on interior cells; nonzero physical
boundary jumps are separate. The orientation fixture has constant tensor magnitude
at interpolation nodes but Ipoly=1.4886: tensor differences are used, not differences
of scalar magnitudes. The coarse smooth zero-crossing fixture is flagged rather
than divided by a signed/zero mean. No extra non-proportional transport family
was needed after local common-weight/basis checks; general tensor transport remains
limited to this evidence.

Fresh original PF4 operator/source/publication control has byte-identical matrix,
loads, lifecycle, accepted fields **and all metrics**. Independent matrix/correction/
inflow errors≤1.56e-15; polynomial derivative4.33e-15. All four new unlimited jump
fields are byte-identical to PF4; smooth fields differ by≤4.56e-16 in scaled
coefficients (≤2.23e-14Pa m in full L2, harmless observer time-expression rounding).
Native cohesive/Maxwell628 assertions in8 cases and PF1's94 checks pass. Both
production translation units compile independently without unity/PCH. PF2/PF3
reports/evidence are preserved; their full native regressions were not repeated
for this private limiter comparison.

## Transport tradeoff

Same physical initial projection, native velocities, T=.2s, SSPRK3, source/inflow
and sampled CFL≤.025 for all three paths. Smooth velocity(.4,.2), signed jump(.4,0).
Time steps are .2/(26,52,103,205) for the smooth pulse and
.2/(13,26,52,103) for the jump. No limiter-specific time refinement or temporal
order claim. Extrema use PF4's Gauss-Lobatto9×9 cell samples. L2/L1 use full embedded
tensor errors; jump quadrature is split at exact translated fronts before
Gauss24/48 integration (L2 differences≤1.09e-12Pa m).

Raw initial projections are recorded separately, without limiting:

| Grid | Smooth initial xx range, Pa | Jump initial xx range, Pa |
|---|---:|---:|
|4×8|[-47.541,1013.958]|[-1000,1000]|
|8×16|[-.04371,991.811]|[-1000,1000]|
|16×32|[-2.91e-10,999.504]|[-1000,1000]|
|32×64|[1.99e-14,999.607]|[-1000,1000]|

Jump projection errors in the listed extrema are≤1.6e-12Pa. Subsequent jump
oscillation and norm changes therefore come from evolution/limiting, not an
initial Gibbs projection at these cell-aligned jumps.

| Grid/path | Relative tensor L2 | Overshoot / undershoot | Quadratic-norm loss |
|---|---:|---:|---:|
|4 unlimited|.38165|52.916% /29.355%|12.826%|
|4 KXRCF|.67873|0 /0|53.071%|
|4 Fu–Shu+safeguard|.67877|0 /0|53.057%|
|8 unlimited|.32255|24.411% /29.211%|10.408%|
|8 KXRCF|.47411|0 /4.429%|40.139%|
|8 Fu–Shu+safeguard|.47492|0 /3.912%|40.229%|
|16 unlimited|.21813|14.152% /28.331%|5.940%|
|16 KXRCF|.47179|.0487% /.8956%|43.010%|
|16 Fu–Shu+safeguard|.47144|.0477% /.0372%|43.040%|
|32 unlimited|.17755|16.503% /15.822%|3.574%|
|32 KXRCF|.37629|.0161% /.0241%|28.948%|
|32 Fu–Shu+safeguard|.37658|.0488% /.0223%|28.903%|

Fine-grid full-tensor L1 errors are67.331/227.178/227.125Pa m², respectively;
Gauss24/48 uncertainties .00542/.0000275/.0000181Pa m². Fine front widths are
.02353/.09967/.09686m at the first front; the other two are
.02353/.09985/.09671 and .02353/.09695/.09662m. Width is the total10–90% transition
length in a fixed±.115m window around each exact front, sampled1024 midpoints per
cell; width resolution is h/1024 (coarse windows can truncate broad fronts).
These are measured smearing indicators, not a mechanical admissibility criterion.

| Smooth relative tensor L2 |4×8|8×16|16×32|32×64|Finest rate|
|---|---:|---:|---:|---:|---:|
|Unlimited|.075205|.018173|.0020976|.00026275|2.997|
|KXRCF|.075205|.018168|.0026892|.00026515|3.342|
|Fu–Shu+safeguard|.29254|.086065|.0074830|.00029650|4.658|

All final-three errors decrease and finest rates exceed the retained2.5 gate.
High apparent limiter rates reflect changing activation, not proof of fourth-order
accuracy. Fine smooth errors increase .91%/12.84% over unlimited; coarse Fu–Shu
limiting is much more intrusive. Full L1/norm/extrema results for every case are
in the [assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/assessment.json).

The stage limiter's maximum cell-mean error is1.07e-15 in the common nodal
amplitude scale, separately from transport. Independent local **physical** mean
comparisons also pass. Fine jump net domain xx integrals change from approximately0
to .000554/.000850/.000850Pa m². In this divergence-free, source-free test the
weak operator leaves boundary advection as the net global-mean term, up to these
limiter roundoff errors; differing numerical outflow tails are allowed. This is
not a periodic-domain exact conservation claim. Quadratic-norm loss is not a
full mechanical energy balance.

Saved-snapshot reconstruction diagnosis explains the similar detector outcomes:
one Fu–Shu pass on the unlimited fine jump reduces selected subcell variance to
.97740 of its incoming value;21.05% of selected cells give a neighbor weight>.5,
with median center weight .97539. Repeating reconstruction through309 jump
stage candidates damps modes while preserving means. Both limiters yield nearly
the same strongly broadened transport result despite different masks. The mixed-xy
example demonstrates the opposite limitation: detection with very weak mode
removal. This implicates reconstruction/resolution tradeoffs; no lower-C search
or changed weights are justified by this comparison.

## Costs, ownership, reproduction and stop

Instrumented advancement costs on32×64 (seconds; setup/export excluded):

| Case/path | Limited stage-cell visits | Safeguard-only | Detector | Reconstruction | Total advance |
|---|---:|---:|---:|---:|---:|
|Smooth unlimited|0|0|0|0|.848|
|Smooth KXRCF|28.44%|—|2.638|1.085|4.832|
|Smooth Fu–Shu+safeguard|41.32%|.703%|3.052|1.577|5.740|
|Jump unlimited|0|0|0|0|.405|
|Jump KXRCF|48.10%|—|1.063|.913|2.513|
|Jump Fu–Shu+safeguard|78.87%|3.995%|2.115|2.206|5.089|

KXRCF does not compute Fu–Shu diagnostic indicators in its timed branch. Detector
cost includes common patch scaling/extrapolation; allocation/copy overhead also
appears in total. These short instrumented serial times are not general parallel
cost estimates. Whole validation runs4/8/16/32 take4.024/4.522/10.983/42.143s,
peak child RSS210772/222088/258676/449516KiB; all below180s. One scheduling
exception briefly overlapped the short operator control with the16/32 pair;
no timeout or memory failure. No wider campaign followed.

`dg_limiter.cc` owns static stencils, common detector/weights and scratch
reconstruction. `dg_history_internal.h` shares only this private comparison data
and statistics. `dg_transport.cc` adds an optional stage hook; accepted ownership,
Maxwell source, constitutive law and publication stay unchanged. The observer
and independent checker are `limiter_checks.cc` and `assess_limiter.py`.
Matching design/spec addenda describe this user-selected extension.

Remaining particle dependencies are unchanged: no particle reads in advancement
or limiting, but operator startup validates fixed particles; initialization H/CPDI/
domains, support/association, surface material projection and derived sampling
still require particle infrastructure. Native surface-history/phase-bound
transport, acceptance/rollback publication, restart, moving geometry, birth,
AMR/MPI/periodic/coupled transport remain guarded or unimplemented. No chi limiting,
component clipping, new stress bounds, spin or physical-law changes.

Reproduce with an unused run label, existing `transport_{4,8,16,32}.prm`,
`run_fixed_profile.py`, plugin `liblimiter_checks.release.so`, postprocessor
`PF4 limiter checks`, and180s limit. For local/snapshot checks use
`limiter_calibration.prm` and `limiter_calibration_32.prm`; run
`assess_limiter.py --calibration`. The production build command is unchanged.
The [portable evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/README.md)
contains exact commands, threshold timing, all runs' identities, independent
assessments, replayable CSVs, source preservation and a [measured comparison figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf4_limiter/comparison.png).

**Recommendation:** retain unlimited DGQ2 for the qualified resolved smooth
controls. Neither tested limiter earns adoption for nonsmooth histories on these
meshes. Keep KXRCF as the cheaper comparison and the safeguarded Fu–Shu experiment
as a diagnostic for complete-polynomial blind spots; its extra detections do not
establish better transport. Mechanical overshoot tolerance is unestablished, and
neither result qualifies coupled use. Stop here; no BP3, PF5 or propagation run.
