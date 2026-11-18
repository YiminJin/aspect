# Single-turn RSF closure: bounded constitutive comparison

11 November 2026. **The fixed-direction control is closed by user agreement.
A curved material-section reference passes its straight, rotation, reversal,
work and history checks, but exposes a constitutive choice before native free
turning can be enabled.** Two work-adjoint slip definitions coincide on a
straight fault yet differ at a bend by 0.952% in stress, 1.398% in age and 1.487%
in added signed slip. The difference survives timestep halving.

[Formulation and equation-to-code links](SINGLE_TURN_RSF_FORMULATION.md),
[evidence and executable inputs](../../../benchmarks/reconstructed_fault/particle_free/evidence/single_turn/),
[comparison figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/single_turn/curved-closure-comparison.png).

## What was executed

Root `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, revision
`b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`, with all incoming manual work
preserved. Entry patch/status/hashes are recorded. Protected executable SHA256
`034d2a848a3a10c496fa097475f31497c00bc68c3456fb6d87c1ee700f453c27` and checker
`b800ef431a2469760efd16befcba94e8cd743156629b55e4c921f769255d9d84` are unchanged.
The independent C++ oracle was built separately with one worker from the current
unchanged `bulk_tip.cc`; its SHA256 is
`235ad4327f7a9a6502433a10abf87da53ca39f299d9b0d1afd7576cbc704a0dc`.
No native simulator or MPI run, core rebuild, checkpoint conversion or commit.

The Python reference couples full-tensor native-form point responses through
section RSF roots under prescribed affine strain and fixed phase. It does
**not** solve momentum, pressure or the phase VI. Six paths compare straight,
rigidly rotated (.37 rad), and curved prescribed charts under both closure
choices. Two further paths halve time only on the curved pair. Reference
discard/serialization/retry is independent-array evidence, not native restart
qualification. The physical law comparison uses the same incoming state.

| Fixed input | Value |
|---|---|
| Material/RSF | Same G=1e6, eta=1e8, m=4, pN=200, mu0=.6, a=.015, b=.02, V0=.001, Dc=.0005, radiation=1000 as peak-.60 control |
| Material chart | s∈[−.4,.4] m; n∈[−.125,.125] m; κ=0 or .8/m; J=1−κn∈(.9,1.1) |
| Quadrature | Composite Q6: 4 axial × 8 normal cells, 24 sections × 48 points; 1,152 full tensor histories |
| Prescribed phase | .6 max(0,1−abs(n−.025)/.1)²; asymmetry exposes curvature's first moment; Δφ=0 throughout |
| Incoming age/slip | θ=.4+.3(s+.4)/.8 s; δ=.0001 sin(3s) m |
| Incoming tensors | τ=700S+rotated diag(100,−150,50) Pa; A=.0002(φ/.6)(1+.3n/.125)S |
| Loading | Affine divergence-free engineering shear .04/s to .05 s, then −.06/s to .1 s; rotated with chart for covariance test |
| Time | Midpoint .00125 s; curved repeat .000625 s; exact constant-stage-rate aging |

These deliberately loaded initial tensors are not an equilibrated specimen or
an RSF-consistent accepted simulator endpoint. Their initial relaxation pulse
is part of this material-reference fixture, not evidence of unstable sliding.
All prescribed geometry/phase is labelled as such in the figure. There is no
new contour advance, added fracture energy, or resolved turn-onset window.

## Results and checks

The two laws are V_area=∫Jw dn, f_loc=f and V_normal=∫w dn, f_loc=f/J.
The same scalar pN law and full bulk tensor updates are used. Relative norms
below use the finer material-area solution; slip is normalized by its change
from the common initial signed slip. Raw tensor norms use physical area; age
and slip use reference ds. No per-section or local-rate clipping is applied.

| Endpoint difference at .1 s | Between laws, dt=.000625 | Each law's .00125→.000625 difference |
|---|---:|---:|
| Stress L2 (Pa m) | 3.487853 (0.9518%) | .0001273 / .0001445 |
| Cohesive A L2 (m) | 1.743926e-6 | 6.36352e-11 / 7.22376e-11 |
| Age L2 (s sqrt(m)) | .003762127 (1.3975%) | 2.75661e-5 / 3.02373e-5 |
| Signed-slip L2 (m sqrt(m)) | 6.145376e-7 (1.4875% of added slip) | 1.21780e-10 / 1.25842e-10 |

This is a constitutive discrepancy, not an accuracy ranking. The two laws give
bitwise-identical straight and rotated-straight trajectories. A symmetric
band could obscure the first-moment difference; the asymmetric prescribed band
is an intentional sensitivity test, not a proposed growth specimen.

| Independent check | Measured maximum / outcome |
|---|---|
| Python against unchanged C++ point law, 270 samples | Stress 4.55e-13 Pa; A amplitude 1.08e-18; w 3.47e-16/s; Y 5.33e-15 Pa |
| Rigid coordinate rotation, full trajectory | Stress L2 1.85e-12 Pa m; age L2 1.28e-16 s sqrt(m) |
| Reflected tensor chart; signed-frame reversal; record permutation | Exact equality in checked next-state arrays |
| Fixed-chart bulk/fault friction-power difference | 8.88e-16 W/m |
| Overlapping maps on Cartesian native Q6 weights | Power defect 6.94e-17 W/m; wrong unweighted transpose gives −.08605 W/m |
| Weighted condensed mobility | Symmetry defect 4.34e-19; minimum eigenvalue 3.02e-5 |
| Fixed-phase midpoint physical-work balance | Max step error 1.55e-16 J/m; max accumulated absolute 3.08e-15 J/m |
| Virgin prospective age at .08 s | .58 s from .5; error 1.11e-16 s; V and prior slip zero |
| Independent discard/restore/retry | Bitwise equal; accepted arrays untouched |
| Virgin plane selection vs continuous-angle scan | 0°, +12°, −12°; tied conjugates retained and explicit preference tie unresolved |

The signed cumulative and accumulated absolute work errors are both exported;
no large cancellation hides in the totals. For these **fixed-phase prescribed
strain** paths midpoint exactly accounts quadratic storage up to roundoff. This
does not qualify physical-work accuracy under changing phase or loading boundary
reactions. Coulomb pressure derivatives, finite rotation and native MPI are
outside these checks.

During reversal all 24 sections can contain opposite-sign local fault rates,
with aggregate negative local power reaching −.11464 W/m while section-integrated
fV remains nonnegative. The diagnostic finds at most one connected active
component per section (abs(w)>1e-10 times that state's maximum rate); this
threshold only counts components and never modifies a rate or integral.
Thus a single band can already exhibit cancellation and local negative power.
The reference retains the full bulk cohesive energy and exports the positive
energy exceeding a one-moment section reduction. A deliberately forbidden
projection of loaded A onto a new tangent would delete 6.13531e-5 J/m.
Averaging two ages .2/.8 to .5 s also changes common-speed resistance by
.88977 Pa. Work-adjoint mapping alone does not authorize either operation.

The first independent implementation used an absolute-rate Brent tolerance,
which could not resolve cold creep: its integrated-rate gate failed although
its point work ledger closed. That evidence is retained under
`failed_absolute_rate_root/`. Switching the **reference** to the same asinh
rate coordinate already used by native `FaultFriction` resolves it, with no
relaxed gate or native change. Final rate, oracle and work checks pass.
Eight short reference paths and checks take about 2.5 s, single numerical worker;
build and plotting cost is separate. There was no overnight/long campaign.

## Decision and next experiment

The formulation note gives a candidate **virgin intact-reaction Y0 maximum**,
retained established planes, explicit ties, prospective age and a weighted work
adjoint. It also identifies why the old finite-phase maximum-speed event cannot
select a plane at exact intactness: w=0 for every candidate. A shear plane and
an evolving absolute phase contour remain distinct observations.

The unresolved scientific choice is the material slip carrier at a bend and its
RSF rate measure. I recommend retaining material-area/reference-length as the
first candidate to review because it preserves uniform bulk resistance, with
normal-integral rate with f/J resistance as the explicit alternative. This is a
constitutive proposal, not a selection justified by the native equations.
It must include a coherent, nonoverlapping virgin-section extension rule while
old material retains membership; a contour alone cannot supply that rule.

Accordingly the conditional native free-phase turn/reflected pair was **not
launched**. The task's allowed minimal two-law comparison exposes the choice
without silently embedding it in a new simulation. The next justified experiment,
after that carrier is selected, is one small opt-in native virgin-cap admission
and reflected off-axis loading pair, preserving accepted planes/age, bulk tensors,
and the phase VI. Track original and current absolute contours, local plane,
width, old/new section state, work and spatial distinguishability separately.
Do not declare a turn from a plane-angle plot. Unstable sliding, contact physics,
branching and another straight refinement campaign remain deferred.

Reproduce the small calculation with the commands in
[`single_turn/README.md`](../../../benchmarks/reconstructed_fault/particle_free/single_turn/README.md).
Current source implementation remains the straight native control. Curved
closure is independently verified reference code only; no coupled turning
qualification or production readiness is claimed. Previous failures, checkpoints,
and cleanup records/data-loss statements are preserved.
