# Overnight free-turn validation

The bounded comparisons support **qualitative load-steered, connected extension
through 0.0775 s** in the selected synthetic material-strip model. Holding all
planes horizontal removes the lateral steering under identical loading. Normal
refinement leaves the common phase contour close, but changes when some planes
first register at tiny positive phase; those planes subsequently carry slip.
Quantitative advance, onset and structural directions are not established as
converged. The smaller-timestep and finer-mesh 0.080 s endpoints remain missing.

## Scope, identity and budget

The worktree remains `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`,
HEAD `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`, with its incoming manual changes
preserved. No C++/CMake change or rebuild was made. All physical runs use the
protected first-free-turn executable, SHA256
`089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`, and checker
`81efcd6e5e0013f4124bf1e1ba857005a2a23447985f69b16464fbc4d3131d69`.
The executable is identified independently of HEAD. The entry patch, protected
binary identities, unchanged original parent/failure hashes, incoming source
comparison and new checkpoints are in
[completed-preservation.json](../../../benchmarks/reconstructed_fault/particle_free/evidence/overnight_free_turn/completed-preservation.json).

One budget starts conservatively at **06:57 UTC, 12 November 2026** and ends at
14:57 UTC. The simulation cutoff is 14:12 UTC, reserving 45 minutes for closeout.
Native numerical work closed by **14:10:35 UTC**; the host process check confirms
no overnight native solver remains. Setup, failed attempts, compression and
postprocessing all count against the same budget; it was never reset. Actual
final elapsed time is recorded in `budget.json` and `final-artifacts.json`.

Each simulation used four bound physical cores, one numerical thread per rank;
only one simulation ran at a time. Heavy analysis/compression followed MPI exit.
The six process segments below form three selected physical comparisons:

| Segment | Elapsed minutes, including wrapper/archive overhead | Last accepted time | Outcome |
|---|---:|---:|---|
| A: four-quarter-step attempt | 90.41 | 0.0775 | Wall limit interrupts pending 0.07875 |
| B: fresh horizontal prefix | 36.94 | 0.0750 | Complete |
| B: own-parent half-step tail | 24.09 | 0.0800 | Complete |
| C: fresh fine prefix (`c-pilot`) | 133.97 | 0.0700 | Planned checkpoint handoff |
| C: unchanged-dt restart (`c-prefix`) | 41.74 | 0.0750 | Complete |
| C: own-parent half-step tail | 79.53 | 0.0775 | Wall limit interrupts pending 0.080 |

C's planned signal exit 143 is **not a convergence failure**. Its complete
accepted 14/0.070 checkpoint was verified before stopping only its four native
ranks. This avoided losing a long 0.075 solve to the pilot's earlier allocation.
An ordinary same-mesh, same-law, unchanged-0.005 restart reproduced the initial
residuals to printed precision. The common deadline stayed fixed. Fine accepted
steps 15 and 16 cost 2470.65 and 2651.98 s, versus 1099.89 and 692.243 s in the
retained four-rank coarse controls: factors 2.25 and 3.83, with different numbers
of nonlinear/search iterations. These are measured interval costs, not a general
scaling law.

The physical inputs retain G=1e6 Pa, eta=1e8 Pa s, Gc=2/3 J/m², ell=0.125 m,
peak-0.60 equilibrated seed, 8×4 m body, initial age 0.5 s, Dc=0.0005 m,
prescribed confinement 200 Pa and radiation coefficient 1000 Pa s/m. Native DGQ2
Bernstein phase, midpoint, six-point native tensor-product quadrature, mechanics
1e-9, phase KKT 1e-7, original iteration cap/search and energy checks are retained.
This is the **fixed-reference bulk-tip closure**, with physical true tau/A and
one RSF age per section. It is separate from mature moving-fault production,
transport/spin qualification and a general turning/branching theory. Newly formed
planes freeze their directions; phase contours and structural directions are
separate observables. See [the formulation](FIRST_FREE_TURN_FORMULATION.md).

## Common-time results

Offsets below are relative to the physical initial endpoint (6,2), not the seed's
initial absolute phi=0.05 contour. Added fracture is relative to each own t=0 seed.
W is the central excess-compliance FWHM; h_n is the corresponding grid spacing.
All reported fronts have an independently verified superlevel connection to the
original seed. The original recovered positive/reflected 0.080 pair remains the
reflection evidence; no new reflected refinement was run.

| Comparison | Accepted time (s) | Path | Axial offset (m) | Lateral offset (m) | Added fracture (J/m) | W/h_n |
|---|---:|---|---:|---:|---:|---:|
| A | .0775 | 128², dt=.0025 | .118129 | .028665 | .046206 | 1.9983 |
| A | .0775 | 128², dt=.00125 | .104011 | .027081 | .046845 | 1.9925 |
| B | .0800 | Free planes, 128² | .243096 | .048261 | .078252 | 2.1014 |
| B | .0800 | Horizontal planes, 128² | .117770 | .000000 | .060633 | 2.0926 |
| C | .0775 | Free planes, 128² | .118129 | .028665 | .046206 | 1.9983 |
| C | .0775 | Free planes, 128×256 | .118773 | .027804 | .046242 | 3.9957 |

The [full CSV table](../../../benchmarks/reconstructed_fault/particle_free/evidence/overnight_free_turn/comparison-table.csv)
includes cell units, stage times, error-reference intervals, cumulative signed
and absolute work errors, slip and cancellation. The figure has equal physical
axes, identical view limits and local mesh lines; each panel states its actual
common time. Dense observations do not turn subcell advance into resolved growth.

![Native absolute phase contours](../../../benchmarks/reconstructed_fault/particle_free/evidence/overnight_free_turn/contour-comparisons.png)

[PDF overlay](../../../benchmarks/reconstructed_fault/particle_free/evidence/overnight_free_turn/contour-comparisons.pdf)
and [centerline/width figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/overnight_free_turn/centerline-width-comparisons.png)
are available alongside the machine-readable A/B/C comparisons.

### A: late temporal sensitivity, partial through 0.0775

Both paths copy the exact original positive accepted 15/0.075 parent, slot 01,
and inherit the earlier 0.005 history. The quarter-step path accepts 0.07625 and
0.0775, then its 90-minute allocation interrupts 0.07875. Its last logged phase
KKT is 2.91235e-7, mechanics 6.22589e-16. This is a wall-time interruption, not an
80-iteration failure. No unaccepted field is used; the quarter-step 0.080 result
is unavailable. This tests a late window, not full-history temporal convergence.

At 0.0775, the phase L2 difference is 0.00196775 m: **14.548% of the increment
since the common 0.075 parent**, versus only 0.5303% of the entire seeded field.
Halving dt reduces axial advance by 14.118 mm (0.2259 h_x) and foremost-point
lateral offset by 1.584 mm (0.05069 h_n). Connected advance and upward steering
both survive.

The band itself steers, beyond the foremost lobe. At x=6.03125 and 6.09375 m,
quarter/half contour midpoints are 2.01763763/2.01790401 and
2.02543802/2.02581246 m. Their differences are only -0.266/-0.374 mm, while both
remain 17–26 mm above y=2. Widths change .145379→.140125 and .084410→.056978 m.
The front/width is substantially more time-sensitive than those midpoints.
Additional fixed 1/2 mm cuts near x=6 are retained in the CSVs, without implying
that their subcell features are spatially resolved.

Fracture added during the same .075–.0775 window changes .01894199→.01958134 J/m
(+3.38%). Signed and accumulated-absolute physical-work errors over that window
change 4.01408e-6→2.87050e-6 J/m. Small work errors do not eliminate the phase
increment discrepancy. Maximum endpoint age/slip differences are .00503647 s
and 4.45417e-6 m. Mean rates from signed-slip increments over the common .0025
window differ by at most .00178167 m/s. Terminal-stage times differ
(.07625/.076875), so their instantaneous rates and cancellation snapshots are
not compared as simultaneous data. In particular, zero mixed sections in the
quarter export versus 64 in the half export does not establish elimination of
cancellation by refinement.

### B: the plane/metric closure contributes decisively

B starts fresh with horizontal planes and the **entire positive off-axis boundary
velocity**, not the earlier straight loading. It follows .005 through .075,
then two .0025 intervals. No material-strip checkpoint is reinterpreted.
At zero angle, ds=dx and dn=JxW/dx: membership, rate integration, uniform
work-adjoint resistance and initial tau/A/age reduce to the horizontal law.
Angle and its arclength factor are linked parts of the selected closure.

The extra traceless affine diagonal rate has zero contraction with horizontal
S. For constant G/eta, its stress is a homogeneous Maxwell background with zero
divergence; horizontal A and phase driving decouple. Confinement stays prescribed
independently of Stokes pressure. The discrete recurrence is
beta=1/(1+G dt/(2 eta)), tau_bg_new=(2 beta-1)tau_bg_old+2G beta dt D_diag.
This prediction is checked against the retained straight-load fields **only
through .075**, where complete timestep histories match. Maximum phase, age,
slip and rate differences are 3.29e-9, 1.29e-9 s, 1.16e-12 m and 2.25e-10 m/s;
coordinates/measures match exactly. Subtracting tau_bg_xx=-293.633482 Pa,
tau_bg_yy opposite, leaves maximum shear-stress error 5.36e-6 Pa and A_xy error
8.20e-12. Work-error difference is 3.40e-11 J/m.

The maximum raw resistance difference, 8.29 Pa, is retained: its cold rates are
2.22e-13 versus 3.53e-12 m/s, slips 1.11e-15 versus 1.76e-14 m. There is no rate
cutoff. Rate-weighted resistance RMS is 2.07e-5 Pa and section-power difference
2.17e-8 W/m. Pointwise cold resistance equality is not claimed. The straight
run's final .005 interval has a different history and is excluded from the late
reduction check.

At .080, horizontal-plane midpoints at x=6.03125 and 6.09375 are y=2 to roundoff;
free-plane values are 2.02059280 and 2.02864074 m. Thus measured lateral steering
vanishes when planes remain horizontal under matched mechanical forcing.
Their widths are .158873/.215025 and .089538/.199019 m (fixed/free). Integrated
section slips are .00201562/.00206284 m². Phase-increment L2 difference is
32.21% at .080 and 10.42% at .075; through .060 the phase fields differ by only
4.53e-15 m in L2. Separation begins in the first new-plane admission interval,
.065. Maximum age/slip differences at .080 are .0486241 s/4.80610e-5 m; maximum
rate difference is .0129056 m/s at the same stage time .07875. These are
constitutive-control differences, not numerical error estimates.

B has no mixed-sign sections at that stage. Free planes have 88, maximum
cancellation .727%, negative local power -.00125965 W/m against positive
9.24065 W/m. Full-history signed/absolute physical-work errors are
6.47439e-5/6.47439e-5 J/m for B and 5.23115e-5/6.75089e-5 for free planes.
Neither small work error nor positive integrated dissipation implies pointwise
positive fault deformation or equivalent geometry.

### C: normal refinement supports the common contour, with a history limitation

C holds h_x=.0625 m and all 768 x-Gauss section labels fixed, halves h_n from
.03125 to .015625 m, and independently equilibrates the same physical seed.
Initial phase L2 difference is 5.66e-17 m. Initial fracture energies are
1.1209136166920064/1.1209136166920435 J/m (coarse/fine), and both initial widths
are .0435645354 m: W/h_n=1.39407/2.78813. Thus the comparison does not change
peak, physical seed shape or label spacing.

Early onset is already resolution sensitive: at .040 the fine central band has
maximum phase increment 2.91325e-4 and added fracture 3.49239e-6 J/m, while the
coarse phase remains exactly at its seed. The close later contour does not
establish a precise onset time.

The matched history completes through .0775. Pending .080 is interrupted at the
original wall limit, last Newton14 mechanics 1.19968e-11, phase KKT 2.63163e-4.
This is not an 80-iteration failure. No fine .080 endpoint is available.

At .0775, fine-minus-coarse front offsets are +.644 mm axially (.01030 h_x) and
-.861 mm laterally (.02755 coarse h_n). Fine advance is 1.90037 h_x and its
lateral shift .027804 m, or 1.77945 fine cells. Both remain connected and upward.
At x=6.03125/6.09375, fine midpoints are 2.01752476/2.02525960 m, only
-.379/-.553 mm from coarse. Width differences are -.141/+.926 mm. The matched
contour and along-band steering survive this normal refinement.

Phase L2 difference is .000575664 m, **1.814% of growth since each own seed**. For a
consistent late-window scale, the endpoint difference is **4.256% of the coarse
.075–.0775 increment**; subtracting each path's own .075 field gives a
**7.644% difference between late increments**. These quantities differ because
inherited spatial errors partly cancel later ones. The 1.814% seed-normalized
number must not be compared directly with A's 14.548% late-window number.
Maximum age/slip differences are .000431591 s/.395217 micrometres; stage-rate
difference is .000157687 m/s at the common .07625 stage.

Fine/coarse added fracture is .04624171/.04620571 J/m (+.0779%). Its primary
regional contributions are central band .02561851/.02561583, old taper
.01282427/.01282240 and initially intact axial region .00779894/.00776747 J/m.
Remote/boundary increments and phase remain zero in these accepted controls.
Central physical widths are .06243324/.06244765 m, W/h_n=3.99573/1.99832.
Excess-compliance integrals are 2.01925/2.01677 m. Signed/absolute full-history
work errors through this common time are 6.00282e-5/6.00282e-5 and
5.99102e-5/5.99102e-5 J/m. Positive/negative local powers are
6.92516/-.00104533 and 6.91789/-.00109549 W/m; mixed counts 62/64, maximum
cancellation .957%/.814%. Signed local deformation persists under refinement.
No cancellation or negative local power is clipped away.

**First registration is not mesh independent.** Both grids have 480 formed
sections at .0775, including 96 outside the original seed. Twelve register at
.070 on the fine mesh versus .075 on coarse: fine first-export phase maxima
1.40e-15–1.15e-12, coarse 4.31e-6–3.53e-3. Their frozen angles differ by up to
.0163012 rad (.934 degrees). Slip-weighted angle RMS over all newly formed
sections is .006186 rad (.354 degrees). These are endpoint samples at first
accepted registration, not precise physical nucleation times or a contour rule.

This difference reaches slipping material. At the common endpoint, the affected
fine sections carry 9.19–26.67 micrometres of slip (1.8–5.3% of Dc), 14.49% of
new-section integrated absolute slip and .1149% of the whole-fault total. At
x=6.148793, fine first-export phase is 4.71e-13, frozen angle .172916 rad, versus
coarse .00144964 and .189170 rad; final slips are 19.219/18.896 micrometres.
In the **retained coarse-only .080 follow-up**, those labels carry 52.7–75.6
micrometres and stage rates .0175–.0197 m/s. That is not a matched fine .080
comparison, but shows the labels are not permanently negligible. No threshold,
angle reset or altered admission law was introduced. The data associate earlier
registration with later active material; they do not isolate registration as
the sole cause of spatial differences.

## Measurement and lifecycle checks

Native Q2 roots retain both DG traces, never an averaged field. Vertical cuts
are at identical physical x coordinates; a reported piece must have an exact
horizontal superlevel connection to an interior seed point. Multiple or
unverified pieces remain explicit. An independent analytic ellipse, an ellipse
with unequal DG traces at y=2, and constant-field common-area L2 checks pass on
both normal spacings; see `observation-verification.json`.

Fracture uses authoritative native SIPG derivatives, including faces. For this
AT1 fixture Kf=Gc/((8/3)ell)=2 Pa, so Efr=.5 B dot(gradient+2 mass) includes the
linear potential and quadratic gradient energy exactly. Seed/current totals
are checked against accepted energies. Differences are assigned by fixed
physical cell-center regions; outer-taper diagnostics overlap the old taper.
Physical-work error means Wext minus endpoint storage change and Db+Dc+Df;
signed cumulative and accumulated-absolute errors are separate from the small
algebraic ledger remainder.

Independent readback passes bounds, physical-obstacle irreversibility, positive
age, unchanged original formed angles/measures, ds=dx/cos(angle), virgin age
.5+t, original residual gates and one publication per accepted step. Raw minima
are retained: the smallest
coefficient is -5.42e-19 (roundoff), minimum physical-obstacle gap is zero; no
postprocessing clipping is used. The original solver gates are unchanged. All four
restarts reproduce native phase/solution coordinates and section state exactly.
Every copied parent checkpoint is byte-identical. B additionally has an exact
full-point tau/A comparison; sparse A/fine parent endpoints lack corresponding
point frames, so no separate full-point comparison is claimed there. Rejection
by process interruption publishes no trial; this does not add a new in-process
rollback qualification. See `A/B/C-integrity.json`, `restore-comparisons.json`
and `completed-preservation.json` in the evidence directory.

For disk room, older closed gzip CSVs were losslessly repacked (596442606 bytes
saved), and 48 closed raw CSVs archived with verified round-trip hashes and exact
restoration commands (360.74 MiB saved). `repacked-exports.json` and
`cold-export-compression.json` record compressed-hash/path changes. No executable,
runtime library or checkpoint was removed. The final alias audit finds exactly
the same 64 pre-existing broken CSV aliases, with no new losses; historical
cleanup availability losses remain unrecovered and documented. New closed large
exports are also verified gzip archives. Scripts, fixtures, manifests, accepted
checkpoints, failed/interrupted logs and figure sources are retained; see the
[reproduction guide](../../../benchmarks/reconstructed_fault/particle_free/overnight_free_turn/README.md).

## Manuscript decision and next experiment

Retain a **synthetic model-verification demonstration** of load-steered connected
extension: steering survives the completed late time comparison and normal
refinement through .0775, and the matched horizontal control demonstrates the
contribution of the plane/metric closure. Do not assign converged advance,
onset time, full-.080 refinement, or mesh-independent structural directions.
The first-positive-phase orientation memory is an additional constitutive
assumption with measured resolution sensitivity, even where contour and global
energy agree closely. This is neither general turning/branching validation nor
production readiness.

The next justified experiment is **one fine .0025 interval from the retained
accepted16/.0775 checkpoint (slot02) to .080**, with the same protected physical
pair and original gates. The parent is
`build-particle-free/bulk-tip/runs/overnight-free-turn-c-half/output/restart/02`;
retain the .0025 pending timestep and use a fresh output label. One bounded
90-minute attempt would test this interval without another prefix or a solver
change. The front is approaching x=6.127–6.185, where the two
meshes froze directions at different registration times. Completing this missing
endpoint will test the effect when that material becomes strongly active; use
the retained coarse .080 field as the comparison. A's two remaining .00125
intervals from its own17/.0775 checkpoint are also explicitly unqualified, and
are needed before a complete late temporal conclusion. Neither continuation is
automatically launched.

An unstable-RSF experiment additionally needs an operating-point section-stiffness
assessment. The nominal logarithmic reference is kc=sigma_n(b-a)/Dc=2000 Pa/m
for a=.015, b=.020, confinement200 Pa and Dc=.0005 m. Compare it with actual bulk,
band and boundary compliance using the regularized law, state and radiation.
With eta_rad=1000 Pa s/m, the steady slope is approximately
eta_rad-sigma_n(b-a)/|V|, crossing zero at |V|=.001 m/s; the retained free peak
stage rate is .022958 m/s. Velocity weakening alone does not establish instability
at this operating point. Numerical state/shape accuracy and registration
sensitivity must be bounded before selecting that experiment. No instability
loading, general refactor, branching case or production run was selected.
