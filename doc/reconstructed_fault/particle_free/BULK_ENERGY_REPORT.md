# Aligned-wide physical work and temporal accuracy

Entry f1f9f29cf plus retained local work; 20 October 2026.
[Pre-run selection](BULK_ENERGY_SELECTION.md) fixes the scope and comparisons.
Reuse the original 48×64 wide result at h=.0025 s, and run only h/2 and h/4 to
.16 s from the identical continuum prescription and unloaded histories. The
loading is unchanged: shear rate .018 s^-1 on (.05,.15], then fixed boundary
motion through .16 s. Bulk law, seed, mesh, Gauss8/source3, unlimited DG, BE,
solver tolerances, physical obstacle and BP3 are unchanged.

## What work is being measured

F_n is the accepted **unconstrained nodal boundary reaction**, including physical
pressure. Let du_n be the prescribed boundary displacement increment. On this
one-parameter affine boundary, du_n=b dgamma_n, b_x=y-.5 m, b_y=0, so Q_n=F_n.b
is the generalized strain-conjugate reaction, in J/m per unit shear strain.
Positive F_n.du_n is work supplied to the specimen (unit out-of-plane thickness).
The two distinct work estimates are

    W_BE,n = F_n.du_n,
    W_tr,n = (F_n+F_(n-1)).du_n/2.

The first is the exact endpoint work in the native backward-Euler ledger. The
second integrates the measured force-displacement path with a trapezoid; it does
not change the constitutive update or make the BE state second order. Native
physical viscous dissipation retains endpoint rates, and friction dissipation
retains the actual signed/absolute shear increments.

For E=Eb+Ec+Ef and D=Db+Dc+Df, keep

    r_BE,n = W_BE,n - dE_n - D_n
           = Lstress,n + Ltraction,n + Lphase,n + Lfracture,n
             - gphi_n.dphi_n - Rmechanical,n.du_n,
    r_tr,n = W_tr,n - dE_n - D_n.

Here the stored native mechanical-work residual also includes pressure-constraint
work: Rvelocity.du = mechanical_work - integral(p div du). Both pieces are
reported separately. Plane strain includes zz=-(xx+yy) and both xy contributions.
There is no extra energy assigned to the native source complement.

The four explicit numerical terms are the integrated |tau_n-tau_(n-1)|²/(4G),
h_(n-1)(q_n-q_(n-1))²/(2G),
q_n²[h_n-h_(n-1)-h'_n dphi]/(2G), and Kf ell²|grad dphi|². The phase term is
signed; it is not clipped or relabeled physical dissipation. Exact algebraic
closure of this ledger alone does not establish physical energy accuracy.

The saved h result contains endpoint work but not individual nodal forces.
During loading, Q_n is recovered exactly as W_BE,n/dgamma_n. Independent saved
DG/local histories reproduce the stress moment integral(tau_xy), differing by
at most 9.64e-8 J/m from that generalized boundary reaction during loading
(<4.2e-11 relative to its maximum). During the hold, zero work is **not** treated
as zero force: the saved stress moment supplies the hold diagnostic. Free
mechanical residuals bound the difference between that volume moment and the
boundary functional. The refined runs save the actual nodal forces and increments
at every accepted state, including zero-motion states.

The read-only decoder explicitly distinguishes DG active-cell/Morton ordering
from the numeric CellId ordering of local q/a histories. Four complete old native
QP exports validate its nine-mode decoding to 5.15e-12 Pa. The initial reader
ordering error was caught by this comparison and corrected before analysis;
no native state or physical input changed.

## Measured result

Only the requested two fresh trajectories were run. Initial physical phase is
bitwise identical in all cases, with zero unloaded drift; no accepted field or
quadrature history was imported. The original affine pure-DG result is reused:
the preceding source-history control already demonstrated roundoff agreement
with the qualified FE-complement route on this same affine mesh. All native
source, mechanical/KKT and once-only publication checks pass at unchanged limits.
Actual refined nodal-force work matches native endpoint work within 1.74e-17 J/m.

All energy/work entries below are J/m, except the stated percentages.

| Measure at .16 s | h=.0025 | h/2=.00125 | h/4=.000625 |
|---|---:|---:|---:|
| Endpoint work | 2.328670618 | 2.302686796 | 2.289694188 |
| Trapezoidal work | 2.276564165 | 2.276633442 | 2.276667479 |
| Storage change | 1.891982305 | 1.892002788 | 1.892013029 |
| Physical dissipation | .384694014 | .384673245 | .384662941 |
| Endpoint residual | .0519942991 | .0260107632 | .0130182175 |
| **Original endpoint screen** | **2.23279%, fail** | **1.12958%, fail** | **.568557%, pass** |
| Signed trapezoidal residual | -1.12154036e-4 | -4.25907087e-5 | -8.49125288e-6 |
| Sum of absolute per-step trapezoidal residuals | 1.12163494e-4 | 4.25954376e-5 | 8.49361742e-6 |
| Absolute residual variation / absolute work | .00492688% | .00187098% | .000373072% |

The historical screen keeps its exact old definition,
|sum r_BE|/max(1 J/m,|sum W_BE|)<=1%. Its original failure remains in the old
report/data; no native acceptance rule or tolerance is replaced.

For constant dgamma=.018 dt during loading, the difference of work estimates
has the exact observable identity

    sum(W_BE-W_tr) = dgamma [Q(.15)-Q(.05)]/2.

The measured differences .0521064531, .0260533539 and .0130267088 J/m satisfy
this to <=2.85e-16 J/m. This explains the dominant O(dt) endpoint excess without
changing an energy term or discarding numerical losses. The remaining physical
work estimate is still based on BE reaction states: observed orders for W_tr,
storage change, dissipation and peak growth are respectively **1.025, 1.000,
1.011 and 1.000**, not second order. Final-value first-order extrapolation gives
remaining W_tr uncertainty about 3.29e-5 J/m; the practical criterion below uses
unextrapolated common-time component differences instead.

| Separate contribution | h | h/2 | h/4 |
|---|---:|---:|---:|
| Bulk storage change | 1.877377905 | 1.877397889 | 1.877407881 |
| Cohesive storage change | .0140469061 | .0140473432 | .0140475617 |
| Fracture storage change | .000557493829 | .000557555314 | .000557586057 |
| Bulk viscous dissipation | .00182285062 | .00179914737 | .00178737450 |
| Cohesive viscous dissipation | 7.22968768e-6 | 7.04881141e-6 | 6.95976724e-6 |
| Friction dissipation | .382863934 | .382867049 | .382868607 |
| Stress BE remainder | .0512201268 | .0256218929 | .0128224887 |
| Traction BE remainder | .000748441376 | .000375519547 | .000188926188 |
| Phase Taylor remainder | -4.22170133e-7 | -2.10307311e-7 | -1.04622172e-7 |
| Fracture BE remainder | 2.61530752e-5 | 1.35611111e-5 | 6.90728440e-6 |

Native exact ledger residual-work contributions remain tiny: summed phase work
is -1.86e-12 / -1.11e-11 / -4.60e-12 J/m and velocity residual work is
7.67e-12 / 2.37e-11 / 1.81e-11 J/m. Pressure-constraint work magnitudes are
<=1.11e-18 J/m. Their actual signs and per-step histories are retained. They
do not explain the original ~.052 J/m endpoint gap.

Independent analysis reconstructs all baseline stresses/storage/stress losses,
and all cases' fracture/cohesive energies, physical rate/increment dissipation,
traction/phase/fracture remainders. Baseline bulk storage differs by <=9.22e-13
J/m; refined full stress/storage is independently checked at initialization,
loading stop and final snapshots (<=4.09e-14 J/m). Refined per-step stress
remainders come from the unchanged native ledger; the compact exports do not
claim an independent stress-loss reconstruction at every refined step.

## No cancellation hidden by the final time

Work, energy and every remainder are exported **per accepted increment** and
cumulatively. During loading, r_tr is negative on every nontrivial interval.
The positive fixed-hold residuals total 4.729e-9 / 2.364e-9 / 1.182e-9 J/m.
Fractions canceled between steps are only **.00843%, .01110%, .02784%** of the
absolute residual variation. Maximum cumulative magnitudes are 1.12158765e-4,
4.25930732e-5 and 8.49243515e-6 J/m. A small final sum therefore does not hide
large alternating errors. Cross-resolution storage/dissipation changes are also
compared separately rather than allowed to cancel in their sum.

All grids integrate exactly the same prescribed total dgamma=.0018 within
2e-18. The interval ending at .05 s is unloaded; loading starts with the next
positive increment. The interval ending at .15 s is the last loaded interval;
subsequent boundary increments and both work estimates are exactly zero. The
native 1e-12 switch convention is unchanged and all step grids align with the
switches.

The hold retains physical stress. At h/4, Q falls only from 2315.85934 to
2315.62777 J/m, while storage decreases by .000378327738 J/m and physical
viscous dissipation is .000378326556 J/m. Friction dissipation on hold is zero;
phase and fracture storage remain unchanged. This is Maxwell/cohesive memory
relaxation at fixed boundary displacement, not removal of the load or stress.
The original h hold similarly has storage change -.000378318064 J/m and
positive dissipation .000378313335 J/m.

The quarter-step run has 256 regular steps plus a native 6.661338147750939e-16 s
terminal clock-closing step, ending at exactly .16 s. **All 257 accepted
publications and all its terms are retained.** That step has zero work, zero
phase change and 2.52e-17 J/m physical dissipation; no end-time or integration
logic was changed to suppress it. Common-time budgets include the last matching
accepted state and its accumulated contributions.

![Work, physical storage/dissipation and separate BE remainders](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_energy/work-ledger.png)

*The same shear history was imposed; reactions and energies were calculated.
Solid/dotted work curves distinguish force-displacement quadrature from endpoint
accounting. The negative phase remainder and tiny residual work are kept, and
absolute cumulative errors expose cancellation. Nonzero reaction after the
second vertical marker is the fixed-displacement hold.*

## Actual response and temporal uncertainty

Comparisons use physical fields at common times and the measured **growth**
delta_phi=phi-phi_initial. Cap means x>.5 and radius<.35 m about (.5,.5).
The L2 denominator is the finer cap-growth norm with the same positive physical
quadrature; Linf/peak differences use the finer peak growth. The large .6 seed
never normalizes these errors. Zero-growth states have no invented relative
error. Initial representation error is unchanged (.33972% whole / .74985% cap
L2 from the preceding independent seed comparison), not a temporal error.

| Response | h | h/2 | h/4 |
|---|---:|---:|---:|
| First-positive cap bracket, s | (.135,.1375] | (.13625,.1375] | (.13625,.136875] |
| First native node angles | ±40.6013°, ±42.8789°, ±49.3987° | same six | ±42.8789° |
| Final peak cap increment | .001671204838 | .001671325459 | .001671385770 |
| Final physical cap-growth L2 norm, m | .000107289725 | .000107300093 | .000107305277 |

The earlier apparent simultaneous first activation of six nodes is resolved
into a first pair by h/4. No mirror branch is registered or selected. At common
physical time .1375 s, the h/2 versus h/4 growth L2 difference is .05662% and
Linf difference .04873% of the growth. At .16 s they are **.005071% L2 and
.003608% Linf**. The h versus h/4 differences are .16986% L2 at .1375 s and
.015211% L2 at .16 s; final peak difference is .010825%. Thus these comparisons
are not made artificially small by the initial seed.

Original-QP driving is evaluated before the existing radius-.0625 m physical
kernel average; no local history is interpolated into a different constitutive
state. At .135 s h/2/h4 angular Y differs by .001806% relative L2 across the
three sampled arcs; at .16 s it differs by **.001123%**. The ±43° outer-cap
peaks (and ±45° at r=.225 m) are unchanged on the 1° angle grid. Growth stops on
the final hold. These are temporal comparisons on one mesh; the documented
orientation/spatial onset sensitivity remains.

![Actual growth and common-kernel temporal response](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_energy/temporal-response.png)

*The imposed seed, mesh and loading are identical. Colours show actual native
nodal increments; white contours show phase and red circles mark first activation.
The curves nearly coincide, so the right-hand panel quantifies differences
against growth itself. Timing remains bracketed by accepted states, not claimed
exact from a smoother plot.*

## Recommended physical-energy accuracy criterion

Keep the exact BE ledger and its original numerical tolerances as mandatory
implementation checks. Retain the old 1% endpoint screen as historical data,
including its failures above. For **physical work accuracy**, recommend the
following additional reporting/qualification rule on a specified nonzero-work
event, supported by a decreasing three-level timestep hierarchy:

    W_scale = sum_n |W_tr,n|,
    C_E = sum_n |W_tr,n-dE_n-D_n| / W_scale,
    C_T = max_(common times) ( |W_tr,f-W_tr,c|
                              +|dE_f-dE_c|+|D_f-D_c| ) / W_scale,
    B_E = C_E + C_T <= 0.01.

W_tr, dE and D in C_T are cumulative histories at common physical times, with
f the reported fine run and c its double-step control. This prevents both
between-step cancellation and cancellation of temporal errors between storage
and dissipation. W_scale is the event's actual absolute force-displacement work;
it is not initial fracture storage or a dimensional floor. Report each stage
and max cumulative signed error alongside C_E. C_T is a measured Cauchy
uncertainty, not an unconditional mathematical error bound.

Require response agreement too: growth-relative L2 and peak/Linf differences
<=5% at declared common positive-growth times, and report compatible shrinking
onset brackets. The same force-displacement test must not be called second order
because it uses trapezoids; these data show first-order BE states.

For h/4 against h/2, **C_E=.0003731%, C_T=.0023978%, B_E=.0027709%**.
Maximum selected positive-growth differences are .05662% L2 and .04873% Linf,
well below 5%; all onset brackets overlap. The bounded work/temporal criterion
therefore passes. Even h versus h/4 is already close for energy/growth amplitudes;
the finer calculation mainly supplies the .000625 s onset bracket. The old 2.23279%
endpoint failure is chiefly a work-quadrature excess, not evidence of a comparable
unresolved physical balance defect. This conclusion rests on the measured
refinement, independent reactions and response, not on subtracting a loss term.

This resolves the selected fixed-mesh work/temporal question. It does not remove
spatial/mesh-orientation findings, validate material physics, or qualify physical
propagation. The model, integrator and production defaults remain unchanged.
`strict_reference_match_passed=false` is retained. No additional native case,
server job, limiter, mesh or constitutive experiment was run.

Evidence and commands: [assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_energy/assessment.json),
[time-series/replay guide](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_energy/README.md).
