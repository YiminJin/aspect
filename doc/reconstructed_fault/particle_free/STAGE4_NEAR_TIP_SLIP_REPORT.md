# Stage 4 near-tip prepared sliding

The user selected a new slipping-state preparation at the **same** elastic
preload gamma_e=.000365, followed by a bounded fixed-phase weakening/b=a pair
and a conditional free-phase decision. The
[uniform-preload calibration](STAGE4_PRELOAD_CALIBRATION_REPORT.md) is closed;
no new preload multiplier is tuned here. The physical solver, material data,
pressure convention, native Bernstein phase, quadrature and tolerances remain
unchanged. This is an explicit synthetic initial-state change.

## What changed physically

The new nominal rate is 0.2 mm/s times a C1 profile: zero through x=3 m,
sin²(pi (x-3)/2) on 3<x<4, one on 4<=x<=6, cos²(pi (x-6)/.25) on
6<x<6.125, and zero beyond. All nonzero-rate sections are verified to be
formed. The old phase coefficients, stored planes and section measures are
retained exactly; the cutoff lies within existing formed support. No sliding
in intact material or geometric continuation is invented.

The independent 1% sin² age perturbation moves from x=2.5..5.5 to x=3..6 m.
It vanishes at x=6, while the nominal rate there remains 0.2 mm/s. The existing
native helper equilibrates stress/cohesive memory at the unperturbed target
resistance, then applies the age perturbation and inverts the owned friction
law at that common resistance. The two laws therefore share initial stress
and damage margins, but their actual perturbed initial rates differ slightly.
Uniform unperturbed age, maximum nominal rate, material parameters and Dc are
unchanged. The imposed physical shear/extension velocity is unchanged.

The actual initial weakening-law rates, interpolated at identical physical
coordinates through the native section values, are:

| x (m) | Old V (mm/s) | New V (mm/s) |
|---:|---:|---:|
| 2.0 | .002398 | 0 |
| 5.5 | .046320 | .200627 |
| 5.75 | .025623 | .200168 |
| 5.9 | .007760 | .200027 |
| 6.0 | .002398 | .199982 |
| 6.05 | .001137 | .130855 |

The x=6 entry is interpolated from section points on either side of the join;
the prescribed nominal function is exactly .2 mm/s at the join. This is a
change in actual taper sliding, not merely relocation of a vanishing age bump.
Spatially integrated initial rate changes from .000498592 to .000516283 m²/s,
only +3.55%, while the taper rates increase by much larger factors.

The [nominal profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_near_tip/prepared-rate-profiles.png)
and [actual initial rates](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_near_tip/initial-rates.png)
show both the complete slipping region and an enlarged right taper. PDF
companions and exact CSV tables are retained.

## Re-equilibrated margins, before evolution

The independent zero-interval native preflight completed normally in 83.17 s.
Its mixed-equilibrium residual is 7.61e-12, static relative work error 2.04e-14,
local traction defect 7.11e-13 Pa, and friction inversion error 2.85e-14 Pa.
The original equilibrium and one-sided phase-direction checks pass unchanged.

Positive net driving means an admissible infinitesimal damage direction lowers
the reduced functional. Negative means that direction resists growth. The
native full AT1/SIPG gradient is paired with the actual mechanical driving;
normalization by each direction's weighted area gives Pa. Smooth probes and
individual nonnegative Bernstein rows are kept distinct, as in the calibration
report. At t=0 the virgin-section direction diagnostic uses the analytic native
objective without publishing a new history; later stages use actual native
selected planes. No physical history is changed by observation.

| Initial measure (Pa) | Old preparation | New preparation |
|---|---:|---:|
| Most favorable right axial-tip row | -.040334 | -.061951 |
| Most favorable left axial-tip row | -.040334 | +.319523 |
| Right smooth tip-spanning probe | -1.773933 | -1.795150 |
| Left smooth tip-spanning probe | -1.773933 | -1.442019 |
| Right immediately-ahead probe | -1.421664 | -1.412057 |
| Left immediately-ahead probe | -1.421664 | -1.122910 |
| Maximum over all exactly intact rows | -1.276638 | -1.024516 |

The right tip becomes slightly **more resistant** despite the higher prepared
rate, while three left axial-tip rows become favorable. Broad negative probe
averages do not negate these favorable left rows. Old-tail readjustment modes
also remain. The changed resistance distribution alters the equilibrated
cohesive strain and stress, even at the same affine boundary displacement.
This is exactly why the old right-tip margin cannot be reused.

For example, at the section point x=1.998945 m the old prepared rate was
2.34126e-6 m/s and its resistance 108.269 Pa; both become zero in the new
compact preparation. At x=5.998945 m the resistance instead rises from
108.412 to 121.809 Pa. The regularized RSF law has zero resistance at zero
rate. Thus moving this compact slipping state also changes the left-side
resistance substantially, despite its small old rate. Neither a static-friction
floor nor a clipped signed rate was added to conceal that consequence.

The prepared phase is inherited, not globally re-equilibrated. Any free left
response must distinguish this already favorable common initial condition from
a law-specific effect. The left tip remains a useful comparison, but is not
an initially stable negative control. No exactly intact or remote row is
initially favorable.

## Fixed-phase pair and release decision

Both laws completed eight midpoint intervals at dt=8e-5 s, to .00064 s, with
four ranks sequentially. The last **stage** is at .00060 s. Rates, resistance
and driving below refer to that stage; cumulative slip refers to the .00064 s
endpoint. The entire phase coefficient array is exactly unchanged in every
accepted state. Initial mechanical margin mismatch between laws is only
5.89e-14 Pa. The pair therefore compares the same preparation and physical
locations without conflating stage and endpoint time.

| Final quantity | Weakening | Matched b=a | Relative increase |
|---|---:|---:|---:|
| Peak V (mm/s) | 2.170110 | 1.667319 | 30.16% |
| Integrated V (m²/s) | .004825643 | .003823326 | 26.22% |
| Integrated cumulative slip (m²) | 1.248676e-6 | 1.103572e-6 | 13.15% |
| V at x=5.75 (mm/s) | 1.642456 | 1.333770 | 23.14% |
| V at x=6.0 (mm/s) | .358430 | .346011 | 3.59% |
| Slip at x=5.75 (micrometres) | .453481 | .406085 | 11.67% |
| Slip at x=6.0 (micrometres) | .183520 | .180706 | 1.56% |
| Resistance at x=5.75 (Pa) | 126.603788 | 126.708608 | — |
| Resistance at x=6.0 (Pa) | 123.177864 | 123.213840 | — |

The new preparation produces substantial sliding and law separation inside the
formed taper. The separation diminishes near its endpoint. The CSV histories
also include x=1.95, 2, 2.05, 5.5, 5.9 and 6.05 m. Local driving and fracture
derivatives use nonnegative Bernstein probes centered at these same coordinates,
not a peak chosen separately for each law. Probe coefficients sample a tent;
it is exactly piecewise Q1 when its breaks align with cell faces. For example,
at x=5.75 the final mean mechanical driving is .383941/.383680 Pa against a
common fracture derivative 4.110861 Pa; at x=6 it is .426623/.426502 Pa against
2.269417 Pa. These averages do not replace the individual admissible-row test.

| Final net driving (Pa) | Weakening | Matched b=a | Weakening minus control (microPa) |
|---|---:|---:|---:|
| Initially limiting right-tip row | -.049215254 | -.049384214 | +168.959 |
| Initially limiting left-tip row | +.319510200 | +.319510396 | -.196 |
| Right tip-spanning probe | -1.783244751 | -1.783263633 | +18.882 |
| Right immediately-ahead probe | -1.404253786 | -1.404072559 | -181.227 |
| Right exactly-intact probe | -1.411420197 | -1.411449293 | +29.096 |

The fixed right row is the Bernstein row j=0 in the cell centered at
(6.015625,1.984375) m, initially B=.04139415. The fixed left row is j=8 at
(1.984375,2.015625) m. They remain the most favorable rows in their respective
axial-tip cells. The right-row law excess is positive at every sampled stage,
increasing to .000168959 Pa; the other right-tip rows have small mixed signs
early, but all 18 have positive excess at the last stage (92.13..199.53 microPa).
The broader right cap has both signs, -267.22..199.53 microPa. Thus relocation
reverses the old limiting-row effect, but does **not** produce uniformly
enhanced driving everywhere ahead of the tip.

The control's right-row deficit falls from .061950885 to .049384214 Pa. The
law-specific excess is **0.342% of the remaining deficit**, or **1.34% of the
common control's approach toward fracture**. The negative immediately-ahead
probe difference is only .0129% of its 1.404 Pa deficit. There is no admissible
row that becomes favorable only under weakening in either tip cap, either
intact-ahead region, all exactly intact cells, or remote material. All exactly
intact and remote rows remain unfavorable. The three right-tail and nine
left-cap favorable rows remain common to both laws; three of the latter are
in the left axial-tip cells and were already favorable initially.

**Decision: close this preparation and do not run a free-phase pair.** The
positive right-tip interaction is numerically distinguishable but small
relative to the remaining fracture deficit. The negative ahead-tip interaction
also does not supply an appreciable delay near onset. Neither sign meets the
user's release criterion in this window. The much larger global slip-rate
difference alone is insufficient. Uniform-preload tuning remains closed.

This is a bounded-window conclusion, not a claim that later acceleration could
never affect growth, nor a temporal/spatial convergence qualification. Before
another growth experiment, the justified next preparation check is a
phase-stable specimen/seed design that retains appreciable right-taper slip
and checks both tip margins and signed deformation. In particular, it must not
mistake zero integrated left slip for absence of local deformation. Continuing
this trajectory or tuning its preload still closer to a single-row threshold
is not selected here.

## Signed deformation, work and cost

At the final stage the right taper has no negative fault deformation in either
law: its positive integrated rates are .000719871/.000586862 m²/s. Across the
whole fault, positive and negative contributions are .004871110 and
-.0000454666 m²/s for weakening, versus .003858859 and -.0000355333 for b=a.
The corresponding cancellation fractions, 1-|P+N|/(P-N), are 1.850% and 1.825%.

The left taper is different: its positive and negative contributions are
approximately ±1.69911e-6 m²/s under weakening and ±1.49244e-6 under b=a.
They cancel almost completely. Its near-zero integrated slip does not imply
locally quiescent fault deformation. The section closure applies one signed
force to that signed integral; opposing local power contributions cancel in
its friction/radiation work. This is retained explicitly, without clipping,
replacing signed slip by absolute slip, or suppressing local cohesive history.
It limits interpretation of the left tip as an independent physical control.
The outputs retain positive/negative rates and powers separately.

| Verification/cost | Weakening | Matched b=a |
|---|---:|---:|
| Accepted intervals / exit code | 8 / 0 | 8 / 0 |
| Maximum mechanical residual | 1.024e-11 | 9.964e-12 |
| Added fracture inventory (J/m) | 0, phase fixed | 0, phase fixed |
| Signed physical-work error (J/m) | +8.8645e-11 | +6.7527e-11 |
| Accumulated absolute physical-work error (J/m) | 8.8645e-11 | 6.7527e-11 |
| Absolute error / external work | 7.176e-10 | 5.466e-10 |
| Total section friction/radiation work (J/m) | .000156795 | .000138450 |
| Maximum discrete work-adjoint defect (J/m) | 8.81e-20 | 1.63e-19 |
| Wall time (s), four ranks | 1008.36 | 996.77 |

Physical-work error is Wext minus the bulk/cohesive/fracture energy change and
all physical dissipation, before numerical ledger corrections. Both signed
and absolute errors are reported; their equality here means there is no
inter-step cancellation. Each accepted step publishes once. The zero fixed-phase
KKT residual and zero fracture increment are imposed-control facts, not proof
of physical phase stability. The preflight plus pair cost about 34.8 minutes
of numerical wall time. No new convergence/restart/production qualification is
claimed, and no free-phase continuation or checkpoint conversion was performed.

## Figures and machine-readable evidence

* [Actual old/new initial V](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_near_tip/initial-rates.png), including the enlarged taper.
* [Matched slip and driving histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_near_tip/fixed-histories.png), including signed right/left law differences.
* [Local cumulative slip and resistance](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_near_tip/fixed-local-slip.png).
* [Full comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_near_tip/fixed-comparison.json), per-law CSV histories and compressed native row margins, with PDF figure companions.

## Ownership and reproduction

The benchmark is in
[`stage4_near_tip`](../../../benchmarks/reconstructed_fault/particle_free/stage4_near_tip/README.md).
`prepare.py` writes a new background-rate table and builds a separate checker
from the protected preload checker. The only changed checker expression moves
the age-perturbation support; existing phase/margin observers and native static
mechanics are reused. Production source and checkpoint meanings are unchanged.
The legacy `mean_patch_resistance` output still denotes x=2.5..5.5; analysis
renames it explicitly and separately computes the new support/taper averages.

Worktree/branch: `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`, revision
`b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`, preserving incoming manual changes.
The retained executable is `first-free-turn/tested/aspect-release`, SHA256
`089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`;
its identity is independent of HEAD. `build-particle-free/stage4-near-tip/`
records entry/build identities, exact case lists, logs and task state.
Original run outputs/checkpoints remain in the existing bulk-tip run tree.
Earlier cleanup losses are unchanged. No commit or production job is selected.

[`preservation.json`](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_near_tip/preservation.json)
records the exact source/runtime/input identities and hashes of checkpoints
retained in place. The evidence directory contains the three run commands,
expanded parameter files, compact native observations and logs. Full raw
rank-local scans and phase coefficients remain in the original run directories;
the compressed margin table retains the law comparison without copying those
large raw files. `analyze.py initial` and `analyze.py fixed` regenerate the
figures and tables after the numerical ranks exit.
