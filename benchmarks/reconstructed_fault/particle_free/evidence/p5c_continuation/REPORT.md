# P5c continued loading: connected process-zone extension

Completed on `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local
work. **The radial trajectory reaches the declared extension target at .170 s.**
Growth enters initially intact material and remains seed-connected. Neither mode
nucleates a disconnected region or reaches the boundary stopping condition.
Both complete their own .005 s displacement holds without further phase growth.
The bounded campaign and focused timestep checks are complete; quantitative
spatial accuracy and propagation speed remain unqualified.

## Retained problem and selected stopping rules

Resume each mode's own 192×128/four-rank checkpoint127/.13875 s, retaining all
tensor histories and clocks. The finite 1.5×1 m specimen, released lateral
horizontal motion/vertical restraint, continuum seed and material tuple are
unchanged: G=1e6 Pa, eta=1e8 Pa s, strength drop2000 Pa, R=1000 Pa, Gc=1 J/m²,
ell=.046875 m, Kf=8 Pa, m=4. Retain unlimited DGQ2/native FE source memory,
Gauss8/source3, Δx=Δy=.0078125 m and Δt=.000625 s. Grip shear rate is .018/s
from .05 s through the selected endpoint, followed by zero grip increment.
[Parameters](parameters.csv).

The inherited fresh initialization has .500005% relative L² representation
error, ≤1.11e-16 nodal prescription defect, zero unloaded change/KKT and initial
fracture storage .5695867164 J/m. No initialization or histories were replaced.
All primary damage increments subtract that original represented seed.

[Predeclared rules](SELECTION.md) use four-neighbor nodal
connectivity and Q1 edge crossings. The represented support ends at x=1.09375 m;
the target is connected φ=.05 reaching1.140625 m, one ell farther. Intact
activation uses φ>1e-8 in cells whose four initial vertices are exactly zero.
Remote/boundary thresholds and the1024-critical-master limit were unchanged.
.155/.175 s were observations, not imposed holds; the earlier target ends loading.

## Measured response

The table reports **each mode's own loading endpoint**, followed by its hold.
Fixed target detection requested termination at175; the permitted one-step
latency ended loading at176. The different endpoints are not matched load states.

| Quantity | Fixed-direction tensor flow | Radial tensor flow |
|---|---:|---:|
| First intact activation bracket [s] | (.140625,.141250] | (.142500,.143125] |
| Target crossing bracket [s] | (.168125,.168750] | (.169375,.170000] |
| Actual loading / hold end [s] | .169375 / .174375 | .170000 / .175000 |
| Final grip displacement [mm] | 2.14875 | 2.16000 |
| Integrated Δφ [m²] | 9.321807e-4 | 9.850269e-4 |
| Peak Δφ | .1568841 | .1432395 |
| Integral in initially-zero cells [m²] | 4.249133e-4 | 3.923565e-4 |
| Added fracture storage from original seed [J/m] | .00703993 | .00677812 |
| φ=.01 / .05 / .1 fronts [m] | 1.194567 / 1.154049 / 1.119748 | 1.183564 / 1.143368 / 1.110408 |

At the **common loading time .169375 s**, integrated growth is
9.321807e-4/8.743674e-4 m² fixed/radial: radial reduction **6.20%**; peak reduction
**14.49%**. Radial flow still delays growth, but the earlier31.41% integrated
reduction does not persist unchanged. Comparing its later endpoint directly
against the stopped fixed case would reverse the integral comparison misleadingly.
[Common-time data](matched-mode-comparison.csv).

Radial φ=.05 advances1.0585 ell past the represented support. Initially-zero
cells contribute39.83% of its integrated increment. This is connected weak
process-zone extension with cap amplification, not just initial-cap relaxation.
The φ=.2 front advances .01734 m, φ=.4 only .000402 m, and φ=.6 stays at x=1 m.
The symmetric trajectory does not branch. No disconnected φ=.01/.05/.1 component
appears; newly grown material remains ≥.3203/.3125 m from monitored boundaries
radial/fixed, versus the .09375 m stop distance. Outside-initial-support growth
is labeled separately from **remote nucleation** in the CSVs.

![Common loading-state comparison](fields-0176-tip.png)

Continued grip motion was imposed; the native coupled solver calculated phase,
physical deviatoric stress τ, driving and slip-strain rate. Cyan marks initial
represented support, green initially-zero cells now activated, and white phase
contours. The unchanged high-phase front and advancing low-phase fronts separate
cap/process-zone growth from developed crack advance. Constitutive flow direction
is not crack angle. [Whole body](fields-0176.png)
and [one radial GIF](radial-continuation.gif)
retain boundary/remote visibility. GIF labels give actual times/displacements;
nonuniform frame sampling and equal playback duration do not measure speed.

## Focused temporal and energy checks

Replay the same accepted incoming histories at Δt=.0003125 s in short onset
windows and pre-rapid/hold windows. The latter start at .168750 s fixed and
.169375 s radial; each preserves its physical loading endpoint and full hold.
No reflection or branch alignment is applied. Norms use exact consistent-Q1
integration and normalize by **growth since the common incoming state**.

| Local check | Fixed | Radial |
|---|---:|---:|
| Maximum onset growth-relative L² difference [%] | .0001994 | .01803 |
| Maximum rapid/hold growth-relative L² difference [%] | .0001464 | .03686 |
| Maximum rapid/hold peak-field difference / growth [%] | .0006507 | .04449 |
| Final φ=.05 front difference [m] | 5.06e-9 | 2.91e-6 |
| Max accepted Δφ: coarse / half-step | .0106769 / .0054532 | .0105290 / .0054229 |
| Rapid/hold physical-energy B_E [%] | .001207 | .003379 |

Half-step intact-activation brackets are (.140625,.1409375] fixed and
(.1428125,.143125] radial. The selected5% local growth criterion passes. Near
halving of per-step growth and matching spatial patterns support gradual
advancement in this window, rather than a persistent timestep-independent jump.
These are local two-step Cauchy checks, not a globally refined trajectory,
convergence order or validated propagation speed.

Physical work uses independently summed native boundary reactions:
W_end=Σ f_n·Δu_n and W_tr=Σ(f_n+f_(n−1))·Δu_n/2. Keep the exact BE ledger and
its separate stress/traction/phase/fracture remainders and residual-work terms.
For each replay, C_E=Σ|W_tr,n−ΔE_n−D_n|/W_scale and
C_T=max_common(|δW_tr|+|δΔE|+|δD|)/W_scale, with actual absolute boundary work
as W_scale. B_E=C_E+C_T passes the retained1% practical criterion locally.
Full onset/rapid results and separate terms are in temporal-summary.json,
*-work.csv and the physical histories; no small signed total substitutes for
the sum of absolute step remainders.

From .13875 s through each final hold, W_tr/ΔE/D are
1.35317470/1.11523648/.23794341 J/m fixed and
1.38118307/1.13362821/.24756027 J/m radial. Signed physical remainders are
−5.1903e-6/−5.4066e-6 J/m. Every hold has exactly zero phase change, slip rate,
K/L increment, frictional expenditure and boundary work. Stress stays nonzero:
its mean tensor scales by beta^8=.999950001406, with full-3D relative defects
≤6.44e-16. Viscous dissipation is .00027088/.00027231 J/m fixed/radial.
Phase remained free; stopping motion did not remove stress.

[Temporal maps](temporal-damage-comparison.png),
[energy accounting](energy-accounting.png),
[reaction/growth](reaction-growth-work.png).

## Native checks, cost and provenance

All163 accepted intervals across the selected continued trajectories and local
forks pass unchanged gates: mechanics≤2.78e-12 (bound1e-9), phase KKT≤9.67e-8
(1e-7), source≤4.44e-15 (1e-12), and BE ledger≤3.06e-15 J/m (existing scaled
1e-10 gate). Every interval publishes once; dissipation is nonnegative. Relaxed
curvature is ≥2.83213 on at most621 critical masters; no stability retry or
capacity failure occurs. All221184 diagnostic particles retain exact ID/coordinate
fingerprints. Existing rejection qualification is reused; no new forced rejection
is claimed. Checkpoints before intact activation and rapid growth, and all
terminal histories, are copied outside the rolling ring with verified hashes.

Nonoverlapping native wall costs total18490.99 s (5.14 h), including3554.96 s
completed before shutdown. Assembly/linear/stability/history sums are
3196.98/3309.03/7828.31/92.34 s; rounded checkpoint timers total3482 s. Peak
accepted-state rank RSS is4.86 GiB. The fixed/radial loading stages take
6060.76/8928.74 s, and the four focused replays3048.77 s combined. These are
measured costs, not speedup claims: checkpoint I/O timing varies substantially.
[Per-stage costs](stage-costs.csv)
and activity-cost.csv retain active sizes and solver/stability costs by step.

The original interactive observer died on turn interruption while native MPI
continued to .175 s. Its extra states, positive1014-master certificate and
8413.95 s run are preserved separately, not selected as the experiment endpoint.
Native replay from the verified153/.155 s checkpoint recovered177/.170 s with
**96 bitwise-identical exports**. A persistent controller prevents that observer
lifetime failure in the remaining runs. No native equation/source/binary changed.

Three benchmark-only corrections are documented: CSV export
handles initialization rows lacking energy fields; outside-support growth is
not mislabeled remote; replay target suppression no longer hides a simultaneous
remote/boundary flag. The last correction passes explicit simultaneous-event
controls. Actual replays had **zero** remote/boundary flags, so it changes none
of these results; the supplemental observer started after completion and exited.
No native MPI trigger test is claimed for that helper correction.

The41 protected/BP3 hashes, qualified executable/plugin, three native source
hashes and prior evidence archives remain unchanged. All native commands ended.
Full solver archives/checkpoints remain under build-particle-free/p5c-campaign;
the compact review archive (this package)
contains PNG/GIF/CSV/JSON, inputs, commands and review context.

The original P4 temporal/spatial failures8.302%/7.442%, boundary sensitivity57.26%,
P5c axial growth differences28.68%/37.97% and
**strict_reference_match_passed=false** remain. This is a finite physical
specimen. Recommend designing one targeted spatial check of the identified
connected-growth interval, first checking capacity under the unchanged1024
limit; qualify any required capacity extension separately. No further mesh,
loading, RSF, transport, AMR, geometry or propagation campaign is begun.
