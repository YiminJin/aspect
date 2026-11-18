# Stage 4: loaded-seed preparation and friction/growth comparison

The selected overnight task is
[`CODEX_STAGE4_OVERNIGHT_PREPARATION.txt`](CODEX_STAGE4_OVERNIGHT_PREPARATION.txt).
It preserves the closed [near-tip fixed-phase result](STAGE4_NEAR_TIP_SLIP_REPORT.md)
and prepares a different, phase-stable initial state before comparing the laws.
The common preload remains gamma_e=.000365, with the same near-tip target-rate
profile and 256×128 mesh. At most four numerical ranks run in aggregate. The
eight-hour allocation starts 14 November 2026 at 07:34:30 UTC, ending no later
than 15:34:30 UTC, including reduction and retention.

**Completed decision:** retain this phase-stable preparation as a controlled
friction/growth comparison for further spatial validation. Weakening produces
earlier detected motion and a larger right connected-contour displacement;
the local timestep check preserves that ordering. The motion is confined to
the inherited diffuse band/tail, with no change in exactly intact material,
no new formed section and no new turning. It establishes a discrete frictional
effect on cap evolution, not spatially qualified finite propagation or an
autonomous event. The driven window is closed at 6.08 ms; no job remains active.

The synthetic material has G=1e6 Pa, eta=1e8 Pa s, Gc=2/3 J/m,
strength=1000 Pa, phase length=.125 m, h=.03125 m and
Dc=3.22111261263e-7 m. Both laws use a=.015 and pressure=200 Pa;
(b,mu_ref) are (.020,.6) and (.015,.6080471895621705), matching resistance
at the common unperturbed age=.001610556306315 s. The inherited boundary
velocity is v_x=.01(y-2)-.0041954981558864(x-4),
v_y=.0041954981558864(y-2), in m/s. It includes normal extension as well as
shear. The held preload uses the same affine ratio, with gamma_e=.000365.
No material, mesh, loading, quadrature or physical-step tolerance is tuned.

## Static preparation and history meaning

At held boundary displacement the initial mechanical equilibrium and native
irreversible phase constraint are solved together. The inherited native
Bernstein phase is the lower obstacle; one is the physical upper bound. The
ordinary fracture matrix and distributed mechanical/Stokes operators are reused,
with the existing phase obstacle solver. The benchmark helper adds no physical
time integrator or material law. All original tolerances are retained.

The phase and plane catalog are inherited from the retained native .075 s
specimen used in Stage4. They are geometric/structural input here, not a resumed
material checkpoint. Formed planes range from 0 to10.855 degrees, and the
prepared right-tip section angle is about7.227 degrees. The normal tip offsets
are therefore inherited geometry.

The prescribed data are the held elastic boundary displacement, section planes,
unperturbed age and target V(x). The friction API supplies their common resistance
f(x). Physical bulk stress tau, cohesive strain A and phase are equilibrated at
that force. Cumulative slip increments start at zero relative to this preloaded
state; the nonzero initial elastic A remains in the stored energy. Age does not advance, and preparation neither
relaxes Maxwell history nor increments accepted-step/publication clocks. After
preparation the same 1% age perturbation is applied to both laws; each friction
law is inverted at the common retained resistance. The perturbation vanishes
at x=6, while the prescribed taper sliding there remains .2 mm/s.

With E=dev(sym grad u), S:S=1/2 and
H=m phi(1+phi)/[m phi(1+phi)+(1-phi)^2], elimination gives

    F = 2G E:S-f,       e=H F/G,       q=(1-H)F,
    A=e S,              tau=2G E-2G e S,       Q=2q S.

Here dev is the full 3D plane-strain deviator, including the zz component.

The regular form also applies at intact phase: e=0, q=F and cohesive energy
e q/2=0. For C>0 it equals G e²/(2C), with C=H/(1-H); no normalization floor
or intact-phase division is introduced.

The static functional is the native fracture energy plus the integral of bulk
and cohesive storage **and f e**. Its phase derivative is

    Gamma'_i - integral N_i H'(phi) (2G E:S-f)^2/(2G).

The constant-resistance work term is essential to the equilibrium/phase
derivative; stored energy alone would prescribe a different preparation.
The native point update is evaluated with displacement coordinates u/beta, so
its stress predictor is exactly 2G E and beta cancels. Its positive algebraic
stage duration is an evaluation device, not an elapsed physical interval.
The subsequent physical midpoint equations, pressure reference, quadrature,
section closure and acceptance criteria remain unchanged.

The static iteration uses the native fracture-plus-mass search metric and
physical obstacle, re-equilibrates mechanics for every phase trial and accepts
the reduced static functional by Armijo search. Final acceptance requires the
native physical phase KKT tolerance and the ordinary mechanical residual gate.
Preparatory fracture change/front motion is separate from later growth.

## Preparation evidence

The held-displacement preparation converged in three phase corrections. Its
physical phase KKT fell .866824 → .000675524 → 5.4045e-6 → 4.3285e-8 against
the unchanged 1e-7 tolerance; final mechanical residual was 7.50e-17. Static
traction error is 6.54e-13 Pa, force inversion error 2.85e-14 Pa and relative
static work error 1.26e-14. Time, accepted-step count and publication count
remain zero, and all section cumulative slips remain zero.

| Preparation measurement | Result |
|---|---:|
| Right connected absolute phi=.05 tip | 6.00205437994 m, unchanged |
| Left connected tip | 1.99794141510 m |
| Left preparation advance | 4.20507e-6 m = .00013456 cell |
| Maximum phase coefficient increment | .000658131 |
| Initially intact coefficient changes | exactly zero |
| Detached phi=.05 components | none |
| Right axial-tip maximum net driving | -.061947664 Pa |
| Left axial-tip maximum net driving | +1.04e-10 Pa, within tolerance |
| Old right/left cap maximum net driving | +1.20e-10 / +3.03e-8 Pa, within tolerance |
| Exactly intact maximum net driving | -1.023596839 Pa |
| Remote maximum net driving | -1.466767055 Pa |
| Prepared V at x=5.5,5.75,5.9 m | .2 mm/s |
| Prepared V at measured right tip | .199765 mm/s |

The complete initially intact coefficient array is unchanged, including values
below plotting thresholds. No specimen revision is needed. The left tip is
locally stationary at the prepared phase, although it is not strictly below
every damage threshold: free rows naturally have near-zero derivatives.
This is native first-order constrained stability, not a proof of the global
minimum of the nonconvex phase functional.

The phase increase smooths old band/tail material. Fracture inventory changes
from 1.14817764085426 to 1.14817762366003 J/m, a **decrease** of 1.71942e-8 J/m;
the gradient/face part can decrease under an irreversible phase increase.
Preparation is an initial-state selection, not a physical growth trajectory
with a measured slip/aging work ledger. That inventory is the common baseline
for subsequent physical fracture changes.

Connected contour roots retain both DG traces. Sampling at four and eight
points per cell gives identical axial endpoints here. The inherited right/left
endpoint locations have normal offsets near y=2.009365/1.990614 m. These
belong to the supplied phase field and are not new turning. The reducer
retains normal ranges if extrema tie and reports upper/lower-half extents
separately; native structural-plane angles remain distinct from contour geometry.

The first attempt reached the same KKT tolerance but crashed in the new
post-preparation export because its FEValues object omitted update_values.
The flag was corrected and the entire preparation completed normally on retry
in 270.85 s. The failed checker, input and log are preserved separately. No
physical solver or tolerance was changed to repair this export defect.

The exact prepared coefficients are now the common fresh input for the matched
law pair. The pair is selected through 32 midpoint intervals, dt=8e-5 s,
t=2.56 ms; the first eight intervals provide the old common-window comparison.
The new native state dump permits pointwise initial tau/A comparison. Static
preparation assigns V by the owned friction-law inverse; signed local fault
deformation is measured from the first accepted physical stage, not invented
from static strain divided by an arbitrary preparation time.

At the first solved physical stage, t=40 microseconds, the weakening/control
rates are .22719/.22677 mm/s at x=5.5 m, .22562/.22535 at x=5.75 m,
.22067/.22049 at x=5.9 m, and .21208/.21198 at the measured right tip.
Thus appreciable taper sliding survives the actual coupled rate solve, rather
than existing only as prescribed initial data. The native left-taper positive
and negative fault-deformation integrals are +2.34183e-7/-2.34183e-7 and
+2.34463e-7/-2.34463e-7 m²/s, respectively. These use fault deformation alone;
ordinary elastic and viscous bulk deformation is excluded.

The comparison includes both broad probes and a profile-shaped nonnegative
increment: the positive part of a one-cell axial shift of the prepared
Bernstein field, multiplied by a fixed quartic window with axial half-width
.5 m at each tip. Its mechanical driving and full native fracture cost use
the same row/quadrature measures. This is a read-only admissible direction;
it does not translate the phase, evolve geometry or establish finite connected
growth by itself. Release selection must consider a neighborhood, common
loading motion, slip/Dc and the time/contour resolution, including a possible
negative/delaying law effect.

## Completed fixed-phase comparison and release selection

Both laws completed all 32 midpoint intervals through 2.56 ms (last stage
2.52 ms), in 2377.42/2356.36 s on four ranks sequentially. Every held phase
coefficient is exactly unchanged. The exact common phase is verified across
laws. All native quadrature-point tau and A components were compared with
matching rank/CellId maps: maximum differences are 1.089e-9 Pa in tau and
7.915e-18 in A, relative maxima 1.976e-12 and 4.707e-15. Initial section
resistance is exactly equal. These pass the declared 1e-9 relative comparison
tolerance; that check does not change the physical solver's tolerances.

| Endpoint window | Right-tip row excess, weakening minus control (Pa) | Control's remaining row deficit (Pa) | Excess right-tip slip / Dc |
|---|---:|---:|---:|
| .64 ms, step8 | .000168987 | .049381329 | .009623 |
| 1.28 ms, step16 | .003334164 | .028302088 | .158372 |
| 1.92 ms, step24 | .006061961 | .001835215 | see full CSV |
| 2.56 ms, step32 | .006554558 | none; row favorable by .024069991 | .270689 |

Rates/driving use the common midpoint stage, while slip uses the listed
endpoint. The tip station is the measured prepared x=6.00205437994 m;
x=5.75 m samples the taper interior. Final slips there are 3.918237/3.647548 Dc
at the tip and 28.652405/24.898839 Dc inside the taper. The row excess grows
with local slip/state separation. At step24 it exceeds the control deficit;
at the final stage all 18 checked axial-tip rows have positive law excess,
5.88..7.79 mPa. The initially limiting row is favorable by .030624549 Pa under
weakening versus .024069991 Pa in the control.

The first favorable right-tip directions are bracketed by stage times
1.72..1.80 ms under weakening and 1.88..1.96 ms for b=a. Counts use 2e-7 Pa,
the native 1e-7 KKT tolerance times Kf=2 Pa, rather than counting roundoff-scale
signs as damage instability. These nonoverlapping intervals suggest roughly
two timesteps of separation **for infinitesimal directions**, not yet for a
connected front.

The whole-profile advance direction is still unfavorable: final mean net
driving is -2.248673/-2.255772 Pa, though its law excess is +.007099 Pa.
The immediately-ahead probe also resists damage (-1.351458/-1.353523 Pa).
Exactly intact and remote rows remain unfavorable in the held-phase calculation.
Thus favorable coefficients alone cannot establish finite connected advance.

**Release selected:** the coherent right-neighborhood excess, measurable local
slip/state difference and separated direction-activation brackets justify one
fresh coupled pair. Both laws start at the same prepared t=0 state, with phase
and ordinary native virgin-plane selection enabled. They do not release from
a fixed-phase endpoint. The first selected free window is the same32 intervals;
its cost and response determine whether a longer window or local half-timestep
check is justified within the existing allocation. A snapshot is requested just
before the apparent separation window. The physical solver, pressure reference,
quadrature, constitutive law and tolerances are unchanged.

The held-phase signed physical-work errors are +7.8681e-10/+3.2894e-10 J/m;
accumulated absolute errors have the same values. These are Wext minus all
physical energy change and dissipation, before numerical ledger corrections.
Zero added fracture inventory in these runs is a consequence of holding phase.

The final held-phase right-taper positive/negative deformation integrals are
.00186346/0 for weakening and .00182783/0 m²/s for the control. On the left
they are +3.38513e-5/-3.38513e-5 and +3.11529e-5/-3.11529e-5 m²/s: cancellation
exceeds 99.99999999998%. The corresponding signed left force powers cancel
as well (+/-.00102516 and +/-.00091454 W/m). Thus a quiet net left-section
rate conceals local deformation. The nonlocal section closure dissipates by
the integrated signed rate, not by the integral of its absolute value. This
limitation is retained; no clipping or localization rule has been introduced.
Maximum global work-adjoint defects are 1.00e-18/7.05e-19 in the held pair.

The first freely evolving weakening history identifies the earlier favorable
row as B0 in the lower axial-tip cell, centered at (6.015625,1.984375) m. Its
increment is 6.6124e-7 at step26, while every coefficient of the upper cell
containing the actual outer tip is unchanged. The absolute connected contour
does not advance. This is cap adjustment, not connected extension. The least
resistant upper-cell direction is B6; its held-phase final deficits are
.12446/.13225 Pa, substantially larger than the lower-row deficits quoted above.
A linear extrapolation of the last nine held-phase samples puts its zero near
5.18/5.40 ms. This is only an observation-window estimate. It supports following
the same free pair longer within budget, without adjusting the specimen or load.

## Free phase: first common window and continued comparison

Both fresh free histories completed32 intervals through2.56ms, in3962.63/
3844.53s on four ranks sequentially. Initial phase and age are exactly equal,
slip is zero, and native tau/A and resistance pass the same comparison as the
held pair. The common age perturbation gives a maximum initial rate difference
6.34648e-7m/s through the separate owned friction-law inverses; it does not
change the common initial stress or resistance.

| Free response at2.56ms | Weakening | Matched b=a |
|---|---:|---:|
| Right connected contour advance | 0 | 0 |
| Left contour advance (m) | 5.29e-9 | 4.01e-9 |
| Maximum coefficient increment | 3.92324e-5 | 3.28215e-5 |
| Exactly intact / remote increment | 0 / 0 | 0 / 0 |
| Added fracture inventory (J/m) | 7.20704e-8 | 5.93073e-8 |
| Wholly ahead of right front, row partition (J/m) | 5.32354e-10 | 5.14831e-10 |
| Right front-straddling cell partition (J/m) | 4.62691e-10 | 4.19736e-10 |
| Signed physical-work error (J/m) | 2.68402e-13 | 1.36465e-13 |
| Accumulated absolute error (J/m) | 2.69026e-13 | 1.37405e-13 |

No detached contour or newly formed section appears, and all old formed-plane
angles remain exactly unchanged. Both DG traces give identical endpoints at
sampling densities4/8. This is contour-extraction consistency, not a spatial
accuracy estimate. Regional fracture values are native quadratic **row
partitions**, including face terms; they are not unique local energy densities.
Their total matches the inventory change within1.99e-15J/m. Wholly ahead
material includes the inherited diffuse tail and is distinct from exactly
intact cells. Left-taper signed cancellation persists; right-taper deformation
remains one-signed. The tiny work error does not establish state accuracy.

The coupled outer-cell deficits at the last stage are .1244637/.1322545Pa.
Their measured recent slopes place zero near5.180/5.404ms, consistent with the
held estimate. **Continue the same free pair to6.08ms**, without a specimen
revision, retaining an accepted4.8ms parent for a local timestep comparison.
The short initial observation window has not tested connected extension yet.

A5.13s restart attempt was rejected by the clipped-terminal guard before any
physical step. That scheduling contract is inapplicable to this snapshot.
The existing ordinary restart callback successfully restores accepted parent32
at2.56ms and schedules step33 at2.64ms, dt80microseconds; the original saved
pending clock was not independently dumped by the failed attempt.
The failed command/log and original snapshots remain intact; no core source,
checkpoint payload or acceptance tolerance changed.

Both continuations completed all 44 additional intervals through 6.08 ms,
in 3521.69/3434.60 s. The combined histories contain every accepted state from
the fresh t=0 release; `free76` is an explicitly labelled analysis view of the
original native runs, rather than another numerical trajectory.

| Complete free response at 6.08 ms | Weakening | Matched b=a |
|---|---:|---:|
| Largest logged stage rate (mm/s) | 15.1823 | 11.3253 |
| Time of that rate maximum (ms) | 3.72 | 3.88 |
| Final peak stage rate (mm/s) | 11.7217 | 11.3245 |
| Spatially integrated cumulative slip (m²) | 1.1602398e-4 | 1.0856892e-4 |
| Slip / Dc at the prepared right tip | 9.48981 | 9.22408 |
| Slip / Dc at x=5.75 m | 71.5917 | 67.7089 |
| Right connected contour advance (micrometres) | .612735 | .501525 |
| Right contour maximum's y-coordinate shift (micrometres) | 3.06264 | 2.50678 |
| Left contour advance (nanometres) | 17.8074 | 16.3380 |
| Added fracture inventory (J/m) | 2.429574e-7 | 2.223095e-7 |
| Signed physical-work error (J/m) | 6.486457e-13 | 4.916014e-13 |
| Accumulated absolute work error (J/m) | 6.492695e-13 | 4.925406e-13 |

The right-front law difference is .111210 micrometres, or 3.55872e-6 cell
widths. It must be compared with the local timestep sensitivity below.
The native reducer uses its existing 1 nm tied-extrema tolerance as a numerical
motion-detection floor, separately from .001/.01/.05-cell growth thresholds.
The first coarse endpoints exceeding that floor are 5.20 ms and 5.44 ms,
following stationary endpoints at 5.12 and 5.36 ms. These are **detected discrete
contour-motion brackets**, not bounds on a continuum crack-onset time.
At 5.20 ms the weakening displacement is only 7.02 nm; at 5.28 ms it is .10082
micrometres. Small changes of detection level can therefore move the reported
first endpoint. The local refinement is needed to assess the time sensitivity.

At the common 4.8 ms parent, both right contours are unchanged and their outer
stage-row deficits are .0192832/.0275971 Pa. The law excess .00831392 Pa is
30.13% of the control deficit, but only 4.12% of the control's change from its
initial .229197 Pa deficit. Shared loading supplies most of the approach to
activation. The main rate transient precedes this cap response; continued
boundary driving prevents an autonomous-event interpretation.

Advance values here are changes of global x-extents; the y shifts are coordinate
changes, not displacements projected onto a material normal or tangent.
The right lower-half contour extent remains unchanged. No new formed section
appears and every old formed-plane angle remains exactly unchanged. The y
shift of the contour maximum is a change of the diffuse shape, rather than a
newly selected turning angle. Exactly intact material remains unchanged.
These subcell displacements alone establish no spatially qualified crack growth.

## Completed local timestep and restart check

Each law restarts from its own unchanged accepted step60 at 4.8 ms, retaining
the complete coarse prefix. The local check requested twenty 40-microsecond
intervals through 5.6 ms. Both runs ended normally at their native wall-time
limit after **sixteen intervals through 5.44 ms**, in 2166.52/2254.48 s.
The 35-minute limit is checked at accepted boundaries, so the final interval
can extend beyond it. Neither a zero exit code nor the planned endpoint is
used to claim twenty completed intervals. No additional continuation was run.
Integrator, physical inputs, quadrature and all acceptance tolerances are
unchanged.

Every recorded free-phase KKT satisfies the existing 1e-7 gate: maxima are
7.69e-8/8.70e-8 on the full coarse pair and 1.14e-8/9.12e-9 on the local
fine pair. Coefficient increments are nonnegative at every retained interval.

| Common endpoint 5.44 ms | Weakening | Matched b=a |
|---|---:|---:|
| Right advance, dt80 microseconds (micrometres) | .202168 | .0931956 |
| Right advance, local dt40 microseconds (micrometres) | .184016 | .0680390 |
| Fine minus coarse advance (nanometres) | -18.1524 | -25.1565 |
| Maximum phase coefficient difference | 7.13603e-7 | 5.58467e-7 |
| Phase L2 difference (m) | 1.14351e-8 | 5.11074e-9 |
| Difference / full phase-increment L2 | .1101% | .05353% |
| Maximum endpoint section-slip difference (nanometres) | .650219 | .581816 |
| Maximum endpoint section-age difference (nanoseconds) | 22.3099 | 30.0716 |
| Maximum interval-average rate difference (m/s) | 3.50617e-7 | 4.54654e-7 |
| Local added fracture, coarse (J/m) | 3.388117e-8 | 3.259535e-8 |
| Local added fracture, fine (J/m) | 3.383047e-8 | 3.259600e-8 |
| Local signed work error, coarse (J/m) | +7.25230e-14 | +6.90735e-14 |
| Local signed work error, fine (J/m) | -7.48651e-15 | -4.40179e-15 |
| Local accumulated absolute work error, fine (J/m) | 8.13695e-15 | 6.19842e-15 |

The law difference is .108972 micrometres on the coarse calculation and
.115977 micrometres on the local fine calculation: a 7.00419 nm, **6.43%**
change. Its sign survives. The sum of the individual front changes is 43.31 nm;
this is measured local sensitivity, not a rigorous error bound. Fine detected
motion brackets are 5.16..5.20 ms and 5.32..5.36 ms, reducing the difference
between first detected endpoints from .24 to .16 ms. Thus the ordering is
supported locally, while a precise onset-time difference is not qualified.
Both histories remain below every .001/.01/.05-cell advance criterion.
The common 0..4.8 ms prefix and spatial discretization have not been refined.

Rate and resistance comparisons average two fine midpoint observations over
each matching coarse interval; endpoint phase, age and slip compare at equal
physical times. They are not comparisons of coincident midpoint stages.
The local fracture partition is formed relative to the actual 4.8 ms parent;
its inventory identity error is at most 7.69e-16 J/m. The smaller physical-work
error accompanies a state comparison, rather than replacing one.

The restore observer records accepted60 at 4.8 ms/publication60, with saved
pending61 at 4.88 ms and dt80 microseconds. The existing callback schedules
pending61 at 4.84 ms with dt40 microseconds; the first solved clock agrees.
Restored phase history and the native FE solution agree exactly. Rebuilding
the input's actual 6x6 Gauss rule gives coordinate error below 9e-15 and phase
reconstruction error below 1.12e-15. Full 3D deviator traces are below 3e-12 Pa
for tau and exactly zero for A. Endpoint age, cumulative slip, section measure,
accepted rate and force match the parent exactly, and every copied parent
checkpoint file has the source SHA256. No history is reset or re-equilibrated.
The initial-phase reference rebuilt for reporting is read-only and does not
replace the physical obstacle. No deliberate new rejection or birth test is
claimed.

The independent signed-rate audit uses native volume measures and the matching
section ds. Positive and negative volume integrals agree with the native
section observations within 6.8e-20 and 6e-21 m²/s, respectively. Tau/A, phase,
age and slip are endpoint data; stored local deformation/reaction belong to the
last midpoint stage, here 4.76 ms at the restored parent. Confusing those clocks
would manufacture a restart mismatch.

## Spatial distribution, closure limitation and interpretation

At 6.08 ms the maximum phase-coefficient increases are .000116706/.000107918.
Exactly intact and remote coefficients remain **exactly unchanged** throughout,
with one connected sampled component and no detached absolute phi=.05 contour.
Right wholly-ahead fracture row partitions are 1.63759e-9/1.61236e-9 J/m;
front-straddling partitions are 1.86940e-9/1.77381e-9 J/m. Left wholly-ahead
partitions are zero, and left front-straddling values are 1.44206e-11/
1.32499e-11 J/m. All regional rows sum to the total change within 2.34e-15 J/m.
Most added fracture inventory is in the old band. The nonzero ahead values
include the existing diffuse tail, and do not imply damage of initially intact
material. Face terms make these reproducible row partitions rather than a
unique pointwise fracture-energy density.

Final right-taper positive/negative deformation integrals are
.00183280/0 and .00183254/0 m²/s. The left-taper values are approximately
+3.18781e-5/-3.18781e-5 and +3.18485e-5/-3.18485e-5 m²/s: about
99.999997% cancellation. In the left-tip neighborhood cancellation is still
99.9988%/99.9984%; the right-tip neighborhood stays one-signed. The native
section power is the resistance times the **signed integrated** fault rate.
It is not the integral of a positive pointwise friction dissipation. Opposing
left deformation therefore remains a limitation of this nonlocal closure,
even though its work-adjoint identity passes. No clipping, hidden band
selection or elastic/viscous bulk deformation is included in these integrals.

The right profile-shaped admissible increment remains unfavorable at the final
stage (-2.13985/-2.14812 Pa). Local cap rows can evolve while a broader forward
increment still costs energy. The connected absolute contour changes by a
measurable amount within this discrete representation; its tiny size alone is
not a proof of numerical error, since a quadratic field resolves subcell level
sets. What remains missing is spatial convergence, refinement of the earlier
time history and damage of initially intact material. The different peak rates,
tip slip/state histories, coherent driving excess and local timestep ordering
support a frictional effect on **diffuse cap evolution**. They do not yet answer
the stronger Example 3 question of finite connected propagation or direction.
The rate maximum precedes detected contour motion; continual boundary driving
and the dominance of the common loading change prevent an autonomous-event
claim.

## Cost, decision and next experiment

There are 248 successful physical intervals: 64 fixed-phase, 152 in the coarse
free pair and 32 in the local fine pair. Numerical wall time, including both
failed attempts and static preparation, is 24465.05 s = **6.796 h**, with
27.183 aggregate rank-hours. Only one four-rank job ran at once; threaded
libraries were limited to one worker per rank. The late coarse free intervals
average 78.28/76.30 s, versus 133.09/138.59 s for local half steps. Thus local
halving costs about 3.4..3.6 times as much per simulated interval of equal
physical duration. It improves the cap trace and work accuracy, but does not
remove spatial or shared-prefix uncertainty. The total elapsed budget also
includes building the separate checker, reductions, documentation and retention;
the final elapsed time is recorded in `preservation.json`, below eight hours.

**Decision:** keep the selected gamma_e=.000365, phase-stable, right-taper-sliding
preparation as a controlled Example 3 validation specimen. Close the present
driven history at 6.08 ms and retain both laws, failures and all restart parents.
Do not extend it simply to obtain visible growth, or call it a qualified turning
or finite-propagation example. Zero purposeful specimen revisions were used.
The work errors are small compared with added fracture inventory, so that ledger
is not the obstruction to the next decision.

**Next justified experiment, not launched:** one matched pair on 256x256 with
unchanged axial spacing and surface labels, halving normal spacing to .015625 m.
Exactly prolong the inherited Bernstein lower obstacle, preserve the physical
plane geometry, and repeat zero-clock preparation at the same load on that mesh;
then run from fresh t=0 with dt80 microseconds through 5.44 ms. Compare the
prepared margins, profile, physical tau/A, signed rates and the law difference
in cap advance with this 256x128 result. Keep the physical model, six-point
quadrature rule and criteria fixed, and budget from the first accepted intervals.
This addresses the remaining profile/cap sensitivity without another preload
scan, a long driven continuation or checkpoint conversion. Demonstrating an
autonomous event would separately require a controlled loading-history test.

## Reproduction and visualization

The [benchmark reading path](../../../benchmarks/reconstructed_fault/particle_free/stage4_preparation/README.md)
describes the preparation, frozen checkers, run labels, compression, analysis
views and restart audit. Each retained case has its exact input, native command,
log, scalar histories and SHA256 identities; `preservation.json` lists the raw
run locations and checkpoint hashes. Raw observations and expanded checkpoint
payloads remain in this worktree's build tree. Only diagnostic duplicates were
losslessly compressed after verifying their decompressed hashes. Earlier cleanup
losses remain as documented; no deleted historical run was recreated.

| Evidence | Contents |
|---|---|
| [Preparation](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/preparation.png) | Phase stability, tip margins and actual initial sliding |
| [Complete free histories](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/free76-growth.png) | Rate transient, both contour motions, phase change, fracture and physical work |
| [Tip profiles](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/free76-profiles.png) | Prepared phi0, final phi and delta-phi through both tips on their prepared planes |
| [Tip map](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/free76-tip-map.png) | Absolute phi=.05 contours and spatial phase increments |
| [Slip/state comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/free76-local-slip.png) | Tip/taper slip and state by physical station |
| [Local time comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/half-time-check.png) | Coarse/fine contour, phase, slip and physical-work differences |
| [Signed deformation](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/half-signed-deformation.png) | Opposing left deformation and one-signed right deformation at the retained parent |
| [Machine-readable comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/free76-comparison.json) | Driving, resistance, signed section integrals and common-time differences |
| [Growth data](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/free76-growth.json) | Contour detection brackets, energy partitions and limits |
| [Local time data](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/half-time-check.json) | Matched endpoint/rate errors and local work budgets |
| [Restart/measure audit](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/half-restore-audit.json) | Native quadrature, signed measure identity and parent checkpoint verification |
| [Cost](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/cost.json) | Actual intervals, elapsed numerical time and solver effort |
| [Preservation manifest](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/preservation.json) | Protected runtimes, input/checker identities and source preservation |
| [Final checks](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preparation/final-verification.json) | Entry-hash, acceptance/bounds, local restart, report-link and Python checks |

PNG figures have PDF companions; per-law CSV files retain every accepted
observation. `reduction-commands.json` records the exact five final reductions.
Joined `free76` and `half` directories are labelled analysis views, not numerical
runs or checkpoints. These results qualify the benchmark preparation and a
bounded local comparison only. They neither test the current manual production
source against HEAD nor change production readiness or the closed mature-fault
baseline.

## Source and runtime identity

Root/branch: `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`; HEAD
`b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`, preserving incoming manual changes.
The main executable is the protected `first-free-turn/tested/aspect-release`,
SHA256 `089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`.
Its banner revision is `5f1fc6f3c`; its identity is independent of HEAD. The
frozen campaign checker SHA256 is
`a176d36184fdc600d812f37e34a0b5251ce4a43d1ae568980fa0efad9713fd6c`.
The successful preparation checker is separately retained with SHA256
`d01670eefe18072a355cff4bf1f83533add0229d9883463c36ed45f396fa3296`.
The new checker and exact commands live
under `build-particle-free/stage4-preparation/`; benchmark support is
[`stage4_preparation`](../../../benchmarks/reconstructed_fault/particle_free/stage4_preparation/README.md).
No production source, constitutive law, checkpoint meaning or prior runtime is
replaced. Earlier cleanup losses remain unchanged. No commit is requested.
