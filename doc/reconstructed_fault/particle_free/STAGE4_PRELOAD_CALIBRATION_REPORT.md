# Stage 4 native preload calibration and tip discrimination

**Decision: close preload tuning; do not run the conditional phase-release
pair.** The selected near-threshold preparation separates slip rates but gives
no weakening-specific excess at the tip. Every checked axial-tip row has
slightly lower driving in weakening than in b=a at every accepted stage.
The next experiment should bring the prepared slipping state toward the formed
tip, rather than raise the common preload or lengthen this trajectory.

This task varies one elastic preparation amplitude, measures admissible damage
directions using the native mechanical response, and compares weakening with
matched b=a at one common subcritical preparation. It follows the
[completed phase-release pair](STAGE4_PHASE_RELEASE_REPORT.md). The physical
solver, constitutive laws, midpoint update, native Bernstein phase, quadrature,
pressure convention and acceptance tolerances are unchanged.

## Preparation and native calibration

The sole varying parameter is gamma_e in the affine elastic boundary
displacement

    u_x = gamma_e (y-2) - k gamma_e (x-4),
    u_y = k gamma_e (y-2),       k = .41954981558864.

The existing native static solve equilibrates true stress, cohesive memory and
pressure at the exact retained phase and common initial section resistance.
The subsequent boundary velocity stays at .01/s shear with the same extension
ratio. Material data, section age and nominal slip-rate preparation, seed,
formed directions and the 1% sin² age perturbation over x=2.5..5.5 m are held
fixed. This is a family of fresh synthetic initial states, not continuation of
an old checkpoint or a replay of an unrecorded loading history. Phase is
inherited, not re-equilibrated as gamma_e changes.

Three four-rank static preparations at gamma_e=.0002, .0005 and .0008 completed
normally. They use zero physical intervals and pass the existing native mixed
equilibrium, local traction, static work and section-friction inversion gates.
The parameter is calibrated against these mechanical solutions, not against a
ratio of the earlier 288 Pa effective force to the 1000 Pa strength.

At fixed phase and prescribed section resistance the static mechanical problem
is linear: its physical stress predictor is affine in gamma_e. Thus the driving
of a fixed damage direction on formed planes is quadratic in gamma_e. The
three native solves determine that quadratic, and a fresh solve at the selected
amplitude checks its prediction. This does not approximate the subsequent
nonlinear time-dependent solve by that polynomial.

## What constitutes the damage margin

For a nonnegative admissible phase increment chi, positive

    D_net = integral chi Y dx - chi . gradient E_fracture

means that the reduced functional initially decreases. The full native
AT1/SIPG gradient supplies the second term, including consistency and penalty
faces. Y is the existing eliminated local driving. At t=0 it uses the
equilibrated predictor T=tau+2G A and resistance f:

    Y = H'(phi) (T:S-f)^2 / (2G),
    H = m phi(1+phi) / [m phi(1+phi)+(1-phi)^2],  m=4.

It includes the intact right limit, rather than setting Y=0 wherever A=0.
After physical intervals, the measurement uses the actual accepted midpoint
stage. Endpoint slip, phase and energy keep their distinct times. The existing
one-sided finite-difference checks of the smooth probes remain enabled.

The new scan assembles integral N_i Y and integral N_i for each of the nine
nonnegative Bernstein directions on each cell, and pairs them with the native
fracture-gradient row. A positive row below the physical upper obstacle is an
admissible infinitesimal damage mode. It is not, by itself, a finite crack
extension or a new phi=.05 component. The artificial fixed-phase upper bound
does not redefine physical admissibility.

Smooth probes are nonnegative combinations of those same rows. Their axial
half-width is .0625 m and their transverse support is y=1.875..2.125 m. Three
useful centers are x=6.03125 (spans the old tip), 6.09375 (immediately beyond the
initial .05 contour), and 6.375 (entirely intact band). The original rightmost
connected .05 position is x=6.00205437994 m. The very small represented tail
through x=6.3125 is retained rather than clipped away.

For wholly virgin sections phi=A=V=f=0, the existing direction objective at
zero elapsed time reduces to maximizing the section integral of
[u sin(2 alpha)+v cos(2 alpha)]², with u=-(tau_xx-tau_yy)/2 and v=tau_xy.
The observer evaluates its analytic maximizer from the section moments without
publishing a new plane. It also integrates the pointwise envelope
m(u²+v²)/(2G), which bounds every common section orientation. Formed planes
remain unchanged; later observations use actual native plane selection.
Selected-plane interpolation is predictive only if the maximizing plane varies;
the envelope remains quadratic in the affine native stress response.

## Why a broad-probe threshold is insufficient

The native selected-plane mean net driving, in Pa, is:

| gamma_e | Tip-spanning probe | Immediately-ahead probe | Fully intact probe |
|---:|---:|---:|---:|
| .0002 | -2.17449 | -1.80793 | -1.84356 |
| .0005 | -1.17705 | -.85404 | -.90534 |
| .0008 | +1.01662 | +1.21800 | +.87916 |

The corresponding zero estimates are .00068057, .00064385 and .00066968.
However, individual intact rows near x=6.328125, y=2.109375 reach zero around
.0006327; the conservative all-orientation envelope reaches zero at .00063212.
Thus calibrating solely to failure of the broad tip-spanning shape would
already admit detached intact damage farther ahead. The final report must not
describe that preparation as a clean extension experiment.

The opposite averaging problem also occurs: individual increments in the
existing taper become favorable much earlier than these broad averages. Some
off-axis tail rows are already favorable at the original .0002 preparation,
consistent with the small adjustment in the previous free pair. In the two
cells containing the axial end, the first of their 18 Bernstein rows crosses
zero at **.0003753278135**. Its cell center is (6.015625,1.984375), local index 0;
the left-end mirror agrees to 3.1e-13 in gamma_e. The row's support includes
part of the existing diffuse cap and material beyond it. Its infinitesimal
activation is a local cap response, not proof of appreciable connected advance.
The larger smooth-probe thresholds must not conceal it.

At gamma_e=.0002, favorable rows already occur in the central band (maximum
.1325 Pa) and old taper (1.3660 Pa), while all 18 axial-tip-cell rows are
unfavorable (maximum -.4757 Pa). The first unconstrained response is therefore
band/taper readjustment, not nucleation in fully intact material. As load rises,
local axial-cap modes become favorable before detached intact rows; a broad
tip-spanning increment becomes favorable later still. These distinct events
must not be collapsed into a single "fracture load."

These are native first-variation thresholds on the retained mesh, not a
mesh-independent critical load or a measured propagation onset. In particular,
the inherited seed already permits some band/taper readjustment. Neither a
favorable row nor a negative broad average establishes the eventual free
response. The fixed-phase solve reports zero phase KKT because its artificial
upper and lower bounds coincide; this must not be cited as physical damage
stability. The unconstrained directional measurements supply that information.

The common comparison therefore selects **gamma_e=.000365**, with extension
.41954981558864 gamma_e. All 18 rows in those axial-tip cells are initially
unfavorable; the limiting row has net driving **-.040333635 Pa**. Fully intact
material farther ahead also remains unfavorable. The fresh native preparation
verifies this margin. This is a deliberately local stability statement: the
inherited band and off-axis tail are not claimed to be globally phase-equilibrated.

## Law comparison and decision

Both laws completed all eight fixed-phase midpoint intervals of 8e-5 s, to
.00064 s, using four ranks sequentially. Neither required a retry or changed
tolerance. The weakening/control elapsed times were 974.9/956.0 s; static
calibration took 233.9 s. Total numerical time was about **36.1 minutes**.
The three static preparations' affine-response prediction is independently checked
by the selected fresh native states: the maximum fixed-plane row-density error
over the entire mesh is **2.40e-11 Pa**. Initial margins match across laws to
roundoff. The age perturbation is identical; the corresponding instantaneous
rate changes differ because the friction laws differ, as intended.

| Quantity | Weakening | Matched b=a |
|---|---:|---:|
| Final stage time (s) | .00060 | .00060 |
| Peak stage rate (mm/s) | 2.13458 | 1.64934 |
| Integrated stage rate (m²/s) | .004570115 | .003795009 |
| Integrated endpoint slip (m²) | 1.197190e-6 | 1.088788e-6 |
| Mean patch resistance (Pa) | 126.50268 | 126.53454 |
| Limiting tip-row net driving (Pa) | -.040028970 | -.040006376 |
| Smooth tip-spanning net driving (Pa) | -1.77366157 | -1.77364116 |
| Immediately-ahead probe net driving (Pa) | -1.42141918 | -1.42139978 |
| Fully intact probe net driving (Pa) | -1.41530759 | -1.41531540 |
| Maximum net driving over all exactly intact rows (Pa) | -1.27643801 | -1.27642025 |

Weakening increases final peak rate by **29.42%** and integrated slip by
**9.96%**. Yet its limiting tip row has **22.5943 microPa less driving**, only
.0564% of the remaining approximately .040 Pa deficit and with the opposite
sign to the desired promotion of tip damage. The smooth tip-spanning and
immediately-ahead differences are likewise negative, -20.4117 and -19.3996
microPa. Across all 36 rows in the two axial-tip cells at each end and all eight
stages, every weakening-minus-control difference is negative; the maximum
absolute difference is 24.1696 microPa. No such row becomes favorable in either
law. The complete histories, rather than only their endpoints, support this
decision.
The final scan also checks the wider immediate cap x=6..6.15625,
y=1.875..2.125: every row there has a negative weakening-minus-control
difference, ranging from -24.17 to -15.37 microPa. This guards against selecting
only the limiting row while missing a different near-tip positive excess.

The intact probe farther ahead does acquire a small **positive** weakening
excess, +7.8075 microPa at the final stage. That is not connected-tip promotion:
its net margin remains about -1.415 Pa. All exactly intact rows and all remote
rows remain unfavorable. Three favorable old-tail rows remain in the right
tail mask, with maximum net driving about +1.379 Pa; they are existing low-phase
readjustment modes, not concealed intact damage or evidence of a globally
equilibrated seed.

Thus the conditional release criterion is not met. The comparison rejects
**preload adjustment alone** as the next way to make this preparation
distinguish the laws at the tip. It does not prove that the laws would have
identical free responses indefinitely. No phase-release run or further preload
case was launched.

One concrete next preparation is to move the three-metre smooth perturbation
support from x=2.5..5.5 to **x=3.0..6.0 m**, and prepare a compatible slipping
state that remains appreciably active into the formed taper near x=5.5..6.0.
Moving only the age bump while retaining a nearly quiescent tip state may be
insufficient. Keep material parameters, Dc, mesh, dt and gamma_e=.000365 fixed;
re-equilibrate the new common stress/cohesive preparation and verify its
connected/intact margins before one short fixed-phase law pair. This changes
the initial slipping-state location, not the constitutive law or a checkpoint
label. It is the next proposed experiment, not an additional case in this task.

## Work, bounds and qualification

Every native phase coefficient is bitwise unchanged at every recorded stage.
The inherited coefficients range from -2.71e-19 to .6; the roundoff-sized
negative coefficient and tiny tail coefficients are retained, not clipped.
"Exactly intact" uses literal all-zero cell coefficients and does not silently
reclassify these small inherited values.
Consequently connected contour advance and added fracture inventory are exactly
zero **by the fixed-phase constraint**, not a prediction of the free response.
Both probes and all native rows use the same tested quadrature and SIPG
gradient. The static equilibrium residual is below 8.45e-12, static relative
work defect below 3.18e-14, and local traction defect below 1.47e-12 Pa.
The smooth-probe one-sided derivative checks pass without relaxed gates.

The physical-work error is Wext-delta E-Db-Dc-Df, before cohesive or residual
ledger corrections. Signed cumulative / accumulated absolute values are
**+8.89365e-11 / 8.89365e-11 J/m** for weakening and
**+7.37704e-11 / 7.37704e-11 J/m** for b=a. Absolute external work is about
.1239824 J/m, giving relative defects 7.17e-10 and 5.95e-10. Added fracture work
is zero here, so no error-to-added-fracture ratio is defined. These balances do
not qualify temporal accuracy or a free-growth calculation.

Maximum accepted mechanics residuals are 1.078e-11 and 1.051e-11. Signed local
deformation is retained: final positive/negative integrated rates are
.004574763/-4.647915e-6 m²/s and .003796572/-1.563373e-6 m²/s, respectively.
Their cancellation fractions are .2030% and .0823%; no clipping removes the
opposite-sign contribution. Accepted-state publication checks pass throughout.
Checkpoints are preserved in place, but no restart or additional refinement
test is claimed.

## Figures and compact data

* [Preload calibration and distinct activation thresholds](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preload/calibration.png)
* [Slip-rate and tip-margin histories, including signed law differences](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preload/law-comparison.png)
* [Spatial row-margin map and ahead-tip probes](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_preload/margin-map.png)

Each has a PDF companion. `calibration.json`, `comparison.json`, the two
`*-history.csv` files and compressed arrays retain the measurements. The exact
inputs, run commands/logs, original per-step scans and checkpoints are preserved;
`preservation.json` identifies the tested checker, protected runtimes and
checkpoint hashes. Full histories remain under `build-particle-free/bulk-tip/runs/`.

## Reproduction and preservation

The benchmark is in
[`stage4_preload`](../../../benchmarks/reconstructed_fault/particle_free/stage4_preload/README.md).
It builds a separate checker, reuses the protected main executable, and retains
exact commands, generated inputs and run outputs under the existing build tree.
Source root/branch are `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`,
revision `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`, with incoming manual/source
changes preserved. Executable identity is independent of HEAD: the retained
`first-free-turn/tested/aspect-release` SHA256 is
`089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`.

The generated preload header differs only in reading gamma_e from the recorded
benchmark environment. The added scan is observational; it does not change
physical histories or the native equations. No production source, default,
checkpoint meaning or earlier evidence is replaced. There is no new restart,
mesh/time convergence or production propagation qualification. Historical
cleanup losses remain as previously recorded. No commit was requested.
