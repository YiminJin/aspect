# Local acceleration and moving slip-front diagnostic — 2026-10-15

**Both coupled runs pass correctness and produce a bounded accelerating slip
pulse. The accuracy exit is not met:** the one tighter-timestep comparison has
material common-time field and front-timing differences. The endpoint pressure
feature also remains spatially unqualified. No further variation is launched.

Baseline `fault-propagation` at `c33edaaeb145e5d3dce155bb815552ef80efc5f9`, with all
incoming dirty work, P2/stationary BP3, 4×4 and 5×5 evidence preserved. Provenance,
actual artifacts, source snapshots/diff, numerical assessment and figures are in
`build-bp3-native/slip-front/`. Exact launch commands, environment, logs, exits
and measured times are `refactoring_boundary/evidence/bp3-slip-front-*.{json,log}`.
Core executable is the unchanged `managed-coupled/final/aspect-release`, SHA256
`bf6f605431cddf13ccc5dab8db1814a915c21b3220ce45ab7888258105be333d`.
Local compiler/dependencies: GCC12.4.0, deal.II9.6.2, Trilinos14.2.0, p4est2.8.7;
ASPECT3.1.0-pre, one MPI rank per run, at most two simulations concurrently.

## Explicit diagnostic initial data

This is **not official BP3 initialization or a first-event test**. The isolated
`ASPECT_BP3_LOCAL_FRONT_TEST` plugin/observer build removes the old bottom-test's
forced uniform strengthening. It retains the ordinary horizontal friction
assignment, with `Weakening region length=0` and the original 3-km transition:

- On this L=1154.70054-m fault, a(s)=.010+.015s/3000, b=.015: weakening for
  s<1000m, strengthening over the final154.7m. The truncated local transition
  is explicitly admitted; it is not rescaled to fit the box.
- The common particle/surface initializer selects
  Theta0(s)=(Dc/Vinit) exp[ln(10) exp(-((s/L-.35)/.25)^2)]. Its e-fold half-width
  is288.675m and centre404.145m. Initial nominal Omega rises toward10.
- Initial fault shear is explicitly prescribed by nodal values of
  sigma0 mu(Vinit,Theta0)+damping Vinit, then frozen. It ranges29.735–32.792MPa;
  normal background remains50MPa. This is a prescribed diagnostic traction,
  without recalibration to the computed FE solution. The maintained ordinary
  plugin actually keeps uniform BP3 prestress; older weak-equilibrium
  calibration belongs to the historical reference implementation.

Keep physical plate/initial reference velocity1e-9m/s, Dc=.008m, G=32038120320Pa,
eta=1e26Pa s, length scale20m, peak phase.6, damping4624440Pa s/m, full bottom
velocity and effective Helmholtz filter20m. Retain the translated60-degree
2000×1000m geometry,4136 graded Q2/Q1 cells,149538 DoFs,59 surface nodes at
19.90863m, native3×3 mechanical quadrature and general consistent normalization.
Phase/mesh/geometry are fixed. Native RK2,4×4/min12/max24 and population management
stay enabled; no density sweep or new5×5 run.

Particle stress remains authoritative, with the existing non-objective law:

  tau_p(new)=beta tau_p(old)+2 eta_ve [dev3(symgrad u_h)(x_p)-chi_p V_h S_p],
  beta=exp(-G dt/eta), eta_ve=-eta expm1(-G dt/eta), chi=h(phi)/I_h.

Current strain comes from FE velocity. Old particle stress is neither replaced
by FE history nor deviated again. Surface theta and cumulative slip publish once;
mature cohesive traction is zero and H remains inert. Bulk perturbation history
and slip start at zero; background traction never enters bulk history.
The initial constitutive interval stays200000s in both runs. Step zero produces
no slip or stress-history increment. Its computed V is.9801–1.0363Vp, shared
exactly by the pair; prescribed nodal resistance is not an exact weak initial
root for arbitrary represented fields. All initial phase, mesh, quadrature,
particle checksum, theta and background-input files match byte for byte.

## The single targeted comparison and actual evolution

The primary profile stays smooth while acceleration makes the state controller
active. Choose a temporal comparison: reduce its **predictive logarithmic state
bound .2→.1**. Keep native convection/reconstructed-fault controllers, CFL=.5,
first/global caps.2e6/2e6s,91% growth, all physical data and solver tolerances.
No function timestep is active. This is a tighter controller, not a uniform
halving of every accepted dt. There is no convergence-order claim.
Both inputs request T=2e7s; primary/comparison bounds are120/240 positive steps,
1200/1800s internal walls and1320/1920s external limits. Checkpoint/resume and
unexpected-failure retry/cutback are disabled.

| Actual result | Primary (.2) | Tighter (.1) |
|---|---:|---:|
| Positive steps, T=2e7s reached |41, yes|67, yes|
| Runtime [s], exit |851.45,0|1169.73,0|
| Positive dt range [s] |126216–1573586|103659–838336|
| Peak V/Vp |21.2534|20.1062|
| Sampled peak time [s] |12235670|11214093|
| Minimum theta [s] |420968|441249|
| Maximum actual abs(log(theta_new/theta_old)) |.27172|.14554|
| Maximum predicted log-state change |.20000|.10000|
| Maximum step slip/Dc |.42242|.29487|
| Maximum normalized nonlinear residual, target<1e-8 |5.451e-9|4.903e-9|
| Maximum surface RMS [Pa] |.01686|.00426|
| Newton / Krylov totals, including step zero |127 /6218|200 /9243|
| Maximum particle Maxwell defect [Pa] |1.245e-8|9.955e-9|
| Maximum theta relative defect |2.221e-16|2.221e-16|

Each run exhibits increasing V, a finite peak and subsequent decay; this is a
slow accelerating slip pulse, with no demonstrated dynamic-rupture threshold.
The spatial pattern is measured with slip-rate threshold contours on the fixed
fault, not by changing phase/geometry. All59 nodes cross2Vp;5Vp reaches s=796.345m
and10Vp reaches537.533m in both runs. For example, the2Vp arrival at the lower
endpoint shifts13.29988e6→12.30275e6s; accepted interpolation brackets are
.17815e6/.18536e6s. The5Vp arrival at796.345m shifts13.46800e6→12.44742e6s.
Sampled peak amplitude changes−5.40%, and peak time−1.02158e6s (−8.35%).

Primary final ranges: V=.8574–1.4640Vp, theta=3.435–5.654e6s,
Omega=.4001–.8441, slip=.02652–.09450m. Tighter final ranges:
V=.5781–1.1460Vp, theta=3.990–5.864e6s, Omega=.3276–.7849,
slip=.02659–.09274m. Omega initially peaks9.9843; it remains finite and positive.
All41/67 accepted terminal publications, including every particle and derived
sample, pass the original recurrence/state/slip, residual and frozen-field gates.
Domain coverage and CPDI gradient defects remain<=1.60e-15. I_h is unchanged,
finite and positive (minimum12188.324m). The selected off-trace G checks pass
at their unchanged2e-7 relative criterion (worst4.57e-8).

## Common-time sensitivity

Final fields below are at the **exact same accepted endpoint**, on identical
physical nodes. Scales: Vp; Dc/Vp=8e6s;1; primary final maximum slip.094503m;
and50MPa for all stress contributions. RMS uses the retained Q1 surface metric.

| Final field | Max absolute difference | Max/scale | RMS/scale |
|---|---:|---:|---:|
| V |3.51078e-10m/s|35.11%|28.99%|
| theta |799753s|10.00%|7.19%|
| Omega |.105544|10.55%|7.62%|
| Slip |.00176094m|1.86%|.88%|
| Shear |.15130MPa|.303%|.112%|
| Raw normal |.25784MPa|.516%|.067%|
| Filtered normal |.16274MPa|.325%|.062%|
| Pressure contribution |.16961MPa|.339%|.053%|
| Deviatoric normal contribution |.08823MPa|.176%|.022%|

Initial discrepancies are zero. Transient discrepancies are larger: maximum
common-time V difference9.1474Vp (43.0% of the primary pulse peak), theta3.436e6s,
Omega1.474, slip.019687m, shear1.670MPa, raw normal2.839MPa, filtered1.745MPa,
pressure1.887MPa and tensor normal.952MPa. These maxima occur at different times.
The primary accepted times are matched by linear interpolation of finer saved
fields, with brackets recorded. At the largest V discrepancy, the finer bracket
is106172s and the linear/quadratic interpolation indicator is.01818Vp, much
smaller than9.1474Vp. That indicator is not an error bound or convergence order.
The physical timing discrepancy is substantial even though final slip is closer.

The controller predicts theta change using the preceding committed V; actual
accepted state uses the new V. Its .2/.1 contracts therefore do not bound actual
changes during acceleration. The measured .2717/.1455 changes identify this
limitation without a recurrence failure. The comparison includes RSF splitting,
bulk history evolution and transport/reconstruction in time; it does not isolate
all sensitivity to one of those mechanisms. No law, controller or tolerance is
modified to improve agreement.

## Endpoint/interior disturbance diagnosis

Both retain66176 particles: **zero cell crossings, births, removals, outflows
and multiple-trace events**. Particle movement/domains are real, but displacement
is too small to exercise those events. The old host/average trace discrepancy
is only<=1.48e-8Pa here. Thus population/eligible-trace changes cannot explain
this pulse's disturbances; this is not a qualification of those mechanisms in
accelerating flow. Their earlier moving/population evidence remains preserved.

Final local-chord roughness, primary/tighter:

| Region / field | Primary | Tighter |
|---|---:|---:|
| Lower200m V/Vp |.01057|.00832|
| Interior V/Vp |.002515|.001471|
| Lower200m raw normal [MPa] |3.68595|3.69018|
| Lower200m filtered normal [MPa] |.26015|.26420|
| Interior raw /filtered normal [MPa] |.01489 /.00873|.01432 /.00845|

The lower-end feature is localized in the final interval. Primary final projected
pressure at s=1134.792/1154.701m is−1.576/−8.719MPa; tensor normal is−.317/−.851MPa.
At the penultimate node, raw chord3.68595MPa comprises pressure3.62774MPa plus
tensor.05820MPa. This is chiefly a sharp pressure drop adjacent to the full-bottom
boundary, not evidence of a resampling jump. Raw/filtered normal remain monotone
there; pressure has a small penultimate reversal, and V one small final-interval
rebound. The large chord must not automatically be labelled an alternating wave.
No growing alternating slip-rate ringing is observed during this short pulse.
That observation does not establish a cure or accuracy of the endpoint feature.

Raw and effective normal stay compressive: primary final ranges40.430–50.062MPa
and44.558–50.055MPa; tighter40.415–50.060 and44.720–50.054MPa. Pressure and tensor
are retained separately. Final lower-window cancellation ratio is1.0945/1.0959;
the dominant last-row pressure drop is not hidden by extreme tensor cancellation.
Existing three interior normal cuts show pressure continuity to roundoff and
finite one-sided constitutive normal-stress jumps; they do not sample the
endpoint itself. Boundary completion, FE gradients, quadrature and reconstruction
remain possible spatial contributors requiring a fixed-state accuracy audit.

The raw spectral maximum remains73.4m (only3.69 surface intervals), already
present at initialization. Its windowed global amplitude is6.19/5.95kPa, despite
the much larger endpoint chord: the Hann window suppresses endpoints. Chord and
spectral quantities are roughness diagnostics, not independent error estimates.
The endpoint feature persists under tighter timesteps; spatial accuracy stays open.

`disturbances.csv` aligns roughness, V, actual/predicted state changes, dt and
population counts. Event correlations are undefined because every event count
is zero. Lower-V roughness changes correlate negatively with dt (−.651/−.592),
while raw-traction roughness changes have weak dt correlations (.039/−.039).
These are descriptive: the controller responds to the evolving state. The raw
endpoint feature grows through accumulated slip and persists after the V peak;
correlation with adaptive dt does not establish a numerical cause.

## Implementation, verification and disposition

Only four maintained plugin files change, under the isolated compile definition:
`bp3_model.h` (common diagnostic theta), `bp3.cc` (diagnostic fixed initial shear
and compact initialization CSV), `monitor.cc` (reuse Local state disturbance),
`configuration.cc` (explicit diagnostic identity). Two PRMs, the analysis script,
this report and two status notes are added/updated. The existing coupled observer
is reused unchanged. No core algorithm, history owner, geometry, population rule,
sampling rule, production parameter/default or solver tolerance changes.

Build with `-DCMAKE_CXX_FLAGS=-DASPECT_BP3_LOCAL_FRONT_TEST` in the isolated
`slip-front/plugin` and `slip-front/observer` directories; the observer additionally
uses `-DBP3_TRANSITION_CHECK=ON` and links that exact BP3 library. Both initial
and final styled plugin/observer builds pass. The final one-positive-step gate
exits0 and matches every initial/step1 profile column exactly. Final source differs
from the trajectory source only in whitespace. ASPECT astyle2.04 new-code checks,
Python syntax/analysis and whitespace checks pass. Twelve compiler-preprocessed
comparisons preserve all tokens (including strings) in ordinary, old bottom-test
and old oscillation branches. No full core rebuild, complete unit campaign,
MPI transport, restart, cutback or full-model run is claimed.

The three actual launches total2050.86s (34.18min), excluding compilation, within
the local budget. Original source/artifact preservation is recorded in
`preservation.json`. No checkpoint-dependent continuation or extra physical
variation is used. Figures: `fields.pdf`, `front-and-timestep.pdf`; compact native
endpoint snapshots: `endpoint-profiles.csv`; full metrics: `assessment.json`.
The constant50MPa normal background is fixed exactly in the input property;
its diagnostic mass-solve projection varies by roundoff, checked against
64 epsilon times50MPa, rather than requiring bitwise equality of that projection.

The next proposed task is temporal resolution of this same pulse, consistently
tightening the active restrictions and retaining sparse complete states for a
subsequent endpoint-pressure audit. Do not change particle density or physics to
mask these differences. This task stops with an observed accelerating/slipping
regime and a failed accuracy qualification, not a resolved BP3 first event.
