# Native RSF acceleration and early cohesive activation

Completed bounded comparisons, 2026-10-30 UTC. Baseline `f2f730529` plus preserved
local portability work. **RSF feedback is demonstrated; quantitative coupled
propagation is not qualified.** The tighter replay encountered a numerical
rate-controller floor before its requested endpoint. Production/BP3, physical
laws, defaults and acceptance tolerances remain unchanged.

[Comparison PNG](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_instability/review/comparison.png) ·
[GIF](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_instability/review/evolution.gif) ·
[Uncertainty](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_instability/review/uncertainty.png) ·
[Energy](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_instability/review/energy.png) ·
[Controller failure](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_instability/controller-floor.png) ·
[Indexed CSV/JSON and provenance](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_rsf_instability/INDEX.md).
Original criteria, equations, references and failed first-step arithmetic are
preserved in `evidence/cohesive_rsf_instability/SELECTION.md`.

## Specimen and implementation

This is a finite-wall, prescribed straight-interface experiment: 16×12 m domain;
Γ on x=4–10, y=0; mature damage a=1 on x=5–8; one-metre underloaded shoulders;
initially tied material on x=9–10. Surface damage a is distinct from bulk φ.
Fresh fields prescribe equilibrated uniform shear 1345.39035 Pa, normalized age
10 (600 s physical age), zero jump/Maxwell displacement memory, and cohesive
reference −C0 a τ0. Initial damage does not relax before the pulse. Initial bulk/cohesive/fracture
inventories are173.7672/.452519/12 J/m; fracture is counted once.

| Physical input | Value |
|---|---:|
| G; Maxwell viscosity | 1e6 Pa; 1e12 Pa s |
| Gc; intact strength | 3 J/m²; 2000 Pa |
| Background confinement | 10000 Pa, with solved pressure perturbation |
| RSF a, b, μ0 | .005, .02, .1 |
| V0; Dc | 1e−6 m/s; 6e−5 m |
| Radiation impedance | 25836.0214 Pa s/m |
| Constant engineering shear rate | 1e−7 /s |
| Physical R; numerical lifting half-width | .75 m; .75 m, separate meanings |

The finite-domain native stiffness minimum is 0.81145 MPa/m versus critical
2.49996 MPa/m, including radiation. Native K/current-normal reaction/Maxwell/state
linearization has a growing mode .00207442 /s; the matched neutral control decays.
Four eigenvalues are unstable on each probed mesh; high-stiffness modes decay.
The Jacobian action check errs 2.65e−9. Pressure coupling is retained; symmetry
makes its contribution negligible for this straight mature-patch perturbation.
These steady-rate diagnostics do not predict an exact nonlinear onset.

The selected uniform 128×96 grid has Δx=Δy=.125 m, surface spacing .0625 m,
12,288 cells and 185,667 reported native DoFs. Estimated nucleation and minimum
cohesive scales are .8/.75 m; the evolving rate half-maximum width is about1.5 m.
The uniform grid is affordable for a bounded mechanism test, without established
nonlinear spatial accuracy. Quadrature and DGQ2/native FE-remainder histories remain.

Native adaptive stepping bounds slip/Dc, log age, log(|V|+V0), and damage increments;
original limits .1/.1/.2/.02, safety .8, growth1.25, maximum20 s. Typed local solver
failures use native cutback. All trials anchor accepted histories and publish once.
V0 regularizes the change indicator as well as appearing in the established RSF
law; the indicator does not alter friction. No inertia or wave-speed claim.

## Measured response and comparisons

At the common primary/control endpoint 374.578558977 s:

| Quantity | State weakening | Matched b=a control |
|---|---:|---:|
| Maximum observed slip rate (m/s) | .0446365 | 1.62959e−7 |
| Mean mature slip / Dc | 19.0776 | .785081 |
| Integrated additional damage (m) | 1.37353e−4 | 0 |
| Maximum damage increment | .00290960 | 0 |
| Fully decohered length (m) | 3, unchanged | 3, unchanged |

The control changes b to .005 and μ0 to .1345387764, preserving initial friction;
initial surface, geometry and bulk exports are bitwise identical. Acceleration
precedes **any** damage change under constant loading. During the rapid window,
bulk storage falls3.2900 J/m while boundary work is only .002100 J/m. Stored
mechanical energy and state weakening drive the pulse. Adjacent intact records
activate near x=9, remaining below a=.01; no fully decohered advance or new formed
geometry is demonstrated.
The primary stops at600 steps while accelerating; its largest speed is not a
completed pulse maximum or evidence of arrest.

The replay halves all adaptive increment limits and maximum dt from the identical
accepted step368, t=374.567041686 s. It reaches step660/t=374.578519544 s before
stopping. Earlier history is shared, not independently time-refined. Comparisons
below use that common physical endpoint, with diagnostic interpolation only:

| Primary versus tighter replay | Difference |
|---|---:|
| Integrated added damage; growth-relative field L2 | .583%; .597% |
| Mean mature slip | .0392% |
| Maximum rate at endpoint; maximum window difference / peak rate | .835%; 1.20% |
| Mean mature state at endpoint; maximum relative window difference | 7.47%; 9.04% |
| Mature-state field L2 | 14.0% |
| Cell-centre full-tensor change norm | .560% |
| Maximum Gaussian-averaged resistance difference, σ=.375 m | 1.99 Pa |

The stated1% history comparison criterion **does not pass**. Full-plane-strain
stress diagnostics include zz and both xy contributions. The replay does not
qualify absolute acceleration timing. A numerical intact-damage integral threshold
1e−8 m brackets onset at [374.567049797,374.567051404] s in the primary and
[374.567050411,374.567051316] s in the replay; these are discrete brackets on the
shared trajectory, not physical timing accuracy.

Frozen common-memory bulk comparisons retain1.58% storage spread,25.2% unfiltered
weak-load norm spread and12.53 Pa maximum averaged shear-load difference between
96×72 and128×96. Earlier point/ahead-normal failures and5.5% linear growth-rate
spread remain. Tiny local damage cannot be declared spatially qualified from
integrated work agreement. Raw tied reactions are retained, including oscillations.

## Energy and lifecycle

Primary totals: boundary work9.80457, added bulk/cohesive storage5.55176/.211803,
friction2.20164, radiation1.69916, Maxwell.133811, new fracture.000412060 J/m.
The absolute physical trapezoidal-work remainder is .00613931 J/m: **.157% of
sliding dissipation, but14.9 times new fracture work**. The fracture-relative
screen fails. Signed and absolute sums are both supplied to expose cancellation.
On the common replay window, the physical remainder decreases .00150899→.000774703
J/m, while new fracture work is .000407227/.000409615 J/m. This supports a temporal
remainder diagnosis, not completed fracture-energy qualification.

Exact BE ledger residuals remain below9.18e−14 J/m against the unchanged1e−10-scaled
bound; mechanical weak residual <=1.42e−10 Pa, surface <=9.992e−9 Pa, source scaled
<=1.39e−14. Physical work is the work-conjugate bulk virtual-work functional for
the prescribed affine boundary increment. BE and surface numerical remainders
remain separate from physical dissipation.

Forced rejection/retry and small same-rank restart retain every bulk coefficient
and surface history bitwise. Publication count equals accepted step throughout.
The failed replay leaves all13 checkpoint files unchanged; its diagnostic restart
reproduces the failure with identical incoming physical exports. The retained
portable five-state control reproduces surface histories, geometry and bulk-centre
exports bitwise after the diagnostic-only source change.

## Specific stopping cause and next action

At attempted step661, material `seed:243`, surface cell81, x=9.069543854 m:
a=4.52856e−15→5.39069e−14, Δj=1.48135e−16 m, dt=3.17839e−10 s.
The rate changes5.34299e−8→4.66069e−7 m/s. Its regularized log change .330533
exceeds the tighter .1 limit, requesting7.69276e−11 s below the unchanged1e−10 s
floor. Independent60-digit arithmetic reproduces the rate within4.82e−23 m/s.
Other normalized indicators are <=4.77e−6. The failure occurs before publication;
contact, source, mechanics and energy have passed.

Thus the current all-point rate-change controller stalls at incipient near-zero
damage. This is not evidence of a physical loss of admissibility. These data do
not independently separate an exact admission rate jump from algebraic active-set
noise at such tiny slip. No floor was lowered, damage clipped, history reset or
newborn point exempted. The smallest useful next numerical task is an independently
checked admission/event-aware error estimate, with residual/tangent-derived rate
uncertainty and all continuous-history checks retained. Do that before claiming
completed acceleration–growth coupling; no new constitutive assumption is proposed.

## Cost and reproducibility

Two MPI ranks throughout. The remaining control took11.26 min; tighter replay
38.37 min; exact failure reproduction3.44 min; retained regression9.72 s.
No long invocation exceeded90 min. Peak reported memory is about697 MiB/rank.
The original primary segments took50.84+33.09 min; the finer initialization probe
3.11 min. This is measured local cost, without a full-event performance guarantee.
Native checkpoints, original failures and full solver outputs stay under
`build-particle-free/cohesive-rsf-instability/`; compact review data are indexed above.

Inputs and direct commands: `benchmarks/reconstructed_fault/particle_free/cohesive_rsf_instability/README.md`.
Changed implementation: native owner/accessor metrics, default-off timestep plugin,
typed cutback exceptions, small-operator diagnostics and factored storage arithmetic;
the latter preserves the original failed subtraction and identical work bound.
This final pass adds only pre-publication failure operands and bounded-run controls,
plus comparison/plot scripts and status documentation. No automatic continuation.
