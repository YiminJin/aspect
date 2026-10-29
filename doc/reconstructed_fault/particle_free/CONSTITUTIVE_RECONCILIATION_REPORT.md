# Constitutive reconciliation: branch kinetics improve, continuation remains open

The endpoint discrepancy is causally explained, but **the proposed continuation
rule does not resolve all endpoint ambiguity**. Positive RSF rate slope and
radiation damping remove the tested smooth-branch mixed-mechanical singularity;
they do not establish a globally unique changing-direction law or a resolved
slip width. No propagation or native/BP3 change follows.

[Complete equations](CONSTITUTIVE_RECONCILIATION_FORMULATION.md),
[predeclared scope](CONSTITUTIVE_RECONCILIATION_SELECTION.md),
[CSV/PNG/code package](../../../benchmarks/reconstructed_fault/particle_free/evidence/constitutive_reconciliation.zip).
Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532`, branch
`pf-rsf-particle-free`, plus preserved local work.

| Question / comparison | Measured outcome | Conclusion |
|---|---|---|
| Exact printed Feng angle/update vs global condition | Original phi=.8 endpoint: stationary residual <1e-12 Pa, other maximum +.0144094 Pa | Stationarity is insufficient. |
| Complete trigonometric branches | Recover the independent roots, including the reversed-memory case and C1=0 mature limit | Fixes branch enumeration, not the constitutive global test. |
| Loading refinement, 32→2048 steps | phi=.8 peak violation .0292189→.0297097 Pa | The inadmissibility does not disappear with dt. |
| RSF+radiation, full pressure/history/state | Bounded smooth-branch spectrum; mature maximum .00697007/s, strict reversed-memory .00695514/s | Conditional kinetic regularity, not global stability. |
| Radiation only / no kinetics | Bounded but fast growth up to 6.09486/s / constitutive rate matrix singular | Damping controls rates; it does not guarantee decay. |
| Previous-step pressure vs coupled pressure | Both tend to the same continuous mixed generator | Pressure lag is not a separate continuum remedy. |
| Freely changing directions with pressure lag | Mature nonlinear endpoints rejected; 80-digit calculation confirms violations | Fei's temporal rule is not automatically a valid free-direction closure. |

The sources are Feng's [energetic formulation](https://doi.org/10.1016/j.jmps.2024.105687)
and specifically Fei's [**in-plane** quasi-dynamic formulation](https://doi.org/10.1002/nag.70416).
The latter's Appendix A14 uses previous-step normal traction. Its complete
stress/phase/direction algorithm is not transplanted into our Maxwell model.

## Endpoint and continuous-history findings

The literal Feng closed-contact angle formula reproduces the earlier nonglobal
endpoints. Its arcsine shortcut also misses a stationary branch in the reversed
memory example; the assumed shear sign then disagrees with the computed one.
Enumerating all solutions of the same Appendix E equation removes this
algorithmic issue. The printed physical-shear sign in equation 73 is separately
distinguished from relative shear in equation 59. Neither correction makes
stationary roots automatically globally admissible. Two different globally
admissible reversed-memory endpoints remain, and preference among tied
directions **at one state** does not choose between those states.

Fresh zero histories under pre-existing uniform damage obstacles reach first
activation at approximately **.832655 s** in the prescribed shear ramp. At
that instant A=0, Y=0, and the physical phase force is **+8 Pa**. For either
single conjugate flow, the other yield function increases at
**9.47315/.163330/.000300525 Pa/s** for phi=.2/.8/.99. Thus the homogeneous
single-ray continuation fails its tangent admissibility condition before any
phase growth is permitted. The exact mature limit is different. The recorded
post-failure printed-update paths are diagnostics, not accepted trajectories.
The synthetic reversed-memory states are obstacle-compatible; their
reachability from a fully admissible single-direction path is not established.

![Endpoint reconciliation](../../../benchmarks/reconstructed_fault/particle_free/evidence/constitutive_reconciliation/endpoint-reconciliation.png)

*Imposed homogeneous loading and unchanged phase obstacles expose the distinction
between a smooth stationary path and admissible evolution. No conjugate average,
phase relaxation or increased tolerance repairs the missing continuation.*

## Kinetics and pressure treatment

The independent candidate uses asinh RSF and aging with radiation
`eta_rad=G/(2 cs)`. A normalized density from the retained AT1 phase profile
maps `W=chi V m`; at the mature profile center chi=16/m. No arbitrary reference
width or accumulated-K storage is introduced. The integrated dissipation
matches interface friction plus radiation under the explicitly coherent-profile
assumption. This is a kinematic/power identity, not a resolved cross-band
mechanical solution. For varying direction or velocity across the band, local
V remains interface-equivalent; intact normalization/state birth is unresolved.

Inputs retain G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², ell=.046875 m and m=4.
Exploratory mu0=.1, a=.005, b=.01, V0=5e-6 m/s, Dc=.001 m and rho=2670 kg/m³
give eta_rad=25836.0214 Pa s/m. These are explicitly uncalibrated candidate
kinetics, not production parameter changes.

Full pressure, changing directions, Maxwell tensor histories, state and
incompressibility are retained. Six frozen states and both conjugate branch
limits were checked, including the earlier oblique failure normals. Generalized
eigenvalues of the uneliminated mixed pencil agree with independent elimination
to **2.67e-13** relative; coupled current/previous-state derivatives agree with
finite differences to **2.65e-12**. The discrete block update agrees to
**3.19e-16**. These pass the retained 1e-10/1e-6 checks.

Finite rates remain bounded over wavelengths **29.45 m to .2945 mm** and are
independent of wavelength in this local quasistatic law. With state feedback
removed, tested RSF branches decay (largest eigenvalue -.005/s, including the
state clock). RSF without radiation is already regular on these branches;
radiation modestly lowers growth here. Constant friction with radiation remains
rate-regular on an active branch; removing both rate slope and radiation is
singular. None of this selects a band width or proves regular switching at ties.

At the same reversed-memory wave normal of 90°, radiation-only growth at
dt=.1 s is **9.40281/s coupled** versus **3.18335/s lagged**. At dt=.0001 s
it is **6.09668/s** versus **6.08915/s**, approaching **6.09482/s**. RSF+radiation
shows much smaller differences and the same convergence. Including the old
pressure perturbation is essential to this comparison.

![Kinetics and pressure](../../../benchmarks/reconstructed_fault/particle_free/evidence/constitutive_reconciliation/kinetics-pressure.png)

*These are frozen-state mechanical/state perturbations, not crack-front speeds.
Positive bounded growth does not establish a physical finite-wavelength RSF
instability or a nonlinear propagation certificate.*

## Small nonlinear checks and limits

The fresh exact-mature point completes .3 s of loading/reversal at
dt=.0125/.00625/.003125 s with fully coupled pressure and all tensor histories.
At common times, successive full-tensor L2 differences are **1.09% and .89%**;
final shear differences are **4.12 and 1.20 Pa**. Work differs by **4.68% and
2.07%**. The BE remainder falls **.46163→.23503→.11895 J/m³**, while ledger
defects remain below **7.54e-16 J/m³**. This is bounded temporal evidence,
not a physical-energy accuracy claim.

The free-direction pressure-lag paths stop at .025/.0125/.009375 s. At the
coarsest failure, the independently reevaluated global violation is
**4.87016e-8 Pa** against **1.00467e-11 Pa** allowance. Eighty-digit roots
confirm the smaller-step failures too. No complete matched lagged trajectory
is claimed. In a separate coherent fixed-plane velocity-step/reversal/stop
control, traction error halves with dt: current-pressure **.1503→.07559→.03791 Pa**,
lagged **2.6872→1.3459→.6732 Pa**. This control isolates kinetic/state timing;
it does not impose permanent plane locking on the candidate.

![Point and interface controls](../../../benchmarks/reconstructed_fault/particle_free/evidence/constitutive_reconciliation/point-and-interface.png)

Final verification cost **7.23 s** locally, plus figure generation. Two numerical
verification issues were corrected without changing physical tolerances:
unit-equilibration of both sides of the mixed QZ pencil, and measuring the
complete Jacobian FD error rather than dividing an analytically zero column
by a tiny arbitrary denominator. Original failed logs are retained.

**Decision:** the smooth-branch kinetic prerequisite improves; the global
continuation prerequisite does not pass. Before native integration, resolve
the single-direction evolution at conjugate ties and between distinct return
states, together with a consistent direction/pressure time discretization.
Do not resume propagation, reinterpret a mixture as Feng flow, or use pressure
lag as the missing constitutive rule. Production/BP3, the retired K-gradient
experiment, and earlier physical-energy/spatial/branch failures remain intact.
