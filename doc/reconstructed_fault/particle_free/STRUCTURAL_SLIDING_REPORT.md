# Structural-plane sliding: qualified fixed-phase response, intact admission unresolved

**Substantial sliding and the mixed numerical checks are complete. A small
propagation demonstration was not started: the naive intact extension of this
rate-dependent law fails the phase-admission test.** The fixed-phase working
hypothesis remains useful; it does not yet define a consistent new-fault birth
law. This is an independent reference at
`f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved work. No native,
production, BP3, prior evidence or checkpoint is changed.

[Formulation and causal derivations](STRUCTURAL_SLIDING_FORMULATION.md) ·
[declared criteria](STRUCTURAL_SLIDING_SELECTION.md) ·
[indexed PNG/GIF/CSV evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/structural_sliding/delivery/INDEX.md) ·
[archive](../../../benchmarks/reconstructed_fault/particle_free/evidence/structural_sliding.zip).

## 1. Sliding of several Dc

Fresh .25 m strips retain G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², ell=.046875 m,
m=4, p0=10000 Pa, mu0=.1, a=.005, b=.01, V0=5e-6 m/s, Dc=.001 m and
rho=2670 kg/m³. The two fixed profiles have peak phase .8 and 1. A mature
center does not remove cohesive storage from its less-damaged flanks.
Boundary velocity is .001 m/s to 2 s, .002 m/s to 4 s, zero to 4.5 s,
then -.001 m/s to 6 s. All transitions align exactly with the timestep.
No history is reset at a velocity change or reversal. The fixed seed fracture
storages at 512 cells are .715544789 and 1.000003391 J/m²; their changes are zero
and they are not charged as work during sliding.

At 4 s, the integrated signed slips are **5.092 Dc and 5.716 Dc**. Over the
whole run, spatially integrated absolute slip reaches 6.163 Dc and 6.999 Dc;
this includes local counterflow and is distinguished from signed interface
slip. At 6 s the signed slips are 4.138 Dc and 4.671 Dc. State changes strongly,
with minimum Theta .0006864/.0002575. The experiments therefore exercise much
more than the earlier .00030 Dc weak-sliding test.

The initial 64/128/256-cell comparison exposed unresolved profile widths;
a focused 512-cell follow-up resolved the declared criteria. No parameters
were changed. Relative differences below use history maxima, with declared
absolute floors; no monotonic trend is required below numerical uncertainty.

| Measure | Developed: 256→512 | Mature-center: 256→512 | Worst dt .002→.001 s at 512 |
|---|---:|---:|---:|
| Traction | .1423% | .2884% | .1218% |
| Integrated slip | .02497% | .01639% | .00697% |
| Endpoint work | .1003% | .02994% | .03182% |
| Rate participation width | .9696% | .4371% | .1840% |
| Chi-weighted age | .07766% | .01300% | .02567% |

At the end of forward loading, widths are 32.366 and 9.654 mm. They evolve
physically but do not collapse to a cell width in this experiment. Maximum
local signed accumulated engineering shear magnitude is .808 in the saved
profiles, despite only 2.4% macroscopic shear strain. This is a mathematical
small-strain model test, **not** finite-deformation/objective-rate qualification.

The mature interface reference uses the same Maxwell block, imposed motion,
pressure and RSF/radiation parameters, with no arbitrary width and no cohesive
spring. A second reference retains the coherent-profile spring
C=G integral chi²/h dy. At 4 s:

| Response | Developed band | Mature-center band | Mature interface |
|---|---:|---:|---:|
| Traction (Pa) | 3549.70 | 1090.94 | 833.92 |
| Signed slip (mm) | 5.0919 | 5.7165 | 5.7810 |
| Minimum age | .0006864 | .0002575 | .0055903 |

The 30.8% mature-center traction difference is **not discretization error**.
Its work-conjugate traction separates into 127.07 Pa cohesive, 620.10 Pa
frictional and 343.77 Pa radiation contributions. Local speeds concentrate
near the center, approaching .02 m/s while the interface speed is about
.002 m/s. Local aging and quadratic radiation loss therefore differ from one
interface state/speed. For the developed band, cohesive resistance alone is
2691.56 Pa. This is retained A storage, not the retired K-gradient hardening.

Trapezoidal physical-work balance errors, normalized by accumulated sliding
friction+radiation, are **.00609%/.00595%**; sums of absolute step remainders
are **.00925%/.01162%**. Thus a small final error is not hiding large cancelling
steps. The corresponding final balance errors relative to external work are
.00335%/.00573%. Endpoint work, BE numerical remainder, Maxwell loss, friction,
radiation and both storages remain separate in the CSVs. Early ratios to almost
zero creep dissipation are not an energy-accuracy claim. The 1% criterion is
for the complete sliding experiment. Exact BE ledger defects stay below
3.4e-13 J/m², and local rate residuals below 1e-9 Pa against the retained
1e-8 Pa allowance.

![Substantial sliding](../../../benchmarks/reconstructed_fault/particle_free/evidence/structural_sliding/delivery/sliding_response.png)

[Saved-profile animation](../../../benchmarks/reconstructed_fault/particle_free/evidence/structural_sliding/delivery/sliding_profiles.gif)
shows physical saved times only. The phase profile stays fixed; the changing
rate field is not a crack front. Hold motion does not remove stored stress,
and developed cohesive memory can drive reverse local slip while the physical
traction remains positive.

## 2. Mixed mechanics: convergent instability, not a solver failure

The complete displacement-pressure-flow-Maxwell-state pencil retains actual
plane pressure and full plane-strain tensors. It agrees with an independent
constrained elimination to **3.49e-13 relative**; finite differences of its
pressure/state flow row agree to **7.10e-13**. This includes oblique waves,
finite/developed/mature phase, fresh/steady state and speeds up to .02 m/s.
No pressure lag or old symmetric-Hessian certificate is used.

At mature V=.0005 m/s, an oblique disturbance at 2.29° from the fault tangent
has growth rate **1.32056/s**, independent of wavenumber. Suppressing state
perturbations still leaves a maximum **.73508/s** pressure-feedback mode at
this same base state. Those suppression controls diagnose the mechanism;
they are not proposed laws or mandatory successful limits.

A nonlinear equilibrated oblique laminate, with the fault plane held horizontal,
reproduces the instability. A 1% state perturbation grows to 1.8340% after
1 s at dt=.0005 s. Its pressure and actual-plane-pressure ranges are 37.90 and
75.68 Pa. Refining 32/64/128 points gives the same resolved mode; timestep
halving converges to the continuous prediction. Momentum and flow residuals
are below 1.2e-11 Pa, and energy defects below 3e-16 J/m³. A separate spatial
wave-symbol test reduces nonzero errors from 1.177% to .2964% to .07425%.
This is substantially more than rotating the symmetric band, but remains an
independent laminate/symbol qualification, not a full native 2-D solver test.

The important distinction: this is **bounded continuum growth**, not a growth
rate diverging as wavelength tends to zero, nor loss of nonlinear convergence.
Positive growth alone is not a rejection rule. Nevertheless, the distributed
model has internal modes absent from a geometric closed-interface RSF model.
The latter has stiffness G|k| in the incompressible identical-halfspace limit;
our Maxwell reference is stable at sufficiently short wavelengths. The bulk
symbol has a wavelength-independent coefficient instead. The interface's
short-wave property cannot simply be inherited from its friction formula;
see [Rice, Lapusta & Ranjith (2001)](https://authors.library.caltech.edu/records/5efnc-nm026)
and the separate derivation in the formulation. Whether these internal bulk
modes are intended fault physics remains a constitutive interpretation issue.
It is not, by itself, the reason for stopping this task.

![Mixed response](../../../benchmarks/reconstructed_fault/particle_free/evidence/structural_sliding/delivery/mixed_response.png)

## 3. Why phase-active progression stops here

For positive phase the energy force and normalization derivatives can be
written consistently; the formulation records their full work balance. The
remaining problem is **exactly intact material with zero fault history**.
The old scalar constant-resistance law supplied an intact cohesive multiplier.
The structural-plane rate law does not yet supply an equivalent birth rule.

The audit freezes the same physical trial shear at 4000 Pa, A=K=0 and Theta=1,
then evaluates the one-sided positive-phase return. On a ray with existing
normalization support, chi/h has a finite limit. The returned traction is
Qxy=G beta dt (chi/h) V, so its phase driving tends to zero with dt. There is
no accepted A history at exact phi=0 in which that prospective traction can
accumulate.

| Trial dt (s) | Intact-limit driving Y (Pa) | Fracture contribution minus Y (Pa) |
|---:|---:|---:|
| .050 | 10.0419 | -2.0419 |
| .025 | 7.3261 | +.6739 |
| .00625 | 2.1370 | +5.8630 |
| .00078125 | .07695 | +7.92305 |

A coarse step would admit homogeneous damage and a fine step would not.
This is an admission/initial-state problem, not permission to pick a coarse
step. An entirely new uniformly incipient ray instead has I→0, giving the
one-sided driving limit **32 Pa**; at exactly zero support Gamma/I is undefined.
These are distinct support topologies, explicitly labeled, and their limits
are resolved down to trial phase 1e-24 without a phase or normalization floor.

![Intact admission](../../../benchmarks/reconstructed_fault/particle_free/evidence/structural_sliding/delivery/intact_admission.png)

This does **not** prove every seeded front is immobile: the fracture gradient
can seed adjacent material. A front-only model with absorbing homogeneous
intact material is possible, but changes the earlier intact-nucleation
semantics. It must not be adopted silently just to generate an animation.
Likewise, a latent traction stored at phi=0, or damage-created recoverable
compliance distinct from frictional slip, requires new constitutive evolution
and birth work; neither follows from the accepted T/A/K tensors alone.

**Recommended physical choice:** preserve the verified existing-fault sliding
law, but separate fracture admission from frictional sliding. Derive the
intact traction/compliance and its energy calibration, then transfer the
unchanged Maxwell tensors and existing frictional age into a new structural
record. Specify genuinely new age and direction explicitly. At the same time,
decide whether the intended mature limit is a distributed bulk material or a
geometric interface rate/state/contact model. Do not remedy these differences
with pressure lag, extra damping, K reset or an unexplained phase threshold.

That is the substantive obstruction to a scientifically interpretable tip
experiment. No phase-active run is claimed, no production law is replaced,
and the original accepted structural-plane evidence remains intact.

Implementation/evidence: six new solver/analysis references plus the plotting
script, three PNGs, one GIF and eight indexed CSVs. Source, parameters, exact
commands, timings and preservation hashes accompany the archive. Measured
sequential solve/audit time is about 62.3 s (excluding figures and file packaging),
not a native/server performance prediction. The native
solver, production checkpoints and protected BP3 artifacts are unchanged.
