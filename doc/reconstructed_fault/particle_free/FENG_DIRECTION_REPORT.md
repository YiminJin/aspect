# Single-direction reconciliation: redistribution removed, propagation blocked

The independent Feng direction/dissipation reference removes the old mature
two-system redistribution mode. **The full pressure-coupled mechanical
localization defect remains**, including for a unique maximizing direction.
No propagation test, native integration, production or BP3 change was made.
The prior gradient-hardening and Pranger results remain preserved.

[Equations and source audit](FENG_DIRECTION_FORMULATION.md),
[predeclared checks](FENG_DIRECTION_SELECTION.md),
[compact evidence and runnable reference](../../../benchmarks/reconstructed_fault/particle_free/evidence/feng_direction.zip).
Baseline: `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local work,
branch `pf-rsf-particle-free`. This is a fixed-phase, closed-contact,
small-strain constitutive/mechanical audit, not a crack-path prediction.

## What changes

Feng's actual normalized crack-strain increment N must maximize
`F=(sigma-Q):N/sqrt(2)-mu pn(N)` and dissipates
`sqrt(2) mu pn(N) norm(W)`, where pn is the smaller positive compression on
the two perpendicular axes of N. The paper's rule is not a sum of conjugate
flows. Its elastic model and phase energy are distinguished from our explicit
Maxwell/history mapping; the latter retains all current parameters.
[Feng et al. (2024)](https://doi.org/10.1016/j.jmps.2024.105687).

The old spin selector was recovered from `pf-cpdi`. With positive CCW spin
`omega=(L21-L12)/2`, it intended to choose a conjugate branch by total-spin
sign, with a +1 zero-spin fallback. Superposing rigid observer spin flips its
normal by **84.2894°** at unchanged instantaneous stress/strain rate. A separate
literal 2-D NaN guard prevents the intended sign selection from executing.
Both behaviors, zero spin, velocity reversal and stress reversal are recorded;
neither is adopted as a constitutive requirement.

Among genuine F ties, an existing plane/history chooses the closest unoriented
normal. A resolvably larger F overrides history. Rotating the stress axes through
40° rotates the selected maximum through 40°; there is no permanent lock.
No preference or an equal-distance preference leaves symmetry unresolved.
Passive basis rotations transform histories and directions together; this
does not qualify finite-rotation physics.

![Candidate and selected directions](../../../benchmarks/reconstructed_fault/particle_free/evidence/feng_direction/candidate-directions.png)

*The imposed mature stress gives two equal candidate planes. A supplied
near-horizontal plane resolves this tie. Arrows are plane normals, lines are
slip planes; neither is a computed phase front. The lower panels show why spin
sign is unsuitable and how continuity follows moving maxima without locking.*

## Measured checks

| Check | Result |
|---|---:|
| Global direction maximum vs independent scalar optimization | 2.27e-13 Pa |
| Direction / complete point-return basis covariance | 2.83e-16 / 1.61e-16 |
| 17 archived strict-single states: physical-stress difference | <=4.82e-13 Pa |
| Same states: tangent difference normalized by G | <=6.66e-16 |
| Full changing-direction/pressure tangent finite difference | 4.87e-11 relative; allowance 1e-6 |
| Independent rate formula / mixed-null residual | 6.11e-16 / 6.19e-17; allowance 1e-12 |
| Independent algebraic single-return backstress check | 5.85e-14 Pa |

The 17 states use identical archived incoming histories and trial stress: six
phase values on both normal branches, plus five forward timestep probes.
The old native single-system output is a comparison, not an accuracy reference.

At mature pure shear, p=10000 Pa, mu=.1 and tau_xy=995.03719 Pa, the normal
angles are 2.85530° and 87.14470°. Each single flow satisfies actual-direction
power. The equal conjugate mixture at equivalent rate 1e-4/s gives physical
work and old summed friction **0.099503719 W/m³**, but Feng dissipation
**0.100000000 W/m³**, a **0.496281%** deficit. Its actual direction has
F=-4.96281 Pa. It cannot be admitted as a Feng active direction.

The old 45° zero-stress redistribution perturbation is outside the selected
single-flow tangent: relative projection mismatch .995037; its new mixed
coefficient is 1.0 G. But at phi=.8 the minimum **full mixed** finite-step
coefficient is **-0.00195453 G**, while the fixed-pressure check gives
**+0.00467879 G**. Full mixed nulls occur at wave normals 165.60683° and
168.27995° from +x. This is an admissible fixed-phase obstacle control.
The continuous-rate threshold is **phi=.76761128** (m=4, mu=.1).
Forward probes at the same incoming state approach **-0.003015 G** as dt
decreases to 1e-8 s; reducing the timestep does not remove the defect.

![Mechanical and power comparisons](../../../benchmarks/reconstructed_fault/particle_free/evidence/feng_direction/mechanical-comparison.png)

*The left panels retain solved pressure and changing directions. The right
panels distinguish developed-damage localization from the removed mixture
freedom. These are point/symbol calculations, not mesh or propagation results.*

There is also a finite-step closure limitation. For zero incoming backstress
and a pure-shear trial of 1100 Pa, regular single-return roots at phi=.2/.8/.99
satisfy their own plane equation but violate final global maximization by
**.870256/.0144094/2.64938e-5 Pa**, versus **2.21e-11 Pa** roundoff allowance.
They are rejected. The exact mature limit admits a continuity-selected root.
At phi=.99 with Qref_xy=-.05 Pa, two distinct globally admissible return states
remain; direction ties at one state cannot select between different states.
These are bounded endpoint tests, not a claim that Feng's complete published
model has no solutions. No tolerance was enlarged or conjugate average used.

The Maxwell and elastic-control BE ledgers close within 8.33e-17 J/m³.
In the Maxwell control, work=.2187503, storage change=.0005678, viscous
loss=.0001265, friction=.1172271 and BE remainder=.1008290 J/m³. The large
numerical remainder is explicit; ledger closure is not physical energy accuracy.
The final verification took **3.47 s**, figures **2.17 s**, single-process
local wall time. Existing failed accuracy/branch findings are retained.

## Decision and handoff

Do not proceed to propagation. A single actual maximizing direction resolves
the specific conjugate-redistribution artifact, not pressure-induced loss of
mechanical ellipticity or every finite-step branch ambiguity. Before native
integration, a scientific choice must address mechanical regularity while
preserving mature sliding, and define admissible continuation where no unique
single endpoint exists. Restoring spin selection, locking a plane, averaging
conjugates or importing the old symmetric stability certificate is not a
justified remedy. No additional model is selected automatically.

Changed files are the three `feng_direction` reference/check/plot scripts,
this report, formulation and selection, three status/audit index additions,
and compact evidence/PNG/archive files. Native sources, checkpoints, defaults
and protected BP3 artifacts are unchanged; the package records preservation
hash checks and reproducible commands.
