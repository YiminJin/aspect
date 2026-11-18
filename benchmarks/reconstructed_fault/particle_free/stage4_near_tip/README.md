# Near-tip prepared sliding, common elastic preload

The uniform-preload calibration is closed. This task changes the *prepared
slipping state* and the age-perturbation location at gamma_e=.000365. It retains
the exact phase/formed planes, native mechanics, material parameters, pressure
reference, quadrature, midpoint dt=8e-5 s and acceptance tolerances. Four ranks
run sequentially. No checkpoint conversion or production source change occurs.

The nominal V0(x), before the age perturbation, is 2e-4 m/s times

* zero for x<=3;
* sin²(pi (x-3)/2) for 3<x<4;
* one for 4<=x<=6;
* cos²(pi (x-6)/.25) for 6<x<6.125;
* zero for x>=6.125.

It is continuously differentiable at the joins and remains nonzero through the
formed taper x=5.5..6.0. Every section with nonzero target rate is checked to be
formed. The short right taper ends within existing formed material; no fault
continuation or intact sliding is invented. The initial unperturbed age and
maximum nominal rate are unchanged. This is an explicit synthetic initial-state
change, not transport of the old state or a constitutive modification.

The independently specified age perturbation is
theta0(x)=theta_base [1-.01 sin²(pi (x-3)/3)] on 3<x<6, and theta_base elsewhere.
The native preload helper first equilibrates mechanics at the resistance of
the unperturbed V0/theta_base, then applies this age perturbation and inverts
each law at the retained common resistance. Therefore the actual perturbed
initial rates differ slightly between laws, while their initial stresses and
damage margins match. At x=6 the age perturbation vanishes but V0 remains
2e-4 m/s. That distinction is the purpose of this preparation.

`prepare.py` constructs exact inputs and a separate benchmark checker from the
protected preload checker. The only changed checker expression moves the age
perturbation from [2.5,5.5] to [3,6]. Existing scan/release observers are reused.
`run.py` and `batch.py` provide fresh bounded cases and durable sequential
execution, with failure stopping the batch. An initial zero-interval native
preflight measures the changed equilibrium and tip margins before the pair.

Fixed-phase comparison: eight stages per law, t=.00064 s, gamma_e=.000365.
Both right and left tip rows/probes and local rates are compared at identical
physical positions and stage times. The held phase cannot advance; its zero
fracture increment is not a free-response prediction. The initial state is not
silently phase-equilibrated. Existing favorable tail modes must be kept visible.

A short free-phase pair is conditional on a resolved law-specific change large
enough to matter to the right-tip fracture margin, including a substantial
reduction/delay. Tiny differences close this preparation. No further uniform
preload sweep or longer campaign is selected. The native diagnostic equations
and signed work conventions are documented in `../stage4_preload/README.md`.

## Completed outcome

Preflight and both eight-interval fixed-phase laws exit zero. Weakening raises
peak rate 30.16% and integrated slip 13.15%, but the right-tip row excess is
.000168959 Pa against a .049384214 Pa control deficit. The ahead probe has a
small opposite effect. No direction becomes favorable only under weakening.
No free-phase pair is selected; this preparation and uniform-preload tuning are
closed. Three left-tip rows are already favorable initially, and opposing local
left-taper deformation nearly cancels in its section-integrated rate.

After numerical ranks exit, `analyze.py initial` and `analyze.py fixed` reproduce
the old/new rate figure, matched local histories, full native row differences,
signed deformation and work checks. `retain.py` verifies incoming source and
protected runtimes, stores compact inputs/logs/observations and hashes retained
checkpoint payloads without duplicating them. See
`doc/reconstructed_fault/particle_free/STAGE4_NEAR_TIP_SLIP_REPORT.md` for the
release decision, limits and exact evidence links. The optional release runner
is retained support, not evidence of an executed or authorized continuation.
