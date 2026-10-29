# Mature-fault sliding: constant K-gradient hardens

**The selected 6 mm K-gradient option is incompatible with friction-only mature
sliding over the tested regime. Coupled phase–microforce progression is not
started.** The 3 mm option also fails the declared finite-slip resistance screen.
This does not undo the accepted fixed-phase regularization tests: the new result
identifies what their recoverable gradient storage does during continued slip.
No coefficient was degraded, K reset, friction recalibrated or tolerance changed.

## Reduction and literature

For S=sym(t⊗n), S:S=1/2, define engineering shear k=2K:S and rate w=2W:S.
Then accumulated interface slip delta=∫k dr is in metres, and V=∫w dr is in
m/s. V excludes elastic/Maxwell boundary deformation. For a fixed normalized
profile k=delta rho, ∫rho dr=1 and rho has units 1/m:

```
Egrad/area = (G ell_s²/2) delta² integral (rho')² dr
Rgrad = G ell_s² delta integral (rho')² dr.
```

Thus a nonuniform fixed-width band acquires linear accumulated-slip hardening.
The coefficient does not vanish when h→infinity and cohesive Q→0. Current
native mature-fault semantics instead have zero cohesive resistance/storage,
with separate auxiliary Maxwell/slip memory. An RSF steady state at fixed V
does not contain this unbounded delta term. This test uses constant mu, the
necessary a=b=0 friction limit, not an implementation of full RSF.

[The derivation](MATURE_GRADIENT_FORMULATION.md) includes varying-profile terms,
microboundary work, exact 1-D mixed mechanical elimination and source/tangent
equations. It compares Gurtin (2003), the two-length gradient-plasticity/fracture
construction of Miehe et al. (2016), Pranger et al. (2022)'s spatial rate/state
interaction and the Fei antiplane/in-plane phase-field RSF formulations, with
primary links and explicit distinctions. Their mature-friction or variational
properties are not inherited by our added energetic K-gradient.

## Bounded experiment

The .25 m cross-fault specimen has a prescribed near-mature core phi=.999 on
|r|≤ell/2, smooth quintic shoulders to |r|=ell, and stronger phi=.02 surroundings.
G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², ell=.046875 m, Kf=8 Pa, m=4,
mu=.1, c_r=0 and mean p0=10000 Pa are retained. Strength-drop=2000 Pa remains
distinct from c_r. The core h≈7.988e6 makes its cohesive traction negligible.
The surroundings have higher cohesive stiffness, not a different friction law.
They are nearly intact, not exactly virgin. The bulk API's forbidden phi=1
endpoint is not inserted, and no new damage-to-mature transition is defined.

Each case starts fresh with zero histories, fixed physical phase and microfree
outer ends. Both tensor components/conjugate systems remain free. The mixed
equations enforce uniform physical shear traction, normal equilibrium and
the same mean perturbation-pressure reference as the native confined block.
The pressure derivative is included even though this symmetric solution has
p'=0. This is a mechanically equilibrated independent P1 strip, using the
qualified C++ normal-map projection; it is not the older prescribed-strain
prototype or a new ASPECT propagation run.

Impose relative boundary velocity 1 mm/s for 2 s, dt=.01 s. Local, 1/3/6 mm
lengths use N=257; 6 mm additionally uses N=129/513 and dt=.005 s. Maximum
accumulated engineering shear stays below 3.6%. The seven controls take 73.25 s
wall time in total. All 1600 intervals are accepted within the retained checks.

At the common final imposed motion of 2 mm:

| ell_s | Actual plastic slip [mm] | Traction [Pa] | Gradient / cohesive / friction [Pa] | Rate width [mm] |
|---|---:|---:|---:|---:|
| Local | 1.7466 | 995.042 | 0 / .0045 / 995.037 | 49.14 |
| 1 mm | 1.7329 | 1049.02 | 47.18 / 7.35 / 994.48 | 61.37 |
| 3 mm | 1.6603 | 1337.21 | 296.65 / 49.54 / 991.02 | 68.33 |
| 6 mm | 1.4934 | 1998.76 | 865.78 / 153.16 / 979.82 | 73.88 |

The resistance terms are actual power/V, not point samples. Their sum matches
traction within 1.13e-11 Pa. For 6 mm, only .0021 Pa of the cohesive term comes
from the mature core; 142.87 Pa comes from shoulders and 10.29 Pa from outside
the initial support. The dominant gradient term is not residual cohesive
storage in an insufficiently mature core. Local X can assist flow on the
flanks while resisting it in the core; its integrated power is positive.

Widths use `(∫|w|dr)²/∫w²dr`. They settle early: over delta=.25–1 mm, the
6 mm rate width grows only .056%, while gradient resistance grows from
145.19 to 580.14 Pa. This is persistent hardening at effectively fixed width,
not continued widening that explains the traction rise. Relative to the local
reference, the gradient law also establishes a wider flow profile inside the
unchanged phase band; it does not create new phase damage.

At matched delta=1 mm, gradient resistance is 2.74%, 17.97% and 58.30% of
the initial friction traction for lengths 1/3/6 mm. The declared 5% resistance
screen therefore fails for 3/6 mm. **The 1 mm case passes that narrow finite-slip
screen** and has negligible width drift within it; this is an approximate
window, not a non-hardening mature limit or a selection/calibration of 1 mm.
Its established flow width is already about 25% greater than the local one.
For any retained nonzero length and nonuniform fixed shape, the analytic
hardening continues with delta. Shrinking the length to hide it over one
window would not settle the intended RSF constitutive decision.

The free conjugate-plane law has an additional known distinction from an
aligned interface: with Q=X=0 it gives mu p0/sqrt(1+mu²)=995.037 Pa rather
than 1000 Pa. Its actual plane-normal friction changes slightly as traction
changes. No angle lock or mu correction is applied to conceal this difference.

## Numerical and energy checks

For 6 mm at 2 s, N=257→513 changes traction .01494% and the continuous P1
rate-profile L2 norm .01884%; N=129→513 gives .07480%/.09025%. The dt halving
changes traction .000515% and rate profile .000211%. These are far below the
measured hardening. Initial phase representation errors are separately
.2163%, .05431% and .01359% for N=129/257/513. The frozen continuum fracture
storage is 2.64449 J/m²; its N=257 representation is 2.64093 J/m² and stays
constant. This prescribed profile is not claimed to be phase equilibrium.

Maximum full-tensor relative force error is 3.27e-13 (bound 1e-12), source
recurrence error 1.12e-16, and mechanical compatibility error 3.02e-18 m.
All actual points remain compressive, minimum 8000.94 Pa. The complete joint
K/shear/pressure derivative agrees with finite differences at 9.99e-9
(bound 1e-6); the local reference matches an independent 50-digit mechanical
elimination to 3.23e-13 at retained stress scales. Immutable trials/retries
are bitwise on T/A/K/Q/L. This reuses prior native lifecycle evidence rather
than claiming new native/MPI/restart qualification.

Bulk, cohesive and gradient storage; physical friction and both Maxwell losses;
and BE stress/cohesive/gradient remainders are separate in the CSVs and plots.
Maximum BE ledger and gradient-work identity defects are 9.65e-15 and
4.13e-15 J/m². Final 6 mm gradient storage is .64695 J/m². Algebraic closure
is not by itself physical energy qualification; the focused timestep control
and the retained earlier P4 energy/accuracy limitations remain separate.

## Constitutive decision and one alternative

The selected constant, non-relaxing K-gradient cannot serve as the intended
mature RSF regularizer. It is a defensible gradient-hardening material model,
but its friction-only mature limit fails. Stage 4's compatibility condition
is not met, so no coupled phase implementation is begun.

Recommend **a Pranger-type spatial interaction in rate/state kinetics**, rather
than recoverable energy in total accumulated K. The next small scientific task
would derive the normalized bulk-to-interface rate/state mapping, no-flux
spatial interaction and integral calibration; then test constant-velocity
steady sliding and velocity steps against the same interface RSF parameters.
K remains physical accumulated history. This is a different constitutive
option, initialized fresh, not resetting K or converting stored gradient
energy in existing checkpoints. Merely replacing K by W in a gradient term
would add a finite rate-dependent traction and is not automatically RSF-equivalent.
Pranger's normalization/calibration and reported transient limitations matter.
Thermodynamic admissibility, physical-pressure/direction tangents and the mixed
short-wave audit must be redone for that extension before phase coupling.

![Traction and resistance](../../../benchmarks/reconstructed_fault/particle_free/evidence/mature_gradient/traction-slip.png)

![Cross-band fields](../../../benchmarks/reconstructed_fault/particle_free/evidence/mature_gradient/cross-band.png)

![Separate energies](../../../benchmarks/reconstructed_fault/particle_free/evidence/mature_gradient/energy.png)

The phase and boundary motion were imposed; stress, slip distribution and
resistances were solved. Shapes mostly scale after onset, so a GIF adds little.
Compact inputs, PNG/CSV/JSON, code and commands are in
`benchmarks/reconstructed_fault/particle_free/evidence/mature_gradient.zip`.
BP3, production source/defaults, accepted prior evidence and unresolved P4/P5c
findings are preserved. No large specimen, MPI expansion or free-direction
propagation run is started.
