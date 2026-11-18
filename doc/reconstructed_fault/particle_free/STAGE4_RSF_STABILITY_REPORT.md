# Stage 4 — steering endpoint and bulk RSF stability

12 November 2026. Selected task: `CODEX_STAGE4_RSF_STABILITY.txt`.

**Decision.** Retain the present specimen as a qualitative growth/steering
control, not as an unstable-RSF specimen. Its measured restoring stiffness is
hundreds of times the nominal RSF critical scale. A synthetic reduction of
`Dc` produces clear weakening/control separation in a measured-mechanics Ritz
diagnostic, but also excites a faster cell-scale mode. Do not start a free-phase
instability campaign on this evidence. Resolve that spatial instability and the
native preparation/cost issue first. No completed slip event is claimed.

## 1. Fine endpoint: capped; accepted .0775 comparison retained

The fresh `stage4-free-turn-fine-endpoint` copied only fine128×256 accepted16,
slot02, from `overnight-free-turn-c-half`. It used the original physical pair,
four bound cores, midpoint .0025 and unchanged physical gates. It stopped at
5400.37 numerical seconds without accepting .080. The final printed trial was
Newton34, mechanics `2.10106e-15`, phase KKT `1.55216e-7`, above the `1e-7`
phase gate. Trial fields are excluded. Compression/closure took another ten
seconds; the cap was not extended. The .0775 parent and its copied checkpoint
remain byte-identical, with exactly reproduced native phase and section state.
The two outstanding .00125 intervals remain separate and deferred.

These are matched **retained** .0775 results, not a new endpoint qualification:

| Quantity | 128×128 | 128×256 |
|---|---:|---:|
| Right front x (m), absolute phi=.05 | 6.118129275 | 6.118773080 |
| Signed front y−2 (m) | +.028664979 | +.027803980 |
| Band center y−2 at x=6.03125 (m) | +.017904007 | +.017524757 |
| Band center y−2 at x=6.09375 (m) | +.025812456 | +.025259603 |
| Added fracture energy (J/m) | .0462057065 | .0462417101 |
| Added fracture energy in axial-ahead material (J/m) | .0077674740 | .0077989360 |
| Signed physical-work error, full retained history (J/m) | 5.99102e-5 | 6.00282e-5 |
| Accumulated absolute physical-work error (J/m) | 5.99102e-5 | 6.00282e-5 |

Fine-minus-coarse front offsets are **+.643805 mm axially, −.860999 mm
laterally**. Band-center differences at the two tabulated stations are
−.379250 and −.552853 mm. Native phase L2 difference is .1551% of the field,
or 1.814% of growth from the seed. Work errors are about .130% of added fracture.
No sign reversal of steering is seen; this is not spatial/time convergence.

The exact 12 earlier-registration labels from the overnight report are retained.
Their complete slip, age, stage rate, frozen angle and ds are in
[affected-registration-sections.csv](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/affected-registration-sections.csv).
They have 9.19–26.67 micrometers of fine cumulative slip, ages .54757–.56701 s,
and stage rates .003675–.010282 m/s. Frozen-angle differences reach .934 degrees.
Using the original paired-grid weighting, they carry **14.4877% of new-section
absolute integrated slip and .114915% of all paired slip**. Separate-grid shares
are also recorded; their denominator includes all sections, and is explicitly
separate from the historical paired-registration convention.

Their measured global steering effect is small at .0775, while their structural
memory and later slip are real. The already available coarse .080 follow-up has
larger slip on these labels; without the fine .080 endpoint we cannot conclude
that their later influence remains small.

[Endpoint figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/endpoint-comparison.png)
([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/endpoint-comparison.pdf)).

## 2. Native restoring measurement and continuous stability

The successful `stage4-fixed-phase-stiffness-homogeneous` takes one fixed-phase
coarse interval from accepted15/.075 to .0775, then performs seven read-only
restoring actions. The physical phase is the .075 profile. Formed planes retain
their history; the usual virgin-plane update finishes during the physical solve.
**All planes, phase and section measures are fixed during the restoring actions.**
This diagnostic is not the free-phase .0775 state. The physical solve took
34.25 s; the complete native run, seven actions and output took 218.66 s.

The response uses actual native DG Bernstein phase, FE strain, full 3D
plane-strain stress, bulk/cohesive energy, quadrature JxW, section ds and work
adjoints. The six smooth traction patterns occupy established material
x=[2.5,5.5] m. Outside-patch resistance perturbations vanish, but their mechanical
slip response remains. Dirichlet displacement increments vanish on the body
boundary; prescribed confinement is 200 Pa, separate from Stokes pressure.
Friction slopes are removed only from the diagnostic restoring operator.

Let h=m phi(1+phi)/(1−phi)^2, H=h/(1+h). Equilibrated elastic response is
`delta z = -C delta f`, with clamped compliance `sum dn H/G` plus finite-body
compliance. The normalized traction basis P and slip basis Psi=C P (P' W C P)^−1
use W=diag(ds). Positive K is the restoring stiffness: the signed change of
driving traction is **−K delta z**. It is neither the imposed boundary slope
nor the algorithmic Newton Jacobian. The native stage factor is removed exactly.
[Derivation and units](../../../benchmarks/reconstructed_fault/particle_free/stage4_rsf/DERIVATION.md)
give the constrained energy, projection and Maxwell equations.

| Measured response | Result |
|---|---:|
| Six smooth generalized stiffnesses, MPa/m | 1.24181, 1.92163, 2.38760, 2.71065, 2.90380, 2.97182 |
| Full-profile cohesive lower bound, MPa/m | .509957 |
| Nominal p(b−a)/Dc, MPa/m | .002 |
| Clamped share of first traction-pattern compliance | 31.449% |
| Equilibrated finite-body share of that compliance | 68.551% |
| Lowest smooth slip-mode RMS width | .70908 m = 11.35 x cells |
| Smallest smooth traction half wavelength | .5 m = 8 x cells |
| Lowest-mode squared-slip norm outside loading patch | 1.141% |
| Cell-scale probe effective stiffness, MPa/m | 1.14279 |
| Minimum with that probe added | 1.14279 MPa/m, 7.97% lower |

The full-profile bound follows from cohesive energy and Cauchy–Schwarz,
`K >= diag(G / integral h dn)`; it is not inferred from the center peak.
All seven equilibrated mixed residuals are below `9.69e-12` relative, independent
point-update response errors below `2.65e-16`, and integrated energy versus
half-force/slip-work errors below `1.90e-13` relative. Reciprocity also passes.
Persistent physical state, accepted coordinates and accepted evaluation are
unchanged by the diagnostic. The native interval publishes once and exits cleanly.

The initial checker omitted the homogeneous-increment constraint context needed
by the existing Stokes inverse after postprocessing restored physical boundary
values. Two diagnostic residual checks failed; normalization alone did not fix
it, and a residual-correction attempt exhausted its linear iterations. Those
runs/checkers remain evidence. Saving/restoring constraints, Newton flags and
linear tolerance, and temporarily installing the same inverse context used by
the coupled solver, fixes the diagnostic. The final solve needs no extra defect
correction. No production solver, physical gate or tolerance was modified.

The actual law has a=.015, b=.020, Dc=.0005 m, p=200 Pa, radiation=1000 Pa s/m,
G=1e6 Pa and eta=1e8 Pa s. Native friction values/partials agree with the independent
formula to `2.59e-15` relative; a finite-difference Jacobian action agrees to
`2.63e-11`. The fixed-phase interval has no opposite-sign normal deformation and
no section cancellation. This does not erase the mixed-sign deformation observed
in the separate free-phase history.

For the common Maxwell rate r=G/eta=.01/s, the physical dynamic stiffness is
`K(lambda)=lambda K/(lambda+r)`. Aging is `theta_dot=1-|V|theta/Dc`.
The following scalar uniform-state diagnostics use the measured smooth stiffness
spectrum and actual regularized friction/radiation derivatives. They are not
claimed to be native steady trajectories:

| Operating point | V (m/s) | theta (s) | fV (Pa s/m) | ftheta (Pa/s) | Largest growth rate (1/s) | Decay time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Low-speed, steady age | 1e-6 | 500 | 3.001e6 | .008 | −.00194912 | 513.05 |
| Nucleation-speed, steady age | 1e-4 | 5 | 31000 | .8 | −.200514 | 4.987 |
| Fast, steady age | .0105329 | .0474704 | 1284.82 | 84.263 | −21.1231 | .04734 |
| Fast observed transient | .0105329 | .202090 | 1284.82 | 19.7932 | −21.1231 | .04734 |

The last state's age derivative is −3.25717, so it is distinctly transient.
The actual spatial stage rates/ages were also used together, keeping every
responsive section age. The maximum real part is approximately zero from
quiescent age modes (−1.04e-14/s), with no growing mode in this measured space.
It must not be interpreted as a meaningful trillion-second decay time or a
full native-spectrum proof. Maxwell relaxation matters for the slow case;
it is negligible on the prospective .04–.07 s instability timescale. Fast
observed slip exceeds the .001 m/s radiation crossover, giving a strengthening
steady slope. Nonlinear convergence alone was not used as a stability argument.
These conclusions concern the sampled fixed profile and operating states. They
do not exclude instability after substantial profile evolution or in a different,
very slow Maxwell-dominated state. The full native spectrum is not computed.

[Stiffness/stability figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/stability.png),
[response table](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/stability-responses.csv),
and [complete measurements](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/stability.json).

## 3. Economical pair: resolved in time, incomplete in space

One synthetic material adjustment was selected from the measured margin:
**Dc=3.2211126e-7 m**, about 1552 times smaller. At initial peak V=2e-4 m/s,
`[p(b−a)−radiation V]/Dc = 2 K_min` for the smooth patch.
All responsive sections start with uniform theta=.0016105563 s; this is steady
only at the peak. The b=a control uses one scalar mu0=.60804718956, which matches
the entire initial instantaneous friction curve, traction, rate and age.
Both receive the same generalized force kick (initial peak rate change 2.0e-9
m/s) and the same constant generalized drive. Each is compared with its own
unperturbed transient background. No rate clipping or cold-section cutoff is used.

This is a **six-mode Ritz diagnostic**, not a native coupled run. It retains
native-measured cohesive/finite-body mechanics, all responsive ages, signed
nonlinear friction, common Maxwell relaxation and midpoint/exact-stage aging.
Its generalized preload and drive have not been realized as native uniform-shear
boundary/history data. The initial coefficient-frozen prediction is +15.124/s
(e-fold .06612 s); the neutral spectrum has no positive mode beyond roundoff.

The pair runs .1323072 s: 1643 intervals at 8.05278e-5 s, and the matching
half-step histories. The timestep is half the native scalar-root maximum and
initially resolves the fastest state time with 20 steps. The backgrounds evolve,
so neither a constant exponential nor the initial eigenvalue is imposed on them.

| Measured perturbation response | Weakening | Matched b=a |
|---|---:|---:|
| Maximum rate-norm amplification | 29.484 | 1.44635 |
| Final rate-norm / initial | 3.16589 | .00234848 |
| Final maximum relative age perturbation | 5.33258e-5 | 7.14416e-8 |
| Maximum magnitude of cumulative section slip / Dc | 82.0500 | 80.2204 |
| Final rate-perturbation change on halving dt | .1524% | .0250% |
| Final age-perturbation change on halving dt | .1297% | .00725% |

A differentiated full discrete step follows each evolving background; finite
perturbation versus variational rate errors stay below .208% across both time
resolutions. Peak `dt |V|/Dc` reaches .319 in weakening (.160 on the half step);
accuracy is supported by the measured state/rate comparison, not exact aging
alone. Signed and accumulated absolute reduced energy errors stay below
1e-14 J/m. This quadratic-reservoir ledger is **not** the native fracture ledger.
Profiles retain signed positive and negative rate perturbations; acceleration
has a spatial structure, rather than being inferred from a single section.
Background pulses and small-perturbation separation do not qualify a native
completed event or growth/acceleration sequence. Phase is fixed here.

[Pair figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/reduced-pair.png)
([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/reduced-pair.pdf)),
[full pair results](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/reduced-pair.json).

**Spatial warning changes the adoption decision.** At the identical smooth
initial rate/age, adding the already measured sine48 action raises the largest
real growth rate to **25.781/s**, with imaginary part 146.245/s. The rate mode has
99.918% of its squared norm outside the six smooth modes. Its envelope is broad
(.846 m RMS), but its half wavelength is one .0625 m x cell. Envelope width is
therefore not a resolution certificate. Baseline and neutral checks still have
no positive growth beyond roundoff. A reduced smooth-space pass cannot justify
suppressing this faster oscillatory response in a native calculation.

[Mode-resolution figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/mode-resolution.png)
([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/mode-resolution.pdf)),
[augmented-spectrum check](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/probe-stability.json).

## 4. Selected configuration, cost and next experiment

Retain the smaller Dc as a **fixed-phase instability diagnostic candidate**;
reject its adoption as a qualified free-growth specimen on the current mesh.
The next justified scientific experiment is a targeted spatial-mode comparison:
keep the profile, planes and this RSF law fixed, resolve the .125 m wavelength
with finer tangential spacing, and check whether growth rate and mode structure
converge rather than moving to the next grid scale. This is a mechanical/RSF
resolution question, not another general steering or propagation campaign.
A native matched preload/drive must also be established before transferring the
reduced growth timescale to a physical trajectory.

Concrete native input alternatives and a runner are prepared in
[`stage4_rsf/`](../../../benchmarks/reconstructed_fault/particle_free/stage4_rsf/README.md):
`weakening-background.prm`, `weakening-perturbed.prm`, `neutral-background.prm`,
`neutral-perturbed.prm`. They use the selected Dc, ages and scalar friction
coefficients, fresh equilibrated peak-.60 seeds, original uniform/off-axis
velocity held constant from t0, and the measured admissible timestep. Their
common perturbation is +1% initial section age. **They are cold-start conditioning
controls, not realizations of the reduced slipping operating point.** In
particular, the original boundary speed may carry them into radiation-dominated
sliding. No old checkpoint is relabeled or converted, and no native pair was run.

For reproducibility, the prepared conditioning input is runnable as:

```sh
python3 benchmarks/reconstructed_fault/refactoring_boundary/run_logged.py stage4-conditioning-weakening 9300 python3 benchmarks/reconstructed_fault/particle_free/stage4_rsf/run_native_candidate.py weakening-background --label stage4-conditioning-weakening --seconds 9000
```

The .020 s prefix requires 249 intervals: **2.37 hours per history**, or about
9.48 hours for a pair with its two backgrounds, extrapolating the measured
34.25 s fixed-phase interval. A native .1323 s comparison at the selected step
would be **15.63 hours per history, 62.52 hours total**, before half-step checks.
Cold-start and later solver costs can differ. These costs, missing matched
native preparation and the faster cell-scale mode rule out an economical native
instability claim in this bounded task. The runner can prepare a fresh free-phase
counterpart with `--free-phase`, but it is conditional on those qualifications;
it was not launched. Late release of a frozen checkpoint is not substituted.

## 5. Provenance, preservation and implementation boundary

Root `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, HEAD
`b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`. Incoming dirty/manual work is retained.
The physical executable independently has SHA256
`089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`;
its protected original checker is
`81efcd6e5e0013f4124bf1e1ba857005a2a23447985f69b16464fbc4d3131d69`.
The executable banner is 5f1fc6f3c, not an assertion that current HEAD matches it.
The separate diagnostic checker is generated/build-recorded under
`build-particle-free/stage4-rsf/`; failed versions remain in its named subfolders.

The task changes benchmark support and documentation only. Native DG Bernstein,
midpoint/BE conventions, physical inputs of the endpoint attempt, quadrature,
constitutive laws, state publication and physical acceptance criteria are
unchanged. The measured new Dc is confined to diagnostic/prepared inputs.
No production source/header/CMake change, commit or server job was made.
At most four numerical/build workers ran concurrently; analyses used one thread.

Evidence is in [`evidence/stage4_rsf/`](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_rsf/):
entry root/revision/patch, launch commands, full numeric tables, plots,
`preservation.json` and verified compression records. Native command manifests
retain inputs and binary/checker hashes separately. One MPI startup was denied
sandbox sockets; it performed no simulation and remains recorded. Final checker
and run artifacts are separately identifiable from all failed diagnostic trials.
Original source/checkpoints/binaries passed preservation checks. No historical
data were removed; existing cleanup losses and broken aliases remain explicitly
unrecovered. Only new, closed CSVs were losslessly compressed after hash checks.

Implemented: benchmark-only native restoring measurement and independent RSF
analysis. Locally verified: work adjoint/energy, true mixed residual, point
response, state preservation, friction derivatives, reduced temporal and
variational comparisons. Physically compared: retained .0775 mesh endpoints.
**Not qualified:** fine .080, outstanding .00125 endpoints, full native stability
spectrum, resolved unstable mode, native matched unstable pair, or free-growth
instability/event production readiness.
