# Restored DGQ2 coupled validation

**Completed 7 November 2026.** The four runs are stable and preserve the native
material phase balance, but the coupled solution is **not yet spatially
qualified for a paper-oriented server pilot**. Bulk refinement changes final
central/endpoint surface stress by **5.48%/3.71%**, compared with **0.54%/0.89%**
under physical-timestep halving. Endpoint slip rate and age change by
**2.73%/4.13%** under mesh refinement. Phase L2 changes by only .414% under that
same refinement, and normalization I by less than .068%. Thus excellent profile
balance is not evidence that stress and surface state are resolved.

There are two different sensitivities: bulk resolution dominates several final
mechanical observables, while physical timestep remains important during the
initial state transient. Localized-work quadrature is still measurably sensitive;
it limits interpretation of small energy/normalization differences. The matrix
does not separate tensor approximation, mechanical discretization and fixed
20-label resolution, so none is asserted to be the sole cause. No universal
percent-error acceptance threshold was introduced.

## Frozen baseline, inputs and execution

The existing worktree was `aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, documentation HEAD `36de6bcf3`, with no tracked changes
at entry. Numerical source/CMake matched native integration **`84f57503c`**
after normal revert `319b10b94`; shared-assembler review tag
`review/shared-scalar-dg-3a21d59cf` remains preserved and inactive. All 950
recorded source/include/unit hashes and seven binary/plugin hashes matched.
The same retained executable and matching energy-checker/background plugins
were used throughout; no compilation or numerical source change occurred.
Executable SHA256:
`6ea374c5b9b50a20af00fa6e2c365df61d7ac59ed33ad26c7b338d7bb7ba67d9`.
The executable's embedded Git banner predates its recorded local integration
changes; the hash/source manifest, not that banner, establishes identity.

One semantic `phase_field` composition, DGQ2, SSPRK3/native subcycling, Bernstein
BP bounds, phase-specific mapped assemblers and zero particle managers are
retained. True direct stress, ordinary DGQ2 tensor transport, accepted-state
bulk-slip-subtracted spin, endpoint forcing/RSF and acceptance criteria are
unchanged. The generic startup rotation banner still says total spin; resolved
`Constitutive spin = bulk slip subtracted` and the per-step `Accepted bulk spin`
diagnostics establish the selected closure. That old banner was not edited.

All cases freshly initialize the same analytic profile, curve and physical
fields on [-6,6]×[-2,2], with 20 fixed material labels, G=10,
eta(y)=5.5+4.5 tanh(y/.35), background loading 3.7, a=.025, b=.015,
matching prescribed exterior stress/top traction and unchanged pressure reference.
Growth and adaptive physical timestepping are disabled. C changes only the
uniform drift and matching provider identity, not shear loading or fault frame.
D uses nine-by-three box repetitions and level four, giving h=1/12; no checkpoint
interpolation, width change or integral rescaling was used.

| Case | U | Actual cells | Physical dt | Accepted steps | Minutes | Phase substeps, total |
|---|---:|---|---:|---:|---:|---:|
| A | 2 | 96×32 | 0.0125 | 64 | 33.15 | 192 |
| B | 2 | 96×32 | 0.00625 | 128 | 57.43 | 256 |
| D | 2 | 144×48 | 0.00625 | 128 | 123.25 | 256 |
| C | -2 | 96×32 | 0.0125 | 64 | 32.90 | 192 |

Run order was A, B, D, C, one simulation at a time, four MPI ranks and one
numerical thread per rank. Every case reached T=.8 (within floating-point
accumulation), with **zero retries, zero rejected line-search candidates, and
one joint history/geometry publication per step**. Exterior recurrence errors
are below 3.2e-14; integrated velocity compatibility is within 1.53e-13 and
provider clocks match exactly. Geometry/age replay errors are at most
2.30e-16/8.89e-16. Minimum curve-to-box clearance exceeds 1.50 throughout.

A/C use three phase substeps per physical step (internal dt=.00416667); B/D use
two (dt=.003125). Halving physical dt therefore does not halve the internal
phase substep. B/D have identical phase substep schedules. No CFL or quadrature
rule was adjusted. Native checkpoints are retained every 16 steps and at final
termination. No restart trajectory or extra diagnostic solve was needed.

## Initialization, profile and localization

A/B initial bulk and surface observations are byte-identical. Relative initial
FE phase L2 error against the analytic profile decreases from **.21870%** on
96×32 to **.06277%** on 144×48 (Gauss16 observer). The analytic integral is
.194180740740741; represented coarse/fine integrals are
.194180674520619/.194180748765034. The B/D initial phase difference is .22456%.
Initial normalization differs by .02784% centrally and .01497% at the ends;
initial slip rate differs by .03941%/.70742%. Geometry agrees exactly. Mesh
comparison therefore includes initialization as well as evolution; it does not
start from an interpolated identical FE field.

| Case | Integral change (%) | Final sampled peak | Final normal width | Accumulated absolute native balance defect |
|---|---:|---:|---:|---:|
| A | -0.005305 | 0.350293 | 0.139047 | 2.231e-15 |
| B | -0.007571 | 0.350415 | 0.139117 | 3.192e-15 |
| D | -0.001142 | 0.350074 | 0.138809 | 3.215e-15 |
| C | -0.004641 | 0.350381 | 0.139028 | 2.148e-15 |

Width is the phase-weighted standard deviation about the mean curve normal,
not a fitted physical length-scale parameter. Its initial value is .155209 on
the coarse mesh and .155136 on the fine mesh. Deformation changes that width;
its decline alone is not a diffusion/error measure. Peaks use 81 samples per
cell, including separate face traces, supplementing the native bound guarantee.
Minimum samples are nonnegative up to roundoff (largest negative magnitude below
2.8e-20 initially and 7.2e-19 finally). The [0,1] physical bound was retained;
no artificial initial-peak cap was imposed.

The signed SSPRK balance defects for A/B/D/C are respectively
-2.231e-15/-2.982e-15/-3.006e-15/-2.115e-15; absolute defects are listed above.
Integrated phi div(v) sources are -1.03006e-5, -1.47023e-5, -2.21737e-6 and
-9.01112e-6. Outward phase flux magnitudes are at most 9.71e-15. These sources
account for the integral changes: numerical Stokes velocity is not pointwise
divergence-free, and DGQ2 phase is not a pressure test function. Raw integral
change is not the transport defect. In this incompressible model, this is a
discretization contribution, not physical compressibility or evidence of exact
continuum phase-integral conservation.

Limiter cell-stage counts are 659,562 / 886,807 / 1,877,304 / 655,458, with minimum
alpha .0572963. The fine mesh has smaller peak fluctuations. These observations
include the limiter, subcycling and coupled velocity changes; no unlimited
coupled trajectory was selected to isolate limiter causality.

## Matched coupled comparisons

Common times are 0, .2, .4, .6 and .8, with all accepted label histories also
retained. Bulk norms integrate on the union of both cell partitions; surface
rules split at labels, mesh crossings and region boundaries. Central labels
are [.2,.8]; ends are their complement. Absolute difference, reference norm and
relative L2 are supplied for every reported field in the linked CSV/JSON.
Cumulative slip has no relative value at zero initial increment.

The final **relative L2 differences in percent** are:

| Observable | A/B central | A/B ends | B/D central | B/D ends |
|---|---:|---:|---:|---:|
| Surface physical stress | 0.538 | 0.892 | 5.478 | 3.714 |
| Total driving shear | 0.073 | 0.131 | 0.943 | 0.657 |
| Work-projected total driving shear | 0.037 | 0.058 | 0.042 | 0.137 |
| Pressure-inclusive normal traction | 0.380 | 1.354 | 2.520 | 4.952 |
| Effective compression (includes background) | 0.052 | 0.181 | 0.348 | 0.665 |
| Slip rate | 0.428 | 0.998 | 0.570 | 2.727 |
| Cumulative slip | 0.800 | 1.403 | 0.392 | 2.045 |
| Age | 0.445 | 1.917 | 0.980 | 4.131 |
| Normalization I | 0.155 | 0.279 | 0.067 | 0.060 |
| Geometry | 0.176 | 0.167 | 0.005 | 0.010 |

Work-projected total driving shear changes much less than pointwise stress and
traction; agreement in that work measure does not resolve the observed raw
field oscillations. For scale, B/D central stress has absolute/reference L2 .0698862/1.275877,
and endpoint age .00223612/.0541312. Total driving shear includes prescribed
background loading: the much larger raw perturbation-shear difference
(31.53% centrally for B/D) is not a 31.53% error in the total driving shear.
Normal traction retains its sign and pressure; effective compression also
includes the prescribed background normal load.

Whole-domain final A/B versus B/D differences are **.2448% versus .5403%** for
stress, **1.1102% versus .4139%** for phase, and **1.6406% versus 3.1689%** for
pressure. The final localization L2 differences are 8.80% and about 3.6%; sharp
finite support ends and their displacement contribute. B/D Gauss8/12/16
localization L2 differences are 3.561%/3.655%/3.591%, so the last digits are not
quadrature-converged. This is not evidence of a comparable normalization-I error.

Final-time differences understate the temporal transient: sampled A/B age
sensitivity reaches **9.67% centrally at .2 and 12.32% at the ends at .4**;
slip-rate sensitivity is about 3.7% at .2. B/D endpoint age reaches **8.32% at
.4**. Thus smaller final temporal differences do not establish full-trajectory
accuracy. Two bulk meshes provide a refinement difference, not an asymptotic
order or an exact error estimate; the unchanged 20-label surface may limit
further bulk refinement. This is not a phase-only mesh study or third-order
qualification of the complete coupled algorithm.

C is stable and admissible, with compatible inflow and bounded endpoints. At
final time A/C central/end slip-rate differences are .799%/1.963%, cumulative
slip .225%/1.548%, and age .631%/2.445%. Drift is removed only for comparative
geometry/field displays and the explicitly marked overlap-domain comparison;
work and traction calculations retain physical coordinates. Oblique geometry,
shear and finite boundaries do not imply an exact reversal symmetry. The pair
shows no sign-specific solver failure, but does not distinguish all physical
finite-domain effects from directional discretization errors.

Accumulated slip is substantial: final label ranges span .0793–.1521 m across
the four runs, about 9.9–19.0 Dc. Sampled B/D accepted bulk-spin exposure ranges
are -.1888 to -.01775 rad / -.1908 to -.01535 rad; total-material exposures are
about -.36 to -.14 rad. The sums of dt times the domain maximum absolute bulk
spin are .481/.515/.669/.486 rad for A/B/D/C; these are upper envelopes, not a
single material rotation angle. No total-spin/no-spin trajectory was added.

## Energy, quadrature and cost

Endpoint and trapezoidal diagnostics retain separate signed and absolute sums.
Units are energy per out-of-plane length; reaction projection is a separate
inventory, not added again to the total balance.

| Case | Endpoint signed | Endpoint absolute | Trapezoid signed | Trapezoid absolute | Endpoint absolute / active work (%) | Reaction-projection loss |
|---|---:|---:|---:|---:|---:|---:|
| A | -0.00340545 | 0.0751197 | 0.000243606 | 0.00261689 | 0.2686 | 2.979e-05 |
| B | -0.00162172 | 0.0383207 | 0.000192005 | 0.000697962 | 0.1370 | 1.626e-05 |
| D | -0.000681148 | 0.0392295 | 0.00112733 | 0.0011626 | 0.1403 | 1.154e-05 |
| C | -0.0107014 | 0.068207 | -0.00675504 | 0.00902983 | 0.2457 | 3.071e-05 |

Active absolute work scales are 27.966/27.965/27.965/27.763; final stored energies
are 2.67246/2.67195/2.67202/2.67822. Endpoint absolute remainders correspond to
2.81%/1.43%/1.47%/2.55% of final storage. A/B halving nearly halves the absolute
endpoint remainder, while bulk refinement does not improve it. C's larger
trapezoidal remainder is retained, not hidden by signed cancellation or called
a direction-specific instability. Transport-stage raw storage changes and
rotation-projection losses are separately retained; the former include physical
boundary flux and are not numerical dissipation alone. Rotation-projection
losses total 2.47e-7/1.29e-7/9.06e-8/2.45e-7. Reaction losses are all nonnegative,
so their signed and absolute sums coincide.

The final accepted B and D fields were evaluated at Gauss8/12/16 without another
mechanical solve or any production quadrature change:

| Quantity | B: q8 / q12 / q16 | D: q8 / q12 / q16 |
|---|---|---|
| Localization measure | 2.068017 / 2.069416 / 2.069653 | 2.071869 / 2.071524 / 2.071619 |
| Measure/curve-length defect (%) | -.62294 / -.55567 / -.54430 | -.45737 / -.47390 / -.46935 |
| Friction power | 1.273510 / 1.274307 / 1.274831 | 1.269131 / 1.268678 / 1.268851 |
| Slip power | .0689365 / .0689615 / .0690146 | .0682703 / .0682368 / .0682859 |

Storage and phase integrals agree near roundoff, but localized work does not.
The q12→q16 maximum work-traction coefficient change is **.00237 Pa for B** and
**.000322 Pa for D**; B's q8→q12 change is .00229 Pa, so monotone convergence is
not demonstrated. The native q8 surface work residual is near 1e-10, while q16
gives -4.26e-5/+2.45e-5. This tests the integration of the existing discrete
solution, not failure of its unchanged native acceptance gate. The B/D signed
energy and normalization differences cannot be promoted to exact accuracy
estimates, and this final-state check does not certify every earlier energy
increment.

Linear iteration totals are 14,813 / 24,354 / 24,352 / 14,850. B and D have nearly
identical iteration counts, but D costs 2.15 times as much with 2.25 times the
cells. Phase transport accounts for only .67–.72% of runtime (14.3/23.0/51.2/13.9 s).
The main queue took 14,873.9 s including reference checks and compression;
main reductions/figures took 451.1 s and final frozen-observer/figure update
17.0 s. Two sets of pre-step failures consumed about 240 s: blocked sandbox MPI
sockets, then an inherited unconditional `Particles/Load balancing strategy`
guard. Both are preserved. Retaining the inert `none` setting fixed the input
without instantiating particles or changing production code. Total measured
run/reduction overhead is about **4 h 20 min**, below eight hours; compilation
cost is zero. No diagnostic replay, extra trajectory, server job or checkpoint
conversion occurred.

## Paper readiness and next single experiment

| Paper configuration | Present capability/evidence | Remaining blocker |
|---|---|---|
| Stationary BP3 first event | Retained production stress/RSF path and supplied first-event prefix | Event completion/remote manifests remain unestablished; incompressible diffuse-fault and finite-domain differences prevent strict reference equivalence. This task does not qualify a new production BP3 trajectory. |
| Fixed fault with viscous lower crust | Layered Maxwell coefficients and compatible stress/traction construction have local algorithmic evidence | A genuinely fixed-geometry layered control, dimensional viscosity/loading/prestress choices and its accuracy qualification remain; U=0 here still permits deformation. |
| Materially advected counterpart | Zero-particle DGQ2 phase, direct objective stress, complete material transaction, both drift directions and retained restart infrastructure | Several-percent spatial mechanical/state differences, temporal state transient, unresolved localized-work integration and fixed-label resolution; no earthquake nucleation or moving-cohesive qualification. |

See the established [BP3 review](BP3_REVIEW.md) and
[pilot mapping](LAYERED_BP3_PILOT_MAPPING.md); the latter's older Q1 remapping
findings remain historical, not the present DG phase balance. The current
velocity-strengthening fixture is not the dimensional paper experiment.

**Recommendation:** address coupled spatial accuracy before a paper server
pilot. The next single justified comparison is a fresh **U=+2, h=1/16,
dt=.00625, T=.8, 20-label** case against D, with the same frozen algorithm,
physical initialization and native subcycling. It would test whether the
measured stress/endpoint-state differences decrease or approach a fixed-surface/
integration floor; two existing meshes cannot answer that. This recommendation
does not authorize that run, quadrature optimization, architecture changes,
additional campaigns or new physical parameters.

## Evidence and reproducibility

- [Exact four inputs](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/inputs/) and [resolved differences](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/input-differences.json).
- [Source/binary/plugin identities](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/identity.json), [queue ledger](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/queue.json),
  [progress history](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/progress.jsonl), [analysis ledger](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/reduction-ledger.json).
- [Compact summary CSV](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/comparison-summary.csv), [full summary JSON](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/comparison-summary.json),
  [absolute/reference/relative comparison norms](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/comparison-norms.csv),
  [frozen quadrature](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/frozen-quadrature.json).
- [Profile balance and shape](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/profile-balance-shape.png).
- [Central/end traction, slip, age and normalization](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/coupled-traction-slip-state.png).
- [Moving-region fields, including reverse drift](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/moving-region-fields.png).
- [Energy, runtime, solver and subcycle costs](../../../benchmarks/reconstructed_fault/particle_free/evidence/dgq2_overnight/energy-runtime-subcycles.png).

Run directories are
`build-particle-free/transported-objective/stage-c/material-runs/layered-phase-integration-overnight-{a,b,d,c}/`.
Each retains exact command/input/binary hashes, complete log, all accepted
observations and rotating native checkpoints. The new rank CSVs are losslessly
gzip-compressed after completion, with verified uncompressed hashes in
`compressed-fields.json`; no accepted state was deleted. Original historical
outputs/checkpoints and the shared-assembler review archive remain unchanged.
Generated artifacts stay local and outside source commits.

Reusable entry points are `dgq2_phase_field/run_overnight.py prepare`, then `run`,
and `dgq2_phase_field/analyze_overnight.py` after completion. Preparation refuses
to overwrite cases. The actual bounded launch is recorded in
`refactoring_boundary/evidence/dgq2-overnight-restored-queue02.json`;
analysis and final observer commands have analogous records. The new harness
reuses the established fixture, complete-Q2 observers and comparison rules;
it changes no numerical source. Plain and compressed historical data remain
readable. All four PNGs were visually inspected.

Architecture, quadrature, constitutive conventions and checkpoint meanings
remain frozen. No jobs remain active. All other experiments remain deferred.
Inherited availability losses remain **39/45 native-pilot frames and earlier
temporary logs**; none was reconstructed or relabeled as available.
