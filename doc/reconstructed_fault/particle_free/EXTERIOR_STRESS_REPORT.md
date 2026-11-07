# Coupled prescribed exterior stress and shear benchmark — 5 November 2026

The native prescribed-exterior option and the five selected coupled runs are
complete. Prescribed inflow substantially reduces the original corner
amplification; compatible top loading is also needed to remove the remaining
large corner concentration. The compatible temporal sequence shows decreasing
stress/energy differences. The single spatial comparison measures the remaining
fixed-mesh limitation; it is not a spatial convergence order or an absolute
continuum-error guarantee.

[Protocol](EXTERIOR_STRESS_PROTOCOL.md) ·
[Evidence and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/exterior_stress/README.md) ·
[Figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/transported_objective/exterior_stress/exterior-stress.png)

## Source baseline and implementation

Actual root `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, revision `cf06525471ce89789f1ac10905dc79f3aac7562a`,
with seven pre-existing tracked documentation changes and retained untracked
experiments. Entry hashes precede changes. Native source changes are confined to
`phase_field_fault.h`, `phase_field_fault.cc`, `material_history.cc` and
`stress_evolution.cc`; no solver, tensor operator, spin, endpoint, quadrature,
convergence criterion, physical coefficient or profile-transport implementation
changed. The new benchmark has its own CMake target and provider DSO.

`Material tensor inflow = prescribed exterior stress` and the required
`Material exterior stress identity` select the feature. A public material setter
attaches a physical-stress function of point, accepted-parent time, RK stage time
and attempted physical dt. The existing `TensorAdvection` boundary callback
calls it on **every incoming physical face**, including locally incoming top
portions. The function supplies the exterior for pure transport; ordinary
rotation and endpoint reaction remain downstream and occur once.

Only **direct physical stress** is supported by this option. One physical tensor
does not define both separated exterior histories. Unsupported storage, absent
or mismatched identity and a missing provider are rejected explicitly. Default
zero and accepted-trace paths and their checkpoint identities are retained.
The new exterior identity enters the existing material-law checkpoint identity;
no schema conversion is introduced and the callback is not serialized. A matching
provider must be reattached on restart. No physical history pointer is exposed to
or published by the benchmark provider; the existing transaction remains owner.

The core was rebuilt with four workers and saved as
`build-particle-free/transported-objective/stage-c/exterior-stress/aspect-release`.
The retained `build-particle-free/native/aspect-release` was backed up and restored
byte-for-byte. The new candidate requires the matching checker under
`exterior-stress/checks/`; retained checker binaries were not overwritten.
HEAD and the retained executable do **not** identify the new dirty-source binary.
Exact command, binary, source and input hashes are recorded in the evidence.

## Split-consistent exterior, loading and initialization

For the selected homogeneous background, G=eta=10, imposed shear gamma=.5,
p_ref=0, and zero stored initial stress. Define

    z_n = (tau_xx-tau_yy)/2 + i*tau_xy
    r = exp((-G/eta-i*gamma)*dt)
    F = eta*gamma*(1-exp(-G*dt/eta))
    z_n = iF*(1-r^n)/(1-r), z_0=0.

Then tau=(Re z,-Re z,Im z), with tau_zz=0. During transport step n+1 the provider
returns **z_n**, unchanged at all RK stages because this background is spatially
homogeneous. It never feeds z_(n+1) into transport and then rotates/reacts it again.
The independently evaluated real-component recurrence checks the complex formula.

Compatible top traction uses (Im z_(n+1),-Re z_(n+1)) at p=0, exactly the prior
isolation's discrete background formula. Pressure normalization remains `no`.
At simulator step zero, initial mechanics instead uses its existing .1-s
constitutive interval, so initial top traction is (5*(1-exp(-.1)),0). That solve
publishes **no accepted stress**. Consequently the first physical transport
receives zero, despite the nonzero initial mechanical response. The benchmark
provider validates G, eta, imposed boundary shear and the pressure/initialization
conventions. It supports its declared uniform accepted timestep schedule only;
it is not an adaptive exterior-history integrator.

The five runs retain the same mature finite-fault fixture, positive translation
.4 plus shear .5, compact bounded Q1 profile, growth disabled, ordinary DGQ2,
SSPRK3, accepted-gradient spin lag and projection-corrected coupled solve.
All run from zero to .4 s with **18 fixed initial material labels at .1 m spacing**.
The fine case changes only bulk mesh resolution at the smallest timestep.
Temporal differences also contain the convergence of the discrete background
loading, since the same physical background is integrated with each dt.

## Crossed boundary comparison at 32², dt=.025

The two retained numerical-parent-trace controls are reused without rerunning.
All peak norms below use common Gauss4 sampling of the full 3D plane-strain
tensor, and are sampled maxima rather than certified continuum suprema.

| Top loading | Inflow | Final peak stress (Pa) | Endpoint energy remainder / active work | Final raw-face minus weak mechanical power (W/m) |
|---|---|---:|---:|---:|
| Original zero | Retained trace | 58.796 | 11.0005% | +.634805 |
| Original zero | Prescribed exterior | 7.93205 | 4.09090% | +1.12441 |
| Compatible background | Retained trace | 2.343 (near fault) | 3.16046% | -.00224515 |
| Compatible background | Prescribed exterior | 2.34311 (near fault) | 3.15920% | -.0000795618 |

The original zero-top/prescribed case still peaks in the upper-left corner near
(-1.9913,1.9913). Its peaks at t=.1,.2,.3,.4 are 2.320,4.374,6.252,7.932 Pa.
The extreme 58.8-Pa response is strongly suppressed, but neither that observation
nor acceptance proves convergence of the original mechanical boundary problem.
Its raw face-work mismatch actually increases. Compatible loading and prescribed
inflow remove that large corner concentration; the peak shifts near the fault.

The evidence supports **both** a substantial numerical-trace feedback contribution
and a remaining mechanical-loading contribution. It does not justify attributing
the entire issue to one alone, or adopting the original zero-top problem as a
converged shear reference. Changing top loading changes the boundary-value problem
and fault response; those crossed differences are not discretization errors.

The stage ledger also distinguishes the mechanisms: original trace transport
adds .295721 J/m, versus -.00486493 J/m with prescribed exterior and zero top.
The corresponding rotation/projection energy loss falls from .00862174 to
9.25e-6 J/m. With compatible loading/prescribed exterior, net transport change
is 1.19e-5 J/m and rotation/projection change is roundoff. These are measured
stage changes, not newly introduced physical dissipation terms.

Prescribed versus trace inflow changes final central/endpoint slip rate by
5.07%/4.51% under zero top loading, but only .0138%/.0384% under compatible
loading. Changing top loading with prescribed inflow changes those rates by
99.4%/233.6% relative to the compatible case. Thus the two loading models cannot
be conflated even though the stable compatible trace/exterior controls are close.

## Temporal and spatial accuracy

The fixed-32² temporal initial bulk coefficients and surface fields are bitwise
identical. The initial states **and first physical steps** of both new .025 runs
also match their respective retained trace controls bitwise. That independent
first-step comparison confirms the unloaded transport start: a reacted exterior
at the first transport stage would fail it. The fine mesh uses the same physical
initial functions and exactly the same surface coordinates/labels; its FE
profile representation naturally differs with h.

Central labels are a in [.2,.8]; ends are the two outer 20% regions, also retained
separately in JSON. Stress uses a common .2-m tube about the finest temporal
reference curve; surface quadrature is split at both grids' crossings and uses
the same reference measure. Normal traction includes pressure. All entries below
are relative L2 **percent**, written central / ends. These are pairwise
sensitivities, not exact-solution errors.

| Observable | dt .05 → .025 | dt .025 → .0125 | 32² → 64² at dt .0125 |
|---|---:|---:|---:|
| Stress | 0.55245 / 0.61734 | 0.27933 / 0.31199 | 0.074852 / 0.046141 |
| Point shear traction | 0.15488 / 0.3885 | 0.087709 / 0.20312 | 0.059396 / 0.031575 |
| Work-adjoint shear traction | 0.1832 / 0.36855 | 0.10061 / 0.19191 | 0.036426 / 0.0080224 |
| Pressure-inclusive normal traction | 0.83065 / 0.90095 | 0.41489 / 0.4483 | 0.043606 / 0.041928 |
| Slip rate | 2.9432 / 5.9343 | 1.5351 / 3.0108 | 0.35381 / 0.099297 |
| Cumulative slip | 13.922 / 17.42 | 7.0515 / 8.6189 | 0.20434 / 0.054514 |
| Surface age | 5.6674 / 2.8165 | 2.726 / 1.2931 | 0.075613 / 0.0074424 |
| Localization normalization I | 0.033486 / 0.029208 | 0.044568 / 0.0085158 | 3.2486 / 1.6819 |

Whole-box temporal stress differences decrease .60781% → .30606%; corresponding
corner differences decrease .61748% → .31097%. The spatial differences are
.01998% whole-box and .04334% corner. The original large corner amplification
is absent in this consistent-boundary sequence.

Orders from absolute differences on common regions are .982/.984 for central/end
stress, .996/1.001 for normal traction, .960/1.021 for slip rate and 1.080/1.134 for
cumulative slip. Point shear orders are .821/.938, and age orders 1.017/1.105.
Geometry differences are small but have orders .804/.701. **Not every quantity
shows clean time refinement:** central I sensitivity grows .0335%→.0446%
(order -.412), while endpoint I decreases. This is retained rather than called
universal first-order convergence.

Final geometry L-infinity differences (central / ends, metres) are:

| Comparison | Geometry difference (m) |
|---|---:|
| time050_025 | 1.75973e-05 / 1.76627e-05 |
| time025_0125 | 1.04295e-05 / 1.06576e-05 |
| space32_64 | 1.51141e-05 / 1.12632e-05 |

The spatial initial slip-rate difference is only .0203% central/.00788% ends,
but initial I differs 2.921%/1.090%. Final I differences therefore include the
changed Q1 representation at initialization; they cannot all be assigned to
repeated transport. Initial normal profile variance is .0326292 m² on 32² versus
.0306761 m² on 64². Final variances are .0296368/.0276321 m². The profile deforms
under shear, so those changes are not interpreted as diffusion alone.

| Compatible case | Phase mass change (%) | Peak phase | Final normal variance (m²) | Localization measure / curve length − 1 (%) |
|---|---:|---:|---:|---:|
| 32², dt=0.05 | 0.00900185 | 0.345761 | 0.02963566 | 0.0129997 |
| 32², dt=0.025 | 0.00728199 | 0.345728 | 0.02963613 | 0.0165204 |
| 32², dt=0.0125 | 0.00109218 | 0.345198 | 0.02963684 | 0.0171315 |
| 64², dt=0.0125 | -0.00181252 | 0.347911 | 0.02763206 | -0.00727745 |

Mean accumulated label spin is -.10175, -.10192, -.10202 rad through the temporal
sequence and -.10211 rad on the fine mesh (about 5.85° clockwise). The curve stays
about .992 m from the boundary. Across all five runs, independent geometry replay
is within 1.12e-16 m, Q1 profile replay 5.0e-16, donor-bound violation 3.34e-16,
age source 1.78e-15 s and auxiliary memory source 1.39e-17 Pa m. All 104 selected
physical steps publish history and geometry once without retries. These identities
verify implementation; they do not replace the measured accuracy comparisons.

## Work, energy and flux

The endpoint active-work remainder decreases 6.2206%→3.1592%→1.5916%; the fine
mesh gives 1.5911%. The corresponding trapezoidal diagnostic is .13942%→.043143%
→.018392%, with .017992% on the fine mesh. The trapezoidal calculation is an
observer only; no endpoint evolution or acceptance rule was changed.

| Compatible case | Endpoint signed / absolute (J/m) | Trapezoidal signed / absolute (J/m) | Raw face − weak mechanical power (W/m) |
|---|---:|---:|---:|
| 32², dt=0.05 | -0.2143841 / 0.2143841 | 0.003837699 / 0.004804834 | -0.0005848948 |
| 32², dt=0.025 | -0.1088336 / 0.1088336 | 0.0002336816 / 0.001486281 | -7.95618e-05 |
| 32², dt=0.0125 | -0.05482003 / 0.05482003 | -0.0002983184 / 0.0006335026 | 0.0003044951 |
| 64², dt=0.0125 | -0.05480543 / 0.05480543 | -0.0002841191 / 0.0006197016 | 0.0002147022 |

Absolute totals sum absolute physical-step remainders; signed trapezoidal errors
show cancellation and would understate the error. The raw face-work defect is
**not monotone in dt** on 32²; it reduces from 3.04e-4 to 2.15e-4 W/m with mesh
refinement at the smallest dt, but one pair does not establish its order. Actual
weak mechanical power is about 13.09 W/m and is never assumed zero. Native q8
surface-work identities are checked with the existing observer; q12 sensitivity
in the final surface-work residual is about 9.14e-8 W/m on 32² and 1.41e-8 on 64².

Interior-trace energy flux is retained for comparison with previous reports.
A separate observer uses the prescribed **physical endpoint background** on
incoming faces and the numerical trace on outgoing faces. At32²/dt=.0125, final
interior flux is +2.9734e-4 W/m while this supplied-background flux is -2.3286e-4;
on 64² they are +4.8183e-4 and -2.2946e-4. This distinction matters even though the
magnitudes are small. Substituting the supplied flux in the endpoint continuum
diagnostic gives 1.59296% at32², versus1.59157% with interior trace. Neither flux
observer is the time-integrated SSPRK transport-stage flux: that stage uses the
**parent** exterior, and its saved ledger measures actual before/after energy.
The report does not claim an exact conservative stage balance from endpoint data.

## Cost and practical boundary

| New case | Mesh | dt (s) | Steps | Wall time (s), four ranks | Linear solves / iterations |
|---|---:|---:|---:|---:|---:|
| exterior-zero025 | 32² | 0.025 | 16 | 244.67 | 119 / 4354 |
| exterior-compatible025 | 32² | 0.025 | 16 | 225.99 | 106 / 3987 |
| exterior-compatible050 | 32² | 0.05 | 8 | 133.01 | 65 / 2445 |
| exterior-compatible0125 | 32² | 0.0125 | 32 | 399.92 | 180 / 6766 |
| exterior-fine0125 | 64² | 0.0125 | 32 | 1225.04 | 171 / 6683 |

All runs are sequential; at most four build/test workers were active. Native
CFL substeps per physical step are 6/3/2 on 32² for dt=.05/.025/.0125, and 3 on 64²
at dt=.0125. Thus the spatial comparison fixes physical dt and labels while
retaining native CFL subcycling. Core build took 180.14 s; the unit suite 8.83 s.
The finer mesh costs about 3.06× the 32² run at the same timestep, with similar
linear iteration count and much smaller slip sensitivity than timestep halving.

## Verification, preservation and limits

The new executable passes **183170 assertions in 152 existing unit cases**.
No dedicated rotation campaign was repeated. The separate provider/checker build
uses matching headers. Selection tests verify rejection of empty identity,
unsupported separated histories, mismatched attachment and missing provider.
Changing a copied prescribed-inflow checkpoint to accepted trace is rejected by
the existing material/quadrature identity gate.

A same-identity 4-to-2-rank restart successfully reloads the new checkpoint and
publishes one short .025-s step. The native resume loop advances before checking
the end time, so the initially intended reload-only probe advanced from .4 to
.425 despite End time=.4. This is retained explicitly as a lifecycle check and
excluded from the five-run accuracy matrix; original checkpoints were untouched.
It is not a restarted-versus-uninterrupted accuracy comparison. The guard helper
now names that interval explicitly. The trial source/geometry ownership is
unchanged; existing rejection evidence remains applicable. The benchmark's
uniform-schedule restriction explicitly rejects smaller adaptive retry intervals.

The restart probe exposed an omitted header in the new provider's diagnostic CSV.
The logging-only fix was compiled before the last three comparison runs; the
first DSO/source and all earlier logs are preserved. Every run records which
provider DSO it used. The physical callback and traction formula are identical
across these two logger builds. No solver adjustment, tolerance relaxation,
added dissipation, initialization change or failed scientific run was concealed.

The maintained design/specification and reading guide document the new callback,
split timing, direct-only restriction and checkpoint identity. STATUS,
SOURCE_AUDIT, REVIEW_PLAN and RESEARCH_HANDOFF distinguish this implementation
from the earlier frozen replay and defer the previously proposed frozen audit.
Prior binaries/runtime libraries, outputs, checkpoints, failed attempts and
manual documentation work remain. The inherited loss of **39/45 native-pilot
frames and earlier temporary logs** remains unrepaired.

**Implemented:** opt-in prescribed physical exterior, benchmark provider and
compatible loading, restart identity and five-run analysis. **Locally verified:**
unit suite, split/initialization checks, accepted publications, provider guards
and a short same-identity restart. **Physical comparison:** boundary-model
responses and bounded temporal/spatial sensitivities for this mature constant-
viscosity fixture. **Production readiness:** not established. The provider does
not qualify adaptive schedules, separated exterior histories, variable-viscosity
exterior models or production checkpoint conversion. Production BP3, cohesive
physical histories, unresolved cohesive energy/tip failures and the broader
Stage C roadmap remain separate.

## Concrete conclusion and next justified experiment

Retain prescribed parent exterior plus compatible endpoint traction as the
explicit reference boundary model for this bounded shear fixture. The five-run
matrix is complete; it does not support using the original zero-top/held-trace
case as a converged reference. No further frozen mechanical audit was launched.

The dominant remaining error for cumulative slip is **temporal**: its latest
7.05%/8.62% central/end sensitivity is far larger than the .204%/.0545% mesh
sensitivity. Stress and traction show much smaller sensitivities, while profile
normalization and raw boundary work retain their stated qualifications.

The next justified experiment is **one 32², dt=.00625 run with the same compatible
exterior/loading and fixed 18 labels**, compared with the completed .0125 case.
It would test whether the observed first-order cumulative-slip trend persists
and quantify the accuracy/cost of the retained endpoint scheme. This is a
recommendation, not an additional run or automatic continuation of the five-case
task. A broader spatial campaign, finite-end implementation or constitutive/
time-integrator change is not justified solely by a smoother plot.
