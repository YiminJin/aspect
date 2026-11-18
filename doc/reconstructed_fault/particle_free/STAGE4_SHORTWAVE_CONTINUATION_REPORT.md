# Stage 4 matched short wavelengths and native pilot

This continuation supersedes the previous wavelength task's decision to defer
the native pilot. The user selected a same-disturbance 256/512 comparison and
allowed a finite short-wave growth plateau to justify a short fixed-phase test.
Spontaneous patch spacing need not be settled first. The native comparison is
now complete and supports a bounded phase-release control as the next experiment.
No free growth was run and no prior growth checkpoint was converted.

## Matched mechanical and spectral comparison

The native .075 phase is subdivided exactly in x, with normal spacing 1/32 m,
the same planes, section measures, confinement, and physical rate/age profile.
The force patch remains x=2.5–5.5 m. All responsive ages enter the spectrum.
These are continuous-time frozen spectra, not measured nonlinear event rates.

| Mesh | Wavelength (m) | x cells/wave | Force pattern | Stiffness (MPa/m) | Growth (1/s) |
|---|---:|---:|---|---:|---:|
| 256×128 | .0625 | 2 | sine | .815391 | 34.06896 |
| 512×128 | .0625 | 4 | sine | .815556 | 33.90412 |
| 256×128 | .0625 | 2 | cosine | .649774 | 37.61975 |
| 512×128 | .0625 | 4 | cosine | .813800 | 32.80711 |
| 512×128 | .03125 | 2 | sine | .656157 | 38.03029 |
| 512×128 | .03125 | 2 | cosine | .576556 | 39.85660 |

Resolving yesterday's .0625 disturbance raises the cosine restoring stiffness
25.24% and lowers its growth 12.79%. The sine stiffness changes only .0202%.
Their pair's leading growth decreases 37.61975→33.90412/s. In identical
six-smooth-plus-.0625-pair spaces it decreases 37.61875→33.90306/s. The smooth
six-mode control changes only 15.11809→15.11944/s. The newly accessible .03125
pair gives 39.85660/s; its two-cell result is not a limiting-rate measurement.
The six smooth modes plus both pairs give 39.83873/s, with signed wavelength
.0312498 m. The six smooth modes plus the shorter pair alone give 39.85652/s.
Thus adding the neighboring measured pair changes growth only .045%. Matched
b=a and original-Dc controls have no positive growth above roundoff in either
pair or the ten-pattern coupled space (maximum 2.56e-13/s).

The joined 512 data contain five complete smooth actions from the shutdown run,
the fully repeated sixth action, and four completed new sine/cosine actions.
The 3,034 complete old rows of the sixth action agree **exactly** with its repeat.
The partial buffered tail and interrupted commands remain unchanged. The joined
dataset is explicitly a collection of frozen actions, not a successful physical
trajectory attributed to an interrupted process.
Maximum native equilibrium, work and point-derivative errors over the ten
actions are 9.33e-12, 1.40e-12 and 6.17e-16. The single-worker full-age spectrum
analysis took 427 s. [Matched comparison figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_shortwave/matched-shortwave.png)
is also available as PDF beside the PNG.

## Compatible short-wave prediction

[Derivation and scope](SHORTWAVE_COMPATIBILITY.md) use the actual transverse DG
phase polynomial and incompressible compatible displacement, with clamped top
and bottom. They retain the tilted-plane elastic penalty. At a frozen section,

    C∞ = cos(a)/G ∫ h/(1+h sin²(2a)) dy,
    h = 4 phi(1+phi)/(1-phi)².

The fixed 3 m force carrier, with the actual varying V and theta and every age,
has limiting growth **41.13219/s**, imaginary part **90.96370/s**. The normal-Q2
mass projection gives **41.13223/s**. The compatible slow-x finite-wavelength
reference approaches that limit: 33.3954, 37.8756, 39.9250, 40.7422, 41.0184/s
at wavelengths .0625, .03125, .015625, .0078125, .00390625 m respectively.
This slow-x approximation is not an exact finite-body native spectrum.

A separately localized packet can approach the fastest local principal-symbol
edge, **61.67179/s**. That value now has a compatible mechanical derivation;
the earlier uncoupled number alone did not establish it. Neither this packet
edge nor the fixed-carrier rate equals the general **155.2/s upper bound**.
Allowing arbitrary envelopes and restricting a carrier to a fixed patch are
different spectral questions. No selected wavelength follows from either limit.

Checks: normal 256→512 reference refinement changes compliance by at most
3.15e-6 relative over the tested wavelengths. At .03125, quadrature 6/10/16
agrees to roundoff. Maximum equilibrium and independent work errors are
1.04e-13 and 3.12e-13. The full 480-age matrix reproduces the rational carrier
root within 4.41e-13/s; matched b=a and original-Dc carrier controls decay.
All native ages, including passive zero modes outside the patch, remain in the
native spectra, where stable controls can have roundoff-positive zero roots.
The final limit integration splits at all 336 actual interpolation/profile
intervals. Along-fault orders 6→10 and normal orders 10→16 change the root by
less than 3e-14/s. The older unsplit order-10 estimate, retained in
`compatible-limit.json`, differs by .000736/s; `limit-quadrature.json` supplies
the final numbers and figure. Independently substituting all ten growing native
roots into the small dispersion matrices in original force coordinates gives
relative smallest singular values at most 1.13e-15. This checks the roots without
relying on the dimensionally scaled full-age matrix's normalized residual.

## Native preparation and current boundary of evidence

A separate benchmark checker prepares a fresh loaded native state using the
ordinary mixed operator and ASPECT Stokes inverse. It prescribes the same smooth
V(x), uniform theta, and signed friction/radiation force, then equilibrates
physical tau and cohesive A under an affine elastic boundary displacement.
The static boundary shear is gamma=2e-4 and extension=.4195498156 gamma;
subsequent velocity is the retained uniform shear .01/s and extension
.004195498156/s. This is a declared synthetic initial condition, not the .075
growth history's stress or a checkpoint conversion. Phase and planes are the
retained specimen. Full native cohesive history and energy are retained.
The preload is a static boundary-displacement problem, not an integrated
loading/aging trajectory. Its identity `2 storage + integral(f e) = boundary
work` tests the static variational solution; the `f e` term is not a claimed
frictional dissipation history. Instantaneous rate follows from the prescribed
age and equilibrated `(tau-Q):S`. Subsequent intervals are actual native
physical evolution. Reachability of this synthetic initial condition under a
specified past loading trajectory is not established.
Initial section rates are instantaneous inversions of the equilibrated signed
law. Pointwise last-stage rate observations are not invented at t=0; the first
physical interval supplies them and checks the actual integrated-rate/power
identities. `pilot-sections-0000.csv` gives the actual perturbed age field;
the generic `tip-initial.csv` age column is only the nominal input parameter.

The preflight checks the free mixed residual, local tau−Q traction, static
boundary work, initial age and intended rate. The first attempt stopped before
mechanics because the benchmark called the positive-rate friction API below
its minimum. The corrected benchmark evaluates the existing signed section law
at precisely the prescribed age (analytically undoing its BE age map for this
fixed-state evaluation). No rate floor, law change or tolerance change is used.
The failed checker and log are preserved.

The corrected two-rank preflight passed and accepted one physical interval:
mixed residual 8.26e-12, initial work error 1.54e-14, point traction error
1.85e-13 Pa and initial rate error 2.63e-18 m/s. Its initial bulk plus cohesive
storage is 1.0538063 J/m. The first midpoint rate peaks at .000226598372 m/s
at stage time .00004 s; its endpoint is .00008 s. Initial preparation, one
interval and two checkpoints took 311.53 s on two ranks. Repeating this first
interval on four ranks changes its peak rate by 4.16e-12 m/s. Neither check
establishes temporal convergence. Native direction selection remains enabled
in wholly intact, zero-slip sections; formed active planes and phase stay fixed.

The common perturbation is a 1% age decrease proportional to
sin²(pi(x−2.5)/3), supported on the same physical patch, applied after the
matched preload. Each law has its own undisturbed background. The matched b=a
reference coefficient .6080471895621705 gives the same initial traction as
the weakening law. Midpoint dt=8e-5 s keeps the existing monotone section-root
condition. Measured native cost selected a common four-interval window
(.00032 s) for the two-mesh/law/perturbation comparison, with four ranks per
run. The initial 256 weakening background completed its unchanged
original eight-interval schedule to .00064 s; comparisons use its common
prefix. This short window measures early forced acceleration and rate doubling,
not a complete autonomous instability event. All eight histories completed;
the seven supervised cases exited zero within the existing caps.

The first 256 weakening background completed eight intervals with a final
snapshot: at endpoint .00064 s, the last stage rate peaks at .00210180 m/s,
integrated slip is 1.18725e-6 m² and mean section resistance has risen 5.60154 Pa.
Accumulated absolute/signed physical-work errors are 8.70e-11/−8.70e-11 J/m,
against .069442 J/m accumulated absolute external work. The last integrated
positive/negative local rates are .00453155/−4.54068e-6 m²/s: cancellation is
.2002%, retained explicitly. At the common four-interval endpoint, integrated
slip is 2.72169e-7 m² and the last stage peak is .000508494 m/s. The diagnostic
4e-4 m/s crossing is a rate-doubling marker, not a physical onset criterion.

The launch wrapper was terminated with signal 15 during interval two; the
detached four native ranks continued. Normal termination, all eight accepted
records, completed snapshots and rank exit were verified separately in
`completion-observed.json`. No successful process exit code was invented for
the interrupted wrapper. The remaining sequential cases use a durable
supervisor, per-run time caps and a two-hour aggregate batch cap; a `stop-batch`
file prevents the next case from starting. No more than four ranks run together.

The undisturbed native background doubles its peak rate in roughly .21 ms,
whereas a frozen 41.13/s perturbation would double in 16.85 ms. These are
different measurements: the fast loaded-background acceleration must not be
reported as the frozen RSF growth rate. The four-interval pilot tests spatial
consistency of this early forced response. Both absolute perturbation changes
and perturbations relative to each law's own evolving background are retained.
No exponential fit to the total rate is used to claim an instability.

## Completed native comparison and decision

The table uses the common endpoint .00032 s; rates and resistance are at the
last midpoint stage .00028 s. Traction here means the section resistance
`(tau-Q):S`, averaged over x=2.5–5.5 m, not the bulk shear stress alone.

| Mesh | Law/case | Peak stage rate (mm/s) | Integrated slip (m²) | Mean resistance change (Pa) | 4e-4 m/s marker (ms) |
|---|---|---:|---:|---:|---:|
| 256 | weakening/background | .508493745 | 2.721694003e-7 | 2.711693314 | .211184069 |
| 512 | weakening/background | .508493869 | 2.721693255e-7 | 2.711693293 | .211184027 |
| 256 | weakening/perturbed | .515058843 | 2.741012610e-7 | 2.711080654 | .208003108 |
| 512 | weakening/perturbed | .515058969 | 2.741011867e-7 | 2.711080632 | .208003068 |
| 256 | b=a/background | .491862423 | 2.703698674e-7 | 2.712208387 | .216333655 |
| 512 | b=a/background | .491862538 | 2.703697829e-7 | 2.712208368 | .216333610 |
| 256 | b=a/perturbed | .496231342 | 2.717319439e-7 | 2.711772118 | .213869813 |
| 512 | b=a/perturbed | .496231459 | 2.717318597e-7 | 2.711772098 | .213869768 |

Across all four paired histories, maximum mesh differences are 2.46e-7 relative
for peak rate, 3.40e-7 for integrated rate, 3.13e-7 for integrated slip, and
2.12e-8 Pa for the resistance change. The interpolated rate marker differs by
at most 4.57e-11 s; this is spatial agreement of a diagnostic, not onset-time
accuracy at an 8e-5 s timestep. Whole-section profile comparisons, interpolating
coarse Gauss observations onto fine ones, give relative weighted L2 differences
up to 1.76e-4 in rate, 1.79e-4 in slip, 2.74e-5 in age and 3.78e-3 in resistance.
The latter includes sharp transitions and sampling/interpolation differences;
these are not exact FE-field errors or a claim of pointwise traction convergence.

The same age perturbation produces different initial rate perturbations under
the two laws. Normalize each relative rate perturbation by its own initial
value and its own evolving background. On 512 columns this amplification is
**1.04425542 for weakening** and **.95642232 for b=a**; the corresponding
256 values are 1.04425531 and .95642220. Absolute integrated-rate perturbations
increase by factors 2.548424 and 2.297570 respectively because both backgrounds
accelerate strongly. Thus the native pilot distinguishes the laws in the early
response without misidentifying forced background acceleration as an autonomous
RSF event. [Native comparison figure](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_shortwave/native-pilot/early-native-comparison.png)
and its PDF retain all four perturbed histories; dashed coarse and marked fine
curves overlap at this scale.

All histories have one publication per accepted interval. Maximum accepted
mechanics residual is 1.08e-11; the preload residual across meshes/laws is at
most 1.83e-11. Power-adjoint and section-residual work errors are at most
1.46e-19 and 2.08e-17 respectively. Fracture inventory changes by at most
6.22e-15 J/m with phase frozen. Initial discrete fracture inventories differ
by 6.22e-7 J/m (5.42e-7 relative) between meshes despite exactly subdividing
the phase polynomial; their discrete SIPG integration/penalty is mesh-dependent.
Over the common four intervals, signed physical-work errors range from
−6.48e-11 to −6.15e-11 J/m; accumulated absolute errors equal their magnitudes,
against .034443 J/m absolute external work. Their small net values therefore
do not conceal cancellation. Final local-rate cancellation is .035–.052% in
the four-interval cases and .2002% at the longer background endpoint. The
signed positive/negative integrals remain in `runs.csv`.

**Decision for Example 3:** retain the smaller Dc for a bounded test of whether
this spatially consistent acceleration drives connected tip damage. Do not claim
a preferred patch spacing, temporal convergence, or a completed autonomous
instability. The specific next experiment is a fresh t=0 phase-release pair on
256×128, with this exact equilibrated preload, phase obstacle, 1% smooth age
perturbation and uniform boundary velocity: weakening versus matched b=a,
midpoint dt=8e-5 s, at most eight intervals to .00064 s. Keep solver tolerances
unchanged, four aggregate ranks and a 45-minute cap per case; stop at the cap
rather than extend into a campaign. Track central/taper/ahead/remote phase and
added fracture energy, connected axial increments and the absolute phi=.05
contour, signed section deformation, slip, traction and physical work. Retain
the current fixed-phase histories as controls. This tests acceleration-driven
damage, including the possibility of no growth; a later autonomous-event claim
would require a longer qualified observation window and time refinement.
This experiment is proposed, **not launched**.

The seven-case batch took 6949 s (115.8 min), including startup and checkpoints.
Individual four-interval costs were 495–536 s on 256 columns and 1315–1414 s
on 512 columns. The first eight-interval background reported approximately
1000 s native wall time; its interrupted wrapper remains separately recorded.
The two-rank preflight took 312 s. All solver ranks and the batch supervisor
have exited. No production source, solver criterion, or protected runtime changed.

## Reproduction and preservation

Root/branch were verified as `aspect-pf-rsf-particle-free` /
`pf-rsf-particle-free`, HEAD `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`.
Incoming dirty source/manual changes are preserved. The protected executable
SHA256 is `089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`;
its banner is older than HEAD. Production sources and that executable were not
rebuilt. Benchmark checker build manifests identify each generated source and
shared library independently. The four new short actions took 492.32 s including
startup on four ranks. Aggregate numerical/build workers never exceed four.

Scripts live under `benchmarks/reconstructed_fault/particle_free/stage4_wavelength/`:
`continue_analysis.py` joins and analyzes complete actions, `shortwave.py` and
`check_shortwave.py` establish the compatible reference, `plot_shortwave.py`
generates figures, and `prepare_pilot.py` / `run_pilot.py` prepare/run the separate
native experiment. Compact tables, provenance and figures are under
`benchmarks/reconstructed_fault/particle_free/evidence/stage4_shortwave/`.
`check_native_spectra.py` and `check_limit_quadrature.py` provide the final cheap
checks; `analyze_pilot.py` produces common-window and own-background comparisons.
`retain_pilot.py` preserves the tested checker, exact inputs and hashes of retained
restart payloads in place; it does not copy those payloads into compact evidence.
Exact commands and input/runtime hashes remain in each run's `command.json`.
No historical output, checkpoint, protected binary or manual work was deleted.
Earlier cleanup availability losses remain in their original cleanup records.
