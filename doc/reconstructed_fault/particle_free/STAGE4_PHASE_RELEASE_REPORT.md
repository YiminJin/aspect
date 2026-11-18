# Stage 4 bounded phase-release pair

The user selected the 256×128 weakening / matched b=a phase-release pair from
the completed short-wave/native pilot, adding a measurement of the phase driving
for an admissible increment just ahead of the initial tip. Both histories use
the exact same fresh static preload, retained phase/planes, 1% smooth physical
age perturbation, uniform shear/extension velocity, material data and midpoint
dt=8e-5 s. The target is eight intervals to .00064 s, four aggregate ranks,
45 minutes per case. The existing coupled phase obstacle is enabled by removing
only the fixed-phase benchmark constraint. Production code, equations, solver
criteria and prior checkpoints are unchanged. This is a synthetic initial
condition, not a restart of the .075 growth trajectory or a reconstructed past
loading/aging history. The preload equilibrates mechanics; it does not perform
an additional phase equilibration before t=0.

## Driving measurement

The [benchmark derivation and reproduction guide](../../../benchmarks/reconstructed_fault/particle_free/stage4_release/README.md)
defines two fixed nonnegative Bernstein directions. Their transverse tent is
y=1.875–2.125 m. Probe 0 covers x=6.03125–6.15625 m, just beyond the initial
absolute phi=.05 contour. Probe 1 covers x=6.3125–6.4375 m, the first completely
intact band. Tiny represented phase coefficients beyond the visible taper are
retained; neither the contour nor a threshold is used to silently delete them.

For a probe chi, positive net driving

    D_net = integral chi Y dx - chi . gradient E_fracture

means this direction lowers the reduced phase functional. The native fracture
gradient includes every AT1/SIPG cell and face term. The probe's weighted area
is .0078125 m²; dividing by it gives the reported mean net driving in Pa.
Nonnegative coefficients respect the lower obstacle; the available upper-bound
amplitude is recorded. After a physical interval the diagnostic uses the actual
accepted midpoint state. Endpoint phi/slip/energy retain their separate times.

At t=0, T=tau+2G A and the verified section resistance f define F=T:S-f. The
phi-dependent relaxed local potential is -H(phi)F²/(2G), with
H=m phi(1+phi)/[m phi(1+phi)+(1-phi)²]. Thus Y=H'F²/(2G), including the intact
right limit m(T:S-f)²/(2G). Holding A=0 and reporting zero at intact points would
omit the admissible coupled deformation. This uses the selected section closure's
intact reaction, not a new constitutive rule. The observer advances no time and
publishes no probe state. It checks one-sided differences of the actual native
fracture energy plus this local potential at amplitudes 1e-4, 1e-5 and 1e-6.
The initial measurement uses the stored section planes, without an additional
orientation optimization. Later measurements use the solver's actual accepted
planes, including ordinary selection in wholly intact material. Their changes
can therefore include that selection as well as loading. Formed planes keep
their existing history meaning.

The weakening t=0 values are:

| Probe | Mechanical driving (J/m per amplitude) | Fracture derivative | Net driving | Mean net driving (Pa) |
|---|---:|---:|---:|---:|
| Ahead of phi=.05 | .00054498044 | .01466947029 | −.01412448985 | −1.80793470 |
| Fully intact band | .00115310689 | .01562500000 | −.01447189311 | −1.85240232 |

Both directions initially oppose damage. Their one-sided derivative errors
decrease approximately tenfold with each tenfold amplitude reduction, reaching
7.40e-6 and 7.23e-6 relative to the declared force/area scale. The b=a preload
has matched initial force and mechanical history; its independent diagnostic is
retained in its run. Negative results for these two shapes do not prove that all
admissible directions elsewhere are unfavorable. The first coupled trial's
nonzero phase residual explicitly makes that distinction necessary.

## Completed response

Both runs exited zero with all eight intervals accepted. The initial phase
coefficients are identical across laws. Initial stress, resistance and driving
also match to roundoff. The weakening and b=a runs took 1092 and 1134 s,
respectively; the entire sequential numerical batch took 37.1 minutes. No run
needed its 45-minute cap, tolerance changes, a retry, or an extra case.

| Quantity | Weakening | Matched b=a |
|---|---:|---:|
| Endpoint time (s) | .00064 | .00064 |
| Last rate/driving stage time (s) | .00060 | .00060 |
| Peak stage slip rate (mm/s) | 2.135124 | 1.649643 |
| Spatially integrated endpoint slip (m²) | 1.19776809e-6 | 1.08920244e-6 |
| Mean patch section resistance (Pa) | 126.503263 | 126.535169 |
| Right connected phi=.05 location (m) | 6.00205437994 | 6.00205437994 |
| Left connected phi=.05 location (m) | 1.99794562016 | 1.99794562016 |
| Advance of either contour end / h_x | 0 | 0 |
| Maximum increment in exactly intact cells | 0 | 0 |
| Added fracture inventory (J/m) | 8.98235960e-8 | 8.98234933e-8 |
| Final mean probe-0 net driving (Pa) | −1.80783869 | −1.80783110 |
| Final mean probe-1 net driving (Pa) | −1.83460442 | −1.83460860 |

The smaller-Dc weakening law produces a 29.4% larger final peak rate than b=a,
but it does not produce a distinguishable phase-growth response here. The final
phase-coefficient difference between laws is at most 8.74e-11; its exact affine
Q2 L2 norm is 6.22e-12 m. Their added fracture inventories differ by 1.03e-13 J/m.
Relative to each law's retained **perturbed fixed-phase** control over the common
first four intervals, enabling growth changes peak rate by at most 8.35e-5,
integrated rate by 3.71e-4 and integrated slip by 3.21e-4 relative. The old
eight-interval unperturbed background is not substituted for the missing longer
perturbed fixed-phase controls. This is early forced acceleration, not a measured
autonomous RSF instability or a fit to the frozen spectral growth rate.

The phase response is a small adjustment of the existing band and taper,
concentrated in the first three intervals and unchanged thereafter to the
recorded precision. Quadrature-sampled maximum increments are 4.30473e-4 in
the central band and 2.27856e-4 in the old taper/tail. The global Bernstein
coefficient increment reaches 9.05971e-4; it is a coefficient bound, not a
pointwise phase maximum. Initially intact surroundings and remote material
have zero increments and zero added fracture work.

There are small **off-axis** changes in the pre-existing low-phase tail beyond
the .05 contour: changed coefficient lattice positions reach x=6.0625 m, in
cells whose support ends at x=6.09375 m. The
largest coefficient increment at or ahead of the initial front's cell is
1.10553e-4, at a lattice position near y=1.90625 m; it must
not be hidden by a contour-only claim of no phase change. However, the axial
y=2 trace is unchanged for x≥6 m, no initially zero cell acquires damage, and
neither connected .05 endpoint advances. The axial increment profile has a
small localized change near x=5.875 m within the old taper. The near-identical
response of both laws and its early plateau are consistent with an initial
constrained band readjustment, not demonstrated weakening-driven extension.

Connected components are identified from the seed, then their extreme x
locations are obtained from the native cell polynomials, retaining separate DG
traces. Four and eight samples per cell for the connectivity search give the
same initial/final extrema; h_x=.03125 m. All observed states have one sampled
.05 component. Figures are sampled visualizations; the reported contour
positions use cell-local polynomial roots, not interpolation across cell faces.
These measurements do not establish a general crack-front definition or
arbitrary turning/branching capability.

## Fracture work and physical work

For weakening, the AT1 potential increases by **2.75836751e-5 J/m**, while
gradient/face energy decreases by **2.74938515e-5 J/m**. Their net is the
8.98236e-8 J/m fracture-inventory increase above. This physical cancellation
reflects profile adjustment; it is distinct from a numerical work defect.
The exact row partition assigns 7.99769e-8 J/m of the net increase to the
central band and 9.84674e-9 J/m to the old taper/tail, with zero in the other
two regions. The b=a values agree to the precision needed for this conclusion.

Signed cumulative / accumulated absolute physical-work errors are
**+2.42495e-11 / 2.42495e-11 J/m** for weakening and
**+1.06052e-11 / 1.06052e-11 J/m** for b=a. They are respectively .0270% and
.0118% of the added fracture inventory, against about .069442 J/m absolute
external work. Numerical work loss does not dominate this small fracture change;
no time-accuracy claim follows from that fact. Mechanical, cohesive midpoint,
phase-residual and total physical-work terms remain separately available.

Signed local deformation is retained: final integrated positive/negative rates
are .00457753/−4.62457e-6 m²/s for weakening and .00379823/−1.55096e-6 m²/s
for b=a. Their cancellation fractions are .2019% and .0816%. No clipping removes
opposite-sign deformation. All publication, native fracture-partition and
irreversibility checks pass. The initial directional checks pass independently
for both laws. Restart is not exercised; the benchmark observer explicitly
requires its fresh initial reference, while physical checkpoint meanings are
unchanged.

## Decision and next justified experiment

**This preload/window is under-driven for axial fracture.** For the intact
probe, the actual material relation m=2 G Kf/strength² gives
D_mech/D_frac=weighted_mean[(F/1000 Pa)²]. Its weighted RMS effective force
rises only from about 272 to 288 Pa; even the final integrated mechanical
driving supplies only 8.27% of the fracture derivative. Both fixed probe
directions stay admissible throughout and their net driving stays negative.
This is a measured deficit for these shapes and the native selected planes,
not proof against every possible shape/orientation or future loading history.

The pair therefore demonstrates slip-law separation with no intact-material
extension in the selected window. It does not establish Example 3's desired
instability-growth interaction. Extending this same run or refining its mesh
is not the next justified growth experiment. First perform one fixed-phase
**preload calibration** using the actual mechanical response: bring the chosen
intact ahead-tip probe close to D_net=0, check central-band and remote driving,
and specify how phase is equilibrated or inherited before selecting another
free interval. Target the model's effective-force scale, not a guessed multiple
of boundary shear; redistribution and section resistance also enter F. That
calibration is proposed only. No additional run is launched.

## Deliverables and reproduction

Compact histories, initial/final coefficient arrays, commands and figures are
under `benchmarks/reconstructed_fault/particle_free/evidence/stage4_release/`:

* [Driving, slip, contour and work history](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_release/release-history.png)
* [Phase and phase-increment fields](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_release/release-phase.png)
* [Initial/final axial phase and increment](../../../benchmarks/reconstructed_fault/particle_free/evidence/stage4_release/release-tip-profile.png)

Each figure also has a PDF. `weakening-history.csv`, `neutral-history.csv` and
`comparison.json` contain the quantitative observations. `analyze.py` reproduces
them from native coefficient/energy records; `retain.py` verifies unchanged
incoming source and protected runtimes, saves exact inputs and the tested checker,
and hashes retained checkpoints in place. Original step fields and snapshots
remain in the run directories; no cleanup or failed-evidence deletion occurred.

## Ownership and preservation

Entry root/branch: `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`, HEAD
`b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`. Incoming source/manual work is retained.
The main executable remains the protected `first-free-turn/tested/aspect-release`
with SHA256 `089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`;
its identity is independent of current HEAD. The separate checker and input
identities are recorded in `build-particle-free/stage4-release/entry.json` and
`build-command.json`; individual run commands also record source/runtime hashes.
The old pilot and original first-free-turn checker are preserved.

`observe.h` reads accepted histories and the existing accepted-stage evaluation.
The initial field/gradient retained by this observer are diagnostic references,
not another physical history owner. It exports native coefficients compactly
and uses the exact quadratic identity delta B_i(g_i(initial)+g_i(endpoint))/2
to partition fracture-energy change across fixed regions, including face work.
The regional sum is independently checked against the native total. Regional
row assignments are explicit and need not be positive. The reported physical
work error is Wext−delta E−Db−Dc−Df before subtracting cohesive midpoint or
residual remainders, with both signed and accumulated absolute errors.

Raw outputs, failed evidence, executable identities and checkpoints are kept;
previous cleanup availability losses remain unchanged. No commit is requested.
