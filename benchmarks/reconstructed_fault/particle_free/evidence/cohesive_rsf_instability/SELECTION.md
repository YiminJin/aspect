# Native RSF instability coupled to growth — work in progress

Baseline f2f730529 plus preserved portable-workflow/domain-protection edits.
Production/BP3 and the qualified pilot remain unchanged. This is a separate
finite-wall quasi-dynamic experiment, not a seismic-wave calculation.

## Selection and criteria, declared before runs

Inputs: `benchmarks/reconstructed_fault/particle_free/cohesive_rsf_instability/`.
The 16 × 12 m prescribed-displacement rectangle contains a straight 6 m seed,
x=4–10, y=0. Mature a=1 on x=5–8; one-metre cohesive shoulders; a=0 on x=9–10.
All histories start from explicit physical fields. Uniform initial shear is the
regularized RSF resistance at V=1e−7 m/s, normalized age=10; q=(1−a)tau0,
reference=−C0 a tau0, initial jump/Maxwell displacement memory zero. The initial
rate predictor is zero; the first resolved interval establishes its physical
rate. This startup is measured separately, not called a nucleation event.
Uniform initial stress is mechanically equilibrated. The permanent seed endpoint
and finite walls are physical constraints of this specimen.

G=1e6 Pa, eta=1e12 Pa s, p0=1e4 Pa, peak=2000 Pa, Gc=3 J/m²;
RSF a=.005, b=.02, mu0=.1, V0=1e−6 m/s, Dc=6e−5 m;
radiation impedance=25836.0213655 Pa s/m. Pure engineering shear rate=1e−7/s
is constant, without an imposed acceleration. Physical R=.75 m and resolved
numerical lifting half-width=.75 m retain their separate meanings.

An elastic single-mode linearization at steady V, ignoring negligible asinh
correction only for this scale estimate, gives

`(pa/V+eta_rad) lambda² + [k+(pa+eta_rad V−pb)/Dc] lambda + k V/Dc=0`.

Thus `kc=[p(b−a)−eta_rad V]/Dc ≈ 2.5e6 Pa/m`. The mode-II scale
`[G/(1−nu)] Dc/[p(b−a)] = .8 m` (nu=.5) is an order-of-magnitude estimate,
not a substitute for the actual finite-domain, current-pressure native operator.
Maxwell time eta/G=1e6 s substantially exceeds the declared 2000 s horizon.
Radiation provides rate resistance, not inertia or resolved waves. The original
pilot's much larger Dc gives kc=5e4 Pa/m and does not fit the same instability.
The numerical finite-domain spectral check will retain K, pressure reaction N,
state dynamics, and radiation; small perturbations need no cohesive softening.

Minimum cohesive scale estimate G Gc/peak²=.75 m (the actual residual-strength
difference increases it). df=2Gc/peak=.003 m=50 Dc. Primary uniform mesh96×72,
Delta x=Delta y=1/6 m; surface spacing1/16 m. This resolves support by4.5 bulk
cells and the smaller estimated process length by12 surface intervals. Affordability
and adequate approximation require measured setup cost and a targeted comparison;
these counts alone do not establish convergence. A 128×96 common-field reference
is planned only if the primary operators support progression. No AMR/history
conversion is part of this task.

Native adaptive stepping targets slip/Dc<=.1, |Delta log theta|<=.1,
|Delta log(|V|+V0)|<=.2, and Delta a<=.02; safety .8, growth1.25, dt max20 s,
initial .01 s. Oversized trials reject before publication; the same immutable
incoming state is retried. The regularization in the speed *indicator* never
alters friction. Native weak/source/contact/work tolerances are unchanged.
The retained endpoint symmetric tangent is a discrete step guard, not a claim
that physically unstable RSF trajectories cannot exist.

Computational cap:600 accepted steps or2000 physical seconds; checkpoint every
accepted state. Initial probe budget15 minutes, main run budget60 minutes; extend
only within this bounded event investigation, not a full-event/server campaign.
Physical targets:>=100× initial-reference rate, state weakening on the original
mature patch, several Dc slip and measurable cohesive/full-front change. Stop
and diagnose contact/domain/solver limits. Save the acceleration window for
half-step/increment replay. Report front brackets, growth/slip and speed/state
uncertainty, with a1% physical-work screen against sliding dissipation and
fracture separately. Failure of this screen remains visible. A velocity-neutral
control b=a preserves initial friction exactly by compensating mu0 at theta=10;
its law differs only in state feedback after that common initialization.

Reference basis: [Rice & Ruina (1983)](https://esag.harvard.edu/rice/105_RiceRuina_StabFrictSlip_JAM83.pdf),
[Gu et al. (1984)](https://esag.harvard.edu/rice/111_GuRiRuinaTse_RateStateStab_JMPS84.pdf),
and the [SEAS quasi-dynamic benchmark conventions](https://strike.scec.org/cvws/seas/benchmark_descriptions.html).
The above finite-step implementation and eigenanalysis are computed independently
for this law; benchmark literature is not evidence of this specimen's stability.

## Results

Native build passes. Forced outer rejection/retry gives bitwise identical DG/native
bulk coefficients, all six surface histories, geometry and outputs at six accepted
states (.005–.05 s). Restart at the penultimate rolling checkpoint repeats both
remaining endpoints with bitwise physical agreement and unchanged publication counts.

The actual 96×72 mature-patch restricted operator has minimum generalized stiffness
821340.714253 Pa/m, below critical2499956.939964 Pa/m. Full K/N/Maxwell/state/radiation
linearization gives maximum real growth rate .001966337858/s (e-fold508.5596 s).
The matched b=a control has −.000821841334/s. Pressure is included; its relative
contribution on this symmetric straight-patch subspace is1.83e−16. This vanishing
coupling is a consequence of symmetry, not imposed friction pressure. The nonlinear
algebraic action finite-difference error is2.14e−8; independent scalar elastic
characteristic roots agree within1.06e−16/s. This is a steady-rate diagnostic on
admissible mature-patch perturbations, not a claim that the nonuniform startup is
an exact steady sliding solution. Native evolving feedback remains to be tested.

The first selected-mesh step initially failed the exact BE guard. Its legacy
bulk subtraction was .000248176311487214 J/m, whereas the independently factored
increment was .000248176059653541 J/m. Subtracting two ~173.767 J/m independently
summed storages introduced2.51834e−10 J/m. With extended accumulation of storage
and `(tau_new+tau_old):(tau_new−tau_old)/(4G)`, the unchanged guard residual is
−2.89536e−14 J/m instead of−2.51863e−10 J/m. No mechanics/history/tolerance changed.
The old arithmetic remains an exported diagnostic; the original failed run and
initial accepted checkpoint are retained. Setup104.75 s, peak memory470.06 MiB/rank;
source1.014e−15 scaled, mechanical weak1.675e−12 Pa, surface weak2.665e−12 Pa.
The one-step trapezoidal physical remainder is1.0554e−12 J/m. This is not an event
energy qualification.

### Mesh selection after the bounded probes

The 128×96 uniform control costs186.77 s including setup and one step. Its
minimum mature stiffness811449.860 Pa/m differs by1.22%; the growth rate
.002074416818/s differs by5.50% (e-fold482.06 s). Neither mesh changes the
instability/stability classification. Common smooth jump Gaussian widths .4/.8 m
give storage/work differences .54/.92%, but the complete Q1 weak-load mass norm
differs19.1/12.1%. High-frequency surface loads are substantially less resolved
than integrated work. The truncated1.2 m Gaussian also has an artificial cutoff
and is not used as an accuracy target. These failures are retained.

Select **128×96, Delta x=Delta y=.125 m**, surface spacing .0625 m, for the
native event (`event.prm`). Fixed .75 m support/cohesive scale now has6 bulk cells
and12 surface intervals; estimated .8 m nucleation scale6.4 bulk cells. Uniform
mesh cost/memory is affordable for the bounded event; no AMR is introduced.
This improves the approximation but does not claim point-load convergence or
1% onset-time accuracy. The 5.5% linear growth-rate spread and old traction
findings remain spatial uncertainty. No physical parameters changed after the
probes. A full uniform-refinement campaign is outside this event demonstration.

The cohesive length estimate depends on the instantaneous residual strength:
at the initial slow reference rate, tau_r≈1345.39 Pa gives
G Gc/(peak−tau_r)^2≈7.00 m, while strong RSF weakening reduces this toward the
.75 m lower bound. The latter sets the demanding spatial resolution, not the
initial zone's equilibrium extent. The prescribed one-metre damaged shoulders
are an admissible underloaded pre-existing damage obstacle (e/df=a tau0/peak<a),
not an asserted steady process-zone solution. This is an explicitly finite-wall,
prescribed-path specimen. The initial x=9–10 tied interface is installed geometry;
damage/full-decohesion growth there must be distinguished from later geometric
append. The event may test coupled growth without claiming a distant-boundary
process zone or a newly selected free path.
