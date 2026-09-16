# K5: frozen-Airy initialization-interval sensitivity

## Decision

The 1 s artificial Maxwell interval explains the enormous **rate-form bulk
velocity**, but increasing it to 4e6 s does not restore the intended BP3
initial state. The corrective velocity scales almost exactly inversely with
the interval; its displacement-like field persists. Surface equilibrium with
the official Vinit/Theta0 improves partially, but the largest free V remains
12.665 Vinit and the shear-target error remains above 1 MPa. Retain the
discrete-prestress mismatch as unresolved. This experiment neither establishes
the necessity nor verifies the sufficiency of a prestress-offset redesign.
No real BP3 steps were run and no production algorithm was changed for this test.

## Controlled experiment

Reuse the completed `initialization04` 1 s run. Run one complete initialization
at 4e6 s using the saved, corrected Airy plugin source
`evidence/initialization04-bp3.cc`, in the isolated `airy_dt0/airy.cc` plugin.
The existing unfinished offset implementation is not selected. Its generic
optional API is present in the current executable, but no background property
is supplied; this experiment retains full Airy particle Maxwell history and
the full bottom Airy traction, not stress-change boundary conditions.

An essential control: the saved initializer originally computed its Airy
shear augmentation from the dt0-dependent evaluated cohesive response.
Changing only the input parameter would therefore change the Airy field.
The isolated plugin instead reads the **entire saved 1 s target curve** from
`frozen_airy.csv`, with the original radial integration/evaluation unchanged.
The CSV is byte-identical to the baseline target (SHA256
`4e65940fc002615b351f15cc0e6029117ad85f68d049b8f954ad42971474fcac`).
It is not retuned to the 4e6 s cohesive response.

Resolved `parameters.prm` files differ only in Initial time step, library and
output paths (and printing whitespace). Physical parameters, pressure
normalization=no, initial phase, mesh, support, fault grid, solver tolerances,
iteration budgets and real timestep controls are unchanged. Both have 36,106
cells, 1,282,726 total DoFs, 324,954 particles and 1,155 fault elements.
All saved bulk old-stress/phase/initial-state arrays and surface geometry,
Theta, stored C and Ih arrays compare exactly at their exported precision.
The current binary and plugin hashes are recorded in the resource JSON;
the baseline predates the dormant optional offset API and is not represented
as the identical executable.

The realized side velocities are left
`(+2.5e-10,+4.330127018922193e-10)` m/s and right the opposite. Both exported
solutions satisfy them within 7.13e-18 m/s (Float32 output error). Define
u_rigid as these piecewise rigid side translations, with speed 5e-10 m/s.

## Bulk scaling

G=3.203812032e10 Pa, eta=1e26 Pa s. Beta=exp(-dt0 G/eta) and
kappa=-eta expm1(-dt0 G/eta), with no cutoff active here.

| Quantity | dt0=1 s | dt0=4e6 s |
|---|---:|---:|
| beta | 0.9999999999999997 | 0.9999999987184752 |
| kappa [Pa s] | 3.203812032e10 | 1.281524812e17 |
| max / RMS speed [m/s] | 0.0568484 / 0.0154413 | 1.37722e-8 / 4.00261e-9 |
| max / RMS speed divided by Vp/2 | 1.13697e8 / 3.08827e7 | 27.5444 / 8.00522 |
| max / RMS correction speed [m/s] | 0.0568484 / 0.0154413 | 1.42082e-8 / 3.86248e-9 |
| max / RMS correction divided by Vp/2 | 1.13697e8 / 3.08827e7 | 28.4164 / 7.72497 |
| max dt0 abs(u-u_rigid) [m] | 0.05684842 | 0.05683272 |
| RMS dt0 abs(u-u_rigid) [m] | 0.01544133 | 0.01544994 |

The expected inverse velocity ratio is 3,999,999.997; measured max/RMS
correction ratios are 4,001,105 and 3,997,772. The two displacement-correction
fields have correlation 0.9999623 and relative RMS difference 0.8704%
(absolute RMS difference 1.3440e-4 m). Their amplitudes and spatial pattern
therefore survive almost unchanged. These are artificial-interval corrections,
not actual elapsed physical displacement or accepted real-step motion.

RMS is physical-volume weighted from the saved Q2 polynomial on each original
cell. A manufactured polynomial integral test passes. 3x3 versus 8x8 Gauss
integration agrees to 7.5e-9 relative or better for these correction RMS values.
Maxima are sampled at support points and 8x8 Gauss points, not certified
continuous extrema. Bulk output is Float32; this accuracy is sufficient to
distinguish the observed scaling and persistent correction.

## Fault equilibrium and physical pressure

| Quantity | dt0=1 s | dt0=4e6 s |
|---|---:|---:|
| free V/Vinit range | 0.97418--22.55105 | 0.96766--12.66497 |
| free nodal RMS of V/Vinit-1 | 1.30050 | 0.76390 |
| actual sigma_n range [MPa] | 48.17535--52.62727 | 45.63407--52.60817 |
| weak weighted mean sigma_n [MPa] | 50.010035 | 50.010535 |
| free q-target range [MPa] | -0.015879--1.123178 | -0.014562--1.026059 |
| test-weighted nodal RMS q-target [MPa] | 0.141494 | 0.130701 |
| top q-target [MPa] | 0.901376 | 0.183181 |
| bulk physical p range [MPa] | 26.835674--35.175468 | 26.835670--35.175128 |

Deep prescribed V and retained Theta0 match exactly in both cases. The raw
sigma_n range is from actual particle/domain constitutive evaluation, not a
normal-column bulk average. Shear q is the consistent Q1 representation of
the production weak shear load; its target is the same frozen Airy Q1 curve.
The nodal error RMS uses positive test-function integral weights, not a claim
of an exact continuous error norm. The new plugin exports the unchanged mass
matrix. Its row sums match both runs' normal diagnostic weights; the same
matrix recovers consistent p/sigma profiles for both runs. These projected
values should not be confused with the earlier report's endpoint test-weighted
means. Full profiles are in each run's `dt0_surface_comparison.csv`.

At fixed stored C0 and Ih the **evaluated** cohesive response still depends on
dt0 through kappa and beta (at initialization, C_eval=(kappa V+beta Ih C0)/Ih).
Thus fault mechanics need not be an exact inverse-rate rescaling, even with
the identical Airy target. Its partial improvement does not restore official
Vinit equilibrium: the peak remains 12.7x, the maximum shear error only falls
about 8.6%, and the lower normal-stress extreme becomes worse.

## Genuine convergence and cost

| Final diagnostic | dt0=1 s | dt0=4e6 s |
|---|---:|---:|
| logged final Newton iteration | 5 | 4 |
| bulk absolute assembled norm | 8.06832e-4 | 9.50503e-4 |
| fixed bulk scale | 7.54715e11 | 7.54719e11 |
| bulk relative residual | 1.06906e-15 | 1.25941e-15 |
| velocity / scaled-continuity norm | 8.06825e-4 / 3.27773e-6 | 9.48927e-4 / 5.47157e-5 |
| surface RMS residual [Pa] | 3.00569e-5 | 0.136432 |
| fixed surface scale [Pa] | 1.70867e7 | 1.69954e7 |
| surface relative residual | 1.75908e-12 | 8.02756e-9 |
| last linear estimated / fresh | 3.0081680e-13 / 3.0081683e-13 | 3.9797784e-8 / 3.9797791e-8 |
| last linear requested target | 8.06832e-13 | 1.26120e-7 |
| elapsed [s] / peak RSS [GiB] | 273.18 / 3.895 | 273.75 / 3.947 |

Both final nonlinear blocks pass the unchanged 1e-8 criteria. Every returned
linear direction passes its fresh residual check (six baseline, five new).
Convergence of the discrete equations is **not** agreement with the intended
BP3 initial state. There was one new initialization attempt, no retry and no
real timestep. No extra logarithmic sweep is needed to establish this result.

## Evidence and reproduction

From the repository root:

```sh
cmake -S benchmarks/reconstructed_fault/bp3/airy_dt0 -B benchmarks/reconstructed_fault/bp3/airy_dt0/build -DAspect_DIR=/home/ein/repository/aspect/build-pf-cpdi
cmake --build benchmarks/reconstructed_fault/bp3/airy_dt0/build --target bp3_airy.release -j4
python3 benchmarks/reconstructed_fault/bp3/airy_dt0/run.py
python3 benchmarks/reconstructed_fault/bp3/airy_dt0/test_analysis.py
python3 benchmarks/reconstructed_fault/bp3/airy_dt0/analyze.py
```

Build PASS; initialization PASS numerical convergence; manufactured integration
test PASS (one test); analysis PASS frozen-input, side-velocity, mesh, mass-row
and nonlinear/fresh-linear checks. The runner refuses to overwrite its saved
attempt. `comparison.json` contains unrounded measurements and every linear
check; `dt4e6.log` and `dt4e6.resources.json` preserve the log, cost and source/
binary hashes. Baseline artifacts remain under `initialization04`.

Only an isolated benchmark plugin/input, frozen target, run/analysis/test
scripts and documentation were added for this experiment. Existing unfinished
production/prestress-offset changes remain preserved, unselected and suspended
for review. Do not resume dynamics or infer that the offset redesign has been
approved by this initialization test.
