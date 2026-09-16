# K5: fully prescribed mature sliding and bottom traction

## Decision

The early bottom stress feature does **not require RSF-controlled slip rates**.
With every fault vertex prescribed to `Vp=1e-9 m/s`, zero initial perturbation
Maxwell history, and the mature law, the feature remains almost identical to
the saved mature RSF run. One fresh four-rank Release run completed initialization
and two real steps in **114.27 s**. No physical equation, support, normalization,
boundary condition, timestep tolerance, or solver tolerance was changed.

The bottom localization is not compatible with the infinite straight-fault
uniform-sliding reference: the boundary-truncated, projected `I_h` decreases
strongly toward the tip, increasing `chi`. The measured strain mismatch is much
larger there than in the interior. This identifies an RSF-independent
endpoint/localization effect, not a demonstrated production-code defect or a
quantitative attribution of all later bottom stresses to a single cause.

## Experiment and verification

Reused `mature-fault-50-local4/run.prm`, its fixed background coefficients,
42,880-cell saved mesh, and 1,236 fault vertices. The nominal 50-m case has local
refinement around the RSF junction; the bottom region retains approximately
97.7-m bulk cells and 100-m fault elements. It is not uniformly a 50-m grid.

The only mechanical experiment switch is
`ASPECT_BP3_UNIFORM_SLIDING=1`: prescribe **all** manager-owned V DoFs to Vp.
The default BP3 mask remains unchanged when this switch is absent. Benchmark
output now obtains prescribed/free classification from the actual manager mask.
No free RSF rows remain; background, friction, Theta and damping contribute to
the reported prescribed-rate reaction, not to a slip-rate equation determining V.
Thus zero constrained surface residual is **not** a claim that the unreplaced
frictional force balance vanishes.

| Accepted step | Physical time (s) | dt (s) | Final relative bulk residual | Constrained surface residual |
|---|---:|---:|---:|---:|
| 0 | 0 | artificial Maxwell interval 4,000,000 | 5.36413e-10 | 0 |
| 1 | 2,487,214.2056652424 | 2,487,214.2056652424 | 7.24815e-10 | 0 |
| 2 | 4,970,143.2669182401 | 2,482,929.0612529977 | 9.10376e-10 | 0 |

All six returned-direction fresh linear checks passed; worst fresh/target ratio
was 0.910903. All 1,236 V values equal Vp exactly, with zero free/lower-active
nodes. The original convergence targets are unchanged. Recorded physical times
agree with the saved prefix; the existing CFL/fault controllers remained active.

The retained initial particle stress and initial working FE history are zero.
All 385,920 stable particle IDs retain initialized H exactly through both steps;
C remains zero. Nodal Theta passes its independent split aging check at 1e-12
relative tolerance, and accumulated slip matches the accepted time integration.
Fault coordinates and I_h are identical at all three accepted states. Timestep
zero is not counted as elapsed slip/state evolution. Ordinary ASPECT output is
unchanged, and no further timestep was run.

## Compatible reference and boundary data

In ASPECT coordinates take

\[
 s=(1/2,\sqrt3/2),\quad n=(-\sqrt3/2,1/2),\quad
 r=(x_{\rm trace}-x)\sqrt3/2-(H-y)/2,\quad
 S=\operatorname{sym}(s\otimes n).
\]

For the prescribed stationary distance profile, define

\[
 h_\infty(r)=1/g(\phi_\infty(|r|))-1,\quad
 I_\infty=\int_{-\infty}^{\infty}h_\infty(r)\,dr,
\]
\[
 u_{\rm plate}(r)=V_p s\left[
 \int_{-\infty}^r\frac{h_\infty(\rho)}{I_\infty}\,d\rho-\frac12\right].
\]

Then div u=0 and `sym grad u = chi_infinity Vp S` exactly. With zero retained
stress this reference has zero perturbation stress and pressure, and hence zero
perturbation traction on both top and bottom. The compatible diffuse velocity
is smooth through the fault, not a discontinuous piecewise-rigid field there.

The exported prescribed profile has radius 790.58328 m and independently
integrated I_infinity=13,053.8401613 m. A piecewise-linear h and its exactly
integrated piecewise-quadratic primitive define the diagnostic reference.
Simpson versus trapezoidal integration differs by 0.000277816 m (2.13e-8
relative). A Cartesian finite-difference check of the reference symmetric
gradient agrees to 6.35e-9 relative at five transverse locations. This reference
does not replace production FE phase, I_h, support or quadrature.

The nearest lateral boundary point is 18,301.27 m from the fault, well outside
that profile. Thus the reference gives the existing exact translations:

- left: `(2.5e-10, 4.330127018922193e-10)` m/s;
- right: their negatives;
- top and bottom: zero perturbation traction, not prescribed velocity.

The initialized physical constraint lift was sampled at 96 entries per side;
maximum discrepancy was exactly zero. The test retains these conditions.

### Why the production tip cannot satisfy that identity everywhere

At the bottom fault vertex production I_h=6,647.462047 m; it rises to
11,111.332785 m at y=86.568 m and 12,348.330749 m at y=173.136 m, approaching
approximately 12,692 m away from the tip. At the separately more refined 40-km
junction it is approximately 12,960–12,965 m. The latter spatial differences
are ordinary FE/profile representation effects, kept distinct from tip truncation.

Even with an exactly transverse phase profile, `chi(s,r)=h(r)/I_h(s)` is not
independent of s near an open boundary. A target strain with only
`epsilon_sn=Vp*chi/2` must satisfy Saint-Venant compatibility. Its compatibility
defect is `-Vp*partial_s partial_r chi`, generally nonzero when I_h varies in s.
Consequently there need not be *any* velocity field satisfying the proposed
zero-elastic-strain identity for this actual endpoint localization. Compatible
outer boundary data do not remove that internal incompatibility.

## Bottom measurements

Actual constitutive particle/domain samples, using the retained history and
the accepted mechanics, give the following total normal traction. These are
not bulk-column pressure averages.

| Step | Uniform Vp raw bottom sigma_n range (MPa) | Saved mature RSF range (MPa) | Uniform row-weighted mean range (MPa) |
|---|---:|---:|---:|
| 0 | 49.8532583–50.0754047 | 49.8532583–50.0754048 | 49.9634341–50.0014363 |
| 1 | 49.9087549–50.0468865 | 49.9087550–50.0468865 | 49.9772632–50.0008931 |
| 2 | 49.8318547–50.0987614 | 49.8317484–50.0986546 | 49.9562245–50.0020174 |

At step 2 the minimum occurs at a domain quadrature point mapped to the bottom
tip `(21132.4865405,0)` m, xd=115470.053838 m. Its parent is at y=81.381472 m:

\[
 \sigma_n=50\,\mathrm{MPa}
 -57{,}833.3080\,\mathrm{Pa}
 -110{,}312.0005\,\mathrm{Pa}
 =49.8318546915\,\mathrm{MPa}.
\]

Here phi_parent=0.504034324 and chi=0.00278501187 1/m. The positive extreme
maps to y=7.641236 m, with delta p=6067.03753 Pa,
`-delta tau:N=92694.32733 Pa`, and chi=0.00450915233 1/m. These point values
use parent-P0 bulk/history and surface-coordinate localization: the parent need
not lie at its quadrature point's projected fault position.

The maximum bulk-QP chi in the first 200 m is 0.00457861733 1/m, versus
0.00274231916 1/m at the 59–61-km interior control. Using every associated
production Stokes quadrature point and its JxW:

| Region | Step-0 RMS strain mismatch (1/s) | Step-0 mismatch/crack RMS | Step-2 mismatch/crack RMS |
|---|---:|---:|---:|
| Bottom y<200 m | 1.54732e-13 | 14.5518% | 14.1086% |
| Bottom 200–1000 m | 1.91121e-14 | 2.5508% | 2.8056% |
| Bottom 1000–2000 m | 9.96467e-15 | 1.3544% | 1.3840% |
| Interior xd=59–61 km | 9.87515e-15 | 1.3427% | 1.3667% |
| Refined junction xd=37–43 km | 2.52349e-15 | 0.3401% | 0.3534% |

Mismatch means `|sym grad u - chi Vp S|`, not a stress-component proxy. At step
0 velocity-reference RMS/Vp is 1.6622% in the first 200 m versus 0.25845% at
the interior control. The bottom layer relaxes toward the interior mismatch
level within roughly 1 km. No mesh study or decomposition into separate
truncation, projection and bulk FE error shares was performed.

## Existing 15–18-km and 40-km profiles: no reruns

Extracted saved cohesive states 0,10,12,13 and mature states 0,2,10. CSVs and
figures retain separate:

1. Actual constitutive samples selected by the saved extrema exporter.
2. Positive test-weight averages `sum(w N_i sigma)/sum(w N_i)`.
3. Consistent Q1 coefficients obtained from `M sigma_hat = weak_sigma`.

All weak moments were MPI-summed before averaging or solving. Raw files contain
the retained 20 extrema per rank/support class, **not every constitutive point**;
absence of points in a region is reported as missing raw coverage, never zero
variation. In particular the cohesive step-12 transition retains only one side
of its regional extrema. Weak profiles cover every node. Markers identify
15,18,40 km, with windows 13–20 and 37–43 km.

At the accepted mature step 10 (29.24190894 yr):

| Window | Delta p weak peak-to-peak (MPa) | -Delta tau:N weak peak-to-peak (MPa) | Total sigma_n weak peak-to-peak (MPa) | Retained raw sigma range (MPa) |
|---|---:|---:|---:|---:|
| 13–20 km | 0.114003 | 0.025760 | 0.130883 | 47.820968–52.266418 |
| 37–43 km | 0.009267 | 0.024293 | 0.016476 | 44.623888–55.258675 |

Those weak normal ranges are respectively 50.013446–50.144329 MPa and
49.993075–50.009552 MPa. Raw and weak fields must not be conflated.

At a retained transition minimum, xd=17.852239 km,
`delta p=-4.791240 MPa` and `-delta tau:N=+2.612208 MPa`: cancellation leaves
`sigma_n=47.820968 MPa`, not `50+delta p=45.208760 MPa`.
At a retained junction minimum, xd=39.975019 km,
`delta p=-4.526710 MPa` and `-delta tau:N=-0.849402 MPa` reinforce, giving
`sigma_n=44.623888 MPa`. Multiple quadrature coordinates can share a parent-P0
normal traction; the quoted coordinate identifies a retained sample, not a
unique spatial singularity.

In the older cohesive step 12 the retained junction minimum is genuinely
tensile: `50-51.272650-7.534578=-8.807228 MPa`. Its region's weak normal
averages remain 50.002195–50.181853 MPa. The tensile test-weight fraction in
37–43 km is 0.000614408. A positive weak average does not erase local tension;
neither a point extreme nor a small weak resultant alone diagnoses the error.

For this straight geometry the relevant full contraction is

\[
 \Delta\tau:N=\tfrac34\Delta\tau_{xx}
 +\tfrac14\Delta\tau_{yy}-\tfrac{\sqrt3}{2}\Delta\tau_{xy}.
\]

Pressure or individual Cartesian deviatoric components cannot establish a
fault-normal traction error without that signed combination, the physical
background and the correct history timing. Published FE stress-history arrays
are not silently interpreted as accepted current stresses here.

## Diagnostic correction, files and reproducibility

Only benchmark code changed: `bp3.cc`, new `uniform_sliding.h`,
`run_uniform_sliding.py`, `analyze_uniform_sliding.py`, and this report.
The new observer initially used `deviator(strain)` in its reconstructed *bulk
diagnostic stress*, whereas production uses the raw symmetric strain; FE
incompressibility is not pointwise. It did not feed mechanics. Original files
and their source hash remain preserved. The offline analysis corrects the
three normal/diagonal values by `kappa*trace(strain)*Identity`, guarded by that
specific original observer hash, and exports original and corrected quantities.
The C++ observer is fixed and rebuilds successfully. No ASPECT rerun was needed:
strain, kappa, pressure, and old FE history were already saved. Actual production
surface stress samples, all strain-mismatch numbers and solver results were
unaffected. The corrected observer binary was not used for an additional run.

Commands:

```sh
cmake --build benchmarks/reconstructed_fault/bp3/build --target bp3.release -j4
python3 benchmarks/reconstructed_fault/bp3/run_uniform_sliding.py
MPLCONFIGDIR=/tmp/bp3-uniform-mpl python3 benchmarks/reconstructed_fault/bp3/analyze_uniform_sliding.py
```

The runner refuses an existing output directory: the command records the one
completed experiment, not permission to overwrite or continue it. Analysis is
repeatable and read-only with respect to simulation artifacts. Its assertions,
the Release build, Python compilation and `git diff --check` passed. No restart,
additional rank count, refinement, or later-time experiment was run.

Evidence lives under `benchmarks/reconstructed_fault/bp3/uniform-sliding-50-local4/`:

- `run.log`, `execution.json`, `provenance.json`, resolved parameters, saved clock;
- `velocity_constraints.csv`, `fault_*.csv`, `stress_samples_*`, `stress_weak_moments_*`;
- `uniform_bulk_*_rank*.csv`: original exact-QP observations, including old FE history;
- `analysis/uniform_bottom.png`, `uniform_bulk_summary.csv`, `uniform_bulk_extrema.csv`;
- `analysis/uniform_bottom_bulk_*.csv`: offline production-consistent tensor diagnostics;
- `analysis/compatible_reference_profile.csv`, `summary.json`;
- `analysis/existing_*`: requested old profiles and seven figures, without reruns.

**Recommended next decision:** treat the bottom effect as an RSF-independent
open-tip/localization compatibility question. Review the intended normalization
of a through-going diffuse fault at a physical boundary before proposing a
different policy. This experiment does not authorize changing I_h, support,
endpoint topology, stress transfer, or the BP3 boundary conditions.
