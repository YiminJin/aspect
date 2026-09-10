# Bounded K2.1 prescribed-normal-stress pilot

Subsequent explicit review: K2.1 is accepted as a completed feasibility pilot,
not Gate K2. The user revised the original 1e-6 omitted-profile target to a
provisional 1e-4 allowance for this fixed-profile, prescribed-pressure K2.1/
K2.2 family only, retaining the separate 1e-4 actual normalization requirement.
K2.2 is now proceeding under those conditions. The measurements and original
gate failure recorded below remain historical evidence of the approximation;
same-support refinement does not establish equivalence to the full profile.

2026-09-09. The full coupled pilot completes initialization and four real
0.5 s timesteps through 2 s. **This is not Gate K2 or a converged K2 numerical
reference.** K2's original containment target remains unmet; the K1-only
allowance has not been transferred. No production changes were made after
freezing the K1 correctness baseline in `stage_K1_verification.md`.

## Fixture and realized initial state

`benchmarks/reconstructed_fault/uniform_shear/nonuniform/pilot.prm` includes
the validated finest K1 fixture: 64x256 bulk cells, 1/128 m fault spacing,
147456 particles, ell=.15625 m, prescribed normal pressure 1000 Pa, identical
loading and physical coefficients. The same original H initializer is solved
to convergence; the benchmark's existing constraints then freeze only the
Q1 phase field. H is deliberately frozen as in K1. Temperature/composition,
initial stress, artificial initialization interval, full I_h and support
remain unchanged. Boundary tangential velocities are the original ramp;
normal velocity is zero and x is periodic.

The sole physical change is the compact C3 initial-state bump

    Theta0(x)=200 [1 + .05 ((1+cos(pi*(x-.125)/.0625))/2)^2] s

inside |x-.125|<.0625 m, and 200 s elsewhere. It is constant near the
periodic boundaries. Full width .125 m spans 32 bulk cells and 16 fault
intervals. At fixed V the analytic peak friction increment is about .6343
Pa. The expected diagnostic was an order 1e-7--1e-6 m/s slip response and a
measurable response outside the perturbed interval, not an imposed sign or
a per-vertex K1 root. ell/.0625=2.5: this is a finite-width implementation
pilot, not a thin-fault approximation.

Measured consistent Q1 projection:

- Theta0 minimum 199.999860872 s, maximum 210.132028816 s. Its 0.132029 s
  overshoot relative to the analytic maximum is a projection effect,
  not a changed input or physical evolution at timestep zero.
- Independent volume-weighted projection using actual particle associations
  agrees with production to 1.42e-13 s. Initial Theta is retained, not aged
  through the artificial 2 s Maxwell interval.
- Endpoint Theta0 is 200.000000033 s; at |x-.125|>=.09375 m the largest
  projection tail is 2.88e-6 s. Thus the projected field is not perfectly
  compact even though the physical initial function is.
- Initial H, retained particle stress=1500 Pa and retained C0 are identical
  to the homogeneous K1 baseline. The entire initialized Q1 phi and geometry
  exports are bit-for-bit identical. Initial V is solved and committed;
  its evaluated mechanical stress is distinct from retained particle stress0.

## Coupled response

Comparisons below subtract the **fully coupled homogeneous ASPECT run** at
the same mesh/time/loading, not a scalar reference applied independently to
fault vertices. They measure the realized perturbation, not K2 discretization
error. The remote window is |x-.125|>=.09375 m.

| t (s) | minimum delta V (m/s) | maximum delta V (m/s) | remote max abs(delta V) (m/s) | max abs(delta ux) (m/s) | max raw abs(delta tau_xy) (Pa) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | -2.41162e-6 | 5.85371e-7 | 3.64326e-7 | 3.79960e-8 | 11.8867 |
| .5 | -3.89734e-6 | 7.25573e-7 | 5.82126e-7 | 9.88598e-8 | 4.89769 |
| 1 | -2.52901e-6 | 6.08208e-7 | 4.45333e-7 | 3.47581e-8 | 8.06304 |
| 1.5 | -1.80995e-6 | 4.72926e-7 | 3.20056e-7 | 2.74656e-8 | 10.2968 |
| 2 | -1.52642e-6 | 4.16790e-7 | 2.61958e-7 | 2.45170e-8 | 12.1515 |

The central slip slows while surrounding slip increases. At t=2 the
arclength-weighted accumulated-slip difference is 1.82860e-6 m RMS; maximum
bulk pressure difference is 8.65089 Pa. That pressure does **not** enter
friction in this prescribed-pressure branch. The raw local stress signal is
larger than the initial .6343 Pa friction change; no equality between a bulk
QP stress and an averaged surface traction is assumed. These raw oscillations
and signals require K2.2 convergence, not smoothing or a production correction
in this pilot.

The remote initial projection tail would change friction by only about
1000*.013*(2.88e-6/200)=1.87e-7 Pa at fixed V, far smaller than the imposed
central signal. The sizeable remote delta V is evidence of a coupled response,
but not by itself proof that this response is spatially converged.

## Independent assumption and lifecycle measurements

Measured on this pilot, not presumed from K1:

- Half-width .3088215939070757 m; independent complete Q1 I_h is
  108.144833843526 m, relative discrepancy from production <=3.46e-12.
  Omitted fraction 5.913959442e-5 **fails K2's original 1e-6 requirement**.
- Actual integrated slip/accepted V ratio is .999939935728. Every measured
  normal column and every accepted step has normalization error <=6.00643e-5,
  passing the unchanged 1e-4 requirement. Full I_h is retained in mechanics;
  the omitted tail is not absorbed by renormalization.
- Both endpoint projection masses are .00159610 m2 and all particle-domain
  volumes remain positive. Domain-volume sum lies between .249999996346
  and .250000000000392 m2 (maximum relative error 1.47e-8). This is measured
  after advection; periodic particle-domain behavior is not inferred from
  absence of particle wrapping.
- Prescribed boundary-velocity error <=4.07e-20 m/s. Original Q1 phi and
  particle H remain unchanged. Independent vertex aging-state updates agree
  to <=1.42e-13 s, starting once from the independently projected Theta0.
- All 25 returned linear directions pass fresh/compatibility checks. Final
  bulk/surface normalized residuals at t=2 are 1.20448e-13 / 2.87754e-12;
  dimensional bulk residual is 1.70228e-12 and surface RMS 4.79367e-10 Pa.
  Separate surface convergence and original budgets/tolerances are retained.

## Outputs, commands and cost

All paths below are under `benchmarks/reconstructed_fault/uniform_shear/`:

- `nonuniform/output/solution.pvd`: native bulk VTU time series.
- `nonuniform/output/reconstructed_faults.pvd`: native fault VTU time
  series with committed V, state, cohesive history and previous I_h.
- `nonuniform/measurements/surface_*.csv`: common native surface locations,
  V, Theta, C, prescribed sigma, slip and differences from the coupled baseline.
- `nonuniform/measurements/raw_qps_*.csv`: unfiltered complete evaluated
  Maxwell shear stress, velocity and pressure. ParaView can load these with
  Table To Points using x/y. Bulk tau_xy composition instead denotes history.
- `near_fault_qps_*.csv` and `transverse_x*_*.csv`: actual sampled locations,
  not interpolated/smoothed curves. `normal_columns_*.csv` is explicitly a
  full bulk normal average, **not** the production particle-based surface
  traction. A direct surface-traction export remains a K2.2 diagnostic need.
- `coupled_profiles.png`, `report.json`, `nonuniform/residuals.json`,
  `pilot.log`, `pilot.resources.json`: plot, measurements and solver provenance.

Run command is in `nonuniform/README.md`; analysis:

```sh
python3 nonuniform/analyze_pilot.py nonuniform/output \
  residual-floor/convergence/space64_dt05
python3 summarize_residual_floor.py nonuniform/pilot.log --nx 64
```

One-rank Release: **414.14 s**, **933372 KiB** peak RSS (about 912 MiB),
within the predeclared 400--600 s / ~1 GiB estimate and 1200 s runner cap.
Both PVD files contain all five physical output times; all referenced files
exist. No K2 MPI or refinement run has been made.

## Stop point and proposed K2.2

The controlled sequence is in `nonuniform/README.md`: three spatial levels
32x128, 64x256, 128x512 with corresponding fault spacing at fixed ell, then
dt=.5,.25,.125 on the resolved configuration, not a Cartesian product.
Use the finest full coupled nonuniform solution as reference, common physical
sampling and weighted norms; separate initial profile/C0/Theta projection
changes. Add a fault-only refinement only if it identifies a limiting error.

Before that campaign, review the K2 containment budget. This pilot supplies
no authority to relax it or alter support. No K2.2, true-normal-stress branch,
orientation study, K3 or server job was started. No broader production
correction is proposed from a single unrefined nonuniform run.
