# Bounded timestep-zero attribution

**Qualified follow-up:** the selected [filter-derivative correction](../filter_derivative/README.md) resolves all remaining comparisons (927/927 pass). This report retains its earlier evidence.

**Cause: order-dependent Q1 derivative assignment in the normal-stress filter,
not solver stopping error.** The corrected source admissions remain unchanged.
This diagnosis changes no production source, parameters, ownership, pressure
reference, comparator, or valid association; Section 3 remains unstarted.

All signed differences below are **reversed minus forward**, at the same
physical vertices. The largest slip-rate discrepancy is at
`(48611.111111111109, 1388.8888888888889) m`, down-dip distance
`68746.492615358773 m`. Forward/reversed rates there are
`1.0005795427822510e-9` / `1.0005795357117757e-9 m/s`.

| Physical location (x,y), m | ΔV, m/s | Discrete prediction, m/s | Δ incoming Θ, s | Δ projected shear, Pa | Δ friction-input coefficient z, Pa |
|---|---:|---:|---:|---:|---:|
| (48611.111111, 1388.888889), maximum | -7.070475385e-18 | -7.070474857e-18 | 0 | +8.327886e-5 | +0.0167966411 |
| (50000, 0), lower endpoint | +6.628296062e-18 | +6.628281060e-18 | 0 | -7.688627e-5 | -0.0157525539 |
| (25000, 25000), central anchor | +1.649604937e-21 | +1.647419277e-21 | 0 | -5.960464e-8 | -4.060566e-6 |

Incoming nodal Θ is `8000000.000000014 s` throughout, with no timestep-zero
aging. Both cases use the same regularized law: a=0.025, b=0.015, mu0=0.6,
V0=1e-6 m/s, Dc=0.008 m, damping=4624440 Pa s/m. The two material parameter
sets coincide for these quantities; sampled mixtures are recorded. Implemented
common-rate mu values differ by at most 3.34e-16, their V derivatives by
7.46e-9 (against 2.5e7), solely interpolation/roundoff.

The actual friction input is `N_q z`, where `(M+20² K)z=b_sigma`.
There is no normal-stress clamp in this path, and all inputs remain compressive;
V is safely above its 1e-20 m/s lower bound. The standard projected shear and
velocity are **not pointwise constitutive partners**. The table's z values are
the actual Q1 input coefficients; full samples record the consumed `N_q z`.

## Discrete sensitivity and upstream attribution

The offline calculation assembles the implemented-law derivative
`D_ij = sum_q w_q N_i N_j (sigma_filtered,q * mu_V,q + damping)`.
Using the measured weak shear and normal-input changes, it predicts
`D ΔV = ΔQ - M_mu Δz - Δ_geometry - ΔR`.
`Δ_geometry` evaluates both quadratures at common forward nodal V,z, including
weights/basis changes; `ΔR` retains the freshly assembled discrete residual
rather than assuming exact zeros. At the maximum, prediction error is
5.29e-25 m/s. Omitting the residual correction predicts -7.070451754e-18 m/s.
The direct friction+damping reassembly closes within 5.47e-5 Pa m in loads of
order 1e11 Pa m. This is a conditional constitutive sensitivity prediction,
not a separate solution of the entire coupled perturbation system.

At a **common initial bulk iterate and V=1e-9 m/s**:

| Compared object | Matched result |
|---|---|
| Physical bulk DoF map, homogeneous Newton constraints, lifted initial iterate, incoming V/Θ | byte-identical / exact |
| All four A blocks (574260 nonzero entries total) | exact |
| Source mass M / friction mass M_mu | max relative differences 4.55e-15 / 4.76e-15 |
| Assembled bulk RHS | relative difference 6.55e-16 |
| Complete B / K_V | relative differences 7.90e-15 / 3.74e-15 |
| Filter stiffness K | **5.87325e-5 absolute, 5.32519% of maximum entry** |
| G on nine matched physical velocity/pressure probes | up to 2.80e-6 of the combined maximum action |

The first non-roundoff operator difference is K. In
`source/reconstructed_fault/surface_system_bulk_work.cc`, strict
`along<0` / `along>length` tests independently decide whether a continued
endpoint has zero derivative. Seven admitted endpoint-plane samples receive
zero derivative forward and nonzero derivative reversed. Seventeen additional
samples on the central-vertex normal plane select opposite adjacent segments:
the Q1 values coincide, but the one-sided Q1 derivatives do not. Independent
sample reassembly reproduces each K within 4.34e-19. Endpoint and interior-tie
contributions account for the K difference; remaining geometric roundoff is
below 4.67e-18. No source association was changed to obtain this evidence.

At the maximum-V-discrepancy location, the filtered-normal difference splits
into **+0.0168150444 Pa from K**, -1.239175e-5 Pa from pressure, and
-5.913111e-6 Pa from deviatoric stress. Background/mass contributions are below
3e-8 Pa each; decomposition closure is below 9.67e-8 Pa over all nodes.
The K contribution here is from endpoint classification; the central tie's
contribution at this location is only 2.70e-15 Pa. Pressure and deviatoric
components were propagated through the same filter separately. Both runs use
true-normal friction, the same 50 MPa background and `Pressure normalization=no`:
**no independent mean subtraction or gauge shift** was made. The small pressure
and deviatoric changes are downstream responses, not the dominant forcing.

## Solver-accuracy control and limits

The separately labelled tight pair changes only diagnostic stopping tolerances:
linear 1e-9 → 1e-12, nonlinear 1e-8 → 1e-11. Both retain two Newton updates;
Krylov totals increase from 46 to 59. Fresh first-direction residuals decrease
from about 1.912 to 0.001516, and second-direction residuals from 5.67e-9 to
2.24e-12. Accepted normalized nonlinear residuals are 1.13–1.56e-14 and surface
RMS residuals 1.44–1.98e-8 Pa. All fresh checks pass. The maximum rate discrepancy
is **7.070498753e-18 m/s**, essentially unchanged: solver stopping is not its
explanation. The original qualification threshold still fails and is untouched.

The production-tolerance observer runs reproduce both prior serial profile CSVs
**byte for byte**. A/B/K_V were compared completely; G was tested on nine
constraint-compatible physical probes, not exhaustively materialized. These
are short serial timestep-zero runs only, as requested. Diagnostic setup/build
mistakes (include/constness, support-point-map API, an unregistered intermediate
include, and a trimmed rather than full B vector) are retained in run metadata;
only the four final `endpoint-t0-*` runs support this attribution.

Recommended next bounded task: agree and implement a consistent filter-derivative
convention at endpoint planes and shared vertices, preserving valid source
associations and native input order, then rerun the unchanged qualification.
That correction is **not implemented or implicitly approved here**.

## Reproduction

`plugin/timestep_zero.cc` uses existing read-only observers and the existing
material test-access seam. Build target `timestep_zero` in `build/observer`.
Run `run_cases.py --batch endpoint-t0-forward:1 endpoint-t0-reverse:1
endpoint-t0-tight:1 endpoint-t0-reverse-tight:1`, then `diagnosis/analyze.py`.
Use fresh paths to repeat; the runner preserves logs. `results/analysis.json`,
`matched_locations.csv`, compact initial/final node and matrix snapshots,
run metadata and preservation hashes retain the attribution. Ignored raw output
contains the complete bulk matrices, samples, physical DoF maps and constraints.
