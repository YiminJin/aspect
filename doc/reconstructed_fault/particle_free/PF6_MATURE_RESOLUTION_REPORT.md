# Mature mechanical/profile-resolution separation

17 October 2026; branch `pf-rsf-particle-free`, base
`a8707307008401ac9cff702a369e2e2a611fa0f7` plus preserved incoming work.
[Declared formulation](PF6_MATURE_RESOLUTION_FORMULATION.md),
[source audit](SOURCE_AUDIT.md),
[evidence/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6_mature_resolution/README.md).

**The selected six-case comparison is complete. Both mechanical resolution and
phase/localization representation matter; their effects partially cancel.**
Retain unlimited DGQ2. These measurements do not establish continuum accuracy or
favor either history backend as an accuracy reference. No limiter change, phase
step, transport, moving geometry, AMR or full event run occurred.

## Cases, ownership and exact prolongation

Use the retained straight 60-degree mature fault, homogeneous matched coefficients,
physical seed/RSF disturbance/loading, stationary points and 59-node fixed surface.
Coarse/fine meshes have 4136/11528 graded Cartesian cells, finest h=7.8125/3.90625 m;
mechanical DoFs are 39607/108621. All six fresh serial trajectories use Gauss6
bulk mechanics, dt=200000 s, the same 200000 s initial constitutive interval and
three positive steps to T=600000 s. Gauss3 DG source projection is unchanged.
CC means coarse mechanics/coarse phase, FC fine mechanics/coarse phase,
FF fine mechanics/fine phase. Each has a matched native particle control.

CC and FF retain the native physical initialization prescription. FC imports
**only the coarse step-zero physical Q1 phase**, including hanging values:
T and Z still start unloaded; surface state, loading and geometry are unchanged.
Fine cells lie inside coarse cells. Prescribe every fine vertex by the complete
coarse bilinear polynomial and retain native hanging constraints. Preserve the
coarse 7.8125 m exterior Q1 lattice as explicit immutable exterior data. Do not
resample the analytic seed at fine vertices or change the boundary loading.
The native normalizer recomputes I with its existing integrand, projection,
completion and derivatives; **no I values are imported or overridden**.

The fresh serial frozen-mature reference hook verifies all physical constraints,
affine cells, reference-field agreement, aligned boundary subdivision and the
coarse exterior trace. The ordinary automatic-completion admission remains.
Measured independent checks, for both DG and particles:

- Coarse Q1 trace discrepancy, including hanging interfaces: exactly zero.
- All fine physical phase vertices reproduce the reference exactly; common
  nonnodal phase defect <=2.22e-16.
- CC/FC chi Linf difference <=1.11e-16 per m (1.96e-15 relative).
- CC/FC nodal I difference <=9.09e-12 m (7.46e-16 relative); all accepted phase/I
  remain exactly fixed within each trajectory.
- Common-rule phase representation errors against the native physical profile
  are 2.5128% for CC/FC and .64663% for FF. FC changes mechanics, not this error.

The established mature law remains
T_new=beta*T_old+2*kappa*D3(symgrad u),
Z_new=beta*Z_old+kappa*V, tau=T-2*chi*Z*S.
Z is auxiliary slip memory; mature cohesive resistance/storage are zero.
Particles retain their actual stress history/LLS transfer with the existing
optional D3 convention. Particle T/Z are absent, not inferred from zero cohesion.
DG exports read accepted reconstructed tau; particle exports evaluate the actual
accepted mechanical operator from immutable incoming transferred stress plus
current native FE forcing. They do not apply another update or substitute a
postpublication particle remap for the stress used in the accepted solve.

## Common stress measurements

Use 103752 common physical Gauss3 points on the fine overlay, explicit matching
cell traces and identical positive diagnostic weights. This diagnostic rule is
separate from Gauss6 mechanics. The full 3D tensor metric counts zz=-(xx+yy) and
both xy contributions. Local weighting uses the same tabulated physical-profile
h on every case, independently of represented chi. Cauchy differences also
include pressure: |delta(tau-p Id)|²=|delta tau|²+3(delta p)².
These sampled norms are not independently converged continuum error integrals.

| DG effect at T=600000 s | Whole tau RMS difference, kPa | Local tau RMS difference, kPa | Local Cauchy RMS difference, kPa |
|---|---:|---:|---:|
| CC→FC: mechanics at fixed phase/chi | 3.125 | 24.417 | 49.188 |
| FC→FF: profile/chi at fixed fine mechanics | 3.944 | 35.536 | 56.428 |
| CC→FF: combined | 3.540 | 29.163 | 30.166 |

Local mechanical/profile/combined differences grow through the three positive
times as 8.140/11.848/9.724, 16.279/23.694/19.445 and
24.417/35.536/29.163 kPa. The final effect-vector cosine is **-.5811**; the
combined change is not the sum of their magnitudes. Refining mechanics at fixed
coarse phase increases the local physical tau signal from 30.205 to 36.375 kPa;
FF has 8.174 kPa. This is not a monotonic stress-error estimate. Local T/physical-tau
norm ratios are 33.42/27.76/127.33 for CC/FC/FF, confirming strong cancellation.

The retained interior normal cuts preserve both face traces. Maximum DG tensor
jumps are 91.395/33.013/14.822 kPa for CC/FC/FF; raw-normal jumps are
45.738/15.752/7.754 kPa. Jump reduction alone does not establish accuracy.
Whole/local tau norms, maxima, pressure, unfiltered point tractions and symmetric
relative differences are retained at all matching times in the measured JSON.

## Weak loads, state and endpoints

Declare endpoint windows s<125 m or s>L-125 m, containing the native 121.407 m
completion influence. They contain 14 surface nodes and 28430 bulk points;
interior contains 45 nodes and 75322 points. No face traces are averaged.
Raw nodal tractions below are consistent weak-load projections, not point extrema.

The native **bulk-work mass depends on chi and bulk quadrature partition**, even
on the fixed surface. Export actual M and b. For each case separately form
b_pert=b-M*1*t_background, then normalize pair differences by the larger
background-subtracted perturbation amplitude of the two cases in that region.
Also evaluate a common functional
b_common=M_geometric*(M_native^-1*b-1*t_background),
using the same geometric surface Q1 mass. This diagnostic does not replace the
native residual, filter or work measure.

| DG comparison, final | max dV/Vp, % | max dTheta, s | max dslip, micrometers | shear/raw/filter projection difference, Pa |
|---|---:|---:|---:|---:|
| CC→FC | .02905 | 113.32 | .1163 | 175.89 / 425.89 / 344.40 |
| FC→FF | .32923 | 1317.23 | 1.3569 | 4252.20 / 6052.31 / 993.68 |
| CC→FF | .33216 | 1327.90 | 1.3679 | 4351.41 / 6196.95 / 864.57 |

| DG comparison | Native perturbation weak-load change, shear/raw/filter %: interior | Endpoints |
|---|---:|---:|
| CC→FC | 15.97 / 54.78 / 53.70 | 3.33 / 15.44 / 55.33 |
| FC→FF | 70.82 / 70.18 / 67.62 | 72.73 / 72.06 / 70.24 |
| CC→FF | 66.55 / 37.48 / 29.59 | 72.60 / 81.88 / 60.23 |

For CC/FC, the maximum mass-one relative difference is 6.19e-5. The raw-normal
background-measure difference reaches 61636.6 Pa m, versus **7725.94 Pa m** for
the perturbation weak-load difference. For FC/FF these are 1.269e-3,
1263002 Pa m and **34344.38 Pa m**. Comparing total weak loads against a
perturbation scale would wrongly attribute the background-measure contribution
to changing physical traction. Common geometric raw-load changes in interior/
endpoints are 54.78/16.47% (CC/FC), 70.18/81.69% (FC/FF), and
37.48/85.06% (CC/FF). All absolute native and common-functional loads are retained.

Mechanics-only raw-normal projection changes are 425.89 Pa in the interior and
401.33 Pa at endpoints. Profile-only changes are **554.00 versus 6052.31 Pa**;
dV/Vp is .07655% versus .32923%. The combined interior raw-normal difference
shrinks to 144.91 Pa despite substantial separate effects. Endpoint profile
sensitivity and cancellation explain why a combined coarse/fine comparison or
small V difference can hide important resolution dependence.

## Matched particle controls: differences, not accuracy errors

Mesh, initial physical phase, actual mechanical quadrature and common physical
particle identity files match byte for byte within all three DG/particle pairs.
Native I and bulk-work mass also match exactly within each pair. At the first
positive step, local tensor differences are <=1.20e-9 Pa and V differences
<=6.83e-24 m/s. Differences emerge at step two and grow with repeated history
publication/transfer; no particles move and no transport occurs.

| Final DG–particle difference | Whole/local tau RMS, kPa | Local symmetric relative difference, % | max dV/Vp, % | max dTheta, s | raw/filter projection difference, Pa |
|---|---:|---:|---:|---:|---:|
| CC | 2.294 / 18.593 | 80.03 | .05416 | 160.87 | 324.56 / 64.36 |
| FC | .9935 / 8.689 | 26.38 | .02045 | 60.48 | 175.79 / 66.72 |
| FF | .6297 / 5.252 | 84.81 | .009660 | 28.74 | 176.75 / 68.45 |

Local Cauchy differences are 18.622/11.772/5.253 kPa. Final native shear/raw/filter
perturbation-load changes are 12.38/6.49/8.56% (CC), 4.57/3.06/10.49% (FC), and
9.60/6.87/10.54% (FF) over the whole surface. Interior/endpoint raw-normal
projection differences are 25.81/324.56, 76.57/175.79 and 7.16/176.75 Pa.
Particle mechanical/profile local effects are 21.931/28.856 kPa, with cosine
-.9189 and a combined difference of 12.273 kPa. Their smaller combined change
also contains cancellation; it is not evidence of greater accuracy.

## Verification, source changes and cost

- Independent saved-coefficient evaluation retains all nine DGQ2 modes. Maximum
  scaled T defect is 7.14e-15; reconstructed tau component defect <=6.64e-9 Pa.
- All six native fixed phase/I/H/geometry, Maxwell, positive state, work and single
  publication checks pass. Maximum sampled recurrence defect is 3.80e-9 Pa,
  maximum fraction of the retained allowance .01049; Theta-aging defect <=2.23e-16,
  Z recurrence defect <=3.73e-9 Pa m and normalized nonlinear residual <=2.99e-9.
  No rejection/restart qualification is inferred from these successful steps.
- All 66176/184448 IDs and coordinates remain fixed within each coarse/fine
  trajectory; no insertion/deletion. Effective integration is `BP3 stationary
  samples`; `Evolve phase field=false` alone is not a movement freeze.
- Current CC/FF reproduce **32 retained field/profile CSVs byte for byte**.
  A separate default Gauss3 particle control reproduces retained step-0/1 surface
  fields exactly. All 34 material defaults match entry; 55 unit cases and 2625
  assertions pass. Each primary case pins the same core/plugins and 1031 source
  hashes, input, command, output hashes and cost.

Source changes are confined to the immutable benchmark phase-reference hook,
completion qualification/exterior spacing, explicit mature particle quadrature
selection, maintained BP3 Q1 reference reader/prescription/configuration identity,
and diagnostic/driver exports. No native normalization integrand/projection/
derivative, constitutive recurrence, source quadrature, transport, limiter,
checkpoint, transfer, particle-domain or acceptance allowance changes occur.
The existing quadrature parameter now applies an explicit value above three to
mature particle controls; its default three retains the original native rule.
The reference option defaults empty and is unsupported on restart/AMR/MPI.

| Runtime including startup/exports, s | DG | Particle |
|---|---:|---:|
| CC | 57.30 | 48.22 |
| FC | 159.84 | 131.23 |
| FF | 164.44 | 135.50 |

Primary runs total 696.52 s. Eight native attempts, including the original CSV
header failure before setup and the default regression, total **743.68/3600 s**.
All are serial, one thread; peak RSS is about 1071 MiB. The preliminary CRLF input,
failed log and corrected LF inputs are preserved separately. The plugin/core
stack is GCC12.4/deal.II9.6.2. Current core SHA256 is
`eb90af8d0a93242aa423e563681f4cc0cd161aa33e7ef6eca7e33d5d149c4719`;
full plugin identities and rebuild/replay commands are in the evidence README.
Costs include diagnostics and do not establish production performance rankings.

## Limits and next task

The retained coarse Gauss6/8 and dt/dt2 controls remain unchanged. In particular,
the failed 1% temporal perturbation-load screen remains: changes reached 2.48%.
At the final horizon its shear difference is 141.26 Pa, comparable to the new
mechanics-only 175.89 Pa; raw/filter differences are 88.04/33.94 Pa. Thus the
small mechanics-only load corrections are **not temporally qualified**. Large
stress/profile changes identify sensitivity but do not establish spatial order;
new fine/FC timestep uncertainty is unmeasured. No new timestep sweep was made.

Retain `strict_reference_match_passed=false`, every original native-quadrature,
publication and legacy-domain error, and all earlier PF0–PF6/WENO evidence.
DG setup still uses fixed diagnostic/independent-publication particle points,
particle initialization and persistence infrastructure; it reads no particle
stress/material/domain into accepted mechanics. Particle controls intentionally
retain their legacy domain/CPDI/material/history transfer dependencies. Mature
restart/MPI, phase evolution, heterogeneous materials, moving geometry and
propagation remain unqualified; full PF5 completion is not claimed.

**Recommendation:** keep unlimited DGQ2 and both WENO experiments unchanged.
Do not select a backend by reduced jumps or smaller combined differences.
Next bound one finer mechanical control with **the FF physical Q1 phase and
exterior lattice exactly retained**, and a timestep-halving control at that
chosen fixed profile/mesh before ranking perturbation loads. Only then select a
separate profile refinement if its measured contribution still dominates. Keep
native weak/background-measure and common-functional diagnostics together.
No such next task, limiter change or full event is automatically launched.
