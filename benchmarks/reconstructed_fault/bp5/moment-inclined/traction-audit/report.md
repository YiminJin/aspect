# Bounded traction-definition audit: inclined A/B

## Answer

The small transfer-induced pattern survives the **consistent-Q1 projection of
mechanical traction**, but **does not enter frictional normal stress in this
fixture**. Both saved runs explicitly select adiabatic friction pressure,
zero gravity, surface pressure 1000 Pa, and no background-traction property.
Thus the actual friction input is 1000 Pa everywhere in both branches. All V
nodes are prescribed as well. The earlier inclined comparison demonstrates a
raw/diagnostic stress effect, not normal-stress feedback through friction.

This qualification is important: in true-normal-pressure mode, production
friction uses the raw pointwise mechanical normal stress before any projection.
Small projected roughness alone would not establish a small friction effect
when the friction coefficient varies over the integration points.

## Production trace

1. `source/material_model/phase_field_fault.cc:574`,
   `PhaseFieldFault::evaluate_reconstructed_fault_point` evaluates the current
   tensor using the frozen incoming history. At lines 651–702 it forms

       sigma_f = sigma_background +
                 (adiabatic_mode ? p_adiabatic(x) : p(x)-tau(x):N),
       traction_friction = mu(V(x), Theta_in(x), f_Gamma(x)) * sigma_f,
       R_density = shear - cohesion - traction_friction - damping.

   The background is a separate fixed fault property, not a shift hidden in
   the bulk pressure; `reconstructed_fault_background_tractions` (line 749)
   returns zero when no selector is attached, as in these A/B plugins.
   Mechanics interpolates incoming nodal Theta, not newly aged output state.

2. `source/reconstructed_fault/surface_system.cc:781`,
   `assemble_bulk_work_system` uses the actual constrained bulk FE state at
   the native Stokes QPs. At lines 1008–1088:

       w_q = JxW_q * chi_q,
       friction_load_i = sum_q w_q N_i(q) mu_q sigma_f,q,
       M_ij = sum_q w_q N_i(q) N_j(q).

   Locally owned QPs contribute once; fault-sized arrays are MPI-summed.
   There is **no mass inversion or row normalization before friction**.
   The consistent mass enters convergence measures and diagnostic conversion
   to Pa; it does not replace the pointwise constitutive input. Prescribed V
   constraints restrict mechanical directions/free equations, not the
   observational traction mass matrix.

3. `benchmarks/reconstructed_fault/bp3/bp3.cc:570` projects the assembled
   `weak.normal_traction` by `solve_tridiagonal_system(M_diag,M_off,b)`.
   That field represents sigma_f; for this fixture it would be 1000 Pa.
   `bp5/normal_stress_diagnostic.cc:298` uses the same consistent inverse for
   separate p, d and background loads. Its production split-capture path is
   explicitly restricted to **true normal traction**
   (`surface_system.cc:809`), not this adiabatic fixture.

4. The inclined benchmark's `moment_cycle.cc` instead exported raw mechanical
   p, d=-n^T tau n and sigma_mech=p+d. Its Python row-average plot was
   `b_i/(M*1)_i`. Neither that row-average field nor its consistent projection
   is the actual sigma_f for the present parameter selection.

5. `source/adiabatic_conditions/compute_profile.cc:111–135` starts at surface
   pressure 1000 Pa and adds rho*g*dz; g=0, so p_adiabatic=1000 exactly.
   The saved resolved files also disable the surface-condition function.
   `source/simulator/helper_functions.cc:833–836` instead normalizes the
   **mechanical** pressure to volume mean zero. We retain that gauge without
   adding 1000 Pa or a BP5 50-MPa background to the saved mechanical data.

## Operators and data reuse

The four existing raw exports (A/B, steps 1/4) already contain every associated
QP's JxW, chi, segment and xi. They are sufficient to reconstruct the **full**
mass matrix; row sums alone would not be. For this source map,
`manager.cc:1914–1917` defines N=(1-xi,xi). There is no selected endpoint
continuation or independent trace in this fixture.

The audit exports diagonal/off-diagonal entries and all RHS vectors in
`*_assembly.csv`. It assembles the full open-fault 38x38 M before choosing the
same 12 interior nodes |s|<=0.2 m. No artificial interior-window boundary
condition or prescribed-V row replacement is applied to this projection.
All four matrices agree bitwise; condition number is 4.11622.

The new isolated test plugin calls the **actual compiled production**
`ReconstructedFaultUtilities::solve_tridiagonal_system` during ASPECT
`--validate`. This executes only small offline linear solves—no Simulator
trajectory, material preparation, history update or mechanical solve. A dense
independent solve verifies its output. The actual constant friction input is
derived from the implemented branch and the saved fully resolved parameters;
it is not a newly replayed material response.

## Numerical comparison

All values below are Pa. Means and mean-removed RMS use the same native row
masses. Chord RMS is the deviation from the neighboring-node chord, not a
claim that all broad curvature is numerical noise.

At the first real step (0.1 s), A and B coincide exactly for every exported
sample, RHS and projected value:

| Projection | mean p | mean d | mean p+d | p+d chord RMS |
|---|---:|---:|---:|---:|
| Row average | 153.985076 | 16.271693 | 170.256769 | 0.125014 |
| Consistent Q1 | 154.064856 | 16.166753 | 170.231609 | 0.123157 |

At the final step (0.4 s), the total mechanical normal traction is:

| Projection | A mean | B mean | A chord RMS | B chord RMS |
|---|---:|---:|---:|---:|
| Row average | 680.229222 | 681.027075 | 0.498819 | 0.500057 |
| Consistent Q1 | 680.130375 | 680.926434 | 0.490174 | 0.492628 |

Final **A minus B**, separating mean from variation:

| Projection / quantity | Mean difference | Mean-removed RMS | Chord RMS |
|---|---:|---:|---:|
| Row / p | -0.821544 | 0.014534 | 0.001539 |
| Consistent / p | -0.821038 | 0.014464 | 0.001531 |
| Row / d | +0.023690 | 0.029820 | 0.004999 |
| Consistent / d | +0.024979 | 0.029468 | 0.005908 |
| Row / p+d | -0.797854 | 0.044314 | 0.006429 |
| Consistent / p+d | -0.796060 | 0.043858 | 0.007040 |

**Actual friction normal input:** mean 1000 Pa for A and B at both steps;
pointwise and spatial A-B difference zero. Projecting this constant only adds
roundoff. The fully prescribed rates also mean there is no free RSF response
from which to infer feedback, even independently of the pressure selection.
Transfer may still affect shear driving and bulk mechanics; this audit does
not assert that its entire mechanical effect vanishes.

## Checks and artifact provenance

- Constant 1 reproduction: maximum consistent error 4.78e-15.
- Constant 1000-Pa reproduction: maximum error 5.58e-12 Pa.
- p+d=sigma_mech: raw samples exact; row-average closure <=7.28e-12 Pa;
  consistent-Q1 closure <=1.46e-11 Pa over the **whole fault**.
- Maximum ||M q-b||_infinity normalized rowwise by mass: 9.11e-13 Pa.
- Production and dense independent inverse agree within the tested
  2e-14 relative / 2e-11 Pa absolute allowance.
- Four offline matrices/five RHS each passed. No new trajectory or MPI study.

One figure: `traction_definition.png`. Full first/final A/B/A-B means,
mean-removed/chord RMS, constant/closure checks are in `results.json`;
profiles are in `*_comparison.csv`. Input hashes are in `provenance.json`.
`validation.log` records the no-simulation invocation.

ASPECT executable SHA256 remains
`d053c7a596887b6e630574c8db660520957a890d672efa8a83ec22087392006c`.
Offline projection plugin SHA256 is
`6292afe97a73eea074ced0623d58802e2d1261ce11a4acec92b0429fffb8de4e`.
New benchmark files are `audit_inclined_traction.py` and
`test_traction_projection.cc`, with a dedicated CMake target. No production
source, parameter, pressure mode, solver tolerance or existing result changed.

Reproduction from the repository root (use a fresh audit directory for prepare):

```sh
cmake -S benchmarks/reconstructed_fault/bp5 -B benchmarks/reconstructed_fault/bp5/build
cmake --build benchmarks/reconstructed_fault/bp5/build --target test_traction_projection -j4
python3 benchmarks/reconstructed_fault/bp5/audit_inclined_traction.py prepare benchmarks/reconstructed_fault/bp5/moment-inclined /tmp/traction-audit
ASPECT_TRACTION_PROJECTION_AUDIT=/tmp/traction-audit build-pf-cpdi/aspect-release --validate /tmp/traction-audit/validate.prm
python3 benchmarks/reconstructed_fault/bp5/audit_inclined_traction.py analyze benchmarks/reconstructed_fault/bp5/moment-inclined /tmp/traction-audit
```

**Stop point:** no inference about BP5 true-normal feedback is warranted from
this prescribed-normal, prescribed-slip comparison. The projected mechanical
pattern is real but small; the frictional normal input in these runs is constant.
