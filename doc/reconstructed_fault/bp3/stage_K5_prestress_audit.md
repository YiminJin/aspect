# BP3 initialized-prestress audit, before mechanics

## Outcome

The initialized particle history is **not missing or corrupted**: it equals
the current analytic Airy deviatoric stress exactly at every sampled particle.
Replaying the normal distance-weighted transfer reproduces the production
published FE history exactly. The large discrepancy appears in the represented
bulk equilibrium of the variable radial prestress, not in an uninitialized
particle array.

The current shear augmentation varies by only 82.4 kPa along the fault, but
its radial derivative enters the Airy tensor as `r*qprime`, reaching about
56 MPa. That structure extends through coarse bulk cells, not just the fine
fault strip. 93.13% of the squared assembled variable-field equilibrium-defect
norm is attributed to 6250 m cells. Direct evaluation without particle transfer
also has a large production-quadrature defect, and its strong quadrature
dependence prevents treating the direct Q3 result as an exact reference.

No nonlinear residual evaluation, Newton direction, linear solve, mechanical
update, or real timestep was run. No production code, equations, support,
history semantics, pressure treatment or solver settings were changed.

## Execution and controls

The audit includes the current isolated `airy_dt0/airy.cc`, with its frozen
radial curve. It starts from `distance_initialization.prm`: dt0=4e6 s,
distance weighted average with linear weights, the same 36,106-cell mesh,
fixed fault/phase geometry and the same boundary conditions. The resolved
parameter differences are only the diagnostic library and output directory.

The existing `set_assemblers` signal runs the audit during coupled setup,
after normal particle-to-FE transfer and before the first nonlinear residual.
After writing and closing the output files, it intentionally throws
`PRESTRESS_AUDIT_STOP_BEFORE_STOKES`. The completed marker, checks and absence
of nonlinear/linear solve messages establish completion, **not exit zero**.
The simulator exits 1 by design, and its existing failure path restores the
coupled lifecycle. No diagnostic field is published into the production
solution or working vector. Only the three particle Maxwell entries are
temporarily replaced for the constant control; scoped restoration protects
them on interpolation failure and they are checked against the analytic field
after restoration.

An earlier harness attempt used the no-Stokes scheme and was rejected during
setup because that scheme forbids the retained boundary conditions. Its log,
input and resources remain preserved (`audit.log`, `audit.prm`, `output/`).
It took 1.47 s and ran no mechanics. The corrected attempt is `audit2.prm`,
`audit2.log`, `output2/`, `resources2.json`; it took **198.11 s**, peak RSS
**3,756,192 KiB (3.582 GiB)**, below the 900 s cap.

## What the fields and loads mean

Airy supplies total tensile-positive stress `sigma`. The particle property
stores its 2-D deviatoric part `tau`; the corresponding positive-compression
analytic pressure is `p_A=-trace(sigma)/2`. These are not interchangeable.

The common grid is the 324,954 actual particle locations, with stable IDs and
host cell IDs. `particle_common.csv/.vtp` contains analytic total stress,
analytic deviatoric stress, retained particle stress, published FE history,
constrained working FE history, analytic pressure, and all three pairwise
deviatoric differences. `quadrature.csv` supplies the corresponding analytic
and FE fields on the actual bulk Q3 quadrature, including the actual entry
pressure. It is zero everywhere at this pre-solve entry; it is not silently
replaced by analytic pressure in the live vector.

The independently assembled, noncommitting prestress-only residual functionals
use the production quadrature, velocity shape gradients, homogeneous Newton
test constraints, ADD assembly, and
`PhaseFieldFault::evaluate_frozen_maxwell_stress()`:

```
L_hist(w) = integral beta*tau_old : eps(w)
L_p(w)    = -integral p_A div(w)
L_b(w)    = -integral prescribed_traction . w
E(w)      = L_hist(w) + L_p(w) + L_b(w)
```

The sign is the residual sign; the frozen history RHS in production is its
negative. `L_hist` alone is not expected to vanish. `E` uses the matching
analytic pressure solely to audit total-stress equilibrium; that pressure
need not be representable exactly by the pressure FE space. The actual-entry
load with entry pressure zero is exported separately. Neither quantity is
claimed to be the complete coupled residual: current strain, slip/profile
terms and surface friction are deliberately outside this prestress audit.

Cases:

- **a:** current variable `q=tau0+C0(r)`, retained particle history and the
  normal production transfer, then constraints on a private copy.
- **b:** old-style `q=tau0`, using the same interpolator and incident-cell
  ADD/count transfer. Analytic pressure is that of this constant-shear field.
- **c:** current variable Airy tensor directly at bulk QPs, without transfer.

All cases retain the **same actual variable-field bottom traction**, both
lateral velocities and the free top. No RSF equilibrium is required of b.
Its predictable boundary mismatch is explicitly isolated below.

## Field and radial measurements

| Check | Result |
|---|---:|
| variable transfer replay minus published history, maximum [Pa] | 0 |
| retained particle minus analytic deviatoric stress, maximum [Pa] | 0 |
| constrained FE minus particle, max Frobenius norm [MPa] | 4.24048 |
| same, particle-sample RMS [MPa] | .245334 |
| constrained FE minus analytic, bulk-Q3 volume RMS [MPa] | .967289 |
| same, bulk-Q3 maximum [MPa] | 6.57066 |
| working minus published FE, particle-grid maximum [MPa] | .367737 |
| same, particle-sample RMS [MPa] | .004863 |
| beta everywhere | .9999999987184752 |

The point-sample RMS weights particles equally; the bulk-Q3 RMS uses actual
JxW weights. The latter samples an underresolved analytic field and is not a
converged continuous error norm. Working/published differences are the existing
constraint application, not a new transfer or history update.

| Radial quantity over the fault | Minimum | Maximum |
|---|---:|---:|
| frozen C0=q-tau0 [MPa] | .418899 | .501307 |
| q [MPa] | 26.965022 | 27.047429 |
| qprime [Pa/m] | -485.978 | 486.125 |
| r*qprime [MPa] | -50.6672 | 56.1328 |

`radial.csv` exports each interval's two endpoints and midpoint, preserving
both derivative traces at knots, and separately exports stored cohesive C.
The frozen augmentation and stored C differ by only a few millipascals from
the retained 1 s target construction; they are not recalculated using a new
4e6 s evaluated cohesive response.

The current source linearly extrapolates the final radial interval beyond
the physical fault length 115470.05 m. The box reaches r=127358.10 m; there,
q reaches 32.82650 MPa, C0 reaches 6.28038 MPa and r*qprime reaches
61.91190 MPa. `radial_full_box.csv` records that existing extension. It was
not clipped or changed. Its bottom traction consequence is especially visible
in the constant-field control; the main variable-field defect is not confined
to that extrapolation region.

## Global and per-cell weak loads

Norms below are Euclidean norms of assembled constrained velocity-DoF loads,
in N/m for this 2-D per-unit-thickness problem. No new acceptance threshold is
introduced.

| Case | norm L_hist | norm E | max abs(E_i) |
|---|---:|---:|---:|
| a: transferred variable | 7.52827e11 | **9.29817e10** | 1.39584e10 |
| b: transferred constant, same BC | 7.50253e11 | **3.62873e10** | 2.09378e10 |
| c: direct variable, Q3 | 7.59945e11 | **1.26879e11** | 2.22584e10 |

The pressure-variable and boundary norms are 7.64566e11 and 4.70223e10.
The actual-entry history/zero-pressure/boundary load norm is 7.57183e11.
Large individual terms must therefore not be mistaken for an equilibrium
error before their signs and cancellation are included.

`cells.csv` gives raw local weak-load norms for every cell. For example:

| Case | maximum raw cell E norm | root sum of squared raw cell E norms |
|---|---:|---:|
| a | 4.37920e11 | 6.54820e12 |
| b | 4.43577e11 | 6.52337e12 |
| c | 4.38005e11 | 6.54704e12 |

These raw local terms do **not** include cancellation against neighboring
cells. `cell_correlations.csv/.vtp` also attributes each assembled nodal
squared load equally to incident cells, conserving the complete global
squared norm. This is a spatial diagnostic, not a modified quadrature rule.
For a, 93.13% of this quantity belongs to h=6250 m cells and 5.30% to h=3125 m
cells. Only .200% of the global squared norm is at nodes within 1500 m of the
fault; the top 1 km accounts for .00199%. The largest attributed cell is
centered at (90625,3125) m, with norm share 1.77122e10 N/m and saved
displacement-correction RMS .03865 m.

## Direct quadrature and constant-field boundary controls

| Analytic variable integration | norm E [N/m] |
|---|---:|
| production Q3 | 1.26879e11 |
| diagnostic Q6 | 5.01805e10 |
| diagnostic Q12 | 3.21488e10 |

The Q3-to-Q6 and Q6-to-Q12 **vector changes** are 1.38300e11 and 6.25025e10,
respectively. Q12 is not a converged direct reference. These checks refine
only the volume integration; the actual production boundary quadrature/load
is held fixed. Consequently remaining volume-versus-boundary integration
uncertainty is explicit. Replacing beta by one changes the direct-Q3 load
by only 973.89 N/m, negligible on these scales.

The current analytical formula passes cheap constant-mode recovery, free-top
traction, smooth-variable divergence and two-sided fault-traction checks.
For piecewise linear q, q and its radial integral are continuous; the radial
normal/shear traction is continuous even when qprime jumps. The Airy continuum
equilibrium construction is therefore distinct from its underresolved cell
quadrature/FE representation. These checks do not certify a converged
numerical global weak integral of the current finely varying radial data.

For constant shear evaluated directly with Q12, the same-BC load is
3.57759e10 N/m. The independently assembled mismatch between constant-field
boundary traction and the unchanged prescribed traction is 3.57727e10 N/m.
Subtracting that known mismatch **offline only** leaves 4.80573e8 N/m;
99.9999999994% of its squared norm is in the near-fault strip, consistent with
the remaining unsplit-interface integration issue. No boundary condition was
changed to obtain this diagnostic. The actual transferred b run retains its
3.62873e10 N/m load and makes no claim about RSF equilibrium.

## Correlation with the saved correction, not a new solve

Use the saved distance-weighted initialization's Q2 velocity and
`D=dt0*(u-u_rigid)`. For nodal weak work, u_rigid is represented at the same
Q2 support nodes; constrained rows carry zero residual. For cell RMS plots,
the physical piecewise rigid reference is evaluated at quadrature points as
in the previous dt0 comparison. These different diagnostic representations
are kept explicit.

| Load | work on saved D [J/m] | global load/D cosine | cell norm vs D_RMS correlation |
|---|---:|---:|---:|
| a | -5.01033e9 | -.01263 | .1063 |
| b, same BC | -4.38084e8 | -.00283 | .0214 |
| c, Q3 | -1.33760e9 | -.00247 | .1047 |
| a-c: same-Q3 transfer contribution | -3.67274e9 | -.00936 | .1053 |
| direct variable Q12 | -7.15672e8 | -.00522 | .0995 |

The transfer contribution accounts for **73.3% of a's signed work on D** in
this exact Q3 decomposition. This is accounting on the saved field, not proof
that it explains 73.3% of the velocity: elasticity is nonlocal, pressure is an
unknown, and the scalar correlations are modest. Neither the direct analytic
field at Q3 nor changing the interpolator alone removes the represented
equilibrium defect. The coarse bulk resolution of radially extended qprime
structure is the principal newly identified limitation. An additional
mechanical solve would not separate that from quadrature error.

## Artifacts, commands and review boundary

All new files are benchmark-local under
`benchmarks/reconstructed_fault/bp3/prestress_audit/`; production source was
not edited in this task. Existing unrelated work is preserved.

```sh
cmake -S benchmarks/reconstructed_fault/bp3/prestress_audit -B benchmarks/reconstructed_fault/bp3/prestress_audit/build -DAspect_DIR=/home/ein/repository/aspect/build-pf-cpdi
cmake --build benchmarks/reconstructed_fault/bp3/prestress_audit/build --target bp3_prestress_audit.release -j4
python3 benchmarks/reconstructed_fault/bp3/prestress_audit/run.py audit2
python3 benchmarks/reconstructed_fault/bp3/prestress_audit/analyze.py
c++ -std=c++17 -O2 benchmarks/reconstructed_fault/bp3/prestress_audit/test_airy.cc -o benchmarks/reconstructed_fault/bp3/prestress_audit/build/test_airy
benchmarks/reconstructed_fault/bp3/prestress_audit/build/test_airy
```

Build and audit assertions PASS. Scalar formula checks: constant recovery
7.45e-9 Pa, smooth-variable divergence 1.04e-7 Pa/m, top traction zero,
two-sided offset normal/shear errors .000250/.000403 Pa. Analysis verifies
mesh identity with the saved correction, norm-share conservation and absence
of mechanical iterations. Output is about 433 MiB including CSV and compressed
ParaView point clouds; the runner refuses overwrite. `analysis.json`,
`analysis.log` and `resources2.json` contain full numerical values and hashes.

Review point: no lost prestress history or failure of the normal transfer
workflow is demonstrated. There are substantial approximation errors in the
current radial-variable prestress on the graded bulk mesh, including direct
quadrature without transfer. The exact partition of the remaining direct
load between volume and boundary quadrature is not resolved by Q12. A future
discrete-equilibrium treatment must address that radial structure; this audit
does not authorize smoothing C0, changing the mesh, altering the Airy field,
changing boundary tractions, or proceeding with a prestress-offset redesign.
