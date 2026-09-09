# K1: initial-stress audit and first-step line-search diagnosis

Status: diagnosis, not a corrected mechanical trajectory or K1 acceptance.
The production executable and equations, retained initialization, association
width, full `I_h`, physical parameters and solver tolerances are unchanged.
The current dirty production changes from earlier work were preserved.

## Scope and evidence provenance

Before editing the benchmark diagnostics, the saved accepted
`diagnostics/mechanical-pilot/bulk_0.csv` was audited read-only. Its 9,216 rows
were grouped by their nine-point cell quadrature and matched against **every**
native cell's vertex coordinates and tensor-product weights in
`diagnostics/results-corrected/raw/pre_mechanics_cells.csv` and
`pre_mechanics_nodes.csv`. This was not a comparison of global extrema.

The old CSV lacked explicit deal.II CellIds. The only C++ change in this pass
adds a `bulk_cell_ids_N.csv` provenance sidecar to `uniform_shear.cc` containing
row number, actual `cell->id()`, QP number and native vertex indices. The
unchanged pilot was replayed under GDB. Its accepted `bulk_0.csv` is byte-for-byte
identical to the original saved data. The sidecar therefore supplies direct
CellId provenance for the saved audit without changing its stress values.

All paths below beginning `diagnostics/` are relative to
`benchmarks/reconstructed_fault/uniform_shear/`.
The unchanged debug executable SHA256 is
`5022898bddbc9762595547433056e3d09e8fe06bb9bbb812038d80b3ceca33cd`.

## 1. The initial oscillation is a cellwise quadrature-orthogonal mode

At the assembler's actual QGauss(3) points evaluate

\[
q=\tau_{xy}=\kappa(u_{x,y}+u_{y,x}-\chi V-\upsilon^{\rm hist})
  +\beta\tau^{\rm old}_{xy}.
\]

Use normalized cell coordinates `(xi, eta)` in `[-1,1]^2` and divide each
weighted moment by the cell's total weight; the reported moments have units Pa.
Subtracting each cell's own mean makes its constant moment zero by definition,
so that fact alone would not establish equilibrium. Instead, check that all
cell means agree, and compute the constant and linear moments about the single
**global** weighted mean as well.

For 1,024 actual cells / 9,216 QPs:

| Measurement | Value |
|---|---:|
| Global weighted `q` mean | 1040.01656699455 Pa |
| Range of the 1,024 cell means | 4.657e-10 Pa |
| Maximum constant moment about the global mean | 2.442e-10 Pa |
| Maximum absolute linear moment, either coordinate | 3.988e-11 Pa |
| Maximum departure from `[-0.8,1,-0.8]` transverse deviations | 3.345e-11 Pa |
| Raw minimum / maximum | 1032.59010734904 / 1049.29964155140 Pa |
| Within-cell RMS / maximum deviation | 2.79736188063 / 9.28307455680 Pa |

For example, **CellId `2_4:0123`**, native vertices `[983,726,535,204]`,
center `(0.0859375,0.0546875)` m, has transverse Gauss-row deviations
`[-7.426459645445, +9.283074556789, -7.426459645419]` Pa. Their `5:8:5`
weighted constant moment and signed linear moment vanish to roundoff.
The full nine QPs, rather than just these three transverse averages, enter
the moment checks. `diagnostics/stress-coarse/cell_stress_moments.csv`
records every CellId and its measurements.

These **sampled** deviations are proportional to `1-3 eta^2`, orthogonal to
the constant/linear normal derivatives of the Q2 shear test functions under
the actual Gauss rule. This explains weak equilibrium despite pointwise
stress oscillations. It does not assert that the continuous evaluated stress
is a quadratic polynomial: `h(phi_h)` is nonlinear. There is also a separate
nonzero scalar-reference bias; cancelling cell moments does not remove it.

## 2. Diagnostic stress, bulk assembly, and particle traction

Production bulk constitutive viscosity is `kappa`, not `eta`. The ordinary
Newton Stokes residual supplies `2 kappa epsilon_dot`; the reconstructed-fault
assembler adds the RHS counterpart of
`-2 kappa (chi V + upsilon_hist) S`. For this horizontal fault `S_xy=1/2`,
which gives the engineering-shear formula above without an extra factor two.
The benchmark obtains `chi`, `kappa` and `upsilon_hist` from the production
bulk point response on the exact cached Stokes quadrature. At zero, its FE
old shear stress and every retained particle old shear stress are 1500 Pa.
The spatial oscillations are **evaluated responses**, not changes to that
retained initial stress history.

The surface system does not sample the bulk Gauss stress. It reevaluates the
FE velocity gradient and phase field at associated particle positions, reads
the particle's retained stress, calls `evaluate_reconstructed_fault_point`,
then assembles `sum_p volume_p N_i F_p`. The regular initial particles are at
cell fractions `1/6,1/2,5/6`, not Gauss points. The independent audit reconstructs
the Q2 velocity gradient from all nine QPs and reevaluates nonlinear `h(phi)`
at the particles; it does **not** interpolate the Gauss stresses themselves.

The coarse associated-particle weighted traction is **1040.29349115113 Pa**,
0.27692415658 Pa above the bulk QP mean. The independent reconstructed surface
residual has consistent-mass strong RMS **5.34746e-8 Pa**. Its particle-level
tractions and weak-form reconstruction are saved in
`diagnostics/stress-coarse/particle_surface_traction.csv`. Thus the different
quadratures affect the converged traction/slip, while the oscillations remain
compatible with the actual discrete bulk equilibrium.

### Separate, unfixed assembly discrepancy

The constitutive law and surface point response include `beta tau_old`.
However, inspection of the active `NewtonStokesIncompressibleTerms`,
`ReconstructedFaultStokes::execute`, and `PhaseFieldFault::evaluate` finds
**no frozen `beta tau_old` contribution to the bulk weak residual**.
The exported parameters have both `Enable elasticity=false` and
`Enable additional Stokes RHS=false`; no alternate elastic-force path supplies it.
The comment in `PhaseFieldFault::evaluate` claiming its viscosity is not used
by assemblers is also stale: the ordinary Newton assembler uses it.

This conflicts with the full Maxwell constitutive weak form. No fix is made
here. In the present zero and first real step, `beta tau_old` is a spatially
constant tensor. Its contraction with gradients of admissible homogeneous
test functions integrates to zero (periodic sides; prescribed top/bottom
velocity). It therefore cannot explain these initial cell oscillations or
the first-step line-search failure. Nonuniform later histories will require
a separate assembly correction and a manufactured nonuniform-old-stress
regression, including points outside fault support. Do not hide it by changing
the diagnostic stress or smoothing histories.

## 3. First real timestep: every proposed candidate

The replay uses the actual first step `t=2 s`, `dt=2 s`, loading
`U=1.1e-4 m/s`. It has no active vertices, all recovered `delta V` values are
positive, and the fraction-to-boundary limit is 1. No rejection is due to a
bound, singular constitutive evaluation, non-finite result or exception.

At Newton iteration zero:

- `V` is approximately `3.1731792608273e-4 m/s`;
  `delta V` ranges from `9.0987947524e-6` to `9.0988660960e-6 m/s`.
- Bulk norm and scale: `112.18594578208965` in the solver vector convention.
- Surface strong RMS: `2.49613133348732e-6 Pa`;
  fixed surface scale: `1.61613832255358e-5 Pa`.
- Initial relative residuals `(1, 0.154450...)` give merit
  `0.511927455668990`.

| alpha | Bulk norm | Surface RMS (Pa) | Relative surface | Trial merit | Armijo upper bound | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.11633e-7 | 1.008525e-2 | 624.034 | 194709.2064 | 0.511876263 | reject: Armijo |
| 2/3 | 37.3953153 | 4.510442e-3 | 279.088 | 38945.0062 | 0.511893327 | reject: Armijo |
| 4/9 | 62.3255254 | 2.013082e-3 | 124.561 | 7757.9033 | 0.511904703 | reject: Armijo |
| 8/27 | 78.9456656 | 8.972401e-4 | 55.5175 | 1541.3459 | 0.511912287 | reject: Armijo |
| 16/81 | 90.0257590 | 3.995489e-4 | 24.7224 | 305.9216 | 0.511917344 | reject: Armijo |
| 32/243 | 97.4124879 | 1.778341e-4 | 11.0036 | 60.9170 | 0.511920714 | reject: Armijo |

The bound is `(1-1e-4 alpha) Phi_0`. Production correctly reports exhaustion
and does not accept the last candidate. The original unsupplemented trace is
`diagnostics/line-search-trace.log`; the completed signed-vector repeat is
`diagnostics/line-search-signed-complete.log`. GDB only prints existing variables;
it does not assign program state or change the line-search budget.

### Directional Jacobian/Taylor check on the actual production trials

The signed replay exports the base and every trial weak nodal vector, together
with the actual surface mass diagonal/off-diagonal. Their independently
computed strong RMS norms reproduce the reported production norms. The base
**signed mean** is `4.50571534646e-8 Pa`, not its `2.496e-6 Pa` RMS; treating
the latter as a signed constant would give a misleading remainder check.

For this frozen prescribed-pressure problem, the bulk stress and cohesive
response are affine in the direction; only friction is nonlinear. Let `M` be
the captured mass matrix and `r_0` the signed base vector. Compute independently
the nodal friction remainder `e_mu(alpha)` below, then test

\[
 r_\Gamma(\alpha)=(1-\alpha)r_0+M e_\mu(\alpha).
\]

Across all six actual production trials,
`||r_Gamma(alpha)-(1-alpha)r_0-M e_mu(alpha)||_Gamma / alpha`
is **at most 9.744e-13 Pa**. At alpha=1 it is `2.975e-13 Pa`.
This is a directional Taylor verification along the recovered Newton
direction, using the known analytic remainder instead of subtracting nearly
equal values at an arbitrarily tiny FD step. It is not a claim to have checked
every possible Jacobian direction. Using Q1 nodal remainders with the mass
matrix differs from evaluating the remainder at each particle by less than
`2e-13 Pa` here: the adjacent `delta V` variation is below `7.14e-11 m/s` and
the interpolation-error bound is `max|e_mu''|*(delta deltaV)^2/8`.
The asinh-to-log remainder approximation is below `1e-23 Pa` for the actual
regularized-friction arguments, which exceed `1e12`.

The bulk norm is `(1-alpha)` times its initial norm to roundoff for the five
partial steps; the full-step linear residual is `9.951e-10` relative. Thus
the observed trials are consistent with the recovered directional Jacobian,
not a sign/scaling error in `B`, `G`, or `K_V` along this direction. Machine-
readable acceptance and remainder measurements are in
`diagnostics/line-search-analysis/candidates.csv` and `summary.json`, produced
by `analyze_line_search.py` from the signed transcript.

### Cause and smallest proposed correction

The initial surface equation is already almost balanced, while the changed
boundary loading requires a bulk update. The implementation sets
`s_Gamma=max(initial_surface_norm, floor_factor*characteristic_surface_norm)`,
with `floor_factor=max(linear_tolerance,sqrt(machine_epsilon))`. Here the floor
suppresses an approximately 1084.57 Pa characteristic traction scale to
`1.616e-5 Pa`. Ordinary second-order friction curvature is then amplified
into a huge normalized merit. For nearly uniform `V,dV`, that remainder is

\[
  1000 a\,[z-\log(1+z)],\qquad z=\alpha\delta V/V,
  \qquad a=0.025.
\]

It is about 0.0100853 Pa at the full step, matching the observed fault
residual while the bulk full-step residual falls by nine orders of magnitude.
The current convergence rule also demands a final surface RMS below
`1.616e-13 Pa`, despite subtraction of tractions of order 1000 Pa.
Increasing the backtracking budget would not cure that precision/scale
problem. An isolated exactly balanced two-block model needs 11 reductions
before accepting a step near 0.01156, even with its exact Jacobian.

**Proposal for review, not implemented:** use the already computed physical
surface characteristic `||K_V V_char||_Gamma` without the roundoff multiplier
as the fixed surface normalization, taking the maximum with the initial
surface RMS. Keep the bulk convention, separate block convergence tests,
configured nonlinear/linear tolerances, Armijo rule and maximum reductions.
Use that same fixed scale in convergence and merit; changing only merit would
leave the roundoff-level convergence demand unresolved. No new dimensional
parameter, support adjustment or residual smoothing is needed. This changes
the documented normalization contract in current_design section 24 and the
corresponding specification paragraph, so it requires explicit approval.

The focused production regression should use a converged first state with a
nearly balanced surface block, then change prescribed boundary loading away
from the fault. Check nonzero homogeneous Newton directions, directional
surface/bulk Jacobian agreement, successful Armijo progress with the existing
budget, both final dimensional and normalized residuals, correct retained
zero-time histories and committed real-step histories. Retain the existing
forced-exhaustion/rollback tests: the correction must not accept a failed
candidate. Repeat the small fixture on one and two ranks. The unmodified K1
first step is the end-to-end acceptance test, not the isolated scalar model.

## 4. Fixed-length-scale refinement

The additional fixture `diagnostics/refined_initial.prm` includes the unchanged
pilot, doubles uniform bulk resolution from `16x64` to `32x128`, and stops at
`End time=0`. Thus `ell=0.15625 m` stays fixed while normal `h` halves from
`0.015625` to `0.0078125 m` (`ell/h` from 10 to 20). The same nine particles
per cell, initialization formula, activation semantics, fault support,
constitutive parameters and nonlinear/linear tolerances remain in effect.
It is an initial mechanical comparison, not a temporal refinement campaign.

The refined run **passed**, exit 0, in **509.85 s wall / 392.98 s user CPU**,
maximum RSS **626,680 KiB**. It overlapped the supplemental GDB replay, so the
wall time is not an isolated performance benchmark. The phase solve reached
`1.220e-9` after 14 residual evaluations; mechanics converged after eight
accepted Newton updates with relative residuals `(9.795e-15,6.430e-11)`.
All eight steps were accepted without backtracking.

| Raw initial mechanics measurement | 16x64 | 32x128 |
|---|---:|---:|
| Independent full `I_h` (m) | 108.07223820379 | 108.09809507226 |
| Retained initial `C_0` (Pa) | 313.51176014712 | 318.74765576757 |
| Independent evaluated `q_0` (Pa) | 1040.26507957880 | 1044.93071086855 |
| Evaluated QP mean (Pa) | 1040.01656699455 | 1044.87236310022 |
| QP minimum / maximum (Pa) | 1032.590107 / 1049.299642 | 1042.993276 / 1047.221222 |
| Raw RMS error against reference (Pa) | 2.80837889 | 0.69765096 |
| Raw maximum error against reference (Pa) | 9.03456197 | 2.29051098 |
| Within-cell RMS deviation (Pa) | 2.79736188 | 0.69520674 |
| Within-cell maximum deviation (Pa) | 9.28307456 | 2.34885875 |
| Range of cell means (Pa) | 4.657e-10 | 3.747e-10 |
| Particle weighted surface traction (Pa) | 1040.29349115113 | 1044.93979717690 |
| Mean `V_0` error (m/s) | 1.44129072e-7 | 4.59543835e-8 |

The RMS and maximum raw stress errors decrease by factors **4.025 and 3.944**;
the within-cell RMS decreases by **4.024**. No smoothing or stress projection
was performed. The independently computed reference is initialized once from
each resolution's actual retained initial `C_0`, `Theta_0`, and 1500 Pa stress,
using its full FE-profile `I_h`. The change in `C_0` arises from applying the
unchanged initialization/projection on a finer particle/FE grid. Therefore the
shift in absolute mean stress between meshes is not, by itself, a stress-error
measure against one identical prescribed scalar initial state. The table
separates this initialization effect from the raw within-cell oscillations.

Artifacts: `diagnostics/refined-initial.log`, raw accepted CSVs under
`diagnostics/refined-initial/`, and `diagnostics/stress-refined/summary.json`,
`cell_stress_moments.csv`, `raw_stress_error_qps.csv`, and
`particle_surface_traction.csv`. Matching coarse raw-error tables are under
`diagnostics/stress-coarse/`. Every raw stress-error row retains its CellId,
QP number, physical coordinates and integration weight.

## 5. K1 containment status

The user's provisional **K1-only** omitted-fraction allowance is now `1e-4`,
conditional on the completed trajectory and refinement checks. At the saved
coarse initial state, the independently measured `5.64918e-5` omitted fraction
meets that provisional number. The actual initial integrated bulk/surface
slip ratio is `0.9999413015338395`; its deficit `5.86985e-5` meets the unchanged
`1e-4` normalization requirement. Both use the **full** independent `I_h`;
neither renormalizes the supported slip nor changes the scalar reference.

These initial measurements do not establish a completed-trajectory gate.
The refined initial omitted fraction is `5.83027365745e-5`, also below the
provisional allowance, and the measured slip ratio is `0.999947635129399`
(deficit `5.23648706010e-5`). The refined geometry keeps exactly the same
`0.3088215939070757 m` half-width and nine fault vertices; prescribed velocity
error is `2.033e-20 m/s`. The refinement does not drive the fixed-support
omission to zero; it tests this provisional finite allowance only.
No positive-time state has been accepted, K1 remains incomplete, and this
allowance must not be inherited by K2 or later stages. Older artifacts that
report the original `1e-6` containment failure remain valid historical
measurements; their acceptance label predates this provisional permission.

## Verification commands and files

From the repository root unless another directory is specified:

```sh
cmake --build benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build -j4
python3 benchmarks/reconstructed_fault/uniform_shear/audit_stress.py \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/line-search-output \
  --output benchmarks/reconstructed_fault/uniform_shear/diagnostics/stress-coarse
```

The benchmark plugin build succeeded; the main executable was not rebuilt
or changed. The cell audit succeeded with the measurements above.
From `benchmarks/reconstructed_fault/uniform_shear`:

```sh
python3 -m unittest -v test_line_search_diagnosis test_support_resolution
```

**7 tests passed**: four new diagnostic tests (Gauss moment cancellation,
consistent-mass norm, second-order friction remainder, exhaustion with an
exact Newton direction) and three existing tests preserving the distinction
between the full-Ih reference and the retained-support diagnostic. Log:
`diagnostics/line-search-analysis-tests.log`. These are diagnosis checks, not
evidence that a production correction has passed.

The unchanged production unit command
`build-pf-cpdi/aspect --test 'Stage-I*'` also passed: **31 assertions in seven
test cases**. It covers local bounds, release, exact fraction-to-boundary,
two rejections then acceptance, exhaustion without acceptance, and zero/tiny
scale arithmetic. The last two tests do not test a nearly balanced block
coupled to a loading change in the other block; that is the regression gap
the proposed production test must close.

This pass changes only the benchmark QP provenance export and adds the stress
audit, signed line-search analysis, diagnostic tests, two diagnostic parameter
overrides, GDB script and this report. It does not change production source,
the authoritative normalization specification, or existing acceptance tests.
No complete ASPECT suite was run.

Commands from `diagnostics/pilot-build` (one MPI rank):

```sh
timeout 180 gdb -q -batch -x ../line_search.gdb --args \
  /home/ein/repository/aspect/build-pf-cpdi/aspect ../line_search.prm
timeout 600 /home/ein/repository/aspect/build-pf-cpdi/aspect ../refined_initial.prm
```

The first command originally used the trace-only version of the GDB file and
completed the intended nonlinear failure (inferior exit 1; GDB itself returns
0). A subsequent signed-vector capture with a 240-second wall cap expired
before the first real-step residual while overlapping the refinement; its
exit 124 is a diagnostic-run timeout, not a solver result. It is preserved as
`line-search-signed-trace.log`. The isolated signed repeat uses the same
command with a 300-second cap and writes `line-search-signed-complete.log`.
It completed: the inferior exited 1 through the intended nonlinear-failure
path, not the wall-time cap; GDB returned 0. All six rejections and signed
vectors were captured. Analysis command from the repository root:

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/analyze_line_search.py \
  benchmarks/reconstructed_fault/uniform_shear/diagnostics/line-search-signed-complete.log \
  --output benchmarks/reconstructed_fault/uniform_shear/diagnostics/line-search-analysis
```

The existing `analyze.py diagnostics/refined-initial` returns **2**, as expected:
this intentionally `End time=0` run has no completed six-second trajectory,
and that historical analyzer still checks the original `1e-6` containment
threshold. Its dimensional measurements are reused above under the explicitly
approved provisional K1-only allowance; the analyzer/reference and their
thresholds were not silently changed. Log: `refined-initial-analysis.log`.
