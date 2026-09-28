# BP5 initialization: normal-feedback discrimination

## Decision

**Normal-stress feedback is not the principal cause of the initialization
teeth.** Prescribing the total frictional normal traction to 50 MPa leaves
their locations and amplitudes nearly unchanged. The native perturbation-shear
chord RMS increases 0.331%; the velocity chord RMS decreases 0.618%.
This is a fresh zero-history comparison, not a loaded trajectory with inherited
teeth. No real timestep was run, and no tighter tolerance was needed.

The result supports the earlier spatial-representation diagnosis. It does not
separate bulk FE accommodation from localization/work quadrature, nor prove
that normal feedback is negligible later in the evolving trajectory. Those
are outside this initialization-only test.

## Identical inputs and exactly one new mechanical initialization

Reference: `weakening30-dc010-ell100/loading-startup/startup/`, accepted step 0.
Control: `weakening30-dc010-ell100/loading-normal-control/initial/`.

The saved reference executable/library and parameter/input hashes were checked.
The original executable and surface-system source are preserved under the
control study's `provenance/`. The control used the same four ranks, Release
build configuration, 114984-cell mesh, 1156-node continuous fault, endpoint
treatments, loading and physical constraints. Artificial dt0 remains 1e6 s;
both accepted states have physical time and reported physical dt equal to zero.

The independently rebuilt projected mixture and initial state are checked
against the reference before mechanics. The saved effective background shear
coefficients are **imported unchanged**, not recalculated. Their input CSV is
byte-identical to the reference. QP locations, associations, phase, chi, I_h and
JxW match (the latter three have zero maximum difference); native masses agree.
Owned initial-mesh exports are byte-identical. Initial nodal state agrees
bitwise; the output state is unchanged; slip is zero; retained particle Maxwell
stress is zero. Recovered working FE old-stress components are below
6.46e-11 Pa, consistent with zero to arithmetic accuracy.

Only four explicit parameter entries differ: diagnostic plugin, output path,
`Use adiabatic pressure in fault friction = true`, and `Last accepted step = 0`.
Surface pressure and gravity already equal zero. Thus the control uses

\[
 \sigma_n^{friction}=50\,{\rm MPa}+p_{ad}=50\,{\rm MPa},\qquad p_{ad}=0,
\]

while solving and exporting the independent mechanical quantity
`p - tau:N`. No shear-background recalibration, pressure shift, bulk constraint
change, aging, slip accumulation or Maxwell history advancement was performed.
The existing timestep-zero kinematic publication was retained.

## Same native measure and chord diagnostic

Use the earlier report's 27–29.8 km window (29 nodes, locally 100-m spacing).
Native averages use `W=JxW*chi`, the actual Q1 test functions and native row mass
`m_i=sum(W N_i)`. As before,
`c_i(z)=z_i-(z_(i-1)+z_(i+1))/2`; the RMS uses the same row masses.
Neither plots nor measurements smooth or resample the original curves.

| Quantity | Normal feedback | Fixed 50 MPa | Change |
|---|---:|---:|---:|
| Native perturbation-shear chord RMS, Pa | 9.053775 | 9.083765 | +0.3312% |
| Maximum absolute shear chord, Pa | 13.311969 | 13.361314 | +0.3707% |
| Velocity chord RMS, m/s | 1.311954e-13 | 1.303839e-13 | -0.6185% |
| Velocity chord RMS / Vp | 1.311954e-4 | 1.303839e-4 | -0.6185% |
| Mechanical normal chord RMS, Pa | 0.1283024 | 0.1283177 | +0.01192% |

The strongest absolute shear and velocity chord departures both remain at
**29.7 km**. Shear/velocity chord correlations are **0.9999981 / 0.9999967**;
none of the 29 chord signs reverses. The largest matched changes in the fields
themselves are 0.2254 Pa in shear and 1.4418e-14 m/s in velocity.

The table's shear is the **mechanical perturbation after subtracting the same
native background load**, matching the preceding report. Full total shear is
also retained in the CSV/summary; its chord includes curvature/tails of the
fixed projected background near the transition and is not a clean measure of
mechanically generated teeth. Its chord RMS changes from 1222.3519 to
1222.3499 Pa. Nothing was subtracted inside mechanics.

Secondary 30.2–32.8 km window: native perturbation-shear chord RMS changes
12.62447 to 12.62944 Pa (+0.0394%); velocity chord RMS changes 4.19385e-14 to
4.17975e-14 m/s (-0.3361%). Peak locations and all chord signs are unchanged.

## Actual mechanical stress versus frictional normal traction

In the primary window, the actual native `p - tau:N` ranges:

- reference: **3.403811–3.710280 Pa**;
- control: **3.404643–3.711127 Pa**.

The control therefore has a nonuniform mechanical normal stress despite fixed
frictional normal traction. Its corresponding total mechanical normal traction
is 50 MPa plus that small perturbation. Pressure and signed deviatoric-normal
contributions are exported separately; neither is substituted for their sum.

The raw and native comparison CSVs label `mechanical_normal` / total mechanical
normal separately from `friction_total_normal`. In the control's original
`profiles/fault_0.csv`, `sigma_n_weak_Pa` is the consistent Q1 **frictional** normal
traction. In `work_weak_0.csv` and `work_qp_0_rank*.csv`, stress is the **actual
current constitutive** mechanical stress, not the retained zero particle stress.
This deliberate distinction is essential when reading those files.

## Narrow implementation prerequisite and verification

The common bulk-work path previously rejected the public adiabatic-pressure
selector and hard-coded `false` in its cached G coefficients. It now permits
the existing selector and carries `response.uses_adiabatic_friction_pressure`
into both sparse/reference G. The true-pressure path is unchanged. The point
residual and K already used the correct total prescribed normal traction.
No constitutive equation, work weight, solver tolerance or line-search rule
was changed. The specification now documents this explicit diagnostic mode.

A separate, excluded-from-default-build `bp5_normal_control` plugin imports
saved background data, checks initial inputs, checks G, and distinguishes the
two normal quantities. The qualified steady/server plugin binary and package
were not rebuilt or replaced. Diagnostic selection is initialization-only.

Verification on the actual four-rank control:

- Pure 100-Pa bulk pressure direction: G and fresh residual derivative **zero**.
- Homogeneously constrained velocity direction: max G/m = 22169.746 Pa;
  central residual-difference error **4.005e-10 Pa**; sparse/reference G error
  **1.211e-10 Pa** (existing 1e-5-Pa allowance).
- Independent native QP friction reconstruction error: **1.0014e-7 Pa**;
  observer shear/prescribed-normal consistency error **2.6822e-7 Pa**.
- Final normalized bulk/surface residuals **4.024896e-10 / 1.169624e-9**;
  surface RMS **0.00222405 Pa**. Both fresh linear checks passed.
- One accepted Newton update; 42 total Krylov iterations; alpha=1;
  1156 free nodes, zero lower-active nodes. Only accepted timestep 0 exists.
- Elapsed **117.24 s**; peak reported child RSS **2639304 KiB (2.52 GiB)**,
  not an aggregate MPI-memory measurement.

The sandbox's first MPI launch failed before ASPECT execution because sockets
were denied. That log is preserved separately; the authorized MPI launch is
the single completed simulation. No baseline rerun, trajectory, MPI-size study
or full test suite was performed. Both Release and diagnostic plugin builds
passed with `-j4`; `git diff --check` passed.

## Artifacts and commands

Under `weakening30-dc010-ell100/loading-normal-control/`:

- `initial/run.log`, `accepted_steps.csv`, `normal_control_G.csv`, `execution.json`;
- `initial/launch.json`, resolved parameters, imported background and `source-tested/`;
- `comparison/initial-teeth.png`: native shear/velocity/normal fields and chords;
- `comparison/normal-decomposition.png`: signed pressure, deviatoric and sum;
- `comparison/{feedback,prescribed50MPa}_{native,raw}.csv`, `summary.json`.

From the repository root (preparation is deliberately exclusive; do not
overwrite the completed case):

```sh
cmake --build build-pf-cpdi -j4
cmake -S benchmarks/reconstructed_fault/bp5 -B benchmarks/reconstructed_fault/bp5/build
cmake --build benchmarks/reconstructed_fault/bp5/build --target bp5_normal_control -j4
python3 benchmarks/reconstructed_fault/bp5/run_normal_control.py prepare
python3 benchmarks/reconstructed_fault/bp5/run_normal_control.py run
python3 benchmarks/reconstructed_fault/bp5/analyze_normal_control.py
```

Tested executable SHA256:
`96fd6828d904f4a926f1d970b72e4dd7adfc4bdc2d2abbd4981e351e5f9b0088`.
Diagnostic plugin SHA256:
`7d40ed8906032ae20cf859a1f62cd1da35ed6c095bbb66a1324d3b2573de5b4d`.
These are a working-tree test, not a new commit.

**Next decision:** keep the true-normal-stress research model unchanged. If
further attribution is requested, distinguish localization/work quadrature
from bulk FE accommodation; do not change frictional normal stress to remove
these teeth. This bounded task stops here.
