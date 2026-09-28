# One frozen bulk refinement: shallow mechanical modes

## Decision

Halving the local bulk-cell size **does not restore a stronger alternating
response**. The broad and six-node responses change by -0.00675% and -0.3486%,
whereas the alternating-mode mechanical shear stiffness **decreases by
35.843%**. Its positive restoring response remains, but becomes still smaller
relative to the longer modes. This contradicts the simple explanation that
the coarse bulk mesh was failing to supply restoring traction that would
recover upon refinement.

The measured change is predominantly greater bulk relaxation of the imposed
crack strain, not a changed physical phase/localization field or I_h.
It is consistent with short-wavelength accommodation by the diffuse
eigenstrain representation. It does not yet separate finite-width behaviour
from the fixed fault-Q1/source discretization, establish asymptotic bulk
convergence of the alternating mode, or prove the cause of the later
state-evolving instability. No further mesh level, loading replay, temporal
comparison, stabilization or production change was made.

## Precisely what changed

One nested mesh was prepared from the accepted 300-km Box leaf tree using
deal.II's existing refinement and grading flags. Cells intersecting the
buffer 10–23 km down dip and approximately +/-2.5 km normal distance were
marked for one refinement. Mandatory grading adds neighbouring cells;
endpoints and boundary conditions remain unchanged.

| Quantity | Coarse | Fine |
|---|---:|---:|
| Total active bulk cells | 36282 | 48699 |
| Bulk Stokes Q2 velocity DoFs | 303310 | 404446 |
| Bulk pressure DoFs | 37961 | 50603 |
| Cell side at every nonzero-phase QP in the 15–18-km depth window | 97.65625 m | 48.828125 m |
| Reconstructed fault vertices | 1156 | 1156 |
| Local fault spacing in probe patch | 100 m | 100 m |
| ell | 400 m | 400 m |
| Nonzero-phase QPs in that depth window | 4869 | 19476 |

Fault coordinates and all nodal perturbation values are identical. The
three shapes remain the tapered 3-km bump, 600-m six-node variation, and
200-m neighbouring-node alternation from the preceding response test.
The same production B, shear G, constitutive evaluation, bulk quadrature,
homogeneous perturbation constraints and physical-pressure conversion are
used. Bulk quadrature points change with refinement; the quadrature rule
and numerical tolerances do not.

The ordinary production timestep controller is retained, unmodified. Both
local runs stop within timestep zero; no real timestep is selected or
advanced, and no saved-clock replacement controller is used. The artificial
initialization interval remains 4e6 s, so the actually assembled bulk
coefficient is identical on both meshes:

    kappa0 = 1.2815248119788472e17 Pa*s.

As in the preceding validated linear-response construction, uniform Maxwell
coefficients make mechanical shear responses scale exactly with kappa.
For direct comparison with the existing step-11 table, **both** results
below are multiplied by the same 38.33411519295163 factor, giving

    kappa11 = 4.912611976502280e18 Pa*s.

The table is not a fine loading trajectory or a new step-11 history/state
solve. Frozen history adds an affine bulk load and does not affect these
uniform-coefficient mechanical derivatives. No initial/updated friction
response is substituted for mechanical shear. Native unscaled coefficients
are also stored in `comparison.json`.

## Prolongation and normalization checks

Graphical phase output is Float32 and was deliberately **not** used.
A 53.40-s initialization-only capture exported the actual working Q1 phase
coefficients at all coarse cell vertices, in double precision, together
with the current surface geometry and nodal I_h. It stopped before
publication. This was a field snapshot, not another loading prefix.

On the fine mesh, each phase support point is evaluated in its captured
coarse parent Q1 polynomial. Existing hanging-node interpolation is retained.
The test-only constraint callback overrides the BP3 analytic profile values:
there is **no new analytic stationary-profile interpolation on the fine mesh**.

Measured checks:

- Maximum difference between fine FE phi and the captured parent Q1 function,
  at all owned fine Stokes QPs: **1.2157e-14**.
- Global integral of phi changes by **-8.33e-15 relative**; integral of phi^2
  changes by **+3.77e-15 relative**.
- Every nodal current I_h is an **exact copy** of the coarse snapshot;
  previous I_h is also retained. Fault coordinates match exactly.
- Ordinary fine preparation did compute an unused I_h with a maximum
  3.88264e-6 relative difference. The test seam explicitly replaces it with
  the saved values, and checks exact equality immediately before the probes.
  That unused projection does not enter the reported mechanical response.
- An independent algebraic check evaluates h/I_h from the captured coarse
  polynomial and saved surface coefficients, at the actual exported QPs.
  Relative JxW-weighted L2 errors versus production chi are **9.96e-16**
  for 11911 coarse samples and **2.69e-14** for 47649 fine samples. Maximum
  absolute errors are 7.81e-18 and 1.47e-16 1/m. This also checks that the
  new snapshot reproduces the localization in the previously qualified
  coarse response, not just that the fine run agrees with itself.

The independent check uses the configured AT1 law

    h = m phi (1+p phi)/(1-phi)^2,
    m = Gc / [(8/3) ell Hc],  Hc = cohesion^2/(2G),  p=1,

with the bounded physical phi from the snapshot. Material degradation is
uniform for this fixture. It is a diagnostic formula, not a replacement
of production evaluation.

The **physical localization function and full denominator are unchanged**.
The discrete modal work masses differ by -0.0000824%, -0.004926%, and
-0.049814% because the same integration rule visits different QPs/basis
breaks. These are reported as bulk quadrature differences, not a change
to I_h or a claim that a trajectory's support-normalization gate was tested.
Using the coarse mass denominator for both meshes gives essentially the
same alternating stiffness change, -35.875% instead of -35.843%.

## Separated mechanical response

For each prescribed dV, the diagnostic solves

    A dx = B dV

on private vectors. Define positive restoring coefficients

    k_direct = integral(w dV^2 2 kappa chi S:S) / (dV^T M dV),
    k_relax  = dV^T G_shear dx / (dV^T M dV),
    k_shear  = k_direct - k_relax.

Here w=JxW*chi and M is the native consistent surface mass. The table excludes
sigma_n*mu_V, normal-friction feedback and damping. Values are in
**1e15 Pa/(m/s)** at the common step-11 coefficient.

| Mode | Coarse direct source | Fine direct source | Coarse bulk relaxation | Fine bulk relaxation | Coarse net shear | Fine net shear | Net change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Broad, 3 km | 8.408510 | 8.408508 | 5.555822 | 5.556012 | 2.852689 | 2.852496 | -0.00675% |
| Six nodes, 600 m | 8.408325 | 8.408292 | 5.295412 | 5.306229 | 3.112913 | 3.102062 | -0.34856% |
| Alternating, 200 m | 8.409719 | 8.405305 | 7.246752 | 7.659181 | 1.162967 | 0.746124 | **-35.84304%** |

For the alternating mode, the direct-source change accounts for about
1.06% of the net stiffness reduction; increased bulk relaxation accounts
for the other 98.94%. The retained fraction of direct stiffness falls from
13.829% to **8.877%**. Its stiffness relative to the six-node response falls
from 37.36% to about **24.05%**.

Thus the longer responses are insensitive to this one refinement, whereas
the alternating response is still bulk-resolution-sensitive in the
**opposite direction to a resolution cure**. The fine bulk space can
accommodate more of this imposed short-wavelength diffuse crack strain.
It would be incorrect to conclude from this that the fine short mode is
converged, that all remaining effects are physical, or that changing the
fault grid/regularization is already justified.

The plot `bulk_refinement_shear.png` compares identical nodal directions
and mass-row weak shear responses. Its 1e-12-m/s amplitude is a tangent
normalization, not a finite trial to apply to nearly locked target nodes.

## Numerical checks, execution and lifecycle

All three fine private solves passed the same fresh residual and
production-action checks as the coarse probes:

| Check | Broad | Six nodes | Alternating |
|---|---:|---:|---:|
| Private A solve iterations | 18 | 17 | 17 |
| Fresh relative residual | 6.4130e-11 | 2.8349e-11 | 5.3076e-11 |
| Shear work-pairing relative error | 8.14e-16 | 2.04e-15 | 7.02e-16 |
| Full decomposed/native action relative error | 6.81e-16 | 1.79e-15 | 1.31e-14 |
| Constitutive finite-difference relative error | 1.51e-10 | 2.20e-10 | 4.44e-10 |

These checks include the full constitutive action in the diagnostic harness,
but only its mechanical shear part is used to compare the meshes. The
native weak residual reproduction check also passes. Production solution,
working vector and manager V are unchanged by the probes. An intentional
exception invokes the ordinary rollback path before timestep publication.
Exit code 1 is expected only with the explicit verification markers; it is
not interpreted as a physical trajectory's success or failure.

- Capture: **53.40 s**, maximum child RSS 1260980 KiB.
- Fine response run: **70.31 s**, maximum child RSS 1446596 KiB (~1.38 GiB).
- Four MPI ranks; **123.70 s total ASPECT execution**. RSS is a maximum
  child-process measurement, not summed four-rank memory.
- Both runs completed their intended diagnostics on the first launch;
  no retry, loading prefix or additional level was run.
- Build: `-j4`; Python syntax checks and `git diff --check` pass.

## Files and reproduction

No production numerical source changed. Task additions/changes are:

- `tests/reconstructed_fault_frozen_profile.h`: captured-Q1 prolongation,
  test-only exact normalization retention and preservation checks.
- `tests/reconstructed_fault_refine_mesh.cc`: one nested mesh using existing
  deal.II grading, with coarse-tree identity checked before refinement.
- `tests/reconstructed_fault_mechanical_modes.cc`: opt-in capture/verification
  callbacks around the existing qualified response probes.
- `benchmarks/reconstructed_fault/performance/CMakeLists.txt`: mesh-preparation target.
- `benchmarks/reconstructed_fault/bp3/run_frozen_bulk_refinement.py` and
  `analyze_frozen_bulk_refinement.py`: preserved inputs, explicit launch,
  no automatic retry, offline comparison and independent chi check.
- This report and a follow-up pointer in the preceding response report.

Artifacts are under
`benchmarks/reconstructed_fault/bp3/first_long_run/mechanical-bulk-refinement/`:
`capture/`, `fine/`, `comparison.json`, and `bulk_refinement_shear.png`.
The exact coarse snapshot is `capture/phase_cells.csv` plus `surface.csv`;
the fine run retains `mechanical_modes.csv`, nodal/raw-QP outputs and logs.
Input/binary hashes and execution results are preserved in each launch/run
record. Existing unrelated working changes and previous evidence are retained.
`frozen-bulk-refinement-sources.tar.gz` preserves the task implementation and
report alongside those inputs.

```sh
cmake --build benchmarks/reconstructed_fault/performance/build-gmg \
  --target fault_mechanical_modes fault_refine_mesh -j4

# Completed runs; do not repeat for report completeness:
python3 benchmarks/reconstructed_fault/bp3/run_frozen_bulk_refinement.py capture --execute
python3 benchmarks/reconstructed_fault/bp3/run_frozen_bulk_refinement.py fine --execute

# Offline report reproduction only:
MPLCONFIGDIR=/tmp/aspect-bp3-slip-plot python3 \
  benchmarks/reconstructed_fault/bp3/analyze_frozen_bulk_refinement.py
```

The run driver deliberately refuses to overwrite existing logs. Mesh and
input preparation commands are recorded by its code and retained files;
the commands above are provenance, not an instruction to repeat completed
experiments. This task stops at the requested one-refinement comparison.
