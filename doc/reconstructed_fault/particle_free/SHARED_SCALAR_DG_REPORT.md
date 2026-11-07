# Shared scalar DG assembly

## Baseline and scope

Started from clean tracked revision `84f57503c8fa63196e073eec330101b70ff2b966`,
branch `pf-rsf-particle-free`, in the existing worktree. Retained outputs,
checkpoints and manual files remain in place. No checkout/reset, production run,
checkpoint conversion, commit or push is part of this task.

Retained native integration executable SHA256:
`6ea374c5b9b50a20af00fa6e2c365df61d7ac59ed33ad26c7b338d7bb7ba67d9`.
The earlier candidate remains
`5e1e8620a21a5c6dca3a093c5f99d5e31e1d573e167f20e4807130acbba6f137`.
New builds use `build-particle-free/shared-scalar-dg/`; retained executable paths
are never rebuilt. Detailed local evidence is under
`benchmarks/reconstructed_fault/particle_free/evidence/shared_scalar_dg/` and
logged-run records use the `shared-dg-` prefix. Generated evidence is not a
source-commit deliverable. Inherited losses remain 39/45 native-pilot frames and
earlier temporary logs.

Local records: [final operator commands/results](../../../benchmarks/reconstructed_fault/particle_free/evidence/shared_scalar_dg/operators04/runs.json),
[native/unit commands/results](../../../benchmarks/reconstructed_fault/particle_free/evidence/shared_scalar_dg/native-runs.json),
[source/binary identity](../../../benchmarks/reconstructed_fault/particle_free/evidence/shared_scalar_dg/source-identity.json),
[ordinary comparisons](../../../benchmarks/reconstructed_fault/particle_free/evidence/shared_scalar_dg/ordinary-final-comparison.json),
and [lifecycle comparisons](../../../benchmarks/reconstructed_fault/particle_free/evidence/shared_scalar_dg/lifecycle-final-comparison.json).
These links refer to retained local artifacts, not files selected for version control.

## Implementation

The ordinary advection cell and face classes now provide spatial operations
independent of time integration. The native phase manager registers these actual
classes; standalone checks call their same static spatial entries. Ordinary
BE/BDF2, thermal capacities, source terms, SUPG and ALE remain in their adapters.
Phase retains its conservative weak form with the divergence source, local mass
inverse, SSPRK3 subcycling/stage-time inflow and Bernstein bound scaling.

The face classes share face/subface traversal, quadrature, upwind terms, both SIPG
consistency terms, penalty, boundary terms and neighbor contributions. ASPECT's
single-face owner assembles four blocks, including off-process rows. One common
copier inserts neighbor contributions into ordinary block matrices and native
spatial matrices. The old phase private flux/face loops are removed, as is its
private cell assembler. Independent energy integration remains benchmark support.

Fracture supplies `k=2 A ell^2`, unit scale/capacity and the retained trace penalty.
No timestep or temperature factor enters its stiffness. Natural phase diffusion
and transport inflow are separate policies; the optional elliptic-test Dirichlet
condition alone supplies Nitsche terms. Local fracture potential, mechanical
coupling, obstacle/KKT solve and acceptance stay phase-specific.

The [authoritative design](DGQ2_PHASE_FIELD_DESIGN.md) states the weak-form identity
and quadrature requirements. The supported Q2 rules remain Lobatto4 mass,
Lobatto5 volume/face transport and Gauss5 fracture. The phase velocity degree is
supplied explicitly. Mapped SIPG has rational integrands; Gauss5 is retained
oversampling, not an exactness claim. No lower-order quadrature is adopted here.

## Verification

All 27 final operator runs passed (50.01 s including launcher overhead). Nine
matching-quadrature matrix/action checks span Cartesian, affine and bilinear
meshes, one/two/four ranks and static nonmatching interfaces. Maximum matrix
error is **2.78e-16**, stage-action error **1.11e-16**, and material balance error
**8.64e-16**. The nonzero-divergence constant remains constant to L2 **2.21e-16**.

Nine paired retained/new cases cover bounded positive/reverse transport,
nonzero divergence, a smooth unlimited control, SIPG refinement and stationary/
transported obstacles. All recorded numerical comparisons meet a 2e-12 relative
or unit-scale screen; subcycles, elliptic/obstacle iterations and active sets
agree. Transport summaries differ by at most 6.67e-16; obstacle coefficients by
2.28e-15. Limiter activation counts differ by -10 to +7 at roundoff (for example,
716 versus 723); minimum alpha and physical profiles agree. No limiter criterion
was altered to make these integer counts identical.

Mapped SIPG refinement gives L2 3.7261e-3 to 5.1136e-4. Independent energy identity
error is at most 2.90e-15; finite-difference residual/tangent errors are at most
1.42e-11/6.64e-12. The moving obstacle check gives 7.55e-15 error and 6.29e-15 KKT.
This confirms equivalence to retained quadrature, not reduced-quadrature accuracy.

Ordinary temperature and composition tests run three physical steps, including
BE and BDF2, nonmatching initial meshes, two-rank assembly and serial periodic
AMR topology. Printed numeric statistics, solver counts, and every stored
Float32 visualization field match the retained binary exactly. The full unit
suite passes **155 cases / 185,551 assertions**.

The four-step layered mature comparison (four ranks, dt=.0125, t=.05) passes
with one joint publication per accepted step and no retries. Against the retained
integration run, maximum phase difference is 1.53e-16, stress 2.06e-15 Pa,
velocity 1.78e-15 m/s, pressure 1.01e-15 Pa, slip rate 3.34e-16 m/s, age 3.56e-15 s,
normalization 7.11e-15 m and curve position 2.78e-17 m. Transport balance remains
within 3.82e-17. This is a numerical regression, not independent physical
qualification of the moving-fault model.

Forced rejection and its matched accepted-time schedule give bitwise-identical
bulk and surface outputs. Both accepted histories publish once per step; there
is exactly one retry. The final four-to-two-rank restart completes the remaining
step with bulk/surface maxima 1.60e-14/1.07e-14 versus uninterrupted evolution.
Configured nonzero transport inflow passes. After the cost correction below,
all recorded operator metrics (excluding timing) and all 27 mature physical CSVs
are bitwise unchanged; the final unit suite, four-step mature run, full growth
run and rank-changing restart were repeated successfully.

The fixed-fault cohesive case completes 17 loaded steps to t=.53125. Its phase
integral differs from the retained integration result by at most 2.90e-14,
fracture energy by 8.51e-13, and maximum coefficient growth by 5.03e-13. Final
KKT is 3.46e-14 and the final positive integral increment is .00176223239310.
Independent fracture/normalization/mechanical derivative and trace checks pass.
The nonlinear path is not bitwise identical: at step 13 the retained/new runs
use five/four endpoint iterations, and at step 17 their obstacle iteration totals
are 19/21. Both reach the unchanged gate; final-step KKT is 2.27e-14/3.46e-14.
The logged decision comparison localizes this sensitivity to the converged
endpoint search (e.g. step-13 iteration-3 KKT 6.34e-12 versus 5.74e-14), not a
changed tolerance or physical input. The source assembly/summation order changes
at roundoff; no solver modification was made to force identical iteration paths.
A fresh retained-binary growth run reproduces the historical trajectory.

Four ordinary regression fixtures pass: the two nonmatching 2-D cases, serial
periodic composition, and a two-step 3-D spherical-shell diffusion case with
nonplanar faces. Final printed statistics and stored fields are identical to
their retained-binary controls. Two 3-D fixture setup failures (missing prescribed
velocity and incompatible velocity boundary data) were fixed in fresh inputs;
those failed inputs/logs remain available.

Including intermediate failures, baseline controls and cost-correction reruns,
the local ledger contains 59 operator attempts (113.87 s) and 28 native/unit
attempts (684.14 s): **798.01 s total**, excluding builds. No test exceeds 120 s;
each native attempt had a 600 s ceiling. Jobs/builds ran sequentially with at
most four workers/ranks and one numerical thread per rank. Final source and
binary hashes are in the local `source-identity.json`; the final core SHA256 is
`8777e0917d794989990896c5a48c950797f6ffaddd32e1be33d2f644ad7fc9f5`.

### Cost and defects found during verification

The paired operator fixtures use unchanged subcycles. Rank-zero reported operator
storage rises about 1.9-2.2% from distributed sparse-matrix bookkeeping. Short
operator timings are noisy; no throughput claim is made. The final four-step
mature run takes 51.95 s versus retained 56.49 s, with .232 s versus .263 s for
all phase transport calls. This establishes bounded cost, not a general speedup.

An intermediate 3-D comparison exposed a performance defect: ordinary face
scratch was being rebuilt per face and shape/gradient extraction repeated inside
the quadratic matrix loops. Borrowing the existing WorkStream FEValues and
caching per-point shapes/normal derivatives reduces temperature assembly from
3.93 s to **.159 s**, compared with retained **.197 s**, with identical numerical
outputs. The complete operator comparison was repeated after this correction.

**Remaining cost limitation:** the final loaded cohesive case takes 105.79 s
versus a fresh matched-stack retained control's 88.44 s (about 20% longer).
Approximately 97% of both totals is the existing endpoint phase solve. The new
run has fewer logged linear solves/Krylov iterations (668/10,300 versus
699/10,809), so iteration counts alone do not explain the wall-time difference.
The cache correction leaves its diagnostic/physical CSVs bitwise unchanged.
Do not describe the full cohesive route as performance-neutral or attribute
this remaining difference to quadrature without a targeted profile.

Three implementation defects were found and fixed, preserving their failed logs:

- Adopting single-face ownership required writable ghost-row sparsity for all
  four blocks, rather than the old owned-row-only graph.
- Only some MPI ranks have incoming boundary loads. Cell mass-inverse Insert
  operations must finish collectively before boundary Add operations; otherwise
  Trilinos can enter implicit assembly collectives in different orders.
- Coupled Stokes rebuilds the assembler manager. Cached phase operators must
  refresh their borrowed assembler addresses on each registration; otherwise
  the initial cached operator retains deleted assemblers at the first transport.

No physical parameters, damping, timestep rules or acceptance tolerances were
changed to repair these defects. Early compiler/configuration failures are also
retained in the local build logs. Original legacy P1 and I_h smoke failures remain
historical evidence; this task does not relabel them as passes.

## Unchanged limits

The complete coupled tensor history remains Cartesian, despite mapped scalar
support. Mature moving growth, cohesive tip failures and endpoint-energy accuracy
remain separate open work. Production BP3 and cohesive histories are distinct.
Dynamic AMR history transfer, native periodic topology, ALE and 3-D phase are not
newly qualified. No physical law, stress spin lag, stress/surface quadrature,
convergence threshold or checkpoint meaning has changed.

## Outcome and next bounded experiment

Shared spatial cell/interior/boundary assembly is implemented and locally
verified for the stated ordinary/native configurations. The retained numerical
phase results support using it as the current implementation. These are numerical
regressions and operator references, not new earthquake/propagation physics or
production-readiness evidence. Source defaults, checkpoint meanings, SSPRK3/BP,
stress integration and endpoint conventions remain selected.

The next justified experiment is a fixed-input operator/cost comparison: retain
Lobatto5 transport while comparing Gauss3/4 fracture against Gauss5 and a higher
integration reference on affine/bilinear meshes. Profile the remaining cohesive
endpoint runtime separately before claiming a cost benefit. Neither lower
quadrature nor another coupled campaign is enabled by this report.
