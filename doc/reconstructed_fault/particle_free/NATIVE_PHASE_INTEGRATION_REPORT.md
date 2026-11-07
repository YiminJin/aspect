# Native phase ASPECT integration

**6 November 2026, local worktree.** Implements the user-selected
[task](CODEX_NATIVE_PHASE_ASPECT_INTEGRATION.txt). The authoritative algorithm and
migration instructions are in [DGQ2_PHASE_FIELD_DESIGN.md](DGQ2_PHASE_FIELD_DESIGN.md).
The [earlier DGQ2 report](DGQ2_PHASE_FIELD_REPORT.md) remains evidence for its
separate-block binary and completed long comparisons.

## Baseline and preservation

Verified root `/home/ein/repository/aspect-pf-rsf-particle-free`, branch
`pf-rsf-particle-free`, revision `e4291922732a4f238c7bfb5b07ef82eb3964a9c5`.
Tracked source was clean at entry; retained untracked work was preserved. During
implementation, no checkout, reset, commit or push was performed. The old tested candidate remains
`build-particle-free/dgq2-phase/candidate/aspect-release`, SHA256
`5e1e8620a21a5c6dca3a093c5f99d5e31e1d573e167f20e4807130acbba6f137`.
All **1,020 protected artifacts** match their preservation hashes. Inherited
availability losses remain **39/45 native-pilot frames and earlier temporary logs**.

New core/plugins are isolated under `build-particle-free/native-phase-integration/`.
The tested native executable hash is
`6ea374c5b9b50a20af00fa6e2c365df61d7ac59ed33ad26c7b338d7bb7ba67d9`.
A separate Voro++ build verifies legacy CPDI; the native executable itself requires
no Voro++ or particle manager. The [identity record](../../../benchmarks/reconstructed_fault/particle_free/evidence/native_phase_integration/source-identity.json)
records 950 source/include/unit hashes and matching binary/plugin identities.
HEAD alone does not identify these binaries. Detailed logs and hash ledgers
linked here are retained local artifacts, excluded from the source commit.

## Implementation

- Native phase is one composition named `phase_field`, of semantic type/method
  `phase field`. Its ordinary composition block, FE degree/family, initial data,
  boundary manager and visualization are authoritative. There is no native
  `VariableDeclaration("phase_field")`, proxy field or DG-to-Q1 projection.
  Chemical indices exclude it; unit-sum normalization, particle mapping and
  generic composition reactions cannot consume it. Legacy CPDI retains its Q1 block.
- Ordinary simulator field dispatch transports phase once per attempt from the
  accepted phase/full velocity, before geometry binding. Material geometry receives
  the transported field explicitly. Generic composition solves and Stokes outer
  iterations do not repeat it. Rejection/restart retain the existing joint transaction.
- The ordinary assembler manager owns `PhaseTransport`, with ASPECT scratch/copy
  bases and a spatial cell kernel shared with the operator executable. Mapped
  face/subface scratch is shared by phase transport and SIPG. The scalar symmetric
  penalty/Nitsche kernel is also used by the ordinary advection boundary assembler.
  SSPRK3, local mass inverses, bounds and constrained growth remain phase-specific;
  the implicit BE/BDF2 advection cell solve is not substituted for them.
- Phase operators use mapped normals, gradients, JxW and physical-point neighbor
  matching, including static nonmatching faces. The mapped Jacobian/trace bound
  replaces the Cartesian timestep restriction; whole-cell Bernstein scaling
  preserves the physical mean. Stage-time boundary functions replace implicit zero
  inflow. Constant mass and fracture operators are cached; velocity blocks rebuild
  per attempt and mesh-change signals invalidate spatial caches.
- Native initialization and FE/surface publication require **zero particle managers**.
  Optional particle histories are diagnostic samples, with rollback handling.
  No dummy H, hidden manager or freeze environment variable is used. Fixed-fault
  cohesive growth remains separate from mature moving, growth-disabled evolution.
- Shared DG observations choose the greatest persistent CellId string before
  nonlinear localization. Normal-ray interval ownership and derivatives use the
  same rule. All active source/test/benchmark phase-index consumers use the accessor;
  inactive `tmp/` PhaseFieldRSF archival code is not a native entry point.

Historical benchmark accessor substitutions are API migrations, not qualification
of those old launchers with the new layout. Only the listed new fixtures use
rebuilt matching plugins.

Stress discretization/SSPRK3, accepted bulk-spin lag, selected spin law, endpoint
forcing, RSF, plane-strain convention, stress quadrature and acceptance criteria
are unchanged. This is field/assembly integration, not a new constitutive law.

## Bounded verification

Commands, environments, exits and wall times are in the [run ledger](../../../benchmarks/reconstructed_fault/particle_free/evidence/native_phase_integration/runs.json)
and `benchmarks/reconstructed_fault/refactoring_boundary/evidence/phase-integration-*`.
Every simulation/operator command had a 600-second limit, at most four ranks and
one thread per rank; builds used at most four workers, sequentially with tests.
The timing sample was 31.59 seconds for two layered steps; four steps took 56.49
seconds. Total new simulation/operator/unit runtime was **451.00 seconds (7.52 minutes)**
across 33 attempts, including setup failures and expected rejections; the largest
was 88.94 seconds. Compilation is excluded. The 45-minute budget was not approached.

| Check | Measured result |
|---|---|
| Full existing unit suite | **155 cases, 185,551 assertions pass**; includes semantic registration/chemical exclusion and conflicting-family rejection |
| Cartesian, sheared/rotated and bilinear constant transport | L2 error ≤4.17e−16 with nonzero divergence; physical material balance at roundoff |
| Positive/negative affine transport, bilinear nonzero inflow, static nonmatching faces | Bounds hold; smooth-inflow one/two-rank L2 difference 5.2e−18; maximum step balance ≤1.63e−15 |
| Mapped manufactured SIPG refinement | Bilinear L2 5.970e−3 →8.229e−4 (7.25×); symmetry error ≤1.17e−15; coarse smallest eigenvalue .19034 |
| Independent mapped energy integral vs matrix, residual and tangent | Energy ≤2.90e−15, first derivative ≤1.42e−11, second derivative ≤2.15e−11; includes bilinear nonmatching faces |
| Nonconstant mapped obstacle | 1,120 active/320 free coefficients, KKT 1.08e−14, no negative obstacle gap; positive integral growth .03443 |
| Transported manufactured obstacle | Error 3.64e−15; KKT 5.09e−15; 298 active/62 free coefficients |
| Native fixed-fault early growth, no particles | Positive coefficient growth 8.67e−5 and integral increase 1.81e−6; no healing beyond 4.45e−16; final KKT 6.94e−18 |
| Native fracture/normalization and face trace | Gauss12/32 ray difference ≤9.16e−16; fracture energy/tangent differences ≤4.83e−11/3.53e−13; normalization action ≤4.77e−10 |
| Mature layered DGQ2, no particles | Four steps to .05, one publication per step, no retries; phase balance ≤3.80e−17 |
| Configured time-dependent native exterior phase | One coupled step, exterior `.001+.001*t`; net flux −1.06027877e−4 and balance −2.33e−17; independent incoming-flux integral −1.06028007e−4 (outflow is included only in the net value) |
| Four-to-two-rank native restart | Bulk physical observations differ by ≤1.28e−14; surface state ≤8.89e−15 |
| Forced rejection and identical accepted schedule | All six accepted records are bitwise identical in bulk/surface observations; exactly one retry |
| Native Q1 comparison, no particles | One coupled step passes with the retained remapper |
| Ordinary DG temperature / composition | Temperature physical statistics match retained Blankenbach values at printed precision; clock difference ≤1.01e−17. Two-step composition control passes |
| Legacy CPDI frozen-input operator | Two-rank existing test passes through .02; independent AT1 residual and timestep-independent Jacobian verified |
| Old separate-DG checkpoint | Private copy rejects before deserialization with storage/index/family/degree diagnostic |

The early growth check precedes mechanical loading; its mechanical derivatives
are nearly zero. The final two-rank check therefore continued the existing small
fixture for 17 steps to .53125 (88.94 seconds). With zero particle managers it
reaches integral .15888770005961966, positive final increment .00176223239307260,
KKT 2.27e−14 and no healing beyond roundoff. Nonzero mechanical energy/action
differences are ≤3.94e−10/3.08e−10; Gauss12/32 differs by ≤1.25e−15. Against the
retained fixed-fault growth trajectory, phase integral differs by ≤2.51e−14 and
fracture energy by ≤1.39e−12. This verifies the new particle-free publication
path under loading without changing the physical inputs or endpoint convention.
See [loaded-growth comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/native_phase_integration/loaded-growth.json).

### Comparison with the retained moving baseline

The new four-step case reuses `layered-dgphase-life-control01` physical inputs,
mesh, initial data and dt=.0125. Initial physical observations are identical.
At t=.05, maximum differences are phase 2.93e−4, stress component 1.96e−5 Pa,
velocity 1.52e−6 m/s, pressure 3.27e−6 Pa, normalization .001638 m, age 3.01e−5 s,
slip rate 3.04e−6 m/s, memory 6.02e−7 Pa m and geometry 9.53e−9 m.
The [comparison](../../../benchmarks/reconstructed_fault/particle_free/evidence/native_phase_integration/retained-comparison.json)
uses common physical observations, not changed block-vector layouts.

The new mapped mean bound selects **two** SSPRK3 substeps versus the retained
**four**, so this is not a bitwise transport comparison. Different limiter-stage
sampling is an expected contributor to the small differences; it is not isolated
as their sole cause. Runtime is 56.49 versus retained 65.49 seconds, but removing
particle/unused composition work, caching and changing subcycles all contribute.
This short observation is not an isolated speedup or physical convergence claim.

### Retained failures and setup corrections

One MPI launch was blocked by sandbox sockets and rerun with the existing local
MPI authorization. The first CPDI attempt correctly rejected missing Voro++; a
separate build supplied that dependency. Build-time list access, Jacobian API and
checker vector-proxy errors were corrected; their failed logs remain.

The P1 controlled-history equality test still fails physical admissibility
(`phi_min≈−.007184`, with measurable healing). Phase extrema, increments and crack
integral match [retained P1 evidence](../propagation/P1_report.md) within 8.66e−15.
No bound, tolerance or physical-law adjustment was made to turn it into a pass.
The standalone legacy `phase_field_fault_ih` smoke fixture also fails: its initial
10-iteration phase solve exhausts the existing budget, leaving no reconstructed
geometry for its postprocessor. This fixture is not reported as passing or as
newly qualified. The independent legacy frozen-input residual/Jacobian check
passes; the native no-particle growth/lifecycle checks are separate evidence.

## Migration and remaining work

Use composition degree/family and initial/boundary models. Deprecated phase
settings default to 0/auto and reject explicit conflicts. Retired private native
seed inputs must be moved to the initial-composition model; the fixture preparer
clears them. New `phase-space.txt` identifies composition-v2, index, degree and
family before loading vectors. No old separate-block or production checkpoint
conversion is provided; legacy Q1 retains its historical layout interpretation.

Mapped **scalar phase** is implemented and locally verified. The complete coupled
path remains Cartesian: `phase_field_fault/fe_strain.cc` guards history/quadrature,
and `material_model/history/tensor_advection.cc` retains Cartesian tensor faces.
Normalization's existing mapped fallback does not remove these restrictions.
Dynamic AMR history transfer, ALE, 3-D and periodic topology are not qualified.
Mature moving growth stays disabled; cohesive-tip and endpoint-energy failures
remain independent unresolved scientific issues. Production BP3 and cohesive
histories/pressure conventions remain distinct.

The short comparisons support using the integrated architecture for further local
work. They do not qualify a production earthquake trajectory or independent
coupled energy/slip accuracy. Reverse translation, h=1/12 coupled refinement,
full T=.8 reruns, the former long campaign and server experiments remain **deferred**.
The next justified accuracy experiment is a frozen accepted-velocity replay with
matched subcycles to separate the limiter-stage effect from storage/assembly
changes before drawing physical conclusions from coupled differences.
