# Phase-field fault working record

## Baseline and source identity

On 2026-11-21 the target branch `pf-rsf` and the sibling checkout `../aspect-pf-rsf-particle-free` both had HEAD `0d97d2d3311154ca3f01d801b18f3945dc8b8bad`. The peeled release object exists and is the target's ancestor. The local tag `v3.1.0` is absent. The target already had edits to AGENTS.md and the untracked refactoring kit. The sibling had untracked research reports and benchmarks. Its old build directory is gone. The preserved archive is [reference.tar.gz](../../../aspect-pf-rsf-reference-20261121/reference.tar.gz), SHA256 `a82d362c3504be188e9602cf10c93cf51a26544be5e6d4a7677016cc52981c57`. All replays and source overlays went into /tmp/aspect-slice1. Neither sibling was changed.

From the target root, with new destinations for replays:

```sh
git rev-parse HEAD
git merge-base --is-ancestor 0d97d2d3311154ca3f01d801b18f3945dc8b8bad HEAD
git --no-optional-locks -C ../aspect-pf-rsf-particle-free status --short --branch
(cd ../aspect-pf-rsf-reference-20261121 && sha256sum -c reference.tar.gz.sha256)
D=/tmp/aspect-slice1
R=$D/reference
mkdir -p "$D"
tar -xzf ../aspect-pf-rsf-reference-20261121/reference.tar.gz -C "$D"
(cd "$R" && sha256sum --quiet -c SHA256SUMS)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
```

All 144 files matched. The [archive guide](../../../aspect-pf-rsf-reference-20261121/reference/README.md) identifies the commands, dependency versions, runtime, checker and three mature plugin hashes via SHA256SUMS. For the mature overlay, extract `git archive 3928edaf7e6a59383ef4bf65aeb22f34ff723a4c`, then `patch -p1 < "$R/mature/entry.patch"`. For the tip overlay use `c407e612bf7c9aa1a9c0fc60fd623208dfd53b39`, `tip/qualified.patch` and then unpack `tip/qualified-additions.tgz`. All 16 mature and 24 tip files listed under verified_source_files in their respective source-identity.json matched. The tip manifest lists four subsequent documentation changes separately.

| Route | Inputs, comparison and result |
| --- | --- |
| Mature | One rank, fresh initialization. `mature/input.prm`, `fault.txt`, `runtime/`, `expected/`. Ran `python3 "$R/analysis/replay_midpoint_reference.py" "$R" mature "$D/replay-mature-mpi"`, then `python3 "$R/analysis/check_midpoint_reference.py" "$R" mature "$D/replay-mature-mpi/output"`. All retained columns at steps 0–2 matched exactly, excluding peak RSS per the archive checker. Two publications, zero retries. |
| Tip | Two ranks. `runs/localized-onecut/{input.prm,command.json}`, `tip/runtime`, saved three-step output, forced rejection, tangent and work checks. `localized-oneresume` takes the last step from onecut generation 03. The 18-strip pair branches from `localized-oneo1/output/restart/02` at 5 ms. Re-ran the archive README commands for `midpoint_timestep.py`, `midpoint_audit.py`, `midpoint_closeout.py`. Their outputs were byte-identical to expected/{timestep,point-audit,diagnosis}.json. The timestep diagnostic and stage update are frozen. |
| Fixed reduction | [Replay script](replay_fixed_reference.py) contains the complete input delta and launches the archived mature core with its material transaction observer. `python3 refactoring/phase-field-fault/replay_fixed_reference.py "$R" "$D/fixed-tight"`. The original loose-tolerance reduction was repeated in $D/fixed-repeat and its final physical field CSV was byte-identical to $D/fixed. The vertical fault lies on the clamped left edge of a 1 m square. Two labels, four Q2/Q1 cells, constant phase 0.35, constant viscosity, accepted-trace tensor inflow. The observer's prescribed variation becomes constant at x=0 (3.85 Pa shear, 5 Pa compression, 10 s age). Apply shear traction 0.02+t Pa on the right. Both labels stay fixed, phase differs from 0.35 by at most 1.2e-16. At 12.5 and 25 ms the bulk shear gradients are +0.0980362191757956 and −0.0633663089202283 /s. |

The first fixed reduction used the archive's 1e−8 nonlinear tolerance. Its t=0 rate, 0.17074374609569581 m/s, missed the target by 2.075e−9 m/s (allowance 1.807e−9). At this geometry the independently solvable equation is 5μ(V,10)+V=3.85−0.02 Pa. The donor had a −3.594e−9 Pa residual and its last relative fault residual was 4.323e−9. Tightening **both solvers** to nonlinear 1e−12 and linear 1e−13 gives an archived-core rate of 0.17074374817067778 against independent root 0.17074374817067803. The three exported donor states in $D/fixed-tight supply the test values. The moving control, its thresholds, and its expected outputs were untouched. Failed target evidence is in $D/lifecycle/clean/run.log.

## One ordinary accepted step

[Benchmark input](../../benchmarks/fixed_fault/fixed_fault.prm) → [material and section interface](../../include/aspect/material_model/fixed_fault.h) → [implementation](../../source/material_model/fixed_fault.cc). Scope is the homogeneous mode of one vertical, fixed Q1 section, constant positive phase below unity, affine tangential shear, full normal cross-section. The uniformity check at publication is important. A nonuniform profile, two independently evolving endpoint ages or a curved fault needs a different transfer operation. The target uses block Picard in the ordinary release solver, with a scalar Newton solve at fixed bulk iterate. The archived core solves the coupled equations by condensed Newton. Compare converged fields and independently check the resistance derivative.

1. `Simulator::start_timestep` calls `material_model->update()`. `FixedFault::update` reads the previous accepted velocity gradient and the three static DGQ2 stress fields from old_solution. In this shear the parent tensor and phase are spatial constants, so their transport with compatible inflow is the identity. Rotate the tensor by half the antisymmetric part of the **accepted bulk gradient minus the accepted slip gradient**, times this attempt's duration. Apply Maxwell relaxation β=exp(−GΔ/η), κ=−η expm1(−GΔ/η). At t=0 only the mechanics uses the initialization interval 0.1 s.
2. `solve_timestep → solve_no_advection_iterated_stokes → assemble_and_solve_stokes` calls `FixedFault::prepare_stokes(current_linearization_point)`. `section_mean` integrates on owned cells with the donor's 8×8 rule, then reduces on the simulator communicator. For constant phase, χ=h/∫h dn=1/width. Both Q1 residuals have the same constant traction density. Their normalized equation uses the χ JxW mean. With tangent +y and normal −x, shear is background shear minus τxy, and compression is background compression plus p minus τxx. No pressure normalization. At fixed bulk iterate `section_rate` solves against RSF at **accepted age**, with radiation damping and κχV. Its scalar derivative includes κχ.
3. `assemble_stokes_system → FixedFault::evaluate → Assemblers::StokesIncompressibleTerms::execute` uses κ for the velocity matrix. The elastic force supplies the negative rotated/relaxed parent and localized slip stress to the velocity test gradients. Solve the velocity and pressure blocks, update the linearization point, repeat. At the first physical endpoint the measured shear traction is 0.0325 Pa. The shear rate without the current slip term at this parent would be 0.2616283854 /s. Including slip subtracts 0.1635921662 /s from that response. This change is visible in the solved interior velocity.
4. `Simulator::run → TimeStepping::Manager::update`. A repeat skips `accept_step`. The next attempt rebuilds its coefficients and rate from the original parent. On acceptance `FixedFault::accept_step` checks the converged bulk field and rate, writes the homogeneous L² projection of physical spatial stress to the DGQ2 fields, accumulates ΔV, and integrates the constant-speed aging ODE exactly over Δ. Initial equilibrium publishes a rate with zero physical interval. Section age, slip and rate are shared by the two equivalent material endpoints. Bulk zz is −xx−yy.
5. `postprocess → advance_time → maybe_write_checkpoint → create_snapshot`. The simulator rotates the solution vectors and archives them with the mesh. `FixedFault::save/load` writes the accepted section and its representation and geometry identity to resume.z. On restart old_solution contains the accepted stress and spin-driving velocity. The staged offset and trial rate are recomputed. [Fault statistics](../../source/postprocess/fixed_fault_statistics.cc) reads the accepted endpoint.

The three core edits connect section reduction before Picard assembly, publication after the actual repeat decision, and a model-specific physical-state trailer on checkpoint and restore. Feature-off writes retain the release format. The optional trailer carries the existing Plugins::InterfaceBase save/load map and its material type. This also detects loading a stateful checkpoint under another material. The type check assumes a compatible C++ ABI. A post-nonlinear-solver publication would age the fault before the timestep plugins can reject the solve. The donor instead stages complete solution and linearization backups and a deferred publication callback. Both extra block-vector copies disappear in this fixed slice.

## Retained operations and deferred paths

The reconstructed trees under `$D/mature` and `$D/tip-source` have separate entry points. In the mature tree, the input selects the phase-field material and reconstructed fault. In `source/simulator/core.cc::solve_timestep`, `prepare_material_history` and `transport_material_history` take the accepted DG parent. `solve_native_phase_transport`, `stage_material_geometry` and `rotate_material_history` prepare the mechanical state. `source/simulator/solver/reconstructed_fault_stokes.cc` assembles the condensed section/bulk problem. Its `pending_dg_endpoint_publication` is consumed by `Simulator::run` after the repeat decision, and `finish_dg_material_step` releases the trial. The checkpoint calls `serialize_dg_accepted_history` in `phase_field_fault/dg_accepted.cc`.

The tip input selects the bulk cohesive phase and localized rule. In `source/simulator/solver_schemes.cc::solve_single_advection_iterated_newton_stokes`, `BulkTipCoupling` holds the accepted point/strip parent. `solver/bulk_tip_assembly.cc` evaluates the reduced stage system, including the point constitutive response from `material_model/phase_field_fault/bulk_tip.cc`. After convergence a pending callback installs the candidate `bulk_cohesive_state` on acceptance. The checkpoint calls `serialize_bulk_cohesive_history` and `restore_bulk_cohesive_history`. The localized-onecut replay and its original comparator both passed, including a tangent error of 7.53e−9.

| Donor source in reconstructed mature tree | Disposition |
| --- | --- |
| `source/material_model/phase_field_fault/{constitutive,stress_reaction}.cc`, `rheology/fault_friction.cc`, `history.cc::compute_surface_state_candidates` | Carry physical stress, exact Maxwell coefficients, positive regularized RSF at committed age, exact publication age. The regularized law permits very small positive speeds, and the admissible minimum speed is enforced. |
| `surface_system_bulk_work.cc::assemble_bulk_work_system`, `stage_material_geometry`, `sample_accepted_bulk_spin` | Carry the full-section work measure and the bulk-slip-subtracted spin in the constant mode. General projected reaction, Q1 geometry/normalization, transport and independent labels belong to later increments. Direct DG stress has no separated residual tensor. Donor Z is diagnostic on this route. |
| Transported-objective material observer, layered-background velocity/traction/exterior stress provider and stress-advection comparison DSO | Frozen fixtures. Only the first is needed for the fixed donor reduction. No runtime dependency in the target. CPDI, particle-domain averaging, exact-line comparators and separated stress controls stay in the donor. |
| Tip native point/cohesive memory, strip reduction and return, localized orientation, bounded Bernstein obstacle, stage publication | Preserve as later Example 2 evidence. Its stage age and midpoint publication equations cannot substitute for the mature branch. The standalone driver and earlier orientation hypotheses stay historical. |

For moving stress, release `AdvectionSystem::execute` assembles a scalar timestep matrix/RHS using AdvectionField and old scalar solution blocks. `AdvectionSystemInteriorFace::execute` adds both cell and neighbor matrix contributions, `AdvectionSystemBoundaryFace::execute` uses the active scalar boundary manager, and `Simulator::copy_local_to_global_advection_system` scatters into the global constrained system. The donor needs a spatial residual at *each* SSPRK3 stage, tensor-component exterior traces, neighbor fluxes and a cell inverse mass. Reuse mapping, FEValues and quadrature here. No face operation is needed on the homogeneous fixed slice. The previous shared-assembler experiment 3a21d59cf changed 30 files (+1371/−1417) and was reverted by 319b10b94. A shared flux alone would leave the stage and scatter mismatches.

## Build and verification

GNU 12.4.0, deal.II 9.6.2, Trilinos 14.2.0, p4est 2.8.7 and Open MPI 5.0.6. Both target builds use shared plugin support, World Builder, unity compilation and precompiled headers. The link line supplies the installed MPI library explicitly. Release built with at most three workers, Debug with one. Every runtime comparison used one OpenMP, BLAS and deal.II thread per rank.

```sh
D=/tmp/aspect-slice1
R=$D/reference
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
build_aspect() {
  cmake -S . -B "$1" -DCMAKE_BUILD_TYPE="$2" -DDEAL_II_DIR=/opt/dealii/9.6-local \
    -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
    -DCMAKE_CXX_STANDARD_LIBRARIES=/opt/openmpi/5.0.6/lib/libmpi.so
  cmake --build "$1" --target aspect -j"$3"
}
build_aspect build-slice Release 3
build_aspect build-slice-debug Debug 1
cmake --build build-slice --target setup_tests
cmake -S tests -B build-slice/tests
cmake -S tests -B build-slice-debug/tests -DAspect_DIR="$PWD/build-slice-debug" \
  -DASPECT_BINARY="$PWD/build-slice-debug/aspect" -DASPECT_RUN_ALL_TESTS=ON \
  -DASPECT_COMPARE_TEST_RESULTS=ON
build-slice/aspect --validate benchmarks/fixed_fault/fixed_fault.prm
build-slice/aspect "$D/release-benchmark/input.prm"
build-slice/aspect --test
build-slice-debug/aspect --test
ctest --test-dir build-slice -R '^fixed_fault$' --output-on-failure
ctest --test-dir build-slice-debug/tests -R '^fixed_fault$' --output-on-failure
python3 benchmarks/fixed_fault/check_lifecycle.py build-slice/aspect \
  build-slice/tests/libfixed_fault.release.so "$D/lifecycle-tight"
python3 benchmarks/fixed_fault/check_lifecycle.py build-slice-debug/aspect \
  build-slice-debug/tests/libfixed_fault.debug.so "$D/lifecycle-debug"
/opt/openmpi/5.0.6/bin/mpirun -np 2 -x DEAL_II_NUM_THREADS -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \
  build-slice/aspect "$D/mpi-distributed/input.prm"
python3 "$R/analysis/replay_midpoint_reference.py" "$R" localized-onecut "$D/replay-tip"
python3 "$R/analysis/check_midpoint_reference.py" "$R" localized-onecut "$D/replay-tip/output"
python3 "$R/analysis/replay_midpoint_reference.py" "$R" localized-oneresume "$D/replay-tip-restored"
for field in "$D/replay-tip/output"/tip-*-0003*.csv; do
  cmp "$field" "$D/replay-tip-restored/output/${field##*/}"
done
```

Each replay and lifecycle script takes a fresh final directory. The archived tip restart reproduced the uninterrupted step 3 plane, section, point and reaction files byte for byte on both ranks. For an ordinary run, `build-slice/aspect benchmarks/fixed_fault/fixed_fault.prm` writes to output-fixed_fault. The measured benchmark includes that file and overrides only its output path. The two-rank check includes tests/fixed_fault.prm, loads the Release probe DSO, sets its own output directory and increases initial global refinement from one to two so both ranks own cells.

| Check | Measured result |
| --- | --- |
| Release and Debug | Each passed the complete unit suite, 2,302 assertions in 46 cases. Five assertions cover the new resistance, linearization and aging law. The scoped integration passed in both builds. The ordinary benchmark also validated and ran using built-in plugins alone. |
| Archived fixed fields | Matched cell ID, quadrature index and position in all 108 samples over initialization and two physical endpoints. Largest differences from the donor CSV were 3.29e−14 m/s in velocity, 1.30e−16 Pa in pressure, 4.17e−16 Pa in physical stress and 1.12e−16 in phase. Original field tolerance was 1e−10 absolute plus 1e−8 relative. |
| Retry | The test requested 25 ms, rejected the solved step at the native decision, and repeated with 12.5 ms. Time, duration, all eight solution components, section age, cumulative slip and rate matched the clean shorter trajectory exactly at all three outputs in both builds. |
| Restart | Copied only generation 02 at the first physical endpoint and advanced to the second. The same comparison had maximum absolute difference zero. A changed representation version produced the expected diagnostic. |
| Parallel reduction | Two ranks, 16 cells, 72 samples on each rank. Both ranks checked the same donor section state and their own 72 complete bulk samples at each of the three times. The four-cell case also passed with an empty cell partition on one rank. |
| Feature disabled | Built and copied the pristine release executable to `$D/aspect-v3.1.0` before editing core. `matrix_nonzeros_1` passed on both executables against the existing upstream reference. The final target also resumed a pristine release checkpoint from the same Simple-model input extended to 25 ms. All 12 endpoint statistics columns matched the pristine continuation. |

```sh
cmake -S tests -B "$D/release-tests" -DAspect_DIR="$PWD/build-slice" \
  -DASPECT_BINARY="$D/aspect-v3.1.0" -DASPECT_RUN_ALL_TESTS=ON \
  -DASPECT_COMPARE_TEST_RESULTS=ON -DCMAKE_BUILD_TYPE=Release
ctest --test-dir "$D/release-tests" -R '^matrix_nonzeros_1$' --output-on-failure
ctest --test-dir build-slice -R '^matrix_nonzeros_1$' --output-on-failure
```

Earlier harness attempts are retained under `$D`. MPI socket warnings contaminated the first matrix test's stdout in the sandbox. Running the same comparator with functioning local MPI sockets produced an empty diff before and after the change. Other baseline probes hit a Debug/Release test plugin mismatch or near-zero numeric differences using plain diff on this installation. Their outputs were retained. The earlier loose-tolerance fixed-field mismatch and its independent scalar diagnosis are recorded above.

### Storage and cost

On four cells, ASPECT owns 108 scalar DGQ2 stress DoFs and 36 phase DoFs. These use its normal distributed solution and time-history vectors. The section has three replicated accepted doubles per rank. The attempt stores two coefficients for the bulk response (a scalar and a symmetric tensor) and a trial speed. Quadrature scratch is reused across locally owned cells. There is one reduction of the weighted sums per Picard preparation, with additional extrema checks on parents and endpoints. Publication adds no full-field candidate, checkpoint gather, or per-cell material archive. Generation 02 occupies 7,536 bytes for the target and 6,337 for the reduced donor, including mesh and metadata. The target stores its DG stress in the ordinary mesh solution transfer, the donor stores its tensor separately. The target's compressed resume.z is 636 bytes.

Three alternating, single-rank Release runs used the fixed-tight donor input and the clean target lifecycle input, each in a fresh directory under `$D/cost`. Mesh, physical steps, tolerances, field sampling and checkpoint frequency matched. Each implementation used its own observer and nonlinear scheme. Median ASPECT wall time was 0.169 s on target and 0.221 s on donor. Median end-to-end wall time was 1.525 s and 1.581 s respectively, including startup. The first target launch took 3.490 s end to end. Median process peak RSS from wait4 was 204,020 and 211,852 KiB respectively. The target used 28, 12 and 13 Picard assemblies. The donor used 13 condensed linear solves in total. These four-cell timings include initialization and do not establish scaling.

### Architecture review

Read [the public header](../../include/aspect/material_model/fixed_fault.h), then [the equations, measure and commit](../../source/material_model/fixed_fault.cc). The review decision is whether the fixed material should retain the section reduction and accepted section state when geometry acquires independent labels. The Simulator currently owns the global vectors and the decision to accept. The material exposes a preparation call before assembly, a publication call and read-only accepted section access. The fixed case establishes homogeneous Stokes feedback and transaction lifetime. The next action is the author's review of this implementation at the end of increment 1.
