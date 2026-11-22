# Phase-field fault working record

## Stationary review fixes · From `0928a0fa6`

The seven review findings are addressed on `pf-rsf`. Commit `4d579d91d` covers the fault configuration, admissibility, geometry, homogeneous clock and regression cases. Commit `a8222c3e9` covers replay validation. Both donor references and their comparison bounds are frozen. The geometry and phase are still stationary.

| Finding | Fix and discriminating evidence |
| --- | --- |
| Closed-fault predictor | `FaultStokesCoupling::sample_bulk` collectively checks for finite, strictly positive normal compression before evaluating friction, on the predictor and again on the solved fields. Failure throws ASPECT's recoverable nonlinear exception. A triangular 3 Pa tensile traction peaked at 12.5 ms. At the 25 ms predictor ASPECT cut the 12.5 ms step to 6.25 ms and accepted 18.75 and 25 ms. The clean run prescribed these steps explicitly. Both produced exactly four publications including initialization. All labels and 36 bulk samples matched exactly through 18.75 ms. At 25 ms the largest Release differences were 5.56e−17 m/s in label rate, zero in cumulative slip, 2.20e−15 m/s in velocity, 1.91e−17 Pa in pressure and 1.67e−16 Pa in physical stress. The overall Debug maximum was 8.94e−15. A 6 Pa tensile initial load aborted before any publication. |
| Pressure datum | During assembler registration, after BC parsing, `has_normal_traction_boundary` checks the box faces and ASPECT's prescribed velocity component masks. At least one nonperiodic face must admit normal test velocity, so total normal traction fixes the pressure used by friction. This includes a zero natural traction. The original case prescribes tangential velocity on the horizontal faces and passes. Changing only its initial surface pressure from 0 to 0.1 Pa changed the sampled values by at most 1.05e−14 across the three clocks. Closing the normal boundaries failed during setup at initial surface pressures of 0 and 0.1 Pa. A box closed by no-slip, free-slip and periodicity also failed. The release pressure-RHS compatibility flag serves a different purpose and does not resolve component masks. |
| Maxwell ownership | `PhaseFieldFault::parse_parameters` requires the fault to be the simulator's top-level material. The 2× depth-dependent viscosity test failed before Stokes assembly. The averaging wrapper parses its base and does not forward initialization. The first initialization-only guard missed it in the development Debug run. The parse-time check diagnoses both wrappers. This increment supports direct material use. |
| Initial topography | The same parser requires the release zero-topography model before computing the box measure. This uses the same plugin identification as the box mapping. A horizontal section in a unit box with +0.25 m initial topography failed before a width or history was used in a solve. |
| Homogeneous clock | `FixedFaultProbe::execute` checks both time and duration before writing a snapshot or indexing the donor table. A first step of 15 ms failed at step 1. Only the initial publication was recorded. Cases exercising an independent trajectory explicitly disable this table. |
| Replay validation | Every required inline parameter is checked before destination creation, compilation or launch. The script also validates the homogeneous shear expression and the three pinned observer substitutions. Separate trials inherited either the output directory or stress rotation through an include. Both ordinary and optimized Python diagnosed the missing key. All four trials using the real archived parent preserved its output-tree digest `a94031b969011a92d9ae991600f428a0012054a6a9e434c3359d574b4165a1fa`. A standalone Python test covers both modes without donor dependencies. |
| Box endpoints | `StationaryFault::initialize` compares endpoint coordinates to the box faces at a tolerance scaled by coordinate magnitude and extent. The test accepts labels 0.1, 0.2, 0.3 m in both orientations for a 0.1 m origin and 0.2 m extent. The ordinary Release run on this translated box published all three clocks. Both the unit test and the native run rejected a section truncated to 0.299 m. The stored material coordinates and strict ordering are preserved. |

**One ordinary timestep.** The public state is still `StationaryFault::Label` in `include/aspect/reconstructed_fault/stationary_fault.h`. `PhaseFieldFault::update` prepares the Maxwell coefficient and immutable stress parent. At each Picard preparation `sample_bulk` validates the dynamic normal traction and `solve_rates` uses the accepted ages to assemble Q1 label forces. `FaultStressLoad::execute` returns the spatial stress offset to the velocity RHS using the test gradient at each quadrature point. After bulk convergence, the ordinary solver asks the coupling to check and stage the rates. The existing timestep manager decides on repeat or advance. Only the advance calls `accept_step` and then `StationaryFault::accept` for DG stress, age, slip and rate. The first donor-matched physical timestep at 12.5 ms retained rates 0.09123104, 0.09510370 and 0.08832194 m/s from bottom to top. No change to a public interface, generic Simulator hook, weak form or aging equation was required. A single integer closure reduction replaces the previous minimum-double reduction. There are no additional persistent or candidate field copies.

**Native regression results.** Both parameter files validated. Both scoped CTests and both lifecycle trajectories passed in Release and Debug. Each unit suite passed 2,306 assertions in 47 cases. Against the nonuniform Release donor, the largest differences were 4.43e−15 m/s in rate, 3.56e−14 s in age, 3.21e−17 m in slip, 2.61e−14 m/s in bulk velocity, 9.88e−16 Pa in pressure and 3.37e−15 Pa in physical stress. The original forced repeat retained exact clean-run values at all three clocks in both builds. The one-/two-rank 16-cell comparison checked 144 points at each clock, distributed 72/72. Its maximum differences were 4.49e−16, 3.56e−15 and 2.77e−15. On the four-cell pulse (36/0 points on two ranks), both ranks returned from the recoverable failure. The run published four endpoints, agreeing with the clean serial fields to 2.20e−15.

**Reproduction.** Fresh outputs and logs are under `/tmp/fault-review.tVp3VSMv` for this run. Set `D` to a new directory to reproduce. The two native build trees use the GNU 12.4.0 / deal.II 9.6.2 / Open MPI 5.0.6 configurations below. Each process used one thread. The changed C++ files were formatted with the checkout's `indent_file` and astyle 2.04.

```sh
D=$(mktemp -d /tmp/fault-review.XXXXXXXX)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
for mode in release debug; do
  build=build-slice; config=Release
  if [ "$mode" = debug ]; then build=build-slice-debug; config=Debug; fi
  cmake --build "$build" --target aspect -j1
  cmake -S tests -B "$D/final-tests-$mode" -DAspect_DIR="$PWD/$build" \
    -DASPECT_BINARY="$PWD/$build/aspect" -DCMAKE_BUILD_TYPE="$config" \
    -DASPECT_RUN_ALL_TESTS=ON -DASPECT_COMPARE_TEST_RESULTS=ON
  cmake --build "$D/final-tests-$mode" --target fixed_fault stationary_fault -j1
  for test_case in stationary_fault fixed_fault; do
    python3 -O benchmarks/fixed_fault/check_lifecycle.py "$build/aspect" \
      "$D/final-tests-$mode/libfixed_fault.$mode.so" "$D/final-$test_case-$mode" --case "$test_case"
  done
  ctest --test-dir "$D/final-tests-$mode" -R '^(fixed_fault|stationary_fault)$' --output-on-failure
  (cd "$build/unit_tests" && ../aspect --test)
done
build-slice/aspect --validate benchmarks/fixed_fault/fixed_fault.prm
build-slice/aspect --validate benchmarks/fixed_fault/stationary.prm
python3 refactoring/phase-field-fault/test_replay_stationary_reference.py -v
ctest --test-dir "$D/final-tests-release" -R '^(matrix_nonzeros_1|viscoelastic_numerical_elastic_time_step)$' --output-on-failure
```

The MPI input files include `tests/stationary_fault.prm`, set the Release probe library shown above and use separate outputs. The 16-cell pair sets initial global refinement to 2. The four-cell two-rank pulse includes the lifecycle harness's pulse-cutback input with its own output directory. For the native decimal check, set the box's Y origin to 0.1 and Y extent to 0.2, with vertices at x = 0.5 and y = 0.1, 0.2, 0.3. The short variant replaces only the final y with 0.299.

```sh
build-slice/aspect "$D/rank1/input.prm"
/opt/openmpi/5.0.6/bin/mpirun -np 2 build-slice/aspect "$D/rank2/input.prm"
python3 -O benchmarks/fixed_fault/compare_snapshots.py "$D/rank1/output" "$D/rank2/output" --points 144
/opt/openmpi/5.0.6/bin/mpirun -np 2 build-slice/aspect "$D/pulse-rank2/input.prm"
PYTHONPATH=benchmarks/fixed_fault python3 - "$D" <<'PYTHON'
from compare_snapshots import compare_step
from pathlib import Path
import sys
d = Path(sys.argv[1])
for step in range(4):
    print(compare_step(d/"final-stationary_fault-release/pulse_clean/output", d/"pulse-rank2/output", step))
PYTHON
build-slice/aspect "$D/decimals/input.prm"
build-slice/aspect "$D/short/input.prm"   # rejected before step 0
build-slice/aspect "$D/open-datum/input.prm"   # stationary fixture, initial surface pressure 0.1 Pa
python3 benchmarks/fixed_fault/compare_snapshots.py "$D/final-stationary_fault-release/clean/output" "$D/open-datum/output"
```

The Release matrix control passed with ordinary MPI socket access. A preliminary sandbox run differed in eight launcher-warning lines. The original-elasticity depth-average CTest again failed its checked-in shear at 1,000 years. The five depth averages match the local pristine 3.1.0 results investigated in the preceding maintenance record. The top bin is −0.0346938 Pa against +0.0967461 Pa in the checked-in reference. Both normal stresses agree at ±4.79488e6 Pa.


**Independent positive replay.** The retained sibling archive passed its manifest check. The archived mature executable produced a fresh homogeneous parent. Rebuilding the stationary donor at `663937bed` and running the repaired replay script under optimized Python regenerated all three checked-in three-label snapshots byte for byte. Both the donor source extraction and all generated outputs lived under `$D`.

```sh
mkdir "$D/stationary-source"
git archive 663937bed9305df2823ac77a3fc2c3925bb94caa | tar -xf - -C "$D/stationary-source"
cmake -S "$D/stationary-source" -B "$D/stationary-build" -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/opt/dealii/9.6-local -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DCMAKE_CXX_STANDARD_LIBRARIES=/opt/openmpi/5.0.6/lib/libmpi.so
cmake --build "$D/stationary-build" --target aspect -j2
tar -xzf ../aspect-pf-rsf-reference-20261121/reference.tar.gz -C "$D"
(cd "$D/reference" && sha256sum --quiet -c SHA256SUMS)
python3 refactoring/phase-field-fault/replay_fixed_reference.py "$D/reference" "$D/homogeneous"
python3 -O refactoring/phase-field-fault/replay_stationary_reference.py \
  "$D/stationary-source" "$D/stationary-build" "$D/homogeneous/input.prm" "$D/replay-verified"
diff -qr benchmarks/fixed_fault/stationary-reference "$D/replay-verified/comparison"
```

**Next action.** Author review of the stationary configuration checks and the predictor failure path. Inclined geometry remains a separate increment.

## Stationary coupling repair · Initial implementation

**Implemented from `e91e801ba`.** The numbered second download contained the requested patch and task, and the unnumbered ZIP contained the earlier kit. Both were extracted under `/tmp/stationary-maintenance.Rai2boDP`. The newer patch passed `git apply --check` and applied cleanly to the author's path instructions and recorded results. Commit `79df67037` moves the unchanged friction and aging equations into paired rheology files and names the traction and fixed-age speed derivative. Commit `49bc11838` implements the coupling repair, removes prototype persistence and adds the discriminating tests. The checked-in donor snapshots, homogeneous expected values and comparison bounds remain frozen. Both ordinary parameter files run with original elasticity disabled. All native fault runs begin at physical time zero in fresh directories.

### One accepted stationary step

1. `source/simulator/core.cc`, `start_timestep` calls `PhaseFieldFault::update`. The material computes the Maxwell relaxation and effective viscosity. `FaultStokesCoupling::prepare_parent` samples accepted velocity, stress and label rate, asks the material for the rotated/relaxed physical tensor, and projects it to owned DGQ2 cells. Spin uses the accepted bulk gradient minus the localized slip gradient. Initialization uses 0.1 s for Maxwell mechanics and zero physical time for rotation, age and slip.
2. `solve_no_advection_iterated_stokes` emits `prepare_stokes_picard_iteration`. The coupling samples the current bulk solution at 8×8 Gauss points. `solve_rates` uses the two neighboring Q1 labels and the unchanged work measure and consistent tangent. `StationaryFault` owns the oriented vertices, projection and each accepted age, slip and rate. The coupling owns trial rates and cell-local scratch. The material owns these components and the constitutive law. Only the four section vectors are reduced across ranks.
3. The material supplies $\kappa$ through ordinary viscosity and the spatial offset through `FaultStressOffset` on request. The offset is the projected parent minus $2\kappa(h/I)V\,\mathrm{sym}(t\otimes n)$. The registered `FaultStressLoad::execute` adds $-\sum_q JxW_q\,\tau_{\rm offset}:\varepsilon(v_i)_q$ to the velocity RHS. Each symmetric gradient comes from FEValues at the current quadrature point. The same operation runs on RHS-only assembly. Registration also adds ASPECT's existing $-2\kappa/3\,\mathrm{div}(u)\mathrm{div}(v)$ correction exactly once, using the matching preconditioner when needed. This keeps the 3-D deviator consistent with pointwise traction and physical stress in 2-D plane strain. Continuity remains incompressible.
4. When the bulk stopping condition succeeds, `check_stokes_picard_convergence` calls `FaultStokesCoupling::check_convergence`. It recomputes rates from the solved velocity and pressure. Each rate must agree with its assembly rate to relative 1e−8. Disagreement continues the ordinary Picard loop under its iteration limit and failure policy. Agreement stages the converged rates. After the timestep manager decides to advance, `post_accept_timestep` calls `accept_step` to write the physical DG stress blocks, then `StationaryFault::accept` accumulates $\Delta t V_j$ and advances accepted age once. A repeat skips publication and reconstructs the next attempt from the same parent.
5. ASPECT's **named additional outputs** visualizer asks the material for `fault_tau_xx`, `fault_tau_yy` and `fault_tau_xy` (Pa). They read the accepted, named DG components. The material looks up the three stress roles and the generic phase role by name and validates their types. The linearized incompressible EOS receives one output slot for background lithology. No stress or phase value enters density mixing.

**Architecture review.** The normal caller of `PhaseFieldFault` is the material manager/Simulator. Its local Maxwell operations and the paired `RateAndStateFriction` law follow the release material/rheology split. `FaultStokesCoupling` is the FE/MPI owner for prep, coupled convergence and writeback. It connects to the existing assembler registration signal in the same manner as the anisotropic-viscosity example. `FaultStressLoad` uses the additional-output request pattern of `tests/additional_outputs_03.cc`. The three new signals are the shared-core changes relative to release. They expose a pre-assembly point, a check within the solver's stopping decision, and a publication point after the repeat decision. No shared solver or assembler selects a concrete fault material. The named-output reader and label statistics serve normal runs. The analytic probe, prescribed work test, deliberate rejection and snapshot checker live with the fixtures. The straight, mesh-aligned section and single-lithology material are limitations of this increment. The ordinary release checkpoint code and default assembler selection are byte-identical to `0d97d2d33`. The material diagnoses resume, timed/step/scheduled checkpoints and checkpoint-on-termination while its label history is in memory.

### Evidence from fresh runs

| Check | Result |
| --- | --- |
| Homogeneous and three-label | Debug and Release fault CTests passed. Each unit suite passed 2,302 assertions in 46 cases. In the nonuniform Release comparison, maximum errors were 4.43e−15 m/s in label rate, 3.56e−14 s in age, 3.21e−17 m in slip, 2.61e−14 m/s in velocity, 9.88e−16 Pa in pressure and 3.37e−15 Pa in physical stress. Debug maximum stress error was 3.45e−15 Pa. All three times and 36 interior samples were checked. |
| Work and graphical stress | On a Q2 polynomial test velocity the production stress assembler matched independently integrated signed work within 1e−12 for both rebuild flags, with zero pressure load. Two equivalent clamped-wall nonuniform runs gave zero sampled differences with 47 versus 3 preconditioner rebuilds. At 12.5 ms all three named stress outputs equaled the accepted DG values at all 36 graphical points. Homogeneous shear was 0.0325 Pa. The viscous contribution alone was about 0.01218 Pa. |
| EOS and field order | With a 123 kg/m³ composition contrast, positive xx stresses of 1 and 1e6 Pa at 293 K and fixed pressure both produced density 3300 kg/m³. The clean trajectory also survived permutation to phase, xy, xx, yy with zero numerical differences. |
| Retry and convergence | Both cases and builds gave zero difference at every clock, label and bulk sample after a rejected 25 ms step retried at 12.5 ms. Bulk tolerance 0.05 exercised the rate veto 20 times on the three-label Release case and 23 on the homogeneous case. A one-iteration cap failed before publication when the bulk criterion passed and rates disagreed. Requests for disk continuation or a checkpoint failed before solving. |
| Python and MPI | A copied step-1 shear reference perturbed by 0.01 Pa failed in ordinary and optimized Python at the same cell, quadrature point and column. With 16 cells and 144 points, a one-/two-rank match by material vertex and cell/quadrature identity gave maximum differences of 4.49e−16, 3.56e−15 and 2.77e−15 at steps 0, 1 and 2. Each of two ranks owned 72 sample points. |

**Fault-off controls.** The Release `matrix_nonzeros_1` CTest passed. Two original-elasticity CTests completed their solves and failed comparisons with their checked-in regressions. Both local builds, the repair and a fresh archive of unmodified ASPECT 3.1.0 (`0d97d2d33`), agree on the following outputs.

- In `viscoelastic_numerical_elastic_time_step` the complete depth-average files are byte-identical. Both agree with the archive at time zero. At 1,000 years and 10 km depth the reference shear is +0.0967461 Pa and both local binaries give −0.0346938 Pa. The normal stresses agree at ±4.79488e6 Pa. On the pristine source alone, reducing the linear Stokes tolerance from 1e−7 to 1e−12 reduces the maximum absolute shear over the five depths from 0.0618006 to 2.20614e−7 Pa. This isolates the solver-accuracy sensitivity of the near-zero shear. The historical build environment for the checked-in values was not reconstructed.
- In `visualization_shear_stress_viscoelastic` all 576 numerical rows are identical between local binaries at each of steps 0 and 1. The first checked-in difference at step 1 is the near-zero pressure at (0.5, 0), 6.58453e−14 versus 1.16127e−14. The four shear-stress output columns agree at that point. Both failed diff files and the pristine control outputs are retained under `$D`.

The initial development executable checked the mass-conservation selector before ASPECT resolved its default `ask material model` value and stopped before step zero. The model now accepts either explicit incompressible continuity or the selector resolved from its own incompressible EOS. The rejected run is in `$D/pilot`. The donor comparisons use the corrected executable.

**Cost.** Four owned cells require 108 rotated DG parent coefficients per attempt and 256 owned section quadrature samples per rate solve. The endpoint stages three owned scalar stress blocks. The normal material output allocates the offset only when Stokes requests it. The physical diagnostic allocates three scalar arrays only when a caller requests named outputs. There is no fault checkpoint collection. Ordinary one-rank Release runs with visualization took 53 bulk solves for the homogeneous case and 46 for the three-label case. One process measurement including startup and output gave 1.957 s / 206,296 KiB and 4.543 s / 206,224 KiB respectively during concurrent build activity. These single measurements have different output settings from the historical checkpointed timings.

### Reproduction

The native builds use the existing GNU 12.4.0, deal.II 9.6.2, Open MPI 5.0.6 and ASPECT Release/Debug configurations recorded below. At most four workers total were used and each simulation had one thread per rank. The checkout's `indent_file` from `contrib/utilities/indent` was applied to the changed C++ files with `/opt/astyle/2.04/bin/astyle` and `contrib/utilities/astyle.rc`.

```sh
D=/tmp/stationary-maintenance.Rai2boDP   # choose a new directory for a replay
unzip -q "$HOME/Downloads/aspect31-pf-refactor-kit.zip" -d "$D"
unzip -q "$HOME/Downloads/aspect31-pf-refactor-kit (1).zip" -d "$D/updated"
# At the original e91e801ba worktree before implementation
git apply --check "$D/updated/pf-refactor-kit/update-stationary-coupling.patch"
git apply "$D/updated/pf-refactor-kit/update-stationary-coupling.patch"
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for mode in release debug; do
  build=build-slice; config=Release
  if [ "$mode" = debug ]; then build=build-slice-debug; config=Debug; fi
  cmake --build "$build" --target aspect -j1
  cmake -S tests -B "$D/tests-$mode" -DAspect_DIR="$PWD/$build" \
    -DASPECT_BINARY="$PWD/$build/aspect" -DCMAKE_BUILD_TYPE="$config" \
    -DASPECT_RUN_ALL_TESTS=ON -DASPECT_COMPARE_TEST_RESULTS=ON
  cmake --build "$D/tests-$mode" --target fixed_fault stationary_fault -j1
  ctest --test-dir "$D/tests-$mode" -R '^(fixed_fault|stationary_fault)$' --output-on-failure
  (cd "$build/unit_tests" && ../aspect --test)
  python3 benchmarks/fixed_fault/check_lifecycle.py "$build/aspect" \
    "$D/tests-$mode/libfixed_fault.$mode.so" "$D/$mode-finish-fixed"
  python3 -O benchmarks/fixed_fault/check_lifecycle.py "$build/aspect" \
    "$D/tests-$mode/libfixed_fault.$mode.so" "$D/$mode-finish-stationary" --case stationary_fault
done
build-slice/aspect --validate benchmarks/fixed_fault/fixed_fault.prm
build-slice/aspect --validate benchmarks/fixed_fault/stationary.prm
python3 benchmarks/fixed_fault/compare_snapshots.py benchmarks/fixed_fault/stationary-reference "$D/release-finish-stationary/clean/output"
ctest --test-dir "$D/tests-release" -R '^(matrix_nonzeros_1|viscoelastic_numerical_elastic_time_step)$' --output-on-failure
ctest --test-dir "$D/tests-release" -R '^visualization_shear_stress_viscoelastic$' --output-on-failure
```

The rank input files under `$D/one-rank` and `$D/two-ranks` include `tests/stationary_fault.prm`, load the Release probe, set initial global refinement to 2 and use separate output directories. Run `build-slice/aspect "$D/one-rank/input.prm"` and `/opt/openmpi/5.0.6/bin/mpirun -np 2 build-slice/aspect "$D/two-ranks/input.prm"`, then `python3 benchmarks/fixed_fault/compare_snapshots.py "$D/one-rank/output" "$D/two-ranks/output" --points 144`. For the negative Python check, copy the checked-in three files to `$D/perturbed` and add 0.01 to the xy column of the first bulk point in step 1. Run the same comparison in `python3` and `python3 -O` using that reference directory and the clean three-label output.

For the independent fault-off control, the two pristine input files include the original test input by absolute path and set a fresh output directory. The third includes the original numerical test with only the linear Stokes tolerance changed to 1e−12. The raw CTest differences are in `$D/upstream-release.log` and `$D/upstream-viz.log`.

```sh
mkdir "$D/release-source"
git archive 0d97d2d33 | tar -xf - -C "$D/release-source"
cmake -S "$D/release-source" -B "$D/pristine-build" -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/opt/dealii/9.6-local -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DCMAKE_CXX_STANDARD_LIBRARIES=/opt/openmpi/5.0.6/lib/libmpi.so
cmake --build "$D/pristine-build" --target aspect -j2
for run in pristine-numerical pristine-viz pristine-tight; do
  "$D/pristine-build/aspect" "$D/$run/input.prm" > "$D/$run/run.log" 2>&1
done
cmp "$D/pristine-numerical/output/depth_average.txt" "$D/tests-release/output-viscoelastic_numerical_elastic_time_step/depth_average.txt"
for step in 00000 00001; do
  file=solution/solution-$step.0000.gnuplot
  diff <(rg -v '^#' "$D/pristine-viz/output/$file") <(rg -v '^#' "$D/tests-release/output-visualization_shear_stress_viscoelastic/$file")
done
```

**Next action.** Author architecture review of the stationary stress return, immutable parent and publication points above. Inclined geometry starts in a subsequent assignment.

## Increment 2 · Stationary reconstructed mature fault (historical result)

The target starts at `2a74e2666` on `pf-rsf`, descended from release `0d97d2d33`. The homogeneous regression retains the input, archived donor values and tolerances from `88cd4eee1`. The stationary implementation handles an ordered, straight, axis-parallel Q1 fault spanning a Cartesian box. Every interior knot is at a transverse bulk face. Phase is fixed at a spatial constant strictly between zero and one. The two tests use the left wall and the interior line x = 0.5 m respectively.

### Stationary donor and independent comparison

The archived mature executable selects transported material history. Its homogeneous reduction remains the reference for increment 1. For the stationary experiment use its ancestor `663937bed9305df2823ac77a3fc2c3925bb94caa`, where the separated stationary DG history, accepted slip-subtracted spin and bulk-work residual are all available together. Reconstruct and build that tree in a separate directory. The replay script adapts the observation plugin's selector and publication checks and prescribes 0.35 on the donor's continuous Q1 phase through the BP3 constraints signal. The compiled equations, geometry, Stokes assembly and time integration come directly from that Git tree.

In the donor, follow `ReconstructedFaultSurfaceSystem::assemble_bulk_work_system` in `source/reconstructed_fault/surface_system_bulk_work.cc`. Each associated bulk quadrature point supplies dynamic pressure, strain, stress history and its two Q1 shapes. The constitutive response uses the interpolated age and rate, and deposits shear minus resistance into both rows with $N_i(h/I)JxW$. The native solver in `source/simulator/solver/reconstructed_fault_stokes.cc` returns the slip stress to the momentum equation and stages the converged endpoint. The repeat decision in `source/simulator/core.cc` controls publication. `phase_field_fault/dg_objective.cc` rotates the accepted split tensors with the spin of the previous bulk velocity minus slip. The physical stress in the donor CSV is $T-R-B(Z)$. With constant viscosity, constant localization and cell-resolved Q1 knots, $B(Z)$ is in the cell DGQ2 space. Linear projection of the rotated physical tensor gives the same endpoint in a single physical-stress representation.

Use the 1 m square, four Q2/Q1 cells and three labels at y = 0, 0.5 and 1 m. They start at the same age (11.682941969615793 s) and background shear (3.7106105802501554 Pa). Apply $0.02+t+0.12(y-0.5)$ Pa tangentially at the right boundary. The donor published two stress endpoints and zero geometry changes. The target reads fixed phase in its native static DGQ2 field. The donor uses continuous Q1 for the same constant. Compare physical velocity, pressure, all three stress components and phase at the same 36 QGauss(3) locations at each time. Temperature is prescribed at 293 K. The donor exports age and rate directly. The slip reference sums recorded endpoint rate times accepted duration.

| Time (s) | Rates from bottom to top (m/s) | Ages from bottom to top (s) |
| --- | --- | --- |
| 0 | 0.0966529145, 0.1002292839, 0.0943642083 | 11.68294197, 11.68294197, 11.68294197 |
| 0.0125 | 0.0912310449, 0.0951036956, 0.0883219360 | 10.14246038, 10.08130943, 10.18864051 |
| 0.025 | 0.0903738068, 0.0943475169, 0.0874068945 | 8.81843454, 8.71111890, 8.89966709 |

The initial run in `$D/target` differed by up to 5.60e−5 m/s in velocity and 1.43e−4 Pa in pressure at initial equilibrium. Its full incompressible velocity matrix omitted the plane-strain trace term used in the traction and stress. Discrete Q2 velocity is only weakly divergence free. The stationary donor explicitly adds the existing ASPECT strain-rate trace assembler while retaining incompressible continuity. Activating that same assembler closed the comparison with the original donor data. All runs and both sets of snapshots remain separate.

### An ordinary accepted step, in source order

1. `source/simulator/core.cc`, `start_timestep` → `source/material_model/fixed_fault.cc`, `FixedFault::update`. Read accepted velocity and the three physical stress components from `old_solution`, and rates from `StationaryFault::labels()`. Rotate the spatial stress with the accepted slip-subtracted spin, relax it, and L²-project the result onto the owned DGQ2 cells. Initialization uses the Maxwell initialization interval and zero physical duration for rotation, slip and aging.
2. `source/simulator/solver_schemes.cc`, `assemble_and_solve_stokes` → `FixedFault::prepare_stokes` and `sample_bulk`. At each owned QGauss(8) point form current driving shear and compression from velocity, pressure and the projected parent. `include/aspect/reconstructed_fault/stationary_fault.h` defines the material vertex and projection. `source/reconstructed_fault/stationary_fault.cc`, `project` and `solve_rates`, assemble the two-node work residual, its diagonal and off-diagonal tangent with the same Q1 interpolation. Friction reads the accepted age. Solve the positive tridiagonal Newton correction at fixed bulk iterate.
3. `FixedFault::stress_offset` interpolates the two current rates back to each Stokes evaluation point and adds $-2\kappa(h/I)V\,\mathrm{sym}(t\otimes n)$ to the projected parent. `evaluate` supplies the negative of this offset via `ElasticOutputs`. `source/simulator/assemblers/stokes.cc` contracts it against the test gradient. `source/simulator/assembly.cc` includes the plane-strain trace correction for this material. The ordinary velocity-pressure solve updates the Picard iterate.
4. After convergence `Simulator::run` calls `time_stepping_manager.update()`. On a repeat the simulator bypasses the entire publication and rebuilds from the accepted parent at the new duration. On acceptance `FixedFault::accept_step` checks every rate against the converged bulk solution and writes the three owned DG stress blocks. `StationaryFault::accept` advances each vertex's age with the exact aging law and accumulates $\Delta t V_j$ once. Ordered index and fixed coordinates identify each vertex in the versioned material checkpoint. ASPECT checkpoints bulk stress and velocity with the mesh and solution histories.

### Verification and reproduction

The implementation and native regressions are committed at `fba90b134`. The two Release and two Debug fault CTests passed. Both complete unit runs passed 2,302 assertions in 46 cases. The feature-off Release `matrix_nonzeros_1` comparison passed on rerunning with local MPI socket access. Its sandbox run differed only by launcher warnings in screen output. The two fault CTests run with the normal checkout and checked-in controls, independently of the donor.

| Maximum absolute donor difference, all three states | Release | Debug |
| --- | ---: | ---: |
| Label rate (m/s) | 4.43e−15 | 4.27e−15 |
| Age (s) | 3.56e−14 | 3.74e−14 |
| Cumulative slip (m) | 3.21e−17 | 3.39e−17 |
| Bulk velocity (m/s) | 2.61e−14 | 2.65e−14 |
| Pressure (Pa) | 9.88e−16 | 1.04e−15 |
| Physical DG stress (Pa) | 3.37e−15 | 3.45e−15 |

The homogeneous two-label check uses the original values and tolerances. For both cases in both builds, a fully solved 25 ms attempt was rejected and retried at 12.5 ms. Every sampled field and every label matched the clean trajectory exactly at all three output times. Restoring generation 02 at 12.5 ms and continuing to 25 ms also gave zero difference. Each harness rejected an unknown material schema and a checkpoint with the same label count but different positions. An additional homogeneous compatibility check encoded a generation-02 material record using the `88cd4eee1` Boost types and field order. Loading that v1 record reproduced the final v2 fields within 8.89e−16. This was a format migration test using a reconstructed record.

The one- and two-rank Release runs used global refinement 2, giving 16 cells and 144 sample points (72 per rank for the distributed run). Matching by cell ID and quadrature index gave at most 2.96e−15 in the bulk and 3.56e−15 in the label history across decompositions. All numerical runs used one thread per rank.

From the target root, use fresh replay destinations. The archive manifest and sibling identity are recorded below under increment 0.

```sh
D=/tmp/stationary-fault
R=$D/reference
mkdir -p "$D/stationary-source"
tar -xzf ../aspect-pf-rsf-reference-20261121/reference.tar.gz -C "$D"
(cd "$R" && sha256sum --quiet -c SHA256SUMS)
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 refactoring/phase-field-fault/replay_fixed_reference.py "$R" "$D/homogeneous"
git -C ../aspect-pf-rsf-particle-free archive 663937bed9305df2823ac77a3fc2c3925bb94caa | tar -xf - -C "$D/stationary-source"
cmake -S "$D/stationary-source" -B "$D/stationary-build" -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/opt/dealii/9.6-local -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DCMAKE_CXX_STANDARD_LIBRARIES=/opt/openmpi/5.0.6/lib/libmpi.so
cmake --build "$D/stationary-build" --target aspect -j3
python3 refactoring/phase-field-fault/replay_stationary_reference.py \
  "$D/stationary-source" "$D/stationary-build" "$D/homogeneous/input.prm" "$D/replay-final"
diff -qr benchmarks/fixed_fault/stationary-reference "$D/replay-final/comparison"

# Reuse the two native ASPECT builds configured below, or configure them with build_aspect.
for build in build-slice build-slice-debug; do
  cmake --build "$build" --target aspect -j2
  cmake -S tests -B "$build/tests" -DAspect_DIR="$PWD/$build" -DASPECT_BINARY="$PWD/$build/aspect" \
    -DASPECT_RUN_ALL_TESTS=ON -DASPECT_COMPARE_TEST_RESULTS=ON
  cmake --build "$build/tests" --target fixed_fault stationary_fault -j2
  "$build/aspect" --test
  ctest --test-dir "$build/tests" -R '^(fixed_fault|stationary_fault)$' --output-on-failure
  mode=release
  if [ "$build" = build-slice-debug ]; then mode=debug; fi
  for case_name in fixed_fault stationary_fault; do
    python3 benchmarks/fixed_fault/check_lifecycle.py "$build/aspect" "$build/tests/libfixed_fault.$mode.so" \
      "$D/final-$mode-$case_name" --case "$case_name"
  done
done
build-slice/aspect --validate benchmarks/fixed_fault/stationary.prm
ctest --test-dir build-slice/tests -R '^matrix_nonzeros_1$' --output-on-failure
```

For the rank comparison, both inputs include `tests/stationary_fault.prm`, set the Release probe library, override initial global refinement to 2 and write to their respective fresh destinations. Run `build-slice/aspect "$D/mpi-one/input.prm"` and `/opt/openmpi/5.0.6/bin/mpirun -np 2 build-slice/aspect "$D/mpi-two/input.prm"`. The measured donor executable had SHA256 `3e9d16949fc0d91c2304fada8c6fb0c49c573213d733efb682bd78c793b8179b`. The observing DSO used for the qualified run had SHA256 `738fa8a1e3a4407dd03c1f39e5e2c464d035f1573ff487fa9095c05070f3d7ce`.

**Resource use.** In the four-cell Release case the target stored 108 physical tensor coefficients in three static DGQ2 solution fields and 15 doubles of label positions and accepted physical state. It stores 108 additional rotated-parent coefficients while evaluating the mechanics. Each Picard iterate samples 256 owned quadrature points. Each section Newton iteration reduces four arrays of length proportional to the number of labels (the off-diagonal has one fewer entry). The endpoint stages three owned stress blocks. Bulk fields stay rank-local apart from ordinary ASPECT ghosts. The donor reported 216 owned tensor coefficients and six additional surface scalars. Single checkpointed runs, including startup and the respective observation plugins, measured 1.943 s and 203,836 KiB peak RSS on the target, 4.023 s and 211,572 KiB on the donor. The target performed 46 bulk Picard solves and the donor performed 13 condensed linear solves. Generation 02 occupied 7,601 bytes on the target and 5,600 bytes on the donor, including 701 and 1,782 bytes respectively in the compressed simulator record. ASPECT also checkpoints previous bulk solution vectors. These are single small-run measurements with different observers.

**Review at this revision.** The stationary comparison qualified the aligned constant-phase case. The resulting architecture repair is the active assignment above.

## Increment 0 · Baseline and source identity

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

## Increment 1 · Homogeneous accepted step (historical)

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

### Increment 1 architecture review (historical)

Read [the public header](../../include/aspect/material_model/fixed_fault.h), then [the equations, measure and commit](../../source/material_model/fixed_fault.cc). The review decision is whether the fixed material should retain the section reduction and accepted section state when geometry acquires independent labels. The Simulator currently owns the global vectors and the decision to accept. The material exposes a preparation call before assembly, a publication call and read-only accepted section access. The fixed case establishes homogeneous Stokes feedback and transaction lifetime. The next action is the author's review of this implementation at the end of increment 1.
