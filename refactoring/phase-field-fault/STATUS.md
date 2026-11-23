# Phase-field fault working record

## Increment 4A · Accepted DG stress transport

**Ready for author review.** The [4A assignment](INCREMENT_4A_TASK.md) runs from local `pf-rsf` at `72c5a9c2d`. Both Release and Debug measured the transported stress in an ordinary Stokes assembly and published the independently predicted physical endpoint. A second calculation using the old receiving-cell stress failed at that assembly. The 3B phase-range diagnostic, immutable phase model, affine AT2 CTest and installed canonical CSV remain in the 3B commits. The corrected label CSV still has SHA-256 `98a5e5929b43b7877d8ee3c426c50758c3dc874289dd38ec6315f4022206f662`. The incoming assignment and local FE verification skill were preserved.

### From accepted coefficients to a new endpoint

1. `Simulator::start_timestep` calls `PhaseFieldFault::update` once per attempt. Initialization uses the specified Maxwell interval and zero physical transport time. With `Material model / Phase field fault / Advect physical stress = true`, `FaultStokesCoupling::prepare_parent` passes the role-resolved stress fields, ghosted old solution (including accepted velocity), prescribed exterior and physical interval to `Transport::advect_stress`. Its returned map holds locally owned DGQ2 coefficients in Pa. The three static stress compositions are the sole accepted store. The default preserves the stationary control.
2. For each Cartesian cell, scalar DGQ2 trial stress component $s$ and test $v$, the spatial action is

   $$
   (v,\dot s)_K=(s,u^n\cdot\nabla v+v\,\nabla\cdot u^n)_K
        -\langle v (u^n\cdot n_K)\widehat s\rangle_{\partial K}.
   $$

   Cell and face JxW measure m² and m, respectively. The mass-inverted action has units Pa/s. Q8 evaluates the accepted continuous FE velocity on both cells and faces. Each owned cell assembles its own row. Positive outward flux takes the local trace. Negative flux takes the upstream polynomial at the same physical point, or the exterior parent on a physical face. The material-form divergence term uses the mapped gradient of the accepted FE velocity.
3. The local inverse mass multiplies the self block, incoming neighbor blocks and prescribed-face loads once per attempt. The substep count is `max(1, ceil(dt * max_K sum_d(max|u_d| / h_d) / 0.1))`, using component maxima over cell **and face** points and an MPI maximum. The donor budget diagnoses more than 4,096 substeps. Every SSPRK3 stage imports the current tensor coefficients using `GridTools::exchange_cell_data_to_ghosts`. An Euler update is blended from the substep parent with coefficients 1, 1/4 and 2/3. The exterior function receives times $t_k$, $t_k+h$ and $t_k+h/2$ in seconds, positions in metres and returns xx, yy, xy in Pa. A quiescent velocity returns the original coefficients exactly.
4. `prepare_parent` interpolates the transported stress before rotating it with the accepted `grad(u) - slip_gradient`, relaxing by `exp(-G dt / eta)` and projecting onto the existing DG parent. `prepare_iteration → solve_rates → sample_bulk` resolves that parent in the section frame. `PhaseFieldFault::evaluate → stress_offset → FaultStressLoad::execute` adds its signed weak load during the Stokes solve. Both read the same attempt-local coefficients. The Picard convergence veto resamples the converged bulk fields. Each retry rebuilds from the old solution and accepted labels at the revised duration.
5. After the repeat decision, `post_accept_timestep → FaultStokesCoupling::accept_step` projects the complete viscous, slip and history response into the simulator solution, then `StationaryFault::accept` commits the label age, slip and rate. The FE transport operation has no fault geometry, friction, crack-law or observer dependency. The numerical load observer and synthetic transport fields live in the test plugin.

```text
include/aspect/transport/dg_stress.h
source/transport/dg_stress.cc
source/material_model/phase_field_fault.cc
source/reconstructed_fault/stokes_coupling.cc
source/simulator/assemblers/fault_stress.cc
source/simulator/solver_schemes.cc
source/simulator/core.cc
source/reconstructed_fault/stationary_fault.cc
```

### Donor and release assembly decisions

The sibling donor checkout currently points at the release. The recovered closeout archive in the sibling reference directory passed its SHA-256 check. The frozen mature base is `3928edaf7e6` plus `mature/entry.patch`. Its direct transport and rotation source and distributed tensor operator are Git blobs `69606797c5f1dc171075876f2158f1a8797917de` and `62dcef69abb8923d760be8ce6b78b1c58a29fca2`, also present in `pf-rsf-particle-free`. The mature patch leaves those two files intact. The archived input selects Q8, Jaumann DG material Lie, bulk-slip-subtracted spin and prescribed exterior stress (layered Maxwell provider). In `transport_material_history` the direct stress has zero transport source and zero in-transport relaxation. `TensorAdvection::assemble/advance` uses SSPRK3 with the 0.1 target, below its 0.125 ceiling. `rotate_material_history` consumes the transported primary tensor and the frozen accepted spin. The 4A fixture supplies its own analytic parent exterior.

| Release piece | Required action and implementation decision |
| --- | --- |
| `AdvectionField`, advection scratch and `AdvectionSystem::execute` | The release scalar assembly uses a `fem_field`, old scalar fields, current velocity, physical timestep and BDF mass terms. The stress needs an unscaled material residual for three fields at each explicit stage. Reuse the native composition role, FE, mapping, velocity extractor and mapped FEValues. Assemble the signed volume block locally. |
| `AdvectionSystemInteriorFace::execute` | Its face owner contributes timestep-scaled self and neighbor **matrix** blocks to the global system. The stress requires the neighbor's current SSPRK coefficients. Use the same upwind trace convention, physical normals and face JxW. Assemble only locally owned rows and import stage ghosts before applying the blocks. |
| `AdvectionSystemBoundaryFace::execute` | The scalar boundary manager supplies Dirichlet composition with the simulator clock. The stress has three exterior physical components on any locally incoming face, sampled at three SSPRK stage times. Reuse FEFaceValues and the parsed-function implementation. Evaluate the prescribed tensor only on inflow. |
| Local mass and `Simulator::copy_local_to_global_advection_system` | Release scatters the constrained timestep matrix and RHS. DGQ2 stress has disjoint cell coefficients. Use deal.II FullMatrix for the physical cell mass inverse and block products, and the release's deal.II cell-data exchange facility for the halo. This adds no global scalar solve, replicated bulk field, second DoFHandler or release assembler edit. |

The shared-assembler experiment `3a21d59cf` and revert `319b10b94` were inspected. Stage values, clock, boundary data and scatter would all still have required separate handling after extracting a scalar face sign.

### Measured checks

| Check | Discriminating result |
| --- | --- |
| Affine translation and material derivative | In the two-cell fixture at 25 ms, slopes differ for all three components. For +8 m/s the Release max nodal errors in xx, yy, xy were 5.55e−16, 7.77e−16 and 1.78e−15 Pa. For −8 m/s they were 6.66e−16, 1.33e−15 and 1.55e−15 Pa. Zero velocity gave three exact identities. For $u=(x,0)$, constant physical stress stayed constant to 8.88e−16 Pa despite nonzero divergence. |
| Independent face transfer | For $u=(8,0)$ the upstream stress jump is $(1,2,3)(1+y+y^2)$ Pa at $x=0.5$. Three single-substep receiver masses at 1, 2 and 3 ms recover the signed initial derivative exactly for a cubic RK update. Independent face Gauss integration gives +14.6666666667, +29.3333333333 and +44 Pa m²/s. Release errors were −5.33e−15, −1.07e−14 and −3.55e−14 Pa m²/s. A separate strong-form P2 lifting polynomial and cubic Taylor update checked every discontinuous nodal trace at 24 ms over **four** substeps. Maximum error was 7.55e−15 Pa. Each rank owns one of the two cells. All serial and MPI spatial coefficients matched exactly by CellId, component and node. |
| Ordinary Stokes load | The accepted bulk velocity is $u=(8,0.4x)$ m/s, the section is vertical and the phase is constant. The initial diagonal stresses have a common 0.3 Pa/m slope and a 0.6 Pa difference. In the right cell at 12.5 ms, apply the test displacement $w=(x,0)$ to the actual Q3 local vector contributed to the ordinary Stokes RHS. Slip has zero xx here. Release measured **−0.29364437004440330 Pa m²**, against **−0.29364437004440386** analytically. Disabling advection changed the actual vector action to **−0.30845803705181202**, against the old-parent prediction of −0.30845803705181207, and the same test failed during assembly. The missing translation accounts for 0.0148136670074 Pa m². |
| Rotation, section and accepted stress | Independent scalar bisection gives the initial label rate 0.108817998188 m/s. In this geometry spin is $(0.4+V^n)/2$. At 12.5 ms the positive rotation angle is 0.003180112489 rad, raising the rotated shear from 0.05 to 0.051907043321 Pa. After relaxation the spin signal is +0.001883353648 Pa. The solved endpoint rate is 0.385391084593 m/s and the accepted xy is 0.148825091104 Pa. The largest error over **all 54 accepted stress coefficients** was 1.62e−14 Pa in Release and 8.88e−15 Pa in Debug. Both labels passed independent age and slip checks. |
| Retry and MPI | The rejected 25 ms trial reached the receiving-cell assembly. A 12.5 ms retry and the next 12.5 ms interval produced **exactly** the clean stress coefficients, age, slip and rate. A signal subscriber counted one publication at each of 0, 12.5 and 25 ms per rank. Release serial vs two ranks differed by at most 4.67e−15 Pa in accepted stress and 7.36e−16 m/s in label rate. Debug also passed the partition, clock and rollback checks. |
| Stationary AT1 and section tangent | All 38 corrected labels matched exactly at all three clocks. All six serial label/bulk exports matched the 3B closeout byte for byte. The review's passively captured, globally reduced row 19 action was 0.00559276415135455 Pa m. At $h=10^{-5}$, the independent centered-difference defects over **all 38 rows** were 4.11e−14 and 4.13e−14 Pa m on one and two ranks. Enabling transport with zero exterior on the same fitted case gave an action of 0.00559276562991257 Pa m, defects 4.14e−14 and 4.14e−14 and a serial–MPI action difference of 1.54e−17 Pa m. The two ranks supplied 1,260 and 1,447 points to row 19. Independent dense solves recovered the next damped iterates within 4.17e−17 m/s across the four runs. The 25 ms solved label rates changed by up to 0.00128003793 m/s when advection was enabled. |
| Existing callers | Release passed six scoped CTests, including the fault-off matrix control. Debug passed five scoped CTests. Both unit suites passed 2,359 assertions in 51 cases. Both Release fault lifecycle suites passed. The three-label tensile cutback differed from its clean endpoint by at most 1.09e−14. |

The Cartesian affine and P2 face references have exact spatial moments at the chosen rules. The nodal budget of 2e−12 Pa allows rounding through the dense mass inverse and RK stages on O(1–5 Pa) stresses. The signed face budget of 2e−11 Pa m²/s includes division of mass differences by 1 ms. The assembled work budget of 2e−11 Pa m² and accepted stress budget of 2e−10 Pa include the Stokes solve and DG projection. The stale-cell work defect is over eight orders larger than the work budget. The tangent uses the existing relative budget of 2e−8 times the action infinity norm and a decreasing-perturbation sweep. Its independent residual takes sampled tractions and support identities as inputs. The direct Stokes and face checks supply independent evidence about the parent entering these mechanics.

In the uninstrumented one-rank 32² fitted case the base and transported runs both took 143 Stokes solves. Elapsed process times were 24.00 and 25.74 s, and peak RSS was 226,908 and 227,452 KiB. The timer attributed 0.0773 s to the two physical transports. This is one timing pair on the laptop. For nine DG nodes, each local tensor field has 27 doubles per cell. At most three RK fields coexist (two can carry halos), plus a scalar 9×9 self block and a 9×9 block per incoming neighbor per owned cell, and nine weights per incoming physical quadrature point. The phase, solver vector and full accepted solution are never cloned by transport.

### Reproduce 4A

The following commands run from the target root. Use a fresh results directory for a new run. Outputs cited above use `D=build-4a`, GNU 12.4, deal.II 9.6.2, Trilinos and Open MPI 5.0.6. Every solve used one thread per rank. The public skill, verification scripts, fixture and references have no runtime role in an installed fault simulation.

```sh
W=$PWD
D=build-4a
export OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1
export PATH=/opt/openmpi/5.0.6/bin:$PATH
mkdir -p "$D"
(cd ../aspect-pf-rsf-reference-20261121 && sha256sum -c reference.tar.gz.sha256)
tar -xOzf ../aspect-pf-rsf-reference-20261121/reference.tar.gz reference/mature/input.prm
git show 3928edaf7e6:source/material_model/phase_field_fault/stress_evolution.cc
git show 3928edaf7e6:source/material_model/history/tensor_advection_distributed.cc
git show --stat 3a21d59cf
git show --stat 319b10b94

for mode in Release Debug; do
  if [ "$mode" = Release ]; then
    build=build-slice
    t="$D/tests"
    variant=release
    select='^(dg_stress_transport|fixed_fault|stationary_fault|phase_field_model_without_reporter|postprocess/phase_field_surface_energy_at2|matrix_nonzeros_1)$'
  else
    build=build-slice-debug
    t="$D/debug-tests"
    variant=debug
    select='^(dg_stress_transport|fixed_fault|stationary_fault|phase_field_model_without_reporter|postprocess/phase_field_surface_energy_at2)$'
  fi
  cmake --build "$build" --target aspect -j2
  cmake -S tests -B "$t" -DAspect_DIR="$W/$build" -DASPECT_BINARY="$W/$build/aspect" \
    -DCMAKE_BUILD_TYPE="$mode" -DASPECT_RUN_ALL_TESTS=ON -DASPECT_COMPARE_TEST_RESULTS=ON
  cmake --build "$t" --target dg_stress_transport fixed_fault stationary_fault phase_field_model_without_reporter -j1
  ctest --test-dir "$t" -R "$select" --output-on-failure
  (cd "$build/unit_tests" && ../aspect --test)
  python3 benchmarks/fixed_fault/check_transport.py "$build/aspect" "$t" "$D/sheared-$variant" \
    --mpiexec /opt/openmpi/5.0.6/bin/mpirun
done
for fixture in fixed_fault stationary_fault; do
  python3 -O benchmarks/fixed_fault/check_lifecycle.py build-slice/aspect \
    "$D/tests/libfixed_fault.release.so" "$D/${fixture%_fault}-lifecycle" --case "$fixture"
done
```

The first sandbox MPI launch failed at the PMIx socket, and UCX permission warnings changed the release matrix control's screen diff. Both ran successfully with local socket access. Re-running the initial transport CTest in the same directory exposed an appended fixture summary. The final plugin opens a fresh publication log and summary per simulation. Repeated CTest runs exercise the corrected observer.

The retained tangent scripts rebuild the review binary in scratch and leave production sources untouched. The captured bands are taken inside `solve_rates` after MPI reduction and before elimination. The checker independently reconstructs the perturbed residual, including phase weights, Q1 interpolation and the direct asinh law. The former scratch scripts remain in `/tmp/pf-rate-tangent-pU2nnuH3`. The worktree copy retains the equations and adds optimized-Python checks for the residual, tangent and next Newton iterate.

```sh
r=benchmarks/fixed_fault/phase_ridge
v="$W/refactoring/phase-field-fault/rate_tangent"
cmake -S "$r" -B "$D/ridge" -DAspect_DIR="$W/build-slice" -DCMAKE_BUILD_TYPE=Release
cmake --build "$D/ridge" -j1
mkdir -p "$D/review"
(cd "$D/review" && python3 "$v/capture.py" "$W")
for ranks in 1 2; do
  suffix=
  if [ "$ranks" = 2 ]; then suffix=-mpi2; fi
  python3 "$r/run.py" "$D/review/aspect-review" "$D/ridge/libphase_ridge.release.so" \
    "$D/free-32$suffix" free 32 --ranks "$ranks"
done
PYTHONPATH="$r" python3 -c 'from pathlib import Path; import sys; from check import check_case,rank_check; root=Path(sys.argv[1]); check_case(root,"free",32); rank_check(root)' "$D"
python3 -O "$v/check_row.py" "$D/free-32" "$D/free-32-mpi2"
python3 - "$D" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
original = (root / 'free-32/input.prm').read_text()
for name in ('advected-32', 'advected-32-mpi2'):
    dest = root / name
    dest.mkdir()
    old = str(root / 'free-32/output')
    if original.count(old) != 1:
        raise ValueError('expected one inline output directory')
    (dest / 'input.prm').write_text(original.replace(old, str(dest/'output')) + """
subsection Material model
  subsection Phase field fault
    set Advect physical stress = true
    subsection Exterior parent stress
      set Function expression = 0;0;0
    end
  end
end
""")
PY
"$D/review/aspect-review" "$D/advected-32/input.prm" > "$D/advected-32/run.log"
mpirun -n 2 "$D/review/aspect-review" "$D/advected-32-mpi2/input.prm" > "$D/advected-32-mpi2/run.log"
python3 -O "$v/check_row.py" "$D/advected-32" "$D/advected-32-mpi2"
```

The timing pair used `build-slice/aspect` directly with the same serial parameter files, changing only the output paths to `cost-free-32/output` and `cost-advected-32/output`. Elapsed time and peak RSS came from the individual child's monotonic interval and `wait4().ru_maxrss`. The six accepted label/bulk files in each final uninstrumented run matched its captured run byte for byte.

The following reproduces the timing measurement with a fresh output directory for each case.

```sh
python3 - "$D" <<'PYTHON'
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
root = Path(sys.argv[1]).resolve()
for name in ('free-32', 'advected-32'):
    previous = root / name
    dest = root / ('cost-' + name)
    dest.mkdir()
    text = (previous/'input.prm').read_text()
    out = str(previous/'output')
    if text.count(out) != 1:
        raise ValueError(out)
    (dest/'input.prm').write_text(text.replace(out, str(dest/'output')))
    start = monotonic()
    with (dest/'run.log').open('w') as log:
        proc = subprocess.Popen(['build-slice/aspect', dest/'input.prm'],
                                stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        if proc.returncode:
            raise RuntimeError(proc.returncode)
    print(name, monotonic()-start, usage.ru_maxrss)
    for step in range(3):
        for table in ('bulk', 'labels'):
            filename = f'{table}-{step}-0.txt'
            if (dest/'output'/filename).read_bytes() != (previous/'output'/filename).read_bytes():
                raise ValueError(filename)
PYTHON
```

**Review boundary.** All stationary 4A acceptance checks passed. Supported production faces are conforming and nonperiodic on the stationary 2-D Cartesian box. The later layered moving-geometry calculation, phase transport and joint moving publications at 12.5 and 25 ms remain open under Increment 4 continued.

## Increment 3B review closeout

**Ready for author review.** The four findings against `e4338ddfe` are closed. The model ownership change and its reporter-disabled test are isolated in `d748dd201`. Collective phase validation, the AT2 CTest and installed reference lookup are in `9dd92e321`. The original 3B measurements follow this record.

| Finding | Discriminating result |
| --- | --- |
| Physical phase domain | The pre-fix installed AT1 run gave −0.06000000000000011 J/m for constant φ = −0.01. The repaired run failed before publishing energy, with finite quadrature extrema −0.010000000000000004 and −0.01. On two ranks the bilinear field 1.1xy/2 exceeds one only in the upper-right cell. The diagnostic counted **1 of 2 ranks** and reported **0.00087323949482400961 to 1.0388873235362321**. The Q3 point nearest the opposite corner supplies the minimum. Both extrema matched an independent calculation from the Gauss abscissa. Exact unity and ±4ε excursions about the relevant endpoints passed. The −4ε case measured −5.32907051820075 × 10⁻¹⁵ J/m, as expected when using the admitted field directly. |
| Configured model ownership | With the energy reporter disabled, an ordinary assembler-setup callback read the simulation's CZM model, resolved the named second composition `damage` at index 1, and obtained Gc = 3.5 J/m² and ℓ = 0.125 m. At φ = 0.25 its α, α′ and α″ were 0.4375, 1.5 and −2. The existing zero-composition disabled input also passed. |
| Affine AT2 CTest | The quick-test registration runs an ordinary parameter file. Its oracle reads **local, gradient and total by statistics column name** and compares at 14 significant digits. The analytic values are 7/3, 5/32 and 239/96 J/m. Release measured 2.3333333333333339, 0.15624999999999986 and 2.4895833333333339 J/m. Swapping only the first two columns in an output copy failed the comparator while preserving the total. |
| Installed free-32 checker | The examples installer ships one regular CSV beside the checker. The historical results path resolves to the same bytes. The installed file, the historical path and the original Git blob have SHA-256 `98a5e5929b43b7877d8ee3c426c50758c3dc874289dd38ec6315f4022206f662`. A fresh 32² free run and checker executed outside the checkout, with inputs and observer from the installation. All 38 IDs at 0, 12.5 and 25 ms matched. Maximum absolute differences for x, y, normal integral, age, slip and rate were all zero. The 38 label records and 9,216 bulk samples per clock were also byte-identical to the previous 3B output. |

### The instance available to an assembler

1. `Simulator` constructs its `Parameters<dim>` member before initializing plugins. The `Parameters` constructor initializes its **const, by-value** `surface_energy_model` member with the sole call to `PhaseField::SurfaceEnergyModel::parse_parameters(prm)`. The phase-side implementation declares and reads the existing `Postprocess / Phase field surface energy` keys, preserving one spelling for each setting. The model stores the named composition, Gc, ℓ and a const reference to the immutable crack law.
2. `SimulatorAccess::get_parameters()` returns a const reference to those same parameters. An assembler can bind `const auto &model = this->get_parameters().surface_energy_model;` during setup, call `model.validate()`, and resolve `model.composition` through introspection. Normal postprocessor initialization and execution use the same member. The simulator owns its lifetime on each rank, independently of the postprocessor list. The test exercises this access through the release `set_assemblers` signal. The increment adds no signal or simulator-aware handler. The diff against the reviewed base contains no change to `StationaryFault::fit`.
3. The reporter admits finite **quadrature values** in [−64ε, 1+64ε], with ε the double-precision machine epsilon. It integrates the sampled field and its mapped broken gradient directly. One batched deal.II collective carries the two integrals, extrema and a flag per rank. Every rank receives the range and number of ranks with a violation before the statistics write. The bound applies at the integration points.

```text
include/aspect/phase_field/surface_energy_model.h
source/phase_field/surface_energy_model.cc
include/aspect/parameters.h
source/simulator/parameters.cc
source/simulator/simulator_access.cc
source/postprocess/phase_field_surface_energy.cc
tests/phase_field_model_without_reporter.cc
tests/postprocess/phase_field_surface_energy_at2.prm
benchmarks/fixed_fault/phase_ridge/3a-repair-free-labels.csv
```

### Closeout verification

The following commands reproduce the checks from the worktree root. The recorded outside directory was `/tmp/pf-rsf-3b-closeout-pVlRDaM0`. Runtime, public headers, the top-level examples data install rule and the ridge observer's own examples install rule were exercised. The outside prefix contained no project working-record directory. Open MPI 5.0.6 used one thread per rank and eight owned cells per rank in the two-rank energy runs. Debug and Release each passed 2,359 unit assertions in 51 cases, the two new CTests, the homogeneous and stationary CTests, and both fault lifecycle checkers. Release also passed the upstream matrix control and the AT2 CTest in a fresh quick-suite configuration (one selected, one passed). All three affine laws passed in Release on one and two ranks and in Debug on one. Straight-crack, named-field, 3D-units and invalid-input controls passed on both builds.

```sh
W=$PWD
D=build-3b-closeout
mkdir -p "$D"
stage=$(mktemp -d /tmp/pf-rsf-3b-closeout-XXXXXXXX)
echo "$stage" > "$D/outside-dir"
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for mode in Release Debug; do
  build=build-slice
  variant=release
  if [ "$mode" = Debug ]; then
    build=build-slice-debug
    variant=debug
  fi
  if [ "$mode" = Release ]; then
    cmake --build "$build" --target aspect -j3
    cmake --build "$build" --target aspect -j1
  else
    cmake --build "$build" --target aspect -j1
    cmake --build "$build" --target aspect -j2
  fi
  (cd "$build/unit_tests" && ../aspect --test)
  t="$D/$variant-tests"
  cmake -S tests -B "$t" -DAspect_DIR="$W/$build" -DASPECT_BINARY="$W/$build/aspect" \
    -DCMAKE_BUILD_TYPE="$mode" -DASPECT_RUN_ALL_TESTS=ON -DASPECT_COMPARE_TEST_RESULTS=ON
  cmake --build "$t" --target phase_field_model_without_reporter fixed_fault stationary_fault -j1
  if [ "$mode" = Release ]; then
    ctest --test-dir "$t" -R '^(phase_field_model_without_reporter|postprocess/phase_field_surface_energy_at2|fixed_fault|stationary_fault|matrix_nonzeros_1)$' --output-on-failure
  else
    ctest --test-dir "$t" -R '^(phase_field_model_without_reporter|postprocess/phase_field_surface_energy_at2|fixed_fault|stationary_fault)$' --output-on-failure
  fi
  for fixture in fixed_fault stationary_fault; do
    python3 -O benchmarks/fixed_fault/check_lifecycle.py "$build/aspect" "$t/libfixed_fault.$variant.so" \
      "$D/lifecycle-$variant-$fixture" --case "$fixture"
  done
done
ctest --test-dir "$D/release-tests" -R '^matrix_nonzeros_1$' --output-on-failure
cmake -S tests -B "$D/quick-tests" -DAspect_DIR="$W/build-slice" -DASPECT_BINARY="$W/build-slice/aspect" \
  -DCMAKE_BUILD_TYPE=Release -DASPECT_RUN_ALL_TESTS=OFF -DASPECT_COMPARE_TEST_RESULTS=ON
ctest --test-dir "$D/quick-tests" -R '^postprocess/phase_field_surface_energy_at2$' --output-on-failure
cmake -S . -B "$D/examples" -DDEAL_II_DIR=/opt/dealii/9.6-local -DCMAKE_BUILD_TYPE=Release \
  -DASPECT_INSTALL_EXAMPLES=ON -DCMAKE_INSTALL_PREFIX="$stage/install"
cmake -DCMAKE_INSTALL_LOCAL_ONLY=ON -DCMAKE_INSTALL_COMPONENT=examples -P "$D/examples/cmake_install.cmake"
cmake --build "$D/examples" --target phase_ridge -j1
cmake --install "$D/examples/benchmarks/fixed_fault/phase_ridge" --component examples
cmake --install build-slice --prefix "$stage/install" --component runtime
cmake --install build-slice --prefix "$stage/install" --component includes
energy="$stage/install/benchmarks/phase_field_surface_energy/check.py"
python3 "$energy" "$stage/install/bin/aspect-release" "$D/serial-final"
python3 "$energy" "$stage/install/bin/aspect-release" "$D/mpi" --mpi /opt/openmpi/5.0.6/bin/mpirun
python3 -O benchmarks/phase_field_surface_energy/check.py build-slice-debug/aspect "$D/debug-serial"
(
  cd "$stage"
  export ASPECT_SOURCE_DIR="$stage/install"
  ridge=install/benchmarks/fixed_fault/phase_ridge
  python3 -I "$ridge/run.py" install/bin/aspect-release "$ridge/libphase_ridge.release.so" free-32 free 32
  python3 -I -c 'import sys; from pathlib import Path; sys.path.insert(0,str(Path("install/benchmarks/fixed_fault/phase_ridge").resolve())); from check import check_case; check_case(Path.cwd(),"free",32)'
)
ref=refactoring/phase-field-fault/results/3a-repair-free-labels.csv
sha256sum "$ref" "$stage/install/benchmarks/fixed_fault/phase_ridge/3a-repair-free-labels.csv"
git cat-file blob "e4338dd:$ref" | sha256sum
for step in 0 1 2; do
  for table in bulk labels; do
    cmp "$stage/free-32/output/$table-$step-0.txt" "build-3b/free-32/output/$table-$step-0.txt"
  done
done
```

The signed negative control and the swapped-column comparator can be replayed using the pre-closeout installation and the completed serial run, respectively. The column test compares an isolated altered row to the tracked analytic CTest output.

```sh
PYTHONPATH=benchmarks/phase_field_surface_energy python3 - "$D" <<'PY'
from pathlib import Path
import sys
from check import phase_input, run
old = Path("build-3b/install/bin/aspect-release").resolve()
s = run([old], Path(sys.argv[1]).resolve(), "old-negative", phase_input("-0.01"))
print(s["Crack surface energy total (J/m)"])
PY
mkdir -p "$stage/swapped-terms/output-phase_field_surface_energy_at2"
awk '/^#/ {print; next} {tmp=$9; $9=$10; $10=tmp; print}' \
  "$D/serial-final/AT2/output/statistics" > "$stage/swapped-terms/output-phase_field_surface_energy_at2/statistics"
(cd "$stage/swapped-terms" && "$W/tests/postprocess/phase_field_surface_energy_at2.sh" screen-output > screen-output)
diff -u tests/postprocess/phase_field_surface_energy_at2/screen-output "$stage/swapped-terms/screen-output"
# Expected exit status 1, with differences in local and gradient only.
```

The first Release closeout run caught a stale unity object compiled while the reporter was being edited. Recompiling that translation unit, relinking and reinstalling produced the measured rejection. The sandbox blocked the MPI launch and added socket warnings to the upstream matrix screen comparison. Both were rerun with local sockets available. The original failed logs are retained under the closeout build directory. All four findings meet their tests. Review stops at 3B. The previously recorded free-RSF donor gap and coupled AT2/CZM calibration boundary still apply.

## Increment 3B · Crack geometry and phase-only surface energy

**Original results at `e4338ddfe`.** The [3B assignment](INCREMENT_3B_TASK.md) starts from the completed 3A repair at `5bed31e63`. Commit `3c5b7ddc1` contains the constitutive interface and unit tests. Commit `2e7f0b1e5` contains the installed postprocessor, analytic fixtures and corrected-label regression.

### Measured energies

The phase-only inputs use ASPECT's simple material, zero fault objects, original elasticity disabled and a static DGQ2 composition. Under `Postprocess / Phase field surface energy`, the fixtures use `Phase composition = phase_field`, `Fracture energy = 2` J/m² and `Length scale = 0.25` m. The three affine inputs differ only in `Crack geometry` and output directory. The straight crack uses the AT1 default. The plugin defaults for fracture energy and length scale are 1 J/m² and 1 m.

On the 2 m × 1 m rectangle, φ = x/4 + y/2 has ∫|∇φ|² dx = 5/8 and ∫α dx = 1, 7/12, 17/12 m² for AT1, AT2, CZM. The aligned straight AT1 crack has two flanks, each of width 0.5 m, and a length of 1 m. All entries below are J/m of out-of-plane thickness.

| Case | Analytic (local, gradient, total) | Installed Release local | Gradient | Total |
| --- | --- | ---: | ---: | ---: |
| Affine AT1, 1 rank | 3, 15/128, 399/128 | 3 | 0.11718749999999989 | 3.1171875 |
| Affine AT2, 1 rank | 7/3, 5/32, 239/96 | 2.3333333333333339 | 0.15624999999999986 | 2.4895833333333339 |
| Affine CZM, 1 rank | 34/(3π), 5/(16π), 559/(48π) | 3.6075120434162926 | 0.09947183943243450 | 3.7069838828487272 |
| Affine AT1, 2 ranks | 3, 15/128, 399/128 | 3 | 0.11718750000000004 | 3.1171875 |
| Affine AT2, 2 ranks | 7/3, 5/32, 239/96 | 2.3333333333333330 | 0.15625000000000006 | 2.4895833333333330 |
| Affine CZM, 2 ranks | 34/(3π), 5/(16π), 559/(48π) | 3.6075120434162953 | 0.09947183943243462 | 3.7069838828487298 |
| Straight AT1, 1 rank | 1, 1, 2 | 0.99999999999999944 | 0.99999999999999944 | 1.9999999999999989 |

Release used one thread per rank. The one-rank affine mesh had 16 owned cells. The two-rank run had eight on each rank according to ASPECT's load-balance statistics. Every interior quadrature point has positive α and |∇φ|², so each rank contributes to both sums. Maximum affine absolute error across both layouts was 1.78 × 10⁻¹⁵ J/m. Maximum serial–parallel difference was 2.67 × 10⁻¹⁵ J/m. The checker uses relative tolerance (number of cells × 3ᵈ + 32)ε + 0.5 × 10⁻¹⁶ for Q3 accumulation, FE evaluation, the reduction and 16-place scientific output.

The same serial checker passed on Debug. Both builds passed a continuous Q2 second field named `damage` with a constant decoy in the first slot. A 3 m extrusion gave 8.999999999999963, 0.3515625000000066 and 9.351562499999970 J in Release, against 9, 45/128 and 1197/128 J. The feature-disabled zero-composition control passed. Selecting an unknown law, an absent field, zero fracture energy or a negative length each produced the expected diagnostic. The first disabled fixture omitted ASPECT's required gravity model and stopped at initialization. The corrected input specifies zero vertical gravity. The first MPI launch encountered the sandbox's PMIx socket restriction. The recorded two-rank results come from Open MPI 5.0.6 with local sockets enabled.

### Production call path and mathematical ownership

1. ASPECT's postprocessor manager creates the selected plugin and calls `PhaseFieldSurfaceEnergy::parse_parameters`. It resolves the named composition through introspection, validates positive finite Gc and ℓ and retains a reference through a pointer to the selected immutable law. The selector and its parameter grammar use the same table.
2. `PhaseField::CrackGeometry` owns α, α′, α″ and c₀. The selector maps AT1, AT2 and CZM to the quadratic α(φ) = φ[ξ + (1−ξ)φ] with ξ = 1, 0, 2. The normalization constants are 8/3, 2 and π. The source reference is the donor's `GeometricFunction` in Git object `abbc9b0c0`. The public law needs only its header and a selection name. A phase assembler can retain the same const reference and use its first and second derivatives for residual and tangent. Unit tests cover both endpoints, two interior points and the independent integral 4∫₀¹√α du.
3. `Simulator::postprocess` calls the release manager, which dispatches `PhaseFieldSurfaceEnergy::execute`. Like the release `CompositionStatistics` plugin, it reads the current solution on locally owned cells. The selected field's Gauss rule has p+1 points per coordinate on the box. The code accumulates α(φ) and the squared mapped gradient with physical JxW, reduces the two scalars together, and applies Gc/(c₀ℓ) and Gcℓ/c₀. It adds the two energies and their sum to the ordinary statistics table. For DG, the gradient is elementwise and the functional omits face jump and flux terms.
4. ASPECT owns the field and the statistics table. This diagnostic adds two quadrature arrays per rank, a cell-local FEValues object and a two-double reduction per evaluation. The phase law has no mesh or MPI state. The existing Simulator and fault classes acquire no fixture-specific hooks or state.

```text
include/aspect/phase_field/crack_geometry.h
source/phase_field/crack_geometry.cc
include/aspect/postprocess/phase_field_surface_energy.h
source/postprocess/phase_field_surface_energy.cc
benchmarks/phase_field_surface_energy/affine.prm
benchmarks/phase_field_surface_energy/straight_crack.prm
benchmarks/phase_field_surface_energy/check.py
```

### Mature control and reproduction

The 32² free 60° AT1 case began from the prescribed initial conditions and solved all 38 label rates. IDs and clocks were exact at 0, 12.5 and 25 ms. Every recorded coordinate, normal integral, age, slip and rate matched the corrected CSV exactly at the retained absolute/relative tolerances of 10⁻¹⁰/10⁻⁸. All 9,216 Q3 bulk records per clock matched the repaired archive by cell and quadrature ID with zero maximum difference. This includes the three accepted DG stress components. The independent Q2 stress-moment observer on 782 cut cells measured scaled errors 2.52440 × 10⁻⁹ and 1.51997 × 10⁻⁹ at the two physical endpoints. Both builds passed the homogeneous and three-label CTests, both lifecycle checkers and the full unit suite (2,359 assertions in 51 cases per build). The forced retry reproduced all three accepted snapshots exactly. At the fourth tensile-pulse endpoint the largest clean/cutback differences were 1.08 × 10⁻¹⁴ in Release and 8.85 × 10⁻¹⁵ in Debug. The stationary replay-isolation test passed.

Commands below run from the worktree root with a fresh destination. Builds used GCC 12.4, deal.II 9.6.2 and at most four local workers. The energy checks use an ordinary installed executable. Only the mature regression loads the existing 3A observer.

```sh
D=build-3b
mkdir -p "$D"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
for mode in Release Debug; do
  build=build-slice
  suffix=release
  if [ "$mode" = Debug ]; then
    build=build-slice-debug
    suffix=debug
  fi
  cmake --build "$build" --target aspect -j2
  (cd "$build/unit_tests" && ../aspect --test)
  tests="$D/${suffix}-tests"
  cmake -S tests -B "$tests" -DAspect_DIR="$PWD/$build" -DASPECT_BINARY="$PWD/$build/aspect" \
    -DCMAKE_BUILD_TYPE="$mode" -DASPECT_RUN_ALL_TESTS=ON -DASPECT_COMPARE_TEST_RESULTS=ON
  ctest --test-dir "$tests" -R '^(fixed_fault|stationary_fault)$' --output-on-failure
  for fixture in fixed_fault stationary_fault; do
    python3 -O benchmarks/fixed_fault/check_lifecycle.py "$build/aspect" "$tests/libfixed_fault.${suffix}.so" \
      "$D/lifecycle-${suffix}-$fixture" --case "$fixture"
  done
done
cmake --install build-slice --prefix "$PWD/$D/install" --component runtime
cmake --install build-slice --prefix "$PWD/$D/install" --component includes
python3 benchmarks/phase_field_surface_energy/check.py "$D/install/bin/aspect-release" "$D/installed"
python3 benchmarks/phase_field_surface_energy/check.py "$D/install/bin/aspect-release" "$D/mpi" --mpi /opt/openmpi/5.0.6/bin/mpirun
python3 -O benchmarks/phase_field_surface_energy/check.py build-slice-debug/aspect "$D/debug-final"
python3 refactoring/phase-field-fault/test_replay_stationary_reference.py -v
python3 benchmarks/fixed_fault/phase_ridge/run.py build-slice/aspect build-3a-repair/plugin/libphase_ridge.release.so "$D/free-32" free 32
PYTHONPATH=benchmarks/fixed_fault/phase_ridge python3 - "$D" <<'PY'
from pathlib import Path
import sys
import numpy as np
from check import bulk, check_case
from moments import check_moments
root = Path(sys.argv[1])
check_case(root, "free", 32)
for step in (1, 2):
    check_moments(root/"free-32", 32, step, .0125)
old = Path("build-3a-repair/qualification/free-32/output")
for step in range(3):
    a, b = [np.sort(bulk(output, step), order=["cell", "q"]) for output in (old, root/"free-32/output")]
    for key in a.dtype.names:
        if key in ("cell", "q", "phi"):
            np.testing.assert_array_equal(a[key], b[key])
        else:
            np.testing.assert_allclose(a[key], b[key], atol=1e-10, rtol=1e-8)
PY
```

All three crack geometries are qualified for the phase-only functional. Coupled coverage remains the mature AT1 case with its established degradation magnitude. The donor relation m = Gc α′(0)/(c₀ℓHc) gives zero for AT2 at positive finite Hc. The old mature localization therefore cannot be assumed for that calibration. The independent free-RSF donor trajectory and the original-elasticity comparison remain the recorded 3A baseline limitations. All 3B criteria are met. The next action is author review.

## Increment 3A repair · Stationary mature AT1 fault

**Author review.** The [assignment installed from the repair kit](INCREMENT_3A_REPAIR.md) starts at reviewed commit `707ce9a9a`. The historical 3A measurements and CSV below retain their original meaning. The corrected free solution has its own label history. The supported combination is the stationary mature AT1 distance profile, curvature-one energetic degradation with positive magnitude, a frozen generic DGQ2 composition, and finite in-box fault transfer. Increment 3B awaits review.

### Findings and discriminating tests

| Finding | Change and discriminating result |
| --- | --- |
| P1 · Fitted contacts | Each crossing is the maximum of the one-sided DGQ2 face trace in the seed window. Both crossings constrain the regularized fit. Only two-sided interior stations contribute centroids. Both along-face errors fell from approximately 0.078 m on the historical free curve to 0.004598 m on 32² and 0.0003832 m on 64². Moving either contact independently to x = ±0.48 m fails. Results for the four adjacent labels are below. |
| P1 · Fitted stress | Split the accepting DG projection on segment ends, support edges and overlapping-strip distance bisectors. A vertical-slice observer checks nine moments for each of three physical components. At 64² the displaced, imposed-rate fit has E = 9.44e−10. On precisely the same geometry and bulk fields the former unsplit Q8 would give E = 0.01914, exceeding the 0.01 budget. |
| P2 · Width truncation | Integrate discarded weight independently on both sides of the association strip. Constant phase with a 0.1 m half-width in a 1 m box rejects at segment 0, station 0 with omitted fraction 0.8. Full width passes. Both centered vertical profiles integrate to 0.005 m/s. The translated 32² and 64² transects give 0.004999986842 and 0.004999999942 m/s, consistent with the independently measured central Q1 normalization error. |
| P2 · Underresolved normal | Each owning-cell interval and each width edge is a quadrature break. Compare Q12 on a panel with its two halves and subdivide to a bounded depth. For the one-cell peak-0.99 quadratic at m = 1, the projected integral is 1534.6965325 m versus 755.6377468 m with unsplit Q12. Maximum error against Q256 is 2.64e−11 m. Exactly singular phase rejects setup. |
| P2 · Actual Stokes load | The prescribed fixture instruments the installed fault assembler on a knot-crossed cell at initialization, when the physical parent stress is zero. Its actual Q3 RHS dotted with the Q2 nodal values of v = (y+xy, 0.4x) agrees with independent signed contraction using the frozen FE phase. In the fine fitted case both give −0.616531092518. Reversing just the supplied stress and changing only the phase passed to material assembly to zero each trigger the signed-work assertion. Common-point Q8 adjointness and Q3–Q8 differences are reported separately. |
| P2 · Prescribed rate in convergence | The ordinary Picard preparation and veto call the section solver directly. The imposed fixture uses its configured initial label rate, registers the real assembler and accepted publication, and omits RSF solve/veto registration. Its row imbalance is labelled imposed. All 38 ordinary free rates solve the independent row balance. Switching the observed free material to the built-in material gives byte-identical label files at all three clocks. Both ordinary stationary inputs also run without a test library. |
| P2 · Collective getter | Removed the collective material getter. The free solver and the all-rank fixture probe call the coupling's documented collective sampling operation during the lifetime of the relaxed parent. Reading accepted labels is local. The probe traversed the same active Q8 samples used by the two-sided traction and independent residual checks. |
| P2 · Donor complement | Recover the full donor (CellId, q) keys from its Cartesian level-four coordinates, coarse Z-order and Q3 abscissae. Match all 36,864 identities and coordinates. At both compared clocks all 23,984 donor traction points match segment, station and imposed rate. The actual three-component slip stress is exactly zero at every one of the remaining 12,880 points. An injected 1 Pa shear load on one complement sample fails. The independent finite-strip admission check distinguishes associated zero-phase points (zero such points in this Q1 case). The matched cell-interval donor differs in χ by at most 2.37e−13 m⁻¹. The archived remote backend differs by 5.88e−6 m⁻¹. |
| P2 · Initial history | Independently assert 11.682941969615793 s and zero cumulative slip at all 38 labels at time zero. Advancing just one initial exported age by 0.01 s fails the check. The initialization Maxwell interval is 0.1 s. Compare the analytic fixed-rate age update and Δs = VΔt at 12.5 and 25 ms. One solved 25 ms attempt was rejected. All labels and all published bulk fields by cell/sample match the clean path exactly at all three clocks. |
| P3 · Distributed work | Each of two ranks contributed 22,365 positive-weight Q8 samples at each clock, summing to the serial 44,730. Setting one rank’s active count to zero fails with both full bulk shards still present. Both replicated label tables and all 9,216 Q3 points were compared by identity. The largest DG component difference was 4.26e−14 Pa. |
| P3 · Normalization storage | Assemble one diagonal and one off-diagonal and eliminate the consistent Q1 mass in O(n) work and storage on each rank. A single DGQ2 cell, unequal intervals of 0.3 and 0.7 m, and a phase varying in both directions give nodal values 1.95438975, 1.72938523 and 0.50769628 m. They agree with independent Q256 and a dense solve to 4.44e−16 m. Geometry still uses its separate second-difference fit. |

### Contact measurements

Both inclined seeds have analytic length 1.154700538379252 m. The displaced seed starts 0.015625 m off in x and 0.01353165 m off in the normal direction. Reported distances are in metres, with the near-end labels listed in material order (1, 2, 35, 36). The crossing coordinates come from the observed FE face. The remaining contact error measures the discrete phase maximum's displacement from the analytic ridge. A flat bulk phase keeps the supplied trace.

| Seed | Grid | Absolute x error, bottom / top | Near-end normal errors | Maximum interior | Length |
| --- | --- | --- | --- | ---: | ---: |
| Centered | 32² | .004598 / .004598 | .003104, .002269, .002269, .003104 | .003104 | 1.150194 |
| Centered | 64² | .0003832 / .0003832 | .0002555, .0001832, .0001832, .0002555 | .0002555 | 1.154318 |
| Displaced | 32² | .004598 / .004598 | .003064, .002193, .002303, .003123 | .003123 | 1.150197 |
| Displaced | 64² | .0003832 / .0003832 | .0002629, .0001968, .0001858, .0002570 | .0002629 | 1.154318 |

The independent Q32 ray integral at the *original* half-width measures the omitted fractions below. The measured historical matched-Q1 tail of 2.033332 parts per million sets the 2.1 ppm relative budget. The bounded quadratic interpolant can spread phase across a whole edge cell. Its benchmark inputs therefore extend the association width by one Cartesian cell diagonal. Every repaired DG run measured zero discarded weight at this wider width. The matched-Q1 run retains the donor's width and both in-box ray lengths. There is no extension past a physical face or finite end.

| Original-width control | 32² omitted (ppm) | 64² omitted (ppm) |
| --- | ---: | ---: |
| vertical | 0.2803 | 0.2106 |
| narrow | 0.3184 | 0.2803 |
| translated | 3.147 | 0.7657 |
| inclined | 4.726 | 0.7257 |
| inclined-fit | 4.897 | 0.6548 |
| inclined-q1 | — | 2.033 |
| free | 5.054 | 0.7252 |

### Accepted stress moments

The production rule clips the unit cell into convex regions of constant finite-strip association and triangulates them. Cells with no internal cut keep Q8. The observer instead slices vertically at intersections of the physical strip edges and equal-distance lines. It reconstructs the smooth viscous plus relaxed-parent polynomial from nine FE samples, then evaluates the rate and discontinuous slip tensor on each slice. Q6 and Q10 product rules on these slices provide an independent integration uncertainty estimate. E is the maximum absolute component/monomial defect divided by the maximum cell-area times sampled absolute reference stress on the same cells and clock. Both dimensional columns have units Pa m². The legacy observer column is the original corner-selected Q16 metric, recomputed on the repaired fields. The independent observer includes finite-end and support cuts even in cells whose corner segment IDs agree.

| Case, grid, physical step | Cells | Max defect / scale | Worst defect cell / scale cell | E | Reference uncertainty / scale | Unsplit Q8 E | Legacy Q16 E |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| inclined-32, 1 | 782 | 1.042e-08 / 4.796 | 15_3:221 / 0_3:110 | 2.17e-09 | 5.93e-12 | 0.02176 | 0.0002186 |
| inclined-64, 1 | 2032 | 2.862e-10 / 1.1242 | 15_4:2303 / 0_4:1011 | 2.55e-10 | 1.24e-13 | 0.01071 | 2.654e-06 |
| inclined-fit-32, 1 | 782 | 8.491e-09 / 4.1225 | 3_3:303 / 0_3:101 | 2.06e-09 | 5.45e-12 | 0.02142 | 0.006857 |
| inclined-fit-64, 1 | 2041 | 1.037e-09 / 1.0982 | 15_4:2223 / 15_4:2232 | 9.44e-10 | 1.22e-13 | 0.01914 | 0.005878 |
| inclined-q1-64, 1 | 1940 | 3.392e-10 / 1.1228 | 15_4:2303 / 0_4:1011 | 3.02e-10 | 3.53e-14 | 0.01071 | 2.614e-06 |
| free-32, 1 | 782 | 2.54e-13 / 0.00010063 | 15_3:221 / 15_3:223 | 2.52e-09 | 4.96e-12 | 0.008554 | 0.004962 |
| free-64, 1 | 2032 | 3.439e-15 / 2.7875e-05 | 15_4:2303 / 15_4:2232 | 1.23e-10 | 9.43e-14 | 0.01456 | 0.00346 |
| free-32, 2 | 782 | 2.769e-13 / 0.00018215 | 15_3:221 / 0_3:110 | 1.52e-09 | 2.51e-12 | 0.005169 | 0.002506 |
| free-64, 2 | 2032 | 1.817e-15 / 5.1701e-05 | 15_4:2303 / 0_4:1101 | 3.51e-11 | 4.9e-14 | 0.007466 | 0.001761 |

### One accepted step and review boundary

1. ASPECT owns the static generic DGQ2 coefficients and their accepted time levels. At the first material update, `StationaryFault::fit` reads the interior DG traces on the seed faces and the two-sided interior FE samples. `FaultPhaseProfile::initialize` uses the resulting segment frames, the same initialized FE field and the phase-side `excess_compliance` to assemble the full normal integral. The stable curvature-one expression is mφ(1+φ)/(1−φ)². The phase-side function admits the infinite limit at one. The fault enforces the narrower inverse-degradation domain [−1e−4,1) and maps a permitted negative undershoot to zero weight.
2. For the 12.5 ms attempt, the material evaluates κ and the coupling rotates and relaxes the accepted DG parent with the accepted slip removed from the bulk spin. Picard preparation samples Q8 pressure and strain, gathers Q1 balance against accepted label age, solves rates, and returns the negative slip stress offset through the material to the actual Stokes RHS. The converged iterate must also satisfy the coupling veto. Generic Stokes assembly, fixed-box pressure datum and the plane-strain correction retain their Increment 2 roles.
3. Only after the simulator's repeat decision does `accept_step` project endpoint stress into the three owned DGQ2 blocks and publish label rate, slip and age together. At the sampled central Q8 point in the 32² free run, the 12.5 ms accepted DG xx, yy and xy are 0.02163087, −0.02198014 and 0.03737990 Pa. Mechanical pressure is −0.02623878 Pa, friction compression 4.99541727 Pa, and driving traction including the slip response 3.67313684 Pa. The scaled independent section residual is 6.02e−15.
4. At 25 ms the 32² rates at labels 0, 19 and 37 are 0.07301323, 0.08554975 and 0.07116843 m/s. Turning off only the loading ramp leaves initialization identical and changes label 1's 25 ms rate from 0.10509809 to 0.12191597 m/s. The complete corrected 32² trajectory is retained alongside the original CSV. The historical donor prescribes V and disables fitting. It supplies independent common-input source and stress identities, but no independent time trajectory for free rates, DG stress, age or slip.

The rank-local attempt retains one relaxed-parent DG tensor per support point. Publication stages three owned DG blocks. Geometry, ages, slip, rates and O(n) normalizer rows are replicated. Face-maximum candidates are gathered once at initialization. The dense regularized geometry fit and its storage are separate from the tridiagonal normalizer. Cut polygons, quadrature and projection matrices live only for the current owned cell. Instrumented serial runs with one thread used 225,316 KiB / 79.8 s at 32² and 323,088 KiB / 161.1 s at 64², including diagnostics and concurrent local verification. The accepted projection adds no persistent per-cell cut cache.

The coupling's Picard and accept hooks all have production callers. The phase-side law has its own fault-independent unit test. The imposed material and collective observation adapter exist only in the benchmark. The release analogues are ASPECT's existing Stokes assembler interfaces and deal.II's cell-local projection from quadrature points. No Simulator hook was added.

```text
include/aspect/phase_field/mature_degradation.h
include/aspect/reconstructed_fault/stationary_fault.h
source/reconstructed_fault/stationary_fault.cc
source/reconstructed_fault/phase_profile.cc
source/material_model/phase_field_fault.cc
source/reconstructed_fault/stokes_coupling.cc
source/simulator/assemblers/fault_stress.cc
benchmarks/fixed_fault/phase_ridge/phase_ridge.cc
benchmarks/fixed_fault/phase_ridge/moments.py
refactoring/phase-field-fault/results/3a-repair-free-labels.csv
refactoring/phase-field-fault/results/3a-repair-transects.png
```

### Reproduction and preserved discrepancies

The archive contained the instruction patch and its standalone assignment. It was extracted under `/tmp/pf-rsf-3a-repair-wK3AAaA1`. `git apply --check` succeeded on the clean reviewed head before `git apply`. The instruction commit is `efd41ab1b`, the geometry and normalization commit is `15cbc8a88`, the coupling commit is `a91d55f63`, and the verification commit is `26ce90102`.

All new results are fresh starts under the ignored worktree directory below. The independent Q1 donor was built from a Git archive in that directory. The historical donor input and archived output were read without modification. The original free and displaced geometry, pre-fix Q8/Q16/Q32 comparisons, and original label CSV remain in the following historical section. The prior session's temporary run tree was absent on arrival.

```text
build-3a-repair/qualification-review.log
build-3a-repair/qualification
build-3a-repair/preflight
build-3a-repair/mutations
build-3a-repair/donor-cell
```

Both first corrected free initial solves reached the 80-iteration allowance while still decreasing the relative Stokes residual (4.08202e−10 and 4.54965e−10). Only the free fixture allowance was increased to 180. Stokes tolerance 1e−12, section and clock checks were retained. The first vertical-slice observer missed events where a cut entered the *horizontal cell edge*. Its Q6/Q10 difference on the coarse fit reached 1.66%. Including these events before ordering the vertical panels reduced the scaled uncertainty to 5.45e−12 on that control. The initial probe placement at the origin fell between knots on the fine displaced grid. Selecting the cell containing a particular interior material knot exercises an actual cut on both grids. The first sandbox MPI launcher had no usable local socket interface. The two-rank qualification used the local Open MPI launcher with socket access.

The comparison used one deal.II, OpenMP and BLAS thread per rank, one rank unless indicated, and up to four total local workers. Reproduce in a new empty evidence directory. The names below correspond to the completed runs.

```sh
D=$PWD/build-3a-repair
Q=$D/qualification
B=benchmarks/fixed_fault/phase_ridge
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
cmake --build build-slice --target aspect -j2
cmake -S "$B" -B "$D/plugin" -DAspect_DIR="$PWD/build-slice"
cmake --build "$D/plugin" -j1
cmake --build build-slice/tests --target fixed_fault stationary_fault -j1
so=$D/plugin/libphase_ridge.release.so
for c in vertical narrow translated inclined inclined-fit free; do
  for g in 32 64; do python3 "$B/run.py" build-slice/aspect "$so" "$Q/$c-$g" "$c" "$g"; done
done
python3 "$B/run.py" build-slice/aspect "$so" "$Q/inclined-q1-64" inclined-q1 64
PATH=/opt/openmpi/5.0.6/bin:$PATH python3 "$B/run.py" build-slice/aspect "$so" "$Q/free-32-mpi2" free 32 --ranks 2
python3 "$B/run.py" build-slice/aspect "$so" "$Q/free-32-retry" free 32 --retry-plugin build-slice/tests/libfixed_fault.release.so
python3 -O "$B/check_normalization.py" build-slice/aspect "$so" "$D/normalization-bounds"
archive=../aspect/benchmarks/reconstructed_fault/bp5/moment-inclined/native_history_reference/output-native_history_reference
python3 -u -O "$B/check.py" "$Q" --retry --donor "$archive" --donor-cell "$D/donor-cell/output" \
  --history "$D/free-labels.csv" --plot "$D/transects.png" > "$D/qualification-review.log"
python3 refactoring/phase-field-fault/test_replay_stationary_reference.py -v
```

To regenerate the cell-interval donor, use the historical build commands below with source revision `82e43c2669b92ac92a1c62d47f5dd79ac3b29c06`, the original input from `abbc9b0c0` and the `cell` override. The repaired comparison checks the unmodified archived remote results as well as this fresh interval replay. The known plane-strain convention shift at identical donor gradients remains 362.498 Pa, with a maximum common-convention normal defect of 4.61e−11 Pa. Neither pressure boundary data nor free rates are inferred from this substitution.

The signed RHS mutations used separate copies of the fixture. Immediately before its isolated `FaultStressLoad::execute` call, one negated every actual offset tensor. The other set only the scratch material-input phase values to zero and called `m.evaluate` again. Both copies were built against the same Release executable and run with `run.py … inclined-fit 32`; both aborted at the signed-work assertion. The normal-material comparison included the exact free-32 input, changed only the material name to `phase field fault` and the postprocessors to `fixed fault statistics`, and compared its three `fault-labels` files byte for byte. The steady-load sibling removed just the traction ramp.

### Retained stationary controls

Release and Debug each passed 2,318 unit assertions in 50 cases, including the phase-only law and a strip that spans 20% of a cell whose four corners all have the same segment ID. Each build passed both stationary integration CTests and both lifecycle harnesses. The original tolerances and clocks were used for the homogeneous and three-label donor files, physical DG output, EOS contrast, field reordering, RHS-only assembly, retry and pressure-pulse cutback. The replay substitution checks passed in normal and optimized Python. A Debug 32² inclined fitted, imposed-rate run also completed through physical publication.

The grid, phase, support, compression datum and thermal/elasticity settings for the two original native material inputs are unchanged. The free localized fixture keeps its original friction reference speed of 0.001 m/s. In the imposed diagnostic the fixture sets the initial label rate through the existing reference-speed parameter to 0.005 m/s. Its measured imbalance has no authority over convergence. On the fitted 64² run, common-point Q8 work differed by 1.79e−16 on a 0.008715 scale; the Q3–Q8 difference was −1.15e−6.

The deliberately ambiguous, laterally flat face maximum rejects fitting. An admissible peak of 0.999999999999 tests the bounded quadrature failure, including on two ranks with just one owned cell in total. Both the singular and unresolved paths terminate before the section solve.

The feature-disabled matrix statistics comparator passed with local MPI sockets. The original-elasticity numerical-timestep CTest still differs from its checked-in near-zero shear values under this installation's plain `diff`. A fresh local build of the pinned pristine 3.1.0 commit gave a byte-identical complete depth-average file to the repaired executable (SHA-256 `e203bf08c3b307f6fa248adf53765e15d82362172c72b870253be1f877f7d5b5`). The first shear is −0.0346938 Pa in both local binaries and 0.0967461 Pa in the upstream golden file. This reproduces the pre-repair baseline discrepancy recorded below. All eleven repair checks meet their fixture criteria. An independently solved free-RSF donor trajectory remains unavailable.

```sh
for mode in release debug; do
  build=build-slice; tests=build-slice/tests
  if [ "$mode" = debug ]; then build=build-slice-debug; tests=$D/debug-tests; fi
  cmake --build "$build" --target aspect -j1
  (cd "$build/unit_tests" && ../aspect --test)
  ctest --test-dir "$tests" -R '^(fixed_fault|stationary_fault)$' --output-on-failure
  for c in fixed_fault stationary_fault; do
    python3 -O benchmarks/fixed_fault/check_lifecycle.py "$build/aspect" "$tests/libfixed_fault.$mode.so" \
      "$D/replay-$mode-$c" --case "$c"
  done
done
ctest --test-dir "$D/off-tests" -R '^matrix_nonzeros_1$' --output-on-failure
ctest --test-dir build-slice/tests -R '^viscoelastic_numerical_elastic_time_step$' --output-on-failure
cmp "$D/pristine-numerical/output/depth_average.txt" build-slice/tests/output-viscoelastic_numerical_elastic_time_step/depth_average.txt
```

The Debug and feature-off test trees were configured from `tests` with the matching `Aspect_DIR`, executable and build type, `ASPECT_RUN_ALL_TESTS=ON` and `ASPECT_COMPARE_TEST_RESULTS=ON`. The pristine comparator used release commit `0d97d2d3311154ca3f01d801b18f3945dc8b8bad`, the same GCC 12.4.0, deal.II 9.6 and Open MPI 5.0.6, and the original parameter file with only an output-directory override. The implementation and this review stop at the repaired stationary model.

## Increment 3A · Stationary phase ridge

[Original assignment](INCREMENT_3A_TASK.md). **Stopped for architecture review.** Base `c27257269`, geometry `e4806f56f`, localized transfer `9e0e2d793`, admissibility and shared-vertex fixes `37269ee45`, fixtures and accepted data `dcfc854a5`. The ordinary Stokes/RSF path solved all 38 rates on the fitted trace. **The inclined fit reproduces the interior ridge more accurately than the boundary contacts. The fitted trace is 1.25372 m long against the 1.15470 m straight reference.** This comparison remains open.

### Geometry and the first discrepancy

The fixed 60° control has length 1.154700538379252 m, tangent (0.5, √3/2) and maximum label error normal to the line 2.78e−17 m on both grids. The displaced seeds translate by 0.015625 m in x. The initial normal error is 0.015625 m for the vertical line and 0.01353165 m for the incline. Post-fit errors below are absolute distances from the known line. Core means |y| ≤ 0.25 m. Interior includes every label except the two contacts.

| Displaced seed | Grid | Maximum core / interior error (m) | Bottom / top contact error (m) | Fitted length (m) |
|---|---:|---:|---:|---:|
| Vertical ridge at x = 0.07 | 32² | 8.00e−6 / 8.00e−6 | 8.00e−6 / 8.00e−6 | 1 |
| Vertical ridge at x = 0.07 | 64² | 5.78e−6 / 5.78e−6 | 5.78e−6 / 5.78e−6 | 1 |
| Inclined ridge | 32² | 0.002684 / 0.042562 | 0.066618 / 0.067879 | 1.253589 |
| Inclined ridge | 64² | 0.002684 / 0.042547 | 0.066591 / 0.067854 | 1.253546 |

The exact centered seed in the free run provides the clearest diagnosis. Before regularization, an independent sum of the donor fit's Q3 weights gives a mean signed normal offset of **+0.084377 m at seed label 0** and −0.0000175 m at label 19. The box clips the normal profile near its lower face. Even the exact line therefore has a nonzero right-hand side in the weighted least-squares fit. Reintersecting the fitted end segments with the original horizontal faces preserves y = ±0.5 m. The resulting x coordinates are ±0.366402 m, versus ±0.288675 m for the reference. The cropped-end bias persists under bulk refinement and precedes normalization, friction and the Stokes solve. The pinned donor's inclined control explicitly disables fitting. Retain this result for the author's decision on the contact fit.

### One accepted step in source order

1. `Simulator::start_timestep → PhaseFieldFault::update` reads the static DGQ2 `phase_field` from the accepted FE solution. On first entry `StationaryFault::fit` assembles the donor normal-offset fit with cell weights JxW max(φ−0.1,0), Q1 hats, and the unit second-difference penalty. Constant phase retains the seed. Permanent ID is the index in the ordered label vector. A single owner holds position, age (s), cumulative slip (m) and rate (m/s). One segment supplies both consumers with arclength, tangent, normal and projection. The two production consumers are `FaultPhaseProfile` for normal integrals and transfer, and `FaultStokesCoupling` for traction in the local frame and accepted history.
2. `FaultPhaseProfile::initialize` evaluates the owning cell's DG shape functions on physical box-clipped normals. It integrates h = mφ(1+φ)/(1−φ)² with 12 points per cell interval, then consistently projects three Gauss station values per segment onto the Q1 labels. The finite-segment normal foot and profile half width determine mechanical admission. The legacy donor uses stationary-profile support and both available physical ray lengths. At a contact it integrates the in-box portion and supplies no exterior extension. Zero-weight samples carry zero slip. Active samples require a finite positive interpolated integral. A ray coincident with a cell face takes the higher-coordinate DG trace, except at the upper box boundary.
3. `prepare_parent` evaluates and projects the rotated/relaxed accepted stress with spin from the accepted velocity gradient minus the accepted slip gradient. At each Picard preparation, `sample_bulk → BulkPoint::add_balance → solve_rates` computes the Q1 rows and the tridiagonal fixed-age tangent. In each row one χ = h/I enters the integration weight and a second χ enters κχV in the local traction. The material returns −2κ sym(χ V t⊗n) plus the relaxed parent to the fault-owned weak stress load. Generic ASPECT assembles velocity and pressure. The constitutive rate closure is virtual so the diagnostic material can supply 0.005 m/s in the same Stokes assembly. The production material always solves the RSF balance. 3A adds no Simulator hook.
4. The coupled convergence veto checks the updated bulk iterate. After the repeat decision, `post_accept_timestep → accept_step` evaluates the full physical stress at 8×8 points, L²-projects it into the three owned DGQ2 blocks, and calls `StationaryFault::accept`. The xy block receives the xy component of the projected tensor. The label update uses Δs = VΔt and exact constant-rate aging. Initialization commits rate at zero physical duration. One fully solved 25 ms attempt was discarded and repeated at 12.5 ms. Every exported label and bulk value matched the clean free trajectory exactly at all three accepted clocks.

Representative interface, geometry, normal integral, bidirectional transfer and commit, in the refactor worktree:

```text
include/aspect/reconstructed_fault/stationary_fault.h
include/aspect/reconstructed_fault/phase_profile.h
source/material_model/phase_field_fault.cc
source/reconstructed_fault/stationary_fault.cc
source/reconstructed_fault/phase_profile.cc
source/reconstructed_fault/stokes_coupling.cc
source/simulator/assemblers/fault_stress.cc
```

### Independent bulk and section checks

The fixture initializes the donor's tabulated AT1 distance profile at peak 0.6. The first direct nodal DGQ2 interpolant on 32² cells reached −1.99e−4 on the incline and −2.51e−4 after vertical translation. Both exceed the donor's empirical −1e−4 undershoot bound. The final initializer converts the local quadratic nodal values to Bernstein controls and bounds those controls to [0,0.6] before writing ASPECT's DGQ2 support values. Only the initializer knows the analytic ridge. Production checks sampled phase collectively and diagnoses an inadmissible negative field. Prior unbounded runs remain under the evidence root. Exported DG phase was identical at all physical times and across retry.

| 64² central transect | Peak returned engineering slip shear rate (s⁻¹) | Physical integral (m/s) | Independent integral (m/s) |
|---|---:|---:|---:|
| Vertical, ℓ = 0.15625 m | 0.03534196 | 0.004999998685 | 0.004999998947 |
| Vertical, ℓ = 0.078125 m | 0.07068562 | 0.004999998927 | 0.004999998599 |
| Translated, ℓ = 0.15625 m | 0.03509342 | 0.004999996474 | 0.004999996171 |

The independent checker reconstructs each cell polynomial from nine interior FE samples, splits every ray at all Cartesian faces, and uses 32-point Gauss integration with a separate dense Q1 mass solve. Across the controls the maximum nodal I difference is 2.14e−13 m at scales up to about 109 m. At 64² on the fixed incline the central raw ray integral and its interpolated Q1 value are 108.65228 and 108.62350 m. At the first, boundary-truncated Gauss station they are 58.83819 and 59.63594 m. These differences measure the station projection. The direct transect uses 24 points per cell and can straddle the compact support edge. Its tolerance is the independent edge-panel weight bound plus 1e−13 m/s. The plot uses the actual upper DG trace on y = 0.

For the fixed incline at 64² the two common-Q8 work contractions differ by 1.25e−16 on a scale of 0.008740. They use a nonconstant Q1 label vector, a nonconstant virtual traction, the actual section scatter and the returned bulk slip gradient. The work bound is 256 machine epsilons times its magnitude. Sampling that bulk work with the ordinary Q3 Stokes rule gives Q3−Q8 differences of −7.73e−6 and −1.43e−6 on 32² and 64². The assembled positive section tangent differs from a centered residual difference by 6.91e−7 on a scale of 97.87 at 64². The fixed-parent directional probe uses a 1e−5 perturbation and a relative bound of 2e−8, allowing cancellation in the two assembled residuals.

The moment probe integrates the three accepted DG components minus the raw endpoint stress against all nine cell monomials using a separate 16×16 rule. On the straight prescribed incline the largest absolute moment on knot-crossed cells drops from 0.006577 Pa m² (scale 3.922, 496 cells) to 0.00002253 Pa m² (scale 0.8520, 1,288 cells). On the displaced fitted incline the maximum drops from 0.2018 to 0.05724 Pa m², while the maximum cell-area times stress scale drops from 6.440 to 1.468 Pa m². The worst scaled discrepancy therefore remains about 4% at 64². A second observer build used a 32×32 moment rule with the same production Q8 update. It measured 0.006283 and 0.00002193 Pa m² on the straight incline, and 0.2769 and 0.06249 Pa m² on the fitted incline. The fine fitted value is 4.25% of its 1.4714 Pa m² stress scale. This second rule confirms the local discrepancy and the decrease in absolute moment with refinement. Neither quadrature comparison bounds the pointwise stress error on a discontinuous segment boundary.

### Donor comparison

The pinned `abbc9b0c0` builds. Its supplied inclined input aborts during parsing because the four particle compositions omit the now-required DG Bernstein phase field. Adding that field in a separate diagnostic run reaches the explicit rejection of retired particle/CPDI mechanics. The original parameter file SHA-256 is `65da7510a975238d6990445542a8a580065a0279b522df58b4c8fcb499c65f53`, identical to the archived successful input. The archive identifies a historical executable with working changes, reporting `33228369d`. A separately built compatible revision `82e43c2669b92ac92a1c62d47f5dd79ac3b29c06` reproduced the four 0.1 s steps. Across all 36,864 bulk points and five clocks, coordinates and weights matched exactly, maximum velocity difference was 2.26e−17 m/s and maximum physical stress difference was 1.59e−10 Pa. `CleanStressCycle::initialize` prescribes all 38 rates in its temperature `post_advection_solver` callback. Its phase constraints initialize the continuous Q1 profile at support points.

The target prescribed incline matches the box, grid, 38 vertices, support, analytic profile, clock, rate, modulus, viscosity and corresponding prescribed velocity components. It opens the bottom normal component to establish a traction datum and adds 50,000 Pa compression in the fixture. The archived donor prescribes every velocity, normalizes mechanical pressure to volume mean zero, and uses 1,000 Pa adiabatic pressure for strength. The comparison below holds the donor's actual velocity gradient, incoming tensor and mechanical pressure fixed and substitutes the measured target source at matched cell IDs and Q3 indices. The two independently solved pressure fields have different boundary data.

* With the new bounded quadratic phase, the maximum / RMS χ differences on the 23,984 donor-associated points are 0.08934 / 0.01051 m⁻¹ on a peak of 13.3048 m⁻¹. The maximum corresponding shear difference is 44.67 Pa at κ ≈ 100,000 Pa s. On the same historical Q1 phase embedded in DGQ2, these become 5.88e−6 / 5.58e−7 m⁻¹ and 0.00294 Pa. A second donor replay changing only remote-point integration to its cell-interval backend gives the same 5.88e−6 m⁻¹ spread. Against that backend the matched Q1 target differs by just 2.38e−13 m⁻¹. Both the original and matched-backend differences are retained.
* The historical `MomentCycle` diagnostic passes the full symmetric strain directly to its Maxwell stress calculator. The target retains the increment-2 plane-strain deviator. At identical donor gradients this explains a maximum normal-traction difference of 362.498 Pa, equal pointwise to 2κ div(u)/3. Accounting for that term leaves at most 4.61e−11 Pa. The shear contraction has no trace term. The common-input normal and shear identities use a 1e−8 Pa absolute bound. The pinned input supplies no matched free-RSF trajectory.

### Accepted free-RSF states

The applied right-face traction is 0.02 + t + 0.12y Pa in the centered box. Every accepted label, coordinate and normalization is in the CSV below. At 25 ms labels 0, 19 and 37 have rates 0.0744834, 0.0814777 and 0.0716577 m/s. The columns below give the range over all 38 labels and measured values at one actual section Q8 point, x = y = 0.000620470992226 m. DG stress carries the accepted projection. Friction uses the pointwise tensor prior to that projection with σ = 5 + p − n·Tn and τ = 3.71061058025 + t·Tn in Pa.

| Time (ms) | V (m/s) | Age (s) | Slip (m) | DG xx, yy, xy (Pa) | p / σ / τ (Pa) | Independent scaled section residual |
|---:|---|---|---|---|---|---:|
| 0 | 0.03277–0.12319 | 11.68294 | 0 | 0, 0, 0 | −0.085007 / 4.910587 / 3.582045 | 1.31e−16 |
| 12.5 | 0.07953–0.10042 | 9.99799–10.32952 | 0.00099410–0.00125523 | 0.024872, −0.025205, 0.038711 | −0.023337 / 4.997602 / 3.669536 | 8.00e−17 |
| 25 | 0.07166–0.10479 | 8.49954–9.24719 | 0.00188982–0.00256506 | 0.039591, −0.040232, 0.056899 | −0.033363 / 4.995913 / 3.647508 | 4.36e−16 |

A paired run removed only the time-dependent right-face traction term. All initialization rates matched exactly. At label 26, restoring the ramp changed the 12.5 ms rate from 0.0927200 to 0.0869955 m/s and the 25 ms rate from 0.0937409 to 0.0823767 m/s. This isolates a loading response with identical initial geometry, phase and history.

Both ranks contributed 20,850 active Q8 points and exported 4,608 Q3 points each per endpoint. Across all clocks the largest one-/two-rank differences were 3.74e−15 m/s in label rate, 1.25e−14 s in age, 9.76e−18 m in slip, and 6.44e−15 Pa in accepted DG components. Label IDs, cell IDs, sample indices and phase matched. At every active free Q8 point, independent normal and shear contractions agreed with the section loads to 1.78e−15 Pa. Per-label age and slip use independent time-update checks with absolute bounds 2e−13 s and 2e−16 m. The independent section residual bound is 128 machine epsilons.

The instrumented 32² one-rank free run used 226,764 KiB peak RSS and 20.450 s elapsed including startup and diagnostic output. It has 50,628 global DoFs, including 9,216 per DG component. The attempt stores 27 physical relaxed-parent coefficients per owned cell and three owned DG blocks during publication. The 41,700 active section samples are partitioned. Each rank replicates 38 label records, 38 normal integrals and trial rates. Initialization temporarily assembles 38² fit/mass entries and 111 ray integrals. The section solve reduces four linear-sized row arrays per Newton iteration. No full bulk candidate or disk checkpoint is needed.

```text
refactoring/phase-field-fault/results/3a-free-labels.csv
refactoring/phase-field-fault/results/3a-transects.png
benchmarks/fixed_fault/phase_ridge/phase_ridge.cc
benchmarks/fixed_fault/phase_ridge/run.py
benchmarks/fixed_fault/phase_ridge/check.py
```

### Reproduction and retained evidence

Actual scratch root was `/tmp/fault-3a.1B1UmQr7`. Final runs are in `qualification`, with the complete per-rank fields, metrics and inputs. `final-evidence.log` contains the numerical report. `donor-native` and `donor-native-dg` retain the two pinned-revision diagnostics. The initial unbounded runs and the inactive shared-vertex MPI tie are preserved separately. Two neighboring finite segments could name the same nearest clamped vertex differently by rounding when neither admitted the point. Using its stored coordinates on both segments resolved this. The final comparison includes every exported field, including the foot at inactive points.

All solvers used one OpenMP, BLAS and deal.II thread per rank, and at most four total local workers. Build the existing fixed-fault test DSO using the Release regression commands below before the new retry run. Use fresh destinations.

```sh
D=$(mktemp -d /tmp/fault-3a.XXXXXXXX)
Q=$D/qualification
B=benchmarks/fixed_fault/phase_ridge
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
cmake --build build-slice --target aspect -j2
cmake -S "$B" -B "$D/plugin" -DAspect_DIR="$PWD/build-slice"
cmake --build "$D/plugin" -j1
so=$D/plugin/libphase_ridge.release.so
for c in vertical narrow translated inclined inclined-fit; do
  for g in 32 64; do python3 "$B/run.py" build-slice/aspect "$so" "$Q/$c-$g" "$c" "$g"; done
done
python3 "$B/run.py" build-slice/aspect "$so" "$Q/inclined-q1-64" inclined-q1 64
python3 "$B/run.py" build-slice/aspect "$so" "$Q/free-32" free 32
PATH=/opt/openmpi/5.0.6/bin:$PATH python3 "$B/run.py" build-slice/aspect "$so" "$Q/free-32-mpi2" free 32 --ranks 2
python3 "$B/run.py" build-slice/aspect "$so" "$Q/free-32-retry" free 32 --retry-plugin "$D/tests-release/libfixed_fault.release.so"
python3 -O "$B/check.py" "$Q" --retry --plot "$D/transects.png" --history "$D/labels.csv"
r=$Q/free-32-steady-load
mkdir "$r"
cat > "$r/input.prm" <<PRM
include $Q/free-32/input.prm
set Output directory = $r/output
subsection Boundary traction model
  subsection Function
    set Function expression = 0;0.02+0.12*y
  end
end
PRM
build-slice/aspect "$r/input.prm" > "$r/run.log" 2>&1
```

To reproduce the donor comparison, extract each revision to a fresh source tree, leaving all sibling worktrees untouched. In the following commands `D` is the same scratch root. The archived output is optional for normal target tests.

```sh
mkdir "$D/donor" "$D/q1-source"
git archive abbc9b0c0574c9f4de7a64d9662d539b93599f2b | tar -xf - -C "$D/donor"
git archive 82e43c2669b92ac92a1c62d47f5dd79ac3b29c06 | tar -xf - -C "$D/q1-source"
cmake -S "$D/q1-source" -B "$D/q1-build" -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR=/opt/dealii/9.6-local -DCMAKE_CXX_COMPILER=/opt/gcc/12.4.0/bin/g++ \
  -DCMAKE_CXX_STANDARD_LIBRARIES=/opt/openmpi/5.0.6/lib/libmpi.so \
  -DASPECT_WITH_VORO=ON -DVORO_DIR=/home/ein/local/voro++/0.4.6
cmake --build "$D/q1-build" --target aspect -j2
bp=benchmarks/reconstructed_fault/bp5
cmake -S "$D/q1-source/$bp" -B "$D/q1-plugin" -DAspect_DIR="$D/q1-build"
cmake --build "$D/q1-plugin" --target bp5_moment_cycle -j1
. "$D/q1-source/$bp/stress_cycle_env.sh"
export ASPECT_FAULT_COMPATIBILITY_DIAGNOSTIC=1
for mode in remote-voro cell; do
  r=$D/q1-$mode
  mkdir "$r"
  cat > "$r/input.prm" <<PRM
include $D/donor/$bp/inclined_native_history_reference.prm
set Additional shared libraries = $D/q1-plugin/libbp5_moment_cycle.release.so
set Output directory = $r/output
PRM
  if [ "$mode" = cell ]; then
    cat >> "$r/input.prm" <<'PRM'
subsection Material model
  subsection Phase field fault
    set I h integration backend = cell intervals
  end
end
PRM
  fi
  (cd "$r" && "$D/q1-build/aspect" input.prm > run.log 2>&1)
done
archive=../aspect/$bp/moment-inclined/native_history_reference/output-native_history_reference
python3 -O "$B/check.py" "$Q" --retry --donor "$archive" --donor-cell "$D/q1-cell/output"
```

For the independent Q32 moment audit, copy the fixture CMake file and source into a fresh directory. The sole source delta changes the observer's moment rule and its matching order checks. Use the same target executable and unchanged inputs.

```sh
mkdir "$D/moments32"
cp "$B/CMakeLists.txt" "$D/moments32/"
sed -e 's/16u/32u/g' -e 's/order==16/order==32/g' -e 's/order!=16/order!=32/g' "$B/phase_ridge.cc" > "$D/moments32/phase_ridge.cc"
cmake -S "$D/moments32" -B "$D/moments32-build" -DAspect_DIR="$PWD/build-slice"
cmake --build "$D/moments32-build" -j1
for c in inclined inclined-fit; do
  for g in 32 64; do python3 "$B/run.py" build-slice/aspect "$D/moments32-build/libphase_ridge.release.so" "$D/q32/$c-$g" "$c" "$g"; done
done
```

The original homogeneous and three-label lifecycle harnesses, both scoped CTests and the complete unit suite passed in Debug and Release (2,311 assertions, 48 cases per build). The separate frozen three-label snapshot comparison had maximum absolute errors 5.69e−14 in Release and 1.14e−13 in Debug. Both original forced repeats matched every output exactly. The tensile predictor retry matched its clean trajectory within 9.44e−16. The Python replay-isolation regression passed in both Python modes, both original parameter files validated, and the Release feature-off matrix regression passed. A Debug 32² vertical run also completed. No original clock, tolerance or reference file changed.

```sh
for mode in debug release; do
  build=build-slice; config=Release
  if [ "$mode" = debug ]; then build=build-slice-debug; config=Debug; fi
  cmake --build "$build" --target aspect -j1
  cmake -S tests -B "$D/tests-$mode" -DAspect_DIR="$PWD/$build" -DASPECT_BINARY="$PWD/$build/aspect" \
    -DCMAKE_BUILD_TYPE="$config" -DASPECT_RUN_ALL_TESTS=ON -DASPECT_COMPARE_TEST_RESULTS=ON
  cmake --build "$D/tests-$mode" --target fixed_fault stationary_fault -j1
  for c in fixed_fault stationary_fault; do
    python3 -O benchmarks/fixed_fault/check_lifecycle.py "$build/aspect" "$D/tests-$mode/libfixed_fault.$mode.so" "$Q/$c-$mode" --case "$c"
  done
  python3 -O benchmarks/fixed_fault/compare_snapshots.py benchmarks/fixed_fault/stationary-reference "$Q/stationary_fault-$mode/clean/output"
  ctest --test-dir "$D/tests-$mode" -R '^(fixed_fault|stationary_fault)$' --output-on-failure
  (cd "$build/unit_tests" && ../aspect --test)
done
python3 refactoring/phase-field-fault/test_replay_stationary_reference.py -v
build-slice/aspect --validate benchmarks/fixed_fault/fixed_fault.prm
build-slice/aspect --validate benchmarks/fixed_fault/stationary.prm
ctest --test-dir "$D/tests-release" -R '^matrix_nonzeros_1$' --output-on-failure
```

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
