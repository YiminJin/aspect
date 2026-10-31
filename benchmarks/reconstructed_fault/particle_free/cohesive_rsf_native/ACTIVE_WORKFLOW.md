# Active fixed-mesh cohesive–RSF workflow

`active.prm` is the maintained, fresh-history starting point. It needs source,
ASPECT/deal.II and the compiled fixture plugin; it reads no prototype, kernel,
checkpoint, or archived evidence. Its four-step loading/hold interval is an API
smoke experiment, not another propagation qualification. Production/BP3 stays off.

Set `DEAL_II_DIR` to your MPI/Trilinos/p4est/zlib/SUNDIALS-enabled installation;
select a compatible compiler with `CXX` and its matching MPI launcher with
`MPIEXEC`. Build and launch directly (paths are examples relative to the source):

```sh
export SOURCE_ROOT="$PWD"
export ASPECT_BUILD="${ASPECT_BUILD:-$SOURCE_ROOT/build-particle-free/cohesive-rsf-native/native}"
export PLUGIN_BUILD="${PLUGIN_BUILD:-$SOURCE_ROOT/build-particle-free/cohesive-rsf-native/checks-native}"
cmake -S "$SOURCE_ROOT" -B "$ASPECT_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR="$DEAL_II_DIR" -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
  -DASPECT_UNITY_BUILD=ON -DASPECT_PRECOMPILE_HEADERS=ON -DASPECT_WITH_PYTHON=OFF
cmake --build "$ASPECT_BUILD" -j "${BUILD_JOBS:-2}"
cmake -S "$SOURCE_ROOT/benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native" \
  -B "$PLUGIN_BUILD" -DAspect_DIR="$ASPECT_BUILD"
cmake --build "$PLUGIN_BUILD" -j "${BUILD_JOBS:-2}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
# For oversubscribed local OpenMPI runs, match the bounded runner's setting.
export OMPI_MCA_mpi_yield_when_idle="${OMPI_MCA_mpi_yield_when_idle:-1}"
cd "$PLUGIN_BUILD"
"${MPIEXEC:-mpiexec}" -n 2 "$ASPECT_BUILD/aspect-release" \
  "$SOURCE_ROOT/benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/active.prm"
python3 "$SOURCE_ROOT/benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/analyze.py" \
  native-active --active --output-directory "$PLUGIN_BUILD/native-active-output" \
  --out "$PLUGIN_BUILD/native-active-analysis"
```

The input's plugin path is relative to this launch directory. Use a new output
directory for each experiment. Keep two ranks for the qualified restart route;
small ownership controls also cover one/four ranks. No speedup claim is implied.

| Input | Contract |
|---|---|
| Native `Box` extents, repetitions and `Box origin X/Y coordinate` | Physical rectangular domain and fixed mesh |
| `Seed x,y,x length,angle,cells` | Straight initial graph; length is its x projection; angle in degrees |
| `Initial state = prescribed fields` | Explicit fresh initialization at t=0; no imported file |
| `Initial independent tensor` | Three DGQ2 tensor functions Txx,Tyy,Txy in Pa, evaluated in physical x,y |
| `Initial native displacement memory` | Two displacement-memory functions in metres; native FE remainder retained in addition to T_DG |
| `Initial surface histories` | Six functions of normalized seed coordinate z: signed jump, a, normalized age, cohesive reference, relaxed jump, absolute slip |
| `Virgin normalized age` | Initial common age of unformed material; existing implicit aging/birth semantics unchanged |
| Loading `Schedule = function / Strain history` | Total gamma and angle(degrees) as functions of physical `time`; axial strain ratio specifies the retained affine strain basis |
| R, lifting width, spacing/reserve/epoch/refresh, completion budgets, formed-length limit | Separate physical, numerical and computational quantities; all explicit/configurable |

The initialized jump is retained as the existing Gauss offset. Relaxed jump must
vanish at the seed endpoints and be reproducible by the native piecewise Q1
trace; inconsistent data fail the existing 1e−13 m identity check. No history is
silently projected/reset. Intact records require zero cohesive elastic jump.
Native memory means `T_b=T_DG+2G dev eps(d_R)−Pi3[2G dev eps(d_R)]`; it is not a
second addition of its DG projection. Initial stress need not be relaxed by an
unrecorded solve. The input owner must prescribe a physically consistent state.
The example's C0 is explicitly 2Gc/peak² and its pre-existing seed energy is
initial inventory, not new formation work. Changing physical parameters requires
consistent initial histories; the program does not recalibrate them implicitly.

Boundary velocity is the finite difference of the prescribed displacement at
t and t−dt, including a loading-to-hold transition. Initial gamma is represented
by the supplied stress/history, not an impulsive displacement at t=0. The loading
schedule/axial ratio and prescribed setup are checked across native restart.
To extend only the physical endpoint, keep the histories and schedule unchanged.

Before sampling, assembly, admission and restart, the native owner checks the
**entire** disk centered R ahead of the formed tip with radius R, and the complete
vertical lifting strip of every formed/temporary piece. Conservative, outward
interval bounds include polynomial extrema between quadrature points. Actual
bulk-mesh bounds are used; this route supports complete rectangular Box meshes.
`sampling_clearance_m` and `lifting_clearance_m` are exported. Approach within
`Boundary warning distance` is logged to `cohesive-domain.csv`; loss of positive
certified clearance stops before publication. The accepted checkpoint remains
the recovery anchor. Raising the formed-length budget does not bypass this guard.
No sampling normalization on a truncated disk is allowed.
The fixed interval envelope is a conservative safety certificate, not an exact
boundary-contact location or a quadrature change. Formation length remains a
separate computational stop with the retained 7.5 m default.

Optional bounded tooling, without shell execution inside the native solve:

```sh
python3 "$SOURCE_ROOT/benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/qualify.py" \
  active --label smoke --aspect "$ASPECT_BUILD/aspect-release" \
  --plugin "$PLUGIN_BUILD/libcohesive_rsf_native_checks.release.so" --mpiexec "${MPIEXEC:-mpiexec}"
python3 "$SOURCE_ROOT/benchmarks/reconstructed_fault/particle_free/cohesive_rsf_native/analyze.py" \
  smoke --active
```

Runner overrides also include `--work-dir`, `--input`, `--cwd`, `--mpi-args` and
`--np-flag` (empty for a launcher that supplies rank count externally), or
`ASPECT_COHESIVE_WORK_DIR`, `ASPECT_BINARY`, `ASPECT_COHESIVE_PLUGIN`, `MPIEXEC`,
`MPIEXEC_PREFLAGS`, `MPIEXEC_NUMPROC_FLAG`. Analysis uses accepted native outputs
only and can read relocated output directories. NumPy/Matplotlib/Pillow are
analysis dependencies. `analyze.py` without `--active`, `verify.py`, and
`package.py` remain **archived-pilot regression tools**, not fresh-run prerequisites.
`qualify.py portable-check --label CHECK` runs the fixed short initialization,
restart, incompatible-configuration and boundary controls using these same path
overrides. `--regression-from RETAINED_OUTPUT` additionally checks the preserved
pilot checkpoints; it is optional and requires that original archive. The runner
checks the actual reported MPI rank count, rather than trusting the launcher.

The unchanged `pilot.prm`/`resume.prm`, retained loading mode, measured CSV/JSON and
original native anchors preserve the .440–.462375 s qualified regression. Its
optional replay needs the separately retained archive. This workflow does not
claim to recreate its preparation from zero time or resolve the retained
point/normal-traction, spatial, boundary and path-hypothesis limitations.
