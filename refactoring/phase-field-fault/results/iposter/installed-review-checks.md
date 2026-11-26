# Installed inputs and SDK verification

The destination prefix in this check was `build-d-review/install`.
The Release core was already compiled from the matching source. We configured
its CMake build for that destination and installed the runtime, data, headers,
CMake package and examples directly. The layered provider was compiled anew
against the installed SDK. A normal user can substitute a clean core build
configured against their own deal.II installation and build the Release
executable before installation.

~~~bash
W=$PWD
D=$W/build-d-review
I=$D/install
B=$W/build-6b/release
cmake -S "$W" -B "$B" -DCMAKE_INSTALL_PREFIX="$I" -DASPECT_INSTALL_EXAMPLES=ON
cmake -DCMAKE_INSTALL_LOCAL_ONLY=ON -P "$B/cmake_install.cmake"
L=$I/benchmarks/layered_moving_fault
cmake -S "$L" -B "$D/layered-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/layered-plugin" -j1
P=$D/layered-plugin/liblayered_moving_fault.release.so
A=$I/bin/aspect-release
rg 'Aspect_(INCLUDE_DIRS|DIR) ' "$I/lib/cmake/Aspect/AspectConfig.cmake"
realpath "$D/layered-plugin/aspect-release"
rg -o '/[^ ]+/include/aspect/material_model/phase_field_fault.h' \
  "$D/layered-plugin/CMakeFiles/layered_moving_fault.dir/layered_moving_fault.cc.o.d"
sha256sum "$A" "$P" "$L/layered_vertices.txt"
~~~

**Finding 9.** The plugin compiler dependency names
`build-d-review/install/include/aspect/material_model/phase_field_fault.h`.
Its executable symlink resolves to `build-d-review/install/bin/aspect-release`.
The CMake package uses the same prefix for ASPECT and World Builder headers.
The installed executable SHA-256 was
`d75d72aec4714266631244fbe621fb2a007388c0c85abc5197e6c26517792edc`;
the new DSO was
`91c3cc22f606a47549e3e3e26676fded7087a16cf538806eaf1b7e9f0e20a90e`;
the ordered table was
`2aef87b0dd6520c5157023564fff06c07984084a6390b1e9711b7019a5df4dc7`.
The original D staging test copied the SDK with its old embedded paths. Its
numerical observations stand. This in-place installation establishes which
headers, plugin, executable and table the external run actually used.

~~~bash
E=$(mktemp -d /tmp/aspect-d-repairs.XXXXXXXX)
export ASPECT_SOURCE_DIR="$I" DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPIEXEC=/opt/openmpi/5.0.6/bin/mpirun
mkdir "$E/layered"
cp "$L/layered.prm" "$E/layered/input.prm"
echo "set Additional shared libraries = $P" >> "$E/layered/input.prm"
(cd "$E/layered" && "$A" input.prm > run.log)
bash refactoring/phase-field-fault/results/iposter/check-growing.sh "$I" "$E/growing"
python3 -BO refactoring/phase-field-fault/results/iposter/check-installed-vertices.py \
  "$I" "$P" "$E/vertices" --mpiexec "$MPIEXEC"
~~~

**Findings 7 and 14.** The checker sets each child process's input prefix to the
installation containing the checker and its parameter file. Its independent
reference still reads the historical inline lists. With the caller's
`ASPECT_SOURCE_DIR` unset, the negative case increased ID 0's persistent shear by
0.1 Pa in the table under test. ASPECT's resolved parameters name that table.
The public-file initial speed was 0.0625878629817589 m/s against the historical
0.0411706405355091 m/s. The checker failed on the rate difference of
0.0214172224462498 m/s. The positive input included a leading comment, a blank
line, mixed tabs and spaces, and end-of-row comments. It reproduced all 20
historical vertices at the three clocks. On each rank layout the crossing
file and inline histories agreed exactly. ID 9 crossed from rank 0 to rank 1.
The one-to-two-rank restart's maximum absolute difference across the compared
columns was 1.48e−14. All crossing and comparison budgets were unchanged.

**Finding 8.** The three benchmark-dependent CTest commands set their input
prefix to the checkout that supplies their Make dependencies. We ran them with
a deliberately different ambient prefix containing the reviewer's altered
vertex table. All three passed their existing references. The resolved public
vertex path and the Make dependency both named the checkout copy. A GNU Make
dry run that treated just this copy as newer scheduled the solver command.
With just the installed copy treated as newer, Make scheduled zero commands.

~~~bash
cmake -S tests -B "$D/tests" -DAspect_DIR="$I/lib/cmake/Aspect" \
  -DASPECT_BINARY="$A" -DASPECT_COMPARE_TEST_RESULTS=ON
ASPECT_SOURCE_DIR="$E/vertices/changed-shear" ctest --test-dir "$D/tests" \
  --output-on-failure -R '^(layered_moving_fault|layered_fault_traction_step|growing_fault)$'
rg 'Fault vertex file' "$D/tests/output-layered_moving_fault/parameters.prm"
(
  cd "$D/tests"
  rule=CMakeFiles/tests.layered_moving_fault.dir/build.make
  target=output-layered_moving_fault/screen-output
  make -sn -f "$rule" -W "$W/benchmarks/layered_moving_fault/layered_vertices.txt" "$target"
  make -sn -f "$rule" -W "$L/layered_vertices.txt" "$target"
)
~~~

The recorded external output directory was
`/tmp/aspect-d-repairs.TphOTorc`. The CTest run used the reviewer's
`/tmp/reviewD.LsOSQowE/altered` as its ambient prefix. The command above
uses the equivalent scratch installation generated by the negative control.
