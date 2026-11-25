#!/usr/bin/env bash
# From the worktree root: bash sdk.sh BUILD FRESH_PREFIX PLUGIN_BUILD
set -eu
build=$(realpath "$1")
prefix=$(realpath -m "$2")
plugins=$(realpath -m "$3")
cmake -S . -B "$build" -DCMAKE_INSTALL_PREFIX="$prefix" -DASPECT_INSTALL_EXAMPLES=ON
cmake --build "$build" --target aspect explicit_dg_compositions -j3
for component in runtime includes examples Unspecified; do
  cmake -DCMAKE_INSTALL_LOCAL_ONLY=ON -DCMAKE_INSTALL_COMPONENT="$component" -P "$build/cmake_install.cmake"
done
cmake -DCMAKE_INSTALL_COMPONENT=examples -P "$build/benchmarks/explicit_dg_transport/cmake_install.cmake"
for pkg in growing_fault growing_fault/timestep fixed_fault/phase_ridge layered_moving_fault explicit_dg_transport; do
  cmake -S "$prefix/benchmarks/$pkg" -B "$plugins/$pkg" -DAspect_DIR="$prefix/lib/cmake/Aspect" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "$plugins/$pkg" -j2
done

mkdir -p "$plugins/test-src"
cat > "$plugins/test-src/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.13.4)
find_package(Aspect 3.1 REQUIRED)
DEAL_II_INITIALIZE_CACHED_VARIABLES()
project(fault_observers)
foreach(probe fixed_fault pressure_constraint dg_stress_transport dg_stress_spin_order)
  add_library(${probe} SHARED "${TEST_SOURCE}/${probe}.cc")
  ASPECT_SETUP_PLUGIN(${probe})
endforeach()
# This extension contains just a resistance law. Both production routes load it.
add_library(dashpot SHARED "${SDK}/benchmarks/growing_fault/linear_resistance.cc")
ASPECT_SETUP_PLUGIN(dashpot)
CMAKE
cmake -S "$plugins/test-src" -B "$plugins/tests" -DAspect_DIR="$prefix/lib/cmake/Aspect" \
  -DSDK="$prefix" -DTEST_SOURCE="$PWD/tests" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$plugins/tests" -j2
python3 - "$plugins" "$prefix" "$PWD" "$build" <<'PY'
import json,shlex,sys
from pathlib import Path
plugins,prefix,source,build=map(Path,sys.argv[1:])
for record in sorted(plugins.rglob('compile_commands.json')):
    for row in json.loads(record.read_text()):
        args=shlex.split(row['command'])
        includes={Path(t[2:]).resolve() for t in args if t.startswith('-I')}
        if prefix/'include' not in includes or includes.intersection({source/'include',build/'include'}):
            raise ValueError(f'Wrong ASPECT header search path in {record}: {includes}')
print('All external compile commands use the installed ASPECT include directory')
PY

# Exercise the ordinary installed example driver with these SDK-built plugins.
for pkg in growing_fault growing_fault/timestep fixed_fault/phase_ridge layered_moving_fault explicit_dg_transport; do
  install -m 755 "$plugins/$pkg"/lib*.release.so "$prefix/benchmarks/$pkg/"
done
