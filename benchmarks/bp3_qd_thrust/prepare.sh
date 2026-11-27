#!/usr/bin/env bash
# Stage one immutable input set and compile the extension against this installation.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: prepare.sh ASPECT_RELEASE_PREFIX NEW_RUN_DIRECTORY" >&2; exit 2; }
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
prefix=$(realpath -e "$1")
solver=$(realpath -e "$prefix/bin/aspect-release")
sdk=$(realpath -e "$prefix/lib/cmake/Aspect")
mkdir -- "$2"
cd -- "$2"
catalogue="$here/history_events.py"
if [[ ! -f "$catalogue" ]]; then
  catalogue="$here/../../refactoring/phase-field-fault/results/bp3-preparation/history_events.py"
fi
cp "$catalogue" history_events.py
cp "$here"/{history.py,check_pilot.py,native_frames.py,make_vertices.py,README.md} .
mkdir plugin-source
cp "$here"/{CMakeLists.txt,bp3_qd.cc} plugin-source/
cp "$here"/{bp3.prm,pilot.prm,production.prm,vertices.txt,stampede3.sh} .
cmake -S plugin-source -B plugin-build -DAspect_DIR="$sdk"
cmake --build plugin-build --target bp3_qd -j2
[[ $(realpath -e plugin-build/aspect-release) == "$solver" ]] || { echo "SDK and executable prefixes differ" >&2; exit 1; }
cp plugin-build/libbp3_qd.release.so .
printf '%s\n' "$solver" > executable.path
printf '%s\n' "$sdk" > sdk.path
sha256sum "$solver" "$sdk/AspectConfig.cmake" executable.path sdk.path libbp3_qd.release.so \
  plugin-source/{bp3_qd.cc,CMakeLists.txt} bp3.prm pilot.prm production.prm vertices.txt stampede3.sh history_events.py history.py check_pilot.py native_frames.py > inputs.sha256
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$solver" --validate pilot.prm > validate-pilot.log
"$solver" --validate production.prm > validate-production.log
printf 'Prepared %s\nExecutable %s\nSDK %s\n' "$PWD" "$solver" "$sdk"
