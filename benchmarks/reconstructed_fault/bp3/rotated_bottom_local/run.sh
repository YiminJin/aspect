#!/usr/bin/env bash
set -euo pipefail
test_dir=$(cd -- "$(dirname -- "$0")" && pwd)
source "$test_dir/../environment.sh"
case_name=${1:?Specify uniform-A, uniform-B, A, or B}
if [[ "$case_name" == prepare ]]; then
  test ! -e "$test_dir/fixture" || { echo 'Refusing to overwrite fixture'; exit 2; }
  exec /opt/openmpi/5.0.6/bin/mpirun -np 1 "$test_dir/build/prepare_local" \
    "$test_dir/../fixtures/bp3_150x50" "$test_dir/fixture"
fi
test ! -e "$test_dir/output-$case_name" || { echo 'Refusing to overwrite prior output'; exit 2; }
aspect_binary=${ASPECT_LOCAL_BINARY:-"$test_dir/../../../../build-tmp/aspect-release"}
if [[ "$case_name" == production-validate ]]; then
  exec /opt/openmpi/5.0.6/bin/mpirun -np 1 "$aspect_binary" --validate "$test_dir/production-validation.prm"
fi
ranks=${2:-1}
[[ "$ranks" == 1 || "$ranks" == 2 ]] || { echo 'Use one or two MPI ranks'; exit 2; }
timeout --signal=TERM 900 /opt/openmpi/5.0.6/bin/mpirun -np "$ranks" "$aspect_binary" "$test_dir/$case_name.prm"
