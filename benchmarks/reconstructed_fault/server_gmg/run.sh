#!/usr/bin/env bash
# One bounded initialization; run inside an existing compute allocation.
# Usage: run.sh CASE OUTPUT_DIR ASPECT_BINARY PLUGIN [MPI_LAUNCHER ARGS...]
# Example: run.sh gmg /scratch/new-gmg /path/aspect-release /path/plugin.so ibrun -n 2
set -euo pipefail
if (( $# < 4 )); then
  echo 'Usage: run.sh {amg|gmg|gmg_q1|gmg_uninstrumented|coarse|coarse_context} OUTPUT_DIR ASPECT_BINARY PLUGIN [MPI_LAUNCHER ARGS...]' >&2
  exit 2
fi
case_name=$1
case "$case_name" in amg|gmg|gmg_q1|gmg_uninstrumented|coarse|coarse_context) ;; *) echo 'Unknown case' >&2; exit 2 ;; esac
output_dir=$(realpath -m "$2")
aspect_binary=$(realpath -e "$3")
plugin_library=$(realpath -e "$4")
shift 4
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
# Never overwrite outputs, even an empty directory from an earlier attempt.
mkdir -- "$output_dir"
git -C "$source_dir" rev-parse HEAD > "$output_dir/HEAD"
git -C "$source_dir" diff --binary > "$output_dir/working-tree.patch"
git -C "$source_dir" status --short -uno > "$output_dir/tracked-status.txt"

# Preserve the relevant inherited switches before clearing them; do not change
# compiler/MPI library search paths or site launch/binding configuration.
env | LC_ALL=C sort | sed -n '/^ASPECT_/p; /^I_MPI_/p; /^OMP_/p; /^MKL_/p; /^OPENBLAS_/p; /^DEAL_II_/p; /^SLURM_/p' > "$output_dir/environment-before.txt"
while IFS= read -r variable; do unset "$variable"; done < <(compgen -v ASPECT_)
export ASPECT_SOURCE_DIR="$source_dir"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1

cat > "$output_dir/input.prm" <<EOF
include $source_dir/benchmarks/reconstructed_fault/server_gmg/$case_name.prm
set Additional shared libraries = $plugin_library
set Output directory = $output_dir/output
EOF
{
  date -u
  uname -a
  lscpu
  command -v mpicxx || true
  mpicxx -show || true
  mpicxx --version || true
  ldd "$aspect_binary"
  ldd "$plugin_library"
  sha256sum "$aspect_binary" "$plugin_library" "$output_dir/input.prm"
  sha256sum "$source_dir/source/simulator/solver/stokes_matrix_free_local_smoothing.cc" \
    "$source_dir/benchmarks/reconstructed_fault/server_gmg/fault_gmg_diagnostics.h" \
    "$source_dir/source/simulator/solver/matrix_free_operators.cc" \
    "$source_dir/include/aspect/simulator/solver/matrix_free_operators.h" \
    "$source_dir/benchmarks/reconstructed_fault/server_gmg/coarse_diagonal_probe.cc" \
    "$source_dir/benchmarks/reconstructed_fault/server_gmg/$case_name.prm" \
    "$source_dir/tests/phase_field_fault_stage_i.prm" \
    "$source_dir/tests/phase_field_fault_stage_i.cc"
  printf 'Command: '
  printf '%q ' "$@" "$aspect_binary" "$output_dir/input.prm"
  printf '\n'
} > "$output_dir/provenance.txt" 2>&1

# Launcher and arguments remain an array: no eval or shell command strings.
# Retain stdout/stderr on failure and preserve timeout/launcher status.
set +e
timeout --signal=TERM --kill-after=30s 300s "$@" "$aspect_binary" "$output_dir/input.prm" > "$output_dir/run.log" 2>&1
run_status=$?
set -e
printf '%s\n' "$run_status" > "$output_dir/exit-status.txt"
if (( run_status != 0 )); then
  echo "Run failed (status $run_status); preserve $output_dir" >&2
  exit "$run_status"
fi
verification_marker='Reconstructed-fault Stage-I solve: verified'
if [[ "$case_name" == coarse || "$case_name" == coarse_context ]]; then verification_marker='GMG coarse diagonal probe: verified'; fi
if ! grep -q "$verification_marker" "$output_dir/run.log"; then
  echo "No verification marker; inspect $output_dir/run.log" >&2
  printf '%s\n' 'missing verification marker' > "$output_dir/verification-failure.txt"
  exit 1
fi
if [[ "$case_name" == coarse_context ]] && ! bash "$source_dir/benchmarks/reconstructed_fault/server_gmg/check_context_output.sh" "$output_dir/output/coarse_diagonal_probe.txt"; then
  printf '%s\n' 'missing or incomplete context cases' > "$output_dir/verification-failure.txt"
  exit 1
fi
echo "Case $case_name verified: $output_dir"
