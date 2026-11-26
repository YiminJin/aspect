#!/usr/bin/env bash
#SBATCH --job-name=BP3
#SBATCH --nodes=2
#SBATCH --ntasks=160
#SBATCH --ntasks-per-node=80
#SBATCH --cpus-per-task=1
# One launch with the donor's dependency modules already loaded. The submitter
# selects the walltime, partition and account for the replay or continuation.
set -euo pipefail
: "${BP3_RUN:?Set BP3_RUN to the staged donor run}"
: "${SLURM_JOB_ID:?Requires a Slurm allocation}"
: "${SLURM_NTASKS:?Requires a task count}"
[[ $SLURM_NTASKS == 160 ]] || { echo "Expected 160 MPI ranks" >&2; exit 1; }
[[ ${SLURM_JOB_NUM_NODES:?Requires a node count} == 2 ]] || { echo "Expected two nodes" >&2; exit 1; }
case "${1:-}" in
  donor-replay.prm | donor-cycles.prm) input=$1 ;;
  *) echo "Specify donor-replay.prm or donor-cycles.prm" >&2; exit 2 ;;
esac
cd "$BP3_RUN"
if [[ $input == donor-replay.prm ]]; then
  python3 ./donor_parent.py job3616642.prm fault.txt output/restart
fi
source ./environment.sh
solver=$(realpath -e ./plugin/aspect-release)
runtime_plugin=$(realpath -e ./plugin)
# The donor CMake adapter writes these DSOs into its source directory and
# creates aspect-release as a symlink to the executable in its Aspect_DIR.
cache="${BP3_PLUGIN_BUILD:-$runtime_plugin/build}/CMakeCache.txt"
run_log="launch-${SLURM_JOB_ID}-$input"
set -o noclobber
{
  module list 2>&1
  printf 'run_directory=%s\nexecutable=%s\nruntime_plugin_directory=%s\n' "$(pwd -P)" "$solver" "$runtime_plugin"
  if [[ -f $cache ]]; then
    sdk=$(sed -n 's/^Aspect_DIR:[^=]*=//p' "$cache")
    [[ -n $sdk ]] && cmp -s "$sdk/aspect-release" "$solver" || {
      echo "Plugin Aspect_DIR does not identify the selected executable: $cache" >&2; exit 1; }
    printf 'plugin_cache=%s\nplugin_Aspect_DIR=%s\n' "$(realpath -e "$cache")" "$(realpath -e "$sdk")"
  elif [[ -n ${BP3_PLUGIN_BUILD:-} ]]; then
    echo "Missing selected plugin CMake cache: $cache" >&2; exit 1
  else
    echo "Historical plugin build prefix unavailable (no CMake cache at $cache)."
  fi
  for file in "$solver" ./plugin/libbp3_{restore_150x50,coupled_checks}.release.so \
              job3616642.prm "$input" fault.txt environment.sh; do
    sha256sum "$(realpath -e "$file")"
  done
  sha256sum --check <<'DIGESTS'
1c8c078ba8e949787cb5985b35ae1c286516b456bf018cfdbefe9477053c4660  job3616642.prm
5460e34da479ef16e7dcc95f62669250a891ee8c969df5064dd3e4320bea4a24  fault.txt
DIGESTS
} >"$run_log.identity"
ibrun "$solver" "$input" >"$run_log.out" 2>&1
python3 ./donor_startup.py "$run_log.out" fault.txt
