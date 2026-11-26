#!/usr/bin/env bash
# Run inside a 160-rank Stampede3 allocation with the donor's dependency modules
# already loaded. BP3_RUN is the fresh working directory in server.md. One launch.
set -euo pipefail
: "${BP3_RUN:?Set BP3_RUN to the staged donor run}"
: "${SLURM_JOB_ID:?Requires a Slurm allocation}"
: "${SLURM_NTASKS:?Requires a task count}"
[[ $SLURM_NTASKS == 160 ]] || { echo "Expected 160 MPI ranks" >&2; exit 1; }
case "${1:-}" in
  donor-replay.prm | donor-cycles.prm) input=$1 ;;
  *) echo "Specify donor-replay.prm or donor-cycles.prm" >&2; exit 2 ;;
esac
cd "$BP3_RUN"
if [[ $input == donor-replay.prm ]]; then
  python3 ./donor_parent.py job3616642.prm fault.txt output/restart
fi
source ./environment.sh
# Keep executable, both DSOs and their build stack paired with the parent.
run_log="launch-${SLURM_JOB_ID}-$input"
set -o noclobber
{
  module list 2>&1
  sha256sum ./plugin/aspect-release ./plugin/libbp3_{restore_150x50,coupled_checks}.release.so \
            job3616642.prm "$input" fault.txt environment.sh
} >"$run_log.identity"
ibrun ./plugin/aspect-release "$input" >"$run_log.out" 2>&1
