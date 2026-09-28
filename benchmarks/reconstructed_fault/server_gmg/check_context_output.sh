#!/usr/bin/env bash
# Validate the actual cases, not the output directory's name or generic marker.
set -euo pipefail
if (( $# != 1 )) || [[ ! -f "$1" ]]; then
  echo 'Usage: check_context_output.sh coarse_diagonal_probe.txt' >&2
  exit 2
fi
for expected in 'cartesian_active refinements=0' 'q1_parent refinements=6' 'cartesian_parent refinements=6'; do
  if [[ $(grep -Fxc "case=$expected" "$1") != 1 ]]; then
    echo "Missing or repeated context case: $expected" >&2
    exit 1
  fi
done
if [[ $(grep -c '^case=' "$1") != 3 || $(grep -c '^coarse_probe_pass=' "$1") != 3 || $(grep -Fxc 'coarse_probe_pass=1' "$1") != 3 ]]; then
  echo 'Expected exactly three context cases, each with a successful result.' >&2
  exit 1
fi
echo 'All three context cases verified.'
