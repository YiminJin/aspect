#!/usr/bin/env bash
# Recreate the initial-mesh, accepted-history and interpolation-only controls.
set -eu -o pipefail
[[ $# == 3 ]] || { echo "usage: original_pilots.sh WORKTREE OLD_RELEASE_PREFIX OUTPUT_ROOT" >&2; exit 2; }
repo=$(realpath -e "$1")
sdk=$(realpath -e "$2")
work=$(realpath -m "$3")
mkdir -p "$work"
mkdir "$work/baseline"
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for revision in 36725f110 264aedc32; do
  tree="$work/source-$revision"
  mkdir "$tree"
  git -C "$repo" archive "$revision" benchmarks/bp3_qd_thrust refactoring/phase-field-fault/results/bp3-preparation/history_events.py |
    tar xf - -C "$tree"
  stage="$work/stage-$revision"
  bash "$tree/benchmarks/bp3_qd_thrust/prepare.sh" "$sdk" "$stage"
  for ranks in 1 2; do
    launcher=()
    if [[ $ranks == 1 ]]; then layout=serial; else layout=two; launcher=(mpiexec -n 2); fi
    if [[ $revision == 36725f110 ]]; then
      run="$work/original-$layout"
    elif [[ $ranks == 1 ]]; then
      run="$work/log-age"
    else
      run="$work/old-log-two"
    fi
    mkdir "$run"
    cp "$stage"/{pilot.prm,bp3.prm,libbp3_qd.release.so,vertices.txt} "$run"
    (cd "$run" && "${launcher[@]}" "$sdk/bin/aspect-release" pilot.prm >run.log 2>&1)
    if [[ $revision == 36725f110 ]]; then
      cp -a "$run/output" "$work/baseline/$layout"
      capture="$work/old-$layout"
      mkdir "$capture"
      cp "$stage"/{pilot.prm,bp3.prm,libbp3_qd.release.so,vertices.txt} "$capture"
      cat >"$capture/capture.prm" <<'PRM'
include pilot.prm
set End time = 0
subsection Postprocess
  set List of postprocessors = mature fault statistics, visualization
  subsection Visualization
    set List of output variables = named additional outputs
  end
end
PRM
      (cd "$capture" && "${launcher[@]}" "$sdk/bin/aspect-release" capture.prm >run.log 2>&1)
    fi
  done
done
