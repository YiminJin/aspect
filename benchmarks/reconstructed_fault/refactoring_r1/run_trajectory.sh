#!/usr/bin/env bash
# Run after run_focused.sh and building both matching R1 plugins.
set -u
export PATH=/opt/gcc/12.4.0/bin:/opt/openmpi/5.0.6/bin:$PATH
export ASPECT_SOURCE_DIR="$PWD"
source benchmarks/reconstructed_fault/bp3/environment.sh
root=benchmarks/reconstructed_fault/refactoring_r1
runner="$root/run_logged.py"
binary=build-refactor-baseline/aspect-release
python3 "$runner" limiter-probe 300 mpirun -np 1 "$binary" "$root/inputs/limiter-probe.prm"
python3 "$runner" ih-converged 300 mpirun -np 1 "$binary" "$root/inputs/ih-converged.prm"
python3 "$runner" ih-converged-two 300 mpirun -np 2 "$binary" "$root/inputs/ih-converged-two.prm"
python3 "$runner" ih-no-composition-converged 300 mpirun -np 1 "$binary" "$root/inputs/ih-no-composition-converged.prm"
python3 "$runner" ih-cell 300 mpirun -np 1 "$binary" "$root/inputs/ih-cell.prm"
python3 "$runner" ih-cell-two 300 mpirun -np 2 "$binary" "$root/inputs/ih-cell-two.prm"
python3 "$runner" rollback-open-top 300 mpirun -np 1 "$binary" "$root/inputs/rollback-open-top.prm"
python3 "$runner" rollback-open-top-two 300 mpirun -np 2 "$binary" "$root/inputs/rollback-open-top-two.prm"
python3 "$runner" bp3-one 600 mpirun -np 1 "$binary" "$root/inputs/bp3-one.prm"
one_status=$?
python3 "$runner" bp3-two 600 mpirun -np 2 "$binary" "$root/inputs/bp3-two.prm"
if [[ "$one_status" == 0 ]]; then
  checkpoint=$(python3 - "$root/output-bp3-one" <<'PY'
import pathlib, sys
matches = [p.parent.name for p in pathlib.Path(sys.argv[1]).glob('restart/*/bp3_accepted_state.txt')
           if int(p.read_text().split()[0]) == 4]
if len(matches) != 1:
    raise SystemExit('Expected one retained checkpoint whose actual accepted step is four')
print(matches[0])
PY
  ) || exit 1
  python3 "$runner" bp3-branch 60 bash benchmarks/reconstructed_fault/bp3/branch_output.sh \
    "$root/output-bp3-one" "$checkpoint" "$root/output-bp3-split" || exit 1
  python3 "$runner" bp3-split 600 mpirun -np 1 "$binary" "$root/inputs/bp3-split.prm"
fi
