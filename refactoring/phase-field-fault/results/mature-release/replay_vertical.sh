#!/usr/bin/env bash
# Run the original and current vertical-Gaussian fixtures with their own installed data.
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "Usage: bash $0 OLD_PREFIX OLD_LAYERED_DSO CURRENT_PREFIX CURRENT_LAYERED_DSO NEW_RUN_DIR" >&2
  exit 2
fi
old=$(realpath -e "$1")
old_plugin=$(realpath -e "$2")
current=$(realpath -e "$3")
current_plugin=$(realpath -e "$4")
run=$(realpath -m "$5")
scripts=$(cd "$(dirname "$0")" && pwd)
checkout=$(git -C "$scripts" rev-parse --show-toplevel)
old_revision=04aaf6db08abec8a06a27134f803cd265e64f7f9
old_case=$old/benchmarks/fixed_fault/curved_mature
current_case=$current/benchmarks/fixed_fault/curved_mature
if [[ "$old" == "$current" || "$old_plugin" == "$current_plugin" ]]; then
  echo 'Supply the historical and current installations and their separately built observers.' >&2
  exit 2
fi
for prefix in "$old" "$current"; do
  if [[ "$run" == "$prefix" || "$run" == "$prefix/"* ]]; then
    echo "Choose a run destination outside the installation $prefix" >&2
    exit 2
  fi
done

# The observed solve loads the named layered plugin; the work driver also
# loads a transfer probe from the installed example. Check the SDK recorded
# at configuration for both, including the bytes copied into the example.
plugin_sdk() {
  sed -n 's/^Aspect_DIR:[^=]*=//p' "$(dirname "$1")/CMakeCache.txt"
}
verify_observers() {
  local prefix=$1 layered=$2
  local expected="$prefix/lib/cmake/Aspect" sdk
  sdk=$(plugin_sdk "$layered")
  if [[ ! -d "$sdk" || ! "$sdk" -ef "$expected" ]]; then
    echo "Layered observer $layered uses SDK '$sdk', expected $expected" >&2
    return 1
  fi
  local installed="$prefix/benchmarks/fixed_fault/transfer_work/libtransfer_work.release.so" probe
  for probe in "$(dirname "$(dirname "$layered")")"/*/libtransfer_work.release.so; do
    [[ -f "$probe" && -f "$(dirname "$probe")/CMakeCache.txt" ]] || continue
    sdk=$(plugin_sdk "$probe")
    if [[ -d "$sdk" && "$sdk" -ef "$expected" ]] && cmp -s "$probe" "$installed"; then
      echo "Verified observers $layered and $installed against $sdk"
      return 0
    fi
  done
  echo "Missing matching transfer-work build for $installed against $expected" >&2
  return 1
}
verify_observers "$old" "$old_plugin"
verify_observers "$current" "$current_plugin"

# The fixtures share a parameter file, the deck it includes and its vertex table.
# Verify all three byte-for-byte against the scientific input at the historical revision.
for file in curved_mature/curved_mature.prm fixed_fault.prm curved_mature/vertices.txt; do
  git -C "$checkout" show "$old_revision:benchmarks/fixed_fault/$file" |
    cmp - "$old/benchmarks/fixed_fault/$file"
  cmp "$old/benchmarks/fixed_fault/$file" \
      "$current/benchmarks/fixed_fault/${file/curved_mature.prm/vertical_gaussian.prm}"
done
mkdir "$run"
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$run/mpl"
sha256sum "$old/bin/aspect-release" "$old_case/curved_mature.prm" "$old_case/vertices.txt" \
  "$old_case/../fixed_fault.prm" "$old_plugin" \
  "$old_case/../transfer_work/libtransfer_work.release.so" > "$run/historical.sha256"
sha256sum "$current/bin/aspect-release" "$current_case/vertical_gaussian.prm" \
  "$current_plugin" "$current_case/../transfer_work/libtransfer_work.release.so" > "$run/current.sha256"

for driver in run_curve.py check_local_work.py; do
  git -C "$checkout" show "$old_revision:refactoring/phase-field-fault/results/mature-release/$driver" \
    > "$run/$driver"
done
cd "$run"
python3 -BO run_curve.py "$old" "$old_plugin" historical
if ! grep -Eq 'version 3\.1\.0 \([^)]*, (0af424ddf|04aaf6db0)[[:xdigit:]]*\)' historical/public/run.log; then
  echo 'Historical solver must report the original production or example revision.' >&2
  exit 1
fi
python3 -BO "$scripts/run_curve.py" "$current" "$current_plugin" candidate --vertical-gaussian
for side in historical candidate; do
  if [[ "$side" == historical ]]; then
    case_dir=$old_case
    prefix=$old
    work_driver=$run/check_local_work.py
  else
    case_dir=$current_case
    prefix=$current
    work_driver=$scripts/check_local_work.py
  fi
  python3 -BO "$case_dir/check_curve.py" "$side/observed/output" --figure "$side/curve.png"
  python3 -BO "$work_driver" "$prefix" "$side/observed/output" "$side/work"
done
python3 -BO - "$current_case" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from check_restart import accepted_frame, check_clocks, compare

old, new = (Path(side)/'observed/output' for side in ('historical', 'candidate'))
for out in (old, new):
    check_clocks(out, [0, 1, 2])
for step in range(3):
    compare(old, new, step, exact=False)
    a, b = (accepted_frame(out, step, 1)[1] for out in (old, new))
    if any(float(a[key]['phi']) != float(b[key]['phi']) for key in a):
        raise ValueError(f'Historical native phase changed at step {step}')
PY
sha256sum --check historical.sha256
