#!/bin/sh
# Exercise the reversal and strip-geometry checkers against a matching C2 installation.
set -eu
if [ "$#" -ne 5 ]; then
  echo "Usage: sh $0 INSTALL DONOR CORRECTED HISTORICAL FRESH_DIRECTORY" >&2
  exit 2
fi
I=$(realpath "$1")
donor=$(realpath "$2")
new=$(realpath "$3")
old=$(realpath "$4")
D=$(realpath -m "$5")
R=$(realpath "$(dirname "$0")/check_reversal.py")
G=$I/benchmarks/growing_fault
A=$I/bin/aspect-release
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 UCX_LOG_LEVEL=error
mkdir -p "$(dirname "$D")"
mkdir "$D"
mkdir "$D/continued"
cp "$G/adiabatic_pressure.prm" "$D/continued/"
sed '/^subsection Boundary velocity model/,/^end$/s/^\( *set Function expression =\) /\1 \\\n    /' "$G/onecut.prm" > "$D/continued/onecut.prm"
grep -F 'set Function expression = \' "$D/continued/onecut.prm"
(cd "$D/continued" && "$A" --validate adiabatic_pressure.prm)
python3 -BO "$R" "$A" "$D/ordinary"
python3 -BO "$R" "$A" "$D/wrapped" --inputs "$D/continued"
for run in forward reverse; do
  for f in "$D/ordinary/$run/output/"*.csv; do
    cmp "$f" "$D/wrapped/$run/output/$(basename "$f")"
  done
done
python3 -BO "$R" "$A" "$D/ordinary" --replay

# Every mutation writes a fresh file. All linked records are read-only inputs.
python3 - "$D" "$new" "$old" <<'PY'
import csv
import math
import sys
from pathlib import Path

work, corrected, historical = map(Path, sys.argv[1:])

def read(file):
    with file.open() as stream:
        return list(csv.DictReader(stream))

def write(file, rows):
    with file.open('x') as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

for mutation in ('nan-both', 'inf-reverse', 'empty-forward'):
    for run in ('forward', 'reverse'):
        directory = work/mutation/run/'output'
        directory.mkdir(parents=True)
        for file in (work/'ordinary'/run/'output').glob('*.csv'):
            if file.name != 'fields-0001-r0.csv':
                (directory/file.name).symlink_to(file)
                continue
            rows = read(file)
            vertical = [r for r in rows if r['component'] == '1']
            if mutation == 'nan-both':
                for r in vertical:
                    r['value'] = 'NaN'
            elif mutation == 'inf-reverse' and run == 'reverse':
                vertical[0]['value'] = 'Inf'
            elif mutation == 'empty-forward' and run == 'forward':
                rows = [r for r in rows if r['component'] != '1']
            write(directory/file.name, rows)

for name, source in [('corrected', corrected), ('historical', historical)]:
    directory = work/name
    directory.mkdir()
    width = next(float(r['ds']) for r in read(source/'sections-0000.csv') if r['section'] == '8')
    for file in source.glob('*.csv'):
        if file.name not in {f'sections-{step:04}.csv' for step in (1, 2, 3)}:
            (directory/file.name).symlink_to(file)
            continue
        rows = read(file)
        strip = next(r for r in rows if r['section'] == '8')
        strip['beta'] = float(strip['beta']) - 3e-8
        strip['ds'] = width / math.cos(strip['beta'])
        write(directory/file.name, rows)
PY

for mutation in nan-both inf-reverse empty-forward; do
  if python3 -BO "$R" "$A" "$D/$mutation" --replay > "$D/$mutation.log" 2>&1; then exit 1; fi
  case "$mutation" in
    nan-both) expected='forward run: nonfinite velocity component 1' ;;
    inf-reverse) expected='reverse run: nonfinite velocity component 1' ;;
    empty-forward) expected='forward run: missing velocity component 1' ;;
  esac
  grep -F "$expected" "$D/$mutation.log"
done
for mode in corrected historical; do
  set --
  if [ "$mode" = historical ]; then set -- --historical-donor; fi
  python3 -BO "$G/compare.py" "$donor" "$D/$mode" "$@"
  python3 -BO "$G/compare.py" "$donor" "$D/$mode" "$@" --check-corruption
done
