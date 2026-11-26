# The frozen five-ID public curve

The scientific reference is `ae676e24095909196f885073b1146babce50292b` on
pf-rsf. One MPI rank with one numerical thread ran the installed public input
from initial conditions on the 16 × 16 mesh. A companion run of the same
physical input loaded the matching layered observer and selected the two
label/field postprocessors. The three public files from the bare executable
were byte-identical to those from the observed run.

[Reference data and input hashes](../../../../benchmarks/fixed_fault/curved_mature/public-reference/provenance.json)
identify the executable, observer binary and source, public and inherited
parameter files, ordered vertex table, and ten frozen data files. The archive
holds three physical clocks, all five native label records at each clock, all
2,304 native samples per clock in losslessly compressed CSV, and the three
original public text files. The checker verifies the frozen digests on each
invocation.

| Public column | Native column | Physical meaning |
|---|---|---|
| `label` | `i` | Permanent integer material ID, 0 through 4 |
| `x(m)`, `y(m)` | `x`, `y` | Accepted Cartesian coordinates in metres |
| `age(s)` | `Theta_s` | Accepted RSF state age in seconds |
| `slip(m)` | `slip` | Accumulated positive slip distance in metres |
| `rate(m/s)` | `V_m_s` | Accepted positive RSF endpoint speed in metres per second |

Public columns are checked by their header, units, physical ID and values
against both the frozen public files and the native observer. Starting at
zero slip, the integral of the reported endpoint speed over each accepted
interval must reproduce the new public slip distance at every ID. For example,
at 25 ms ID 0 has slip 0.0027263044389429654 m and speed
0.11271557306364266 m/s. The normal field check also compares every column
of the surface CSVs and native cell CSVs, including stresses, velocity
gradients and the shear offset. Each native record is keyed by (CellId, q)
across rank shards. The fixed-grid reference contains q = 0–8 for every cell
at all three clocks. Step IDs and physical keys are exact. Time and interval
length use 10⁻¹⁵ s absolute. Each physical column uses 10⁻¹⁰ absolute in its
units plus 10⁻⁸ times the magnitude of the frozen value.

## Normal candidate verification

Start with the installed candidate example and matching observer described
in the [replay instructions](README.md). The ordinary public verification
reads the installed reference. To check already published frames directly:

```sh
python3 -BO "$C/check_public_reference.py" "$E/curve/observed/output" \
  --public-output "$E/curve/public/output"
```

The fresh reference run and the candidate passed all three clocks. Reversing
the data-row order of all surface, native and public files in a scratch copy
passed. An additional replay of the previously qualified two-rank clean
trajectory passed on the same serial reference. Each frame contained 256
cells and five IDs.

## Discriminating checks

In a fresh directory run the ordinary inputs with a single physical change:

```sh
mkdir "$E/damping"
cat > "$E/damping/case.prm" <<PRM
include $C/curved_mature.prm
set Output directory = output
set Additional shared libraries = $P
subsection Postprocess
  set List of postprocessors = mature fault statistics, moving fault samples
end
subsection Material model
  subsection Phase field fault
    set Radiation damping = 1.01
  end
end
PRM
(cd "$E/damping" && "$I/bin/aspect-release" case.prm > run.log 2>&1)
python3 -BO "$C/check_public_reference.py" "$E/damping/output"
```

The solve completed 25 ms. The checker first rejected the initialization
Stokes/RSF solution. ID 0's speed was 0.09839723054348577 m/s against
0.09877810653069349 m/s, a 0.000380876 m/s difference.

The public-writer mutation uses the valid native trajectory from the same
run and a private copy of the bare public output:

```sh
cp -R "$E/curve/public/output" "$E/swap-public"
python3 - "$E/swap-public/fault-labels-00002.txt" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
header, *data = p.read_text().splitlines()
rows = [line.split() for line in data]
for row in rows:
    row[-2], row[-1] = row[-1], row[-2]
p.write_text('\n'.join([header, *(' '.join(row) for row in rows)])+'\n')
PY
python3 -BO "$C/check_public_reference.py" "$E/curve/observed/output" --public-output "$E/swap-public"
```

At 25 ms the swapped public `slip(m)` was 0.11271557306364266 at ID 0 vs
the true accumulated 0.0027263044389429654 m. The first two clocks and all
native comparisons passed. A separate scratch edit changing only `rate(m/s)`
to `rate(m/yr)` failed the header/units check at the same clock.

## Reproduce the one-time capture

The reviewed run used a fresh dedicated prefix and matching DSO built while
the checkout was at exactly `ae676e240`. To repeat later, use a separate
source tree at this revision and its configured Release build. Keep the
repair scripts in the current worktree. A clean build can use
`-DDEAL_II_DIR=/opt/dealii/9.6-local` on the author's machine. The same
ASPECT local install and external-plugin build steps as in the current
recipe go to a dedicated reference prefix:

```sh
W=$PWD
REV=ae676e24095909196f885073b1146babce50292b
R=$W/refactoring/phase-field-fault/results/mature-release
CAP=$(mktemp -d /tmp/curved-reference.XXXXXXXX)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
# SRC is the separately checked out scientific reference, or the original checkout at REV.
test "$(git -C "$SRC" rev-parse HEAD)" = "$REV" || exit 1
RI=$CAP/reference-install
cmake -S "$SRC" -B "$CAP/core" -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/opt/dealii/9.6-local \
  -DCMAKE_INSTALL_PREFIX="$RI" -DASPECT_INSTALL_EXAMPLES=ON
cmake --build "$CAP/core" --target aspect -j3
cmake -DCMAKE_INSTALL_LOCAL_ONLY=ON -P "$CAP/core/cmake_install.cmake"
cmake -S "$RI/benchmarks/layered_moving_fault" -B "$CAP/observer" -DAspect_DIR="$RI/lib/cmake/Aspect"
cmake --build "$CAP/observer" -j1
RP=$CAP/observer/liblayered_moving_fault.release.so
git show "$REV:refactoring/phase-field-fault/results/mature-release/run_curve.py" > "$CAP/reference-run.py"
python3 -BO "$CAP/reference-run.py" "$RI" "$RP" "$CAP/run"
python3 -BO "$R/freeze_public_reference.py" "$RI" "$RP" "$CAP/run" "$CAP/capture"
python3 -BO "$W/benchmarks/fixed_fault/curved_mature/check_public_reference.py" \
  "$CAP/run/observed/output" --public-output "$CAP/run/public/output"
```

For the recorded capture, the reference install was
`build-curved-baseline/reference-install`, its DSO came from
`build-curved-baseline/reference-layered`, and the fresh run was
`/tmp/aspect-curved-baseline.4PizBGTF/reference`. The installed Release
reported ae676e240. The writer requires an unused destination and validates
the resolved damping, recorded revision, inputs, five IDs, native mesh and
public column binding before creating it. Repeating capture from the
original frames into a new directory reproduced every archived byte.
Attempting to overwrite the frozen directory failed before writing. Removing
the last row of the 12.5 ms native CSV in a scratch copy failed the 256-cell
and nine-point completeness check and left the destination absent.
