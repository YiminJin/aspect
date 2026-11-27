# Mature trace through the unified installed material

The scientific reference for the public curve is `ae676e24095909196f885073b1146babce50292b`.
The installed checker reads its archived material IDs, (CellId, q) fields, clocks and public
label files. The [station table](../mature-release/bend-slip.md) records both the scalar
slip returned to the bulk and the side of the overlap loss. These checks use the original
10⁻¹⁰ absolute plus 10⁻⁸ relative cross-build budgets in the reported units and
10⁻¹⁵ s absolute for accepted clocks. Serial retry and serial continuation compare exactly.

The first installed implementation used SHA-256
`c00d4e21b9c0d0804814909b7969b0cf7424527242b2ea079d41310b049e728d`.
The layered cross-build comparison, two-installation vertical replay, scoped CTests and
Release units below ran on that binary. After publication of implementation commit
`4472fbcf6`, a rebuilt executable reporting that revision (SHA-256
`ad5b91587404af2409ed7e30b0d5f0933c35bbcebce828d2ff3e177de2700930`)
ran the public curve, station and restart comparisons. Both builds used the
new SDK. The separately rebuilt layered observer and transfer-work library had
SHA-256 `4b21717460b9e2a4f97a9bbfbdf6b08d62c00234ea95588c8be5b1d8860f4e67`
and `4bc19f0eb51718fdabdcb94545071f754ea43d21c69afb7b6bc535f5e1f27c38`.

## Release SDK and runs

Run from the worktree root. The candidate executable and examples are installed in
`build-unification/install`. The growing and mature continuation checks use the starting
executable installed separately at `build-unification/starting-install`. Give each run a fresh
directory. Each MPI rank uses one numerical thread.

```sh
W=$PWD
D=$W/build-unification
E=$D/mature-results
I=$D/install
C=$I/benchmarks/fixed_fault/curved_mature
R=$W/refactoring/phase-field-fault/results/mature-release
P=$D/layered-plugin/liblayered_moving_fault.release.so
T=$I/benchmarks/fixed_fault/transfer_work
M=/opt/openmpi/5.0.6/bin/mpirun
export ASPECT_SOURCE_DIR="$I" OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$D/mpl"
mkdir -p "$E"
cmake -S "$I/benchmarks/layered_moving_fault" -B "$D/layered-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/layered-plugin" -j1
cmake -S "$T" -B "$D/transfer-work-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/transfer-work-plugin" -j1
cp "$D/transfer-work-plugin/libtransfer_work.release.so" "$T/"
cmake -S "$W/tests" -B "$D/tests" -DAspect_DIR="$I/lib/cmake/Aspect" \
  -DASPECT_BINARY="$I/bin/aspect-release" -DCMAKE_BUILD_TYPE=Release \
  -DASPECT_COMPARE_TEST_RESULTS=ON -DASPECT_RUN_ALL_TESTS=ON
cmake --build "$D/tests" --target fixed_fault -j1
python3 -BO "$R/run_curve.py" "$I" "$P" "$E/curve"
python3 -BO "$C/check_curve.py" "$E/curve/observed/output" --figure "$E/curve.png"
python3 -BO "$R/measure_slip_return.py" "$I" "$P" "$T/libtransfer_work.release.so" "$E/slip"
cmp "$E/slip/slip-return-candidate.csv" "$R/slip-return.csv"
old=$W/build-mature-release
bash "$R/replay_vertical.sh" "$old/install" "$old/layered-plugin/liblayered_moving_fault.release.so" \
  "$I" "$P" "$E/vertical-pair"
python3 -BO "$C/check_restart.py" "$I" "$P" "$D/tests/libfixed_fault.release.so" "$E/restart" --mpiexec "$M"
python3 -BO "$C/check_public_reference.py" "$E/restart/curved-clean-two/output"
ctest --test-dir "$D/tests" --output-on-failure -j1 -R '^(fixed_fault|stationary_fault)$'
ctest --test-dir "$D/tests" --output-on-failure -j1 -R '^dg_stress_(transport|spin_order)$'
(cd "$W/build-6b/release/unit_tests" && ASPECT_SOURCE_DIR="$W" "$I/bin/aspect-release" --test)
```

The final executable used new destinations `$E/committed-curve`, `$E/committed-curve.png`
and `$E/committed-slip` for the second public run, plot and inventory. The archived public
text, accepted surface CSVs and decompressed native CSVs each matched their fresh file
byte for byte at all three clocks. The [cross-build checkpoint commands](growing-and-restart.md)
continue the separate starting writer’s accepted mature checkpoint on the final executable.

## Moving trace and material overrides

The installed layered input derives its moving material from the unified plugin. Its viscosity
varies in y and its exterior stress uses the benchmark boundary provider. Follow the
accepted parent from the DG stress transport and rotation through the actual Stokes solve.
Compare with the independent starting executable on the same mesh, subject to the same wall
data. At initialization and at the first physical endpoint the wall traction also has an
independent layer-compliance integral. Reconstruct the applied wall velocity from the Q2
field and check the selected traction provider.

```sh
for side in starting candidate; do
  if test "$side" = starting; then
    prefix=$D/starting-install
    plugin=$D/starting-layered/liblayered_moving_fault.release.so
  else
    prefix=$I
    plugin=$P
  fi
  dir=$E/$side-layered
  mkdir "$dir"
  cp "$prefix/benchmarks/layered_moving_fault/layered.prm" "$dir/input.prm"
  echo "set Additional shared libraries = $plugin" >> "$dir/input.prm"
  (cd "$dir" && ASPECT_SOURCE_DIR="$prefix" "$prefix/bin/aspect-release" input.prm > run.log 2>&1)
done
out=output-layered-moving-fault
before=$E/starting-layered/$out
after=$E/candidate-layered/$out
PYTHONPATH="$C" python3 -B - "$before" "$after" <<'PY'
from pathlib import Path
from sys import argv
from check_restart import check_clocks, compare
base, measured = map(Path, argv[1:])
for endpoint in (base, measured):
    check_clocks(endpoint, [0, 1, 2])
for step in range(3):
    compare(base, measured, step, exact=False)
PY
for file in "$before"/background-*.csv "$before"/material-*.csv "$before"/load-*.txt "$before/layered-background.csv"; do
  cmp "$file" "$after/${file##*/}" || exit 1
done
PYTHONPATH="$I/benchmarks/layered_moving_fault" python3 -BO - "$after" <<'PY'
from pathlib import Path
from sys import argv
from check_review import check_wall_loading
check_wall_loading(Path(argv[1]), 2., 2.)
PY
python3 -BO "$I/benchmarks/layered_moving_fault/plot_layers.py" "$after" "$E"
```

## Measured comparisons

- At 0, 12.5 and 25 ms the 16² curved run matched all 5 material IDs and 2,304
  native samples per clock, including the public columns against independent
  native ID records. The naked example and instrumented calculation gave the
  same public labels. The original checks for a repeated edge, a crossing and
  an inadmissible phase fit each reached the intended geometry diagnostic.
  The largest cell-mean rotated-parent shear defect was 6.07 × 10⁻¹⁷ Pa.
  Reversing only the spin sign in the independent reconstruction gave
  8.25 × 10⁻⁴ Pa.
- At 25 ms the regenerated [accepted solution](../mature-release/curve.png)
  matched the committed raster byte for byte. The relative up-dip speeds across
  the four 0.13 m cuts were 16.711, 33.749, 42.545 and 52.663 mm/s.
  The public-bend unit-speed inventory gave a native Q8 return of
  1.21084483627 m and a support-resolved return of 1.21204363318 m for a
  1.26223969471 m physical trace. All 135 station values reproduced the
  committed CSV byte for byte, including the 0.02877787 m negative-side loss
  at ID 2. The measured shortfall is 3.97675% globally and 9.123% at ID 2.
- The two-installation vertical Gaussian replay used the original executable
  at `0af424ddf` and its original input as the reference, including its
  matching layered and transfer observers. At all three clocks FE phase
  matched exactly. The largest native stress difference was 5.378 × 10⁻¹⁶ Pa,
  rate 1.665 × 10⁻¹⁶ m/s and integrated work 1.224 × 10⁻¹⁶ J/m. The transfer
  probe evaluated actual unconstrained Stokes cell vectors at fixed bulk
  iterate and parent, yielding −0.00080620588233 m at the contact and
  −0.00443265883749 m at the bend. Six historical hashes passed after replay.
- All three moving layered frames matched the starting build byte for byte
  for 20 material IDs, 1,728 native samples per frame, the layered background
  and the load. At 25 ms the measured virtual work was 5.65475496073643 J/m
  and ID 9 had moved 52.149417217 mm against 50 mm mean wall translation.
  The recovered side-wall speeds were 1 and 3 m/s. Applied shear traction
  agreed with the independent layer integral at 0.396810330786 Pa on the
  initialization interval and 0.0607273563419 Pa on the first physical step.
- The committed executable's serial checkpoint-only continuation reproduced the
  clean 25 ms state exactly. After a rejected 25 ms attempt, both 12.5 ms
  acceptances reproduced the uninterrupted state exactly. The same serial
  12.5 ms checkpoint continued on two MPI ranks with a maximum difference of
  6.106 × 10⁻¹⁶ Pa in point stress, 6.939 × 10⁻¹⁷ m/s in rate and
  4.491 × 10⁻¹⁸ J/m in shear work relative to the clean two-rank 25 ms run.
  The two-rank clean calculation passed the public archive at every clock,
  with 1,152 owned native points per rank. A seed whose fit moved ID 0 by
  0.169294636 m also resumed exactly. Changing the sense, vertex position,
  background loading or resistance parameters triggered its checkpoint diagnostic.
- All four scoped CTests passed, including both DG stress transport checks.
  The Release unit suite passed 2,605 assertions in 64 cases.

## Maintainer review of the mature observer

The observer now reads Eulerian viscosity (Pa s) at each label and Q3 point through
the active trace material. At zero strain rate, the local endpoint law returns its
additive stress offset (Pa) directly. The accepted DG stress is sampled separately
from the solution. The RSF label CSV retains its header and material IDs. A
state-free law omits the age column in both the native label CSV and public output.

The reviewed installation from production commit `4472fbcf6` reported 10 Pa s for
every label and native point on a curved case with viscosity 24 Pa s, modulus
17 Pa and initialization interval 0.025 s. Its maximum initial offset error
was 0.469 Pa. With the external linear dashpot and the input table's age
column removed, it aborted in `MovingFaultSamples` while reading age at ID 0.
The two reproductions can be run separately with the reviewed executable and
its matching SDK plugins. Use fresh result directories.

```bash
W=$PWD
D=$W/build-unification-review
O=$W/build-unification
C=$W/benchmarks/fixed_fault/curved_mature
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 -BOu "$C/check_observer.py" "$O/install" "$O/layered-plugin/liblayered_moving_fault.release.so" \
  "$O/law-plugin/libgrowing_fault_checks.release.so" "$D/reproduce-eta" --case changed
python3 -BOu "$C/check_observer.py" "$O/install" "$O/layered-plugin/liblayered_moving_fault.release.so" \
  "$O/law-plugin/libgrowing_fault_checks.release.so" "$D/reproduce-age" --case state-free
```

Build both plugins against the repaired installation and run each public case
in a fresh directory. All mature checks here used one MPI rank and one thread.

```bash
I=$D/install
P=$D/layered-plugin/liblayered_moving_fault.release.so
L=$D/law-plugin/libgrowing_fault_checks.release.so
C=$I/benchmarks/fixed_fault/curved_mature
R=$W/refactoring/phase-field-fault/results/mature-release
cmake -S "$I/benchmarks/layered_moving_fault" -B "$D/layered-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/layered-plugin" -j1
cmake -S "$I/benchmarks/growing_fault" -B "$D/law-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/law-plugin" -j1
python3 -BOu "$C/check_observer.py" "$I" "$P" "$L" "$D/observer"
python3 -BOu "$R/run_curve.py" "$I" "$P" "$D/curve"
MPLBACKEND=Agg MPLCONFIGDIR="$D/mpl" python3 -BOu "$C/check_curve.py" "$D/curve/observed/output" --figure "$D/curve.png"
cmp "$D/curve.png" "$R/curve.png"
```

Both parents are initially stress-free, and the initialization solve publishes
zero physical stress. At initialization and the first physical step the entire
offset is therefore the signed slip contribution. The check locates the
independent segment foot, interpolates the accepted label rates and normal
integrals, and evaluates all three components of
`κ χ V (t⊗n + n⊗t)` for the public opposite-tangent convention.
Here `κ = −η expm1(−G Δt/η)` in Pa s. It compares all 2,304 Q3 samples
and checks viscosity at these samples and all five vertices.

| Law and clock | Observed η (Pa s) | Expected κ (Pa s) | Peak absolute offset (Pa) | Maximum error (Pa) |
|---|---:|---:|---:|---:|
| RSF, initialization | 24 | 0.421259093454 | 0.430277990 | 2.22 × 10⁻¹⁶ |
| RSF, 12.5 ms | 24 | 0.211562015191 | 0.219999874 | 1.11 × 10⁻¹⁶ |
| Dashpot, initialization | 10 | 0.951625819640 | 1.12305536 | 3.33 × 10⁻¹⁶ |
| Dashpot, 12.5 ms | 10 | 0.124221995061 | 0.155836974 | 5.55 × 10⁻¹⁷ |

The unmodified public curve passed the original archive at 0, 12.5 and 25 ms.
Public label files and all native label records matched byte for byte.
Sampling the offset directly changed its last bits by at most 1.11 × 10⁻¹⁶ Pa.
Every other native field matched exactly. The independent rotated-parent shear
error was 5.99 × 10⁻¹⁷ Pa, rising to 8.25 × 10⁻⁴ Pa with reversed spin.
The [regenerated figure](../mature-release/curve.png) matched byte for byte.
It shows bulk velocity across the curve at 25 ms, with up-dip differences
of 16.711, 33.749, 42.545 and 52.663 mm/s across the four segments.

The separate layered run exercised a spatial override of the same viscosity
operation. Recreate the current endpoint and compare with the reviewed
`candidate-layered` trajectory recorded above.

```bash
PYTHONPATH="$C:$I/benchmarks/layered_moving_fault" python3 -BOu - "$I" "$P" "$D" \
  "$O/mature-results/candidate-layered/output-layered-moving-fault" <<'PY'
from pathlib import Path
import sys
import numpy as np
from check_restart import run, check_clocks, compare, accepted_frame
from check_review import check_wall_loading
prefix, plugin, root, ref = map(Path, sys.argv[1:])
out = run('layered', prefix/'benchmarks/layered_moving_fault/layered.prm', [plugin],
          root=root, prefix=prefix)
check_clocks(out, [0, 1, 2])
for step in range(3):
    compare(ref, out, step, exact=False)
    for records in accepted_frame(out, step, 1):
        y = np.array([float(row['y']) for row in records.values()])
        eta = [float(row['eta']) for row in records.values()]
        np.testing.assert_allclose(eta, 5.5+4.5*np.tanh(y/.35), atol=1e-14, rtol=0., equal_nan=False)
check_wall_loading(out, 2., 2.)
PY
```

The sampled layer spanned 1.00013512–9.99986488 Pa s. All 20 labels and
1,728 native points per clock matched the domain checks. The differences in
accepted stress, slip rate and integrated shear work were zero. At 25 ms
the work was 5.65475496073643 J/m.
