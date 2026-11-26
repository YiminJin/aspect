# Assignment D frame, loading and permutation checks

The calculations started from `3617163671e7aadbe52385b6f11de946db4a5283`
and used the Release installation in `build-assignment-d/install`. All cases
below used one MPI rank with one numerical thread. The recorded output root
was `/tmp/aspect-frames-d.dCyqX3vz`. To replay, start in the worktree root
and use a fresh directory.

```sh
W=$PWD
I=$W/build-assignment-d/install
A=$I/bin/aspect-release
G=$I/benchmarks/growing_fault
GP=$G/libgrowing_fault_checks.release.so
M=$G/timestep/libmidpoint_fields.release.so
R=$I/benchmarks/fixed_fault/phase_ridge
RP=$R/libphase_ridge.release.so
D=$(mktemp -d /tmp/aspect-frames-d.XXXXXXXX)
export ASPECT_SOURCE_DIR=$I OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
cd "$D"
python3 -B "$G/check_repairs.py" "$A" "$GP" geometry3 geometry
python3 -B "$G/check_repairs.py" "$A" "$GP" companion2 companion
python3 -B "$G/check_repairs.py" "$A" "$GP" friction friction
python3 -B "$G/check_state_free.py" "$A" "$GP" state-free
python3 -B "$G/check_wall_clock.py" "$A" "$GP" wall-clock
```

At 5 ms, the seed check evaluated ASPECT's resolved affine wall function at
the center and at offsets (1,0) and (0,0.5) m. It also compared every prescribed
Q2 velocity on the walls with the independent Cartesian tensor
$R\left(\begin{smallmatrix}0.003&0.020\\0&-0.003\end{smallmatrix}\right)R^T$
applied to position relative to the center. The rates in this tensor are in
s⁻¹. The check compared every initial Bernstein cell mean with the integrated
Q2 interpolant of the independently specified seed (tensor Simpson rule).

| Seed center (m), dip from +x | Velocity at center (m/s) | At (1,0) m offset (m/s) | At (0,0.5) m offset (m/s) | Max applied $v_x,v_y$ errors (m/s) | Max initial phase mean error |
|---|---|---|---|---|---|
| (1,0.5), 30° | (0,0) | (−0.00716025404, −0.00240192379) | (0.00879903811, 0.00358012702) | 3.47e−18, 1.74e−18 | 4.17e−17 |
| (10001,0.5), 30° | (0,0) | (−0.00716025404, −0.00240192379) | (0.00879903811, 0.00358012702) | 4.26e−15, 2.46e−15 | 3.91e−13 |
| (1,0.5), 20° | (0,0) | (−0.00412974277, −0.00041119274) | (0.00979440363, 0.00206487138) | 5.21e−18, 8.68e−19 | 4.17e−17 |

All 4,968 translated native points retained their strip membership. Their
maximum position defect after translation back was 9.10e−13 m. Strip
coordinates and formation flags matched through 15 ms. The tip at 20° was
(1.4698463104, 0.6710100717) m. Keeping only the old wall origin on the
translated seed made the center velocity (−71.60254038, −24.01923789) m/s.
ASPECT accepted the step and the checker rejected both components.

The phase-first, implicit-companion run matched all named phase coefficients
at 0, 5 and 10 ms, and its tracer changed by 1.06395681e−4. The cold-creep
signed RSF defect was 1.42e−14 Pa. The constant-one phase passed and the
out-of-domain initial phase of 1.001 was rejected. Reversing both wall
components reversed the signed dashpot rates. The mature dashpot speeds
were 0.190875 and 0.19025 m/s. At 5 ms the ramp gave the half-load bulk
energy of 0.002724863755109 J/m, compared with 0.01089945502034 J/m for the
full load. The largest midpoint wall-work reconstruction error was
5.2e−18 J/m. The clean and cutback runs matched through 10 ms.

## Boundary-spanning ridge

```sh
for dip in 35 90; do
  python3 -B "$R/run.py" "$A" "$RP" "inclined-$dip" inclined 32 --angle "$dip" --mesh-plugin "$M"
  python3 -B "$R/check_frame.py" "inclined-$dip/output" "$dip"
done
python3 -B "$R/run.py" "$A" "$RP" free-32 free 32
PYTHONPATH=$R python3 -BO -c 'from check import check_case; from pathlib import Path; check_case(Path.cwd(),"free",32)'
python3 -B "$R/run.py" "$A" "$RP" laboratory-35 free 32 --angle 35
python3 -B - <<'PYTHON'
import json
from pathlib import Path
for directory in ('free-32', 'laboratory-35'):
    parameters = json.loads((Path(directory)/'output/parameters.json').read_text())
    wall = parameters['Boundary traction model']
    if (wall['Prescribed traction boundary indicators']['value'] != 'right y:function' or
        wall['Function']['Function expression']['value'] != '0;0.02+t+0.12*y'):
        raise ValueError(f'Laboratory traction changed in {directory}')
PYTHON
for spec in inclined-q1 inclined; do
  python3 -B "$R/run.py" "$A" "$RP" "$spec-64" "$spec" 64
done
donor=$W/../aspect/benchmarks/reconstructed_fault/bp5/moment-inclined/native_history_reference/output-native_history_reference
PYTHONPATH=$R python3 -BO -c 'from check import donor_check; from pathlib import Path; import sys; donor_check(Path.cwd(), *map(Path, sys.argv[1:]))' "$donor" "$W/build-3a-repair/donor-cell/output"
```

On the 1 m square, the 35° trace contacted the side faces at (−0.5,
−0.3501037691) and (0.5, 0.3501037691) m. The 90° trace contacted the
bottom and top at (0,−0.5) and (0,0.5) m. Each prescribed-rate calculation
also checked the phase core at three stations, zero phase beyond the support
and its last cut cell, and the two co-rotating Cartesian components of wall
velocity. The largest wall error at 100 ms was 8.68e−19 m/s across 380
prescribed local x entries and 286 y entries. For the free-RSF control the
laboratory loading remained the vertical right-wall traction
$0.02+t+0.12y$ Pa at both 60° and 35°.

The 38-label free-32 control passed its frozen Q8 check at 0, 12.5 and
25 ms. The maximum differences in position, normal integral and speed
were 5.56e−17 m, 8.53e−14 m and 1.39e−15 m/s. On the qualified 64² Q1
donor control all 23,984 active points matched at both clocks, all 12,880
remaining points returned zero load, and the maximum difference in
normalized weight against the independent cell-interval replay was
2.40e−13 m⁻¹. The historical remote-point quadrature spread was
5.88e−6 m⁻¹. The existing tolerances and references were used throughout.

## C2 reversal and corruption controls

```sh
sh "$W/refactoring/phase-field-fault/results/direction-c2/review_controls.sh" \
   "$I" "$W/build-5a-repair/donor-two/output" "$W/build-direction-c2/nine/principal/output" \
   "$W/build-direction-c1/frozen-ordinary/principal/output" "$D/c2"
```

For both spellings of the wall function (continued or single line), the
maximum forward-plus-reverse velocity defects were 1.73e−18 and
1.95e−18 m/s. The corresponding accepted frames matched byte for byte.
The replay rejected matching vertical NaNs, a reverse infinity and a missing
forward vertical velocity. Both comparison modes rejected their six injected
strip-8 errors at the expected clock and their independent zz-stress mutation.

## Exploratory configurations outside the qualified controls

The first extra growing run used 45° at (1,0.5) m, keeping the compression
adiabatic and constant at 200 Pa. At the first physical solve, strip 0 had
zero in-box segment length on which to evaluate its friction pressure.
ASPECT diagnosed the missing station before acceptance. The retained input
and log are in `geometry1/second-dip` within the recorded output root. The
20° run above exercises a distinct dip with all stations present.

The extra 32² matched-Q1 case stopped at initialization. Segment 3, station
2 exceeded its original normal-tail budget (omitted fraction printed as
0.000002). The pre-D driver gives the same diagnosis. The matched-Q1
reference uses 64².

```sh
python3 -B "$R/run.py" "$A" "$RP" inclined-q1-32 inclined-q1 32
python3 -B "$W/build-branch-readability/install/benchmarks/fixed_fault/phase_ridge/run.py" \
  "$A" "$RP" pre-d-inclined-q1-32 inclined-q1 32
```
