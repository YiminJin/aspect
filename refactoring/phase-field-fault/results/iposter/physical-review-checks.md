# Material geometry, wall load and partitioned ridge checks

| D review | Changed input and result before repair | Discriminating measurement after repair |
| --- | --- | --- |
| 5. Growing material frame | In the 20° seed and wall control, changing only the material angle to 19° passed the tip, phase and wall checks. | The nine friction sampling stations on the incoming graph give 19° and a maximum normal offset of 0.0254497699 m from the intended line. The checker rejects this at the existing geometry tolerance of 2e−12 m. The 20° input passes. At strip 0 the physical depth changes from 0.829717276 to 0.803745926 m and the accepted 5 ms rate from 7.34744142e−14 to 8.65980805e−14 m/s. |
| 6. Layered top shear | With 1 m/s wall contrast, changing the top traction to the zero function passed the diagnostic column and side-wall checks. | The checker reads the resolved provider and component on the box top. It rejects the zero function. The independent 160-point column integral gives 0.198405165393 Pa initially and 0.030363678171 Pa at 12.5 ms. The side-wall end speeds are 1.5 and 2.5 m/s. Despite identical column reports, the changed provider moves ID 9 to x = 0.005927314530 m at 25 ms versus 0.009039004755 m with the prescribed column traction. |
| 13. Partitioned phase samples | Reading only rank zero from the reviewer's split 35° fixture reported a missing ridge at +0.3 m. | All 1,024 cells contain nine distinct native Q3 points. The original split fixture and fresh one- and two-rank runs pass with 9,216 samples and with 4,608 + 4,608 samples, respectively. Both contacts are at ±(0.5, 0.350103769104855) m. Maximum errors in both Cartesian wall velocities are 8.67e−19 m/s. Changing one second-rank index from 1 to 0 fails the cell/point identity check. |

At the first growing solve, `Material::section_pressures` queries
`MaterialSections::pressure_stations` on the accepted incoming graph.
Each returned location bisects its in-box strip arclength. The checker joins
the observed stations to the initial section IDs, checks their order and
normal distance to the intended seed, and measures the angle from the
first and last stations. The cell-mean seed and two independent Cartesian
wall checks still use their original budgets. The +10,000 m translation
moves all 4,968 native positions within 9.10e−13 m and the deliberately
stale wall origin gives errors of 71.60 and 24.02 m/s.

For the layered case `LayeredTraction::boundary_traction` takes horizontal
traction from `LayeredBackground::shear_traction()`. The ordinary
`StokesBoundaryTraction::execute` obtains the selected provider's traction
through the manager and multiplies it by the velocity test function and
face JxW. This check verifies that provider's selection, column traction
and physical side velocities. It makes no independent measurement of
an individual assembled face-vector entry. The two traction and velocity
checks retain the 1e−12 Pa and 1e−12 m/s tolerances.

The mature ridge check joins all bulk shards by (CellId, q) and checks their
coverage against the independent Q2 mesh export before testing core and
tail phase values. The serial 90° control also passes with 9,216 samples,
contacts (0, ±0.5) m, and maximum velocity error 8.67e−19 m/s.

## Replay

From the worktree root with the installed Release executable and benchmark
plugins prepared as in the poster walkthrough. Each solver process used one
numerical thread. The fresh local run used `/tmp/aspect-d-physical.2pkib1kg`.

```bash
W=$PWD
I=$W/build-assignment-d/install
A=$I/bin/aspect-release
G=$I/benchmarks/growing_fault
L=$I/benchmarks/layered_moving_fault
R=$I/benchmarks/fixed_fault/phase_ridge
M=$G/timestep/libmidpoint_fields.release.so
D=$(mktemp -d)
export PATH=/opt/openmpi/5.0.6/bin:$PATH ASPECT_SOURCE_DIR="$I"
export DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python3 -BO "$W/benchmarks/growing_fault/check_repairs.py" "$A" "$G/libgrowing_fault_checks.release.so" "$D/geometry" geometry
python3 -BO "$W/benchmarks/layered_moving_fault/check_review.py" "$A" "$L/liblayered_moving_fault.release.so" "$D/layered"
for run in 35-r1 35-r2 90-r1; do
    angle=${run%-r*}
    ranks=${run#*-r}
    python3 -BO "$R/run.py" "$A" "$R/libphase_ridge.release.so" "$D/inclined-$run" inclined 32 --angle "$angle" --mesh-plugin "$M" --ranks "$ranks"
    python3 -BO "$W/benchmarks/fixed_fault/phase_ridge/check_frame.py" "$D/inclined-$run/output" "$angle"
done
```

The growing and layered drivers each run the valid and deliberately altered
parameter files through the ordinary executable and require the altered
case to fail for its specific physical check. To repeat those four solves
with two ranks using the exact same generated parameter files:

```bash
for case in geometry/second-dip geometry/stale-material-angle layered/half-contrast layered/zero-top-traction; do
    input=input.prm
    if [[ $case == geometry/* ]]; then input=case.prm; fi
    dest=$D/mpi/$case
    mkdir -p "$dest"
    printf 'include %s\nset Output directory = %s\n' "$D/$case/$input" "$dest/output" > "$dest/case.prm"
    mpirun -np 2 "$A" "$dest/case.prm"
done
grow_check() {
    PYTHONPATH=$W/benchmarks/growing_fault python3 -BO -c 'from sys import argv; from pathlib import Path; from check_seed_frame import check_frame, SeedFrame; check_frame(Path(argv[1]), SeedFrame((1., .5), 20.))' "$1"
}
layer_check() {
    PYTHONPATH=$W/benchmarks/layered_moving_fault python3 -BO -c 'from sys import argv; from pathlib import Path; from check_review import check_wall_loading; check_wall_loading(Path(argv[1]), 1., 2.)' "$1"
}
grow_check "$D/mpi/geometry/second-dip/output"
grow_check "$D/mpi/geometry/stale-material-angle/output"   # expected 19° failure
layer_check "$D/mpi/layered/half-contrast/output"
layer_check "$D/mpi/layered/zero-top-traction/output"      # expected provider failure
```

Both two-rank runs reproduce the same 20°/19° and prescribed/zero-provider
decisions. The old one-rank checker passed the reviewer's retained
`stale19` and `wrongtraction` outputs. The new checker also rejects both
retained outputs. The point-identity corruption changes a single index,
retaining the file lengths and global row count:

```bash
good=$D/inclined-35-r2
bad=$D/duplicate
mkdir -p "$bad/output"
ln -s "$good/profile.txt" "$bad/profile.txt"
ln -s "$good"/output/labels-*.txt "$good"/output/mesh-0001-r*.csv "$good/output/bulk-0-0.txt" "$bad/output"
awk 'NR == 3 {$2 = 0} {print}' "$good/output/bulk-0-1.txt" > "$bad/output/bulk-0-1.txt"
python3 -BO "$W/benchmarks/fixed_fault/phase_ridge/check_frame.py" "$bad/output" 35
# Expected: Bulk (cell, q) identities do not cover every mesh cell exactly once
```
