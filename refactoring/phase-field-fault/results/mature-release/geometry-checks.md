# Prescribed mature geometry and straight-trace controls

The Assignment D Release executable on the starting branch, `68f6d1510`,
rejected the five-row curved example during material initialization with
“Initial material vertices must be straight and strictly ordered.” The old
executable predates the slip-sense parameter. To reproduce the geometry
failure, omit only that setting and retain the example's initial state:

```bash
W=$PWD
before=$(mktemp -d)
sed '/set Mature slip sense =/d' benchmarks/fixed_fault/curved_mature/curved_mature.prm > "$before/input.prm"
(cd "$before" && ASPECT_SOURCE_DIR="$W" OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 \
    "$W/build-d-review/install/bin/aspect-release" input.prm > log 2>&1)
rg 'Initial material vertices must be straight' "$before/log"
```

The new initializer treats adjacent rows as a material edge and checks its
length, its relation to the preceding edge and intersections with earlier
nonadjacent edges. Every rank has the same ordered table. Validating simple
topology is quadratic in the number of edges in the worst case, as in the
existing segment rebuild. This change requires no additional collective or
persistent array. It removes the two stored seed directions. Only the
initial straight ridge fit needs them. The incident edge at each endpoint
selects its box face. At a corner the largest absolute normal component
wins, with ties in axis order.

The first independent geometry test uses vertices (0, 0.25), (0.625, 0.25),
(0.375, 0.75) and (1, 0.75) m. Its middle edge has length √5/4 m and zero
projection onto the endpoint chord. Its local normal is (−2, −1)/√5. A
probe at (0.5, 0.5) + (2, 1)/(16√5) m lies exactly on the chord. The actual
projection is halfway along the middle edge, with signed normal offset
−1/16 m. The test also checks the first contact at (0, 0.25) m, the
near-contact normal, the closest segment in an overlap, the earlier
segment at a shared material vertex, and an inactive point just outside
both incident normal fans and the distant segment's support. Five invalid
inputs exercise a crossing, repeated vertex, outside vertex, nonfinite
coordinate and an incident edge tangent to a noncorner contact.

All 54 assertions in seven geometric unit cases passed on one MPI rank
with one numerical thread. This includes the existing straight decimal-face
tolerance, inclined finite support, cut-cell quadrature, material rotation
through 105° and self-intersection checks.

The public interface suite exercises prescribed geometry through the
ordinary parameter file. It retains all three positions and IDs of a bent
three-row table at the three accepted clocks. Requesting the straight ridge fit on that identical
table produces the diagnostic `Set Fit trace to phase = false to use a
prescribed curved trace`. The larger five-row example and its independent
normal integrals are documented in the release demonstration. All 27
interface runs passed their expected comparisons or diagnostics, including
the stateless mature law (0.190875 and 0.19025 m/s at 12.5 and 25 ms) and
both wrong-route checkpoint controls.

## Straight runs from initial conditions

Both historical straight lifecycle suites rebuilt their physical states from
the branch's inputs. The homogeneous and nonuniform three-label examples
both agreed exactly with their forced-retry trajectories on the material
vertices and the 36 native samples at 0, 12.5 and 25 ms. Explicit and
automatic continuation from 12.5 ms matched the uninterrupted endpoint at
25 ms exactly. Both suites wrote their checkpoint on termination. At all
three clocks the displayed deviatoric stress matched the three accepted
DG components. The tensile-pulse cutback in the three-label case agreed
with its clean short-step control to 1.04e−14 or better over the stored
quantities.

| 25 ms three-label control vs frozen reference | Maximum absolute error |
|---|---:|
| Slip speed | 3.61e−16 m/s |
| Accumulated slip | 3.26e−17 m |
| Age | 3.38e−14 s |
| Velocity | 2.50e−14 m/s |
| Pressure | 4.74e−16 Pa |
| DG stress (xx, yy, xy) | 3.47e−16 Pa |
| Temperature | 5.69e−14 K |

The historical native comparison uses 1e−10 absolute or 1e−8 relative and
keys points by cell and quadrature index. The 5.684e−14 maximum reported at
each of the three clocks comes from the 293 K temperature field. No
reference was regenerated.

## Commands

Run from the checkout with the installed Release SDK, binary and growing-law
test plugin prepared in the release directory. Each lifecycle destination
must be fresh. The tests CMake tree is required for the fixed-fault probe.

```bash
W=$PWD
D=$W/build-mature-release
I=$D/install
A=$I/bin/aspect-release
F=$W/benchmarks/fixed_fault
export DEAL_II_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
(cd build-6b/release/unit_tests && "$A" --test 'Prescribed curved fault*,Prescribed mature fault*,Stationary fault spans*,Finite normal association*,Fault strip cuts*,Interior material connectivity*,A material fault retains*')
cmake -S tests -B "$D/tests" -DAspect_DIR="$I/lib/cmake/Aspect" -DASPECT_BINARY="$A" \
  -DCMAKE_BUILD_TYPE=Release -DASPECT_COMPARE_TEST_RESULTS=ON -DASPECT_RUN_ALL_TESTS=ON
cmake --build "$D/tests" --target fixed_fault -j1
for case in fixed_fault stationary_fault; do
  ASPECT_SOURCE_DIR="$W" python3 -Bu "$F/check_lifecycle.py" "$A" "$D/tests/libfixed_fault.release.so" \
    "$D/lifecycle-$case" --case "$case"
done
python3 -B "$F/compare_snapshots.py" "$F/stationary-reference" "$D/lifecycle-stationary_fault/clean/output"
ASPECT_SOURCE_DIR="$I" python3 -Bu benchmarks/growing_fault/check_interface.py "$A" \
  "$D/law-plugin/libgrowing_fault_checks.release.so" "$D/interface"
```
