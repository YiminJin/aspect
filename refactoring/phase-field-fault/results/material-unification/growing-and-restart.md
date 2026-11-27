# Growing-field reference and cross-binary continuation

Before changing material ownership, the Release binary built at
`6466cfe6130e5ea2d1324466b4c530b917247961` ran the installed one-field seed,
its affine-pressure variant and the four-field control from time zero.
The writer reported `6466cfe61`. Its SHA-256 was
`6b39e9d4d5807efa3caffe0bc0793770a7e1c876521588698ab22a2c2ab8307b`.
The one-field and four-field input digests were
`7d75313fe84e48419ea12ac473bc4bea7c0d7c124d249480369cfe10b8f20af1` and
`7f54a525fa05d6f250a1a7a05d8127bd3c1d58f6dfd29cc52e5545c6f116e081`.
The starting executable, both checkpoint generations and all original outputs
live under `build-unification`. The two manifests identify the captured files.
Future runs compare numerical fields within their original budgets.

## Reference capture

Use a separate source checkout of the writer revision (`OLD_SRC`) when replaying
after the refactor. In the original capture, `OLD_SRC=$W` while HEAD was still
`6466cfe61`, and `B=$W/build-6b/release` reused the configured Release build.
Each installation and run directory is separate. Run the commands in Bash
from the worktree root. The source dependencies on this machine use
`DEAL_II_DIR=/opt/dealii/9.6-local`.

```sh
set -eu
W=$PWD
D=$(mktemp -d "$W/build-unification-replay.XXXXXXXX")
B=$D/starting-build
I=$D/starting-install
R=$W/refactoring/phase-field-fault/results/material-unification
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
export MPIEXEC=/opt/openmpi/5.0.6/bin/mpirun
test "$(git -C "$OLD_SRC" rev-parse HEAD)" = 6466cfe6130e5ea2d1324466b4c530b917247961
git -C "$OLD_SRC" diff --exit-code HEAD -- source include CMakeLists.txt benchmarks
cmake -S "$OLD_SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DDEAL_II_DIR="${DEAL_II_DIR:-/opt/dealii/9.6-local}" \
  -DCMAKE_INSTALL_PREFIX="$I" -DASPECT_INSTALL_EXAMPLES=ON
cmake --build "$B" --target aspect -j2
cmake -DCMAKE_INSTALL_LOCAL_ONLY=ON -P "$B/cmake_install.cmake"
export ASPECT_SOURCE_DIR="$I"
bash "$OLD_SRC/refactoring/phase-field-fault/results/iposter/check-growing.sh" "$I" "$D/starting-growing"
for mode in mature growing; do
  case "$mode" in
    mature) base=$I/benchmarks/fixed_fault/curved_mature/curved_mature.prm; stop=.0125;;
    growing) base=$I/benchmarks/growing_fault/growing_seed.prm; stop=.005;;
  esac
  run="$D/starting-parent-$mode"
  mkdir "$run"
  cat > "$run/input.prm" <<PRM
include $base
set End time = $stop
set Output directory = output
subsection Checkpointing
  set Steps between checkpoint = 1
  set Number of checkpoints to keep = 4
end
PRM
  (cd "$run" && "$I/bin/aspect-release" input.prm >run.log 2>&1 && test -f output/restart/02/resume.z)
done
mkdir "$D/starting-growing-long"
cat > "$D/starting-growing-long/input.prm" <<PRM
include $I/benchmarks/growing_fault/growing_seed.prm
set End time = .015
set Output directory = output
PRM
(cd "$D/starting-growing-long" && "$I/bin/aspect-release" input.prm >run.log 2>&1)
```

The two-rank seed, four-field control and affine runs each exported 2,484
native points per rank at both accepted clocks, 0 and 5 ms. The serial
15 ms reference provides the subsequent 10 and 15 ms states for the
growing checkpoint continuation. The 12.5 ms mature and 5 ms growing
parents were both written on one rank. ASPECT writes these accepted parents
as generation 02 with the clock already advanced to the next solve (25 ms
and 10 ms, respectively). Immediately after capture the writer also resumed
copies of both its own parents and reproduced the independent endpoints.

The recorded bytes can be checked in place with `D=$W/build-unification`.
Use field comparisons for independent captures made on another build or MPI layout.

```sh
(cd "$D" && sha256sum --check "$R/starting-frames.sha256" && sha256sum --check "$R/starting-parents.sha256")
```

## Verification on the candidate

The qualified binary reported `4472fbcf6`, SHA-256
`ad5b91587404af2409ed7e30b0d5f0933c35bbcebce828d2ff3e177de2700930`.
The independent frames, both checkpoint continuations, the two-rank lifecycle,
the stateless-law and cold-creep tests, and the figure used this binary.
Each destination below must be fresh.
The comparison reads the original native points by (CellId, q), FE
coefficients by (CellId, component, local basis index), and sections and
pressure stations by strip ID. The original public checker supplies the
signed, relative-only creep budgets and the formation and native-state
checks. The ledger uses 10⁻¹¹ J/m absolute plus 10⁻⁶ relative. Clock,
duration, units and stress conventions agree with the existing
verification guide. The restored mature fields are compared directly with
the archived five-ID reference from `ae676e240`.

The two installations, the original outputs and checkpoints from the capture
above, and the plugins built against the candidate installation suffice for
the public-frame, continuation, retry, state-free and cold-creep checks below.
At 10 and 15 ms the continuation checks both native shear histories with the
fresh-frame relative tolerance of 10⁻¹² and zero absolute tolerance. It also
tests two scratch copies, reversing one signed shear at 10 ms and erasing
one absolute shear at 15 ms. Both mutations must fail at the affected native
point. The captured tables and checkpoint files remain read-only.

```sh
I=$D/install
G=$I/benchmarks/growing_fault
L=$D/layered-plugin/liblayered_moving_fault.release.so
GP=$D/law-plugin/libgrowing_fault_checks.release.so
export ASPECT_SOURCE_DIR="$I"
bash "$W/refactoring/phase-field-fault/results/iposter/check-growing.sh" "$I" "$D/qualified-growing"
python3 -B "$R/cross_build.py" "$I" frames "$D/starting-growing" "$D/qualified-growing"
python3 -B "$R/cross_build.py" "$I" continue "$D" "$L" "$D/qualified-continue"
python3 -B "$G/run.py" "$I/bin/aspect-release" "$GP" "$D/qualified-lifecycle" --ranks 2 --mpiexec "$MPIEXEC"
python3 -B "$G/check_state_free.py" "$I/bin/aspect-release" "$GP" "$D/qualified-state-free"
python3 -B "$G/check_repairs.py" "$I/bin/aspect-release" "$GP" "$D/qualified-cold" friction
MPLBACKEND=Agg MPLCONFIGDIR="$D/mpl" python3 -B "$G/plot_seed.py" \
  "$D/qualified-growing/seed/output" "$D/qualified-figure"
```

For the additional donor comparison, follow the
[pinned two-rank nine-strip regeneration recipe](../../STATUS.md#increment-5a-review-repairs).
It verifies the reference archive and runs `replay_midpoint_reference.py`
and `check_midpoint_reference.py` for `localized-onecut`. Set `DONOR_OUTPUT`
to that replay's `output` directory. The original replay used
`$W/build-5a-repair/donor-two/output`.

```sh
python3 -B "$G/compare.py" "${DONOR_OUTPUT:?Set the path to the verified localized-onecut output}" \
  "$D/qualified-lifecycle/principal/output"
```

The corruption replay changed just the signed rate of strip 8 in a scratch
5 ms section table. The comparator rejected −1.5301387463483072 × 10⁻¹³ m/s
against +1.5301387463483072 × 10⁻¹³ m/s (allowance 1.53 × 10⁻²⁵ m/s).
Changing only the consumed compression of that strip to 201 Pa also failed
against the measured 200 Pa. Both negative controls retain the original
native-point and FE data.

## Measured fields

| Installed comparison | Result |
|---|---|
| Constant 200 Pa and affine 200 + 50 depth Pa, each against its own pre-refactor run | At 0 and 5 ms, the physical-ID comparison passed on both MPI ranks. All 35 native, FE, pressure, section and balance CSVs across the three runs were byte-identical to their respective reference files. Each public frame has 4,968 native points, nine strip IDs and 1,280 FE coefficient rows. The four-field control has 2,144 FE rows. |
| Pressure consumed by signed friction at 5 ms | At strip 0, depth 0.967004940 m, compression changed from 200 to 248.350246981 Pa between the public constant and affine cases; speed changed from 7.34744584329 × 10⁻¹⁴ to 2.86410793411 × 10⁻¹⁵ m/s. At strip 8, depth 0.032995060 m, compression changed from 200 to 201.649753019 Pa and speed from 1.53013874635 × 10⁻¹³ to 1.32712211350 × 10⁻¹³ m/s. Both executables gave the same values. |
| Work and orientation | In the constant case at 5 ms, native bulk storage was 0.010899455020343897 J/m, fracture energy 6.5457126338534932 J/m and external work 0.010899727525304194 J/m. Strip 8 had β = −0.14786810579423598 rad and ds = 0.25074188693430988 m, exactly the approved nine-strip reference. In the affine case, β = −0.14788594422348153 rad and ds = 0.2507425532295271 m. |
| Mature generation 02 from the old executable | The candidate restored just the serial checkpoint from 12.5 ms, rebuilt its trace and published 25 ms. The native bulk and all five surface rows matched the decompressed reference bytes. The public label file also matched byte for byte. |
| Growing generation 02 from the old executable | The candidate restored the serial 5 ms parent and published only 10 and 15 ms. Each of the five tables at both clocks (fields, points, sections, friction pressure and balance) matched the old uninterrupted output byte for byte. At 15 ms strip 8 had V = 4.88659866 × 10⁻⁸ m/s, age = 10.0149993299 s. |
| Two-rank nine-strip continuation | Two successfully solved long attempts were rejected and rebuilt from the accepted parent; their 5, 10 and 15 ms results matched the clean trajectory. Generation 01 restored before the pre-solve observer and generation 03 restored before the last physical solve. The full donor comparison passed at the approved tolerances, with total signed slip 1.69302306477 × 10⁻⁴ m² at 15 ms. |
| External state-free law | The forward/reverse test checked opposite velocity components and strip-rate signs, the empty constitutive state in both archives, and the analytic dashpot solution. Relative centered tangent-action errors were 3.76 × 10⁻¹⁰ and 4.21 × 10⁻¹⁰. The mature rates were 0.190875 and 0.19025 m/s at 12.5 and 25 ms. |
| Cold creep | The largest signed RSF defect was 1.42 × 10⁻¹⁴ Pa at the stage age. Native rate recovery differed by at most 2.16 × 10⁻²⁷ m/s for a 1.35 × 10⁻²⁴ m/s budget. Frictional work was positive at 8.45045 × 10⁻²³ J/m. Constant phase 1 passed and an initial phase mean of 1.001 was rejected. |

**Figure.** The regenerated [growing-seed figure](../iposter/growing-seed.png)
shows accepted FE phase in the 2 m × 1 m box at 5 ms. The line through strip
8 inclines at 21.527781613° to the horizontal, a −8.472218387° turn from the
30° seed. The recorded station-7 to station-8 offset is −0.0369417619 m.
The PNG and PDF matched both the starting executable's regeneration and the
committed figures byte for byte. PNG SHA-256 is
`e3b025b8f2bf5a22c553cf9cb84d1a8606f68abf32cb5e48e0023f24a2a4b7b3`.
