# Frozen 18-strip control

This control uses the 16 × 8 mesh, two MPI ranks and one numerical thread per rank. The ordinary parent writes a checkpoint at 5 ms. The coarse run resumes it for one 5 ms interval. The fine run reproduces that parent from the seed and takes two 2.5 ms intervals. The recorder repeats all three trajectories using its own checkpoint. The donor frames and both sets of canonical tolerances are the ones used at the Increment 5B closeout.

The row recorder replaces one object in the ordinary Release link. It compiles the current midpoint direction source along with a copy of the current midpoint solver containing a diagnostic immediately before candidate field publication. This records the stage obstacle, mass and residual after both convergence tests. The strict audit checks the ordinary and recorder physical files for byte equality before comparing the recorded rows to the independent reconstruction and the donor.

## Replay

From the worktree root, with the C2 Release build, its installed examples and the matching growing-fault plugin, use fresh recorder and run directories.

```sh
build=$PWD/build-6b/release
install=$PWD/build-direction-c2/install
work=$PWD/build-direction-c2/strip18
record=refactoring/phase-field-fault/results
growth=$PWD/build-direction-c2/growth-plugin/libgrowing_fault_checks.release.so

cmake -S "$install/benchmarks/growing_fault/timestep" -B "$work/mesh-plugin" \
  -DAspect_DIR="$install/lib/cmake/Aspect"
cmake --build "$work/mesh-plugin" -j1
python3 -B "$record/strip18/capture_rows.py" "$build" "$work/recorder"
sh "$record/direction-c2/run_strip18.sh" "$install" "$work/recorder/aspect-rows" \
  "$growth" "$work/mesh-plugin/libmidpoint_fields.release.so" \
  build-5b build-5a-repair/reference "$work/runs"
```

The last command uses the installed runner, observer inputs and strict audit. The donor FE exports in `build-5b/donor-{parent,coarse,half}` are the retained 5B replays. Their run commands and executable identities are in the individual run directories and the 5B working record.

## Strip 15

The strip turns at 5 ms and stays formed. The coarse 10 ms and fine 7.5 and 10 ms frames retain the same angle and length. The 5B values below come from the tracked strip-15 table in `results/strip18/strip15.csv`. The donor sets the tolerance in the last column.

| Accepted quantity | Frozen donor | Historical 5B target | C2 ordinary | C2 minus donor | Allowed absolute difference |
|---|---:|---:|---:|---:|---:|
| Angle (rad) | −0.14786132315540235 | −0.14786156004981515 | −0.14786159560289686 | −2.724475e−7 | 1.147861e−7 |
| Arclength (m) | 0.12537081680662576 | 0.12537082123034402 | 0.12537082189425566 | +5.087630e−9 | 1.254708e−8 |

Relative to the historical target, the correction changes strip 15's angle by −3.555308e−8 rad and its arclength by +6.639116e−10 m. Its angle discrepancy at each of the four clocks grows from 2.064 to 2.374 times the original tolerance. Its arclength remains within tolerance.

## Strict audit result

All six ASPECT runs exited 0. Ordinary parent, coarse and fine runs took 17.0, 18.3 and 48.3 s. The matched recorder took 16.7, 17.4 and 47.9 s. At each of the four physical frames the eight ordinary physical files matched the recorder byte for byte. Both MPI ranks recorded 576 returned phase rows at every frame. All 4,608 rows matched the donor classification and passed the assembled residual check.

The audit exited 1 with the four historical strip-15 angle failures. The remaining 383 comparisons passed at their original tolerances. The largest passing fraction of tolerance was 0.49652 for the native out-of-plane deviatoric stress at 5 ms. The accepted 7.5 ms endpoint retained 169 downhill rows and a maximum conditional load of 0.00171295729773436 J/m. No new quantity crossed a tolerance.

`strip18-audit.csv` concatenates the eight per-frame and time-window reports in lexical filename order. Each row preserves the count outside tolerance, maximum absolute error, maximum ratio to the allowance and the donor and target values at the key with that maximum ratio. To reproduce its contents after the strict audit completes:

```sh
{
  printf '%s\n' 'frame,quantity,failed,max_abs,max_ratio,worst_key,donor,target'
  for f in "$work"/runs/strict/*.csv; do
    frame=$(basename "$f" .csv)
    awk -v frame="$frame" 'NR > 1 { print frame "," $0 }' "$f"
  done
} > "$work/replayed-audit.csv"
cmp "$work/replayed-audit.csv" "$record/direction-c2/strip18-audit.csv"
```

Pairing the new ordinary trajectories with the retained pre-C2 recorder was also tested. The audit rejected the 5 ms parent balance file before checking its rows.

```sh
OPENBLAS_NUM_THREADS=1 python3 -B "$install/benchmarks/growing_fault/timestep/audit.py" \
  "$work/runs/ordinary" build-5a-repair/reference "$work/stale-recorder-control" \
  --returned-rows build-5b/rows
```
