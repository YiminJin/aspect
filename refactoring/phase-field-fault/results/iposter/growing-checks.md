# Installed growing-seed checks

The default one-field example, the four-field constant-pressure control and
the one-field pressure-gradient variant each ran for 5 ms on two MPI ranks
with one numerical thread per rank. All three used the installed
`growing fault history` postprocessor. The retained comparison log is
`build-assignment-d/runs/growing/check-5ms.log`. The following commands can
be run from the worktree root with a fresh destination.

```sh
set -e
I=$PWD/build-assignment-d/install
A=$I/bin/aspect-release
G=$I/benchmarks/growing_fault
plugin=$G/libgrowing_fault_checks.release.so
datum=$PWD/build-4c/tests/libpressure_constraint.release.so
export MPIEXEC=/opt/openmpi/5.0.6/bin/mpirun
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 UCX_LOG_LEVEL=error
E=$(mktemp -d /tmp/aspect-iposter.XXXXXX)/growing
bash refactoring/phase-field-fault/results/iposter/check-growing.sh "$I" "$E"

python3 "$G/check_interface.py" "$A" "$plugin" "$E/interface-final" >"$E/interface-final.log"
python3 "$G/check_material_settings.py" "$A" "$E/settings" >"$E/settings.log"

python3 "$G/check_adiabatic_pressure.py" "$A" "$plugin" "$E/profile-r1" profile >"$E/profile-r1.log"
python3 "$G/check_adiabatic_pressure.py" "$A" "$plugin" "$E/profile-r2" profile \
  --ranks 2 --mpiexec "$MPIEXEC" --serial "$E/profile-r1" >"$E/profile-r2.log"
python3 "$G/check_adiabatic_pressure.py" "$A" "$plugin" "$E/pressure-lifecycle" lifecycle \
  >"$E/pressure-lifecycle.log"
python3 "$G/check_physical_inputs.py" "$A" "$plugin" "$E/scalar-pressure" compression \
  --pressure-plugin "$datum" >"$E/scalar-pressure.log"
python3 "$G/check_adiabatic_pressure.py" "$A" "$plugin" "$E/adiabatic-constant" constant \
  --stage-b "$E/scalar-pressure" --pressure-plugin "$datum" >"$E/adiabatic-constant.log"

for mode in gap blocked close overwrite cleanup retry; do
  python3 "$G/check_history_output.py" "$A" "$E/writer-$mode" "$mode" \
    --mpiexec "$MPIEXEC" >"$E/writer-$mode.log"
done
```

The recorded root was `/tmp/aspect-iposter.UpyGyZ/growing`. At both clocks
each rank supplied 2,484 native samples. All four accepted stress components
matched the four-field control exactly, as did all nine strip records and
the ten energy and work columns. Strip 8 selected −0.14786810579423598 rad
relative to the initial seed, with an arclength of 0.25074188693430988 m.
Both values matched the corrected C2 reference exactly.

All 26 interface cases and 14 material-setting cases passed. The affine
profile check compared the consumed pressure at all nine physical stations
and exercised the +0.23 rad off-peak score. Across one and two ranks, the
largest direction and arclength differences were 6.96 × 10⁻⁹ rad and
2.60 × 10⁻¹⁰ m. Raising constant compression from 200 to 250 Pa changed
the strip-0 rate from 7.3474458432859358 × 10⁻¹⁴ to
2.6213457691637103 × 10⁻¹⁵ m/s. With a 100 Pa increase in the Stokes
pressure datum at constant 200 Pa compression, the rate was
7.3474458432858525 × 10⁻¹⁴ m/s. Both pressure-gradient cutbacks matched
their clean 2.5 ms controls. The unchanged affine-pressure checkpoint
resumed through 15 ms.

All six writer modes passed. The initial three-rank export used by
`retry` contained 1,548, 1,728 and 1,692 native points per rank. After
the deliberate rank-1 write failure the old rank-2 shards were intact.
The successful two-rank retry matched clean output on all 4,968 native
points and 1,280 FE records at both 0 and 5 ms.
