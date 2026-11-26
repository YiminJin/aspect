# Curved-fault baseline before material-model unification

The public scientific reference is `ae676e24095909196f885073b1146babce50292b`. A fresh, one-rank Release run supplied the three public label files. A second run with the matching observer supplied the native samples and reproduced those public files byte for byte. The [capture manifest](../../../../benchmarks/fixed_fault/curved_mature/public-reference/provenance.json) binds both binaries, the public input, inherited input, vertex table and observer source to the archived data. Normal verification reads the installed archive. The [one-time capture](public-baseline.md#reproduce-the-one-time-capture) requires an unused destination.

## Results

| Comparison | Observed result |
|---|---|
| Public curved release at 0, 12.5 and 25 ms | All five material IDs and 2,304 native (CellId, q) samples per clock matched. The serial native CSVs and public labels were byte-identical to the reference. A two-rank calculation reached the same endpoints, with 1,152 native samples per rank. At 25 ms the largest differences were 5.83 × 10⁻¹⁶ m/s in rate and 1.51 × 10⁻¹⁵ Pa in stress. |
| Public label semantics | Header, units and values match the accepted position, RSF age, accumulated slip and endpoint speed by ID. The endpoint speed integrated over each accepted interval recovers the new slip. Reordering output rows passed. |
| Bend inventory | All nine lengths for each of the 15 (case, ID) rows matched the committed station table byte for byte. Native Q8 returned 1.21084483627 m. The support-resolved return was 1.21204363318 m for a line measure of 1.26223969471 m, a **3.97675% deficit**. ID 2 at (0.50, 0.69) m was 9.123% low. The nearest-segment choice removed 0.04797303 m on the negative-normal side, including 0.02877787 m at ID 2. The straight chord was 0.01681% high and the horizontal control agreed with unity to roundoff. |
| Historical vertical Gaussian | The historical executable with its original input and the candidate with its named vertical fixture produced identical FE phase across all clocks. Maximum differences were 5.38 × 10⁻¹⁶ Pa in native stress and 1.67 × 10⁻¹⁶ m/s in rate. The per-prefix work checks passed. Historical binary and input hashes matched before and after the runs. |
| Accepted history | On the public case, serial restore and forced retry reproduced accepted history exactly. Restoring the serial 12.5 ms parent on two ranks gave a maximum 6.11 × 10⁻¹⁶ Pa difference in stress from clean two-rank acceptance at 25 ms. The separate vertical fixture also passed serial/parallel continuation, fitted continuation and retry. |

The [public data and column mapping](public-baseline.md), [nine-column station table and integration rules](bend-slip.md) and [historical provenance and commands](vertical-gaussian.md#replay-the-two-installations) are the inputs to a repeat review. Clock tolerance is 10⁻¹⁵ s. Physical columns use 10⁻¹⁰ in the column's units plus 10⁻⁸ times the magnitude of the reference. The independent work, rotation and quadrature checks retain their own tolerances.

For code review, `be68fe244` contains the archived run and its provenance. `add26361a` adds the keyed comparisons and rejects nonfinite rotation errors. The replay procedures make up the final commit of this repair.

## Sensitivity to a single change

| Deliberate change | First discrepancy detected |
|---|---|
| Radiation damping from 1 to 1.01 Pa s/m | At initialization, ID 0 speed changed from 0.09877810653069349 to 0.09839723054348577 m/s. The changed calculation itself completed 25 ms. |
| Exchange only the two public slip and rate values at 25 ms, preserving the header and native output | At ID 0 the public slip became 0.11271557306364266 m against 0.0027263044389429654 m. Independently changing just the unit token also failed. |
| Substitute the complete straight-chord calculation for the public bend | Independent integration agreed with the substituted geometry. The frozen ID 0 line measure detected a +0.023686677 m change. |
| Reverse only the normal-side classification | Every other station column agreed. ID 0 returned zero negative-side loss against 0.000154862055 m. |
| Replace one old velocity gradient with NaN | The old rotation checker returned success with both errors NaN. The repaired checker rejects the nonfinite shear error before applying the two original sign thresholds. |
| Exchange observer DSOs between the historical and current installations | The paired driver rejected the SDK mismatch before creating a run directory. |

The [public corruption commands](public-baseline.md#discriminating-checks) and [two bend substitutions](bend-slip.md#discriminating-the-bend-and-the-side-of-lost-slip) operate in scratch directories. To reproduce the rotation check, set `C` to the candidate's installed curved example, `E` to the fresh run root and `out` to its unmodified observed output:

```sh
out=$E/candidate/observed/output
python3 -BO "$C/check_curve.py" "$out" --figure "$E/accepted-curve.png"
cp -R "$out" "$E/gradient-nan"
python3 - "$E/gradient-nan/material-1-rank0.csv" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
header, *rows = p.read_text().splitlines()
names = header.split(',')
matches = []
for i, line in enumerate(rows):
    cols = line.split(',')
    if cols[names.index('cell')] == '0_4:0000' and cols[names.index('q')] == '0':
        matches.append(i)
        cols[names.index('Lxy')] = 'nan'
        rows[i] = ','.join(cols)
if len(matches) != 1:
    raise ValueError('Expected one old gradient sample')
p.write_text('\n'.join([header, *rows])+'\n')
PY
if python3 -BO "$C/check_curve.py" "$E/gradient-nan" > "$E/gradient.log" 2>&1; then exit 1; fi
grep 'Nonfinite rotated-parent shear error' "$E/gradient.log"
```

At the 25 ms acceptance, the old gradient is the one accepted at 12.5 ms. Its original Lxy at that key was 0.00053147256699930803 s⁻¹. On the clean output the maximum shear error was 6.072 × 10⁻¹⁷ Pa and reversing the old slip spin gave 0.000824991 Pa. The correct-sign ceiling is 2 × 10⁻¹³ Pa and the wrong-sign error must exceed 10⁻⁷ Pa.

## Installed replay and figure

The [installation and driver commands](README.md#reproduce-the-runs) build against each prefix's SDK and create fresh run directories. This execution used `build-curved-baseline/reference-install` (writer reporting `ae676e240`) and `build-curved-baseline/candidate-install` (reporting `be68fe244`, with the repaired example scripts installed). Only the example readers and replay procedures changed. Both builds reused the local Release compiler cache, with separate installations. The historical vertical executable in `build-mature-release/install` reported `0af424ddf`, with input and drivers at `04aaf6db0`. Its SHA-256 remained `4029757b271ef8c016442877529ef8db70cfaf08e20025e468e5696187ce1de5`. Its public input SHA-256 remained `2c2581c77785cc1d4bcecf5094d9f0ba3be83994f3df8e9a3f602d9bc7d4abc0`.

The following are the replay invocations with the paths used in this repair. Use fresh destinations when repeating them. All runs used one numerical thread per MPI rank.

```sh
W=$PWD
D=$W/build-curved-baseline
I=$D/candidate-install
C=$I/benchmarks/fixed_fault/curved_mature
R=$W/refactoring/phase-field-fault/results/mature-release
P=$D/candidate-layered/liblayered_moving_fault.release.so
T=$I/benchmarks/fixed_fault/transfer_work/libtransfer_work.release.so
E=$(mktemp -d /tmp/aspect-curved-baseline.XXXXXXXX)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 ASPECT_SOURCE_DIR="$I"
python3 -BO "$R/run_curve.py" "$I" "$P" "$E/candidate"
mkdir "$E/parallel"
cp "$E/candidate/observed/case.prm" "$E/parallel/"
(cd "$E/parallel" && mpirun -np 2 "$I/bin/aspect-release" case.prm > run.log 2>&1)
python3 -BO "$C/check_public_reference.py" "$E/parallel/output"
python3 -BO "$R/measure_slip_return.py" "$I" "$P" "$T" "$E/slip"
cmp "$E/slip/slip-return-candidate.csv" "$R/slip-return.csv"
old=$W/build-mature-release
bash "$R/replay_vertical.sh" "$old/install" "$old/layered-plugin/liblayered_moving_fault.release.so" \
  "$I" "$P" "$E/paired-vertical"
python3 -BO "$C/check_curve.py" "$E/candidate/observed/output" --figure "$E/accepted-curve.png"
cmp "$E/accepted-curve.png" "$R/curve.png"
```

The recorded fresh destination was `/tmp/aspect-curved-baseline.4PizBGTF`. The regenerated [accepted-solution figure](curve.png) had SHA-256 `2ca158651436220baa8049a67e47e5935532db935f1ad0eb1e1fb2479daa42cc`.

The [remaining installed replay commands](README.md#reproduce-the-runs) also passed. The original and moved phase fields matched exactly on one and two ranks. Moving vertex 2 moved the measured FE ridge by −0.118516463 m. The stale-field control failed by 0.159398075. Both field-role controls and the vertex/constitutive interface suite completed, both scoped CTests passed, and the Release unit suite passed all 2,605 assertions in 64 cases.

On the historical installation, the linked [restart](restart-checks.md), [straight-geometry](geometry-checks.md) and [signed-work](sign-checks.md) replays also passed. The two senses gave ±0.58901895576274 m in independent normalized work. The strict 38-label control differed by at most 1.39 × 10⁻¹⁵ m/s in rate. The legacy layered RSF continuation reproduced 20 IDs and 1,728 native samples exactly, and the state-free continuation gave byte-identical labels.

**Figure.** Accepted phase on the five-ID public fault at 25 ms. The four cuts span 0.13 m across the segments. Their hanging-wall-minus-footwall up-dip velocities, in material order, are 16.711, 33.749, 42.545 and 52.663 mm/s. The station inventory above measures the scalar slip returned by this geometry at unit test speed. It preserves the observed bend deficit for the ownership refactor.
