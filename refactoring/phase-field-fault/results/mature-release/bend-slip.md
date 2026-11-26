# How much scalar slip reaches the bulk?

The experiment applies a spatially constant test speed to the accepted 0 s trace and
DGQ2 phase on the 16 × 16 public mesh. Let `N_i` be the linear arclength hat
of material ID `i`. The independent line reference `L_i = ∫Γ N_i ds` is
half the sum of the incident segment lengths. The measured return is
`B_i = ∫Ω χ(x) N_i(s(x)) dA`, with zero integrand wherever the mechanical
point has no eligible normal foot. Both columns are lengths in m. At a test speed of
1 m/s their numerical values give the corresponding integrated slip rates in m²/s
per unit out-of-plane convention. The diagnostic keeps the speed positive, removing
the signed unit dyad that converts this scalar magnitude into a slip-gradient tensor.
RSF, viscosity and stress do not appear in this kinematic comparison.

The installed `integrated slip transfer` observer evaluates the accepted FE composition at
mapped cell points. It calls the production `FaultPhaseProfile::sample`, then
`Sample::add_load(JxW, rows)`. The quadratures are the ordinary tensor Gauss(8)
used by `FaultStokesCoupling::sample_bulk` and the support-split rule used by
`accept_step` for stress projection (ordinary Gauss(8) on smooth cells). In normal
mechanics the same `sample` supplies `slip_gradient`, which scales the selected
segment's dyad by χ times its interpolated speed. This test measures that scalar
transfer. The [local work test](vertical-gaussian.md) exercises the resulting Stokes load.

An independent integration uses physical arclength and signed normal distance. It
reconstructs each cell's DGQ2 from its nine accepted FE samples, recomputes the
full-ray normal integrals including the Q1 mass solve, and integrates each segment
through the box. Each ray is split at cell faces, perpendiculars through other
segment ends and lines of equal distance to two segments. Along-segment panels
are split at the crossings of these lines. Gauss(6) and Gauss(10) on the smooth
panels check convergence at fixed FE field and fixed geometry. Counting every
segment that admits a point gives the **complete-strip** integral. Subtracting the
nearest-eligible integral measures how much of the normalized strips is removed
where they overlap. Where no strip admits a point both integrals are zero.
The complete-strip integral can itself differ from the exact line length because
the full-ray denominator is projected onto Q1 from three stations per segment and
then enters the return through its reciprocal.

## Measured return

The public bend returns **1.21204363 m of 1.26223969 m**, a shortfall of
0.05019606 m or **3.97675%**. The largest loss is at ID 2, the turn at (0.50, 0.69) m.
Its hat receives **9.1230% less** than its prescribed line integral. The two contact hats
receive 2.8828% and 0.6514% less. The ordinary Q8 sum gives 1.21084484 m; resolving the
support cuts shifts that result by 0.00119880 m. Independent physical-ray integration
agrees with the split-cell result within 2.92 × 10⁻¹¹ m per ID, and Gauss(6) to
Gauss(10) changes each independent row by at most 1.84 × 10⁻¹³ m.

| Public ID | Position (m) | Line (m) | Native Q8 (m) | Support-resolved (m) | Resolved minus line (m) | Loss to overlapping strips (m) |
|---|---|---:|---:|---:|---:|---:|
| 0 | (0, 0.82) | 0.12747549 | 0.12389676 | 0.12380065 | −0.00367484 | 0.00015486 |
| 1 | (0.25, 0.77) | 0.25871954 | 0.24925722 | 0.24917315 | −0.00954639 | 0.00996053 |
| 2 | (0.50, 0.69) | 0.30452128 | 0.27678133 | 0.27673974 | −0.02778154 | 0.02877787 |
| 3 | (0.75, 0.45) | 0.37240031 | 0.36399190 | 0.36450421 | −0.00789610 | 0.00899548 |
| 4 | (1, 0.14) | 0.19912308 | 0.19691763 | 0.19782589 | −0.00129719 | 0.00008429 |
| **Sum** | | **1.26223969** | **1.21084484** | **1.21204363** | **−0.05019606** | **0.04797303** |

For the last column the trace, accepted phase and projected ray integrals all stay
fixed. Retaining *every* eligible strip yields 1.26001667 m. Choosing the nearest
one removes 0.04797303 m from that sum. Splitting the integrals at zero normal offset
locates the entire loss on the negative-normal side, inside the clockwise turns.
This accounts for 95.57% of the net global shortfall. A further 0.00222303 m
shortfall is present with all eligible strips. At the contacts most of the
discrepancy is present even in the complete-strip integral. At ID 2 the complete
strips give 0.30551761 m, slightly above the line integral; the overlap loss
accounts for the subsequent undershoot. Uncovered outer gaps carry no return.

The straight chord uses its own matched Gaussian with the same two contacts.
Its resolved total is **1.20950054 m** for a 1.20929732 m line (**+0.01681%**),
with zero overlap. For IDs 0–4 the resolved-minus-line defects are −0.00062280,
+0.00073440, −0.00010131, +0.00092469 and −0.00073177 m. The horizontal control
has constant normal integral along strike; it returns 1 m to within 1.8 × 10⁻¹⁵ m
and all five IDs agree within 6.2 × 10⁻¹⁶ m. The [full station table](slip-return.csv)
includes the independent Q8 evaluation on exactly the same points as the native
row. Its largest difference is 1.17 × 10⁻¹⁵ m (absolute budget 2 × 10⁻¹³ m).

The bend shortfall survives accurate bulk integration. A regional calculation using
this transfer can accumulate a prescribed fault slip distance while spreading less
scalar slip into the bulk, especially at a turn. The measured percentages apply to
these vertices, phase, box and discretization. Changes to the mechanical normalization
remain a separate design and verification task.

## Replay

From the checkout root use the installed executable, examples and matching
observer DSOs from the [release replay](README.md). Each run starts in a new
directory, stops after its time-zero solve and uses one MPI rank and one numerical
thread. The horizontal control places all vertices at y = 0.5 m; the straight
chord preserves both public contacts. Both retain the five x coordinates, all
material properties, the same phase law and w = 0.15 m.

```bash
D=$PWD/build-curve-phase
I=$D/install
L=$D/layered-plugin/liblayered_moving_fault.release.so
P=$D/transfer-work-plugin/libtransfer_work.release.so
R=$PWD/refactoring/phase-field-fault/results/mature-release
E=$(mktemp -d /tmp/aspect-slip-return.XXXXXXXX)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
python3 -BO "$R/measure_slip_return.py" "$I" "$L" "$P" "$E/slip"
cat "$E/slip/slip-return-candidate.csv"
```

The driver writes the candidate table under the run directory. Both fresh runs
and `--replay` compare all nine values for each (case, material ID) against the
committed [station inventory](slip-return.csv) from
`ae676e24095909196f885073b1146babce50292b`. The cross-build budget is
10⁻¹⁰ m + 10⁻⁸ times the absolute reference value. The independent Q8,
support-split and quadrature checks above retain their own tolerances.
Both observers are needed only for measurement. The public example
runs with the installed executable alone. The fresh runs for this report were at
`/tmp/aspect-curve-phase.McM7BMnA/slip-verified`. Its public-bend time-zero
bulk and label files matched the independent public observed run byte for byte.

The following scratch mutation changed only ID 2's measured Q8 contribution by
10⁻⁹ m. The replay exited with status 1 at the per-station native check, reporting
one mismatch at ID 2 and a 1.00000025 × 10⁻⁹ m defect.

```bash
cp -R "$E/slip" "$E/bad-q8"
python3 - "$E/bad-q8/public-bend/output/slip-transfer-0.csv" <<'PYTHON'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
with p.open() as f:
    rows = list(csv.DictReader(f))
row = next(r for r in rows if r['id'] == '2')
row['Q8_m'] = str(float(row['Q8_m']) + 1e-9)
with p.open('w', newline='') as f:
    writer = csv.DictWriter(f, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
PYTHON
python3 -BO "$R/measure_slip_return.py" "$I" "$L" "$P" "$E/bad-q8" --replay
```

## Discriminating the bend and the side of lost slip

Fresh runs of the three installed cases on one rank, with one thread, reproduced
the committed 15-row station table byte for byte. The SHA-256 of the reference
was `22f3d7ac26adb969056cdc64a584daa3109778637976ce1120c4434f871f4cb3`
before and after these checks. The tests below point the replay at isolated
candidate directories. Run them after the fresh command above.

```sh
mkdir "$E/chord-as-bend" "$E/reversed-side" "$E/reversed-checker"
for geometry in horizontal straight-chord public-bend; do
  ln -s "$E/slip/$geometry" "$E/reversed-side/$geometry"
done
for geometry in horizontal straight-chord; do
  ln -s "$E/slip/$geometry" "$E/chord-as-bend/$geometry"
done
ln -s "$E/slip/straight-chord" "$E/chord-as-bend/public-bend"
if python3 -BO "$R/measure_slip_return.py" "$I" "$L" "$P" "$E/chord-as-bend" --replay \
    > "$E/chord.log" 2>&1; then exit 1; fi
grep 'ValueError: public-bend ID 0 line_m:' "$E/chord.log"

python3 - "$R" "$E/reversed-checker" <<'PYTHON'
import sys
from pathlib import Path
src, scratch = map(Path, sys.argv[1:])
file = 'measure_slip_return.py'
text = (src/file).read_text()
if text.count('midpoint_offset < 0.') != 1:
    raise ValueError('Expected one negative-normal side classification')
(scratch/file).write_text(text.replace('midpoint_offset < 0.', 'midpoint_offset > 0.'))
(scratch/'slip-return.csv').symlink_to(src/'slip-return.csv')
PYTHON
if python3 -BO "$E/reversed-checker/measure_slip_return.py" "$I" "$L" "$P" \
    "$E/reversed-side" --replay > "$E/side.log" 2>&1; then exit 1; fi
grep 'ValueError: public-bend ID 0 negative_side_loss_m:' "$E/side.log"
```

Substituting the whole chord trajectory (vertices, phase, ray integrals and
returned weights) gave self-consistent quadrature. Its measured line length at
bend ID 0 was 0.151162165 m against 0.127475488 m, a discrepancy of
+0.023686677 m. The frozen comparison rejected it. Reversing just the side of
the discarded weight left all other columns unchanged at every station. All
five bend-side entries changed. At ID 0 the reversed classification reported
zero against 0.000154862055 m, and at ID 2 it lost the full 0.0287778685 m
negative-side contribution. The replay failed at ID 0 on
`negative_side_loss_m`.
