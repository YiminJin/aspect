# Strip 8 at the first prospective turn

On the frozen 8 × 4 case, the turn away from zero is resolved. The last digits of the preferred angle vary with the returned bulk stage and with the arithmetic used to score the same stage. Only the two-rank Release target passes every frozen endpoint check. These observations were made at target commit `3c58b5d26` using the 5A donor executable and its qualified source at `c407e612b` with the archived patch and additions.

## Capture before the 5 ms decision

The incoming angle is zero. The first physical interval is 5 ms and its algebraic midpoint duration is 2.5 ms. At entry to the selector, the coupled solve has returned the midpoint displacement increment, scaled pressure and phase. The accepted age is 10 s. The accepted history, disk, strip membership and returned iterate stay fixed during the angle search. A change of trial plane subsequently triggers reassembly.

The observer interrupts the original executable at entry to `BulkTipCoupling<2>::update_virgin_planes` or `MidpointSystem::select_prospective_planes`. It reads the actual argument, records 992 cell-local stage coefficients from all 32 cells, and samples the original strip quadrature on the owning rank. Each of the six captures has the same 180 (CellId, quadrature index) keys from cells 29, 30 and 31. All 180 contribute to the strip integral according to their phase fraction. Exactly 109 have positive disk weight. Rank 1 owns all 180 on the two-rank layout. Both ranks participate in the reductions.

The point files record physical area J, frozen disk kernel ρ, phase fraction F, seed-frame predictor components u and v in Pa, midpoint phase, and Cartesian predictor xx, yy, xy in Pa. The rooted replay uses the complete strip to compute free rate and mobility at arclength Δξ/cos β. Only the direction score uses the disk:

```text
traction(β) = u sin(2β) + v cos(2β)
V + M f(V, θ(h)) = Vfree
S(β) = Σ J ρ (1−F)² [traction(β) − f]²       Pa² m²
```

The compiled observer resamples the original iterate. To distinguish its evaluation roundoff from the executable's result, a second GDB breakpoint reads the original MPI_Allreduce receive buffers on rank zero immediately after each return. Strip 8 supplies nine raw moments and three centered moments from the 81- and 27-entry buffers. For two-rank Release, donor resampling differs in two raw entries (largest change 4.44e−16) and one centered entry. Target resampling differs in three raw entries (largest change 8.88e−16). The two Release selectors on their own **live-reduced moments** each recover their published angle exactly, on one and two ranks. The recaptured native point, section, FE and energy CSVs common to each original run match byte for byte.

## Where the results first separate

At initialization the two-rank native phase already differs by up to 4.16e−17; corresponding point areas differ by up to 2.17e−19 m². At entry to the selectors the two displacement-increment components differ by up to 1.24e−15 and 1.37e−15 m, pressure/G by 9.72e−15 and midpoint phase coefficients by 1.03e−15. On strip 8 this gives differences up to 7.04e−10 and 4.22e−10 Pa in u and v. The calculation localizes this input difference to the stage supplied to selection. It does not resolve its origin within the preceding nonlinear solve.

At the donor's reported angle, substituting only target u and v into the donor samples changes the 65-decimal direct score from 1.08481702050063685e−10 to 1.08481321052822496e−10 Pa² m². Substituting **all** target columns gives 1.08481321052823172e−10. Changing only J and ρ gives 1.08481702050063681e−10, and changing only F gives 1.08481702050064364e−10. The bulk predictor accounts for the 3.81e−16 Pa² m² shift in the absolute score across the two stages. That shift alone does not determine which of two nearby angles wins.

Both Release selectors then scored all six captured stages on three common inputs apiece. Point replay substitutes by key before accumulation. Live-moment replay supplies both selectors with the same twelve entries from the original binary. Ordered replay computes those entries by sorted key, using `math.fsum` and a two-pass centered variance. Each selector uses its own compiled search and linked production scalar resistance solve. The table gives angles returned on the two principal captures in radians. The archive also contains the one-rank and Debug input replays.

| Shared input | Donor selector | Target selector |
| --- | ---: | ---: |
| Donor points | −0.147867690333 | −0.147867543123 |
| Target points | −0.147867600194 | −0.147868802765 |
| Donor live moments | **−0.147868871839** | −0.147869143271 |
| Target live moments | −0.147867599914 | **−0.147868802765** |
| Donor ordered moments | −0.147869191000 | −0.147868670845 |
| Target ordered moments | −0.147868579236 | −0.147868819980 |

On their *own live moments*, the donor gives both reported maxima exactly the same binary64 score. The target gives the donor angle 6.78e−21 Pa² m² more than its own search result when explicitly asked to evaluate both. The golden-section search samples a subset of angles, so its returned angle need not win such a cross-check on a numerically flat peak.

| Probe | Donor live resistance (Pa) | Target live resistance (Pa) | Donor live S (Pa² m²) | Target live S (Pa² m²) |
| --- | ---: | ---: | ---: | ---: |
| Incoming | 49.99864211480510 | 49.99864211486752 | 2.446773190185818e−11 | 2.446760281596438e−11 |
| Donor maximum | 52.19952198834514 | 52.19952198858568 | 1.084817017087283e−10 | 1.084813207238305e−10 |
| Target maximum | 52.19952201921610 | 52.19952201945664 | 1.084817017087283e−10 | 1.084813207170460e−10 |

On identical two-rank donor points the nine raw entries agree exactly. The first differing reduced entries are the three centered moments, by 2.09e−27, 6.98e−28 and 5.18e−27 Pa² m². The donor centers incrementally within each rank and corrects to the global mean. The target sums deviations from the global mean on its second pass. A common set of twelve moments removes this difference. The following values use the same **ordered donor moments** and angle in each column pair.

| Angle (rad) | Donor resistance (Pa) | Target resistance (Pa) | Donor S (Pa² m²) | Target S (Pa² m²) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 49.99864211480509 | 49.99864211480509 | 2.446773191152452e−11 | 2.446773191152452e−11 |
| −0.14786887183904318 (donor maximum) | 52.199521988345126 | 52.199521988345126 | 1.084817017290820e−10 | 1.084817017290820e−10 |
| −0.147868802764826 (target maximum) | 52.19952201921609 | 52.19952201921607 | 1.084817017155129e−10 | 1.084817017358666e−10 |

At the target maximum both scalar solves receive exactly 3.515075060444475e−8 m/s free rate and 6.733892616420192e−10 m/(Pa s) mobility. At this angle their reported resistances differ by three binary64 spacings (2.13e−14 Pa). Substituting each returned resistance into a direct ordered sum of the 180 point contributions gives 1.0848170172173502e−10 and 1.0848170174208870e−10 Pa² m². The difference reproduces the 2.04e−20 Pa² m² spread of the selectors' scores. This is enough to reverse the ranking of the two reported angles even at identical moments. Both scalar routines solve the regularized asinh law with constant-rate aging and a roundoff-scaled certificate. The observation isolates their returned resistance, without assigning the three spacings to an individual operation inside the root.

As an independent control, the offline replay solves the scalar equation with 65 decimal digits and sums the **positive pointwise squares**. It starts from the captured binary64 point data and uses the same two-rank Δξ. The score at zero is 2.44677319208859539e−11 Pa² m². At the donor and target maxima the scores are 1.08481702050063684746e−10 and 1.08481702050770631746e−10. Increasing precision to 85 digits changes these latter scores by at most 1.08e−70. The target-minus-donor gap is **+7.06947e−22 Pa² m²**, or **3.47%** of the observed 2.03537e−20 fixed-input spread. Using the target point data instead gives +7.50862e−22. On the donor points, all six reported angles span only 6.52e−21 Pa² m². By contrast, the score gain from zero is 8.4014e−11, about 516 times the production tie threshold of 1.6272e−13 Pa² m². The donor's normalized gain after multiplication by m/(2G) is 3.36056e−15.

## What reaches accepted history

The endpoint audit compares every frozen native, strip and ledger column at every available clock, including all 4,968 point identities per frame. It also compares incoming strip history, three available midpoint quantities, and derived integrated slip and work defect. The latter derived columns are descriptive. Existing tolerances apply to the original columns, including the added fracture energy at its own scale. The angle and arclength allowances for strip 8 are 1.14787e−7 rad and 2.50842e−8 m. Every angle recorded at 5 ms is retained at 10 and 15 ms.

| Target / reference | Strip 8 angle at 5 ms (rad) | Δ angle (rad) | Δ arclength (m) | Outside tolerance at 5, 10 and 15 ms |
| --- | ---: | ---: | ---: | --- |
| Release 2 / donor 2 | −0.147868802764826 | +6.90742e−8 | −2.57989e−9 | None |
| Release 1 / donor 2 | −0.147867578568529 | +1.29327e−6 | −4.83028e−8 | Angle, arclength |
| Debug 1 / donor 2 | −0.147868295240347 | +5.76599e−7 | −2.15357e−8 | Angle |
| Debug 2 / donor 2 | −0.147867340658143 | +1.53118e−6 | −5.71886e−8 | Angle, arclength |
| Release 1 / donor 1 | −0.147867578568529 | −3.76968e−7 | +1.40794e−8 | Angle |

All nine formed flags and their parent values agree. Strips 0–7 keep zero direction. The six trajectories reproduce the observed strip-8 arclength from their own Δξ/cos β. Thus the geometric discrepancy first seen at 5 ms persists by inheritance; there is no later divergent formation event. Across all four target layouts relative to frozen donor 2, the maximum absolute differences are 4.24e−16 m in strip slip, 7.48e−14 m/s in rate, 4.25e−13 s in age, 3.16e−6 Pa in resistance, 1.80e−6 Pa in native stress, 9.04e−13 in recoverable strain and 2.55e−15 in native phase. Stress reaches at most 7.82% of its pointwise combined budget. Added fracture energy differs by at most 4.00e−14 J/m, external work by 4.33e−14 J/m, friction/radiation loss by 1.21e−14 J/m, and the derived physical work defect by 5.19e−14 J/m. The final integrated signed slip is 1.69302306477e−4 m² with maximum difference 1.46e−16 m². The mechanical residual, phase KKT and transfer-work gates pass at every clock. The canonical comparison and production law retain their existing settings.

## Reproduce the evidence

The compressed evidence contains six pre-turn captures, their original reduced buffers and endpoint plane tables, the 36 common-input replays and all three analysis CSVs. Entries D, T and B denote donor Release, target Release and target Debug, and the digit gives the native capture's rank count. Replays of Debug inputs in the score audit use the Release selectors; the native Debug angles are in the B1 and B2 section tables. Stage coefficients are keyed by cell, FE component and component-local index. Round-trippable point rows start with CellId and integer quadrature index. The archive SHA-256 is `0c7875ccd2451e2864acacb22ef7f9fd7c60cf181158528a69a2a575b0e8d757`.

The following commands reproduce the score and substitution CSVs from the stored data. Run from the worktree root in Bash with Python 3 and mpmath. Endpoints additionally use NumPy and the original 5A run directories.

```bash
P=refactoring/phase-field-fault/strip8
R=$(mktemp -d /tmp/strip8-evidence.XXXXXX)
tar -xzf "$P/evidence.tar.gz" -C "$R"
(cd "$R" && sha256sum --quiet -c SHA256SUMS)
python3 "$P/replay.py" "$R"
python3 -O "$P/replay.py" "$R" --analyze "$R/reproduced.csv"
python3 -O "$P/replay.py" "$R" --stage-controls "$R/check.csv"
cmp "$R/reproduced.csv" "$R/score-audit.csv"
cmp "$R/check.csv" "$R/substitution.csv"
D=$(cat build-5a/scratch)
python3 -O "$P/endpoints.py" "$D" "$R/endpoints-reproduced.csv"
cmp "$R/endpoints-reproduced.csv" "$R/endpoints.csv"
```

The capture used the existing 5A ASPECT executables and original inputs with fresh output directories. For this run D was /tmp/pf-rsf-5a.ZICqAr and W was /tmp/pf-rsf-strip8.3w2akP. The commands below build only the diagnostic shared libraries. They use the x86-64 System V calling convention, GDB, Open MPI, the local compiler and the original build flags. Each launch runs sequentially with one numerical thread per rank. Recreating the captures requires the original qualified build/source and native 5A outputs. The stored score audit can be reproduced independently of them.

```bash
P=refactoring/phase-field-fault/strip8
D=$(cat build-5a/scratch)
W=$(mktemp -d /tmp/pf-rsf-strip8.XXXXXX)
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for role in donor target debug; do python3 "$P/capture.py" "$D" "$W" "$role"; done
python3 "$P/capture.py" "$D" "$W" donor  --run D2 --ranks 2
python3 "$P/capture.py" "$D" "$W" target --run T2 --ranks 2
python3 "$P/capture.py" "$D" "$W" donor  --run D1 --ranks 1
python3 "$P/capture.py" "$D" "$W" target --run T1 --ranks 1
python3 "$P/capture.py" "$D" "$W" debug  --run B2 --ranks 2
python3 "$P/capture.py" "$D" "$W" debug  --run B1 --ranks 1
python3 "$P/replay.py" "$W"
modes=(point-{D2,T2,D1,T1,B2,B1} live-{D2,T2,D1,T1,B2,B1} ordered-{D2,T2,D1,T1,B2,B1})
for role in donor target; do
  python3 "$P/capture.py" "$D" "$W" "$role" --run "replay-$role" --ranks 2 --modes "${modes[@]}"
done
python3 -O "$P/replay.py" "$W" --analyze "$W/score-audit.csv"
python3 -O "$P/replay.py" "$W" --stage-controls "$W/substitution.csv"
python3 -O "$P/endpoints.py" "$D" "$W/endpoints.csv"
```

The target production reading order is the pre-turn selector, the moment-to-score calculation and the signed resistance root. The observer and offline score are confined to this investigation.

```text
source/simulator/solver/midpoint_direction.cc
include/aspect/reconstructed_fault/onset_score.h
source/material_model/growing_fault/section_response.cc
refactoring/phase-field-fault/strip8/capture.py
refactoring/phase-field-fault/strip8/replay.py
refactoring/phase-field-fault/strip8/endpoints.py
```
