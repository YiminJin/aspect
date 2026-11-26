# Growing-seed figure review checks

The regenerated default shows the accepted phase at 5 ms and the initial
φ = 0.05 contour. The red segment runs from station 7 at
(1.64433757, 0.87200847) m to station 8 at (1.87758764, 0.96401879) m.
The slope between these endpoints gives 21.527781613° from +x, a clockwise
turn of 8.472218387° from the 30° seed. Its length is 0.2507418869343099 m.
In this uniform-strip case the station spacing along the seed equals the
strip width, so the segment length matches the accepted strip arclength.
When the segment starts at the tip its length can differ from the full
strip arclength. The caption describes initiation in the first interval.

| Review finding | Edited accepted output | Measurement from the figure |
| --- | --- | --- |
| 1. Signed annotation | Change extension from +0.003 to −0.003 s⁻¹ | +8.472263747°, labeled 8.47° counterclockwise. Absolute direction 38.472263747°. |
| 2. Starting knot | Extension +0.012 s⁻¹ | Strip 7 turns at the first station beyond the tip. The segment starts at “Prescribed tip.” |
| 3. Equal-scale detail | Same +0.012 s⁻¹ run | The arrival at −0.115647853 m and both endpoint labels fit inside the detail. Its vertical limits are −0.171771749 and +0.085405928 m. |
| 4. Phase in a translated box | Add 10,000 m to the x coordinates | Both native frames pass. The largest absolute phase differences were 8.17 × 10⁻¹³ initially and 8.31 × 10⁻¹³ at 5 ms. Interpreting the same coefficients in a nodal Q2 basis fails on all 144 native observations of the first cell, with a 0.064765 maximum absolute error and a 3.29 × 10⁻¹² allowance. |
| 12. Legacy phase selection | Name the composition `damage` and select it under `Postprocess / Phase field surface energy` | Both frames render. The accepted strip-8 turn in this one-rank run is −8.472218854°. |

The plotter evaluates the tensor-product quadratic Bernstein expansion in
each physical cell. At each exported native point the comparison value comes
from ASPECT FEValues on that accepted composition, as written by the growing
history reporter. Inverting the physical coordinate map introduces a local
uncertainty proportional to the coordinate ulp divided by the cell width.
For each direction, twice the largest adjacent coefficient difference bounds
the derivative on the unit cell by the Bernstein convex-hull property.
The checker propagates two coordinate ulps in each direction through these
slopes and retains the original 10⁻¹³ arithmetic floor for the untranslated
8 × 4 box. The independent phase, Stokes and strip acceptance tolerances in
the physics comparisons are unchanged.

## Commands

Run from the worktree root. The number of MPI ranks for the fresh default is
two with one numerical thread per rank. The script below also reads the
reviewer's edited accepted fields, which need only one plotting process.
The regression inspects the signed text and the *drawn* positions of both
markers and annotations. Its negative control evaluates nodal polynomials
on the unchanged translated coefficients. The installed FE solver and its
accepted frames are unaffected by that in-memory substitution.

```sh
prefix=$PWD/build-d-review/install
plot=$prefix/benchmarks/growing_fault/plot_seed.py
figures=$PWD/refactoring/phase-field-fault/results/iposter
review=/tmp/reviewD.LsOSQowE
seed=/tmp/aspect-d-repairs.TphOTorc/growing/seed/output
scratch=$(mktemp -d)
export MPLCONFIGDIR=$scratch MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
python3 -BO "$plot" "$seed" "$figures"
python3 -BO "$figures/check_growing_plot.py" "$plot" "$seed" \
  "$review/reverse/output-growing-seed" "$review/large/output-growing-seed" \
  "$review/geometry/translated/output" "$review/replot-legacy/output-growing-seed"
for edit in reverse large replot-legacy; do
  python3 -BO "$plot" "$review/$edit/output-growing-seed" "$scratch/$edit"
done
python3 -BO "$plot" "$review/geometry/translated/output" "$scratch/translated"
pdftotext -layout "$scratch/reverse/growing-seed.pdf" -
```

The edited outputs can be regenerated with the installed growing-seed input.
For each signed-load control, change only `extension` in the boundary
velocity function constants. For the legacy case, rename the composition
to `damage`, clear the public `Phase field / Phase composition`, and set
the legacy selector to `damage`. Use the ordinary translated run in
`benchmarks/growing_fault/check_seed_frame.py` for the coordinate control.
Pass these output directories in the same order to the regression above.
