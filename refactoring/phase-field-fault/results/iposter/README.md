# Assignment D poster figures

## Growing seed, 0–5 ms

**Caption.** During the first 5 ms at 200 Pa compression, strip 8 selected a direction 8.47° clockwise from the 30° seed, or 21.53° counterclockwise from +x. Blue shows the accepted phase field. The dashed line marks the initial seed at φ = 0.05. The red segment joins accepted material stations 7 and 8. The detail panel shows this segment in seed coordinates with equal distance scales. The measured angle describes initiation over one timestep.

The figure is available in `growing-seed.png` and `growing-seed.pdf`. The input was
`benchmarks/growing_fault/growing_seed.prm` with its output directory set to
`output`. Both the one-field example and the four-field control used the 8 × 4
mesh on two MPI ranks with one numerical thread each.

The plot evaluates the nine exported Bernstein phase coefficients in each
cell at 0 and 5 ms, checking the reconstruction against every native phase
sample by cell and quadrature index. The phase contours are cellwise, so a
DG discontinuity stays at its original face. The red bend follows the
station-to-station construction of `MaterialSections::accepted_graph`.
Starting at the input tip, advance along the seed tangent to the next
formed station and accumulate the transverse displacement using the
arriving station's signed angle. The accepted strip table gives an along-seed
spacing of 0.248005645 m and an angle of −0.14786810579423598 rad at
station 8. The transverse displacement is −0.036941762 m. The published
strip length of 0.25074188693430988 m agrees with the seed strip width
divided by the cosine of that angle. The nine-strip angles, lengths, rates,
resistances and histories, all four stress components at 4,968 matching
native points and the energy and work terms agree exactly with the corrected
four-field control at 5 ms.

## Layered fault, 0–25 ms

**Caption.** The same 20 material vertices are shown at 0, 12.5 and 25 ms on the measured Maxwell viscosity profile. Vertex 9 moved 52.15 mm to the right, 2.15 mm beyond the 50.00 mm mean-wall translation. The right panel subtracts that common motion. Vertex 0 lagged it by 3.33 mm and vertex 19 led it by 7.74 mm. The map uses equal physical distance scales and the true vertex positions.

The figure is available in `layered-motion.png` and `layered-motion.pdf`. The input was
`benchmarks/layered_moving_fault/layered.prm` with its ordered
`layered_vertices.txt` table and benchmark plugin, on one MPI rank with one
numerical thread. The background interpolates the measured column viscosity
in `background-0.csv`. The map connects accepted coordinates by material ID
in each frame. The right panel subtracts the shared wall velocity (2 m/s)
multiplied by the elapsed time from each vertex's measured horizontal
displacement. The end-to-end difference in horizontal displacement was
11.065447 mm at 25 ms.

## Reproduce the figures

The [implementation guide](../../IMPLEMENTATION.md) gives the two installed
ASPECT launches. For the recorded run, the checkout started at
`3617163671e7aadbe52385b6f11de946db4a5283`. The updated parameter files
and benchmark provider in this assignment were installed into
`build-assignment-d/install`. The Release executable's SHA-256 was
`d75d72aec4714266631244fbe621fb2a007388c0c85abc5197e6c26517792edc`,
and the layered plugin's was
`63aeeaa3f50762360db262d3842605b8531cde2fb6340b824e1622d730ef5862`.
The executable reports its configure-time Git revision `1f0f5b413`.

From the worktree root, the following commands render both figures from the
retained external run directories. For a new run, substitute the output
directories produced by the two example launches. The scripts read the
accepted frames, the resolved parameter values and the physical times in
those directories. The 350 dpi PNGs are 4175 × 1713 and 4259 × 2025 pixels
respectively. Both PDFs contain vector geometry and text throughout.

```sh
run=/tmp/aspect-iposter.UpyGyZ
prefix=$PWD/build-assignment-d/install
figures=$PWD/refactoring/phase-field-fault/results/iposter
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
python3 -B "$prefix/benchmarks/growing_fault/plot_seed.py" \
  "$run/growing/seed/output" "$figures"
python3 -B "$prefix/benchmarks/layered_moving_fault/plot_layers.py" \
  "$run/layered-public/output-layered-moving-fault" "$figures"
```
