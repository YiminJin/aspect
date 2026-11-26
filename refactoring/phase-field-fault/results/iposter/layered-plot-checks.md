# Layered poster checks after Assignment D review

The installed plotter selects one accepted frame at each of 0, 12.5 and
25 ms, with the existing absolute clock tolerance of 10⁻¹⁴ s. Each rank's
surface table contains all material vertices, so the plot reads rank zero,
orders coordinates by persistent material ID and matches the IDs across the
three frames. The viscosity comes from the initial column quadrature data.
The frozen donor checker still reconstructs and verifies the bulk solution.

| One-rank calculation | Bulk cells | Plotted steps | Material IDs | Highlighted IDs | Middle horizontal displacement at 25 ms |
| --- | ---: | --- | ---: | --- | ---: |
| Installed default | 192 | 0, 1, 2 | 20 | 0, 9, 19 | 52.149417217 mm |
| Vertex inserted at the origin | 192 | 0, 1, 2 | 21 | 0, 10, 20 | 52.418734671 mm |
| Initial global refinement 2 | 48 | 0, 1, 2 | 20 | 0, 9, 19 | 52.155114807 mm |
| 6.25 ms steps | 192 | 0, 2, 4 | 20 | 0, 9, 19 | 51.497851682 mm |

The previous plotter rejected the second input on its 20-label comparison,
the third on its 192-cell reconstruction and the fourth on its three-entry
clock comparison. Decimal subtraction of the three marked default vertices'
exported coordinates gives 46.670840622, 52.149417217 and 57.736287639 mm.
Subtracting 2 m/s × 0.025 s gives −3.329159378, +2.149417217 and +7.736287639 mm.
These are horizontal material displacements. The initial trace spans the
viscosity transition at y = 0 m. The 512 measured background viscosities span
1.000098617–9.999901383 Pa s and agree with 5.5 + 4.5 tanh(y / 0.35 m) in Pa s
within 1.78 × 10⁻¹⁵ Pa s. The rendered PDF shows the equal-scale map, its
viscosity color bar, three labeled times and the three marked displacement
tracks. The regenerated default PDF and 4259 × 2025 PNG match the original
files byte for byte.

### Reproduce

These commands use the destination-configured SDK, ASPECT executable and
matching layered plugin. From the worktree root:

```bash
W=$PWD
I=$W/build-d-review/install
L=$I/benchmarks/layered_moving_fault
P=$W/build-d-review/layered-plugin/liblayered_moving_fault.release.so
C=$W/refactoring/phase-field-fault/results/iposter/check_layered_plot.py
S=$(mktemp -d)
export MPLCONFIGDIR=$S/mpl MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export DEAL_II_NUM_THREADS=1 ASPECT_SOURCE_DIR=$I
cd "$S"
for name in default moreids coarse shorter; do
  mkdir "$name"
  printf 'include %s/layered.prm\nset Additional shared libraries = %s\n' "$L" "$P" > "$name/input.prm"
done
awk 'NR==1 || $1<10 {print;next} $1==10 {print "10 0 0 10 3.85"} {$1++;print}' \
  "$L/layered_vertices.txt" > moreids/extra.txt
printf 'subsection Material model\n subsection Phase field fault\n set Fault vertex file = %s/moreids/extra.txt\n end\nend\n' "$S" >> moreids/input.prm
printf 'subsection Mesh refinement\n set Initial global refinement = 2\nend\n' >> coarse/input.prm
printf 'subsection Time stepping\n subsection Function\n set Function expression = 0.00625\n end\nend\n' >> shorter/input.prm
for name in default moreids coarse shorter; do
  (cd "$name" && "$I/bin/aspect-release" input.prm > run.log)
  python3 -BO "$L/plot_layers.py" "$name/output-layered-moving-fault" "$name/figures"
done
out=output-layered-moving-fault
python3 -BO "$C" "$L" "default/$out" --steps 0 1 2 --vertices 20
python3 -BO "$C" "$L" "moreids/$out" --steps 0 1 2 --vertices 21
python3 -BO "$C" "$L" "coarse/$out" --steps 0 1 2 --vertices 20
python3 -BO "$C" "$L" "shorter/$out" --steps 0 2 4 --vertices 20
```

The recorded default output is in
`/tmp/aspect-d-repairs.TphOTorc/layered/output-layered-moving-fault`.
The three edited calculations are in
`/tmp/layered-plot-repair.UOKxZXma/solver-{moreids,coarse,shorter}`.
The installed plotter also rendered all three of the reviewer's accepted outputs
from `/tmp/reviewD.LsOSQowE`. For each, both image formats matched the new
calculation. The focused checker copies just the clock and surface files into
a temporary directory, reverses the middle frame's row order and compares
every selected coordinate independently by ID. A missing or duplicate vertex
at 25 ms and a missing or ambiguous 25 ms clock each produced the expected
diagnostic on all four inputs.
