# A mature fault with a prescribed curved phase ridge

This small two-dimensional example runs from 0 to 25 ms in a 1 m square.
Five ordered material vertices define four fault segments. The same table
sets the initial phase ridge. To run from an installed ASPECT with examples,
set `ASPECT_SOURCE_DIR` to the installation prefix. From an empty directory:

```sh
export ASPECT_SOURCE_DIR=/path/to/installed/aspect
case_dir="$ASPECT_SOURCE_DIR/benchmarks/fixed_fault/curved_mature"
"$ASPECT_SOURCE_DIR/bin/aspect-release" "$case_dir/curved_mature.prm"
```

To move the bend, copy the table and edit the `y_m` entry in row 2, for
example from 0.69 to 0.57. Keep the IDs in file order. Point both consumers at
your copy with one override and run in a fresh directory:

```sh
cp "$case_dir/vertices.txt" ./vertices.txt
# Edit row 2 in ./vertices.txt.
cat > moved.prm <<EOF_PRM
include $case_dir/curved_mature.prm
set Output directory = output-moved
subsection Material model
  subsection Phase field fault
    set Fault vertex file = vertices.txt
  end
end
EOF_PRM
"$ASPECT_SOURCE_DIR/bin/aspect-release" moved.prm
```

At each interpolation point, the initial phase is
φ = 0.03 + 0.32 exp[−(d/w)²]. The distance d is the shortest Euclidean
distance to the finite segments. The closest point can be the bend itself,
or either boundary contact. `Gaussian width = 0.15` gives the e-fold distance
in metres on either side of an isolated segment. The finite-element field
interpolates this profile on a 16 × 16 DGQ2 mesh. The phase remains static
as stress and slip evolve. The separate `Profile half width` parameter sets
the admissible normal support for mechanical transfer. The fracture length
scale belongs to the phase functional.

The `background_shear_Pa` and `initial_age_s` columns set persistent loading
and initial age at each ID. The trace is prescribed, and `opposite tangent`
makes the positive-normal side slide up dip. Read the accepted positive RSF
speed and slip distance in `fault-labels-*.txt` and the accepted deviatoric
stress components in the `fault_tau_*` visualization fields. A difference
between two bulk velocities measured across a finite band depends on where
the samples are placed.

The separately named `vertical_gaussian.prm` reproduces the historical
manufactured profile measured in the vertical direction. Its two-sided
normal width decreases as the dip steepens.

## Compare a solver build with the public reference

The installed `public-reference` directory contains the serial 0, 12.5 and
25 ms results from revision `ae676e240`. With the matching `moving fault
samples` observer, compare accepted rows by material ID and native values by
cell ID and quadrature index. Supply the separate bare public output to also
check the files a user gets without an observer:

```sh
python3 -BO "$case_dir/check_public_reference.py" observed/output --public-output public/output
```

The public header is `# label x(m) y(m) age(s) slip(m) rate(m/s)`.
The last two columns are cumulative slip distance and the accepted endpoint
speed. The checker tests those meanings against the native label state and
checks that the speed integrated over each accepted timestep gives the slip
increment. It opens the reference read-only.
