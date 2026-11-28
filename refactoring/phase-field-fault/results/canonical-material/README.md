# A single material input for traces and propagating strips

Both examples select `Material model / Model name = phase field fault`. Set fault inputs inside `Material model / Phase field fault`. A single composition with method `midpoint fault phase` selects the strip formulation. An existing mature trace keeps its supported static or transported phase method. The choice of resistance law is independent of that dispatch.

* [96 × 48 granite with steady drive](../../../../benchmarks/granite_fault_growth/granite_fault_growth.prm) and [48 × 24 fixed-wall control](../../../../benchmarks/granite_fault_growth/granite_prestress_hold.prm). Keep the files together so the relative include resolves. [Native tensor measurements, figure and hold replay](GRANITE.md).
* [One-phase growing seed](../../../../benchmarks/growing_fault/growing_seed.prm) and [prescribed curved mature trace](../../../../benchmarks/fixed_fault/curved_mature/curved_mature.prm). The mature example inherits its material properties from [the common input](../../../../benchmarks/fixed_fault/fixed_fault.prm).

| Material setting | Owner and admission |
| --- | --- |
| Shear modulus, viscosity, friction normal stress, resistance law and its registered coefficients | Both formulations. For omitted values the mature/growing Maxwell defaults are 1e10/1e6 Pa and 5e24/1e8 Pa s. The omitted RSF age, direct effect, evolution effect, reference rate and damping are also formulation-specific. |
| Seed tangent angle, forward seed tip, target strip width, section quadrature order, cohesion, prescribed fault compression and initial deviatoric stress | Growing strips. The optional fresh-start tensor takes Cartesian xx, yy, xy in Pa and sets zz = −(xx + yy). Restart reads the accepted native tensors. |
| Fault geometry, vertices and vertex properties, slip sense, fit and profile controls, boundary-contact completion, stress advection and exterior stress, section diagnostics, initial interval and speed, background shear and compression | Mature traces. Background compression adds to the resolved traction. A growing strip can instead take a prescribed compression as the entire friction pressure. |
| Pressure and crack energy | Adiabatic pressure comes from ASPECT's adiabatic conditions model. The crack geometry, toughness and length scale stay in the Phase field section. Mesh refinement, wall loading and timestep control keep their ASPECT settings. |

For the built-in rate-and-state law, the mature/growing omitted values are 1/10 s for initial age, 0.01/0.015 for direct effect, 0.015/0.02 for evolution effect, 1e−6/0.001 m/s for reference speed and 0/1,000 Pa s/m for damping. Both use a reference coefficient of 0.6 and a slip distance of 0.01 m. An omitted growing tensor is zero, angle is 0°, tip is (0,0) m, strip width is mesh-derived, section quadrature order is 6 and cohesion is 1,000 Pa. A valid physical seed still needs its tip in the supported region.

The historical selector `growing fault` and material subsection `Growing fault` remain readable. On that selector, supplying Simple-model viscosity by itself never changed the growing Maxwell viscosity and still leaves it at 1e8 Pa s. The public selector admits the Simple viscosity as an alias. The earlier `Material model / Friction normal stress` location also remains valid. Duplicates are compared by declared type, including nested numeric lists. Supplying a formulation-specific setting explicitly, even at its default, requires that formulation.

## Follow the input to the accepted fields

1. `PhaseFieldFault::parse_fault_parameters` selects mechanics from the phase method and calls `FaultInputCompatibility::resolve`. The latter distinguishes explicit user entries from omitted route defaults and checks physical admission before material construction. [Material](../../../../source/material_model/phase_field_fault.cc) · [compatibility reader](../../../../source/material_model/phase_field_fault/input_compatibility.cc).
2. The mature reader builds the trace and its law. The ordinary Stokes iteration returns the fault-corrected stress. Read the accepted `fault_tau_xx`, `fault_tau_yy`, `fault_tau_xy` in the VTU and the five material-ID rows in each public label file.
3. The growing reader builds the strip Maxwell and resistance laws. Native-point history owns the initial stress and the later accepted tensors. Read `tauxx,tauyy,tauzz,tauxy` in Pa by `(cell,q)`, `sections-*.csv` by strip ID, and interval work in `balance-*.csv`. Both `phase field fault history` and the former postprocessor name write the same native format. [Strip material](../../../../source/material_model/phase_field_fault/strip_material_history.cc) · [postprocessor](../../../../source/postprocess/growing_fault_history.cc).

## Installation and comparison

The starting revision is `d6c5520759bcafc0d4834f9244507b7dbcce0ed6`. The reference is the pre-change local Release executable and examples taken from that revision. The installed reference executable's SHA-256 is `3d02d8b6edd6c09c22ef46fb4bce90d4e44250f9008836a449624e33bf792522`. Its embedded build banner says `8b6a0e7ff`. The hash identifies the retained AMR build; `d6c552075` identifies the checked-in inputs. The candidate also reads the old input itself to distinguish parameter compatibility from binary identity. The final candidate SHA-256 is `e299e1da1f59c993268a5f02fce2c28cb80050dcd37873e38f5f8b13adfe9355`. Keep the two installations and their outputs separate. Run from the worktree root with a fresh destination, setting `OLD` to this retained installation or an independently built checkout of the starting revision.

```sh
D="$PWD/build-fault-inputs"
I="$D/install"
OLD="$D/baseline"
R="$PWD/refactoring/phase-field-fault/results/canonical-material"
export DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -Bu "$R/compare_inputs.py" "$OLD" "$D/old-runs"

cmake -S . -B build-6b/release -DCMAKE_INSTALL_PREFIX="$I" -DASPECT_INSTALL_EXAMPLES=ON
cmake --build build-6b/release --target aspect -j2
cmake --install build-6b/release
export ASPECT_SOURCE_DIR="$I"
```

Generation 02 of the old growing checkpoint is the 5 ms parent (resume.z SHA-256 `ca93167eda6a73d1641f315406a0a770e83cb74c3f0b1d3b9af9b6cb15f9168c`). The corresponding old curved checkpoint is the 12.5 ms parent (`3c956ba71e5ba299d03823b3f8d3df26c6daaa0d426d66d25cdae698725e9051`). The continuation driver copies only that generation into a fresh directory. It also replays the writer's expanded original input through the candidate and compares against an uninterrupted run with the new input.

The historical executable was installed in the baseline destination **before** reconfiguring the build. Its two public inputs and their dependencies came from the following source tree:

```sh
git archive d6c552075 benchmarks/fixed_fault/fixed_fault.prm \
  benchmarks/fixed_fault/curved_mature/{curved_mature.prm,vertices.txt} \
  benchmarks/growing_fault/{growing_seed.prm,onecut.prm} | tar -x -C "$OLD"
```

Build the two test plugins from the installation against its own SDK. The ordinary test runs use a single rank with one thread. Granite uses two ranks.

```sh
ASPECT_TREE="$PWD"
B="$ASPECT_TREE/build-6b/release"
cd /tmp
cmake -S "$I/benchmarks/growing_fault" -B "$D/law-probe" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/law-probe" -j1
cmake -S "$I/benchmarks/growing_fault/parameter_compatibility" -B "$D/sdk-probe" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/sdk-probe" -j1
"$I/bin/aspect-release" --validate "$I/benchmarks/fixed_fault/curved_mature/curved_mature.prm"
"$I/bin/aspect-release" --validate "$I/benchmarks/growing_fault/growing_seed.prm"
python3 -Bu "$R/compare_inputs.py" "$I" "$D/release-frames" --reference "$D/old-runs"
python3 -Bu "$I/benchmarks/growing_fault/check_interface.py" "$I/bin/aspect-release" \
  "$D/law-probe/libgrowing_fault_checks.release.so" "$D/release-interface"
python3 -Bu "$I/benchmarks/growing_fault/parameter_compatibility/check.py" "$I/bin/aspect-release" \
  "$D/sdk-probe/libfault_parameter_probe.release.so" "$D/release-sdk"
python3 -Bu "$I/benchmarks/growing_fault/check_material_settings.py" "$I/bin/aspect-release" "$D/release-settings"
# A stock utility unit test reads VERSION from the source tree.
ASPECT_SOURCE_DIR="$ASPECT_TREE" "$I/bin/aspect-release" --test
ctest --test-dir "$B" -R '^growing_fault$' --output-on-failure
```

The final Release passed all 3,191 assertions in 66 unit cases. The `growing_fault` CTest passed in 15.88 s. The installed SDK probe exercised nested growing-only and mature-only keys as well as different defaults and admissible ranges for a shared coefficient.

## Accepted output and input admission

| Installed control | Observation |
| --- | --- |
| Mature curved trace, both inputs | At 0, 12.5 and 25 ms all five public history rows and the three accepted stress arrays match the previous executable exactly. At 25 ms ID 4 has 0.0040918276294517209 m slip and 0.16907176889700526 m/s speed. The continued 25 ms frame also matches. |
| Growing seed, both inputs | At 0, 5, 10 and 15 ms the differences in all 4,968 native records, nine strip rows and 1,280 FE coefficients are zero. The physical pressure and work differences are zero. All 18 scientific CSVs are byte-identical in both uninterrupted comparisons, including the historical positive-zero representation of the omitted τzz. All ten physical CSVs match in the continuation. At 15 ms strip 8 has a speed of 4.8865986755578738e−8 m/s. The old 5 ms parent reproduced the 10 and 15 ms frames. |
| Formulation admission | An explicit forward tip on a mature trace fails even at (0,0). The angle, width, quadrature order, cohesion, initial stress and prescribed strip pressure give the growing-only diagnostic there. A growing model rejects the normal integral subdivisions, normal filter, contact completion, convergence and traction reports, background compression, fit and mature-only SDK coefficient. These are material-setup checks. |
| Alias values | Equivalent stress and coordinate lists, including exponent spellings, give identical accepted records. Conflicting tip, stress, cohesion, modulus, viscosity, RSF, friction pressure source and nested SDK coefficients identify the public and historical paths. The default pressure source equals resolved traction for a trace and prescribed constant pressure for a strip. |
| Viscosity | With the historical growing selector, changing Simple viscosity alone leaves the 5 ms native shear at 75.978862635792 Pa. With the public selector, switching Maxwell viscosity from 1e8 to 1e6 Pa s changes that shear to 75.7912838976146 Pa. Doubling the public shear modulus gives 151.954052677592 Pa. |
| External resistance | The installed SDK plugin declares distinct defaults and disjoint valid impedance ranges at the same key. Before the range repair a growing run could reject its valid omitted default of 1,000 Pa s/m against the mature interval [1,500,2,500], even with ordinary RSF selected. Now the active route's bounds apply, and 750 Pa s/m yields measured strip force/speed of 750. The nested strip-only multiplier 2.5 yields 2,500 Pa s/m with either input spelling. Supplying 2,000 on growing or 750 on mature fails the active range diagnostic. Plugins using different, opaque custom validators for the same name can use the typed declaration helper to register both validators. [SDK contract](../../IMPLEMENTATION.md). |
| Restart, constitutive law and output | The installed interface checker also advanced stateless formulations and diagnosed wrong formulations, changed checkpoint law signatures, and incompatible reporters. The old spellings for the mature statistics and growing native-history exporters produced the accepted physical columns. |

The fixed-wall experiment measures the response of the prescribed prestressed seed over 60 µs. Its measured endpoint and the time-zero driven production observation are distinct from a positive-time driven propagation experiment.
