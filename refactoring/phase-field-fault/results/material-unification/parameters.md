# Installed factory, material properties and input compatibility

The material property test loads a small postprocessor from the installed resistance-law test library. At initialization and the first accepted step it checks the exact dynamic type of the material returned by the simulator. It calls the normal named-output creator and `evaluate` with a point at 320 K. The parameter file sets the Simple reference density to 12 kg/m³, reference temperature to 300 K, heat capacity to 127 J/(kg K), conductivity to 2.7 W/(m K) and expansion coefficient to 0.001 K⁻¹ on **both** formulations. Zero gravity would hide a change of density in a trajectory-only comparison.

| At the first physical step | Prescribed public curve | Public growing seed | Historical growing input |
|---|---:|---:|---:|
| Returned material | PhaseFieldFault<2> | PhaseFieldFault<2> | PhaseFieldFault<2> |
| Endpoint (s) | 0.0125 | 0.005 | 0.005 |
| Density (kg/m³) | 11.76 | 1 | 1 |
| Specific heat (J/(kg K)) | 127 | 1 | 1 |
| Conductivity (W/(m K)) | 2.7 | 0 | 0 |
| Thermal expansion (K⁻¹) | 0.001 | 0 | 0 |
| Viscosity (Pa s) | 0.12422199506118572 | 10⁸ | 10⁸ |
| Density dependence flag | temperature | none | none |

The independent trace density is 12(1−0.001(320−300)). Its viscosity is −10 expm1(−Δt) Pa s, with an initialization interval of 0.1 s. The probe also checks all five dependence flags, incompressibility, zero entropy derivatives and reaction terms. Supplying distinguishable stress values to the three trace compositional slots returned the named components `fault_tau_xx`, `fault_tau_yy` and `fault_tau_xy` in order. The strip material returned no named DG stresses. Its physical tensors are published by the growing history postprocessor. On the separately installed starting revision the same exact-type assertion passed on the curve and failed on public growth, which still constructed `GrowingFault::Material<2>`.

## Parameter declarations and admission

The canonical material registration declares both physical parameter groups. The historical material-name registration creates the same concrete type and has an empty declaration callback. The external weighted dashpot checks for duplicate declaration of its nested subsection in each group. ASPECT calls the registered resistance declaration with the real handler, including the current dimension. Equivalent `1,2` and `1.0,2.0` list spellings reproduced the same signed strip response. Different lists were rejected and the diagnostic named both fully qualified paths. At 5 ms, strip 8 had a force-to-speed ratio of 750 Pa s/m for the literal dotted parameter and 1,500 Pa s/m with nested weights. Both spellings matched.

The post-declaration hook in `source/material_model/interface.cc` performs one call to the fault-owned input compatibility code. ASPECT's `Simulator::declare_parameters` first declares the ordinary parameters, including their extension signal, and then the plugin managers. On this installed Release executable, breakpoints on `Simple<2>::declare_parameters` and `PhaseFieldFault<2>::declare_parameters` during validation of the growing seed showed Simple followed by PhaseFieldFault. The material plugin list invokes each declaration in registration order, which can vary with link order or the plugins loaded. This provides no portable ordering between the fault and upstream Simple declarations. The deal.II 9.6 and 9.7 parameter handler documentation places `add_action` after `declare_entry`. Its 9.6 public set-status query only reports missing *mandatory* entries. A value read through `get` cannot distinguish omission from explicitly setting the upstream default. The hook observes Simple viscosity after the full material list, without modifying its declaration. Declarations, observation, numeric reconciliation and conflict handling live with their owning material and laws.

* With the historical material name, adding a Simple viscosity setting reproduced the native 5 ms output byte for byte at both tested Maxwell viscosities.
* On canonical growth, omitting viscosity gave 10⁸ Pa s and a signed native xy stress of 75.978862635792 Pa at (cell 0_0:, q 0). Setting just Simple viscosity to 10⁶ Pa s gave 75.7912838976146 Pa. Supplying the upstream default of 5×10²⁴ Pa s together with a different explicit public viscosity produced the two-path conflict diagnostic.
* Doubling the public modulus from 1 to 2 MPa produced 151.95405267759796 Pa at the same key and clock. Canonical and legacy modulus spellings reproduced the same point, section, pressure and balance files. Conflicting values equal to the two historical defaults were rejected. The two mature viscosity spellings reproduced the same label histories at all three clocks.
* A midpoint field with Simple material, on a different composition from the selected phase, in duplicate, in 3-D or with an incompatible Stokes scheme failed at startup. The historical growing name without a midpoint field and the depth-dependent wrapper around the trace also failed. A fault-free affine AT1 calculation with Simple material and the external law library loaded returned 3.1171875 J/m.
* All 31 installed interface cases passed. Each crossed benchmark dashpot selector failed with the diagnostic for its own formulation. Both positive benchmark dashpot selectors reproduced their explicit-law counterparts. The two wrong-output selections and both cross-formulation restores failed. Restoring each RSF generation with just the resistance law changed to linear dashpot also failed. Growth diagnosed the changed physical parameter signature and the trace diagnosed the changed resistance-law name.

## Reproduction

Use the ordinary executables and separately built extensions. Run from the checkout root with the starting and candidate installations from the main report. The following commands use their installed example sources. Each run directory must be fresh. Qualification used one MPI rank and one numerical thread.

```sh
W=$PWD
D=$W/build-unification
I=$D/install
A=$I/bin/aspect-release
G=$I/benchmarks/growing_fault
P=$D/parameters-plugin
O=$D/starting-install/bin/aspect-release
export ASPECT_SOURCE_DIR="$I" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
cmake -S "$G/parameter_compatibility" -B "$P" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$P" -j1
cmake -S "$G" -B "$D/law-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/law-plugin" -j1
(cd /tmp && python3 -BOu "$G/parameter_compatibility/check.py" "$A" \
  "$P/libfault_parameter_probe.release.so" "$D/qualified-parameters")
(cd /tmp && python3 -BOu "$G/check_material_settings.py" "$A" "$D/qualified-settings")
(cd /tmp && python3 -BOu "$G/check_interface.py" "$A" \
  "$D/law-plugin/libgrowing_fault_checks.release.so" "$D/final-interface")
bash "$G/check_surface_pressure.sh" "$O" "$A" "$W/tests/simpler_box.prm" "$D/simple-control"
```

The candidate executable at `4472fbcf6935e991bad8e42130c435cc877e81de` has SHA-256 `ad5b91587404af2409ed7e30b0d5f0933c35bbcebce828d2ff3e177de2700930`. The starting executable at `6466cfe6130e5ea2d1324466b4c530b917247961` has SHA-256 `6b39e9d4d5807efa3caffe0bc0793770a7e1c876521588698ab22a2c2ab8307b`. The stock Simpler calculation sampled the correct surface pressure at eight upper-boundary points per run. Across 64 pressure samples, the maximum printed error in a 100 Pa shift was 2.84413 × 10⁻⁶ Pa. Its pre-refactor 100 Pa checkpoint continued on the candidate and rejected a change to 0 Pa.
