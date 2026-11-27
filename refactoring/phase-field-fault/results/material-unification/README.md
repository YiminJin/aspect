# One material for prescribed traces and growing strips

Both [the public curve](../../../../benchmarks/fixed_fault/curved_mature/curved_mature.prm) and [the growing seed](../../../../benchmarks/growing_fault/growing_seed.prm) select `Material model / Model name = phase field fault`. ASPECT's ordinary factory now constructs the same concrete `MaterialModel::PhaseFieldFault<2>` for those inputs and for the historical `growing fault` name. The [installed material probe](parameters.md) checks exact dynamic type and independently exercises each material output. The latter name still requires the midpoint field method and keeps its historical treatment of Simple viscosity.

## Maintainer repair and replay

The repaired production and SDK sources are at `82aee1ed3`. The local Release
executable was built before committing that delta and prints the parent
`754e62c4d`. Its SHA-256 is
`bace6a44dbf6522c9d3aed7a6d46fca942ef18b5127b6b1b5af6275cdc182fc7`.
The [installed parameter tests](parameters.md), [mature observer tests](mature.md#maintainer-review-of-the-mature-observer)
and [old-parent replay](growing-and-restart.md#verification-on-the-candidate)
give the SDK build commands and individual controls.

| Input or comparison | Measured result |
|---|---|
| Raise the growing seed's top by 0.02 m with a 50 Pa/m pressure gradient | The reviewed binary accepted 5 ms. Strip 8 used y = 0.969227162 m and 202.538641908 Pa. Using the actual top face would put the station at y = 0.979227162 m and give 202.038641908 Pa. The repaired parser identifies the initial topography model before any native frame. On the ordinary flat box the two-rank affine case used 201.649753019 Pa at depth 0.032995060 m on strip 8. |
| Select midpoint phase with an SDK material that only overrides trace viscosity | The old binary used a homogeneous 10⁸ Pa s despite the 20 Pa s override. The repaired binary diagnoses the selector. The same override on a mature trace matches the ordinary 20 Pa s control at all three clocks. An explicitly admitted strip extension honors the Simple viscosity alias. The historical growing and dashpot selectors keep their original semantics. Calling `sections()` in observer initialization now gives the documented lifetime diagnostic. |
| Restore each formulation's generation 02 from the pre-unification writer | The growing 10 and 15 ms output tables matched byte for byte. The checker rejected a reversal of signed shear and a loss of absolute shear at native point (`20_0:`, 138), with allowances of 5.64 × 10⁻²⁶ and 4.84 × 10⁻²⁵. The mature 25 ms public and observer label files were identical to the reference. Only the observer's stress offset changed at roundoff, at most 1.39 × 10⁻¹⁷ Pa. |
| Observe a 24 Pa s curved trace, G = 17 Pa, initialization interval 0.025 s | The old observer reported 10 Pa s and its initial stress offset was wrong by 0.469 Pa. The corrected observer gave 24 Pa s at all vertices and quadrature points and a maximum initial offset error of 2.22 × 10⁻¹⁶ Pa. The age-free SDK dashpot produced all five material IDs and the age-free CSV header at both clocks. The ordinary curve retained its RSF header and passed the frozen public check. |
| Archive or launch a calculation | [The public instructions](../../IMPLEMENTATION.md#run-the-two-poster-examples) use the selected installation and explain which source inputs to save. Replaying the expanded growing deck passed syntax validation and then diagnosed the explicit mature-only stress-transport setting at startup. The [BP3 readiness assessment](../bp3-preparation/README.md#branch-readiness-for-a-fresh-start) specifies a branch-owned start from time zero. The 160-rank donor recovery remains in the historical runbook. |

Run from the checkout with a fresh destination. The following commands used
the retained pre-unification reference capture and its verified donor replay.
Set `O` and `DONOR_OUTPUT` to the corresponding directories for a new replay.
Each rank used one numerical thread.

```bash
W=$PWD
D=$W/build-unification-review
I=$D/install
G=$I/benchmarks/growing_fault
R=$W/refactoring/phase-field-fault/results/material-unification
O=$W/build-unification
DONOR_OUTPUT=$W/build-5a-repair/donor-two/output
GP=$D/law-plugin/libgrowing_fault_checks.release.so
L=$D/layered-plugin/liblayered_moving_fault.release.so
export ASPECT_SOURCE_DIR="$I" OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPIEXEC=/opt/openmpi/5.0.6/bin/mpirun
cmake -S . -B build-6b/release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$I" -DASPECT_INSTALL_EXAMPLES=ON
cmake --build build-6b/release --target aspect -j3
cmake -DCMAKE_INSTALL_LOCAL_ONLY=ON -P build-6b/release/cmake_install.cmake
cmake -S "$G" -B "$D/law-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/law-plugin" -j1
cmake -S "$I/benchmarks/layered_moving_fault" -B "$D/layered-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/layered-plugin" -j1
(cd /tmp && python3 -BOu "$G/check_interface.py" "$I/bin/aspect-release" "$GP" "$D/final-interface")
bash "$W/refactoring/phase-field-fault/results/iposter/check-growing.sh" "$I" "$D/growing-two"
python3 -B "$R/cross_build.py" "$I" frames "$O/starting-growing" "$D/growing-two"
python3 -B "$R/cross_build.py" "$I" continue "$O" "$L" "$D/old-parent-continuations"
(cd "$O" && sha256sum --quiet -c "$R/starting-frames.sha256" && sha256sum --quiet -c "$R/starting-parents.sha256")
python3 -BOu "$G/run.py" "$I/bin/aspect-release" "$GP" "$D/nine" --ranks 2 --principal-only --mpiexec "$MPIEXEC"
python3 -BO "$G/compare.py" "$DONOR_OUTPUT" "$D/nine/principal/output"
MPLCONFIGDIR="$D/mpl-seed" MPLBACKEND=Agg python3 -B "$G/plot_seed.py" "$D/growing-two/seed/output" "$D/seed-figure"
cmp "$D/seed-figure/growing-seed.png" "$R/../iposter/growing-seed.png"
```

To reproduce the topography failure on the reviewed executable after the
interface check, run its raised-top input in a fresh directory:

```bash
mkdir "$D/before-replay"
(cd "$D/before-replay" && "$O/install/bin/aspect-release" "$D/final-interface/raised-top/input.prm")
tail -1 "$D/before-replay/output/friction-pressure-0001.csv"
```

All 35 fresh constant-pressure, affine and four-field tables matched the
pre-unification capture byte for byte. The nine-strip donor comparison
passed at 5, 10 and 15 ms, ending at 1.69302306477 × 10⁻⁴ m² integrated
slip. The [5 ms growing figure](../iposter/growing-seed.png) shows the
accepted phase and the 21.527781613° inclination at strip 8. Its PNG
and the [25 ms public curved solution](../mature-release/curve.png)
each reproduced the tracked figure byte for byte.

Four SDK CTests passed (`dg_stress_spin_order`, `dg_stress_transport`,
`fixed_fault` and `stationary_fault`). The Release unit suite passed
2,605 assertions in 64 cases. The same-source fitted ridge reproduced all
38 label histories within the frozen Q8 budgets (maximum rate difference
1.31 × 10⁻¹⁵ m/s). The three-label paired work was 0.494235968292736 m on
the Stokes side and 0.494235968292737 m on the section side.

```bash
cmake -S tests -B "$D/tests" -DAspect_DIR="$I/lib/cmake/Aspect" \
  -DASPECT_BINARY="$I/bin/aspect-release" -DCMAKE_BUILD_TYPE=Release \
  -DASPECT_COMPARE_TEST_RESULTS=ON -DASPECT_RUN_ALL_TESTS=ON
ctest --test-dir "$D/tests" -j1 --output-on-failure \
  -R '^(fixed_fault|stationary_fault|dg_stress_transport|dg_stress_spin_order)$'
(cd build-6b/release/unit_tests && ASPECT_SOURCE_DIR="$W" "$I/bin/aspect-release" --test)
```

## Source and state ownership

1. The [simulator constructor](../../../../source/simulator/core.cc) calls the release material factory, gives the selected material simulator access and calls its parameter reader. In [PhaseFieldFault](../../../../source/material_model/phase_field_fault.cc), the method on the **named phase composition** selects trace or strip mechanics. The canonical registration declares both sets of inputs. The historical name supplies the same factory and declares no additional inputs.
2. For a trace, the material allocates the [Stokes coupling](../../../../source/reconstructed_fault/stokes_coupling.cc). It retains the EOS, layered viscosity and exterior-stress overrides, and trace-profile reference binding. The Q1 section equations consume the bulk traction and return a signed weak stress load to Stokes at the existing Q8 transfer points. The returned velocity and rates pass the Picard convergence check before the acceptance callback advances labels, law state and DG stress. Adiabatic friction samples pressure at the physical foot on the trace.
3. For growth, the material allocates [StripMaterialHistory](../../../../include/aspect/material_model/phase_field_fault/strip_material_history.h). Ordinary material evaluation returns the growing Maxwell viscosity, unit density and heat capacity, and zero conductivity. The simulator calls [advance_midpoint](../../../../source/simulator/solver/midpoint_fault.cc). It reads the accepted native tensors, constructs pressure stations on the accepted graph and solves the signed strip balance with stage-aged friction. The direction search, native quadrature, FE assembly and obstacle solve remain with the solver. Only the solver stages a returned endpoint on the material. Only post-acceptance advances the law and publishes the tensors, strip records and work ledger. Restore rebuilds geometry on the restored mesh before pre-solve observers.
4. The material exposes const strip state to the growing observer and trace state to the mature observer. A retry starts from the accepted parent. Each active growing rank holds its own native tensor records and a replica of the nine strip records for this case. The refactor adds one owned strip object per rank, with the existing accepted and trial storage; it adds no native tensor copy or collective. The split transfer and its 3.97675% bend deficit remain part of the scientific reference.

The release material plugin pattern also serves the Simple model with its material-owned equation of state. The strip component directly owns its quadrature state and versioned archive. A shared RSF change belongs in [the resistance implementation](../../../../source/material_model/rheology/rate_and_state_friction.cc). A strip-history wire change belongs in [the strip records](../../../../include/aspect/material_model/phase_field_fault/strip_material_history.h) and [their archive](../../../../source/material_model/phase_field_fault/strip_material_history.cc). The [polyline initializer](../../../../source/initial_composition/fault_polyline_gaussian.cc) owns the initial curved phase.

On the qualified binary, 31 public/legacy interface cases, 23 parameter, identity and admission cases, and 14 material-setting cases passed. Both wrong-output selections, both cross-formulation continuations and both incompatible-law restorations produced their intended diagnostics. The stock Simpler calculation also continued its old checkpoint. Commit `c7f2eaa57` contains the installed checks and the independent field and continuation comparator.

## Changes at the ASPECT boundary

| Release file or interface | Reason and control |
|---|---|
| `Simulator` construction in `source/simulator/core.cc` | Restores `MaterialModel::create_material_model<dim>(prm)`. Field methods select mechanics inside the material. The exact-type test distinguishes the former factory substitution. An ordinary Simple model and a phase-only calculation exercise the original factory path. |
| Material plugin declarations in `source/material_model/interface.cc` | One call after all plugin declarations attaches an action to Simple viscosity. Registration order does not guarantee the Simple entry exists while the fault declares its inputs. An action registered on the declared entry distinguishes an omitted value from an explicit assignment equal to the default. The [explicit-default and external-law tests](parameters.md) cover this. Reconciliation and numeric comparison are in [the fault input code](../../../../source/material_model/phase_field_fault/input_compatibility.cc). |
| Branch midpoint, Picard and publication interfaces | The existing midpoint dispatch validates the active strip formulation, Picard handles the trace coupling, and the simulator signals acceptance after its repeat decision. This repair edits the branch midpoint solver's provider access. It adds no shared solver hook. The [retry, restart](growing-and-restart.md) and [Stokes work](mature.md) controls exercise their consumers. The [release-to-branch inventory](../readability/README.md#where-the-branch-meets-aspect) documents the earlier FE selection, constraint, checkpoint and signal changes. |

## Reproduce the two installations and measurements

The starting revision was `6466cfe6130e5ea2d1324466b4c530b917247961`. Before ownership edits, the public two-rank nine-strip cases and serial checkpoint parents ran with its Release executable. Commit `485f4cabb` records the hashes for 56 reference input/output files and 10 checkpoint files. The [growing and pre-refactor restart commands](growing-and-restart.md) describe the actual inputs and destinations. For mature mechanics the [public frozen trajectory](../mature-release/public-baseline.md) remains the independent scientific reference at `ae676e24095909196f885073b1146babce50292b`.

Commit `6df84149d` relocates input compatibility. Commit `4472fbcf6` unifies the material. On this machine the Release build reused `build-6b/release` with deal.II 9.6.2 and separate installation prefixes. For a new replay the [capture recipe](growing-and-restart.md#reference-capture) verifies the starting source and assigns `D` to a fresh directory. From the candidate checkout, configure a separate build under that same `D`:

```bash
W=$PWD
: "${D:?Use the fresh directory from the reference capture}"
B=$D/candidate-build
I=$D/install
cmake -S "$W" -B "$B" -DCMAKE_BUILD_TYPE=Release -DASPECT_INSTALL_EXAMPLES=ON \
  -DDEAL_II_DIR=/opt/dealii/9.6-local -DCMAKE_INSTALL_PREFIX="$I"
cmake --build "$B" --target aspect -j2
cmake -DCMAKE_INSTALL_LOCAL_ONLY=ON -P "$B/cmake_install.cmake"
sha256sum "$I/bin/aspect-release"
```

Writer binary SHA-256 `6b39e9d4d5807efa3caffe0bc0793770a7e1c876521588698ab22a2c2ab8307b`. Candidate binary SHA-256 `ad5b91587404af2409ed7e30b0d5f0933c35bbcebce828d2ff3e177de2700930`. The candidate SDK plugins were rebuilt in separate build directories against its configured prefix. The [mature replay](mature.md), [growing replay](growing-and-restart.md) and [parameter and admission tests](parameters.md) give commands, tolerances and results. The final executable and the installed examples can be used outside the source checkout.

The changed fitted-ridge extension also passed its installed 32² control. Its 38 label histories at all three clocks met the frozen Q8 budgets. Maximum rate difference was 1.31 × 10⁻¹⁵ m/s and the normalized work difference was 2.78 × 10⁻¹⁷. To reproduce:

```bash
I=$D/install
P=$I/benchmarks/fixed_fault/phase_ridge
export OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ASPECT_SOURCE_DIR="$I"
cmake -S "$P" -B "$D/ridge-plugin" -DAspect_DIR="$I/lib/cmake/Aspect"
cmake --build "$D/ridge-plugin" -j1
E=$(mktemp -d)
python3 -BO "$P/run.py" "$I/bin/aspect-release" "$D/ridge-plugin/libphase_ridge.release.so" "$E/free-32" free 32
PYTHONPATH="$P" python3 -B - "$E" <<'PY'
from pathlib import Path
import sys
from check import check_case
check_case(Path(sys.argv[1]), 'free', 32)
PY
```

**Figure.** In the [regenerated growing-seed plot](../iposter/growing-seed.png), the accepted FE phase and strip stations give a laboratory inclination of 21.527781613° at 5 ms under 200 Pa compression. Both executables produced the same PNG and PDF. The plotting command is in the [growing record](growing-and-restart.md). The [public curved figure](../mature-release/curve.png) also reproduced byte for byte.

These fixtures cover small plane-strain mature and growing calculations. The historical four strip-15 comparisons retain their original 18-strip scope and tolerances. Curved-seed propagation, modified bend normalization and growth under gravity need separate scientific qualification.
