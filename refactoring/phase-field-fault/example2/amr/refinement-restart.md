# Growing-fault mesh resolution and restart

On the loaded 30° test, changing the maximum bulk edge from 0.25 to 0.125 m changed the mesh after the 10 ms solve from 77 to 128 cells. Both runs started from the same 5 ms checkpoint. All 19 physical strip edges, 18 material IDs, ages and signed slips agreed exactly at both clocks. This check used the tip-disk mesh criterion on a scheduled pass, with the timestep predictor disabled. [Loaded input](../../../../benchmarks/growing_fault/loaded_remesh.prm) · [Installed-run checker](../../../../benchmarks/granite_fault_growth/amr/check_refinement_contract.py)

## Where the geometry comes from

| Quantity | Physical owner and consumer |
| --- | --- |
| Direction kernel | The onset score owns the kernel radius of two phase lengths. Midpoint mechanics freezes this disk about the accepted, phase-supported mechanical tip. AMR adds a trigger margin and a replenishment margin, both measured in phase lengths. Their default value of one gives trigger and restored radii of three and four phase lengths. Zero trigger margin is admissible. Replenishment must add positive clearance. [Direction parameters](../../../../include/aspect/reconstructed_fault/onset_score.h) · [Direction solver](../../../../source/simulator/solver/midpoint_direction.cc) |
| Forward sector | Prospective planes range from −π/4 to +π/4 about the seed tangent. The clipped AMR sector uses that same limit. The 128 angular intervals and local search accuracy belong to mechanics. The crack law independently supplies the normal half-width used for the formed band and sector padding. [Marker](../../../../source/mesh_refinement/growing_fault.cc) · [Crack laws](../../../../source/phase_field/crack_geometry.cc) |
| Seed and mechanical tip | Material strip history supplies the seed point and unit tangent. AMR follows the incoming tangent back one phase length, clipped to the box wall, to locate the incoming seed segment in physical space. In mechanics the original discrete anchor is the first strip midpoint at or above the projected seed coordinate minus one phase length. The contiguous formed graph and maximum incident DG trace at 0.05 determine the supported mechanical tip. The independently owned 0.01 AMR contour describes damage requiring fine cells. [Strip history](../../../../source/material_model/phase_field_fault/strip_material_history.cc) · [Damage contour](../../../../include/aspect/mesh_refinement/fault_damage.h) |
| Material station width | MaterialSections divides the projected box extent into equal physical intervals with width at most the requested strip spacing. Edges and their ordering give each strip its identity. The native history reader verifies the realized interval edges, seed angle and quadrature, then restores the records. It still reads native v1/v2 layouts. [Strip partition](../../../../source/reconstructed_fault/material_sections.cc) · [Checkpoint reader](../../../../source/material_model/phase_field_fault/strip_material_history.cc) |
| Bulk resolution | The mesh criterion uses the longest physical Cartesian edge for marking, reserve coverage and coarse-damage detection. Its zero input default inherits the requested strip spacing. A positive override changes only the bulk target. The reachable-level check on restore credits additional refinements already released by the restored clock, using the simulator’s cutoff. [Marker](../../../../source/mesh_refinement/growing_fault.cc) · [Simulator clock](../../../../source/simulator/core.cc) |

ASPECT's box uses a subdivided axis-aligned hyperrectangle. The growing material requires a fixed nonperiodic box with zero initial topography and stationary Q2/Q1 plane strain. MaterialSections checks the orientation and rectangularity of the actual cells. ASPECT uses deal.II’s distributed triangulation, which documents that p4est does not implement anisotropic refinement (deal.II 9.6, `distributed/tria.h`, lines 119–121). When transferring a native point, the material passes the actual parent refinement case to the child-to-parent coordinate map. This preserves the child coordinates on rectangular root cells as well as square ones. [Box implementation](../../../../source/geometry_model/box.cc) · [Native remap](../../../../source/material_model/phase_field_fault/strip_material_history.cc)

The same history owner checks positive station spacing, zero coarsening fraction and a minimum of four section Gauss points. On cut Cartesian cells the phase-weighted moments contain two affine factors, Q2 phase, and the chord Jacobian, giving degree at most seven along the outer integration coordinate. Scheduled midpoint runs check these inputs at startup. Activating the growing mesh criterion also checks them, including a timestep-controlled run with no scheduled refinement. The marker supplied to the timestep model owns the repeat decision. [Midpoint admission](../../../../source/simulator/solver/midpoint_fault.cc) · [Timestep model](../../../../source/time_stepping/growing_fault.cc)

## Installed checks

Commands below run from the worktree root after installing the reviewed executable, SDK and examples. The plugin compiles against that same install and supplies the test-only transfer observer. Each MPI process uses one numerical thread. Use fresh destination directories when repeating the runs.

```sh
prefix="$PWD/build-amr-review/install"
plugin="$PWD/build-amr-review/plugin"
cmake -S "$prefix/benchmarks/growing_fault" -B "$plugin" -DAspect_DIR="$prefix/lib/cmake/Aspect"
cmake --build "$plugin" --parallel 1
driver="$prefix/benchmarks/granite_fault_growth/amr/check_refinement_contract.py"
observer="$plugin/libgrowing_fault_checks.release.so"
OPENBLAS_NUM_THREADS=1 python3 "$driver" "$prefix/bin/aspect-release" "$observer" build-amr-review/policy/admission admission
OPENBLAS_NUM_THREADS=1 python3 "$driver" "$prefix/bin/aspect-release" "$observer" build-amr-review/policy/loaded loaded --ranks 2
```

The one-rank admission suite rejected 12 inputs. Both the scheduled and reaction-only variants rejected zero target strip spacing, order-three quadrature, nonzero coarsening and an unreachable bulk edge. The other four rejected an absent mesh criterion for the selected timestep model, zero replenishment clearance, an unrelated material and a maximum-level function which blocked the final initialized-seed refinement. The blocked pass produced the no-progress diagnostic before the first physical endpoint. Each failure was checked against its own diagnostic and publication clock.

| Two-rank loaded test | Measured result |
| --- | --- |
| Original native transfer after 5 ms | 32 → 77 cells, 12 migrated. Largest phase error 7.29 × 10⁻¹⁶. Largest stress integral error 3.55 × 10⁻¹⁵ Pa m² and pointwise stress reconstruction error 2.70 × 10⁻¹³ Pa. |
| Pre-solve restore at 5 ms in both target runs | Native records and ledger matched the saved parent exactly. Strip coordinates span [0, 2.232050807568877] m. Age range 2.524025433103816 × 10⁻¹¹ s and slip range 2.5233112356966437 × 10⁻¹⁴ m distinguish the loaded strips. |
| Scheduled remesh after 10 ms, bulk targets 0.25 / 0.125 m | 77 → 77 / 77 → 128 cells. Fine run moved 24 cells to another owner. Published IDs, realized edges, ages and signed slips agree by material ID. Each transfer preserved its accepted ledger byte for byte. |
| Fine-run independent reconstruction | Largest phase error 1.33 × 10⁻¹⁵, stress integral error 6.22 × 10⁻¹⁵ Pa m² and pointwise stress error 4.26 × 10⁻¹³ Pa. A cellwise constant stress return would miss by 43.7612 Pa. Measured fracture energy change 0.042639277211455 J/m, independent interior-face penalty change 0.042639277211472 J/m. |
| Checkpoint edge compatibility | Requested spacings of 0.125 and 0.126 m both give 18 identical strips and continuation passed. A request for 0.124 m produced 19 strips and failed. A separate scratch generation retained 18 strips and the identical physical inputs but shifted one archived edge by 0.001 m. It failed the physical-edge diagnostic. |
| Unreleased future level | At restoration the 0.006 s release had occurred. A 0.0625 m target needing the 0.1 s release failed the reachable-level check before the next solve. |

The established two-rank loaded AT1 fixture also matched its pre-repair trajectory. At 5, 10 and 15 ms the measured maximum difference was zero for every native column keyed by cell and quadrature point, all FE coefficients, strip columns and accepted balance. The comparison uses the previously retained parent run in the author’s worktree.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=benchmarks/granite_fault_growth python3 - <<'PY'
from pathlib import Path
from run_transfer_cases import trajectory
old = Path('build-granite-amr/transfer-final/parallel/output')
new = Path('build-amr-review/policy/loaded/parent/output')
for clock, result in trajectory(old, new, (1, 2, 3)).items():
    print(clock, max(result['maximum'].values()))
    assert not result['strict_failures'] and not result['angle_and_length_failures']
PY
```

### Final installed binary and retained checkpoint generations

The final Release build repeated the admission and loaded-target suites. Their result JSONs were byte-identical to the first two runs. We then reused the retained accepted serial, two-rank and stateless parents and ran fresh fine-mesh, order-four and checkpoint continuations. The finer test remeshed 128 → 308 cells, migrating 52. Continuing the same layout reproduced both physical endpoints exactly. Restoring on a different layout moved 37 cells and preserved the accepted values before solving. Every endpoint maximum by column in the same-layout, changed-layout, v1 and v2 comparisons matched the earlier measurements. All new cross-layout continuations passed the physical angle/arclength tolerances. The archived independent one-rank and two-rank fresh starts retain their known strip-16 differences of 2.36894e−7 rad and 1.75210e−8 m before remeshing. The tighter same-layout tolerance is not a claim of bitwise trajectory identity after repartition.

Native v1 passed the strict 15 ms continuation with maximum stress difference 6.25e−13 Pa. Native v2 passed at 10 and 15 ms, with maximum stress differences 4.55e−13 and 7.67e−13 Pa. The parent archives and donor tolerances were preserved.

```sh
r=$PWD/build-amr-review
exe=$r/install/bin/aspect-release
b=$r/install/benchmarks/granite_fault_growth
observer=$r/plugin/libgrowing_fault_checks.release.so
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 "$b/amr/check_refinement_contract.py" "$exe" "$observer" "$r/final-admission" admission
python3 "$b/amr/check_refinement_contract.py" "$exe" "$observer" "$r/final-loaded" loaded --ranks 2
python3 "$b/run_transfer_cases.py" "$exe" "$observer" build-bp3-fresh/install/bin/aspect-release \
  "$r/native-continuations" --parents build-granite-amr/transfer-final \
  --v1-generation build-5a-repair/lifecycle2/clean/output/restart/03
```
