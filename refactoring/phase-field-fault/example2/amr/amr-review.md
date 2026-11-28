# Fault AMR review closeout

The starting branch revision is `3c1ec279d`. The 48 × 24 steady-drive run and earlier AMR outputs are retained. This repair changes the decision to refine a physical cell. A connected trial excursion on an underresolved cell can now veto publication even when that cell's own phase did not change.

| Quantity | Owner and consumer |
|---|---|
| Equilibrium normal half-width (m) | The [crack geometry](../../../../source/phase_field/crack_geometry.cc) supplies the dimensionless profile integral. The [configured energy](../../../../source/phase_field/surface_energy_model.cc) multiplies by ℓ. The marker consumes the physical length for lateral protection and forward-sector padding. [Profile and installed energy tests](support-and-seed.md). |
| Direction-search radius and turn angle | The [direction score](../../../../include/aspect/reconstructed_fault/onset_score.h) supplies 2ℓ and π/4 to mechanics and AMR. Angular sampling, maximization and formation remain in the midpoint solver. |
| Trigger and replenishment | The [mesh criterion](../../../../source/mesh_refinement/growing_fault.cc) adds one ℓ by default to the mechanical radius for the trigger and another ℓ for replenishment. Thus the AT1 granite kernel, trigger and restore radii are 0.125, 0.1875 and 0.25 m. The normal profile half-width independently equals 0.125 m. |
| Material stations and bulk size | MaterialSections partitions the physical strip coordinate. The marker compares the longest physical cell edge with its bulk target in marking, reserve checks and damage screening, with a relative roundoff allowance of 1e−12. Zero inherits Target strip width. [Loaded-state and checkpoint checks](refinement-restart.md). |
| Initial seed, formation and mechanical tip | The [strip material](../../../../source/material_model/phase_field_fault/strip_material_history.cc) supplies the prescribed tip and tangent. The incoming physical anchor is one ℓ back, clipped at the box wall. Mechanics retains its ordered-station lower bound at tip coordinate minus ℓ. The [direction solver](../../../../source/simulator/solver/midpoint_direction.cc) initializes contiguous positive seed stations. During a step, positive native phase on a formed or eligible strip admits history. Irreversibility bounds endpoint phase relative to the parent. The last contiguous formed station whose maximum one-sided phase trace reaches **0.05** locates the mechanical tip. |
| Bulk damage contour | The [mesh excursion](../../../../include/aspect/mesh_refinement/fault_damage.h) uses **0.01** to resolve the seed-connected damaged bulk, independently of the tip or strip formation. The initialized field also determines the contiguous rear of the seed and any connected damage wider than the equilibrium profile. |

The mature trace's prescribed normal support is a kinematic transfer width. The midpoint fracture operator's 2ℓ² is a gradient coefficient. The factor of two in shear rotation, the nine basis coefficients of a DGQ2 quad, the angular scan resolution, quadrature order and floating-point root tolerances also retain their local definitions. The growing midpoint assembly still admits AT1 alone.

## What the geometry check establishes

The production operator first excludes DGQ2 patches whose Bernstein hull cannot reach 0.01. With no possible coarse damage the MPI ranks exchange only patches on the incoming line to find its contiguous extent. On an ambiguous mesh they exchange nonexcluded owned patches and test connectivity. Face intervals come from the quadratic traces on each side in physical coordinates. A connection needs positive-length overlap, including across a hanging subface or partition. Strictly positive-area excursions and constant plateaus at the contour participate. A zero-length corner contact supplies no path. The incoming straight line restricts a biquadratic to degree at most four. During a trial an accepted point and its one-sided cell identity fix the source.

Sign-definite patches and monotone patches with a connected high edge admit a direct test. For the remaining shapes, de Casteljau subdivision bounds each axis to 1/32 of its parent at the terminal level. Unresolved pieces inside one such rectangle can join conservatively. The parent diagonal divided by 32 bounds **one local join**. A chain of such joins can cause a distant extra cell to be refined. No global contour-position error follows from that local tolerance. Graph storage is replicated during the collective search. Native stress, strip history and FE solutions retain their existing owners.

| Manufactured receiving patch | Previous center/increment test | Repaired endpoint decision |
|---|---|---|
| Connected, 0.00999 → 0.01001 → 0.01020 | Both intervals pass without refinement | Below at the parent. Both later endpoints require coverage. |
| Connected, 0 → 0.012(1−x)² on a unit square | Largest center value 0.0091875, no refinement | Refine. The exact active width is 1−√(0.01/0.012) = 0.087129070825 m. |
| Unchanged coarse island at 0.02, fine bridge from 0.00999 to 0.01001 | No coarse sample crosses | The island stays detached before the bridge and requires coverage afterward. |

These are manufactured operator tests on actual rank-local coefficient sets. The compact calculation below exercises the ordinary nonlinear solve, timestep decision, transfer and final publication. Additional manufactured controls exercise disjoint one-sided DG traces on a translated rectangular hanging interface, a split saddle, detached and corner-only components and a new, closer island on the seed line.

## Startup, rollback and accepted work

On early passes the criterion can have a DoFHandler before ASPECT has interpolated the phase. In that case it interpolates the prescribed initial field at the phase element's generalized support points. The optional [initialization check](../../../../source/simulator/helper_functions.cc) inspects the actual initialized field after the scheduled passes. If coverage is missing it remeshes and reinitializes from the same initial data before physical time. An ambiguous coarse exposure without a supported incoming seed is diagnosed. An unattainable target or a cap that prevents progress also produces a diagnostic. Fixed forward-sector refinement uses the mesh criterion alone.

Ordinary positive-time order is midpoint solve → growing fault timestep reaction → evaluate seed-connected trial damage → compare with the accepted parent if coarse damage is exposed. A missed trial remeshes the parent and repeats. An already exposed parent has its own diagnostic. A reserve-only request accepts the endpoint and then refines. Geometry for a repeat consists of physical footprints and survives refinement of their cell IDs.

## Replay

Use a fresh output root. Run from the checkout with one numerical thread per MPI rank. The paths below use the warm Release tree on this machine. The install includes the benchmark input files and Python checkers.

```sh
r=$PWD/build-amr-review
exe=$r/install/bin/aspect-release
amr=$r/install/benchmarks/granite_fault_growth/amr
driver=$amr/../run.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1 MKL_NUM_THREADS=1
cmake -S . -B build-6b/release -DCMAKE_INSTALL_PREFIX="$r/install" -DASPECT_INSTALL_EXAMPLES=ON
cmake --build build-6b/release --target aspect granite_history -j2
cmake --install build-6b/release
"$exe" --test 'Fault damage*' -s
mpiexec -n 2 "$exe" --test 'Fault damage*' -s
```

A fresh build of the full example collection can use the default build target before installation. See the [law and granite checks](support-and-seed.md) and [mesh/strip and checkpoint tests](refinement-restart.md) for their exact commands. The loaded-state observer is built against the same installed SDK.

```sh
python3 "$amr/run_advancing.py" "$exe" "$r/miss" miss --case compact --ranks 2
python3 "$amr/run_advancing.py" "$exe" "$r/uniform" uniform --case compact --ranks 2
python3 "$amr/check_recovery.py" "$r/miss" "$r/uniform" --compact --ranks 2 --steps 2
python3 "$amr/run_advancing.py" "$exe" "$r/resumed" miss --case compact --ranks 2 --restart "$r/miss/output/restart/01"
python3 "$amr/trajectory.py" "$r/resumed/output" --compare "$r/miss/output" --strict --steps 1 2
python3 "$amr/check_recovery.py" "$r/miss" "$r/uniform" --compact --ranks 2 --steps 2 --resumed "$r/resumed"
python3 "$amr/run_advancing.py" "$exe" "$r/prevent" preventive --case compact --ranks 2 --observer "$r/plugin/libgrowing_fault_checks.release.so"
python3 "$amr/check_prevention.py" "$r/prevent" --ranks 2
python3 "$amr/run_advancing.py" "$exe" "$r/shoulder" disk --case shoulder --ranks 2
mkdir "$r/detached"
cp "$amr/detached_island.prm" "$amr/../../growing_fault/growing_seed.prm" "$r/detached/"
python3 "$driver" "$exe" "$r/detached/detached_island.prm" timing --ranks 2
python3 "$amr/check_detached_island.py" "$r/detached/output" --bins 66
```

## Measured solver decisions

| Two-rank calculation | Measured decision and accepted result |
|---|---|
| Miss in the compact propagation control | The compact test started with 92 cells. The phase and all native history components were zero in the twelve parent cells ahead of x = 1.25 m. At 2.5 ms four coarse cells belonged to the trial's incoming component. One repeat refined the parent to 128 cells. Independently starting on the uniform fine mesh gave identical initial data on common cells and virgin data on all 48 receiving children. All fourteen accepted physical CSVs at 2.5 and 5 ms were byte-identical. The keyed test includes 8,640 native points, 15 material rows, FE coefficients and work. |
| Miss after checkpoint restore | Generation 01 reproduced the same source tip (1.25, 0.5) m, four coarse cells, same first interval, one rejection and both strict accepted frames. |
| Sufficient reserve, then preventive refinement | On the 116-cell mesh the clearance fell from 0.500000 to 0.483333 m as the supported tip advanced to x = 1.266667 m. It crossed the 0.495 m trigger. The trial had zero connected coarse cells. The accepted phase trace was 0.2513464073763899 before and after refinement to 128 cells. No repeat occurred. |
| Diffuse shoulder | The 0.05 supported station moved from x = 0.75 to 0.875 m by 2.5 ms. Both physical intervals were accepted. At 5 ms the independent 16-sample-per-fine-edge observation gave forward positions 0.945313 and 0.882813 m for the 0.01 and 0.05 contours (sample diagonal 0.02210 m). |
| Detached damage behind the seed | At 2.5 ms all 16 historical sample crossings belonged to a separate component, x = 0.5625–0.8125 m. The seed-connected count was zero on the independent 66-bin raster. No repeat occurred. |
| Initial finite inclined seed | At the inherited 0.25 m target, the earlier mesh left connected cells up to 0.5 m wide and refined an intact backward witness to 0.25 m. The repaired maximum connected edge was 0.25 m and the witness stayed 0.5 m. Halving only the bulk target resolved all connected cells to 0.125 m and kept every material row. [Measured native field and mesh](support-and-seed.md). |

The two-rank compact recovery took 66.22 s (highest rank 296,536 KiB). The uniform control took 49.77 s (294,188 KiB). These wall times include startup, the nonlinear solves and public output, with another MPI calculation sharing the machine. They measure the whole calculation. The cover check stores up to nine phase coefficients, a physical box and identity per candidate cell. The expensive search can replicate up to 32² local subpatches per ambiguous cell during an assessment. Early exclusions and monotonicity reduce that bound on ordinary meshes. The rank-zero factorization remains the principal solver cost on the physical granite case.

The final installed Release executable, SHA-256 `effadd4cfd941c8d44eca41da6f94afadde1b4f875bd68d79712a886ddff9070`, passed 3,249 assertions in 74 unit cases. The 40 focused assertions in seven geometry cases passed on one rank and independently on each of two ranks. The final executable reproduced both the initial admission and loaded-resolution reports byte for byte. The final geometry also excludes a monotone patch touching the contour on a zero-area line, and carries the same accepted seed identity through the one-dimensional shortcut.

### Parent coverage, shared-core regression and frozen mechanics

We also resumed a deliberately underresolved, already accepted 32-cell compact parent. At 5 ms the reaction identified **unresolved accepted damage** with 24 connected coarse cells. It transferred the parent, repeated once on 128 cells and wrote a single 5 ms balance. In contrast, the historical 18-strip cap prevented the initial seed from being resolved at all. Both ranks diagnosed the blocked initialization and no positive-time balance was written. Resuming the sufficient-reserve case from generation 01 reproduced the preventive decision, its before/after clearance and trace, and both accepted frames.

```sh
mkdir "$r/unresolved-parent"
cp "$amr/compact.prm" "$amr/../../growing_fault/growing_seed.prm" "$r/unresolved-parent/"
cat > "$r/unresolved-parent/run.prm" <<'PRM'
include compact.prm
set Output directory = output
set End time = .0025
subsection Checkpointing
 set Steps between checkpoint = 1
 set Number of checkpoints to keep = 4
end
PRM
python3 "$driver" "$exe" "$r/unresolved-parent/run.prm" timing --ranks 2
python3 "$amr/run_advancing.py" "$exe" "$r/unresolved-resumed" miss --case compact --ranks 2 --restart "$r/unresolved-parent/output/restart/02"
rg 'unresolved accepted damage|Repeating the current' "$r/unresolved-resumed/timing/solver.log"
# Expected failure on the old blocked seed.
python3 "$amr/recovery.py" "$exe" "$r/blocked-legacy" miss --ranks 2
python3 "$amr/run_advancing.py" "$exe" "$r/resumed-prevent" preventive --case compact --ranks 2 --observer "$r/plugin/libgrowing_fault_checks.release.so" --restart "$r/prevent/output/restart/01"
python3 "$amr/check_prevention.py" "$r/prevent" --ranks 2 > "$r/prevention.json"
python3 "$amr/check_prevention.py" "$r/resumed-prevent" --ranks 2 > "$r/prevention-resumed.json"
cmp "$r/prevention.json" "$r/prevention-resumed.json"
python3 "$amr/trajectory.py" "$r/resumed-prevent/output" --compare "$r/prevent/output" --strict --steps 1 2
```

The one shared ASPECT edit is the optional final initialization check in `Simulator::maybe_do_initial_refinement`. The previous hook ran only during the scheduled passes. A fixed sector can run with no timestep plugin, and the physical phase can first become available at the end of those passes. Only a marker implementing the optional target can request another pass. The stock time-dependent minimum-refinement control reproduced the retained statistics byte for byte (SHA-256 `f69c353436301940d12f83b12c3fe93a13823338f9ad5649f2612b2f468fbd80`). The scoped growing-fault CTest passed. The two-rank nine-strip donor comparison passed at 5, 10 and 15 ms, with final integrated slip 1.69302306477e−4 m² at the established tolerance.

```sh
mkdir "$r/stock" "$r/nine"
cp tests/minimum_refinement_function_time_dependent.prm "$r/stock/base.prm"
cp refactoring/phase-field-fault/example2/amr/stock-refinement.prm "$r/stock/run.prm"
python3 "$driver" "$exe" "$r/stock/run.prm" timing --ranks 1
cmp "$r/stock/output/statistics" build-granite-amr/stock2-old/output/statistics
ctest --test-dir build-6b/release -R '^growing_fault$' --output-on-failure
cp "$amr/../../growing_fault/onecut.prm" "$r/nine/"
cat > "$r/nine/run.prm" <<'PRM'
include onecut.prm
set Output directory = output
subsection Postprocess
 set List of postprocessors = growing fault history
end
PRM
python3 "$driver" "$exe" "$r/nine/run.prm" timing --ranks 2
python3 "$amr/../../growing_fault/compare.py" build-5a-repair/donor-two/output "$r/nine/output"
```
