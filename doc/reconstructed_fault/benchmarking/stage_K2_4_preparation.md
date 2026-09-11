# K2.4 preflight: 64x255 normal-grid parity sensitivity

Subsequent execution: the user approved both cases with 300-s individual and
600-s aggregate simulation caps. Both completed and passed their actual
geometry and acceptance checks; see `stage_K2_4_report.md`. The preflight
below is retained as the pre-execution record, not an outstanding run request.

## Status

K2.3 is completed and accepted as feasibility verification with a documented
spatial-resolution limitation. Its endpoint lobe has approximately constant
amplitude and O(h_Gamma) physical width, with strong collapse against normalized
endpoint distance. After excluding three coarse surface elements at each end,
interior Delta sigma_n and -Delta(tau:N) differences remain about 19.6% and
21.9% of the fine interior signals. K2.2/K2.3 references remain provisional;
Gate K2 remains unmet. No 128x512 pair is to be launched now.

The user explicitly authorizes K2.4 as a **bounded alignment-sensitivity check
against the provisional 64x256 baseline**, superseding the original
post-convergence sequencing for this check only. This is not a claim of
converged normal-stress verification. The configuration below uses existing
Box parameters; no production mesh/mapping infrastructure is needed.
The bumped and matched homogeneous fixtures are prepared, but **execution
awaits preflight review**. No ASPECT process has been launched for K2.4.

## Locked 64x256 baseline

Reuse `nonuniform/true-pressure/pilot64.prm` and `homogeneous64.prm` and their
accepted saved outputs. No baseline rerun is required for preparation.

| Quantity | Retained baseline |
|---|---|
| Bulk grid | 64x256; box 0.25 m by 1 m |
| Surface | Open fault, h_Gamma=0.0078125 m, 33 independent vertices |
| Times | Initialization and accepted 0.5, 1 s; dt=0.5 s |
| Physical normal loading | Top total normal traction -1000 Pa |
| Velocity / periodicity | Top tangential velocity, both bottom components; bulk x periodic |
| Pressure | Physical p; sigma_n=p-tau:N; Pressure normalization=no |
| Perturbation | Existing Theta bump and matched homogeneous control |
| Phase/history inputs | Existing initialization; converged initial Q1 phase frozen; unchanged retained/split histories |
| Numerical methods | Accepted domain quadrature, full I_h, support, continuous Q2 ADD/count stress transfer; unchanged tolerances and budgets |

The user explicitly extends the provisional omitted-profile allowance <=1e-4
to this K2.4 check (original target 1e-6), conditional on independent
remeasurement. The separate actual slip-normalization requirement remains
<=1e-4 at measured locations and accepted times. Neither is satisfied merely
by inheritance from K2.3. Keep the support rule and full I_h, with no
renormalization or new support-policy adjustment.

## Existing configuration path and exact changes

`source/geometry_model/box.cc`, `Box::create_coarse_mesh()`, passes the X/Y
repetition counts directly to `GridGenerator::subdivided_hyper_rectangle`,
using the unchanged box origin and extents. It collects periodic faces and
adds their periodicity on that coarse grid. `source/simulator/core.cc`
performs the requested number of initial global refinement passes; zero is
explicitly admitted by `source/simulator/parameters.cc`. The existing fixture
has zero initial adaptive refinement, no subsequent mesh refinement, and
minimum refinement level zero. Its block-AMG Stokes solver does not require
the previous six-level geometric refinement hierarchy.

| Explicit parameter | Accepted baseline | Staggered fixture |
|---|---:|---:|
| Geometry model / Box / X repetitions | 1 | 64 |
| Geometry model / Box / Y repetitions | 4 | 255 |
| Mesh refinement / Initial global refinement | 6 | 0 |

These are the **only mesh-generation overrides**. Each fixture also changes
its output directory to avoid overwriting accepted evidence. The bumped
fixture includes `true-pressure/pilot64.prm`; the homogeneous fixture includes
the staggered bumped fixture and removes only the Theta bump, exactly as in
the accepted matched pair. Relative to each corresponding baseline, all
other explicitly configured parameter values are unchanged.

Fixtures are under
`benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/`:
`pilot.prm` and `homogeneous.prm`.

## Geometry implied by the production Box generator

The physical box remains [0,.25] x [-.5,.5] m; the prescribed fault file still
specifies (0,0) to (.25,0). Neither boundaries nor loading nor the physical
fault have been translated. The active mesh changes from 16384 to 16320 cells.

| Quantity | 64x256 | 64x255 |
|---|---:|---:|
| h_x (m) | .00390625 | .00390625 |
| h_y (m) | .00390625 | .003921568627450980 |
| Normal cell-size change | — | +.3921568627% |
| y=0 relative to normal grid | Face at row boundary 128 | Middle of row 127 (zero-based) |
| Containing-row normal bounds (m) | Adjacent rows meet at 0 | [-.001960784313725490, +.001960784313725490] |
| Reference normal coordinate of y=0 | 0/1 on adjacent cells | .5 |

The new vertex ordinates are y_j=-.5+j/255. Thus there is no vertex row at
zero. The fault runs through cell interiors in the normal direction; it still
crosses vertical cell faces along its length. Structural spacing remains
.0078125 m and surface endpoints remain independent. This is a **parity and
small normal-resolution change**, not a mathematically pure translation.

These are source-derived mesh dimensions, not measurements from an executed
ASPECT mesh. The eventual initialization must confirm the actual reconstructed
line's y coordinates and its cell-relative placement from exported geometry;
the prescribed y=0 line must not be confused with an already measured
reconstruction. If reconstruction fails or materially departs from the intended
alignment, stop rather than moving the fault to repair the fixture.

There are two additional representation effects to retain in the interpretation:

- The new mesh has 16320 level-zero cells rather than four coarse cells with
  six refinement levels. Coarse-grid storage, cell IDs/traversal and algebraic
  ordering can change. No solver option or transfer method changes.
- The unchanged 3x3 reference-cell generator produces 146880 particles instead
  of 147456. Particle ordinates and domains change with mesh parity; the central
  row now includes particles at y=0. Do not reuse incompatible old particle
  checkpoints, freeze volumes, or adjust particle construction. Initialize
  once with the same physical fields and record the changed projections.

The fixed phase-field rule means solving the initial FE problem on this mesh
with the existing initialization and then freezing that result. It does not
mean importing the old mesh's nodal array. Changes in initial phase profile,
I_h, cohesive/state projections, and actual support/normalization must remain
visible as initialization differences, not hidden by resetting histories.

## Required comparison if execution is subsequently authorized

Use actual constitutive surface data, not bulk-column averages. Compare the
offset bumped-minus-homogeneous fields to the saved unshifted pair at common
physical coordinates and accepted times. Keep p and -tau:N signed and separate
so cancellation remains visible. Report sigma_n, V, accumulated slip and the
realized initial projections; retain Theta/cohesive-state and weak-balance
checks. Initial-error subtraction is diagnostic accounting, not removal of
its physical influence.

Keep endpoint and interior norms separate, using fixed physical exclusion
widths such as 0.046875 m per end, plus normalized endpoint-coordinate plots.
The 19.6%/21.9% spatial differences are uncertainty evidence, not new pass
thresholds. A change comparable to the unresolved interior response cannot
be called either an alignment error or a resolved physical effect using the
64 grid alone.

Require genuine final nonlinear and fresh-linear convergence, separate surface
balance, history/lifecycle correctness, and independently measured containment
and actual slip normalization. If the check reveals sensitivity to unresolved
interior tau:N, stop and review whether the matched 128 K2.3 confirmation is
justified. Do not adjust pressure treatment, endpoint topology, quadrature,
history transfer, I_h, or solver tolerances to improve the comparison.

## Resources and preflight verification

Saved 64-case costs were 150.524 and 139.018 s, with about 1.64/1.54 GiB peak
RSS. Similar active-cell and particle counts suggest roughly 150–240 s per
one-rank case and 2–3 GiB memory provision. The much larger coarse grid and
changed initialization can affect those estimates; they are not measurements.
Propose sequential execution with at most 300 s per case, 600 s combined
simulation budget, and separately reserved short saved-data analysis time.
Do not retry a failure or timeout automatically. Execution/caps await review.

Only the saved-configuration preflight was run:

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/nonuniform/alignment/check_preflight.py
```

It exited zero. It expands the simple include/subsection/set syntax actually
used by these fixture files and verifies that each matched-baseline comparison
differs in exactly the three mesh parameters above and the output directory.
It checks the static refinement/solver choices and derives the central-row
bounds using exact rational arithmetic. It is not a full ASPECT parameter
validator or a realized-mesh/initialization test. No solver, integration,
performance, or acceptance test was run, and none is claimed to pass yet.
