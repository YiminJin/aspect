# K1 initialization: visualization and localized CPDI inconsistency

Follow-up: `stage_K1_cpdi_cause_and_correction_plan.md` now establishes the
ownership-gap mechanism left as a hypothesis below, and records the separate
activation-aware stationary-equation audit. This initial diagnostic packet
remains the unchanged pre-correction baseline.

## Outcome and scope

The requested visualization is available persistently at:

`/home/ein/repository/aspect/benchmarks/reconstructed_fault/uniform_shear/diagnostics/results/`

Opening/reproduction instructions are in the adjacent `README.md`. Production
source, fixture parameters, constraints, normalization, tolerances, initialization
semantics and acceptance gates were not changed. No production correction,
full pilot, convergence matrix, full test suite or K2 work was performed.

The original `/tmp/aspect-k1-pilot` artifacts are no longer present. The existing
pilot/prerequisite reports remain the historical record, not substitute field
data. One bounded, isolated **initialization-only** diagnostic was necessary.
It used the same executable (SHA256
`1144c79341f60facac2365791ebdf21f10a2c331cb742aa43beb1f61dfa9311a`)
and included the original pilot input. Its observed phi range, extrema and
reconstructed geometry reproduce the earlier pilot. Accepted t=0/t=2 mechanical
fields are unavailable and have not been reconstructed from summary numbers.

## Observed spatial pattern

See `phase_and_fault.png` and `particles_and_discrete_problem.png`:

- The phase field has two off-center transverse maxima and a trough at y=0.
  Its maximum is 0.402161988463; phi(x,0) varies roughly from 0.168 to 0.228.
  The largest along-x range at a fixed y is 0.0596760807714.
- This is not confined to the periodic edge. Interior x columns differ too.
  The actual reconstructed line is displaced by up to 6.22036e-4 m and has
  a smooth y(x) trend; the nominal y=0 line is supplied separately.
- The initial H profile is uniform along x to about 1.26456e-8 Pa and matches
  the independent intended stationary initializer within 2.72013e-7 relatively.
  Total initial particle-domain volume is 0.25000000000000006 m².
- Particles, their H/volumes, full CPDI arrays, mesh connectivity and DoF
  mapping remain unchanged through the phase solve. The saved before/after
  particle, stencil and connectivity CSVs are byte-identical.

The core input 0.6 is **not a prescribed FE phase value**. Current design
section 13 and `ReconstructedFaultManager::initialize_crack_driving_force`
(`source/reconstructed_fault/manager.cc:317`) use the intended stationary
profile to initialize particle H where phi exceeds activation. Elsewhere the
baseline H remains. `PhaseFieldHandler::stationary_crack_driving_force`
(`source/simulator/phase_field.cc:761`) supplies this H formula. The subsequent
CPDI weak solve is unconstrained by phi_hat except through H. Finite resolution,
the activation cutoff and the discrete transfer can affect its peak. Therefore
0.402 versus 0.6 is kept separate from the confirmed consistency failure; a
peak-equals-0.6 assertion would not be justified.

## Earliest localized inconsistency

At `start_timestep`, **before the initial phase solve**, the FE phase is exactly
zero and the actual CPDI stencils already fail basic consistency:

| Check | Measured result |
| --- | ---: |
| Number of particles | 9,216 |
| Sum of weights = 1, within 1e-10 | 9,148 particles |
| Sum of weights = 5/6 | 29 particles |
| Sum of weights = 2/3 | 39 particles |
| Maximum partition-of-unity error | 1/3 |
| Constant-field gradient failures (>1e-8 m^-1) | Same 68 particles |
| Maximum norm of sum of gradients | 192.0000000000001 m^-1 |
| Locations of affected rows | y=-0.002604166666666668 and +0.0026041666666666665 m |

There are 35 failures in the negative row and 33 in the positive row. Of these,
49 lie in interior columns x in (0.03125,0.21875) m. Thus this is not explained
by merely lacking a periodic image across x=0/0.25. Particle 3836, for example,
at (0.0026041666666666665,-0.002604166666666668), has sum(w)=2/3 and summed
gradient approximately (0,192) m^-1. Every particle's measurements are in
`particle_stencil_checks.csv`; the original stencils remain in `raw/`.

These errors are not ordinary profile-resolution errors: a constant nodal
field must transfer as a constant, and its gradient must vanish. For affected
particles, even a constant field creates a spurious gradient contribution to
the phase energy. The source consumes these arrays directly in
`PhaseFieldHandler::assemble_phase_field_system`
(`source/simulator/phase_field.cc:806`, transfer at lines 845–869).

The actual discrete operator therefore does **not** preserve the intended
along-x translation symmetry. With the exported stencils and the actual
periodic DoF constraints, independent assembly gives:

| Quantity | Result |
| --- | ---: |
| Initial residual Euclidean norm | 6940.56987548539 |
| Maximum initial residual range along x at fixed y | 51.1577207127 |
| First direct-solve Newton direction range along x | 6.05645023290e-4 |
| Final relative residual, independently reassembled | 5.05242289855e-11 |
| Final relative residual, production log | 5.052e-11 |
| One-cell x-translation residual commutator / initial norm | 1.37521772831e-4 |

The 65 exported phase constraints are precisely homogeneous unit-weight
periodic identifications of matching-y endpoints. No phase lifting, unexpected
phase Dirichlet condition or mismatched-y identification was found. The
independent elimination maps these same pairs to canonical free DoFs.

For this homogeneous AT1 fixture, the independent residual is

    R = W^T diag(volume) [H g'(W phi) + 64]
        + 3.125 [X^T diag(volume) X phi + Y^T diag(volume) Y phi].

Here W,X,Y are the **exported** CPDI value/x-gradient/y-gradient matrices after
periodic elimination, not a substituted ideal stencil. At phi=0, g'=-128
and g''=32000. These coefficients follow the same benchmark material inputs
and rational degradation, independently evaluated in Python. A SciPy sparse
direct solve supplies the diagnostic first Newton direction. That direction
is not asserted to be a recorded production iterate. No independent candidate
is written back into ASPECT.

Matching the final residual to the production log shows that the phase solver
really converges this discrete problem. The symmetry defect is already in the
transfer/residual, before AMG, nonlinear stopping, reconstruction, I_h or
mechanics. It is not evidence for loosening tolerances or changing the initial
Newton iterate. The characteristic central trough is consistent with a
spurious gradient penalty in those rows, but its entire peak error is not
quantitatively attributed without a corrected-stencil control calculation.

## Confirmed findings versus remaining hypotheses

**Confirmed:** pre-solve CPDI partition-of-unity and constant-gradient failure;
interior as well as edge failures; resulting loss of discrete x-symmetry;
correctly identified periodic phase constraints; unchanged input histories;
and genuine convergence to the defective discrete problem.

**Most targeted remaining hypothesis:** loss of unique FE-cell ownership of
Voronoi sampling points near the internal face y=0. In
`source/particle/particle_domain.cc:964`, `is_inside_unit_cell` uses strict
half-open comparisons on internal faces, with tolerance only at domain
boundaries. `compute_voronoi_cell` at lines 1102–1107 and 1113–1118 transforms
each polygon vertex/centroid independently and zeros all basis values when
that cell rejects it. A roundoff gap in which every candidate cell rejects a
point would produce missing basis contributions. The observed 1/6 and 1/3
weight deficits at just the adjacent rows are consistent with this mechanism.
However, raw Voronoi vertices and their per-cell ownership decisions were not
exported, so that exact mechanism is **not yet proved**.

A separate orientation observation must not be confused with the main failure:
among the 9,148 partition-consistent particles, the recovered gradient of y is
(0,-1) for 3,932 and (0,+1) for 5,216, to roundoff. `SimplexIntegrator` at
`source/particle/particle_domain.cc:777` integrates values with absolute area,
but returns the signed face-based gradient. A common sign reversal of all
gradients for one particle cancels from its phase residual/Jacobian gradient
bilinear form. It therefore does not, by itself, explain the current phase
failure. It should be included in the focused consistency regression, not
silently fixed as part of this diagnostic.

No conclusion is drawn here about the other K1 failures (I_h accuracy,
association-strip tail, integrated slip, periodic particle domains during
motion, or pilot runtime). They remain separate unresolved gates.

## Smallest next test and proposed correction boundary

Before a production correction, capture one affected initial Voronoi polygon
and a nearby unaffected one from these two rows. At every polygon vertex and
centroid, record the candidate FE cells, transformed reference coordinates,
ownership decisions and total retained basis value. This can be a tiny
particle-domain test: it does not require phase evolution or mechanics.
Exercise exact-face positions and representable perturbations on either side.

If this confirms uncovered samples, propose correcting **unique ownership at
internal FE faces** in the CPDI construction, using mesh-aware ownership and
a deterministic tie-break where tolerance neighborhoods overlap. Merely
loosening both inequalities can double-count samples. Renormalizing weights
after assembly or enforcing phi symmetry would hide rather than repair the
missing transfer contributions. No such correction has been implemented.

Regression proposal: extend the focused particle-domain coverage beyond cache
regeneration (`tests/phase_field_particle_domains.cc` currently checks rebuild
identity, not mathematical consistency). Require sum(w)=1, sum(grad w)=0,
affine reproduction, unique ownership, and invariance under one-cell x shifts
on this regular three-particles-per-direction layout. Include the y=0 shared
face and periodic/nonperiodic edge cases. Then rerun this initialization-only
fixture with unchanged tolerances and compare its discrete translation
commutator and x-variation. Do not assert that its finite-resolution peak must
equal the initializer's 0.6 exactly.

## Deliverables and checks

New files are limited to
`benchmarks/reconstructed_fault/uniform_shear/diagnostics/` (plugin, CMake/input
overlay, analyzer, README, packet tests, saved CSV/VTU/PNG/JSON/raw data) and this
report. Existing Stage-J/K1 working-tree changes remain intact. No new public
production API was added. The packet is about 10 MB; local build files are
ignored, while the saved result data remain persistent repository files.

The plugin reuses production `DataOut` and `ReconstructedFaultOutput`. It stops
after initial cohesive/I_h preparation, before the first mechanical residual,
with an explicit `K1_INITIALIZATION_DIAGNOSTIC_COMPLETE` exception through the
existing rollback path. Initialized Theta/I_h/cohesive fields are not labeled
as accepted mechanics; no V_min pseudo-solution is supplied.

- `cmake --build .../diagnostics/build -j4`: final Debug and Release plugin
  builds passed. No production rebuild was needed.
- `timeout 300 .../aspect ../initialization.prm`: reached the explicit diagnostic
  completion marker, exit 1 by design (not timeout); phase converged in 13
  nonlinear iterations, each with one CG iteration, at original tolerance 1e-8.
  No mechanical residual or positive timestep was run.
- `visualize.py`: succeeded; XML/VTK files were read back with expected counts
  and no nonfinite field data. Both preview plots were visually inspected.
- `python3 -m unittest discover -s .../diagnostics -p test_packet.py -v`:
  **3/3 passed** (0.033 s). These are artifact/reproduction checks, not CPDI
  production acceptance. VTK emits a NumPy deprecation warning, not a failure.
- `git diff --check`: passed.

Initial plugin development encountered a missing include and duplicate signal
registration symbols; these were resolved in diagnostic-only files. A first
launch failed input registration before any solve, and sandbox MPI socket
access required the approved external execution. The actual diagnostic has
one completed initialization-only run; no successful full pilot was repeated.

**Stop for review before any production correction or further K1/K2 work.**
