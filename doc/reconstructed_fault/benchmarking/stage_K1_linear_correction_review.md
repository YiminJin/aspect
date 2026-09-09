# K1 condensed linear correction

## Scope and invariants

This implements the three-part correction approved after
`stage_K1_linearization_diagnosis.md`. The nonlinear normalization formulas,
configured tolerances, line-search budget, loading, initialization, support,
and full I_h are unchanged. No K2 work is included.

- A direction is checked by a fresh application of the **original constrained
  condensed operator**, not only the Arnoldi estimate. Raw, pressure-complement,
  and constant-pressure residual components are retained separately.
- FGMRES and its preconditioner operate on the pressure complement only in a
  verified eligible configuration. This is not physical pressure normalization.
  The existing private physical-base/trial normalization and terminal publication
  remain intact.
- Restarts retain the returned iterate and use the remaining original
  cheap-plus-expensive iteration budget. No extra linear budget or new solver
  tolerance is introduced. Failure still follows the existing bulk/V/history
  restoration path.

## Full constrained nullspace and compatibility

Eligibility is conservative: prescribed adiabatic friction pressure,
incompressibility, no mesh deformation, and closed/periodic velocity boundaries.
Partial prescribed-component masks, open boundaries, and fixed pressure DoFs
are excluded. Detection of a fixed pressure DoF is collective. True-normal-
stress friction is excluded before any projection.

The candidate q represents constant physical pressure in solver coordinates,
has zero constrained entries, and has unit algebraic norm. Continuous pressure
uses constant coefficients; FE_DGP uses its constant local basis coefficient,
following ASPECT's existing pressure-normalization implementation.

For C=A-B K_V^-1 G, the right check applies Cq including the current semantic
surface inverse. The left check is C^Tq=A^Tq **exactly**, since B has velocity
rows only and q has pressure entries only. Both checks use the homogeneous
constrained algebraic space. They must be at most 100 machine epsilons times
the corresponding pressure-coupling block Frobenius norm (including A_pp).
A failed null check disables projection rather than modifying the equations.

Before projecting b, reject |q^T b| above

```
min(100*epsilon*max(initial bulk residual, zero-velocity reference, ||b||),
    nonlinear tolerance * fixed bulk scale).
```

The same bound checks the fresh full residual's null component. This is an
internal backward-error validation, not permission to discard significant
incompatibility. The remaining fresh residual must meet the original
`linear_stokes_solver_tolerance * ||b||`. Thus the raw residual may exceed that
linear target only in the independently bounded null component. Raw values
are reported explicitly, not relabeled as converged full residuals.

## Files and interfaces specific to this correction

- `include/aspect/simulator/solver/reconstructed_fault_linear.h`: shared internal
  algebraic projection, compatibility validation, projected action, and fresh
  residual kernels used by production and isolated tests.
- `include/aspect/simulator/solver/reconstructed_fault_condensed_system.h` and
  its `.cc`: `Linearization::verified_pressure_nullspace(right_error,left_error)`
  and conservative configuration eligibility. No material or surface API change.
- `source/simulator/solver.cc`: budgeted fresh-residual verification/restart and
  pressure-complement FGMRES; dimensional residual diagnostics.
- `unit_tests/reconstructed_fault.cc`: nonsymmetric explicit full-block reference,
  recovery, bad-returned-direction detection, and incompatible RHS rejection.
- `tests/phase_field_fault_surface_system.cc`,
  `phase_field_fault_condensed_adiabatic.cc/.prm`, and the new
  `phase_field_fault_condensed_adiabatic_mpi` fixture: full production nullspace
  check and true-pressure exclusion. The adiabatic fixtures provide one/two-rank
  coverage; the existing two-rank case is retained under the MPI fixture name.
- New `tests/phase_field_fault_linear_exhaustion` fixture: one total linear
  iteration, mandatory failure before acceptance, and production rollback check.
- `benchmarks/reconstructed_fault/uniform_shear/uniform_shear.cc`: diagnostic-only
  MPI support. CSVs have `_rankN` suffixes on multiple ranks; initial phi/H
  snapshots are shared so constraints and particle migration retain the same
  frozen-state checks. Serial output names and physical parameters are unchanged.
- `convergence/run_case.py`: rank-count option, preserving bounded runtime and
  executable/plugin provenance. `linear-correction/one.prm` and `two.prm` include
  the unchanged half-second fixture, overriding only output paths.
- `linear-correction/summarize.py`: parse raw linear/nonlinear logs without
  changing any result; `compare_ranks.py` compares accepted rank-local exports.
  Authority documents record the approved numerical rule; the progress summary
  and benchmark READMEs point to the incomplete K1 outcome.

Existing dirty-tree work, including earlier pressure normalization and stress
corrections, is preserved; the repository diff also contains that earlier work.

## Verification record

Artifacts are in
`benchmarks/reconstructed_fault/uniform_shear/linear-correction/`.

Recoverable intermediate failures are retained rather than overwritten:

- `first-tests.log`: an initial sandboxed run; MPI launches failed due to
  unavailable interfaces. Do not count these as numerical failures.
- `one-initial-guard.log`, `two-initial-guard.log`: caught an implementation
  mistake in the initial compatibility reference. The zero-velocity reference
  vanishes at initialization; using it alone made the roundoff bound shrink
  with the Newton RHS. Including the original initial residual corrects that
  reference, without changing the nonlinear target or the 100-epsilon factor.
- `two-serial-exporter.log`: initial mechanics converged, then the old exporter
  rejected two ranks. The MPI output/snapshot adaptation fixes that fixture
  limitation without changing mechanics.
- The first new MPI surface check attempted a norm on a ghosted vector.
  It now copies into owned layout before reduction; this was a test defect,
  not a nullspace or constitutive failure.

## Unchanged half-second replay: linear repair, separate nonlinear floor

Both runs accept initialization and all real steps through 4 s, then reach
the t=4.5 s linearization. Each has **47 returned linear solves**, all passing
the fresh pressure-complement target and the raw-null compatibility bound.
The maximum fresh/target ratios over the entire runs are 0.94231 (one rank)
and 0.95759 (two ranks). No residual replacement/restart was needed in these
replays. The maximum raw-null/bound ratios are 0.15906 and 0.20006; no
significant incompatible RHS was projected away.

The formerly false-converged Newton iteration 2 now gives:

| t=4.5 s, Newton 2 | One rank | Two ranks |
|---|---:|---:|
| FGMRES iterations | 13 | 13 |
| Estimated residual | 3.8577611e-21 | 8.6512849e-21 |
| Fresh pressure-complement residual | 3.8576517e-21 | 8.6513256e-21 |
| Unchanged linear target | 1.6476709e-20 | 1.6484617e-20 |
| Raw full residual | 6.7976006e-16 | 6.7647497e-15 |
| Compatibility bound | 1.4670116e-13 | 1.4670116e-13 |

For comparison, the diagnosed uncorrected one-rank direction had estimated
1.55177e-20 but fresh **2.11212e-11**. The corrected quotient solve removes
that discrepancy without changing A/B/G/K_V or the nonlinear scale.
At this linearization, the verified right/left null errors are
2.22430e-10 / 2.22430e-10 (one rank) and
1.63438e-10 / 1.62749e-10 (two ranks), within the operator-relative roundoff
checks. These are actions on a unit *solver-coordinate* pressure vector, not
physical pressure errors in Pa.

Both runs take **eight accepted Newton updates** within the unsuccessful
t=4.5 s solve, then exhaust the unchanged six-candidate Armijo search at
Newton iteration 8. The terminal diagnostic is:

| Final t=4.5 s base | One rank | Two ranks |
|---|---:|---:|
| Fixed bulk scale | 5.8727992234e-5 | 5.8727992367e-5 |
| Absolute bulk nonlinear target | 5.8727992234e-13 | 5.8727992367e-13 |
| Absolute bulk residual | 8.8396813e-13 | 9.1764622e-13 |
| Relative bulk residual | 1.5051904e-8 | 1.5625363e-8 |
| Surface RMS, Pa | 1.6853619e-14 | 1.2747856e-14 |
| Fixed surface scale, Pa | 128.0032428671 | 128.0032428671 |
| Last linear estimate | 1.7499631e-22 | 5.7994713e-22 |
| Last fresh pressure-complement residual | 1.7499627e-22 | 5.7994710e-22 |
| Last unchanged linear target | 8.8397464e-22 | 9.1768208e-22 |
| Last raw full linear residual | 1.5369147e-14 | 2.2742901e-15 |

Bulk norms use the assembled velocity/scaled-continuity algebraic norm
(Pa m in this 2-D benchmark), not a pointwise stress norm. Surface convergence
is already far below its target. Even the last **raw** full linear errors are
well below the nonlinear bulk target. The remaining failure is therefore not
the previous false linear convergence or a significant pressure incompatibility.

This is an observed nonlinear residual stagnation/floor, not a proof that no
representable iterate can ever meet the target. The remaining distinction is
how much comes from nonlinear residual assembly cancellation versus finite-
precision representation of the updated bulk iterate. This patch does not
change either assembly or the stopping rule to resolve that distinction.
No further correction is proposed as already authorized: review this evidence
before choosing a residual-evaluation or stopping-scale follow-up.

Neither t=4.5 s state is committed and **neither run advances to 6 s**. Gate K1
remains unmet, and K2 remains unstarted.

### MPI consistency and controls

`parameters-values.diff` and `rank-parameters.diff` show that resolved parameter
values differ from the saved half-second case only in output directory.
`one.residuals.json` / `two.residuals.json` retain every logged linear and
nonlinear residual, not only selected iterations.

`rank-comparison.json` compares accepted exports t=0 through 4 s (a consistency
check, **not** a replacement for the independent scalar physics reference).
Maximum differences over that interval:

- V: 1.90e-17 m/s; Theta: 4.42e-12 s; cohesive traction: 4.55e-12 Pa;
  I_h: 7.11e-14 m.
- QP velocity: 1.23e-18 m/s; phase: 2.22e-16; physical pressure: 7.17e-11 Pa.
- Absolute volume-mean pressure in either run: at most 2.08e-22 Pa.

Each reference history, fixture initialization and support policy is untouched.
No stress smoothing or history reset is used. The serial run loaded the
pre-MPI-exporter plugin; the two-rank run loaded its diagnostic-only MPI
adaptation. Both plugin hashes are recorded. The serial branch keeps the same
CSV schema and frozen-state semantics.

Resources (`*.resources.json`): Debug, exit status 1 for genuine Armijo failure;
one rank **1865.11 s, 476764 KiB**; two ranks **1184.90 s, 460076 KiB** peak
child RSS. These are bounded runs under CPU contention, not isolated performance
timings or summed MPI memory. The 2400 s wall timeout was not reached.

## Focused tests and exact commands

Builds used `-j4`:

```sh
cmake --build build-pf-cpdi --target aspect.exe.debug -j4
cmake --build benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build -j4
cmake --build build-pf-cpdi/tests --target \
  phase_field_fault_condensed_adiabatic phase_field_fault_condensed_adiabatic_mpi -j4
build-pf-cpdi/aspect --test 'Stage-I*'
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_(condensed_(adiabatic(_mpi)?|dynamic)|linear_exhaustion|pressure_gauge.*)$' -j1
```

- Stage-I unit tests: **45 assertions in 9 cases passed**,
  `final-unit-tests.log`. This includes the existing Armijo rejection/exhaustion,
  local-bound/fraction-to-boundary and zero/tiny-block tests. The new small
  nonsymmetric coupled solve agrees with the explicit full-block solve, the
  fresh check rejects a deliberately bad returned vector, and a significant
  null RHS component is rejected before mutation.
- Integration selection: **12/12 passed**, `final-focused-tests.log`.
  It reuses validated output from the actual executions below, rather than
  rerunning already successful simulations merely to consolidate the result.

| Actual simulation | Result | Wall time, s |
|---|---|---:|
| condensed_adiabatic (one rank) | pass | 117.33 |
| condensed_adiabatic_mpi (two ranks) | pass | 75.62 |
| condensed_dynamic (true pressure, no quotient) | pass | 553.59 |
| linear_exhaustion | pass | 97.32 |
| pressure_gauge | pass | 166.56 |
| pressure_gauge_mpi | pass | 147.82 |
| pressure_gauge_no | pass | 234.88 |
| pressure_gauge_rollback | pass | 37.44 |
| pressure_gauge_rollback_mpi | pass | 67.82 |
| pressure_gauge_shifted | pass | 54.47 |
| pressure_gauge_shifted_mpi | pass | 44.51 |
| pressure_gauge_surface | pass | 57.67 |

All names have the `phase_field_fault_` prefix. The first six successful
executions are retained in `focused-tests.log`; its unfinished batch was
interrupted to prioritize the bounded replays. The remaining six executed
sequentially with

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_(condensed_adiabatic(_mpi)?|pressure_gauge_(shifted(_mpi)?|surface|rollback))$' -j1
```

and passed **6/6 in 387.07 s**, `final-missing-tests.log`. The consolidated
12-test check is not another timing measurement of those simulations.

The new forced linear-budget fixture reports exactly **one** outer iteration,
fresh residual 3.2208704e7 versus target 10.8392857, no accepted candidate,
and verified bulk/current-V/committed-V/history restoration. Existing rollback
fixtures independently cover failure **after** acceptance on one and two ranks.
The replays require no residual-replacement restart; the deliberately bad
direction is an isolated fresh-check test, not a claim to deterministically
reproduce the library's internal Arnoldi gap. No separate FE_DGP or open-boundary
campaign was run. `git diff --check` passes. The complete ASPECT suite, the
remaining K1 trajectory beyond the failed step, and K2 were **not** run.
