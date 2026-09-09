# Coupled-solver pressure-normalization correction

Follow-up diagnosis: `stage_K1_linearization_diagnosis.md` records the bounded
replay of the remaining 4.5 s failure. It demonstrates false linear convergence
and proposes a guarded pressure-complement solve; that correction is not yet
implemented. The pressure-normalization findings below remain valid.

This is the approved prerequisite correction, not completion of Gate K1 or
authorization to start K2. Support, full `I_h`, phase/H initialization, physical
parameters, nonlinear/linear tolerances and the Armijo budget are unchanged.
Existing unrelated working-tree changes have been preserved.

## Cause and correction

The ordinary Newton workflow normalizes `current_linearization_point` in
`source/simulator/solver_schemes.cc`. The reconstructed-fault branch returns
through its dedicated solve before that workflow. Its private iterates and
published solution therefore omitted the configured normalization. The saved
coarse dt=2 s Debug/Release trajectories demonstrated different constant
pressure offsets with otherwise matching physical histories; this was a real
configuration defect, not a change to the reference pressure convention.

`Simulator::solve_reconstructed_fault_stokes()` now calls the existing
`normalize_pressure()` on the physically constrained private base and on each
physical trial **before** residual/merit evaluation. The accepted solution and
linearization point are copied from that same normalized state. Newton
directions remain homogeneous; the physical/solver pressure conversion and
the boundary lifting contribution are unchanged.

The operation is guarded by the existing choice of prescribed adiabatic
friction pressure. A constant bulk-pressure shift cannot change that surface
equation. The true-normal-stress branch is deliberately unchanged: shifting
its pressure would change the friction equation and is not a gauge correction.
`volume`, `surface` and `no` use ASPECT's existing implementation and parameter
meanings. No new numerical or physical parameter has been introduced.

Pressure-normalization bookkeeping follows the accepted private iterate and
is published only in the existing terminal mutation phase. All normalization
allocations/collectives happen before history writes. Rejection or nonlinear
failure cannot publish a candidate pressure adjustment, bulk solution or V.

Files changed for this correction (other dirty files predate this task):

- `source/simulator/solver.cc`: private base/trial normalization and terminal
  publication of the corresponding adjustment.
- `include/aspect/material_model/phase_field_fault.h` and its `.cc`: one
  const semantic accessor, `uses_adiabatic_friction_pressure()`, returning the
  existing mode. The material model still owns only pointwise constitutive work.
- `doc/reconstructed_fault/current_design.md` and `specification.tex`:
  the approved gauge rule and lifecycle constraints.
- The eight `tests/phase_field_fault_pressure_gauge*` fixture families listed
  below: new `.cc`, `.prm`, `.sh`, and expected `screen-output` files.
- `benchmarks/reconstructed_fault/uniform_shear/pressure-normalization/`:
  output-only dt=.5/dt=2 wrappers, a read-only before/after comparison script,
  saved outputs, logs, independent-reference analyses and provenance records.
- This review and status links in the existing benchmark records.

## Focused verification

All artifacts below are relative to
`benchmarks/reconstructed_fault/uniform_shear/pressure-normalization/`.
Both production configurations and test/benchmark plugins were built with
`-j4`; only focused tests were selected.

```sh
cmake --build build-pf-cpdi --target aspect.exe.debug -j4
cmake --build build-pf-cpdi --target aspect.exe.release -j4
cmake -S tests -B build-pf-cpdi/tests
cmake --build build-pf-cpdi/tests --target \
  phase_field_fault_pressure_gauge phase_field_fault_pressure_gauge_mpi \
  phase_field_fault_pressure_gauge_shifted phase_field_fault_pressure_gauge_shifted_mpi \
  phase_field_fault_pressure_gauge_surface phase_field_fault_pressure_gauge_no \
  phase_field_fault_pressure_gauge_rollback phase_field_fault_pressure_gauge_rollback_mpi -j4
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_pressure_gauge($|_)' -j2
build-pf-cpdi/aspect --test 'Stage-I*'
```

Both production builds passed. The Stage-I unit selection passed **31
assertions in 7 cases**, retaining the isolated backtracking, exhaustion,
projected-bound and active-set tests (`stage-i-unit.log`). The first integration
attempt exposed a test-only duplicate signal-registration symbol caused by
including an existing fixture. Registration was moved to its own namespace;
the failed compile log is retained as `gauge-tests-initial-build-failure.log`.
No physics or assertion tolerance was changed to address that build error.

All five one-rank integrations passed (volume, shifted, surface, no, rollback).
The first two-rank attempts were blocked before simulation by the sandbox's
unavailable MPI network interfaces. They were rerun with MPI access using:

```sh
ctest --test-dir build-pf-cpdi/tests --output-on-failure \
  -R '^phase_field_fault_pressure_gauge_(mpi|shifted_mpi|rollback_mpi)$' -j2
```

The integration fixtures cover:

| Fixture suffix | Invariant checked |
|---|---|
| base / `_mpi` | Volume mean of both published vectors is zero at t=0 and changed-loading t=2; inherited K1/history assertions still run. |
| `_shifted` / `_shifted_mpi` | Adding 12345 Pa to the initial pressure leaves normalized velocity, V, Theta, cohesive traction, previous-I_h and mean particle stress unchanged. |
| `_surface` | Existing 1000 Pa top-surface target is honored; the same gauge-independent trajectory is recovered. |
| `_no` | Normalization remains disabled and does not silently remove the test offset; physical trajectory agrees. |
| `_rollback` / `_rollback_mpi` | Forced failure after an accepted update restores the deliberately shifted original bulk vector, current/committed V and all captured histories. |

The volume/surface assertions use an independent FE integral with a 1e-8 Pa
absolute allowance. Gauge-independent fingerprints use 1e-9 times their
documented dimensionless scales, not a relaxation of solver tolerances.

**All eight integrations passed.** `gauge-tests.log` records the five successful
one-rank runs (98.87, 132.69, 142.90, 143.73 and 159.83 s) and the blocked MPI
attempts. `gauge-mpi-tests.log` records the successful volume, rollback and
shifted two-rank tests (136.88, 96.90 and 76.08 s; 212.98 s elapsed for that
selection). `gauge-tests-final.log` collects the complete eight-test status by
reusing those up-to-date output comparisons; it is not eight additional solves.

## Unchanged coarse dt=2 pilot

```sh
cmake --build benchmarks/reconstructed_fault/uniform_shear/diagnostics/pilot-build -j4
python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py \
  benchmarks/reconstructed_fault/uniform_shear/pressure-normalization/dt2.prm \
  --configuration Release --timeout 600
```

The pilot completed through 6 s in **31.03 s**, peak RSS **244724 KiB**
(`dt2.resources.json`). At 6 s the raw volume mean is **4.466e-13 Pa**, compared
with **566.739740203 Pa** in the pre-correction Release trajectory and
**-1480.929134687 Pa** in the pre-correction Debug trajectory.

`dt2-release-comparison.json` compares identical quadrature coordinates and
accepted times with the saved same-configuration trajectory. Phase, H and I_h are
identical. Across all four accepted states, the maximum velocity difference
is 2.64e-18 m/s, V difference 2.59e-17 m/s, Theta difference 3.47e-12 s,
cohesive-traction difference 6.83e-13 Pa, and particle stress-component
difference 2.15e-10 Pa. The remaining spatial (nonconstant) pressure difference
is at most 3.55e-10 Pa. The correction removes an arbitrary offset without
changing the measured mechanical/history trajectory.

The saved `dt2-parameters.diff` and `dt05-parameters.diff` compare the resolved
parameter files, ignoring alignment whitespace. In both cases, the output
directory is the only difference from the corresponding pre-correction run.

`dt2-analysis.log` independently initializes and advances the scalar reference
once; `dt2-errors.json` retains raw unsmoothed stress and velocity errors.
Containment remains 5.64918e-5; the actual integrated-slip ratio at t=0 is
0.999941301534. These retain the provisional **K1-only** 1e-4 containment
allowance and separate unchanged 1e-4 normalization requirement. This repeat
does not establish spatial or timestep convergence.

## Unchanged dt=.5 reproducer: remaining failure

```sh
python3 benchmarks/reconstructed_fault/uniform_shear/convergence/run_case.py \
  benchmarks/reconstructed_fault/uniform_shear/pressure-normalization/dt05.prm \
  --timeout 1800
```

The Debug run accepted t=0 through 4 s, then **again exhausted Armijo at
4.5 s**. Exit status was 1, wall time **1051.89 s**, peak RSS **462352 KiB**
(`dt05.resources.json`). No unsuccessful state was exported. The final
linearizations at 4.5 s were:

| Newton iteration | Relative bulk residual | Relative surface residual | Outcome |
|---:|---:|---:|---|
| 0 | 1.000e0 | 4.459e-3 | Full candidate accepted |
| 1 | 3.701e-7 | 3.675e-6 | Full candidate accepted |
| 2 | 1.992e-8 | 2.485e-12 | All admissible Armijo candidates rejected |

The configured convergence threshold remains 1e-8. The corresponding last
pre-correction residuals were bulk 1.754e-8 and surface 4.677e-13. Normalizing
the pressure therefore **does not resolve or adequately explain** this
remaining bulk-residual stagnation. It fixes the independently demonstrated
gauge defect; it is not a justification to relax the convergence test, enlarge
the line-search budget or accept the last unsuccessful candidate.

Across the nine accepted states, the corrected raw volume mean is within
**8.524e-12 Pa** of zero. `dt05-comparison.json` compares the saved old/new
Debug states: phase and I_h are identical, as are initial H and Maxwell stress.
The maximum velocity change is 3.33e-17 m/s, V change 1.48e-16 m/s, Theta change
6.97e-12 s, and cohesive-traction change 1.03e-12 Pa. The accepted histories
have not been repaired, smoothed or reinitialized to obtain that comparison.

The partial trajectory's independent-reference audit is retained in
`dt05-analysis.log`; the analysis exits 2 because the requested end time was
not reached, despite all listed assumption checks passing. The reference was
initialized once from retained initial histories and then advanced using the
actual accepted timestep sequence. `dt05-errors.json` preserves the separate
conditional trajectory and raw unsmoothed stress errors and explicitly records
`completed_to_6_seconds=false`. The measured omitted fraction is 5.64918e-5
and the maximum actual slip-normalization error is 5.86985e-5 **through 4 s
only**, not on a completed trajectory.

The smallest remaining uncertainty is the cause of the nonzero bulk residual
floor at the failing linearization: the present output does not distinguish
residual cancellation/conditioning from an action or linear-solve error.
Absolute residual scales and directional/linear residual diagnostics at this
state are the appropriate next diagnostic evidence, before proposing another
production correction. No such correction was added in this task.

## Handoff

The approved pressure-normalization correction and its focused regressions
are complete. The authoritative rule, implementation, unchanged-input checks,
exact commands, build/test logs and executable/plugin hashes are retained.
`git diff --check` passed. The complete ASPECT suite was not run. True-normal-
stress behavior was not changed or newly certified by these prescribed-mode
gauge tests. No commit was made.

Gate K1 remains **unmet**: the half-second trajectory still fails, and the
remaining spatial/timestep convergence checks have not been completed. K2
has not started. Stop here rather than expanding the production patch.
