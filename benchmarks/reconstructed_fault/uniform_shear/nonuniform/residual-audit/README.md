# Bounded bulk residual consistency audit

This directory preserves the pre-audit working-tree diff (`pre-audit.patch`)
and new evidence separately from `../polyline-quadrature/`. No geometric rule,
input parameter, convergence tolerance, or iteration budget is changed.

`ASPECT_K1_FLOOR_AUDIT=1` instruments Newton iteration 1. The normal residual
still controls acceptance. A copied Stokes matrix survives all shadow/trial
assemblies; the canonical B/G/K_V caches are not rebuilt. Each shadow assembly
restores the original matrix, and the actual normal RHS is restored before
returning the candidate merit.

For this affine, all-prescribed-boundary fixture, the channels are:

- `newton`: ordinary Stokes residual, without the reconstructed-fault load;
- `load`: reconstructed-fault load alone (including beta times old stress
  everywhere, profile history, and the current B V term);
- `separate`: integrate the load separately within each cell, then combine
  with that cell's ordinary residual before constrained global assembly;
- `global_sum`: add independently assembled global `newton` and `load`
  vectors, diagnostic only.

With ASPECT's RHS sign, the affine prediction is
`rhs(x+dx,V+dV) = rhs(x,V) - A dx + B dV`. Bulk dx is measured after actual
trial arithmetic and physical-pressure normalization, with homogeneous
constraint rows zeroed and pressure divided by the production scaling.
The V increment includes the manager's represented trial arithmetic; its
discrepancy from explicit surface V is reported. `represented_action_error`
compares A applied to represented dx against A applied to the requested
alpha-scaled Newton direction. `frozen_load_change` subtracts B dV from the
load difference. Block 0 is velocity, block 1 scaled continuity.

Compare `newton_affine_error`, `frozen_load_change`,
`separate_affine_error`, and `global_sum_affine_error` to locate loss of
consistency; a small linear residual alone does not establish nonlinear
convergence. Active/free counts and unchanged merit quantities are recorded.
This diagnostic is not a general decomposition for arbitrary face loads or
non-affine material models.

Commands (repository root; builds use `-j4`):

```sh
cmake --build build-pf-cpdi -j4
cmake --build build-pf-cpdi/tests --target phase_field_fault_stage_i phase_field_fault_stage_j_temperature phase_field_fault_stage_j_temperature_mpi -j4
ASPECT_K1_FLOOR_AUDIT=1 timeout 1200 build-pf-cpdi/aspect benchmarks/reconstructed_fault/uniform_shear/nonuniform/residual-audit/stage_i.prm
build-pf-cpdi/aspect --test '[fault_domain_quadrature],Stage-I*'
mpirun -np 2 build-pf-cpdi/aspect --test '[fault_domain_quadrature],Stage-I*'
ctest --test-dir build-pf-cpdi/tests --output-on-failure -R '^phase_field_fault_(stage_j_temperature(_mpi)?|stage_i_rollback(_mpi)?|linear_exhaustion)$' -j1
```

The positive lifecycle/temperature/benchmark postprocessors now require
`SolverControl::success` and a finite final maximum normalized residual below
the configured tolerance. Intentional exhaustion/rollback fixtures are
separate and unchanged. No expected output has been refreshed.
