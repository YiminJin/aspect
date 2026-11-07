# One attempted step: state and lifecycle

[Overview](../MATURE_MOVING_FAULT.md) · Prerequisite: [mechanics](mechanics.md)
· Next: [configuration and code map](configuration.md)

## Coherent direct-mode trace

Read the ordinary [Simulator::solve_timestep](../../../../source/simulator/core.cc)
alongside this trace. The material hooks precede the existing mesh/reaction and
solver-scheme dispatch; they do not invoke a special timestep executor.

```text
accepted parent: tau-, phi-, geometry-, age-, slip-, Z-, accepted velocity v-

ONCE PER ATTEMPT (new dt on retry):
  prepare_material_history(v-, endpoint_time, dt)
      retain accepted parent and rollback data; sample accepted bulk spin
      return immutable borrowed inputs
  transport_material_history(inputs)
      advect tau- -> transport.physical_stress
  stage_material_geometry(inputs, solution, linearization)
      move labels/support, pull back phi-/obstacle, bind candidate, rebuild I+
  rotate_material_history(inputs)
      project Q * transported_tau * Q^T -> candidate.objective.tensor = tau_bar

ORDINARY SOLVER-SCHEME DISPATCH:
  solve_reconstructed_fault_stokes(...)
      repeat inner mechanics at fixed correction
      refresh reaction/correction from tau_bar and current (v,V)
      terminal inner solve with converged private V -> pending callback

ORDINARY TIMESTEP ACCEPTANCE:
  accepted: callback installs rates and commits endpoint bulk/surface history;
            finish_dg_material_step(true) releases scratch/rollback owner
  rejected: discard pending candidate, finish_dg_material_step(false) restores
            accepted geometry/history; ordinary solver restores bulk bookkeeping
```

The [discrete operations](discretization.md) and [geometry operations](geometry.md)
read the same parent and velocity. Only the endpoint law sees current Newton
velocity/slip. `objective.tensor` never becomes the preceding outer iterate.
At acceptance `prepare_fe_strain_commit` recomputes the endpoint candidate using
terminal rates and the same fixed anchor. A final solve is not itself permission
to advance age or increment publication counters.

## Actual ownership

The structs are in [material_history_internal.h](../../../../include/aspect/material_model/phase_field_fault/material_history_internal.h),
[fe_strain_internal.h](../../../../include/aspect/material_model/phase_field_fault/fe_strain_internal.h)
and [tensor_history_internal.h](../../../../include/aspect/material_model/phase_field_fault/tensor_history_internal.h).

| Owner/member | Meaning and lifetime | Writer → readers |
|---|---|---|
| `DGMaterialTrial::accepted_history` | Accepted physical parent, retained through attempt | Capture at preparation → all staging, rollback |
| `FEStrainState::bulk_history.tensor` | Tagged DG history; `primary` is physical tau in direct mode | Initialization/accepted commit → transport, physical observations |
| `FEStrainState::bulk_history.native` | Mixed FE vector: phase in DG mode, zero velocity entries | Initialization/profile staging → phase sampling; native alternative has different velocity meaning |
| `DGMaterialTrial::candidate_history` / bound `fe_strain_state` | Private staged profile/geometry state | Geometry staging → rotation, normalization, mechanics |
| `DGMaterialTrial::transport.physical_stress` | Advected tensor, attempt-only scratch | Transport → rotation |
| `FEStrainState::objective.tensor` | Fixed transported/rotated anchor | Rotation once → every reaction and mechanical trial |
| `DGMaterialReaction::{bulk,rates,candidate}` | Frozen correction iterate, not accepted history | `refresh_material_reaction` → inner residual observations |
| `FEStrainUpdate::bulk_history.tensor`, `Z`, `I` | Unaccepted reaction result | `make_material_reaction` → correction or validated commit |
| `DGMaterialTrial::manager` | Before swap: candidate clone; after swap: rollback geometry/properties | Geometry staging → restoration or release |
| Manager fault properties | Age, slip, memory/normalization at material labels | Accepted surface-history commit → incoming friction/history |
| `FEStrainState::observations` | Derived stress/particle/mechanical/tensor-point samples | Sampling → assembly; invalidated on relevant state changes |
| `TensorField::space` | Shared FE infrastructure and numerical geometry cache | Space creation/transfer → projection/advection/evaluation |
| Simulator `pending_dg_endpoint_publication` | Attempt-only callback with terminal bulk/rates and solver diagnostics | Terminal inner solve → ordinary acceptance gate |

`MaterialHistoryInputs` borrows accepted state and ghost-readable velocity;
it copies neither large field and is not stored/serialized. Copying a
`TensorField` copies coefficient vectors and shares `Space`. Copying `DGHistory`
also shares its optional residual field pointer; consumers treat that parent
as immutable and allocate replacement outputs. `FEStrainState` copies its native
vector and stress observation map, while particle/mechanical/tensor-point cache
pointers share their existing objects. These are deliberate lifetimes, not a
promise of a deep clone of the entire simulation.

## Binding, synchronization and failure

`stage_material_geometry` swaps the staged manager and candidate FE state into
view together, writes candidate phase into bulk/linearization vectors and
rebuilds normalization. The point cache contains shape/phase/association samples,
not independent stress history; geometry motion/transfer clears it. Reaction
refresh clears stress/mechanical caches but preserves the fixed objective anchor.
The existing mechanical sampling mutex covers cache construction/access, and
returned cell arrays remain immutable during threaded assembly.

Numerical projection builds owned coefficients then collectively constructs
readable ghosts; remote geometry/profile sampling is collective outside assembly.
No new MPI call is hidden in a cell projection callback. Surface reductions and
bulk work maps retain their established synchronization. Timestep/outer scratch
is prepared before threaded constitutive readers run.

`finish_dg_material_step(false)` restores saved geometry, accepted FE history,
normalization, temperatures and particle diagnostics when a candidate was bound,
then clears correction/transaction data. Inner exceptions restore assembly
controls, constraints and surface nonlinear state. The ordinary timestep owner
handles rejected attempts, recoverable failures and escaping exceptions; it
restores bulk solution bookkeeping and closes the material transaction. The
outer solver now has no recursion flag requiring a separate exception reset.
A retry prepares from the accepted parent with its new interval.

`TimeStepping::Manager::update()` selects the next interval and the current
accept/reject decision **before** the pending callback publishes. The existing
reconstructed-fault age predictor therefore reads the earlier committed age
and rate, not the just-completed endpoint. For an uninterrupted moving DG run,
step n's proposal uses surface n-2 (surface zero for the first two intervals).
Its logarithmic state-change setting bounds that prediction, not the actual
endpoint age change. The layered campaign measures both in
`check_layered_timestep.py`; neither a passed predictor nor nonlinear convergence
is an endpoint accuracy guarantee. Publication remains behind the rejection gate.

Timestep zero follows existing initialization, not material transport or an
extra rotation. Nonpublishing endpoint-phase/benchmark calls retain their scratch
contract and return private rates. The stationary objective branch can reuse its
anchor keyed by parent, step and interval; it is not the moving-step driver.

## Restart boundary

[dg_accepted.cc](../../../../source/material_model/phase_field_fault/dg_accepted.cc)
and [dg_transfer.cc](../../../../source/material_model/phase_field_fault/dg_transfer.cc)
retain the serialization/transfer boundary. Schema7 identifies complete material
state, representation, law/integrator, profile/support and material inputs.
Direct and separated histories are not automatically converted. Internal names,
callback arguments and solver decomposition do not change packet order or
checkpoint meanings. Caches, borrowed views, advected scratch, correction
iterates, accepted-spin quadrature scratch and pending callbacks are rebuilt rather than serialized. The bounded
verification includes a fresh two-rank prefix restarted on four ranks; this is
local lifecycle evidence, not permission to resume production with changed laws.

## Constitutive-spin identity

`Constitutive spin = total material` remains the default and retains old
checkpoint semantics. Moving schema7 bulk-spin packets append the versioned
`constitutive spin:bulk slip subtracted v1` identity to their material-law string.
Stationary total spin keeps schema6; stationary bulk spin uses schema8 with
an explicit material-law identity. Cross-law loading is refused in either
direction; there is no production checkpoint conversion.

At timestep zero the new selector takes the accepted parent without rotation.
The converged initial coupled solve commits its slip rates when it publishes
its accepted velocity; it advances no physical stress history. The first
physical attempt therefore samples that matched pair, not the initial minimum
rate or predictor. `DGMaterialTrial` owns the immutable spin snapshot and drops
it on acceptance or rollback; it never becomes an independent physical history.

Native DGQ2 phase uses the existing material transaction. Transport returns a
candidate phase/obstacle from the immutable accepted parent; staged geometry,
normalization, surface histories and mechanics consume it before the same joint
publication or rollback. Phase assembly does not publish material history.
DG checkpoints carry native phase-space metadata and material schema9, including
all nine phase modes per cell; rank-changing restart preserves the space. Missing
phase-space metadata denotes historical Q1, not permission to reinterpret it as DG.
Adaptive DG phase/obstacle transfer is explicitly rejected. The
[phase report](../DGQ2_PHASE_FIELD_REPORT.md) tracks lifecycle qualification.
