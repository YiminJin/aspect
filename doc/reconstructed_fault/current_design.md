# Reconstructed Fault — Current Design

**Status:** Current implementation decisions.  
**Purpose:** This file is intentionally short. It records only the design decisions that are currently settled.

## Authority

For reconstructed-fault development, use this authority order:

- `doc/reconstructed_fault/pf_rsf.tex` is authoritative for the continuum constitutive model.
- `doc/reconstructed_fault/phase_field_fault_redesign.md` is authoritative for architecture, discretization, lifecycle, MPI design, and staged implementation.
- This file records settled implementation decisions and is subordinate to those two authorities where they overlap.
- `doc/reconstructed_fault/specification.tex` and older redesign notes remain useful background, but are superseded where they conflict with the authorities above.
- The current source tree is authoritative for existing class names, APIs, and reusable ASPECT/deal.II infrastructure.
- If this file conflicts with the current implementation in a way that affects architecture or scientific behavior, report the conflict rather than silently redesigning the method.

## 1. Current scope

The framework is intended to support a sharp reconstructed fault derived from a phase-field model.

The current implementation target is **2-D only** where a dimension-specific algorithm or data structure is required.

The public design should avoid unnecessary assumptions that would make a future 3-D implementation impossible, but the current code does not need to implement 3-D propagation, 3-D surface remeshing, branching, merging, or nucleation away from prescribed faults.

## 2. Initial pre-existing fault geometry

The initial pre-existing fault geometry is **prescribed geometrically by the user/model setup**.

For each pre-existing fault, the prescribed geometry is a connected, non-branching 2-D curve/graph.

The intended initialization sequence is

\[
\text{prescribed fault geometry}
\rightarrow
H
\rightarrow
\text{phase field}
\rightarrow
\text{reconstructed sharp fault}.
\]

The same prescribed fault geometry serves two purposes:

1. It is used to initialize the crack-driving field \(H\), using the appropriate stationary phase-field profile, length scale, geometric model, and core-phase-field value.
2. It provides the reference geometry, topology, ordering, and approximate length for the initial sharp-fault reconstruction.

The reconstructed sharp fault is **not required to coincide exactly with the prescribed graph**. The solved phase field may be used to correct the sharp fault, especially in the direction normal to the prescribed curve.

Because the initial graph already provides topology and length, the initialization does **not** need to discover a seed point, trace an unknown curve, or estimate the number of structural points from the number of active phase-field nodes.

The final structural spacing should be approximately uniform in physical arc length. The exact reconstruction/resampling algorithm is deferred to a later implementation stage.

## 3. `ReconstructedFault<dim>` responsibility

`ReconstructedFault<dim>` is primarily a **geometry/topology container**.

For the current 2-D implementation, the preferred representation is a simple application-owned ordered polyline.

A minimal representation may be based on an ordered sequence of points:

```cpp
std::vector<dealii::Point<dim>> vertices;
```

with consecutive vertices defining fault cells implicitly:

\[
\text{cell } i = (i,i+1).
\]

Do not introduce `dealii::Triangulation<dim-1,dim>` or `dealii::Particles::ParticleHandler` merely to store the reconstructed-fault points unless a later concrete requirement demonstrates a clear advantage.

Do not duplicate geometry unnecessarily between a custom registry and another mesh container.

Persistent committed vertices are append-only. Once committed, old fault vertices do not move.

A geometry-version counter may be used to invalidate geometry-dependent caches. A 64-bit scalar is acceptable for the version counter; ordinary fault vertex/cell indices may use `unsigned int`.

## 4. Geometry versus material-model data

The reconstructed-fault framework must not hard-code material-model-specific quantities such as

- slip rate \(V\),
- state variable \(\theta\),
- friction coefficient,
- normal stress,
- cumulative slip,
- hydraulic variables,
- or any other particular constitutive state.

The geometry class should therefore not define a fixed `FaultState` structure.

A future material model may use RSF, simplified rate-dependent friction, or another law entirely.

When fault-associated properties are needed, use a **generic material-independent property mechanism**. A reasonable design direction is a contiguous property array with runtime-defined property names/component counts, similar in spirit to ASPECT/deal.II particle-property storage, but without inheriting the distributed-particle machinery unless it is actually needed.

Do not implement the full property system in the geometry-only stage unless the current task explicitly requires it.

Tangents, normals, search structures, interpolation weights, and similar quantities are geometric information or reconstructible caches, not constitutive state.

## 5. Why `ParticleHandler` is not the default fault container

The reconstructed fault is intended to be a small structural object, likely replicated on all MPI ranks.

deal.II `ParticleHandler` is designed for particles associated with bulk mesh cells, with distributed ownership, ghost particles, migration, and particle-cell bookkeeping.

Those features are useful for the existing bulk/CPDI particles but are not automatically useful for the reconstructed-fault vertices.

Fault-surface connectivity must also be represented explicitly or implicitly; a set of particles alone does not provide the surface/curve topology required for interpolation.

Therefore, do not use `ParticleHandler` for reconstructed-fault points by default.

The property-storage ideas behind the particle infrastructure may still be reused as design inspiration.

## 6. Existing implementation interfaces

The following modules contain existing functionality that should be inspected and reused where appropriate:

- `source/simulator/phase_field.cc`
- `source/particle/particle_domain.cc`
- `source/material_model/rheology/fault_friction.cc`

The current reconstructed-fault implementation should reuse existing authoritative APIs rather than introduce duplicate physical/numerical parameters.

In particular, inspect the existing phase-field interfaces for quantities such as the acceptable phase-field range and length scale before adding new parameters.

## 7. Existing code that must not define the new architecture

Do not base the reconstructed-fault architecture on:

- core-phase-field extension/reconstruction functions such as `PhaseField::extend_core_phase_field`;
- the deprecated `tmp/material_model/phase_field_rsf.cc` snapshot;
- the deprecated `tmp/particle_property/phase_field_rsf.cc` snapshot.

These belong to the previous phase-field RSF algorithm.

Specific numerical routines from the old implementation may still be reused or refactored when explicitly appropriate. For example, the local return-mapping/root-solving procedure may be reusable.

Reuse an isolated algorithm only when its inputs, outputs, and assumptions remain valid in the reconstructed-fault formulation.

## 8. Phase-field dependency

Reconstructed-fault support requires the phase-field module to be enabled.

Prefer an early assertion or parameter-consistency check if/when a reconstructed-fault enable switch is introduced.

Do not create a separate duplicated phase-field activation mechanism.

## 9. MPI philosophy

The bulk phase field and bulk particles remain distributed.

The reconstructed fault is expected to be small enough to replicate on all MPI ranks unless profiling later demonstrates otherwise.

Do not gather large bulk point clouds or all particles onto one rank.

For later coupling stages, prefer the pattern

\[
\text{local distributed processing}
\rightarrow
\text{MPI reduction of small fault-sized data}
\rightarrow
\text{replicated fault calculation}.
\]

The exact MPI implementation belongs to the stage that introduces the corresponding coupling operation.

## 10. Development workflow

Do not attempt to implement the entire reconstructed-fault framework at once.

For each stage:

1. inspect the relevant current ASPECT/deal.II code;
2. propose the smallest concrete interface needed for that stage;
3. preserve the scientific and architectural requirements in this file;
4. implement only that stage;
5. test it;
6. update this design note when an implementation decision becomes settled.

If a scientific or architectural question is unresolved, report it rather than inventing a permanent design.

Ordinary C++ implementation choices may be made by Codex when they do not alter the stated invariants or future extensibility.

## 11. Minimal geometry implementation

The geometry-only `ReconstructedFault<dim>` implementation establishes the
representation and basic access/update operations needed for a 2-D append-only
polyline.

It does **not** implement:

- initial phase-field reconstruction,
- prescribed-fault-to-\(H\) generation,
- generic fault-property storage,
- particle-to-fault projection,
- fault-to-quadrature-point interpolation,
- PCA,
- smoothing,
- propagation,
- RSF coupling,
- 3-D surface support.

Those will be designed and implemented incrementally after the geometry container is tested.

## 12. Distributed phase-field access

The initial sharp-fault reconstruction evaluates the solved Q1 phase field
directly at ordinary finite-element quadrature points on locally owned cells.
The earlier distributed arbitrary-point sampling wrappers were removed because
they had no independent production caller and are not used by the direct
finite-element reconstruction algorithm.

## 13. Prescribed-fault initialization of particle `H`

Each prescribed initial 2-D fault is represented by an ordered polyline with
one prescribed core phase-field value at each vertex. Core values are
piecewise-linearly interpolated at the closest point on the polyline.

For every locally owned particle, initialization evaluates each
material-specific `PhaseField::PhaseFieldProfile` at the distance to that
closest point. The resulting phase-field values are combined using ASPECT's
volume-fraction-weighted arithmetic average. The existing
`PhaseFieldHandler::stationary_crack_driving_force()` then computes `H` from
that averaged phase field, the interpolated core value, and particle-derived
composition fractions.

A fault contributes where the prescribed stationary profile exceeds the
existing lower acceptable phase-field bound. No contribution leaves the
particle's baseline `H` unchanged; more than one contribution is an error.
This profile initializes particle `H` only. Sharp-fault reconstruction uses the
subsequently solved Q1 phase field and belongs to a later stage.

## 14. Initial sharp-fault reconstruction

`ReconstructedFaultManager<dim>` is owned by `Simulator` when `Formulation /`
`Reconstruct faults from phase field` is enabled. It owns the prescribed faults,
the reconstructed connected faults, reconstruction parameters, and temporary
diagnostics. Individual `ReconstructedFault<dim>` objects remain geometry-only.

Reference faults are resampled at approximately uniform arc-length spacing set
by `Fault reconstruction / Structural point spacing`. Locally owned bulk cells
assemble the direct Q1 phase-field weighted data matrix, right-hand side, total
weight, and structural support. Only these fault-sized quantities are summed
over MPI. The globally assembled system is normalized by total phase-field
weight before adding the second-difference ridge term. `Fault reconstruction /`
`Ridge coefficient` supplies the dimensionless ridge coefficient and defaults
to one.

Prescribed faults are read from the single file named by `Fault reconstruction /`
`Prescribed faults file`. In 2-D, each non-comment line contains

```
x  y  phi_hat
```

in polyline order. A line containing only `---` ends one connected fault and
starts the next; `#` begins a comment. Each fault requires at least two vertices.
Rank zero reads the file once using ASPECT's distributed-file utility and
broadcasts its contents, after which every rank parses the same small replicated
fault description.

The tubular integration radius is the largest relevant stationary-profile
support plus one global cell-diameter margin. Stationary profiles determine the
integration region only; fitted offsets come from the converged Q1 field. The
generic reconstruction operation has no timestep condition. Its initialization
caller runs after the converged timestep-zero phase-field solve.

## 15. Reconstructed-fault visualization output

The `reconstructed faults` postprocessor writes the replicated reconstructed
geometry once from MPI rank zero as VTK unstructured-grid line cells. Each VTU
contains built-in identifiers: `fault_id` and per-fault `vertex_id` on points,
and `fault_id` and per-fault `cell_id` on cells. A PVD file records the time
series. Runtime-registered generic vertex properties are written automatically
as point data. All faults use the same manager-owned property schema and common
property indices. Cell properties and material-model-specific output remain
outside the current scope. When the distinguished slip-rate field is
initialized, its committed value is additionally written as the built-in
`slip_rate` point-data array.

## 16. Generic reconstructed-fault vertex properties

`ReconstructedFaultManager<dim>` owns a minimal material-independent registry
for vertex properties. A property is identified by a runtime name and a
component count. Registration returns a stable property index and must be
completed before reconstructed faults exist. Every fault receives the same
schema and component offsets. Each fault owns one contiguous vertex-major value
array with layout

```
vertex 0 property component 0, ..., vertex 0 property component Np-1,
vertex 1 property component 0, ..., vertex 1 property component Np-1, ...
```

The manager supplies only the total component count when it creates a fault;
individual faults do not duplicate property names, component counts, or
offsets. Appending geometry preserves existing property values and extends the
array with signaling-NaN entries. This storage has no particle
ownership, migration, cell association, constitutive-law assumptions, or MPI
communication of its own; it follows the replicated ownership of the fault.

The generic property pool stores committed physical/material state only.
Temporary Newton quantities, trial constitutive state, friction coefficients,
derivatives, residual coefficients, and similar working data do not belong in
this pool. In particular, future committed values such as `Theta`, `T_coh`,
and `I_h` may use generic property storage, while their trial values and other
temporary constitutive data remain owned by the material model. The generic
property pool does not provide trial or rollback machinery.

## 17. Particle-to-fault property projection

Stage 5 projects generic components from the phase-field-associated particle
manager to registered reconstructed-fault properties by a consistent weighted
Q1 least-squares solve. Only locally owned particles contribute. The sampling
weight is the existing particle-domain volume; within the admitted influence
region the geometric kernel is one. Fault-sized tridiagonal matrices and packed
right-hand sides are summed over MPI, and every rank solves the identical
replicated systems.

Influence half-width is generic per-vertex projection geometry metadata owned
by `ReconstructedFaultManager<dim>`. The current initialization obtains it from
stationary-profile support, caches support for repeated prescribed core values,
and interpolates prescribed endpoint widths onto the resampled fault. The
projection operator itself consumes only the resulting half-width values and
does not depend on stationary-profile formulas or core phase-field values.

Particle association uses finite segment-normal profiles. A segment contributes
only when its unconstrained orthogonal coordinate satisfies `0 <= xi <= 1` and
the normal distance is within the Q1-interpolated half-width. This excludes
tangent extensions beyond the two true open tips. At internal vertices both
incident segments remain candidates, and the candidate with the smallest
absolute normal distance is selected. The current implementation assumes small
enough turning angles that no separate corner construction is needed.

A particle may contribute to at most one fault. Admission to more than one
fault is an error, but the implementation does not attempt exhaustive geometric
detection of overlapping influence regions. Non-overlapping fault influence
regions are a current model assumption. Closed loops, intersections, branching,
joining, coalescence, and 3-D projection are unsupported.

The projection cache stores every locally owned particle's ID, position,
particle-domain volume, and optional fault segment/coordinate association. It
also stores geometry versions and reusable tridiagonal LDL-transpose factors.
Cache reuse is primarily intended within a timestep or nonlinear solve.
Particle advection normally invalidates it; changes in particle IDs, iteration
order, positions, domain volumes, fault geometry versions, or projection
metadata cause a rebuild. Particle property-value changes alone do not.

## 18. Distinguished fault slip-rate field

`ReconstructedFaultManager<dim>` owns the replicated nodal slip-rate field
`V` separately from the generic property array. `ReconstructedFault<dim>`
therefore remains a geometry/property container and does not acquire
constitutive state. The generic property name `slip_rate` is reserved for this
built-in kinematic field.

New reconstructed geometry has no implicit physical slip-rate value. A caller
must initialize one finite nonnegative value per fault vertex. The manager
provides Q1 segment interpolation and three distinct lifecycle states:
timestep-committed `V_k`, current accepted Newton iterate `V_current`, and
line-search candidate `V_trial`. A nonlinear solve begins by copying `V_k` to
`V_current`. Every candidate is formed from `V_current`, so rejected step
lengths cannot accumulate. Accepting a candidate changes only `V_current`;
only successful nonlinear convergence commits it to `V_k`. Rejecting the
nonlinear solve restores `V_current` from `V_k`. The manager enforces `V >= 0`
but does not know the stronger constitutive/numerical bound `V_min`.

Checkpoint/restart stores reconstructed geometry, projection half-widths, the
generic property schema and values, initialization flags, and timestep-
committed `V_k`. Newton iterates, trial values, and all projection/factorization
caches are reconstructed: after load, current `V` equals `V_k` and no nonlinear
solve or trial is active. Reconstructed-fault checkpoints created before this
serialized manager state was introduced are not guaranteed to be compatible;
no backward archive migration is provided.

The existing mutable `get_fault()` and public `ReconstructedFault::append_*()`
interfaces can temporarily bypass manager ownership and make the number of
slip-rate values differ from the number of vertices. Fixed-geometry stages may
continue using the current API. Before fault propagation or any other topology
change is implemented, geometry mutation must be routed through
`ReconstructedFaultManager` so that all manager-owned nodal fields remain
aligned. No appended-vertex physical state rule is inferred here.

The manager archive round-trip is the current restart test. A full Simulator
filesystem checkpoint/restart test remains mandatory before reconstructed-
fault restart support is considered complete, but it does not block the next
fixed-geometry development stage.

## 19. Maxwell particle stress history

The reconstructed-fault material model uses the particle property plugin
`maxwell stress` to store exactly one committed symmetric tensor
\(\boldsymbol\tau_{k-1}\) per bulk particle. It stores no current slip rate,
surface state, cohesive traction, fault direction, second old-stress tensor, or
Newton working value. The plugin has no automatic particle update and applies
no objective rotation.

Every independent tensor component is initialized explicitly from one
particle-advected compositional field of type `stress`. The normal mapped-
particle-property component syntax associates those fields with the components
of the single `maxwell stress` property. The mapped-particle-property parameter
must be nonempty; the implicit one-to-one fallback is not accepted for this
constitutive history. The plugin evaluates the active
initial-composition model at particle creation; missing, duplicate, or
out-of-range component mappings are errors. Late particles interpolate the
current particle history. Particle migration and checkpoint/restart use the
existing particle infrastructure.

`MaterialModel::PhaseFieldFault` owns the non-rotational time-discrete Maxwell
law

\[
\beta=\exp(-\Delta tG/\eta),\qquad
\kappa=-\eta\,\operatorname{expm1}(-\Delta tG/\eta),\qquad
\boldsymbol\tau_k=2\kappa\dot{\boldsymbol\epsilon}^{b}_k
                  +\beta\boldsymbol\tau_{k-1}.
\]

The `expm1` expression is the authoritative evaluation of
\(\eta(1-\beta)\); code must not recover it by subtracting `beta` from one.
The effective bulk strain rate will later include the reconstructed-fault slip
correction. Stage 2 only provides the constitutive operation and does not add a
Stokes or surface coupling.

There is no separate Maxwell-stress transaction. Particle stress remains
unchanged throughout the nonlinear solve, including rejected Newton and line-
search trials. A failed solve therefore has no particle stress to restore.
After mechanical convergence, one pass over locally owned particles evaluates
and writes \(\boldsymbol\tau_k\). This post-convergence pass will be connected
when the later coupling stage provides the converged slip-corrected strain
rate.
