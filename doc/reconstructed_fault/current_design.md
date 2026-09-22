# Reconstructed Fault — Current Design

**Status:** Current implementation decisions.  
**Purpose:** This file is intentionally short. It records only the design decisions that are currently settled.

## Authority

For reconstructed-fault development, this file and
`doc/reconstructed_fault/specification.tex` are authoritative for scientific
algorithms, architecture, ownership, and MPI design. They are maintained as a
consistent pair: this file records the concise settled decisions and the
specification gives their detailed contract.

Older reconstructed-fault redesign notes and `pf_rsf.tex` remain useful
derivation history but are not architectural authorities. An isolated
continuum equation may be promoted from them only by recording the approved
equation in both authoritative documents before implementation.
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

The physical phase-field range, reconstruction activation threshold, and
fault-model upper admissibility threshold are distinct invariants. The
physical range is `[0,1]`. Reconstruction and refinement use the activation
threshold, while prescribed fault-core values and legacy slip-rate
normalization use the activation and upper-admissibility thresholds. These
quantities are obtained through separate `PhaseFieldModel` accessors; none is
duplicated in reconstructed-fault parameters. The existing phase-field length
scale is likewise reused through `PhaseFieldHandler`.

The generic upper-admissibility accessor defaults to the physical upper bound
`1.0`; this does not impose the fault model's interior cutoff on future phase-
field models. Algorithms that require a strict interior endpoint validate that
contract explicitly. `PhaseFieldFault` overrides the default with `0.99` for
the legacy slip-rate normalizer.

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
material model's phase-field activation threshold. No contribution leaves the
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

The BP3 implementation task additionally permits fixed prescribed geometry:
`Fit prescribed geometry to phase field = false` retains the resampled input
polyline, with the unchanged profile-derived admission widths. The default
remains the existing phase-ridge fit. A benchmark may prescribe independent
phase DoFs through the existing constraint signal; this does not constrain
mechanics or change the production phase equation.

Selected surface V DoFs may have prescribed positive values. The manager
lifts them only into the private Newton base, leaving committed V intact until
convergence. Their perturbations vanish and their friction equations are
replaced by the essential kinematic condition: condensation and residual norms
use the same principal free block as the existing active-set machinery.
They cannot be released by the lower-bound active set. Caller-supplied boundary
configuration is reapplied after reconstruction/restart, rather than treated
as constitutive history. Unprescribed DoFs retain existing behavior.

**Approved discrete revision, 2026-09-09:** the domain-integrated rule in
`benchmarking/stage_K2_domain_quadrature_addendum.md` supersedes point-volume
quadrature throughout this document, including surface residual/Jacobian,
G, mass-based norms and diagnostics. The former source obeyed its former
specification. Preserve the admitted center set and each full existing domain
without normal clipping. The objective is now
\(\sum_p\int_{D_p}(z_p-z_\Gamma(s(x)))^2\,dx\); equivalently replace
\(\sum_p m_p N_i(\xi_p)F_p\) by
\(\sum_{p,q}w_{pq}N_i(\xi_{pq})F_{pq}\), with the corresponding second Q1
factor in mass/Jacobian terms. Bulk FE inputs and particle histories remain
parent-center/P0 inputs; surface Q1 fields and nonlinear responses are evaluated
at every integration coordinate. The first integrated geometry is straight
ordered 2-D faults, including inclined/nonuniform collinear segments. Split
domains at fault-node planes and projected polygon vertices. Beyond true tips
use constant endpoint continuation, without admitting new centers or joining
endpoint DoFs. Three-point Gauss integrates the linear transverse width times
Q1 products exactly; nonlinear accuracy is checked separately. Curved domain
partitions were unsupported by the first implementation. The approved extension
in `benchmarking/stage_K2_polyline_quadrature_addendum.md` now defines the
actual open 2-D polyline map: nearest finite segment-normal projection without
reapplying width, with nearest-vertex continuation in corner/tip gaps. Split
at endpoint planes and distance bisectors; use each selected segment's own
frame. Parent admission/fault assignment and full measure remain unchanged.
Overlap, branching, closed loops, self-intersections and 3-D remain unsupported.
History-commit timing is unchanged.

For the approved 2-D Box periodic-domain correction, see
`benchmarking/stage_K3_periodic_domain_addendum.md`. Domains use the global
period and geometry-only image neighbors. Surface quadrature integrates all
physical-box fragments of each admitted real parent's full periodic domain,
once, with the existing open-polyline map. Periodic bulk geometry does not
identify the independent surface endpoint DoFs. CPDI samples use periodically
mapped FE locations and the existing unique-owner rule.

The bounded K2 performance pass caches only the geometric point-location maps
for the adaptive `I_h` batch sequence of the last preparation. Exact coordinate
equality and deal.II's mesh-validity flag are required on every rank before a
batch can be reused; any rank's mismatch causes collective reconstruction.
Phase-field values and cell-diameter reductions are evaluated afresh. Unused
trailing batches are discarded, and mesh-deformation configurations use fresh
lookups. This does not change adaptive panels, quadrature, tail termination,
support, residuals, or history publication.

The approved K5 extension additionally retains a completed surface `I_h` value
field. Reuse requires exact owned phase entries, fault vertices/versions, a valid
mesh lookup, and unchanged degradation-law parameters and projected surface
mixtures on every rank. Mixtures need not match when all material degradation
laws are identical (friction projection still runs). Mesh deformation disables
reuse. Restart and parameter parsing invalidate the transient cache; validity
is published only after successful integration and surface projection. A hit
performs neither adaptive integration nor remote FE sampling. Endpoint truncation,
full integral values, quadrature/tail tolerances and history timing are unchanged.
At identical current/previous local phase and I_h, the history-localization
difference is exactly zero. Moving parents still require fresh local sampling.

Stage 5 projects generic components from the phase-field-associated particle
manager to registered reconstructed-fault properties by a consistent weighted
Q1 least-squares solve. Only locally owned parents contribute. The sampling
measure is their full existing particle domains, with geometric kernel one.
Fault-sized tridiagonal matrices and packed
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

The qualified K5 BP3 bottom continuation is a narrow exception for bulk
mechanics only: a straight through-bottom fault may continue its endpoint
fields into the missing tangent-extension wedge inside the physical Box.
The ordinary normal-width cutoff is not reapplied inside this wedge: Q1 phase
can remain positive beyond it. Outside the wedge the original admission is
unchanged. Actual physical FE phase and constantly
extended completed endpoint I_h define chi there. Existing admitted bulk
values are unchanged. Stokes residual/B assembly and particle Maxwell-history
strain subtraction use the same bulk-source coordinate map. Particle/surface
admission, surface-domain quadrature, ownership and open connectivity do not
change. It is opt-in for the fixed-profile mature BP3 through-boundary fault
with prescribed bottom Vp, not a general internal-tip continuation policy.
`Postprocess/BP3/Bottom normalization completion file` selects the paired
normalization/source treatment. It must reference immutable completion data
for the exact mesh, profile and fault; the same selector is reattached on
restart before rebuilding non-checkpointed caches. The table and path are
external run inputs, not checkpoint payload. The legacy diagnostic switches
remain available for reproducing denominator-only comparisons. See
`bp3/stage_K5_bottom_source_continuation_report.md` for qualification and limits.

The subsequent bounded **top experiment** pairs outside-top normalization
completion with the analogous in-box straight tangent wedge. It is restricted
to frozen mature uniform prescribed slip (`ASPECT_BP3_TOP_SOURCE_EXPERIMENT`).
The endpoint coordinates are the last segment at xi=1, so bulk assembly and
history subtraction read the endpoint field rather than a hardcoded rate.
Surface admission and its weak measure remain unchanged for this experiment.
Success with prescribed slip does not qualify a free endpoint: its continued
bulk virtual work and the corresponding surface traction/G contribution must
be checked before adoption. No free-RSF top continuation is enabled by this
experimental switch.
The uniform experiment passed, including the endpoint B finite difference,
but free-endpoint adoption remains blocked on surface-measure qualification;
see `bp3/stage_K5_top_source_continuation_report.md`. No mature-RSF top replay
is part of that evidence.

**Authorized work-measure qualification (K5):** an explicit simulator-side
`ReconstructedFaultSurfaceSystem::enable_bulk_work_measure()` mode changes
mechanical surface assembly only, for one straight, frozen mature 2-D fault
with true normal stress. Default particle-volume assembly is unchanged.
Visit each owned physical Stokes QP once using its exact existing bulk-source
map, phase, FE material fields and frozen working FE stress. All mechanical
terms use JxW*chi*N_i, including background driving, friction and damping;
K=-dR/dV, G=dR/d(u,p), and the consistent mass matrix use the same measure.
Zero chi contributes nothing. Surface state is interpolated at the same
extended fault coordinate and remains committed/frozen during mechanics.
The same QP positions and frozen coefficients feed the existing G action
and optional sparse matrix machinery (no particle-center substitution).
Only the shear part of G is work-adjoint to B on homogeneous bulk directions;
normal/friction and pressure terms remain non-associated.

Absolute Q1 slip-rate evaluation in the bulk source and both surface integration
paths preserves the nodal convex hull in floating-point arithmetic. Equal nodes
at the lower bound and exact endpoint coordinates retain their stored values;
descending profiles are evaluated from the smaller endpoint to avoid cancellation.
This is not a clamp to the constitutive minimum: invalid nodal inputs are not
repaired. Linear B/G/K actions retain the same mathematical Q1 shape derivatives.

Mechanical weak residuals change from Pa*m^2 to Pa*m in 2-D. The new mass
matrix changes from m^2 to m, so M^-1 R, R_i/(M*1)_i and the existing consistent
RMS norm remain in Pa. K*V supplies the same traction characteristic after
that mass conversion; the fixed merit scales, nonlinear tolerances, bound
and fraction-to-boundary rules are not retuned. Generic property projection,
ownership, topology and history commit remain unchanged. Fixed benchmark
background fields are loaded without recalibration. The BP3 qualification
switch `ASPECT_BP3_WORK_MEASURE=1` currently requires a fresh noncommitting
free-top run with paired boundary completion/source enabled.

The separately authorized committing follow-up is explicitly selected by
`Postprocess/BP3/Committing work-measure replay = true`. It starts fresh,
retains mature friction, immutable initial background, both paired boundary
treatments and ordinary accepted-state publication (nodal split Theta and
particle Maxwell stress; inert H and zero C). It is not a restart conversion
or the noncommitting qualification switch. Existing time-step controllers
may insert smaller steps between saved comparison times. Diagnostic native
work averages and observational legacy parent/domain FE-history averages
are labelled separately; no post-commit reevaluation uses newly published
particle stress as old history. The bounded replay is capped at 40 minutes
on four ranks and approximately 29.24 years physical time.

The separately authorized bounded candidate-state experiment is documented in
`bp3/stage_K5_coupled_substeps_addendum.md`. It permits only two/four substeps
from the saved revised-work step-9 state: candidate nodal Theta(V) is used
during mechanics with its nonsymmetric Jacobian, then ordinary terminal
publication commits that candidate once. This benchmark alternative does not
replace the default split-state algorithm or qualify general replay restart.

The separately authorized frozen BP3 free-trace diagnostic is specified in
`bp3/stage_K5_free_trace_addendum.md`. At the 40-km prescribed/free junction,
one node represents the free trace while the deep trace stays exactly Vp.
Mechanical weights on the adjacent fully prescribed segment become (1,0),
eliminating its constant prescribed trace through its other prescribed node.
Geometric/property interpolation remains unchanged; B, residual, K, G and
norms share the modified kinematic basis. This explicitly selected experiment
must roll back before any history or kinematic publication. It is not a new
default or a general discontinuous/committing surface representation.
The later explicitly authorized seven-step committing comparison is specified
in `bp3/stage_K5_trace_replay_addendum.md`. Its separate benchmark selector
also splits state/slip on the prescribed side, while leaving geometry and
material-property interpolation unchanged.

The later fully frictional comparison is explicitly authorized in
`bp3/stage_K5_fully_frictional_addendum.md`: remove every prescribed deep
rate, retain continuous Q1 fields and the work-measure boundary extensions,
and start from the original physical initial data for seven steps. This is
a changed BP3 boundary condition, not the default or a split-trace method.

**K5 consolidation:** the successful fully frictional configuration is now
selected explicitly by benchmark parameter `Fault loading configuration =
fully frictional`, with `prescribed deep slip` retained as the reference/default.
See `bp3/stage_K5_cleanup_report.md`. The earlier free-trace, candidate-state,
frozen-cohesion and noncommitting A/B descriptions above record historical
experiments, not current runtime choices. Their sources and evidence are
archived; the maintained core uses continuous Q1 kinematics and committed
state during mechanics. The split history cycle and work-measure equations
are unchanged. No unrestricted boundary-extension default is introduced.

Bounded benchmark replays may select `BP3 replay complete`, which terminates
only after the accepted-state history audit reaches the last saved comparison
time, allowing eight machine-epsilon-scaled units of time-representation error.
This prevents a roundoff-sized extra Maxwell interval; it changes neither
physical timestep selection nor nonlinear acceptance criteria.

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

The manager also exposes one constitutively neutral scalar projection for
caller-computed values keyed by stable locally owned particle ID. It reuses
the same cached mass operator and MPI reduction and returns replicated nodal
values without storing them. Active associated particles must have one finite
input value; inactive particles may be omitted. The result includes
volume-weighted RMS and maximum projection residuals so a material model can
diagnose assumptions such as profile-uniform initial cohesive traction.
The reverse local operation Q1-interpolates a registered fault property at
every active locally owned particle's cached fault coordinate and returns the
values keyed by stable particle ID. It performs neither a new geometry search
nor MPI communication.

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

## 20. Distributed adaptive normalization integral

`MaterialModel::PhaseFieldFault` owns the transient, recomputable current
normalization integral

\[
I_h=\int (1/\bar g-1)\,d\zeta.
\]

Every `QGauss<1>(3)` point on a reconstructed-fault segment has one
fault-surface material mixture \(\mathbf f_\Gamma\). Chemical compositional
fields are projected from particles to the generic fault-property storage,
Q1-interpolated at the surface quadrature point, and converted to material
fractions there. That mixture is fixed for the complete two-sided normal
profile. Thus every phase-field sample uses

\[
\bar g(\phi,\mathbf f_\Gamma)=\sum_m f_{\Gamma,m}g_m(\phi).
\]

Composition is never resampled along the normal. This approximation is
applicable only when every material/composition transition has characteristic
width \(L_{\rm mat}\gg\ell\). Models with transitions comparable to or narrower
than the diffuse-fault width are outside the formulation's validity.

Each chemical compositional field is stored as a separate scalar generic
fault property named `phase field fault chemical composition <field-name>`, in
the chemical-field introspection order. This keeps the corresponding output
arrays individually named. If there are no chemical compositional fields, no
chemical fault property is registered and an empty composition vector denotes
the background-only material mixture.

Profiles have deterministic fault-major, segment-major, quadrature-point IDs
and balanced contiguous MPI ownership. Owners retain adaptive state; all ranks
participate in batched distributed Q1 phase-field evaluation. Initial panel
width is one half the smaller of \(\ell\) and the local bulk-cell diameter.
Four- and eight-point Gauss estimates control bisection. Each side terminates
after two independent outer integral windows are negligible; the activation
threshold does not truncate the profile and no monotonic tail is required.
The completed quadrature-point integrals are projected to replicated fault
vertices with the consistent Q1 mass matrix.

The along-fault projection may use `I h surface quadrature subdivisions`
equal panels per element, with three Gauss points on each panel (default one
for compatibility). This resolves tangential bulk-FE column variation without
altering the Q1 space or mass matrix. Profile IDs are ordered by segment, then
panel, then Gauss point. The rule is fixed during a run; normal integration,
surface mixture sampling and any outside-boundary completion all use those
same origins and weights. Completion files with old counts/origins are rejected.
The resulting single projected field is used by every constitutive and
coupling consumer; neither a smoothed denominator nor a separate mechanical
normalization is introduced. Completed-value caches are invalidated at parameter
parsing; cell-interval caches additionally compare profile origins and normals.

When a physical boundary is located, its coordinate is independent of the
adaptive panel width. If the boundary-truncated panel is bisected for accuracy,
accepted subpanels continue until the stored boundary coordinate is reached;
accepting the first refined subpanel must not discard the remaining interval.

An approved opt-in cell-interval backend is specified in
`bp3/stage_K5_cell_profile_addendum.md`. It replaces per-sample distributed
location with cached ray/cell intersections and bulk-cell-owned quadrature,
not with cached evolving phase values. The remote backend remains the default
and validation reference. The equations, material mixture, tail coefficient,
physical boundary truncation and surface projection above are unchanged.

The authorized K5 bottom-normalization experiment is an opt-in exception,
not a changed default policy. In a fresh frozen mature uniform-sliding BP3
test only, `ASPECT_IH_BOTTOM_COMPLETION_DIAGNOSTIC` supplies fixed auxiliary
outside-profile integrals at the actual three-point surface quadrature origins.
Their profile owners add them to the in-box integrals before the existing
weighted Q1 RHS assembly; the surface mass and every physical mechanical
quadrature remain unchanged. The benchmark extends the bottom-resolution Q1
nodal phase grid below the box, validates it against saved physical FE samples,
and integrates only the missing bottom interval. Complete columns and the top
receive no virtual contribution. This fixed input is immutable during the
process; restart and evolving-phase use are rejected. Completed-value cache
hits reuse the augmented result. See `bp3/stage_K5_bottom_normalization_report.md`.

Following the paired source-support verification, the BP3 parameter
`Bottom normalization completion file` selects this same preprojection
completion together with bottom bulk-source continuation (described above).
Unlike the legacy diagnostic selector, it may be reattached on restart and
does not require all RSF nodes to be prescribed; the continued bottom endpoint
must still be prescribed to Vp. Only the tested frozen mature, straight,
through-bottom geometry is qualified. Completion tables must be regenerated
for a changed mesh/profile/fault, not reused as universal analytic constants.

Cold point location may conservatively reject points outside the global
reference-tolerance-expanded mesh enclosure. This optimization is limited to
exact `MappingCartesian` or degree-one `MappingQ`/`MappingQ1` on axis-aligned affine cells; unsupported
maps/shapes use the original lookup. The enclosure includes the existing
deal.II reference-cell tolerance plus outward roundoff padding. It is reduced
across MPI owners and invalidated with the geometric cache. Surviving requests
retain their order and ordinary shared-face/MPI ownership search, and their
samples are restored to the original request indices. Proved-outside samples
remain missing. No profile, physical, or integration tolerance changes.

The physical lower bound is zero, but the unconstrained Q1 solve may produce a
small negative numerical undershoot. For `I_h` only, degradation is evaluated
at

\[
\phi_{\mathrm{eff}}=\max(\phi_h,0).
\]

The activation threshold is not used for this clamp. The minimum raw sampled
phase field is tracked across all profiles and MPI ranks. The initial
empirical dimensionless error-detection threshold is \(10^{-4}\). It is an
internal invariant guard, not a physical parameter, solver tolerance,
convergence criterion, or user-adjustable numerical convergence parameter. A
global minimum below \(-10^{-4}\) is an invariant failure that reports the raw
value, threshold, and sample location. All raw samples must be finite. No analogous
upper clipping is permitted: a raw value above one is an invariant failure,
while \(\bar g=0\), including a possible \(\phi=1\) case, is reported explicitly
as an `I_h` singularity rather than a phase-field range violation. A tail value
\(\phi_h=0\) is admissible. Current \(I_h\) is private transient material-model
data in this stage and is not connected to the mechanical solve.

The consistent Q1 projection of positive quadrature-point normalization
integrals is retained. Because consistent projection is not positivity
preserving in general, positive nodal coefficients are a tested property of
the smooth intended profiles, not a generic mathematical guarantee.

Stage C accuracy is checked separately from fault reconstruction: the adaptive
kernel is compared with an analytic profile integral under quadrature- and
tail-tolerance refinement, and the distributed Q1 path is compared with an
independently integrated Q1 reference under bulk-mesh refinement. The Voro
lifecycle smoke fixture retains a low activation threshold only to bootstrap
geometry; reconstruction behavior, location, and accuracy are not under test
in that fixture.

## 21. Common cohesive state

`MaterialModel::PhaseFieldFault` owns the common cohesive law independently of
the selected fault-friction law. It registers two scalar generic fault
properties: committed cohesive traction $T^{\rm coh}_{k-1}$ and committed
previous normalization $I_{h,k-1}$. The manager stores and checkpoints these
values but does not interpret them. Current $I_{h,k}$ and all non-committed
cohesive responses remain private transient material-model data.

For fixed current phase field and history, the non-committing update is

\[
T^{\rm coh}_k=
\frac{\kappa_k V_k+\beta_k I_{h,k-1}T^{\rm coh}_{k-1}}{I_{h,k}},
\]

and the exact diffuse crack-strain-rate magnitude is

\[
\upsilon_k=
\frac{h_k}{I_{h,k}}V_k+
\frac{\beta_kT^{\rm coh}_{k-1}}{\kappa_k}
\left(h_k\frac{I_{h,k-1}}{I_{h,k}}-h_{k-1}\right).
\]

Consequently the history correction integrates to zero and
$\int\upsilon_k\,d\zeta=V_k$. At fixed history,
$\partial\upsilon_k/\partial V_k=h_k/I_{h,k}$. The previous pointwise
$h_{k-1}$ is derived from the previous FE phase-field solution; it is not an
additional persistent fault property.

For an initially reconstructed pre-existing fault, the committed state is not
zeroed. At each associated particle, form

\[
q_p=\bar g(\phi_{\rm eff,p},\mathbf f_\Gamma(s_p))
\sqrt{2\bar G(\mathbf f_\Gamma(s_p))H_p},\qquad
\phi_{\rm eff,p}=\max(\phi_p,0),
\]

using the prescribed initial particle crack-driving force and the Stage C
bounded-negative rule. The particle's $H_p$ and $\phi_p$ remain local, but
both $\bar G$ and $\bar g$ use the same surface mixture
$\mathbf f_\Gamma(s_p)$ obtained by interpolating the already projected Q1
chemical-composition fields at the particle's cached fault coordinate. A
particle-local composition must not be used for either constitutive mixture.
Consistently project $q$ to the replicated Q1 fault to obtain nodal
$T^{\rm coh}_0$, and commit current $I_{h,0}$ in the same initialization
operation. Volume-weighted RMS and maximum projection residuals diagnose the
profile-uniform-$q$ assumption but do not introduce a rejection threshold.

Stage D provides an explicit, unwired commit operation for testing and future
coupling. No timestep or nonlinear-solver signal commits cohesive history;
that can occur only after a later coupled slip-rate solve accepts a mechanical
timestep.

## 22. Generic fault friction

`MaterialModel::Rheology::FaultFriction` selects either the existing stateful
rate-and-state law or a stateless rate-dependent weakening law. The
rate-and-state equations, regularization, exact aging update, parameter
averaging, slip-rate bounds, and timestep restriction are unchanged.

The rate-dependent law is

\[
\mu(V)=\mu_d+\frac{\mu_s-\mu_d}{1+V/V_c},
\qquad
\frac{\partial\mu}{\partial V}
=-\frac{(\mu_s-\mu_d)V_c}{(V_c+V)^2}.
\]

Before evaluating this nonlinear law, the surface material fractions
arithmetic-average each parameter independently:

\[
\bar\mu_s=\sum_m f_{\Gamma,m}\mu_{s,m},\qquad
\bar\mu_d=\sum_m f_{\Gamma,m}\mu_{d,m},\qquad
\bar V_c=\sum_m f_{\Gamma,m}V_{c,m}.
\]

The existing `Reference friction coefficients` parameter supplies \(\mu_0\)
for rate-and-state friction and \(\mu_s\) for rate-dependent friction. The new
`Dynamic friction coefficients` and `Characteristic weakening slip rates`
parameters supply \(\mu_d\) and \(V_c\), with defaults `0.4` and `1e-6` m/s.
Rate-dependent inputs require \(V_c>0\) and
\(0\leq\mu_d\leq\mu_s\) componentwise.

Stateful friction is evaluated through overloads that include `theta`; the
stateless overloads omit it. Calling an overload that does not match the
selected law is an API error. `has_state_variable()` is the solver-facing
dispatch operation. Both laws require the physical/numerical lower bound
(V\geq V_{\min}>0). There is no constitutive upper slip-rate clamp: nonlinear
trial values above the lower bound are evaluated as supplied. The stateless
law evolves no constitutive state and therefore imposes no fault-friction
timestep restriction: `compute_time_step()` returns the largest finite
`double`.

Stage E remains constitutive-only. It does not register `Theta`, evaluate a
surface residual, couple slip rate to Stokes, or add commit/rollback hooks.

## 23. Coupled-solver architecture and Stage-F surface system

The coupled implementation separates material mechanics from finite-element
assembly. `MaterialModel::PhaseFieldFault` directly exposes the narrow input,
response, and three semantic operations needed to evaluate one particle-point
surface response without committing history. There is no abstract
reconstructed-fault constitutive base class. A solver-local simulator helper,
`ReconstructedFaultSurfaceSystem`, checks that the selected material model is
`PhaseFieldFault` once when the helper is constructed and retains that concrete
reference for its lifetime. The simulator owns one canonical surface-system
instance and one canonical reconstructed-fault Stokes-coupling instance when
this coupling is available. Solver helpers retain references to these objects;
they do not construct duplicate stateful instances. The surface helper owns
particle/Q1 surface assembly, MPI
reduction, and the surface factorization. It is not an ordinary ASPECT
assembler because this workflow does not use cell-local Scratch/CopyData.
The material-model interface does not expose `apply_B()` or `apply_G()`
operations. The condensed operator remains solver-owned.

At an associated particle \(p\), with the surface Q1 slip rate \(V_p\), the
non-committing constitutive response is

\[
F_p=t_p-T^{\rm coh}_p-\mu_p\sigma_{n,p}-\eta^d_pV_p,
\qquad
t_p=\boldsymbol\tau_p:\boldsymbol S_p.
\]

The response uses the committed particle Maxwell stress, committed fault
history, current transient \(I_h\), and fixed rate-and-state `Theta`. None of
these values is changed by residual or Jacobian evaluation. For timestep zero,
the configured positive `Initial time step` supplies the Maxwell interval;
later timesteps use the current simulator timestep. Surface evaluation is
admissible only after current \(I_h\), cohesive history, slip rate, and, for
rate-and-state friction, a positive committed `Theta` have been initialized.
Stage F diagnoses missing state but does not invent a `Theta` initialization or
commit rule; that lifecycle belongs to Stage I.

The parameter `Use adiabatic pressure in fault friction` belongs to
`PhaseFieldFault` and defaults to false. Its two meanings are:

\[
\begin{aligned}
\text{false:}\quad &\sigma_n=p-\boldsymbol\tau:\boldsymbol N,\\
\text{true:}\quad  &\sigma_n=p_{\rm ad}(\boldsymbol x).
\end{aligned}
\]

The second mode uses the adiabatic pressure as the complete normal pressure in
the friction term; it does not add dynamic pressure or deviatoric normal
traction.

Only locally owned associated particles contribute to the projection-
consistent weak residual and Jacobian,

\[
(R_\Gamma)_i=\sum_{p,q} w_{pq}N_i(\xi_{pq})F_{pq},
\]

\[
(K_V)_{ij}
=\sum_{p,q} w_{pq}N_i(\xi_{pq})N_j(\xi_{pq})
\left[
2\kappa_p\chi_p\boldsymbol S_p:\boldsymbol S_p
+\frac{\kappa_p}{I_{h,p}}
+\sigma_{n,p}\left.\frac{\partial\mu}{\partial V}\right|_{\Theta}
+\eta^d_p
\right],
\qquad K_V=-\frac{\partial R_\Gamma}{\partial V},
\]

where all bracketed coefficients use the same (p,q) evaluation as the residual
and \(\chi=h/I_h\). The rate-and-state derivative holds `Theta` fixed;
the rate-dependent law uses its exact signed derivative. The replicated
ordered Q1 segments and segment-local integration points make each
per-fault block symmetric tridiagonal, but the friction term means it is not
assumed positive definite. Fault-sized vectors and tridiagonal coefficients
are reduced over MPI and replicated. Each fault block has a reusable
indefinite-capable direct factorization. Factorization failure and an
excessive scaled solve backward error are explicit numerical failures.

The qualified K5 surface inverse prefers LAPACK GTTRF/GTTRS
adjacent-pivoting tridiagonal LU. Active rows split each fault into contiguous
principal free blocks; each block is factored once and reused for all right-hand
sides. Pivoting supports nonsingular indefinite matrices, including zero initial
diagonal entries. UMFPACK remains the reference. The semantic surface solve,
exact zero active increments, generation validity, and numerical failure checks
are unchanged; no SPD-only path or new condition estimator is introduced.

The next approved K5 prototype may assemble the identical B/G actions as
sparse rectangular matrices for one coupled linearization, retaining the
quadrature actions as references. A solver-side few-mode correction may
precondition the nonsymmetric condensed operator, never replace it or its
fresh residual tests. The bounded ownership, constraint, pressure and mode
rules are recorded in `bp3/stage_K5_coupling_addendum.md`.

`R_\Gamma` is a nested fault-major vector with one entry per reconstructed-
fault vertex. It is accompanied by volume-weighted per-fault and global RMS
diagnostics. `linearize_surface_system()` invalidates the previous
linearization, builds and factors a complete candidate, and publishes it only
after every fault block succeeds. The semantic surface `solve()` operation applies the
stored replicated \(K_V^{-1}\). The residual evaluator is non-committing and
accepts explicit bulk and slip-rate trial states, so line searches can
evaluate trial \(V\) without changing manager-owned current or committed slip
rate. The helper is a simulator-owned, non-checkpointed computational
component. Its surface linearization is valid only as part of one coupled
linearization and introduces no constitutive state or additional timestep
lifecycle.

Stage F ends at this surface system. It does not assemble the slip-dependent
bulk residual, implement \(B\) or \(G\), alter the Stokes operator, run a
coupled nonlinear solve, or commit/rollback constitutive state.

## 24. Reviewed boundaries for Stages G--I

**Approved BP3 stress-change extension (2026-09-12).** Optional fixed Q1
background tractions are stored as a generic two-component manager property
(positive shear, compressive normal) selected by PhaseFieldFault. They are
frozen through mechanics and never enter bulk assembly or Maxwell history.
The surface residual is tau_pre + Delta_tau - C - mu*(sigma_pre+Delta_sigma_n)
- damping*V. K_V uses total normal traction in sigma_n*dmu/dV; B and G keep
their stress-change derivatives. Diagnostics distinguish total normal stress
from incremental p and tau:N. With no property selected the previous equations
are unchanged. The caller initializes/reselects the property; offset data may
not be mutated during a nonlinear solve.

The explicitly selected BP5 initialization normal-feedback control uses the
existing prescribed-pressure branch with zero adiabatic pressure and the same
50 MPa background. Thus friction sees 50 MPa while the bulk still solves its
own pressure and deviatoric stress. The native bulk-work surface path carries
the constitutive pressure-mode selector into G: in this control only its shear
term remains. K uses the prescribed total normal traction. Fixed background,
work weights and all true-normal-stress default equations are unchanged.

For BP3, particle Maxwell history begins at zero stress change, side velocities
retain the official translations, top/bottom have zero stress-change traction,
and pressure is the unshifted incremental pressure. The analytic Airy field is
not an initial bulk history or boundary load. For the approved discrete BP3
initialization, the plugin solves the existing consistent surface mass system
`M tau_bg = weak(C_eval,0 + mu(Vinit,Theta0)*sigma_bg + damping*Vinit)`.
The same admitted domains and nonlinear quadrature as the surface residual
are used. Here `C_eval,0 = kappa_Gamma*Vinit/Ih0 + beta_Gamma*C0`, with the
supplied stored C0 retained at timestep zero. This is an exact weak initial
root **at zero stress perturbation**, not a prescription that an arbitrary
finite-element velocity field has zero constitutive stress. Supplied nodal
Theta0 is not modified. The difference from `tau0_BP3 + projected(C_eval,0)`
arising from the represented surface state/composition is exported explicitly.
The background is initialized once and is not recalibrated at real timesteps.

The BP3 first-event continuation serializes its benchmark accumulated slip,
history-audit state, accepted-step index, event observer and output schedule.
The manager remains owner of frozen background data and physical histories;
the benchmark reselects the background property and reapplies deep prescribed
rows after restart, without recalibration. See
`bp3/stage_K5_first_cycle_preparation.md` for the checkpoint/output contract and
the status of full filesystem restart qualification. Accepted-solver statistics
and completed-checkpoint notifications are observational only.

Stage G introduces `Assemblers::ReconstructedFaultStokes` as a genuine
cell/QP `Assemblers::Interface` implementation. It owns the slip-dependent
bulk Stokes residual and the fault-to-bulk \(B\) action. The particle-based
bulk-to-surface \(G\) action instead extends
`ReconstructedFaultSurfaceSystem`, where it reuses the Stage-F point response,
particle associations, and replicated fault-vector reduction. The solver
orchestrates these independent actions while leaving condensation and
nonlinear lifecycle absent. The known actions are

\[
R_{\rm fault}(\boldsymbol w;V)
=-\int_\Omega 2\kappa(\chi V+\upsilon^{\rm hist})
\boldsymbol S:\dot{\boldsymbol\epsilon}(\boldsymbol w)\,d\Omega,
\]

\[
\delta\boldsymbol\tau=-2\kappa\chi\boldsymbol S\,\delta V_\Gamma
\quad\text{for }B,
\]

so the standalone residual operation overwrites its destination with
\(R_{\rm fault}\), `apply_B()` overwrites its destination with
\(+B\,\delta V\), and the ordinary assembler adds \(-R_{\rm fault}\) to the
cell/global Stokes right-hand side. In particular, the frozen
\(-2\kappa\upsilon^{\rm hist}\boldsymbol S\) term remains in the absolute bulk
residual even though it has no \(B\) derivative.

The ordinary reconstructed-fault assembler also adds the frozen Maxwell
history load \(-\int_\Omega\beta\boldsymbol\tau_{\rm old}:
\dot{\boldsymbol\epsilon}(\boldsymbol w)\,d\Omega\) to the Newton RHS,
exactly once and at every bulk QP, including outside fault support. The
positive counterpart belongs to the absolute bulk residual. The explicitly
mapped particle-stress compositional fields supply the frozen FE history in
the current working bulk vector; they are not taken from an older FE timestep
or updated during Newton. `PhaseFieldFault` evaluates the pointwise frozen
stress with the local bulk composition/temperature and the same Maxwell
interval as the viscosity. The slip-only residual and B interfaces do not
include this V-independent bulk history load. Initial history retention and
terminal publication semantics are unchanged.

Particle-to-FE transfer averages the incident-cell interpolator proposals at
each shared continuous DoF: sum values and counts from locally owned cells
using MPI ADD, then divide at the owning DoF. Each cell contributes once;
unshared/DG DoFs retain their single proposal. This is not an L2 projection
or a change to the configured particle interpolator. The published FE history
and the privately constrained working history remain distinct. The generic
rule also applies to other particle-mapped continuous compositional fields.

Keep the genuinely frozen Maxwell/profile RHS load in a separate accumulator
from the unknown-dependent residual through cell quadrature, homogeneous
constraint distribution and MPI compression. Combine only completed global
vectors. The positive RHS contribution B V stays unknown-dependent, even for
an all-active set whose current direction has dV=0. The same assembly path
serves every Newton base and non-committing trial. This arithmetic separation
prevents loss of the small residual in cancelling stress loads; it changes
neither the equations nor convergence/line-search criteria.

When dynamic pressure is used,

\[
G\delta x=
2\kappa(\boldsymbol S+\mu\boldsymbol N):\delta\dot{\boldsymbol\epsilon}
-\mu\,\delta p.
\]

When adiabatic pressure is used, the prescribed \(p_{\rm ad}\) has no bulk
variation and

\[
G\delta x=2\kappa\boldsymbol S:\delta\dot{\boldsymbol\epsilon};
\]

there is no \(\mu\boldsymbol N\) contribution and no \(-\mu\delta p\) term.
Both pressure modes require finite-difference tests of the non-committing
residual and their corresponding block actions. The Stokes assembler performs
no fault-vector MPI reduction and the surface helper performs no cell-local
bulk weak-form assembly.

Two caches have different invariants. The manager-owned geometry cache maps
each locally owned cell and each point of the exact production Stokes velocity
quadrature, in its original order, to the fault/segment, Q1 coordinate and
weights, signed distance, position, tangent, and normal. It persists while the
mesh, mapping-relevant geometry, fault geometry, projection widths, and
quadrature identity are unchanged. Mesh refinement, restart loading, mesh
deformation, or fault/projection mutation invalidates it. Debug builds verify
the exact reference points, weights, number, and physical QP order at every
consumer boundary. The assembler-owned linearization cache stores
\(2\kappa\chi\boldsymbol S\) at those associated QPs. It is rebuilt once from
an explicit physical bulk linearization state and thereafter `apply_B()` uses
only frozen coefficients and cached geometry; Krylov applications do not
reevaluate the material model or reconstruct fault associations. A geometry
cache change makes that linearization unusable.

ASPECT stores pressures in the full solution, old solution, and nonlinear
linearization vectors in physical units. Its Stokes solver vector instead uses
the scaled unknown

\[
\widehat p=p_{\rm physical}/s_p,
\qquad s_p=\texttt{get\_pressure\_scaling()}.
\]

The Stage-G non-committing residual and `apply_G()` interfaces accept physical
full-system vectors. Stage H must therefore convert a solver direction by
\(\delta p_{\rm physical}=s_p\delta\widehat p\) before applying \(G\), or
equivalently use
\(G_{\rm solver}=G_{\rm physical}\operatorname{diag}(I_u,s_p I_p)\). Thus the
dynamic-pressure contribution for a solver pressure direction is
\(-\mu s_p\delta\widehat p\); adiabatic-pressure mode still has no pressure
contribution. Stage G does not implement this Stage-H solver-vector adapter.

Stage H adds the solver-side exact condensation only. The assembled bulk
operator (A), the frozen (B) coefficients, and the surface (G/K_V)
linearization have the lifetime of one coupled linearization, independently of
the lifetime of the simulator-owned components or the condensed-system helper.
Starting another linearization invalidates every earlier view. With

\[
\begin{bmatrix}A&-B\\G&-K_V\end{bmatrix}
\begin{bmatrix}\delta x\\\delta V\end{bmatrix}
=-
\begin{bmatrix}R_{\rm bulk}\\R_\Gamma\end{bmatrix},
\]

the condensed equation and recovery are

\[
(A-BK_V^{-1}G)\delta x
=-R_{\rm bulk}+BK_V^{-1}R_\Gamma,
\qquad
\delta V=K_V^{-1}(R_\Gamma+G\delta x).
\]

The condensed Krylov operator references the canonical simulator-owned surface
and coupling components and owns no constitutive state. It depends on a
semantic surface solve operation rather than the unrestricted factorization
itself. Stage H supplies the unrestricted implementation; Stage I may replace
it with an active/free-set solve without changing the condensation algorithm.

Bulk Krylov vectors represent homogeneous perturbations. Constrained algebraic
entries are zero for the \(A\) action. Before the physical \(G\) action,
homogeneous hanging-node and periodic constraints are distributed and all
inhomogeneous boundary values remain zero. The \(B\) action is assembled with
the same homogeneous constraint semantics, and condensed results have zero
constrained algebraic entries.

At the start of each coupled mechanical solve, distribute the current physical
inhomogeneous constraints into a private owned copy of the bulk base iterate.
This includes nonzero initial and time-dependent prescribed velocities. Then
assemble Newton residuals and matrices with homogeneous Stokes constraints:
the physical lift is already represented in the iterate and must not be
subtracted again during local-to-global assembly. Preserve auxiliary-field
constraints and restore the caller's constraints on exit. The lift does not
publish bulk state; failure restores the pre-solve solution and committed V.

When fault friction uses prescribed adiabatic pressure, honor ASPECT's existing
`Pressure normalization` on the private physical base and every physical trial
before residual/merit evaluation. `volume`, `surface`, and `no` retain their
ordinary meanings; this is not a change to pressure scaling or homogeneous
Newton-direction constraints. Both published bulk vectors inherit the same
normalized accepted iterate. Keep the normalization adjustment private until
the terminal commit, so rejected trials and failed solves leave its published
bookkeeping unchanged. All normalization allocations and MPI collectives occur
before history/V writes. Do not apply this gauge-only operation to
true-normal-stress friction, whose surface residual depends on bulk pressure;
that pressure-mode formulation is unchanged by this correction.

The assembled iterative path uses FGMRES. This is required because the
condensed operator is generally nonsymmetric: no identity \(B=G^T\) is
assumed. A returned linear direction must pass a freshly computed residual
test, not merely the Arnoldi estimate. Residual replacement/restart shares
the existing total linear iteration budget.

The bounded K5 GMG experiment changes only the velocity-block preconditioner
inside the assembled block-Schur inverse. It reuses ASPECT's velocity GMG
hierarchy while retaining assembled fine A, sparse B/G, the surface inverse,
pressure-block treatment and outer FGMRES. The synchronous post-linear-solve
observer permits tests to compare the same frozen RHS/operator before any
trial/history commit. Reference-action comparisons belong in tests, not in
production vmult loops. See `bp3/stage_K5_gmg_prototype.md`; this is not support
for a full matrix-free reconstructed-fault solver or coupled multigrid levels.

For incompressible prescribed-friction-pressure coupling in a closed/periodic
domain, the solver may work on the algebraic constant-pressure complement.
First verify both null identities for the full homogeneous constrained
condensed operator. The left identity uses the fact that B has no pressure
rows; the right identity includes G and the current surface solve. Do not
apply this to open or absolute-pressure-dependent configurations. Project
operator/preconditioner actions consistently, without changing physical
pressure normalization or homogeneous Newton constraints. Reject any removed
RHS or full-residual null component exceeding either 100 machine epsilons
times the maximum of the initial bulk residual, zero-velocity reference,
and current condensed RHS norm, or the unchanged absolute nonlinear bulk
target. This is an internal backward-error check, not a new parameter
or permission to discard significant incompatibility. Normal convergence output
reports the fresh residual and requested target. Detailed diagnostic mode also
reports the Arnoldi estimate, raw full residual and compatibility components;
silencing diagnostic prose never disables these acceptance checks.

Block-action, condensed-action, right-hand-side,
and recovery signs must be verified against centered finite differences of a
single non-committing coupled residual evaluator. Tests cover velocity-only,
pressure-only, slip-only, and mixed directions; both pressure modes; nonzero
cohesive/profile history; one and two MPI ranks; and step sizes showing the
expected truncation-error regime followed by roundoff saturation. A small
explicit full two-block solve is compared with condensation and recovery, and
a separate fixture covers more than one reconstructed fault.

Stage I alone connects nonlinear trial evaluation and lifecycle. At the lower
bound, it first forms a projected Newton direction: a degree of freedom at
\(V_{\min}\) whose Newton direction points below the bound is held active,
while the remaining free direction is solved consistently. Fraction-to-
boundary limiting is then applied only to the free direction. Thus a bound-
active outward direction does not force a zero step. Trial values are evaluated
through the non-committing residual interface and manager-owned trial slip
rate; only accepted line-search trials replace the current iterate, and only
nonlinear convergence commits timestep state. Rejected trials and failed
timesteps leave committed `Theta`, cohesive history, particle Maxwell stress,
and committed \(V\) unchanged.

Bound-contact candidates retain absolute nodal values: at the limiting local
fraction set V_trial exactly to V_min; non-contact entries keep their ordinary
affine trial. Exact surface endpoint evaluation likewise reads the stored
absolute node directly, rather than reconstructing it by left+(right-left).
The same validated vector is copied into manager trial state,
residual evaluation and accepted state without subtract/add reconstruction.
This fixes cancellation when V_min is tiny relative to the base rate; the
fraction formula, active-set tolerance and acceptance criteria are unchanged.

The bound test is local to each vertex:
\(V_i-V_{\min}\leq 100\epsilon_{\rm mach}\max(V_{\min},|V_i|)\).
Within one Newton iteration the active set starts empty and can only grow; it
is rebuilt from empty at the next Newton iteration. The restricted semantic
surface solve uses the principal free block \(K_{\mathcal F\mathcal F}\),
projects its right-hand side, and returns exact zero on active vertices. The
line search holds this active set fixed, starts from the exact free-set
fraction-to-boundary step, and accepts only when
\(\Phi_{\rm trial}\leq(1-10^{-4}\alpha)\Phi_k\), where
\(\Phi=(r_b^2+r_\Gamma^2)/2\). An exhausted line search is a nonlinear
failure; its last candidate is not accepted.

The bulk and free-surface residuals have separate fixed normalization scales
and must both satisfy the configured nonlinear tolerance. The relative bulk floor
retains the existing initial Stokes residual convention, multiplied by the
larger of the linear Stokes tolerance and \(\sqrt{\epsilon_{\rm mach}}\).
The bulk criterion includes an independent absolute precision scale. With
\(A_0\) the first homogeneous-constraint bulk matrix and \(x_0=(u_0,p_0/s_p)\)
the first physical iterate in solver coordinates, define
\[
 d_i=\sum_{c\in\{u,p\}}\sum_{j\in c}|(A_0)_{ij}|\,
          \|(x_0)_c\|_\infty,\qquad
 \rho_b=\epsilon_{\rm mach}\|d\|_2.
\]
This is the absolute-row-sum bound on A's response to one machine-precision
bulk perturbation at the initial block magnitudes, not a measured stagnation
floor or an estimate of physical discretization error. Only owned rows enter
the MPI sum. The mixed bulk target is
\(\epsilon_{\rm nl}s_b+\rho_b\); equivalently use the fixed scale
\(s_b^{\rm mixed}=s_b+\rho_b/\epsilon_{\rm nl}\) for both convergence and
merit. Neither scale is updated from subsequent residuals or trial states.
The original relative bulk target remains the cap on pressure-compatibility
projection; the absolute allowance does not authorize incompatible pressure
loads or weaken fresh linear checks. Surface convergence is unchanged.
Cellwise constant velocity is removed before evaluating the bulk residual's
FE strain and divergence, reducing cancellation without changing the affine
Maxwell law or its Jacobian. This does not remove rounding in the represented
velocity iterate itself. A residual materially above the mixed target is a
failure even if it stagnates.
The fixed surface scale is the maximum of the initial free-surface residual
norm and the two full-surface reference norms
\(\|R_{\Gamma,0}\|_\Gamma\) and \(\|K_V V_{\rm char}\|_\Gamma\), evaluated
at the first stabilized linearization, with
\(V_{{\rm char},i}=\max(V_{\min},|V_i|)\). This physical traction scale is
**not** multiplied by a roundoff factor. The same fixed surface scale is used
for convergence and the Armijo merit, including when the initial surface
equation is almost balanced but the bulk loading changes. No dimensional
tuning parameter or change to the configured tolerances/search budget is added.
The surface norm is the RMS norm of the consistent-Q1 strong residual: for the
weak nodal vector \(r_{\mathcal F}=P_{\mathcal F}R_\Gamma\) and the particle-
quadrature mass matrix, it is
\([r_{\mathcal F}^T M_{\mathcal F\mathcal F}^{-1}r_{\mathcal F}/
({\boldsymbol 1}_{\mathcal F}^T M_{\mathcal F\mathcal F}
{\boldsymbol 1}_{\mathcal F})]^{1/2}\). Both operands use the stabilized free
set, and the all-active norm is zero.
The simulator keeps a separate accepted bulk iterate, and every trial is
evaluated from that same base state through the non-committing coupled
residual path.

At a fresh timestep zero, mechanical preparation recomputes transient
\(I_h\), initializes missing cohesive history, projects a user-supplied
positive \(\Theta_0\) from exactly one generic particle-advected
compositional field mapped to particle property `phase field fault state`
component zero, and initializes \(V=V_{\min}\). Restart and later-time paths
must already contain complete committed state and never reconstruct it from
initial fields.

## Explicit mature frictional specialization (K5)

`Material model / Phase field fault / Fault constitutive mode` selects
`cohesive` (unchanged default) or `mature frictional`. Maturity is explicit,
not inferred from a frozen phase field. This task introduces no transition
between modes. The mature mode applies to every vertex of the pre-existing
fixed fault, including prescribed-rate vertices, and requires `Evolve phase
field = false`. Phase is prescribed at initialization and thereafter retained;
geometry changes and conversion of cohesive checkpoints are rejected.

The reduced law is
\[
\upsilon=\chi V,\quad\chi=h/I_h,\qquad C=0,\quad
F=\tau_{\rm bg}^{\rm eff}+\Delta\tau:S
 -\mu(V,\widehat\Theta)\bigl(\sigma_{n,\rm bg}+\Delta p-\Delta\tau:N\bigr)
 -\eta^d V.
\]
The bulk Maxwell law and its retained stress are unchanged. The positive
minus-V derivative is `2 kappa_bulk chi S:S + sigma_n mu_V + eta_d`:
only the cohesive `kappa_Gamma/I_h` term is absent. G and B keep their
existing discretizations, signs, pressure convention and true-normal feedback.

There is no recoverable cohesive energy, `W_coh=0`, and no cohesive driving
candidate for H. Initialized H remains inert profile/irreversibility metadata
(`H_k=H_0`, not growing shadow cohesive work); it does not enter mechanics.
The phase/gradient energy is constant because the profile is fixed. Bulk
elastic storage and Maxwell dissipation remain; frictional power is
`mu sigma_n V` and radiation loss is `eta_d V^2`. No positivity is inferred
for frictional work if the signed normal traction is tensile. Fixed prestress
is external/background work, not a recoverable cohesive spring.

For the bounded BP3 comparison initialize the fixed effective background as
`tau_bg_old(s)-C_star(s)`, using the previously captured **evaluated** initial
resistance, not retained nodal C0. A generic optional background correction
is represented by three checkpointed Q1 coefficient fields `a,b,d`, evaluated
as `a(s)+b(s)/d(s)` **after interpolation**. For this fixture they are
`a=beta0 C0`, `b=kappa0 V0_accepted`, `d=I0`. They are prestress data, never
updated with slip, C, temperature or time; do not project the ratio back into
Q1. The background selector is reattached by the owning plugin on restart.
The mature geometry marker and identically zero C distinguish compatible
histories. Theta keeps its original nodal split update and timestep-zero
retention semantics. Restart does not reload the initial snapshot file.

The modified fully frictional research fixture's original restart audit was
blocked by cold-Ih bitwise reproducibility (see the K5 research restart report).
The long-run preparation adds the approved restricted restoration: after
restart, cold-recompute and compare with persisted previous-Ih using the
unchanged quadrature/tail budget; require fixed mature geometry, identical
current/previous frozen phase and composition-independent degradation, then
restore the persisted values exactly before publishing the cache key. This
does not restore search caches or overwrite accepted history. Changed inputs
are rejected rather than silently accepting a new normalization.
Benchmark checkpoint version 4 preserves
the original inert-H, fixed-geometry and completed-Ih audit baseline, in
addition to accumulated slip, preceding Theta, last accepted step and output
state. It verifies the fully-frictional/work-measure selectors before reuse;
it does not convert earlier investigation checkpoints. Runtime work/source
selectors and the fixed background-property indices are reattached before
mechanics, without reinitializing histories or recalibrating background.
Immutable mesh, fault, prestress and paired completion inputs reside in
`benchmarks/reconstructed_fault/bp3/fixtures/modified_bp3/` with SHA-256 provenance.
The seven-step saved-clock regression and explicitly bounded ordinary-adaptive
continuation are separate benchmark execution modes. Neither changes the
split constitutive cycle or global endpoint/mature defaults.

The separately selected modified-BP3 long-run fixture removes junction-only
refinement, not the physical junction law or the ordinary fault band. Its
version-5 checkpoints additionally retain coordinated output clocks/slip
references and native fault time-series metadata. Only strengthening is
spatially refreshed by the opt-in initial-composition property; unrelated
histories are untouched, and old particle-layout checkpoints are not converted.
A benchmark observer chooses heavy output after accepted history publication;
native bulk/particle/fault writers share the decision, and a final observer
advances its per-node slip reference only after all writers return. Rejected
states cannot advance it. Restart creates a new output branch and restores
the selected checkpoint's metadata prefix, preserving prior evidence.
The BP3 cumulative-slip CSV records every accepted vertex state, independently
of visualization throttling, with stored-order arclength and physical down-dip
coordinate. Its restart branch streams only the selected accepted prefix;
the checkpointed slip vector, not the CSV, remains constitutive/output history.
Reconstructed-fault VTU names are output-only aliases with underscore separators;
optional property exclusions do not change registry names, values or checkpoints.
See `bp3/stage_K5_long_run_preparation.md` for the bounded verification and
limitations; this is not first-event or recurrence-cycle qualification.

### Selected BP5 steady-sliding initialization

The benchmark-only `bp5_steady_initialization` plugin selects the approved
alternative initial data, not a new mature constitutive law. Set all initial
particle and nodal state to the configured `Dc/Vinit` (1e8 s for the current
fixture). After the normal surface chemical projection, use the production
surface residual evaluator's native bulk-QP work mass and resistance loads to
solve `M tau_bg = friction_load + damping_load` at this constant state/rate and
50 MPa background compression. Both continued endpoint wedges participate.
Exclude the constitutive probe's crack-induced shear; it is not initial
prestress. The captured shear and its correction are replaced together: the
correction is identically zero, while the background is spatially varying Q1.
Bulk Maxwell stress and pressure remain perturbations initialized to zero.

This explicitly selected variant bypasses both captured-prestress import and
nodal/weak inverse-state initialization. It rejects a nonempty captured
prestress filename. Benchmark checkpoint metadata identifies the initial-data
variant; incompatible checkpoints are rejected in either direction. Restore
the manager's background, state, slip and histories and reattach runtime
selectors, without reinitialization (including a timestep-zero checkpoint).
Subsequent split aging, timestep safeguards and history publication are unchanged.
The original inverse-state plugin and its verification artifacts remain separate.

The benchmark's explicitly selected loading-driven variant sets
`Weakening initial state ratio = 0.8` (default `1` preserves the steady variant).
After material projection, use production material fractions to set
`Theta_i = (Dc/Vinit) R_VW^(1-f_i)`, where the arithmetic two-material friction
mixture gives `f_i = (a_i-a_VW)/(a_VS-a_VW)`, its strengthening fraction.
The native weak prestress initializer then evaluates the actual Q1 state,
not an analytic state evaluated independently at quadrature points. Initial
particle inputs use the corresponding horizontal-depth extension; the final
projected-mixture nodal state is authoritative. Restart identity includes the
profile version and ratio and rejects incompatible initial data; no histories
or prestress are recalibrated on restart. This changes initial data only,
not split aging, the work measure, mechanical equations or solver tolerances.

## 25. Stage-J constitutive history feedback

### Phase nonlinear precision criterion

The phase equations and Newton/line-search algorithm are unchanged. Phase
convergence uses the mixed absolute residual target
`max(relative_tolerance * initial_residual, roundoff_allowance)`, frozen at
entry to each phase solve. The configured relative tolerance retains its
meaning; the internal allowance is not a physical/convergence parameter and
is not inferred from stagnation or an unsuccessful iteration.

For each owned particle let `P = sum_j |w_j phi_j|` and
`D_d = sum_j |grad(w_j)_d phi_j|`. Before cancellation, assemble the positive
nodal scale

\[
 S_i=\sum_p m_p\left\{|w_i|\left(a_p+b_p P_p\right)
       +|F_p|\sum_d|\partial_d w_i|D_{p,d}\right\},
\]

where `F` is the phase gradient coefficient,
`a = sum_m f_m (|H g'_m| + |E_m alpha'|)`,
`b = sum_m f_m (|H g''_m| + |E_m alpha''|)`, and `E_m=Gc_m/(c0 ell)`.
Apply the same nonnegative Q1 hanging-node/periodic constraint weights and MPI
ADD ownership as the residual. The allowance is `8 epsilon_double ||S||_2`.
This is a conservative first-order roundoff estimate including represented
nodal-value sensitivity, with a modest arithmetic safety factor; it is not a
rigorous error bound for arbitrary ill-conditioned problems. It scales with
the physical coefficients and discrete weak-load measure, not a fixed
dimensional constant. Reject nonfinite targets. A tiny initial residual may
skip Newton only if it meets this same residual criterion. Keep iteration
exhaustion/failure handling and line-search logic unchanged.

The K3 phase-entry/exit checks use this production scale while retaining the
1e-8 relative target. Physical benchmark criteria (admissibility, feedback,
support/normalization, lifecycle and homogeneity) are separate and unchanged.
See `benchmarking/stage_K3_phase_precision_audit.md` for the exact failed-state
audit, independent verification and the scope of the precision estimate.

The fixed-fault timestep cycle is indexed as

\[
H_{k-1}\longrightarrow\phi_k\longrightarrow (u_k,p_k,V_k)
\longrightarrow\{\Theta_k,T_k^{\rm coh},\tau_k,H_k,I_{h,k}\}
\longrightarrow\phi_{k+1}.
\]

Thus the phase field and fault geometry are fixed during the complete coupled
mechanical solve. Residual, Jacobian, Krylov, and line-search evaluations do
not modify history. Only a converged mechanical state may enter a terminal
commit, and all failure-capable calculations and MPI validation precede the
first persistent write.

Timestep zero has initialization semantics rather than physical history
evolution. Its mechanical solve commits the initial kinematic solution
$V_0$, but retains the user-supplied $\Theta_0$, initialized irreversible
$H_0$, initialization-specific $T_0^{\rm coh}$, and user-initialized Maxwell
stress. The current $I_{h,0}$ is stored as the previous-normalization snapshot
needed by the first real step. In particular, neither `Theta` nor $H$ is
advanced through the artificial positive `Initial time step`.

The initial mechanical evaluation uses the converged $\phi_0$ as both current
and previous profile, paired with the initialized previous-$I_h$ snapshot.
The zero `old_solution` phase block is an initialization placeholder, not a
physical earlier profile. Apply this rule in the shared pointwise localization
evaluation used by surface and bulk coupling; do not overwrite the old FE
vector or advance any retained history. For $k>0$, use the actual previous FE
phase field as before, including after restart.

For $k>0$, bulk Maxwell coefficients use particle-local bulk composition and
temperature. Surface cohesive coefficients use the projected Q1 surface
composition and a fault-surface temperature obtained by sampling the frozen
FE temperature at fault vertices and interpolating it in the fault Q1 space.
Consequently all particles with the same fault coordinate share surface
coefficients even if their transverse bulk temperatures differ. Surface
coefficients are used consistently in cohesive mechanics and history, while
bulk Maxwell, $B$, and $G$ retain bulk coefficients.

The accepted particle cohesive-traction samples are consistently projected to
the replicated fault Q1 space. This projected traction, interpolated back to a
particle's cached fault coordinate, enters the exact finite-step driving-force
candidate. For $g_k<1$, evaluate it deliberately as

\[
a=\frac{T_k^{\rm coh}}{g_k},\qquad
b=\frac{\beta_{\Gamma,k}h_{k-1}T_{k-1}^{\rm coh}}{1-g_k},\qquad
\mathcal H_k=\frac{\Delta t_k}{2\kappa_{\Gamma,k}}(a-b)(a+b),
\]

and commit $H_k=\max(H_{k-1},\mathcal H_k)$. A negative candidate is not
independently clipped to zero. At an exactly intact current point with
$g_k=1$ and $h_k=h_{k-1}=0$, use the removable limit

\[
\mathcal H_k=\frac{\Delta t_k}{2\kappa_{\Gamma,k}}
(T_k^{\rm coh})^2.
\]

The case $g_k=1$ with $h_{k-1}>0$ is inadmissible healing and is diagnosed.
Rate-and-state `Theta` is updated at Q1 vertices from accepted $V_k$ through
the existing exact aging law and the same surface-state path used by friction.
The current law has one global $D_c$, so no composition-dependent $D_c$ or
duplicate parameter is introduced. Rate-dependent friction has no state
update.

The existing law-specific RSF timestep restriction is exposed as an ordinary
ASPECT time-stepping plugin named `reconstructed fault time step`. It uses
timestep-committed Q1 $V$ and the Q1 surface mixture, ASPECT's global CFL
number, and operator-splitting semantics. It is opt-in: the time-stepping
manager does not add it implicitly and preserves standard explicit model-list
selection. Stage J adds no post-solve cutback or repeat operation.
