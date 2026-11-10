# Geometry and physical history

This page describes the retained **BE default**. The selectable midpoint stage,
exact aging, endpoint publication and work identity are in
[time integration](time_integration.md); do not apply BE loss formulas to that mode.

[Guide](../TIP_TREATMENT.md)

The accepted bulk owner stores true τ and physical cohesive tensor A at the
fixed native mechanical quadrature points, plus signed/absolute fault strain and
last rate. The authoritative phase field is DGQ2. Full native strain samples are
used for increments; no uncut DGQ2 projection of nonlinear stress or cohesive
history is performed. This preserves the fixed-quadrature native information,
but does not define an arbitrary-point stress evaluator or a transport/AMR rule.
The old bulk branch retains its independent trial tensor and native remainder.

Each full x-Gauss section owns its age, signed slip, last rate and resistance.
The x coordinate and ds are fixed metadata; JxW/ds supplies its normal measure.
The supported adapter admits aligned Cartesian cells and fixed direction. Its layout
must be identical along each normal section; adaptive and moving meshes are
unsupported. MPI owns cell histories uniquely and collectively reduces section
integrals. Sections are small replicated state, not reconstructed surface nodes.

The ordinary simulator solver scheme owns the coupled iteration. Every evaluation reads
the same accepted parent. Newton iterates, local τ/A responses and phase
candidates remain scratch. On convergence and energy checks, `PhaseFieldFault::prepare_bulk_tip_candidate`
constructs one prepared owner
containing all new physical histories, section ages/slip and phase. The existing
pending-publication callback installs it once on acceptance; rejection leaves
the accepted owner untouched and the ordinary solver transaction restores fields.

The physical checkpoint law is `bulk tip physical tau A DGQ2 sections BE v1`.
It stores accepted time/publication, full cell τ/A histories, phase obstacle,
section metadata/age/slip/rate/resistance, quadrature and material/RSF identity.
It checks physical geometry and field values when restoring. There is no old
interface, mature or legacy-bulk conversion. Changing quadrature changes the
history representation and cannot silently resume this state.

Reconstruction is disabled in the supported tip control. A phase contour can be sampled
at any number of points without modifying energy, age or mechanics: it is an
observer only. Intact quadrature points receive Y before any curve exists.
A contour moving is not by itself a fully developed fault advance. New damage
outside the initial support, threshold motion, φ≈1 advance, and slip on old
damage must be reported separately.

The numerical phase basis is guarded by outer `phase-space.txt` metadata before
checkpoint deserialization. Native Bernstein obstacles contain coefficients,
not nodal samples; [basis and import rules](../NATIVE_DG_BERNSTEIN.md) distinguish
the bounded benchmark conversion from ordinary production restart.


## Initialization and transaction order

At timestep zero, `initialize_fe_shared_phase` takes the semantic composition's
interpolated phase and applies the existing polynomial bound limiter.
`initialize_bulk_cohesive_history` creates zero tau/A, fixed section metadata and
uniform configured age. When enabled, `BulkTipCoupling::equilibrate_seed` solves
the fracture-only coefficient obstacle above the supplied seed, verifies the
unmodified fracture KKT and makes that result the initial accepted phase. It
can add damage above the supplied coefficient obstacle; it does
not relax an already accepted loaded state or advance age. The simulator then
synchronizes its current/old/old-old phase copies at the same zero clock.

During a physical attempt the accepted `BulkCohesiveState` remains immutable.
`Evaluation::points` holds fresh tau/A responses; `Evaluation::laws` holds fresh
section roots. `prepare_bulk_tip_candidate` copies the parent, updates every
point tensor and signed/absolute fault strain, copies the accepted phase
coefficients, and updates section slip and age using the integrated signed rate.
The root-resolved rate and integrated rate remain distinct diagnostics because
summation cancellation can make them differ numerically. No rate clipping or
band selection is applied during publication.

The solver's `pending_dg_endpoint_publication` verifies parent identity and
installs the candidate only when called with acceptance. A false callback leaves
the owner unchanged. The ordinary simulator transaction restores trial fields on
rejection. Caches and search ordering are attempt-local, not checkpoint history.
Native one-to-two-rank restart plus forced reject/retry is qualified in the
[native-basis report](../NATIVE_DG_BERNSTEIN_REPORT.md).

The mature serializer's generalized-anchor correction belongs to mature schema9,
not to this tip law's point-history schema. Both routes share the outer basis
metadata guard but keep different histories and acceptance transactions. The
explicit same-mesh parent import in the benchmark checker converts all phase
copies for an old/new comparison; it is not an ordinary restart conversion.
