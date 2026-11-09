# Geometry and physical history

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
The first adapter admits aligned Cartesian cells and fixed direction. Its layout
must be identical along each normal section; adaptive and moving meshes are
unsupported. MPI owns cell histories uniquely and collectively reduces section
integrals. Sections are small replicated state, not reconstructed surface nodes.

The ordinary simulator solver scheme owns the coupled iteration. Every evaluation reads
the same accepted parent. Newton iterates, local τ/A responses and phase
candidates remain scratch. On convergence and energy checks, `PhaseFieldFault::prepare_bulk_tip_candidate`
constructs one prepared owner
contains all new physical histories, section ages/slip and phase. The existing
pending-publication callback installs it once on acceptance; rejection leaves
the accepted owner untouched and the ordinary solver transaction restores fields.

The new checkpoint law is `bulk tip physical tau A DGQ2 sections BE v1`.
It stores accepted time/publication, full cell τ/A histories, phase obstacle,
section metadata/age/slip/rate/resistance, quadrature and material/RSF identity.
It checks physical geometry and field values when restoring. There is no old
interface, mature or legacy-bulk conversion. Changing quadrature changes the
history representation and cannot silently resume this state.

Reconstruction is disabled in the first control. A phase contour can be sampled
at any number of points without modifying energy, age or mechanics: it is an
observer only. Intact quadrature points receive Y before any curve exists.
A contour moving is not by itself a fully developed fault advance. New damage
outside the initial support, threshold motion, φ≈1 advance, and slip on old
damage must be reported separately.

The numerical phase basis is guarded by outer `phase-space.txt` metadata before
checkpoint deserialization. Native Bernstein obstacles contain coefficients,
not nodal samples; [basis and import rules](../NATIVE_DG_BERNSTEIN.md) distinguish
the bounded benchmark conversion from ordinary production restart.
