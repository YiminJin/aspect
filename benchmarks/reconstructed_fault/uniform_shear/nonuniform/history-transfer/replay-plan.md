# Approved bounded representation and timeline replay

DGQ1 only for the three Maxwell fields was tried first. The debug run aborts
in PhaseFieldHandler::make_sparsity_pattern: a full component index is used
as a vertex-local DoF index, invalid with preceding DG fields. Independently,
interpolate_particle_properties requires all batched particle fields to share
one base element; theta_initial must remain continuous Q2. Do not repair these
broader DG compatibility issues here, and do not run DGQ2.

Select continuous Q2 with the approved generic incident-cell ADD/count transfer.
No averaging patch existed before this task. The patch also removes competing
writes from the particle-mapped continuous theta_initial field. Its realized
initial projection may change; record that, never reset it from old outputs.
The old INSERT audit/source is retained in insert-baseline.tar.gz.

Replay only the existing 32x128 K2 case to t=1 s (.5-s real steps), with the
same physical initial data, phase profile, surface rule, support, full I_h and
solver settings. Existing old-rule results through 1 s are the comparison;
no new baseline replay is needed. Add diagnostic output and particle VTU only.
Expected cost 30--60 s and less than 1 GiB, hard cap 120 s. The new one-/two-
rank small audit and this replay must stay below 600 s aggregate execution.
No automatic timeout retry, further convergence run or K2.3 pilot is permitted.

An existing set_assemblers signal adds a read-only zero-contribution observer.
For every cell it records the actual first assembly's FE history and current
particle history; subsequent assembly visits verify unchanged particle stress.
The generic pre_assemble_stokes_system signal is bypassed by the coupled path,
so it is deliberately not used. The post_restore_particles signal records
history before advection; the observer records it again after advection and
transfer, at the positions used by mechanics. IDs join these to the preceding
terminal commit. Surface_system.cc reads these same immutable parent particle
properties directly; no particle-stress update exists between the observed
bulk assembly and surface evaluations. Cached association geometry is recorded
separately from the published FE field.

The postprocessor records actual terminal properties, without applying any
Maxwell update. Standard bulk composition VTU arrays and particle property VTU
arrays are inspected separately. No live history is refreshed for output.
The independent timeline check reconstructs nodal means from the actual
post-advection particle data and applies the fixture's existing periodic
history constraints, then compares with first-assembly values. This tests
consumption, not an inference from post-solve output. Timestep zero must retain
the supplied stress; steps 1/2 must consume the previous committed history.
