# Bulk energetic tip treatment

The selected propagation control solves damage in the **bulk**, including intact
material ahead of a finite damaged patch. The diffuse fracture force competes
with the cohesive elastic driving force; its constrained equilibrium determines
where damage grows. A threshold drawn through that field is an observation, not
an admission criterion or an extra fracture inventory.

The new `bulk tip sections` option uses direct physical deviatoric stress and a
physical cohesive strain tensor. Storing stress directly does not remove the
elastic energy stored in that cohesive strain. The historical interface pilot
instead evolves surface damage and charges a surface fracture integral; its
checkpoints and qualification do not transfer to this model.

The user selected an additional nonlocal constitutive closure for this bounded
control: one signed fault-deformation integral and one RSF age per complete
straight cross-section. Nothing is solved on a reconstructed curve. This is a
fixed-reference, closed Mode-II specimen, with prescribed positive compression,
matched constant coefficients, zero spin and no transport. It does not implement
general turning, branching, material motion, current-pressure friction or the
mature production law.

- [Physical model, intact reaction and mature limits](tip_treatment/model.md)
- [Section friction, direction and power](tip_treatment/friction_and_direction.md)
- [Discretization and implementation owners](tip_treatment/discretization.md)
- [History, geometry and lifecycle](tip_treatment/geometry_and_history.md)
- [Energy accounting and validation](tip_treatment/energy_and_validation.md)
- [Configuration](tip_treatment/configuration.md)
- [Limitations and future material motion](tip_treatment/limitations.md)

Implementation and measured qualification are distinguished in the
[TIP_TREATMENT_IMPLEMENTATION_REPORT.md](TIP_TREATMENT_IMPLEMENTATION_REPORT.md).
The supplied [continuum draft](fault_tip_treatment.tex) supplies the energetic
core; the section closure is an additional assumption, not its consequence.
