# Bulk energetic tip treatment

Start with the [active implementation reading path](ACTIVE_RESEARCH_MODEL.md);
[consolidation evidence](TIP_CONSOLIDATION_REPORT.md) closes the midpoint repair
attempt and compact-loading experiment. BE remains the reference.

Current implementation: `e8b6dd091` plus midpoint and active-path consolidation, 9 November 2026. This guide
describes `Bulk tip solver = ASPECT` with the active native Bernstein DGQ2 phase
fixtures. DG Bernstein is the supported phase basis. Numerical reports
identify their tested executables separately from this source revision.

The selected propagation control solves damage in the **bulk**, including intact
material ahead of a finite damaged patch. The diffuse fracture force competes
with the cohesive elastic driving force; its constrained equilibrium determines
where damage grows. A threshold drawn through that field is an observation, not
an admission criterion or an extra fracture inventory.

The `bulk tip sections` option uses direct physical deviatoric stress and a
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
- [Default BE step, equations and executable operations](tip_treatment/discretization.md)
- [Midpoint candidate: stages, endpoints and physical work](tip_treatment/time_integration.md)
- [State dictionary and worked point trace](tip_treatment/state_and_trace.md)
- [History, geometry and lifecycle](tip_treatment/geometry_and_history.md)
- [Energy accounting and validation](tip_treatment/energy_and_validation.md)
- [Configuration](tip_treatment/configuration.md)
- [Limitations and future material motion](tip_treatment/limitations.md)

The implementation has three distinct layers: the material point returns physical
tau/A and derivatives; `BulkTipCoupling` assembles and condenses section feedback;
the ordinary simulator owns nonlinear iteration, acceptance and publication.
[Native coefficient meaning and transfer](NATIVE_DG_BERNSTEIN.md) are shared with
the phase infrastructure. The tip calculation itself has no phase advection or
SSPRK3 stage; those belong to the separately qualified moving mature route.

The selectable time-update decision is in the
[time-update and localized-growth report](TIP_TIME_UPDATE_AND_GROWTH_REPORT.md).
The other step/trace pages retain the BE default; midpoint differences are
authoritatively derived on the time-integration page.

Prior representation qualification is in the
[native Bernstein report](NATIVE_DG_BERNSTEIN_REPORT.md): element/operator checks,
same-parent difficult step, localized growth, and MPI restart/rejection pass.
The difficult-step phase/stress differences from the retained nodal control are
4.68e-16 / 1.59e-12 Pa. Performance is mixed: 109.94 versus 101.36 s for that
step, but 10.61 versus 12.49 s for the localized control. A new representation
is not a new time-accuracy or propagation result.

The [efficiency/growth report](TIP_EFFICIENCY_AND_GROWTH_REPORT.md) retains the
preceding search-order improvement (200.51→101.36 s), small-rate resistance
sensitivity, common-parent timestep comparison and .09 localized diagnostic
stop. BE work error remains large relative to added fracture; the localized
probe adds damage without resolved front-contour advance. The
[solver-integration report](TIP_SOLVER_INTEGRATION_REPORT.md) and
[original implementation report](TIP_TREATMENT_IMPLEMENTATION_REPORT.md) retain
earlier solver generations, including the original refined failure. They are
evidence, not competing descriptions of the current algorithm.

The supplied [continuum draft](fault_tip_treatment.tex) supplies the energetic
core; the section closure is an additional assumption, not its consequence.
