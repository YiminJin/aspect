# Discrete equations and owners

[Guide](../TIP_TREATMENT.md)

For fixed accepted parent, let β=(1+ΔtG/η)⁻¹, e⁻=2A⁻:S, and
P=β(τ⁻+2GΔε). The point elimination at current φ and section resistance f is
\[
c=m\phi(1+\phi),\quad d=(1-\phi)^2,\quad H=c/(c+d),
\quad z=P:S+G\beta e^- -f,
\]
\[
e^+=Hz/G,\quad q^+=dz/(c+d),\quad
\tau^+=P-2G(e^+-\beta e^-)S,\quad
w^+=(e^+/\beta-e^-)/\Delta t.
\]
The stable phase driving is
\[
Y=\frac{m(1+3\phi)(1-\phi)}{(c+d)^2}\frac{z^2}{2G}.
\]
These expressions include φ=0 and φ=1. The tangent differentiates z, H and Y
with accepted history fixed; it includes the current section resistance.
Neither viscosity nor friction loss is differentiated as stored energy.

The existing bulk coupled solver assembles Q2 mechanics/Q1 pressure and the
native authoritative DGQ2 phase field. It adds one section equation to the
mixed solve. Its extra unknown is section resistance divided by G. Local point
elimination makes the integrated rate affine: V=B−C f, where B is the free rate
and C=Σdn H/(GβΔt). The section material evaluator solves V+C R(V)=B in a scaled
asinh coordinate and returns f*=R(V). The mixed equation is (f−f*)/G=0.
Its consistent derivative is
\[
\delta f^*=\frac{R'}{1+C R'}(\delta B-f^*\delta C).
\]
This is algebraically the same closure as f=R(Σdn w). It avoids evaluating a
very steep RSF function on a cancellation-sized rate during Newton. The first
control requires the proven monotonicity bound
η_rad ≥ p_N Δt/Dc (max(0,√b−√a))²; unsupported multiple-root cases reject rather
than silently select a root. Steady velocity weakening b>a remains present.

The resolved scalar rate, measured bulk integral, their difference and RSF
re-evaluated on that integral are exported separately. Below arithmetic
resolution the latter can differ by pascals even though the absolute slip and
power defects are tiny. Ages and cumulative slip use the actual integrated rate.
Physical friction/radiation diagnostics use the resolved section solution;
its difference from bulk fw work is reported explicitly as section residual work.
The mechanical and phase tolerances are unchanged. Signed local work is not
required nonnegative for this nonlocal law; physical section losses are.

The phase owner supplies the AT1/SIPG fracture functional, its raw gradient and
sparse Hessian, using the retained Gauss5 cell/face rules. Mechanical energy,
Y, work and section integrals all use the configured native Gauss rule (Gauss6
in the fixture). These rules are not reduced in this task. Gauss5 exactly
integrates the constant-coefficient Q2 polynomial fracture forms on the affine
fixture; nonlinear cohesive rational factors are not generally integrated
exactly by Gauss6. Quadrature convergence remains a separate qualification.

Phase unknowns are cell Bernstein coordinates B, with FE values φ=EB. The
entire residual and Jacobian, including section coupling, use EᵀR and EᵀJE.
The obstacle is accepted B, and both lower and upper bounds are included in
projected Newton/KKT. The tip option also resolves the linearized Newton system
when its predicted phase value crosses a bound, supplying the coupled force
correction for that bound displacement. Projecting phase alone after the solve
can destroy descent; no tolerance or physical obstacle is changed. The full
nonlinear KKT conditions are rechecked after every accepted iteration. This is a
sufficient polynomial bound/irreversibility
cone, narrower than all pointwise-admissible Q2 polynomials. There is no growth
post-limiter retaining an invalid old KKT certificate. The initial field follows
the existing native initialization limiter and is then frozen as the obstacle.

| Equation/responsibility | Source owner/entry |
|---|---|
| Endpoint τ,A,Q,Y and derivatives | `material_model/phase_field_fault/bulk_tip.cc`: `Tip::update`, `derivative` |
| Signed RSF and BE age, existing constants | `rheology/fault_friction.cc`: `signed_endpoint` |
| Physical quadrature histories and section measures | `bulk_cohesive_history.cc`: `initialize_bulk_cohesive_history` |
| Mixed mechanics, phase and section residual | `simulator/solver/bulk_cohesive_phase.cc`: `solve_bulk_cohesive_phase` |
| Fracture functional and SIPG faces | `phase_field/fracture.cc`, `phase_field/dg_sipg.cc` |
| Bernstein primal/dual transformation | `reconstructed_fault/endpoint_phase_constraints.h` |
| Publish/rollback | Ordinary simulator pending endpoint publication callback |

The bounded solver retains distributed integration, replicated sparse mixed
algebra, and a root UMFPACK solve with broadcast. It is not a scalable production
solver. No ordinary scalar-advection face assembler was changed. No new timestep
driver or alternate simulator pipeline was introduced. The retained old bulk
potential's stability certificate does not include the new section dissipation
primitive and is not applied to this route; KKT/residual convergence
alone does not establish stability or uniqueness.
