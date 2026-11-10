# Physical model

[Guide](../TIP_TREATMENT.md)

All energies are per unit out-of-plane thickness. The tensors use the full
three-dimensional plane-strain metric, including the zz deviatoric component.
The supported control fixes \(S=(e_x\otimes e_y+e_y\otimes e_x)/2\),
\(S:S=1/2\). Write \(D_f=wS\), \(A=eS\); w is the fault-deformation
rate in s⁻¹, distinct from the total strain rate. Physical stress is \(\tau\);
cohesive conjugate stress is
\(Q=2GA/h\), and \(q=Q:S=Ge/h\).

\[
\psi=\frac{\tau:\tau}{4G}+\frac{G A:A}{h(\phi)},\qquad
h=m\frac{\phi(1+\phi)}{(1-\phi)^2},\qquad
Y=\frac{G h' A:A}{h^2}.
\]

\[
\frac{\dot\tau}{2G}+\frac{\tau}{2\eta}=D_3-wS,
\qquad \dot A+\frac G\eta A=wS.
\]

These are fixed-reference small-strain equations. Zero spin is a restriction of
this qualification, not a selection of a general objective rate. The tensor A
is kept even though the control restricts it to span(S); later changing a scalar's
direction would not be a valid tensor-memory update.

The single fracture functional is
\[
E_f=\int_\Omega K[\phi+\ell^2|\nabla\phi|^2]dx,
\qquad K=G_c/[(8/3)\ell].
\]
Its DG discretization includes SIPG faces. The force is the derivative of this
same functional minus Y, constrained in the implemented FE solve by the accepted
Bernstein coefficients
\(B^-\le B\le1\). This sufficient whole-cell obstacle is more restrictive
than all pointwise-admissible Q2 polynomials; see [discretization](discretization.md).
There is no surface damage variable or surface fracture charge. The configured
`Cohesions` value calibrates \(m=K/(c_{cal}^2/(2G))\); it is not an added
yield resistance.

At φ=0, A=0 is a constraint. The section closure selects its reaction
\(q=\tau:S-f_s\). For an entirely intact section V=0, smooth regularized RSF gives
f=0, so Y(0)=m(τ:S)²/(2G). There is no imported Coulomb sticking interval. If other
parts of the section deform, their common f also enters the intact reaction.
This nonlocal influence is an explicit consequence of the selected closure.

At φ=1 and finite A, inverse compliance, Q, cohesive energy and Y all vanish.
The implementation uses analytic endpoint elimination, not division by zero or
a phase cap. Cohesive memory can remain finite at φ=1 and continues to relax.
A mature growth-disabled profile with φ<1 and manually absent cohesion is not
this limit. Only a fully developed region with Q=0, a compatible fault-rate
profile, identical friction/age conventions, and separately matched mechanical
time integration could reproduce a mature reduction.

The implemented material operations are
[`Tip::update`, `Tip::derivative`, `Tip::account`](../../../../source/material_model/phase_field_fault/bulk_tip.cc).
Their input strain increment is the full plane-strain deviator of the native
velocity/displacement FE gradient. `update` receives an explicit accepted point,
trial phase and section force; it neither solves a section nor publishes history.
The [discrete update](discretization.md#point) gives the finite BE formulas,
including their intact and fully damaged limits.
