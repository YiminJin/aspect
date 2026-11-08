# Physical model

[Guide](../TIP_TREATMENT.md)

All energies are per unit out-of-plane thickness. The tensors use the full
three-dimensional plane-strain metric, including the zz deviatoric component.
The first control fixes (S=(e_x\otimes e_y+e_y\otimes e_x)/2), (S:S=1/2).
Write (D_f=wS), (A=eS); (w) is a fault-deformation rate in s⁻¹, not total
strain rate. Physical stress is \(\tau\), cohesive conjugate stress is
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
same functional minus Y, constrained by accepted damage ≤ new damage ≤ 1.
There is no surface damage variable or surface fracture charge. The configured
`Cohesions` value calibrates (m=K/(c^2/(2G))); it is not an added yield resistance.

At φ=0, A=0 is a constraint. The section closure selects its reaction
(q=\tau:S-f_s). For an entirely intact section V=0, smooth regularized RSF gives
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
