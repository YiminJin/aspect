# Fixed material sections for one fault

`Bulk tip reference angle` declares the in-plane **initial or incoming** tangent in degrees counterclockwise from positive global x. The material model reads it once. The default is zero. Set the seed, boundary velocities and tensor initial data in the same physical frame in the input. The model has no need to infer a tangent from small phase values ahead of the seed.

Let the input Cartesian coordinate origin be \(X_0=(0,0)\). For declared angle \(\alpha\), set \(t_0=(\cos\alpha,\sin\alpha)\), \(n_0=(-\sin\alpha,\cos\alpha)\), \(\xi=(X-X_0)\cdot t_0\), and \(\zeta=(X-X_0)\cdot n_0\). The continuum material section spans the specimen at fixed \(\xi\). Its coordinate origin only labels stations. A vertical seed gives horizontal sections. The physical frame of its structural plane is

\[
t=\cos\beta\,t_0+\sin\beta\,n_0,\quad n=-\sin\beta\,t_0+\cos\beta\,n_0,\quad
S=(t\otimes n+n\otimes t)/2 .
\]

The seed has \(\beta=0\). Fixed direction retains that value. Both selectable virgin-plane laws use the existing sector \(|\beta|<\pi/4\). In the reference basis define \(u=(\tau_{n_0 n_0}-\tau_{t_0 t_0})/2\), \(v=\tau_{t_0 n_0}\). Then \(\tau:S=u\sin(2\beta)+v\cos(2\beta)\). This gives the intact-driving and accepted-stress Coulomb selectors their original signed stress moments in the correct basis. Stored physical stress and recoverable strain remain full plane-strain tensors in global coordinates.

## Integration and return of resistance

The aligned path retains Gauss stations on complete lines, including the historical tensor-product ordering on horizontal Cartesian cells. **Bulk tip oblique integration = local strips** uses a piecewise constant section response on a generic oblique mesh. For span L and minimum affine cell altitude h, the base count is ceil(L/h). The positive integer `Bulk tip strip subdivisions` multiplies this count (default 1). A factor of two bisects each original strip on the same mesh. The line integration paths are independent of this setting. Equal projected widths define one age, signed slip, resistance and plane per strip. This is a complete transverse strip average of the continuum line law.

Inside each intersected cell, split a strip only at that cell’s projected vertices. Integrate each resulting interval with the configured longitudinal Gauss rule and each cell chord with the same order of transverse Gauss integration. Their physical product is material area Jq. FEValues receives that area divided by the affine determinant. Initialization checks the physical area and strip membership. Every point keeps that definition through stress evolution and restart. **Bulk tip oblique integration = exact lines** retains the earlier global projected-vertex construction and its archive format for comparisons. The two discretizations have different station and native history identities.

A selected plane gives \(d s_k=a_k/\cos\beta_k\). This is the **reference arclength quadrature weight** along the structural plane. The transverse chord length, or its strip average, is \(\sum_{q\in k}b_q\). The signed material fault-deformation rate \(w_q\) alone enters the section law. With membership matrix \(P_{qk}=1\) at a point's station and zero elsewhere,

\[
V=D_s^{-1}P^T Jw,\qquad f_q=(P f)_q,\qquad
V^TD_s f=w^T J P f .
\]

Here \(J\) and \(D_s\) are diagonal area and arclength measures. The solver uses the same \(J_q\) in the bulk residual, tangent and work ledger. With plane and parent fixed, the local law takes the form \(w_q=w_q^{\rm free}-m_q f_k\). After integration the scalar equation is \(V_k=V_k^{\rm free}-M_k f_k(V_k)\). The integrated fixed-force variation includes the derivative of the point mobility. Its multiplier for the induced resistance is \(f'_k/(1+M_k f'_k)\). Returning that variation to the same points gives the existing condensed tangent. The fracture energy and its SIPG quadrature retain their own phase-field operator.

## Ownership and limits

The material owner constructs the station map and Gauss points once per fixed mesh. Reference membership never follows the evolving damage contour. These lines couple distributed slip to a single age and resistance per station. They impose no mature phase profile ahead of the tip. If a virgin structural plane turns, the material section remains along \(n_0\) and its effective normal weight is \(J_q/ds_k\).

Every prospective station starts with the configured age and zero slip. All stations age through accepted intervals, including before registration. The full-section direction law retains its positive-phase registration. The localized forward-tip option has the lifecycle below. Solver trials hold the accepted point and station histories immutable. A candidate stores accepted age, slip, plane and formed status together with global tensor history. Cut rules have their own checkpoint signature with the declared angle, Gauss order, ordered stations, cell vertices and point counts. Rebuilding the same rules restores their material-point identities. Changing frame or quadrature requires a fresh run.

Use this route for a fixed affine 2-D mesh, one incoming direction and complete sections of one specimen. Each section sums signed deformation across the entire transverse span, including opposite-signed bands. Mesh alignment economizes on points. The strip width must resolve along-fault variations in the nonlinear section response. The archive still gathers native point histories across ranks. Production budgeting must include candidate and trial fields, FE operators and archive replication.

## Localization on one forward front

Under **Material model / Phase field fault**, enable **Localize prospective tip direction**, declare **Initial direction selection tip** in global Cartesian coordinates, and set **Direction selection radius in phase lengths**. The initial trial radius is twice the phase length. **Bulk tip section closure = material strips** selects the integrated onset law. Restart identity includes the law, localization, seed coordinates, radius, and material discretization.

The first interval uses the known seed tip. Subsequently, *accepted_selection_center* starts one phase length inside the seed. Ahead of the seed it constructs a piecewise linear graph by accumulating Δξ tan β from the accepted formed planes. At consecutive stations it evaluates the accepted Bernstein polynomial on the graph, taking the larger trace at a shared DG face. The center advances from the declared seed tip to the last consecutive formed station where absolute phase reaches 0.05. The first gap ends the scan. This station-resolved indicator assumes an anchor within the seed core and one forward graph. It cannot measure substation motion or follow a separate flank off the graph. The 0.05 level applies solely to tracking.

*prepare_tip_neighborhood* freezes the positive compact kernel ρ = max(0, 1 − |X − Xtip|²/R²)² at the fixed material quadrature points. The direction measure is Jq ρq. A virgin strip is eligible when some material sample lies inside the open disk. Every angle and Newton trial uses the same weights. A retry or restart reconstructs them from the immutable parent. The declared initial seed keeps its prescribed orientation. This option tracks one forward tip. A second independently activating tip requires its own neighborhood or a separately selected formation law.

For a virgin plane, let F = c/(c+d) be the native relaxed fraction and p the physical Maxwell stress predictor. The integrated onset score is

**Jk(a) = [h′(0⁺)/(2G)] Σ Jq ρq (1−Fq)² [pq:S(a)−fk(a)]².**

The free-rate and mobility moments for the candidate RSF solve integrate Jq Fq over the **entire eligible strip**. A remote stress can therefore affect the score indirectly through the section resistance. Only the direct squared-traction moments carry the kernel. Under cancellation the solver uses centered variance plus a squared mean. The original signed branch, angular sector and roundoff tie rules remain in effect. A converged physical trial residual can test a proposed map before assembling its new tangent. Final acceptance checks mechanics and phase on the final plane and retains its assembled tangent.

At initialization the material seed is the contiguous exact-positive section support containing the anchor one phase length inside the declared tip, truncated at the tip. It registers the incoming plane. In subsequent steps, positive phase can register any previously unformed section only when that section is eligible. The same condition applies on either side of the seed. A candidate that would leave nonzero recoverable fault strain or accumulated absolute fault shear in an unregistered exterior section fails before publication. This is an explicit single-front restriction. A later direction change cannot preserve an already developed Mode-II tensor on a different plane. Prospective sections continue to age.

The same localization switch on **material strips Coulomb formation** weights the preceding accepted signed physical stress by Jq ρq. Its representative stress is the weighted area mean. The positive normalization and isotropic pressure cancel in the angle maximization. The prescribed coefficient is μ₀. The existing selector evaluates both tension-positive Coulomb conjugates, keeps the admissible one closest to the accepted signed plane, and prefers positive shear on a tie. Degenerate in-plane deviatoric stress retains the incoming preference. The mechanical RSF law still uses the prescribed normal pressure. This alternative orientation hypothesis has its own checkpoint identity.

[Measured rule comparison, geometry cost and limitations](../TIP_DIRECTION_AND_STRIP_RESOLUTION.md).

[Bounded verification, covariance and mesh cost](../TIP_REFERENCE_SECTIONS_REPORT.md).

The geometry and FE rule are in [tip_section_geometry.h](../../../../include/aspect/material_model/phase_field_fault/tip_section_geometry.h). Initialization, candidate history and archives are in [bulk_cohesive_history.cc](../../../../source/material_model/phase_field_fault/bulk_cohesive_history.cc). The two section reductions, point return, selection and tangent action are in [bulk_tip_assembly.cc](../../../../source/simulator/solver/bulk_tip_assembly.cc).
