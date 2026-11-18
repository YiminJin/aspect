# Single-turn RSF: material carrier and formation proposal

11 November 2026. [Task](CODEX_SINGLE_TURN_RSF_CLOSURE.txt),
[measured reference comparison](SINGLE_TURN_RSF_REPORT.md).
The user closes the [straight peak-.60 control](RESOLUTION60_OVERNIGHT_REPORT.md)
for qualitative connected extension. This note proposes an extension; it does
not change the native fixed-direction law or enable a turning calculation.

## Existing equations and the transferable part

In the fixed reference configuration, with full three-dimensional plane-strain
contractions, let S=sym(t⊗n), S:S=1/2, and D_f=wS. Retain

\[
 \dot\tau/(2G)+\tau/(2\eta)=D_3-wS,\quad
 \dot A+(G/\eta)A=wS,\quad Q=2GA/h(\phi),
\]
\[
 \psi=\tau:\tau/(4G)+GA:A/h,\quad
 Y=Gh'A:A/h^2,\quad h=m\phi(1+\phi)/(1-\phi)^2.
\]

The sole fracture inventory is the native AT1/SIPG functional
K∫(φ+ell²|∇φ|²)dx, K=Gc/[(8/3)ell]. Its Bernstein coefficient variational
inequality, not a contour crossing or a new geometric segment, decides growth.
Full τ and A, including transverse variation across a section, stay in the bulk.

The [P5c comparison](P5C_REPORT.md) demonstrates that changing the local tensor
flow law changes cap driving without making a turning phase path. Its radial
tensor angle is not a crack tangent. The
[structural-plane reference](DIRECTIONAL_CONTINUATION_FORMULATION.md) supplies
a useful **additional constitutive assumption**: an established material plane
keeps its orientation and age, admits signed reversal, and need not maximize
over competing planes on every increment. Its prescribed formation event and
local normalized speed/state do not supply the present bulk section birth rule.
The sharp cohesive-interface pilot instead owns surface damage/fracture energy;
none of that inventory or checkpoint meaning is imported here.

## What RSF would belong to at a bend

Candidate ownership is one immutable material cross-section (a set of reference
material points), its signed frame convention, and its reference arclength
measure ds. It owns θ in seconds and cumulative signed slip δ in metres.
Bulk points retain τ, A and accumulated fault deformation. Labels are persistent
material identities, not CellId, contour-node or reconstructed-segment identities.
Only fault deformation enters V; ordinary elastic/viscous D is excluded.

For a prescribed smooth single-branch chart X(s,n)=r(s)+n n(s), with s reference
arclength, curvature κ, and J=1−κn>0, dx=J ds dn. Nonintersecting normals and
unique ownership are admission requirements, not normalization floors. There
are at least two distinct extensions of the straight closure:

| Candidate | RSF rate V(s) | Bulk scalar resistance f_loc | Physical meaning assumed |
|---|---|---|---|
| Material area / reference length | ∫J w dn | f(s) | Fault deformation per unit material reference length |
| Normal integral | ∫w dn | f(s)/J | Normal integral of engineering fault deformation |

Both reduce exactly to the current law for κ=0. Write either as
V=∫J g w dn and f_loc=g f, with g=1 or 1/J. Then

\[
 (\tau-Q):S=g f(V,\theta),\qquad
 \int_\Omega g f w\,dx=\int f V\,ds.
\]

Use the same signed regularized RSF plus radiation and scalar pN>0 as the
straight control. Its midpoint stage age is the existing exact constant-speed
aging over Δt/2; publish θ using the full Δt and δ+=δ−+Δt V_m. Neither positive
section dissipation nor this power identity chooses g. A normal integral is
also not automatically a literal displacement jump on a curved diffuse band:
2D_tn=∂n u_t+(∂s u_n+κu_t)/J, and D_f alone need not be a compatible gradient.
Calling either generalized speed an RSF slip rate is a constitutive assumption.

With native bulk masses Wb=diag(JxW_q), section masses Ws=diag(ds_s), and rate
map V=Lw, the required force map is **f_loc=Wb⁻¹ Lᵀ Ws f**. For fitted sections,
L_sq=(JxW_q/ds_s)g_q; for overlapping interpolation use the same interpolation
in this weighted transpose. A convenient interpolation is not itself physical
material ownership. The reference checks both fitted Q6 charts and overlapping
maps on actual Cartesian Q6 quadrature weights.

The existing point response is affine w=b−c f_loc at fixed strain/phase,
c=fraction/(G β Δt_stage). Thus disjoint sections retain
V=B−C f, B=Σdn J g b, C=Σdn J g² c. For overlapping sections the condensed
matrix is L diag(c) Wb⁻¹ Lᵀ Ws; it is positive semidefinite in the Ws metric.
The squared g in C and the metric factors cannot be omitted. The current
midpoint/BE stage equations and tolerances need no change for a frozen map.

## Candidate formation criterion, with its limits

Keep the declared seed plane throughout its original diffuse support, including
the taper. Once material has fault history, freeze its plane; a different local
maximum is a competing-fracture observation, not permission to erase A.
For still-intact, history-free material, a candidate incipient rule is to
maximize the **intact cohesive constraint reaction driving**

\[
 q_0(a)=\tau:S(a)-g f_s,\qquad Y_0(a)=m q_0(a)^2/(2G).
\]

This acts at native bulk points before any reconstructed segment exists. For an
entirely prospective intact section V=0 and smooth RSF gives f=0. The maximum
is then maximum resolved shear magnitude of physical τ, not auxiliary T. In
partially active sections the common f affects intact reactions and must come
from the coupled section solve; using f=0 there would change the model.

At exact φ=0 every candidate has w=0. Consequently the earlier prototype's
maximum *instantaneous speed* formation test is completely tied there. Nor can
one maximize the already-relaxed (τ−Q):S, which equals zero on an entirely
intact prospective section. Using Y0 instead is a new formation assumption,
not an algebraic continuation of that prototype. In the diffuse cap that is
already assigned, retain its plane and use its actual full A driving. The phase
VI still evaluates local/fracture-gradient competition; Y0>K alone is not its
discrete acceptance test.

For an implemented joint trial, τ in this criterion must be the mechanical
trial's physical stress and f its consistent section force, evaluated from the
same accepted parent. Freezing an older stress maximizer would be a separate
lagged approximation. Angular and map dependence would enter the coupled
residual/tangent or an explicitly converged outer iteration; new plane/map/phase
and history would publish together only after acceptance. Those operations are
not implemented by this frozen-chart reference.

For pure symmetric shear there are two orthogonal tied geometric planes, with
opposite S. Choose the plane closest to the established incoming tangent only
as an explicitly declared continuity preference. At an exact preference tie
retain ambiguity; at zero deviatoric driving all angles tie. The point check
recovers a horizontal plane under straight shear, and ±12° under the reflected
off-axis stresses. These are plane selections, **not predictions of a phase
contour or proof that a finite symmetric specimen stays straight**.

The scalar pN has no orientation dependence; no Coulomb angle follows. Mechanical
pressure remains its existing constrained gauge variable and does not enter
this law. An actual-plane contact option would introduce
pn=p0+p′−n·τn, a physical pressure reference, pressure/orientation derivatives,
closed/open rules and a new frictional constitutive assumption. The earlier
[zero-dilation Coulomb audit](COULOMB_LOCALIZATION_REPORT.md) found mixed
short-wave loss at developed phase (φ>.76761 for its m=4, μ=.1 single-system
example). Phase-gradient energy and Maxwell relaxation did not repair that
specific defect. It is not a theorem about the present fixed-pN nonlocal RSF
law, nor permission to import a pressure law without its own audit.

## State before activation, map changes and acceptance

Prospective material is not newborn at an arbitrary contour threshold: initialize
its declared age at t=0. With zero prior section rate, θ=θ0+t, δ=0. A newly
activated record inherits that history, recovering the straight control's
ahead-of-tip ages. Existing nonuniform ages cannot be averaged into a new record
merely because a curve extraction changed. Pure relabelling permutes complete
records and maps, preserving physical ds; under ξ=ξ(s), include ds/dξ. Reversing
one frame vector changes S,w,V,δ,f signs together, leaving θ and physical τ/A
unchanged. Reversing both t and n leaves S unchanged.

Birth can provisionally record a candidate plane when phase evolution admits
previously intact material with its first nonzero fault deformation; no extra
fracture inventory is charged for the label. Phase positivity alone, especially
diffuse tails, is not proof of a developed sliding plane. Zero-force/tied points
require an explicit continuation rule. This proposal does **not yet determine
a unique coherent section foliation through a newly evolving diffuse cap** from
pointwise maxima, or authorize reassignment of established material. Neighboring
point optima need not be normals of the same chosen section. That carrier must
be specified together with its generalized slip measure.

For a fixed chart the reference preserves all tensor storage. Projecting A onto
a changed tangent deletes physical energy. Even holding τ/A/φ fixed, replacing
L changes V and the friction law's future evolution. If δ were redefined as
L(t)k from accumulated scalar fault strain k, its derivative would be
L kdot+Ldot k; a changing S adds the corresponding tensor contraction derivative.
Ignoring these terms is not pure relabelling. Any genuine repartition needs an
explicit history/work rule; append only disjoint history-free material until
one is selected. No work is assigned to diagnostic contours.

## Code boundary and the decision now exposed

| Existing owner / operation | Reuse and restriction |
|---|---|
| [`Tip::update`, `midpoint_endpoint`, `midpoint_account`](../../../source/material_model/phase_field_fault/bulk_tip.cc) | Physical point elimination and accounting; current guard requires A parallel to its recorded S |
| [`BulkTipCoupling::evaluate`](../../../source/simulator/solver/bulk_tip_assembly.cc) | Two-pass bulk/section root, mechanics/phase rows and adjoints; current S and Cartesian sections fixed |
| [`FaultFriction::update_state`, `evaluate_resistance`, `solve_section_slip_rate`](../../../source/material_model/rheology/fault_friction.cc) | Same age, signed RSF, radiation and monotone scalar elimination |
| [`initialize_bulk_cohesive_history`, `prepare_bulk_tip_candidate`](../../../source/material_model/phase_field_fault/bulk_cohesive_history.cc) | Accepted bulk/section ownership and publication; current checkpoint rejects changed section geometry |
| [`single_turn/reference.py`](../../../benchmarks/reconstructed_fault/particle_free/single_turn/reference.py) | Explicit frozen-chart alternatives, independent midpoint quadrature reference only |

**One scientific choice:** identify the material slip carrier at a bend, including
whether its RSF rate is the normal integral or material-area/reference-length
integral. The latter is the closest algebraic extension of uniform bulk section
resistance; the former retains the literal normal-integral definition but needs
the 1/J work adjoint. Neither is selected by bulk energy or the straight evidence.
The measured discrepancy warrants deciding this before native implementation.
Future integration belongs in the existing material/history and coupling owners,
opt-in with a distinct law identity, under the ordinary simulator transaction.
No alternate timestep driver, new native selector, contact law or checkpoint
conversion is introduced by this reference.
