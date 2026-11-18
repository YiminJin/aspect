# First native free-turn closure

The user selects material-area/reference-arclength RSF for this experiment
([task](CODEX_FIRST_FREE_TURN.txt)). This is an opt-in constitutive assumption,
not an exact transformation of the straight law. See
[SINGLE_TURN_RSF_FORMULATION.md](SINGLE_TURN_RSF_FORMULATION.md) for the retained
normal-integral alternative and the distinction between planes and phase paths.
The implementation and native qualification are described separately in the
[first-free-turn results report](FIRST_FREE_TURN_REPORT.md). Mature production and the surface-damage pilot
are unchanged.

Current bounded qualification is the
[overnight time/plane/normal comparison](OVERNIGHT_FREE_TURN_VALIDATION_REPORT.md).
The equations here are unchanged. Qualitative connected steering survives the
completed comparisons through .0775; first registration at tiny phase is
resolution sensitive in material that subsequently slips. This limits claims
about structural directions and a fully refined .080 trajectory.

## Material strips and their measure

Keep the complete vertical Cartesian material sections, including all quiescent
bulk, of the straight control. Each x-Gauss section owns exactly its original
native quadrature points and horizontal measure dx_s. The chosen structural
plane has t=(cos a,sin a), n=(-sin a,cos a), S=sym(t tensor n). Assign the reference
measure ds_s=dx_s/cos a. This is the arclength metric of a forward graph with
r'(x)=(1,tan a); its height can be integrated continuously from the seed. The
graph describes the metric/structural frame, **not** the phase contour. Its
height does not own histories, move material, or enter the phase functional.

These are skew material sections, not rotating straight normal lines. With
X(s,z)=(x(s), y(s)+z), det(dX/d(s,z))=cos a, with the z limits determined by the
fixed top and bottom of the body. Thus area is covered exactly once even when
the selected plane is oblique. Neighboring material strips meet at their fixed
vertical faces; no normal-line fan, near-tip cutoff, covered-area floor, overlap
or gap is inserted. The native quadrature is unchanged. At discrete level a
section is its x-Gauss material line and quadrature measure; ds is an
arclength quadrature weight, not the distance between neighboring Gauss lines.
It is not a new FE cell or a particle-derived support. The formal graph is piecewise smooth and
its angles are not filtered. Smoothness and the actual phase path are outcomes
to examine, not imposed curve shapes.

For local fault shear rate w, use

    V_s = sum_(q in s) JxW_q w_q / ds_s,
    (tau_q-Q_q):S_s = f_s(V_s,theta_s),
    sum_q JxW_q f_s w_q = sum_s ds_s f_s V_s.

The return map is Wb^-1 L^T Ws for V=Lw, Wb=diag(JxW), Ws=diag(ds). It returns
one uniform scalar resistance to every point of a section. Elastic and viscous
bulk deformation never enter V. This generalized slip rate differs from a
literal displacement jump on a curved diffuse band. Uniform bulk resistance
and exact actual-measure power are the motivation for this selected closure;
its nonlocal material-strip interpretation is an additional limitation.

The material strip remains identifiable at arbitrarily small phase. Prospective
sections exist from t=0 with the configured age, zero cumulative slip and no
recorded plane. At V=0, age increases by elapsed time. Formed sections retain
membership, angle, ds, age and signed slip; a contour redraw or mesh/segment
label cannot alter these quantities. There is no material transport or AMR.

## Virgin plane selection and actual coupled evaluation

The declared equilibrated seed records its original horizontal plane. In each
physical trial a prospective section retains one coherent plane across its
entire height. From the same accepted parent and current mechanical/phase trial,
choose the angle maximizing its physical-area integral of

    Y0 = m [(tau_m(a):S(a)) - f_s(a)]^2/(2G).

All points contribute, including intact bulk. This full-section integrated
criterion is an explicit nonlocal extension of the pointwise intact-reaction
proposal. It does not promise that a point's own optimum is the section's
optimum. For each candidate angle compute the physical material response and
consistent section RSF root; do not use auxiliary stress or a fixed old f.
For a virgin section A_n=0, with midpoint duration h=dt/2, beta=1/(1+hG/eta),
F=h_c(phi_m)/(1+h_c(phi_m)), P=beta(tau_n+2G D3(h v_m)),

    w = F (P:S-f)/(G beta h),
    tau_m:S-f = (1-F)(P:S-f),
    V = B-C f, B=sum JxW F(P:S)/(G beta h ds),
               C=sum JxW F/(G beta h ds).

Nine native quadrature moments evaluate B, C and the integrated Y0 for angular
candidates; MPI sums these small section arrays. A 128-interval angular scan
brackets the best maximum, then golden-section refinement resolves its angle
to 1e-9 radians. The admitted conjugate sector is (-pi/4,pi/4), closest to the
horizontal seed; hitting its edge rejects admission with the section identity,
coordinate, force objective, age and slip printed. This bounded single-forward-
bend hypothesis is stricter than the graph singularity cos(a)=0. It is not a
general direction or branching model. Roundoff-equivalent maxima retain the
incoming preference. Zero driving retains that preference without claiming
mechanical direction selection.

Plane selection is an outer constitutive iteration within the existing ASPECT
coupled solve. After mechanics and the phase VI pass their unchanged gates for
fixed trial angles, recompute virgin angles. A change above 1e-7 radians requests
another coupled iteration; at most 12 orientation updates are admitted, within
the original nonlinear iteration budget. All trials read the same accepted
bulk/section state. The point response and condensed tangent remain exact for
that frozen map. No claim of a monolithic angular tangent or global energy
minimization is made. A failure of this outer iteration preserves the parent.

The actual phase force remains the full bulk G h_c' A:A/h_c^2 and native AT1/SIPG
functional. The Bernstein phase VI decides advance, including through the old
seed taper. The maximized Y0 is a **plane** criterion, not a growth threshold or
a replacement phase force. The scalar prescribed pN remains orientation
independent; pressure is the existing constrained mechanical multiplier. No
Coulomb angle, current-normal-pressure law, opening or contact is inferred.

## Registration, work and lifecycle

A prospective section records its selected plane when accepted phase first
becomes positive at any of its native points. There is no positive phase cutoff.
Its age and signed slip come from the existing immutable-parent midpoint update:
stage friction uses the stage rate and half-step exact age; publication advances
age over the full step and adds dt V_m to slip exactly once.

Registration cannot use a positive phase cutoff. The first native append
control recorded planes at phase about 1e-12. The current-row natural-map
classification alone suggested that these values could be placed on the zero
bound, but the actual zero-state trial raised phase KKT to about .06: the
regularized section friction reaction changes sharply across its cold-creep
scale. Small phase is not proof of a removable numerical tail.

The opt-in solver therefore **tests** an exact active-bound candidate inside
the ordinary constrained iteration. It accepts that candidate only if its
recomputed mechanics and phase residuals pass the original gates, then checks
again before publication. Otherwise it retains the admissible positive-state
solution. No phase cutoff, tolerance relaxation, obstacle change, post-publication
clipping, discarded A or age reset is introduced. Zero-clock seed equilibration
also finishes its active bounds before recording the initial seed. The default
straight solver ordering remains unchanged. Both the original tiny-registration
evidence and the unsuccessful unconditional-polishing attempt are retained.
The timing of infinitesimal formation remains distinct from resolved phase-path
advance and requires explicit reporting.

Full physical tau/A stay at native bulk quadrature points. Existing plane memory
is never projected. Changing a virgin section's metric cannot hide old slip or
storage: the section is required to have exactly zero A and accumulated absolute
fault deformation before a new plane is considered. The reference material
membership never changes. The existing physical-work ledger therefore accounts
for the entire bulk response; there is no extra contour or graph-shape work and
no separate surface fracture inventory. Across-band cancellation and negative
local f*w remain observable; only section-integrated fV is constrained positive.

`Bulk tip section closure = material strips` requires the integrated native
midpoint route. Defaults remain `fixed direction` and BE. The new physical
checkpoint tag is `bulk tip physical tau A material strips midpoint v1`, with
angle, formed flag and reference measure added to each section record. Old
straight checkpoints are rejected under the new selector. Restore checks the
fixed material x identity and dx=ds cos(a); it restores the actual weights and
all full tensor and section histories. No automatic conversion is supplied.

## Formulation-to-code map

| Responsibility | Existing owner / changed operation |
|---|---|
| Selector and admission | `PhaseFieldFault::declare_parameters/parse_parameters` |
| Immutable material points, angle, measure, age/slip and checkpoint | `BulkCohesiveState::Section`; `bulk_cohesive_history.cc` serializer/restore |
| Physical stress/A update, stage age and RSF | Unchanged `Tip::update`, `midpoint_endpoint`, `FaultFriction` |
| Native-weight rate and work adjoint | `BulkTipCoupling::evaluate`, using trial `section_angles` and `section_measures` |
| Coherent Y0 selection | `BulkTipCoupling::update_virgin_planes` |
| Admissibility-tested active-bound candidate | `BulkTipCoupling::finish_strip_active_bounds`; zero-clock `equilibrate_seed` |
| Ordering and coupled recheck | Existing Newton loop in `solver_schemes.cc` |
| Single candidate/rollback/publication | `BulkTipCoupling::candidate`, `prepare_bulk_tip_candidate`, existing pending publication callback |
| Read-only plane/state/work exports | Benchmark `bulk_tip/checks.cc`, `tip-planes-*.csv` and existing phase/power records |

The test selects a 20-degree change in the homogeneous strain-rate plane after
.04 s, keeping engineering shear rate .01/s and adding a traceless normal rate.
The reflected load transforms the **whole** velocity history across y=2, including
its initial shear sign. The phase contour, any lateral displacement and actual
new material activation must be measured from the coupled solution.

## Current qualification boundary

The [original coarse history](FIRST_FREE_TURN_REPORT.md) accepts through .075 s
and fails the .080 phase-KKT gate. The [bounded recovery and reflected comparison](RECOVER_FREE_TURN_STEP16_REPORT.md)
complete both signs through .080 with two .0025 midpoint intervals from independent
.075 parents, preserving the original executable, equations and acceptance gates.
No derivative or search-rule correction was adopted. Near-intact nonlinear trial
rows obstruct the coarse search even when the current worst row follows its tangent.

At .080 the connected absolute phi=.05 contour extends .243096 m beyond x=6,
with lateral displacement ±.0482614 m under load reflection. These are3.89 axial
and1.54 normal cells; central W/hy is2.10. Reflection agrees closely in native
phase, physical tau/A and section state, but turn magnitude/angle is not qualified
in time or space. The inherited old/new structural-plane jump remains about.156 rad.
Some sections still register at infinitesimal phase, and mixed-sign local deformation
occurs in88 sections, with maximum integrated cancellation .727%. Negative local
work is retained under the whole-section nonlocal law; section power stays nonnegative.
This is neither a smooth-plane result nor general branching qualification.

Original failure evidence, accepted parents and complete checkpoint histories
remain intact. The proposed next numerical check is four .00125 intervals from
the same positive .075 parent, compared with the two recovered half-steps. It has
not been launched. The mature-fault production path and its qualification are separate.
