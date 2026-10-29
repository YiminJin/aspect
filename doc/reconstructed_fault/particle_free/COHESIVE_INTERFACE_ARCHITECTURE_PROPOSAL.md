# Proposed roles of phi, a, Gamma and the jump

This is a forward architecture proposal for the accepted physical separation,
not a new production algorithm. The current test evolves **surface damage a**
on a fixed prescribed candidate Γ; no bulk phi evolution or new path is enabled.

| Object | Proposed responsibility | Authoritative physical history |
|---|---|---|
| Bulk | Maxwell equilibrium, pressure, recoverable tensor memory | Full physical bulk tensor/memory, independent of surface visualization |
| Γ | Oriented material surface and exact-jump/work-adjoint kinematics; initially tied candidates may be represented | Reference geometry, normals, topology, birth/reference clock; no free fracture energy from metadata alone |
| Jump δ | Actual tangential displacement discontinuity; zero normal gap in the present closed-contact scope | Signed accumulated geometric displacement, relaxed jump memory; signed V=δdot and absolute slip are distinct |
| Surface a | Irreversible creation of compliance and progressive activation of frictional contacts | Gauss-owned a, bond reference b, one interface RSF age Θ per surface point |
| Bulk phi | Initially a **geometric/path representation**, reconstructed from Γ and surface history for search/visualization | No second independent degradation or fracture-energy reservoir in this proposal |

At partial damage the jump stretches recoverable bonds while their damaged
fraction also slides frictionally. This is the selected parallel bond/contact
model, not an unannounced series plastic-slider split. At a=1, bond storage and
cohesion vanish exactly; Γ retains contact, one signed velocity and one age.
The bulk tensor memory persists and continues Maxwell relaxation. Direction is
an established material record; reversal changes signed velocity, not its normal.

## Formation of a path

1. In virgin material, assess candidate closed planes using the **maximum-shear
   intact release** c0(t·τn)^2/2. Admission reaches Gc at |t·τn|=τp=3000 Pa.
   This fixed reference-calibrated strength is not a dynamically recalibrated
   Coulomb angle/strength. Solved normal pressure must remain compressive.
2. Declare a geometric/history preference for exact maximum-shear ties; preserve
   unresolved symmetry if none exists. Do not average conjugate flows, infer a
   normal solely from the changing slip vector, or rotate established records
   merely to increase their instantaneous driving force.
3. Create prospective **tied** Γ metadata with a=0, δ=b and zero new bond/fracture
   energy. Existing bulk tensor histories are preserved. The prospective contact
   clock is inherited from the agreed material-age convention, not reset to
   steady state. Existing Γ records retain all accepted histories.
4. In the joint mechanical/admission solve, enforce zero jump while a record is
   tied. A weak multiplier in the intact strength interval balances its load.
   Loss of weak admissibility allows a>0 and a geometric jump; bond energy,
   fracture increment and contact work are then accounted together. This is not
   insertion of a finite damaged/slipping segment for free.
5. Commit geometry, jump, age/damage and bulk history jointly once. Rejection
   restores all accepted owners and geometry/cache generations. A new numerical
   trace space may change FE approximation even at zero jump: measure that
   remapping defect separately from physical formation and do not label it free
   fracture energy. Restart must identify both the physical records and the
   discrete trace space.

This specifies local admission and history handling, but **not a unique global
free-path topology algorithm**. Candidate generation, junctions, overlapping
records, secondary fracture and moving material orientations require explicit
subsequent qualification. The present weak-front test only checks candidates
already on a straight line; its mirrored/periodic response is not a path-choice
prediction. A stress direction is not the geometry of a phase contour.

## Count fracture energy once

The selected inventory is Gc∫Γ a ds, with bond storage
(δ−b)^2/[2 c0 a/(1−a)] and bulk Maxwell energy separately. The local contact
power is a[pn μ(|V|,Θ)|V|+ηrad V²]. **Progressive contact activation is exactly a**:
there is no friction before formation and full geometric-interface RSF at a=1.
At fixed geometry the energy/work account is the one tested in the current
reports; backward-Euler remainders remain numerical, not physical loss.

For a reconstructed diffuse phi used only to locate Γ, do **not** also add the
old bulk phase-gradient/fracture energy or degrade the bulk again. Its smoothing
width is numerical geometry support, not another physical Gc or the slip process
zone. The exact-jump test establishes why changing that representation must not
change nonuniform mechanical stiffness.

If future work instead makes phi a thermodynamic free-path variable, it must
regularize **this same surface-creation measure**, with a derived coupling and
an explicit transfer of the fracture inventory to Γ/a. One cannot independently
minimize the old phi energy and simultaneously charge Gc∫Γ a. That alternative
requires a new variational derivation before implementation; it is not enabled
by this proposal. In particular, irreversible old bulk phi cannot simply be
reinterpreted as this surface a without an energy/history-accounted mapping.

Recommendation: keep phi geometric for the next bounded path-representation/
transfer study, keep a as the sole fracture-energy inventory, and validate tied
record insertion and explicit material-normal/age ownership before any free-path
propagation. Finite physical tips and secondary/junction formation remain open.
Production/BP3's established phi/Γ semantics are not changed by this proposal.
