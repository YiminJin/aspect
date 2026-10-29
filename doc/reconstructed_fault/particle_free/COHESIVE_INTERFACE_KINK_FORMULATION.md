# Bounded non-collinear path admission and transfer

This is an isolated native deal.II mixed-FE control, not a production topology
change or a selected propagation direction. The accepted cohesive–RSF parameters
and point law remain unchanged. The earlier 65.4% early-growth spatial failure
and unqualified first-activation timing remain evidence.

## Contract selected before computation

Use a fitted candidate polyline with a horizontal old segment ending at
(0.5,0) m and a prescribed slope-1/2 append ending at (0.75,0.125) m.
The old segment starts at (0.25,0). The block is [0,1] × [-0.5,0.5] m.
The fitted bulk mesh is fixed during insertion. This kink is prescribed to
exercise vector traces; it is not selected from a coarse tip stress.

Bulk displacement is Q2, with cell-local discontinuous linear pressure
(Q2/P1disc). The latter is a deliberate reference discretization, distinct from
the preceding Q2/Q1 component. It represents independent pressure traces on
both branches, including the prospective tied append. It is fixed before and
after insertion; changing a pressure approximation is not a zero-energy birth.
On the fitted seam, duplicate the upper displacement trace and impose

    u+ − u− = t δ,       n·(u+ − u−)=0.

Retain both components of the rotated tangent and the complete pressure/divergence
coupling. Outside installed records δ=0 by representation. Newly installed
intact records impose δ=0 through weak bounded shear multipliers. Existing
damaged records retain their material tangent, a, bond reference b, age, slip,
absolute slip and Maxwell memory. Bulk tensor history is retained in physical
FE coefficients, not rotated to the new tangent. The vertex at the kink has
zero vector jump: continuous traces with zero normal gap on two nonparallel
branches imply this. This is a bounded closed-corner rule, not a qualification
of sliding through a corner with opening or a displacement singularity.

New records receive a=δ=b=relaxed jump=absolute slip=0 and the accepted virgin
material age. Geometry insertion advances no clock. Its fracture and bond
storage increments are exactly zero. The inventory remains Gc integral(a ds),
with actual oblique arclength. Phi is geometric metadata only, with no second
fracture/degradation term. Old material record identities and Gauss locations
are immutable; a new segment owns its own Gauss3 rule, not interpolated ages.

For assembled mixed bulk A, jump coupling B and Gram D, eliminate only the
bulk mechanical unknowns. Use the full mixed response, including pressure:
K=D−B^T A^{-1}B, with shear force −(Kδ+f). Surface normal pressure is evaluated
from both physical traces of sigma=tau−(p0+p')Id. Closed contact requires both
normal pressures positive. No fixed-pressure substitute is permitted.

At insertion, preserve the full physical state and test existence of Gauss
multipliers |lambda|≤3000 Pa satisfying weak equilibrium on the newly available
trace variations. Reject if infeasible. An overloaded append requires physical
admission with damage, work and dissipation; it cannot be inserted for free.
This task tests and rejects that case, rather than inventing a joint birth law.
Future event-resolved admission must bracket loss of this certificate, then
solve the existing irreversible return with the full pressure response and
count fracture work once. Negative curvature is not cured by topology changes.

## Topology and direction design

Records have immutable material IDs, oriented endpoints and Gauss histories.
Prospective edges are separate from damaged edges. Permit one append to a free
endpoint; reject crossings, overlap, zero length and secondary fracture through
an established record. Existing normals never rotate during candidate search.
At virgin points the intended admission criterion remains maximum-shear intact
release, c0(t·tau n)^2/2=Gc, under compressive contact. A declared continuation
preference resolves genuinely tied candidates only; it must not overrule a
resolved maximizing direction. A physical finite averaging region and a focused
resolution comparison are required before treating computed near-tip directions
as path evidence. This control uses radius 0.1 m, centered at the original tip,
and compares 16×8 and 32×16 fitted meshes. It does not qualify pointwise singular
stresses, exact activation times or a unique global path.

Stage geometry, jump-space constraints, pressure trace maps and Gauss owners
together. Validate stress/storage/history and the tied certificate, then publish
one geometry generation. Rejection restores the old generation and every
history; deterministic retry and serialized restart must reproduce the accepted
state. Geometry-only insertion is distinct from a physical timestep.

## Gates

Before runs: unchanged-state field/energy transfer relative error ≤1e−11
(absolute stress allowance 1e−7 Pa); weak-force certificate ≤1e−8 Pa after
division by nodal arclength; contact positive; old histories bitwise unchanged;
exactly one geometry publication; restart coefficients identical. Compare an
unloaded tied case, a nonzero loaded case with nonuniform old histories, and an
overloaded new segment that must reject without publication. Verify the mixed
operator/pressure response and independent higher-quadrature energy. Local
direction evidence is only called resolved if common-kernel principal-axis
differences are ≤5 degrees and stress-tensor differences ≤5%; otherwise report
the discrepancy without another mesh campaign. This is a mechanics/transfer
control, not a new evolution or solver-stability qualification.

## Implemented reference details and interpretation

The control uses native deal.II MappingQ1, Q2 vector displacement, P1disc
pressure, SparseDirectUMFPACK and full three-dimensional plane-strain deviators.
All boundary displacements are prescribed; the pressure perturbation has zero
volume mean and p0=10000 Pa is the background confinement. The displacement
correction is continuous except for duplicated upper seam nodes. Exact normal
constraints are imposed with the **segment** tangent, not a smoothed vertex
normal. A pressure trace exists on each side before insertion. The reference
uses the mean of the two current normal tractions in the retained point return
and checks compressive contact on both sides independently.

The old synthetic loaded snapshot is prepared in mechanical equilibrium at
boundary shear strain .001. Its continuum prescribed histories are

    a(x)=.4+.1 sin(4 pi (x−.25)),
    b(x)=2e−5 sin²(4 pi (x−.25)) m,
    Theta(x)=1.2+.1 sin(8 pi x),        .25<x<.5.

The existing bond reference and pre-aged material are deliberate nonuniform
incoming histories, not a claimed fresh loading trajectory. Old damage already
has its fracture inventory; the elastic extension is below the retained damage
obstacle, so this is an admissible unloading state. Solve old jump and the full
mixed bulk response jointly with the fixed old compliance. New material starts
at the virgin material clock Theta=1.001 (200.2 s dimensional age). The physical
snapshot time .2 s is unchanged by geometry insertion. All nine Cartesian tensor
entries are represented through the full native FE strain; no scalar stress
rotation or projected source is used.

A subsequent small **reference** physical step uses dt=.002 s and affine strain
increment .0006, to exercise active RSF response and its pressure derivative.
The existing C++ point return is checked independently against every implemented
partial in the reference harness. Bulk memories advance by

    c_new=beta(c_old+Delta c_displacement), beta=1/(1+G dt/eta).

This large diagnostic strain increment is a tangent/restart check, not a
physical temporal-accuracy study. Endpoint work, Maxwell/sliding losses and the
BE and surface remainders are reported separately. No energetic accuracy claim
is made from the algebraic ledger.

The deliberately inadmissible append uses a **different**, fresh homogeneous
axial loading u=(gamma x,−gamma y), gamma=.0025. It produces tau_xx=5000 Pa,
tau_yy=−5000 Pa, tau_xy=0. The old horizontal plane has zero shear; the append
has |t·tau n|=4000 Pa and pn=13000 Pa. Thus the old tied state is admissible and
only the newly available oblique slip direction violates the intact bound.
The experiment rejects it; no damage, slip, fracture work or clock is invented.
A preliminary simple-shear overload also overloaded the old tied plane and was
therefore discarded as an isolated admission test, with its diagnostic retained.

The topology checks allow one forward graph append only. Zero-length edges,
backtracking/crossing proposals and additional edges reject. A future free-path
implementation must do geometric intersection/contact checks for its broader
candidate family; these checks do not qualify arbitrary networks.

The radius-.1 m compact kernel is (1−r²/R²)² inside the disk, integrated with
physical Gauss6 weights and normalized by its integral. It is a **resolution
diagnostic**, not a new nonlocal strength law or a replacement for the actual
weak admission certificate. Its principal axes imply two conjugate shear planes;
report both. No path is selected from these averages in this task.

The native manager adapter imports the independently solved mechanical snapshot,
then stages its geometry, trace layout and complete coefficient/history payload.
It exercises actual manager swaps (which invalidate native geometry associations),
rollback/retry, duplicate rejection and real ASPECT checkpoint/restart. The
fixture's full mixed kernel is assembled on the fixed candidate mesh beforehand;
only the admitted trace constraints change during insertion. This is not a
native outer mechanical solve, MPI topology transaction, pressure-space remap,
arbitrary DG-history conversion, or a test of populated production projection
caches. No normal-profile normalization enters the exact-jump operator.

The native manager uses .02 m search-support metadata only. It is not used by
the fitted exact-jump mechanical operator, a normalization floor, a phase length,
or an additional energy term. Free paths outside the fitted candidate mesh would
need a separately qualified cut/enrichment or mesh-transfer representation.
