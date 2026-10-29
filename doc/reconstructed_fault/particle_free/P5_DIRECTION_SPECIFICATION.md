# P5 scientific recommendation — tensor memory with changing flow direction

Selected by the user for P5a on 22 October 2026. The experimental stateless
point return and small point-history fixture are now implemented and verified;
see [P5a report](P5A_POINT_REPORT.md). Native history/checkpoint integration and
the full directional specimen remain subsequent tasks. The original reviewed
draft is preserved in bulk_stability.zip and the P5a entry snapshot.

The selected option permits new flow directions in material already carrying
cohesive history, retaining that history as tensors in fixed physical coordinates.
Virgin-only direction locking is not selected. The later specimen is a finite
physical specimen with stated grips and released lateral motion, not an
approximation to distant boundaries. P4's temporal/spatial failures and measured
57.26% boundary effect remain; point qualification does not remove them.

## Retained problem and the missing choice

Keep small strain, the full 3-D plane-strain tensor metric, physical phase
irreversibility, AT1 storage, matched Maxwell times, homogeneous coefficients,
fixed mesh/geometry and accepted/trial ownership. No transport, RSF, AMR or birth
of a computational surface is needed to evaluate the bulk phase driving.
Currently S=(t tensor n+n tensor t)/2 is prescribed and A_c=a S, with
S:S=1/2. T is independent auxiliary stress; tau=T-2G A_c is physical stress.
Here the bulk a, k and L are dimensionless strain memories and w has units
1/s; they are distinct from the earlier surface slip [m] and cohesive Z [Pa m].
Changing S while retaining scalar a changes tau and stored energy. In rates,
dot(a S)+a S/lambda=w S+a dot(S): simply replacing S inserts the additional
orientation strain/work term a dot(S), which the current scalar flow law does
not determine. Rotating T
to hide that change erases established physical memory. Neither is an exact
variable transformation.

Two scientifically different extensions must be distinguished before coding:

* A material direction is chosen only in genuinely virgin material and then
  retained. Existing nonzero cohesive memory keeps its established direction.
  This permits a curved sequence of newly activated material directions but
  does not reorient damaged material. A precise virgin/birth criterion and the
  treatment of the pre-existing diffuse cap are required; phi>0 in the cap is
  already prescribed damage, not newly born surface.
* The active slip direction can change in material carrying cohesive history.
  This requires an orientation-independent persistent memory representation
  and a law for the new active flow direction. It cannot be implemented by
  replacing S in the current scalar correction.

Virgin-only locking would be a different, more restricted law: it needs a
birth criterion and a decision to lock the entire pre-existing diffuse seed
before loading. It is not the recommended extension. The tensor option avoids
rotating or discarding existing strain memory and has an explicit fixed-direction
restriction recovering the current scalar model. It still changes general-stress
yield/driving physics; the user approved that choice for the P5a experiment.

## Selected tensor-memory point law (native integration pending)

Retain A_c as a full symmetric trace-free tensor, not a rotating scalar:

    tau=T-2G A_c,       Q=2G A_c/h,
    dot T+T/lambda=2G D,
    dot A_c+A_c/lambda=w S(theta),
    tau:S(theta)-Q:S(theta) in R partial|w|.

Storage is ||tau||_3D^2/(4G)+G A_c:A_c/h. The phase driving is
G h' A_c:A_c/h^2; bulk/cohesive viscosity and frictional dissipation are
||tau||_3D^2/(2eta), 2G^2 A_c:A_c/(eta h), and R|w|. For A_c=aS and fixed
S these reduce exactly to the present equations, energy and rates. Older
components of A_c persist under the Maxwell recurrence when theta changes;
they do not disappear into a new scalar a. Retain accumulated inelastic strain
as a second tensor K with dot(K)=w S(theta), and Ldot=|w|. At fixed S these
reduce to K=kS and the existing signed/absolute shear histories. A scalar signed
k in a changing director chart is not an invariant net slip; its old value must
not be rotated into a new direction or confused with A_c. Report 2K:S0 only as
a fixed-axis comparison diagnostic. L and frictional work remain invariant
under S/w sign relabeling. At h=0 the intact multiplier must be rederived as a constrained
limit for the chosen orientation rule, not computed as 0/0.

This isotropic tensor cohesive storage is a new constitutive extension. Its
admissible shear space is in-plane trace-free: A_c,xx=-A_c,yy and A_c,zz=0,
with two independent components; full 3-D contraction includes both xy terms.
The full plane-strain T/tau, including zz, remains unchanged. Multiple permanently
oriented systems or out-of-plane flow would be different models.

### Concrete free-flow proposal

This proposal permits reorientation of new flow in loaded material without an
independently rotated director history. Let P map
a symmetric stress to its in-plane trace-free part, leaving the full 3-D T and
physical tau intact. For X=P(X), define

    M(X)=sqrt(((Xxx-Xyy)/2)^2+Xxy^2)=||X||_3D/sqrt(2).

The yield set is M(P(tau-Q))<=R. At fixed phi the local BE update minimizes

    ||Tplus-2G A_c||_3D^2/(4G)+G A_c:A_c/h
       +R sqrt(2) ||A_c-beta A_cn||_3D,

with A_c-beta A_cn restricted to the in-plane shear subspace. The Maxwell-scaled
older components are retained. This convex local problem determines a tensor
increment; its active direction maximizes resolved effective shear. At h>0 put
c=2G(1+1/h), Xtrial=P(Tplus-c beta A_cn). Stick if M(Xtrial)<=R. Otherwise

    A_c=beta A_cn+(1-R/M(Xtrial))*Xtrial/c.

The physical rate tensor is (A_c-beta A_cn)/(beta dt). Its scalar magnitude
|w|=sqrt(2)||A_c-beta A_cn||/(beta dt) advances L and frictional dissipation;
the same tensor rate advances K. The complete derivative includes the radial
direction's response. At a yield tie use a declared semismooth active-set rule,
not a frozen direction derivative. Fixed-direction restriction recovers the
current scalar return exactly; the unrestricted return is additional physics.

In genuinely intact material h=0, A_cn=0 and the constrained limit is A_c=0.
The in-plane cohesive multiplier is
Q=P(Tplus)-projection_{M<=R}(P(Tplus)), and
Y=h'(0)||Q||_3D^2/(4G). Thus intact material receives driving without a locked
director; no phase floor or inserted surface is required. Under pure initial
xy shear the existing intact threshold R+sqrt(2G Kf/m)=3000 Pa is recovered.
Under general stress this uses maximum in-plane effective shear rather than
the old prescribed xy component: that is the identified constitutive choice.

For this proposal the unloaded initial seed keeps its physical phase/obstacle
and zero A_c/K/L, while subsequent flow directions may change there as well as
ahead of it. A nonzero loaded state's A_c and K are retained as tensors, never
rotated to a newly selected plane. Report S=P(tau-Q)/(2M(P(tau-Q))) only where
flow is active; below yield it is an undefined diagnostic, not missing history.
Conjugate plane labels representing the same tensor increment are physically
equivalent in this isotropic model. No rotation-based geometric path selector
is required or justified by this tensor update alone. Any distinct geometric
plane selection would be a separate proposed law.

This is the point law selected for P5a. Implementation remains confined to the
stateless point API and fixture; native P4 histories and checkpoints are unchanged.
The pre-existing seed keeps exactly its physical phase and obstacle. It does
**not** acquire a locked material direction merely because phi>0: this proposed
isotropic tensor law has no persistent director to lock. With unloaded A_c/K/L=0,
flow may later change direction inside the seed and ahead of it. This explicitly
supersedes the old draft's unconditional statement that the seed keeps S0. Under
the unselected virgin-locking alternative that would instead be a separate
initial material assignment. Existing nonzero tensor memory is retained in
both cases; it is never replaced by a newly aligned scalar.

## Direction diagnostics and branch convention for this proposal

Under the present constant R=mu*p_N, p_N is prescribed and angle-independent.
A maximization of admissible in-plane shear over theta uses the physical
stress/backstress, not auxiliary T alone. Introducing angle-dependent normal
friction from the mechanical pressure would change the current law and is
outside this proposed extension. Principal directions of a symmetric stress
are axes: eigenvector sign is not a physical branch. Any plotted director is
modulo pi and is not persistent memory. Reversing S with w preserves tensor
flow, tau, storage and absolute slip; do not transform old A_c or K with it.

With constant angle-independent R and isotropic AT1 storage, conjugate planes
can give S2=-S1 and w2=-w1: the same wS, stress, storage and Y. A director label
alone then does not select a distinct phase-growth path. A geometric plane rule
or additional directional coupling must be declared rather than inferred from
that relabeling.

For the recommended isotropic return, conjugate plane labels describe the
same tensor increment; no rotation-based selector is needed to determine the
constitutive response. Do not introduce one as an undeclared geometric law.
At zero effective shear the flow is zero and the direction diagnostic is
undefined. At yield, declare the semismooth stick/slip derivative convention and
check both sides. Geometry/ridge interpretation and a physical rule selecting
one of several distinct slip systems would be subsequent scientific choices.

## Discrete and lifecycle contract

Intact material ahead of the seed is old bulk material, already carrying T
from mechanical loading; activation must preserve that T. It is not born with
zero stress. A_c/K/L are zero only where their actual prior histories are zero. The intact
limit above defines driving before damage; no direction-locking event or phase
floor is introduced. Phase is prescribed damage, not an extra director history.

Store A_c and K with native CellId/quadrature ownership and checkpoint them;
L and T retain their meanings. No orientation/locking status is needed by this
proposal. All Newton trials start from the same accepted T, A_c/K/L and physical
phase obstacle.
Compute stress, phase driving, energy, return tangent and direction derivatives
from the same state. A direction-dependent local elimination must include its
consistent response; freezing a selected theta while claiming a complete
coupled tangent is insufficient. Independent-master phase constraints remain
C^T g/C^T H C. No added geometry may supply missing intact driving.

Mesh/mapping/basis caches keep their existing fixed-geometry lifetime. Effective
stress, active direction and its derivatives are trial-dependent and must be
refreshed within the local return, not cached as geometry. Accepted-memory
caches key on the publication counter. Initially retain the qualified native
quadrature/evaluation layout; an off-layout physical-history query needs an
explicit representation check, not interpolation into a fictitious constitutive
state. No quadrature-history transfer is implicitly authorized by P5.

Rejected steps restore every tensor history and cache together. Retry is
deterministic and publishes once after outer acceptance. Restart must reproduce
physical stress and energy, not merely the orientation coefficients. Existing
DG/native-FE T representation and independent source coverage are retained.

A fixed-direction checkpoint identity control may map old a to A_c=aS0 while
preserving T, physical phase, k/L, stress and storage. Enabling free direction
from an already loaded P4 checkpoint would be a declared model transition, not
a continuation of the same problem. The directional comparison should normally
start freshly from matched physical initialization; no automatic checkpoint
conversion is authorized here.

## P5a point qualification (completed within its report scope)

Reuse the current stateless point return and benchmark driver. Add an explicitly
selected tensor-return experiment; leave the native production law/checkpoint
unchanged initially. No free-direction 2-D trajectory is part of this first task.

1. Restrict Delta A_c to span(S0) and map a -> A_c=aS0, k -> K=kS0, retaining
   T/L/phi. Compare stress, q=Q:S0, Y, storage, dissipation, return and complete
   tangent against the current scalar implementation on stick/slip, relaxation
   and non-proportional T paths. This is the fixed-direction identity control.
   Unrestricted flow under arbitrary non-proportional stress is intentionally
   different physics; do not demand it reproduce the old fixed-xy law.
2. Enable the in-plane radial return on one loaded non-proportional path. Show
   retention of the old A_c/K components as the new increment turns; verify
   rigid basis covariance, S/w sign equivalence and L/friction invariance.
3. Check the intact constrained limit, pure-xy 3000 Pa threshold, general-stress
   change, yield ties, point potential/energy and directional derivatives of
   the complete return (including radial-direction and phase response).
4. In a small point-history fixture, reject/retry, publish once and serialize/
   restore all tensor histories. Require identical physical stress/energy after
   restart. Reuse existing consistency/derivative bounds; no relaxed tolerance
   or projected fictitious constitutive state.

The P5a report records these checks and proposes a bounded P5b native owner/
checkpoint, coupled-tangent and stability qualification. A full directional
finite-specimen comparison follows that qualification; it is not begun here.
Changing law on an already loaded P4 checkpoint is a model transition,
not a same-problem restart. Preserve the P4 fixed-direction control and all its
accuracy limitations. Do not begin directional evolution, RSF, AMR, transport or
geometry evolution as part of the present recommendation.
