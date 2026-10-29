# P5 specification draft — free direction is an additional constitutive step

Prepared during the P4 continuation check; no directional implementation or run
is authorized by this document. The current bulk model already permits phase
evolution throughout the domain. P5 does not remove an activation mask. The old
surface/RSF roadmap's principal-stress/rotation selector is a design motivation,
not an existing law for this bulk cohesive memory.

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

The first option is the smallest P5 candidate for freeing only newly growing
material, conditional on declaring the virgin rule. If the intended P5 really
allows reorientation of loaded material, the recommended proposal is the tensor
A_c candidate below with maximum resolved driving over admissible in-plane S;
selecting that extension and its plane/locking rule is a scientific decision
before implementation. The second is described below as a consistent candidate extension,
not selected physics. P4 branch/accuracy failures remain blockers to interpreting
any directional trajectory as validated propagation.

## Tensor-memory candidate if reorientation is selected

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

This isotropic tensor cohesive storage is a new constitutive extension. Other
choices, e.g. multiple permanently oriented slip systems, represent different
models. Selection must precede implementation and be stated explicitly.

### Concrete free-flow proposal, conditional on selecting that extension

For a specification that permits reorientation in already loaded material, use
the second option without an independently rotated director history. Let P map
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
Y=h'(0)||Q||_3D^2/(4G). Thus intact material receives driving before a director
is locked; no phase floor or inserted surface is required. Under pure initial
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

This is the concrete tensor-memory P5 proposal for scientific selection, not
an implemented default or authorization to change the P4 runs. A virgin-only
locking implementation remains an alternative, not an interchangeable shortcut.

## Direction and branch specification required for either option

Under the present constant R=mu*p_N, p_N is prescribed and angle-independent.
A maximization of admissible in-plane shear over theta uses the physical
stress/backstress, not auxiliary T alone. Introducing angle-dependent normal
friction from the mechanical pressure would change the current law and is
outside a direction-only extension. Principal directions of a symmetric stress
are axes: eigenvector sign is not a physical branch. Store directors modulo pi;
normalize the (t,n,S,w,a/k) sign convention so reversing S reverses signed
quantities and preserves tau, storage and absolute slip.

With constant angle-independent R and isotropic AT1 storage, conjugate planes
can give S2=-S1 and w2=-w1: the same wS, stress, storage and Y. A director label
alone then does not select a distinct phase-growth path. A geometric plane rule
or additional directional coupling must be declared rather than inferred from
that relabeling.

Conjugate shear planes require a declared branch selector. A rotation-based
selector may use the accepted native displacement-gradient rotation only to
choose between admissible branches; it is not objective transport of history.
At negligible rotation/eigenvalue separation use a declared deterministic
continuation toward an established nearby direction, then a fixed physical-axis
fallback. Quantify the ambiguity instead of discarding one branch. Tie handling,
spatial regularization (if any), locking threshold and whether theta is solved
jointly or selected externally are modeling choices still requiring selection.

## Discrete and lifecycle contract

Intact material ahead of the seed is old bulk material, already carrying T
from mechanical loading; activation must preserve that T. It is not born with
zero stress. Virgin A_c/k/L are zero only where their actual prior histories
are zero. Pre-existing diffuse seed material keeps the prescribed seed direction
even while initially unloaded. The orientation used to evaluate intact driving
must be defined before damage, separately from the event that locks a direction.
No numerical phase floor is an acceptable substitute for declaring that event.

Store orientation/locking status and any new tensor components with native
CellId/quadrature ownership and checkpoint them. All Newton trials start from
the same accepted T, A_c/K/L (or a/k/L with locked S), direction and phase obstacle.
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

Rejected steps restore direction and every history/cache together. Retry is
deterministic and publishes once after outer acceptance. Restart must reproduce
physical stress and energy, not merely the orientation coefficients. Existing
DG/native-FE T representation and independent source coverage are retained.

A fixed-direction checkpoint identity control may map old a to A_c=aS0 while
preserving T, physical phase, k/L, stress and storage. Enabling free direction
from an already loaded P4 checkpoint would be a declared model transition, not
a continuation of the same problem. The directional comparison should normally
start freshly from matched physical initialization; no automatic checkpoint
conversion is authorized here.

Minimum bounded P5 checks: fixed-direction identity, rigid in-plane basis
covariance, director/sign equivalence, one non-proportional memory rotation test,
local stress/energy/tangent checks, intact degeneracy/branch ties, then one short
comparison against P4 with unchanged load/mesh/time sequence. Exercise rejection,
retry and restart. Report any branching; do not silently turn this into a
network/geometry reconstruction task. No P5 run begins in the current task.
