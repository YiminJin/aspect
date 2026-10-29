# Finite interface insertion and localized cohesive growth

**Tied-record insertion passes, and the selected native component specimen shows
both process-zone expansion and advancement of the fully decohered region.**
Later integrated responses meet the declared spatial/temporal targets. The first
tiny activation remains mesh-sensitive. This is a fixed straight candidate in a
finite periodic strip, not a free-path or production/BP3 propagation solver.

[Selection and gates](COHESIVE_INTERFACE_INSERTION_SELECTION.md) ·
[transfer/equations](COHESIVE_INTERFACE_INSERTION_FORMULATION.md) ·
[figures and indexed data](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_insertion/INDEX.md).

## What changed and what was preserved

The optional native FE fixture now owns a **finite** surface, exact zero jump at
its endpoints and outside it, and a variable installed jump subspace. New tied
records inherit the virgin material aging clock; existing Gauss histories retain
their material locations. The enlarged tied space is re-certified at the same accepted state and rate,
before any geometry commit or time advancement. Native geometry and constraint-map rollback, geometry
version checks, deterministic retry, once-only geometry/physical publication and
ASPECT restart are exercised. The mixed bulk FE/pressure space remains fixed,
including its candidate-line pressure traces: simultaneous bulk pressure-space
remapping is explicitly not qualified.

G=1 MPa, eta=100 MPa s, Gc=1 J/m², peak strength=3000 Pa, p0=10000 Pa,
RSF (a,b,mu0)=(.005,.01,.1), V0=5e−6 m/s, Dc=.001 m and the existing radiation
coefficient are unchanged. Bulk/source Gauss3 and independent Gauss6 work checks
are retained. Full current-pressure coupling, plane-strain stress and existing
force/energy tolerances remain intact. The worst retained coupled finite-difference
tangent error is 4.22e−7 relative, below the unchanged 1e−6 criterion. No artificial slip, phase floor, history
reset or additional dissipation is introduced.

The repeated QP·N pressure contraction is cached with its original summation
order. Cached and original trajectories are bitwise identical. The retained
pre-task full-interface executable also reproduces the current full-interface
control bitwise. All 1,162 protected source/header/BP3/prior-front files are
unchanged; the worktree's existing manual changes remain present.

## Insertion evidence

On 32×40 cells in the original 1 m × .125 m strip, compare the interface
[.125,.875] m installed initially against extension from [.25,.75] m at
.10, .18 and .28 s, and from [.3125,.6875] m at .10 s. Thus both the added
length (.25 or .375 m total) and insertion time vary. Existing Gauss layouts are
not remeshed. A no-extension transaction and a separate smaller/larger retained
tied-buffer comparison also pass.

At .28 s the incoming state has partial cohesive damage, peak slip **.08491 mm**
and peak weak shear **2995.60 Pa**. Insertion enlarges the finite jump subspace
from 15 to 23 interior nodes. All physical bulk stress components before/after
are bitwise identical. Bulk storage change, fracture inventory change, old-history
error, shear-load change and pressure change are **exactly zero**. No physical
clock advances during insertion. New dimensionless Theta is 1.0014, obtained
from the accepted virgin aging clock, not a steady-state reset. The immediate
zero-time weak birth certificate has residual 7.06e−9 Pa, before publication.

Physical trajectories match the initially installed control bitwise through
.32 s. Smaller/larger tied buffers match through .38 s while separated from
growth. Tied multipliers are nonunique and can differ by 34.81 Pa between
representations; their work is zero and physical stress is identical. This is
not a physical traction-transfer error.

Actual native-manager swap/rollback and retry are bitwise. Old-geometry trials
and duplicate physical publication are rejected. Restart across insertion
reproduces histories, stress, rates, work and publication counts bitwise. Changed
finite configuration is rejected before publication, with all 13 checkpoint
files unchanged. Independent bounded-multiplier feasibility and direct weak
residual checks pass at the original 1e−8 Pa allowance. This is component-owner
lifecycle evidence, not production outer-cutback or MPI qualification.

## Localized specimen and measured response

Use a fresh **8 m × 1 m** periodic strip, interface y=.4 m, installed interval
[1,7] m, mature seed [2.5,5.5] m, top relative displacement rate .001 m/s and
fixed bottom. The background pressure is 10 kPa. The initial seed inventory is
exactly 3 J/m on both meshes; all other histories start fresh. Merely extending
the old shallow specimen would not provide this release: the elastic estimate
with approximately 1300 Pa seed friction gives 4331 Pa tip stress, 2741 Pa remote
stress and .98 J/m² residual-stress release at nominal 2700 Pa loading.

The 64×40 run stops before publishing trial t=2.44 s because its cohesive zone
reaches the declared .25 m candidate-end buffer. Last acceptance is 2.435 s.
The two finer controls stop at the common **2.42 s**, under the same continuing
loading; the originally planned later hold is not exercised. No endpoint is
moved to evade the guard.

| Native case | Δx, Δy (m) | Δt (s) | First admission bracket (s) | New full-decohesion bracket (s) | Added fracture at 2.42 s (J/m) |
|---|---|---|---|---|---|
| 64×40 | .125, .025 | .005 | [1.625,1.630] | [2.380,2.385] | .860805 |
| 128×80 | .0625, .0125 | .005 | [1.515,1.520] | [2.380,2.385] | .857385 |
| 128×80 | .0625, .0125 | .0025 | [1.5175,1.5200] | [2.3800,2.3825] | .851236 |

On the finest trajectory, the left a≥.05 front reaches approximately x=1.493 m;
the right reaches x=6.094 m. Full decohesion advances only on the left. The actual
Q1 jump threshold δ=2Gc/peak gives **x=2.25708 m**, or **.24292 m advance** beyond
the original seed. At least three complete fine elements (.1875 m) satisfy that
threshold everywhere. The new-material nodal rates are nonnegative throughout,
so this continuous threshold reconstruction follows the retained irreversible
law without interpolating age or evaluating histories in another state. Gauss
history fronts and their sampling uncertainty are also exported separately.

The resolved a≥.01 cohesive region spans approximately 1.52 m total—about
24 fine cells—rather than a single activated node. No disconnected a≥.01 region
appears on the installed candidate. The off-midheight geometry and complete
normal-pressure coupling are retained; the left/right difference is not averaged
away. No free direction is inferred from these straight-interface fronts.

Remote weak shear stays below strength. Independent physical-stress evaluation
at all 368,640 Gauss6 points gives remote in-plane maximum shear of **2098.76 Pa
at 2.00 s** and **2643.24 Pa at 2.42 s**, for x<1 or x>7. These are bulk checks
at those saved states, not an all-time/free-path nucleation proof. The smallest
sampled bulk normal pressure is 8721.97 Pa; actual surface-side contact remains
compressive. Physical shear maps are distinct from projected weak tractions and
nonunique intact multipliers.

## Accuracy, work and curvature

At 2.42 s, spatial refinement changes integrated growth by **.399%**, integrated
slip by **.0986%**, common-box growth L2 by **.383%**, and the continuous
full-decohesion front by **3.81 mm**. Timestep halving changes these by **.722%**,
**.188%**, **.801%**, and **5.20 mm**, respectively. These satisfy the selected
late-response targets (5% spatial, 2% temporal); two meshes do not establish an
order. The .11 s spatial shift in first tiny activation remains unresolved and
is not used as a propagation metric. **The full-interval spatial target fails:** integrated growth differs by 65.4%
at 1.8 s (absolute difference 0.0055521 m), and growth L2 still differs
by 13.5% at 2.0 s. The later response is supported; early process-zone development
is not spatially qualified. These failures are retained, not removed by the
choice of comparison endpoint.

At the finest endpoint, bulk/bond storage are 16.63090/.244831 J/m, added fracture
is .851236 J/m, sliding loss is 3.696999 J/m, total physical loss is 4.019245 J/m
and trapezoidal work is 21.74147 J/m. The cumulative **absolute** physical-work
remainder is .0047602 J/m: **.559% of new fracture** and **.129% of sliding loss**.
It approximately halves with Δt. The coarser-time fracture-normalized value,
1.127%, fails the selected 1% target and is retained. BE ledger error remains
below 4.1e−13 J/m; it is not substituted for physical work accuracy.

Independent Gauss6 bulk integration differs by 1.09e−10 Pa in work-adjoint load
and 4.62e−14 J/m in storage. Exact integration of the reconstructed monotonic
piecewise-linear damage profile, explicitly splitting at δ=delta_f, changes the
finest fracture inventory by −2.64e−5 J/m (.0031%) and bond storage by
+2.63e−5 J/m (.0107%). This diagnoses the surface kink integration; it does not
remap RSF Gauss histories or separately qualify all nonlinear friction quadrature.
The native runs retain their original Gauss3 histories and energy ledger.

All admitted reduced curvatures remain positive. The coarse last accepted
unrestricted release eigenvalue is −1.4931e6 Pa/m, but 93.66% of its mode lies
on tied nodes. Its Rayleigh contributions are bulk +2.5547e6, prescribed cohesive
softening −4.3919e6, rate/contact +.3460e6 and pressure −.00182e6 Pa/m. The actually
admissible minimum is **+2.6713e6 Pa/m**. This is inadmissible release through the
retained softening law, not a negative admitted tangent or an insertion defect.
An independent dual solve gives an 85.53 Pa strict intact-strength margin.
Both signs of the negative mode have positive first-order admission gaps
(1532.33 and 105.88 N/m for unit Euclidean mode norm), so they are outside the
critical cone. No law or tolerance was changed in response.

## Cost, limitations and next step

The completed growth runs cost 72.5 s (64×40 to its buffer guard), 142.5 s
(128×80) and 190.9 s (half Δt), sequentially. Fine setup costs 87–89 s. A final 18.25 s coarse replay reproduces the
complete common accepted trajectory bitwise after the birth-certificate factoring.
An earlier uncached fine run was intentionally terminated with its accepted
outputs/checkpoints preserved; the equivalent contraction cache removed that
avoidable runtime burden. Fine runs contain a final rounding-sized timestep
(~3–4e−14 s) to reach the exact requested end time; its damage increment is below
5e−13 and is retained in the raw history rather than hidden.

The evidence supports finite tied-record insertion and **bounded straight
cohesive propagation in this specimen**, not quantitative general rupture speed,
domain independence, MPI, changing the bulk pressure space, arbitrary imported
history conversion or automatic path creation. The growth comparison uses a
finite candidate installed initially; dynamic candidate extension is qualified
separately in the loaded-state tests.

Keep phi provisional geometric metadata. Surface a owns compliance, progressive
contact activation and the single fracture inventory Gc∫a; Gamma owns material
orientation/topology and the jump is the actual displacement discontinuity.
Future virgin paths must be selected by maximum-shear intact admission with an
explicit symmetry preference, closed contact and energy-accounted new tied
records. Existing damaged normals/history must not rotate or reset. Pure tied insertion
requires a feasible unchanged-state certificate: an already overloaded new path
needs event-resolved or jointly energy-accounted admission, not a free history reset. Recommend a bounded
path-admission/topology design and one non-collinear transfer control before
free-path propagation; do not add a second phi fracture reservoir. Production,
BP3 and the earlier unresolved findings remain unchanged.
