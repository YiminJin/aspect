# Aligned-wide cap damage: one uniform refinement

21 October 2026 closeout; selected 20 October, entry f1f9f29cf plus preserved local work.
[Selection](BULK_SPATIAL_SELECTION.md) and
[inputs/checks/replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_spatial/README.md).

**The spatial changes are not small enough for quantitative cap-onset or
growth predictions.** Loading-driven off-axis activation and the broad driving
directions persist, but onset advances seven timesteps and the final cap-growth
L2 difference is 35.86%. Two meshes establish no convergence order, and neither
mesh is declared the accuracy reference.

## Same physical problem, fresh initialization

Reuse the existing aligned 48x64 trajectory and run one fresh 96x128 trajectory
on the same [0,1.5]x[0,1] m domain. Both have **\(\Delta t=.00125\) s**, 128 accepted
steps to .16 s, and bitwise-identical actual timestep sequences. The prescribed
shear rate is .018 s^-1 on (.05,.15], with fixed-displacement holds before and
after. The length scale .15625 m, continuum seed/profile table (peak .6,
original endpoint (.5,.5), compact radius .30882159 m), homogeneous material law,
unlimited DGQ2 plus native FE source complement, ordinary FE phase, Gauss8/source3,
irreversibility, constitutive update and all solver tolerances are unchanged.
There is no transport, runtime mesh change or state transfer. The same 3x3
stationary diagnostic-particle layout per cell remains; domains/CPDI are disabled.
All parsed parameter values agree except mesh repetitions, diagnostic DSO and
output directory. The native executable is unchanged (SHA256 31ad1e59...b5a).

Each mesh samples the same continuum prescription at its own vertices, with
zero DG/local/native-FE histories. Shared vertex values agree bitwise; independent
nodal seed errors are <=7.92e-16. Both initial holds have **zero phase drift** and
native phase KKT=0. Initial Gauss24/32 diagnostic errors agree within 2.5e-10 in
absolute relative-L2 units; these reference integrations do not change native
quadrature. The common-point initial cap L2 difference is 4.63791e-4 m. Initial
fracture storage differs by .0121112308 J/m; it is reported separately, never
counted as fracture growth. No mesh is relaxed into a different initial problem.

## Common measures and compact comparison

Use each trajectory's own increment dphi=phi(t)-phi(0). Cap is x>.5 and
r<.35 m about (.5,.5), split into lower (y<.5) and upper (y>.5) flanks.
Both physical Q1 increments and gradients are evaluated on the common fine
Gauss8 rule. Growth L2 and Linf differences use fine **growth**, not the initial
seed, for normalization. Peak differences compare actual native peaks; first
nodes are not matched. Added fracture storage integrates
Kf[dphi+ell^2(|grad phi(t)|^2-|grad phi(0)|^2)] with the same physical measure.

Values at .15 s remain unchanged through the hold to .16 s. L/U below means
lower/upper; both are shown even when they coincide at the displayed precision.
The relative-difference column uses the refined quantity as its scale.

| Measure | 48x64 | 96x128 | Difference / refined scale |
|---|---:|---:|---:|
| \(\Delta x,\Delta y\) [m] | .03125, .015625 | .015625, .0078125 | both halved |
| Initial seed L2 error, whole / cap | .33972% / .74985% | .085286% / .188486% | initialization only |
| Initial fracture storage [J/m] | 9.435349953 | 9.447461184 | +.0121112308 J/m |
| Cap onset bracket L/U [s] | (.13625,.1375] / same | (.1275,.12875] / same | 7 \(\Delta t\) earlier |
| Peak dphi L/U [1] | .00167133 / .00167133 | .00197972 / .00197972 | 15.58% / 15.58% |
| Cap-flank growth L2 L/U [m] | 7.58726e-5 / same | 1.08077e-4 / same | **field difference 35.86% / 35.86%** |
| Integrated cap growth L/U [m2] | 7.21074e-6 / same | 1.09559e-5 / same | 34.18% / 34.18% |
| Added cap fracture storage L/U [J/m] | 1.56854e-4 / same | 2.24468e-4 / same | 30.12% / 30.12% |
| Added whole-domain fracture storage [J/m] | 5.57555e-4 | 7.10026e-4 | 21.47% |
| Angular Y peaks at outer two radii L/U | -43 / +43 degrees | -43 / +43 degrees | unchanged on 1-degree samples |
| Angular Y L2 difference at .15 s, three radii L/U | — | — | 2.87, 2.74, 2.29% / same |

The field L2 difference is the norm of the difference, not the difference of
the two scalar norms. At .13 and .135 s only the refined cap grows; its
growth-normalized differences are 100%. At .1375, .14, .145 and .15 s the
per-flank L2 differences are **95.46%, 78.57%, 49.25%, 35.86%**. Final field
Linf difference is .000877495, or 44.32% of refined peak growth. Thus the
disagreement includes shape, not only a shifted first activation or peak value.
Neither onset nor late amplitude passes the predeclared practical 5% target.
No mirror registration is applied; original branch findings remain.

Driving is Y=h'(phi)q^2/(2G), evaluated from each state's **original native
QP histories before averaging**. Apply the same physical kernel
K=(1-rho^2/a^2)^3_+, a=.0625 m, normalized by actual positive JxW, on arcs
r=.225,.2881107643,.2985152059 m and angles -90 to +90 degrees. No local
history is interpolated into another constitutive state. Native quadrature
point counts differ, but do not determine the average. Constant-kernel integral
errors relative to pi a^2/4 are <=2.91e-6 and <=1.30e-7 respectively.

The maximum angular-Y L2 difference over positive common-time curves is 5.92%
at .1275 s; at .135 s it is 5.27%, and over the declared established-growth
times (.14 onward) it is <=4.18%. At .15 s the three values are in the table.
Outer lobe directions stay at +/-43 degrees; the inner-radius peak changes
from +/-45 to +/-44 degrees. This supports loading-oriented broad lobes,
with modest amplitude changes, while the damage response is much more sensitive.
The pair does not isolate mechanical approximation from phase-force/obstacle
discretization and initialization representation; it cannot attribute the
growth discrepancy to just one of these effects.

![Full specimen on common scales](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_spatial/spatial-domain.png)

*The same seed and shear history were imposed. Colours show the represented
initial phase, actual damage increment and calculated driving at .15 s.
Dashed white/solid cyan are initial/evolved phase contours; magenta marks the
cap close-up. Damage outside the cap, including the left-boundary patches,
is retained. The first refined global damage is at .1225 s near x=0,
y=.796875 m; it is distinct from cap onset.*

![Both cap flanks on identical scales](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_spatial/spatial-fields.png)

*Mesh lines, phase contours and fields use common physical axes and colour
scales. Red circles mark each mesh's own first-growing cap nodes, not required
to coincide; the cross marks the original band endpoint. Native-QP Y is shown
as constant midpoint bins for display, with no history interpolation. These
display-bin areas are not quadrature weights. The cap colour scale is the same
as the full-domain figure, so larger boundary increments are not hidden.*

![Growth and same-kernel angular driving](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_spatial/spatial-response.png)

*Lower/upper curves are independently calculated and overlap at this scale.
Growth subtracts each mesh's own initial field. Solid/dotted Y curves are .15
and .135 s, respectively. Lines connect selected common times; onset brackets
use every accepted step. Similar broad driving directions do not make cap
onset or growth amplitudes spatially resolved, nor demonstrate centerline
advance or propagation.*

## Energy/history checks and the one diagnostic repair

The native source/history, physical bounds, irreversibility and once-only
publication checks pass on both meshes. Refined accepted maxima are mechanical
residual 7.49e-10 <=1e-9, phase KKT 7.92e-8 <=1e-7, source discrepancy
7.33e-15 <=1e-12, and exact BE ledger error 2.31e-13 J/m. The native tangent
action check is 2.74e-9 <=1e-6. All 128 refined publications occur once.
Independent sampling matches dense q bitwise, phi to 2.23e-16, and Y to
3.52e-16 relative. Full physical stress includes zz and both symmetric xy
contributions; independent dense bulk-storage errors are <=5.62e-14 J/m.
Fracture/cohesive/friction and traction/phase/fracture BE terms are checked
from q/phase every step. Bulk viscous rates use native per-step bulk storage;
independent full-stress storage checks cover initial/stop/final snapshots.
The affine Maxwell prefix additionally checks both meshes through .1 s, with
bulk-storage error <=2.71e-12 J/m. No tolerance was relaxed.

Endpoint work is 2.302686796 / 2.303277906 J/m; trapezoidal work is
2.276633442 / 2.277212555 J/m. Storage changes are 1.892002788 / 1.893647396
J/m, and physical dissipation .384673245 / .383607667 J/m. Absolute per-step
trapezoidal balance defects are .00187098% / .00186691% of event work.
The historical endpoint screens still **fail at 1.12958% / 1.12982%**; the
original 2.23279% coarse-timestep failure and the separate temporal study
remain evidence. Exact BE closure is not substituted for physical-energy
accuracy. Fine-mesh temporal uncertainty is not separately qualified here.
The final hold has zero further phase growth on both meshes, with retained
stress and viscous relaxation, not removal of stress.

The first refined invocation stops at initialization in a read-only area
observer: sequential double addition of 786432 positive quadrature weights
misses 1.5 m2 by -1.6698e-12, outside the original 1e-12 check. Independent
accurate summation gives 1.5 exactly. Only this diagnostic accumulator changes
to extended precision, in a separate DSO; the tolerance, JxW, constitutive
integration and normalization remain unchanged. Failed output/log and old DSO
are retained. A fresh retry completes in 3272.18 s; there is one evolving
refined trajectory, not a parameter search. Native ownership, checkpoint and
restart implementation are unchanged; prior lifecycle qualification is
preserved rather than expanded by new restart/MPI runs.

**Interpretation:** the physical loading produces off-axis damage on both
freshly initialized meshes, without unloaded cap relaxation, and broad driving
directions are similar. Quantitative cap onset, shape, integrated growth and
fracture increments are not sufficiently stable under this refinement.
Small whole-domain work differences do not qualify these local quantities.
Keep the model and seed unchanged; this comparison does not justify a seed
correction or a claim of spatial convergence/propagation readiness. Previous
orientation/boundary findings and `strict_reference_match_passed=false` remain.
BP3/defaults are protected. Stop here; no extra mesh or timestep case is selected.

Machine-readable [assessment](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_spatial/assessment.json),
[initial reference](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_spatial/initial-reference.json),
and [interpretation](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_spatial/interpretation.json).
The [preservation record](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_spatial/preservation.json)
checks 634 entry-pinned files: all except the six intended source/documentation
edits are unchanged, including the protected BP3 files, native core and old DSOs.
