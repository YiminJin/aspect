# One bounded mesh-orientation control — blocked before onset

20 October 2026; entry f1f9f29cf40c32d1ddcea000b724375b64853532 plus retained
local work. The [pre-run contract](BULK_ORIENTATION_SELECTION.md) selects one
45° near-tip control of the existing wide specimen. **The valid rotated mesh
advances to .105 s, then the unchanged source-publication guard stops the
.1075 s trial. It does not reach flank activation. Neither the broad-lobe
orientation nor the first-growing-node orientation is qualified.** This is an
infrastructure limitation, not evidence for or against physical grid bias.

## What was held fixed and what was qualified

The physical domain remains [0,1.5]×[0,1] m. The same 5000-point continuum
profile is sampled freshly at vertices about (.5,.5); no accepted phase or QP
history is transferred. The seed, loading, all four physical velocity boundaries,
global shear tensor Sxy=.5, G=1e6 Pa, eta=1e8 Pa s, ell=.15625 m, cohesive/
friction laws, normal compression, unlimited DGQ2, Gauss8 mechanics/fracture,
Gauss3 source, dt=.0025 s and numerical/publication tolerances are unchanged.
Requested end time was .145 s, shortly after reference onset .1375 s.

The 800 core cells are exactly the reference .03125×.015625 m rectangles
rotated 45° about the physical endpoint. The core is the rotation of
|u|,|v|<=.3125 m, containing the entire ideal cap radius .30882159 m. A
conforming 24-ring annulus reaches the original rectangular boundary: 3680
cells in total, versus reference 3072. All cells stay fixed, with no hanging
nodes or runtime refinement. Local core resolution is matched; far-field cell
shapes/resolution differ. Even a successful test would not be a full spatial
convergence result or an identical far-field mechanics discretization.

The existing zero-function topography option, with height identically zero,
selects native MappingQ1. Default MappingCartesian is invalid for this mesh;
no production mapping selector was changed. An identity-map control passes,
with identical initial physical history/phase. Its first-step history difference
from the saved wide control is at most 3.46e-12 in stored values (mapping/
assembly roundoff); full solution difference is 1.05e-11. Both inputs parse.

Mesh qualification independently checks every bilinear cell's determinant at
all four reference corners. Since that determinant is affine, positive corners
prove positivity throughout each cell. Minimum is **2.67275e-5 m²**; native
quadrature gives minimum 2.68482e-5 m² and area **1.5 m²** within 1.1e-15.
Physical affine-gradient reproduction errs by at most 1.82e-12 against the
expected constant gradient (1,2).

An earlier cubic-twist transition had inverted cells and stopped at
initialization on the existing positive-weight check. Its source, outputs and
log remain evidence. Annulus import then exposed ulp-scale unsnapped outer
vertices; preserving the exact final ring coordinates corrected that mesh-file
construction error. Neither attempt advanced physical time. No physical
parameter or acceptance threshold was tuned.

The Cartesian normal-column observer cannot be used on rotated cells. Only the
isolated benchmark DSO replaces it by the same positive-weight moment identity
over original bulk QPs, with explicit area coverage. These are labeled **area
moments**, not surface normalization/history. No constitutive point is moved.
The original fixture DSO, its column coverage checks, core solver and BP3
binaries remain unchanged.

## Why the source check stops the comparison

For an affine cell, the physical gradient of a reference Q2 polynomial belongs
to the complete DGQ2 space, including mixed modes. Constant rotation does not
change that property. For a non-affine bilinear cell,

    grad_x v = J(xi)^(-T) grad_xi v.

The varying inverse Jacobian generally makes this a rational function, outside
DGQ2. Retaining all nine coefficients does not make it exactly representable.
The native residual/local update uses this actual gradient; publication tests
the projected DG tensor against that independent native update at Gauss8 QPs.

The benchmark independently projects **every one of nine scalar Q2 basis
gradients**, using both source3 and source8. Scalar basis ordering and linear
algebra are independently repeated in NumPy.

| Region/control | Maximum basis-gradient projection discrepancy [m⁻¹] |
|---|---:|
| 800 rigidly rotated affine cells, native source3 | 1.03e-12 |
| 2880 non-affine transition cells, native source3 | 88.7713375820 |
| Same transition maximum, native source8 | 88.7713375820 |
| Independent source3, worst cell 2873_0: | 88.7713375820 |

The independent worst-cell discrepancy is 5.993% of its maximum tested basis
gradient, **not a measured physical stress error**. Source3/source8 coefficients
differ by only 7.19e-11 m⁻¹ there, relative to gradients up to 1481.28 m⁻¹.
The independent affine control has relative error 1.38e-14.

This is representation error, not missing integration order: the bilinear-map
weighted Q2 mass integrand has degree at most five per reference coordinate;
in the source RHS, det(J) cancels the gradient denominator. Gauss3 already
integrates these polynomial products. Higher source quadrature cannot supply
the absent gradient remainder. No epsilon, tolerance or source rule was changed.

![Mesh and projection-capacity diagnostic](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_orientation/projection-blocker.png)

*Color shows a synthetic basis-gradient projection discrepancy, not physical
stress or damage. The rigid core passes; the non-affine annulus does not.*

The actual native run accepts 42 steps through **.105 s**, with 42 publications.
Maximum accepted mechanical residual is 3.69e-15, phase KKT is zero and
source-check error is 5.42e-14 (limit 1e-12). At step 43, **.1075 s**, Newton
converges mechanically to 1.78e-18, but
`accepted.projection_error<=1e-12` fails before publishing history or a pending
commit. No step-43 accepted record exists. The unmodified exception does not
print its numeric error: the synthetic table is not substituted for that value.
Valid prior accepted checkpoints are retained; no cutback or restart claim is
added for this new mesh.

The timing is consistent with first departure from homogeneous sticking.
Before yield, the analytic Maxwell recurrence is
Txy+=beta(Txy_n+G dt gamma_dot), beta=(1+dt G/eta)^(-1). At .100 s it predicts
899.763793306 Pa. The exported full plane-strain tensor differs by at most
4.43e-8 Pa; bulk energy is .607181162808693 versus analytic .607181162808673
J/m. At .1075 s the affine predictor first exceeds the 1000 Pa resistance.
The resulting heterogeneous native gradient need not fit DGQ2 in the annulus.
The analytic and all-basis controls separate this limitation from a wrong
physical rotation, loading or Maxwell coefficient.

## Valid physical-coordinate comparisons

Both accepted trajectories have q=Y=0 through the last common time .105 s;
all rotated accepted phase vectors are **bitwise unchanged** from initialization,
with 0<=phi<=.6. There is no artificial unloaded relaxation or growing cap node.
Initialization error is measured independently against the same tabulated
continuum prescription, using Gauss24 diagnostic integration:

| Relative L2 representation error | Wide reference | Rotated |
|---|---:|---:|
| Whole domain | .339721% | .565614% |
| Forward cap window x>.5, r<.35 m | .749854% | .736098% |

The whole-domain increase includes the different annulus representation of the
straight band; it is not subsequent evolution. Initial fracture storage is
9.43534995312 versus 9.42341195240 J/m. Core cap representation is comparable,
but the two initial FE fields/energies are not asserted identical.

Physical comparisons use one common positive radial kernel of radius
a=.0625 m, K_z=(1-|x-z|²/a²)^3_+, normalized by its physical integrated weight:

    qbar=integral K q / integral K,   Ybar=integral K Y / integral K,
    Fbar=Kf integral[K+2 ell² grad(phi).grad(K)] / integral K,
    Rbar=Fbar-Ybar.

Original accepted q and Y samples remain at original QPs. No averaged history
is return-mapped and Y is never calculated from averaged q. The common kernel
is a diagnostic functional, distinct from each mesh's native admissible nodal
growth condition. Independent native weak gradients agree to 5.06e-15 in
integrated units; the assessor also retains the established normalized 1e-10
verification screen.

At r=.288111 m the initial common-kernel fracture actions span 19.1473–23.8301
Pa on the reference and 19.8635–23.8300 Pa on the rotated mesh; maximum angular
difference is 1.47934 Pa. Kernel-averaged phase differs by at most .00069693.
This variation is already present before mechanics. At 44 fixed diagnostic
arc points, Gauss8/16 reintegration of the Q1 field/fracture action differs by
at most .00702 Pa; extra quadrature never re-evaluates a constitutive history.
That check is sampled diagnostic uncertainty, not a global quadrature bound.

Native KKT retains its existing positive cell-area/4 nodal scale. On this
bilinear mesh the exact integral of N_i divided by that scale ranges
.94761–1.02821. Scaling does not change the KKT sign condition, but the two
normalizations are not identified. Native KKT and the common physical kernel
are reported separately; a nonzero-growth tolerance assessment is not reached.

![Equal-scale physical overlays](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_orientation/physical-overlays.png)

*Rows show the same physical seed on the aligned and rotated meshes. White
contours show actual phase geometry. Driving and damage increment are exactly
zero at the last common accepted time; the tiny positive increment display
range is only a plotting scale, not observed growth. The .1075 s failed trial
is not displayed as accepted evolution.*

![Common physical-kernel arcs](../../../benchmarks/reconstructed_fault/particle_free/evidence/bulk_orientation/common-kernel.png)

## Conclusion and smallest next task

**Broad driving lobes tied to physical loading:** undecided by this control;
the guard stops before nonzero accepted Y. **First-growing locations following
the grid:** also undecided; no cap node grows. Initialization curvature changes
under orientation, but it does not establish the orientation of later growth.
The prior wide-case lobes/onset remain evidence, with their original limitations.

Do not change the seed on this basis. The next prerequisite is a bounded
non-affine source-history qualification: explicitly retain the native-gradient
projection remainder in the fixed-mesh history (or choose another exact native
representation), including its once-only Maxwell recurrence, tangent, rollback
and checkpoint semantics. Merely using source8 or loosening the comparison
would not preserve the current equations. That representation extension is
not implemented in this task; it should precede repeating this one physical
orientation comparison. No general refactor, new physical law or mesh sweep
is warranted by the present evidence.

For constant coefficients the required remainder recurrence is explicitly
R+=beta[R_n+2G(Id-P_h) dE], with reconstruction T_phys=T_DG+R and the same
source3 projector P_h. A future implementation needs a point-evaluable owner
(the retained native FE history is a possible source), not only more quadrature
samples. Its ownership and independent checks must be qualified before adoption.

The existing wide physical-energy screen failure **2.2328%**, failed temporal
qualification and `strict_reference_match_passed=false` remain unchanged.
BP3 source/artifacts and previous angular evidence are preserved. Commands,
source/input identities, valid and failed logs, common-kernel CSVs and selected
original history/geometry files are in `evidence/bulk_orientation/`; replay
instructions are in `bulk_cohesive/ORIENTATION_README.md`.
