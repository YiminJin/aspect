# PF6a bounded fixed-geometry spatial diagnostic — 16 October 2026

**The cancellation error is reproduced by the independent mechanical reference
and decreases under normal mechanical refinement.** Matched DG and accepted FE
history give the same result in this fixture. Retain unlimited DGQ2 and the
current constitutive equations. The 128-cell event remains spatially sensitive;
this diagnostic does not qualify propagation or complete general PF5 transfers.

## Scope, state and implementation

Root `aspect-pf-rsf-particle-free`, branch `pf-rsf-particle-free`, revision
`a8707307008401ac9cff702a369e2e2a611fa0f7` plus the preserved incoming work.
[Formulation](PF6A_SPATIAL_FORMULATION.md) was written before the measurements;
`frozen-screen.json` records the 32/64 result and decision before the only new
normal level, 128, and the conditional short event. Serial affine 2×N cells,
fixed reference surface, homogeneous matched material, no bulk/surface advection.

This task adds a benchmark observer, preparation/assessment/plot scripts and
one benchmark CMake target. **No production source, physical law, default,
limiter setting, solver tolerance or acceptance rule changed.** The reused
PF6a native executable SHA256 is
`c994b6b809bde83f456ebdf2d0c573ea2ff8dbf69427add898982a3b573a9693`.
Each run pins its executable, observer DSO, source hashes/patch, exact observer
source, input, clock and resource record. Its cached banner is not the source
revision. Preliminary compiler and assessment failures are retained.

Common snapshots come from the retained quadrature-resolved 2×32 PF6a run
(`pf6a-final-32-4-q12`): final physical phase/T/Z at step 72, the same phase
with incoming histories from step 71, and common-entry phase/T/Z at step 68.
All x/y Q2 tensor coefficients are imported; no scalar-only history is substituted.
The equivalent continuous Q2 accepted FE displacement is computed independently
from every saved velocity, using
`d_R(k)=beta*d_R(k-1)+(kappa/G)*u(k)`. Its full plane-strain tensor `2G D3(e(d_R))`
is checked against actual DG history at nine unisolvent Gauss3×Gauss3 points per
cell. `common-verified/preparation.json` records the check; its text data are
byte-identical to the original `common/common.txt` used by the runs. This
compatible zero-start, constant-coefficient case does not define a general
DG-to-FE or checkpoint conversion.

## Identical physical fields versus resolved mechanics

First, identical physical fields are evaluated at the same 18,432 physical
points on every mesh and backend. Stress, energy and phase gradient pulled back
to the same 32-normal-cell Q1 test space agree within the frozen 1e-9 scaled
screen (stress scale 1000 Pa). This is solely an evaluation/representation test.
The cancellation condition `(||Txy||+||Z chi||)/||tau_xy||` is **113.066** for
this identical final field, independent of the evaluation mesh.

Second, the physical phase from step 72 and incoming T/Z from step 71 are held
fixed and **native mechanics is re-solved** at t=.5625 s, dt=1/128 s. Trials
leave the incoming histories unchanged and publish nothing. The native Maxwell
and cohesive endpoint equations, normalization, surface loading and rate
convention are retained. In the verified uniform-shear reduction, with P the
cellwise L2 projection onto normal P1 derivatives of Q2 velocity,

```
a = beta Tn; beta = exp(-dt/100); kappa = -1e8 expm1(-dt/100)
Te = a + kappa u_y; Ze = beta Zn + kappa V; chi = h(phi)/I
mean_tau = (integral a + kappa U - Ze integral chi)/L
physical_tau = mean_tau + (1-P)a - Ze (1-P)chi
```

The retained inexpensive `propagation/localization_1d.py::Reference`, with
`normal_resolution.py` derivative/moment tools, independently solves the
mechanical/contact equations at the supplied fields. It bypasses historical
initialization and trajectories. Here `(1-P)a` is roundoff: the nonpolynomial
localization remainder remains after cancellation even though independent DGQ2
history retains every mode. This is a mechanical approximation error, not an
error established by re-evaluating the same final histories.

The constant-stress continuum control uses the same physical phase, incoming
histories, loading and constitutive law: tau=1293.275208 Pa,
Z=39130.886655 Pa m, bulk/cohesive/fracture energies
0.836280382/5.738059463/15.808320432 J/m. It is a symmetric 1D frozen control,
not a full 2D or evolving exact solution.

| Normal cells | Whole-domain tensor L2 error (Pa) | Localization-weighted tensor RMS (Pa) | Max interface jump (Pa) | Common phase-gradient error, Linf |
|---:|---:|---:|---:|---:|
| 32 | 783.652758 | 1419.285788 | 2276.438380 | 1.422883505 |
| 64 | 200.261460 | 366.311065 | 686.960639 | .095063787 |
| 128 | 50.352509 | 92.356761 | 188.202372 | .006044751 |

Whole tensor norm is `xx²+yy²+(xx+yy)²+2xy²`, including zz and both shear
entries. Whole error is the physical-domain L2 norm (unit area here); localized
RMS uses `integral chi*|error|² / integral chi`. Gradients are energy derivatives
pulled back to identical physical Q1 test functions, not unmatched coefficient
vectors. Continuum-reference gradient columns use x-constant perturbations
(sum over the tangential nodal functions); the matched DG/FE comparison retains
all common 2D test functions. Stress errors decrease by factors 3.91 and 3.98; the common gradient
bias decreases by 14.97 and 15.73. These rates concern this aligned, piecewise
smooth frozen profile only.

Native shear stress agrees with the independent discrete mechanical solution
to <=7.22e-7 Pa. The independent projection identity closes to <1e-6 Pa.
Actual native bulk defect norms are 2.98e-6/2.32e-6/1.72e-6; surface RMS defects
are <=2.19e-11 Pa. These residual norms have their native algebraic units and
are not physical stress errors.

`field.csv` exports Txy, Z chi and actual reconstructed stress separately;
`cell-moments.csv` exports all nine physical Legendre moments; `traces.csv` and
`interface-jumps.csv` retain one-sided traces/jumps. Resolved physical cell
mean ranges are <=6.22e-7 Pa. Maximum normal P2 moments of Txy are <=4.83e-9 Pa,
while those of physical tau are 615.305/184.074/50.581 Pa: the curvature left by
subtracting Z chi is explicit. Z chi is continuous across these interfaces;
the mechanical stress jumps come from the piecewise polynomial T. Cancellation
and spatial resolution are not established from the smoothness of a plot.

Bulk energies are .999881006/.846964344/.836955811 J/m, approaching the
continuum value above; cohesive energies are
5.735775508/5.737910295/5.738050033 J/m. The physical phase and its fracture
energy remain identical in this frozen experiment.

## Retained-history and fracture comparisons

DG and accepted FE history use matching physical T/Z/phase and the same ordinary
FE diagnostic functional. After independently re-solving mechanics, maximum
shear differences are 3.38e-9/3.28e-8/7.05e-8 Pa; common gradient differences
are <=9.77e-10 and scaled energy differences <=6.52e-11. Both pass the predeclared
screens. There is no evidence here that changing the history backend fixes the
normal mechanical projection remainder.

Actual retained FE+CPDI fracture is evaluated **separately** by its native CPDI
assembler. Its energy difference from common ordinary FE fracture is <=1.4e-13
J/m and gradient difference <=7.3e-13 in this aligned, homogeneous, phi(y)
fixture. CPDI domains lie inside cells and the fracture integrands have the
required simple form. This is not general equivalence for distorted domains,
unmatched seeds, transverse phase variation or moving geometry. No
particle-authoritative stress/history comparison was made; those equations and
histories would need matched states and actual particle transfer at common
points. The old independently evolved FE/particle trajectories are not used as
a matched-history reference.

Native normal quadrature 12 versus 24 changes resolved stress by <=1.59e-9 Pa,
common gradient by <=1.51e-12 and scaled energy by <=6.22e-13. Independent
mechanical quadrature 24/48 changes mean stress by <=7.82e-9 Pa and common
gradient by <=7.11e-11; continuum 48/96 energy uncertainty is <=1.1e-14 J/m.
These are far below the spatial differences. Existing transport/source rules
remain unchanged. Frozen diagnostic wall times, including initialization and
large exports, are 7.43/12.24/26.81 s for DG and 7.58/12.59/27.01 s for FE;
they are not isolated solver timings.

## Conditional short native event and uncertainty

Only after the frozen diagnostic identified the localization projection
remainder did the short native coupled event proceed. All meshes start at
.53125 s from **identical imported physical** phi/T/Z at step 68 and advance to
.5625 s with four dt=1/128 steps. Boundary loading is the exact interval mean
of the unchanged linear ramp; the half-step control uses its own interval mean.
The observer imports the entry once, then observes the actual retained
consistent-endpoint/reduced-Newton outer loop. It does not advance phase first
or combine noncommitting observers into an evolution scheme.

The common entry is the coarse-grid physical solution, not an equilibrium in
the enriched fine-grid test spaces. Initial fine-grid phase KKT defects are
6.55 and 5.42 (64/128), versus 2.66e-11 on 32 cells. The common-space functional
still agrees at entry. Consequently the event includes a resolved relaxation
of that imported coarse state; it is not the earlier uninterrupted fine-grid
trajectory or a production restart/AMR transfer. Persistent accumulated slip,
particle H and auxiliary sampling are not imported as matching physical
histories; current constant Coulomb resistance does not depend on those values.
T, physical phase and elastic slip Z/G are matched. This scope limitation is
separate from the previously verified PF6a restart lifecycle.

| Normal cells | Final phase peak | Final Z (Pa m) | Total energy (J/m) | Diagnostic wall time (s) |
|---:|---:|---:|---:|---:|
| 32 | .6203813424 | 39123.098125 | 22.5439769463 | 18.50 |
| 64 | .6468976106 | 39150.568353 | 22.2702765556 | 69.59 |
| 128 | .6511292289 | 39155.995650 | 22.2300922473 | 113.98 |

Relative to the 128-cell event, 32/64 full-tensor stress differences are
784.382/292.106 Pa (relative L2 .436914/.162708), localized differences
1388.811/585.372 Pa, phase L2 differences .0139740/.00150595 and total-energy
differences .3138847/.0401843 J/m. The 128-cell event is a bounded comparison,
not a converged evolving reference; the stress difference remains substantial.

The 128-cell half-step control (eight dt=1/256 steps, 137.30 s) changes whole
stress by .00765845 Pa (relative L2 4.27e-6), localized stress by .0150170 Pa,
phase L2 by 4.92e-6, phase Linf by 1.28e-5, phase peak by 2.42e-6 and energy
by 4.60e-7 J/m. These differences are far below the 64/128 spatial effects.
The common phase-gradient temporal difference is .0001631 versus spatial
.0369262. One half-step comparison bounds uncertainty for this diagnostic; it
is not a new temporal-order qualification.

Every accepted step has zero healing and exactly one native publication (4/4/4
and 8); frozen trials have zero. Native accepted phase KKT <=3.41e-11.
Independent differentiation of the native Q2 velocities verifies the full
Maxwell tensor recurrence at **all nine nodes**, max error 1.57e-10 Pa; the
surface recurrence error is <=1.1e-11 Pa m. Common high-order integration agrees
with native stored energy to 5.26e-13 J/m. Reusing the 1D functional for the
measured near-uniform shear fields gives gradient disagreement <=4.87e-9 and
24/48 quadrature differences <=4.19e-12 (gradient) / 6.04e-14 J/m (energy).
Discarded transverse phase/tensor variation and normal P2 shear defects are
reported explicitly in `final-event.json`; this reduction is not a general
non-proportional 2D reference. Both quadrature and temporal differences are
well below the measured spatial differences, without relaxed tolerances.

## Recommendation, evidence and limits

Keep unlimited DGQ2 and the existing constitutive/normalization equations.
The frozen cancellation is explained by the Q2 mechanical derivative space:
changing compatible history storage or limiting independent history does not
remove the missing localization curvature. Normal refinement reduces this
specific error predictably, and the short event retains measurable phase and
stress sensitivity even at the bounded finer level. Do not treat smoother
phase, native KKT convergence or the 128-cell result as spatial qualification.

A concrete next selected task would establish a matched fixed-geometry evolving
spatial reference, retaining the documented physical initialization/obstacle,
and compare mechanical approximation against the same inexpensive 1D model.
Use the measured localization projection remainder to select the mechanical
resolution question; evaluate an increased mechanical polynomial order only
as a separately selected comparison. Keep history/constitutive equations and
limiter defaults fixed. No further mesh, order, transport or propagation run
is authorized by this recommendation.

[Portable evidence and replay](../../../benchmarks/reconstructed_fault/particle_free/evidence/pf6a_spatial/README.md)
contains common data, independent assessments, the pre-refinement decision,
raw component/moment/interface exports, exact source/input/resource provenance,
failed preliminary logs and standalone plots. The original PF0–PF6a and WENO
evidence is preserved. Existing unchanged-core PF6a unit/operator/lifecycle
results remain evidence for their original scope; they were not relabeled as
new tests here. New checks test the spatial diagnostics and actual short events.

Particles still supply startup H/CPDI/domains, material projection,
association/support, fixed-particle guards, derived samples and persistence.
Their stress does not feed independent history, mechanics or fault loads.
This task does not exercise matching particle-authoritative history, general
CPDI transfer, AMR/MPI, surface transfers, moving geometry, birth, RSF evolution,
coupled transport or propagation. General PF5 completion remains outstanding.
