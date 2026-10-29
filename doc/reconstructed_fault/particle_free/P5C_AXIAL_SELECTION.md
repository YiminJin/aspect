# P5c targeted axial refinement — declared comparison

Selected 22 October 2026, f1f9f29cf40c32d1ddcea000b724375b64853532 plus
preserved local work. Compare the accepted P5c 96×128 half-step trajectories
against one fresh 192×128 trajectory per constitutive mode, four ranks each.
This is an axial sensitivity test, not a convergence-order study.

Keep the continuum seed, zero incoming physical histories, 1.5×1 m body,
released lateral horizontal/retained vertical boundary conditions, material
tuple (G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², cohesion=2000 Pa, ell=.046875 m,
m=4, residual resistance=1000 Pa), Gauss8/source3, unlimited DGQ2 plus native
source complement, physical obstacle and all solver/stability bounds unchanged.
Δx changes from .015625 to .0078125 m; Δy remains .0078125 m.

Both new cases start directly from the same continuum prescription. No coarse
accepted state is interpolated. Native continuation uses Δt=.00125 s through
.11875 s (step95), a same-mesh/same-mode checkpoint, then Δt=.000625 s to
.145 s (step137). The existing native restart hook reduces only the pending
interval, without editing accepted clocks or histories. Shear rate .018/s
from .05 to .140 s, then hold. Expected total wall cost 2–4 hours; runs are
sequential, individually timeout-bounded, with checkpoint rings retained.

Measure each field increment relative to its own represented initialization.
Report continuum representation error separately. Compare onset brackets at
increment>1e-8, exact Q1 phase integrals and common-physical-point growth
differences, peak increments, growing-node coordinates/extents, connected
contours, initially intact/remote activation and added fracture storage. Show
native weak mechanical driving (-g_mech/mass) and fracture resistance
(g_frac/mass) separately. Use a common physical averaging kernel for physical
stress/driving comparisons; do not reconstruct laws from averaged histories.

Check all native mechanics/KKT/source/energy-ledger acceptance, publication
counts, fixed particles and mechanically relaxed stability before interpretation.
No tolerance is changed and no additional spatial gate is imposed. Determine
whether the sign of the radial-versus-fixed reduction survives and report its
changed magnitude, without treating either mode as an accuracy reference.
No prescribed small spatial-error target is inferred from two meshes.

Figures use physical equal-aspect axes, common scales, initial contours and
actual near-cap mesh nodes. Tensor flow angle describes a constitutive
increment, not the geometry of a crack. Preserve P4 failures, boundary
sensitivity, strict_reference_match_passed=false and BP3. Recommend a bounded
loading extension only if supported; do not execute it in this task.
