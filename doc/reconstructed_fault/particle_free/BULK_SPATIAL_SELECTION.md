# One aligned-wide spatial comparison

20 October 2026; entry f1f9f29cf plus retained local work. Reuse the completed
48x64 `bulk-energy/runs/half` and add exactly one fresh 96x128 native trajectory.
The same physical domain is [0,1.5]x[0,1] m. Delta x, Delta y are
.03125,.015625 m and .015625,.0078125 m respectively; Delta t=.00125 s in both.
End time .16 s retains the prescribed .05/.15 s loading switches and final hold.
No accepted field or local history is transferred. The native continuum seed,
unloaded initial histories, length scale, bulk law, constraints, unlimited DGQ2
plus native FE source complement, Gauss8/source3 and all checks are unchanged.
Reuse the existing diagnostic DSO and native core without rebuilding either.

Evaluate initial Q1 representations separately against the same independently
sampled continuum seed and against each other at common physical points.
Check nodal prescription, physical bounds and unloaded drift. Later comparisons
use each trajectory's own delta_phi=phi(t)-phi(0), not final-field differences
normalized by the large initial seed. Evaluate both Q1 increments on the fine
mesh quadrature, which also integrates the nested coarse Q1 field exactly for
the required polynomial norms. Cap is x>.5 and r<.35 m about (.5,.5), split
at y=.5 into upper/lower flanks; a centerline of zero area is not a third flank.
Report physical L2 and peak differences, integrated cap growth, added cap and
whole-domain fracture energy, and onset brackets with the existing 1e-12 native
nodal growth threshold. Do not require common first nodes or align mirror
branches. Initial fracture representation changes are not added fracture growth.

At common times .05,.10,.1275,.13,.135,.1375,.14,.145,.15,.1525,.16 s,
retain native phase/KKT conditions and evaluate Y on each original mechanical
quadrature layout before applying the same physical radial kernel:
K=(1-rho^2/a^2)^3_+, a=.0625 m, actual positive JxW normalization.
Use the established arcs r=.225,.2881107643,.2985152059 m and both angular
halves, without interpolating local histories to a different constitutive state.
Near-onset comparisons and undefined zero-growth relative norms are explicit.

For quantitative amplitude interpretation, preselect a practical 5% target for
growth-relative L2, peak, integrated growth, added fracture energy and averaged
Y once both meshes have established measurable growth. Report all earlier
differences too; do not hide a one-mesh-only onset. Report onset intervals and
whether they overlap or differ by more than one Delta t. Broad angular maxima
within 5 degrees is a separate qualitative lobe check, not proof of accurate
first activation or propagation. These targets guide interpretation only; no
native tolerance is changed. Two meshes cannot establish a convergence order.

Preserve exact BE and physical force-displacement ledgers, source consistency,
mechanical/KKT and once-only publication checks. Bound the one serial run by
14400 wall seconds; no additional mesh, timestep or model trajectory is selected.
Keep BP3 and all earlier failures/evidence. Deliver equal-scale initial phase,
damage-increment and driving overlays, one compact table, and a scoped conclusion.

Initialization-only diagnostic correction: the first attempt has no positive
physical step and fails the unchanged 1e-12 area check. Sequential double
addition of 786432 positive native weights gives 1.4999999999983302 m2;
independent math.fsum gives exactly 1.5 and extended accumulation differs by
only -3.21e-17 m2. Correct only that observer's area accumulator, not its
tolerance, weights or constitutive integration. Build a separate spatial DSO
to preserve the old energy DSO; retain the first output/log and restart from
fresh zero histories in `refined-qualified`. No second physical trajectory or
parameter-search comparison is introduced.

Before inspecting refined cap activation: fix the established-growth interpretation times
to .14,.145,.15,.1525,.16 s, following the retained coarse onset. All earlier
common-time differences remain reported. Absence of growth on one mesh at any
of these times does not pass the quantitative growth target. No additional
trajectory or tuning is selected by these interpretation thresholds.
