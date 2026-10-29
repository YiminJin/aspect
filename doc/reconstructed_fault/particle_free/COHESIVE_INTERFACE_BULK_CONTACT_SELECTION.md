# Fixed-path bulk/contact diagnostic

User-selected task after the accepted causal-extension milestone. Baseline
f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved work. Production/BP3,
cohesive–RSF calibration, p0=10000 Pa, R=.75 m, persistence, fracture inventory
and all solver/acceptance tolerances remain unchanged.

First diagnose the saved work96 evolved geometry and physical Q1 relaxed-jump
memory at .445 s. All approximations receive exactly the same scalar field
along the same curved Γ and the same affine Maxwell strain coefficient.
Each independently solves its native mixed bulk response. This is a frozen
operator comparison, never checkpoint conversion or accepted history transfer.

Separate numerical lifting width from bulk spacing. Compare the retained
64²/96² Q2 and 64² Q3 results, then hold lifting width and normal-test width
fixed at .75 m. Use common Q1 tangential test functions and common normal
bubbles; compare physical integrated loads, not differing mass-normalized node
labels. Examine whole-domain and fault-near storage, jump-induced storage,
weak and strong incompressibility, current normal reactions and raw sided
tractions. The large homogeneous background must not hide perturbation errors.

Add an independently fitted-grid approximation of the actual saved quartic
graph, with all Γ knots respected, exact quartic geometry mapping, and the
same physical analytical jump. Resolve interface and lifting support on mesh
lines. Use two fitted resolutions; higher degree alone is not an accuracy
reference. The fitted diagnostic may depend on Γ, so it must not be advertised
as preserving arbitrary old Maxwell history when Γ changes.

Bound the first matrix to eight frozen controls: Q2 64/96 with width .75,
Q3 64 with width .75, a 96-cell fixed .1875-width control, fitted Q2 64/96,
and the two original mesh-scaled widths. The old 64/Q3 trajectories have
slightly different evolved geometry: rebuild at the common work96 geometry
rather than calling them identical-field controls. One fitted Q3 control is
allowed only if the fitted reference still needs qualification.
One quadrature/cut-placement check is allowed for a demonstrated discrepancy.
No broad mesh/parameter search. Native kernel timeout 300 s, task raw-data
budget 1.5 GiB, preserve every failure and original error.

The eight frozen controls identify support-edge aliasing. An independent 1D
pressure-free FE/projection calculation predicts the old-region load change
to .047 Pa (0.037%). Add one declared off-grid support control, width .6875 m
on the 96 mesh, with unchanged .75 m normal test. This deliberately places
support edges halfway through cells; it tests robustness of the .75 m
numerical correction rather than optimizing a width for physical growth.
The fitted-Q3 check confirms the separately refined fitted reference; no
pressure-space modification is warranted before these measurements.
One final Q3/64 reconstruction at the original .1875 m lifting width completes
the direct Q2/Q3 comparison on identical physical geometry (the archived Q3
trajectory has a different final geometry). Reuse its operators for the cheap
99-step fixed-path check. No further mesh/width/quadrature sweep follows.

The common imported .42025 s history is not surface-equilibrated after bulk
reconstruction (measured free residuals, not just missing tied reactions).
Keep these controls diagnostic and their initialization energy/force changes
explicit. Complete four cheap fresh fixed-path trajectories using existing
wide64/96 and fitted64/96 operators, with the same prescribed initial seed,
zero Maxwell jump memory and homogeneous initial physical stress. Verify the
initial bounded tied multiplier and true weak equilibrium. This removes
imported-history relaxation as an explanation of the evolution comparison;
it adds no mesh, law, loading or operator search. Preserve the original coarse
history and every diagnostic result. Stop after this focused verification.

Frozen practical targets before evolution: <2% common tangential work/load
and normal perturbation changes on meaningful fixed regions; <2% jump-induced
bulk storage change. Report absolute Pa/N/J per thickness as well as relative
values; near-zero test loads use absolute comparisons. Incompressibility is
reported physically, with weak residual separated from pointwise divergence.
Keep existing Gauss6/Gauss8 independent integration, true residual, full-pressure
FD and energy checks. No tolerance relaxation.

If a correction or adequate approximation is demonstrated, use the same fixed
path for a short loading/hold comparison .42025–.445 s (load ends .44 s),
Delta_t=.00025 s, retaining .000125 s as the focused time control. For this
diagnostic evolution, reconstruct the declared common incoming physical
surface/Maxwell field independently in each approximation; label it separately
from the original accepted trajectories. Do not silently publish it over them.
Compare growth-relative integrals and a=.05/.5/1 front positions. Retain the
previous <5% growth/slip and one fine surface-cell front criteria. Only repeat
the complete causal workflow if its fixed-background correction supports the
history-preservation contract; fitted-only improvement cannot authorize
arbitrary evolved-memory remapping. Stop and diagnose if these conditions fail.

Cleanup preceding the science is independently logged in
benchmarks/reconstructed_fault/maintenance/disk_cleanup_2026_10_27. Source,
accepted histories, failure evidence and BP3 remain protected. Original ZIPs
were removed only if every member still existed byte-identically elsewhere;
large raw tables are losslessly compressed with SHA256/restore mappings.
