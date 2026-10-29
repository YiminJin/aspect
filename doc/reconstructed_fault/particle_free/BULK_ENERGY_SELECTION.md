# Bounded aligned-wide temporal / work control

Entry f1f9f29cf plus the retained source-history qualification, 20 October 2026.
Reuse the existing aligned 48x64, [0,1.5]x[0,1] m result through .16 s at
h=.0025 s (`bulk-sensitivity/runs/right-distance`). Run **only** h/2=.00125
and h/4=.000625 s, from the same fresh continuum seed and zero histories.
The final time remains .16 s so all cases include the .05 s loading start,
.15 s loading stop and .01 s fixed-displacement hold. Source-history's later
.145 s control demonstrated roundoff agreement of the original pure-DG and
qualified DG+native-FE representations on this affine mesh; the old .16 s
trajectory is reused without checkpoint conversion. No scientific source,
quadrature (mechanics8/source3), loading, physical parameter, time integrator,
acceptance tolerance, mesh or BP3 change is made.

The only C++ addition is a test-only DSO diagnostic: accepted nodal boundary
reactions from the actual full stress/physical pressure and displacement
increments, with compact QP q histories and full phase exports. At h, the saved
endpoint work determines the generalized reaction Q_n=W_BE,n/dgamma_n during
loading. Independent saved-stress moments cross-check it. During zero increments
work is exactly zero; reaction is not inferred to be zero. The saved histories
will measure the retained stress/reaction on hold. The two new runs export
reactions at every accepted state, including both switching times and holds.

Compute W_BE,n=F_n.du_n and W_tr,n=(F_n+F_(n-1)).du_n/2. Preserve the exact native
BE ledger independently, with physical bulk/cohesive/friction dissipations,
storage changes and each stress/traction/phase/fracture numerical remainder.
Endpoint external-work residual is a numerical BE quantity, not automatically
an error in physical work. Trapezoidal work is a force-displacement diagnostic,
not a replacement integrator. BE physical dissipation still uses endpoint rates.

Preselect cancellation tests: every increment, signed cumulative remainder,
maximum cumulative magnitude, total variation sum|r_n|, and separate initial
hold/loading/final-hold budgets. Normalize by the actual total absolute
trapezoidal work over this prescribed event, not initial fracture storage or a
dimensional floor. Explicitly account for the .05/.15 s switches, measured total
dgamma=.0018 and zero final-hold work despite nonzero stress and dissipation.

Compare common physical times through onset and growth. Cap is x>.5,
r=sqrt((x-.5)^2+(y-.5)^2)<.35 m; first growth threshold remains 1e-12 in nodal
phase. Compare initial fields exactly. Normalize phase differences by **cap
growth delta_phi=phi-phi_initial** on the finest available result: physical
Q1 L2 norm, Linf and peak attenuation/change. Separate the two mirror flanks;
perform no branch registration. Use the established physical radial kernel
radius .0625 m on the identical native QPs for angular driving at r=.225,
.2881107643,.2985152059 m. Average Y after evaluation from each state's q/phi;
never interpolate constitutive histories into a new state.

Record h/h2/h4 differences and observed orders only where differences support
an estimate. As provisional accuracy targets for interpretation (not solver
acceptance changes), inspect 1% physical-work variation/temporal uncertainty,
5% growth-normalized response uncertainty and onset brackets shrinking with
time step. Recommend a precise physical-energy criterion from the measured
budgets, including cancellation control; retain the original 1% endpoint-screen
failure verbatim as historical evidence. A small signed final residual alone
cannot qualify energy or the response. No further timestep, mesh or model run
is authorized by this selection.
