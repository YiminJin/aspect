# Structural-direction continuation: bounded selection before tests

Independent reference only, baseline f1f9f29cf40c32d1ddcea000b724375b64853532
plus preserved local work. Fully coupled pressure and single-direction
asinh RSF/radiation are the candidate. Constant-Coulomb and pressure-lagged
counterexamples remain diagnostics, not gates that the new candidate must pass.

Constitutive hypothesis: a pre-existing fault has a material director supplied
by its geometry/history. Its flow is W=chi v S, S=sym(t tensor n), with signed
v and the actual pressure pn=p-n.tau.n. It need not maximize the force over
all other planes; neither the other-plane yield inequalities nor the angular
stationarity row apply to sliding on an existing fault. This explicitly changes
the previous global-maximizing increment requirement and its min-perpendicular-
pressure interpretation. No numerical phase threshold chooses when to lock.

An explicit formation event on previously unassigned material selects a
direction maximizing the instantaneous kinetic speed at the unchanged incoming
stress/history/state. A declared geometric preference resolves genuine ties;
without one, report symmetry and admit both alternatives for comparison.
The new record thereafter represents an existing plane. A pre-existing diffuse
seed must receive its geometry director at initialization, including its whole
represented support. Do not wait for creep to invent a direction. Direction
selection is not a new phase-growth criterion. No phase is created by this
fixture; selection at a prescribed formation event is a conditional point test.
Secondary fracture through already assigned material needs additional structural
records and a fracture-energy rule, and is outside this single-plane closure.

Keep T/A/K and frictional age Theta unchanged at admission. Their subsequent
Maxwell/flow/aging updates retain their established equations. Geometry is
fixed in the small-strain material chart; passive basis changes rotate tensors
and director together. No finite-rotation/objective-rate claim. Reject loss of
closed contact, invalid state, or an inconsistent record, rather than choosing
a different plane. Preserve physical stress, dissipation and the BE remainder.

Point gate: finite phase .2/.8/.99 and analytic exact maturity, both symmetry
preferences, load/reversal/nonproportional drive, a non-proportional incoming
tensor memory, and competing-plane diagnostics. Use dt=.0125/.00625/.003125 s
over .3 s (load reverses at .15 s). Compare common times; require decreasing
successive stress/slip errors and finer-pair full-tensor error <=2%. Verify
full stress/pressure tangent by FD <=1e-6 relative, force residual <=1e-8 Pa,
energy ledger <=1e-10 relative, basis covariance <=1e-10, immutable trials,
rejection/retry and independent fixture serialization. These are point-fixture
checks, not native lifecycle qualification.

Only after this gate passes: one fixed horizontal band, H=.25 m,
phi(y)=(sqrt(.8)-|y|/(2ell))_+², ell=.046875 m, initialized freshly with zero
tensors and Theta=1. Existing plane n=(0,1) everywhere in its support. Reuse
the mass-lumped P1 strip / exact 1-D mixed equilibrium reduction, with 64/128/256
cells, fixed G=1e6 Pa, eta=1e8 Pa s, Gc=1 J/m², m=4, p0=10000 Pa, and the
previous candidate's a=.005,b=.01,mu0=.1,V0=5e-6 m/s,Dc=.001 m,rho=2670 kg/m³.
No K gradient or other hardening is added. Normal traction and mean pressure
follow mixed equilibrium; fresh aligned histories make pressure constant as a
solution symmetry, not a frozen-pressure approximation.

Impose relative boundary velocity .001 m/s to .25 s, then hold to .30 s.
Keep phase fixed. Shared dt=.00078125 s; halve only on the finest mesh and
allow one further halving if necessary. Require medium/fine traction, slip,
work and rate-participation-width differences <=1%, with decreasing mesh
differences. Separate initial phase/normalization representation errors.
Temporal changes at the finest mesh must be <=.25% for traction/slip/width
and <=.5% for cumulative work (growth/slip measures only after active sliding,
not ratios of roundoff-level early creep). Assess whole active history as well
as endpoint. If a gate fails, report it without further searches or changing
physical parameters/tolerances. Expected cost: seconds for point checks and
at most a few minutes for the vectorized independent band; bounded timeouts.

Report PNGs, compact CSV/JSON, numerical error versus symmetry/bifurcation
versus constitutive failure, and the limits of the 1-D fixed-profile reduction.
Do not begin propagation, native integration, MPI or a new specimen campaign.

Before band execution, define the active comparison window as .15–.30 s,
using common accepted times and normalizing each maximum absolute history
difference by that quantity's maximum absolute reference value in this window.
This avoids division by near-zero early creep or an instantaneous hold rate.
Also report endpoint values and field profiles; the rate width is
(integral |w| dy)^2 / integral w^2 dy, w=chi v in s^-1.

Measured temporal follow-up: the original dt/half pair has 0.3185% slip
change, above 0.25%; the permitted half/quarter pair gives 0.1583%. Match
64 and 128 cells to the already computed dt/2 before the final spatial
comparison. This is the same three meshes/physical experiment, not a parameter
search. Retain all initial results and the literal nondecreasing work-difference
failure (differences <5e-11 relative); report absolute arithmetic bounds instead
of changing tolerances or pretending the strict decreasing-difference gate passed.
