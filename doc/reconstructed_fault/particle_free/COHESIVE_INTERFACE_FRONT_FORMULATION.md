# Weak intact admission on the fixed candidate surface

Use the [exact-jump mixed mechanics](COHESIVE_INTERFACE_EXACT_FORMULATION.md) and
unchanged [bond/contact law](COHESIVE_INTERFACE_FORMULATION.md). Gauss records own
physical a, Θ, b and jump/memory; Q maps nodal jump to those records, W contains
surface quadrature weights, D contains nodal row weights. Define R=D^-1 Q^T W.
For given accepted state and timestep, mechanics gives nodal shear s and current
normal pressure pn as before. On damaged records use r=q+a Rfriction.

## Correct tied and admission conditions

Let T be the records with a_old=0 that remain intact, and D the other records.
The complete discrete conditions are

    Q_T δ=b_T,
    s−R_D r_D=R_T λ_T,       −τp≤λ_q≤τp.

This is an **existence** condition for bounded multipliers, not an equality of
each multiplier to an arbitrarily projected physical trace. A tied region may
have nonunique multiplier distributions; all do zero work because Q_T δdot=0.
The uniquely reconstructed bulk stress is exported separately. On newly forming
records, the same point return determines a, q and the progressive contact term.

Equivalently, after accounting for damaged-record work, every admissible nodal
direction v must obey

    v^T D (s−R_D r_D) ≤ τp Σ_(q in T) W_q |(Q_T v)_q|.

A nodal peak cutoff is not equivalent. Alternating nodal directions can have
small Qv and violate this inequality even when all nodal loads are below τp.
Conversely, projecting an admissible bounded multiplier field can overshoot τp.
The independent examples retain both counterexamples.

At continuum level the bound follows from the one-sided complementary release
Y0=c0 λ²/2≤Gc. At a=0 it is a constraint, not a penalty spring or phase floor.
Maximum-shear selection supplies a prospective material plane; on the fixed
candidate line here that orientation is already prescribed.

## Native solver and certification

For the selected positive loading, a positive first-formation predictor solves
the smooth one-sided bond return with nodal bounds only where virgin Gauss records
require them. These bounds are an **algorithmic candidate search**, not a new
constitutive direction lock or phase-growth mask. Established damaged records
remain free to reverse. Acceptance additionally requires a solution of the full
unrestricted box-multiplier equations above, including negative/alternating
virtual directions. If no certificate is found, the candidate is not published.

The native certificate minimizes the convex squared weak residual over the Gauss
box. Accelerated projected iterations use the bound ||R||1||R||infinity for their
step, stop at the unchanged 1e−8 Pa force tolerance, and have a finite iteration
limit. Clipping **dual intact multipliers to their constitutive strength box**
is not clipping physical stress, slip, damage or phase. An independent linear
program verifies feasibility; direct exported native multipliers verify the
actual residual to the original allowance plus explicit floating-point evaluation
roundoff. No linear-program tolerance substitutes for native acceptance.

On this monotone Q1 front each intact element retains all three Gauss equalities.
They imply both endpoint variations vanish. The reduced tangent therefore uses
the exact nullspace of those equalities, excluding constrained rows. A partial
Gauss tie (e.g. an interior sign-changing zero) is explicitly rejected by this
fixture because it would need the general constraint nullspace, not both endpoint
constraints. The current test does not qualify that different situation.

The complete relaxed tangent retains current pressure, state and bond/damage
partials. It is finite-differenced in admissible free directions. Record both
its reduced symmetric-part curvature and the one-sided formation-release
curvature over all nodal directions. A negative reduced curvature is rejected,
with diagnosis required before altering the law. A negative unrestricted value
alone in a strictly tied region would not establish an admissible instability.
No old symmetric phase-potential Hessian is reused.

## History, work and lifecycle

Every trial reads the same accepted nodal Maxwell memories and Gauss physical
histories. Tied records retain exactly zero jump/a and their material age advances
at zero rate. Multipliers are equilibrium reactions, not additional physical
memory. Newly admitted records inherit their age and immutable b, then pay bond
storage, GcΔa and contact dissipation. The accepted seed's .25 J/m initial fracture
inventory is reported separately from new fracture.

The experimental owner (not the production outer cutback driver) exercises
rejected candidate/unchanged owner at the activating step, deterministic
retry and exactly one commit. Actual ASPECT restart brackets that activation;
output branches join at the saved accepted time. Loading ends at .84 s and its
displacement increment is integrated exactly over a step that might touch the
transition; the hold lasts through .86 s. The BE ledger, signed bond endpoint
remainder and cumulative absolute trapezoidal physical-work error remain separate.

The first two independent semismooth attempts stalled at activation. They were
solver failures, not evidence of constitutive instability. Logs and the retained
failed reference are preserved. The certified predictor produces positive tested
relaxed curvatures without changing material parameters or tolerances.
