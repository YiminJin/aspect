# P5a bounded point qualification — selected before running

Entry f1f9f29cf40c32d1ddcea000b724375b64853532 plus preserved local work.
The user selects the tensor-memory/changing-flow option in the P5 specification.
Implement a separate stateless experimental API and a small point owner only.
Do not connect it to the native solver or production checkpoint format.

Retain G=1e6 Pa, eta=1e8 Pa s, m=4, R=1000 Pa, Kf=8 Pa, Gc=1 J/m2,
ell=.046875 m. Full 3-D contractions include zz and both xy terms. A/K/Q live
in the in-plane trace-free space. T/A/K/Q/phi/L are accepted history, never a
rotating scalar correction. Exact yield equality selects the sticking member
of the generalized derivative. At h=0 use the constrained return and admissible
right phase derivative, without a phase floor. Plot direction only for |W|>0.

Fixed-S identity covers intact/tiny h, finite nonzero histories, stick, positive
and negative sliding, full plane-strain T/increments, dt=.00125/.1/1 s, and
both strain and phase tangent actions. Free and restricted responses also
receive complete finite-difference checks, rotations of coordinates by .37 and
1.11 radians, and an S/-S sign control. These are passive basis changes, not
finite-rotation or objective-rate qualification.

Use the established defect norm |error|/max(physical scale,|reference|), target
1e-12 for algebraic identities/covariance; complete return FD <=1e-6 with
epsilon=1e-4 and 5e-5, central in smooth interiors and second-order one-sided
at the intact phase bound. Scales: 1000 Pa stress/Q/T, .001 A/K/L, .001/s flow,
8 Pa driving, 1 J/m3 energy and .01 W/m3 dissipation. BE ledger retains the
existing 1e-10 scaled bound. Independent potential gradients retain 1e-7,
with no tolerance increase after measurement. Inspect both sides of yield;
do not claim a single classical derivative at the kink.

One demonstration path from fresh T/A/K/Q/L=0 and phi=.2: dt=.25 s to 200 s.
Engineering xy strain rate is 1e-4/s through 50 s, then zero. Engineering
normal-difference rate (Dxx-Dyy) is 1e-4/s in (90,150] s, then zero; all
increments are trace-free. Phase is fixed, not a KKT equilibrium test. Lambda
is 100 s, so the holds resolve visible memory relaxation with the same eta/G.
Run identical imposed increments with radial and fixed-S returns. A/k/L are
strain histories, not surface slips. Energy/dissipation are per bulk volume.

At 120 s serialize/restart both point owners and compare every actual field,
including full Tzz, not just a serialized-string comparison. Discard two trials,
retry and accept once during non-proportional loading. Reject duplicate/stale
publications; compare every later state with uninterrupted advancement. Include
a separately populated full-tensor record to expose omitted serialization fields.
Retain complete original scalar tests and P4/BP3 hashes. No native/PDE run,
spatial or temporal campaign, direction-dependent friction or free specimen.
