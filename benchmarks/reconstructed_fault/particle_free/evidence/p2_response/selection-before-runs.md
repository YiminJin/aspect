# P2 bounded physical-response qualification

Selected 19 October 2026 from the completed serial hanging-node qualification.
Base revision `612b75d3570d81763407875f3a12c12999b9fa8e`, preserving the incoming
worktree in `build-particle-free/p2-response/entry.patch`. No production source,
constitutive law, solver tolerance, acceptance rule or default change is planned.
This task tests the current homogeneous domain-free DGQ2 / ordinary-FE cohesive
consistent endpoint on a fixed through-going fault. Geometry, mesh and diagnostic
particles stay fixed. No RSF, transport, AMR/MPI expansion or P3 is selected.

## Selection and practical criteria (before trajectories)

Use the retained single continuum seeded-H BVP as **pre-existing physical
damage**, not as stored elastic or crack-driving energy. Sample its physical
values at native nodes with zero initial T/Z/slip. The original 512-node table
must equal the BVP at all native nodes exactly. Do not resample its literal
piecewise-linear interpolant at new reference nodes; evaluate the same BVP.
Do not relax separately on each mesh. Equilibrium at zero load means KKT of
ordinary-FE fracture subject to this prescribed lower damage obstacle.

Select a uniform 2×512-cell mesh: dy=1/512 m, eight fixed surface segments,
Q2 velocity / Q1 pressure and phase, all nine modes per DGQ2 tensor component,
Gauss12 physical / source3. Uniform refinement avoids an unnecessary new mesh
preparation path. Existing 1D evidence predicts resolved initialization and
whole/localized stress error 0.294%/0.611% against continuum mechanics with
1024 phase cells; this prediction must be checked on the new interval.
Native along-fault variation will determine applicability of that reference.

Keep G=1e6 Pa, eta=1e8 Pa s, Gc=26.6666666667 J/m², ell=.15625 m,
Lorentz p=1, fixed mu=.1, prescribed normal pressure=10000 Pa, natural phase
boundary and compatible lateral velocity conditions. Exact interval mean of
u_x=.256*t*(y-.5) runs through t=.5625 s; then boundary velocity is zero through
**t=.59375 s**. Histories continue advancing in the hold; no stress is reset.

Run exactly these finite controls, serial/one thread:

- Fresh native unloaded control through .015625 s, dt=1/128, same mesh and seed.
- Fresh native loaded/hold case through .59375 s, dt=1/128.
- One native dt halving, same loaded/hold case, dt=1/256.
- Corresponding inexpensive 512-phase/512-mechanics references at those dts,
  and one 1024-phase/continuum-mechanics spatial reference at dt=1/256.
  No mechanical/phase resolution sweep; previous decomposition remains evidence.

Native budgets: unloaded 180 s, primary 1200 s, dt/2 2400 s. Each reference
budget 240 s. Build budget 300 s. No budget enlargement or further case is
implicit if a run fails. Existing transaction/restart evidence is reused.

Initialization targets: phase relative L2 <=1e-4, gradient relative L2 <=2%,
0<=phi<1 and positive I; constrained unloaded gradient/A <=1e-11 in the
independent model (native arithmetic target unchanged), and unloaded native
phase drift <=1e-8. Loaded growth above the original seed must exceed .001,
start after mechanical loading, and exceed unloaded drift by at least 100×.
Require physical irreversibility, native accepted KKT/mechanical/contact checks,
and once-only publication without changing any native tolerance.

Retain the previous spatial practical screens on common fields/times: whole
full-tensor stress <=1%, localization-weighted RMS <=2%, phase relative L2 <=1%
and Linf <=.005, cohesive traction/slip/total energy <=1%, common phase driving
<=2%, onset within one coarse dt. Use fixed reference peak scales over loading,
not near-zero startup denominators. Preserve every unaligned branch difference.
The old strict 1e-6 native/reference branch flag stays separate and false.

Temporal screens: phase Linf <=.001 and relative L2 <=.2%; traction <=1 Pa
and .2% of fixed reference peak; cumulative slip <=.1%, common-interval rate
<=.5%, whole/local stress <=.2%/.4%, storage <=.2%, driving <=.4% of their fixed
reference scales. These are comparison screens, not solver tolerances. Compare
only states at equal physical times and interval-average rates on equal intervals.
For tiny branch/spatial differences report absolute uncertainty without an
unsupported error-order claim.

Energy: report storage, boundary/pressure work, bulk/cohesive viscosity, friction,
phase work and free-mechanical/contact residual works separately using the
existing PF6a diagnostic interval path. Sum absolute external remainders over
loading/hold separately; require loading sum / max(peak storage, absolute boundary
work) <=.5% and reduction under dt halving for a physical-work readiness claim.
Also report signed sums and the largest interval remainder. Algebraic identity
closure alone does not pass this screen. Hold response must be interpreted
against retained stress and Maxwell decay, with contact complementarity, not
against an assumption that zero boundary velocity removes stress.

Verify selected native snapshots with independent normal integration and Gauss24/32
physical functionals, retaining existing 1e-9 scaled energy/load screens. Snapshot
times cover startup, early loading, pre-onset, onset, growth and hold. All compact
endpoint fields, gradients, velocity and metrics remain exported each accepted
step; only dense point/state diagnostics are scheduled. Empty schedule retains
the observer's old behavior.

Results and P2 readiness will be recorded below after the bounded experiment.
