# Coupled bounded-profile implementation protocol

The current user selects implementation and a coupled accuracy comparison,
following the completed boundary/profile isolation. Entry is branch
`pf-rsf-particle-free`, revision `27178813cebd471b93a7a9308f41215b71b4a20b`,
with incoming dirty source preserved. `bounded_profile/entry.json` and its patch
record this starting point; the current core and four checker libraries were
copied and hashed before rebuilding. No production checkpoint is used.

## Recorded support correction and final comparison

The first `bp-*` matrix exposed a support mismatch: the inherited stationary
association tube truncated the compact profile although I integrated its full
normal section. Those runs and their source/executable pairs are retained as
truncated-support diagnostics. The corrected primary `bpf-*` matrix uses the
existing `full normal cross-section` policy, now explicitly admitted for the
transported mature selector with finite endpoints. New nondefault profile or
inflow options require this policy. Its identity is included in checkpoints;
old truncated checkpoints are rejected, not converted. The original default
Q1/zero-inflow/stationary-support control remains available.

The corrected matrix ends at 0.2 s (two to eight physical steps), still crossing
cells in both directions, with the same initialization interval and h/dt
refinements listed below. Every corrected run exports pressure. Two additional
32², dt=.05 runs halve the *initial* surface-label spacing from .1 to .05 m;
there is no within-run resampling. The inverse-transpose material support-width
map and finite tangential endpoints are unchanged. Normal support loss is
measured independently. Early full-support checkpoints without the final
policy identity are evidence only and are not silently converted.

## Selected method and boundary problem

Keep Q1 phase storage and the accepted-velocity characteristic displacement.
Let A-minus be Q1 pullback at x-dt*v_old(x), A-plus the corresponding return
lookup at x+dt*v_old(x). The candidate is
`A-minus phi + (phi - A-plus A-minus phi)/2`, clamped to the extrema of
accepted Q1 vertices of the admitted departure cells. Where return coverage is
missing, use the forward value. Preserve the existing remote-search tolerance,
multiple-donor averaging and zero incoming phase. This is a donor-bounded
MacCormack correction, not a conservative flux scheme or a higher-order
trajectory integrator. The same transported phase is the material obstacle in
the admitted growth-disabled configuration. It is never maximized against the
old Eulerian profile.

Add explicitly selected frozen accepted-trace inflow for T and U=R+P B.
This specifies constant-normal continuation of the incoming accepted tensors;
reaction and rotation still occur once in their original stages. It is a
finite-domain boundary approximation, not analytic infinite-domain inflow.
Use pure translation (+/-0.4,0) on left/right/bottom and the existing natural
top boundary, removing the earlier imposed box-wide shear source. Initial
stress is zero; the compact oblique profile remains well inside the box.
Reference calculations use this identical boundary problem. Neither a tensor
boundary selection nor a profile selection changes the Maxwell law, total
Jaumann spin, surface state source, endpoint solve or acceptance tolerances.

The old Q1/zero-inflow defaults remain. Nondefault selectors enter the existing
schema7 material identity: incompatible new-local restarts must fail, with no
conversion. No new independently stored history is introduced.

## Comparison and gates

Use the previous finite oblique fault, nonuniform age/traction and RSF inputs,
G=10, eta=10 or eta=10*(1+.5*tanh(4*x)), and the compact profile used in the
isolation study. The original truncated-support matrix integrated to 0.4 s, with common initialization interval 0.1 s.
Both methods and both directions run at n=16, dt=.1 and n=32, dt=.05.
Positive-direction separate h and dt halvings distinguish the two effects;
n=64, dt=.025 bounded runs supply references for each coefficient/direction.
These are short synthetic local calculations, not production qualification.

Independently reconstruct Q2 bulk tensors and Q1 phase from exported samples.
Check native profile replay and donor bounds at 2e-11 absolute, surface
age/memory source identities at 1e-11, and full publication/rollback counters.
Compare profile L2/peak/mass/centroid/variance, normalization I, physical stress,
point shear traction, velocity, labelled geometry, V, age and auxiliary Z.
Compare methods at equal mesh, dt and MPI size, retaining wall time and solve
counts. Improvement must be measured in the coupled fields, not inferred from
the prescribed translation isolation alone. A finer bounded reference is not
an exact continuum solution; retain both methods' refinement differences.

For the mature bulk branch, independently integrate psi=|tau|_3^2/(4G),
mechanical power tau:D3, slip power tau:L[V], Maxwell loss |tau|_3^2/(2eta),
dilatation psi*div(v), and outward material-energy flux (v.n)*psi. Report
endpoint and trapezoidal continuum-balance remainders, signed sums and sums of
absolute step errors. This measures splitting/projection/transport/time errors;
it is not the cohesive total-energy ledger or a fitted numerical dissipation.
Check quadrature sensitivity before drawing energy conclusions.

Repeat completed-trial rejection against its matched accepted schedule and
new-local two-to-four-rank restart through cell crossing for both coefficient
cases. Check method identity mismatch rejection. Run the core unit suite and
the retained-default comparison. At most four local build/test workers run at
once. Reuse isolation evidence; preserve failures and unrelated cohesive gates.
