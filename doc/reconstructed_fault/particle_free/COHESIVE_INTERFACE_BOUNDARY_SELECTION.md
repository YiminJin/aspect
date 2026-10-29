# Boundary mechanism and finite-strip continuation selection

Baseline f1f9f29cf40c32d1ddcea000b724375b64853532 plus retained local work.
Protect production/BP3 and all previous evidence. No constitutive change.

## Mechanism first

Use the existing exact incompressible layer DtN reference and its Q1 Galerkin
projection, at identical saved physical relaxed-jump fields m(x). Separate K/N
changes from subsequent nonlinear history evolution. With H=t+b, d=t−b,

    K(0)=G/H, N(0)=0,
    K(k→0,k≠0)=G/H+3Gd²/H³,
    N(k→0)=−i 6Gd/(H³ k)+O(k).

Check mean/long modes, alias tails, centred/reflected layers, and walls moved
equally away while keeping physical offset. For changed H, compare the same
background Maxwell stress (wall displacement scales with H), not fixed wall speed.
Check physical p0 separately from the algebraic pressure pin/global mean removal.

## One native diagnostic, then retained finite specimen

The inverse-gradient response is a physical consequence of finite clamped,
impermeable walls and an off-centre fault. Increasing period alone cannot turn
this into an isolated fault in an unbounded body. Select an **explicit finite-height
periodic specimen**, L=16 m, H=1 m, fault y=.4 m, physical x=[−4,12], seed
[2.5,5.5], Γ=[−3.5,11.5]. Boundaries are part of this benchmark; do not claim
distant-boundary accuracy. Do not replace it by a centred fault.

First run one **diagnostic** 8×1 m native control at y=.6, same positive loading,
seed and Γ as the old 8 m experiment. A proper 180° rotation maps it to the
retained y=.4 experiment: x→8−x, y→1−y; damage/jump/age and physical stress
map accordingly. K is unchanged, N changes sign, and the developed front should
swap sides. Require existing comparison allowances: stress/pressure ≤1e−7 Pa,
jump/memory ≤1e−10 m, damage/Θ ≤1e−9, original residual/derivative/work gates.
No pressure is prescribed to enforce this result. The diagnostic does not
replace the intended off-centre longer specimen.

## Bounded longer run

After the reflection check passes, resume the retained 16 m accepted checkpoint
near 2.2975 s, **before** the old loading endpoint. Extend only its future loading
plan: .001 m/s to 2.55 s, then displacement hold to **2.58 s**. Document a guarded
forward-only loading-plan extension; accepted times after the old load endpoint
must never be reinterpreted. Preserve all histories, physical clocks and publication.
Compare the shared prefix through 2.45 s against the retained original trajectory.

Use Δt=.00125 s through 2.45 s, then .000625 s; the first pending checkpoint
interval remains its original .0025 s. Target ≥1.5 m new fully decohered advance,
a cohesive zone with ≥8 cells, and measurable mature sliding or arrest during
the .03 s hold. Keep Δx=.0625, Δy=.0125, Gauss3/6, physical material/RSF parameters,
single Gc∫a fracture inventory and all current-pressure/history semantics.
Initial seed inventory remains 3 J/m. No phase mask or extra dissipation.

Preserve a valid checkpoint before 2.43 s outside rolling overwrite, then replay
the new event/hold with .000625/.0003125 s (halved intervals). The initial pending
checkpoint interval may be shared; report it explicitly. Compare common times,
growth-relative errors, continuous full fronts and physical work, not first nodes.
Practical targets: ≤1% integrated growth/slip, ≤.015625 m full-front change,
raw sum|work remainder| / new fracture and / sliding ≤1%, also hold separately.
Preserve failed targets. At most one further local halving if a genuinely
unresolved event requires it; no mesh campaign or extra physical parameter sweep.

Stop at the declared endpoint or existing closed-contact, weak/derivative/ledger,
curvature, remote intact-strength, detached a≥.01 or .25 m candidate-buffer guards.
Preserve failed trials and last accepted/checkpoint state. No limit is relaxed.
Bound sequential serial runs at 900 s (reflection), 1500 s (longer), 1500 s
(replay), 6 GiB observed RSS. Estimate roughly 5+15+17 minutes, including repeated
native setup. No server job, MPI expansion, free path or bend sliding.

Report this as a prescribed-path finite-period/finite-height mechanical specimen,
not a boundary-independent earthquake speed. Retain the 65.4% early spatial failure,
previous boundary sensitivity and branch/time limitations. Report absolute and
relative errors, actual cost, accepted publication and restart evidence.

## Computational follow-up after the actual wall stop

The longer command hit its 1500 s administrative wall bound at accepted
2.57625 s (trial header 2.576875 s), without a physical/solver guard failure.
Its 745.9 s rebuild and repeated complete derivative checks exceeded the initial
cost estimate. Preserve that stopped run. Complete the last 3.75 ms from the
last valid native checkpoint at 2.574375 s in one separate **1100 s** bounded
command; require identical replay of its overlap. The physical plan, histories,
grid, timestep and all tolerances remain unchanged. Give the not-yet-started
half-step replay **2100 s**, based on measured setup/check costs. Retain the same
6 GiB RSS bound. This changes computational budgets only, not acceptance targets.
