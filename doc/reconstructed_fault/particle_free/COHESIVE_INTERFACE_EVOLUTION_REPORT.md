# Native evolution and bounded straight propagation

**Native C++ now computes its own coupled accepted states. The straight specimen
advances a fully decohered region about .75 m beyond its mature seed, through a
resolved cohesive zone, and continues sliding during a short displacement hold.**
The candidate-end buffer limits further continuation. Early spatial sensitivity
remains a failed criterion; this is a prescribed-path demonstration.

[Scope and decisions](COHESIVE_INTERFACE_EVOLUTION_SELECTION.md) ·
[figures, animation, indexed CSVs and commands](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_evolution/INDEX.md).
Baseline: `pf-rsf-particle-free`, `f1f9f29cf40c32d1ddcea000b724375b64853532`
plus preserved local work. Production/BP3, the C++ cohesive–RSF law and previous
archives remain unchanged. No new constitutive choice was required.

## Native solver and checks

The opt-in fitted owner uses `CI_NATIVE_EVOLUTION=1`; inputs contain native mixed
operators and **one initial physical snapshot**, with no target trajectory.
The straight native fixture uses `CI_SIGNED_SOLVER=1` and assembles its bulk
operators from ASPECT's mesh/FE space. Both use the shared signed solve and the
unchanged `CohesiveInterface::evaluate` point return. ASPECT supplies scheduling,
mesh and checkpoint infrastructure; host Stokes/advection are disabled in this
component fixture. Mechanics is solved through exact mixed Schur condensation,
including pressure; physical stress uses the full native FE response. Production
solver integration and automatic path creation remain separate engineering work.

At each trial, c_new=beta(c_old+(Delta boundary amplitude,Delta nodal jump)),
beta=(1+G dt/eta)^−1. The oblique boundary amplitude is dimensionless strain; the straight amplitude
is relative wall displacement in metres. The native operators retain those units.
The complete residual and tangent use current sided normal
pressure, signed slip, implicit RSF aging and irreversible Gauss damage. Intact
points have bounded weak multipliers, without a compliance floor. General
C=ker(Q_tied) reduction retains C^T F and C^T J C, followed by the unrestricted
weak certificate. Q2 shape signs are retained. Geometry/layout maps are fixed
or invalidated at the geometry generation; constitutive values are recomputed
from the immutable accepted history in every nonlinear trial.

The straight route reuses the established front predictor. Its monotone proposal
is a solver strategy; the complete signed law and unconstrained admissible
reduction determine acceptance. Accepted-rate continuation handles failed predictor
searches. Row/column equilibration changes the linear solve's conditioning;
convergence is checked in the original equations. Contact-violating Newton
proposals are rejected. Optional bounded numerical cutback halves the physical
interval from the unchanged accepted state (depth ≤4, minimum half interval
.0003125 s). Curvature/contact/buffer failures are not waived by cutback.

| Focused native verification | Measured result |
|---|---|
| Oblique reference, tied insertion through activation and hold | Max six-history discrepancy **1.11e−15** |
| Independent native coupled derivative | **2.06e−8** relative, allowance 1e−6 |
| Oblique complete weak residual | **9.91e−9 Pa**, allowance 1e−8 Pa |
| Oblique BE ledger | **1.12e−14 J/m**, unchanged scaled allowance 1e−10 |
| Actual zero-time geometry insertion | .25 s, 20 physical steps; rollback/retry/duplicate checks pass |
| Oblique ASPECT restart | Accepted archive and physical history rows **bitwise** |
| Loaded straight rejection/cutback | One injected failure before publication; matching two-half-step control gives **bitwise accepted archive** |
| Retained default-off straight route | 20 matched accepted steps: physical columns **bitwise** against archive |

The oblique trajectory carries negative slip; the longer straight trajectory
carries positive slip. All six Gauss histories, nodal jump, Maxwell coefficients,
physical clocks and publication counters are retained. Rejected work is not
advanced or published. The lifecycle injection is a test mechanism, not a claim
that a physical instability occurred. The native input fingerprint and existing
geometry/backend restart guards remain in force.

Development failures are preserved in the evidence/work directories: a
primal-only tie test misclassified arithmetic remnants at tiny rates; the cold
semismooth system was poorly conditioned and later exhausted its iteration
budget; a fixed-row reaction derivative was initially omitted; an extrapolated
predictor lost contact near full decohesion. The implemented corrections are
exact tie certification, primal–dual classification, algebraic scaling, the
complete reaction map, and accepted-rate continuation/contact globalization.
The general cold solver is not claimed globally convergent for arbitrary paths.
No strength, material parameter, law or acceptance tolerance was tuned.

## Specimen and imposed loading

Use the retained finite **8×1 m periodic strip**, Γ at y=.4 m. The initially
mature seed is [2.5,5.5] m. Γ=[.5,7.5] m is installed tied ahead of it, with
exact zero jump at its endpoints/outside. The bulk pressure approximation and
mesh remain fixed. The prior [1,7] m candidate is the buffer control. Through
2.42 s its state/storage agrees with the enlarged tied representation to about
4e−12 J/m; extra records remain intact. Changes after that are physical admission
on the additional candidate, rather than transfer error.

Fresh data: zero jump/Maxwell memory, Θ=1, a=1 on the 3 m mature seed and a=0
elsewhere. Thus initial bulk/cohesive storage is zero and pre-existing fracture
inventory is **3 J/m** on both meshes. This is an explicit mature-seed
initialization, with identical representation and no imported coarse relaxation.
New fracture excludes that initial inventory. Phi remains provisional geometry;
Gc∫a on Γ is the single fracture reservoir.

| Quantity | Retained value |
|---|---:|
| G; eta | 1e6 Pa; 1e8 Pa s |
| Gc; intact peak | 1 J/m²; 3000 Pa |
| Background confinement p0 | 10000 Pa |
| RSF direct/evolution/mu0 | .005 / .01 / .1 |
| V0; Dc | 5e−6 m/s; .001 m |
| Radiation | G/(2 sqrt(G/2670)) |
| delta_f=2Gc/peak | .0006666667 m |
| Fine mesh Δx, Δy | .0625 m, .0125 m (128×80) |
| Coarse control Δx, Δy | .125 m, .025 m (64×40) |

The straight fixture retains Q2 bulk displacement, Q1 pressure with independent
seam traces, Q1 jump, Gauss3 bulk/source/surface histories, and independent Gauss6
bulk work/stress. The oblique reference uses its retained Q2/P1disc bulk and Q2
jump. The .2 m lifting support is a numerical exact-jump representation parameter,
not a new phase/process length. The fine bulk mixed system has **91,903 masters**;
the host has 134,580 DoFs including unused temperature DoFs.

Impose top/bottom relative speed .001 m/s until **2.45 s**, then hold displacement.
The endpoint is selected after the coarse forward pilot first achieved ≥.5 m
full advance. A requested .05 s hold meets the existing buffer guard at trial
2.4825 s, last accepted 2.48 s. Continued-loading pilot: guard at 2.48 s, last
accepted 2.475 s. Both stopped states/trials are retained. Γ is not extended again
to evade those guards. Select **2.47 s** as the common resolved comparison endpoint,
including .02 s of hold. Per-run timeout is 900 s; runs are serial and sequential.

## What propagated and its numerical sensitivity

At 2.47 s the Δt=.0025 fine solution has added fracture **1.52627 J/m**, peak jump
**1.09862 mm** (exceeding Dc), and a seed-connected fully decohered left extension.
The last intact/full Gauss bracket is x∈[1.74296,1.75704] m. Independently, the
actual Q1 jump reaches delta_f at **x=1.74460 m**, giving **.75540 m advance**.
New-material nodal increments are nonnegative throughout, so this reconstruction
uses the actual continuous jump and damage law; no age interpolation is involved.
**Twelve complete fine cells (.75 m) newly reach full decohesion.** This is not
inference from one activated node. The right side mainly develops a cohesive zone.

The left a≥.01 zone starts near x=.868 m. Its distance to the full front is
approximately .89 m, or **14.2 fine cells**. Right a≥.01 reaches x≈6.344 m. No
disconnected a≥.01 region appears on the installed Γ. The phase-ridge/path is
prescribed straight; tangential slip direction does not select a crack direction.

| Comparison at 2.47 s | Equal-Δt spatial refinement | Fixed-mesh event-window halving |
|---|---:|---:|
| Integrated growth difference | .002072 m (**.136%**) | .005036 m (**.330%**) |
| Growth L2, common .125 m physical boxes | .240% | .421% |
| Integrated slip difference | .0651% | .114% |
| Continuous full-front difference | **3.19 mm** | **3.08 mm** |

Relative response differences use the 128×80 Δt=.0025 trajectory as denominator; window sliding uncertainty uses the half-step loss.

The fine replay begins from the *same actual native checkpoint*, physical time
2.2975 s, using Δt=.00125 through the event and hold. Its final full advance is
**.75233 m**. It changes window sliding loss .645%, below the selected 1% target.
Its prefix is shared; this is temporal evidence for the event/hold window,
not a fresh half-step qualification of first onset.

**The early spatial criterion remains failed:** at 1.8 s the growth difference
is .005551 m (**65.4%**), common-box L2 66.9%; at 2.0 s L2 remains 13.6%.
Numerical first-damage brackets are [1.625,1.6275] s coarse and [1.5175,1.5200] s
fine at the diagnostic a>1e−8 threshold. This threshold only labels onset; it is
not a damage floor. Precise initiation timing and propagation speed remain
unqualified. Two meshes do not establish an order. The passing late comparisons
support developed advance, not the entire evolution interval.

## Energy, physical stress and the hold

At 2.47 s the baseline fine accounting [J/m] is: trapezoidal boundary work
22.21210; bulk storage 15.60189; cohesive storage .275027; new fracture 1.52627;
friction+radiation loss 4.48363; Maxwell loss .338476. The signed physical
remainder is −.0132015; the **sum of absolute per-step remainders** is .0132205:
**.866% of fracture and .295% of sliding loss**. No numerical term is subtracted.
BE ledger errors stay ≤4.33e−13 J/m and remain a separate algebraic check.

The focused half-step window halves its absolute remainder, .0126169→.00629039
J/m. **Hold-only baseline errors fail 1%**: 1.260% of hold fracture and 1.064%
of hold sliding. The half-step values pass: **.633% and .537%** respectively.
These failures are retained in `loading-hold.csv`; global closure is not used to
hide the hold error. The half-step hold adds .297428 J/m fracture and dissipates
.351013 J/m sliding, with zero external work. Stored energy supplies this growth.
Baseline hold work is ≤6.5e−13 J/m from clock roundoff. Stopping motion does not
remove stress. No blanket hold-growth prohibition or artificial damping is used.

Actual Gauss6 fields contain 368,640 physical points per saved state. At
2.30/2.45/2.47 s, remote x<.5 or x>7.5 maximum in-plane shear is
**2460/2707/2734 Pa**, below 3000 Pa. Native remote weak-plane shear is checked at
every accepted step. These saved full-field checks are not an all-time theorem
about off-path nucleation. Minimum sampled bulk horizontal normal pressure is
8682 Pa; current interface-side pressure remains compressive. Solved p′ has zero
volume mean to 4e−14 Pa; p0 is a separate background pressure. Bulk work error
is 1.02e−10 Pa and stored-energy discrepancy ≤2.14e−13 J/m under independent
Gauss6 integration. Figure colors show cell averages; extrema use unaveraged
quadrature and actual weights, not point counts.

All accepted complete weak residuals remain ≤1e−8 Pa; maximum coupled derivative
error is 4.10e−7 relative. Minimum fine admissible symmetric tangent is
3.525e6 Pa/m (half-step 4.933e6). This retained coercivity test is not a general
continuum or global branch-stability certificate for pressure-coupled RSF.
The finite strip, off-midheight fault, normal-pressure redistribution and
periodic images are part of the chosen problem. They are not distant-boundary
accuracy references. Local bulk stress can exceed intact strength near the
cohesive zone; this fixed-path run does not qualify a free-path choice there.

## Cost, lifecycle and conclusion

Native fine prefix to 2.3 s: **210.0 s**; checkpoint continuation to 2.47 s:
**115.1 s** (including an **84.7 s** operator rebuild). Half-step replay:
**145.9 s**. Coarse hold/buffer run: **33.0 s**; physical-field reconstruction:
**92.8 s**. All use one MPI rank and one BLAS/OpenMP thread. Checkpoints are
preserved outside rolling overwrite. Numerical final rounding-sized steps are
retained in raw histories; their changes are negligible and not hidden.

Rejection checks cover immutable nonlinear trials, rejected invalid histories,
stale/duplicate publication, actual cutback and deterministic retry. Oblique
restart is bitwise; the loaded straight cutback matches its equivalent
uninterrupted accepted-step control bitwise. A final-source replay of the fine
checkpoint window reproduces all Gauss/nodal/bulk histories and physical
history columns **bitwise over 70 accepted steps**. Rebuilt fixed operators reproduce
accepted physical fields without transferring Gauss histories between layouts.

**Qualified:** native fixed-mesh, homogeneous, serial coupled component evolution;
tied insertion/activation/hold; restart and once-only publication; and developed
straight prescribed-path propagation through a resolved cohesive zone.
**Remaining:** the 65.4% early spatial failure, precise front speed, candidate/end
and outer-boundary influence, general cold-solver globalization, free paths,
bends, MPI/AMR/transport and production integration. BP3 and prior failures remain
protected. Phi has not acquired a second fracture-energy role.

The next useful scientific task is one enlarged-domain/tied-buffer control,
with unchanged seed, near-tip resolution and equivalent loading, before extending
the event further. For example, double the periodic x span about the same seed
while retaining H=1 m and fixed material values; compare the same physical times
to quantify image/end influence. This remains a finite-strip study. No new corner,
direction-selection or fracture-energy law should be inferred from the present
result, and no next run starts automatically.
