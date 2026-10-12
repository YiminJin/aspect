# Initial P2: resolved mechanical feedback, normalization admissibility failure

Current explicit governing revision: [recoverable-energy P2 report](P2_recoverable_report.md).
Local energy history and unloaded initialization pass, but three coupled timesteps
remain unqualified because of traction-transfer and mechanical work discrepancies.
The high-load evidence below remains historical and unchanged in qualification.

**New bounded material study:** [P2_material_report.md](P2_material_report.md)
uses fresh nominal core 0.6 and the three permitted modest load amplitudes.
It avoids saturation and large normalization error, but fourfold Gc stays
inactive and reference feedback nearly disappears at half dt. The original
high-load results below retain their failed qualification status.


**Normalization follow-up (2026-10-06):** the fine failed-entry field has exactly
phi=1 and g=0 at both side endpoints. Its transverse and along-fault normalization
diverge; prior finite Gauss samples do not establish integrability. Optional
I_h-only row-sum lumping passes restricted P1 implementation checks, but the
finite coarse field still has 66% / 35% relative L1 representation error at
0.1 / 0.05 m spacing. No new coupled paths are justified. See the
[normalization report](normalization_report.md). Historical evidence below is
retained; the newer diagnosis supersedes a projection-only interpretation.


2026-10-06. P1 passed under the user-selected fixed full-cross-section support
policy, and P2 followed without another approval gate. **The initial P2 package
does not qualify coupled evolution:** both timestep paths grow native mechanical
H and consume it in a resolved phase response, but the current consistent Q1
normalization projection produces negative nodal I_h before the requested
one-second interval completes. No normalization floor, activation mask, history
law change or compensating factor was introduced. A scientific/discretization
decision about positive normalization representation is required before continuing.

## Scope and production path

Same worktree/branch/HEAD as P1, with preserved incoming work. Final core
`build-propagation-integration/aspect-release` SHA256
`ab3e920506cbc26025ed9547a38d5cf184ce79330ebd24572c60f2dfb3868e5c`;
GCC 12.4.0, deal.II 9.6.2, OpenMPI 5.0.6, Trilinos 14.2/Epetra, Voro 0.4.6,
Release/unity. The observer uses the same compiler with finite-value checks and
FP contraction disabled. Final inputs, source, binaries and all attempts are
pinned in the [evidence manifest](../../../benchmarks/reconstructed_fault/propagation/coupled_evidence/manifest.json).

The setup is one horizontal prescribed fault from (0,0.5) to (1,0.5), a fixed
32×32 unit Box mesh, 9,216 fixed particle parents, 0.1 structural spacing and
full-normal association width 0.50000000000001588. There is no tip motion, new
geometry, branching, multiple fault interaction or bulk-nucleation model.
AT1/Lorentz, length 0.15625, curvature 1, G=1e6, viscosity=1e8, cohesion=1000
and Gc=26.666666666666667 retain the P1 material/phase scales. Fresh native
initialization supplies H; the relaxed phi maximum is 0.8084775462894394,
current/previous I_h≈366.20755 and projected initial cohesive q≈463.56835 Pa.
Timestep zero preserves initialized H and Maxwell history.

The diagnostic is quasistatic (radiation damping zero). Stateless rate-dependent
friction has equal reference/dynamic coefficients 0.1, so no velocity weakening
or RSF state evolution is present. Friction uses prescribed adiabatic pressure
10000 Pa with zero gravity; temperature is fixed at 293 K. All four boundaries
use ux=0.4*t*(y−0.5), uy=0, an identical smooth shear-rate ramp for both dt=0.25
and dt=0.125 s. Its intended total shear displacement at one second is 0.2.
This deliberately simple local control is not an earthquake-event model or an
independent physical validation.

The existing named RK2 control `ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION=1`
freezes particle positions without changing constitutive dt. IDs/positions are
checked; load balancing remains `none` and no birth/replenishment occurs. This
result does not qualify transported histories, inflow/outflow particles or
replenishment. The full cross-section intentionally remains eligible for the
native associated-particle update, including initially intact material.

The observer never writes H. Native sequencing remains incoming committed H →
phase solve → mechanics → terminal H/stress/q/previous-I_h commit. No outer
iteration was added. At pre-mechanical observations and every observed linear
solve, incoming H, Maxwell stress, q and previous I_h remain unchanged. Phase
KKT is measured against that incoming H. Post-commit candidate replay uses the
same native point sampler, surface coefficients and driving function, together
with captured incoming q and newly projected q. It checks H_new=max(H_in,candidate)
within 128 double epsilons times max(1,|expected H|); maximum observed absolute
replay error in the final accepted records is about 2.04e-9. This allowance only
checks floating-point replay; it is not an H-evolution parameter. Normalization
observation refreshes transient data; independent reprojection and perturbed
phase probes restore their scratch values and never commit physical history.

## Bounded construction controls and numerical defect

The first pair used the small inherited ramp 0.004*t and radiation damping 1e7.
Both completed one second with H/phi unchanged: q decreased from 463.568 to
about 459 Pa and the driving candidate never exceeded stored H. These retained
subcritical controls do not demonstrate feedback loss. One explicitly revised
quasistatic ramp was sized using the measured G/I_h≈2731 per unit slip to obtain
resolved feedback. After it exposed the failure below, its parameters were kept
fixed; no tuning was used to conceal off-band activation or failed normalization.

The zero-velocity initial solve also exposed a routine pressure-compatibility
check defect. The RHS/base-iterate arithmetic scale was zero, but the fresh
linear residual C*direction−rhs had a roundoff null component 1.7487e-15. The
old check incorrectly required exact zero for this additional matrix-vector
operation. The correction adds the constrained direction's continuity scale
and the corresponding dot-product input scale to the existing gamma bound.
The RHS admission check, nonlinear cap, linear/nonlinear tolerances and physical
pressure normalization are unchanged. A focused regression checks this zero-base,
nonzero-correction case and rejection of real incompatible flux. All 15 Stage-I
unit cases pass (20,141 assertions). This fix is separate from the H/normalization
formulation and does not make the later failure disappear.

## Measured feedback and one-step lag

| Measurement | dt=0.25 | dt=0.125 |
|---|---:|---:|
| Last fully accepted time | 0.25 | 0.25 |
| Cumulative volume-integrated H increase at that time | 25158.86946 | 18636.97874 |
| Current maximum phase at that time | 0.8084775463 | 0.8950330445 |
| Cumulative volume-integrated phase increase | 0 | 0.0046317073 |
| Maximum off-original-band particle phase at that time | 0 | 0.001523468 |
| H increases outside the original prescribed active band | None | None |
| Next phase-consumption time (mechanics then fails) | 0.5 | 0.375 |
| KKT defect of that consumed phase | 8.526e-16 | 2.409e-15 |
| Nodal phase range at failed entry | [0,0.9970866752] | [0,1] |
| Maximum phase increment at failed entry | 0.220314558 | 0.117317597 |
| Maximum incremental healing at failed entry | 0 | 4.2683e-11 |
| Minimum projected current I_h | −7474.77608 | −10087.93124 |

The phase at time t consumes H from t−dt. Thus the two *accepted* phase fields
at 0.25 have consumed different physical histories: the coarse one still uses
initial H, while the fine one uses committed H at 0.125. Comparing those phase
increments directly would misidentify the intentional lag as feedback loss.
The next phase entries consume each path's H at the common time 0.25; those
phase solves converge but their dependent normalization is inadmissible, so
neither is an accepted complete physical timestep.

At the common accepted time, increased-H radial bins extend from the fault
center through the bin ending at normal distance 0.3125, inside the original
active profile at the actual particle samples. Maximum replayed candidates are
3.8230e6 and 6.3352e6 on the coarse/fine paths, respectively: lower cumulative H
is not uniformly lower peak driving.

Cumulative H at the common accepted time is about 25.9% smaller with half dt.
The available data do not separate nonlinear splitting effects, boundary-driven
spatial variation and the historical candidate/max-history small-increment loss.
**Refinement consistency is inconclusive**, and the one-second comparison is
incomplete. No third timestep was used: the resolved normalization failure
already blocks the coupled trajectory; a further dt comparison would not qualify
the requested interval without first resolving that representation issue.

## Why the trajectories stop

The consistent surface mass projection is not positivity preserving for these
strongly varying along-fault integrals. The largest phase/normalization response
occurs near the imposed-velocity side boundaries; it is not a new sharp tip.
Every native transverse integral passed its strict positive-integral check before
projection. Independent panelwise Q16/Q32 integration, using the same native
degradation and Q1 field, confirms positive integrals and negative projected nodes:

| Diagnostic | dt=0.25 failed entry | dt=0.125 failed entry |
|---|---:|---:|
| Minimum independent positive profile integral | 495.16746 | 406.67326 |
| Maximum Q16/Q32 integral difference | 4.403e-4 | 25.9551 |
| Maximum native/Q32 projected-node difference | 1.1485e-4 | 7.0705e-4 |
| Q32-projected minimum | −7474.77607 | −10087.93109 |

Q16 is insufficient for high-accuracy integration of the most concentrated
fine-path profile; this is reported rather than called a tolerance pass. Both
integration orders and native adaptive integration nevertheless reproduce the
large negative projection, far beyond the measured integration discrepancy.
The sign diagnosis is resolved. At x=0.1 and 0.9 the coarse projection is negative
despite positive profile integrals; this is not association omission or an I_h
floor issue. The native material admissibility guard correctly stops mechanics.

The failed-phase Jacobian corrections are 8.64e-17 and 1.24e-16; identical-H
perturbed repeats differ by 1.11e-16 on both paths. These support uncertainty
estimates, not rigorous bounds. The fine-path 4.27e-11 healing exceeds the retained
straight allowance 4.922e-12 and is separately reported, even though very small
relative to the resolved phase increase. It was not removed by a previous-phase
constraint. Native phase relative tolerance is 1e-12, CG 2e-7, KKT gate 1e-11;
mechanical nonlinear tolerance is 1e-8, with at most 30 iterations. Accepted
mechanical residuals are below that target; their convergence does not qualify
the later dependent normalization or general irreversibility.

At the failed phase entries, maximum nodal phi outside the original prescribed
active band is 0.01308695 (coarse) and 0.004064795 (fine), with active nodal normal
distances reaching 0.40625 and 0.375. These are resolved diffuse tails outside
the original band, not new reconstructed geometry. Through the last common
accepted time, no off-band particle H increases were observed. Thus spread of
the elliptic phase field is distinguished from new local off-band driving.
These failed-entry fields were never accepted as complete mechanical timesteps.

The initial full-support policy remains implemented consistently and passes P1.
The next scientific question is how positive normalization should be represented
on the fixed fault under this along-fault phase variation, including whether the
structural discretization resolves it. No change to that projection or to H is
selected here. The half-step path also develops a small off-band phase tail;
no new mask is used to hide it. Spatial bins and failed-state spread diagnostics
in the evidence distinguish this diffuse response from off-band H activation.

## Qualification and remaining work

This is a failed initial coupled admissibility test with resolved native feedback,
not full P2 qualification and not proof that the H law is increment consistent.
[P1 remains qualified for its prescribed-H, fixed-support controls](P1_integration_report.md).
All attempts and the original failed equality evidence are preserved. Final
coupled runs are local and short (about 17–25 s each, below 270 MiB child RSS and
about 2.2 MB output per case). No production/cluster run, commit or push occurred.

The [commands and checker](../../../benchmarks/reconstructed_fault/propagation/COUPLED.md)
retain a failed P2 gate; solver output matching alone cannot pass it. Remaining
P2 requirements include resolution of normalization/healing admissibility, a
complete same-interval feedback comparison, physical rejected-step rollback,
coupled restart, transported/birth histories, and broader distributed coupling.
The predefined single-fault region is not a general admission policy for multiple
faults, nucleation or propagating endpoints. No P3 or fault propagation was begun.
