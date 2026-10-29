# Boundary-driven causal extension: stress response and reserve dependence

Baseline `f1f9f29cf40c32d1ddcea000b724375b64853532` plus preserved local work.
[Inputs/gates](COHESIVE_INTERFACE_LOAD_TURN_SELECTION.md) ·
[PNG/GIF/data index](../../../benchmarks/reconstructed_fault/particle_free/evidence/cohesive_interface_load_turn/INDEX.md).

**New cohesive geometry responds to the changed accepted stress, but propagation
is materially dependent on previously reserved geometry and scheduling.**
Actual records selected during the loading turn subsequently damage, reaching
a=.783 and .325 on two successive appends. This is more than drawing a tied tail.
The bounded tests do not establish reserve-independent adaptive propagation.
Intact-birth guards and the remaining mechanical accuracy failures are retained.

## Physical experiment and implementation

The finite specimen remains `[0,12] × [−6,6] m`, with original seed `(3,0)–(6,0)`.
Retain the prepared, equilibrated seed histories, G=1e6 Pa, eta=1e8 Pa s,
Gc=1 J/m², peak strength 3000 Pa, p0=10000 Pa, existing RSF/radiation law,
closed contact, Q2/P1disc bulk, Q1 jump and Gauss3 surface history ownership.
Bulk quadrature is Gauss6. Physical R=.75 m and the **numerical vertical lifting
and normal-test half-width w=.75 m** stay fixed independently of mesh and reserve.

Two fixed laboratory strain bases are
`E0=[[-b,.5],[.5,b]]`, `E1=[[-.5,-b],[-b,.5]]`, b=.5 tan40°.
Impose `uD=(a0 E0+a1 E1)x` on the physical outer boundary, where
`a=gamma(cos(2 alpha),sin(2 alpha))`, gamma=.0003+.004 min(t,.455).
Alpha changes from 0 to −20° with quintic smoothstep over .410–.440 s.
Displacement holds over .455–.460 s. The control keeps alpha=0, with the same
strain norm and clock. At .410 s, new damage integral is already .037414 m
and the a=.05 process front has advanced .215748 m.

Both affine Maxwell memories advance as `b_new=beta(b_old+Delta a)`;
every signed jump-memory mode retains its original recurrence. Existing stress,
geometry, damage, age and six surface histories are never rotated or reset.
Native mechanics, current-pressure coupling, admission and tangent checks
compute each accepted state. Work sums both conjugate reactions:
`Wend=sum_alpha (K c_new)_alpha Delta a_alpha`; trapezoidal work is evaluated
separately. Confinement contributes zero work to these trace-free outer motions
and zero normal jump. Mean p′ stays zero; friction uses the actual normal reaction.

The homogeneous maximum-shear preference changes +20°→0°. Measured accepted
preference follows it, reaching about +.5°, while old formed tangents remain fixed.
The same accepted-stress R-disk sample, nearest-conjugate history preference and
`dtheta/ds=(theta_star−theta)/R` hypothesis are retained. This is not Coulomb-angle
intact admission or a global incremental potential.

## Actual fronts and stops

Fronts a=.05, .5 and 1 are bracketed by **owned irreversible Gauss records**;
current slip cannot locate them reliably during unloading. Typical brackets
span .02–.04 m. No interpolation into a different constitutive state is used.
There is no detached above-threshold Gauss activation in these runs.

| Case | Last accepted t, s | New ∫a ds, m | Developed advance along Γ, m | Stop |
|---|---:|---:|---:|---|
| Turned, S=.1875 m | .44625 | 1.10444 | .65573 | Tied birth needs 3065.40 Pa |
| Turned, S=.75 m | .46000 | 1.38075 | 1.00202 | Declared endpoint |
| Unchanged, S=.1875 m | .43775 | .62394 | .02862 | Tied birth needs 3004.07 Pa |
| Unchanged, S=.75 m | .45500 | 2.34135 | 1.40594 | Declared 4.125 m append capacity |

All rejected insertions remain unpublished, with saved incoming state, proposed
geometry, coefficients, weak force and required multiplier. No damaged birth,
strength adjustment or tolerance relaxation repairs them. Capacity termination
is not physical arrest. Both large-reserve cases compare through the loading
endpoint; only the turned case completes the hold. It advances another .13008 m
during hold: stopping boundary motion does not remove the accepted stress.

Rotation initially accelerates development, then reduces later growth relative
to the unchanged control. At .455 s the turned large-reserve case has 47.65%
less new damage and .5340 m less developed advance. Rotation changes the resolved
shear and normal loads as well as direction; this is not an equal-work experiment.

## Reserve, scheduling and numerical uncertainty

Reserve S is geometric end minus the connected a=.05 front, distinct from R and w.
Primary appends have length .375 m, with unchanged .09375 m surface cells.
Committed virgin directions remain permanently frozen, as in the retained
append-only contract. No new scientific assumption is adopted.

The third large-reserve append commits at .3455 s with preference 19.971°;
the corresponding short-reserve append commits at .42075 s with preference
15.088°. At common .44425 s it has peak a=.783, and the fourth append has
peak a=.325 (integrated damage .2083/.0714 m respectively). The differing directions therefore belong to actual damaged material.
The primary developed fronts still traverse geometry committed before any
substantial load turn. In the half-quantum control, .0676 m of weighted saturated
records enters an append selected during the turn (preference 17.611°), a limited
developed response; it does not qualify the full requested turn.

| Comparison | Common t, s | New-damage relative difference | Process-front difference, m | Developed-front difference, m |
|---|---:|---:|---:|---:|
| Short versus large reserve | .445 | +10.94% | +.22440 | +.03640 |
| Fresh 96² versus 64² bulk | .445 | +1.22% | +.05763 | Same Gauss bracket |
| Half timestep versus original | .445 | −.456% | Same Gauss bracket | Same Gauss bracket |
| Half future append quantum | .44425 | +5.77% | +.15861 | +.03687 |

The original timestep is .00025 s; the half-step and half-quantum replays start
from the **same accepted .410 s checkpoint**, preserving the already committed
.375 m append. Fresh mesh initialization uses the same physical seed prescription.
Mesh and temporal differences lie below the reserve effect. Append-quantum
sensitivity itself exceeds the 5% integral and one-surface-cell process-front
targets near termination. It changes accepted sampling times and realized reserve
within `[S,S+Delta_s)`. Thus both scheduling and permanent virgin commitment matter;
the experiment does not isolate a unique reserve-independent path. At .445 s the two primary reserves differ only .0125° in current sampled preference, despite their growth difference. Exact matching
of front brackets is not sub-grid accuracy. The fine and half-step cases reproduce
the birth limit at .446 and .446375 s; the half-quantum case stops at .44425 s.

## Mechanical, history and energy checks

The common .440 s physical jump/Maxwell field is independently reconstructed on
Cartesian64/96 and fitted64/96. These are diagnostics, not accepted-state conversions.
Relative full plane-strain tensor errors in the **selection disk** are .000370%
and .0000480% for Cartesian64/96 versus fitted96; angle errors .000104°/.0000107°.
Common physical Q1 virtual-jump tests give Cartesian96 developed/zone/ahead shear
differences .59%/.75%/.40%. Fitted jump-self storage changes .256% between its
meshes. Affine, jump-self and cross storage are exported separately.

This does not establish point accuracy. Current developed-region raw normal-jump
RMS is **520 Pa Cartesian versus 16.6 Pa fitted96**. The historical tied-ahead
normal-perturbation failure **2.85%, 2.29 Pa** remains open. The new old-region
weak normal perturbation differs **3.56%, .118 Pa** on Cartesian96; it also misses
a 2% relative target, despite its small absolute value. Neither is normalized by
10000 Pa to claim success.

The two-component homogeneous tensor error is <8.49e−13 Pa. Restricting to the old
loading reproduces the native solve, and seven default operator tables are bitwise
identical. Loaded coupled FD error is 7.60e−10 against 1e−6. Native contact-loss
rejection leaves the accepted state unchanged; retry, rebuilt-context JSON restart,
duplicate rejection and exactly-once publication pass. Every successful append
preserves old response/stress-pressure columns and all histories bitwise, with
zero measured transfer storage/stress defect and zero new fracture inventory.
True weak residuals remain below 1e−8 Pa; maximum exact BE ledger error 1.21e−12 J/m.
The restricted tangent diagnostic remains distinct from a general RSF stability proof.

Summed absolute physical trapezoidal-work remainders are .149% of sliding loss /
.670% of added fracture for turned-large, and .389%/.864% for unchanged-large.
The half-quantum **suffix** gives .254%/**1.000894%**, narrowly **failing** the fixed
1% fracture screen; it is not rounded into a pass. Full-run and suffix budgets
are labeled separately in CSV. Algebraic BE closure is not that physical screen.

Forty-eight cached native kernels, including rejected proposals and diagnostics,
take 1281 s in their native timers (0.69–68.3 s each; at most 115337 bulk masters).
Trajectory wall times are 13.5–403 s with explicit cache reuse; the 13.5 s case is
a cached exact reproduction, not a speedup claim. Timed runs do not overlap.
Raw work uses about 858 MiB, below the declared 2 GiB budget.

## Conclusion and next decision

This demonstrates stress-responsive **cohesive formation**, with bounded evidence
of developed advance into a newly selected segment, and substantial reserve/scheduling
dependence. It does not establish a robust adaptive mature path or precise speed.
Production/BP3, previous failures and accepted histories are preserved.

Before another propagation run, resolve the tradeoff exposed here: a long permanently
committed virgin reserve fixes direction too early, while short reserves can require
inadmissible tied birth. A justified next design is to distinguish temporary zero-damage,
zero-slip, zero-memory candidate geometry from persistent formed geometry, committing
direction at energy-accounted admission. That would **change the current permanent
virgin-direction assumption**, not merely tune S. It requires explicit basis/history
preservation and simultaneous admission/work rules, retaining the present birth guard;
formed/damaged geometry must never be rewritten. It is proposed, not implemented.
