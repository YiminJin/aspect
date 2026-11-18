# Free-turn step16 recovery

**Recovered: both loading signs accept through 0.080 s.** The selected recovery
uses two midpoint intervals of .0025 s from each accepted .075 s parent. The
original .005 s failure remains unresolved and preserved. No derivative fix,
search-rule change, phase cutoff, tolerance relaxation or increased iteration
cap was adopted. All physical intervals use the original protected executable.

The independent reflected history starts at t=0 with reversed initial shear,
then the consistently reflected off-axis loading. Both histories use .005 s
through .075, followed by .0025 to .0775 and .080. The positive early history
used two ranks before the retained four-rank parent; the reflected history uses
four throughout. Their actual field differences are measured below, rather
than assuming bitwise symmetry from identical physical inputs.

## Accepted endpoints and figures

| Sign | Accepted time (s) | Phase KKT | Mechanics | Added fracture (J/m) | Interval physical-work error (J/m) |
|---|---:|---:|---:|---:|---:|
| Positive | .0775 | 5.37581e-8 | 4.41785e-17 | .04620570649 | +4.01408e-6 |
| Reflected | .0775 | 5.37577e-8 | 4.33772e-17 | .04620570647 | +4.01408e-6 |
| Positive | .080 | 5.51483e-9 | 1.07362e-14 | .07825205489 | -7.59873e-6 |
| Reflected | .080 | 5.51483e-9 | 1.07362e-14 | .07825205487 | -7.59873e-6 |

The unchanged gates are mechanics1e-9 and phase KKT1e-7. Each accepted step has
exactly one publication. The common .075 positive parent had added fracture
.0272637130 J/m and phase KKT6.07147e-8; its original checkpoint and failed
coarse-step logs remain untouched.

- [Matched absolute phi=.05 contours and local mesh](../../../benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16/free-turn-contours-0017.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16/free-turn-contours-0017.pdf)).
- [Material planes, age, signed slip and rate](../../../benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16/free-turn-histories-0017.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16/free-turn-histories-0017.pdf)).
- [Obstructing rows, tangent probes and recovery convergence](../../../benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16/recovery-diagnosis.png)
  ([PDF](../../../benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16/recovery-diagnosis.pdf)).

At .080, direct local Q2-root refinement gives:

| Sign | Rightmost absolute-contour point (m) | Extension beyond x=6 / hx | Signed lateral displacement / hy |
|---|---|---:|---:|
| Positive | (6.243095748, 2.048261387) | 3.889532 | +1.544364 |
| Reflected | (6.243095748, 1.951738612) | 3.889532 | -1.544364 |

Thus extension is .243096 m and signed lateral displacement is ±.0482614 m.
Both contours connect to the seed: sampled superlevel checks at two resolutions
find one connected component ahead, and an independent horizontal native-Q2
root cut through each extremum has one interval reaching the seed. Adjacent
pieces join only when both independent DG traces exceed .05. Contour subdivision
64→128 changes front x by5.93e-8 m; local root refinement adds1.17e-7 m.
This refines the observation, not the physical mesh (hx=.0625, hy=.03125 m).

Maximum reflected phase-coefficient difference is1.0741e-9; native-weight L2
error is2.0181e-10, or4.1557e-9 relative to the phase increment from the seed.
Physical tau and cohesive A relative L2 errors are2.8688e-10 and6.7349e-10.
Maximum age/slip/rate reflection errors are2.69e-10 s,2.86e-13 m and6.00e-11 m/s.
Plane-angle error is1.84e-8 rad, with **no formed-flag disagreements**.
The corresponding .075 parent phase/angle errors were2.75e-10 and1.60e-8 rad.
This establishes a loading-reflected connected response in this bounded model.
It does not establish a converged turn angle or general branching capability.

## Histories, signed deformation and work

All384 original planes remain exactly horizontal. Previously formed angles and
reference measures are unchanged through both new acceptances. There are528
formed sections at.080 (144 after the initial seed), and240 still prospective.
The two recovery intervals register36 then48 sections with maximum native phase
3.84e-12 and8.19e-11. These small-phase registrations remain visible and are
not equated with contour advance. The inherited old/new angle jump remains
.155927 rad. Prospective age remains .5+t without a birth reset.

Opposite signs occur locally within88 sections at.080. For the positive case,
positive/negative integrated local rates are .0658635 and -9.15082e-6 m²/s.
Corresponding powers are +9.24065 and -.00125965 W/m; summed section power is
+9.23939 W/m, with every section nonnegative. Negative contributions are .0139%
of positive integrated rate and.0136% of positive power. The maximum section
cancellation `1-|integral w|/integral |w|` is.726876%, near x=5.74789 m and
its left counterpart. Large negative-power contributions occur in the old taper,
including x=5.72621 m, where reverse-rate samples occupy the flanks. No clipping
or new localization rule is used.

Exact quadrature-sign inspection finds up to two positive and three negative
sampled components in the positive case, exchanged under reflection. Twenty-two
sections have multiple positive components, with maximum phase .182–.475;
these are not solely tiny-phase tails. They are sampled sign runs, not an
independent continuous-band or branching reconstruction. The whole-section
work-adjoint law permits local negative work while retaining positive integrated
section resistance. The CSV records every section, including these cancellations.
Native rate-adjoint errors are below1.74e-17 m/s and power-adjoint errors below
6.22e-14 W/m at the positive recovered endpoints.

Full-history physical-work error is +5.23115e-5 J/m signed and6.75089e-5 J/m
accumulated absolute: .06685% and.08627% of added fracture. Both signs agree to
3.41e-13 J/m in signed error. The recovery window alone has -3.58465e-6 signed
and1.16128e-5 absolute J/m; the two interval errors have opposite signs. Small
algebraic ledger errors are not substituted for these physical-work errors.
No same-parent accepted coarse .080 state exists for a temporal-accuracy claim.
At.080 the added fracture partitions into central band .0334025, old taper
.0233836 and initially intact axial region .0214660 J/m. Remote/boundary added
fracture remains zero. Central W/hy is2.1014, still a normal-resolution limit.

Positive restore phase, section and plane errors are exactly zero. Original
parent15 lacks a point export, so a new independent tensor-to-export comparison
for that parent is unavailable; its complete checkpoint and prior rank-transfer
controls remain available. The reflected parent has a point export: full physical
tensors, phase, section state and planes all restore exactly. Endpoint coefficient increments have minimum0; coefficients span -5.42e-19
(arithmetic residue retained from the seed) to .60. Thus bounds hold to arithmetic
precision without clipping; old planes/measures remain exactly unchanged.

## Cost, decision and next experiment

The positive recovery takes2092.07 coupled-solver seconds (692.24,1399.83);
reflected recovery takes2101.29 (718.53,1382.76). Whole-process times including
compression are2140.34 and2151.52 s, about35.7–35.9 minutes. The reflected
initialization-to-.075 prefix takes3202.56 s including compression. All use four
bound cores with one thread/rank; analysis follows MPI completion. Sparse export
cadence affects observations only. No numerical job remains.

**Decision:** retain the original solver and use the documented half-step
schedule for this recovered comparison. The evidence supports a stiff nonlinear
near-intact search obstruction, not a demonstrated tangent defect. Successful
recovery and reflection do not remove the cold-creep/admission sensitivity or
qualify the turn's time/space accuracy. This is the synthetic fixed-reference
bulk-cohesive material-strip control, not the mature-fault production route.

The next justified experiment is one positive, matched-parent window from.075
with **four .00125 s midpoint intervals**, compared against these two half-steps.
Measure phase increment, contour/plane/state changes and signed/absolute physical
work before allocating another mesh or longer campaign. It is **proposed, not
launched**. Normal resolution and the tiny-phase plane-registration assumption
remain separate limitations; no general solver redesign is justified by this task.

## Cause and evidence

The .005 s replay uses the copied step15, restart slot01 from
`first-turn-positive-fourcore-end`, four ranks, original search ordering and
unchanged gates: mechanics 1e-9, phase KKT 1e-7, cap80. An explicit diagnostic
stop at iteration40 leaves all diagnostic trials unpublished. The original
80-iteration failure remains intact. See [prior report](FIRST_FREE_TURN_REPORT.md)
for the physical history and its subcell accuracy limit.

All iteration coefficients below are midpoint **stage** coefficients. The
reported natural-map residual uses the solver-scaled phase gradient R:
`x - clip(x - R/m, lower, upper)`. Here the fracture scale Kf=2, so the
physical stage-functional gradient divided by row mass is `2 R/m`.
The sparse CSV records both scales, bounds and active/free status; these are
not endpoint phase increments.

At iteration30 the current worst phase row is119863, cell13318_0, coefficient1,
at (6.15625,2.03125) m. Its stage value is .03617611795, lower obstacle0,
and mass .0002170138889. Its projected residual is 7.422115742e-7.
For alpha=.0625 its actual/predicted residuals are 6.958233409e-7 and
6.958233296e-7. This row follows the tangent closely.

The larger-step limiter is a different row:120060, cell13340_0, coefficient0,
at (6.375,2.0625) m, with stage value/obstacle gap 8.47588753e-12 and
Newton displacement +2.181464608e-10. It remains free at alpha=.0625.
Its current gradient divided by mass is -2.544363771e-7; the linear prediction
is -2.385311662e-7, but the actual trial is -4.338089526e-6.
At alpha=.5 it reaches -2.729619174e-4. At alpha1 another row120075
switches from lower-active to free and reaches magnitude .0648982.
These are nonlinear trial failures, not failure of the linear VI certificate.

Contributing prospective strips612–617 have zero parent phase. At the sampled
native quadrature points, stage phase is of order 1e-11, section rate is
1.86e-12–3.42e-12 m/s, resistance about60–62 Pa, and resistance derivative
with respect to free rate about8.75e11–1.61e12. Opposing stripwise changes
partly cancel at first order; finite relative changes in these very small phase
coefficients produce a substantial nonlinear driving remainder. No phase cutoff
or registration threshold is introduced.

At iteration36 the one-sided probe on row120060 gives:

| Fraction of feasible Newton displacement | Actual change in R/m | Tangent change | Relative discrepancy |
|---:|---:|---:|---:|
| 1e-5 | 6.95306e-12 | 7.05076e-12 | 1.386% |
| 1e-6 | 7.03562e-13 | 7.04936e-13 | .195% |
| 1e-7 | 7.09433e-14 | 7.07266e-14 | .305% |

The mirrored blocking row behaves similarly. Smaller probes become dominated
by assembly arithmetic, including cancellation among contributions much larger
than the residual. The CSV's scalar subtraction estimate alone is **not** a
bound on this assembly uncertainty. These measurements support the analytic
fixed-plane tangent on the implicated row; they do not prove all derivatives
or justify an exact finite-displacement prediction.

Near iteration36–37 the maximum changes to120060. By39, alpha1/1024 raises
the maximum from7.072007562e-7 to7.074400673e-7, whereas1/2048 lowers it
only to7.070879308e-7. This explains the observed nearly stationary tail.

The late-stage functional Armijo test allows the rejected larger trials; the
max-residual tie-breaker blocks them. Independently accumulated local energy
differences and direct Gauss32/64 resistance increments were evaluated rather
than subtracting only global functionals. At iteration30, alpha=.0625 gives
DeltaF=-5.73791e-15 versus slope increment -5.78236e-15, but the energy-scale
arithmetic estimate is2.93e-13 and the existing search allowance3.27e-13.
These tiny signs cannot justify a changed energy acceptance rule. Earlier search
iterations do have genuine functional rejections; the diagnosis is specific to
the late stall.
The fracture difference uses its exact quadratic increment: the existing raw
phase gradient with locally reintegrated mechanical driving removed, plus half
the fracture-Hessian quadratic form. It is not a second independent fracture
assembler. Storage differences are accumulated pointwise in extended precision;
the underlying point states and MPI sums remain double precision.

A read-only comparison at iteration38 evaluates the existing shift1 direction.
Every fraction1 through1/128 fails the original search test; at1/128 its scaled
merit is229.55 versus current about7.07. No alternative is published. This
does not exclude every conceivable globalization method, but provides no
evidence for adopting the tested alternative. The selected recovery is therefore
the authorized **distinct discretization**, dt=.0025 from .075, preserving
equations, bounds, tolerances, iteration cap and history ownership.

## Provenance and reproduction

Worktree/branch: `aspect-pf-rsf-particle-free` / `pf-rsf-particle-free`;
HEAD `b32a2d4f6e21c219f9af786d1b6e3725ea78a5a4`. Incoming manual work is
recorded in `evidence/recover_free_turn_step16/entry.patch` and preserved.
Binary identities are independent of HEAD and the stale configured banner.

The physical fallback uses protected core SHA256
`089824ae44c9c53b57921fe0c5019c59f4a37cf48e1f753917e232f4e4c22378`
and checker `81efcd6e5e0013f4124bf1e1ba857005a2a23447985f69b16464fbc4d3131d69`.
Diagnostic core SHA256 is
`8bcc4ece5c74c797208144e691f5cd8fab8ddf30ac16357d383a5c52662f3602`.
The only new C++ behavior is opt-in observation: sparse row/strip tracing,
independent functional differences, feasible probes, read-only existing-direction
comparison, and an explicit unpublished diagnostic stop. Physical assembly,
search acceptance and publication are unchanged. The diagnostic build completed
with two build workers; native runs use four bound cores and one thread/rank.

`benchmarks/reconstructed_fault/particle_free/recover_free_turn/run.py` selects
one window. Its launch manifests record exact commands and environments; each
run's `command.json` records executable/checker/input/checkpoint hashes. Use
`analyze_trace.py recover-free-turn-trace-complete` for the row/functional tables.
Closed large CSVs are compressed only after MPI exits, with roundtrip SHA checks.
No historical data loss is repaired or erased from the cleanup records.

Diagnostic execution costs: the first, deliberately stopped instrumentation
attempt482.50 s; completed replay through39 and explicit stop at40 takes1978.47 s
including wrapper overhead. Its nonzero exit is deliberate, not a new physical
convergence failure. No diagnostic state is an accepted endpoint.


The final [preservation manifest](../../../benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16/completed-preservation.json)
verifies all four protected executables/checkers, the original .075 checkpoint
and failure logs, the tested diagnostic source, and all new physical-run inputs,
logs and checkpoints. All three physical processes exit0. Restart slot03 in each
recovery output holds accepted17 at.080; slot02 holds accepted16 at.0775.
The only cleanup removed the regenerated854139194-byte precompiled header after
building, with a hash record; future compilation regenerates it. No executable,
runtime library, output or checkpoint was removed. Existing cleanup loss records
remain unchanged. `git diff --check` and Python syntax checks pass. No commit
was requested or made.

[Runner and analysis instructions](../../../benchmarks/reconstructed_fault/particle_free/recover_free_turn/README.md).
Compact CSV/JSON tables, source manifests and PNG/PDF figures are under
`benchmarks/reconstructed_fault/particle_free/evidence/recover_free_turn_step16/`.
