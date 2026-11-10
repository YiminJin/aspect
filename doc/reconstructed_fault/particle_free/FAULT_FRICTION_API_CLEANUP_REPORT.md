# Fault-friction API cleanup

> Historical first cleanup stage. The subsequent [public-interface review](FAULT_FRICTION_INTERFACE_REVIEW.md)
> consolidates coefficient overloads and supersedes this stage’s stateless
> state-partial rejection. Its original verification evidence remains intact.

Starting revision: `3928edaf7e6a59383ef4bf65aeb22f34ff723a4c`, branch
`pf-rsf-particle-free`, with no tracked changes at entry. Retained executables,
checkpoints and outputs are preserved. This is a bounded API cleanup; the
previous material formulas and scalar-obstacle correction remain in place.

## Interface and ownership

| Previous interface | Current interface |
|---|---|
| evaluate_age and two forwarding update_state overloads | update_state(speed, accepted_age, duration, AgingMethod) returning StateUpdate |
| SignedEndpoint / signed_endpoint | ResistanceResponse / evaluate_resistance |
| SectionEndpoint / section_endpoint | SectionSlipSolution / solve_section_slip_rate |
| section_primitive | compute_resistance_potential |

There is one aging implementation and no compatibility wrapper. The common
updater accepts zero speed, rejects stateless laws and invalid inputs, and
computes candidates without publishing history. Mature publication and its
logarithmic-state timestep predictor enforce their positive-rate/Vmin contract
at the call site. Per-vertex composition gathering and fraction construction
used solely for aging were deleted: Dc is global.

The [equation-to-caller table](tip_treatment/friction_and_direction.md#shared-rheology-ownership)
is the current reading path. Mature publication/prediction select exact aging;
BE trials/publication select full-duration BE; midpoint resistance uses exact
half-duration aging and publication exact full-duration aging from the same
accepted age. Publication still uses the independently integrated section rate.

ResistanceResponse distinguishes resistance, derivative_wrt_slip_rate, age,
friction and radiation. SectionSlipSolution exposes resistance_response,
slip_rate and resistance_derivative_wrt_free_rate. The coupling stores
section_solutions, removing the former law.law nesting. Physical history fields
retain their meanings. The search potential integrates resistance over trial
rate with fixed incoming history; it is never stored energy or fV work.

## Shared private operations and readability

`rate_state_parameters` owns checked stateful parameter mixing;
`rate_state_coefficient` owns value and fixed-age partial derivatives. The signed
resistance adds the aging chain rule. `rate_dependent_parameters` now shares the
previously duplicated stateless static/dynamic coefficient and weakening-speed
mixture. It preserves active-component selection and arithmetic ordering.
Public rate restrictions remain distinct from common parameter validation.

Root elimination and the search integral remain separate mathematical operations:
the former retains its asinh bracket, monotonicity guard, stopping criterion and
iteration budget; the latter retains Gauss48. Exact aging retains expm1, zero
speed/duration and small/subnormal-argument limits. No input selector, checkpoint
meaning, constitutive equation, physical quadrature or acceptance tolerance changes.
The header and source group state evolution, coefficient evaluation, resistance,
section elimination, search potential and existing utilities in matching order.
Touched files use the repository's Artistic Style 2.04 configuration. The source
explains the small-speed derivative, signed chain rule and safeguarded coordinate.

All maintained source/test/benchmark callers were searched, including mature
publication, timestep prediction, BP3/BP5 diagnostics and phase-field tests.
The BP3 within-step diagnostic still referenced the retired derivative-only
helper; it now obtains the derivative from the canonical StateUpdate result.
Historical reports and archived outputs retain their original API names.

## Verification

The final core builds, as do the bulk-tip checker and all eight diagnostic
consumer targets (stateful/stateless tests, Stage I, state limiter, BP3, both
BP5 diagnostics and mechanical-mode/length-scale checks). The required mature
energy checker also builds. Full units pass: 189151 assertions in 169 cases.
The focused two-rank audit passes BE/exact aging, zero speed/duration, small
speed, derivative/semigroup, signed resistance and condensed derivatives,
mixtures, a=0, constant-law timestep and stateless rejection. Added stateless
mixture checks cover the newly shared computation with independent formulas,
including inactive components. The canonical updater rejects stateless aging.

| Short control | Comparison with retained evidence |
|---|---|
| BE .0575→.06, two ranks, rejection/retry | Every exported endpoint point/section field identical |
| Midpoint dt=.005, t=0→.09 control, one rank, tangent checks | Every exported endpoint point/section field identical |
| Midpoint restart to .08, two ranks, rejection/retry | Every exported endpoint point/section field identical |
| Mature layered transport, retained short input | All compared material/background fields identical; only diagnostic peak RSS differs |
| Existing timestep-limiter oracle, two ranks | Independent analytic bound, tiny timestep, equilibrium, both aging directions, nonmutation and MPI checks pass |

The limiter oracle is hosted in the supported mature fixture, retains its
original maximum timestep of 100000 s and runs only the zero-clock observation.
Its probes restore accepted state and velocity. No physical history is advanced
by that test. The retained point/section comparisons have zero maximum absolute
phase, shear-stress, age and slip differences. Memory counters are explicitly
excluded from the mature physical comparison; no timing speedup is claimed.

The older stateful friction input is rejected by the existing particle/Q1
retirement guard, before reaching the API test. Its consumer compiles, and the
relevant law assertions run in the supported focused audit. Initial limiter
hosting attempts omitted the required background provider and inherited the
mature fixture's smaller timestep ceiling; both failed attempts are retained.
The final input restores the oracle's original ceiling and activates the
background provider. No solver tolerance or oracle tolerance was relaxed.

One unrelated historical aggregate build target remains unsupported:
`transported_bulk_spin` includes `spatial_relaxation.h`, which still references
the removed `TensorAdvection::distributed` member. The required mature energy
checker builds and its control passes. No old friction API was retained to
conceal this historical dependency. No maintained friction caller remains on
the retired API; historical reports and archived sources are unchanged.

Entry identities, actual commands, source checksums, build logs, failed attempts
and numerical comparisons are in
[friction_api evidence](../../../benchmarks/reconstructed_fault/particle_free/evidence/friction_api/README.md).
The full growth histories, expensive fine interval and production jobs were
not repeated. Existing physical/spatial accuracy limitations remain unchanged.
After the final reading-order and Doxygen pass, the core, checker and all eight
consumers were rebuilt. The final executable repeats units, the focused audit,
short BE/midpoint intervals and the two-rank limiter successfully. Its BE and
midpoint point/section exports again match the retained references exactly.
Actual final core SHA256: `b80ab805067ec9e6e0ebb5cc9766746c7764225505e26e6db25b9dc1fdc88953`.
The per-control metadata distinguishes earlier recovery/mature checks from this
final build; source/binary identity is not inferred from HEAD.

No commit or push is part of this task.
