# FaultFriction public-interface review

This follow-up starts on `pf-rsf-particle-free` at
`3928edaf7e6a59383ef4bf65aeb22f34ff723a4c` **plus** the completed, uncommitted
[aging/resistance cleanup](FAULT_FRICTION_API_CLEANUP_REPORT.md). The entry patch
and binary hashes are retained in
`benchmarks/reconstructed_fault/particle_free/evidence/friction_interface/`.
The tested incoming executable (`b80ab805…`) and checker (`38c533d9…`), plus
consumer/provider libraries, are copied to
`build-particle-free/friction-interface/reference/`. No checkout/reset or
checkpoint conversion was performed. The previous report describes its own
completed stage, including the now-superseded stateless state-partial rejection.

## Public contract and grouping

| Quantity | Incoming interface | Reviewed interface |
|---|---|---|
| Coefficient | Separate two-/three-argument overloads selected by caller | One `friction_coefficient(fractions, speed, optional_age)` |
| Speed partial, age fixed | Separate two-/three-argument overloads selected by caller | One `friction_coefficient_derivative_wrt_slip_rate(fractions, speed, optional_age)` |
| Age partial, speed fixed | Stateful-only operation, stateless law rejected | One `friction_coefficient_derivative_wrt_state(fractions, speed, optional_age)`; stateless result zero |
| Candidate age and speed derivative | One explicit-method `update_state` | Retained: stateful-only, zero speed/duration admitted |
| Signed resistance and total rate derivative | `evaluate_resistance` | Retained: regularized RSF, trial aging, signed/zero rates |
| Section rate and eliminated tangent | `solve_section_slip_rate` | Retained: safeguarded monotone root |
| Search integral | `compute_resistance_potential` | Retained: fixed incoming age, not physical work |
| Initialization inversion / timestep / identity | Distinct quantities | Retained, no alternative wrappers |

`std::optional<double>` is passed by value, with **no default argument**.
`std::nullopt` explicitly means there is no state. The configured object owns
law dispatch: stateful evaluation requires finite positive age; a stateless
law ignores supplied age and its age partial is identically zero. All three
coefficient operations retain the finite speed >= Vmin contract. The explicit
signed resistance operation separately admits zero rate. No caller may select
a different law by choosing an overload or omitting state.

Declarations and definitions now follow configuration, aging, coefficient and
partials, initialization inversion, signed resistance, section solve, search
potential, then timestep/identity utilities. Small accessors remain inline.
The private coefficient helpers are contiguous beside the coefficient methods:
`rate_dependent_parameters` and `rate_state_parameters` mix their actual law's
parameters; `rate_state_coefficient` evaluates fixed-state RSF value/partials
and is shared with signed resistance; `coefficient_at_state` owns public
validation and dispatch. It returns the existing small value/partials record.
No history owner, mutable cache, allocation or generic strategy was added.

## Concrete caller simplification

`PhaseFieldFault::evaluate_reconstructed_fault_point` formerly performed three
law tests and contained four coefficient/derivative call expressions. It now
performs **one** history-availability test and **two** unconditional calls:

```cpp
std::optional<double> age;
if (fault_friction.has_state_variable())
  age = interpolate_fault_scalar(/* existing accepted surface property */);
const double mu = fault_friction.friction_coefficient(fractions, speed, age);
const double dmu_dV = fault_friction.friction_coefficient_derivative_wrt_slip_rate(
                       fractions, speed, age);
```

This block shrinks from 21 lines to 10 (recorded in
`caller-simplification.json`). The dummy zero age and two duplicated dispatch
branches are gone. Known
stateless calls in material setup, FE phase energy and bulk cohesive history
pass `std::nullopt`. Stateful diagnostics pass the age they already own; there
is no extra argument preparation. Test references remain law-specific because
they check different mathematical formulas. The complete maintained caller
search found no other coefficient-law dispatch branches to remove.

Remaining `has_state_variable()` checks protect actual state ownership,
initialization, publication, timestep policy or supported combinations. Removing
those would access nonexistent histories or change supported physics. The prior
cleanup's removal of per-vertex composition/mixture preparation for global-Dc
aging is retained.

## Verification

The core, bulk-tip checker, mature checker, background provider and eight
diagnostic consumer targets build successfully. The consumers cover stateful
and stateless friction, Stage I, the state limiter, BP3, BP5 startup and normal
stress, and mechanical-mode diagnostics. The public API requires rebuilding
plugins that called the removed coefficient overload symbols.

All 189151 unit assertions in 169 cases pass. The two-rank focused rheology
audit passes with its original mathematical references and tolerances. New
contract checks verify all three stateful operations reject absent, zero and
nonfinite age, stateless value/rate partial ignore supplied age, and stateless
age partial is zero with or without age. Existing tests retain BE/exact aging,
zero duration/speed, derivative limits, signed symmetry, mixture/inversion,
a=0, total versus partial tangents, root and search integral checks.

Commands and outcomes are in `controls.json`; final numerical differences are
in `comparisons.json`. No long history or production run is selected. The
two cleanup passes are included in the accompanying ASPECT-baseline source
commit; generated evidence and binaries remain local.


| Bounded control | Ranks | Result | Elapsed seconds | Retained comparison |
|---|---:|---|---:|---|
| Unit suite | 1 | 189151 assertions / 169 cases pass | 6.98 | Existing expectations |
| Rheology audit, two-material fixture | 2 | All gates pass | 2.17 | Independent formulas and new API contracts |
| BE interval .0575–.06, restart and rejection | 2 | Pass | 80.43 | `friction-api-be01`, step 24 |
| Midpoint to .09, tangent at step 8 | 1 | Pass | 9.39 | `friction-api-midpoint-final`, step 18 |
| Midpoint restart/rejection to .08 | 2 | Pass | 4.03 | `friction-api-midpoint-recovery01`, step 16 |
| Mature layered-viscosity transport | 1 | Pass | 15.30 | `friction-api/mature`, all three exported states |
| Actual state timestep-limiter oracle | 2 | Pass | 3.53 | Existing analytic assertions on supported mature fixture |

The maximum absolute difference is **zero in every compared finite physical
column**: bulk stress/cohesive tensors, phase, driving and local deformation;
section rate/age/slip/resistance; accepted energy/work ledgers and acceptance
residuals; mature tensor/geometry/profile/age/slip-memory fields and publication
counts. Nonfinite masks match (the intentionally unavailable endpoint diagnostic
remains unavailable). Peak RSS is recorded separately and excluded from physical
equivalence. Inputs, commands, exact field names and per-column differences are
retained; the controls total about 122 seconds including launch overhead.

Final core SHA256: `5e9bd0623611e0f7aa65929bc52cde522bc8cbf3435149abc84b5849e2a440da`.
Final checker SHA256: `f7b3b05e9e6fef98f11b25d344c71a7516d66e84cc671326f06db130f36da83c`.
`verification.json` records source hashes independently of the executable's
cached configure-time revision banner. No result identity is inferred from HEAD.

The inherited historical particle/Q1 fixtures remain compile-only; the real
limiter runs on the supported native mature configuration. The previously
documented obsolete aggregate checker dependency was not rebuilt or reenabled;
the required mature target was built explicitly. No test tolerance, mathematical
formula, quadrature, convergence criterion, checkpoint meaning, spin, time
location or history transaction changed. The twelve retained method bodies are
identical after whitespace normalization.

This closes the API/caller cleanup. It establishes local numerical equivalence,
not new propagation accuracy or production readiness. The prior failed fine
continuation and unresolved growth/time-integration limits remain unchanged;
longer experiments stay deferred. Both cleanup passes are consolidated under the existing ASPECT baseline.
Any further numerical campaign requires a separately selected task.
